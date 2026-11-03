"""Narrowband pass detector, with emission-time curve validation.

Frequency alone cannot distinguish a synthesizer imitating a pass from a real
engine. Classification describes acoustic evidence, not source identity.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import signal, optimize, ndimage, stats
from .types import DetectConfig, Track

_C = 331.3 * np.sqrt(1 + 15 / 273.15)


def ideal_curve(t, f0, v, d_cpa, t_cpa):
    """Exact subsonic retarded-time solution (CPA is emission time)."""
    u = np.asarray(t) - t_cpa
    a = _C**2 - v**2
    te = (_C**2 * u - np.sqrt(_C**2 * v**2 * u**2 + a * d_cpa**2)) / a
    r = np.hypot(d_cpa, v * te)
    return f0 * _C / (_C + v**2 * te / r)


def _spectrogram(x, sr, cfg):
    rate = max(cfg.work_sr, int(np.ceil(2.2 * cfg.fmax)))
    g = math.gcd(rate, sr)
    y = signal.resample_poly(x, rate // g, sr // g).astype(np.float32)
    n = max(cfg.nperseg, int(2 * rate))
    hop = int(round(cfg.hop_s * rate))
    if hop < 1 or hop > n:
        raise ValueError('hop_s must be positive and shorter than the STFT window')
    if len(y) < n:
        return None
    sos = signal.butter(4, [cfg.fmin, cfg.fmax], btype='bandpass', fs=rate, output='sos')
    y = signal.sosfiltfilt(sos, y).astype(np.float32)
    freq = np.fft.rfftfreq(n, 1 / rate)
    band = (freq >= cfg.fmin) & (freq <= cfg.fmax)
    blocks = []
    count = (len(y) - n) // hop + 1
    for start in range(0, count, 1024):
        stop = min(count, start + 1024)
        _, _, z = signal.stft(y[start*hop:(stop-1)*hop+n], rate, nperseg=n,
                              noverlap=n-hop, boundary=None, padded=False)
        blocks.append((2 * np.abs(z[band])).astype(np.float32))
    amp = np.concatenate(blocks, axis=1)
    # filtfilt applies the band filter twice; restore original signal units.
    _, response = signal.sosfreqz(sos, worN=freq[band], fs=rate)
    amp /= np.maximum(np.abs(response[:,None])**2, .1)
    db = 20 * np.log10(amp + 1e-15)
    width = max(1, round(600 / cfg.hop_s))
    centers, medians = [], []
    for start in range(0, count, width):
        stop = min(count, start + width)
        centers.append((start + stop - 1) / 2)
        medians.append(np.median(db[:, start:stop], axis=1))
    medians = np.array(medians)
    for j in range(len(db)):
        db[j] -= np.interp(np.arange(count), centers, medians[:, j])
    return (n/2 + np.arange(count)*hop)/rate, freq[band], amp, db


def _fit(t, f):
    span = max(np.ptp(t), 1)
    mid = float(np.median(t))
    best = None
    for v in (25., 75., 150.):
        p = [np.median(f), v, max(v*span/8, 5), mid]
        r = optimize.least_squares(lambda p: ideal_curve(t, *p)-f, p,
            bounds=([1, 5, 5, t[0]-span], [10000, 200, 100000, t[-1]+span]),
            loss='soft_l1', f_scale=.2, max_nfev=120)
        rms = float(np.sqrt(np.mean((ideal_curve(t, *r.x)-f)**2)))
        if best is None or rms < best[0]:
            best = rms, r.x
    rms, p = best
    return dict(zip(('f0','v','d_cpa','t_cpa'), map(float, p)), rms_hz=rms)


def _sample(s, f, ff):
    idx = (ff-f[0])/(f[1]-f[0])
    lo = np.clip(np.floor(idx).astype(int), 0, len(f)-2)
    w = np.clip(idx-lo, 0, 1)
    out = s[lo, np.arange(s.shape[1])] * (1-w) + s[lo+1, np.arange(s.shape[1])] * w
    return np.where((ff>=f[0]) & (ff<=f[-1]), out, 0.)


def _brightness(s, f, ff):
    """Curve evidence relative to displaced-frequency controls."""
    values = _sample(s, f, ff)
    controls = np.array([_sample(s, f, ff+d) for d in (-5., -3., 3., 5.)])
    return values - np.mean(controls, axis=0)


def _llr(s, expected_snr=1.):
    # Per-bin median whitening: Exp(1)'s median is ln(2).
    z=np.log(2.)*10**(np.clip(s,-40,25)/10)
    return z*expected_snr/(1+expected_snr)-np.log1p(expected_snr)


def _harmonic_spectrum(s, f, count=8):
    """Sum whitened FFT power evidence at integer harmonic frequencies.

    Variance normalization prevents a low fundamental's larger harmonic bank
    from receiving a systematically larger noise score.
    """
    power=np.clip(_llr(s),-1,20)
    out=np.zeros_like(s);norm=np.zeros(len(f))
    for k in range(1,count+1):
        index=np.rint((k*f-f[0])/(f[1]-f[0])).astype(int)
        good=index<len(f)
        weight=1/np.sqrt(k)
        out[good]+=weight*power[index[good]]
        norm[good]+=weight**2
    return out/np.sqrt(np.maximum(norm[:,None],1))


def _zoom_refine(tr,x,sr):
    """Dechirped one-second ZoomFFT: dense samples without a long window.

    Dense FFT sampling removes bin quantization; it does not create additional
    physical resolving power. Only high-confidence local peaks adjust the fit.
    """
    rate=2000
    g=math.gcd(sr,rate)
    y=signal.resample_poly(x,rate//g,sr//g)
    n=rate;window=signal.windows.hann(n,sym=False)
    offsets=[];weights=[]
    chirp_rate=np.gradient(tr.f,tr.t)
    cache={}
    zoom=signal.ZoomFFT(n,[-1.5,1.5],m=121,fs=rate)
    residual_grid=np.linspace(-1.5,1.5,121,endpoint=False)
    dt=(np.arange(n)-n/2)/rate
    for time,freq,slope in zip(tr.t,tr.f,chirp_rate):
        length=np.clip(1/np.sqrt(max(abs(slope),.25)),.25,1.)
        n=max(64,round(length*rate/32)*32)
        if n not in cache:cache[n]=(signal.ZoomFFT(n,[-1.5,1.5],m=121,fs=rate),signal.windows.hann(n,sym=False))
        zoom,window=cache[n]
        dt=(np.arange(n)-n/2)/rate
        center=round(time*rate);left=center-n//2
        if left<0 or left+n>len(y):offsets.append(0.);weights.append(0.);continue
        local_f=np.interp(time+dt,tr.t,tr.f)
        phase=2*np.pi*np.cumsum(local_f)/rate
        spectrum=abs(zoom(y[left:left+n]*window*np.exp(-1j*phase)))
        j=int(np.argmax(spectrum));contrast=spectrum[j]/max(np.median(spectrum),1e-15)
        offsets.append(residual_grid[j] if contrast>1.15 and 0<j<120 else 0.)
        weights.append(float(contrast>1.15 and 0<j<120))
    offsets=np.array(offsets);weights=np.array(weights)
    smooth=ndimage.gaussian_filter1d(offsets*weights,4)
    norm=ndimage.gaussian_filter1d(weights,4)
    tr.f+=np.where(norm>.6,smooth/np.maximum(norm,1e-6),0)


def _bank(t, f, s):
    stride = max(1, len(t)//100)
    tt = t[::stride]
    smooth = ndimage.gaussian_filter1d(np.clip(s, 0, 18), 1.5/(f[1]-f[0]), axis=0)
    ss = smooth[:, ::stride]
    harmonic = ndimage.gaussian_filter1d(_harmonic_spectrum(s,f),1.5/(f[1]-f[0]),axis=0)[:,::stride]
    base = np.arange(max(30., f[0]), f[-1], 1.)
    cols = np.arange(len(tt))[None, :]
    candidates = []
    for v in (20., 40., 65., 90., 120., 160.):
        for tau in (2., 5., 12., 30., 70.):
            for heard in np.linspace(t[0]+.1*np.ptp(t), t[-1]-.1*np.ptp(t), 9):
                d = v*tau
                tc = heard-d/_C
                ratio = ideal_curve(tt, 1., v, d, tc)
                scores = []
                for k in (1,2,3,4):
                    ix = np.rint((k*base[:,None]*ratio-f[0])/(f[1]-f[0])).astype(int)
                    valid = (ix>=0)&(ix<len(f))
                    values = ss[np.clip(ix,0,len(f)-1),cols]
                    scores.append(np.mean(np.where(valid, values, -6),axis=1))
                scores = np.sort(scores,axis=0)
                ix=np.rint((base[:,None]*ratio-f[0])/(f[1]-f[0])).astype(int)
                valid=(ix>=0)&(ix<len(f))
                harmonic_score=np.mean(np.where(valid,harmonic[np.clip(ix,0,len(f)-1),cols],-6),axis=1)
                score = scores[-1]+.35*np.maximum(scores[-2],0)+.2*harmonic_score
                for j in np.argpartition(score,-min(3,len(score)))[-3:]:
                    # Include the harmonic responsible for the strongest score.
                    candidates.append((float(score[j]),np.array([base[j],v,d,tc])))
    return candidates


def _search(t, f, work, candidates, calibration=False):
    likelihood=np.clip(_llr(work),-1,12)
    smooth = ndimage.gaussian_filter1d(likelihood, 1.2/(f[1]-f[0]),axis=0)
    ranked = []
    # Re-score the cached bank after each CLEAN subtraction.
    for _,p in candidates:
        ff = ideal_curve(t,*p)
        scores = [np.mean(_sample(smooth,f,k*ff)) if np.all(k*ff<=f[-1]) else -10 for k in (1,2,3,4)]
        k = int(np.argmax(scores))+1
        pp=p.copy();pp[0]*=k
        ranked.append((max(scores),pp))
    chosen=[]
    for score,p in sorted(ranked,key=lambda q:-q[0]):
        if all(abs(p[0]-q[0])>2 or abs(p[3]-q[3])>12 for q in chosen):
            chosen.append(p)
        if len(chosen)>=32:
            break
    best=None;raw_best=None
    for p in chosen:
        def objective(p,data):
            if not (f[0]<p[0]<f[-1] and 5<p[1]<200 and 5<p[2]<50000 and t[0]-100<p[3]<t[-1]+100):
                return 100.
            ff=ideal_curve(t,*p)
            if np.any((ff<f[0])|(ff>f[-1])):
                return 100.
            return -float(np.mean(_sample(data,f,ff)))
        for data in (smooth,likelihood):
            opt=optimize.minimize(lambda p:objective(p,data),p,method='Nelder-Mead',options={'maxiter':180,'xatol':.01})
            p=opt.x
        ff=ideal_curve(t,*p)
        if p[2]/p[1]<.8 or np.ptp(ff)<max(1.,.015*np.median(ff)):
            continue
        # Independent harmonic channels increase integration significance.
        hypotheses=[]
        for divisor in (1,):
            pp=p.copy();pp[0]/=divisor
            base_curve=ideal_curve(t,*pp)
            channels=[];reverses=[];supports=[];llr_channels=[];llr_nulls=[]
            for k in range(1,9):
                curve=k*base_curve
                if np.min(curve)<f[0] or np.max(curve)>f[-1]:continue
                ev=_brightness(work,f,curve)
                mean=float(np.mean(np.clip(ev,-12,18)))
                if mean<.9:continue
                offsets=np.arange(-3,4)*(f[1]-f[0])
                nearby=np.array([_sample(work,f,curve+d) for d in offsets])
                residual=offsets[np.argmax(nearby,axis=0)]
                support=float(np.mean(abs(residual)<.75))
                if support<.50:continue
                channels.append(ev);supports.append(support)
                llr_channels.append(_brightness(likelihood,f,curve))
                llr_nulls.append(_brightness(likelihood[:,::-1],f,curve))
                reverses.append(_brightness(work[:,::-1],f,curve))
            if not channels:continue
            ev=np.mean(channels,axis=0)
            score=4*float(np.mean(llr_channels))*np.sqrt(len(channels))
            null=4*float(np.mean(llr_nulls))*np.sqrt(len(channels))
            if raw_best is None or score>raw_best[0]:raw_best=(score,pp.copy())
            if score<3.5 or score-null<2.5:continue
            if len(channels)==1 and (supports[0]<.65 or np.ptp(t)<20):continue
            if sum(np.mean(q)>1 for q in np.array_split(ev,4))<3:continue
            hypotheses.append((score,pp))
        for score,pp in hypotheses:
            if best is None or score>best[0]:best=(score,pp)
    return raw_best if calibration else best


def _ridge_vehicles(t,f,amp,s,cfg):
    """Strong, continuously supported RPM ridges; no ideal-pass assumption."""
    active=[];finished=[]
    for i in range(len(t)):
        peaks,_=signal.find_peaks(s[:,i],height=max(10,cfg.snr_db),distance=3)
        used=set()
        for path in sorted(active,key=lambda q:-len(q)):
            prev,j=path[-1]
            if i-prev>2:
                finished.append(path);continue
            pred=f[j]
            if len(path)>2:
                i2,j2=path[-3];pred+=(f[j]-f[j2])/(prev-i2)*(i-prev)
            options=[k for k in peaks if k not in used and abs(f[k]-pred)<1.5]
            if options:
                k=min(options,key=lambda k:abs(f[k]-pred));path.append((i,k));used.add(k)
        active=[p for p in active if i-p[-1][0]<=2]
        active.extend([[(i,k)] for k in peaks if k not in used])
    finished.extend(active)
    out=[]
    for path in finished:
        ij=np.array(path);ii,jj=ij.T
        if len(ii)<max(32,cfg.min_duration_s/cfg.hop_s):continue
        ff=f[jj];tt=t[ii]
        lag=min(len(ff)-1,round(20/cfg.hop_s))
        drop=np.max((ff[:-lag]-ff[lag:])/ff[:-lag]) if lag>0 else 0
        if np.mean(abs(np.diff(ff))>.15)<.4 or np.mean(np.diff(ff)<-.15)<.2:continue
        if drop<.03 or np.mean(s[jj,ii])<18 or len(ii)/(ii[-1]-ii[0]+1)<.9:continue
        # Reject changing musical notes joined by a spurious linker.
        if np.mean(abs(np.diff(ff))>2)> .08:continue
        out.append(Track(tt.copy(),ff.copy(),amp[jj,ii].copy(),.7,None,kind='doppler'))
    return out


def detect_tracks(x,sr,cfg=DetectConfig()) -> list[Track]:
    """Whole-file whitening followed by harmonic CLEAN in 120 s windows.

    Blind search uses summed exponential-power log likelihoods, validated
    against reversed-time controls and an empirical Gumbel maximum model.
    Every in-band predicted harmonic of an accepted source is returned.
    The nominal 0.1 false source/hour calibration is not a universal guarantee.
    """
    x=np.asarray(x)
    if x.ndim==2:x=x.mean(axis=1)
    if x.ndim!=1 or sr<=0 or not 0<cfg.fmin<cfg.fmax or not np.isfinite(x).all():
        raise ValueError('expected finite mono/stereo audio, positive rate and frequency band')
    spec=_spectrogram(x,sr,cfg)
    if spec is None:return []
    t,f,amp,s=spec
    tracks=[];sources=[]
    # Persistent lines are recovered before whitening hides them.
    profile=np.median(amp,axis=1)
    floor=ndimage.median_filter(profile,size=31)
    peaks,_=signal.find_peaks(profile/np.maximum(floor,1e-15),height=4,distance=3)
    for j in peaks:
        if j==0 or j==len(f)-1:continue
        a,b,c=np.log(profile[j-1:j+2]+1e-15)
        off=np.clip(.5*(a-c)/(a-2*b+c+1e-15),-.5,.5)
        tracks.append(Track(t.copy(),np.full(len(t),f[j]+off*(f[1]-f[0])),amp[j].copy(),kind='stationary'))
    try:
        from .ridges import track_ridges
    except ImportError:
        track_ridges=None
    null_maxima=[]
    for start in np.arange(t[0],t[-1],60.):
        m=(t>=start)&(t<start+120)
        if m.sum()<48:continue
        tt=t[m];original=s[:,m];work=original.copy()
        candidates=_bank(tt,f,work)
        null=_search(tt,f,original[:,::-1],candidates,calibration=True)
        null_threshold=max(5., (null[0]+2.) if null is not None else 5.)
        null_maxima.append(null[0] if null is not None else 0.)
        if len(null_maxima)>=8 and np.std(null_maxima)>.05:
            # Foreground transients violate the IID null model. Fit the
            # central null population; reject a tail fit inconsistent with the
            # local reversed control rather than censoring the rest of a file.
            values=np.asarray(null_maxima)
            median=np.median(values);mad=np.median(abs(values-median))
            central=values[values<=median+3*max(mad,.05)]
            location,scale=stats.gumbel_r.fit(central)
            # 0.1 expected false source/hour, 60 windows/hour, 8 CLEAN trials.
            threshold=stats.gumbel_r.isf(.1/(60*8),loc=location,scale=scale)
            if np.isfinite(threshold) and threshold<=2*null_threshold:
                null_threshold=max(null_threshold,float(threshold))
        for iteration in range(8):
            found=_search(tt,f,work,candidates)
            if found is None or found[0]<null_threshold:break
            score,p=found
            # Probe subharmonics to recover the source's actual fundamental.
            for divisor in (3,2):
                sub=p.copy();sub[0]/=divisor
                curve=ideal_curve(tt,*sub)
                if np.min(curve)>=f[0] and np.mean(_brightness(original,f,curve))>3 and np.mean(_sample(original,f,curve)>0)>.75:
                    p=sub;break
            curve=ideal_curve(tt,*p)
            # Consolidate the same physical source across neighboring windows.
            source=None
            for item in sources:
                old=item['p']
                if abs(p[3]-old[3])<3:
                    ratio=np.median(curve/ideal_curve(tt,*old))
                    if min(abs(ratio-round(ratio)),abs(1/ratio-round(1/ratio)))<.025:
                        source=item;break
            if source is None:
                source={'p':p.copy(),'group':len(sources),'tracks':{}}
                sources.append(source)
            scale=p[0]/source['p'][0]
            if scale<.9 and abs(1/scale-round(1/scale))<.03:
                factor=round(1/scale)
                source['tracks']={h*factor:q for h,q in source['tracks'].items()}
                for h,q in source['tracks'].items():q.harmonic=h
                source['p']=p.copy();scale=1
            harmonic_base=max(1,round(scale))
            maximum_harmonic=int(cfg.fmax/max(np.min(curve),1))
            for k in range(1,maximum_harmonic+1):
                harmonic=k*harmonic_base
                ff=k*curve
                valid=(ff>=f[0])&(ff<=f[-1])
                if valid.sum()<8:continue
                # All predicted harmonics are removal targets, even unobserved.
                grid=np.arange(np.ceil(tt[0]*4),np.floor(tt[-1]*4)+1)/4
                freq=ideal_curve(grid,*(p*np.array([k,1,1,1])))
                in_band=(freq>=cfg.fmin)&(freq<=cfg.fmax)
                grid=grid[in_band];freq=freq[in_band]
                if len(grid)<8:continue
                amplitudes=np.interp(grid,tt,_sample(amp[:,m],f,ff))
                fit=dict(zip(('f0','v','d_cpa','t_cpa'),map(float,p*np.array([k,1,1,1]))),rms_hz=0.)
                tr=Track(grid,freq,amplitudes,min(.99,.5+score/20),fit,harmonic,source['group'],'doppler')
                old=source['tracks'].get(harmonic)
                if old is not None:
                    all_t=np.r_[old.t,tr.t];order=np.argsort(all_t,kind='stable')
                    unique,ix=np.unique(all_t[order],return_index=True)
                    old.f=np.r_[old.f,tr.f][order][ix];old.amp=np.r_[old.amp,tr.amp][order][ix];old.t=unique
                else:source['tracks'][harmonic]=tr
                # CLEAN all possible harmonics, including undetected ones.
                bins=np.rint((ff-f[0])/(f[1]-f[0])).astype(int)
                for offset in range(-3,4):
                    work[np.clip(bins+offset,0,len(f)-1),np.arange(len(tt))]=0
        vehicles=_ridge_vehicles(tt,f,amp[:,m],original,cfg) if track_ridges is None else []
        for tr in vehicles:
            if not any(np.mean(abs(tr.f-np.interp(tr.t,q.t,q.f)))<2 for src in sources for q in src['tracks'].values()):
                tr.group=len(sources)+len(tracks);tracks.append(tr)
    tracks.extend(q for src in sources for q in src['tracks'].values())
    if track_ridges is not None:
        ridge_tracks=track_ridges(s,t,f,cfg)
        offset=max((q.group for q in tracks),default=-1)+1
        for tr in ridge_tracks:
            tr.group+=offset
            if not np.any(tr.amp):
                m=(t>=tr.t0)&(t<=tr.t1)
                if m.any():
                    curve=np.interp(t[m],tr.t,tr.f)
                    tr.amp=np.interp(tr.t,t[m],_sample(amp[:,m],f,curve))
        tracks.extend(ridge_tracks)
    # Compare model-to-local-peak residuals on overlap when two methods agree.
    def error(q):
        if q.kind=='stationary':return 0.
        m=(t>=q.t0)&(t<=q.t1)
        if m.sum()<8:return np.inf
        curve=np.interp(t[m],q.t,q.f)
        shifts=np.arange(-3,4)*(f[1]-f[0])
        nearby=np.array([_sample(s[:,m],f,curve+d) for d in shifts])
        residual=shifts[np.argmax(nearby,axis=0)]
        return float(np.mean(residual**2) - np.mean(_brightness(s[:,m],f,curve)))
    unique=[]
    for q in sorted(tracks,key=error):
        duplicate=False
        for old in unique:
            if old.kind!=q.kind:continue
            m=(q.t>=old.t0)&(q.t<=old.t1)
            if m.sum()<8:continue
            if np.mean(abs(q.f[m]-np.interp(q.t[m],old.t,old.f))<1)>.7:
                duplicate=True
                # Keep the lower-error frequency estimates, extend its span.
                outside=(q.t<old.t0)|(q.t>old.t1)
                tt=np.r_[old.t,q.t[outside]];order=np.argsort(tt)
                old.f=np.r_[old.f,q.f[outside]][order];old.amp=np.r_[old.amp,q.amp[outside]][order];old.t=tt[order]
                break
        if not duplicate:unique.append(q)
    tracks=unique
    for tr in tracks:
        grid=np.arange(np.ceil(tr.t0*4),np.floor(tr.t1*4)+1)/4
        tr.f=np.interp(grid,tr.t,tr.f);tr.amp=np.interp(grid,tr.t,tr.amp);tr.t=grid
    for group in {q.group for q in tracks if q.kind=='doppler' and q.fit is not None}:
        members=[q for q in tracks if q.group==group and q.fit is not None]
        parent=min(members,key=error)
        before=parent.f.copy()
        left=max(0,int((parent.t0-1)*sr));right=min(len(x),int((parent.t1+1)*sr))
        parent.t-=left/sr
        _zoom_refine(parent,x[left:right],sr)
        parent.t+=left/sr
        for tr in members:
            if tr is not parent:
                tr.f*=np.interp(tr.t,parent.t,parent.f/np.maximum(before,1e-12),left=1.,right=1.)
    return sorted(tracks,key=lambda q:(q.group,q.harmonic,q.t0))


def main():
    import argparse
    import subprocess
    import json
    import time
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file')
    parser.add_argument('--window',nargs=2,type=float,action='append',metavar=('START','END'),help='plot a time range; may be repeated')
    parser.add_argument('--output',help='PNG output prefix (default: FILE.tracks)')
    parser.add_argument('--json',dest='json_path',help='write full Track arrays to JSON')
    args=parser.parse_args()
    started=time.monotonic()
    raw=subprocess.run(['ffmpeg','-v','error','-i',args.file,'-ac','1','-ar','4000','-f','f32le','-'],check=True,capture_output=True).stdout
    x=np.frombuffer(raw,dtype='<f4')
    tracks=detect_tracks(x,4000)
    elapsed=time.monotonic()-started
    for tr in tracks:
        print(f'{tr.kind:10} group={tr.group} harmonic={tr.harmonic} span={tr.t0:.2f}..{tr.t1:.2f}s frequency={tr.f[0]:.3f}..{tr.f[-1]:.3f}Hz score={tr.doppler_score:.3f} fit={tr.fit}')
    print(f'Runtime including decode: {elapsed:.2f}s')
    if args.json_path:
        Path(args.json_path).write_text(json.dumps({'runtime_s':elapsed,'tracks':[tr.to_dict() for tr in tracks]}))
    import matplotlib.pyplot as plt
    for lo,hi in args.window or [(0,len(x)/4000)]:
        left=max(0,int(lo*4000));right=min(len(x),int(hi*4000))
        spec=_spectrogram(x[left:right],4000,DetectConfig())
        if spec is None:continue
        t,f,_,s=spec;t+=left/4000
        fig,ax=plt.subplots(figsize=(14,6))
        ax.pcolormesh(t,f,s,vmin=-5,vmax=15,shading='auto',rasterized=True,cmap='magma')
        for tr in tracks:
            if tr.kind=='doppler' and tr.t1>=lo and tr.t0<=hi:
                ax.plot(tr.t,tr.f,lw=.7,color='cyan',alpha=.8)
        ax.set(xlabel='Time (s)',ylabel='Frequency (Hz)',xlim=(lo,hi),ylim=(30,1000),title=f'{Path(args.file).name}; {elapsed:.1f}s analysis')
        prefix=args.output or args.file+'.tracks'
        suffix=f'.{lo:g}-{hi:g}' if args.window else ''
        fig.savefig(prefix+suffix+'.png',dpi=140)
        plt.close(fig)


if __name__=='__main__':
    main()
