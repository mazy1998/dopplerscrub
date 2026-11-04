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


class _AnalysisSpectrum(np.ndarray):
    """Long-window matched-search view plus fused native-ridge observations."""
    def __new__(cls,long_view,ridge_view,ridge_amplitude):
        obj=np.asarray(long_view).view(cls)
        obj.ridge_view=ridge_view;obj.ridge_amplitude=ridge_amplitude
        return obj


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
    adaptive=amp.copy()
    # High orders smear rapidly in a two-second window. Keep the low-band
    # resolution, but measure higher bands with shorter centered windows.
    analysis_f=freq[band]
    for lower,upper,seconds in ((0,700,.5),(700,np.inf,.25)):
        rows=(analysis_f>lower)&(analysis_f<=upper)
        if not rows.any():continue
        short=max(hop+1,int(round(seconds*rate)))
        for first in range(0,count,512):
            stop=min(first+512,count)
            offset=first*hop+n//2-short//2
            segment=y[offset:offset+(stop-first-1)*hop+short]
            # Dense FFT sampling of the short window avoids the 2--4 Hz
            # staircase produced by interpolating coarse FFT magnitudes.
            _,_,z=signal.stft(segment,rate,nperseg=short,nfft=n,noverlap=short-hop,boundary=None,padded=False)
            adaptive[rows,first:stop]=2*np.abs(z[band][rows])
    # filtfilt applies the band filter twice; restore original signal units.
    _, response = signal.sosfreqz(sos, worN=freq[band], fs=rate)
    correction=np.maximum(np.abs(response[:,None])**2,.1)
    amp/=correction;adaptive/=correction
    def whiten(magnitude):
        db=20*np.log10(magnitude+1e-15)
        width=max(1,round(600/cfg.hop_s));centers=[];medians=[]
        for start in range(0,count,width):
            stop=min(count,start+width)
            centers.append((start+stop-1)/2);medians.append(np.median(db[:,start:stop],axis=1))
        medians=np.array(medians)
        for j in range(len(db)):db[j]-=np.interp(np.arange(count),centers,medians[:,j])
        return db
    long_view=whiten(amp);short_view=whiten(adaptive)
    # The matched search retains its original long-view null statistic.
    # Ridges need both views: short windows resolve low-band CPA chirps,
    # while long windows retain faint harmonics throughout the whole band.
    db=_AnalysisSpectrum(long_view,np.maximum(long_view,short_view),np.maximum(amp,adaptive))
    return (n/2+np.arange(count)*hop)/rate,freq[band],amp,db



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
    # Blind source fundamentals are searched in the attacker band; observed
    # higher orders are supplied by harmonic summation and explicit probes.
    base = np.arange(max(30., f[0]), min(450., f[-1])+1e-6, 1.)
    cols = np.arange(len(tt))[None, :]
    candidates = []
    for v in (20., 40., 65., 90., 120., 150.):
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


def _refine_curves(t,f,data,seeds,iterations=180):
    """Nelder-Mead simplices evaluated together, with no Python curve calls.

    Each seed retains its own simplex and convergence state. Batching changes
    evaluation cost rather than the search space or the acceptance statistic.
    """
    seeds=np.asarray(seeds,float)
    if not len(seeds):return seeds
    simplex=np.repeat(seeds[:,None,:],5,axis=1)
    for j in range(4):simplex[:,j+1,j]=np.where(seeds[:,j]!=0,seeds[:,j]*1.05,.00025)
    columns=np.arange(len(t))[None,:]
    def evaluate(parameters):
        shape=parameters.shape[:-1];p=parameters.reshape(-1,4)
        ok=(p[:,0]>max(40,f[0]))&(p[:,0]<f[-1])&(p[:,1]>5)&(p[:,1]<=150)&(p[:,2]>=50)&(p[:,2]<50000)&(p[:,3]+p[:,2]/_C>t[0]+5)&(p[:,3]+p[:,2]/_C<t[-1]-5)
        # Invalid simplex vertices never enter the retarded-time formula.
        result=np.full(len(p),100.)
        if ok.any():
            q=p[ok];curve=ideal_curve(t[None,:],*[q[:,j,None] for j in range(4)])
            bins=(curve-f[0])/(f[1]-f[0]);low=np.clip(np.floor(bins).astype(int),0,len(f)-2);weight=np.clip(bins-low,0,1)
            values=data[low,columns]*(1-weight)+data[low+1,columns]*weight
            result[ok]=np.where(np.all((curve>=f[0])&(curve<=f[-1]),axis=1),-np.mean(values,axis=1),100.)
        return result.reshape(shape)
    values=evaluate(simplex)
    for _ in range(iterations):
        order=np.argsort(values,axis=1);simplex=np.take_along_axis(simplex,order[:,:,None],axis=1);values=np.take_along_axis(values,order,axis=1)
        active=(np.max(abs(simplex[:,1:]-simplex[:,:1]),axis=(1,2))>.01)|(np.max(abs(values[:,1:]-values[:,:1]),axis=1)>1e-4)
        if not active.any():break
        ids=np.flatnonzero(active);x=simplex[ids];y=values[ids];center=x[:,:4].mean(axis=1)
        reflected=2*center-x[:,4];rv=evaluate(reflected)
        expanded=3*center-2*x[:,4];ev=evaluate(expanded)
        outside=rv<y[:,4]
        contracted=np.where(outside[:,None],1.5*center-.5*x[:,4],.5*(center+x[:,4]));cv=evaluate(contracted)
        improve=rv<y[:,0];ordinary=(rv>=y[:,0])&(rv<y[:,3])
        contract=(~improve)&(~ordinary)&np.where(outside,cv<=rv,cv<y[:,4])
        replacement=np.where((improve&(ev<rv))[:,None],expanded,reflected)
        replacement=np.where(contract[:,None],contracted,replacement)
        cost=np.where(improve&(ev<rv),ev,rv);cost=np.where(contract,cv,cost)
        accepted=improve|ordinary|contract
        simplex[ids[accepted],4]=replacement[accepted];values[ids[accepted],4]=cost[accepted]
        shrink=ids[~accepted]
        if len(shrink):
            simplex[shrink,1:]=.5*(simplex[shrink,1:]+simplex[shrink,:1]);values[shrink,1:]=evaluate(simplex[shrink,1:])
    return simplex[np.arange(len(seeds)),np.argmin(values,axis=1)]


def _search(t, f, work, candidates, calibration=False):
    likelihood=np.clip(_llr(work),-1,12)
    smooth = ndimage.gaussian_filter1d(likelihood, 1.2/(f[1]-f[0]),axis=0)
    # Re-score the cached bank in batches, including harmonic hypotheses.
    ranked=[]
    parameters=np.asarray([p for _,p in candidates])
    columns=np.arange(len(t))[None,:]
    for start in range(0,len(parameters),64):
        pp=parameters[start:start+64]
        curves=ideal_curve(t[None,:],*[pp[:,j,None] for j in range(4)])
        scores=[]
        for k in (1,2,3,4):
            curve=k*curves;bins=(curve-f[0])/(f[1]-f[0])
            low=np.clip(np.floor(bins).astype(int),0,len(f)-2);weight=np.clip(bins-low,0,1)
            values=smooth[low,columns]*(1-weight)+smooth[low+1,columns]*weight
            values=np.where((curve>=f[0])&(curve<=f[-1]),values,0.)
            scores.append(np.where(np.all(curve<=f[-1],axis=1),np.mean(values,axis=1),-10))
        scores=np.asarray(scores);harmonics=np.argmax(scores,axis=0)+1
        for j,p in enumerate(pp):
            q=p.copy();q[0]*=harmonics[j];ranked.append((scores[harmonics[j]-1,j],q))
    chosen=[]
    for score,p in sorted(ranked,key=lambda q:-q[0]):
        if all(abs(p[0]-q[0])>2 or abs(p[3]-q[3])>12 for q in chosen):
            chosen.append(p)
        if len(chosen)>=16:
            break
    best=None;raw_best=None
    chosen=_refine_curves(t,f,smooth,chosen)
    chosen=_refine_curves(t,f,likelihood,chosen)
    for p in chosen:
        ff=ideal_curve(t,*p)
        if p[2]<50 or p[1]>150 or p[0]<40 or np.ptp(ff)<max(1.,.015*np.median(ff)):
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


def _bright_ridges(t,f,amp,s,cfg):
    """Observed peak tracker with a constant-velocity predictor; no orders inferred."""
    band=f<=min(cfg.fmax,450.)
    f=f[band];amp=amp[band];s=s[band]
    df=f[1]-f[0];active=[];finished=[]
    raw=20*np.log10(amp+1e-15)
    tonal=raw-ndimage.median_filter(raw,size=(25,1),mode='nearest')
    # Time whitening hides nearly steady aircraft lines. Restore measured
    # narrowband contrast; truly steady lines still fail the movement test.
    s=np.maximum(s,tonal-4)
    shape=ndimage.gaussian_filter1d(np.maximum(s,0),.6/df,axis=0)
    high=f>=250
    shape[high]=ndimage.gaussian_filter1d(np.maximum(s,0),1.8/df,axis=0)[high]
    for col,time in enumerate(t):
        peaks,_=signal.find_peaks(shape[:,col],height=5,distance=2,prominence=1.5)
        if len(peaks)>100:peaks=peaks[np.argsort(s[peaks,col])[-100:]]
        if not len(peaks):
            for path in active:
                if col-path[-1][0]>5:finished.append(path)
            active=[path for path in active if col-path[-1][0]<=5]
            continue
        aa=shape[np.clip(peaks-1,0,len(f)-1),col]
        bb=shape[peaks,col]
        cc=shape[np.clip(peaks+1,0,len(f)-1),col]
        off=np.clip(.5*(aa-cc)/(aa-2*bb+cc+1e-15),-.5,.5)
        hz=f[peaks]+df*off
        offers=[]
        for i,path in enumerate(active):
            last=path[-1];gap=col-last[0]
            if gap>5:finished.append(path);continue
            recent=np.array(path[-8:])
            slope=0.
            if len(recent)>=3:
                slope=np.polyfit(t[recent[:,0].astype(int)]-t[last[0]],recent[:,2],1)[0]
            slope=np.clip(slope,-8,8)
            prediction=last[2]+slope*(time-t[last[0]])
            tolerance=.65+.25*abs(slope)+.25*(gap-1)
            for k in np.flatnonzero(abs(hz-prediction)<tolerance):
                cost=abs(hz[k]-prediction)/tolerance-.02*min(s[peaks[k],col],20)-.05*min(len(path),25)
                offers.append((cost,i,int(k)))
        used_paths=set();used_peaks=set()
        for _,i,k in sorted(offers):
            if i in used_paths or k in used_peaks:continue
            active[i].append((col,int(peaks[k]),float(hz[k])))
            used_paths.add(i);used_peaks.add(k)
        active=[path for path in active if col-path[-1][0]<=5]
        active.extend([[(col,int(peaks[k]),float(hz[k]))] for k in range(len(peaks)) if k not in used_peaks])
    finished.extend(active)
    output=[]
    for path in finished:
        points=np.array(path);ii=points[:,0].astype(int);jj=points[:,1].astype(int);ff=points[:,2]
        if len(ii)<24 or t[ii[-1]]-t[ii[0]]<max(8,cfg.min_duration_s):continue
        if len(ii)/(ii[-1]-ii[0]+1)<.72 or np.mean(s[jj,ii])<7:continue
        smooth=ndimage.gaussian_filter1d(ff,1.5)[::4]
        if np.ptp(ff)<max(cfg.min_sweep_hz,1.5,.02*np.median(ff)):continue
        if np.sum(abs(np.diff(smooth)))>2.5*max(np.ptp(smooth),.1):continue
        grid=np.arange(np.ceil(t[ii[0]]*4),np.floor(t[ii[-1]]*4)+1)/4
        frequency=np.interp(grid,t[ii],ff)
        fit=_fit(t[ii],ff)
        if fit['rms_hz']<1. and fit['v']<=150 and fit['d_cpa']>=50:
            pp=[fit[k] for k in ('f0','v','d_cpa','t_cpa')]
            frequency=ideal_curve(grid,*pp)
        amplitude=np.interp(grid,t[ii],amp[jj,ii])
        kind='doppler' if np.ptp(frequency)>=max(cfg.min_sweep_hz,.004*np.median(frequency)) else 'stationary'
        output.append(Track(grid,frequency,amplitude,.85 if kind=='doppler' else 0.,None,1,len(output),kind))
    return output


def _steady_lines(t,f,amp,cfg):
    """Report locally persistent lines, including tones flattened by whitening."""
    out=[];df=f[1]-f[0]
    for start in np.arange(t[0],t[-1],8.):
        left=np.searchsorted(t,start);right=np.searchsorted(t,start+16)
        if right-left<24:continue
        profile=np.median(amp[:,left:right],axis=1)
        floor=ndimage.median_filter(profile,size=31)
        peaks,_=signal.find_peaks(profile/np.maximum(floor,1e-15),height=4,distance=3)
        for peak in peaks:
            if not 85<=f[peak]<=220 or peak<2 or peak>=len(f)-2:continue
            cols=np.arange(left,right)
            candidates=np.arange(peak-2,peak+3)
            rows=candidates[np.argmax(amp[candidates[:,None],cols],axis=0)]
            a,b,c=np.log(amp[rows-1,cols]+1e-15),np.log(amp[rows,cols]+1e-15),np.log(amp[rows+1,cols]+1e-15)
            delta=np.clip(.5*(a-c)/(a-2*b+c+1e-15),-.5,.5)
            ff=ndimage.gaussian_filter1d(f[rows]+df*delta,2)
            tt=t[left:right]
            grid=np.arange(np.ceil(tt[0]*4),np.floor(tt[-1]*4)+1)/4
            frequency=np.interp(grid,tt,ff)
            amplitude=np.interp(grid,tt,amp[rows,cols])
            # Report small local drift; source classification requires the
            # longer moving-line tracker or confirmed harmonic family.
            out.append(Track(grid,frequency,amplitude,0.,None,1,len(out),'stationary'))
    return out


def _motion_kind(track,cfg,allow_maneuver=False):
    """Classify whole-track motion, independently of its detector provenance."""
    if len(track.t)<2 or track.t1-track.t0<cfg.min_duration_s:return None
    median=float(np.median(track.f));minimum=max(1.5,.01*median,cfg.min_sweep_hz)
    smooth=ndimage.gaussian_filter1d(track.f,2)
    excursion=float(np.ptp(smooth))
    if excursion<minimum:
        return 'stationary' if track.t1-track.t0>=5 and np.std(track.f)<.5 else None
    # A single deviant frame or staircase jump is not sustained source motion.
    robust=float(np.percentile(smooth,95)-np.percentile(smooth,5))
    if robust<minimum:return None
    # A held note followed by a discontinuous note change is not source
    # motion. This catches abrupt steps without penalizing bin quantization
    # in an otherwise smooth slow factor.
    steps=abs(np.diff(track.f))
    if np.mean(steps<max(.15,.0007*median))>.55 and np.max(steps)>max(2.,.02*median):return None
    # A verified source factor is classified once at its lowest order.
    # Higher orders inherit that shape; estimation jitter scales with k.
    if allow_maneuver=='verified':return 'doppler'
    # An uncorroborated rise/arc can be machinery drift or voiced pitch.
    # Straight passes must fall; non-monotonic maneuvers need a shared factor
    # observed in another harmonic, rather than a detector's inherited label.
    if not allow_maneuver and smooth[0]-smooth[-1]<.5*robust:return None
    coarse=smooth[::4]
    travel=float(np.sum(abs(np.diff(coarse))))
    if travel>(4. if allow_maneuver else 2.8)*robust:return None
    jump=np.max(abs(np.diff(smooth))) if len(smooth)>1 else 0
    if jump>max(1.,.12*robust):return None
    # Passes and smooth maneuvers have few substantial turns; noise-floor
    # walks and notes have repeated reversals or an isolated frequency step.
    prominence=max(.5,.15*minimum)
    turns=len(signal.find_peaks(coarse,prominence=prominence,distance=3)[0])+len(signal.find_peaks(-coarse,prominence=prominence,distance=3)[0])
    if turns>max(3 if allow_maneuver else 2,int((track.t1-track.t0)/(8 if allow_maneuver else 20))+1):return None
    return 'doppler'


def _motion_families(tracks):
    confirmed=set();groups={}
    for q in tracks:
        if q.kind=='doppler' and len(q.t)>=2:groups.setdefault(q.group,[]).append(q)
    for group,members in groups.items():
        for i,a in enumerate(members):
            for b in members[:i]:
                if a.harmonic==b.harmonic:continue
                lo=max(a.t0,b.t0);hi=min(a.t1,b.t1)
                if hi-lo<4:continue
                grid=np.arange(lo,hi,.25)
                error=np.interp(grid,a.t,a.f)-np.interp(grid,b.t,b.f)*a.harmonic/b.harmonic
                if np.sqrt(np.mean(error**2))<.5:
                    confirmed.add(group);break
            if group in confirmed:break
    return confirmed


def _consolidate_families(tracks,cfg,trusted_groups):
    """Unify source labels before expanding their shared harmonic factor.

    An independently found higher harmonic must not seed a second complete
    comb. Compare factors at the lower source frequency, so measurement error
    at high harmonic orders does not defeat source deduplication.
    """
    groups={}
    for q in tracks:
        if q.kind=='doppler' and len(q.t)>=2:groups.setdefault(q.group,[]).append(q)
    representatives=[]
    for group,members in groups.items():
        q=min(members,key=lambda q:(q.harmonic,-len(q.t)))
        if np.median(q.f)/q.harmonic>=40:representatives.append(q)
    canonical=[]
    for q in sorted(representatives,key=lambda q:np.median(q.f)/q.harmonic):
        matched=False
        for base in canonical:
            if q.group==base.group:matched=True;break
            lo=max(q.t0,base.t0);hi=min(q.t1,base.t1)
            if hi-lo<max(8.,.5*min(q.t1-q.t0,base.t1-base.t0)):continue
            tt=np.arange(lo,hi,.25)
            a=np.interp(tt,q.t,q.f)/q.harmonic;b=np.interp(tt,base.t,base.f)/base.harmonic
            k=round(float(np.median(a/b)))
            if k<1 or np.std(b)<.5:continue
            if np.sqrt(np.mean((a/k-b)**2))>.65 or np.corrcoef(a,b)[0,1]<.97:continue
            loser=q.group;winner=base.group
            if loser in trusted_groups:trusted_groups.add(winner)
            for member in tracks:
                if member.group==loser:member.group=winner;member.harmonic*=k
            matched=True;break
        if not matched:canonical.append(q)
    return tracks


def _dedupe_tracks(tracks,cfg,trusted_groups=None):
    """Merge coincident measurements across detector and source labels."""
    merged=[]
    bounds={id(q):(float(np.min(q.f)),float(np.max(q.f))) for q in tracks}
    for tr in sorted(tracks,key=lambda q:(q.fit is None,-(q.t1-q.t0))):
        match=None
        for old in merged:
            if old.kind!=tr.kind:continue
            a,b=bounds[id(tr)];c,d=bounds[id(old)]
            if a>d+1 or c>b+1:continue
            lo=max(tr.t0,old.t0);hi=min(tr.t1,old.t1)
            if hi-lo<max(2.,.5*min(tr.t1-tr.t0,old.t1-old.t0)):continue
            grid=np.arange(lo,hi+.01,.25)
            difference=np.interp(grid,tr.t,tr.f)-np.interp(grid,old.t,old.f)
            if np.sqrt(np.mean(difference**2))<.5 and np.mean(abs(difference)<1)>.9:
                match=old;break
        if match is None:merged.append(tr);continue
        if trusted_groups is not None and tr.group in trusted_groups:trusted_groups.add(match.group)
        if tr.group!=match.group and tr.harmonic==match.harmonic:
            loser=tr.group
            for member in tracks:
                if member.group==loser:member.group=match.group
        outside=(tr.t<match.t0)|(tr.t>match.t1)
        times=np.r_[match.t,tr.t[outside]];order=np.argsort(times)
        match.f=np.r_[match.f,tr.f[outside]][order];match.amp=np.r_[match.amp,tr.amp[outside]][order];match.t=times[order]
        bounds[id(match)]=(min(bounds[id(match)][0],bounds[id(tr)][0]),max(bounds[id(match)][1],bounds[id(tr)][1]))
        match.doppler_score=max(match.doppler_score,tr.doppler_score)
    return merged


def _probe_known_harmonics(tracks,t,f,amp,s,cfg):
    """Probe every in-band harmonic of an established sweep; no blind births."""
    found=[]
    # A confirmed ridge can have a short foreground occlusion. Probe its
    # continuous shared factor, rather than dropping harmonics in that gap.
    chains={}
    for tr in tracks:
        if tr.kind=='doppler' and tr.harmonic==1 and len(tr.t)>=2:
            chains.setdefault(tr.group,[]).append(tr)
    families={}
    for tr in tracks:
        if tr.kind=='doppler' and len(tr.t)>=2:families.setdefault(tr.group,[]).append(tr)
    family_confirmed=set()
    for group,members in families.items():
        for i,a in enumerate(members):
            for b in members[:i]:
                if a.harmonic==b.harmonic:continue
                start=max(a.t0,b.t0);end=min(a.t1,b.t1)
                if end-start<4:continue
                grid=np.arange(start,end,.25)
                error=np.interp(grid,a.t,a.f)-np.interp(grid,b.t,b.f)*a.harmonic/b.harmonic
                if np.sqrt(np.mean(error**2))<.5:
                    family_confirmed.add(group);break
            if group in family_confirmed:break
        if group not in chains and group in family_confirmed:
            lowest=min(members,key=lambda q:q.harmonic)
            root=lowest.f/lowest.harmonic
            if np.median(root)>=40:
                chains[group]=[Track(lowest.t.copy(),root,lowest.amp.copy(),lowest.doppler_score,None,1,group,'doppler')]
    parents=[]
    for group,pieces in chains.items():
        pieces=sorted(pieces,key=lambda q:q.t0)
        current=pieces[0]
        for nxt in pieces[1:]:
            gap=nxt.t0-current.t1
            if 0<=gap<=6 and abs(nxt.f[0]-current.f[-1])<=max(3,.03*np.median(current.f)):
                grid=np.arange(current.t0,nxt.t1+.01,.25)
                times=np.r_[current.t,nxt.t];order=np.argsort(times)
                current=Track(grid,np.interp(grid,times[order],np.r_[current.f,nxt.f][order]),np.interp(grid,times[order],np.r_[current.amp,nxt.amp][order]),max(current.doppler_score,nxt.doppler_score),None,1,group,'doppler')
            else:
                parents.append(current);current=nxt
        parents.append(current)
    for parent in parents:
        if parent.kind!='doppler' or parent.harmonic!=1 or len(parent.t)<2 or (parent.t1-parent.t0<(8 if parent.group in family_confirmed else 20)) or np.median(parent.f)<40:continue
        left=np.searchsorted(t,parent.t0);right=np.searchsorted(t,parent.t1,side='right')
        tt=t[left:right]
        if len(tt)<16:continue
        base=np.interp(tt,parent.t,parent.f)
        local=s[:,left:right]
        if parent.group not in family_confirmed:
            # An isolated ridge, especially a short bird/voice fundamental,
            # cannot seed a lattice. Confirm a sibling without searching a
            # frequency corridor or selecting only its bright frames.
            sibling=False
            for order in range(2,13):
                candidate=order*base;inside=(candidate>=f[0])&(candidate<=f[-1])
                if np.mean(inside)<.8:continue
                evidence=_sample(local,f,candidate)[inside]
                if np.median(evidence)>=5 and np.mean(evidence>=5)>=.65:
                    sibling=True;break
            if not sibling:continue
        for k in range(2,int(cfg.fmax/max(np.min(base),1))+1):
            curve=k*base
            values=_sample(local,f,curve)
            contrast=_brightness(local,f,curve)
            # Test an unbiased whole-parent statistic before selecting bright
            # islands; conditioning on positive noise alone invents siblings.
            if np.mean(contrast)<.6:continue
            support=(ndimage.uniform_filter1d(values,size=7,mode='nearest')>=0.)&(ndimage.uniform_filter1d(contrast,size=7,mode='nearest')>=.5)
            support&=(curve>=cfg.fmin)&(curve<=cfg.fmax)
            # An observed harmonic confirms its order, not only a collection
            # of disconnected islands. Follow the parent's continuous factor.
            support=ndimage.binary_closing(support,structure=np.ones(3))
            evidence_indices=np.flatnonzero(support)
            runs=np.split(evidence_indices,np.flatnonzero(np.diff(evidence_indices)>1)+1)
            observed=any(len(run)>=2 and tt[run[-1]]-tt[run[0]]>=cfg.min_duration_s and np.mean(values[run])>=.75 and np.mean(contrast[run])>=1. for run in runs)
            if not observed:continue
            valid=(curve>=cfg.fmin)&(curve<=cfg.fmax)
            for part in np.split(np.flatnonzero(valid),np.flatnonzero(np.diff(np.flatnonzero(valid))>1)+1):
                if len(part)<2:continue
                grid=tt[part];frequency=curve[part]
                amplitude=_sample(amp[:,left:right],f,curve)[part]
                if any(q.harmonic==k and q.group==parent.group and q.t0<=grid[0] and q.t1>=grid[-1] for q in found):continue
                found.append(Track(grid.copy(),frequency.copy(),amplitude.copy(),max(cfg.doppler_threshold,parent.doppler_score),None,k,parent.group,'doppler'))
    return found


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
        if m.sum()<max(48,int(30/cfg.hop_s)):continue
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
                if sub[0]>=40 and np.min(curve)>=f[0] and np.mean(_brightness(original,f,curve))>3 and np.mean(_sample(original,f,curve)>0)>.75:
                    p=sub;break
            curve=ideal_curve(tt,*p)
            measurements=[]
            audio_left=max(0,int((tt[0]-2)*sr));audio_right=min(len(x),int((tt[-1]+2)*sr))
            for k in range(1,5):
                ff=k*curve
                if np.min(ff)<f[0] or np.max(ff)>f[-1]:continue
                brightness=_sample(original,f,ff)
                strong=brightness>5
                if strong.sum()<32 or np.mean(brightness)<2:continue
                measured=Track(tt.copy()-audio_left/sr,ff.copy(),np.zeros(len(tt)))
                _zoom_refine(measured,x[audio_left:audio_right],sr)
                measurements.append((k,measured.f,strong))
            confirmed=False
            for i,(k,frequency,mask) in enumerate(measurements):
                for h,other,other_mask in measurements[:i]:
                    common=mask&other_mask
                    if common.sum()<24:continue
                    mismatch=frequency[common]-other[common]*k/h
                    if np.sqrt(np.mean(mismatch**2))<.5:
                        bank=np.stack([frequency/k,other/h])
                        observed_curve=np.mean(bank,axis=0)
                        fit=_fit(tt[common],observed_curve[common])
                        if fit['rms_hz']<.5 and fit['v']<=150 and fit['d_cpa']>=50 and fit['f0']>=40:
                            p=np.array([fit[key] for key in ('f0','v','d_cpa','t_cpa')])
                            curve=ideal_curve(tt,*p)
                        else:
                            curve=observed_curve
                        confirmed=True;break
                if confirmed:break
            if not confirmed:
                bins=np.rint((curve-f[0])/(f[1]-f[0])).astype(int)
                for delta in range(-3,4):work[np.clip(bins+delta,0,len(f)-1),np.arange(len(tt))]=0
                continue
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
                freq=np.interp(grid,tt,k*curve)
                in_band=(freq>=cfg.fmin)&(freq<=cfg.fmax)
                grid=grid[in_band];freq=freq[in_band]
                if len(grid)<8:continue
                amplitudes=np.interp(grid,tt,_sample(amp[:,m],f,ff))
                shifts=np.arange(-3,4)*(f[1]-f[0])
                nearby=np.array([_sample(original,f,ff+delta) for delta in shifts])
                residual=shifts[np.argmax(nearby,axis=0)]
                measured=valid & (_sample(original,f,ff)>3)
                rms=float(np.sqrt(np.mean(residual[measured]**2))) if measured.any() else None
                fit=dict(zip(('f0','v','d_cpa','t_cpa'),map(float,p*np.array([k,1,1,1]))),rms_hz=rms)
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
    tracks.extend(q for src in sources for q in src['tracks'].values() if len(q.t)>=2)
    stationary=_steady_lines(t,f,amp,cfg)
    stationary_offset=max((q.group for q in tracks),default=-1)+1
    for tr in stationary:tr.group+=stationary_offset
    tracks.extend(stationary)
    bright=_bright_ridges(t,f,amp,s,cfg)
    bright_offset=max((q.group for q in tracks),default=-1)+1
    for tr in bright:tr.group+=bright_offset
    tracks.extend(bright)
    trusted_groups=set()
    if track_ridges is not None:
        # Native ridges already verify multiresolution peaks and harmonic
        # families. Rechecking them against a narrow long-window peak loses
        # steep maneuvers and breaks continuous harmonic proposals.
        try:
            from .ridges import RidgeSpectrum
        except ImportError:
            ridge_s=getattr(s,'ridge_view',s)
        else:
            ridge_s=RidgeSpectrum(getattr(s,'ridge_view',s),getattr(s,'ridge_amplitude',amp))
        ridge_tracks=track_ridges(ridge_s,t,f,cfg)
        ridge_tracks=[q for q in ridge_tracks if q.kind!='doppler' or q.doppler_score>=cfg.doppler_threshold]
        offset=max((q.group for q in tracks),default=-1)+1
        ridge_tracks=[q for q in ridge_tracks if len(q.t)>=2 and len(q.f)==len(q.t) and len(q.amp)==len(q.t)]
        native_families={}
        for q in ridge_tracks:
            if q.kind=='doppler':native_families.setdefault(q.group,[]).append(q)
        smooth_groups=set()
        for group,members in native_families.items():
            parent=min(members,key=lambda q:(q.harmonic,-len(q.t)))
            # The producer has already checked trajectory smoothness and
            # source coherence. Only enforce whole-track motion here.
            if _motion_kind(parent,cfg,'verified')=='doppler':smooth_groups.add(group+offset)
        accepted_ridges=[]
        for tr in ridge_tracks:
            # Preserve measured harmonic orders even for sub-40Hz families.
            # Relabeling each observed order as h1 would create new comb seeds.
            tr.group+=offset
            if tr.kind=='doppler' and tr.group in smooth_groups and np.any(amp):trusted_groups.add(tr.group)
            if not np.any(tr.amp):
                m=(t>=tr.t0)&(t<=tr.t1)
                if m.any():
                    curve=np.interp(t[m],tr.t,tr.f)
                    tr.amp=np.interp(tr.t,t[m],_sample(amp[:,m],f,curve))
            accepted_ridges.append(tr)
        tracks.extend(accepted_ridges)
    tracks=[q for q in tracks if len(q.t)>=2 and len(q.f)==len(q.t) and len(q.amp)==len(q.t)]
    # Measured nonparametric spans are final. Extending their nearest peak
    # can latch onto a stationary neighbor before or after the actual sweep.
    # Compare model-to-local-peak residuals on overlap when two methods agree.
    def error(q):
        if q.kind=='stationary':return 0.
        left=int(np.searchsorted(t,q.t0));right=int(np.searchsorted(t,q.t1,side='right'))
        if right-left<8:return np.inf
        curve=np.interp(t[left:right],q.t,q.f)
        low=max(0,int(np.searchsorted(f,np.min(curve)-7)))
        high=min(len(f),int(np.searchsorted(f,np.max(curve)+7))+1)
        if high-low<2:return np.inf
        observations=getattr(s,'ridge_view',s) if q.group in trusted_groups else s
        local=observations[low:high,left:right];frequencies=f[low:high]
        shifts=np.arange(-3,4)*(f[1]-f[0])
        nearby=np.array([_sample(local,frequencies,curve+d) for d in shifts])
        residual=shifts[np.argmax(nearby,axis=0)]
        return float(np.mean(residual**2) - np.mean(_brightness(local,frequencies,curve)))
    unique=[]
    for q in sorted(tracks,key=error):
        duplicate=False
        for old in unique:
            if old.kind!=q.kind:continue
            m=(q.t>=old.t0)&(q.t<=old.t1)
            if m.sum()<8:
                first,second=(old,q) if old.t1<q.t0 else (q,old)
                gap=second.t0-first.t1
                if old.fit is None and q.fit is None and 0<gap<=1.5:
                    slope=np.polyfit(first.t[-min(8,len(first.t)):]-first.t1,first.f[-min(8,len(first.t)):],1)[0]
                    if abs(first.f[-1]+slope*gap-second.f[0])<.8:
                        tt=np.r_[old.t,q.t];order=np.argsort(tt)
                        old.f=np.r_[old.f,q.f][order];old.amp=np.r_[old.amp,q.amp][order];old.t=tt[order]
                        duplicate=True;break
                continue
            residual=q.f[m]-np.interp(q.t[m],old.t,old.f)
            if np.sqrt(np.mean(residual**2))<.5 and np.mean(abs(residual)<1)>.9:
                duplicate=True
                if q.group in trusted_groups:
                    trusted_groups.add(old.group)
                    old.group=q.group;old.harmonic=q.harmonic
                if old.fit is None and q.fit is not None:
                    old.group=q.group;old.harmonic=q.harmonic
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
    tracks=[q for q in tracks if len(q.t)>=2]
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
    motion_families=_motion_families(tracks)|trusted_groups
    classified=[]
    for tr in tracks:
        kind=_motion_kind(tr,cfg,'verified' if tr.group in trusted_groups else tr.group in motion_families)
        if kind is None:continue
        tr.kind=kind
        if kind=='stationary':tr.doppler_score=0.
        classified.append(tr)
    tracks=_consolidate_families(classified,cfg,trusted_groups)
    tracks=_dedupe_tracks(tracks,cfg,trusted_groups)
    # Label only harmonics directly confirmed by frequency agreement.
    ordered=sorted([q for q in tracks if q.kind=='doppler'],key=lambda q:np.median(q.f))
    for i,q in enumerate(ordered):
        if q.fit is not None:continue
        for base in ordered[:i]:
            if np.median(base.f)<40:continue
            lo=max(q.t0,base.t0);hi=min(q.t1,base.t1)
            if hi-lo<20:continue
            grid=np.arange(lo,hi,.25)
            bf=np.interp(grid,base.t,base.f);qf=np.interp(grid,q.t,q.f)
            k=round(float(np.median(qf/bf)))
            if 2<=k<=int(cfg.fmax/max(np.min(bf),1)) and np.sqrt(np.mean((qf-k*bf)**2))<.5 and np.corrcoef(qf,bf)[0,1]>.95:
                if q.group in trusted_groups:
                    trusted_groups.add(base.group);motion_families.add(base.group)
                q.group=base.group;q.harmonic=k*base.harmonic
                break
    probes=_probe_known_harmonics(tracks,t,f,amp,s,cfg)
    tracks.extend(probes)
    probe_ids={id(q) for q in probes}
    confirmed_groups={source['group'] for source in sources}
    supported_groups=trusted_groups|_motion_families(tracks)|{q.group for q in tracks if q.kind=='doppler' and q.t1-q.t0>=20}
    final=[]
    for tr in tracks:
        if len(tr.t)<2 or len(tr.f)!=len(tr.t) or len(tr.amp)!=len(tr.t):continue
        valid=np.isfinite(tr.t)&np.isfinite(tr.f)&np.isfinite(tr.amp)&(tr.f>=cfg.fmin)&(tr.f<=cfg.fmax)
        if tr.kind=='stationary':
            columns=np.clip(np.searchsorted(t,tr.t),0,len(t)-1)
            bins=(tr.f-f[0])/(f[1]-f[0])
            low=np.clip(np.floor(bins).astype(int),0,len(f)-2);w=np.clip(bins-low,0,1)
            brightness=s[low,columns]*(1-w)+s[low+1,columns]*w
            valid&=ndimage.uniform_filter1d(brightness,size=7,mode='nearest')>=3
        if tr.kind=='doppler' and tr.group not in trusted_groups and id(tr) not in probe_ids:
            # Predictions outside the attacker band require independent local
            # evidence in 1.5 s neighborhoods. Never bridge unsupported gaps.
            indices=np.clip(np.searchsorted(t,tr.t),0,len(t)-1)
            bins=(tr.f-f[0])/(f[1]-f[0])
            lo=np.clip(np.floor(bins).astype(int),0,len(f)-2)
            weight=np.clip(bins-lo,0,1)
            brightness=s[lo,indices]*(1-weight)+s[lo+1,indices]*weight
            local=ndimage.uniform_filter1d(brightness,size=7,mode='nearest')
            verified=(local>=3)&(ndimage.uniform_filter1d((brightness>0).astype(float),size=7,mode='nearest')>=.65)
            if id(tr) in probe_ids:
                verified|=(local>=0.)
            valid&=(tr.f<=450)|verified
            if tr.harmonic>10:valid&=verified
            if tr.fit is not None and tr.group in confirmed_groups:
                # Limit even a confirmed asymptote using geometric spreading
                # calibrated to observed excess power, not Track.amp guesses.
                center=np.clip(np.rint(bins).astype(int),0,len(f)-1)
                offsets=np.array([-16,-12,-8,8,12,16])[:,None]
                noise=np.median(amp[np.clip(center[None,:]+offsets,0,len(f)-1),indices]**2,axis=0)
                measured_amp=amp[lo,indices]*(1-weight)+amp[lo+1,indices]*weight
                speed=tr.fit['v'];distance=tr.fit['d_cpa'];u=tr.t-tr.fit['t_cpa']
                denominator=_C**2-speed**2
                emission=(_C**2*u-np.sqrt(_C**2*speed**2*u**2+denominator*distance**2))/denominator
                gain=distance/np.hypot(distance,speed*emission)
                observed=(brightness>=3)&(gain>.15)
                excess=np.maximum(measured_amp**2-noise,0)/np.maximum(gain**2,1e-9)
                reference=np.median(excess[observed]) if observed.any() else 0.
                valid&=verified|(reference*gain**2>=noise)
            if tr.fit is None or tr.group not in confirmed_groups:
                center=np.clip(np.rint(bins).astype(int),0,len(f)-1)
                offsets=np.array([-16,-12,-8,8,12,16])[:,None]
                flanks=amp[np.clip(center[None,:]+offsets,0,len(f)-1),indices]
                measured_amp=amp[lo,indices]*(1-weight)+amp[lo+1,indices]*weight
                tonal=20*np.log10((measured_amp+1e-15)/(np.median(flanks,axis=0)+1e-15))
                raw_verified=ndimage.uniform_filter1d(tonal,size=7,mode='nearest')>=4
                valid&=verified|raw_verified
        indices=np.flatnonzero(valid)
        for part in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            minimum_duration=cfg.min_duration_s
            if len(part)<2 or tr.t[part[-1]]-tr.t[part[0]]<minimum_duration:continue
            piece=Track(tr.t[part],tr.f[part],tr.amp[part],tr.doppler_score,tr.fit,tr.harmonic,tr.group,tr.kind)
            kind=_motion_kind(piece,cfg,'verified' if tr.group in trusted_groups else tr.group in motion_families)
            if kind is None:continue
            piece.kind=kind
            if kind=='stationary':
                piece.doppler_score=0.
                if tr.t[part[-1]]-tr.t[part[0]]<5 or np.std(tr.f[part])>=.5:continue
                columns=np.clip(np.searchsorted(t,tr.t[part]),0,len(t)-1)
                bins=(tr.f[part]-f[0])/(f[1]-f[0])
                low=np.clip(np.floor(bins).astype(int),0,len(f)-2);w=np.clip(bins-low,0,1)
                snr=s[low,columns]*(1-w)+s[low+1,columns]*w
                if np.mean(snr)<3:continue
            final.append(piece)
    merged=[]
    for tr in sorted(_dedupe_tracks(final,cfg,trusted_groups),key=lambda q:(q.group,q.harmonic,q.kind,q.t0)):
        previous=merged[-1] if merged else None
        if previous is not None and (tr.group,tr.harmonic,tr.kind)==(previous.group,previous.harmonic,previous.kind) and tr.t0<=previous.t1+.26:
            lo=max(previous.t0,tr.t0);hi=min(previous.t1,tr.t1)
            sample=np.arange(lo,hi+.01,.25)
            match=(len(sample)>0 and np.max(abs(np.interp(sample,previous.t,previous.f)-np.interp(sample,tr.t,tr.f)))<.5)
            if match:
                times=np.r_[previous.t,tr.t];order=np.argsort(times,kind='stable');grid,keep=np.unique(times[order],return_index=True)
                previous.f=np.r_[previous.f,tr.f][order][keep];previous.amp=np.r_[previous.amp,tr.amp][order][keep];previous.t=grid
                previous.doppler_score=max(previous.doppler_score,tr.doppler_score)
                continue
        merged.append(tr)
    # Merging independently steady pieces must not create a staircase.
    kept=[]
    for q in merged:
        if len(q.t)<2:continue
        kind=_motion_kind(q,cfg,'verified' if q.group in trusted_groups else q.group in motion_families)
        if kind is None:continue
        q.kind=kind
        if kind=='doppler' and q.group not in supported_groups:continue
        if q.kind=='stationary':
            q.doppler_score=0.
            if q.t1-q.t0<5 or np.std(q.f)>=.5:continue
            columns=np.clip(np.searchsorted(t,q.t),0,len(t)-1)
            bins=(q.f-f[0])/(f[1]-f[0]);low=np.clip(np.floor(bins).astype(int),0,len(f)-2);w=np.clip(bins-low,0,1)
            if np.mean(s[low,columns]*(1-w)+s[low+1,columns]*w)<3:continue
        if q.kind=='doppler':q.doppler_score=max(cfg.doppler_threshold,q.doppler_score)
        kept.append(q)
    return sorted(kept,key=lambda q:(q.group,q.harmonic,q.t0))


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
    decode_sr=max(DetectConfig().work_sr,int(np.ceil(2.2*DetectConfig().fmax)))
    started=time.monotonic()
    raw=subprocess.run(['ffmpeg','-v','error','-i',args.file,'-ac','1','-ar',str(decode_sr),'-f','f32le','-'],check=True,capture_output=True).stdout
    x=np.frombuffer(raw,dtype='<f4')
    tracks=detect_tracks(x,decode_sr)
    elapsed=time.monotonic()-started
    for tr in tracks:
        print(f'{tr.kind:10} group={tr.group} harmonic={tr.harmonic} span={tr.t0:.2f}..{tr.t1:.2f}s frequency={tr.f[0]:.3f}..{tr.f[-1]:.3f}Hz score={tr.doppler_score:.3f} fit={tr.fit}')
    print(f'Runtime including decode: {elapsed:.2f}s')
    if args.json_path:
        Path(args.json_path).write_text(json.dumps({'runtime_s':elapsed,'tracks':[tr.to_dict() for tr in tracks]}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for lo,hi in args.window or [(0,len(x)/decode_sr)]:
        left=max(0,int(lo*decode_sr));right=min(len(x),int(hi*decode_sr))
        spec=_spectrogram(x[left:right],decode_sr,DetectConfig())
        if spec is None:continue
        t,f,_,s=spec;t+=left/decode_sr
        fig,ax=plt.subplots(figsize=(14,6))
        ax.pcolormesh(t,f,s,vmin=-5,vmax=15,shading='auto',rasterized=True,cmap='magma')
        for tr in tracks:
            if tr.kind=='doppler' and tr.t1>=lo and tr.t0<=hi:
                ax.plot(tr.t,tr.f,lw=.7,color='cyan',alpha=.8)
        ax.set(xlabel='Time (s)',ylabel='Frequency (Hz)',xlim=(lo,hi),ylim=(DetectConfig().fmin,DetectConfig().fmax),title=f'{Path(args.file).name}; {elapsed:.1f}s analysis')
        prefix=args.output or args.file+'.tracks'
        suffix=f'.{lo:g}-{hi:g}' if args.window else ''
        fig.savefig(prefix+suffix+'.png',dpi=140)
        plt.close(fig)


if __name__=='__main__':
    main()
