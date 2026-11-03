"""Non-parametric frequency-line tracking for maneuvering harmonic sources.

The spectrogram result is an ndarray subclass carrying pre-whitening peak
magnitudes in ``.amplitude``. Preserve that object when calling track_ridges;
plain external dB arrays are accepted, but their Track.amp is zero (unknown).
No ideal-pass fit or attack-library dependency is used here. Physics prior:
120 m/s maximum speed, 200 m minimum slant range, c=343 m/s; transitions
are truncated at three standard deviations. A Hann periodogram is used for
calibration; multitaper whitening is left for future variance reduction.
"""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np
from scipy import signal, ndimage
from .types import DetectConfig, Track


class RidgeSpectrum(np.ndarray):
    """Whitened dB array with aligned, calibrated sinusoidal peak magnitudes."""
    def __new__(cls, values, amplitude):
        obj = np.asarray(values, dtype=np.float32).view(cls)
        obj.amplitude = np.asarray(amplitude, dtype=np.float32)
        return obj

    def __array_finalize__(self, obj):
        self.amplitude = getattr(obj, 'amplitude', None)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, RidgeSpectrum) and self.amplitude is not None:
            result.amplitude = self.amplitude[key]
        return result


def ridge_spectrogram(x, sr, cfg):
    """Return (S, t, f), mono, 0.25 s frames, multiresolution STFT.

    Low band uses >=2 s Hann windows (~0.5 Hz physical resolution). Above
    250/700 Hz windows shorten to 0.5/0.25 s to follow rapid passes. Higher
    bands are interpolated onto the common low-band grid; interpolation does
    not increase their physical resolution. Frame times are sample-centered;
    zero padding includes both clip edges. Native signal units are retained.
    """
    x = np.asarray(x)
    if sr <= 0 or x.ndim not in (1, 2) or not np.isfinite(x).all():
        raise ValueError('finite mono or samples-by-channels audio and positive sr required')
    if not 0 < cfg.fmin < cfg.fmax or cfg.fmax >= sr/2:
        raise ValueError('require 0 < fmin < fmax < input Nyquist')
    y = np.mean(x, axis=1) if x.ndim == 2 else x
    rate = max(int(cfg.work_sr), int(math.ceil(cfg.fmax*2.4)))
    g = math.gcd(int(sr), rate)
    if len(y):
        y = signal.resample_poly(y, rate//g, int(sr)//g)
    nlong = max(int(cfg.nperseg), 2*rate)
    df = rate/nlong
    f = np.arange(math.ceil(cfg.fmin/df), math.floor(cfg.fmax/df)+1)*df
    t = np.arange(0, len(y)/rate, .25)
    if not len(t):
        return RidgeSpectrum(np.empty((len(f), 0)), np.empty((len(f), 0))), t, f
    amp = np.zeros((len(f), len(t)), np.float32)
    for lo, hi, n in [(0, 250, nlong), (250, 700, round(.5*rate)),
                       (700, np.inf, round(.25*rate))]:
        band = (f > lo) & (f <= hi)
        if not band.any():
            continue
        window = signal.windows.hann(n, sym=False)
        ff = np.fft.rfftfreq(n, 1/rate)
        padded = np.pad(y, (n//2, n))
        for j, time in enumerate(t):
            start = round(time*rate)
            # FFT is unnormalized: 2|Z| / sum(window) gives sinusoid peak.
            a = 2*np.abs(np.fft.rfft(padded[start:start+n]*window))/window.sum()
            amp[band, j] = np.interp(f[band], ff, a)
    db = 20*np.log10(amp+1e-12)
    # Long blocks preserve maneuvering lines; blend medians to avoid seams.
    width = 2400  # 600 s at fixed 0.25 s hop
    centers = []; medians = []
    for start in range(0, len(t), width):
        stop = min(start+width, len(t))
        centers.append((start+stop-1)/2)
        medians.append(np.median(db[:, start:stop], axis=1))
    medians = np.array(medians)
    baseline = np.stack([np.interp(np.arange(len(t)), centers, medians[:, k])
                         for k in range(len(f))])
    return RidgeSpectrum(db-baseline, amp), t, f


def _evidence(S, f, amplitude):
    """Remove broadband foreground while preserving stationary raw ridges."""
    result = np.empty_like(S, dtype=np.float32)
    for lo, hi, width in [(0,250,18.), (250,700,40.), (700,np.inf,80.)]:
        mask = (f > lo) & (f <= hi)
        bins = max(5, int(round(width/np.median(np.diff(f)))) | 1)
        local = ndimage.median_filter(S, size=(bins,1), mode='nearest')
        result[mask] = S[mask]-local[mask]
    if amplitude is not None:
        raw = 20*np.log10(amplitude+1e-12)
        tonal = raw-ndimage.median_filter(raw, size=(25,1), mode='nearest')
        # Persistent raw lines vanished under time whitening. Reintroduce only
        # bins with a strong time-median spectral peak, e.g. hum or chords.
        persistent = np.median(tonal, axis=1) > 12
        result[persistent] = np.maximum(result[persistent], tonal[persistent])
    if amplitude is not None:
        # Ignore floating-point/codec residue >100 dB below the loudest peak.
        result[amplitude < max(float(np.max(amplitude))*1e-5, 1e-12)] = -20
    return np.clip(np.nan_to_num(result, nan=-20, posinf=30, neginf=-20), -20, 30)


def _llr(db, snr=3.):
    # Median-whitened periodogram has median 1; ln(2) restores Exp(1) mean.
    z = np.log(2)*10**(np.clip(db,-30,20)/10)
    return np.minimum(z*snr/(1+snr)-np.log1p(snr), 12.)


def _viterbi_lines(evidence, t, f, threshold, limit=70, observation=None, harmonic_suppression=False):
    """Sparse-bin Viterbi with zero-score births, deaths, Gaussian transitions.

    Local maxima are the frequency states (at most 96/frame), rather than
    spending quadratic work on bins with no ridge evidence. Paths compete
    through iterative corridor suppression and are split at unsupported gaps.
    """
    F,T = evidence.shape
    if T < 3 or F < 3:
        return []
    dt = float(np.median(np.diff(t)))
    candidates=[]
    for j in range(T):
        peaks = signal.find_peaks(evidence[:,j], height=max(1., threshold*.6), distance=2)[0]
        if len(peaks)>96:
            peaks=peaks[np.argsort(evidence[peaks,j])[-96:]]
        candidates.append(peaks)
    N=max(map(len,candidates),default=0)
    if N==0:
        return []
    idx=np.zeros((T,N),int);valid=np.zeros((T,N),bool)
    for j,p in enumerate(candidates):
        idx[j,:len(p)]=p;valid[j,:len(p)]=True
    hz=f[idx]
    llr = _llr(evidence) if observation is None else observation
    emission=llr[idx,np.arange(T)[:,None]].astype(float)
    emission[~valid]=-1e6
    transitions=[]
    for j in range(1,T):
        delta=hz[j,:,None]-hz[j-1,None,:]
        center=(hz[j,:,None]+hz[j-1,None,:])/2
        # Physics: sigma_Hz = f*v_max^2/(c*d_min)*hop.
        # 120 m/s and 200 m permit rapid low passes without supersonic jumps.
        sigma=np.maximum(np.median(np.diff(f)),center*120.**2/(343.*200.)*dt)
        cost=.5*(delta/sigma)**2 + np.log(np.sqrt(2*np.pi)*sigma/np.median(np.diff(f)))
        cost[np.abs(delta)>3*sigma]=1e6
        transitions.append(cost)
    paths=[]
    for _ in range(limit):
        scores=np.zeros((T,N));back=np.full((T,N),-1,np.int16)
        # Explicit null state: score 0, null->line uniform-bin birth cost log(F);
        # line->null death cost 0. Carrying score <=0 selects null.
        birth=math.log(max(F,3))
        scores[0]=np.maximum(0,emission[0]-birth)
        for j in range(1,T):
            previous=scores[j-1][None,:]-transitions[j-1]
            best=previous.argmax(axis=1);value=previous[np.arange(N),best]
            continuing=value>-birth
            scores[j]=np.maximum(0,np.maximum(value,-birth)+emission[j])
            back[j,continuing]=best[continuing]
        end,state=np.unravel_index(np.argmax(scores),scores.shape)
        if scores[end,state] < 24.:
            break
        points=[]
        j=int(end);k=int(state)
        while j>=0 and scores[j,k]>0:
            points.append((j,int(idx[j,k])))
            old=int(back[j,k]);j-=1
            if old<0: break
            k=old
        points.reverse()
        if not points: break
        times=np.array([p[0] for p in points]);bins=np.array([p[1] for p in points])
        # Suppress roughly one resolution element, wider in the short-window bands.
        for frame,bin_ in points:
            corridor=max(1., 2. if f[bin_]<=250 else 5. if f[bin_]<=700 else 9.)
            emission[frame,np.abs(hz[frame]-f[bin_])<=corridor]=-1e6
            if harmonic_suppression:
                for ratio in (.25,1/3,.5,2/3,.75,1.5,2.,3.,4.):
                    emission[frame,np.abs(hz[frame]-ratio*f[bin_])<=max(corridor,.008*ratio*f[bin_])]=-1e6
        support=evidence[bins,times]>=threshold
        jump=np.abs(np.diff(f[bins]))
        allowed=np.maximum(3.,.06*(f[bins[1:]]+f[bins[:-1]])/2)
        cuts=np.flatnonzero((np.diff(times)>1)|(jump>allowed))+1
        for segment in np.split(np.arange(len(times)),cuts):
            good=segment[support[segment]]
            if len(good)<3: continue
            a,b=good[0],good[-1]+1
            if t[times[b-1]]-t[times[a]]>=1.75:
                paths.append((times[a:b], bins[a:b]))
    return paths


def _refine_path(bins, frames, evidence, f):
    lo=np.maximum(bins-1,0);hi=np.minimum(bins+1,len(f)-1)
    a=evidence[lo,frames];b=evidence[bins,frames];c=evidence[hi,frames]
    den=a-2*b+c
    shift=np.divide(.5*(a-c),den,out=np.zeros_like(b),where=np.abs(den)>1e-6)
    return f[bins]+np.clip(shift,-.5,.5)*np.median(np.diff(f))


def _moving(freq, cfg):
    return np.ptp(freq)>=max(cfg.min_sweep_hz,.005*np.median(freq))


def _speech_like(tt, ff):
    # A sustained voiced vowel can be a moving harmonic stack too. Reject rapid
    # repeated pitch oscillations, not one/two turns in an aircraft maneuver.
    smooth=ndimage.gaussian_filter1d(ff,1.2)
    prominence=max(.8,.008*np.median(ff))
    peaks=signal.find_peaks(smooth,prominence=prominence)[0]
    valleys=signal.find_peaks(-smooth,prominence=prominence)[0]
    return len(peaks)+len(valleys)>=3 and (len(peaks)+len(valleys))/(tt[-1]-tt[0])>.38


def _compatible(a,b):
    lo=max(a.t0,b.t0);hi=min(a.t1,b.t1)
    if hi-lo<max(4., .6*min(a.t1-a.t0,b.t1-b.t0)):
        return False
    tt=np.arange(lo,hi+.01,.25)
    aa=np.interp(tt,a.t,a.f);bb=np.interp(tt,b.t,b.f)
    ratio=float(np.median(aa/bb))
    rational=Fraction(ratio).limit_denominator(12)
    if rational.numerator>16 or abs(float(rational)-ratio)>max(.004*ratio,.6/np.median(bb)):
        return False
    error=aa-float(rational)*bb
    if np.quantile(np.abs(error),.8)>max(.6,.003*np.median(aa)):
        return False
    if np.ptp(aa)>2 and np.ptp(bb)>2:
        da=np.diff(ndimage.gaussian_filter1d(aa,1.5));db=np.diff(ndimage.gaussian_filter1d(bb,1.5))
        return np.corrcoef(aa,bb)[0,1]>.9 and np.corrcoef(da,db)[0,1]>.7
    return True


def _track_chunk(S, t, f, cfg):
    """Extract non-parametric lines, recover harmonic members, classify stacks.

    Return 0.25-second Track samples. Group ids are local to this call. amp is
    in input-signal peak units only when S retains RidgeSpectrum.amplitude;
    arbitrary external whitened arrays cannot determine absolute amplitude.
    cfg.hop_s is intentionally not used: this API's output cadence is 0.25 s.
    """
    amplitude=getattr(S,'amplitude',None)
    S=np.asarray(S,dtype=float);t=np.asarray(t,dtype=float);f=np.asarray(f,dtype=float)
    if S.shape!=(len(f),len(t)) or len(f)<3 or np.any(np.diff(f)<=0) or np.any(np.diff(t)<=0):
        raise ValueError('S must have shape (len(f), len(t)) with increasing grids')
    if not np.allclose(np.diff(f),np.median(np.diff(f)),rtol=.01):
        raise ValueError('frequency grid must be uniform')
    if len(t)<3: return []
    if amplitude is not None and amplitude.shape!=S.shape:
        amplitude=None
    evidence=_evidence(S,f,amplitude)
    threshold=max(float(cfg.snr_db),4.)
    paths=_viterbi_lines(evidence,t,f,threshold)
    # Harmonic summation in POWER, with cap to stop one very loud order
    # masquerading as an entire stack. Variance-normalize the harmonic bank.
    # The physical fundamental can be BELOW cfg.fmin (rotors, engine orders).
    bf=np.arange(8.,min(500.,f[-1]/2),np.median(np.diff(f)))
    harm=np.zeros((len(bf),len(t)));weights=np.zeros(len(bf))
    observations=_llr(evidence)
    for h in range(1,25):
        indices=np.rint((h*bf-f[0])/(f[1]-f[0])).astype(int)
        good=(indices>=0)&(indices<len(f));weight=1/np.sqrt(h)
        harm[good]+=weight*observations[indices[good]];weights[good]+=weight**2
    harm/=np.sqrt(np.maximum(weights[:,None],1))
    # Harmonic observations sum LLRs at k*f0 under one shared frequency state.
    seeds=_viterbi_lines(harm,t,bf,2.,limit=50,observation=harm,harmonic_suppression=True) if len(bf)>3 else []
    # Recover physical members from summed-power templates, checking each
    # member's local contrast. This fills short gaps but never invents a tone.
    for frames,bins in seeds:
        fundamental=bf[bins]
        for h in range(1,25):
            center=h*fundamental
            if np.median(center)>f[-1]: break
            middle=np.rint((center-f[0])/(f[1]-f[0])).astype(int)
            radius=max(2,int(round((1.5 if np.median(center)<250 else 5)/(f[1]-f[0]))))
            offsets=np.arange(-radius,radius+1)
            options=np.clip(middle[:,None]+offsets,0,len(f)-1)
            value=evidence[options,frames[:,None]]
            best=np.argmax(value-.15*offsets[None,:]**2,axis=1)
            selected=options[np.arange(len(frames)),best]
            supported=evidence[selected,frames]>=max(3.,threshold*.75)
            if supported.mean()<.65: continue
            good=np.flatnonzero(supported)
            if len(good)>2:
                a,b=good[0],good[-1]+1
                paths.append((frames[a:b],selected[a:b]))
    lines=[];strength=[]
    for frames,bins in paths:
        tt=t[frames];ff=_refine_path(bins,frames,evidence,f)
        ff=ndimage.median_filter(ff,size=5,mode='nearest')
        if tt[-1]-tt[0]<1.75: continue
        # Ignore rapidly jumping speech paths and wandering noise maxima.
        sm=ndimage.gaussian_filter1d(ff,1.)
        jitter=float(np.sqrt(np.mean((ff-sm)**2)))
        if jitter>max(1.4,.008*np.median(ff)) or _speech_like(tt,ff): continue
        grid=np.arange(math.ceil(tt[0]*4)/4,tt[-1]+1e-6,.25)
        if len(grid)<2: continue
        frequency=np.interp(grid,tt,ff)
        amp=np.interp(grid,tt,amplitude[bins,frames]) if amplitude is not None else np.zeros(len(grid))
        tr=Track(grid,frequency,amp,kind='unknown')
        duplicate=False
        for old in lines:
            lo=max(old.t0,tr.t0);hi=min(old.t1,tr.t1)
            if hi-lo<.7*min(old.t1-old.t0,tr.t1-tr.t0): continue
            overlap=np.arange(lo,hi,.25)
            if len(overlap) and np.median(np.abs(np.interp(overlap,old.t,old.f)-np.interp(overlap,tr.t,tr.f)))<max(1.5,.003*np.median(ff)):
                duplicate=True;break
        if not duplicate:
            lines.append(tr);strength.append(float(np.median(evidence[bins,frames])))
    # Anchor each family to a long line: pairwise union-find would let
    # incompatible rational ratios chain into an absurd subharmonic lattice.
    groups={};unassigned=set(range(len(lines)))
    while unassigned:
        reference=max(unassigned,key=lambda i:lines[i].t1-lines[i].t0)
        members=[reference]+[i for i in unassigned if i!=reference and _compatible(lines[i],lines[reference])]
        groups[len(groups)]=members
        unassigned.difference_update(members)
    output=[]
    for group,members in enumerate(groups.values()):
        long=[i for i in members if lines[i].t1-lines[i].t0>=cfg.min_duration_s]
        if not long: continue
        moving=[i for i in long if _moving(lines[i].f,cfg)]
        # A stack requires actual shared motion, rather than a random ridge
        # near some rational multiple of a steady hum line.
        coherent=(len(moving)>=3 or (len(moving)==2 and min(strength[i] for i in moving)>=8.))
        reference=max(members,key=lambda i:lines[i].t1-lines[i].t0)
        ratios=[]
        for i in members:
            lo=max(lines[i].t0,lines[reference].t0);hi=min(lines[i].t1,lines[reference].t1)
            tt=np.arange(lo,hi+.01,.25)
            ratio=float(np.median(np.interp(tt,lines[i].t,lines[i].f)/np.interp(tt,lines[reference].t,lines[reference].f)))
            ratios.append(Fraction(ratio).limit_denominator(16))
        denominator=math.lcm(*(r.denominator for r in ratios))
        orders=[r.numerator*(denominator//r.denominator) for r in ratios]
        divisor=math.gcd(*orders)
        orders=[order//divisor for order in orders]
        # Reject inconsistent quantized ratios rather than predict a lattice.
        if max(orders)>32 or np.median(lines[reference].f)/orders[members.index(reference)]<8:
            coherent=False
            orders=[1]*len(members)
        divisor=1
        emitted=[]
        for i,order in zip(members,orders):
            tr=lines[i];motion=_moving(tr.f,cfg)
            duration=tr.t1-tr.t0
            if duration<2. and not coherent: continue
            if motion:
                if duration<cfg.min_duration_s and not coherent: continue
                # Single lines need high contrast and substantial, smooth
                # motion; stack members may be weak (the threat model).
                travel=np.sum(np.abs(np.diff(ndimage.gaussian_filter1d(tr.f,1.))))
                efficiency=np.ptp(tr.f)/max(travel,1e-9)
                if not coherent and (strength[i]<10 or efficiency<.4 or np.ptp(tr.f)<max(3.,.02*np.median(tr.f)) or np.max(np.abs(np.diff(tr.f)))>max(2.,.025*np.median(tr.f))): continue
                tr.kind='doppler';tr.doppler_score=.9 if coherent else .65
            else:
                if duration<cfg.min_duration_s: continue
                tr.kind='stationary';tr.doppler_score=0.
            tr.group=group;tr.harmonic=int(order//divisor)
            output.append(tr);emitted.append(tr)
        moving_emitted=[tr for tr in emitted if tr.kind=='doppler']
        if coherent and len(moving_emitted)>=2:
            # Shared fractional-frequency trajectory, robust across members.
            grid=np.arange(min(tr.t0 for tr in moving_emitted),max(tr.t1 for tr in moving_emitted)+.01,.25)
            bank=np.stack([np.interp(grid,tr.t,tr.f/tr.harmonic,left=np.nan,right=np.nan) for tr in moving_emitted])
            latent=np.nanmedian(bank,axis=0)
            finite=np.isfinite(latent)
            grid=grid[finite];latent=latent[finite]
            for h in range(1,int(cfg.fmax/np.min(latent))+1):
                frequency=h*latent
                valid=(frequency>=f[0]) & (frequency<=min(f[-1],cfg.fmax))
                if valid.sum()<2:continue
                # Extend measured members to cover the stack's gaps too.
                existing=[tr for tr in moving_emitted if tr.harmonic==h]
                gt=grid[valid];gf=frequency[valid]
                if existing:
                    tr=existing[0]
                    supported=(gt>=tr.t0)&(gt<=tr.t1)
                    gf[supported]=np.interp(gt[supported],tr.t,tr.f)
                    tr.t=gt;tr.f=gf
                else:
                    tr=Track(gt,gf,np.zeros(len(gt)),.55,None,h,group,'doppler')
                    output.append(tr)
                if amplitude is not None:
                    times=np.clip(np.searchsorted(t,tr.t),0,len(t)-1)
                    bins=np.clip(np.rint((tr.f-f[0])/(f[1]-f[0])).astype(int),0,len(f)-1)
                    tr.amp=amplitude[bins,times].copy()
                else:tr.amp=np.zeros(len(tr.t))
    contiguous=[]
    for tr in output:
        for part in np.split(np.arange(len(tr.t)),np.flatnonzero(np.diff(tr.t)>.26)+1):
            if len(part)<2:continue
            contiguous.append(Track(tr.t[part],tr.f[part],tr.amp[part],tr.doppler_score,
                                    tr.fit,tr.harmonic,tr.group,tr.kind))
    return sorted(contiguous,key=lambda tr:(tr.group,tr.harmonic,tr.t0))


def track_ridges(S, t, f, cfg):
    """Track arbitrary sweeps and harmonic stacks; return 0.25 s Track grids.

    Twenty-second overlapping inference windows prevent a loud early event
    from monopolizing the path budget of a long recording. Shared-factor
    groups are stitched across overlaps. Missing harmonics of confirmed
    moving stacks are predicted up to cfg.fmax (score .55 versus .9 measured).
    Preserve ridge_spectrogram's S object to retain calibrated amplitudes;
    plain whitened arrays yield zero/unknown amplitudes.
    """
    t=np.asarray(t,dtype=float)
    if len(t)<=96:
        return _track_chunk(S,t,f,cfg)
    all_tracks=[];representatives={};offset=0
    for start in np.arange(t[0],t[-1],10.):
        indices=np.flatnonzero((t>=start)&(t<start+20.))
        if len(indices)<3 or t[indices[-1]]-t[indices[0]]<cfg.min_duration_s:continue
        lo,hi=indices[0],indices[-1]+1
        tracks=_track_chunk(S[:,lo:hi],t[lo:hi],f,cfg)
        for tr in tracks:
            tr.group+=offset
            previous=representatives.get(tr.group)
            if previous is None or (tr.kind=='doppler' and tr.harmonic<previous.harmonic):
                representatives[tr.group]=tr
        all_tracks.extend(tracks)
        offset=max(representatives,default=-1)+1
    parents={g:g for g in representatives}
    def root(g):
        while parents[g]!=g:
            parents[g]=parents[parents[g]];g=parents[g]
        return g
    keys=list(representatives)
    for n,g in enumerate(keys):
        a=representatives[g]
        if a.kind!='doppler':continue
        for h in keys[:n]:
            b=representatives[h]
            if b.kind!='doppler':continue
            lo=max(a.t0,b.t0);hi=min(a.t1,b.t1)
            if hi-lo<2:continue
            grid=np.arange(lo,hi,.25)
            af=np.interp(grid,a.t,a.f)/a.harmonic
            bf=np.interp(grid,b.t,b.f)/b.harmonic
            if np.quantile(np.abs(af-bf),.8)<max(.6,.01*np.median(af)):
                parents[root(g)]=root(h)
    grouped={}
    for tr in all_tracks:
        tr.group=root(tr.group)
        grouped.setdefault((tr.group,tr.harmonic,tr.kind),[]).append(tr)
    merged=[]
    for tracks in grouped.values():
        tracks.sort(key=lambda tr:tr.t0)
        current=tracks[0]
        for nxt in tracks[1:]:
            if nxt.t0>current.t1+.26:
                merged.append(current);current=nxt;continue
            grid=np.arange(current.t0,max(current.t1,nxt.t1)+.01,.25)
            freq=np.interp(grid,current.t,current.f,left=np.nan,right=np.nan)
            amp=np.interp(grid,current.t,current.amp,left=np.nan,right=np.nan)
            other=np.interp(grid,nxt.t,nxt.f,left=np.nan,right=np.nan)
            oa=np.interp(grid,nxt.t,nxt.amp,left=np.nan,right=np.nan)
            freq=np.nanmean(np.stack([freq,other]),axis=0)
            amp=np.nanmean(np.stack([amp,oa]),axis=0)
            current=Track(grid,freq,amp,max(current.doppler_score,nxt.doppler_score),None,
                          current.harmonic,current.group,current.kind)
        merged.append(current)
    return sorted(merged,key=lambda tr:(tr.group,tr.harmonic,tr.t0))
