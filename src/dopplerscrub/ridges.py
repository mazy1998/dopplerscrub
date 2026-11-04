"""Non-parametric frequency-line tracking for maneuvering harmonic sources.

The spectrogram result is an ndarray subclass carrying pre-whitening peak
magnitudes in ``.amplitude``. Preserve that object when calling track_ridges;
plain external dB arrays are accepted, but their Track.amp is zero (unknown).
No ideal-pass fit or attack-library dependency is used here. Physics prior:
120 m/s maximum speed, 500 m minimum slant range, c=343 m/s; transitions
are truncated at three standard deviations. A Hann periodogram is used for
calibration; multitaper whitening is left for future variance reduction.
"""
from __future__ import annotations
from fractions import Fraction
import math
from functools import lru_cache
import numpy as np
from scipy import signal, ndimage, stats
from .types import DetectConfig, Track


class RidgeSpectrum(np.ndarray):
    """Whitened dB array with aligned, calibrated sinusoidal peak magnitudes."""
    def __new__(cls, values, amplitude):
        obj = np.asarray(values, dtype=np.float32).view(cls)
        obj.amplitude = np.asarray(amplitude, dtype=np.float32)
        obj.calibration = []
        return obj

    def __array_finalize__(self, obj):
        self.amplitude = getattr(obj, 'amplitude', None)
        self.calibration = getattr(obj, 'calibration', [])

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, RidgeSpectrum) and self.amplitude is not None:
            result.amplitude = self.amplitude[key]
        return result


def ridge_spectrogram(x, sr, cfg):
    """Return (S, t, f), mono, 0.25 s frames, multiresolution STFT.

    All bands retain a >=2 s Hann view (~0.5 Hz resolution). The
    short view uses 0.5 s below 700 Hz and 0.25 s above to follow rapid
    passes. Short spectra are fused with the long view on the common grid; interpolation does
    not increase their physical resolution. Frame times are sample-centered;
    zero padding includes both clip edges. Native signal units are retained.
    """
    x = np.asarray(x)
    if sr <= 0 or x.ndim not in (1, 2) or not np.isfinite(x).all():
        raise ValueError('finite mono or samples-by-channels audio and positive sr required')
    if not 0 < cfg.fmin < cfg.fmax or cfg.fmin >= sr/2:
        raise ValueError('require 0 < fmin < fmax and fmin < input Nyquist')
    fmax=min(cfg.fmax,np.nextafter(float(sr)/2,0.))
    y = np.mean(x, axis=1) if x.ndim == 2 else x
    rate = max(int(cfg.work_sr), int(math.ceil(fmax*2.4)))
    g = math.gcd(int(sr), rate)
    if len(y):
        y = signal.resample_poly(y, rate//g, int(sr)//g)
    nlong = max(int(cfg.nperseg), 2*rate)
    df = rate/nlong
    f = np.arange(math.ceil(cfg.fmin/df), math.floor(fmax/df)+1)*df
    t = np.arange(0, len(y)/rate, .25)
    if not len(t):
        return RidgeSpectrum(np.empty((len(f), 0)), np.empty((len(f), 0))), t, f
    amp = np.zeros((len(f), len(t)), np.float32)
    long_amp=np.zeros_like(amp)
    for lo, hi, n in [(0, np.inf, nlong), (0, 700, round(.5*rate)),
                       (700, np.inf, round(.25*rate))]:
        band = (f > lo) & (f <= hi)
        if not band.any() and n!=nlong:
            continue
        window = signal.windows.hann(n, sym=False)
        # Dense FFT sampling avoids artificial 2–4 Hz frequency steps in
        # short-window peaks. It interpolates the spectrum, not its magnitude.
        ff = np.fft.rfftfreq(nlong, 1/rate)
        padded = np.pad(y, (n//2, n))
        for j, time in enumerate(t):
            start = round(time*rate)
            # FFT is unnormalized: 2|Z| / sum(window) gives sinusoid peak.
            a = 2*np.abs(np.fft.rfft(padded[start:start+n]*window,n=nlong))/window.sum()
            amp[band, j] = np.interp(f[band], ff, a)
            if n==nlong:long_amp[:,j]=np.interp(f,ff,a)
    def whiten(magnitude):
        db=20*np.log10(magnitude+1e-12)
        width=2400
        centers=[];medians=[]
        for start in range(0,len(t),width):
            stop=min(start+width,len(t))
            centers.append((start+stop-1)/2)
            medians.append(np.median(db[:,start:stop],axis=1))
        medians=np.array(medians)
        baseline=np.empty_like(db)
        for k in range(len(f)):
            baseline[k]=np.interp(np.arange(len(t)),centers,medians[:,k])
        return db-baseline
    adaptive=whiten(amp);long_view=whiten(long_amp)
    magnitude=np.maximum(long_amp,amp)
    # Keep faint higher harmonics visible to the attacker, alongside short-
    # window peaks needed at maximum chirp rate. Reference nulls include this
    # same max operation; its distribution is not assumed to be pure Exp(1).
    return RidgeSpectrum(np.maximum(adaptive,long_view),magnitude),t,f


def _evidence(S, f, amplitude, base=None):
    """Remove broadband foreground while preserving stationary raw ridges."""
    if base is None:
        result=np.empty_like(S,dtype=np.float32)
        for lo,hi,width in [(0,450,18.),(450,700,40.),(700,np.inf,80.)]:
            indices=np.flatnonzero((f>lo)&(f<=hi))
            if not len(indices):continue
            bins=max(5,int(round(width/np.median(np.diff(f))))|1)
            left=max(0,int(indices[0])-bins//2)
            right=min(len(f),int(indices[-1])+bins//2+1)
            local=ndimage.median_filter(S[left:right],size=(bins,1),mode='nearest')
            result[indices]=S[indices]-local[indices-left]
    else:result=np.asarray(base,dtype=np.float32).copy()
    if amplitude is not None:
        raw = 20*np.log10(amplitude+1e-12)
        tonal = raw-ndimage.median_filter(raw, size=(49,1), mode='nearest')
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
    z = np.log(2)*10**(np.clip(db,-300,300)/10)
    return z*snr/(1+snr)-np.log1p(snr)


def _robust_llr(db, snr=3.):
    # Winsorize individual observations so one clipped/saturated harmonic
    # cannot impersonate an entire coherent stack. Nulls use the same cap.
    return np.minimum(_llr(db,snr),12.)


def _transition_cost(delta, center, dt, df):
    # A slow Gaussian core provides loose continuation of faint lines. The
    # physics-width Gaussian tail permits fast flybys without raising the
    # continuation cost of an almost stationary distant aircraft.
    sigma=np.maximum(df,center*120.**2/(343.*500.)*dt)
    slow=np.maximum(df,.005*center*dt)
    a=np.log(.9)-.5*(delta/slow)**2-np.log(np.sqrt(2*np.pi)*slow/df)
    b=np.log(.1)-.5*(delta/sigma)**2-np.log(np.sqrt(2*np.pi)*sigma/df)
    cost=-np.logaddexp(a,b)
    return np.where(np.abs(delta)<=3*sigma,cost,1e6)


def _viterbi_lines(evidence, t, f, threshold, limit=70, observation=None, harmonic_suppression=False, score_only=False, min_score=24.):
    """Sparse-bin Viterbi with zero-score births, deaths, Gaussian transitions.

    Local maxima are the frequency states (at most 96/frame), rather than
    spending quadratic work on bins with no ridge evidence. Paths compete
    through iterative corridor suppression and are split at unsupported gaps.
    """
    F,T = evidence.shape
    if T < 3 or F < 3:
        return 0. if score_only else []
    dt = float(np.median(np.diff(t)))
    candidates=[]
    for j in range(T):
        peaks = signal.find_peaks(evidence[:,j], height=max(1., threshold*.6), distance=2)[0]
        if len(peaks)>96:
            peaks=peaks[np.argsort(evidence[peaks,j])[-96:]]
        candidates.append(peaks)
    N=max(map(len,candidates),default=0)
    if N==0:
        return 0. if score_only else []
    idx=np.zeros((T,N),int);valid=np.zeros((T,N),bool)
    for j,p in enumerate(candidates):
        idx[j,:len(p)]=p;valid[j,:len(p)]=True
    hz=f[idx]
    llr = _robust_llr(evidence) if observation is None else observation
    emission=llr[idx,np.arange(T)[:,None]].astype(float)
    emission[~valid]=-1e6
    # Sparse edges: the hard physics bound usually admits only 2–15 of
    # 96 predecessor peaks. Do not evaluate a dense N*N*beam tensor only to
    # discard almost all of it afterwards.
    edges=[]
    df=float(np.median(np.diff(f)))
    for j in range(1,T):
        delta=hz[j,:,None]-hz[j-1,None,:]
        center=(hz[j,:,None]+hz[j-1,None,:])/2
        sigma=np.maximum(df,center*120.**2/(343.*500.)*dt)
        allowed=np.abs(delta)<=3*sigma
        counts=allowed.sum(axis=1)
        width=max(1,int(counts.max()))
        neighbors=np.zeros((N,width),int);present=np.zeros((N,width),bool)
        row_ids,columns=np.nonzero(allowed)
        starts=np.cumsum(counts)-counts
        positions=np.arange(len(columns))-np.repeat(starts,counts)
        neighbors[row_ids,positions]=columns;present[row_ids,positions]=True
        row=np.arange(N)[:,None]
        dd=delta[row,neighbors];cc=center[row,neighbors];ss=sigma[row,neighbors]
        acceleration=np.maximum(df,.25*cc*120.**3/(343.*500.**2)*dt**2)
        core_norm=np.log(.95)-np.log(np.sqrt(2*np.pi)*acceleration/df)
        reset=np.log(.05)-.5*(dd/ss)**2-np.log(np.sqrt(2*np.pi)*ss/df)
        edges.append((neighbors,present,dd,acceleration,core_norm,reset))
    paths=[]
    beams=min(2,N)
    rows=np.arange(N)[:,None]
    for _ in range(limit):
        # Frequency plus two competing slope hypotheses. Negative active
        # scores remain alive while accumulating evidence against null=0;
        # they pay a birth cost once, rather than being reset every frame.
        scores=np.full((T,N,beams),-1e6)
        velocity=np.zeros((N,beams))
        back=np.full((T,N,beams),-1,np.int16)
        birth=math.log(max(F,3))
        scores[0,:,0]=emission[0]-birth
        for j in range(1,T):
            neighbors,present,delta,acceleration,core_norm,reset=edges[j-1]
            width=neighbors.shape[1]
            innovation=delta[:,:,None]-velocity[neighbors]
            core=core_norm[:,:,None]-.5*(innovation/acceleration[:,:,None])**2
            value=scores[j-1][neighbors]+np.logaddexp(core,reset[:,:,None])
            value=np.where(present[:,:,None],value,-1e6)
            best_beam=value.argmax(axis=2)
            best=np.take_along_axis(value,best_beam[:,:,None],axis=2)[:,:,0]
            options=np.column_stack((best,np.full(N,-birth)))
            previous=np.argpartition(-options,beams-1,axis=1)[:,:beams]
            selected=np.take_along_axis(options,previous,axis=1)
            scores[j]=selected+emission[j,:,None]
            active=previous<width
            clipped=np.minimum(previous,width-1)
            old_state=neighbors[rows,clipped]
            old_beam=best_beam[rows,clipped]
            back[j]=np.where(active,old_state*beams+old_beam,-1)
            velocity=np.where(active,delta[rows,clipped],0.)
        end,state,beam=np.unravel_index(np.argmax(scores),scores.shape)
        maximum=max(0.,float(scores[end,state,beam]))
        if score_only:return maximum
        if maximum<min_score:break
        points=[]
        j=int(end);k=int(state);h=int(beam)
        while j>=0:
            points.append((j,int(idx[j,k])))
            previous=int(back[j,k,h]);j-=1
            if previous<0:break
            k,h=divmod(previous,beams)
        points.reverse()
        if not points: break
        times=np.array([p[0] for p in points]);bins=np.array([p[1] for p in points])
        # Suppress roughly one resolution element, wider in the short-window bands.
        for frame,bin_ in points:
            corridor=max(1., 2. if f[bin_]<=450 else 5. if f[bin_]<=700 else 9.)
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
    if np.quantile(freq,.95)-np.quantile(freq,.05)<max(cfg.min_sweep_hz,1.5,.01*np.median(freq)):
        return False
    steps=np.abs(np.diff(freq))
    if len(steps) and np.mean(steps<max(.15,.0007*np.median(freq)))>.55 and np.max(steps)>max(2.,.02*np.median(freq)):
        return False  # held musical notes joined by a discrete pitch change
    if len(freq)>=16:
        if (np.quantile(freq,.95)-np.quantile(freq,.05))/np.median(freq)<.03:
            window=min(17,len(freq) if len(freq)%2 else len(freq)-1)
            slope=signal.savgol_filter(freq,window,2,deriv=1,delta=.25)
            magnitude=np.abs(slope)
            curved=np.quantile(magnitude,.9)>1.8*max(np.median(magnitude),.02)
            turns=np.quantile(slope,.1)<-.04 and np.quantile(slope,.9)>.04
            if not curved and not turns:return False
        # Measure note-hold concentration at a physical time scale. Half-Hz
        # quantization makes a genuine slow root look like a staircase before
        # its higher harmonics are combined; do not count bin plateaus as holds.
        smooth=ndimage.gaussian_filter1d(ndimage.median_filter(freq,3,mode='nearest'),1.5)
        local_span=ndimage.maximum_filter1d(smooth,3)-ndimage.minimum_filter1d(smooth,3)
        flat=np.mean(local_span<max(.3,.0015*np.median(freq)))
        variation=np.abs(np.diff(smooth))
        top=max(1,int(np.ceil(.2*len(variation))))
        concentration=np.sort(variation)[-top:].sum()/max(variation.sum(),1e-9)
        # A sequence of sustained notes connected through their decay tails
        # is not a continuously moving ridge. Keep slow straight drifts:
        # those distribute their motion throughout the linked track.
        if flat>.4 and concentration>.65:
            active=variation>max(.25,.3*np.mean(variation))
            # Separate genuine multi-second chirps from a one-frame note
            # change. Close only isolated 0.25-second estimation dropouts.
            active=ndimage.binary_closing(active,structure=np.ones(3,bool))
            indices=np.flatnonzero(active)
            longest=max((len(part)*.25 for part in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)),default=0.)
            if longest<cfg.min_duration_s:
                return False
    return True


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
    if np.quantile(np.abs(error),.8)>max(1.2,.006*np.median(aa)):
        return False
    if np.ptp(aa)>2 and np.ptp(bb)>2:
        da=np.diff(ndimage.gaussian_filter1d(aa,1.5));db=np.diff(ndimage.gaussian_filter1d(bb,1.5))
        return np.corrcoef(aa,bb)[0,1]>.88 and np.corrcoef(da,db)[0,1]>.6
    return True


def _track_chunk(S, t, f, cfg, base_evidence=None):
    """Extract non-parametric lines, recover harmonic members, classify stacks.

    Return 0.25-second Track samples. Group ids are local to this call. amp is
    in input-signal peak units only when S retains RidgeSpectrum.amplitude;
    arbitrary external whitened arrays cannot determine absolute amplitude.
    cfg.hop_s is intentionally not used: this API's output cadence is 0.25 s.
    """
    amplitude=getattr(S,'amplitude',None)
    calibration_log=getattr(S,'calibration',None)
    S=np.asarray(S,dtype=float);t=np.asarray(t,dtype=float);f=np.asarray(f,dtype=float)
    if S.shape!=(len(f),len(t)) or len(f)<3 or np.any(np.diff(f)<=0) or np.any(np.diff(t)<=0):
        raise ValueError('S must have shape (len(f), len(t)) with increasing grids')
    if not np.allclose(np.diff(f),np.median(np.diff(f)),rtol=.01):
        raise ValueError('frequency grid must be uniform')
    if len(t)<3: return []
    if amplitude is not None and amplitude.shape!=S.shape:
        amplitude=None
    moving_evidence=_evidence(S,f,None) if base_evidence is None else np.asarray(base_evidence,dtype=np.float32).copy()
    evidence=_evidence(S,f,amplitude,base=moving_evidence)
    threshold=max(float(cfg.snr_db),4.)
    reference=_reference_calibration(tuple(np.round(f,6)),len(t))
    paths=_viterbi_lines(evidence,t,f,threshold,min_score=reference['single_cutoff'])
    high=np.flatnonzero(f>=450.)
    if len(high)>3:
        paths.extend((frames,bins+high[0]) for frames,bins in
                     _viterbi_lines(evidence[high],t,f[high],threshold,limit=35,min_score=reference['high_cutoff']))
    # Harmonic summation in POWER, with cap to stop one very loud order
    # masquerading as an entire stack. Variance-normalize the harmonic bank.
    # The physical fundamental can be BELOW cfg.fmin (rotors, engine orders).
    bf=np.arange(8.,min(500.,f[-1]/2),np.median(np.diff(f)))
    harm=np.zeros((len(bf),len(t)));weights=np.zeros(len(bf))
    if amplitude is not None:
        moving_evidence[amplitude<max(float(np.max(amplitude))*1e-5,1e-12)]=-20
    observations=_robust_llr(moving_evidence)
    for h in range(1,25):
        indices=np.rint((h*bf-f[0])/(f[1]-f[0])).astype(int)
        good=(indices>=0)&(indices<len(f));weight=1/np.sqrt(h)
        harm[good]+=weight*observations[indices[good]];weights[good]+=weight**2
    harm/=np.sqrt(np.maximum(weights[:,None],1))
    maxima=reference['maxima'];cutoff=reference['cutoff']
    single_maxima=reference['single_maxima'];single_cutoff=reference['single_cutoff']
    high_cutoff=reference['high_cutoff']
    if calibration_log is not None:
        calibration_log.append(dict(start_s=float(t[0]),end_s=float(t[-1]),
                                    reference='seeded white/pink noise and hum',**reference))
    # Harmonic observations sum LLRs at k*f0 under one shared frequency state.
    seeds=_viterbi_lines(harm,t,bf,2.,limit=35,observation=harm,harmonic_suppression=True,min_score=24.) if len(bf)>3 else []
    # Recover physical members from summed-power templates, checking each
    # member's local contrast. This fills short gaps but never invents a tone.
    for frames,bins in seeds:
        fundamental=bf[bins]
        for h in range(1,25):
            center=h*fundamental
            if np.median(center)>f[-1]: break
            middle=np.rint((center-f[0])/(f[1]-f[0])).astype(int)
            radius=max(2,int(round((1.5 if np.median(center)<450 else 5)/(f[1]-f[0]))))
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
    seed_lines=[]
    for seed_id,(frames,bins) in enumerate(seeds):
        leader=np.max(harm[:,frames],axis=0)
        if np.mean(harm[bins,frames]>=np.maximum(3.,.65*leader))<.55:continue
        root_curve=_refine_path(bins,frames,harm,bf)
        root_curve=ndimage.median_filter(root_curve,3,mode='nearest')
        ti=t[frames]
        score=float(np.sum(harm[bins,frames])-_transition_cost(np.diff(root_curve),(root_curve[1:]+root_curve[:-1])/2,.25,bf[1]-bf[0]).sum()-np.log(len(bf)))
        strict=score>cutoff
        if not strict and score<24.:continue
        estimates=[];members=[]
        for k in range(1,33):
            center=k*root_curve
            if np.median(center)>f[-1]:break
            if np.median(center)<f[0]:continue
            radius=max(2,int(np.ceil(max(1.5,k*(bf[1]-bf[0]))/(f[1]-f[0]))))
            middle=np.rint((center-f[0])/(f[1]-f[0])).astype(int)
            offsets=np.arange(-radius,radius+1)
            options=np.clip(middle[:,None]+offsets,0,len(f)-1)
            values=moving_evidence[options,frames[:,None]]
            best=np.argmax(values-.2*(offsets[None,:]/radius)**2,axis=1)
            selected=options[np.arange(len(frames)),best]
            frequency=_refine_path(selected,frames,moving_evidence,f)
            support=moving_evidence[selected,frames]>=max(2.,cfg.snr_db*.7)
            if support.mean()<.45:continue
            estimate=frequency/k;estimate[~support]=np.nan
            estimates.append(estimate);members.append((k,frequency,selected,support))
        if len(members)<2:continue
        bank=np.stack(estimates)
        observed=np.sum(np.isfinite(bank),axis=0)>0
        root_curve[observed]=np.nanmedian(bank[:,observed],axis=0)
        root_curve=ndimage.median_filter(root_curve,3,mode='nearest')
        if _speech_like(ti,root_curve):continue
        grid=np.arange(math.ceil(ti[0]*4)/4,ti[-1]+1e-6,.25)
        if len(grid)<2:continue
        for k,frequency,selected,support in members:
            # One frequency factor fills weak intervals; observations refine
            # it only when they remain consistent with the shared stack.
            target=k*root_curve
            measured=support&(np.abs(frequency-target)<=max(1.5,.004*np.median(target)))
            target[measured]=frequency[measured]
            amp=np.interp(grid,ti,amplitude[selected,frames]) if amplitude is not None else np.zeros(len(grid))
            seed_lines.append(Track(grid,np.interp(grid,ti,target),amp,.95 if strict else .45,None,k,seed_id,'candidate'))
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
        if not members: continue
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
        anchor=lines[reference]
        root_freq=anchor.f/orders[members.index(reference)]
        fi=np.clip(np.rint((root_freq-bf[0])/(bf[1]-bf[0])).astype(int),0,len(bf)-1)
        ti=np.clip(np.searchsorted(t,anchor.t),0,len(t)-1)
        joint_score=float(np.sum(harm[fi,ti])-np.sum(np.log(np.maximum(1.,root_freq*.25*120**2/(343*500)/(bf[1]-bf[0])))))
        if calibration_log is not None:
            calibration_log[-1].setdefault('groups',[]).append(dict(base=float(np.median(root_freq)),score=joint_score,orders=list(map(int,orders)),coherent=bool(coherent)))
        calibrated=joint_score>cutoff
        coherent=coherent and calibrated
        emitted=[]
        for i,order in zip(members,orders):
            tr=lines[i]
            ti_line=np.clip(np.searchsorted(t,tr.t),0,len(t)-1)
            fi_line=np.clip(np.rint((tr.f-f[0])/(f[1]-f[0])).astype(int),0,len(f)-1)
            llr=_robust_llr(moving_evidence[fi_line,ti_line])
            transition=_transition_cost(np.diff(tr.f),(tr.f[1:]+tr.f[:-1])/2,.25,f[1]-f[0])
            single_score=float(llr.sum()-transition.sum()-np.log(len(f)))
            strong_single=single_score>(high_cutoff if np.median(tr.f)>=450 else single_cutoff)
            if calibrated or strong_single:
                # Classification is deferred until all fragments are linked.
                tr.kind='candidate';tr.doppler_score=.9 if coherent else .65
            elif not _moving(tr.f,cfg):
                tr.kind='stationary';tr.doppler_score=0.
            else:continue
            tr.group=group;tr.harmonic=int(order//divisor)
            output.append(tr);emitted.append(tr)
    output.extend(seed_lines)
    contiguous=[]
    for tr in output:
        for part in np.split(np.arange(len(tr.t)),np.flatnonzero(np.diff(tr.t)>.26)+1):
            if len(part)<2:continue
            contiguous.append(Track(tr.t[part],tr.f[part],tr.amp[part],tr.doppler_score,
                                    tr.fit,tr.harmonic,tr.group,tr.kind))
    return sorted(contiguous,key=lambda tr:(tr.group,tr.harmonic,tr.t0))


def track_ridges(S, t, f, cfg):
    """Track arbitrary sweeps and harmonic stacks; return 0.25 s Track grids.

    Forty-second overlapping inference windows prevent a loud early event
    from monopolizing the path budget of a long recording. Shared-factor
    groups are stitched across overlaps. Missing harmonics of confirmed
    moving groups emit continuous harmonic proposals through cfg.fmax.
    Continuous orders inherit their confirmed source confidence; the downstream
    coherence gate decides whether an unobserved harmonic is present. Parent spans are clipped
    before expansion, so no source is extended beyond its trusted moving span.
    Preserve ridge_spectrogram's S object to retain calibrated amplitudes;
    plain whitened arrays yield zero/unknown amplitudes.
    """
    t=np.asarray(t,dtype=float)
    if hasattr(S,'calibration'):S.calibration.clear()
    f=np.asarray(f,dtype=float)
    base=_evidence(np.asarray(S,dtype=np.float32),f,None) if len(t)>=3 and len(f)>=3 and np.shape(S)==(len(f),len(t)) else None
    if len(t)<=96:
        return _finalize(_track_chunk(S,t,f,cfg,base),S,t,f,cfg,base)
    all_tracks=[];representatives={};offset=0
    covered=t[0]-.25
    for start in np.arange(t[0],t[-1],30.):
        if covered>=t[-1]-.01:break
        indices=np.flatnonzero((t>=start)&(t<start+40.))
        if len(indices)<3 or t[indices[-1]]-t[indices[0]]<cfg.min_duration_s:continue
        lo,hi=indices[0],indices[-1]+1
        covered=t[hi-1]
        tracks=_track_chunk(S[:,lo:hi],t[lo:hi],f,cfg,None if base is None else base[:,lo:hi])
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
    return _finalize(merged,S,t,f,cfg,base)


def _join(a, b):
    grid=np.arange(min(a.t0,b.t0),max(a.t1,b.t1)+.01,.25)
    # Linear interpolation in <=1.5 s unsupported intervals is the slope-
    # bridging prediction; supported overlap is averaged in original units.
    fa=np.interp(grid,a.t,a.f,left=np.nan,right=np.nan)
    fb=np.interp(grid,b.t,b.f,left=np.nan,right=np.nan)
    aa=np.interp(grid,a.t,a.amp,left=np.nan,right=np.nan)
    ab=np.interp(grid,b.t,b.amp,left=np.nan,right=np.nan)
    count=np.isfinite(fa).astype(int)+np.isfinite(fb)
    freq=np.divide(np.nan_to_num(fa)+np.nan_to_num(fb),count,out=np.full(len(grid),np.nan),where=count>0)
    amp=np.divide(np.nan_to_num(aa)+np.nan_to_num(ab),count,out=np.full(len(grid),np.nan),where=count>0)
    ok=np.isfinite(freq)
    freq=np.interp(grid,grid[ok],freq[ok]);amp=np.interp(grid,grid[ok],amp[ok])
    return Track(grid,freq,amp,max(a.doppler_score,b.doppler_score),None,a.harmonic,a.group,a.kind)


def _consolidate(tracks, binwidth, bridge=True):
    """Join by indexed time/frequency bounds before comparing sample arrays."""
    tracks=sorted(tracks,key=lambda tr:(tr.doppler_score>=.9,tr.t1-tr.t0),reverse=True)
    def metadata(tr):
        def slope(tt,ff):
            tt=tt-np.mean(tt)
            return float(np.dot(tt,ff-np.mean(ff))/max(np.dot(tt,tt),1e-12))
        n=min(5,len(tr.t))
        first=slope(tr.t[:n],tr.f[:n]);last=slope(tr.t[-n:],tr.f[-n:])
        low=min(float(np.min(tr.f)),tr.f[0]-1.75*first,tr.f[-1]+1.75*last)-8
        high=max(float(np.max(tr.f)),tr.f[0]-1.75*first,tr.f[-1]+1.75*last)+8
        return (int(round(tr.t0*4)),int(round(tr.t1*4)),float(np.median(tr.f)),first,last,
                int(np.floor(low/8)),int(np.floor(high/8)))
    changed=True
    while changed:
        changed=False;result=[];meta=[];buckets={}
        for tr in tracks:
            info=metadata(tr)
            candidates=set()
            for bucket in range(info[5],info[6]+1):candidates.update(buckets.get(bucket,()))
            match=None
            for j in sorted(candidates):
                old=result[j];oi=meta[j]
                if tr.kind!=old.kind:continue
                lo=max(info[0],oi[0]);hi=min(info[1],oi[1])
                if hi>=lo:
                    a0=lo-info[0];b0=lo-oi[0];length=hi-lo+1
                    if length<3:continue
                    aa=tr.f[a0:a0+length];bb=old.f[b0:b0+length]
                    if len(aa)!=length or len(bb)!=length:
                        grid=np.arange(lo,hi+1)/4
                        aa=np.interp(grid,tr.t,tr.f);bb=np.interp(grid,old.t,old.f)
                    center=(info[2]+oi[2])/2
                    resolution=binwidth if center<=450 else max(binwidth,2. if center<=700 else 4.)
                    if np.mean(np.abs(aa-bb)<=resolution)>.5:match=j;break
                elif bridge:
                    a,b,am,bm=(tr,old,info,oi) if tr.t1<old.t0 else (old,tr,oi,info)
                    delta=b.t0-a.t1
                    if delta>1.75:continue
                    center=(a.f[-1]+b.f[0])/2
                    resolution=binwidth if center<=450 else max(binwidth,2. if center<=700 else 4.)
                    uncertainty=center*120**2/(343*500)*delta*.1
                    tolerance=max(2*binwidth,.006*center,min(uncertainty,max(3.,2*resolution)))
                    if abs(a.f[-1]+am[4]*delta-b.f[0])<=tolerance and abs(b.f[0]-bm[3]*delta-a.f[-1])<=tolerance:
                        match=j;break
            if match is None:
                position=len(result);result.append(tr);meta.append(info)
            else:
                position=match;result[position]=_join(result[position],tr)
                info=metadata(result[position]);meta[position]=info;changed=True
            for bucket in range(info[5],info[6]+1):buckets.setdefault(bucket,set()).add(position)
        tracks=result
    return tracks


def _finalize(tracks, S, t, f, cfg, base_evidence=None):
    """Link measurements first, then predict each source's orders only once."""
    if not tracks:return []
    df=float(np.median(np.diff(f)))
    tracks=_consolidate(tracks,df)
    tracks=[tr for tr in tracks if tr.t1-tr.t0>=cfg.min_duration_s and (tr.kind!='candidate' or tr.doppler_score>=.65)]
    for tr in tracks:
        if tr.kind=='candidate':tr.kind='doppler' if _moving(tr.f,cfg) else 'stationary'
        elif tr.kind=='doppler' and not _moving(tr.f,cfg):tr.kind='stationary'
    unassigned=set(range(len(tracks)));families=[]
    while unassigned:
        anchor=max(unassigned,key=lambda i:(tracks[i].doppler_score>=.9,tracks[i].t1-tracks[i].t0))
        members=[anchor]+[i for i in unassigned if i!=anchor and tracks[i].kind==tracks[anchor].kind and _compatible(tracks[i],tracks[anchor])]
        families.append(members);unassigned.difference_update(members)
    amplitude=getattr(S,'amplitude',None)
    output=[]
    next_independent_group=len(families)
    verification=_evidence(np.asarray(S,dtype=float),f,None) if base_evidence is None else np.asarray(base_evidence,dtype=np.float32)
    for group,members in enumerate(families):
        anchor=tracks[members[0]]
        ratios=[]
        for i in members:
            tr=tracks[i];lo=max(anchor.t0,tr.t0);hi=min(anchor.t1,tr.t1)
            grid=np.arange(lo,hi+.01,.25)
            ratio=np.median(np.interp(grid,tr.t,tr.f)/np.interp(grid,anchor.t,anchor.f))
            ratios.append(Fraction(float(ratio)).limit_denominator(16))
        denom=math.lcm(*(r.denominator for r in ratios))
        orders=[r.numerator*(denom//r.denominator) for r in ratios]
        divisor=math.gcd(*orders);orders=[k//divisor for k in orders]
        base=float(np.median(anchor.f))/orders[0]
        coherent=(len(members)>=2 and max(orders)<=32 and base>=8 and anchor.kind=='doppler' and
                  (len(members)>=3 or max(orders)<=12))
        for i,k in zip(members,orders):
            tr=tracks[i];tr.group=group
            if coherent:tr.harmonic=int(k)
            # An incompatible family does not turn observed h2/h3 into h1.
            # Preserve its measured order so it cannot seed a virtual root.
        if not coherent:
            for i in members:
                tracks[i].group=next_independent_group;next_independent_group+=1
                output.append(tracks[i])
            continue
        grid=np.arange(min(tracks[i].t0 for i in members),max(tracks[i].t1 for i in members)+.01,.25)
        bank=np.stack([np.interp(grid,tracks[i].t,tracks[i].f/k,left=np.nan,right=np.nan) for i,k in zip(members,orders)])
        counts=np.sum(np.isfinite(bank),axis=0)
        # Avoid all-NaN median warnings in source birth/death gaps.
        latent=np.full(len(grid),np.nan)
        latent[counts>0]=np.nanmedian(bank[:,counts>0],axis=0)
        for missing in np.split(np.flatnonzero(counts==0),np.flatnonzero(np.diff(np.flatnonzero(counts==0))>1)+1):
            if 0<len(missing)<=6 and missing[0]>0 and missing[-1]<len(grid)-1:
                latent[missing]=np.interp(grid[missing],grid[np.isfinite(latent)],latent[np.isfinite(latent)])
        finite=np.isfinite(latent)
        for k in range(1,int(cfg.fmax/np.nanmin(latent))+1):
            frequency=k*latent
            valid=finite&(frequency>=f[0])&(frequency<=min(cfg.fmax,f[-1]))
            ti=np.clip(np.searchsorted(t,grid),0,len(t)-1)
            fi=np.clip(np.rint((np.nan_to_num(frequency,nan=f[0])-f[0])/df).astype(int),0,len(f)-1)
            # Check one independent analysis bin either side, then smooth for
            # 1.5 s. High orders need their own stronger evidence.
            radius=1 if np.nanmedian(frequency)<=450 else int(np.ceil((2. if np.nanmedian(frequency)<=700 else 4.)/df))
            offsets=np.arange(-radius,radius+1)
            choices=np.clip(fi[:,None]+offsets,0,len(f)-1)
            support=np.max(verification[choices,ti[:,None]],axis=1)
            support=ndimage.uniform_filter1d(support,size=7,mode='nearest')
            valid &= support >= max(cfg.snr_db,3.)
            indices=np.flatnonzero(valid)
            for part in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
                if len(part)<2 or grid[part[-1]]-grid[part[0]]<cfg.min_duration_s:continue
                gt=grid[part];gf=frequency[part]
                if not _moving(gf,cfg):continue
                measured=[tracks[i] for i,order in zip(members,orders) if order==k]
                score=.55
                for tr in measured:
                    supported=(gt>=tr.t0)&(gt<=tr.t1)
                    observed=np.interp(gt,tr.t,tr.f)
                    resolution=df if np.median(gf)<=450 else max(df,2. if np.median(gf)<=700 else 4.)
                    supported &= np.abs(observed-gf)<=resolution
                    gf[supported]=observed[supported]
                    score=max(score,tr.doppler_score)
                if amplitude is None:amp=np.zeros(len(gt))
                else:
                    ti=np.clip(np.searchsorted(t,gt),0,len(t)-1)
                    fi=np.clip(np.rint((gf-f[0])/df).astype(int),0,len(f)-1)
                    amp=amplitude[fi,ti].copy()
                output.append(Track(gt,gf,amp,score,None,k,group,'doppler'))
    output=_consolidate(output,df)
    bounded=[]
    for tr in output:
        sigma=np.maximum(df,(tr.f[1:]+tr.f[:-1])/2*120**2/(343*500)*np.diff(tr.t))
        cuts=np.flatnonzero(np.abs(np.diff(tr.f))>3*sigma)+1
        good=np.ones(len(tr.t),bool)
        if tr.kind=='doppler':
            ti=np.clip(np.searchsorted(t,tr.t),0,len(t)-1)
            fi=np.clip(np.rint((tr.f-f[0])/df).astype(int),0,len(f)-1)
            resolution=df if np.median(tr.f)<=450 else max(df,2. if np.median(tr.f)<=700 else 4.)
            radius=max(1,int(np.ceil(resolution/df)))
            choices=np.clip(fi[:,None]+np.arange(-radius,radius+1),0,len(f)-1)
            strength=np.max(verification[choices,ti[:,None]],axis=1)
            strength=ndimage.uniform_filter1d(strength,5,mode='nearest')
            good &= strength>=max(3.,cfg.snr_db)
        valid=np.flatnonzero(good)
        boundaries=np.flatnonzero(np.diff(valid)>1)+1
        segments=[]
        for supported in np.split(valid,boundaries):
            for part in np.split(supported,np.searchsorted(supported,cuts)):
                if len(part):segments.append(part)
        for part in segments:
            if len(part)<2 or tr.t[part[-1]]-tr.t[part[0]]<cfg.min_duration_s:continue
            frequency=tr.f[part]
            if tr.kind=='doppler' and not _moving(frequency,cfg):continue
            if tr.kind=='stationary' and np.quantile(frequency,.95)-np.quantile(frequency,.05)>=max(cfg.min_sweep_hz,1.5,.01*np.median(frequency)):continue
            bounded.append(Track(tr.t[part],frequency,tr.amp[part],tr.doppler_score,None,tr.harmonic,tr.group,tr.kind))
    verified=_clip_verified(bounded,verification,t,f,cfg)
    return _continuous_combs(verified,S,t,f,cfg,verification)


# Offline null calibration for the default 30–2000 Hz grids. These are
# exactly the eight seeded white/pink/hum trials produced by the fallback
# below (seed 90917+160, 40 s, v_max=120, d_min=500, two velocity beams).
# Refit after changing framing, evidence, or the HMM. Storing the learned
# maxima avoids recalibrating the same model at every CLI process startup.
_DEFAULT_NULL_CALIBRATION = {0.5: {'maxima': [0.0, 2.7047182286586935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       'cutoff': 40.0,
       'single_maxima': [16.068665031019123,
                         15.294031742546034,
                         4.259255417975385,
                         8.256705125286787,
                         10.976035729027865,
                         10.603175727774989,
                         11.478782473090924,
                         7.46723531299401],
       'single_cutoff': 60.0,
       'high_maxima': [16.30837488911368,
                       6.3620466860168525,
                       4.49896527606994,
                       8.496414983381342,
                       11.21574558712242,
                       10.842885585869544,
                       11.71849233118548,
                       7.386963933860974],
       'high_cutoff': 48.65971682937699},
 1.0: {'maxima': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
       'cutoff': 40.0,
       'single_maxima': [18.13069510723692,
                         6.6561868301955025,
                         5.462674936353334,
                         7.794740917632538,
                         9.042107865985667,
                         4.7009962461923065,
                         7.721410165705482,
                         12.610259897808795],
       'single_cutoff': 60.0,
       'high_maxima': [18.3703362512131,
                       6.8958279741716835,
                       5.702316080329515,
                       7.647175485921191,
                       9.28174900996185,
                       4.9406373901684875,
                       7.961051309681663,
                       7.618567243892188],
       'high_cutoff': 40.0}}

def _reference_calibration(frequencies, frames):
    # A longer null trial bounds all shorter edge windows and avoids fitting
    # another eight backgrounds for each final partial window.
    if frames<=160 and len(frequencies)>1 and frequencies[0]==30. and frequencies[-1]==2000.:
        spacing=round(frequencies[1]-frequencies[0],6)
        if spacing in _DEFAULT_NULL_CALIBRATION:
            return _DEFAULT_NULL_CALIBRATION[spacing]
    return _reference_calibration_cached(frequencies,max(160,frames))


@lru_cache(maxsize=16)
def _reference_calibration_cached(frequencies, frames):
    """Gumbel null-maxima fit on aircraft-free STFTs, not on the target audio.

    Deterministic white/pink realizations, half with stationary hum, preserve
    the STFT's frequency/time correlations. Cached by grid and frame count.
    Speech/music are separate adversarial null regressions because stationary
    tonal maxima are not samples from the noise-only exponential model.
    """
    f=np.asarray(frequencies);dt=.25
    cfg=DetectConfig(fmin=float(f[0])-.001,fmax=float(f[-1])+.001)
    rate=max(4000,int(np.ceil(2.4*cfg.fmax)))
    t=np.arange(frames)*dt
    bf=np.arange(8.,min(500.,f[-1]/2),np.median(np.diff(f)))
    rng=np.random.default_rng(90917+frames)
    maxima=[];single=[];high_scores=[]
    n=int((frames*.25+4)*rate)
    for trial in range(8):
        audio=rng.normal(size=n)
        if trial%2:
            ff=np.fft.rfftfreq(n,1/rate)
            audio=np.fft.irfft(np.fft.rfft(audio)/np.sqrt(np.maximum(ff,1)),n=n)
            audio/=max(np.std(audio),1e-12)
        if trial%3==0:
            audio+=2*np.sin(2*np.pi*60*np.arange(n)/rate)
        spec,_,rf=ridge_spectrogram(audio,rate,cfg)
        raw=np.asarray(spec[:,8:8+frames])
        if len(rf)!=len(f) or not np.allclose(rf,f,atol=1e-4):
            raw=np.stack([np.interp(f,rf,raw[:,j]) for j in range(raw.shape[1])],axis=1)
        evidence=_evidence(raw,f,None)
        obs=_robust_llr(evidence)
        bank=np.zeros((len(bf),frames));norm=np.zeros(len(bf))
        for h in range(1,25):
            index=np.rint((h*bf-f[0])/(f[1]-f[0])).astype(int)
            good=(index>=0)&(index<len(f));w=1/np.sqrt(h)
            bank[good]+=w*obs[index[good]];norm[good]+=w*w
        bank/=np.sqrt(np.maximum(norm[:,None],1))
        maxima.append(_viterbi_lines(bank,t,bf,2.,limit=1,observation=bank,score_only=True))
        single.append(_viterbi_lines(evidence,t,f,4.,limit=1,score_only=True))
        high=np.flatnonzero(f>=450.)
        high_scores.append(_viterbi_lines(evidence[high],t,f[high],4.,limit=1,score_only=True) if len(high)>3 else 0.)
    def quantile(values,floor):
        if np.ptp(values)<1e-6:return max(floor,max(values)+20.)
        loc,scale=stats.gumbel_r.fit(values)
        return max(floor,float(stats.gumbel_r.ppf(1-1e-6,loc=loc,scale=scale)))
    return dict(maxima=list(map(float,maxima)),cutoff=quantile(maxima,40.),
                single_maxima=list(map(float,single)),single_cutoff=quantile(single,60.),
                high_maxima=list(map(float,high_scores)),high_cutoff=quantile(high_scores,40.))


def _clip_verified(tracks, evidence, t, f, cfg):
    """Final unbiased on-curve verification; no maximum over nearby bins.

    Maximizing over a wide frequency corridor also maximizes noise and can
    falsely verify every high order. Interpolate the exact frequency instead.
    Shared-factor predictions pay no further path-search penalty; independent
    lines must retain positive continuation evidence as their source fades.
    """
    groups={}
    for tr in tracks:
        if tr.kind=='doppler':groups.setdefault(tr.group,set()).add(tr.harmonic)
    out=[];df=float(np.median(np.diff(f)))
    for tr in tracks:
        if tr.kind!='doppler':out.append(tr);continue
        ti=np.interp(tr.t,t,np.arange(len(t)))
        fi=np.clip((tr.f-f[0])/df,0,len(f)-1)
        t0=np.floor(ti).astype(int);t1=np.minimum(t0+1,len(t)-1)
        f0=np.floor(fi).astype(int);f1=np.minimum(f0+1,len(f)-1)
        wt=ti-t0;wf=fi-f0
        snr=(1-wt)*((1-wf)*evidence[f0,t0]+wf*evidence[f1,t0])+wt*((1-wf)*evidence[f0,t1]+wf*evidence[f1,t1])
        llr=_robust_llr(snr)
        if len(groups.get(tr.group,()))<2:
            smooth=ndimage.gaussian_filter1d(tr.f,1.)
            acceleration=np.r_[0.,0.,np.diff(smooth,n=2)]
            sigma=np.maximum(df,.25*tr.f*120**3/(343*500**2)*.25**2)
            llr-=.5*(acceleration/sigma)**2+np.log(np.sqrt(2*np.pi)*sigma/df)
        visible=(ndimage.uniform_filter1d(snr,7,mode='nearest')>=cfg.snr_db)&(ndimage.uniform_filter1d(llr,7,mode='nearest')>0)
        if len(tr.t)>=9:
            window=min(17,len(tr.t) if len(tr.t)%2 else len(tr.t)-1)
            slope=signal.savgol_filter(tr.f,window,2,deriv=1,delta=.25)
            active=np.abs(slope)>=max(.04,.0002*np.median(tr.f))
            active=ndimage.binary_closing(active,structure=np.ones(7,bool))
            visible &= active
        indices=np.flatnonzero(visible)
        for part in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            if len(part)<2 or tr.t[part[-1]]-tr.t[part[0]]<cfg.min_duration_s:continue
            if not _moving(tr.f[part],cfg):continue
            score=.70 if tr.doppler_score==.55 else tr.doppler_score
            out.append(Track(tr.t[part],tr.f[part],tr.amp[part],score,None,tr.harmonic,tr.group,'doppler'))
    return sorted(_consolidate(out,df,bridge=False),key=lambda tr:(tr.group,tr.harmonic,tr.t0))


def _continuous_combs(tracks, S, t, f, cfg, evidence=None):
    """Expand only a confirmed solid, observed h1, never an inferred subharmonic.

    Continuous proposals deliberately bypass per-order SNR clipping: removal's
    coherence gate decides which proposed orders are actually present. Parent
    support and shape are checked independently to prevent speech/bird fragments
    or a virtual 41 Hz subharmonic from seeding a large comb.
    """
    output=[tr for tr in tracks if tr.kind!='doppler' or tr.doppler_score>=cfg.doppler_threshold]
    families={}
    for tr in output:
        if tr.kind=='doppler':families.setdefault(tr.group,[]).append(tr)
        else:tr.doppler_score=0.
    amplitude=getattr(S,'amplitude',None);df=float(np.median(np.diff(f)))
    if evidence is None:evidence=_evidence(np.asarray(S,dtype=np.float32),f,None)
    for group,members in families.items():
        # Score .70 marks a reconstructed order, not an observed root.
        h1=[tr for tr in members if tr.harmonic==1 and not np.isclose(tr.doppler_score,.70)]
        if not h1:continue
        begin=min(tr.t0 for tr in h1);end=max(tr.t1 for tr in h1)
        if end-begin<max(10.,2*cfg.min_duration_s):continue
        grid=np.arange(begin,end+.01,.25)
        h1_bank=np.stack([np.interp(grid,tr.t,tr.f,left=np.nan,right=np.nan) for tr in h1])
        observed=np.sum(np.isfinite(h1_bank),axis=0)>0
        if observed.mean()<.65:continue
        anchor=np.full(len(grid),np.nan)
        anchor[observed]=np.nanmedian(h1_bank[:,observed],axis=0)
        # Sub-40 Hz roots are too ambiguous for blind comb expansion (wind,
        # codec residue, and virtual subharmonics); retain observed lines only.
        if np.median(anchor[observed])<40.:continue
        ti=np.clip(np.searchsorted(t,grid[observed]),0,len(t)-1)
        loc=np.clip((anchor[observed]-f[0])/df,0,len(f)-1)
        low=np.floor(loc).astype(int);high=np.minimum(low+1,len(f)-1);weight=loc-low
        snr=(1-weight)*evidence[low,ti]+weight*evidence[high,ti]
        if np.median(snr)<max(7.,cfg.snr_db+3) or np.mean(snr>=max(4.5,cfg.snr_db+1.5))<.7:continue
        # Require an observed sibling, independently of whether the line
        # tracker fragmented it. A solitary chirp does not establish a comb.
        # Interpolate exactly on each harmonic; never maximize over noise bins.
        sibling=False
        for order in range(2,13):
            frequency=order*anchor[observed]
            valid=(frequency>=f[0])&(frequency<=f[-1])
            if valid.mean()<.8:continue
            location=(frequency[valid]-f[0])/df
            lower=np.floor(location).astype(int);upper=np.minimum(lower+1,len(f)-1)
            fraction=location-lower
            contrast=(1-fraction)*evidence[lower,ti[valid]]+fraction*evidence[upper,ti[valid]]
            threshold=max(5.,cfg.snr_db+2.)
            if np.median(contrast)>=threshold and np.mean(contrast>=threshold)>=.65:
                sibling=True;break
        if not sibling:continue
        # Jointly refine the shared factor where other confirmed orders bridge
        # a short h1 fade, while retaining h1's observed time bounds.
        bank=np.stack([np.interp(grid,tr.t,tr.f/max(1,tr.harmonic),left=np.nan,right=np.nan) for tr in members])
        present=np.sum(np.isfinite(bank),axis=0)>0
        latent=np.full(len(grid),np.nan)
        latent[present]=np.nanmedian(bank[:,present],axis=0)
        missing=np.flatnonzero(~present)
        for gap in np.split(missing,np.flatnonzero(np.diff(missing)>1)+1):
            if 0<len(gap)<=6 and gap[0]>0 and gap[-1]<len(grid)-1:
                latent[gap]=np.interp(grid[gap],grid[present],latent[present]);present[gap]=True
        indices=np.flatnonzero(present)
        for span in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            if len(span)<2 or grid[span[-1]]-grid[span[0]]<max(10.,2*cfg.min_duration_s):continue
            base=latent[span];gt=grid[span]
            smooth=ndimage.gaussian_filter1d(base,2.)
            excursion=float(np.quantile(smooth,.95)-np.quantile(smooth,.05))
            prominence=max(1.5,.01*np.median(smooth))
            turns=len(signal.find_peaks(smooth,prominence=prominence)[0])+len(signal.find_peaks(-smooth,prominence=prominence)[0])
            if turns/(gt[-1]-gt[0])>.1 or np.sum(abs(np.diff(smooth)))>4*max(excursion,1e-6):continue
            if not _moving(base,cfg):continue
            confidence=max(cfg.doppler_threshold,.90,max(tr.doppler_score for tr in h1))
            for order in range(1,int(cfg.fmax/np.min(base))+1):
                frequency=order*base
                valid=(frequency>=cfg.fmin)&(frequency<=min(cfg.fmax,f[-1]))
                selected=np.flatnonzero(valid)
                for part in np.split(selected,np.flatnonzero(np.diff(selected)>1)+1):
                    if len(part)<2 or gt[part[-1]]-gt[part[0]]<cfg.min_duration_s:continue
                    tt=gt[part];ff=frequency[part]
                    if np.quantile(ff,.95)-np.quantile(ff,.05)<max(cfg.min_sweep_hz,1.5,.01*np.median(ff)):continue
                    sigma=np.maximum(df,(ff[1:]+ff[:-1])/2*120**2/(343*500)*.25)
                    for piece in np.split(np.arange(len(tt)),np.flatnonzero(abs(np.diff(ff))>3*sigma)+1):
                        if len(piece)<2 or tt[piece[-1]]-tt[piece[0]]<cfg.min_duration_s:continue
                        time=tt[piece];freq=ff[piece]
                        if amplitude is None:amp=np.zeros(len(time))
                        else:
                            ti=np.clip(np.searchsorted(t,time),0,len(t)-1)
                            fi=np.clip(np.rint((freq-f[0])/df).astype(int),0,len(f)-1)
                            amp=amplitude[fi,ti].copy()
                        output.append(Track(time,freq,amp,confidence,None,order,group,'doppler'))
    return sorted(_consolidate(output,df,bridge=False),key=lambda tr:(tr.group,tr.harmonic,tr.t0))
