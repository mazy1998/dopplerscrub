"""Non-parametric ridge correctness, negative controls, and optional real views."""
import json
from pathlib import Path
import numpy as np
import pytest
from dopplerscrub.ridges import ridge_spectrogram, track_ridges, RidgeSpectrum
from dopplerscrub.types import DetectConfig
from dopplerscrub.synth import mix


def _stack():
    t=np.arange(0,24,.25);f=np.arange(30,1600,.5)
    fundamental=85+11*np.sin(2*np.pi*t/19)+.3*t
    S=np.random.default_rng(22).normal(0,1,(len(f),len(t)))
    for k in range(1,13):
        # Two-second gap in one weak member; the other orders remain visible.
        level=18 if k<8 else 11
        line=level*np.exp(-.5*((f[:,None]-k*fundamental[None,:])/1.1)**2)
        if k==9: line[:,(t>10)&(t<12)]=0
        S+=line
    return S,t,f,fundamental


def test_maneuvering_stack_and_gap():
    S,t,f,fundamental=_stack()
    tracks=track_ridges(S,t,f,DetectConfig(fmax=1600))
    moving=[tr for tr in tracks if tr.kind=='doppler']
    coverage=[]
    for k in range(1,13):
        errors=[]
        for tr in moving:
            if tr.t1-tr.t0<10: continue
            errors.append(np.median(abs(tr.f-k*np.interp(tr.t,t,fundamental))))
        coverage.append(bool(errors) and min(errors)<2)
    assert all(coverage), coverage
    assert all(np.allclose(np.diff(tr.t),.25) for tr in tracks)
    assert all(np.all(tr.amp==0) for tr in tracks)  # absolute units unknowable from dB
    assert len({tr.group for tr in moving})==1


def test_magnitude_calibration_and_stationary():
    sr=4000;t=np.arange(sr*12)/sr
    x=.1*np.sin(2*np.pi*97.65625*t)
    cfg=DetectConfig(fmax=600)
    S,tt,f=ridge_spectrogram(x,sr,cfg)
    assert isinstance(S,RidgeSpectrum)
    assert np.median(np.diff(f))<=.5
    assert np.max(S.amplitude)==pytest.approx(.1,rel=.01)
    assert S[1:4,2:6].amplitude.shape==(3,4)
    tracks=track_ridges(S,tt,f,cfg)
    assert tracks and not any(tr.kind=='doppler' for tr in tracks)
    strongest=max(tracks,key=lambda tr:np.max(tr.amp))
    assert np.median(strongest.amp)==pytest.approx(.1,rel=.02)


@pytest.mark.parametrize('background',['pink','speech','music','hum'])
def test_negative_backgrounds(background):
    sr=4000;x=np.zeros(sr*20)
    levels=dict(pink=0,wind=0,hum=0,speech=0,music=0)
    if background=='hum':levels.update(hum=.03,mains_hz=60)
    else: levels[background]=.04
    x=mix(x,sr,seed=890,**levels)
    cfg=DetectConfig(fmax=1500)
    S,t,f=ridge_spectrogram(x,sr,cfg)
    tracks=track_ridges(S,t,f,cfg)
    assert not [tr for tr in tracks if tr.kind=='doppler'], [(tr.t0,tr.t1,np.ptp(tr.f),tr.harmonic) for tr in tracks if tr.kind=='doppler']


def test_birth_death_and_short_rejection():
    t=np.arange(0,20,.25);f=np.arange(30,450,.5)
    S=np.zeros((len(f),len(t)))
    for center in [100,200,300]:
        ridge=25*np.exp(-.5*((f[:,None]-(center+2*t)[None,:])/.6)**2)
        ridge[:,(t<5)|(t>6)]=0
        S+=ridge
    assert not track_ridges(S,t,f,DetectConfig(fmax=450))


def test_invalid_grid():
    with pytest.raises(ValueError):track_ridges(np.zeros((2,3)),[0,1,2],[1,2],DetectConfig())


@pytest.mark.slow
def test_real_overlays():
    """Run explicitly with -k real_overlays to regenerate inspected artifacts.

    This verifies bounded valid outputs, not ground-truth recall: annotations
    describe bands/windows, not digitized frequency curves.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import subprocess
    from dopplertrack.audio import compute_spectrogram
    root=Path(__file__).resolve().parents[1]
    clips=sorted((root/'data/real/airshow').glob('*.wav'))
    clips += [root/'data/real'/f'{n}.wav' for n in ['prop_flyover','prop_pass','f1_pass']]
    clips=[p for p in clips if p.exists()]
    if not clips:pytest.skip('local real-audio fixtures are not downloaded')
    cfg=DetectConfig(fmax=2000)
    results=[]
    for path in clips:
        raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-ac','1','-ar','4800','-f','f32le','-'])
        x=np.frombuffer(raw,dtype='<f4')
        S,t,f=ridge_spectrogram(x,4800,cfg)
        tracks=track_ridges(S,t,f,cfg)
        assert all(np.isfinite(tr.f).all() and np.isfinite(tr.amp).all() for tr in tracks)
        assert all(tr.kind in ('doppler','stationary') for tr in tracks)
        fig,axes=plt.subplots(2,1,figsize=(14,8))
        fig.subplots_adjust(left=.07,right=.92,bottom=.07,top=.9,hspace=.3)
        # The underlay is the actual attacker's spectrogram, not our multiresolution STFT.
        from scipy.signal import resample_poly
        attacker=compute_spectrogram(resample_poly(x,5,6),4000,fmin=30,fmax=2000)
        for ax,hi in zip(axes,[450,2000]):
            ax.imshow(attacker.S,origin='lower',aspect='auto',extent=[attacker.t[0],attacker.t[-1],attacker.f[0],attacker.f[-1]],vmin=-5,vmax=18,cmap='magma')
            for tr in tracks:
                ax.plot(tr.t,tr.f,color='cyan' if tr.kind=='doppler' else 'lime',lw=.65,alpha=.8)
            ax.set(ylim=(30,hi),xlabel='Time (s)',ylabel='Hz',title=f'30–{hi} Hz; cyan=moving, green=stationary')
        fig.suptitle(path.stem+' — non-parametric ridge overlays')
        target=root/'docs/img'/f'ridges_{path.stem}.png';fig.savefig(target,dpi=110);plt.close(fig)
        results.append(dict(path=str(path.relative_to(root)),moving=sum(tr.kind=='doppler' for tr in tracks),stationary=sum(tr.kind=='stationary' for tr in tracks),tracks=[tr.to_dict() for tr in tracks]))
        print(path.stem,results[-1]['moving'],flush=True)
    (root/'data/real/ridge_results.json').write_text(json.dumps(results,indent=2)+'\n')


@pytest.mark.parametrize('seed',[12,39,702])
def test_combined_negatives(seed):
    sr=4800
    x=mix(np.zeros(sr*18),sr,pink=.015,wind=.005,hum=.004,mains_hz=60,
          speech=.025,music=.015,seed=seed)
    cfg=DetectConfig(fmax=2000)
    S,t,f=ridge_spectrogram(x,sr,cfg)
    tracks=track_ridges(S,t,f,cfg)
    assert not [tr for tr in tracks if tr.kind=='doppler']


def test_audio_stack_missing_orders():
    sr=4800;tt=np.arange(sr*24)/sr
    base=90+10*np.sin(.23*tt)+.4*tt
    phase=2*np.pi*np.cumsum(base)/sr
    # Orders 2 and 7 are deliberately absent. A confirmed stack must still
    # emit their shared-factor predictions for removal/anti-forensic checking.
    x=sum(.025/k*np.sin(k*phase) for k in range(1,13) if k not in (2,7))
    cfg=DetectConfig(fmax=1600)
    S,t,f=ridge_spectrogram(x,sr,cfg)
    tracks=track_ridges(S,t,f,cfg)
    moving=[tr for tr in tracks if tr.kind=='doppler']
    for order in range(1,13):
        matches=[tr for tr in moving if tr.t1-tr.t0>10 and
                 np.median(abs(tr.f-order*np.interp(tr.t,tt,base)))<3]
        assert matches, f'missing order {order}'
