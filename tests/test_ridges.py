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
    for tr in tracks:
        sigma=np.maximum(.5,(tr.f[1:]+tr.f[:-1])/2*120**2/(343*500)*np.diff(tr.t))
        assert np.all(np.abs(np.diff(tr.f))<=3*sigma+1e-9)
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
    assert all(tr.doppler_score==0 for tr in tracks)
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
    clips += [root/'data/real'/f'{n}.wav' for n in ['prop_flyover','prop_pass','f1_pass','colsto_2050_2250','colsto_2980_3170']]
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
                ax.plot(tr.t,tr.f,color='cyan' if tr.kind=='doppler' else 'lime',lw=.7,alpha=.5 if tr.doppler_score==.7 else .8,ls='--' if tr.doppler_score==.7 else '-')
            ax.set(ylim=(30,hi),xlabel='Time (s)',ylabel='Hz',title=f'30–{hi} Hz; cyan=moving (dashed=verified prediction), green=stationary')
        fig.suptitle(path.stem+' — non-parametric ridge overlays')
        target=root/'docs/img'/f'ridges_{path.stem}.png';target.parent.mkdir(parents=True,exist_ok=True);fig.savefig(target,dpi=110);plt.close(fig)
        results.append(dict(calibration=S.calibration,path=str(path.relative_to(root)),moving=sum(tr.kind=='doppler' for tr in tracks),stationary=sum(tr.kind=='stationary' for tr in tracks),tracks=[tr.to_dict() for tr in tracks]))
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
    # Orders 2 and 7 are absent. The latest diagnosis requires locally
    # evidenced proposals, so missing orders must not create corridors.
    x=sum(.025/k*np.sin(k*phase) for k in range(1,13) if k not in (2,7))
    cfg=DetectConfig(fmax=1600)
    S,t,f=ridge_spectrogram(x,sr,cfg)
    tracks=track_ridges(S,t,f,cfg)
    moving=[tr for tr in tracks if tr.kind=='doppler']
    for order in [k for k in range(1,13) if k not in (2,7)]:
        matches=[tr for tr in moving if tr.t1-tr.t0>10 and
                 np.median(abs(tr.f-order*np.interp(tr.t,tt,base)))<(4.1 if order>=8 else 3)]
        assert matches, f'missing order {order}'
    assert all(tr.doppler_score>=cfg.doppler_threshold for tr in moving)


def test_link_gaps_and_duplicate_corridors():
    from dopplerscrub.ridges import _consolidate
    from dopplerscrub.types import Track
    def line(t, delta=0):return Track(t,100+2*t+delta,np.ones(len(t))*.01,.9,harmonic=1,group=0,kind='doppler')
    a=line(np.arange(0,5,.25));b=line(np.arange(6,12.25,.25))
    duplicate=line(np.arange(2,8,.25),.2)
    linked=_consolidate([a,b,duplicate],.5)
    assert len(linked)==1
    assert linked[0].t0==0 and linked[0].t1==12
    assert np.allclose(np.diff(linked[0].t),.25)
    assert np.max(abs(linked[0].f-(100+2*linked[0].t)))<=.2+1e-10
    separated=_consolidate([a,line(np.arange(8,12,.25))],.5)
    assert len(separated)==2  # no prediction across a >1.5-second missing interval


@pytest.mark.slow
@pytest.mark.parametrize('mode',['noise','hum','music','speech'])
def test_two_minute_null_calibration(mode):
    sr=4000;t=np.arange(sr*120)/sr
    x=np.random.default_rng(218).normal(0,.03,len(t))
    if mode=='hum':x+=.08*np.sin(2*np.pi*60*t)+.03*np.sin(2*np.pi*120*t)
    if mode=='music':
        for j in range(30):
            base=[110,130.81,164.81,146.83,110,98][j%6]
            m=(t>=4*j)&(t<4*(j+1));envelope=np.sin(np.pi*(t[m]-4*j)/4)**2
            for k in (1,2,3):x[m]+=.03/k*envelope*np.sin(2*np.pi*k*base*t[m])
    if mode=='speech':x=mix(x,sr,pink=0,wind=0,hum=0,speech=.06,seed=24)
    cfg=DetectConfig()
    S,tt,f=ridge_spectrogram(x,sr,cfg);tracks=track_ridges(S,tt,f,cfg)
    moving=[tr for tr in tracks if tr.kind=='doppler']
    result=dict(mode=mode,doppler=len(moving),stationary=sum(tr.kind=='stationary' for tr in tracks),calibration=S.calibration)
    out=Path(__file__).resolve().parents[1]/'data/real'
    out.mkdir(parents=True,exist_ok=True)
    (out/f'ridges_null_{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not moving, [(tr.t0,tr.t1,np.ptp(tr.f)) for tr in moving]


def test_exponential_power_llr():
    from dopplerscrub.ridges import _llr
    db=np.array([-50.,-10.,0.,10.,20.,30.])
    for snr in (1.,3.,10.):
        z=np.log(2)*10**(db/10)
        np.testing.assert_allclose(_llr(db,snr),z*snr/(1+snr)-np.log1p(snr))


def test_high_only_and_native_nyquist_cap():
    sr=4000;t=np.arange(sr*3)/sr
    x=.1*np.sin(2*np.pi*700*t)
    S,tt,f=ridge_spectrogram(x,sr,DetectConfig(fmin=500,fmax=2000))
    assert S.shape==(len(f),len(tt))
    assert f[0]>=500 and f[-1]<sr/2
    assert np.isfinite(S).all()
    assert np.max(S.amplitude)==pytest.approx(.1,rel=.02)


def test_motion_rejects_flat_drift_and_accepts_pass():
    from dopplerscrub.ridges import _moving
    cfg=DetectConfig();t=np.arange(0,35,.25)
    assert not _moving(230+.10*t,cfg)
    assert not _moving(180+.4*np.sin(t),cfg)
    assert _moving(230-25*np.tanh((t-17)/6),cfg)
    assert _moving(110+10*t/t[-1],cfg)


def test_confirmed_source_keeps_supported_approach_and_recede_plateaus():
    from dopplerscrub.ridges import _clip_verified
    from dopplerscrub.types import Track
    cfg=DetectConfig();t=np.arange(0,60,.25);f=np.arange(30,450,.5)
    freq=np.where(t<20,230,np.where(t>40,200,230-1.5*(t-20)))
    S=18*np.exp(-.5*((f[:,None]-freq[None,:])/.8)**2)
    tr=Track(t,freq,np.full(len(t),.01),.95,harmonic=1,group=0,kind='doppler')
    out=_clip_verified([tr],S,t,f,cfg)
    assert out and min(q.t0 for q in out)==t[0] and max(q.t1 for q in out)==t[-1]
    assert all(q.t1-q.t0>=cfg.min_duration_s for q in out)


@pytest.mark.parametrize('anchor', ['solid', 'short', 'virtual', 'weak', 'sub40', 'reconstructed', 'isolated'])
def test_continuous_comb_requires_solid_observed_root(anchor):
    from dopplerscrub.ridges import _continuous_combs
    from dopplerscrub.types import Track
    cfg=DetectConfig(fmax=650)
    t=np.arange(0,20.25,.25);f=np.arange(30,650.5,.5)
    root=(35 if anchor=='sub40' else 70)+4*np.tanh((t-10)/5)
    S=np.zeros((len(f),len(t)))
    for order in ((1,) if anchor=='isolated' else (1,3)):
        S+=(2 if anchor=='weak' else 18)*np.exp(-.5*((f[:,None]-order*root[None,:])/.7)**2)
    selected=t<=6 if anchor=='short' else np.ones(len(t),bool)
    orders=(3,5) if anchor=='virtual' else ((1,) if anchor=='isolated' else (1,3))
    tracks=[Track(t[selected],order*root[selected],np.zeros(selected.sum()),.70 if anchor=='reconstructed' else .95,
                  harmonic=order,group=7,kind='doppler') for order in orders]
    out=_continuous_combs(tracks,S,t,f,cfg)
    if anchor=='solid':
        assert not [tr for tr in out if tr.harmonic==2]  # absent, never proposed
        assert all(tr.doppler_score>=cfg.doppler_threshold for tr in out)
        assert all(tr.t0>=t[0] and tr.t1<=t[-1] for tr in out)
    else:
        assert len(out)==len(tracks), 'unconfirmed root seeded a harmonic comb'


def test_motion_does_not_treat_quantized_slow_sweep_as_note_holds():
    from dopplerscrub.ridges import _moving
    cfg=DetectConfig();t=np.arange(0,12,.25)
    root=np.round((58+3/(1+np.exp(-(t-8)/1.1)))*2)/2
    assert _moving(root,cfg)
    held_notes=np.where(t<6,110.,130.)
    assert not _moving(held_notes,cfg)


def test_orchestrator_music_note_sequence_is_not_doppler():
    # Exact red-team fixture: unwindowed four-second notes plus seed-nine
    # noise. The 110 -> 98 Hz note change used to become a false HMM chirp.
    sr=4000;t=np.arange(sr*24)/sr
    x=np.random.default_rng(9).normal(0,.02,len(t))
    for j,note in enumerate([110,130.81,164.81,146.83,110,98]):
        mask=(t>=j*4)&(t<(j+1)*4)
        x[mask]+=.06*np.sin(2*np.pi*note*t[mask])
    cfg=DetectConfig(fmax=300)
    S,tt,f=ridge_spectrogram(x,sr,cfg)
    assert not [tr for tr in track_ridges(S,tt,f,cfg) if tr.kind=='doppler']


def test_final_motion_guard_checks_each_returned_segment():
    from dopplerscrub.ridges import _motion_guard
    from dopplerscrub.types import Track
    t=np.arange(0,10,.25)
    flat=Track(t,376+.3*np.sin(t),np.zeros(len(t)),.95,kind='doppler')
    moving=Track(t,180-3*t,np.zeros(len(t)),.95,kind='doppler')
    out=_motion_guard([flat,moving],DetectConfig())
    assert out[0].kind=='stationary' and out[0].doppler_score==0
    assert out[1].kind=='doppler'


def test_faint_harmonic_is_emitted_only_over_its_observed_span():
    from dopplerscrub.ridges import _observed_harmonics
    from dopplerscrub.types import Track
    t=np.arange(0,24,.25);f=np.arange(30,650,.5)
    base=80+8*np.tanh((t-12)/4)
    S=18*np.exp(-.5*((f[:,None]-base[None,:])/.8)**2)
    visible=(t>=6)&(t<=18)
    S+=6*np.exp(-.5*((f[:,None]-2*base[None,:])/.8)**2)*visible
    parent=Track(t,base,np.zeros(len(t)),.95,harmonic=1,group=0,kind='doppler')
    out=_observed_harmonics([parent],S,t,f,DetectConfig(fmax=650),S)
    second=[tr for tr in out if tr.harmonic==2]
    assert second and second[0].t0>=5 and second[-1].t1<=19
    assert not [tr for tr in out if tr.harmonic>=3]


def test_continuation_ends_instead_of_jumping_against_predicted_slope():
    from dopplerscrub.ridges import _continue_lines
    from dopplerscrub.types import Track
    t=np.arange(0,30,.25);f=np.arange(30,450,.5)
    E=np.zeros((len(f),len(t)))
    visible=(t>=15)&(t<=20)
    E+=18*np.exp(-.5*((f[:,None]-(100+2*t)[None,:])/.5)**2)*visible
    E+=25*np.exp(-.5*((f[:,None]-132)/.5)**2)*(t>20)
    seed=Track(t[visible],100+2*t[visible],np.zeros(visible.sum()),.95,kind='candidate')
    out=_continue_lines([seed],E,t,f,DetectConfig())
    assert out[0].t1==20


def test_duplicate_merge_never_averages_divergent_frequency_branches():
    from dopplerscrub.ridges import _consolidate
    from dopplerscrub.types import Track
    t=np.arange(0,10,.25);frequency=100+2*t
    a=Track(t,frequency,np.full(len(t),.02),.95,kind='doppler')
    b=Track(t,frequency+np.where(t<6,.2,7.),np.full(len(t),.001),.95,kind='doppler')
    out=_consolidate([a,b],.5)
    assert len(out)==1
    assert np.max(abs(out[0].f-frequency))<=.2


def test_small_frequency_estimation_jitter_is_not_source_motion():
    from dopplerscrub.ridges import _moving
    f=np.array([165.16,164.29,164.67,163.19,163.29,163.29,164.79,164.79,
                163.82,164.79,164.79,165.28,164.79,164.79,164.79,164.79,
                164.79,164.73,164.68,164.41,164.21])
    assert np.ptp(f)>1.5
    assert not _moving(f,DetectConfig())


def test_sparse_tracker_cannot_alias_across_dark_harmonic_gaps():
    from dopplerscrub.ridges import _viterbi_lines
    t=np.arange(0,8,.25);f=np.arange(80,120,.5)
    E=np.zeros((len(f),len(t)));amp=np.zeros_like(E)
    a=np.argmin(abs(f-100));b=np.argmin(abs(f-105))
    E[a,::2]=17;E[b,1::2]=17
    amp[a,:]=1; amp[b,:]=1
    assert _viterbi_lines(E,t,f,4,limit=1,min_score=24)
    assert not _viterbi_lines(E,t,f,4,limit=1,min_score=24,amplitude=amp)


def test_reconstructed_order_cannot_redefine_unobserved_source_root():
    from dopplerscrub.ridges import _regroup_sources
    from dopplerscrub.types import Track
    t=np.arange(0,20,.25);base=100+10*np.tanh((t-10)/4)
    tracks=[Track(t,k*base,np.zeros(len(t)),score,harmonic=k,group=2,kind='doppler')
            for k,score in [(2,.95),(4,.95),(3,.70)]]
    out=_regroup_sources(tracks)
    assert not any(tr.harmonic==1 for tr in out)
    assert len({tr.group for tr in out})==1
    assert sorted(tr.harmonic for tr in out)==[2,3,4]


def test_continuation_stops_at_frequency_grid_boundary():
    from dopplerscrub.ridges import _continue_lines
    from dopplerscrub.types import Track
    t=np.arange(0,30,.25);f=np.arange(30,150,.5)
    E=18*np.exp(-.5*((f[:,None]-(100+2*t)[None,:])/.6)**2)
    E[-1,t>24]=20
    selected=(t>=15)&(t<=20)
    seed=Track(t[selected],100+2*t[selected],np.zeros(selected.sum()),.95,kind='candidate')
    out=_continue_lines([seed],E,t,f,DetectConfig(fmax=150))
    assert np.min(out[0].f)>=f[0] and np.max(out[0].f)<=f[-1]


def test_constant_five_hz_sideband_is_a_separate_shared_factor_family():
    from dopplerscrub.ridges import _regroup_sources, _joint_source_factors
    from dopplerscrub.types import Track
    t=np.arange(0,30,.25);base=100+12*np.tanh((t-15)/5)
    primary=[Track(t,k*base,np.zeros(len(t)),.95,harmonic=k,kind='doppler') for k in (1,2,3)]
    side=[Track(t,k*base+5,np.zeros(len(t)),.95,harmonic=k,kind='doppler') for k in (2,3)]
    out=_regroup_sources(primary+side)
    assert len({tr.group for tr in primary})==1
    assert len({tr.group for tr in side})==1
    assert primary[0].group!=side[0].group
    factors=_joint_source_factors(out)
    assert len(factors)==2
    for tr in factors:
        np.testing.assert_allclose(tr.f,np.interp(tr.t,t,base),atol=1e-8)


@pytest.mark.parametrize('bins',[25,81,321])
def test_fast_frequency_median_matches_original_rank_filter(bins):
    from scipy import ndimage
    from dopplerscrub.ridges import _frequency_median
    values=np.random.default_rng(71).normal(size=(500,17)).astype(np.float32)
    expected=ndimage.median_filter(values,size=(bins,1),mode='nearest')
    np.testing.assert_array_equal(_frequency_median(values,bins),expected)
