import numpy as np
import pytest
from dopplerscrub.detect import detect_tracks, ideal_curve
from dopplerscrub.types import DetectConfig


def scene(db=-25, harmonics=(1,)):
    sr = 4000
    t = np.arange(40*sr)/sr
    rng = np.random.default_rng(8)
    # Pink background, normalized in the original signal's units.
    z = np.fft.rfft(rng.normal(size=len(t)))
    z /= np.sqrt(np.maximum(np.fft.rfftfreq(len(t),1/sr),1))
    noise = np.fft.irfft(z,n=len(t))
    noise *= .1/np.std(noise)
    f = ideal_curve(t,110,65,350,18)
    phase = 2*np.pi*np.cumsum(f)/sr
    x = noise + .02*np.sin(2*np.pi*60*t)
    for k in harmonics:
        x += .1*10**(db/20)*np.sqrt(2)*np.sin(k*phase)
    return x,sr


def test_physics():
    t = np.linspace(-20,20,100)
    v,d,tc = 65,350,0
    c = 331.3*np.sqrt(1+15/273.15)
    te = np.linspace(-100,100,100000)
    r = np.hypot(d,v*te)
    expected = np.interp(t,te+r/c,110*c/(c+v*v*te/r))
    np.testing.assert_allclose(ideal_curve(t,110,v,d,tc),expected,atol=1e-5)


@pytest.mark.parametrize('db',[-25,-35,-40])
def test_flyover(db):
    x,sr = scene(db)
    if db<=-35:
        from dopplertrack.audio import compute_spectrogram
        sp=compute_spectrogram(x,sr)
        curve=ideal_curve(sp.t,110,65,350,18)
        contrast=float(np.mean(sp.sample(sp.t,curve))-.5*np.mean(sp.sample(sp.t,curve-5)+sp.sample(sp.t,curve+5)))
        if contrast<1:
            pytest.xfail(f'attacker-view contrast {contrast:.3f} dB < 1 dB visibility criterion')
    tracks = [q for q in detect_tracks(x,sr,DetectConfig(fmax=300)) if q.kind=='doppler']
    assert tracks
    errors = [np.sqrt(np.mean((q.f-ideal_curve(q.t,110,65,350,18))**2)) for q in tracks]
    assert min(errors) < .3


def test_harmonics():
    x,sr = scene(-25,(1,2))
    tracks = [q for q in detect_tracks(x,sr,DetectConfig(fmax=300)) if q.kind=='doppler']
    assert any(a.group==b.group and a.harmonic!=b.harmonic for a in tracks for b in tracks)


@pytest.mark.parametrize('mode',['noise','hum','music'])
def test_negative(mode):
    sr=4000
    t=np.arange(sr*24)/sr
    x=np.random.default_rng(9).normal(0,.02,len(t))
    if mode=='hum':
        x+=.1*np.sin(2*np.pi*60*t)
    if mode=='music':
        for j,f in enumerate([110,130.81,164.81,146.83,110,98]):
            m=(t>=j*4)&(t<(j+1)*4)
            x[m]+=.06*np.sin(2*np.pi*f*t[m])
    assert not any(q.kind=='doppler' for q in detect_tracks(x,sr,DetectConfig(fmax=300)))


def test_empty_and_stereo():
    assert detect_tracks(np.zeros((100,2)),4000)==[]
    with pytest.raises(ValueError):
        detect_tracks(np.array([np.nan]),4000)


def test_stationary_reported():
    sr=4000
    t=np.arange(sr*30)/sr
    x=(.001+.099*(t<10))*np.sin(2*np.pi*60*t)
    tracks=detect_tracks(x,sr,DetectConfig(fmax=200))
    assert any(q.kind=='stationary' and abs(np.median(q.f)-60)<.1 for q in tracks)


def visible_metrics(truth, predicted, t, f, spectrogram):
    from dopplerscrub.detect import _sample
    visible=[]
    for tr in truth:
        curve=np.interp(t,tr['t'],tr['f'])
        mask=(_sample(spectrogram,f,curve)>0)&(t>=tr['t'][0])&(t<=tr['t'][-1])&(curve>=f[0])&(curve<=f[-1])
        if mask.any():visible.append((tr,curve,mask))
    hits=set();false=0
    for q in predicted:
        fractions=[]
        for tr,curve,mask in visible:
            close=(abs(np.interp(t,q.t,q.f)-curve)<1)&(t>=q.t0)&(t<=q.t1)
            fractions.append(np.mean(close[mask]))
        if fractions and max(fractions)>.7:hits.add(int(np.argmax(fractions)))
        else:false+=1
    fundamental=[j for j,(tr,_,_) in enumerate(visible) if tr['harmonic']==1]
    return np.array([len(hits),len(visible),sum(j in hits for j in fundamental),len(fundamental),false])


@pytest.mark.slow
def test_shared_dataset():
    import json
    from pathlib import Path
    import soundfile as sf
    from dopplerscrub.detect import _spectrogram
    path=Path('data/synth/truth.json')
    if not path.exists():pytest.skip('shared synthesis fixtures not generated')
    totals=np.zeros(5,dtype=int)
    for row in json.loads(path.read_text()):
        x,sr=sf.read(row['path'])
        predicted=[q for q in detect_tracks(x,sr) if q.kind=='doppler']
        t,f,_,s=_spectrogram(x.mean(axis=1) if x.ndim==2 else x,sr,DetectConfig())
        result=visible_metrics(row['tracks'],predicted,t,f,s)
        totals+=result
        print(row['path'],result.tolist(),flush=True)
    recall=totals[0]/max(totals[1],1);fundamental=totals[2]/max(totals[3],1)
    print('visible recall',recall,'fundamental recall',fundamental,'false',totals[4],flush=True)
    assert fundamental>=.9 and recall>=.7 and totals[4]<=1


def test_ridge_merge_contract(monkeypatch):
    import sys,types
    from dopplerscrub.types import Track
    called=[]
    def tracker(s,t,f,cfg):
        called.append(s.shape)
        grid=np.arange(2.,8.,.25)
        return [Track(grid,np.linspace(100,90,len(grid)),np.ones(len(grid))*.01,.8,kind='doppler')]
    monkeypatch.setitem(sys.modules,'dopplerscrub.ridges',types.SimpleNamespace(track_ridges=tracker))
    out=detect_tracks(np.zeros(4000*10),4000,DetectConfig(fmax=200))
    assert called and not any(q.kind=='doppler' for q in out)


def test_visible_metric_ignores_outside_band():
    t=np.arange(0,10,.25);f=np.arange(30.,200.)
    truth=[dict(t=t.tolist(),f=np.full(len(t),20.).tolist(),harmonic=1)]
    result=visible_metrics(truth,[],t,f,np.ones((len(f),len(t))))
    assert result[1]==0


@pytest.mark.slow
@pytest.mark.parametrize('mode',['noise','hum','music'])
def test_two_minute_null(mode):
    sr=4000
    t=np.arange(sr*120)/sr
    x=np.random.default_rng(218).normal(0,.03,len(t))
    if mode=='hum':
        x+=.08*np.sin(2*np.pi*60*t)+.03*np.sin(2*np.pi*120*t)
    if mode=='music':
        for j in range(30):
            base=[110,130.81,164.81,146.83,110,98][j%6]
            m=(t>=4*j)&(t<4*(j+1))
            envelope=np.sin(np.pi*(t[m]-4*j)/4)**2
            for k in (1,2,3):x[m]+=.03/k*envelope*np.sin(2*np.pi*k*base*t[m])
    found=detect_tracks(x,sr)
    assert not any(q.kind=='doppler' for q in found)


def test_llr_formula_and_shared_harmonics():
    from dopplerscrub.detect import _llr,_harmonic_spectrum
    db=np.array([0.,10.])
    np.testing.assert_allclose(_llr(db,1),np.log(2)*10**(db/10)/2-np.log(2))
    f=np.arange(30.,301.)
    s=np.zeros((len(f),8));s[np.isin(f,[50,100,150]),:]=10
    h=_harmonic_spectrum(s,f)
    assert h[np.argmin(abs(f-50))].mean()>h[np.argmin(abs(f-70))].mean()


@pytest.mark.slow
@pytest.mark.parametrize('start,end',[(2050,2250),(2980,3170)])
def test_real_colsto_events(start,end):
    from pathlib import Path
    import subprocess,json,time
    if not Path('data/real/colsto.wav').exists():pytest.skip('real fixture unavailable')
    sr=16000
    raw=subprocess.run(['ffmpeg','-v','error','-ss',str(start),'-t',str(end-start),'-i','data/real/colsto.wav','-ac','1','-ar',str(sr),'-f','f32le','-'],check=True,capture_output=True).stdout
    x=np.frombuffer(raw,dtype=np.float32)
    begin=time.monotonic();found=detect_tracks(x,sr);runtime=time.monotonic()-begin
    moving=[q for q in found if q.kind=='doppler']
    print('real crop',start,end,'runtime',runtime,'moving',len(moving),flush=True)
    Path(f'/tmp/colsto_{start}_tracks.json').write_text(json.dumps({'runtime_s':runtime,'tracks':[q.to_dict() for q in found]}))
    assert any(q.t1-q.t0>30 and 95<np.median(q.f)<200 and q.f[0]-q.f[-1]>15 for q in moving)
    if start==2980:
        for when,lower,upper in [(3050,310,355),(3097,215,240)]:
            assert any(q.harmonic==2 and q.t0<=when-start<=q.t1 and lower<=np.interp(when-start,q.t,q.f)<=upper for q in moving)
    for q in found:
        if q.kind=='stationary':assert q.t1-q.t0>=5 and np.std(q.f)<.5
        assert np.all(np.isfinite(q.f)) and np.max(q.f)<=DetectConfig().fmax
        if q.fit:
            assert q.fit['v']<=150 and q.fit['d_cpa']>=50
            assert q.fit['f0']/q.harmonic>=40


def test_empty_external_tracks_are_filtered(monkeypatch):
    import sys,types
    from dopplerscrub.types import Track
    def tracker(s,t,f,cfg):
        return [Track(np.array([]),np.array([]),np.array([]),kind='doppler')]
    monkeypatch.setitem(sys.modules,'dopplerscrub.ridges',types.SimpleNamespace(track_ridges=tracker))
    out=detect_tracks(np.zeros(4000*10),4000,DetectConfig(fmax=200))
    assert all(len(q.t)>=2 for q in out)


def test_stationary_hygiene():
    from dopplerscrub.detect import _spectrogram,_sample
    sr=4000;t=np.arange(sr*30)/sr
    x=(.001+.099*(t<10))*np.sin(2*np.pi*60*t)
    frequency=150+3*np.floor(t/2)
    x+=.02*np.sin(2*np.pi*np.cumsum(frequency)/sr)
    cfg=DetectConfig(fmax=300)
    tracks=detect_tracks(x,sr,cfg)
    stationary=[q for q in tracks if q.kind=='stationary']
    assert stationary
    times,f,_,s=_spectrogram(x,sr,cfg)
    for q in stationary:
        assert q.t1-q.t0>=5 and np.std(q.f)<.5
        m=(times>=q.t0)&(times<=q.t1)
        assert np.mean(_sample(s[:,m],f,np.interp(times[m],q.t,q.f)))>=3


def test_known_harmonics_bridge_parent_gap():
    from dopplerscrub.detect import _probe_known_harmonics
    from dopplerscrub.types import Track
    t=np.arange(0,30,.25);f=np.arange(30.,400.,.5)
    fundamental=130-t
    s=np.full((len(f),len(t)),-1.)
    for k in (2,3):
        s+=8*np.exp(-.5*((f[:,None]-k*fundamental[None,:])/.4)**2)
    a=t<=10;b=t>=13
    parents=[Track(t[m],fundamental[m],np.ones(m.sum())*.01,.9,None,1,7,'doppler') for m in (a,b)]
    found=_probe_known_harmonics(parents,t,f,np.ones_like(s)*.01,s,DetectConfig(fmax=400))
    for k in (2,3):
        assert any(q.harmonic==k and q.t0<10 and q.t1>13 and q.group==7 for q in found)


def test_verified_harmonic_order_above_ten():
    from dopplerscrub.detect import _probe_known_harmonics
    from dopplerscrub.types import Track
    t=np.arange(0,30,.25);f=np.arange(30.,2000.,.5)
    fundamental=110-.2*t
    s=np.full((len(f),len(t)),-1.)
    s+=9*np.exp(-.5*((f[:,None]-17*fundamental[None,:])/.4)**2)
    s+=9*np.exp(-.5*((f[:,None]-2*fundamental[None,:])/.4)**2)
    parent=Track(t,fundamental,np.ones(len(t))*.01,.9,None,1,4,'doppler')
    found=_probe_known_harmonics([parent],t,f,np.ones_like(s)*.01,s,DetectConfig())
    assert any(q.harmonic==17 and np.max(q.f)>1800 and q.group==4 for q in found)


def test_multiresolution_analysis_covers_two_khz():
    from dopplerscrub.detect import _spectrogram
    sr=8000;t=np.arange(4*sr)/sr
    result=_spectrogram(.1*np.sin(2*np.pi*1900*t),sr,DetectConfig())
    times,f,amp,s=result
    assert f[-1]>=1999 and amp.shape==s.shape==(len(f),len(times))
    assert abs(f[np.argmax(np.median(amp,axis=1))]-1900)<1


def test_confirmed_harmonic_follows_whole_parent_span():
    from dopplerscrub.detect import _probe_known_harmonics
    from dopplerscrub.types import Track
    t=np.arange(0,30,.25);f=np.arange(30.,400.,.5);base=130-.3*t
    s=np.full((len(f),len(t)),-1.)
    visible=(t>=5)&(t<=20)
    s+=4*np.exp(-.5*((f[:,None]-2*base[None,:])/.4)**2)*visible
    parent=Track(t,base,np.ones(len(t))*.01,.9,None,1,4,'doppler')
    sibling=Track(t,3*base,np.ones(len(t))*.01,.9,None,3,4,'doppler')
    tracks=_probe_known_harmonics([parent,sibling],t,f,np.ones_like(s)*.01,s,DetectConfig(fmax=400))
    tracks=[q for q in tracks if q.harmonic==2]
    assert tracks and all(q.t0==parent.t0 and q.t1==parent.t1 for q in tracks)


def test_final_motion_gate_and_cross_detector_dedupe():
    from dopplerscrub.detect import _motion_kind,_dedupe_tracks
    from dopplerscrub.types import Track
    cfg=DetectConfig();t=np.arange(0,40,.25)
    flat=Track(t,230+.1*np.sin(t),np.ones(len(t)),.9,None,1,1,'doppler')
    moving=Track(t,260-.8*t,np.ones(len(t)),.9,None,1,2,'doppler')
    duplicate=Track(t.copy(),moving.f+.1,np.ones(len(t)),.9,None,1,9,'doppler')
    jitter=Track(t,110+4*np.sin(2*np.pi*t/2),np.ones(len(t)),.9,None,1,3,'doppler')
    assert _motion_kind(flat,cfg)=='stationary'
    assert _motion_kind(moving,cfg)=='doppler'
    assert _motion_kind(jitter,cfg) is None
    notes=Track(t,np.where(t<20,110.,98.),np.ones(len(t)),.95,None,1,4,'doppler')
    assert _motion_kind(notes,cfg,'verified') is None
    assert len(_dedupe_tracks([moving,duplicate],cfg))==1


def test_batched_simplex_matches_scalar_search():
    from scipy.optimize import minimize
    from dopplerscrub.detect import _refine_curves,_sample,_C
    f=np.arange(30.,300.,.5);t=np.arange(1.,60.,.25)
    curve=ideal_curve(t,110,65,350,25)
    s=np.exp(-.5*((f[:,None]-curve)/.8)**2)
    seeds=np.array([[111.,62.,360.,26.],[109.,68.,330.,24.]])
    def objective(p):
        if not(40<p[0]<f[-1] and 5<p[1]<=150 and 50<=p[2]<50000 and t[0]+5<p[3]+p[2]/_C<t[-1]-5):return 100.
        q=ideal_curve(t,*p)
        if np.any((q<f[0])|(q>f[-1])):return 100.
        return -float(np.mean(_sample(s,f,q)))
    for seed,result in zip(seeds,_refine_curves(t,f,s,seeds)):
        scalar=minimize(objective,seed,method='Nelder-Mead',options={'maxiter':180,'xatol':.01}).x
        assert abs(objective(result)-objective(scalar))<1e-8


def test_verified_maneuver_merge_preserves_continuous_comb(monkeypatch):
    import sys,types
    from dopplerscrub.types import Track
    sr=4000;time=np.arange(sr*32)/sr
    base=100+12*np.sin(2*np.pi*time/16)
    x=.08*np.sin(2*np.pi*np.cumsum(base)/sr)
    grid=np.arange(2.,30.,.25);root=np.interp(grid,time,base)
    def tracker(s,t,f,cfg):
        return [Track(grid.copy(),k*root,np.ones(len(grid))*.01,.7,None,k,2,'doppler') for k in (1,2,3)]
    monkeypatch.setitem(sys.modules,'dopplerscrub.ridges',types.SimpleNamespace(track_ridges=tracker))
    out=detect_tracks(x,sr,DetectConfig(fmax=400))
    moving=[q for q in out if q.kind=='doppler']
    assert moving
    assert all(q.doppler_score>=.5 for q in moving)
    for k in (1,2,3):
        assert any(q.harmonic==k and q.t0<=2 and q.t1>=29.75 for q in moving)


def test_duplicate_harmonic_does_not_seed_second_source():
    from dopplerscrub.detect import _consolidate_families,_dedupe_tracks
    from dopplerscrub.types import Track
    t=np.arange(0.,40.,.25);root=130-.6*t
    tracks=[Track(t,k*root,np.ones(len(t)),.8,None,k,2,'doppler') for k in (1,2,3)]
    tracks.append(Track(t,2*root+.2,np.ones(len(t)),.8,None,1,9,'doppler'))
    trusted={9}
    tracks=_consolidate_families(tracks,DetectConfig(),trusted)
    result=_dedupe_tracks(tracks,DetectConfig(),trusted)
    assert len(result)==3 and {q.group for q in result}=={2}
    assert 2 in trusted


def test_fused_ridge_view_resolves_fast_low_band_chirp():
    from dopplerscrub.detect import _spectrogram,_sample
    sr=4000;time=np.arange(sr*10)/sr;frequency=80+20*time
    rng=np.random.default_rng(112)
    x=.02*np.sin(2*np.pi*np.cumsum(frequency)/sr)+.05*rng.normal(size=len(time))
    t,f,amp,long_view=_spectrogram(x,sr,DetectConfig(fmax=400))
    curve=80+20*t
    long_brightness=_sample(long_view,f,curve)
    fused_brightness=_sample(long_view.ridge_view,f,curve)
    assert np.mean(fused_brightness-long_brightness)>2
    assert np.all(long_view.ridge_view>=long_view)
    assert np.all(long_view.ridge_amplitude>=amp)


def test_measured_sub40_family_is_not_relabelled_as_new_fundamentals(monkeypatch):
    import sys,types
    import dopplerscrub.detect as detector
    from dopplerscrub.types import Track
    t=np.arange(1.,13.,.25);f=np.arange(30.,201.,.5);root=37-.25*t
    amp=np.ones((len(f),len(t)))*.01;s=np.ones_like(amp)*-1
    observed=[Track(t.copy(),k*root,np.ones(len(t))*.02,.9,None,k,4,'doppler') for k in (2,3)]
    monkeypatch.setattr(detector,'_spectrogram',lambda *args:(t,f,amp,s))
    monkeypatch.setattr(detector,'_bright_ridges',lambda *args:[])
    monkeypatch.setattr(detector,'_steady_lines',lambda *args:[])
    monkeypatch.setitem(sys.modules,'dopplerscrub.ridges',types.SimpleNamespace(track_ridges=lambda *args:observed))
    result=detect_tracks(np.zeros(14*4000),4000,DetectConfig(fmax=200))
    assert len(result)==2 and {q.harmonic for q in result}=={2,3}
    assert all(q.kind=='doppler' and q.doppler_score>=.5 for q in result)


def test_isolated_parent_does_not_probe_scattered_noise_into_a_comb():
    from dopplerscrub.detect import _probe_known_harmonics
    from dopplerscrub.types import Track
    t=np.arange(0.,60.,.25);f=np.arange(30.,1000.,.5);root=120-.3*t
    rng=np.random.default_rng(661)
    s=20*np.log10(np.maximum(rng.rayleigh(size=(len(f),len(t))),1e-9)/np.sqrt(2*np.log(2)))
    parent=Track(t,root,np.ones(len(t))*.01,.9,None,1,5,'doppler')
    result=_probe_known_harmonics([parent],t,f,np.ones_like(s)*.01,s,DetectConfig(fmax=1000))
    assert result==[]


def test_source_grouping_compares_overlap_not_whole_span_medians():
    from dopplerscrub.detect import _consolidate_families
    from dopplerscrub.types import Track
    t=np.arange(0.,100.,.25);root=200-t;early=t<=20
    tracks=[Track(t,root,np.ones(len(t)),.9,None,1,3,'doppler'),Track(t[early],root[early],np.ones(early.sum()),.9,None,1,8,'doppler')]
    result=_consolidate_families(tracks,DetectConfig(),{8})
    assert len({q.group for q in result})==1
