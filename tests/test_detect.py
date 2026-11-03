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
    t=np.arange(sr*15)/sr
    tracks=detect_tracks(.1*np.sin(2*np.pi*60*t),sr,DetectConfig(fmax=200))
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
    assert called and any(q.fit is None and q.kind=='doppler' for q in out)


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
