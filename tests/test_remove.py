import time

import numpy as np
import pytest
from scipy.signal import stft

from dopplerscrub.remove import remove_tracks
from dopplerscrub.types import RemoveConfig, Track

SR = 8000


def doppler_f(t, f0=120.0, v=60.0, d=300.0, tc=20.0, c=343.0):
    x = v * (t - tc)
    r = np.sqrt(x ** 2 + d ** 2)
    return f0 / (1 + v * x / (r * c))


def tone(f, sr, amp=0.05, ph=0.3, span=None):
    """Tone following f; if span=(a, b) it fades in/out (1 s) and exists only there."""
    y = amp * np.cos(2 * np.pi * np.cumsum(f) / sr + ph)
    if span is not None:
        t = np.arange(len(f)) / sr
        g = np.clip(np.minimum(t - span[0], span[1] - t) + 0.5, 0, 1)
        y *= 0.5 - 0.5 * np.cos(np.pi * g)
    return y


def speechlike(n, sr, seed=0):
    """Harmonic voice with gliding f0, syllabic AM, formant-ish tilt, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.3 * t) + 10 * np.sin(2 * np.pi * 1.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    v = sum((1.0 / k) * np.cos(k * ph + k) for k in range(1, 20))
    am = np.clip(np.sin(2 * np.pi * 4 * t + rng.uniform(0, 6)), 0, None) ** 2
    am *= (np.sin(2 * np.pi * 0.21 * t) > -0.3)
    return (0.03 * v * am + 0.003 * rng.standard_normal(n))


def mk_track(t, f, kind="doppler", amp=None):
    return Track(np.asarray(t, float), np.asarray(f, float),
                 np.zeros(len(t)) if amp is None else np.full(len(t), amp), kind=kind)


def db(e, r):
    return 10 * np.log10(np.sum(np.abs(e) ** 2) / np.sum(np.abs(r) ** 2) + 1e-30)


def outside_corridor_db(y, b, sr, tracks_f, guard=8.0):
    """Energy of (y - b) outside +-guard Hz of every track curve, relative to b's
    energy in the same TF region."""
    f, t, E = stft((y - b).astype(float), fs=sr, nperseg=4096, noverlap=3072, axis=0)
    _, _, B = stft(b.astype(float), fs=sr, nperseg=4096, noverlap=3072, axis=0)
    keep = np.ones((len(f), len(t)), bool)
    for fn in tracks_f:
        fc = fn(t)
        keep &= np.abs(f[:, None] - fc[None, :]) > guard
    if E.ndim == 3:
        keep = keep[:, None, :]
    return db(E * keep, B * keep)


@pytest.fixture(scope="module")
def scene():
    dur = 40
    n = SR * dur
    tt = np.arange(n) / SR
    tg = np.arange(5, 35, 0.25)
    s = tone(doppler_f(tt), SR, span=(tg[0], tg[-1]))
    b = speechlike(n, SR)
    return dict(n=n, tt=tt, s=s, b=b, tg=tg, x=(s + b).astype(np.float32))


@pytest.mark.parametrize("jitter", [0.0, 0.3])
def test_subtract_suppresses_pure_tone(scene, jitter):
    rng = np.random.default_rng(3)
    tg, tt, s = scene["tg"], scene["tt"], scene["s"]
    tr = mk_track(tg, doppler_f(tg) + jitter * rng.standard_normal(len(tg)))
    y = remove_tracks(s.astype(np.float32), SR, [tr])
    m = (tt > tg[0]) & (tt < tg[-1])
    drop = db(y[m], s[m])
    assert drop < (-30 if jitter == 0 else -20), drop
    assert drop < -40  # actual margin we get (~-55 dB)


@pytest.mark.parametrize("method,harden", [("subtract", False), ("mask", False), ("subtract", True)])
def test_speech_background_preserved(scene, method, harden):
    tg, tt, s, b, x = scene["tg"], scene["tt"], scene["s"], scene["b"], scene["x"]
    rng = np.random.default_rng(4)
    tr = mk_track(tg, doppler_f(tg) + 0.3 * rng.standard_normal(len(tg)))
    cfg = RemoveConfig(method=method, harden=harden)
    y = remove_tracks(x, SR, [tr], cfg)
    assert y.shape == x.shape and y.dtype == x.dtype
    m = (tt > tg[0]) & (tt < tg[-1])
    # tone gone: residual tone ~ remove(s + b) - remove(b) (processing is ~linear)
    if not harden:
        yb = remove_tracks(b.astype(np.float32), SR, [tr], cfg)
        assert db((y.astype(float) - yb)[m], s[m]) < -30
        # total error incl. background removed *inside* the narrow corridor
        assert db((y.astype(float) - b)[m], s[m]) < -20
    # outside the corridor: background untouched
    assert outside_corridor_db(y, b, SR, [doppler_f]) < -40
    # outside the padded span: bit-exact
    a = max(0, int((tg[0] - cfg.pad_s - 3.5) * SR))
    z = int((tg[-1] + cfg.pad_s + 3.5) * SR)
    assert np.array_equal(y[:a], x[:a]) and np.array_equal(y[z:], x[z:])


def test_harden_fills_notch_at_background_level():
    """After harden the corridor power ~ the flanking background (no notch, no tone)."""
    sr, dur = 8000, 40
    n = sr * dur
    tt = np.arange(n) / sr
    rng = np.random.default_rng(5)
    b = 0.02 * rng.standard_normal(n)
    s = tone(doppler_f(tt), sr, amp=0.1)
    tg = np.arange(4, 36, 0.25)
    tr = mk_track(tg, doppler_f(tg))
    for harden in (False, True):
        y = remove_tracks((s + b).astype(np.float32), sr, [tr], RemoveConfig(harden=harden))
        f, t, Y = stft(y.astype(float), fs=sr, nperseg=8192, noverlap=8192 - 2000)
        sel = (t > 8) & (t < 32)
        fc = doppler_f(t[sel])
        P = np.abs(Y[:, sel]) ** 2
        d = np.abs(f[:, None] - fc[None, :])
        inside = P[d < 0.6].mean()
        flank = P[(d > 10) & (d < 30)].mean()
        ratio = 10 * np.log10(inside / flank)
        if harden:
            assert -3 < ratio < 3, ratio
        else:
            assert ratio < 3, ratio  # tone gone (it was ~+40 dB)


def test_stereo_and_mono_shapes(scene):
    tg, s, b = scene["tg"], scene["s"], scene["b"]
    x2 = np.stack([s + b, 0.5 * s - b[::-1]], axis=1).astype(np.float32)
    tr = mk_track(tg, doppler_f(tg))
    y2 = remove_tracks(x2, SR, [tr])
    assert y2.shape == x2.shape and y2.dtype == np.float32
    b2 = np.stack([b, -b[::-1]], axis=1)
    yb = remove_tracks(b2.astype(np.float32), SR, [tr])
    m = (scene["tt"] > tg[0]) & (scene["tt"] < tg[-1])
    for c, (sc, bc) in enumerate([(s, b), (0.5 * s, -b[::-1])]):
        assert db((y2[:, c] - yb[:, c])[m], sc[m]) < -30
        assert outside_corridor_db(y2[:, c], bc, SR, [doppler_f]) < -40
    y1 = remove_tracks(x2[:, 0], SR, [tr])
    assert y1.shape == (len(x2),)
    # stereo shares one frequency-refinement across channels, so ~equal, not exact
    assert db(y1 - y2[:, 0], s) < -30


def test_track_at_file_edges():
    sr, dur = 8000, 20
    n = sr * dur
    tt = np.arange(n) / sr
    fA = lambda t: doppler_f(t, f0=150, tc=2.0)
    fB = lambda t: doppler_f(t, f0=90, tc=18.0)
    ta = np.arange(0, 8, 0.25)
    tb = np.arange(12, 20 - 1e-9, 0.25)
    sA = tone(fA(tt), sr, span=(-5, 8))      # already sounding at the first sample
    sB = tone(fB(tt), sr, ph=1.0, span=(12, 25))  # still sounding at the last sample
    s = sA + sB
    y = remove_tracks(s.astype(np.float32), sr, [mk_track(ta, fA(ta)), mk_track(tb, fB(tb))])
    assert db(y[: 7 * sr], sA[: 7 * sr]) < -30           # including the very first samples
    assert db(y[13 * sr:], sB[13 * sr:]) < -30           # through the last sample
    assert np.all(np.isfinite(y))


def test_overlapping_and_crossing_tracks(scene):
    tt, b = scene["tt"], scene["b"]
    f1 = lambda t: doppler_f(t, f0=120, tc=20)
    f2 = lambda t: doppler_f(t, f0=115, v=80, d=200, tc=22)   # crosses f1
    f3 = lambda t: 2 * f1(t)                                  # harmonic
    tg = scene["tg"]
    sp = (tg[0], tg[-1])
    s = (tone(f1(tt), SR, span=sp) + tone(f2(tt), SR, amp=0.03, ph=2, span=sp)
         + tone(f3(tt), SR, amp=0.02, ph=1, span=sp))
    trs = [mk_track(tg, fn(tg)) for fn in (f1, f2, f3)]
    y = remove_tracks((s + b).astype(np.float32), SR, trs)
    m = (tt > 8) & (tt < 32)
    assert db((y - b)[m], s[m]) < -20
    assert outside_corridor_db(y, b, SR, [f1, f2, f3]) < -40


def test_kind_filter_and_noop(scene):
    x, tg = scene["x"], scene["tg"]
    tr = mk_track(tg, doppler_f(tg), kind="stationary")
    assert np.array_equal(remove_tracks(x, SR, [tr]), x)
    assert np.array_equal(remove_tracks(x, SR, []), x)
    y = remove_tracks(x, SR, [tr], RemoveConfig(remove_kinds=("doppler", "stationary")))
    assert not np.array_equal(y, x)
    with pytest.raises(ValueError):
        remove_tracks(x, SR, [mk_track(tg, doppler_f(tg))], RemoveConfig(method="bogus"))


def test_short_and_degenerate_tracks(scene):
    x = scene["x"]
    trs = [mk_track([10.0], [120.0]), mk_track([10.0, 10.5], [120.0, 119.0]),
           mk_track([5, 6, 7], [np.nan, 100, 101]), mk_track([5, 6], [5000.0, 5000.0])]
    y = remove_tracks(x, SR, trs)
    assert y.shape == x.shape and np.all(np.isfinite(y))


@pytest.mark.slow
def test_speed_long_stereo():
    """10 min @ 48 kHz stereo, 20 tracks of 60 s: should take a few seconds
    (=> roughly 1 h with 20 tracks in well under a minute or two)."""
    sr, dur = 48000, 600
    rng = np.random.default_rng(0)
    x = (0.01 * rng.standard_normal((sr * dur, 2))).astype(np.float32)
    trs = []
    for k in range(20):
        t0 = rng.uniform(0, dur - 60)
        tg = np.arange(t0, t0 + 60, 0.25)
        trs.append(mk_track(tg, doppler_f(tg, f0=rng.uniform(50, 400), tc=t0 + 30)))
    t = time.time()
    y = remove_tracks(x, sr, trs)
    el = time.time() - t
    assert y.shape == x.shape
    assert el < 120, el


def test_harden_fill_level_per_track_with_coloured_background():
    """Two simultaneous tracks over a background ~12 dB louder at low frequency:
    each corridor must be refilled at ITS OWN local level (regression: the
    loudest track's level used to be applied to every corridor)."""
    from scipy.signal import butter, sosfilt
    sr, dur = 8000, 40
    n = sr * dur
    tt = np.arange(n) / sr
    rng = np.random.default_rng(7)
    w = rng.standard_normal(n)
    b = 0.005 * w + 0.06 * sosfilt(butter(2, 120, fs=sr, output="sos"), rng.standard_normal(n))
    fa = lambda t: doppler_f(t, f0=80)
    fb = lambda t: doppler_f(t, f0=320)
    tg = np.arange(4, 36, 0.25)
    s = tone(fa(tt), sr, 0.1, span=(4, 36)) + tone(fb(tt), sr, 0.05, span=(4, 36))
    for method in ("subtract", "mask"):
        y = remove_tracks((s + b).astype(np.float32), sr, [mk_track(tg, fa(tg)), mk_track(tg, fb(tg))],
                          RemoveConfig(method=method, harden=True))
        f, t, Y = stft(y.astype(float), fs=sr, nperseg=8192, noverlap=8192 - 2000)
        sel = (t > 8) & (t < 32)
        P = np.abs(Y[:, sel]) ** 2
        for fn in (fa, fb):
            d = np.abs(f[:, None] - fn(t[sel])[None, :])
            ratio = 10 * np.log10(P[d < 0.6].mean() / P[(d > 8) & (d < 16)].mean())
            assert -3 < ratio < 3, (method, ratio)
