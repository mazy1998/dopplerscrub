import time

import numpy as np
import pytest
from scipy.signal import stft

from dopplerscrub.remove import remove_tracks
from dopplerscrub.types import RemoveConfig, Track

SR = 8000


def doppler_f(t, f0=120.0, v=60.0, d=300.0, tc=20.0, c=343.0):
    x = v * (t - tc)
    r = np.sqrt(x**2 + d**2)
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
    am *= np.sin(2 * np.pi * 0.21 * t) > -0.3
    return 0.03 * v * am + 0.003 * rng.standard_normal(n)


def mk_track(t, f, kind="doppler", amp=None, score=1.0):
    """Track as the detector would hand it over; score >= 0.5 = confirmed."""
    return Track(
        np.asarray(t, float),
        np.asarray(f, float),
        np.zeros(len(t)) if amp is None else np.full(len(t), amp),
        doppler_score=score,
        kind=kind,
    )


def db(e, r):
    return 10 * np.log10(np.sum(np.abs(e) ** 2) / np.sum(np.abs(r) ** 2) + 1e-30)


def tone_residual_db(e, comps, m, sr=SR, blk_s=0.5):
    """Residual of the tones themselves in error signal e (= y - background):
    per 0.5 s block, coherent projection of e onto each tone's analytic signal.
    Ignores background removed inside the corridor (which a gated, nonlinear
    removal makes impossible to cancel by differencing)."""
    from scipy.signal import hilbert

    e = np.asarray(e, float)
    num = den = 0.0
    B = int(blk_s * sr)
    for s in comps:
        sa = hilbert(s)
        for k0 in range(0, len(e) - B + 1, B):
            sl = slice(k0, k0 + B)
            mk = m[sl]
            if not mk.any():
                continue
            ss = np.sum(np.abs(sa[sl][mk]) ** 2)
            if ss <= 0:
                continue
            c = 2 * np.sum(e[sl][mk] * np.conj(sa[sl][mk])) / ss
            num += np.abs(c) ** 2 * ss / 2
            den += ss / 2
    return 10 * np.log10(num / den + 1e-30)


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


METHODS = ["vk", "subtract"]
CROSS = (
    lambda t: doppler_f(t, f0=120, tc=20),
    lambda t: doppler_f(t, f0=115, v=80, d=200, tc=22),  # crosses the first
    lambda t: 2 * doppler_f(t, f0=120, tc=20),
)  # harmonic of the first


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("jitter,limit", [(0.0, -50), (0.3, -45), (1.0, -40)])
def test_pure_tone_suppression(scene, method, jitter, limit):
    """Spec: > 30 dB with exact f(t), > 20 dB with 0.3 Hz RMS jitter (harden off,
    so the residual tone itself is measured)."""
    rng = np.random.default_rng(3)
    tg, tt, s = scene["tg"], scene["tt"], scene["s"]
    tr = mk_track(tg, doppler_f(tg) + jitter * rng.standard_normal(len(tg)))
    y = remove_tracks(s.astype(np.float32), SR, [tr], RemoveConfig(method=method, harden=False))
    m = (tt > tg[0]) & (tt < tg[-1])
    assert db(y[m], s[m]) < limit


@pytest.mark.parametrize("method", ["vk", "subtract", "mask"])
@pytest.mark.parametrize("harden", [False, True])
def test_speech_background_preserved(scene, method, harden):
    tg, tt, s, b, x = scene["tg"], scene["tt"], scene["s"], scene["b"], scene["x"]
    rng = np.random.default_rng(4)
    tr = mk_track(tg, doppler_f(tg) + 0.3 * rng.standard_normal(len(tg)))
    cfg = RemoveConfig(method=method, harden=harden)
    y = remove_tracks(x, SR, [tr], cfg)
    assert y.shape == x.shape and y.dtype == x.dtype
    m = (tt > tg[0]) & (tt < tg[-1])
    if not harden:
        # y - b = residual tone + speech removed inside the narrow corridor (inherent)
        assert tone_residual_db(y.astype(float) - b, [s], m) < -25
        # total error incl. background removed *inside* the narrow corridor
        assert db((y.astype(float) - b)[m], s[m]) < -20
    # outside the corridor: background untouched (harden on or off)
    assert outside_corridor_db(y, b, SR, [doppler_f]) < -40
    # outside the padded span: bit-exact
    a = max(0, int((tg[0] - cfg.pad_s - 3.5) * SR))
    z = int((tg[-1] + cfg.pad_s + 3.5) * SR)
    assert np.array_equal(y[:a], x[:a]) and np.array_equal(y[z:], x[z:])


@pytest.mark.parametrize("method", ["vk", "subtract", "mask"])
def test_harden_fills_notch_at_background_level(method):
    """harden=True: corridor power ~ flanking background (no tone, no notch).
    harden=False: tone gone but a deep notch remains (why harden is the default)."""
    sr, dur = 8000, 40
    n = sr * dur
    tt = np.arange(n) / sr
    rng = np.random.default_rng(5)
    b = 0.02 * rng.standard_normal(n)
    s = tone(doppler_f(tt), sr, amp=0.1)
    tg = np.arange(4, 36, 0.25)
    tr = mk_track(tg, doppler_f(tg))
    for harden in (False, True):
        y = remove_tracks(
            (s + b).astype(np.float32), sr, [tr], RemoveConfig(method=method, harden=harden)
        )
        f, t, Y = stft(y.astype(float), fs=sr, nperseg=8192, noverlap=8192 - 2000)
        sel = (t > 8) & (t < 32)
        fc = doppler_f(t[sel])
        P = np.abs(Y[:, sel]) ** 2
        d = np.abs(f[:, None] - fc[None, :])
        inside = P[d < 0.6].mean()
        flank = P[(d > 10) & (d < 30)].mean()
        ratio = 10 * np.log10(inside / flank)
        if harden:
            assert -2 < ratio < 2, ratio
        else:
            assert ratio < -10, ratio  # tone gone (was ~+40 dB) and a notch is left


def test_harden_fill_level_matches_true_background():
    """The fill replaces the removed corridor with the background's own level
    (compared with the untouched background in the same band, which is the only
    unbiased reference: two different 2 Hz bands of gusty noise easily differ by
    1.5 dB by chance)."""
    from scipy.signal import hilbert

    sr, dur = 8000, 60
    n = sr * dur
    tt = np.arange(n) / sr
    rng = np.random.default_rng(9)
    b = 0.02 * (1 + 0.8 * np.sin(2 * np.pi * 0.2 * tt) ** 2) * rng.standard_normal(n)
    f = lambda t: 150.0 + 0 * t
    tg = np.arange(5, 55, 0.25)
    y = remove_tracks(
        (tone(f(tt), sr, amp=0.1) + b).astype(np.float32),
        sr,
        [mk_track(tg, f(tg))],
        RemoveConfig(method="vk", harden=True),
    ).astype(float)

    def band_env(v, lo, hi):
        V = np.fft.rfft(v)
        fr = np.fft.rfftfreq(len(v), 1 / sr)
        V[(fr < lo) | (fr > hi)] = 0
        return np.abs(hilbert(np.fft.irfft(V, len(v))))[10 * sr : 50 * sr]

    e_y, e_b = band_env(y, 149, 151), band_env(b, 149, 151)
    assert (
        abs(10 * np.log10(np.mean(e_y**2) / np.mean(e_b**2))) < 1.0
    )  # tone (+37 dB) gone, level kept


@pytest.mark.parametrize("method", METHODS)
def test_stereo_and_mono_shapes(scene, method):
    tg, s, b = scene["tg"], scene["s"], scene["b"]
    m = (scene["tt"] > tg[0]) & (scene["tt"] < tg[-1])
    tr = mk_track(tg, doppler_f(tg))
    cfg = RemoveConfig(method=method, harden=False)
    # suppression per channel, over a low floor (different tone level per channel)
    wn = 1e-3 * np.random.default_rng(8).standard_normal((len(s), 2))
    x2 = (np.stack([s, 0.5 * s], axis=1) + wn).astype(np.float32)
    y2 = remove_tracks(x2, SR, [tr], cfg)
    assert y2.shape == x2.shape and y2.dtype == np.float32
    for c, sc in enumerate([s, 0.5 * s]):
        assert tone_residual_db(y2[:, c] - wn[:, c], [sc], m) < -45
    y1 = remove_tracks(x2[:, 0], SR, [tr], cfg)
    assert y1.shape == (len(x2),)
    # stereo shares one frequency-refinement across channels, so ~equal, not exact
    assert db(y1 - y2[:, 0], s) < -30
    # speech-like stereo background, harden off and on: confined per channel
    b2 = np.stack([b, -b[::-1]], axis=1)
    xs = (np.stack([s, 0.5 * s], axis=1) + b2).astype(np.float32)
    for harden in (False, True):
        yh = remove_tracks(xs, SR, [tr], RemoveConfig(method=method, harden=harden))
        for c in range(2):
            assert outside_corridor_db(yh[:, c], b2[:, c], SR, [doppler_f]) < -40


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("harden", [False, True])
def test_track_at_file_edges(method, harden):
    sr, dur = 8000, 20
    n = sr * dur
    tt = np.arange(n) / sr
    fA = lambda t: doppler_f(t, f0=150, tc=2.0)
    fB = lambda t: doppler_f(t, f0=90, tc=18.0)
    ta = np.arange(0, 8, 0.25)
    tb = np.arange(12, 20 - 1e-9, 0.25)
    sA = tone(fA(tt), sr, span=(-5, 8))  # already sounding at the first sample
    sB = tone(fB(tt), sr, ph=1.0, span=(12, 25))  # still sounding at the last sample
    rng = np.random.default_rng(6)
    b = 1e-3 * rng.standard_normal(n)
    y = remove_tracks(
        (sA + sB + b).astype(np.float32),
        sr,
        [mk_track(ta, fA(ta)), mk_track(tb, fB(tb))],
        RemoveConfig(method=method, harden=harden),
    )
    assert np.all(np.isfinite(y))
    lim = -30 if not harden else -25  # harden: fill noise at the (low) floor level
    assert db((y - b)[: 7 * sr], sA[: 7 * sr]) < lim  # incl. the very first samples
    assert db((y - b)[13 * sr :], sB[13 * sr :]) < lim  # through the last sample


@pytest.mark.parametrize("method", ["vk", "subtract", "mask"])
@pytest.mark.parametrize("harden", [False, True])
def test_overlapping_and_crossing_tracks(scene, method, harden):
    tt, b, tg = scene["tt"], scene["b"], scene["tg"]
    sp = (tg[0], tg[-1])
    s = (
        tone(CROSS[0](tt), SR, span=sp)
        + tone(CROSS[1](tt), SR, amp=0.03, ph=2, span=sp)
        + tone(CROSS[2](tt), SR, amp=0.02, ph=1, span=sp)
    )
    trs = [mk_track(tg, fn(tg)) for fn in CROSS]
    y = remove_tracks(
        (s + b).astype(np.float32), SR, trs, RemoveConfig(method=method, harden=harden)
    )
    m = (tt > 8) & (tt < 32)
    if not harden:
        assert db((y - b)[m], s[m]) < -20
    # mask's notch (and the fill covering it) widens with the sweep rate within
    # its 1 s STFT window: f2 sweeps ~10 Hz/s, so use a wider guard there
    guard = 8.0 if method != "mask" else 13.0
    assert outside_corridor_db(y, b, SR, list(CROSS), guard=guard) < -40


def test_vk_beats_subtract_on_crossing_tracks(scene):
    """Joint (coupled) VK separates crossing tones; per-track subtraction can't.
    (Low floor, so the residual of the tones themselves is measured.)"""
    tt, tg = scene["tt"], scene["tg"]
    b = 1e-3 * np.random.default_rng(10).standard_normal(len(tt))
    sp = (tg[0], tg[-1])
    comps = [tone(CROSS[0](tt), SR, span=sp), tone(CROSS[1](tt), SR, amp=0.03, ph=2, span=sp)]
    s = comps[0] + comps[1]
    trs = [mk_track(tg, fn(tg)) for fn in CROSS[:2]]
    m = (tt > tg[0]) & (tt < tg[-1])
    res = {}
    for method in METHODS:
        y = remove_tracks(
            (s + b).astype(np.float32), SR, trs, RemoveConfig(method=method, harden=False)
        )
        res[method] = tone_residual_db(y.astype(float) - b, comps, m)
    assert res["vk"] < -45 and res["vk"] < res["subtract"] - 3, res


def test_kind_filter_and_noop(scene):
    x, tg = scene["x"], scene["tg"]
    tr = mk_track(tg, doppler_f(tg), kind="stationary")
    assert np.array_equal(remove_tracks(x, SR, [tr], RemoveConfig(harden=True)), x)
    assert np.array_equal(remove_tracks(x, SR, []), x)
    y = remove_tracks(
        x, SR, [tr], RemoveConfig(remove_kinds=("doppler", "stationary"), harden=False)
    )
    assert not np.array_equal(y, x)
    with pytest.raises(ValueError):
        remove_tracks(x, SR, [mk_track(tg, doppler_f(tg))], RemoveConfig(method="bogus"))


@pytest.mark.parametrize("method", ["vk", "subtract", "mask"])
@pytest.mark.parametrize("harden", [False, True])
def test_short_and_degenerate_tracks(scene, method, harden):
    x = scene["x"]
    trs = [
        mk_track([10.0], [120.0]),
        mk_track([10.0, 10.5], [120.0, 119.0]),
        mk_track([5, 6, 7], [np.nan, 100, 101]),
        mk_track([5, 6], [5000.0, 5000.0]),
    ]
    y = remove_tracks(x, SR, trs, RemoveConfig(method=method, harden=harden))
    assert y.shape == x.shape and np.all(np.isfinite(y))


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
def test_speed_long_stereo(method):
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
    y = remove_tracks(x, sr, trs, RemoveConfig(method=method, harden=True))
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
    for method in ("vk", "subtract", "mask"):
        y = remove_tracks(
            (s + b).astype(np.float32),
            sr,
            [mk_track(tg, fa(tg)), mk_track(tg, fb(tg))],
            RemoveConfig(method=method, harden=True),
        )
        f, t, Y = stft(y.astype(float), fs=sr, nperseg=8192, noverlap=8192 - 2000)
        sel = (t > 8) & (t < 32)
        P = np.abs(Y[:, sel]) ** 2
        for fn in (fa, fb):
            d = np.abs(f[:, None] - fn(t[sel])[None, :])
            ratio = 10 * np.log10(P[d < 0.6].mean() / P[(d > 8) & (d < 16)].mean())
            assert -3 < ratio < 3, (method, ratio)


# ----------------------------------------------------------------------------- do no harm


def _pink(n, rng, sr):
    F = np.fft.rfft(rng.standard_normal(n))
    F /= np.sqrt(np.maximum(np.fft.rfftfreq(n, 1 / sr), 1.0))
    p = np.fft.irfft(F, n)
    return p / np.std(p)


def _garbage_tracks(rng, dur, k=12, score=0.0):
    """What a broken detector might return: random Doppler-ish sweeps, flat lines,
    steep (near-vertical) ramps and random walks, 3-30 s long, 30 Hz-3.5 kHz."""
    trs = []
    for _ in range(k):
        t0 = rng.uniform(0, dur - 5)
        tg = np.arange(t0, min(dur, t0 + rng.uniform(3, 30)), 0.25)
        kind, f0 = rng.integers(4), rng.uniform(45, 900)
        if kind == 0:
            f = f0 + rng.uniform(5, 40) * np.tanh(
                (tg - tg.mean()) / rng.uniform(1, 6)
            ) * rng.choice([-1, 1])
        elif kind == 1:
            f = np.full(len(tg), f0)
        elif kind == 2:
            f = f0 + rng.uniform(-120, 120) * (tg - tg[0])
        else:
            f = f0 + np.cumsum(rng.normal(0, 0.5, len(tg)))
        trs.append(
            Track(
                tg, np.clip(f, 30, 3500), np.full(len(tg), 0.5), doppler_score=score, kind="doppler"
            )
        )
    return trs


def _stft_pair(x, y, sr, blk):
    nper = int(sr * blk)
    f, t, X = stft(x, fs=sr, nperseg=nper, noverlap=nper // 2)
    _, _, Y = stft(y, fs=sr, nperseg=nper, noverlap=nper // 2)
    return f, t, X, Y


def _corridor(tr, t, f, blk, base, pad=2.0):
    """Bins of tr's corridor per frame: f(t) held beyond the span through the pads
    (extrapolation moves <= 2 Hz), half-width base + 2 Hz + in-frame sweep."""
    fc = np.where((t >= tr.t[0] - pad) & (t <= tr.t[-1] + pad), np.interp(t, tr.t, tr.f), np.nan)
    slope = np.max(np.abs(np.gradient(tr.f, tr.t))) if len(tr.t) > 2 else 0.0
    half = base + 2.0 + slope * blk / 2
    return fc, half


def _gains(x, y, trs, sr, blk=1.0):
    """(max per-frame corridor gain, max per-track gain, pooled gain) in dB."""
    f, t, X, Y = _stft_pair(x, y, sr, blk)
    frame_max, track_max, EX, EY = -99.0, -99.0, 0.0, 0.0
    for tr in trs:
        fc, half = _corridor(tr, t, f, blk, 4.0)
        ex_t = ey_t = 0.0
        for k in np.flatnonzero(np.isfinite(fc)):
            mk = np.abs(f - fc[k]) <= half
            ex, ey = np.sum(np.abs(X[mk, k]) ** 2), np.sum(np.abs(Y[mk, k]) ** 2)
            if ex > 0:
                frame_max = max(frame_max, 10 * np.log10(ey / ex))
            ex_t, ey_t = ex_t + ex, ey_t + ey
        if ex_t > 0:
            track_max = max(track_max, 10 * np.log10(ey_t / ex_t))
        EX, EY = EX + ex_t, EY + ey_t
    return frame_max, track_max, 10 * np.log10(EY / EX)


def _outside_err_db(x, y, trs, sr, blk=0.5, guard=8.0):
    f, t, X, Y = _stft_pair(x, y, sr, blk)
    keep = np.ones((len(f), len(t)), bool)
    for tr in trs:
        fc, half = _corridor(tr, t, f, blk, guard)
        act = np.isfinite(fc)
        keep[:, act] &= np.abs(f[:, None] - fc[None, act]) > half
    return 10 * np.log10(np.sum(np.abs((Y - X)[keep]) ** 2) / np.sum(np.abs(X[keep]) ** 2) + 1e-30)


@pytest.fixture(scope="module")
def garbage_scene():
    sr, dur = 8000, 60
    n = sr * dur
    rng = np.random.default_rng(1)
    tt = np.arange(n) / sr
    b = 0.02 * _pink(n, rng, sr) + speechlike(n, sr, 1)
    for fm in (196.0, 261.6, 329.6):  # strong stationary partials (music)
        b += 0.01 * np.sin(2 * np.pi * fm * tt)
    return sr, b.astype(np.float32), rng


@pytest.mark.parametrize("method", ["vk", "subtract", "mask"])
def test_garbage_tracks_never_add_a_curve(garbage_scene, method):
    """Red-team regression: wrong tracks must never inject energy along their path.
    Unconfirmed tracks (doppler_score 0): no corridor gains energy, frame by frame.
    Falsely confirmed tracks: the noise fill replaces noise with noise at the flank
    level, so single frames fluctuate, but no curve appears systematically."""
    sr, x, rng = garbage_scene
    xf = x.astype(float)
    for score in (0.0, 1.0):
        trs = _garbage_tracks(np.random.default_rng(int(score) + 11), len(x) / sr, score=score)
        y, st = remove_tracks(
            x, sr, trs, RemoveConfig(method=method, harden=True), return_stats=True
        )
        frame_max, track_max, pooled = _gains(xf, y.astype(float), trs, sr)
        if score == 0.0:
            assert frame_max <= 0.5, (score, frame_max)
            assert all(not t.get("filled") for t in st["tracks"])
        else:
            # a fill lifts a wrong corridor at most to its local flank level (it
            # may sit in a slightly quieter valley); no curve above the surroundings
            assert pooled <= 0.5 and track_max <= 1.5, (pooled, track_max)
        assert _outside_err_db(xf, y.astype(float), trs, sr) < -40
        assert 0.0 <= st["tf_fraction"] <= 1.0 and "tracks" in st


def test_do_no_harm_reverts_energy_increase():
    """Directly: if a step makes a corridor louder than the input, it is pulled back."""
    import dopplerscrub.remove as R

    sr, n = 8000, 8000 * 20
    rng = np.random.default_rng(3)
    x0 = 0.01 * rng.standard_normal((n, 1))
    tg = np.arange(4, 16, 0.25)
    it = R._select([mk_track(tg, np.full(len(tg), 300.0), score=0.0)], RemoveConfig(), sr)[0]
    tt = np.arange(n) / sr
    y = x0 + 0.05 * np.sin(2 * np.pi * 300 * tt)[:, None] * ((tt > 6) & (tt < 14))[:, None]
    R._do_no_harm(x0, y, sr, it, RemoveConfig(), False, 4.0)
    sel = (tt > 7) & (tt < 13)
    assert 10 * np.log10(np.mean(y[sel] ** 2) / np.mean(x0[sel] ** 2)) < 0.3


def test_coherence_gate_noise_only_track_is_untouched(scene):
    """(c) A confirmed-looking track over pure noise/speech without a tone:
    subtraction removes nothing (no curve-shaped notch)."""
    tg, b = scene["tg"], scene["b"].astype(np.float32)
    tr = mk_track(tg, doppler_f(tg))
    for method in METHODS:
        y, st = remove_tracks(
            b, SR, [tr], RemoveConfig(method=method, harden=False), return_stats=True
        )
        assert st["tracks"][0]["coherent_frac"] < 0.1
        assert db(y.astype(float) - b, b) < -40


def test_stats_report_tf_fraction(scene):
    tg, x = scene["tg"], scene["x"]
    y, st = remove_tracks(x, SR, [mk_track(tg, doppler_f(tg))], RemoveConfig(), return_stats=True)
    assert 0 < st["tf_fraction"] < 0.01  # one ~40 s corridor of ~10 Hz in 4 kHz x 40 s
    assert 0 < st["tf_fraction_40_1000"] < 0.05
    t0 = st["tracks"][0]
    assert t0["filled"] and t0["coherent_frac"] > 0.9 and t0["modified_s"] > 20


@pytest.mark.parametrize("background", ["pink", "gusty"])
def test_fill_is_statistically_identical_to_flanks(background):
    """Red-team ENF check: in the attacker's whitened spectrogram (8192-pt @ 4 kHz,
    0.25 s hop), corridor bins after harden vs flanking bins: KS p > 0.05, mean
    difference < 0.3 dB, std ratio 0.9-1.1. Samples come from a decorrelated grid
    (non-overlapping 2 s frames, every 2nd bin; on the raw 8x-overlapped grid the KS
    test rejects even the untouched background) along 8 well-separated tracks, so
    the mean difference has a standard error of ~0.1 dB."""
    from scipy.stats import ks_2samp
    from dopplerscrub.evaluate import attack_spectrogram

    sr, dur = 4000, 300
    n = sr * dur
    tt = np.arange(n) / sr
    rng = np.random.default_rng(1)
    b = 0.02 * _pink(n, rng, sr)
    if background == "gusty":
        b *= 0.3 + np.abs(np.sin(2 * np.pi * 0.13 * tt)) + 0.5 * np.sin(2 * np.pi * 0.71 * tt) ** 2
    fns = [
        (lambda f0: lambda t: doppler_f(t, f0=f0, v=40, tc=150))(f0)
        for f0 in (150, 230, 310, 390, 470, 550, 630, 710)
    ]
    tg = np.arange(20, 280, 0.25)
    x = b + sum(tone(fn(tt), sr, amp=0.05, ph=i, span=(20, 280)) for i, fn in enumerate(fns))
    y = remove_tracks(
        x.astype(np.float32),
        sr,
        [mk_track(tg, fn(tg)) for fn in fns],
        RemoveConfig(method="vk", harden=True),
    )
    S = attack_spectrogram(y.astype(float), sr, fmax=800)
    cor, fla = [], []
    for k in np.flatnonzero((S.t > 22) & (S.t < 278))[::8]:
        for fn in fns:
            d = np.abs(S.f - fn(S.t[k]))
            cor.append(S.S[d <= 1.5, k][::2])
            fla.append(S.S[(d >= 4.5) & (d <= 10.0), k][::2])
    c, g = np.concatenate(cor), np.concatenate(fla)
    assert ks_2samp(c, g).pvalue > 0.05
    assert abs(c.mean() - g.mean()) < 0.3
    assert 0.9 <= c.std() / g.std() <= 1.1


def test_pipeline_style_harden_needs_original_reference():
    """Pipeline: removal passes first, then harden_tracks on the cleaned audio.
    With x_ref=original the fill restores the background; against the cleaned
    audio alone, do-no-harm sees the removed tone's corridor as a valley and caps
    the fill to it (a curve-shaped notch: the colsto 245 Hz regression)."""
    from dopplerscrub.remove import harden_tracks

    sr, dur = 8000, 60
    n = sr * dur
    tt = np.arange(n) / sr
    b = 0.01 * np.random.default_rng(4).standard_normal(n)
    fn = lambda t: 245.0 + 0.02 * (t - 30)
    tg = np.arange(5, 55, 0.25)
    x = (tone(fn(tt), sr, amp=0.05, span=(5, 55)) + b).astype(np.float32)
    trs = [mk_track(tg, fn(tg))]
    y1 = remove_tracks(x, sr, trs, RemoveConfig(method="vk", harden=False))

    def notch_db(y):
        f, t, Y = stft(y.astype(float), fs=sr, nperseg=8192, noverlap=8192 - 2000)
        sel = (t > 10) & (t < 50)
        P = np.abs(Y[:, sel]) ** 2
        d = np.abs(f[:, None] - fn(t[sel])[None, :])
        return 10 * np.log10(P[d < 0.6].mean() / P[(d > 8) & (d < 20)].mean())

    assert notch_db(y1) < -10  # VK alone: hole
    assert abs(notch_db(harden_tracks(y1, sr, trs, RemoveConfig(), x_ref=x))) < 2.0


# ----------------------------------------------------------------------------- per-track width


def _jittery(n, sr, fc, spread_hz=6.0, seed=3, amp=0.06):
    """A prop-like tone: centre fc(t) plus a slow random FM of +-spread_hz (~3 Hz rate),
    so its dechirped spectrum is ~+-spread_hz wide with a decaying skirt."""
    rng = np.random.default_rng(seed)
    k = max(1, sr // 50)
    jit = np.repeat(rng.standard_normal(n // k + 2), k)[:n]
    from scipy.ndimage import gaussian_filter1d

    jit = gaussian_filter1d(jit, sr / 6)
    jit *= spread_hz / (np.std(jit) * 1.5 + 1e-12)
    return amp * np.cos(2 * np.pi * np.cumsum(fc + jit) / sr)


def _dechirped_profile(sig, sr, t, f, offs):
    from dopplerscrub.remove import _freq_curve

    a, b = int(t[0] * sr), int(t[-1] * sr)
    tq = np.arange(a, b) / sr
    z = sig[a:b] * np.exp(-2j * np.pi * np.cumsum(_freq_curve(t, f, tq)) / sr)
    n = sr
    P = 0
    for s0 in range(0, len(z) - n, n // 4):
        P = P + np.abs(np.fft.fft(z[s0 : s0 + n] * np.hanning(n))) ** 2
    F = np.fft.fftfreq(n, 1 / sr)
    return np.array([10 * np.log10(P[np.abs(np.abs(F) - o) <= 1].mean()) for o in offs])


def test_width_none_or_default_is_bit_identical():
    n = 30 * SR
    tt = np.arange(n) / SR
    x = (tone(doppler_f(tt), SR) + 0.003 * np.random.default_rng(0).standard_normal(n)).astype(
        np.float32
    )
    tg = np.arange(2, 28, 0.25)
    base = remove_tracks(
        x,
        SR,
        [Track(t=tg, f=doppler_f(tg), amp=np.ones(len(tg)), kind="doppler", doppler_score=0.9)],
    )
    assert not np.array_equal(base, x)
    for w in (None, 2.0, 1.0):  # 1.0 < cfg.width_hz -> floored to it
        tr = Track(t=tg, f=doppler_f(tg), amp=np.ones(len(tg)), kind="doppler", doppler_score=0.9)
        tr.width_hz = w
        assert np.array_equal(remove_tracks(x, SR, [tr]), base)


def test_measured_width_removes_jittery_tone_without_notch():
    n = 40 * SR
    tt = np.arange(n) / SR
    fc = doppler_f(tt, f0=300.0, v=80.0, d=150.0, tc=20.0)
    bgn = 0.003 * np.random.default_rng(1).standard_normal(n)
    x = (_jittery(n, SR, fc, amp=0.003) + bgn).astype(
        np.float32
    )  # ~22 dB over background, like p51
    tg = np.arange(4, 36, 1 / 16)  # fine 1/16 s grid, as chirps.py emits
    offs = [0, 3, 6, 9, 12, 16, 20, 30, 45]

    def run(w):
        tr = Track(
            t=tg, f=np.interp(tg, tt, fc), amp=np.ones(len(tg)), kind="doppler", doppler_score=0.9
        )
        tr.width_hz = w
        y, st = remove_tracks(x, SR, [tr], return_stats=True)
        return y, st["tracks"][0]

    y0, _ = run(None)
    y1, s1 = run(10.0)
    p_bg = _dechirped_profile(bgn, SR, tg, np.interp(tg, tt, fc), offs)
    p0 = _dechirped_profile(y0, SR, tg, np.interp(tg, tt, fc), offs)
    p1 = _dechirped_profile(y1, SR, tg, np.interp(tg, tt, fc), offs)
    # core and skirt brought to the background with the measured width ...
    assert np.max(p1 - p_bg) < 3.0, (p1, p_bg)
    assert np.max(p1[:3] - p_bg[:3]) < np.max(p0[:3] - p_bg[:3]) - 6.0
    # ... never notched below it
    assert np.min(p1 - p_bg) > -2.0, (p1, p_bg)
    assert "skirt_hz" in s1
