"""Track-locked removal of narrowband (Doppler) tones.

``remove_tracks(x, sr, tracks, cfg)`` returns a signal with the same shape and
dtype as ``x`` in which only the tracked tones are removed. Samples outside each
track's padded time span are bit-exact; inside, only a narrow corridor around
f(t) is affected.

Methods
-------
"subtract" (default): coherent cancellation. For each track:
  1. f(t) is interpolated to the sample rate (cubic spline; outside the
     detected span it is extrapolated with a decaying end slope) over
     [t0 - pad_s, t1 + pad_s], and integrated to the phase phi(t).
  2. The signal is heterodyned to baseband, z = x * exp(-j phi), and low-passed
     to a narrow bandwidth (1-3 Hz, wider for tracks with steeper sweeps) to
     get the slowly varying complex envelope a(t) of the tone.
  3. Frequency refinement: a first, wider (2x) pass measures the residual
     phase of a(t); that phase (i.e. the detector's frequency error) is
     folded into phi and the envelope is re-estimated at the narrow bandwidth.
     That keeps cancellation locked when the detected f(t) has error.
  4. The tone 2 Re(a(t) exp(j phi)) is subtracted, weighted by a raised-cosine
     fade over the pad regions.
  Tracks are processed one after another on the running residual (strongest
  first), so overlapping or crossing tracks and harmonics work.
  The low-pass is three cascaded centered box filters (sinc^3, deep nulls
  at multiples of ~100 Hz) followed by decimation to ~100 Hz, a zero-phase
  Butterworth at the envelope bandwidth, and cubic interpolation back to full
  rate. Cost is O(n) per track per channel over the track span only.

"mask": STFT-domain time-varying notch of half-width ``cfg.width_hz`` (+ the
  frequency sweep within one STFT window) around f(t). It runs on a decimated
  copy of the low band; only the *removed component* is upsampled and
  subtracted from the full-rate signal, so everything outside the corridor
  (all of the high band, and low-band content away from f(t)) reconstructs
  exactly, apart from the tiny corridor leakage.

``cfg.harden``: after "subtract" (or as part of "mask"), the corridor is
  masked and refilled with complex Gaussian noise whose level matches the
  local background in the bins just outside the corridor ("inpainting"). That
  leaves neither a residual tone nor a visible notch, which an attacker could
  otherwise trace as the Doppler curve. Uses a fixed RNG seed for repeatable
  output.

Per-track width: a track may carry ``width_hz`` (half-width of its measured
  spread, e.g. a jittery prop tone that is +-10 Hz wide after dechirping);
  ``None``/absent = ``cfg.width_hz``. It is read with ``getattr`` (floored at
  ``cfg.width_hz``, capped at WIDTH_MAX_HZ) and sizes that track's corridor
  everywhere: coherence flanks, fill, do-no-harm and the TF statistics. Coherent
  subtraction cannot follow a wide component, so for a wide track the fill
  (harden) does the work; in "mask" mode wide tracks take the VK + fill path.

Tracks whose ``kind`` is not in ``cfg.remove_kinds`` are ignored.
``Track.amp`` is not used; the envelope is estimated from the signal.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, uniform_filter1d
from scipy.linalg import solveh_banded
from scipy.signal import butter, istft, resample_poly, sosfiltfilt, sosfreqz, stft

from .types import RemoveConfig, Track

ENV_RATE = 100.0  # Hz, decimated rate of the complex envelope
RS_WINDOW = ("kaiser", 10.0)  # polyphase resampling window (deep image rejection)
EXTRAP_TAU = 1.0  # s, decay constant of the end-slope extrapolation
EXTRAP_MAX_HZ = 2.0  # extrapolated f(t) never moves more than this off the end value
REFINE = True  # local frequency refinement before the final demod (subtract, vk)
REFINE_SNR = (10.0, 100.0)  # tone/background power ratio: refinement trust ramps 0 -> 1


# ----------------------------------------------------------------------------- helpers


def _clean_track(tr: Track):
    t = np.asarray(tr.t, float).ravel()
    f = np.asarray(tr.f, float).ravel()
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    if len(t) == 0:
        return None
    t, idx = np.unique(t, return_index=True)
    return t, f[idx]


def _end_slope(t, f, at_end: bool) -> float:
    if len(t) < 2:
        return 0.0
    if at_end:
        m = t >= t[-1] - 1.0
    else:
        m = t <= t[0] + 1.0
    if m.sum() < 2:
        m = slice(-2, None) if at_end else slice(0, 2)
    tt, ff = t[m], f[m]
    if np.ptp(tt) <= 0:
        return 0.0
    return float(np.polyfit(tt, ff, 1)[0])


_CURVES: dict = {}  # (id(t), id(f)) -> (t, f, spline, s0, s1); cleared per _run


def _curve_parts(t, f):
    key = (id(t), id(f))
    hit = _CURVES.get(key)
    if hit is not None and hit[0] is t and hit[1] is f:
        return hit[2:]
    cs = CubicSpline(t, f, bc_type="not-a-knot") if len(t) >= 4 else None
    parts = (cs, _end_slope(t, f, False), _end_slope(t, f, True))
    if len(_CURVES) > 20000:
        _CURVES.clear()
    _CURVES[key] = (t, f) + parts
    return parts


def _freq_curve(t, f, tq):
    """Instantaneous frequency at times tq: cubic spline inside [t0, t1],
    decaying-slope extrapolation outside. (Spline cached per track.)"""
    tq = np.asarray(tq, float)
    t0, t1 = t[0], t[-1]
    cs, s0, s1 = _curve_parts(t, f)
    if cs is not None:
        out = cs(np.clip(tq, t0, t1))
    elif len(t) >= 2:
        out = np.interp(tq, t, f)
    else:
        out = np.full_like(tq, f[0], dtype=float)
    lo, hi = tq < t0, tq > t1
    # never run far off the detected span (wrong/steep fits): |shift| <= EXTRAP_MAX_HZ
    if lo.any():
        d = t0 - tq[lo]
        out[lo] = f[0] - np.clip(
            s0 * EXTRAP_TAU * (1 - np.exp(-d / EXTRAP_TAU)), -EXTRAP_MAX_HZ, EXTRAP_MAX_HZ
        )
    if hi.any():
        d = tq[hi] - t1
        out[hi] = f[-1] + np.clip(
            s1 * EXTRAP_TAU * (1 - np.exp(-d / EXTRAP_TAU)), -EXTRAP_MAX_HZ, EXTRAP_MAX_HZ
        )
    return out


def _fade(tq, t0, t1, pad, file_dur):
    """1 on [t0, t1], raised-cosine ramps over the pads. No ramp at a file edge
    the track touches (within pad): the tone presumably continues past it."""
    w = np.ones_like(tq)
    if pad > 0:
        if t0 - pad > 0:
            m = tq < t0
            w[m] = 0.5 - 0.5 * np.cos(np.pi * np.clip((tq[m] - (t0 - pad)) / pad, 0, 1))
        if t1 + pad < file_dur:
            m = tq > t1
            w[m] = 0.5 - 0.5 * np.cos(np.pi * np.clip(((t1 + pad) - tq[m]) / pad, 0, 1))
    return w


def _env_bandwidth(t, f) -> float:
    """Envelope low-pass cutoff (Hz), 1-3 Hz, wider for steeper/curvier sweeps
    (where detected f(t) is less accurate)."""
    if len(t) < 3:
        return 1.5
    tt = np.linspace(t[0], t[-1], max(8, int((t[-1] - t[0]) * 4) + 1))
    ff = _freq_curve(t, f, tt)
    dt = tt[1] - tt[0] if len(tt) > 1 else 1.0
    slope = np.abs(np.gradient(ff, dt))
    curv = np.abs(np.gradient(np.gradient(ff, dt), dt))
    return float(
        np.clip(1.0 + 0.25 * np.percentile(slope, 95) + 0.5 * np.percentile(curv, 95), 1.0, 3.0)
    )


def _lowpass_env(z: np.ndarray, sr: int, bw: float, pre=None):
    """Narrow zero-phase low-pass of complex baseband z (n, ch): boxcar^3 +
    decimation to ~ENV_RATE, then a Butterworth-4 run forwards and backwards.
    ``pre`` = an existing (zbox, D, start) from _boxcar3_decimate.

    Returns (env_dec, D, start): envelope sampled at indices start + k*D.
    """
    zd, D, start = pre if pre is not None else _boxcar3_decimate(z, sr)
    fs_d = sr / D
    if zd.shape[0] > 15:
        sos = butter(4, min(bw, 0.45 * fs_d) / (fs_d / 2), output="sos")
        padlen = min(zd.shape[0] - 1, int(3 * fs_d / max(bw, 0.1)))
        zd = sosfiltfilt(sos, zd.real, axis=0, padlen=padlen) + 1j * sosfiltfilt(
            sos, zd.imag, axis=0, padlen=padlen
        )
    return zd, D, start


def _proximity_weight(t, f, others, tq, bw, pad):
    """0..1 per time in tq: 0 where another track is within ~2*bw of this one
    (refinement would lock onto it), 1 once it is >= 4*bw + 2 Hz away."""
    if not others:
        return None
    fi = _freq_curve(t, f, tq)
    wgt = np.ones_like(tq)
    for to, fo in others:
        act = (tq >= to[0] - pad) & (tq <= to[-1] + pad)
        if not act.any():
            continue
        dist = np.abs(fi[act] - _freq_curve(to, fo, tq[act]))
        wgt[act] = np.minimum(wgt[act], np.clip((dist - 2 * bw) / (2 * bw + 2.0), 0, 1))
    return wgt


def _phase_correction(z: np.ndarray, sr: int, bw: float, weight: np.ndarray | None = None):
    """Local frequency refinement (a smoothed, SNR-gated PLL step).

    Measures the residual phase slope of the tone in baseband z (n, ch) with a
    2*bw envelope filter and returns a phase correction (n,) to add to phi.
    The correction is limited to +-bw Hz and weighted by the tone's local SNR
    against the surrounding +-50 Hz baseband, so where no tone is present it does
    nothing (otherwise it would lock onto a neighbouring speech/music partial).
    """
    n = z.shape[0]
    D = max(1, int(round(sr / ENV_RATE)))
    fs_d = sr / D
    zd, D, st = _lowpass_env(z, sr, 2 * bw)
    m = zd.shape[0]
    if m < 8:
        return None
    # broadband reference: boxcar-decimated baseband (~ +-fs_d/2 around f(t))
    zr = z.real, z.imag
    wide = np.stack([uniform_filter1d(p, D, axis=0, mode="reflect")[st::D] for p in zr], -1)
    wide = wide[..., 0] + 1j * wide[..., 1]
    k = min(m, wide.shape[0])
    zd, wide = zd[:k], wide[:k]
    smooth = max(1, int(fs_d * 1.0))
    p_tone = uniform_filter1d((np.abs(zd) ** 2).sum(1), smooth, mode="nearest")
    p_wide = uniform_filter1d((np.abs(wide) ** 2).sum(1), smooth, mode="nearest")
    dens = np.maximum(p_wide - p_tone, 1e-30) / fs_d  # per-Hz background
    snr = p_tone / (dens * 4 * bw) - 1.0
    lo, hi = REFINE_SNR
    conf = np.clip((snr - lo) / (hi - lo), 0.0, 1.0)
    if weight is not None:
        conf = conf * weight[np.minimum(st + D * np.arange(k), n - 1)]
    comb = zd.sum(axis=1)
    dph = np.angle(comb[1:] * np.conj(comb[:-1]))  # wrapped phase increments
    df = np.clip(dph * fs_d / (2 * np.pi), -bw, bw) * np.minimum(conf[1:], conf[:-1])
    df = uniform_filter1d(df, max(1, int(fs_d * 0.25)), mode="nearest")
    ph = np.concatenate([[0.0], np.cumsum(df) * 2 * np.pi / fs_d])
    idx = st + D * np.arange(k)
    if k < 4:
        return None
    return CubicSpline(idx, ph, bc_type="natural")(np.arange(n))


def _upsample_env(zd, D, start, n):
    idx = start + D * np.arange(zd.shape[0])
    if zd.shape[0] >= 4:
        cs = CubicSpline(idx, zd, axis=0, bc_type="natural", extrapolate=True)
        return cs(np.arange(n))
    out = np.empty((n, zd.shape[1]), complex)
    for c in range(zd.shape[1]):
        out[:, c] = np.interp(np.arange(n), idx, zd[:, c].real) + 1j * np.interp(
            np.arange(n), idx, zd[:, c].imag
        )
    return out


def _span(t, pad, sr, n):
    a = max(0, int(np.floor((t[0] - pad) * sr)))
    b = min(n, int(np.ceil((t[-1] + pad) * sr)) + 1)
    return a, b


@dataclass(eq=False)
class _Item:
    """One selected track: cleaned (t, f), detector confidence, per-track stats."""

    t: np.ndarray
    f: np.ndarray
    score: float
    strength: float
    index: int  # position in the caller's list
    stats: dict = field(default_factory=dict)
    fill: tuple | None = None  # (times, per-Hz level) of a harden fill
    width: float = 2.0  # corridor half-width (Hz) for this track
    wide: bool = False  # width_hz measured > cfg.width_hz (skirt step)
    skirt: tuple | None = None  # (times, extent Hz) of the skirt zone

    @property
    def tf(self):
        return self.t, self.f


WIDTH_MAX_HZ = 20.0  # cap on a per-track width_hz (flanks must fit the ~100 Hz baseband)


def _track_width(tr, cfg) -> float:
    """Corridor half-width for one track: its own ``width_hz`` if set (measured
    spread), never narrower than cfg.width_hz, at most WIDTH_MAX_HZ."""
    w = getattr(tr, "width_hz", None)
    try:
        w = float(w) if w is not None else None
    except (TypeError, ValueError):
        w = None
    if w is None or not np.isfinite(w):
        return float(cfg.width_hz)
    return float(np.clip(w, cfg.width_hz, WIDTH_MAX_HZ))


def _select(tracks, cfg, sr) -> list[_Item]:
    out = []
    for i, tr in enumerate(tracks):
        if cfg.remove_kinds and tr.kind not in cfg.remove_kinds:
            continue
        c = _clean_track(tr)
        if c is None:
            continue
        t, f = c
        if np.nanmax(f) <= 0 or np.nanmin(f) >= sr / 2:
            continue
        a = np.asarray(tr.amp, float)
        strength = float(np.nanmean(a)) if a.size and np.isfinite(a).any() else 0.0
        out.append(
            _Item(
                t,
                np.clip(f, 1.0, sr / 2 - 1.0),
                float(tr.doppler_score or 0.0),
                strength,
                i,
                width=_track_width(tr, cfg),
                wide=_track_width(tr, cfg) > cfg.width_hz + 1e-9,
            )
        )
    out.sort(key=lambda it: (-it.score, -(it.t[-1] - it.t[0])))
    out = _dedupe(out)
    out.sort(key=lambda it: -it.strength)  # strongest first (sequential methods)
    return out


DEDUPE_HZ = 1.0  # tracks closer than this over most of their overlap are one tone


def _dedupe(items: list[_Item]) -> list[_Item]:
    """Drop tracks that follow (almost) the same tone as an already kept, better
    track (higher score / longer) over >= 80% of their span: detectors return
    near-duplicates, and solving for the same tone twice is wasted work (and
    ill-posed in a joint VK solve). The kept track is extended to cover the
    union of both spans if the duplicate reaches further."""
    kept: list[_Item] = []
    for it in items:
        dup = False
        imn, imx = float(np.min(it.f)) - DEDUPE_HZ, float(np.max(it.f)) + DEDUPE_HZ
        for k in kept:
            lo, hi = max(it.t[0], k.t[0]), min(it.t[-1], k.t[-1])
            span = it.t[-1] - it.t[0]
            if hi <= lo or (hi - lo) < 0.8 * max(span, 1e-9):
                continue
            if float(np.min(k.f)) > imx or float(np.max(k.f)) < imn:
                continue
            tq = np.linspace(lo, hi, max(4, int((hi - lo) * 2)))
            if (
                np.median(np.abs(_freq_curve(it.t, it.f, tq) - _freq_curve(k.t, k.f, tq)))
                < DEDUPE_HZ
            ):
                dup = True
                break
        if not dup:
            kept.append(it)
    return kept


# ----------------------------------------------------------------------------- baseband tools

FILL_EDGE_HZ = 0.75  # raised-cosine transition of the corridor filter
FILL_FLANK_HZ = 8.0  # flanking bands reach out to +-8 Hz from f(t)
FILL_SUB_HZ = 1.0  # flank sub-band width
FILL_GAP_HZ = 1.0  # gap between corridor edge and the first flank (estimator skirts)
FILL_MIN_FLANK_HZ = 6.0  # flank width per side, at least (>= 12 bins for the median)
FILL_FRAME_S = 0.25  # level/envelope tracking time constant (attacker hop)
FILL_LONG_S = 2.0  # time scale of the robust (quantile) level
FILL_QUANTILE = 35.0  # robust low quantile (mask method's flank estimate) ...
FILL_Q_CAL = 1.30  # ... times this = the mean for pure noise (calibrated: q35 ~ 0.77 mean)
FILL_MED_CAL = 1.31  # median-of-sub-bands -> mean (calibrated below; ~1/ln 2 for raw powers)
CROSS_DB = 8.0  # corridor this far above the flank level = crossing partial
CROSS_MAX_S = 1.5  # ... but only for brief excursions (residual tones persist)
# A track with a measured width_hz > cfg.width_hz gets NO crossing-partial keep: its
# corridor IS the jittery target, whose bursts sit 10-20 dB over the (skirt-scaled)
# flanks and would be read as crossings. Cost: a partial crossing a <= +-12 Hz core
# is filled for <= CROSS_MAX_S.
COH_SMOOTH_S = 1.5  # coherence decision time scale
COH_ENV = (1.8, 3.5)  # envelope power / expected noise power: subtract nothing .. all
COH_CORR = (1.6, 2.6)  # same for the whole corridor (method-agnostic gate)
COH_MIN_S = 2.0  # coherent episodes shorter than this are foreground crossings
DNH_LIMIT_DB = 0.2  # do-no-harm: corridor energy may not rise more than this
DNH_BLOCKS_S = (0.25, 0.75)  # ... on every one of these block scales
DNH_VALLEY_DB = 3.0  # fill <= input corridor's own ~2 s level + this (no painting valleys)
DNH_VALLEY_SHORT_DB = 6.0  # ... and <= its ~0.25 s level + this
DNH_FILL_MARGIN_DB = (3.0, 1.5)  # per scale: natural fluctuation allowed to a noise fill
# above max(input, flank level). (A strict per-block
# "fill <= input" keeps only low draws: a ~2 dB notch.)
SAFE_EDGE_HZ = 1.0  # transition of the do-no-harm corridor


class _BB:
    """Reflect-padded FFT of a complex baseband z (m, ch) at rate fs, for
    zero-phase frequency-domain band filtering. With boxcar=True the boxcar^3
    droop of _boxcar3_decimate is compensated."""

    def __init__(self, z: np.ndarray, fs: float, boxcar: bool = False):
        m = z.shape[0]
        P = min(m - 1, int(4 * fs)) if m > 1 else 0
        zp = np.concatenate([z[P:0:-1], z, z[-2 : -P - 2 : -1]], axis=0) if P > 0 else z
        self.m, self.P, self.fs, self.L = m, P, fs, zp.shape[0]
        self.z, self.boxcar = z, boxcar
        self.F = np.fft.fftfreq(self.L, 1 / fs)
        self.Z = np.fft.fft(zp, axis=0)
        self.droop = np.maximum(np.sinc(self.F / fs) ** 3, 1e-3) if boxcar else np.ones(self.L)

    def filt(self, mask: np.ndarray) -> np.ndarray:
        return np.fft.ifft((mask / self.droop)[:, None] * self.Z, axis=0)[self.P : self.P + self.m]

    def corridor(self, W: float, E: float) -> np.ndarray:
        aF = np.abs(self.F)
        return np.where(
            aF <= W, 1.0, np.where(aF < W + E, 0.5 + 0.5 * np.cos(np.pi * (aF - W) / E), 0.0)
        )

    def noise_bw(self, mask: np.ndarray) -> float:
        return float(np.sum(mask**2) * self.fs / self.L)


def _gsm_nan(v: np.ndarray, n: int) -> np.ndarray:
    """Gaussian smoothing along axis -1 that ignores NaNs."""
    ok = np.isfinite(v)
    num = gaussian_filter1d(np.where(ok, v, 0.0), max(n, 1) / 2.0, axis=-1, mode="nearest")
    den = gaussian_filter1d(ok.astype(float), max(n, 1) / 2.0, axis=-1, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.2, num / np.maximum(den, 1e-12), np.nan)


def _flank_density(bb: _BB, W: float, E: float, others_off, nf: int):
    """Background next to a track, from the flank bins W+E+gap .. ~+-9 Hz.

    One short-time FFT of the baseband (1 Hz bins = 1 s Hann window, hop ~nf/4)
    gives every flank bin's power per frame. Level per frame: MEDIAN across the
    flank bins (ignores tonal peaks), corrected to a mean (FILL_MED_CAL ~ 1/ln 2
    for exponential bin powers). Bins overlapping another track's corridor
    (others_off: list of (m,) offsets, NaN where inactive) are excluded. No
    long-scale smoothing or capping: the level follows the flanks' own dynamics
    (syllables, gusts). The inter-channel structure (complex coherence) comes
    from slowly smoothed cross-spectra and is scaled to that level.
    Returns (dens (m, ch, ch) per-Hz covariance, level (m,) per-Hz trace).
    """
    z, fs = bb.z, bb.fs
    m, ch = z.shape
    nper = max(8, int(round(fs / FILL_SUB_HZ)))
    if m < nper:
        return np.zeros((m, ch, ch), complex), np.zeros(m)
    hop = max(1, nf // 4)
    win = np.hanning(nper + 2)[1:-1]
    zp = np.pad(z, ((nper // 2, nper - nper // 2), (0, 0)), mode="reflect")
    centers = np.arange(0, m, hop)
    frames = np.lib.stride_tricks.sliding_window_view(zp, nper, axis=0)[centers]  # (T, ch, nper)
    X = np.fft.fft(frames * win, axis=-1)
    fr = np.fft.fftfreq(nper, 1 / fs)
    norm = np.full(nper, fs * np.sum(win**2))
    if bb.boxcar:
        norm = norm * np.maximum(np.sinc(fr / fs) ** 6, 1e-6)  # boxcar^3 power droop
    afr = np.abs(fr)
    lo0 = W + E + FILL_GAP_HZ
    # at least FILL_MIN_FLANK_HZ per side, however wide the corridor: the median
    # must have enough bins to ignore a strong neighbouring partial
    flank = (afr >= lo0) & (afr <= max(FILL_FLANK_HZ + 1.0, lo0 + FILL_MIN_FLANK_HZ))
    if not flank.any():
        return np.zeros((m, ch, ch), complex), np.zeros(m)

    def usable(sel):
        ok_ = np.ones((len(centers), int(sel.sum())), bool)
        ffl_ = fr[sel]
        for off in others_off:
            oc = off[centers]
            with np.errstate(invalid="ignore"):
                ok_ &= ~(np.abs(ffl_[None, :] - oc[:, None]) < W + E + FILL_SUB_HZ / 2 + 0.5)
        P_ = (np.abs(X[:, :, sel]) ** 2).sum(1) / norm[sel][None, :]
        return P_, ok_

    def med(P_, ok_):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(np.where(ok_, P_, np.nan), axis=1)

    Xf = X[:, :, flank]  # (T, ch, K)
    P, ok = usable(flank)  # (T, K) per-Hz trace
    lev_f = med(P, ok)
    # other tracks' corridors can cover every flank bin (a line detected many
    # times, a dense comb): widen the flanks for those frames, and failing that
    # carry the level over in time. (A zero level here = no fill = a notch.)
    thin = ok.sum(1) < 4
    if thin.any():
        wide = (afr >= lo0) & (afr <= lo0 + 3 * FILL_MIN_FLANK_HZ)
        Pw, okw = usable(wide)
        lw = med(Pw, okw)
        use_w = thin & (okw.sum(1) >= 4)
        lev_f = np.where(use_w, lw, np.where(thin, np.nan, lev_f))
    good = np.isfinite(lev_f)
    if not good.any():
        return np.zeros((m, ch, ch), complex), np.zeros(m)
    if not good.all():
        lev_f = np.interp(np.arange(len(lev_f)), np.flatnonzero(good), lev_f[good])
    lev_f = lev_f * FILL_MED_CAL
    level = np.interp(np.arange(m), centers, lev_f)
    if ch == 1:
        return level[:, None, None].astype(complex), level
    # coherence structure from non-excluded, non-tonal flank bins, smoothed ~2 s
    with warnings.catch_warnings():  # frames with every bin excluded
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(ok, P, np.nan), axis=1)
    use = ok & (P <= 4 * np.nan_to_num(med)[:, None] + 1e-30)
    C = np.einsum("tik,tjk,tk->tij", Xf, np.conj(Xf), use / norm[flank][None, :])
    nl = max(1, int(FILL_LONG_S * fs / hop))
    C = _gsmooth(C.real, nl) + 1j * _gsmooth(C.imag, nl)
    trc = np.real(np.einsum("tii->t", C))
    shape = C / np.maximum(trc, 1e-30)[:, None, None]
    idx = np.clip(np.searchsorted(centers, np.arange(m)), 0, len(centers) - 1)
    return shape[idx] * level[:, None, None], level


def _near(items, i, pad, df_hz=15.0):
    """(t, f) of the other tracks that overlap item i in time (+pad) and come
    within df_hz of its frequency range: the only ones that can matter to it."""
    it = items[i]
    lo, hi = it.t[0] - pad, it.t[-1] + pad
    fmn, fmx = float(np.min(it.f)) - df_hz, float(np.max(it.f)) + df_hz
    return [
        o.tf
        for j, o in enumerate(items)
        if j != i
        and o.t[-1] + pad >= lo
        and o.t[0] - pad <= hi
        and float(np.max(o.f)) >= fmn
        and float(np.min(o.f)) <= fmx
    ]


def _others_offsets(item_f, others, tq, pad):
    """Frequency offsets (other - this) of other tracks at times tq (NaN where
    the other track is inactive)."""
    fi = _freq_curve(*item_f, tq)
    out = []
    for to, fo in others:
        act = (tq >= to[0] - pad) & (tq <= to[-1] + pad)
        if not act.any():
            continue
        off = np.full_like(tq, np.nan)
        off[act] = _freq_curve(to, fo, tq[act]) - fi[act]
        out.append(off)
    return out


def _ramp_gate(
    ratio: np.ndarray, lo_hi, n_erode: int, n_smooth: int, n_min_open: int = 0
) -> np.ndarray:
    """0..1 weight from a power ratio: 0 below lo, 1 above hi. Open episodes
    shorter than n_min_open samples are discarded (a Doppler tone persists; a
    brief coherent episode is a speech/music partial crossing the corridor).
    Eroded (never opens wider than the evidence) and smoothed (crossfades)."""
    lo, hi = lo_hi
    g = np.clip((ratio - lo) / (hi - lo), 0.0, 1.0)
    if n_min_open > 1:
        g[_short_runs(g > 0.5, n_min_open - 1)] = 0.0
        g[g <= 0.5] = np.minimum(g[g <= 0.5], 0.0)  # no partial opening outside episodes
    g = minimum_filter1d(g, max(1, n_erode), mode="nearest")
    return np.clip(_gsmooth(g, n_smooth), 0.0, 1.0)


def _coherence(env: np.ndarray, level_per_hz: np.ndarray, ch: int, Bn: float, fs: float):
    """(c) Envelope coherence gate: estimated tone envelope power vs the noise
    power the estimator would pass with no tone present (flank density x its
    noise bandwidth). ~1 means the 'envelope' is just noise -> subtract nothing."""
    ns = int(COH_SMOOTH_S * fs)
    pe = _gsmooth((np.abs(env) ** 2).mean(1), ns)
    # compare like with like: the expected noise on the same time scale as pe
    exp_noise = np.maximum(_gsmooth(level_per_hz, ns) / ch * Bn, 1e-30)
    ratio = pe / exp_noise
    return _ramp_gate(ratio, COH_ENV, int(0.5 * fs), int(0.25 * fs), int(COH_MIN_S * fs)), ratio


def _pad_continuity(gate: np.ndarray, tq: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """In the pads (outside the detected span) a tone may only be removed if it
    was coherent at the adjacent end of the span: the extrapolated f(t) must not
    'discover' an unrelated stationary partial (e.g. a music note) to subtract."""
    g = gate.copy()
    for side, sel_pad, sel_in in (
        (0, tq < t0, (tq >= t0) & (tq <= t0 + 0.5)),
        (1, tq > t1, (tq <= t1) & (tq >= t1 - 0.5)),
    ):
        if sel_pad.any():
            lim = float(gate[sel_in].min()) if sel_in.any() else 0.0
            g[sel_pad] = np.minimum(g[sel_pad], lim)
    return g


# ----------------------------------------------------------------------------- subtract


def _butter_enbw(bw: float, fs_d: float) -> float:
    sos = butter(4, min(bw, 0.45 * fs_d) / (fs_d / 2), output="sos")
    w, h = sosfreqz(sos, worN=8192, fs=fs_d)
    return float(2 * np.trapezoid(np.abs(h) ** 4, w))  # zero-phase: |H|^2, power |H|^4


def _subtract_one(
    y: np.ndarray,
    sr: int,
    item: _Item,
    pad: float,
    refine: bool = True,
    others=(),
    width_hz: float = 2.0,
):
    """Subtract one tracked tone from y (n, ch) in place, only where its
    estimated envelope is coherent (a tone is really there)."""
    t, f = item.tf
    n = y.shape[0]
    a, b = _span(t, pad, sr, n)
    if b - a < 8 * sr / ENV_RATE:
        return
    tq = (a + np.arange(b - a)) / sr
    fi = _freq_curve(t, f, tq)
    phi = 2 * np.pi * np.cumsum(fi) / sr
    phi -= phi[0]
    w = _fade(tq, t[0], t[-1], pad, n / sr)
    seg = y[a:b].astype(np.float64)
    ch = seg.shape[1]
    bw = _env_bandwidth(t, f)
    others = list(others)
    if refine:
        z = seg * np.exp(-1j * phi)[:, None]
        corr = _phase_correction(z, sr, bw, _proximity_weight(t, f, others, tq, bw, pad))
        if corr is not None:
            phi = phi + corr
    z = seg * np.exp(-1j * phi)[:, None]
    pre = _boxcar3_decimate(z, sr)
    zd, D, st = _lowpass_env(z, sr, bw, pre=pre)
    fs_d = sr / D
    # (c) coherence gate against the flank noise level
    td = tq[np.minimum(st + D * np.arange(zd.shape[0]), len(tq) - 1)]
    bb = _BB(pre[0], fs_d, boxcar=True)
    _, level = _flank_density(
        bb,
        width_hz,
        FILL_EDGE_HZ,
        _others_offsets((t, f), others, td, pad),
        max(1, int(FILL_FRAME_S * fs_d)),
    )
    gate, _ = _coherence(zd, level, ch, _butter_enbw(bw, fs_d), fs_d)
    gate = _pad_continuity(gate, td, t[0], t[-1])
    item.stats["coherent_frac"] = (
        float(gate[(td >= t[0]) & (td <= t[-1])].mean())
        if ((td >= t[0]) & (td <= t[-1])).any()
        else 0.0
    )
    if gate.max() < 1e-3:
        return
    env = _upsample_env(zd * gate[:, None], D, st, b - a)
    tone = 2.0 * np.real(env * np.exp(1j * phi)[:, None])
    y[a:b] = seg - w[:, None] * tone


# ----------------------------------------------------------------------------- low band


def _work_rate(sr: int, fmax: float) -> int:
    """Decimated rate for low-band processing. fmax must sit below 0.65*Nyquist,
    where the kaiser-10 polyphase filters are flat to < 0.01 dB."""
    for fw in (1000, 2000, 4000, 8000, 16000, 24000):
        if fw >= sr:
            break
        if 0.65 * fw / 2 >= fmax:
            return fw
    return sr


def _resampler(sr: int, fw: int):
    r = Fraction(fw, sr).limit_denominator(1000)
    up, down = r.numerator, r.denominator
    if (up, down) == (1, 1):
        return (lambda v: v.copy()), (lambda v: v)
    return (
        lambda v: resample_poly(v, up, down, axis=0, window=RS_WINDOW),
        lambda v: resample_poly(v, down, up, axis=0, window=RS_WINDOW),
    )


def _groups(items, pad, extra):
    """Cluster tracks whose padded spans overlap (processed jointly)."""
    items = sorted(items, key=lambda it: it[0][0])
    groups, cur, cur_end = [], [], -np.inf
    for t, f in items:
        if cur and t[0] - pad - extra > cur_end:
            groups.append(cur)
            cur, cur_end = [], -np.inf
        cur.append((t, f))
        cur_end = max(cur_end, t[-1] + pad + extra)
    if cur:
        groups.append(cur)
    return groups


def _boxcar3_decimate(z: np.ndarray, sr: int):
    """Three centred box filters + decimation of complex baseband z (n, ch) to
    ~ENV_RATE. Returns (zd, D, start)."""
    n = z.shape[0]
    D = max(1, int(round(sr / ENV_RATE)))
    zr = np.empty((n, z.shape[1], 2), np.float64)
    zr[..., 0], zr[..., 1] = z.real, z.imag
    for _ in range(3):
        zr = uniform_filter1d(zr, D, axis=0, mode="reflect")
    st = D // 2
    return zr[st::D, :, 0] + 1j * zr[st::D, :, 1], D, st


def _short_runs(mask: np.ndarray, max_len: int) -> np.ndarray:
    """Keep only True-runs of length <= max_len."""
    out = np.zeros_like(mask)
    if not mask.any():
        return out
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for s0, e0 in zip(starts, ends):
        if e0 - s0 <= max_len:
            out[s0:e0] = True
    return out


def _gsmooth(v: np.ndarray, n: int) -> np.ndarray:
    """Gaussian smoothing along axis 0 with roughly the support of an n-box
    (smooth spectrum: modulating a narrowband fill by it does not spread it)."""
    return gaussian_filter1d(v, max(n, 1) / 2.0, axis=0, mode="nearest", truncate=4.0)


# ----------------------------------------------------------------------------- mask (STFT notch)


def _mask_segment(y, sr, items, cfg):
    """STFT-domain time-varying notch for a group of overlapping tracks. Only the
    removed (corridor) component is computed at the work rate and upsampled, so
    everything outside the corridor reconstructs exactly."""
    n = y.shape[0]
    pad = cfg.pad_s
    fmax = max(float(np.max(f)) for _, f in items) + cfg.width_hz + 10.0
    fw = _work_rate(sr, fmax)
    down_fn, up_fn = _resampler(sr, fw)
    nper = int(2 ** np.round(np.log2(fw * 1.0)))  # ~1 s window, ~1 Hz bins
    hop = nper // 4
    margin = (2 * nper / fw) + 0.5
    a = max(0, int((min(t[0] for t, _ in items) - pad - margin) * sr))
    b = min(n, int(np.ceil((max(t[-1] for t, _ in items) + pad + margin) * sr)))
    if b - a < 16:
        return
    seg = y[a:b].astype(np.float64)
    low = down_fn(seg)
    m = low.shape[0]
    fr, tt, S = stft(
        low,
        fs=fw,
        window="hann",
        nperseg=nper,
        noverlap=nper - hop,
        axis=0,
        boundary="even",
        padded=True,
    )  # (nf, ch, nt)
    tt_abs = tt + a / sr
    df = fr[1] - fr[0]
    win_s = nper / fw
    C = np.zeros((len(fr), len(tt)))
    for t, f in items:
        fc = _freq_curve(t, f, tt_abs)
        slope = np.abs(np.gradient(fc, tt_abs)) if len(tt_abs) > 1 else np.zeros_like(fc)
        half = cfg.width_hz + 0.5 * slope * win_s
        w = _fade(tt_abs, t[0], t[-1], pad, n / sr)
        d = np.abs(fr[:, None] - fc[None, :])
        taper = np.clip((half[None, :] + df - d) / df, 0, 1)
        taper = 0.5 - 0.5 * np.cos(np.pi * taper)
        # coherence gate: notch only where the corridor holds more than background
        P = (np.abs(S) ** 2).mean(axis=1)
        inner = d <= half[None, :]
        flank = (d > half[None, :] + 2 * df) & (d <= half[None, :] + 2 * df + 6.0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pin = np.nanmean(np.where(inner, P, np.nan), axis=0)
            pfl = np.nanpercentile(np.where(flank, P, np.nan), FILL_QUANTILE, axis=0) * FILL_Q_CAL
        ratio = np.nan_to_num(pin / np.maximum(pfl, 1e-30))
        gate = _ramp_gate(
            ratio, COH_CORR, max(1, int(0.5 / (hop / fw))), 1, int(COH_MIN_S / (hop / fw))
        )
        C = np.maximum(C, taper * (w * gate)[None, :])
    if not C.any():
        return
    _, d_low = istft(
        C[:, None, :] * S,
        fs=fw,
        window="hann",
        nperseg=nper,
        noverlap=nper - hop,
        time_axis=-1,
        freq_axis=0,
        boundary=True,
    )
    d_low = np.moveaxis(d_low, -1, 0)[:m]
    d = up_fn(d_low)
    L = b - a
    if d.shape[0] < L:
        d = np.concatenate([d, np.zeros((L - d.shape[0], d.shape[1]))], axis=0)
    y[a:b] = seg - d[:L]


def _vk_components(items: list[_Item], pad):
    """Connected components (lists of indices) of tracks that overlap in time AND
    come within VK_COUPLE_HZ of each other: only those need the joint solve."""
    K = len(items)
    parent = list(range(K))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    fmn = [float(np.min(it.f)) for it in items]
    fmx = [float(np.max(it.f)) for it in items]
    for i in range(K):
        ti, fi = items[i].tf
        for j in range(i + 1, K):
            tj, fj = items[j].tf
            lo, hi = max(ti[0], tj[0]) - pad, min(ti[-1], tj[-1]) + pad
            if hi < lo or fmn[j] > fmx[i] + VK_COUPLE_HZ or fmx[j] < fmn[i] - VK_COUPLE_HZ:
                continue
            tq = np.linspace(lo, hi, max(4, int((hi - lo) * 4)))
            if np.min(np.abs(_freq_curve(ti, fi, tq) - _freq_curve(tj, fj, tq))) < VK_COUPLE_HZ:
                parent[find(i)] = find(j)
    comps: dict[int, list] = {}
    for i in range(K):
        comps.setdefault(find(i), []).append(i)
    out = []
    for c in comps.values():
        if len(c) <= VK_MAX_K:
            out.append(c)
            continue
        c = sorted(c, key=lambda i: float(np.mean(items[i].f)))  # neighbours in frequency together
        out.extend(c[j : j + VK_MAX_K] for j in range(0, len(c), VK_MAX_K))
    return out


# ----------------------------------------------------------------------------- harden (fill)


def _harden_one(y, sr, item: _Item, cfg, rng, extra_hz: float = 0.0, others=()):
    """Replace the +-width_hz corridor around one track with track-locked noise
    matched to the local background (in place).

    Works in the track's baseband (signal heterodyned by its own phase, i.e. the
    emission-time-warped frame where the curve is a constant line at DC):
      * corridor filter H: 1 inside +-W, raised-cosine to 0 over FILL_EDGE_HZ,
        applied as an FFT mask on the ~100 Hz baseband, so it is confined.
      * power-complementary: keep sqrt(1 - H^2) of the original, add H * noise
        (no band-edge step).
      * level, frame by frame (FILL_FRAME_S), from the flanking 1 Hz sub-bands
        out to +-8 Hz, EXCLUDING other tracks' corridors, with a robust low
        quantile (FILL_QUANTILE) capping the long-scale level, so a fill never
        imports a neighbouring tone. Complex ch x ch covariance: the noise gets
        the flanks' inter-channel correlation (mono downmix matches too).
      * envelope matching (Chuang/Garg/Wu): the noise's own envelope is
        normalised away and replaced by the flanks' frame-level envelope.
      * crossing partials: where the corridor is CROSS_DB above the flank level
        for less than CROSS_MAX_S (a speech/music harmonic passing through), the
        original is kept instead of filled.
    """
    t, f = item.tf
    n = y.shape[0]
    pad = cfg.pad_s
    a, b = _span(t, pad, sr, n)
    if b - a < 8 * sr / ENV_RATE:
        return
    tq = (a + np.arange(b - a)) / sr
    phi = 2 * np.pi * np.cumsum(_freq_curve(t, f, tq)) / sr
    phi -= phi[0]
    w = _fade(tq, t[0], t[-1], pad, n / sr)
    seg = y[a:b].astype(np.float64)
    ch = seg.shape[1]
    carrier = np.exp(-1j * phi)
    zd, D, st = _boxcar3_decimate(seg * carrier[:, None], sr)
    fs_d = sr / D
    m = zd.shape[0]
    bb = _BB(zd, fs_d, boxcar=True)
    W = item.width + extra_hz
    H = bb.corridor(W, FILL_EDGE_HZ)
    keep = np.sqrt(np.clip(1 - H**2, 0, 1))
    removed = bb.filt(1 - keep)
    corridor = bb.filt(H)
    nf = max(1, int(round(FILL_FRAME_S * fs_d)))
    td = tq[np.minimum(st + D * np.arange(m), len(tq) - 1)]
    dens, level = _flank_density(
        bb, W, FILL_EDGE_HZ, _others_offsets((t, f), list(others), td, pad), nf
    )
    # cross-check in ABSOLUTE frequency (what the attacker's fixed-frequency STFT
    # sees): for a steep track crossing stationary partials (a harmonic comb),
    # the warped-frame flanks are smeared by the crossings and overstate the
    # between-lines background; then the absolute-frame level wins.
    lev_abs = _abs_flank_level(seg, sr, a, t, f, td, W + FILL_EDGE_HZ + FILL_GAP_HZ)
    if lev_abs is not None:
        # continuous cap (a jump would modulate the fill and splatter it):
        # r = warped/abs level; keep r <= 1.5, taper to 1 (abs level) at r >= 3
        r = level / np.maximum(lev_abs, 1e-30)
        r_new = np.where(r <= 1.5, r, np.where(r >= 3.0, 1.0, 1.5 - (r - 1.5) / 3.0))
        scl = np.clip(_gsmooth(r_new / np.maximum(r, 1e-30), nf), 0.0, 1.0)
        level = level * scl
        dens = dens * scl[:, None, None]
    Hbw = bb.noise_bw(H)

    # crossing-partial detection: corridor power vs flank-predicted power
    pc = _gsmooth((np.abs(corridor) ** 2).sum(1), nf)
    pf = np.maximum(level * Hbw, 1e-30)
    with np.errstate(over="ignore"):
        hot = (pc > pf * 10 ** (CROSS_DB / 10)) & (not item.wide)
    cross = _short_runs(hot, int(CROSS_MAX_S * fs_d))
    cross = np.clip(_gsmooth(cross.astype(float), 2 * nf) * 1.5, 0, 1)
    gate = 1.0 - cross  # 0 = keep original

    # noise: exactly white complex Gaussian (unit variance per sample = 1/fs_d per
    # Hz), shaped by the corridor filter H and coloured by chol(dens * fs_d). No
    # data-dependent normalisation, so its magnitudes stay Rayleigh in any STFT;
    # all level dynamics come from the flanks' frame-level median (dens).
    L = bb.L
    g = (rng.standard_normal((L, ch)) + 1j * rng.standard_normal((L, ch))) / np.sqrt(2)
    g = np.fft.ifft(H[:, None] * np.fft.fft(g, axis=0), axis=0)[bb.P : bb.P + m]
    dens = 0.5 * (dens + np.conj(np.swapaxes(dens, 1, 2))) * fs_d
    tr_d = np.real(np.einsum("mii->m", dens))
    try:
        Lc = np.linalg.cholesky(dens + (tr_d * 1e-6 + 1e-30)[:, None, None] * np.eye(ch))
    except np.linalg.LinAlgError:  # fall back: per-channel, uncorrelated
        Lc = np.sqrt(np.maximum(np.real(np.einsum("mii->mi", dens)), 0))[:, :, None] * np.eye(ch)
    noise = np.einsum("mij,mj->mi", Lc, g)

    R = _upsample_env(removed * gate[:, None], D, st, b - a)
    N = _upsample_env(noise * gate[:, None], D, st, b - a)
    c = 1 - np.sqrt(np.clip(1 - w**2, 0, 1))  # power-complementary time fade
    y[a:b] = seg - 2.0 * np.real((c[:, None] * R - w[:, None] * N) * np.conj(carrier)[:, None])
    item.stats["filled"] = True
    item.stats["fill_gate_mean"] = float(np.mean(gate))
    item.stats["fill_hot_frac"] = float(np.mean(hot))
    item.fill = (td, level * np.interp(td, tq, w) * gate)  # per-Hz fill level actually applied


# ----------------------------------------------------------------------------- skirt

SKIRT_MAX_HZ = 50.0  # skirt zone never reaches further than this from f(t)
SKIRT_BG_HZ = (55.0, 80.0)  # far ring (offsets from f(t)) that defines the background
SKIRT_RATE = 400  # complex baseband rate of the skirt step (Nyquist 200 Hz)
SKIRT_NPER_S = 1.0  # STFT window (1 Hz bins) ...
SKIRT_HOP_S = 0.125  # ... and hop
SKIRT_MEET_DB = 1.0  # the skirt ends where the profile is within this of the background
SKIRT_SMOOTH_S = 0.5  # time smoothing of the power used for the gain
SKIRT_EDGE_HZ = 3.0  # outer taper of the skirt zone
SKIRT_CORE_DB = 6.0  # a frame gets a skirt only if its core stands this far over background
SKIRT_HARD_DB = 10.0  # shoulder this far over background is too strong for an STFT gain
# (~20-25 dB dynamic range): the core fill is widened to cover it


def _skirt_one(y, sr, item: _Item, cfg, others=()):
    """Measured-wide tracks only (width_hz > cfg.width_hz): a jittery tone's
    dechirped spectrum has a slowly decaying skirt well beyond its core width,
    which the attacker's 2 s STFT integrates into a visible streak. In the
    track's warped baseband, for each STFT frame: the background is the median
    power of the far ring SKIRT_BG_HZ (other tracks excluded); the skirt extent
    is where the running minimum of the profile (outward from the core edge, both
    sides averaged) first comes within SKIRT_MEET_DB of it. Frames whose
    profile does not decay to the background within SKIRT_MAX_HZ (dense
    speech/harmonics: no shoulder to remove) or whose core is not
    SKIRT_CORE_DB over it get no skirt. Inside the zone (outside the core) the
    ring at offset r is scaled by sqrt(min(1, background / shoulder(r))), where
    shoulder(r) is that running minimum: the decaying shoulder is brought to the
    background (never below), and content riding above it (a crossing speech
    harmonic) is only attenuated as much as the shoulder under it.
    In place; only the change is added back."""
    t, f = item.tf
    n = y.shape[0]
    pad = cfg.pad_s
    a, b = _span(t, pad, sr, n)
    if b - a < int(4 * sr):
        return
    fv = min(SKIRT_RATE, sr)
    if 0.65 * fv / 2 < SKIRT_BG_HZ[1] + 5:
        return
    tq = (a + np.arange(b - a)) / sr
    phi = 2 * np.pi * np.cumsum(_freq_curve(t, f, tq)) / sr
    phi -= phi[0]
    w = _fade(tq, t[0], t[-1], pad, n / sr)
    carrier = np.exp(-1j * phi)
    seg = y[a:b].astype(np.float64)
    r = Fraction(fv, sr).limit_denominator(1000)
    up, down = r.numerator, r.denominator
    zb = resample_poly(seg * carrier[:, None], up, down, axis=0, window=RS_WINDOW)
    m, ch = zb.shape
    nper = int(round(SKIRT_NPER_S * fv))
    hop = int(round(SKIRT_HOP_S * fv))
    if m < 4 * nper:
        return
    F, tt, X = stft(
        zb,
        fs=fv,
        window="hann",
        nperseg=nper,
        noverlap=nper - hop,
        axis=0,
        return_onesided=False,
        boundary="even",
        padded=True,
    )  # (nf, ch, nt)
    P = (np.abs(X) ** 2).sum(1)  # (nf, nt)
    aF = np.abs(F)
    tl = a / sr + tt
    ring = (aF >= SKIRT_BG_HZ[0]) & (aF <= SKIRT_BG_HZ[1])
    okr = np.ones((ring.sum(), len(tt)), bool)
    for off in _others_offsets((t, f), list(others), tl, pad):
        with np.errstate(invalid="ignore"):
            okr &= ~(np.abs(F[ring][:, None] - off[None, :]) < cfg.width_hz + 4.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bg = np.nanmedian(np.where(okr, P[ring], np.nan), axis=0) * FILL_MED_CAL
    if not np.isfinite(bg).any():
        return
    bg = np.where(np.isfinite(bg), bg, np.nanmedian(bg))
    ns = max(1, int(round(FILL_LONG_S / SKIRT_HOP_S)))
    bg = _gsmooth(bg, ns)
    Ps = gaussian_filter1d(P, SKIRT_SMOOTH_S / SKIRT_HOP_S / 2, axis=1, mode="nearest")
    Ps = uniform_filter1d(Ps, 3, axis=0, mode="nearest")
    core = item.width + FILL_EDGE_HZ
    # per-frame extent from the side-averaged profile, running min outward
    offs = np.arange(int(np.ceil(core)), int(SKIRT_MAX_HZ) + 1)
    prof = np.stack(
        [0.5 * (Ps[np.argmin(np.abs(F - o))] + Ps[np.argmin(np.abs(F + o))]) for o in offs]
    )
    runmin = np.minimum.accumulate(prof, axis=0)
    meet = runmin <= bg[None, :] * 10 ** (SKIRT_MEET_DB / 10)
    pcore = Ps[aF <= core].mean(0)
    ok = meet.any(0) & (pcore >= bg * 10 ** (SKIRT_CORE_DB / 10))
    span = (tl >= t[0]) & (tl <= t[-1])
    # strong shoulder -> widen the core (exact long-FFT fill in _harden_one), so the
    # STFT gain below only handles the weak tail
    hard = runmin <= bg[None, :] * 10 ** (SKIRT_HARD_DB / 10)
    sel = ok & span & hard.any(0)
    if sel.any():
        r_hard = float(np.percentile(offs[np.argmax(hard, axis=0)][sel], 90))
        if r_hard > item.width:
            item.width = min(r_hard, WIDTH_MAX_HZ)
            item.stats["width_widened_hz"] = item.width
            core = item.width + FILL_EDGE_HZ
    ext = np.where(ok, offs[np.argmax(meet, axis=0)], core).astype(float)
    ext = np.clip(_gsmooth(ext, max(1, int(round(0.5 / SKIRT_HOP_S)))), core, SKIRT_MAX_HZ)
    zone = np.clip((ext[None, :] + SKIRT_EDGE_HZ - aF[:, None]) / SKIRT_EDGE_HZ, 0, 1)
    zone = 0.5 - 0.5 * np.cos(np.pi * zone)
    zone[aF <= core] = 0.0  # the core was filled already
    oi = np.clip(np.round(aF - offs[0]).astype(int), 0, len(offs) - 1)
    shoulder = runmin[oi]  # (nf, nt) envelope at |F|
    g = np.sqrt(np.clip(bg[None, :] / np.maximum(shoulder, 1e-30), 0, 1))
    g = 1 - zone * (1 - g)  # 1 outside the zone
    tw = np.interp(tl, tq, w)  # track fade
    g = 1 - tw[None, :] * (1 - g)
    if np.all(g > 0.999):
        return
    _, dz = istft(
        (g - 1)[:, None, :] * X,
        fs=fv,
        window="hann",
        nperseg=nper,
        noverlap=nper - hop,
        time_axis=-1,
        freq_axis=0,
        input_onesided=False,
        boundary=True,
    )
    dz = np.moveaxis(dz, -1, 0)[:m]
    if dz.shape[0] < m:
        dz = np.concatenate([dz, np.zeros((m - dz.shape[0], ch), complex)], axis=0)
    d = resample_poly(dz, down, up, axis=0, window=RS_WINDOW)
    L = b - a
    if d.shape[0] < L:
        d = np.concatenate([d, np.zeros((L - d.shape[0], ch), complex)], axis=0)
    y[a:b] = seg + 2.0 * np.real(d[:L] * np.conj(carrier)[:, None])
    item.stats["skirt_hz"] = float(np.median(ext[span])) if span.any() else float("nan")
    item.skirt = (tl, ext * tw)


# ----------------------------------------------------------------------------- Vold-Kalman

VK_ORDER = 3  # p: VK structural-equation order (2-3 recommended; 3 measured best)
VK_COUPLE_HZ = 8.0  # tracks are solved jointly only if they come this close (~2x max bw)
VK_MAX_K = 4  # larger components are split (joint-solve cost ~ K^3)
VK_MAX_FV = 400  # coupled components needing a faster complex rate are split
VK_CHUNK_S = 120.0  # long groups are solved in overlapping chunks
VK_OVERLAP_S = 10.0


def _vk_r2(bw: float, fs: float, p: int) -> float:
    """Weight r^2 so the VK envelope response 1/(1+r^2 (2-2cos 2 pi df/fs)^p)
    is -3 dB at bw."""
    return (np.sqrt(2) - 1) / (2 - 2 * np.cos(2 * np.pi * bw / fs)) ** p


def _dtd_from_coeffs(m, c):
    """Diagonals 0..p of D^T D, D the difference operator with coefficients c."""
    p = len(c) - 1
    diags = [np.zeros(m - d) for d in range(p + 1)]
    rows = m - p
    if rows <= 0:
        return diags
    for i in range(p + 1):
        for j in range(i, p + 1):
            diags[j - i][i : i + rows] += c[i] * c[j]
    return diags


def _diff_coeffs(p):
    from math import comb

    return np.array([(-1) ** (p - i) * comb(p, i) for i in range(p + 1)], float)


def _vk_solve(low: np.ndarray, thetas, acts, r2s, p: int):
    """Joint (multi-order, 2nd-generation) VK: minimise
        sum_n |y_n - sum_k a_kn x_kn e^{j th_kn}|^2 + sum_k r_k^2 |D^p x_k|^2
    for complex envelopes x_k. Unknowns interleaved (n*K + k) so the normal
    matrix is Hermitian banded with half-bandwidth p*K; solved with LAPACK pbsv.
    low: (m, ch) real or complex baseband; returns x (K, m, ch)."""
    m, ch = low.shape
    K = len(thetas)
    N = m * K
    u = p * K
    ab = np.zeros((u + 1, N), complex)  # upper form: ab[u + i - j, j]
    E = np.stack([a * np.exp(1j * th) for a, th in zip(acts, thetas)])  # (K, m)
    # coupling (same n): M[(n,k),(n,l)] = conj(E_k) E_l for l >= k
    for k in range(K):
        for l in range(k, K):
            j = np.arange(m) * K + l
            ab[u - (l - k), j] += np.conj(E[k]) * E[l]
    c = _diff_coeffs(p)
    for k in range(K):
        diags = _dtd_from_coeffs(m, c)
        for d, dg in enumerate(diags):
            # M[(n,k),(n+d,k)] at column j = (n+d)*K + k, offset d*K
            j = (np.arange(len(dg)) + d) * K + k
            ab[u - d * K, j] += r2s[k] * dg
    # inactive samples (outside a track's padded span) get a unit ridge pulling the
    # envelope to 0 there (its fade weight is 0 anyway); this bounds the smallest
    # eigenvalue at ~1 so pbsv stays accurate even for p = 3 (r^2 ~ 1e12).
    inact = np.stack([1.0 - a for a in acts], axis=1).reshape(N)
    ab[u] += inact + 1e-9
    rhs = (np.conj(E)[:, :, None] * low[None, :, :]).transpose(1, 0, 2).reshape(N, ch)
    x = solveh_banded(ab, rhs, lower=False, check_finite=False)
    return x.reshape(m, K, ch).transpose(1, 0, 2)


def _vk_enbw(r2: float, fv: float, p: int) -> float:
    fr = np.linspace(-fv / 2, fv / 2, 20001)
    h = 1.0 / (1.0 + r2 * (2 - 2 * np.cos(2 * np.pi * fr / fv)) ** p)
    return float(np.trapezoid(h**2, fr))


def _vk_group(y, sr, items: list[_Item], cfg, refine=True, p=VK_ORDER, all_items=()):
    """Joint 2nd-generation VK for one component of nearby tracks.

    The component is heterodyned by a time-varying reference phase (the mean of
    its tracks' f(t), i.e. warped so the component sits at DC) and decimated to
    a small complex rate fv (100 Hz for a single track), so the VK system stays
    well conditioned whatever the sweep. Envelopes that are not coherent (c) are
    not subtracted. Only the removed component is upsampled back."""
    n = y.shape[0]
    pad = cfg.pad_s
    a = max(0, int((min(it.t[0] for it in items) - pad - 1.0) * sr))
    b = min(n, int(np.ceil((max(it.t[-1] for it in items) + pad + 1.0) * sr)))
    if b - a < 8 * sr / ENV_RATE:
        return
    tseg = (a + np.arange(b - a)) / sr
    fks = [_freq_curve(it.t, it.f, tseg) for it in items]
    fref = np.mean(fks, axis=0)
    bh = max(float(np.max(np.abs(fk - fref))) for fk in fks) + 10.0
    fv = 100
    while 0.65 * fv / 2 < bh and fv < sr:
        fv *= 2
    fv = min(fv, sr)
    if fv > VK_MAX_FV and len(items) > 1:  # too wide to couple: solve singly
        for it in items:
            _vk_group(y, sr, [it], cfg, refine, p, all_items)
        return
    r = Fraction(fv, sr).limit_denominator(1000)
    up, down = r.numerator, r.denominator
    seg = y[a:b].astype(np.float64)
    ch = seg.shape[1]
    psi = 2 * np.pi * np.cumsum(fref) / sr
    shift = np.exp(-1j * psi)
    zb = resample_poly(seg * shift[:, None], up, down, axis=0, window=RS_WINDOW)
    m = zb.shape[0]
    tl = a / sr + np.arange(m) / fv
    idx = np.minimum(np.round(np.arange(m) * sr / fv).astype(int), len(tseg) - 1)
    thetas, acts, fades, r2s, bws = [], [], [], [], []
    for k, it in enumerate(items):
        th = (2 * np.pi * np.cumsum(fks[k] - fref) / sr)[idx]  # phase relative to the reference
        bw = _env_bandwidth(it.t, it.f)
        if refine:
            oth = [o.tf for o in items if o is not it]
            corr = _phase_correction(
                zb * np.exp(-1j * th)[:, None],
                fv,
                bw,
                _proximity_weight(it.t, it.f, oth, tl, bw, pad),
            )
            if corr is not None:
                th = th + corr
        thetas.append(th)
        acts.append(((tl >= it.t[0] - pad) & (tl <= it.t[-1] + pad)).astype(float))
        fades.append(_fade(tl, it.t[0], it.t[-1], pad, n / sr))
        r2s.append(_vk_r2(bw, fv, p))
        bws.append(bw)
    X = np.zeros((len(items), m, ch), complex)
    chunk, ov = int(VK_CHUNK_S * fv), int(VK_OVERLAP_S * fv)
    s0 = 0
    try:
        while s0 < m:
            e0 = min(m, s0 + chunk)
            lo, hi = max(0, s0 - ov), min(m, e0 + ov)
            sl = slice(lo, hi)
            x = _vk_solve(zb[sl], [th[sl] for th in thetas], [ac[sl] for ac in acts], r2s, p)
            X[:, s0:e0] = x[:, s0 - lo : e0 - lo]
            s0 = e0
    except np.linalg.LinAlgError:
        if len(items) > 1:
            for it in items:
                _vk_group(y, sr, [it], cfg, refine, p, all_items)
        return  # single track: do no harm, skip
    rem = np.zeros_like(zb)
    for k, it in enumerate(items):
        # (c) coherence of this track's envelope against its own flank noise
        zk = zb * np.exp(-1j * thetas[k])[:, None]
        ai = list(all_items)
        pos = next((j for j, o in enumerate(ai) if o is it), None)
        oth = _near(ai, pos, pad) if pos is not None else []
        _, level = _flank_density(
            _BB(zk, fv),
            it.width,
            FILL_EDGE_HZ,
            _others_offsets(it.tf, oth, tl, pad),
            max(1, int(FILL_FRAME_S * fv)),
        )
        gate, _ = _coherence(X[k], level, ch, _vk_enbw(r2s[k], fv, p), fv)
        gate = _pad_continuity(gate, tl, it.t[0], it.t[-1])
        span = (tl >= it.t[0]) & (tl <= it.t[-1])
        it.stats["coherent_frac"] = float(gate[span].mean()) if span.any() else 0.0
        rem += X[k] * (acts[k] * fades[k] * gate * np.exp(1j * thetas[k]))[:, None]
    d = resample_poly(rem, down, up, axis=0, window=RS_WINDOW)
    L = b - a
    if d.shape[0] < L:
        d = np.concatenate([d, np.zeros((L - d.shape[0], d.shape[1]), complex)], axis=0)
    y[a:b] = seg - 2.0 * np.real(d[:L] * np.conj(shift)[:, None])


# ----------------------------------------------------------------------------- do no harm


def _safe_half_width(width_hz: float, extra_hz: float) -> float:
    return max(width_hz + FILL_EDGE_HZ + extra_hz, 3.5)


def _do_no_harm(x0, y, sr, item: _Item, cfg, confirmed: bool, half: float, others=()):
    """(a) Final invariant, per track and per block (DNH_BLOCKS_S scales): inside the
    track's corridor, blend y back towards the input x0 (largest alpha <= 1 of
    the change) so that the corridor energy never rises more than DNH_LIMIT_DB
    above the input's. For tracks the detector did not confirm, changes are also
    reverted wherever the INPUT corridor holds no coherent tone (c). Crossfaded
    (eroded, then smoothed alpha). Returns (td, modified-mask) for the TF stats."""
    t, f = item.tf
    n = y.shape[0]
    a, b = _span(t, cfg.pad_s + 1.0, sr, n)
    if b - a < 8 * sr / ENV_RATE:
        return None
    tq = (a + np.arange(b - a)) / sr
    phi = 2 * np.pi * np.cumsum(_freq_curve(t, f, tq)) / sr
    carrier = np.exp(-1j * phi)
    zx, D, st = _boxcar3_decimate(x0[a:b].astype(np.float64) * carrier[:, None], sr)
    zy, _, _ = _boxcar3_decimate(y[a:b].astype(np.float64) * carrier[:, None], sr)
    fs_d = sr / D
    bx, bd = _BB(zx, fs_d, boxcar=True), _BB(zy - zx, fs_d, boxcar=True)
    Hs = bx.corridor(half, SAFE_EDGE_HZ)
    cx, cd = bx.filt(Hs), bd.filt(Hs)
    alpha = np.ones(cx.shape[0])
    Ex_scales = []
    td = tq[np.minimum(st + D * np.arange(cx.shape[0]), len(tq) - 1)]
    fillE = None
    if item.fill is not None:  # expected fill energy in this corridor
        fillE = np.interp(td, item.fill[0], item.fill[1], left=0, right=0) * bx.noise_bw(Hs)
    if fillE is not None:
        # a fill may not paint a corridor that was systematically quieter than its
        # flanks (a natural valley between partials): cap it at the INPUT corridor's
        # own long-scale level + DNH_VALLEY_DB (well above noise fluctuation)
        # ... on two scales: ~2 s (+3 dB) and ~0.25 s (+6 dB; e.g. a steep track
        # crossing a harmonic comb, whose flank median is inflated by the comb)
        pcx = (np.abs(cx) ** 2).sum(1)
        for scale_s, vdb in ((FILL_LONG_S, DNH_VALLEY_DB), (FILL_FRAME_S, DNH_VALLEY_SHORT_DB)):
            fillE = np.minimum(fillE, _gsmooth(pcx, max(1, int(scale_s * fs_d))) * 10 ** (vdb / 10))
    for blk, margin in zip(DNH_BLOCKS_S, DNH_FILL_MARGIN_DB):
        nb = max(1, int(blk * fs_d))
        Ex = _gsmooth((np.abs(cx) ** 2).sum(1), nb)
        C = _gsmooth(np.real(np.conj(cx) * cd).sum(1), nb)
        Ed = _gsmooth((np.abs(cd) ** 2).sum(1), nb)
        Lm = Ex * 10 ** (DNH_LIMIT_DB / 10)
        if fillE is not None:
            Lm = np.maximum(Lm, np.maximum(Ex, _gsmooth(fillE, nb)) * 10 ** (margin / 10))
        bad = Ex + 2 * C + Ed > Lm
        if bad.any():
            disc = np.sqrt(np.maximum(C**2 + Ed * (Lm - Ex), 0))
            a_s = np.ones(len(Ex))
            a_s[bad] = np.clip(((-C + disc) / np.maximum(Ed, 1e-30))[bad], 0, 1)
            alpha = np.minimum(alpha, a_s)
        Ex_scales.append((Ex, Ed))
    Ex, Ed = Ex_scales[-1]
    nb = max(1, int(DNH_BLOCKS_S[-1] * fs_d))
    if not confirmed:
        _, level = _flank_density(
            bx,
            half,
            SAFE_EDGE_HZ,
            _others_offsets((t, f), list(others), td, cfg.pad_s),
            max(1, int(FILL_FRAME_S * fs_d)),
        )
        ns = int(COH_SMOOTH_S * fs_d)
        ex_long = _gsmooth((np.abs(cx) ** 2).sum(1), ns)
        ratio = ex_long / np.maximum(_gsmooth(level, ns) * bx.noise_bw(Hs), 1e-30)
        alpha = np.minimum(
            alpha, _ramp_gate(ratio, COH_CORR, int(0.5 * fs_d), 1, int(COH_MIN_S * fs_d))
        )
    alpha = minimum_filter1d(alpha, nb, mode="nearest")  # erode: conservative
    alpha = np.clip(_gsmooth(alpha, nb // 2), 0, 1)  # crossfade
    item.stats["reverted_frac"] = float(np.mean(1 - alpha))
    if np.all(alpha > 0.9999):
        out_alpha = alpha
    else:
        corr = (1 - alpha)[:, None] * cd
        full = _upsample_env(corr, D, st, b - a)
        y[a:b] = y[a:b].astype(np.float64) - 2.0 * np.real(full * np.conj(carrier)[:, None])
        out_alpha = alpha
    td = tq[np.minimum(st + D * np.arange(len(Ex)), len(tq) - 1)]
    mod = (out_alpha**2 * Ed) > 1e-3 * np.maximum(Ex, 1e-30)  # change above -30 dB
    item.stats["modified_s"] = float(mod.sum() / fs_d)
    return td, mod, half


def _tf_fraction(records, items, n, sr):
    """(d) Fraction of the time-frequency plane modified (union of modified
    corridor cells on a 0.25 s x 0.5 Hz grid), over 0..sr/2 and 40..1000 Hz."""
    dur = n / sr
    if not records:
        return 0.0, 0.0
    fmax = max(float(np.max(it.f)) for it in items) + 20.0
    tg = np.arange(0, dur, 0.25)
    fg = np.arange(0, min(fmax, sr / 2), 0.5)
    grid = np.zeros((len(tg), len(fg)), bool)
    for rec, it in zip(records, items):
        if rec is None:
            continue
        td, mod, half = rec
        if not mod.any():
            continue
        tm = np.interp(tg, td, mod.astype(float), left=0, right=0) > 0.5
        rows = np.flatnonzero(tm)
        if not len(rows):
            continue
        fc = _freq_curve(it.t, it.f, tg[rows])
        c0 = max(0, int((fc.min() - half) / 0.5) - 1)
        c1 = min(len(fg), int((fc.max() + half) / 0.5) + 2)
        grid[rows, c0:c1] |= np.abs(fg[None, c0:c1] - fc[:, None]) <= half
    cell = 0.25 * 0.5
    area = grid.sum() * cell
    band = (fg >= 40) & (fg <= 1000)
    area_band = grid[:, band].sum() * cell
    return area / (dur * sr / 2), area_band / (dur * 960.0)


def _abs_flank_level(seg, sr, a, t, f, td, lo0):
    """Per-Hz background level next to f(t) in ABSOLUTE frequency: STFT of the
    (low-band) segment, 1 s Hann, 0.25 s hop; median of the bins lo0..lo0+6 Hz
    either side of f(t) at each frame, times FILL_MED_CAL. Returns it on the
    times td, or None if the segment is too short."""
    n, ch = seg.shape
    nper = int(sr * 1.0)
    hop = max(1, int(sr * 0.25))
    if n < nper + hop:
        return None
    win = np.hanning(nper + 2)[1:-1]
    starts = np.arange(0, n - nper + 1, hop)
    fr = np.fft.rfftfreq(nper, 1 / sr)
    tc = (a + starts + nper / 2) / sr
    fc = _freq_curve(t, f, tc)
    lev = np.empty(len(starts))
    norm = sr * np.sum(win**2) / 2.0  # one-sided rfft power -> two-sided per-Hz density
    for k, s0 in enumerate(starts):
        X = np.fft.rfft(seg[s0 : s0 + nper] * win[:, None], axis=0)
        d = np.abs(fr - fc[k])
        sel = (d >= lo0) & (d <= lo0 + FILL_MIN_FLANK_HZ)
        if not sel.any():
            lev[k] = np.inf
            continue
        lev[k] = np.median((np.abs(X[sel]) ** 2).sum(1)) / norm * FILL_MED_CAL / 2.0
    return np.interp(td, tc, lev)


def harden_tracks(
    x: np.ndarray,
    sr: int,
    tracks: list[Track],
    cfg: RemoveConfig = RemoveConfig(),
    seed: int = 0x5C2B,
    x_ref: np.ndarray | None = None,
):
    """Only the harden step (fill confirmed tracks' corridors with matched noise),
    with the same do-no-harm invariant. Same shape/dtype in and out.

    x_ref: the ORIGINAL recording when x has already been through removal passes
    (as in the pipeline). The do-no-harm check then compares against what was
    really there; against the already-cleaned x, a removed tone's corridor looks
    like a valley and the fill would be capped to it (leaving a notch)."""
    return _run(x, sr, tracks, cfg, steps=("harden",), seed=seed, x_ref=x_ref)[0]


# ----------------------------------------------------------------------------- public


def _run(x, sr, tracks, cfg, steps, seed=0x5C2B, x_ref=None):
    _CURVES.clear()
    x = np.asarray(x)
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32
    empty = {"tf_fraction": 0.0, "tf_fraction_40_1000": 0.0, "tracks": [], "notes": []}
    if x.size == 0 or not tracks:
        return x.astype(dtype, copy=True), empty
    mono = x.ndim == 1
    # memory: x0 is a read-only view of the input; y is ONE working copy in the
    # input's float dtype. Every step converts only its own segment to float64.
    x0 = x[:, None] if mono else x
    items = _select(tracks, cfg, sr)
    if not items:
        return x.astype(dtype, copy=True), empty
    # low-band processing: every tone is far below sr/2, so all steps run on a
    # polyphase-decimated copy of the low band (e.g. 8 kHz instead of 48 kHz);
    # only the DIFFERENCE is upsampled and added back, so untouched samples
    # (everything outside the corridors' time spans) stay bit-exact.
    fmax = max(float(np.max(it.f)) for it in items)
    if any(it.wide for it in items):  # the skirt's background ring must be in-band
        fmax += SKIRT_BG_HZ[1] + 5.0 - 60.0
    fw = _lowband_rate(sr, fmax) if LOWBAND else sr
    ref = None
    if x_ref is not None:
        ref = np.asarray(x_ref)
        ref = ref[:, None] if ref.ndim == 1 else ref
        if ref.shape != x0.shape:
            raise ValueError("x_ref must have the same shape as x")
    if fw < sr:
        rr = Fraction(fw, sr).limit_denominator(1000)
        up, down = rr.numerator, rr.denominator

        def lowband(v):
            out = np.empty((-(-v.shape[0] * up // down), v.shape[1]))
            for o0, blk in _resample_blocks(v, up, down, sr):
                out[o0 : o0 + len(blk)] = blk
            return out

        xw = lowband(x0)
        yw = xw.copy()
        refw = lowband(ref) if ref is not None else xw
        for it in items:
            it.f = np.clip(it.f, 1.0, fw / 2 - 1.0)
    else:
        xw, yw = x0, x0.astype(dtype, copy=True)
        refw = ref if ref is not None else xw
    x_full, sr_full, x0, sr, y = x0, sr, xw, fw, yw
    min_score = getattr(cfg, "harden_min_score", 0.5)
    extra_mask = {}
    for step in steps:
        if step == "subtract":
            for i, it in enumerate(items):
                others = _near(items, i, cfg.pad_s)
                _subtract_one(y, sr, it, cfg.pad_s, refine=REFINE, others=others, width_hz=it.width)
        elif step == "mask":
            # the STFT notch widens with the sweep inside its ~1 s window; for
            # steep tracks it would be a meaningless wide hole -> those use VK
            slow, steep = [], []
            for it in items:
                tt = np.arange(it.t[0], it.t[-1] + 1e-9, 0.25)
                sl = (
                    np.abs(np.gradient(_freq_curve(it.t, it.f, tt), 0.25)) if len(tt) > 2 else [0.0]
                )
                s95 = float(np.percentile(sl, 95))
                if s95 <= MASK_MAX_SLOPE and it.width <= cfg.width_hz:
                    slow.append(it)
                    extra_mask[id(it)] = 1.0 + 0.5 * s95
                else:
                    steep.append(it)
            for g in _groups([it.tf for it in slow], cfg.pad_s, 3.0):
                _mask_segment(y, sr, g, cfg)
            for comp in _vk_components(steep, cfg.pad_s):
                _vk_group(
                    y, sr, [steep[i] for i in comp], cfg, refine=REFINE, p=VK_ORDER, all_items=items
                )
        elif step == "vk":
            for comp in _vk_components(items, cfg.pad_s):
                _vk_group(
                    y, sr, [items[i] for i in comp], cfg, refine=REFINE, p=VK_ORDER, all_items=items
                )
        elif step == "harden":
            rng = np.random.default_rng(seed)
            # measured-wide tracks: scale the skirt down to the far background FIRST,
            # so the core fill below takes its level from flanks that are background
            for i, it in enumerate(items):
                if it.wide and it.score >= min_score:
                    _skirt_one(y, sr, it, cfg, _near(items, i, cfg.pad_s, df_hz=SKIRT_BG_HZ[1]))
            for i, it in enumerate(items):
                if it.score < min_score:
                    it.stats["filled"] = False  # not confirmed by the detector
                    continue
                others = _near(items, i, cfg.pad_s)
                _harden_one(y, sr, it, cfg, rng, extra_mask.get(id(it), 0.0), others)
        else:
            raise ValueError(f"unknown RemoveConfig.method {step!r}")
    records = []
    for i, it in enumerate(items):
        others = _near(items, i, cfg.pad_s)
        half = _safe_half_width(it.width, extra_mask.get(id(it), 0.0))
        rec = _do_no_harm(refw, y, sr, it, cfg, it.score >= min_score, half, others)
        if rec is not None and it.skirt is not None:  # TF stats: count the skirt zone too
            rec = (rec[0], rec[1], max(rec[2], float(np.nanmax(it.skirt[1]))))
        records.append(rec)
    frac, frac_band = _tf_fraction(records, items, x0.shape[0], sr)
    if sr < sr_full:  # back to the full rate: add the difference
        d = y - x0
        nz = np.flatnonzero(np.any(d != 0, axis=1))
        y = x_full.astype(dtype, copy=True)
        if len(nz):
            M = 64
            a_l = max(0, int(nz[0]) - M) // up * up  # start on an integer full-rate sample
            b_l = min(d.shape[0], int(nz[-1]) + 1 + M)
            a = int(round(a_l * sr_full / sr))
            for o0, dup in _resample_blocks(d[a_l:b_l], down, up, sr):  # back up, blockwise
                L = min(dup.shape[0], y.shape[0] - a - o0)
                if L <= 0:
                    break
                blk = y[a + o0 : a + o0 + L]
                touched = np.any(dup[:L] != 0, axis=1)
                if touched.any():
                    blk[touched] = (blk[touched].astype(np.float64) + dup[:L][touched]).astype(
                        dtype
                    )
        sr = sr_full
    stats = {
        "tf_fraction": frac,
        "tf_fraction_40_1000": frac_band,
        "notes": [],
        "tracks": [
            dict(
                index=it.index,
                t0=float(it.t[0]),
                t1=float(it.t[-1]),
                f_mean=float(np.mean(it.f)),
                score=it.score,
                **it.stats,
            )
            for it in sorted(items, key=lambda it: it.index)
        ],
    }
    if frac_band > MAX_TF_FRACTION:
        msg = (
            f"removal modified {100 * frac_band:.1f}% of the 40-1000 Hz time-frequency area "
            f"(> {100 * MAX_TF_FRACTION:.0f}%): many tracks, check the detector output"
        )
        stats["notes"].append(msg)
        warnings.warn(msg, stacklevel=3)
    return (y[:, 0] if mono else y), stats


LOWBAND = True  # process on a decimated low band (see _run)


RS_BLOCK_S = 30.0  # chunk length for full-file resampling (bounded memory)


def _resample_blocks(v: np.ndarray, up: int, down: int, sr_in: float):
    """Polyphase resample of v (n, ch) in overlapping blocks, yielding
    (out_start, out_block) float64 pieces that tile the output exactly as
    resample_poly(v, up, down, axis=0, window=RS_WINDOW) would (an FIR: the
    overlap exceeds its half-length, and block starts keep the polyphase phase).
    Memory stays O(block) instead of a full-length float64 copy."""
    n = v.shape[0]
    n_out = -(-n * up // down)
    B = max(down, int(RS_BLOCK_S * sr_in) // down * down)  # input block, multiple of down
    half = 10 * max(up, down)  # resample_poly's half-length (upsampled domain)
    P = (-(-half // up) // down + 2) * down  # input overlap, multiple of down
    for s0 in range(0, n, B):
        a, b = max(0, s0 - P), min(n, s0 + B + P)
        piece = resample_poly(v[a:b].astype(np.float64), up, down, axis=0, window=RS_WINDOW)
        o0, o1 = s0 * up // down, min(n_out, (s0 + B) * up // down)
        off = o0 - a * up // down
        yield o0, piece[off : off + (o1 - o0)]


def _lowband_rate(sr: int, fmax: float) -> int:
    """Smallest convenient rate whose kaiser-flat passband (0.65 x Nyquist)
    covers the highest track + flanks + margin."""
    for fw in (2000, 4000, 8000, 12000, 16000, 24000, 32000):
        if fw >= sr:
            return sr
        if 0.65 * fw / 2 >= fmax + 60.0:
            return fw
    return sr


MASK_MAX_SLOPE = 6.0  # Hz/s: steeper tracks are handled by VK when method="mask"
MAX_TF_FRACTION = 0.10  # warn when more than 10% of the 40-1000 Hz TF area is modified


def remove_tracks(
    x: np.ndarray,
    sr: int,
    tracks: list[Track],
    cfg: RemoveConfig = RemoveConfig(),
    return_stats: bool = False,
):
    """Remove the tracked tones from x (shape (n,) or (n, ch)) with cfg.method
    ("vk" | "subtract" | "mask"), then, if cfg.harden, fill each confirmed
    track's corridor with matched noise. A final do-no-harm pass guarantees that
    no corridor ends up with more energy than the input had (wrong tracks can't
    add curves). Returns an array of the same shape and dtype, or
    ``(array, stats)`` with ``return_stats=True``: stats has ``tf_fraction``
    (modified share of the whole TF plane), ``tf_fraction_40_1000`` and
    per-track ``coherent_frac`` / ``filled`` / ``reverted_frac`` / ``modified_s``."""
    method = cfg.method.lower()
    if method not in ("subtract", "vk", "mask"):
        raise ValueError(f"unknown RemoveConfig.method {cfg.method!r}")
    steps = (method, "harden") if cfg.harden else (method,)
    y, stats = _run(x, sr, tracks, cfg, steps)
    return (y, stats) if return_stats else y
