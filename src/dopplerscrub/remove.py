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

Tracks whose ``kind`` is not in ``cfg.remove_kinds`` are ignored.
``Track.amp`` is not used; the envelope is estimated from the signal.
"""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.linalg import solveh_banded
from scipy.signal import butter, resample_poly, sosfiltfilt, stft, istft

from .types import RemoveConfig, Track

ENV_RATE = 100.0          # Hz, decimated rate of the complex envelope
RS_WINDOW = ("kaiser", 10.0)  # polyphase resampling window (deep image rejection)
EXTRAP_TAU = 1.0          # s, decay constant of the end-slope extrapolation
REFINE = True             # local frequency refinement before the final demod (subtract, vk)
REFINE_SNR = (1.0, 10.0)  # tone/background power ratio: refinement trust ramps 0 -> 1


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


def _freq_curve(t, f, tq):
    """Instantaneous frequency at times tq: cubic spline inside [t0, t1],
    decaying-slope extrapolation outside."""
    t0, t1 = t[0], t[-1]
    if len(t) >= 4:
        cs = CubicSpline(t, f, bc_type="not-a-knot")
        inside = np.clip(tq, t0, t1)
        out = cs(inside)
    elif len(t) >= 2:
        out = np.interp(tq, t, f)
    else:
        out = np.full_like(tq, f[0], dtype=float)
    s0, s1 = _end_slope(t, f, False), _end_slope(t, f, True)
    lo, hi = tq < t0, tq > t1
    if lo.any():
        d = t0 - tq[lo]
        out[lo] = f[0] - s0 * EXTRAP_TAU * (1 - np.exp(-d / EXTRAP_TAU))
    if hi.any():
        d = tq[hi] - t1
        out[hi] = f[-1] + s1 * EXTRAP_TAU * (1 - np.exp(-d / EXTRAP_TAU))
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
    return float(np.clip(1.0 + 0.25 * np.percentile(slope, 95) + 0.5 * np.percentile(curv, 95),
                         1.0, 3.0))


def _lowpass_env(z: np.ndarray, sr: int, bw: float):
    """Narrow zero-phase low-pass of complex baseband z (n, ch).

    Returns (env_dec, D, start): envelope sampled at indices start + k*D.
    """
    n = z.shape[0]
    D = max(1, int(round(sr / ENV_RATE)))
    zr = np.empty((n, z.shape[1], 2), np.float64)
    zr[..., 0], zr[..., 1] = z.real, z.imag
    for _ in range(3):
        zr = uniform_filter1d(zr, D, axis=0, mode="reflect")
    start = D // 2
    zd = zr[start::D, :, 0] + 1j * zr[start::D, :, 1]
    fs_d = sr / D
    if zd.shape[0] > 15:
        sos = butter(4, min(bw, 0.45 * fs_d) / (fs_d / 2), output="sos")
        padlen = min(zd.shape[0] - 1, int(3 * fs_d / max(bw, 0.1)))
        zd = sosfiltfilt(sos, zd.real, axis=0, padlen=padlen) + \
            1j * sosfiltfilt(sos, zd.imag, axis=0, padlen=padlen)
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
    dens = np.maximum(p_wide - p_tone, 1e-30) / fs_d          # per-Hz background
    snr = p_tone / (dens * 4 * bw) - 1.0
    lo, hi = REFINE_SNR
    conf = np.clip((snr - lo) / (hi - lo), 0.0, 1.0)
    if weight is not None:
        conf = conf * weight[np.minimum(st + D * np.arange(k), n - 1)]
    comb = zd.sum(axis=1)
    dph = np.angle(comb[1:] * np.conj(comb[:-1]))               # wrapped phase increments
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
        out[:, c] = np.interp(np.arange(n), idx, zd[:, c].real) + \
            1j * np.interp(np.arange(n), idx, zd[:, c].imag)
    return out


def _span(t, pad, sr, n):
    a = max(0, int(np.floor((t[0] - pad) * sr)))
    b = min(n, int(np.ceil((t[-1] + pad) * sr)) + 1)
    return a, b


def _select(tracks, cfg, sr):
    out = []
    for tr in tracks:
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
        out.append((strength, t, np.clip(f, 1.0, sr / 2 - 1.0)))
    out.sort(key=lambda r: -r[0])
    return [(t, f) for _, t, f in out]


# ----------------------------------------------------------------------------- subtract

def _subtract_one(y: np.ndarray, sr: int, t, f, pad: float, refine: bool = True, others=()):
    """Subtract one tracked tone from y (n, ch) in place."""
    n = y.shape[0]
    a, b = _span(t, pad, sr, n)
    if b - a < 8:
        return
    tq = (a + np.arange(b - a)) / sr
    fi = _freq_curve(t, f, tq)
    phi = 2 * np.pi * np.cumsum(fi) / sr
    phi -= phi[0]
    w = _fade(tq, t[0], t[-1], pad, n / sr)
    seg = y[a:b].astype(np.float64)
    bw = _env_bandwidth(t, f)

    if refine:
        z = seg * np.exp(-1j * phi)[:, None]
        corr = _phase_correction(z, sr, bw, _proximity_weight(t, f, list(others), tq, bw, pad))
        if corr is not None:
            phi = phi + corr
    z = seg * np.exp(-1j * phi)[:, None]
    zd, D, st = _lowpass_env(z, sr, bw)
    env = _upsample_env(zd, D, st, b - a)
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
    return (lambda v: resample_poly(v, up, down, axis=0, window=RS_WINDOW),
            lambda v: resample_poly(v, down, up, axis=0, window=RS_WINDOW))


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
    nper = int(2 ** np.round(np.log2(fw * 1.0)))      # ~1 s window, ~1 Hz bins
    hop = nper // 4
    margin = (2 * nper / fw) + 0.5
    a = max(0, int((min(t[0] for t, _ in items) - pad - margin) * sr))
    b = min(n, int(np.ceil((max(t[-1] for t, _ in items) + pad + margin) * sr)))
    if b - a < 16:
        return
    seg = y[a:b].astype(np.float64)
    low = down_fn(seg)
    m = low.shape[0]
    fr, tt, S = stft(low, fs=fw, window="hann", nperseg=nper, noverlap=nper - hop,
                     axis=0, boundary="even", padded=True)      # (nf, ch, nt)
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
        C = np.maximum(C, taper * w[None, :])
    if not C.any():
        return
    _, d_low = istft(C[:, None, :] * S, fs=fw, window="hann", nperseg=nper,
                     noverlap=nper - hop, time_axis=-1, freq_axis=0, boundary=True)
    d_low = np.moveaxis(d_low, -1, 0)[:m]
    d = up_fn(d_low)
    L = b - a
    if d.shape[0] < L:
        d = np.concatenate([d, np.zeros((L - d.shape[0], d.shape[1]))], axis=0)
    y[a:b] = seg - d[:L]


# ----------------------------------------------------------------------------- harden (fill)

FILL_EDGE_HZ = 0.75       # raised-cosine transition of the corridor filter
FILL_FLANK_HZ = 8.0       # flanking bands reach out to +-8 Hz from f(t)
FILL_SUB_HZ = 1.0         # flank sub-band width (median across sub-bands)
FILL_FRAME_S = 0.25       # level/envelope tracking time constant (attacker hop)
CROSS_DB = 8.0            # corridor this far above the flank level = crossing partial
CROSS_MAX_S = 1.5         # ... but only for brief excursions (residual tones persist)


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


def _harden_one(y, sr, t, f, cfg, rng, extra_hz: float = 0.0):
    """Replace the +-width_hz corridor around one track with track-locked noise
    matched to the local background (in place).

    Works in the track's baseband (signal heterodyned by its own phase, i.e. the
    emission-time-warped frame where the curve is a constant line at DC):
      * corridor filter H: 1 inside +-W, raised-cosine to 0 over FILL_EDGE_HZ.
        Applied as an FFT mask on the ~100 Hz baseband, so it is confined:
        nothing outside +-(W + edge) of f(t) is touched (apart from the tiny
        sinc^3/spline interpolation error).
      * power-complementary: keep sqrt(1 - H^2) of the original, add H * noise,
        so there is no dip at the edges (no band-edge discontinuity).
      * level, frame by frame (FILL_FRAME_S): complex covariance (ch x ch) of
        flanking 1 Hz sub-bands from W+edge to +-8 Hz on both sides; sub-bands
        above 4x the median (other partials) are excluded. The noise gets that
        PSD and inter-channel correlation, so the mono downmix matches too.
      * envelope matching (Chuang/Garg/Wu): the noise's own envelope is
        normalised away and replaced by the flanks' envelope, so the fill has
        the background's temporal modulation and kurtosis.
      * crossing partials: where the corridor (after removal of the tone) is
        CROSS_DB above the flank level for less than CROSS_MAX_S (a speech or
        music harmonic passing through), the original is kept instead of
        filled, so foreground isn't replaced by noise. Long excursions (a
        residual tone) are always filled.
    """
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
    P = min(m - 1, int(4 * fs_d))
    zp = np.concatenate([zd[P:0:-1], zd, zd[-2:-P - 2:-1]], axis=0) if P > 0 else zd
    L = zp.shape[0]
    F = np.fft.fftfreq(L, 1 / fs_d)
    aF = np.abs(F)
    W = cfg.width_hz + extra_hz
    E = FILL_EDGE_HZ
    H = np.where(aF <= W, 1.0, np.where(aF < W + E, 0.5 + 0.5 * np.cos(np.pi * (aF - W) / E), 0.0))
    droop = np.maximum(np.sinc(F / fs_d) ** 3, 1e-3)          # boxcar^3 response
    Z = np.fft.fft(zp, axis=0)
    keep = np.sqrt(np.clip(1 - H ** 2, 0, 1))
    removed = np.fft.ifft(((1 - keep) / droop)[:, None] * Z, axis=0)[P:P + m]
    corridor = np.fft.ifft((H / droop)[:, None] * Z, axis=0)[P:P + m]

    # flanking sub-band covariances
    nf = max(1, int(round(FILL_FRAME_S * fs_d)))
    lo0 = W + E + 0.25
    edges = np.arange(lo0, max(FILL_FLANK_HZ, lo0 + 2 * FILL_SUB_HZ) - 1e-9, FILL_SUB_HZ)
    covs, bws = [], []
    for lo in edges:
        hi = lo + FILL_SUB_HZ
        for sgn in (1, -1):
            band = (sgn * F >= lo) & (sgn * F < hi)
            if not band.any():
                continue
            sb = np.fft.ifft((band / droop)[:, None] * Z, axis=0)[P:P + m]
            c = sb[:, :, None] * np.conj(sb[:, None, :])
            c = _gsmooth(c.real, nf) + 1j * _gsmooth(c.imag, nf)
            covs.append(c)
            bws.append(band.sum() * fs_d / L)
    if not covs:
        return
    covs = np.stack(covs) / np.asarray(bws)[:, None, None, None]   # per-Hz, (K, m, ch, ch)
    # soft outlier rejection of sub-bands holding another (persistent) partial.
    # Weights come from a ~2 s smoothed power so ordinary noise fluctuations are
    # not rejected (that would bias the level low), and are smooth so the fill
    # level never jumps (a jump would spread the fill spectrally).
    tr = _gsmooth(np.real(np.einsum("kmii->mk", covs)), 8 * nf).T     # (K, m)
    med = np.median(tr, axis=0)
    wk = 1.0 / (1.0 + (tr / (4 * med[None, :] + 1e-30)) ** 4)
    dens = (covs * wk[:, :, None, None]).sum(0) / np.maximum(wk.sum(0), 1e-12)[:, None, None]

    # crossing-partial detection: corridor power vs flank-predicted power
    Hbw = float(np.sum(H ** 2) * fs_d / L)                 # corridor noise bandwidth (Hz)
    pc = _gsmooth((np.abs(corridor) ** 2).sum(1), nf)
    pf = np.real(np.einsum("mii->m", dens)) * Hbw
    hot = pc > pf * 10 ** (CROSS_DB / 10)
    cross = _short_runs(hot, int(CROSS_MAX_S * fs_d))
    cross = np.clip(_gsmooth(cross.astype(float), 2 * nf) * 1.5, 0, 1)   # smooth edges
    gate = 1.0 - cross                                     # 0 = keep original

    # noise: white -> H -> envelope-normalised -> coloured by chol(dens)
    g = (rng.standard_normal((L, ch)) + 1j * rng.standard_normal((L, ch))) / np.sqrt(2)
    g = np.fft.ifft(H[:, None] * np.fft.fft(g, axis=0), axis=0)[P:P + m]
    gpow = _gsmooth((np.abs(g) ** 2).mean(1), nf)
    g *= np.sqrt(Hbw / np.maximum(gpow, 1e-30))[:, None]   # unit density, flat frame envelope
    tr_d = np.real(np.einsum("mii->m", dens))
    Lc = np.linalg.cholesky(dens + (tr_d * 1e-9 + 1e-30)[:, None, None] * np.eye(ch))
    noise = np.einsum("mij,mj->mi", Lc, g)               # PSD + channel correlation of flanks

    R = _upsample_env(removed * gate[:, None], D, st, b - a)
    N = _upsample_env(noise * gate[:, None], D, st, b - a)
    # power-complementary time fade over the pads: original x (1 - c), noise x w
    c = 1 - np.sqrt(np.clip(1 - w ** 2, 0, 1))
    out = seg - 2.0 * np.real((c[:, None] * R - w[:, None] * N) * np.conj(carrier)[:, None])
    y[a:b] = out


def harden_tracks(x: np.ndarray, sr: int, tracks: list[Track], cfg: RemoveConfig = RemoveConfig(),
                  seed: int = 0x5C2B):
    """Only the harden step: replace each track's corridor by matched noise
    (see _harden_one). Same shape/dtype in and out."""
    return _run(x, sr, tracks, cfg, steps=("harden",), seed=seed)


# ----------------------------------------------------------------------------- Vold-Kalman

VK_ORDER = 2              # p: VK structural-equation order (2-3 recommended)
VK_CHUNK_S = 120.0        # long groups are solved in overlapping chunks
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
            diags[j - i][i:i + rows] += c[i] * c[j]
    return diags


def _diff_coeffs(p):
    from math import comb
    return np.array([(-1) ** (p - i) * comb(p, i) for i in range(p + 1)], float)


def _vk_solve(low: np.ndarray, thetas, acts, r2s, p: int):
    """Joint (multi-order, 2nd-generation) VK: minimise
        sum_n |y_n - sum_k a_kn x_kn e^{j th_kn}|^2 + sum_k r_k^2 |D^p x_k|^2
    for complex envelopes x_k. Unknowns interleaved (n*K + k) so the normal
    matrix is Hermitian banded with half-bandwidth p*K; solved with LAPACK pbsv.
    low: (m, ch) real; returns x (K, m, ch)."""
    m, ch = low.shape
    K = len(thetas)
    N = m * K
    u = p * K
    ab = np.zeros((u + 1, N), complex)                 # upper form: ab[u + i - j, j]
    E = np.stack([a * np.exp(1j * th) for a, th in zip(acts, thetas)])   # (K, m)
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


def _vk_group(y, sr, items, cfg, refine=True, p=VK_ORDER):
    n = y.shape[0]
    pad = cfg.pad_s
    fmax = max(float(np.max(f)) for _, f in items) + 10.0
    fw = _work_rate(sr, fmax)
    down_fn, up_fn = _resampler(sr, fw)
    margin = 1.0
    a = max(0, int((min(t[0] for t, _ in items) - pad - margin) * sr))
    b = min(n, int(np.ceil((max(t[-1] for t, _ in items) + pad + margin) * sr)))
    if b - a < 16:
        return
    seg = y[a:b].astype(np.float64)
    low = down_fn(seg)
    m = low.shape[0]
    tl = a / sr + np.arange(m) / fw
    thetas, acts, fades, r2s = [], [], [], []
    for i, (t, f) in enumerate(items):
        th = 2 * np.pi * np.cumsum(_freq_curve(t, f, tl)) / fw
        bw = _env_bandwidth(t, f)
        if refine:
            wgt = _proximity_weight(t, f, items[:i] + items[i + 1:], tl, bw, pad)
            corr = _phase_correction(low * np.exp(-1j * th)[:, None], fw, bw, wgt)
            if corr is not None:
                th = th + corr
        thetas.append(th)
        acts.append(((tl >= t[0] - pad) & (tl <= t[-1] + pad)).astype(float))
        fades.append(_fade(tl, t[0], t[-1], pad, n / sr))
        r2s.append(_vk_r2(bw, fw, p))
    d_low = np.zeros_like(low)
    chunk, ov = int(VK_CHUNK_S * fw), int(VK_OVERLAP_S * fw)
    s0 = 0
    while s0 < m:
        e0 = min(m, s0 + chunk)
        lo, hi = max(0, s0 - ov), min(m, e0 + ov)
        sl = slice(lo, hi)
        x = _vk_solve(low[sl], [th[sl] for th in thetas], [ac[sl] for ac in acts], r2s, p)
        for k in range(len(items)):
            part = 2 * np.real(x[k] * (acts[k][sl] * np.exp(1j * thetas[k][sl]))[:, None])
            part *= fades[k][sl][:, None]
            d_low[s0:e0] += part[s0 - lo:e0 - lo]
        s0 = e0
    d = up_fn(d_low)
    L = b - a
    if d.shape[0] < L:
        d = np.concatenate([d, np.zeros((L - d.shape[0], d.shape[1]))], axis=0)
    y[a:b] = seg - d[:L]


# ----------------------------------------------------------------------------- public

def _run(x, sr, tracks, cfg, steps, seed=0x5C2B):
    x = np.asarray(x)
    if x.size == 0 or not tracks:
        return x.copy()
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32
    mono = x.ndim == 1
    y = (x[:, None] if mono else x).astype(np.float64, copy=True)
    items = _select(tracks, cfg, sr)
    if not items:
        return x.astype(dtype, copy=True)
    for step in steps:
        if step == "subtract":
            for i, (t, f) in enumerate(items):
                others = items[:i] + items[i + 1:]
                _subtract_one(y, sr, t, f, cfg.pad_s, refine=REFINE, others=others)
        elif step == "mask":
            for g in _groups(items, cfg.pad_s, 3.0):
                _mask_segment(y, sr, g, cfg)
        elif step == "vk":
            for g in _groups(items, cfg.pad_s, 1.0):
                _vk_group(y, sr, g, cfg, refine=REFINE, p=VK_ORDER)
        elif step == "harden":
            rng = np.random.default_rng(seed)
            for t, f in items:
                extra = 0.0
                if "mask" in steps:
                    # the STFT notch is wider (sweep within its ~1 s window + 1 bin
                    # taper): widen the fill corridor to cover all of it
                    tt = np.arange(t[0], t[-1] + 1e-9, 0.25)
                    sl = np.abs(np.gradient(_freq_curve(t, f, tt), 0.25)) if len(tt) > 2 else [0.0]
                    extra = 1.0 + 0.5 * float(np.percentile(sl, 95))
                _harden_one(y, sr, t, f, cfg, rng, extra)
        else:
            raise ValueError(f"unknown RemoveConfig.method {step!r}")
    out = y[:, 0] if mono else y
    res = x.astype(dtype, copy=True)
    changed = out.astype(dtype)
    diff = np.any((changed != res).reshape(res.shape[0], -1), axis=1)
    res[diff] = changed[diff]
    return res


def remove_tracks(x: np.ndarray, sr: int, tracks: list[Track], cfg: RemoveConfig = RemoveConfig()):
    """Remove the tracked tones from x (shape (n,) or (n, ch)) with cfg.method
    ("subtract" | "vk" | "mask"), then, if cfg.harden, fill each corridor with
    matched noise. Returns an array of the same shape and dtype."""
    method = cfg.method.lower()
    if method not in ("subtract", "vk", "mask"):
        raise ValueError(f"unknown RemoveConfig.method {cfg.method!r}")
    steps = (method, "harden") if cfg.harden else (method,)
    return _run(x, sr, tracks, cfg, steps)
