"""Fast-chirp tracker for the steep cores of close, fast passes.

The main detectors analyse 0.25 s frames with 0.25--2 s windows. A close pass
can sweep 50--200 Hz/s at its core, which smears across every one of those
windows (and across the attacker's 2 s view, where it is still plainly
visible as a bright diagonal). This module follows such tones on a 1/16 s hop
with chirp-matched short windows (the window that best resolves a sweep of
rate a is about 1/sqrt(a) long), taking the per-cell maximum of several
whitened, frequency-contrast views.

Acceptance is deliberately strict (long, smooth, steep, high integrated
contrast against frequency-shifted controls): the band above 150 Hz also holds
speech, birds and music, whose pitch tracks are short, wobbly or flat.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage, signal

from .types import DetectConfig, Track

HOP = 1 / 16
WINDOWS = (.0625, .125, .25, .5)
CHUNK_S, OVERLAP_S = 90., 10.


def _views(y, rate, fmin, fmax, windows=WINDOWS, hop=HOP, df=2.):
    """Frequency-local contrast (dB) for each window; returns t, f, [C_w]."""
    nfft = int(round(rate / df))
    step = int(round(hop * rate))
    freq = np.fft.rfftfreq(nfft, 1 / rate)
    band = (freq >= fmin) & (freq <= fmax)
    f = freq[band]
    count = max(0, (len(y) - 1) // step + 1)
    t = np.arange(count) * step / rate
    out = []
    for w in windows:
        n = int(round(w * rate))
        pad = np.pad(y, (n // 2, n))
        win = signal.windows.hann(n, sym=False)
        spec = np.empty((band.sum(), count), np.float32)
        for c0 in range(0, count, 512):
            c1 = min(count, c0 + 512)
            idx = np.arange(c0, c1)[:, None] * step + np.arange(n)[None, :]
            spec[:, c0:c1] = np.abs(np.fft.rfft(pad[idx] * win, n=max(nfft, n), axis=1))[:, :len(freq)][:, band].T
        db = 20 * np.log10(spec + 1e-12)
        db -= np.median(db, axis=1, keepdims=True)
        # local contrast: a tone stands above its +-(several main lobes) neighbourhood
        width = int(max(31, 12 / w / df)) | 1
        local = ndimage.median_filter(db, size=(width, 1), mode='nearest')
        out.append((db - local).astype(np.float32))
    return t, f, out


SLOPES = np.r_[0., 3., 6., 9., 13., 18., 24., 31., 40., 50., 62., 76., 92., 110., 130., 155., 185., 220.]
HALF = 8                                   # oriented integration over +-0.5 s


def _oriented(C, df):
    """Mean of C along short straight segments; best slope per cell (Hz/s)."""
    F, T = C.shape
    best = np.full((F, T), -np.inf, np.float32); arg = np.zeros((F, T), np.float32)
    slopes = np.r_[-SLOPES[:0:-1], SLOPES]
    pad = HALF * int(np.ceil(SLOPES[-1] * HOP / df)) + 1
    P = np.pad(np.clip(C, -10, 25), ((pad, pad), (HALF, HALF)), constant_values=0.)
    for s in slopes:
        acc = np.zeros((F, T), np.float32)
        for k in range(-HALF, HALF + 1):
            d = int(round(s * k * HOP / df))
            acc += P[pad + d:pad + d + F, HALF + k:HALF + k + T]
        acc /= 2 * HALF + 1
        better = acc > best
        best[better] = acc[better]; arg[better] = s
    return best, arg


def _track(A, S, t, f, thr, max_gap=2):
    """Greedy multi-target peak tracker on the oriented image, slope-predicted."""
    df = f[1] - f[0]
    active, done = [], []
    for j in range(A.shape[1]):
        col = A[:, j]
        peaks, _ = signal.find_peaks(col, height=thr, distance=max(2, int(6 / df)))
        if len(peaks) > 60:
            peaks = peaks[np.argsort(col[peaks])[-60:]]
        a, b, c = col[np.clip(peaks - 1, 0, len(f) - 1)], col[peaks], col[np.clip(peaks + 1, 0, len(f) - 1)]
        hz = f[peaks] + df * np.clip(.5 * (a - c) / (a - 2 * b + c - 1e-9), -.5, .5)
        offers = []
        for i, p in enumerate(active):
            gap = j - p['j'][-1]
            slope = float(np.median(p['s'][-4:]))
            pred = p['f'][-1] + slope * gap * HOP
            tol = 3. + .03 * abs(slope) + 2. * (gap - 1)
            for k in np.flatnonzero(abs(hz - pred) < tol):
                # the cell's own best slope must agree with the path's
                if abs(S[peaks[k], j] - slope) > 25. + .35 * abs(slope):
                    continue
                offers.append((abs(hz[k] - pred) / tol - .01 * min(len(p['j']), 40), i, k))
        used_p, used_k = set(), set()
        for _, i, k in sorted(offers):
            if i in used_p or k in used_k:
                continue
            q = active[i]
            q['j'].append(j); q['f'].append(float(hz[k])); q['c'].append(float(b[k])); q['s'].append(float(S[peaks[k], j]))
            used_p.add(i); used_k.add(k)
        keep = []
        for p in active:
            (keep if j - p['j'][-1] <= max_gap else done).append(p)
        active = keep + [dict(j=[j], f=[float(hz[k])], c=[float(b[k])], s=[float(S[peaks[k], j])])
                         for k in range(len(peaks)) if k not in used_k]
    return done + active


def _sample(C, f, t, tt, ff):
    fi = np.clip((ff - f[0]) / (f[1] - f[0]), 0, len(f) - 1)
    ti = np.clip(np.rint((tt - t[0]) / HOP).astype(int), 0, C.shape[1] - 1)
    lo = np.floor(fi).astype(int); hi = np.minimum(lo + 1, len(f) - 1); w = fi - lo
    return (1 - w) * C[lo, ti] + w * C[hi, ti]


def _accept(p, C, t, f, min_dur, min_rate):
    j = np.asarray(p['j']); ff = np.asarray(p['f'])
    if len(j) < 16 or (j[-1] - j[0]) * HOP < min_dur:
        return None
    if len(j) / (j[-1] - j[0] + 1) < .75:
        return None
    tt = t[j]
    grid = t[j[0]:j[-1] + 1]
    fg = np.interp(grid, tt, ff)
    win = min(len(fg) // 2 * 2 - 1, 9)
    smooth = signal.savgol_filter(fg, win, 2)
    jitter = float(np.sqrt(np.mean((fg - smooth) ** 2)))
    rate = np.gradient(smooth, HOP)
    steep = abs(rate) >= min_rate
    # the steep core itself must last: a pass core is one sustained sweep
    runs = np.split(np.flatnonzero(steep), np.flatnonzero(np.diff(np.flatnonzero(steep)) > 2) + 1)
    if max((len(r) for r in runs), default=0) * HOP < 2. * min_dur / 3:
        return None
    if np.ptp(smooth) < max(25., .03 * np.median(smooth)):
        return None
    if jitter > 2.5 + .01 * np.percentile(abs(rate), 90):
        return None
    # few turns: pass cores fall once (a maneuver may add one turn on a long
    # track); oscillating pitch (voice, vibrato) reverses every second or two
    coarse = ndimage.gaussian_filter1d(smooth, 4)[::4]
    prom = max(6., .015 * np.median(smooth))
    turns = len(signal.find_peaks(coarse, prominence=prom)[0]) + len(signal.find_peaks(-coarse, prominence=prom)[0])
    if turns > (1 if len(grid) * HOP >= 6 else 0):
        return None
    on = _sample(C, f, t, grid, smooth)
    if np.median(on) < 6. or np.mean(on > 3.) < .75:
        return None
    shifts = (-48., -40., -32., -24., 24., 32., 40., 48.)
    controls = np.array([np.mean(_sample(C, f, t, grid, smooth + d)) for d in shifts])
    # adjacent 1/16 s frames share most of their window: ~4 frames per independent sample
    sd = max(float(np.std(controls, ddof=1)), 4. / np.sqrt(max(len(grid) / 4, 1)))
    excess = float(np.mean(on) - np.median(controls))
    z = excess / sd
    if z < 5. or excess < 5.:
        return None
    return grid, smooth, float(np.median(on)), float(z)


def _spread90(P, ff, radii):
    """Per-frame radius holding 90 % of the excess power (flank floor removed)."""
    floor = np.median(P[(np.abs(ff) >= 20.) & (np.abs(ff) <= 30.)], axis=0)
    excess = np.array([np.maximum(P[np.abs(ff) <= r] - floor, 0).sum(0) for r in radii])
    return radii[np.argmax(excess >= .9 * np.maximum(excess[-1], 1e-30), axis=0)]


def _intrinsic_halfwidth(z, rate, lock, win=.5):
    """Tone half-width (Hz) after dechirping: the 90 %-energy radius in 0.5 s
    baseband windows, with the window's own pure-tone spread removed in
    quadrature. Floors at the remover default (2 Hz)."""
    n = int(win * rate); step = max(1, int(HOP * rate))
    ff, tt, Z = signal.stft(z, rate, window='hann', nperseg=n, noverlap=n - step, nfft=max(n, 4096),
                            return_onesided=False, boundary=None, padded=False)
    radii = np.arange(.5, 15.25, .25)
    r = _spread90(np.abs(Z) ** 2, ff, radii)
    tone = np.exp(2j * np.pi * .0 * np.arange(4 * n) / rate)
    _, _, Zt = signal.stft(tone, rate, window='hann', nperseg=n, noverlap=n - step, nfft=max(n, 4096),
                           return_onesided=False, boundary=None, padded=False)
    r0 = float(np.median(_spread90(np.abs(Zt) ** 2 + 1e-30, ff, radii)))
    good = np.interp(np.arange(len(r)), np.linspace(0, len(r) - 1, len(lock)), lock) > .2
    if not good.any():
        return float('nan')
    return float(max(2., np.sqrt(max(np.median(r[good]) ** 2 - r0 ** 2, 0.))))


def refine_dechirp(y, rate, t, f, iterations=3, spans=(25., 10., 5.), win=.25):
    """Iteratively re-centre f(t) on the tone: heterodyne along the current
    curve, read the residual frequency offset in short baseband windows,
    smooth it in time and fold it back. A sweep of 150 Hz/s smears across
    any fixed-window spectrogram, but is a near-DC line after dechirping, so
    0.25 s baseband windows measure it to a fraction of a Hz."""
    from scipy.interpolate import CubicSpline
    t = np.asarray(t, float); f = np.asarray(f, float).copy()
    a = max(0, int((t[0] - .5) * rate)); b = min(len(y), int((t[-1] + .5) * rate))
    if b - a < int(2 * win * rate) or len(t) < 4:
        return f, np.zeros(len(t)), float('nan')
    ts = np.arange(a, b) / rate
    n = int(win * rate); step = max(1, int(HOP * rate))
    for it in range(iterations):
        fi = CubicSpline(t, f)(np.clip(ts, t[0], t[-1]))
        fi[ts < t[0]] = f[0]; fi[ts > t[-1]] = f[-1]
        z = y[a:b] * np.exp(-2j * np.pi * np.cumsum(fi) / rate)
        nfft = int(8 * rate / 1.)  # 0.125 Hz sampling of the baseband spectrum
        nfft = 1 << int(np.ceil(np.log2(max(nfft // 8, n))))
        ff, tt, Z = signal.stft(z, rate, window='hann', nperseg=n, noverlap=n - step, nfft=nfft,
                                return_onesided=False, boundary=None, padded=False)
        ff = np.fft.fftshift(ff); P = np.abs(np.fft.fftshift(Z, axes=0)) ** 2
        tt = tt + ts[0]
        band = np.abs(ff) <= spans[min(it, len(spans) - 1)]
        Pb = P[band]; fb = ff[band]
        j = np.argmax(Pb, axis=0)
        jj = np.clip(j, 1, len(fb) - 2); cols = np.arange(Pb.shape[1])
        la, lb, lc = (np.log(Pb[jj - 1, cols] + 1e-30), np.log(Pb[jj, cols] + 1e-30), np.log(Pb[jj + 1, cols] + 1e-30))
        off = fb[jj] + (fb[1] - fb[0]) * np.clip(.5 * (la - lc) / (la - 2 * lb + lc - 1e-12), -.5, .5)
        contrast = Pb[j, cols] / np.maximum(np.median(Pb, axis=0), 1e-30)
        w = (contrast > 6.).astype(float)
        sig = max(1., .25 / (tt[1] - tt[0]) if len(tt) > 1 else 1.)
        num = ndimage.gaussian_filter1d(off * w, sig); den = ndimage.gaussian_filter1d(w, sig)
        corr = np.where(den > .3, num / np.maximum(den, 1e-9), 0.)
        f = f + np.interp(t, tt, corr)
    # final per-sample quality: fraction of local baseband power within +-3 Hz
    fi = CubicSpline(t, f)(np.clip(ts, t[0], t[-1]))
    z = y[a:b] * np.exp(-2j * np.pi * np.cumsum(fi) / rate)
    ff, tt, Z = signal.stft(z, rate, window='hann', nperseg=n, noverlap=n - step, nfft=max(n, 2048),
                            return_onesided=False, boundary=None, padded=False)
    P = np.abs(Z) ** 2; near = np.abs(ff) <= 3.; wide = np.abs(ff) <= 30.
    q = P[near].sum(0) / np.maximum(P[wide].sum(0), 1e-30)
    half = _intrinsic_halfwidth(z, rate, q)
    return f, np.interp(t, tt + ts[0], q), half


def detect_chirps(x, sr, cfg: DetectConfig = DetectConfig(), fmin=150., min_dur=3., min_rate=12., thr=5.):
    """Steep, smooth single-tone sweeps as 'doppler' Tracks on a 0.25 s grid."""
    x = np.asarray(x, np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    fmax = float(cfg.fmax)
    rate = int(math.ceil(2.4 * fmax / 100) * 100)
    g = math.gcd(int(sr), rate)
    y = signal.resample_poly(x, rate // g, int(sr) // g)
    dur = len(y) / rate
    out = []
    starts = np.arange(0., max(dur - OVERLAP_S, 1e-9), CHUNK_S - OVERLAP_S)
    for s0 in starts:
        a, b = int(s0 * rate), int(min(dur, s0 + CHUNK_S) * rate)
        if b - a < int(min_dur * 2 * rate):
            continue
        t, f, views = _views(y[a:b], rate, max(fmin, cfg.fmin), fmax)
        t = t + s0
        C = np.max(views, axis=0)
        A, S = _oriented(C, f[1] - f[0])
        for p in _track(A, S, t, f, thr):
            r = _accept(p, C, t, f, min_dur, min_rate)
            if r is None:
                continue
            grid, ff, contrast, z = r
            # 1/16 s samples: a 0.25 s grid cannot carry a 150 Hz/s curve to
            # the remover's +-3 Hz envelope band, even when every sample is exact
            ff, quality, half = refine_dechirp(y, rate, grid, ff)
            tr = Track(grid.copy(), ff, np.zeros(len(grid)), .8, None, 1, -1, 'doppler')
            if np.isfinite(half):
                tr.width_hz = float(np.clip(half, 1., 12.))
            tr._chirp = dict(contrast=contrast, z=z, locked=float(np.median(quality)))
            out.append(tr)
    # chunk overlaps: drop duplicates
    out.sort(key=lambda q: -(q.t1 - q.t0))
    kept = []
    for q in out:
        dup = False
        for k in kept:
            lo, hi = max(q.t0, k.t0), min(q.t1, k.t1)
            if hi - lo >= .5 * (q.t1 - q.t0):
                g2 = np.arange(lo, hi + 1e-9, .25)
                if np.median(abs(np.interp(g2, q.t, q.f) - np.interp(g2, k.t, k.f))) < 6:
                    dup = True; break
        if not dup:
            kept.append(q)
    return kept
