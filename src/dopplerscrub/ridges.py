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
        self.amplitude = getattr(obj, "amplitude", None)
        self.calibration = getattr(obj, "calibration", [])

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
        raise ValueError("finite mono or samples-by-channels audio and positive sr required")
    if not 0 < cfg.fmin < cfg.fmax or cfg.fmin >= sr / 2:
        raise ValueError("require 0 < fmin < fmax and fmin < input Nyquist")
    fmax = min(cfg.fmax, np.nextafter(float(sr) / 2, 0.0))
    y = np.mean(x, axis=1) if x.ndim == 2 else x
    rate = max(int(cfg.work_sr), int(math.ceil(fmax * 2.4)))
    g = math.gcd(int(sr), rate)
    if len(y):
        y = signal.resample_poly(y, rate // g, int(sr) // g)
    nlong = max(int(cfg.nperseg), 2 * rate)
    df = rate / nlong
    f = np.arange(math.ceil(cfg.fmin / df), math.floor(fmax / df) + 1) * df
    t = np.arange(0, len(y) / rate, 0.25)
    if not len(t):
        return RidgeSpectrum(np.empty((len(f), 0)), np.empty((len(f), 0))), t, f
    amp = np.zeros((len(f), len(t)), np.float32)
    long_amp = np.zeros_like(amp)
    for lo, hi, n in [
        (0, np.inf, nlong),
        (0, 700, round(0.5 * rate)),
        (700, np.inf, round(0.25 * rate)),
    ]:
        band = (f > lo) & (f <= hi)
        if not band.any() and n != nlong:
            continue
        window = signal.windows.hann(n, sym=False)
        # Dense FFT sampling avoids artificial 2–4 Hz frequency steps in
        # short-window peaks. It interpolates the spectrum, not its magnitude.
        ff = np.fft.rfftfreq(nlong, 1 / rate)
        padded = np.pad(y, (n // 2, n))
        for j, time in enumerate(t):
            start = round(time * rate)
            # FFT is unnormalized: 2|Z| / sum(window) gives sinusoid peak.
            a = 2 * np.abs(np.fft.rfft(padded[start : start + n] * window, n=nlong)) / window.sum()
            amp[band, j] = np.interp(f[band], ff, a)
            if n == nlong:
                long_amp[:, j] = np.interp(f, ff, a)

    def whiten(magnitude):
        db = 20 * np.log10(magnitude + 1e-12)
        width = 2400
        centers = []
        medians = []
        for start in range(0, len(t), width):
            stop = min(start + width, len(t))
            centers.append((start + stop - 1) / 2)
            medians.append(np.median(db[:, start:stop], axis=1))
        medians = np.array(medians)
        baseline = np.empty_like(db)
        for k in range(len(f)):
            baseline[k] = np.interp(np.arange(len(t)), centers, medians[:, k])
        return db - baseline

    adaptive = whiten(amp)
    long_view = whiten(long_amp)
    magnitude = np.maximum(long_amp, amp)
    # Keep faint higher harmonics visible to the attacker, alongside short-
    # window peaks needed at maximum chirp rate. Reference nulls include this
    # same max operation; its distribution is not assumed to be pure Exp(1).
    return RidgeSpectrum(np.maximum(adaptive, long_view), magnitude), t, f


def _frequency_median(values, bins):
    """Exact column medians using SciPy's fast 1-D rank-filter backend."""
    columns = np.ascontiguousarray(np.asarray(values).T)
    out = np.empty_like(columns)
    for j in range(len(columns)):
        ndimage.median_filter(columns[j], size=bins, mode="nearest", output=out[j])
    return out.T


def _evidence(S, f, amplitude, base=None):
    """Remove broadband foreground while preserving stationary raw ridges."""
    if base is None:
        result = np.empty_like(S, dtype=np.float32)
        for lo, hi, width in [(0, 450, 18.0), (450, 700, 80.0), (700, np.inf, 160.0)]:
            indices = np.flatnonzero((f > lo) & (f <= hi))
            if not len(indices):
                continue
            bins = max(5, int(round(width / np.median(np.diff(f)))) | 1)
            left = max(0, int(indices[0]) - bins // 2)
            right = min(len(f), int(indices[-1]) + bins // 2 + 1)
            local = _frequency_median(S[left:right], bins)
            result[indices] = S[indices] - local[indices - left]
    else:
        result = np.asarray(base, dtype=np.float32).copy()
    if amplitude is not None:
        raw = 20 * np.log10(amplitude + 1e-12)
        tonal = raw - _frequency_median(raw, 49)
        # Persistent raw lines vanished under time whitening. Reintroduce only
        # bins with a strong time-median spectral peak, e.g. hum or chords.
        persistent = np.median(tonal, axis=1) > 12
        result[persistent] = np.maximum(result[persistent], tonal[persistent])
    if amplitude is not None:
        # Ignore floating-point/codec residue >100 dB below the loudest peak.
        result[amplitude < max(float(np.max(amplitude)) * 1e-5, 1e-12)] = -20
    return np.clip(np.nan_to_num(result, nan=-20, posinf=30, neginf=-20), -20, 30)


def _llr(db, snr=3.0):
    # Median-whitened periodogram has median 1; ln(2) restores Exp(1) mean.
    z = np.log(2) * 10 ** (np.clip(db, -300, 300) / 10)
    return z * snr / (1 + snr) - np.log1p(snr)


def _robust_llr(db, snr=3.0):
    # Winsorize individual observations so one clipped/saturated harmonic
    # cannot impersonate an entire coherent stack. Nulls use the same cap.
    return np.minimum(_llr(db, snr), 12.0)


def _transition_cost(delta, center, dt, df):
    # A slow Gaussian core provides loose continuation of faint lines. The
    # physics-width Gaussian tail permits fast flybys without raising the
    # continuation cost of an almost stationary distant aircraft.
    sigma = np.maximum(df, center * 120.0**2 / (343.0 * 500.0) * dt)
    slow = np.maximum(df, 0.005 * center * dt)
    a = np.log(0.9) - 0.5 * (delta / slow) ** 2 - np.log(np.sqrt(2 * np.pi) * slow / df)
    b = np.log(0.1) - 0.5 * (delta / sigma) ** 2 - np.log(np.sqrt(2 * np.pi) * sigma / df)
    cost = -np.logaddexp(a, b)
    return np.where(np.abs(delta) <= 3 * sigma, cost, 1e6)


def _viterbi_lines(
    evidence,
    t,
    f,
    threshold,
    limit=70,
    observation=None,
    harmonic_suppression=False,
    score_only=False,
    min_score=24.0,
    amplitude=None,
):
    """Sparse-bin Viterbi with zero-score births, deaths, Gaussian transitions.

    Local maxima are the frequency states (at most 96/frame), rather than
    spending quadratic work on bins with no ridge evidence. Paths compete
    through iterative corridor suppression and are split at unsupported gaps.
    """
    F, T = evidence.shape
    if T < 3 or F < 3:
        return 0.0 if score_only else []
    dt = float(np.median(np.diff(t)))
    candidates = []
    for j in range(T):
        peaks = signal.find_peaks(evidence[:, j], height=max(1.0, threshold * 0.6), distance=2)[0]
        if len(peaks) > 96:
            peaks = peaks[np.argsort(evidence[peaks, j])[-96:]]
        candidates.append(peaks)
    N = max(map(len, candidates), default=0)
    if N == 0:
        return 0.0 if score_only else []
    idx = np.zeros((T, N), int)
    valid = np.zeros((T, N), bool)
    for j, p in enumerate(candidates):
        idx[j, : len(p)] = p
        valid[j, : len(p)] = True
    hz = f[idx]
    llr = _robust_llr(evidence) if observation is None else observation
    emission = llr[idx, np.arange(T)[:, None]].astype(float)
    emission[~valid] = -1e6
    # Sparse edges: the hard physics bound usually admits only 2–15 of
    # 96 predecessor peaks. Do not evaluate a dense N*N*beam tensor only to
    # discard almost all of it afterwards.
    edges = []
    df = float(np.median(np.diff(f)))
    for j in range(1, T):
        delta = hz[j, :, None] - hz[j - 1, None, :]
        center = (hz[j, :, None] + hz[j - 1, None, :]) / 2
        sigma = np.maximum(df, center * 120.0**2 / (343.0 * 500.0) * dt)
        allowed = np.abs(delta) <= 3 * sigma
        if amplitude is not None:
            midpoint = np.clip(np.rint((center - f[0]) / df).astype(int), 0, F - 1)
            middle = np.maximum(amplitude[midpoint, j - 1], amplitude[midpoint, j])
            endpoint = np.maximum(
                amplitude[idx[j], j][:, None], amplitude[idx[j - 1], j - 1][None, :]
            )
            resolution = np.where(center >= 700.0, 4.0, np.where(center >= 450.0, 2.0, df))
            # A true chirp leaves energy between its endpoint frequencies.
            # Sparse peak states otherwise alias across a stationary comb.
            allowed &= (abs(delta) <= 2 * resolution) | (middle >= 0.10 * endpoint)
        counts = allowed.sum(axis=1)
        width = max(1, int(counts.max()))
        neighbors = np.zeros((N, width), int)
        present = np.zeros((N, width), bool)
        row_ids, columns = np.nonzero(allowed)
        starts = np.cumsum(counts) - counts
        positions = np.arange(len(columns)) - np.repeat(starts, counts)
        neighbors[row_ids, positions] = columns
        present[row_ids, positions] = True
        row = np.arange(N)[:, None]
        dd = delta[row, neighbors]
        cc = center[row, neighbors]
        ss = sigma[row, neighbors]
        acceleration = np.maximum(df, 0.25 * cc * 120.0**3 / (343.0 * 500.0**2) * dt**2)
        core_norm = np.log(0.95) - np.log(np.sqrt(2 * np.pi) * acceleration / df)
        reset = np.log(0.05) - 0.5 * (dd / ss) ** 2 - np.log(np.sqrt(2 * np.pi) * ss / df)
        edges.append((neighbors, present, dd, acceleration, core_norm, reset))
    paths = []
    beams = min(2, N)
    rows = np.arange(N)[:, None]
    for _ in range(limit):
        # Frequency plus two competing slope hypotheses. Negative active
        # scores remain alive while accumulating evidence against null=0;
        # they pay a birth cost once, rather than being reset every frame.
        scores = np.full((T, N, beams), -1e6)
        velocity = np.zeros((N, beams))
        back = np.full((T, N, beams), -1, np.int16)
        birth = math.log(max(F, 3))
        scores[0, :, 0] = emission[0] - birth
        for j in range(1, T):
            neighbors, present, delta, acceleration, core_norm, reset = edges[j - 1]
            width = neighbors.shape[1]
            innovation = delta[:, :, None] - velocity[neighbors]
            core = core_norm[:, :, None] - 0.5 * (innovation / acceleration[:, :, None]) ** 2
            value = scores[j - 1][neighbors] + np.logaddexp(core, reset[:, :, None])
            value = np.where(present[:, :, None], value, -1e6)
            best_beam = value.argmax(axis=2)
            best = np.take_along_axis(value, best_beam[:, :, None], axis=2)[:, :, 0]
            options = np.column_stack((best, np.full(N, -birth)))
            previous = np.argpartition(-options, beams - 1, axis=1)[:, :beams]
            selected = np.take_along_axis(options, previous, axis=1)
            scores[j] = selected + emission[j, :, None]
            active = previous < width
            clipped = np.minimum(previous, width - 1)
            old_state = neighbors[rows, clipped]
            old_beam = best_beam[rows, clipped]
            back[j] = np.where(active, old_state * beams + old_beam, -1)
            velocity = np.where(active, delta[rows, clipped], 0.0)
        end, state, beam = np.unravel_index(np.argmax(scores), scores.shape)
        maximum = max(0.0, float(scores[end, state, beam]))
        if score_only:
            return maximum
        if maximum < min_score:
            break
        points = []
        j = int(end)
        k = int(state)
        h = int(beam)
        while j >= 0:
            points.append((j, int(idx[j, k])))
            previous = int(back[j, k, h])
            j -= 1
            if previous < 0:
                break
            k, h = divmod(previous, beams)
        points.reverse()
        if not points:
            break
        times = np.array([p[0] for p in points])
        bins = np.array([p[1] for p in points])
        # Suppress roughly one resolution element, wider in the short-window bands.
        for frame, bin_ in points:
            corridor = max(1.0, 2.0 if f[bin_] <= 450 else 5.0 if f[bin_] <= 700 else 9.0)
            emission[frame, np.abs(hz[frame] - f[bin_]) <= corridor] = -1e6
            if harmonic_suppression:
                for ratio in (0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.5, 2.0, 3.0, 4.0):
                    emission[
                        frame,
                        np.abs(hz[frame] - ratio * f[bin_])
                        <= max(corridor, 0.008 * ratio * f[bin_]),
                    ] = -1e6
        support = evidence[bins, times] >= threshold
        jump = np.abs(np.diff(f[bins]))
        allowed = np.maximum(3.0, 0.06 * (f[bins[1:]] + f[bins[:-1]]) / 2)
        cuts = np.flatnonzero((np.diff(times) > 1) | (jump > allowed)) + 1
        for segment in np.split(np.arange(len(times)), cuts):
            good = segment[support[segment]]
            if len(good) < 3:
                continue
            a, b = good[0], good[-1] + 1
            if t[times[b - 1]] - t[times[a]] >= 1.75:
                paths.append((times[a:b], bins[a:b]))
    return paths


def _refine_path(bins, frames, evidence, f):
    lo = np.maximum(bins - 1, 0)
    hi = np.minimum(bins + 1, len(f) - 1)
    a = evidence[lo, frames]
    b = evidence[bins, frames]
    c = evidence[hi, frames]
    den = a - 2 * b + c
    shift = np.divide(0.5 * (a - c), den, out=np.zeros_like(b), where=np.abs(den) > 1e-6)
    return f[bins] + np.clip(shift, -0.5, 0.5) * np.median(np.diff(f))


def _moving(freq, cfg):
    motion = ndimage.gaussian_filter1d(np.asarray(freq, dtype=float), 2.0)
    if np.quantile(motion, 0.95) - np.quantile(motion, 0.05) < max(
        cfg.min_sweep_hz, 1.5, 0.01 * np.median(freq)
    ):
        return False
    steps = np.abs(np.diff(freq))
    if (
        len(steps)
        and np.mean(steps < max(0.15, 0.0007 * np.median(freq))) > 0.55
        and np.max(steps) > max(2.0, 0.02 * np.median(freq))
    ):
        return False  # held musical notes joined by a discrete pitch change
    if len(freq) >= 16:
        if (np.quantile(freq, 0.95) - np.quantile(freq, 0.05)) / np.median(freq) < 0.03:
            window = min(17, len(freq) if len(freq) % 2 else len(freq) - 1)
            slope = signal.savgol_filter(freq, window, 2, deriv=1, delta=0.25)
            magnitude = np.abs(slope)
            curved = np.quantile(magnitude, 0.9) > 1.8 * max(np.median(magnitude), 0.02)
            turns = np.quantile(slope, 0.1) < -0.04 and np.quantile(slope, 0.9) > 0.04
            if not curved and not turns:
                return False
        # Measure note-hold concentration at a physical time scale. Half-Hz
        # quantization makes a genuine slow root look like a staircase before
        # its higher harmonics are combined; do not count bin plateaus as holds.
        smooth = ndimage.gaussian_filter1d(ndimage.median_filter(freq, 3, mode="nearest"), 1.5)
        local_span = ndimage.maximum_filter1d(smooth, 3) - ndimage.minimum_filter1d(smooth, 3)
        flat = np.mean(local_span < max(0.3, 0.0015 * np.median(freq)))
        variation = np.abs(np.diff(smooth))
        top = max(1, int(np.ceil(0.2 * len(variation))))
        concentration = np.sort(variation)[-top:].sum() / max(variation.sum(), 1e-9)
        # A sequence of sustained notes connected through their decay tails
        # is not a continuously moving ridge. Keep slow straight drifts:
        # those distribute their motion throughout the linked track.
        if flat > 0.4 and concentration > 0.65:
            active = variation > max(0.25, 0.3 * np.mean(variation))
            # Separate genuine multi-second chirps from a one-frame note
            # change. Close only isolated 0.25-second estimation dropouts.
            active = ndimage.binary_closing(active, structure=np.ones(3, bool))
            indices = np.flatnonzero(active)
            longest = max(
                (
                    len(part) * 0.25
                    for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
                ),
                default=0.0,
            )
            if longest < cfg.min_duration_s:
                return False
    return True


def _speech_like(tt, ff):
    # A sustained voiced vowel can be a moving harmonic stack too. Reject rapid
    # repeated pitch oscillations, not one/two turns in an aircraft maneuver.
    smooth = ndimage.gaussian_filter1d(ff, 1.2)
    prominence = max(0.8, 0.008 * np.median(ff))
    peaks = signal.find_peaks(smooth, prominence=prominence)[0]
    valleys = signal.find_peaks(-smooth, prominence=prominence)[0]
    return len(peaks) + len(valleys) >= 3 and (len(peaks) + len(valleys)) / (tt[-1] - tt[0]) > 0.38


def _compatible(a, b, max_order=16):
    lo = max(a.t0, b.t0)
    hi = min(a.t1, b.t1)
    if hi - lo < max(4.0, 0.6 * min(a.t1 - a.t0, b.t1 - b.t0)):
        return False
    tt = np.arange(lo, hi + 0.01, 0.25)
    aa = np.interp(tt, a.t, a.f)
    bb = np.interp(tt, b.t, b.f)
    ratio = float(np.median(aa / bb))
    rational = Fraction(ratio).limit_denominator(12)
    if rational.numerator > max_order or abs(float(rational) - ratio) > max(
        0.004 * ratio, 0.6 / np.median(bb)
    ):
        return False
    error = aa - float(rational) * bb
    if np.quantile(np.abs(error), 0.8) > max(1.2, 0.006 * np.median(aa)):
        return False
    if np.ptp(aa) > 2 and np.ptp(bb) > 2:
        da = np.diff(ndimage.gaussian_filter1d(aa, 1.5))
        db = np.diff(ndimage.gaussian_filter1d(bb, 1.5))
        return np.corrcoef(aa, bb)[0, 1] > 0.88 and np.corrcoef(da, db)[0, 1] > 0.6
    return True


def _track_chunk(S, t, f, cfg, base_evidence=None):
    """Extract non-parametric lines, recover harmonic members, classify stacks.

    Return 0.25-second Track samples. Group ids are local to this call. amp is
    in input-signal peak units only when S retains RidgeSpectrum.amplitude;
    arbitrary external whitened arrays cannot determine absolute amplitude.
    cfg.hop_s is intentionally not used: this API's output cadence is 0.25 s.
    """
    amplitude = getattr(S, "amplitude", None)
    calibration_log = getattr(S, "calibration", None)
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    if (
        S.shape != (len(f), len(t))
        or len(f) < 3
        or np.any(np.diff(f) <= 0)
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("S must have shape (len(f), len(t)) with increasing grids")
    if not np.allclose(np.diff(f), np.median(np.diff(f)), rtol=0.01):
        raise ValueError("frequency grid must be uniform")
    if len(t) < 3:
        return []
    if amplitude is not None and amplitude.shape != S.shape:
        amplitude = None
    moving_evidence = (
        _evidence(S, f, None)
        if base_evidence is None
        else np.asarray(base_evidence, dtype=np.float32).copy()
    )
    evidence = _evidence(S, f, amplitude, base=moving_evidence)
    threshold = max(float(cfg.snr_db), 4.0)
    reference = _reference_calibration(tuple(np.round(f, 6)), len(t))
    paths = []
    # A single 96-state peak budget over 30–2000 Hz lets bright speech/high
    # orders displace a faint low fundamental (and vice versa). Independent
    # banks preserve seeds in every physical band without lowering null scores.
    for lower, upper in ((0, 450), (450, 900), (900, 1400), (1400, np.inf)):
        band = np.flatnonzero((f >= lower) & (f < upper))
        if len(band) < 3:
            continue
        cutoff = reference["single_cutoff"] if upper <= 450 else reference["high_cutoff"]
        found = _viterbi_lines(
            evidence[band],
            t,
            f[band],
            threshold,
            limit=35,
            min_score=cutoff,
            amplitude=None if amplitude is None else amplitude[band],
        )
        paths.extend((frames, bins + band[0]) for frames, bins in found)
    # Harmonic summation in POWER, with cap to stop one very loud order
    # masquerading as an entire stack. Variance-normalize the harmonic bank.
    # The physical fundamental can be BELOW cfg.fmin (rotors, engine orders).
    bf = np.arange(8.0, min(500.0, f[-1] / 2), np.median(np.diff(f)))
    harm = np.zeros((len(bf), len(t)))
    weights = np.zeros(len(bf))
    harmonic_amplitude = np.zeros_like(harm) if amplitude is not None else None
    if amplitude is not None:
        moving_evidence[amplitude < max(float(np.max(amplitude)) * 1e-5, 1e-12)] = -20
    observations = _robust_llr(moving_evidence)
    for h in range(1, 25):
        indices = np.rint((h * bf - f[0]) / (f[1] - f[0])).astype(int)
        good = (indices >= 0) & (indices < len(f))
        weight = 1 / np.sqrt(h)
        harm[good] += weight * observations[indices[good]]
        weights[good] += weight**2
        if harmonic_amplitude is not None:
            harmonic_amplitude[good] += amplitude[indices[good]] ** 2 / h**2
    harm /= np.sqrt(np.maximum(weights[:, None], 1))
    if harmonic_amplitude is not None:
        harmonic_amplitude = np.sqrt(harmonic_amplitude)
    reference["maxima"]
    cutoff = reference["cutoff"]
    reference["single_maxima"]
    single_cutoff = reference["single_cutoff"]
    high_cutoff = reference["high_cutoff"]
    if calibration_log is not None:
        calibration_log.append(
            dict(
                start_s=float(t[0]),
                end_s=float(t[-1]),
                reference="seeded white/pink noise and hum",
                **reference,
            )
        )
    # Harmonic observations sum LLRs at k*f0 under one shared frequency state.
    seeds = (
        _viterbi_lines(
            harm,
            t,
            bf,
            2.0,
            limit=35,
            observation=harm,
            harmonic_suppression=True,
            min_score=24.0,
            amplitude=harmonic_amplitude,
        )
        if len(bf) > 3
        else []
    )
    # Recover physical members from summed-power templates, checking each
    # member's local contrast. This fills short gaps but never invents a tone.
    for frames, bins in seeds:
        fundamental = bf[bins]
        for h in range(1, 25):
            center = h * fundamental
            if np.median(center) > f[-1]:
                break
            middle = np.rint((center - f[0]) / (f[1] - f[0])).astype(int)
            radius = max(2, int(round((1.5 if np.median(center) < 450 else 5) / (f[1] - f[0]))))
            offsets = np.arange(-radius, radius + 1)
            options = np.clip(middle[:, None] + offsets, 0, len(f) - 1)
            value = evidence[options, frames[:, None]]
            best = np.argmax(value - 0.15 * offsets[None, :] ** 2, axis=1)
            selected = options[np.arange(len(frames)), best]
            supported = evidence[selected, frames] >= max(3.0, threshold * 0.75)
            if supported.mean() < 0.65:
                continue
            good = np.flatnonzero(supported)
            if len(good) > 2:
                a, b = good[0], good[-1] + 1
                paths.append((frames[a:b], selected[a:b]))
    seed_lines = []
    for seed_id, (frames, bins) in enumerate(seeds):
        leader = np.max(harm[:, frames], axis=0)
        if np.mean(harm[bins, frames] >= np.maximum(3.0, 0.65 * leader)) < 0.55:
            continue
        root_curve = _refine_path(bins, frames, harm, bf)
        root_curve = ndimage.median_filter(root_curve, 3, mode="nearest")
        ti = t[frames]
        score = float(
            np.sum(harm[bins, frames])
            - _transition_cost(
                np.diff(root_curve), (root_curve[1:] + root_curve[:-1]) / 2, 0.25, bf[1] - bf[0]
            ).sum()
            - np.log(len(bf))
        )
        strict = score > cutoff
        if not strict and score < 24.0:
            continue
        estimates = []
        members = []
        for k in range(1, 33):
            center = k * root_curve
            if np.median(center) > f[-1]:
                break
            if np.median(center) < f[0]:
                continue
            radius = max(2, int(np.ceil(max(1.5, k * (bf[1] - bf[0])) / (f[1] - f[0]))))
            middle = np.rint((center - f[0]) / (f[1] - f[0])).astype(int)
            offsets = np.arange(-radius, radius + 1)
            options = np.clip(middle[:, None] + offsets, 0, len(f) - 1)
            values = moving_evidence[options, frames[:, None]]
            best = np.argmax(values - 0.2 * (offsets[None, :] / radius) ** 2, axis=1)
            selected = options[np.arange(len(frames)), best]
            frequency = _refine_path(selected, frames, moving_evidence, f)
            support = moving_evidence[selected, frames] >= max(2.0, cfg.snr_db * 0.7)
            if support.mean() < 0.45:
                continue
            estimate = frequency / k
            estimate[~support] = np.nan
            estimates.append(estimate)
            members.append((k, frequency, selected, support))
        if len(members) < 2:
            continue
        bank = np.stack(estimates)
        observed = np.sum(np.isfinite(bank), axis=0) > 0
        root_curve[observed] = np.nanmedian(bank[:, observed], axis=0)
        root_curve = ndimage.median_filter(root_curve, 3, mode="nearest")
        if _speech_like(ti, root_curve):
            continue
        grid = np.arange(math.ceil(ti[0] * 4) / 4, ti[-1] + 1e-6, 0.25)
        if len(grid) < 2:
            continue
        for k, frequency, selected, support in members:
            # One frequency factor fills weak intervals; observations refine
            # it only when they remain consistent with the shared stack.
            target = k * root_curve
            measured = support & (np.abs(frequency - target) <= max(1.5, 0.004 * np.median(target)))
            target[measured] = frequency[measured]
            amp = (
                np.interp(grid, ti, amplitude[selected, frames])
                if amplitude is not None
                else np.zeros(len(grid))
            )
            seed_lines.append(
                Track(
                    grid,
                    np.interp(grid, ti, target),
                    amp,
                    0.95 if strict else 0.45,
                    None,
                    k,
                    seed_id,
                    "candidate",
                )
            )
    lines = []
    strength = []
    for frames, bins in paths:
        tt = t[frames]
        ff = _refine_path(bins, frames, evidence, f)
        ff = ndimage.median_filter(ff, size=5, mode="nearest")
        if tt[-1] - tt[0] < 1.75:
            continue
        # Ignore rapidly jumping speech paths and wandering noise maxima.
        sm = ndimage.gaussian_filter1d(ff, 1.0)
        jitter = float(np.sqrt(np.mean((ff - sm) ** 2)))
        if jitter > max(1.4, 0.008 * np.median(ff)) or _speech_like(tt, ff):
            continue
        grid = np.arange(math.ceil(tt[0] * 4) / 4, tt[-1] + 1e-6, 0.25)
        if len(grid) < 2:
            continue
        frequency = np.interp(grid, tt, ff)
        amp = (
            np.interp(grid, tt, amplitude[bins, frames])
            if amplitude is not None
            else np.zeros(len(grid))
        )
        tr = Track(grid, frequency, amp, kind="unknown")
        duplicate = False
        for old in lines:
            lo = max(old.t0, tr.t0)
            hi = min(old.t1, tr.t1)
            if hi - lo < 0.7 * min(old.t1 - old.t0, tr.t1 - tr.t0):
                continue
            overlap = np.arange(lo, hi, 0.25)
            if len(overlap) and np.median(
                np.abs(np.interp(overlap, old.t, old.f) - np.interp(overlap, tr.t, tr.f))
            ) < max(1.5, 0.003 * np.median(ff)):
                duplicate = True
                break
        if not duplicate:
            lines.append(tr)
            strength.append(float(np.median(evidence[bins, frames])))
    # Anchor each family to a long line: pairwise union-find would let
    # incompatible rational ratios chain into an absurd subharmonic lattice.
    groups = {}
    unassigned = set(range(len(lines)))
    while unassigned:
        reference = max(unassigned, key=lambda i: lines[i].t1 - lines[i].t0)
        members = [reference] + [
            i for i in unassigned if i != reference and _compatible(lines[i], lines[reference])
        ]
        groups[len(groups)] = members
        unassigned.difference_update(members)
    output = []
    for group, members in enumerate(groups.values()):
        long = [i for i in members if lines[i].t1 - lines[i].t0 >= cfg.min_duration_s]
        if not members:
            continue
        moving = [i for i in long if _moving(lines[i].f, cfg)]
        # A stack requires actual shared motion, rather than a random ridge
        # near some rational multiple of a steady hum line.
        coherent = len(moving) >= 3 or (
            len(moving) == 2 and min(strength[i] for i in moving) >= 8.0
        )
        reference = max(members, key=lambda i: lines[i].t1 - lines[i].t0)
        ratios = []
        for i in members:
            lo = max(lines[i].t0, lines[reference].t0)
            hi = min(lines[i].t1, lines[reference].t1)
            tt = np.arange(lo, hi + 0.01, 0.25)
            ratio = float(
                np.median(
                    np.interp(tt, lines[i].t, lines[i].f)
                    / np.interp(tt, lines[reference].t, lines[reference].f)
                )
            )
            ratios.append(Fraction(ratio).limit_denominator(16))
        denominator = math.lcm(*(r.denominator for r in ratios))
        orders = [r.numerator * (denominator // r.denominator) for r in ratios]
        divisor = math.gcd(*orders)
        orders = [order // divisor for order in orders]
        # Reject inconsistent quantized ratios rather than predict a lattice.
        if max(orders) > 32 or np.median(lines[reference].f) / orders[members.index(reference)] < 8:
            coherent = False
            orders = [1] * len(members)
        divisor = 1
        anchor = lines[reference]
        root_freq = anchor.f / orders[members.index(reference)]
        fi = np.clip(np.rint((root_freq - bf[0]) / (bf[1] - bf[0])).astype(int), 0, len(bf) - 1)
        ti = np.clip(np.searchsorted(t, anchor.t), 0, len(t) - 1)
        joint_score = float(
            np.sum(harm[fi, ti])
            - np.sum(
                np.log(np.maximum(1.0, root_freq * 0.25 * 120**2 / (343 * 500) / (bf[1] - bf[0])))
            )
        )
        if calibration_log is not None:
            calibration_log[-1].setdefault("groups", []).append(
                dict(
                    base=float(np.median(root_freq)),
                    score=joint_score,
                    orders=list(map(int, orders)),
                    coherent=bool(coherent),
                )
            )
        calibrated = joint_score > cutoff
        coherent = coherent and calibrated
        emitted = []
        for i, order in zip(members, orders):
            tr = lines[i]
            ti_line = np.clip(np.searchsorted(t, tr.t), 0, len(t) - 1)
            fi_line = np.clip(np.rint((tr.f - f[0]) / (f[1] - f[0])).astype(int), 0, len(f) - 1)
            llr = _robust_llr(moving_evidence[fi_line, ti_line])
            transition = _transition_cost(
                np.diff(tr.f), (tr.f[1:] + tr.f[:-1]) / 2, 0.25, f[1] - f[0]
            )
            single_score = float(llr.sum() - transition.sum() - np.log(len(f)))
            strong_single = single_score > (
                high_cutoff if np.median(tr.f) >= 450 else single_cutoff
            )
            if calibrated or strong_single:
                # Classification is deferred until all fragments are linked.
                tr.kind = "candidate"
                tr.doppler_score = 0.9 if coherent else 0.65
            elif not _moving(tr.f, cfg):
                tr.kind = "stationary"
                tr.doppler_score = 0.0
            else:
                continue
            tr.group = group
            tr.harmonic = int(order // divisor)
            output.append(tr)
            emitted.append(tr)
    # HMM-stack ids and independently grouped line ids are separate sources.
    for tr in seed_lines:
        tr.group += len(groups)
    output.extend(seed_lines)
    contiguous = []
    for tr in output:
        for part in np.split(np.arange(len(tr.t)), np.flatnonzero(np.diff(tr.t) > 0.26) + 1):
            if len(part) < 2:
                continue
            contiguous.append(
                Track(
                    tr.t[part],
                    tr.f[part],
                    tr.amp[part],
                    tr.doppler_score,
                    tr.fit,
                    tr.harmonic,
                    tr.group,
                    tr.kind,
                )
            )
    return sorted(contiguous, key=lambda tr: (tr.group, tr.harmonic, tr.t0))


def track_ridges(S, t, f, cfg):
    """Track arbitrary sweeps and harmonic stacks; return 0.25 s Track grids.

    Forty-second overlapping inference windows prevent a loud early event
    from monopolizing the path budget of a long recording. Shared-factor
    groups are stitched across overlaps. Confirmed sources retain supported
    approach/recede plateaus; their harmonics are emitted only over locally
    supported spans with an integrated fixed-curve control test. Missing orders
    never create blind removal corridors.
    Preserve ridge_spectrogram's S object to retain calibrated amplitudes;
    plain whitened arrays yield zero/unknown amplitudes.
    """
    t = np.asarray(t, dtype=float)
    if hasattr(S, "calibration"):
        S.calibration.clear()
    f = np.asarray(f, dtype=float)
    base = (
        _evidence(np.asarray(S, dtype=np.float32), f, None)
        if len(t) >= 3 and len(f) >= 3 and np.shape(S) == (len(f), len(t))
        else None
    )
    amplitude = getattr(S, "amplitude", None)
    if base is not None and amplitude is not None and amplitude.shape == base.shape:
        base[amplitude < max(float(np.max(amplitude)) * 1e-5, 1e-12)] = -20
    if len(t) <= 96:
        return _finalize(_track_chunk(S, t, f, cfg, base), S, t, f, cfg, base)
    all_tracks = []
    representatives = {}
    offset = 0
    covered = t[0] - 0.25
    for start in np.arange(t[0], t[-1], 30.0):
        if covered >= t[-1] - 0.01:
            break
        indices = np.flatnonzero((t >= start) & (t < start + 40.0))
        if len(indices) < 3 or t[indices[-1]] - t[indices[0]] < cfg.min_duration_s:
            continue
        lo, hi = indices[0], indices[-1] + 1
        covered = t[hi - 1]
        tracks = _track_chunk(
            S[:, lo:hi], t[lo:hi], f, cfg, None if base is None else base[:, lo:hi]
        )
        for tr in tracks:
            tr.group += offset
            previous = representatives.get(tr.group)
            if previous is None or (
                tr.kind in ("candidate", "doppler")
                and (tr.doppler_score, -tr.harmonic, tr.t1 - tr.t0)
                > (previous.doppler_score, -previous.harmonic, previous.t1 - previous.t0)
            ):
                representatives[tr.group] = tr
        all_tracks.extend(tracks)
        offset = max(representatives, default=-1) + 1
    parents = {g: g for g in representatives}

    def root(g):
        while parents[g] != g:
            parents[g] = parents[parents[g]]
            g = parents[g]
        return g

    keys = list(representatives)
    for n, g in enumerate(keys):
        a = representatives[g]
        if a.kind not in ("candidate", "doppler"):
            continue
        for h in keys[:n]:
            b = representatives[h]
            if b.kind not in ("candidate", "doppler"):
                continue
            lo = max(a.t0, b.t0)
            hi = min(a.t1, b.t1)
            if hi - lo < 2:
                continue
            grid = np.arange(lo, hi, 0.25)
            af = np.interp(grid, a.t, a.f) / a.harmonic
            bf = np.interp(grid, b.t, b.f) / b.harmonic
            if np.quantile(np.abs(af - bf), 0.8) < max(0.6, 0.01 * np.median(af)):
                parents[root(g)] = root(h)
    grouped = {}
    for tr in all_tracks:
        tr.group = root(tr.group)
        grouped.setdefault((tr.group, tr.harmonic, tr.kind), []).append(tr)
    merged = []
    for tracks in grouped.values():
        tracks.sort(key=lambda tr: tr.t0)
        current = tracks[0]
        for nxt in tracks[1:]:
            if nxt.t0 > current.t1 + 0.26:
                merged.append(current)
                current = nxt
                continue
            grid = np.arange(current.t0, max(current.t1, nxt.t1) + 0.01, 0.25)
            freq = np.interp(grid, current.t, current.f, left=np.nan, right=np.nan)
            amp = np.interp(grid, current.t, current.amp, left=np.nan, right=np.nan)
            other = np.interp(grid, nxt.t, nxt.f, left=np.nan, right=np.nan)
            oa = np.interp(grid, nxt.t, nxt.amp, left=np.nan, right=np.nan)
            freq = np.nanmean(np.stack([freq, other]), axis=0)
            amp = np.nanmean(np.stack([amp, oa]), axis=0)
            current = Track(
                grid,
                freq,
                amp,
                max(current.doppler_score, nxt.doppler_score),
                None,
                current.harmonic,
                current.group,
                current.kind,
            )
        merged.append(current)
    return _finalize(merged, S, t, f, cfg, base)


def _join(a, b):
    grid = np.arange(min(a.t0, b.t0), max(a.t1, b.t1) + 0.01, 0.25)
    # Linear interpolation in <=1.5 s unsupported intervals is the slope-
    # bridging prediction; supported overlap is averaged in original units.
    fa = np.interp(grid, a.t, a.f, left=np.nan, right=np.nan)
    fb = np.interp(grid, b.t, b.f, left=np.nan, right=np.nan)
    aa = np.interp(grid, a.t, a.amp, left=np.nan, right=np.nan)
    ab = np.interp(grid, b.t, b.amp, left=np.nan, right=np.nan)
    both = np.isfinite(fa) & np.isfinite(fb)
    close = both & (abs(fa - fb) <= 1.0)
    prefer_a = np.nan_to_num(aa) > np.nan_to_num(ab) * 1.05
    tie = np.isclose(np.nan_to_num(aa), np.nan_to_num(ab), rtol=0.05)
    prefer_a |= tie & (a.doppler_score >= b.doppler_score)
    freq = np.where(np.isfinite(fa), fa, fb)
    amp = np.where(np.isfinite(aa), aa, ab)
    freq[both] = np.where(prefer_a[both], fa[both], fb[both])
    amp[both] = np.where(prefer_a[both], aa[both], ab[both])
    # Average only genuinely coincident measurements; divergent observations
    # must never create a non-existent midpoint between two spectral lines.
    freq[close] = 0.5 * (fa[close] + fb[close])
    amp[close] = 0.5 * (aa[close] + ab[close])
    ok = np.isfinite(freq)
    freq = np.interp(grid, grid[ok], freq[ok])
    amp = np.interp(grid, grid[ok], amp[ok])
    joined = Track(
        grid, freq, amp, max(a.doppler_score, b.doppler_score), None, a.harmonic, a.group, a.kind
    )
    if hasattr(a, "_ridge_offset_hz"):
        joined._ridge_offset_hz = a._ridge_offset_hz
    return joined


def _consolidate(tracks, binwidth, bridge=True):
    """Join by indexed time/frequency bounds before comparing sample arrays."""
    tracks = sorted(tracks, key=lambda tr: (tr.doppler_score >= 0.9, tr.t1 - tr.t0), reverse=True)

    def metadata(tr):
        def slope(tt, ff):
            tt = tt - np.mean(tt)
            return float(np.dot(tt, ff - np.mean(ff)) / max(np.dot(tt, tt), 1e-12))

        n = min(5, len(tr.t))
        first = slope(tr.t[:n], tr.f[:n])
        last = slope(tr.t[-n:], tr.f[-n:])
        low = min(float(np.min(tr.f)), tr.f[0] - 1.75 * first, tr.f[-1] + 1.75 * last) - 8
        high = max(float(np.max(tr.f)), tr.f[0] - 1.75 * first, tr.f[-1] + 1.75 * last) + 8
        return (
            int(round(tr.t0 * 4)),
            int(round(tr.t1 * 4)),
            float(np.median(tr.f)),
            first,
            last,
            int(np.floor(low / 8)),
            int(np.floor(high / 8)),
        )

    changed = True
    while changed:
        changed = False
        result = []
        meta = []
        buckets = {}
        for tr in tracks:
            info = metadata(tr)
            candidates = set()
            for bucket in range(info[5], info[6] + 1):
                candidates.update(buckets.get(bucket, ()))
            match = None
            for j in sorted(candidates):
                old = result[j]
                oi = meta[j]
                if tr.kind != old.kind:
                    continue
                lo = max(info[0], oi[0])
                hi = min(info[1], oi[1])
                if hi >= lo:
                    a0 = lo - info[0]
                    b0 = lo - oi[0]
                    length = hi - lo + 1
                    if length < 3:
                        continue
                    aa = tr.f[a0 : a0 + length]
                    bb = old.f[b0 : b0 + length]
                    if len(aa) != length or len(bb) != length:
                        grid = np.arange(lo, hi + 1) / 4
                        aa = np.interp(grid, tr.t, tr.f)
                        bb = np.interp(grid, old.t, old.f)
                    center = (info[2] + oi[2]) / 2
                    resolution = (
                        binwidth if center <= 450 else max(binwidth, 2.0 if center <= 700 else 4.0)
                    )
                    # Merge a majority-coincident overlap, but _join selects an
                    # observed branch rather than averaging divergent samples.
                    if (hi - lo) / 4 >= min(
                        4.0, 0.5 * min(tr.t1 - tr.t0, old.t1 - old.t0)
                    ) and np.mean(np.abs(aa - bb) <= max(1.0, binwidth)) > 0.5:
                        match = j
                        break
                elif bridge:
                    a, b, am, bm = (tr, old, info, oi) if tr.t1 < old.t0 else (old, tr, oi, info)
                    delta = b.t0 - a.t1
                    if delta > 1.75:
                        continue
                    center = (a.f[-1] + b.f[0]) / 2
                    resolution = (
                        binwidth if center <= 450 else max(binwidth, 2.0 if center <= 700 else 4.0)
                    )
                    uncertainty = center * 120**2 / (343 * 500) * delta * 0.1
                    tolerance = max(
                        2 * binwidth, 0.006 * center, min(uncertainty, max(3.0, 2 * resolution))
                    )
                    if (
                        abs(a.f[-1] + am[4] * delta - b.f[0]) <= tolerance
                        and abs(b.f[0] - bm[3] * delta - a.f[-1]) <= tolerance
                    ):
                        match = j
                        break
            if match is None:
                position = len(result)
                result.append(tr)
                meta.append(info)
            else:
                position = match
                result[position] = _join(result[position], tr)
                info = metadata(result[position])
                meta[position] = info
                changed = True
            for bucket in range(info[5], info[6] + 1):
                buckets.setdefault(bucket, set()).add(position)
        tracks = result
    return tracks


def _continue_lines(tracks, evidence, t, f, cfg, amplitude=None):
    """Hysteresis continuation of already significant seeds, never new births.

    Follow frequency and its local slope under the same hard three-sigma
    physics bound. A weaker continuation threshold bridges at most 1.5 s;
    an absent peak or a harmonic jump ends the path. Verification below still
    clips unsupported spans and rejects note changes before emitting tracks.
    """
    df = float(np.median(np.diff(f)))
    result = []
    for tr in tracks:
        if tr.kind not in ("candidate", "doppler") or not _moving(tr.f, cfg):
            result.append(tr)
            continue
        ends = []
        for direction in (-1, 1):
            origin = 0 if direction < 0 else -1
            frame = int(np.argmin(abs(t - tr.t[origin])))
            history = list(tr.f[:6][::-1] if direction < 0 else tr.f[-6:])
            previous = history[-1]
            pending = []
            accepted = []
            missed = 0
            for _ in range(len(t)):
                frame += direction
                if frame < 0 or frame >= len(t):
                    break
                slope = float(np.median(np.diff(history[-5:]))) if len(history) > 1 else 0.0
                sigma = max(df, previous * 120**2 / (343 * 500) * 0.25)
                predicted = previous + np.clip(slope, -3 * sigma, 3 * sigma)
                if predicted < f[0] or predicted > min(f[-1], cfg.fmax):
                    break
                lo = max(1, int(np.floor((previous - 3 * sigma - f[0]) / df)))
                hi = min(len(f) - 1, int(np.ceil((previous + 3 * sigma - f[0]) / df)) + 1)
                if hi <= lo:
                    break
                bins = np.arange(lo, hi)
                value = evidence[bins, frame]
                peaks = (value >= evidence[bins - 1, frame]) & (
                    value >= evidence[np.minimum(bins + 1, len(f) - 1), frame]
                )
                good = (
                    peaks
                    & (value >= max(2.5, cfg.snr_db - 0.5))
                    & (abs(f[bins] - previous) <= 3 * sigma)
                )
                observed = False
                if good.any():
                    choices = bins[good]
                    smooth = max(df, 0.005 * previous)
                    core = np.log(0.95) - 0.5 * ((f[choices] - predicted) / smooth) ** 2
                    reset = np.log(0.05) - 0.5 * ((f[choices] - previous) / sigma) ** 2
                    score = _robust_llr(evidence[choices, frame]) + np.logaddexp(core, reset)
                    best = int(np.argmax(score))
                    # A geometrically possible jump is not automatically a
                    # valid continuation. Reject negative evidence relative
                    # to the slope prediction instead of hopping to a line.
                    if score[best] >= 0:
                        selected = choices[best]
                        frequency = float(
                            _refine_path(np.array([selected]), np.array([frame]), evidence, f)[0]
                        )
                        observed = abs(frequency - previous) <= 3 * sigma
                if not observed:
                    frequency = predicted
                    above_floor = float(np.interp(predicted, f, evidence[:, frame])) >= 1.5
                    missed = 0 if above_floor else missed + 1
                    if missed > 6:
                        break
                else:
                    missed = 0
                if amplitude is None and abs(frequency - previous) > 2 * df:
                    midpoint = 0.5 * (frequency + previous)
                    bridge = max(
                        float(np.interp(midpoint, f, evidence[:, frame])),
                        float(np.interp(midpoint, f, evidence[:, frame - direction])),
                    )
                    if bridge < 1.0:
                        break
                if amplitude is not None:
                    resolution = 4.0 if frequency >= 700 else 2.0 if frequency >= 450 else df
                    if abs(frequency - previous) > 2 * resolution:
                        middle = 0.5 * (frequency + previous)
                        previous_frame = frame - direction
                        bridge = max(
                            float(np.interp(middle, f, amplitude[:, frame])),
                            float(np.interp(middle, f, amplitude[:, previous_frame])),
                        )
                        endpoints = max(
                            float(np.interp(frequency, f, amplitude[:, frame])),
                            float(np.interp(previous, f, amplitude[:, previous_frame])),
                        )
                        if bridge < 0.10 * endpoints:
                            break
                pending.append((frame, frequency))
                history.append(frequency)
                previous = frequency
                if not missed:
                    accepted.extend(pending)
                    pending = []
            if direction < 0:
                accepted = accepted[::-1]
            ends.append(accepted)
        before, after = ends
        time = np.r_[[t[i] for i, _ in before], tr.t, [t[i] for i, _ in after]]
        freq = np.r_[[v for _, v in before], tr.f, [v for _, v in after]]
        amp = np.interp(time, tr.t, tr.amp)
        if amplitude is not None:
            outside = (time < tr.t0) | (time > tr.t1)
            frames = np.clip(np.searchsorted(t, time[outside]), 0, len(t) - 1)
            bins = np.clip(np.rint((freq[outside] - f[0]) / df).astype(int), 0, len(f) - 1)
            amp[outside] = amplitude[bins, frames]
        result.append(
            Track(time, freq, amp, tr.doppler_score, tr.fit, tr.harmonic, tr.group, tr.kind)
        )
    return result


def _motion_guard(tracks, cfg):
    """Last classification check after every split/merge and prediction."""
    out = []
    for tr in tracks:
        if tr.t1 - tr.t0 < cfg.min_duration_s:
            continue
        if tr.kind == "doppler":
            smooth = ndimage.gaussian_filter1d(tr.f, 2.0)
            excursion = float(np.quantile(smooth, 0.95) - np.quantile(smooth, 0.05))
            if excursion < max(1.5, cfg.min_sweep_hz, 0.01 * np.median(tr.f)):
                tr.kind = "stationary"
                tr.doppler_score = 0.0
        out.append(tr)
    return out


def _finalize(tracks, S, t, f, cfg, base_evidence=None):
    """Verify measured source trajectories before fitting harmonic families.

    Geometry is never replaced by an inferred lattice. In particular a good
    source's approach/recede plateaus survive temporal whitening, while every
    new harmonic must pass its own fixed-curve local evidence test.
    """
    if not tracks:
        return []
    df = float(np.median(np.diff(f)))
    amplitude = getattr(S, "amplitude", None)
    evidence = (
        _evidence(np.asarray(S, dtype=np.float32), f, None)
        if base_evidence is None
        else np.asarray(base_evidence, dtype=np.float32)
    )
    if amplitude is not None:
        raw = 20 * np.log10(np.asarray(amplitude, dtype=np.float32) + 1e-12)
        evidence = np.maximum(evidence, _evidence(raw, f, None) - 4.0)
        evidence[amplitude < max(float(np.max(amplitude)) * 1e-5, 1e-12)] = -20
    tracks = [
        tr
        for tr in tracks
        if tr.t1 - tr.t0 >= 1.75 and (tr.kind != "candidate" or tr.doppler_score >= 0.65)
    ]
    tracks = _consolidate(tracks, df)
    continuation = np.maximum(evidence, 0.5 * np.asarray(S, dtype=np.float32))
    tracks = _continue_lines(tracks, continuation, t, f, cfg, amplitude)
    tracks = _consolidate(tracks, df)
    bounded = []
    for tr in tracks:
        tr.kind = "doppler" if _moving(tr.f, cfg) else "stationary"
        tr.doppler_score = (
            max(cfg.doppler_threshold, tr.doppler_score) if tr.kind == "doppler" else 0.0
        )
        sigma = np.maximum(df, (tr.f[1:] + tr.f[:-1]) / 2 * 120**2 / (343 * 500) * np.diff(tr.t))
        for part in np.split(
            np.arange(len(tr.t)), np.flatnonzero(abs(np.diff(tr.f)) > 3 * sigma) + 1
        ):
            if len(part) < 2 or tr.t[part[-1]] - tr.t[part[0]] < cfg.min_duration_s:
                continue
            bounded.append(
                Track(
                    tr.t[part],
                    tr.f[part],
                    tr.amp[part],
                    tr.doppler_score,
                    None,
                    tr.harmonic,
                    tr.group,
                    tr.kind,
                )
            )
    verified = _clip_verified(bounded, evidence, t, f, cfg, np.asarray(S))
    return _continuous_combs(verified, S, t, f, cfg, evidence)


# Offline null calibration for the default 30–2000 Hz grids. These are
# exactly the eight seeded white/pink/hum trials produced by the fallback
# below (seed 90917+160, 40 s, v_max=120, d_min=500, two velocity beams).
# These bound the model without the optional raw-amplitude bridge, so they
# remain conservative when dark harmonic-gap transitions are rejected.
# Refit after changing framing, evidence, or the HMM. Storing the learned
# maxima avoids recalibrating the same model at every CLI process startup.
_DEFAULT_NULL_CALIBRATION = {
    0.5: {
        "maxima": [0.0, 3.110345720873803, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cutoff": 24.0,
        "single_maxima": [
            9.229268440715208,
            16.959251683431454,
            4.031083185345152,
            9.1511609643939,
            7.529675750003804,
            10.127578808841971,
            4.4590978443290386,
            8.905524755082078,
        ],
        "single_cutoff": 34.41082003508983,
        "high_maxima": [
            20.59713134717043,
            6.936228617904458,
            5.8140309137222665,
            15.353139965286069,
            15.709452496505087,
            12.072001404206365,
            17.403408531201553,
            16.226150616913163,
        ],
        "high_cutoff": 55.58338150712444,
    },
    1.0: {
        "maxima": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "cutoff": 24.0,
        "single_maxima": [
            9.106773079353262,
            5.725433717647189,
            5.5664417028357995,
            9.64020064024275,
            8.992516392112954,
            6.1470818525902295,
            3.501092065275496,
            13.999528220696407,
        ],
        "single_cutoff": 29.381241536872583,
        "high_maxima": [
            22.4352430106204,
            7.789252519394255,
            6.380105144658268,
            10.522085954835951,
            16.632426734029714,
            6.3325449564519065,
            10.07013894628761,
            17.237430785124893,
        ],
        "high_cutoff": 47.936539292933986,
    },
}


def _reference_calibration(frequencies, frames):
    # A longer null trial bounds all shorter edge windows and avoids fitting
    # another eight backgrounds for each final partial window.
    if (
        frames <= 160
        and len(frequencies) > 1
        and frequencies[0] == 30.0
        and frequencies[-1] == 2000.0
    ):
        spacing = round(frequencies[1] - frequencies[0], 6)
        if spacing in _DEFAULT_NULL_CALIBRATION:
            return _DEFAULT_NULL_CALIBRATION[spacing]
    return _reference_calibration_cached(frequencies, max(160, frames))


@lru_cache(maxsize=16)
def _reference_calibration_cached(frequencies, frames):
    """Gumbel null-maxima fit on aircraft-free STFTs, not on the target audio.

    Deterministic white/pink realizations, half with stationary hum, preserve
    the STFT's frequency/time correlations. Cached by grid and frame count.
    Speech/music are separate adversarial null regressions because stationary
    tonal maxima are not samples from the noise-only exponential model.
    """
    f = np.asarray(frequencies)
    dt = 0.25
    cfg = DetectConfig(fmin=float(f[0]) - 0.001, fmax=float(f[-1]) + 0.001)
    rate = max(4000, int(np.ceil(2.4 * cfg.fmax)))
    t = np.arange(frames) * dt
    bf = np.arange(8.0, min(500.0, f[-1] / 2), np.median(np.diff(f)))
    rng = np.random.default_rng(90917 + frames)
    maxima = []
    single = []
    high_scores = []
    n = int((frames * 0.25 + 4) * rate)
    for trial in range(8):
        audio = rng.normal(size=n)
        if trial % 2:
            ff = np.fft.rfftfreq(n, 1 / rate)
            audio = np.fft.irfft(np.fft.rfft(audio) / np.sqrt(np.maximum(ff, 1)), n=n)
            audio /= max(np.std(audio), 1e-12)
        if trial % 3 == 0:
            audio += 2 * np.sin(2 * np.pi * 60 * np.arange(n) / rate)
        # Reversed-time controls preserve the noise/STFT correlations while
        # checking both time directions of the maneuver-tolerant tracker.
        if trial % 2:
            audio = audio[::-1].copy()
        spec, _, rf = ridge_spectrogram(audio, rate, cfg)
        raw = np.asarray(spec[:, 8 : 8 + frames])
        if len(rf) != len(f) or not np.allclose(rf, f, atol=1e-4):
            raw = np.stack([np.interp(f, rf, raw[:, j]) for j in range(raw.shape[1])], axis=1)
        evidence = _evidence(raw, f, None)
        obs = _robust_llr(evidence)
        bank = np.zeros((len(bf), frames))
        norm = np.zeros(len(bf))
        for h in range(1, 25):
            index = np.rint((h * bf - f[0]) / (f[1] - f[0])).astype(int)
            good = (index >= 0) & (index < len(f))
            w = 1 / np.sqrt(h)
            bank[good] += w * obs[index[good]]
            norm[good] += w * w
        bank /= np.sqrt(np.maximum(norm[:, None], 1))
        maxima.append(_viterbi_lines(bank, t, bf, 2.0, limit=1, observation=bank, score_only=True))
        low = np.flatnonzero(f < 450.0)
        single.append(
            _viterbi_lines(evidence[low], t, f[low], 4.0, limit=1, score_only=True)
            if len(low) > 3
            else 0.0
        )
        scores = []
        for lower, upper in ((450, 900), (900, 1400), (1400, np.inf)):
            band = np.flatnonzero((f >= lower) & (f < upper))
            if len(band) > 3:
                scores.append(
                    _viterbi_lines(evidence[band], t, f[band], 4.0, limit=1, score_only=True)
                )
        high_scores.append(max(scores, default=0.0))

    def quantile(values, floor):
        if np.ptp(values) < 1e-6:
            return max(floor, max(values) + 20.0)
        loc, scale = stats.gumbel_r.fit(values)
        return max(floor, float(stats.gumbel_r.ppf(1 - 1e-4, loc=loc, scale=scale)))

    return dict(
        maxima=list(map(float, maxima)),
        cutoff=quantile(maxima, 24.0),
        single_maxima=list(map(float, single)),
        single_cutoff=quantile(single, 24.0),
        high_maxima=list(map(float, high_scores)),
        high_cutoff=quantile(high_scores, 24.0),
    )


def _curve_llr_z(evidence, t, f, time, frequency):
    """Fixed-curve LLR contrast against identically shifted control curves."""
    ti = np.clip(np.searchsorted(t, time), 0, len(t) - 1)
    df = float(np.median(np.diff(f)))

    def statistic(shift):
        loc = np.clip((frequency + shift - f[0]) / df, 0, len(f) - 1)
        lo = np.floor(loc).astype(int)
        hi = np.minimum(lo + 1, len(f) - 1)
        w = loc - lo
        value = (1 - w) * evidence[lo, ti] + w * evidence[hi, ti]
        return float(np.sum(_llr(value)))

    controls = [statistic(shift) for shift in (-14.0, -12.0, -10.0, -8.0, 8.0, 10.0, 12.0, 14.0)]
    return (statistic(0.0) - float(np.mean(controls))) / max(
        float(np.std(controls, ddof=1)), 0.1 * np.sqrt(len(time))
    )


def _clip_verified(tracks, evidence, t, f, cfg, power_db=None):
    """Final unbiased on-curve verification; no maximum over nearby bins.

    Maximizing over a wide frequency corridor also maximizes noise and can
    falsely verify every high order. Interpolate the exact frequency instead.
    Shared-factor predictions pay no further path-search penalty; independent
    lines must retain positive continuation evidence as their source fades.
    """
    groups = {}
    for tr in tracks:
        if tr.kind == "doppler":
            groups.setdefault(tr.group, set()).add(tr.harmonic)
    out = []
    df = float(np.median(np.diff(f)))
    for tr in tracks:
        if tr.kind != "doppler":
            out.append(tr)
            continue
        ti = np.interp(tr.t, t, np.arange(len(t)))
        fi = np.clip((tr.f - f[0]) / df, 0, len(f) - 1)
        t0 = np.floor(ti).astype(int)
        t1 = np.minimum(t0 + 1, len(t) - 1)
        f0 = np.floor(fi).astype(int)
        f1 = np.minimum(f0 + 1, len(f) - 1)
        wt = ti - t0
        wf = fi - f0
        snr = (1 - wt) * ((1 - wf) * evidence[f0, t0] + wf * evidence[f1, t0]) + wt * (
            (1 - wf) * evidence[f0, t1] + wf * evidence[f1, t1]
        )
        llr = _robust_llr(snr)
        if len(groups.get(tr.group, ())) < 2:
            smooth = ndimage.gaussian_filter1d(tr.f, 1.0)
            acceleration = np.r_[0.0, 0.0, np.diff(smooth, n=2)]
            resolution = np.maximum(
                df, np.where(tr.f >= 700.0, 4.0, np.where(tr.f >= 450.0, 2.0, df))
            )
            sigma = np.maximum(resolution, 0.25 * tr.f * 120**3 / (343 * 500**2) * 0.25**2)
            llr -= 0.5 * (acceleration / sigma) ** 2 + np.log(
                np.sqrt(2 * np.pi) * sigma / resolution
            )
        strong = np.median(snr) >= max(6.0, cfg.snr_db + 2.0)
        floor = max(1.5, cfg.snr_db - 1.0) if strong else cfg.snr_db
        continuation_llr = -0.7 if strong else 0.0
        visible = (ndimage.uniform_filter1d(snr, 7, mode="nearest") >= floor) & (
            ndimage.uniform_filter1d(llr, 7, mode="nearest") > continuation_llr
        )
        visible &= (tr.f >= max(cfg.fmin, f[0])) & (tr.f <= min(cfg.fmax, f[-1]))
        absent = np.flatnonzero(~visible)
        for gap in np.split(absent, np.flatnonzero(np.diff(absent) > 1) + 1):
            if 0 < len(gap) <= 6 and gap[0] > 0 and gap[-1] < len(visible) - 1:
                visible[gap] = True
        indices = np.flatnonzero(visible)
        for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(part) < 2 or tr.t[part[-1]] - tr.t[part[0]] < cfg.min_duration_s:
                continue
            if not _moving(tr.f[part], cfg):
                continue
            if (
                _curve_llr_z(
                    evidence if power_db is None else power_db, t, f, tr.t[part], tr.f[part]
                )
                < 4.5
            ):
                continue
            score = 0.70 if tr.doppler_score == 0.55 else tr.doppler_score
            out.append(
                Track(
                    tr.t[part],
                    tr.f[part],
                    tr.amp[part],
                    score,
                    None,
                    tr.harmonic,
                    tr.group,
                    "doppler",
                )
            )
    return sorted(
        _consolidate(out, df, bridge=False), key=lambda tr: (tr.group, tr.harmonic, tr.t0)
    )


def _regroup_sources(tracks):
    """Joint integer-order family fit, with a separate 4–6 Hz sideband family.

    Observed frequencies are immutable. Candidate roots come from a measured
    anchor whose measured order is retained, preventing subharmonic aliases.
    The normalized fit
    tolerance is 0.5 Hz in f0; constant additive offsets are fitted separately
    so the secondary fan cannot drag the principal frequency factor sideways.
    """
    moving = [tr for tr in tracks if tr.kind == "doppler"]
    stationary = [tr for tr in tracks if tr.kind != "doppler"]
    unassigned = set(range(len(moving)))
    families = []
    measured_orders = {}
    for tr in moving:
        if not np.isclose(tr.doppler_score, 0.70):
            measured_orders.setdefault(tr.group, set()).add(tr.harmonic)
    while unassigned:
        measured = {i for i in unassigned if not np.isclose(moving[i].doppler_score, 0.70)}
        anchor = max(
            measured or unassigned,
            key=lambda i: (
                moving[i].doppler_score >= 0.9
                and len(measured_orders.get(moving[i].group, ())) >= 2,
                (moving[i].t1 - moving[i].t0) / np.sqrt(np.median(moving[i].f)),
                moving[i].doppler_score,
            ),
        )
        a = moving[anchor]
        pairs = {}
        for i in unassigned:
            tr = moving[i]
            lo = max(a.t0, tr.t0)
            hi = min(a.t1, tr.t1)
            if hi - lo < min(4.0, 0.5 * min(a.t1 - a.t0, tr.t1 - tr.t0)):
                continue
            tt = np.arange(lo, hi + 0.01, 0.25)
            aa = np.interp(tt, a.t, a.f)
            bb = np.interp(tt, tr.t, tr.f)
            sa = ndimage.gaussian_filter1d(aa, 2.0)
            sb = ndimage.gaussian_filter1d(bb, 2.0)
            if (
                i != anchor
                and np.ptp(sa) > 1.5
                and np.ptp(sb) > 1.5
                and np.corrcoef(sa, sb)[0, 1] < 0.9
            ):
                continue
            pairs[i] = (aa, bb, hi - lo)
        best = None
        # Retain the measured source gauge. Re-dividing an accepted order
        # repeatedly lets reconstructed members invent ever-lower roots.
        for divisor in (max(1, a.harmonic),):
            if np.median(a.f) / divisor < 8:
                continue
            members = []
            orders = []
            offsets = []
            support = 0.0
            for i, (aa, bb, duration) in pairs.items():
                base = aa / divisor
                order = max(1, int(round(np.median(bb / base))))
                if order > 256:
                    continue
                residual = bb - order * base
                offset = float(np.median(residual))
                if abs(offset) > 8:
                    continue
                if abs(offset) < 2.0:
                    offset = 0.0
                elif abs(offset) < 3.0:
                    continue
                if np.quantile(abs(residual - offset) / order, 0.8) > 0.5:
                    continue
                members.append(i)
                orders.append(order)
                offsets.append(offset)
                support += duration * (0.1 if np.isclose(moving[i].doppler_score, 0.70) else 1.0)
            observed_orders = [
                k for i, k in zip(members, orders) if not np.isclose(moving[i].doppler_score, 0.70)
            ]
            if not members or not observed_orders:
                continue
            score = (support, len(members), -max(orders))
            if best is None or score > best[0]:
                best = (score, members, orders, offsets)
        if best is None:
            members = [anchor]
            orders = [max(1, a.harmonic)]
            offsets = [0.0]
        else:
            _, members, orders, offsets = best
        if anchor not in members:
            members.append(anchor)
            orders.append(max(1, a.harmonic))
            offsets.append(0.0)
        buckets = {}
        for i, k, offset in zip(members, orders, offsets):
            key = 0 if offset == 0 else 1 if offset > 0 else -1
            buckets.setdefault(key, []).append((i, k, offset))
        for bucket in buckets.values():
            delta = float(np.median([offset for _, _, offset in bucket]))
            families.append((bucket, delta))
        unassigned.difference_update(members)
    for group, (members, offset) in enumerate(families):
        for i, k, _ in members:
            moving[i].group = group
            moving[i].harmonic = k
            moving[i]._ridge_offset_hz = offset
    for group, tr in enumerate(stationary, start=len(families)):
        tr.group = group
        tr.doppler_score = 0.0
    return moving + stationary


def _weighted_source_factor(grid, members):
    bank = np.stack(
        [
            np.interp(
                grid,
                tr.t,
                (tr.f - getattr(tr, "_ridge_offset_hz", 0.0)) / max(1, tr.harmonic),
                left=np.nan,
                right=np.nan,
            )
            for tr in members
        ]
    )
    precision = np.array(
        [
            max(1, tr.harmonic) ** 2 * min(40.0, tr.t1 - tr.t0) * max(0.1, tr.doppler_score)
            for tr in members
        ]
    )
    weights = np.isfinite(bank) * precision[:, None]
    order = np.argsort(np.nan_to_num(bank, nan=np.inf), axis=0)
    values = np.take_along_axis(np.nan_to_num(bank, nan=np.inf), order, axis=0)
    weights = np.take_along_axis(weights, order, axis=0)
    total = np.sum(weights, axis=0)
    present = total > 0
    median_index = np.argmax(np.cumsum(weights, axis=0) >= 0.5 * total, axis=0)
    factor = values[median_index, np.arange(len(grid))]
    present &= factor >= 8.0
    factor[~present] = np.nan
    return factor, present


def _joint_source_factors(tracks):
    families = {}
    for tr in tracks:
        if tr.kind == "doppler":
            families.setdefault(tr.group, []).append(tr)
    factors = []
    for group, members in families.items():
        # A lone measured order is still a valid conditional source template.
        start = min(tr.t0 for tr in members)
        stop = max(tr.t1 for tr in members)
        grid = np.arange(start, stop + 0.01, 0.25)
        factor, present = _weighted_source_factor(grid, members)
        indices = np.flatnonzero(present)
        for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(part) < 2:
                continue
            source = Track(
                grid[part],
                factor[part],
                np.zeros(len(part)),
                max(tr.doppler_score for tr in members),
                None,
                1,
                group,
                "doppler",
            )
            source._observed_members = members
            source._ridge_offset_hz = float(
                np.median([getattr(tr, "_ridge_offset_hz", 0.0) for tr in members])
            )
            factors.append(source)
    return factors


def _source_shape_guard(tracks):
    families = {}
    for tr in tracks:
        if tr.kind == "doppler":
            families.setdefault(tr.group, []).append(tr)
    rejected = set()
    for group, members in families.items():
        start = min(tr.t0 for tr in members)
        stop = max(tr.t1 for tr in members)
        grid = np.arange(start, stop + 0.01, 0.25)
        base, present = _weighted_source_factor(grid, members)
        turns = 0
        travel = 0.0
        duration = 0.0
        values = []
        indices = np.flatnonzero(present)
        for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
            if len(part) < 3:
                continue
            smooth = ndimage.gaussian_filter1d(base[part], 2.0)
            prominence = max(0.8, 0.01 * np.median(smooth))
            turns += len(signal.find_peaks(smooth, prominence=prominence)[0]) + len(
                signal.find_peaks(-smooth, prominence=prominence)[0]
            )
            travel += float(np.sum(abs(np.diff(smooth))))
            duration += len(part) * 0.25
            values.extend(smooth)
        if duration < 0.5 or not values:
            continue
        excursion = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
        if turns / duration > 0.1 or travel > 4 * max(excursion, 1e-6):
            rejected.add(group)
    return [tr for tr in tracks if tr.kind != "doppler" or tr.group not in rejected]


def _observed_harmonics(tracks, S, t, f, cfg, evidence):
    """Recover individually visible orders without a blind lattice expansion.

    This conditional hypothesis has no independent frequency search: the
    parent is already observed, and contrast is interpolated exactly at k*f.
    A faint sibling need not satisfy the stricter whole-span comb seed gate.
    """
    result = []
    df = float(np.median(np.diff(f)))
    amplitude = getattr(S, "amplitude", None)
    for parent in tracks:
        if parent.kind != "doppler" or np.isclose(parent.doppler_score, 0.70):
            continue
        members = getattr(parent, "_observed_members", None)
        offset = getattr(parent, "_ridge_offset_hz", 0.0)
        if parent.t1 - parent.t0 < max(8.0, 2 * cfg.min_duration_s):
            continue
        if np.median(parent.f) < 40 and not (members and len({tr.harmonic for tr in members}) >= 2):
            continue
        ti = np.clip(np.searchsorted(t, parent.t), 0, len(t) - 1)

        def sample(frequency, values=evidence):
            loc = np.clip((frequency - f[0]) / df, 0, len(f) - 1)
            lo = np.floor(loc).astype(int)
            hi = np.minimum(lo + 1, len(f) - 1)
            weight = loc - lo
            return (1 - weight) * values[lo, ti] + weight * values[hi, ti]

        if members:
            observed = []
            for tr in members:
                inside = (parent.t >= tr.t0) & (parent.t <= tr.t1)
                if inside.any():
                    observed.extend(sample(parent.f * tr.harmonic + offset)[inside])
            anchor = np.asarray(observed)
        else:
            anchor = sample(parent.f)
        if (
            not len(anchor)
            or np.median(anchor) < max(5.0, cfg.snr_db + 1)
            or np.mean(anchor >= 3.0) < 0.65
        ):
            continue
        smooth = ndimage.gaussian_filter1d(parent.f, 2.0)
        excursion = float(np.quantile(smooth, 0.95) - np.quantile(smooth, 0.05))
        prominence = max(1.5, 0.01 * np.median(smooth))
        turns = len(signal.find_peaks(smooth, prominence=prominence)[0]) + len(
            signal.find_peaks(-smooth, prominence=prominence)[0]
        )
        if turns / (parent.t1 - parent.t0) > 0.1 or np.sum(abs(np.diff(smooth))) > 4 * max(
            excursion, 1e-6
        ):
            continue
        first_order = 1 if members else 2
        for multiplier in range(
            first_order, min(256, int(min(cfg.fmax, f[-1]) / max(8.0, np.min(parent.f)))) + 1
        ):
            frequency = multiplier * parent.f + offset
            contrast = sample(frequency)
            threshold = max(1.5, cfg.snr_db - 1.5)
            visible = (
                (frequency >= max(cfg.fmin, f[0]))
                & (frequency <= min(cfg.fmax, f[-1]))
                & (ndimage.uniform_filter1d(contrast, 7, mode="nearest") >= threshold)
            )
            absent = np.flatnonzero(~visible)
            for gap in np.split(absent, np.flatnonzero(np.diff(absent) > 1) + 1):
                if 0 < len(gap) <= 6 and gap[0] > 0 and gap[-1] < len(visible) - 1:
                    visible[gap] = True
            indices = np.flatnonzero(visible)
            for part in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
                if len(part) < 2 or parent.t[part[-1]] - parent.t[part[0]] < cfg.min_duration_s:
                    continue
                # Integrated LLR against identically shifted control curves.
                # This tests a fixed conditional frequency, not the maximum
                # of an optimised corridor, and retains faint visible orders.
                controls = []
                for shift in (-14.0, -12.0, -10.0, -8.0, 8.0, 10.0, 12.0, 14.0):
                    controls.append(
                        float(np.sum(_llr(sample(frequency + shift, np.asarray(S))[part])))
                    )
                statistic = float(np.sum(_llr(sample(frequency, np.asarray(S))[part])))
                z = (statistic - float(np.mean(controls))) / max(
                    float(np.std(controls, ddof=1)), 0.1 * np.sqrt(len(part))
                )
                if z < 4.5:
                    continue
                freq = frequency[part]
                time = parent.t[part]
                if np.quantile(freq, 0.95) - np.quantile(freq, 0.05) < max(
                    1.5, cfg.min_sweep_hz, 0.01 * np.median(freq)
                ):
                    continue
                sigma = np.maximum(df, (freq[1:] + freq[:-1]) / 2 * 120**2 / (343 * 500) * 0.25)
                for piece in np.split(
                    np.arange(len(part)), np.flatnonzero(abs(np.diff(freq)) > 3 * sigma) + 1
                ):
                    if len(piece) < 2 or time[piece[-1]] - time[piece[0]] < cfg.min_duration_s:
                        continue
                    tt = time[piece]
                    ff = freq[piece]
                    amp = (
                        np.zeros(len(piece))
                        if amplitude is None
                        else amplitude[
                            np.clip(np.rint((ff - f[0]) / df).astype(int), 0, len(f) - 1),
                            np.searchsorted(t, tt),
                        ]
                    )
                    proposal = Track(
                        tt,
                        ff,
                        amp,
                        max(0.70, cfg.doppler_threshold),
                        None,
                        parent.harmonic * multiplier,
                        parent.group,
                        "doppler",
                    )
                    proposal._ridge_offset_hz = offset
                    result.append(proposal)
    return result


def _continuous_combs(tracks, S, t, f, cfg, evidence=None):
    """Recover locally observed harmonics of jointly fitted moving sources.

    Only locally verified harmonic segments are emitted. Missing orders do not
    create removal corridors, and a virtual subharmonic never seeds a comb.
    """
    output = _regroup_sources(
        [tr for tr in tracks if tr.kind != "doppler" or tr.doppler_score >= cfg.doppler_threshold]
    )
    if evidence is None:
        evidence = _evidence(np.asarray(S, dtype=np.float32), f, None)
    verified_orders = _observed_harmonics(_joint_source_factors(output), S, t, f, cfg, evidence)
    for tr in output:
        if tr.kind != "doppler":
            tr.doppler_score = 0.0
    df = float(np.median(np.diff(f)))
    final = _motion_guard(_consolidate(output + verified_orders, df, bridge=False), cfg)
    return sorted(_source_shape_guard(final), key=lambda tr: (tr.group, tr.harmonic, tr.t0))
