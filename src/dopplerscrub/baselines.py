"""Baseline defences to compare dopplerscrub against.

Each baseline takes (x, sr) with x shaped (n,) or (n, ch) float32 and returns an
array of the same shape and dtype. They answer one question: does targeted removal
hide the Doppler curves as well as blanket methods while damaging far less audio?

  bandstop  - blanket: zero-phase Butterworth band-stop over the attacker's band
              (40-450 Hz). Removes the curves and everything else down there.
  bandstop_brick - blanket, strongest: STFT bins 35-480 Hz zeroed (added because the
              Butterworth band-stop attenuates only ~10-20 dB near its edges, which a
              per-bin-whitening attacker undoes).
  noise     - blanket: Gaussian noise band-limited to < 500 Hz, added at 0 dB SNR
              relative to the signal's own < 500 Hz power. Masks rather than removes.
  jitter    - "obfuscate everything": a smooth random time-warp ("wow") whose
              instantaneous rate wanders within +-1% (around 0.2 Hz modulation), with
              bounded timing error (about 8 ms). Every tone wobbles by +-1%, so the
              curve no longer matches an ideal pass exactly, but the ridge stays.
  oracle    - remove_tracks with the TRUE tracks (synthetic only): the upper bound for
              dopplerscrub's remover. It separates detector errors from remover errors.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .types import RemoveConfig, Track


def _as2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(x)
    return (x[:, None], True) if x.ndim == 1 else (x, False)


def _back(y: np.ndarray, mono: bool, dtype) -> np.ndarray:
    y = np.clip(y, -1.0, 1.0).astype(dtype)
    return y[:, 0] if mono else y


def bandstop(x: np.ndarray, sr: int, lo: float = 40.0, hi: float = 450.0, order: int = 8) -> np.ndarray:
    """Zero-phase Butterworth band-stop (sosfiltfilt, so the effective order is 2x)."""
    X, mono = _as2d(x)
    sos = ss.butter(order, [lo, hi], btype="bandstop", fs=sr, output="sos")
    return _back(ss.sosfiltfilt(sos, X.astype(np.float64), axis=0), mono, np.asarray(x).dtype)


def bandstop_brick(x: np.ndarray, sr: int, lo: float = 35.0, hi: float = 480.0,
                   win_s: float = 0.5) -> np.ndarray:
    """Strongest practical blanket: zero every STFT bin in [lo, hi] (Hann, 50% overlap,
    ~0.5 s frames, so ~2 Hz bins), then overlap-add. Unlike the Butterworth band-stop,
    whose attenuation is only ~10-20 dB near its edges (which a per-bin-whitening
    attacker undoes), this leaves nothing inside the band but leakage from outside it."""
    X, mono = _as2d(x)
    n = int(2 ** np.round(np.log2(win_s * sr)))
    out = np.empty_like(X, dtype=np.float64)
    for c in range(X.shape[1]):
        f, _, Z = ss.stft(X[:, c].astype(np.float64), sr, nperseg=n, noverlap=n // 2)
        Z[(f >= lo) & (f <= hi)] = 0
        _, y = ss.istft(Z, sr, nperseg=n, noverlap=n // 2)
        out[:, c] = y[:X.shape[0]] if len(y) >= X.shape[0] else np.pad(y, (0, X.shape[0] - len(y)))
    return _back(out, mono, np.asarray(x).dtype)


def noise_mask(x: np.ndarray, sr: int, fmax: float = 500.0, snr_db: float = 0.0, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise low-passed to fmax, at snr_db relative to the signal's own
    power below fmax (per channel, whole-clip average)."""
    X, mono = _as2d(x)
    X = X.astype(np.float64)
    sos = ss.butter(8, fmax, btype="lowpass", fs=sr, output="sos")
    low = ss.sosfiltfilt(sos, X, axis=0)
    rng = np.random.default_rng(seed)
    n = ss.sosfiltfilt(sos, rng.standard_normal(X.shape), axis=0)
    p_sig = np.mean(low ** 2, axis=0, keepdims=True)
    p_n = np.mean(n ** 2, axis=0, keepdims=True) + 1e-30
    n *= np.sqrt(p_sig / p_n * 10 ** (-snr_db / 10))
    return _back(X + n, mono, np.asarray(x).dtype)


def jitter(x: np.ndarray, sr: int, depth: float = 0.01, rate_hz: float = 0.2, seed: int = 0) -> np.ndarray:
    """Random time-varying resample ("wow"): output sample n reads input position
    n + d(n), where d is a smooth random displacement band-limited around rate_hz
    (0.1-0.4 Hz), scaled so the instantaneous rate 1 + d'(t) stays within +-depth.
    Every tone's pitch wobbles by up to +-depth (+-1.5 Hz at 150 Hz); because d is
    bounded (about depth / (2 pi rate_hz), roughly 8 ms) the timing never drifts away,
    so aligned fidelity metrics measure the wobble rather than a growing offset.
    Output keeps the input length."""
    X, mono = _as2d(x)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    fc = 50.0                                   # control-signal rate
    m = int(np.ceil(n / sr * fc)) + 8
    sos = ss.butter(2, [rate_hz / 2, rate_hz * 2], btype="bandpass", fs=fc, output="sos")
    u = ss.sosfiltfilt(sos, rng.standard_normal(m + 400))[200:200 + m]
    du = np.gradient(u) * fc                    # d'(t) per second of the unscaled shape
    scale = depth / (np.max(np.abs(du)) + 1e-12)
    d = scale * u * sr                          # displacement in samples
    pos = np.clip(np.arange(n) + np.interp(np.arange(n) / sr * fc, np.arange(m), d), 0, n - 1)
    grid = np.arange(n)
    Y = np.stack([np.interp(pos, grid, X[:, c].astype(np.float64)) for c in range(X.shape[1])], axis=1)
    return _back(Y, mono, np.asarray(x).dtype)


def oracle(x: np.ndarray, sr: int, true_tracks: list[Track], cfg: RemoveConfig | None = None) -> np.ndarray:
    """dopplerscrub's remover driven by the TRUE tracks (synthetic clips only)."""
    from .remove import remove_tracks
    cfg = cfg or RemoveConfig(harden=True)
    tr = [Track(t.t, t.f, t.amp, t.doppler_score or 1.0, t.fit, t.harmonic, t.group, "doppler")
          for t in true_tracks]
    return remove_tracks(x, sr, tr, cfg)


def dopplerscrub(x: np.ndarray, sr: int, harden: bool = True) -> np.ndarray:
    """The method under test (in-memory pipeline.scrub_array, defaults, harden=True)."""
    from .pipeline import scrub_array
    y, _ = scrub_array(x, sr, remove_cfg=RemoveConfig(harden=harden))
    return y


BLANKET = {"bandstop": bandstop, "bandstop_brick": bandstop_brick, "noise": noise_mask, "jitter": jitter}
