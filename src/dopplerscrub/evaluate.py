"""Evaluation: the attacker's view (whitened spectrograms) and cheap numeric checks.

Success is judged VISUALLY: in the attacker's whitened low-band spectrogram,
the Doppler curves must be gone from the cleaned audio and everything else must look
unchanged. We never run dopplertrack's slow fit/search here.

Public API
----------
attack_spectrogram(x, sr, ...)            -> Spectrogram  (exactly dopplertrack's view)
attack_spectrogram_wide(x, sr, ...)       -> Spectrogram  (same, 40-1500 Hz)
attack_views(x_orig, x_clean, sr, ...)    -> dict(orig, clean, diff) on one grid
render_before_after(x_orig, x_clean, sr, out_png, tracks=None, window=None, ...)
render_attacker_views(x_clean, sr, out_png, ...)   adversarial red-team views
track_contrast_db(spec, tracks)           -> float  (ghost > 0, notch < 0, in attacker dB)
residual_db(x_orig, x_clean, sr, tracks)  -> float  (narrowband tone energy after/before)
residual_details(...)                     -> list[dict] per track
fidelity(x_orig, x_clean, sr, exclude_tracks=None) -> dict
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.signal as ss
from scipy.ndimage import uniform_filter1d

from .types import Track

ATTACK_SR = 4000
NPERSEG = 8192
HOP_S = 0.25
FMIN, FMAX, FMAX_WIDE = 40.0, 450.0, 1500.0
WHITEN_S = 600.0


# --------------------------------------------------------------------------- utils

def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.mean(axis=1) if x.ndim == 2 else x


def _resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return np.asarray(x, np.float32)
    g = gcd(int(sr), int(target))
    return ss.resample_poly(x, target // g, sr // g).astype(np.float32)


def _attack_signal(x: np.ndarray, sr: int) -> np.ndarray:
    """Attacker's decode: mono, 4 kHz."""
    return _resample(_mono(x), sr, ATTACK_SR)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


@dataclass
class Spectrogram:
    """Same layout as dopplertrack.audio.Spectrogram: S is (F, T) dB above per-bin median."""
    t: np.ndarray
    f: np.ndarray
    S: np.ndarray

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0]) if len(self.t) > 1 else HOP_S

    @property
    def df(self) -> float:
        return float(self.f[1] - self.f[0])

    def crop(self, t0=None, t1=None, f0=None, f1=None) -> "Spectrogram":
        tm = np.ones_like(self.t, bool)
        fm = np.ones_like(self.f, bool)
        if t0 is not None: tm &= self.t >= t0
        if t1 is not None: tm &= self.t <= t1
        if f0 is not None: fm &= self.f >= f0
        if f1 is not None: fm &= self.f <= f1
        return Spectrogram(self.t[tm], self.f[fm], self.S[np.ix_(fm, tm)])

    def sample(self, t, f, fill=np.nan):
        t = np.asarray(t, float); f = np.asarray(f, float)
        ti = (t - self.t[0]) / self.dt
        fi = (f - self.f[0]) / self.df
        ok = (np.isfinite(ti) & np.isfinite(fi) & (ti >= 0) & (ti <= len(self.t) - 1)
              & (fi >= 0) & (fi <= len(self.f) - 1))
        out = np.full(t.shape, fill, float)
        ti, fi = ti[ok], fi[ok]
        i0 = np.floor(ti).astype(int); j0 = np.floor(fi).astype(int)
        i1 = np.minimum(i0 + 1, len(self.t) - 1); j1 = np.minimum(j0 + 1, len(self.f) - 1)
        wt, wf = ti - i0, fi - j0
        S = self.S
        out[ok] = (S[j0, i0] * (1 - wt) * (1 - wf) + S[j0, i1] * wt * (1 - wf)
                   + S[j1, i0] * (1 - wt) * wf + S[j1, i1] * wt * wf)
        return out


# ------------------------------------------------------------ attacker spectrogram

def _raw_db(x4k: np.ndarray, fmin: float, fmax: float, nperseg=NPERSEG, hop_s=HOP_S,
            t_offset: float = 0.0, chunk_frames: int = 4096):
    """Un-whitened dB STFT, bit-for-bit the same framing as dopplertrack."""
    sr = ATTACK_SR
    hop = max(1, int(round(hop_s * sr)))
    K = (len(x4k) - nperseg) // hop + 1
    if K < 1:
        raise ValueError(f"signal too short for the attacker STFT ({len(x4k)/sr:.1f} s)")
    f = np.fft.rfftfreq(nperseg, 1 / sr)
    band = (f >= fmin) & (f <= fmax)
    parts = []
    for k0 in range(0, K, chunk_frames):
        k1 = min(k0 + chunk_frames, K)
        seg = x4k[k0 * hop:(k1 - 1) * hop + nperseg]
        _, _, Z = ss.stft(seg, sr, nperseg=nperseg, noverlap=nperseg - hop, window="hann",
                          boundary=None, padded=False)
        parts.append((10 * np.log10(np.abs(Z[band]) ** 2 + 1e-20)).astype(np.float32))
    t = t_offset + (nperseg / 2 + np.arange(K) * hop) / sr
    return t, f[band], np.concatenate(parts, axis=1)


def _baseline(L: np.ndarray, whiten_window_s: float | None, hop_s=HOP_S) -> np.ndarray:
    """Per-bin median baseline exactly as dopplertrack.compute_spectrogram builds it."""
    if whiten_window_s is None:
        return np.median(L, axis=1, keepdims=True)
    w = int(whiten_window_s / hop_s)
    if w >= L.shape[1]:
        return np.median(L, axis=1, keepdims=True)
    nb = int(np.ceil(L.shape[1] / w))
    centers, cm = [], []
    for b in range(nb):
        sl = slice(b * w, min((b + 1) * w, L.shape[1]))
        centers.append((sl.start + sl.stop) / 2)
        cm.append(np.median(L[:, sl], axis=1))
    cm = np.asarray(cm)  # (nb, F)
    idx = np.arange(L.shape[1])
    return np.stack([np.interp(idx, centers, cm[:, k]) for k in range(L.shape[0])]).astype(np.float32)


def attack_spectrogram(x: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = FMAX,
                       whiten_window_s: float | None = WHITEN_S):
    """The attacker's view: mono -> 4 kHz -> dopplertrack.audio.compute_spectrogram
    (8192-pt STFT, 0.25 s hop, 40-450 Hz, per-bin running-median whitening).

    Uses dopplertrack itself when it is installed (dev dependency); otherwise an
    identical local implementation.
    """
    y = _attack_signal(x, sr)
    try:
        from dopplertrack.audio import compute_spectrogram
    except ImportError:  # pragma: no cover - dopplertrack is a dev dependency
        t, f, L = _raw_db(y, fmin, fmax)
        return Spectrogram(t, f, (L - _baseline(L, whiten_window_s)).astype(np.float32))
    return compute_spectrogram(y, ATTACK_SR, fmin=fmin, fmax=fmax, whiten_window_s=whiten_window_s)


def attack_spectrogram_wide(x: np.ndarray, sr: int, fmax: float = FMAX_WIDE, **kw):
    """Wider-band variant (40-1500 Hz) for vehicle/F1 harmonics; same STFT otherwise."""
    return attack_spectrogram(x, sr, fmax=fmax, **kw)


def attack_views(x_orig: np.ndarray, x_clean: np.ndarray, sr: int, fmin: float = FMIN,
                 fmax: float = FMAX, whiten_window_s: float | None = WHITEN_S,
                 span: tuple[float, float] | None = None, margin_s: float = 300.0) -> dict:
    """Compute the attacker's view of original, cleaned and removed (orig - clean) once.

    orig and clean are each whitened by their OWN baseline (what an attacker sees).
    diff is shown against the ORIGINAL's baseline, so it is on the same dB scale.
    `span=(t0, t1)` only analyses that span plus `margin_s` of whitening context
    each side (fast zooms into long files; whitening then differs slightly from a
    whole-file run).
    Returns {"orig", "clean", "diff": Spectrogram, "raw_orig", "raw_clean": raw dB arrays}.
    """
    a = _attack_signal(x_orig, sr)
    b = _attack_signal(x_clean, sr)
    a, b = _align(a, b)
    off = 0.0
    if span is not None:
        s0 = max(0, int((span[0] - margin_s) * ATTACK_SR))
        s1 = min(len(a), int((span[1] + margin_s) * ATTACK_SR))
        a, b, off = a[s0:s1], b[s0:s1], s0 / ATTACK_SR
    t, f, La = _raw_db(a, fmin, fmax, t_offset=off)
    _, _, Lb = _raw_db(b, fmin, fmax, t_offset=off)
    _, _, Ld = _raw_db(a - b, fmin, fmax, t_offset=off)
    Ba = _baseline(La, whiten_window_s)
    Bb = _baseline(Lb, whiten_window_s)
    return {"orig": Spectrogram(t, f, La - Ba), "clean": Spectrogram(t, f, Lb - Bb),
            "diff": Spectrogram(t, f, Ld - Ba), "raw_orig": La, "raw_clean": Lb,
            "whiten_window_s": whiten_window_s}


# ------------------------------------------------------------------ track contrast

def _track_points(tracks: Iterable[Track], dt: float = HOP_S):
    for tr in tracks:
        t = np.asarray(tr.t, float); f = np.asarray(tr.f, float)
        if len(t) < 2:
            continue
        tt = np.arange(t[0], t[-1], dt)
        yield tt, np.interp(tt, t, f)


def track_contrast_db(spec: Spectrogram, tracks: Sequence[Track], off_hz=(3.0, 6.0)) -> float:
    """Mean attacker-dB ON the curves minus mean attacker-dB beside them (+-3..6 Hz).

    >0 : a ridge (curve / ghost) is still there.  <0 : a curve-shaped notch.
    ~0 : nothing distinguishes the corridor from its surroundings (the goal).
    NaN if no track point falls inside the spectrogram.
    """
    on, side = [], []
    for tt, ff in _track_points(tracks):
        v = spec.sample(tt, ff)
        offs = [spec.sample(tt, ff + s * o) for o in np.linspace(*off_hz, 4) for s in (-1, 1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nb = np.nanmean(np.stack(offs), axis=0)
        ok = np.isfinite(v) & np.isfinite(nb)
        on.append(v[ok]); side.append(nb[ok])
    if not on or sum(len(o) for o in on) == 0:
        return float("nan")
    return float(np.mean(np.concatenate(on)) - np.mean(np.concatenate(side)))


def smooth_along_time(spec: Spectrogram, frames: int = 9) -> Spectrogram:
    """Attacker trick: average whitened dB over ~2 s to pull faint ridges out of noise."""
    return Spectrogram(spec.t, spec.f, uniform_filter1d(spec.S, frames, axis=1))


# ------------------------------------------------------------------ rendering

def _pool_time(spec: Spectrogram, max_cols: int = 2400) -> Spectrogram:
    """Max-pool along time so long views keep faint thin lines visible on screen."""
    T = spec.S.shape[1]
    if T <= max_cols:
        return spec
    k = int(np.ceil(T / max_cols))
    n = (T // k) * k
    S = spec.S[:, :n].reshape(spec.S.shape[0], -1, k).max(axis=2)
    t = spec.t[:n].reshape(-1, k).mean(axis=1)
    return Spectrogram(t, spec.f, S)


def _imshow(ax, spec: Spectrogram, vmin, vmax, cmap):
    spec = _pool_time(spec)
    dt = spec.dt; df = spec.df
    ext = [spec.t[0] - dt / 2, spec.t[-1] + dt / 2, spec.f[0] - df / 2, spec.f[-1] + df / 2]
    return ax.imshow(spec.S, origin="lower", aspect="auto", extent=ext, vmin=vmin, vmax=vmax,
                     cmap=cmap, interpolation="antialiased")


def _overlay(ax, tracks, window, color_doppler="#00e5ff", color_other="#7CFC00"):
    ylim = ax.get_ylim()  # keep the spectrogram's band; tracks outside it are clipped
    for tr in tracks or []:
        t = np.asarray(tr.t, float); f = np.asarray(tr.f, float)
        c = color_doppler if tr.kind == "doppler" else color_other
        ax.plot(t, f, ls=(0, (4, 3)), lw=0.9, color=c, alpha=0.9)
    ax.set_ylim(*ylim)
    if window is not None:
        ax.set_xlim(window[0], window[1])
        if len(window) == 4:
            ax.set_ylim(window[2], window[3])


def _crop_for_window(spec: Spectrogram, window):
    if window is None:
        return spec
    t0, t1 = window[0], window[1]
    f0, f1 = (window[2], window[3]) if len(window) == 4 else (None, None)
    return spec.crop(t0, t1, f0, f1)


def render_before_after(x_orig, x_clean, sr, out_png, tracks: Sequence[Track] | None = None,
                        window: tuple | None = None, title: str = "Attacker view: before / after",
                        fmax: float = FMAX, vmin: float = -3.0, vmax: float = 15.0,
                        cmap: str = "magma", views: dict | None = None,
                        whiten_window_s: float | None = WHITEN_S,
                        overlay_label: str = "cyan = doppler, green = other") -> dict:
    """Stacked panels on one colour scale: original, cleaned, removed (orig-clean), and,
    when `tracks` is given, a separate copy of the original with tracks dashed on top.
    The cleaned panel is never annotated.

    window: (t0, t1) or (t0, t1, f0, f1) in seconds / Hz.
    views:  precomputed `attack_views(...)` output (reuse it for several windows).
    Returns the views dict (so callers can reuse it).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if views is None:
        span = (window[0], window[1]) if window is not None else None
        views = attack_views(x_orig, x_clean, sr, fmax=fmax, whiten_window_s=whiten_window_s,
                             span=span)
    panels = [("Original (attacker view, whitened)", views["orig"]),
              ("Cleaned (attacker view, whitened) - unannotated", views["clean"]),
              ("Removed content (orig - clean, on original's baseline)", views["diff"])]
    if tracks:
        panels.append((f"Original + tracks (dashed: {overlay_label})", views["orig"]))
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 2.9 * len(panels)), sharex=True,
                             constrained_layout=True)
    im = None
    for i, (ax, (name, spec)) in enumerate(zip(axes, panels)):
        im = _imshow(ax, _crop_for_window(spec, window), vmin, vmax, cmap)
        ax.set_ylabel("Hz")
        ax.set_title(name, fontsize=10, loc="left")
        if tracks and i == len(panels) - 1:
            _overlay(ax, tracks, window)
        elif window is not None:
            ax.set_xlim(window[0], window[1])
            if len(window) == 4:
                ax.set_ylim(window[2], window[3])
    axes[-1].set_xlabel("time (s)")
    fig.colorbar(im, ax=axes, shrink=0.9, label="dB above per-bin median")
    fig.suptitle(title, fontsize=12)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return views


def render_attacker_views(x_clean, sr, out_png, tracks: Sequence[Track] | None = None,
                          window: tuple | None = None, fmax: float = FMAX,
                          title: str = "Red-team views of the cleaned audio",
                          x_orig=None) -> dict:
    """Adversarial looks at the CLEANED audio, the way a determined attacker would:
      a) standard view, tight low floor (-2..8 dB)
      b) diverging map centred at 0 (blue = notch, red = ghost), standard whitening
      c) whole-file (global-median) whitening, diverging
      d) ~2 s time-smoothed, diverging, tight (+-2 dB), pulls faint ghosts out of noise
      e) same as (d) with the tracks dashed on top (to know where to look)
      f) curve-aligned profile: mean attacker dB vs offset from the track (needs tracks).
         A bump at 0 Hz = ghost, a dip = notch. This is the most sensitive look:
         it averages along the entire curve, as an attacker who knows the curve would.
    Returns {"contrast_db": {view: dB}, "profile": {...}} along `tracks` (ghost>0, notch<0).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = (window[0], window[1]) if window is not None else None
    ref = x_orig if x_orig is not None else x_clean
    v600 = attack_views(ref, x_clean, sr, fmax=fmax, whiten_window_s=WHITEN_S, span=span)
    vglob = attack_views(ref, x_clean, sr, fmax=fmax, whiten_window_s=None, span=span)
    clean = v600["clean"]; glob = vglob["clean"]
    sm = smooth_along_time(clean, 9)
    panels = [("a) standard whitening, low floor", clean, -2, 8, "magma"),
              ("b) standard whitening, diverging (blue = notch, red = ghost)", clean, -6, 6, "RdBu_r"),
              ("c) global-median whitening, diverging", glob, -6, 6, "RdBu_r"),
              ("d) 2 s time-smoothed, diverging +-2 dB", sm, -2, 2, "RdBu_r")]
    if tracks:
        panels.append(("e) as (d) with tracks dashed", sm, -2, 2, "RdBu_r"))
    nrow = len(panels) + (1 if tracks else 0)
    fig, axs = plt.subplots(nrow, 1, figsize=(13, 2.7 * nrow), constrained_layout=True)
    axs = np.atleast_1d(axs)
    axes = axs[:len(panels)]
    for ax in axes[1:]:
        ax.sharex(axes[0])
    for i, (ax, (name, spec, lo, hi, cm)) in enumerate(zip(axes, panels)):
        im = _imshow(ax, _crop_for_window(spec, window), lo, hi, cm)
        fig.colorbar(im, ax=ax, pad=0.01, label="dB")
        ax.set_title(name, fontsize=10, loc="left"); ax.set_ylabel("Hz")
        if tracks and i == len(panels) - 1:
            _overlay(ax, tracks, window, color_doppler="k", color_other="#333333")
        elif window is not None:
            ax.set_xlim(window[0], window[1])
            if len(window) == 4:
                ax.set_ylim(window[2], window[3])
    axes[-1].set_xlabel("time (s)")
    prof = {}
    if tracks:
        ax = axs[-1]
        trw = _tracks_in_window(tracks, window)
        prof["offset_hz"], prof["clean"] = curve_profile(clean, trw)
        _, prof["clean_global"] = curve_profile(glob, trw)
        _, prof["clean_smoothed"] = curve_profile(sm, trw)
        per = []
        for k, tr in enumerate(trw[:10]):
            o, pk = curve_profile(clean, [tr])
            per.append({"f_mean": float(np.mean(tr.f)), "profile": pk})
            ax.plot(o, pk, lw=0.6, alpha=0.6, color=plt.cm.viridis(k / max(1, min(len(trw), 10) - 1)),
                    label=f"cleaned, track ~{np.mean(tr.f):.0f} Hz")
        prof["per_track"] = per
        ax.plot(prof["offset_hz"], prof["clean"], color="C3", lw=1.8, label="cleaned (all tracks)")
        ax.plot(prof["offset_hz"], prof["clean_global"], color="C1", lw=1, ls="--",
                label="cleaned (global whitening)")
        if x_orig is not None:
            _, prof["orig"] = curve_profile(v600["orig"], trw)
            ax2 = ax.twinx()
            ax2.plot(prof["offset_hz"], prof["orig"], color="0.6", lw=1, label="original (right axis)")
            ax2.set_ylabel("original dB", color="0.5")
            ax2.legend(loc="upper right", fontsize=8)
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("offset from track frequency (Hz)"); ax.set_ylabel("mean dB")
        ax.set_title("f) curve-aligned profile (bump at 0 = ghost, dip = notch)", fontsize=10, loc="left")
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=12)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    out = {}
    if tracks:
        out = {"standard": track_contrast_db(clean, tracks),
               "global": track_contrast_db(glob, tracks),
               "smoothed": track_contrast_db(sm, tracks)}
        if x_orig is not None:
            out["orig_standard"] = track_contrast_db(v600["orig"], tracks)
    return {"contrast_db": out, "profile": prof}


def _tracks_in_window(tracks, window):
    if window is None:
        return list(tracks)
    res = []
    for tr in tracks:
        m = (np.asarray(tr.t) >= window[0]) & (np.asarray(tr.t) <= window[1])
        if m.sum() >= 2:
            res.append(Track(np.asarray(tr.t)[m], np.asarray(tr.f)[m], np.asarray(tr.amp)[m],
                             kind=tr.kind, harmonic=tr.harmonic, group=tr.group))
    return res


def curve_profile(spec: Spectrogram, tracks: Sequence[Track], max_off_hz: float = 15.0):
    """Mean whitened dB as a function of frequency offset from the tracks (averaged over
    every track point). Returns (offsets_hz, mean_db)."""
    offs = np.arange(-max_off_hz, max_off_hz + 1e-9, spec.df / 2)
    rows = []
    for tt, ff in _track_points(tracks):
        rows.append(np.stack([spec.sample(tt, ff + o) for o in offs], axis=1))  # (n, O)
    if not rows:
        return offs, np.full(len(offs), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return offs, np.nanmean(np.concatenate(rows, axis=0), axis=0)


# ------------------------------------------------------------------ numeric checks

def _heterodyne_energy(x: np.ndarray, sr: int, tt: np.ndarray, ff: np.ndarray,
                       offset_hz: float = 0.0, avg_s: float = 0.5) -> float:
    """Mean power of x in a ~2 Hz band that follows the curve ff(tt) (+offset)."""
    s0 = max(0, int(tt[0] * sr)); s1 = min(len(x), int(tt[-1] * sr))
    if s1 - s0 < int(avg_s * sr) * 2:
        return float("nan")
    ts = np.arange(s0, s1) / sr
    fs = np.interp(ts, tt, ff) + offset_hz
    ph = 2 * np.pi * np.cumsum(fs) / sr
    y = x[s0:s1].astype(np.float64) * np.exp(-1j * ph)
    n = max(1, int(avg_s * sr))
    lp = uniform_filter1d(y.real, n) + 1j * uniform_filter1d(y.imag, n)
    return float(np.mean(np.abs(lp) ** 2))


def _work_signal(x, sr, tracks):
    fmax = max((float(np.nanmax(tr.f)) for tr in tracks if len(tr.f)), default=0.0)
    wsr = ATTACK_SR if fmax < 0.45 * ATTACK_SR else int(sr)
    return _resample(_mono(x), sr, wsr), wsr


def residual_details(x_orig, x_clean, sr, tracks: Sequence[Track], floor_offset_hz: float = 6.0):
    """Per track: narrowband (heterodyne, ~2 Hz) energy along the curve before/after.

    before_db / after_db : 10log10 mean power on the curve
    floor_db             : power at +-floor_offset_hz beside the curve in the ORIGINAL (noise)
    residual_db          : after - before (raw; cannot go much below floor - before)
    tone_snr_before_db   : before - floor (how far above noise the tone was)
    """
    a, w = _work_signal(x_orig, sr, tracks)
    b, _ = _work_signal(x_clean, sr, tracks)
    a, b = _align(a, b)
    out = []
    for tr in tracks:
        t = np.asarray(tr.t, float); f = np.asarray(tr.f, float)
        if len(t) < 2:
            continue
        eb = _heterodyne_energy(a, w, t, f)
        ea = _heterodyne_energy(b, w, t, f)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fl = np.nanmean([_heterodyne_energy(a, w, t, f, s * floor_offset_hz) for s in (-1, 1)])
        db = lambda e: 10 * np.log10(e + 1e-30)
        out.append({"t0": tr.t0, "t1": tr.t1, "f_mean": float(np.mean(f)), "kind": tr.kind,
                    "before": eb, "after": ea, "floor": fl,
                    "before_db": db(eb), "after_db": db(ea), "floor_db": db(fl),
                    "residual_db": db(ea) - db(eb), "tone_snr_before_db": db(eb) - db(fl)})
    return out


def residual_db(x_orig, x_clean, sr, tracks: Sequence[Track]) -> float:
    """Tone energy left along the curves after vs before, in dB (energy-weighted over
    all tracks). -inf..0; e.g. -20 = 99% of on-curve narrowband energy removed.
    Noise-limited: it bottoms out near (noise floor - original tone level)."""
    d = [r for r in residual_details(x_orig, x_clean, sr, tracks) if np.isfinite(r["before"])]
    if not d:
        return float("nan")
    return float(10 * np.log10((sum(r["after"] for r in d) + 1e-30) /
                               (sum(r["before"] for r in d) + 1e-30)))


def _snr(sig_e: float, err_e: float) -> float:
    if err_e <= 0:
        return float("inf")
    return float(10 * np.log10(sig_e / err_e + 1e-30))


def fidelity(x_orig, x_clean, sr, exclude_tracks: Sequence[Track] | None = None,
             corridor_hz: float = 3.0, pad_s: float = 2.0, nperseg: int | None = None) -> dict:
    """How untouched is everything else?

    snr_db             : whole-signal SNR orig / (orig - clean)
    snr_outside_db     : same, in the STFT domain, excluding +-corridor_hz around each
                         excluded track (during its span +- pad_s)
    speech_snr_db      : SNR in 300-3400 Hz (time domain, 4th-order Butterworth)
    octave_change_db   : {centre_Hz: 10log10(E_clean/E_orig)} per octave band
    """
    a, b = _align(_mono(x_orig).astype(np.float64), _mono(x_clean).astype(np.float64))
    d = a - b
    out = {"snr_db": _snr(np.sum(a * a), np.sum(d * d))}

    # outside corridors: chunked STFT so hour-long files stay cheap
    if nperseg is None:
        nperseg = int(2 ** np.ceil(np.log2(sr / 2.0)))  # ~2 Hz bins
    hop = nperseg // 2
    freqs = np.fft.rfftfreq(nperseg, 1 / sr)
    win = ss.get_window("hann", nperseg)
    tracks = list(exclude_tracks or [])
    se = ee = 0.0
    chunk = hop * 256
    for c0 in range(0, max(1, len(a) - nperseg + 1), chunk):
        starts = np.arange(c0, min(c0 + chunk, len(a) - nperseg + 1), hop)
        if len(starts) == 0:
            break
        idx = starts[:, None] + np.arange(nperseg)[None]
        A = np.fft.rfft(a[idx] * win, axis=1)
        D = np.fft.rfft(d[idx] * win, axis=1)
        keep = np.ones(A.shape, bool)
        tc = (starts + nperseg / 2) / sr
        for tr in tracks:
            t = np.asarray(tr.t, float); f = np.asarray(tr.f, float)
            if len(t) == 0:
                continue
            m = (tc >= t[0] - pad_s) & (tc <= t[-1] + pad_s)
            if not m.any():
                continue
            fc = np.interp(tc[m], t, f)
            for k in (1,):  # each harmonic is its own Track in our contract
                keep[m] &= np.abs(freqs[None, :] - k * fc[:, None]) > corridor_hz
        se += float(np.sum(np.abs(A[keep]) ** 2))
        ee += float(np.sum(np.abs(D[keep]) ** 2))
    out["snr_outside_db"] = _snr(se, ee)

    hi = min(3400.0, 0.45 * sr)
    if hi > 300:
        sos = ss.butter(4, [300, hi], btype="band", fs=sr, output="sos")
        sa = ss.sosfilt(sos, a); sd = ss.sosfilt(sos, d)
        out["speech_snr_db"] = _snr(np.sum(sa * sa), np.sum(sd * sd))
    else:
        out["speech_snr_db"] = float("nan")

    fw, Pa = ss.welch(a, sr, nperseg=min(len(a), 1 << 14))
    _, Pb = ss.welch(b, sr, nperseg=min(len(b), 1 << 14))
    oc = {}
    c = 31.25
    while c * np.sqrt(2) <= sr / 2:
        m = (fw >= c / np.sqrt(2)) & (fw < c * np.sqrt(2))
        if m.any():
            oc[int(round(c))] = float(10 * np.log10((Pb[m].sum() + 1e-30) / (Pa[m].sum() + 1e-30)))
        c *= 2
    out["octave_change_db"] = oc
    return out


# ------------------------------------------------------------------ anti-forensics
# Checks from the ENF-removal anti-forensics literature (docs/research.md):
# Chuang, Garg & Wu (CCS 2012 / TIFS 2013), Nishimura (IEICE 2017).

def _lin_on_curve(raw: np.ndarray, t: np.ndarray, f: np.ndarray, tt, ff):
    """Linear power at (tt, ff) on a raw-dB grid: max of the two nearest bins
    (scallop-proof; apply the same rule to flanks so the comparison is fair)."""
    ti = np.round((tt - t[0]) / (t[1] - t[0])).astype(int)
    fi = (ff - f[0]) / (f[1] - f[0])
    j0 = np.floor(fi).astype(int)
    ok = (ti >= 0) & (ti < raw.shape[1]) & (j0 >= 0) & (j0 + 1 < raw.shape[0])
    out = np.full(len(tt), np.nan)
    a = raw[j0[ok], ti[ok]]; b = raw[j0[ok] + 1, ti[ok]]
    out[ok] = 10 ** (np.maximum(a, b) / 10)
    return out


def _ratio_to_floor(raw, t, f, tracks, shift=0.0, flank=(3.0, 6.0)):
    """Mean over all curve frames of P(curve+shift) / mean P(flanks at +-3..6 Hz)."""
    num, n = 0.0, 0
    for tt, ff in _track_points(tracks):
        p = _lin_on_curve(raw, t, f, tt, ff + shift)
        nb = np.stack([_lin_on_curve(raw, t, f, tt, ff + shift + s * o)
                       for o in np.linspace(*flank, 4) for s in (-1, 1)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nbm = np.nanmean(nb, axis=0)
        ok = np.isfinite(p) & np.isfinite(nbm) & (nbm > 0)
        num += float(np.sum(p[ok] / nbm[ok])); n += int(ok.sum())
    return (num / n if n else float("nan")), n


NULL_SHIFTS_HZ = (-14.0, -12.0, -10.0, -8.0, 8.0, 10.0, 12.0, 14.0)


def _curve_vs_floor(raw, t, f, tracks, flank=(3.0, 6.0)):
    """On-curve power vs the local per-bin floor, integrated over EVERY frame of every
    curve (what an attacker summing the whole pass sees). Significance is calibrated
    empirically: the identical statistic on copies of the curve shifted by +-8..14 Hz
    is the null distribution (this accounts for the 8x frame overlap and bin
    correlation). excess is in units of the floor (0 = nothing, 0.1 = -10 dB)."""
    r_on, n = _ratio_to_floor(raw, t, f, tracks, 0.0, flank)
    null = np.array([_ratio_to_floor(raw, t, f, tracks, s, flank)[0] for s in NULL_SHIFTS_HZ])
    null = null[np.isfinite(null)]
    if n < 8 or len(null) < 3 or not np.isfinite(r_on):
        return {"n_frames": n, "excess": float("nan"), "z": float("nan"),
                "excess_db": float("nan"), "upper95_db": float("nan")}
    excess = float(r_on - null.mean())
    sd = float(null.std(ddof=1))
    db = lambda v: float(10 * np.log10(v)) if v > 0 else float("-inf")
    return {"n_frames": n, "excess": excess, "z": excess / sd if sd > 0 else float("nan"),
            "excess_db": db(excess), "upper95_db": db(max(excess, 0) + 1.645 * sd),
            "null_sd_db": db(sd)}


def anti_forensics(x_orig, x_clean, sr, tracks: Sequence[Track], width_hz: float = 2.0,
                   n_harmonics: int = 8, fmax: float = FMAX_WIDE,
                   span: tuple[float, float] | None = None, target_db: float = -10.0) -> dict:
    """Numeric anti-forensic checks of the cleaned audio along `tracks`, on the
    attacker's STFT grid (8192 @ 4 kHz, 0.25 s hop). Companions to the visual verdict.

    1. edge_step_db: mean signed whitened-dB step (inside - outside) across the
       corridor's frequency edges (|f - f(t)| = width_hz) in the cleaned audio, minus
       the same measured on control curves shifted +-8..14 Hz (edge_step_null_sd_db
       is their spread). ~0 = seamless; <0 = notch edge; >0 = fill too loud.
       time_edge_db: same across the corridor's ends in time (2 s in vs 2 s out).
    2. var_ratio_db / kurtosis: corridor-bin statistics (linear power) vs flanking
       bins (width..width+4 Hz). Exponential (Rayleigh-power) noise has
       var/mean^2 = 1 and excess kurtosis 6; a synthetic fill or leftover tone
       changes these. Reported for orig and clean.
    3. harmonics: for every curve (as a fundamental f(t)/harmonic), integrated
       residual at k*f(t), k=1..n_harmonics (inside fmax): excess over floor,
       z-score. Catches faint harmonics nobody detected.
    4. residual: on-curve excess power over the local per-bin floor, integrated
       over ALL frames of all curves (what an attacker summing ~1000 frames sees).
       pass_target = upper95_db <= target_db (default -10 dB).
    """
    tracks = [tr for tr in tracks if len(tr.t) >= 2]
    v = attack_views(x_orig, x_clean, sr, fmax=fmax, span=span)
    so, sc = v["orig"], v["clean"]
    t, f = so.t, so.f
    out: dict = {"width_hz": width_hz, "target_db": target_db}

    # 1. band-edge discontinuities (on whitened dB, bilinear)
    def edge(spec, shift=0.0):
        st = []
        for tt, ff in _track_points(tracks):
            for sg in (-1, 1):
                inside = spec.sample(tt, ff + shift + sg * (width_hz - spec.df))
                outside = spec.sample(tt, ff + shift + sg * (width_hz + spec.df))
                st.append(inside - outside)
        a = np.concatenate(st) if st else np.array([np.nan])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean(a))

    def tedge(spec, shift=0.0):
        st = []
        for tr in tracks:
            for t_edge, sgn in ((tr.t0, 1), (tr.t1, -1)):
                fe = float(np.interp(t_edge, tr.t, tr.f)) + shift
                ins = np.arange(t_edge, t_edge + 2 * sgn, 0.25 * sgn)
                outs = np.arange(t_edge - 0.25 * sgn, t_edge - 2.25 * sgn, -0.25 * sgn)
                a = spec.sample(ins, np.full(len(ins), fe)); b = spec.sample(outs, np.full(len(outs), fe))
                if np.isfinite(a).any() and np.isfinite(b).any():
                    st.append(np.nanmean(a) - np.nanmean(b))
        return float(np.mean(st)) if st else float("nan")

    # baseline: identical measurement on control curves shifted +-8..14 Hz (cleaned audio)
    ctrl = np.array([edge(sc, s_) for s_ in NULL_SHIFTS_HZ])
    ctrl = ctrl[np.isfinite(ctrl)]
    e_c = edge(sc)
    out["edge_step_db"] = e_c - (float(ctrl.mean()) if len(ctrl) else 0.0)
    out["edge_step_null_sd_db"] = float(ctrl.std(ddof=1)) if len(ctrl) > 2 else float("nan")
    tc = np.array([tedge(sc, s_) for s_ in NULL_SHIFTS_HZ])
    tc = tc[np.isfinite(tc)]
    out["time_edge_db"] = tedge(sc) - (float(tc.mean()) if len(tc) else 0.0)
    out["time_edge_null_sd_db"] = float(tc.std(ddof=1)) if len(tc) > 2 else float("nan")

    # 2. variance / kurtosis of corridor vs flanking bins (linear power, floor-normalised)
    def stats(raw):
        cor, fla = [], []
        for tt, ff in _track_points(tracks):
            ti = np.round((tt - t[0]) / (t[1] - t[0])).astype(int)
            okt = (ti >= 0) & (ti < raw.shape[1])
            for k, fk in zip(ti[okt], ff[okt]):
                d = np.abs(f - fk)
                col = 10 ** (raw[:, k] / 10)
                fl_m = (d > width_hz) & (d <= width_hz + 4)
                ring = (d > width_hz + 4) & (d <= width_hz + 10)  # independent normaliser
                if not fl_m.any() or not ring.any():
                    continue
                norm = np.median(col[ring]) / np.log(2)  # exponential: median = mean*ln2
                cor.append(col[d <= width_hz] / norm); fla.append(col[fl_m] / norm)
        if not cor:
            return {}
        res = {}
        for name, arr in (("corridor", np.concatenate(cor)), ("flank", np.concatenate(fla))):
            m = arr.mean(); var = arr.var()
            kurt = float(np.mean((arr - m) ** 4) / var ** 2 - 3) if var > 0 else float("nan")
            res[name] = {"mean": float(m), "cv2": float(var / m ** 2), "kurtosis": kurt}
        res["var_ratio_db"] = float(10 * np.log10(res["corridor"]["cv2"] * res["corridor"]["mean"] ** 2
                                                  / (res["flank"]["cv2"] * res["flank"]["mean"] ** 2)))
        return res
    out["stats"] = {"orig": stats(v["raw_orig"]), "clean": stats(v["raw_clean"])}

    # 3. inter-harmonic consistency
    harm = []
    seen = set()
    for tr in tracks:
        h = max(1, int(tr.harmonic))
        key = (tr.group, round(tr.t0, 1)) if tr.group >= 0 else id(tr)
        if key in seen:
            continue
        seen.add(key)
        f1 = np.asarray(tr.f, float) / h
        for k in range(1, n_harmonics + 1):
            fk = k * f1
            if np.nanmax(fk) > f[-1] - 7 or np.nanmin(fk) < f[0] + 7:
                continue
            tk = Track(np.asarray(tr.t, float), fk, np.zeros(len(fk)))
            rc = _curve_vs_floor(v["raw_clean"], t, f, [tk])
            ro = _curve_vs_floor(v["raw_orig"], t, f, [tk])
            harm.append({"f1_mean": float(np.mean(f1)), "k": k, "orig_z": ro["z"],
                         "clean_z": rc["z"], "clean_excess_db": rc["excess_db"],
                         "leftover": bool(np.isfinite(rc["z"]) and rc["z"] > 3)})
    out["harmonics"] = harm
    out["leftover_harmonics"] = [(h_["f1_mean"], h_["k"]) for h_ in harm if h_["leftover"]]

    # 4. integrated residual vs per-bin floor
    out["residual"] = {"orig": _curve_vs_floor(v["raw_orig"], t, f, tracks),
                       "clean": _curve_vs_floor(v["raw_clean"], t, f, tracks)}
    rc = out["residual"]["clean"]
    out["pass_target"] = bool(np.isfinite(rc["upper95_db"]) and rc["upper95_db"] <= target_db)
    return out
