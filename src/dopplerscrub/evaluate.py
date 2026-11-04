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


FIG_W = 12.0     # inches
PNG_DPI = 90
PNG_COLORS = 80


PNG_MAX_KB = 400


def save_png(fig, out_png, dpi: int | None = None, colors: int | None = None,
             max_kb: float = PNG_MAX_KB):
    """Save a figure as a small palette PNG (target <= max_kb, ~150-400 kB): render at
    `dpi`, quantise to an adaptive palette with PIL (pngquant-style); if still too
    big, step down colours, then dpi (keeps figures readable)."""
    import io as _io
    from PIL import Image
    dpi = dpi or PNG_DPI
    colors = PNG_COLORS if colors is None else colors
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    data = b""
    for d, c in [(dpi, colors), (dpi, 64), (dpi, 48), (int(dpi * 0.85), 48), (int(dpi * 0.75), 40)]:
        buf = _io.BytesIO()
        fig.savefig(buf, dpi=d, format="png")
        if not c:
            data = buf.getvalue()
            break
        buf.seek(0)
        img = Image.open(buf).convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=c)
        out = _io.BytesIO()
        img.save(out, format="png", optimize=True)
        data = out.getvalue()
        if len(data) <= max_kb * 1024:
            break
    Path(out_png).write_bytes(data)


def _imshow(ax, spec: Spectrogram, vmin, vmax, cmap):
    spec = _pool_time(spec)
    dt = spec.dt; df = spec.df
    ext = [spec.t[0] - dt / 2, spec.t[-1] + dt / 2, spec.f[0] - df / 2, spec.f[-1] + df / 2]
    return ax.imshow(spec.S, origin="lower", aspect="auto", extent=ext, vmin=vmin, vmax=vmax,
                     cmap=cmap, interpolation="antialiased", rasterized=True)


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
                        overlay_label: str = "cyan = doppler, green = other",
                        boxes: Sequence[tuple] | None = None) -> dict:
    """Stacked panels on one colour scale: original, cleaned, removed (orig-clean), and,
    when `tracks` (or annotated event `boxes` = (t0, t1, f0, f1)) are given, a separate
    copy of the original with them drawn on top.
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
    if tracks or boxes:
        lab = overlay_label + ("; white boxes = annotated events" if boxes else "")
        panels.append((f"Original + tracks (dashed: {lab})", views["orig"]))
    fig, axes = plt.subplots(len(panels), 1, figsize=(FIG_W, 2.5 * len(panels)), sharex=True,
                             constrained_layout=True)
    im = None
    for i, (ax, (name, spec)) in enumerate(zip(axes, panels)):
        im = _imshow(ax, _crop_for_window(spec, window), vmin, vmax, cmap)
        ax.set_ylabel("Hz")
        ax.set_title(name, fontsize=10, loc="left")
        if (tracks or boxes) and i == len(panels) - 1:
            _overlay(ax, tracks, window)
            for b in boxes or []:   # (t0, t1, f0, f1)
                ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                    (b[0], b[2]), b[1] - b[0], b[3] - b[2], fill=False, ec="white", lw=1.0, ls="--"))
        elif window is not None:
            ax.set_xlim(window[0], window[1])
            if len(window) == 4:
                ax.set_ylim(window[2], window[3])
    axes[-1].set_xlabel("time (s)")
    fig.colorbar(im, ax=axes, shrink=0.9, label="dB above per-bin median")
    fig.suptitle(title, fontsize=12)
    save_png(fig, out_png)
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
    fig, axs = plt.subplots(nrow, 1, figsize=(FIG_W, 2.3 * nrow), constrained_layout=True)
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
    save_png(fig, out_png)
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


def box_stats(spec: Spectrogram, box: tuple, hi_db: float = 8.0) -> dict:
    """Ridge content of an annotated event box (t0, t1, f0, f1) without tracks:
    p99 of whitened dB and the fraction of cells above `hi_db`. Compare orig vs clean;
    a successful scrub pulls both toward the noise values (p99 ~ 7 dB, frac ~ 0.5%)."""
    c = spec.crop(box[0], box[1], box[2], box[3])
    if c.S.size == 0:
        return {"p99_db": float("nan"), "frac_hi": float("nan")}
    return {"p99_db": float(np.percentile(c.S, 99)), "frac_hi": float(np.mean(c.S > hi_db))}


# ------------------------------------------------------------------ baseline-comparison metrics

def ridge_excess_db(spec: Spectrogram, tracks: Sequence[Track], half_hz: float = 8.0,
                    ctrl_shifts=(-30.0, -24.0, 24.0, 30.0)) -> float:
    """Shift-TOLERANT curve visibility: per frame, the max whitened dB within +-half_hz of
    f(t), averaged over the pass, minus the same on control curves shifted 24-30 Hz.
    Unlike track_contrast_db this still sees a ridge that was wobbled or pitch-shifted
    by a few Hz (e.g. the jitter baseline). ~0 = no ridge near the curve."""
    offs = np.arange(-half_hz, half_hz + 1e-9, spec.df)

    def band_max(shift):
        vals = []
        for tt, ff in _track_points(tracks):
            m = np.stack([spec.sample(tt, ff + shift + o) for o in offs])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanmax(m, axis=0)
            vals.append(v[np.isfinite(v)])
        v = np.concatenate(vals) if vals else np.array([])
        return float(np.mean(v)) if len(v) else float("nan")

    on = band_max(0.0)
    ctrl = [band_max(s) for s in ctrl_shifts]
    ctrl = [c for c in ctrl if np.isfinite(c)]
    return on - float(np.mean(ctrl)) if ctrl and np.isfinite(on) else float("nan")


def _stft_power(x, sr, win_s=0.046):
    n = int(2 ** np.round(np.log2(win_s * sr)))
    f, t, Z = ss.stft(x, sr, nperseg=n, noverlap=n // 2)
    return f, t, np.abs(Z) ** 2


def log_spectral_distance(x_orig, x_clean, sr, fmin: float = 0.0, fmax: float | None = None,
                          dyn_db: float = 80.0) -> float:
    """Mean over frames of the RMS (over frequency) dB difference between the ~46 ms power
    spectra of original and cleaned, floored at (global max - dyn_db). 0 = identical.
    Tolerant to ms-scale time shifts (unlike waveform SNR)."""
    a, b = _align(_mono(x_orig), _mono(x_clean))
    f, _, Pa = _stft_power(a, sr)
    _, _, Pb = _stft_power(b, sr)
    band = (f >= fmin) & (f <= (fmax or sr / 2))
    floor = max(Pa.max(), 1e-30) * 10 ** (-dyn_db / 10)
    La = 10 * np.log10(np.maximum(Pa[band], floor)); Lb = 10 * np.log10(np.maximum(Pb[band], floor))
    return float(np.mean(np.sqrt(np.mean((La - Lb) ** 2, axis=0))))


def stoi_like(x_orig, x_clean, sr) -> float:
    """STOI-style intelligibility proxy, implemented from Taal et al., IEEE TASLP 19(7),
    2011 (pystoi/PESQ/ViSQOL are not dependencies): 10 kHz, 256-pt Hann frames with 50%
    overlap, silent frames (>40 dB below the loudest) dropped, 15 one-third-octave bands
    from 150 Hz, 30-frame (384 ms) segments, clean-normalised and clipped (beta = -15 dB)
    envelopes, mean correlation. 1 = identical, about 0.4-0.5 = heavily degraded. It
    measures speech-envelope preservation, not tonal fidelity."""
    fs = 10000
    a, b = _align(_resample(_mono(x_orig), sr, fs).astype(np.float64),
                  _resample(_mono(x_clean), sr, fs).astype(np.float64))
    N, hop = 256, 128
    w = np.hanning(N + 2)[1:-1]
    nfr = (len(a) - N) // hop + 1
    if nfr < 30:
        return float("nan")
    idx = np.arange(nfr)[:, None] * hop + np.arange(N)[None]
    fa, fb = a[idx] * w, b[idx] * w
    en = 20 * np.log10(np.linalg.norm(fa, axis=1) + 1e-20)
    keep = en > en.max() - 40
    fa, fb = fa[keep], fb[keep]
    if len(fa) < 30:
        return float("nan")
    nfft = 512
    Xa = np.abs(np.fft.rfft(fa, nfft, axis=1)) ** 2
    Xb = np.abs(np.fft.rfft(fb, nfft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    cf = 150 * 2 ** (np.arange(15) / 3)
    A = np.zeros((15, len(freqs)))
    for i, c in enumerate(cf):
        lo, hi = c * 2 ** (-1 / 6), c * 2 ** (1 / 6)
        A[i, (freqs >= lo) & (freqs < hi)] = 1
    Ea = np.sqrt(Xa @ A.T).T; Eb = np.sqrt(Xb @ A.T).T          # (15, frames)
    L, beta = 30, 10 ** (15 / 20)
    ds = []
    for m in range(L, Ea.shape[1] + 1):
        sa, sb = Ea[:, m - L:m], Eb[:, m - L:m]
        alpha = np.linalg.norm(sa, axis=1, keepdims=True) / (np.linalg.norm(sb, axis=1, keepdims=True) + 1e-20)
        sbn = np.minimum(sb * alpha, sa * (1 + beta))
        sa0 = sa - sa.mean(1, keepdims=True); sb0 = sbn - sbn.mean(1, keepdims=True)
        num = (sa0 * sb0).sum(1)
        den = np.linalg.norm(sa0, axis=1) * np.linalg.norm(sb0, axis=1) + 1e-20
        ds.append(num / den)
    return float(np.mean(ds))


def tf_modified_fraction(x_orig, x_clean, sr, thresh_db: float = -20.0, dyn_db: float = 60.0,
                         bands=((0.0, None), (40.0, 450.0), (450.0, None))) -> dict:
    """Share of the (~46 ms) time-frequency plane that was modified: cells where
    |orig - clean|^2 > 10^(thresh_db/10) * |orig|^2, among cells within dyn_db of the
    loudest cell (silence ignored). Returned per band {"lo-hi": fraction}."""
    a, b = _align(_mono(x_orig), _mono(x_clean))
    f, _, Pa = _stft_power(a, sr)
    _, _, Pd = _stft_power(a - b, sr)
    floor = Pa.max() * 10 ** (-dyn_db / 10)
    act = Pa > floor
    mod = Pd > 10 ** (thresh_db / 10) * Pa
    out = {}
    for lo, hi in bands:
        hi_ = hi or sr / 2
        m = (f >= lo) & (f <= hi_)
        den = act[m].sum()
        out[f"{lo:.0f}-{hi_:.0f}"] = float((mod[m] & act[m]).sum() / den) if den else float("nan")
    return out


def fidelity_boxes(x_orig, x_clean, sr, boxes: Sequence[tuple]) -> float:
    """SNR (dB) orig vs orig-clean OUTSIDE annotated event boxes (t0, t1, f0, f1)."""
    a, b = _align(_mono(x_orig).astype(np.float64), _mono(x_clean).astype(np.float64))
    f, t, Pa = _stft_power(a, sr)
    _, _, Pd = _stft_power(a - b, sr)
    keep = np.ones(Pa.shape, bool)
    for t0, t1, f0, f1 in boxes:
        keep[np.ix_((f >= f0) & (f <= f1), (t >= t0) & (t <= t1))] = False
    return _snr(float(Pa[keep].sum()), float(Pd[keep].sum()))


# --------------------------------------------------------------------------- dopfast attacker
#
# A matched-filter attacker after the "dopfast" pipeline (Doppler geolocation paper, sec. 3.2
# and 4, Listing 3). It does not look for visible ridges: it correlates a whitened
# log-frequency spectrogram S(s, l) with a bank of exact straight-pass templates
# f0 * g(t - h; v, d) carrying a harmonic comb (offsets log j), maximises over source pitch and
# robustly standardises over time, giving the evidence volume E[v, d, s]. Its peaks are what
# the attacker geolocates from. Because it integrates every frame of the curve and every
# harmonic together, faint leftover plateaus and harmonics add up.
#
# Choices the paper leaves open (documented here; chosen on DEV clips for the strongest
# attacker, see docs/results/dopfast.md): 4 kHz mono, 0.512 s Hann STFT with 0.125 s hop
# (shorter than the stock 2 s view, to keep the steep CPA part sharp); a Gaussian temporal
# weight b(w) with sigma = d / v (the paper's b; the Fisher appendix puts the shape
# information at the steep CPA part, while long plateaus also match the nulls); per-bin
# median whitening over the analysed span; S = positive excess dB clipped at DF_CLIP_DB
# (a loud transient cannot dominate); log grid of DF_DL; Gaussian ridge profile of DF_SIGMA
# (log units); harmonic weights 1/j for j = 1..DF_HARMONICS; templates zero-mean over their
# bounding box (flat broadband energy scores ~0) and unit L2 norm; a placement is valid
# only if the fundamental's whole curve lies inside the band. Calibration is eq. (18)
# (median / 1.4826 MAD over time per shape).

DF_C = 343.0
DF_HOP_S = 0.125
DF_NPERSEG = 2048           # 0.512 s Hann (1.95 Hz bins)
DF_DL = 0.005
DF_SIGMA = 0.012
DF_HARMONICS = 4
DF_CLIP_DB = 20.0
DF_V = (30.0, 45.0, 70.0, 100.0, 140.0, 200.0, 280.0)
DF_D = (40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0)
DF_POLARITY = "pos"        # S from the positive excess ("pos"), the deficit ("neg": an attacker
                           # hunting curve-shaped NOTCHES) or the absolute deviation ("abs")
DF_NULLS = 6               # frame-shuffled copies (plus one time-reversed copy)
DF_ZERO_MEAN = True        # template zero-mean over its box (False: nonnegative, unit total weight)
DF_MAX_HALF_S = 40.0       # template temporal support cap (+-)
DF_CORE = 1.0              # b(w): Gaussian temporal weight, sigma = DF_CORE * d / v (None = flat);
                           # emphasises the steep CPA part, where the shape information is
DF_CORE_MIN_S = 2.0        # b(w) sigma >= this: an aircraft tone persists for seconds around
                           # CPA (a bird chirp or a short glide does not)
DF_CORE_IN = 2.0           # a placement is valid only if +-DF_CORE_IN * sigma_w lies inside the file


def dopfast_g(w: np.ndarray, v: float, d: float, c: float = DF_C) -> np.ndarray:
    """Straight-pass received-frequency ratio g(w; v, d) at received time w = t - h
    (paper eqs. 6 and 8; g(0) = 1, falls through the pass)."""
    beta = v / c
    z = w + d / c
    u = (z - np.sqrt(beta ** 2 * z ** 2 + (1 - beta ** 2) * (d / c) ** 2)) / (1 - beta ** 2)
    return c / (c + v ** 2 * u / np.sqrt(d ** 2 + v ** 2 * u ** 2))


def _df_spectrogram(x: np.ndarray, sr: int, fmin: float, fmax: float):
    """Whitened log-frequency spectrogram S (T, Nl) >= 0, its times and log-frequency grid."""
    a = _attack_signal(x, sr)
    hop = int(round(DF_HOP_S * ATTACK_SR))
    f, t, Z = ss.stft(a, ATTACK_SR, nperseg=DF_NPERSEG, noverlap=DF_NPERSEG - hop, window="hann",
                      boundary=None, padded=False)
    L = 10 * np.log10(np.abs(Z) ** 2 + 1e-20)
    L = L - np.median(L, axis=1, keepdims=True)
    band = (f >= fmin) & (f <= fmax)
    L = L - np.median(L[band], axis=0, keepdims=True)   # per-frame: removes broadband gusts/roar
    lg = np.arange(np.log(fmin), np.log(fmax), DF_DL)
    S = np.empty((L.shape[1], len(lg)), np.float32)
    ff = np.exp(lg)
    for k in range(L.shape[1]):
        S[k] = np.interp(ff, f, L[:, k])
    if DF_POLARITY == "neg":
        S = -S
    elif DF_POLARITY == "abs":
        S = np.abs(S)
    return np.clip(S, 0.0, DF_CLIP_DB), t, lg


def _df_template(v: float, d: float, sigma_bins: float):
    """Template T (Kt, Kl), zero-mean over its box, unit norm; anchor (a0, b0) = the
    (w = 0, fundamental pitch) cell; and the fundamental's log-pitch excursion (lo, hi)."""
    half = float(np.clip(4.0 * d / v, 6.0, DF_MAX_HALF_S))
    w = np.arange(-half, half + 1e-9, DF_HOP_S)
    lgw = np.log(dopfast_g(w, v, d))
    lo, hi = float(lgw.min()), float(lgw.max())
    pad = 3.0 * sigma_bins * DF_DL
    top = hi + np.log(DF_HARMONICS) + pad
    bot = lo - pad
    nl = int(np.ceil((top - bot) / DF_DL)) + 1
    grid = bot + np.arange(nl) * DF_DL
    T = np.zeros((len(w), nl))
    for j in range(1, DF_HARMONICS + 1):
        T += (1.0 / j) * np.exp(-0.5 * ((grid[None, :] - lgw[:, None] - np.log(j)) / (sigma_bins * DF_DL)) ** 2)
    if DF_CORE:
        T *= np.exp(-0.5 * (w / max(DF_CORE * d / v, DF_CORE_MIN_S)) ** 2)[:, None]
    if DF_ZERO_MEAN:
        T -= T.mean()
        T /= np.linalg.norm(T) + 1e-12
    else:
        T /= T.sum() + 1e-12
    a0 = int(np.argmin(np.abs(w)))
    b0 = int(round(-bot / DF_DL))                 # fundamental log-pitch 0 (g = 1) row
    return T, a0, b0, lo, hi


_DF_TEMPLATES: dict = {}


def _df_bank():
    key = (DF_V, DF_D, DF_DL, DF_SIGMA, DF_HARMONICS, DF_HOP_S, DF_ZERO_MEAN, DF_MAX_HALF_S, DF_CORE,
           DF_CORE_MIN_S)
    if key not in _DF_TEMPLATES:
        _DF_TEMPLATES.clear()
        _DF_TEMPLATES[key] = [(v, d, *_df_template(v, d, DF_SIGMA / DF_DL)) for v in DF_V for d in DF_D]
    return _DF_TEMPLATES[key]


def dopfast_evidence(S: np.ndarray, lg: np.ndarray):
    """Evidence volume E (Nv, Nd, T) and the maximising pitch index (Nv, Nd, T) for a
    whitened log-frequency spectrogram S (T, Nl). Placements whose fundamental curve would
    leave the band are invalid (NaN before calibration, -inf after)."""
    bank = _df_bank()
    Tn, Nl = S.shape
    Kt = max(b[2].shape[0] for b in bank)
    Kl = max(b[2].shape[1] for b in bank)
    shape = (Tn + Kt - 1, Nl + Kl - 1)
    FS = np.fft.rfft2(S.astype(np.float64), s=shape)
    E = np.full((len(DF_V), len(DF_D), Tn), -np.inf)
    P = np.zeros((len(DF_V), len(DF_D), Tn), int)
    for v, d, T, a0, b0, lo, hi in bank:
        iv, idd = DF_V.index(v), DF_D.index(d)
        C = np.fft.irfft2(FS * np.conj(np.fft.rfft2(T, s=shape)), s=shape)
        # C[lt, ll] = sum S[a + lt, b + ll] T[a, b] for non-negative lags; event time index
        # = lt + a0, fundamental pitch index = ll + b0
        lt = np.arange(Tn) - a0
        ll = np.arange(Nl) - b0
        ok_l = (np.arange(Nl) + lo / DF_DL >= 0) & (np.arange(Nl) + hi / DF_DL <= Nl - 1)
        if not ok_l.any():
            continue
        Cm = C[np.ix_(lt % shape[0], ll % shape[1])]
        Cm[:, ~ok_l] = -np.inf
        if DF_CORE:
            k = int(np.ceil(DF_CORE_IN * max(DF_CORE * d / v, DF_CORE_MIN_S) / DF_HOP_S))
            ev_t = np.arange(Tn)
            ok_t = (ev_t - k >= 0) & (ev_t + k <= Tn - 1)
        else:
            ok_t = (lt >= -(T.shape[0] - 1) // 2) & (lt + T.shape[0] - 1 <= Tn - 1 + (T.shape[0] - 1) // 2)
        R = Cm.max(axis=1)
        R[~ok_t] = np.nan
        P[iv, idd] = Cm.argmax(axis=1)
        good = np.isfinite(R)
        if good.sum() < 8:
            continue
        med = np.median(R[good])
        mad = 1.4826 * np.median(np.abs(R[good] - med))
        E[iv, idd] = np.where(good, (R - med) / max(mad, 1e-6), -np.inf)
    return E, P


def _df_peak(E):
    i = np.unravel_index(int(np.argmax(E)), E.shape)
    return float(E[i]), i


# ---- events (the headline): how many USABLE pass events remain per recording.
# Paper sec. 7 / 12.1 / App. C: with an unknown recording clock the attacker needs >= 2 usable
# events to intersect; with a known or guessable clock ONE usable event localises to about two
# mirror points. Thresholds below are calibrated on null clips and synthetic passes only
# (scripts/dopfast_calibrate.py, docs/results/dopfast.md) and then frozen.

DF_EVENT_TAU = 12.0        # E peak (robust z) for an event. Frozen from dopfast_calibrate.py:
                           # 0.29 false events/min on all null material (0.14/min on time-reversed or
                           # frame-shuffled dev clips; the rest are synth birds and road traffic),
                           # 9/14 synth_hard straight passes detected, plateau-only passes 0/6 usable
DF_EVENT_SEP_S = 5.0       # peaks closer than this (s) are one event (paper sec. 6.3 grouping)
DF_EVENT_WIN_S = 1.0       # (v, d) profile of an event: max of E within +-this of its time
DF_USE_MARGIN = 0.25       # near-peak region: E >= peak * (1 - margin)
DF_USE_DSPAN = 2           # usable if that region spans <= this many d-grid steps (x4 in d)
                           # and does not touch the largest d (unbounded above)


def dopfast_volume(x: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = 1000.0):
    """(E, P, t, lg) for one signal: evidence volume, maximising pitch index, times, log grid."""
    S, t, lg = _df_spectrogram(_mono(x).astype(np.float64), sr, fmin, fmax)
    E, P = dopfast_evidence(S, lg)
    return E, P, t, lg


def dopfast_event_list(E, P, t, lg, tau: float | None = None, sep_s: float = DF_EVENT_SEP_S,
                       margin: float = DF_USE_MARGIN, dspan: int = DF_USE_DSPAN) -> list[dict]:
    """Events = peaks of max_{v,d} E[v, d, s] over time, >= tau, grouped by non-maximum
    suppression within sep_s. Each event: time s, best (v, d, f0), peak E, and whether it is
    USABLE: the (v, d) profile (max of E within +-DF_EVENT_WIN_S) has its near-peak region
    (E >= peak * (1 - margin)) spanning <= dspan d-grid steps and not reaching the largest d.
    A plateau-only leftover matches every distance about equally: detected, not usable."""
    tau = DF_EVENT_TAU if tau is None else tau
    Nv, Nd, T = E.shape
    flat = E.reshape(Nv * Nd, T)
    em = flat.max(0)
    arg = flat.argmax(0)
    dt = t[1] - t[0] if len(t) > 1 else DF_HOP_S
    order = np.argsort(-em)
    events, taken = [], []
    for k in order:
        if not np.isfinite(em[k]) or em[k] < tau:
            break
        if any(abs(t[k] - t[j]) < sep_s for j in taken):
            continue
        taken.append(k)
        iv, idd = divmod(int(arg[k]), Nd)
        w = max(1, int(round(DF_EVENT_WIN_S / dt)))
        prof = E[:, :, max(0, k - w):k + w + 1].max(axis=2)            # (Nv, Nd)
        near = prof >= em[k] * (1 - margin)
        dn = np.flatnonzero(near.any(axis=0))
        vn = np.flatnonzero(near.any(axis=1))
        d_span = int(dn.max() - dn.min()) if len(dn) else Nd
        usable = bool(d_span <= dspan and dn.max() < Nd - 1)
        events.append({"s": float(t[k]), "E": float(em[k]), "v": DF_V[iv], "d": DF_D[idd],
                       "f0": float(np.exp(lg[P[iv, idd, k]])), "d_span": d_span,
                       "d_range": [DF_D[dn.min()], DF_D[dn.max()]] if len(dn) else None,
                       "v_range": [DF_V[vn.min()], DF_V[vn.max()]] if len(vn) else None,
                       "usable": usable})
    return sorted(events, key=lambda e: e["s"])


def dopfast_file(x: np.ndarray, sr: int, fmin: float = FMIN, fmax: float = 1000.0,
                 nulls: int = 0, seed: int = 0) -> dict:
    """dopfast on one signal: peak evidence, its event (v, d, s, f0) and the null
    calibration (time-reversed + `nulls` frame-shuffled copies of the same spectrogram).
    Keys starting with '_' hold arrays for dopfast(); drop them before serialising."""
    rng = np.random.default_rng(seed)
    S, t, lg = _df_spectrogram(_mono(x).astype(np.float64), sr, fmin, fmax)
    E, P = dopfast_evidence(S, lg)
    pk, (iv, idd, it) = _df_peak(E)
    if nulls > 0:
        null = [_df_peak(dopfast_evidence(S[::-1], lg)[0])[0]]
        for _ in range(nulls):
            null.append(_df_peak(dopfast_evidence(S[rng.permutation(S.shape[0])], lg)[0])[0])
        null = np.array(null)
        mu = float(null.mean())
        sd = float(null.std(ddof=1)) if len(null) > 1 else float("nan")
    else:
        mu = sd = float("nan")
    ev_list = dopfast_event_list(E, P, t, lg)
    return {"peak": pk, "null_mean": mu, "null_sd": sd, "events": ev_list,
            "n_events": len(ev_list), "n_usable": sum(e["usable"] for e in ev_list),
            "z": (pk - mu) / sd if sd and sd > 0 else float("nan"),
            "event": {"v": DF_V[iv], "d": DF_D[idd], "s": float(t[it]),
                      "f0": float(np.exp(lg[P[iv, idd, it]]))},
            "_E": E, "_t": t, "_idx": (iv, idd, it)}


def dopfast(x_orig: np.ndarray, x_clean: np.ndarray, sr: int, fmin: float = FMIN,
            fmax: float = 1000.0, nulls: int = 0, seed: int = 0,
            orig: dict | None = None) -> dict:
    """dopfast attacker before vs after cleaning (JSON-ready dict).

    peak_* : max of E[v, d, s] (robust z units, max over shapes, times and pitches).
    z_*    : (peak - null mean) / null sd, the null being the same peak statistic on the
             same file's time-reversed spectrogram (falling templates cannot match rising
             curves) and frame-shuffled copies (per-frame spectra kept, curves destroyed).
             z > 3: the evidence peak stands out from the file's own non-Doppler structure.
    at_orig_clean / z_at_orig_clean: E of the CLEANED file at the original's peak shape,
             within +-1 s of its event time, and that value against the cleaned file's null
             of PEAKS (conservative): does the event the attacker would geolocate survive?
    `orig` = a previous dopfast_file(x_orig, ...) result to reuse (same fmin/fmax/nulls).

    HEADLINE (events; frozen thresholds DF_EVENT_* / DF_USE_*): n_events_* and n_usable_*
    before/after, the event lists, pass_unknown_clock (<= 1 usable event left: no
    intersection; None = n/a, the original had < 2) and pass_known_clock (0 usable left;
    None = n/a, the original had none). With nulls=0 (default) the per-file
    null z is skipped (nan) and only the events are computed.
    """
    a, b = _align(_mono(x_orig).astype(np.float64), _mono(x_clean).astype(np.float64))
    o = orig if orig is not None else dopfast_file(a, sr, fmin, fmax, nulls, seed)
    c = dopfast_file(b, sr, fmin, fmax, nulls, seed)
    out = {"fmin": fmin, "fmax": fmax}
    for key, r in (("orig", o), ("clean", c)):
        out.update({f"peak_{key}": r["peak"], f"null_mean_{key}": r["null_mean"],
                    f"null_sd_{key}": r["null_sd"], f"z_{key}": r["z"], f"event_{key}": r["event"],
                    f"n_events_{key}": r["n_events"], f"n_usable_{key}": r["n_usable"],
                    f"events_{key}": r["events"]})
    # a criterion only applies if the ORIGINAL failed it (else "pass" says nothing): None = n/a
    out["pass_unknown_clock"] = bool(c["n_usable"] <= 1) if o["n_usable"] >= 2 else None
    out["pass_known_clock"] = bool(c["n_usable"] == 0) if o["n_usable"] >= 1 else None
    iv, idd, it = o["_idx"]
    win = np.abs(c["_t"] - o["_t"][it]) <= 1.0
    out["at_orig_clean"] = float(np.max(c["_E"][iv, idd][win])) if win.any() else float("nan")
    mu, sd = c["null_mean"], c["null_sd"]
    out["z_at_orig_clean"] = (out["at_orig_clean"] - mu) / sd if sd and sd > 0 else float("nan")
    return out
