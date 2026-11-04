"""Regenerate the README figures in assets/.

Each figure shows the attacker's view (whitened spectrogram, see README) before and
after cleaning. Real clips are not redistributed: fetch them first (see README,
"Reproduce the figures"); clips that are missing are skipped.

    uv run python scripts/make_figures.py            # all figures
    uv run python scripts/make_figures.py kingair_b350 c150_lowpass
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from dopplerscrub import io
from dopplerscrub.evaluate import attack_spectrogram
from dopplerscrub.pipeline import scrub_array

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
CACHE = ROOT / "data" / "out" / "figures"

# name -> (path, label, split). "test" = held-out clips, never used for tuning.
CLIPS = {
    "kingair_b350": (
        "data/real/heldout2/kingair_b350.wav",
        "Beechcraft King Air B350 (twin turboprop)",
        "test",
    ),
    "c172_takeoff_talking": (
        "data/real/heldout2/c172_takeoff_talking.wav",
        "Cessna 172 takeoff, people talking",
        "test",
    ),
    "c150_lowpass": ("data/real/heldout2/c150_lowpass.wav", "Cessna 150 low pass", "test"),
    "baron_flyby": (
        "data/real/heldout2/baron_flyby.wav",
        "Beechcraft Baron B55 (twin piston)",
        "test",
    ),
    "pc12_lowpass": (
        "data/real/heldout2/pc12_lowpass.wav",
        "Pilatus PC-12 (single turboprop)",
        "test",
    ),
    "c182p_overhead": ("data/real/props_dev/c182p_overhead.wav", "Cessna 182P overhead", "dev"),
    "caravan_flyby": ("data/real/props_dev/caravan_flyby.wav", "Cessna Caravan flyby", "dev"),
    "seneca_v_flyby": ("data/real/props_dev/seneca_v_flyby.wav", "Piper Seneca V flyby", "dev"),
    "pa28_cherokee_flyby": (
        "data/real/props_dev/pa28_cherokee_flyby.wav",
        "Piper Cherokee flyby",
        "dev",
    ),
}
HERO = ["kingair_b350", "c172_takeoff_talking", "c150_lowpass", "baron_flyby"]
FMIN, FMAX = 40.0, 1000.0
VMIN, VMAX = -3.0, 15.0
CMAP = "magma"


def cleaned(name: str) -> tuple[np.ndarray, np.ndarray, int] | None:
    src = ROOT / CLIPS[name][0]
    if not src.exists():
        print(f"skip {name}: {src.relative_to(ROOT)} not found", file=sys.stderr)
        return None
    x, sr, _ = io.load(src)
    cache = CACHE / f"{name}.clean.wav"
    if cache.exists():
        y, _ = sf.read(cache, always_2d=True)
    else:
        print(f"cleaning {name} ...", file=sys.stderr, flush=True)
        y, _ = scrub_array(x, sr)
        CACHE.mkdir(parents=True, exist_ok=True)
        sf.write(cache, y, sr, subtype="FLOAT")
    return x, y, sr


def _show(ax, x, sr, fmin=FMIN, fmax=FMAX):
    s = attack_spectrogram(x, sr, fmin=fmin, fmax=fmax)
    dt, df = s.t[1] - s.t[0], s.f[1] - s.f[0]
    ext = [s.t[0] - dt / 2, s.t[-1] + dt / 2, s.f[0] - df / 2, s.f[-1] + df / 2]
    return ax.imshow(
        s.S,
        origin="lower",
        aspect="auto",
        extent=ext,
        vmin=VMIN,
        vmax=VMAX,
        cmap=CMAP,
        interpolation="antialiased",
        rasterized=True,
    )


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}", file=sys.stderr)


def figure_clip(name: str) -> None:
    r = cleaned(name)
    if r is None:
        return
    x, y, sr = r
    _, label, split = CLIPS[name]
    fig, axs = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, constrained_layout=True)
    for ax, sig, title in [(axs[0], x, "Before"), (axs[1], y, "After dopplerscrub")]:
        im = _show(ax, sig, sr)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Hz")
    axs[1].set_xlabel("time (s)")
    tag = "held-out test clip (never tuned on)" if split == "test" else "development clip"
    fig.suptitle(f"{label} - {tag}", fontsize=11)
    fig.colorbar(im, ax=axs, label="dB above per-frequency median", shrink=0.9)
    _save(fig, OUT / f"{name}.png")


def figure_hero() -> None:
    rows = [(n, cleaned(n)) for n in HERO]
    rows = [(n, r) for n, r in rows if r is not None]
    if not rows:
        return
    fig, axs = plt.subplots(
        2, len(rows), figsize=(3.4 * len(rows), 5.0), constrained_layout=True, squeeze=False
    )
    for j, (n, (x, y, sr)) in enumerate(rows):
        for i, sig in enumerate((x, y)):
            im = _show(axs[i, j], sig, sr)
            axs[i, j].set_xlabel("time (s)" if i == 1 else "")
        axs[0, j].set_title(CLIPS[n][1].split(" (")[0], fontsize=10)
    axs[0, 0].set_ylabel("BEFORE\nHz", fontsize=10)
    axs[1, 0].set_ylabel("AFTER\nHz", fontsize=10)
    fig.colorbar(im, ax=axs, label="dB above per-frequency median", shrink=0.85)
    _save(fig, OUT / "hero.png")


def figure_speech() -> None:
    """Plain (unwhitened) 0-4 kHz spectrogram of the talking clip: speech left alone."""
    r = cleaned("c172_takeoff_talking")
    if r is None:
        return
    x, y, sr = r
    fig, axs = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    kw = dict(fs=sr, nperseg=2048, noverlap=1536)
    import scipy.signal as ss

    ims = []
    for ax, sig, title in [
        (axs[0], x, "Before"),
        (axs[1], y, "After"),
        (axs[2], x - y[: len(x)], "Removed (before - after)"),
    ]:
        f, t, S = ss.spectrogram(sig.mean(axis=1), **kw)
        keep = f <= 4000
        ims.append(
            ax.pcolormesh(
                t,
                f[keep],
                10 * np.log10(S[keep] + 1e-12),
                shading="auto",
                cmap=CMAP,
                vmin=-110,
                vmax=-40,
                rasterized=True,
            )
        )
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Hz")
    axs[2].set_xlabel("time (s)")
    fig.suptitle(
        "Cessna 172 with people talking: what changed, 0-4 kHz (ordinary spectrogram)", fontsize=11
    )
    fig.colorbar(ims[0], ax=axs, label="power (dB)", shrink=0.9)
    _save(fig, OUT / "c172_speech_band.png")


def main(argv: list[str]) -> int:
    names = argv or list(CLIPS)
    for n in names:
        figure_clip(n)
    if not argv:
        figure_hero()
        figure_speech()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
