"""Shared data types. Every module talks through these; do not change field
names without updating docs/CONTRACTS.md and telling the orchestrator."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """One narrowband tone followed through time (a ridge in the spectrogram).

    t, f, amp are sampled on the same time grid (seconds from file start).
    f is the instantaneous frequency in Hz; amp is the estimated linear
    magnitude of the tone (same units as the input signal, i.e. the peak
    amplitude of a sinusoid), or zeros if unknown.
    """
    t: np.ndarray
    f: np.ndarray
    amp: np.ndarray
    doppler_score: float = 0.0          # 0..1, how Doppler-flyover-like the sweep is
    fit: dict | None = None             # ideal-pass fit: {f0, v, d_cpa, t_cpa, rms_hz}
    harmonic: int = 1                   # 1 = fundamental, k = k-th harmonic of `group`
    group: int = -1                     # tracks from the same source share a group id
    kind: str = "unknown"               # "doppler" | "stationary" | "unknown"
    width_hz: float | None = None       # measured tone half-width (Hz); None = RemoveConfig.width_hz

    @property
    def t0(self) -> float:
        return float(self.t[0])

    @property
    def t1(self) -> float:
        return float(self.t[-1])

    def to_dict(self) -> dict:
        return {
            "t": self.t.tolist(), "f": self.f.tolist(), "amp": self.amp.tolist(),
            "doppler_score": self.doppler_score, "fit": self.fit,
            "harmonic": self.harmonic, "group": self.group, "kind": self.kind,
            "width_hz": self.width_hz,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        return cls(np.asarray(d["t"], float), np.asarray(d["f"], float),
                   np.asarray(d.get("amp", np.zeros(len(d["t"]))), float),
                   d.get("doppler_score", 0.0), d.get("fit"), d.get("harmonic", 1),
                   d.get("group", -1), d.get("kind", "unknown"), d.get("width_hz"))


@dataclass
class DetectConfig:
    fmin: float = 30.0          # Hz, lower edge of the searched band
    fmax: float = 2000.0        # Hz, upper edge (prop blade-pass + harmonics up to ~1.9 kHz on close passes)
    work_sr: int = 4000         # analysis sample rate for the low band
    nperseg: int = 8192         # STFT length at work_sr (~0.49 Hz bins)
    hop_s: float = 0.25         # STFT hop
    min_duration_s: float = 4.0 # ignore ridges shorter than this
    min_sweep_hz: float = 0.5   # ignore ridges whose total frequency change is below this
    snr_db: float = 3.0         # ridge must stand this far above the whitened floor
    doppler_threshold: float = 0.5  # tracks with doppler_score >= this are kind="doppler"


@dataclass
class RemoveConfig:
    method: str = "vk"          # "vk" (joint Vold-Kalman, default) | "subtract" (track-locked cancellation) | "mask" (STFT notch)
    width_hz: float = 2.0       # notch half-width around the track for the mask method
    pad_s: float = 2.0          # extend each track this far before/after its detected span
    harden: bool = True         # notch + noise-fill each corridor so no curve-shaped notch remains
    remove_kinds: tuple[str, ...] = ("doppler",)
    harden_min_score: float = 0.5  # noise-fill only tracks the detector confirmed (doppler_score >= this)


@dataclass
class ScrubReport:
    input: str
    output: str
    sr: int
    duration_s: float
    tracks: list[Track] = field(default_factory=list)
    removed: int = 0
    notes: list[str] = field(default_factory=list)
