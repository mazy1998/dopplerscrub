"""End-to-end scrub: decode -> detect Doppler tones -> remove them -> re-encode."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .types import DetectConfig, RemoveConfig, ScrubReport, Track


def scrub_array(x: np.ndarray, sr: int, detect_cfg: DetectConfig | None = None,
                remove_cfg: RemoveConfig | None = None, passes: int = 2,
                ) -> tuple[np.ndarray, list[Track]]:
    """Detect and remove Doppler tones in an in-memory signal.

    Runs up to `passes` detect/remove rounds: a strong tone can mask a weaker
    one (e.g. a second aircraft or a high harmonic), which only becomes
    detectable once the first is gone.
    """
    from .detect import detect_tracks
    from .remove import remove_tracks

    detect_cfg = detect_cfg or DetectConfig()
    remove_cfg = remove_cfg or RemoveConfig()
    y = x
    found: list[Track] = []
    for i in range(max(1, passes)):
        tracks = detect_tracks(y, sr, detect_cfg)
        targets = [t for t in tracks if t.kind in remove_cfg.remove_kinds]
        if i == 0:
            found.extend(tracks)
        else:
            found.extend(targets)
        if not targets:
            break
        y = remove_tracks(y, sr, targets, replace(remove_cfg, harden=False))
    if remove_cfg.harden:
        # one notch + noise-fill over every corridor found in any pass, so the
        # removed curves leave neither a tone nor a curve-shaped hole
        doppler = [t for t in found if t.kind in remove_cfg.remove_kinds]
        if doppler:
            y = remove_tracks(y, sr, doppler, remove_cfg)
    return y, found


def scrub(in_path: str | Path, out_path: str | Path, detect_cfg: DetectConfig | None = None,
          remove_cfg: RemoveConfig | None = None, passes: int = 2, strip_metadata: bool = True,
          report_path: str | Path | None = None) -> ScrubReport:
    from . import io

    t_start = time.time()
    x, sr, meta = io.load(in_path)
    y, tracks = scrub_array(x, sr, detect_cfg, remove_cfg, passes)
    io.save(in_path, out_path, y, sr, meta, strip_metadata=strip_metadata)
    rcfg = remove_cfg or RemoveConfig()
    removed = sum(t.kind in rcfg.remove_kinds for t in tracks)
    rep = ScrubReport(str(in_path), str(out_path), sr, len(x) / sr, tracks, removed)
    rep.notes.append(f"runtime {time.time() - t_start:.1f}s")
    if strip_metadata:
        rep.notes.append("container/stream metadata (incl. GPS location tags) stripped")
    if report_path:
        Path(report_path).write_text(json.dumps({
            "input": rep.input, "output": rep.output, "sr": rep.sr,
            "duration_s": rep.duration_s, "removed": rep.removed, "notes": rep.notes,
            "tracks": [t.to_dict() for t in tracks],
        }, indent=1))
    return rep
