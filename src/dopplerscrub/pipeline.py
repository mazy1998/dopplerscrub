"""End-to-end scrub: decode -> detect Doppler tones -> remove them -> re-encode."""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .types import DetectConfig, RemoveConfig, ScrubReport, Track


def _quiet(*_a, **_k):
    pass


def scrub_array(x: np.ndarray, sr: int, detect_cfg: DetectConfig | None = None,
                remove_cfg: RemoveConfig | None = None, passes: int = 2,
                log=_quiet, stats_out: dict | None = None,
                ) -> tuple[np.ndarray, list[Track]]:
    """Detect and remove Doppler tones in an in-memory signal.

    Runs up to `passes` detect/remove rounds: a strong tone can mask a weaker
    one (e.g. a second aircraft or a high harmonic), which only becomes
    detectable once the first is gone.
    """
    from .detect import detect_tracks
    from .remove import harden_tracks, remove_tracks

    detect_cfg = detect_cfg or DetectConfig()
    remove_cfg = remove_cfg or RemoveConfig()
    y = x
    found: list[Track] = []
    for i in range(max(1, passes)):
        t0 = time.time()
        log(f"pass {i + 1}: detecting tones ...")
        tracks = detect_tracks(y, sr, detect_cfg)
        targets = [t for t in tracks if t.kind in remove_cfg.remove_kinds]
        log(f"pass {i + 1}: {len(tracks)} tone(s), {len(targets)} to remove "
            f"(detect {time.time() - t0:.0f}s)")
        if i == 0:
            found.extend(tracks)
        else:
            found.extend(targets)
        if not targets:
            break
        t0 = time.time()
        y, st = remove_tracks(y, sr, targets, replace(remove_cfg, harden=False), return_stats=True)
        log(f"pass {i + 1}: removed ({remove_cfg.method}, {time.time() - t0:.0f}s, "
            f"{100 * st['tf_fraction_40_1000']:.1f}% of the 40-1000 Hz TF area touched)")
    if remove_cfg.harden:
        # one notch + noise-fill over every corridor found in any pass, so the
        # removed curves leave neither a tone nor a curve-shaped hole
        doppler = [t for t in found if t.kind in remove_cfg.remove_kinds]
        if doppler:
            t0 = time.time()
            log(f"noise-filling {len(doppler)} corridor(s) ...")
            y = harden_tracks(y, sr, doppler, remove_cfg, x_ref=x)
            log(f"noise fill done ({time.time() - t0:.0f}s)")
    if stats_out is not None:
        stats_out["n_doppler"] = sum(t.kind in remove_cfg.remove_kinds for t in found)
    return y, found


def _strip_notes(st: dict) -> list[str]:
    """Human-readable notes on what the metadata strip removed (only what was there)."""
    notes = []
    if st["location"]:
        notes.append("LOCATION tags stripped: " + ", ".join(st["location"]))
    if st["other"]:
        notes.append("metadata stripped: " + ", ".join(st["other"]))
    if st["chapters"]:
        notes.append(f"{st['chapters']} chapter(s) dropped")
    if st["data_streams"]:
        notes.append("data stream(s) dropped (may hold GPS telemetry): " + ", ".join(st["data_streams"]))
    return notes


def scrub(in_path: str | Path, out_path: str | Path, detect_cfg: DetectConfig | None = None,
          remove_cfg: RemoveConfig | None = None, passes: int = 2, strip_metadata: bool = True,
          report_path: str | Path | None = None, log=_quiet, return_audio: bool = False):
    """Decode -> scrub -> re-encode. Returns the ScrubReport, or
    (report, x, y, sr) with return_audio=True (lets callers plot without
    decoding the files again)."""
    from . import io

    t_start = time.time()
    log(f"decoding {in_path} ...")
    x, sr, meta = io.load(in_path)
    log(f"decoded {len(x) / sr:.0f}s, {sr} Hz, {x.shape[1]} ch"
        + (" (video: will be stream-copied)" if meta.get("has_video") else "")
        + f" ({time.time() - t_start:.0f}s)")
    y, tracks = scrub_array(x, sr, detect_cfg, remove_cfg, passes, log=log)
    t0 = time.time()
    log(f"encoding {out_path} ...")
    io.save(in_path, out_path, y, sr, meta, strip_metadata=strip_metadata)
    log(f"encoded ({time.time() - t0:.0f}s)")
    rcfg = remove_cfg or RemoveConfig()
    removed = sum(t.kind in rcfg.remove_kinds for t in tracks)
    rep = ScrubReport(str(in_path), str(out_path), sr, len(x) / sr, tracks, removed)
    rep.notes.append(f"runtime {time.time() - t_start:.1f}s")
    if strip_metadata:
        rep.notes.extend(_strip_notes(io.sensitive_tags(meta)))
    if report_path:
        Path(report_path).write_text(json.dumps({
            "input": rep.input, "output": rep.output, "sr": rep.sr,
            "duration_s": rep.duration_s, "removed": rep.removed, "notes": rep.notes,
            "tracks": [t.to_dict() for t in tracks],
        }, indent=1))
    return (rep, x, y, sr) if return_audio else rep
