#!/usr/bin/env python
"""Benchmark dopplerscrub on the synthetic set (with truth) and the real clips.

For each clip: run dopplerscrub.pipeline.scrub (defaults, harden=True), then render
before/after attacker-view spectrograms + red-team views, and compute cheap numbers
(no dopplertrack fit/search). Writes docs/results/img/*.png, docs/results/benchmark.json
and docs/results/benchmark.md.

The "visual verdict" column comes from docs/results/verdicts.json ({clip: text}),
which a human (or agent) writes after LOOKING at the PNGs; re-run with --md-only to
refresh the table without recomputing.

Usage:
  uv run python scripts/benchmark.py                 # everything
  uv run python scripts/benchmark.py --only speech f1
  uv run python scripts/benchmark.py --no-colsto     # skip the 1-hour clip
  uv run python scripts/benchmark.py --md-only
  uv run python scripts/benchmark.py --split heldout --blind   # no numbers on stdout
  uv run python scripts/benchmark.py --split heldout2 --blind  # fresh blind set (data/real/heldout2)
  uv run python scripts/benchmark.py --boxfree-from .cache/bench_dev  # add box-free checks
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dopplerscrub import evaluate as ev  # noqa: E402
from dopplerscrub.types import RemoveConfig, Track  # noqa: E402
import boxfree as bf  # noqa: E402

RES = ROOT / "docs" / "results"
IMG = RES / "img"


# ------------------------------------------------------------------ helpers

def load(path):
    try:
        from dopplerscrub import io
        x, sr, _ = io.load(str(path))
        return np.asarray(x, np.float32), int(sr)
    except Exception:  # io not ready -> plain WAV
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32")
        return x, sr


def run_scrub(in_path: Path, out_path: Path):
    """pipeline.scrub (orchestrator) with harden=True; fallback: detect -> remove."""
    cfg = RemoveConfig(harden=True)
    t0 = time.time()
    try:
        from dopplerscrub.pipeline import scrub
        rep = scrub(str(in_path), str(out_path), remove_cfg=cfg)
        x_clean, _ = load(out_path)
        return x_clean, list(rep.tracks), time.time() - t0, "pipeline.scrub"
    except ImportError:
        from dopplerscrub.detect import detect_tracks
        from dopplerscrub.remove import remove_tracks
        x, sr = load(in_path)
        tr = detect_tracks(x, sr)
        y = remove_tracks(x, sr, [t for t in tr if t.kind in cfg.remove_kinds], cfg)
        return y, tr, time.time() - t0, "detect+remove"


def match(true: list[Track], det: list[Track], tol_hz: float = 3.0) -> int:
    """How many true tracks have a detected track within tol_hz (median) over >=50% overlap."""
    n = 0
    for tt in true:
        for d in det:
            lo, hi = max(tt.t0, d.t0), min(tt.t1, d.t1)
            if hi - lo < 0.5 * (tt.t1 - tt.t0):
                continue
            g = np.arange(lo, hi, 0.25)
            if len(g) and np.median(np.abs(np.interp(g, tt.t, tt.f) - np.interp(g, d.t, d.f))) < tol_hz:
                n += 1
                break
    return n


def in_band(tracks, fmax):
    return [t for t in tracks if np.nanmedian(t.f) <= fmax and np.nanmax(t.f) >= ev.FMIN]


def fnum(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–" if v is None or np.isnan(v) else ("∞" if v > 0 else "-∞")
    return f"{v:.{nd}f}"


def evaluate_clip(name, x, y, sr, true, det, window=None, wide=False, tag="", fmax=None,
                  boxes=None):
    """Render + numbers for one clip (or one window of it). boxes: annotated events
    (t0, t1, f0, f1) drawn on the overlay panel and scored with ev.box_stats."""
    doppler = [t for t in det if t.kind == "doppler"]
    ref_tracks = true if true else doppler
    fmax = fmax or (ev.FMAX_WIDE if wide else ev.FMAX)
    stem = f"{name}{tag}"
    span = (window[0], window[1]) if window else None
    views = ev.attack_views(x, y, sr, fmax=fmax, span=span)
    win4 = window if window and len(window) == 4 else window
    if true:   # truth cyan, every detection green
        overlay = list(true) + [Track(t.t, t.f, t.amp, kind="detected") for t in det]
        lab = "cyan = truth, green = detected"
    else:      # real: detected doppler cyan, other detections green
        overlay, lab = list(det), "cyan = detected doppler, green = other detections"
    ev.render_before_after(x, y, sr, IMG / f"{stem}_ba.png", tracks=overlay, overlay_label=lab,
                           boxes=boxes,
                           window=win4, views=views, fmax=fmax,
                           title=f"{stem}: attacker view before / after (40-{fmax:.0f} Hz)")
    rt = ev.render_attacker_views(y, sr, IMG / f"{stem}_redteam.png", tracks=in_band(ref_tracks, fmax),
                                  window=win4, fmax=fmax, x_orig=x,
                                  title=f"{stem}: red-team views of the cleaned audio")
    rb = in_band(ref_tracks, fmax)
    SHEET.append((stem, _crop(views["orig"], win4), _crop(views["clean"], win4)))
    row = {
        "clip": stem,
        "contrast_orig_db": ev.track_contrast_db(views["orig"], rb),
        "contrast_clean_db": ev.track_contrast_db(views["clean"], rb),
        "contrast_clean_smoothed_db": rt["contrast_db"].get("smoothed"),
        "contrast_clean_global_db": rt["contrast_db"].get("global"),
        "img_ba": f"img/{stem}_ba.png", "img_rt": f"img/{stem}_redteam.png",
    }
    if boxes:
        bs = [{"box": list(b), "orig": ev.box_stats(views["orig"], b), "clean": ev.box_stats(views["clean"], b)}
              for b in boxes]
        row["boxes"] = bs
        row["box_p99_orig_db"] = float(np.nanmean([b["orig"]["p99_db"] for b in bs]))
        row["box_p99_clean_db"] = float(np.nanmean([b["clean"]["p99_db"] for b in bs]))
        row["box_hi_orig"] = float(np.nanmean([b["orig"]["frac_hi"] for b in bs]))
        row["box_hi_clean"] = float(np.nanmean([b["clean"]["frac_hi"] for b in bs]))
    return row


SHEET: list = []  # (name, orig spec, clean spec) for the contact sheet


def _crop(spec, window):
    return ev._crop_for_window(spec, window)


def write_sheet(path, items, per_page=6):
    """Original | cleaned attacker views of every clip, one row per clip, same colour scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for p0 in range(0, len(items), per_page):
        chunk = items[p0:p0 + per_page]
        fig, axs = plt.subplots(len(chunk), 2, figsize=(ev.FIG_W, 1.9 * len(chunk)), constrained_layout=True,
                                squeeze=False)
        for (name, so, sc), (a, b) in zip(chunk, axs):
            ev._imshow(a, so, -3, 15, "magma"); a.set_title(f"{name}: original", fontsize=9, loc="left")
            im = ev._imshow(b, sc, -3, 15, "magma"); b.set_title(f"{name}: CLEANED", fontsize=9, loc="left")
        fig.colorbar(im, ax=axs, shrink=0.6, label="dB above per-bin median")
        out = path if p0 == 0 else path.with_name(f"{path.stem}_{p0 // per_page + 1}{path.suffix}")
        ev.save_png(fig, out)
        plt.close(fig)


def metrics(x, y, sr, true, det):
    doppler = [t for t in det if t.kind == "doppler"]
    ref = true if true else doppler
    res = ev.residual_details(x, y, sr, ref) if ref else []
    fin = [r for r in res if np.isfinite(r["before"])]
    out = {
        "n_true": len(true) if true is not None else None,
        "n_detected": len(det), "n_doppler": len(doppler),
        "n_matched": match(true, doppler) if true else None,
        "residual_db": ev.residual_db(x, y, sr, ref) if ref else float("nan"),
        "residual_per_track": [{k: r[k] for k in ("f_mean", "residual_db", "tone_snr_before_db")}
                               for r in fin],
    }
    fid = ev.fidelity(x, y, sr, exclude_tracks=(list(true or []) + doppler))
    out.update(snr_outside_db=fid["snr_outside_db"], speech_snr_db=fid["speech_snr_db"],
               snr_db=fid["snr_db"], octave_change_db=fid["octave_change_db"])
    if ref:
        af = ev.anti_forensics(x, y, sr, ref)
        rc = af["residual"]["clean"]; ro = af["residual"]["orig"]
        st = af["stats"].get("clean", {})
        out["af"] = {"resid_floor_db": rc["excess_db"], "resid_upper95_db": rc["upper95_db"],
                     "resid_z": rc["z"], "orig_z": ro["z"], "pass_target": af["pass_target"],
                     "edge_step_db": af["edge_step_db"], "edge_null_sd_db": af["edge_step_null_sd_db"],
                     "time_edge_db": af["time_edge_db"],
                     "var_ratio_db": st.get("var_ratio_db", float("nan")),
                     "kurt_corridor": st.get("corridor", {}).get("kurtosis", float("nan")),
                     "kurt_flank": st.get("flank", {}).get("kurtosis", float("nan")),
                     "leftover_harmonics": af["leftover_harmonics"],
                     "n_harmonics_checked": len(af["harmonics"])}
    return out


# ------------------------------------------------------------------ clip sets

def synth_clips():
    p = ROOT / "data" / "synth" / "truth.json"
    if not p.exists():
        return []
    out = []
    for c in json.loads(p.read_text()):
        path = ROOT / c["path"]
        name = path.stem
        wide = any(e.get("kind") in ("road", "jet") for e in c.get("events", []))
        out.append({"name": name, "path": path, "true": [Track.from_dict(d) for d in c["tracks"]],
                    "wide": wide, "set": "synth"})
    return out


def real_clips(no_colsto=False):
    d = ROOT / "data" / "real"
    out = []
    for p in sorted(d.glob("*.wav")):
        if p.stem == "colsto":
            if no_colsto:
                continue
            ev_path = d / "colsto_events.json"
            events = json.loads(ev_path.read_text())["events"] if ev_path.exists() else []
            out.append({"name": "colsto", "path": p, "true": None, "wide": False, "set": "real",
                        "events": events})
        else:
            out.append({"name": p.stem, "path": p, "true": None, "wide": "f1" in p.stem, "set": "real"})
    return out


AIRSHOW_FMAX = 1950.0


def airshow_clips():
    p = ROOT / "data" / "real" / "airshow" / "annotations.json"
    if not p.exists():
        return []
    out = []
    for c in json.loads(p.read_text()).get("clips", []):
        path = ROOT / c["path"]
        if not path.exists():
            continue
        boxes = [(e["start_s"], e["end_s"], max(40.0, e["fmin_hz"]), min(AIRSHOW_FMAX, e["fmax_hz"]))
                 for e in c.get("events", [])]
        out.append({"name": f"air_{c['name']}", "path": path, "true": None, "wide": True,
                    "set": "airshow", "boxes": boxes})
    return out


HELDOUT_DIR = ROOT / "data" / "real" / "heldout"
HELDOUT2_DIR = ROOT / "data" / "real" / "heldout2"
# set name -> (directory, clip-name prefix). heldout is burned (results seen during
# development); heldout2 is the fresh blind set collected afterwards.
HELDOUT_SETS = {"heldout": (HELDOUT_DIR, "ho_"), "heldout2": (HELDOUT2_DIR, "h2_")}


def heldout_clips(which: str = "heldout"):
    """Held-out real clips (never used for tuning). <dir>/annotations.json in
    the airshow format ({"clips": [{"name", "path", "events": [{start_s, end_s, fmin_hz,
    fmax_hz}]}]}) if present, else every audio file there (no event boxes)."""
    hdir, prefix = HELDOUT_SETS[which]
    if not hdir.exists():
        return []
    ann = hdir / "annotations.json"
    out, seen = [], set()
    if ann.exists():
        for c in json.loads(ann.read_text()).get("clips", []):
            path = ROOT / c["path"] if not Path(c["path"]).is_absolute() else Path(c["path"])
            if not path.exists():
                continue
            boxes = [(e["start_s"], e["end_s"], max(40.0, e["fmin_hz"]), min(AIRSHOW_FMAX, e["fmax_hz"]))
                     for e in c.get("events", [])]
            out.append({"name": f"{prefix}{c.get('name', path.stem)}", "path": path, "true": None,
                        "wide": True, "set": which, "boxes": boxes})
            seen.add(path.resolve())
    for p in sorted(hdir.iterdir()):
        if p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".mp4", ".mov", ".mkv", ".webm") \
                and p.resolve() not in seen:
            out.append({"name": f"{prefix}{p.stem}", "path": p, "true": None, "wide": True,
                        "set": which, "boxes": []})
    return out


# Research scope: fixed-wing airplanes (propeller and jet). Rotorcraft, drones and road
# vehicles stay listed but are excluded from headline medians/tallies. Synthetic null
# controls (speech, music, long_hum60, stereo) stay in scope: they test do-no-harm.
OUT_OF_SCOPE = {"helicopter", "helicopter_wide", "air_helicopter_fast", "air_military_helicopters",
                "ho_drone", "ho_city_helicopter", "ho_motorcycle"}


def in_scope(clip: str) -> bool:
    """True for fixed-wing clips (and null controls); f1* (cars) and rotorcraft are out."""
    base = clip.split("_ev")[0] if clip.startswith("colsto") else clip
    return not (base in OUT_OF_SCOPE or base.startswith("f1"))


def clip_split(c) -> str:
    """'dev' = everything used during development/tuning (synth, data/real, airshow, colsto);
    'heldout' = data/real/heldout only; 'heldout2' = data/real/heldout2 only."""
    return c["set"] if c["set"] in HELDOUT_SETS else "dev"


# ------------------------------------------------------------------ markdown

COLS = ["clip", "split", "set", "scope", "detected (doppler/all)", "true", "matched", "residual dB",
        "contrast orig→clean dB", "smoothed clean dB", "SNR outside dB", "speech SNR dB",
        "event-box p99 dB (orig→clean)", "box-free ridges z>3: orig → still (new)", "persistent cells % (orig→clean)", "AF z (orig→clean)", "AF ≤-10 dB?", "AF edge dB", "AF var dB / kurt", "AF leftover k·f",
        "runtime s", "visual verdict"]


def af_cells(af):
    if not af:
        return ["–"] * 5
    lh = af.get("leftover_harmonics") or []
    lh_s = ", ".join(f"{k}×{f:.0f}" for f, k in lh[:6]) + ("…" if len(lh) > 6 else "") if lh else "none"
    return [f"{fnum(af['orig_z'])} → {fnum(af['resid_z'])}",
            ("n/a (z<3 before)" if not (af["orig_z"] is not None and af["orig_z"] >= 3)
             else ("yes" if af["pass_target"] else "**no**") + f" ({fnum(af['resid_upper95_db'])})"),
            fnum(af["edge_step_db"]), f"{fnum(af['var_ratio_db'])} / {fnum(af['kurt_corridor'])}", lh_s]


def bf_cells(b):
    if not b:
        return ["–", "–"]
    return [f"{b['n_ridges_orig_z3']} → {b['n_still_z3']} ({b['n_new_z3']})",
            f"{fnum(100 * b['persist_orig'])} → {fnum(100 * b['persist_clean'])}"]


def scope_tally(rows):
    """Headline box-free tally per split, in-scope (fixed-wing) clips only."""
    L = ["## Headline (fixed-wing clips only)", "",
         "Scope is fixed-wing airplanes; clips marked *out of scope* (rotorcraft, drones, cars, "
         "motorcycles) are listed below but excluded here.", "",
         "| split | clips with box-free check | ridges z>3 before | still z>3 | new z>3 |", "|---|---|---|---|---|"]
    for sp in ("dev", *HELDOUT_SETS):
        bs = [r["boxfree"] for r in rows if r.get("split", "dev") == sp and in_scope(r["clip"]) and r.get("boxfree")]
        if bs:
            L.append(f"| {sp} | {len(bs)} | {sum(b['n_ridges_orig_z3'] for b in bs)} | "
                     f"{sum(b['n_still_z3'] for b in bs)} | {sum(b['n_new_z3'] for b in bs)} |")
    return L + [""]


def write_md(rows, verdicts):
    lines = ["# Benchmark: before/after attacker view", "",
             "Generated by `scripts/benchmark.py` (defaults, `harden=True`). No dopplertrack "
             "fit/search is run; success is judged by eye on the attacker's whitened spectrogram.",
             "",
             "- **residual dB**: narrowband (≈2 Hz heterodyne) energy along the curves after/before "
             "(truth tracks for synth, detected for real). Noise-limited: it cannot go below "
             "-(tone SNR).",
             "- **contrast**: mean attacker-dB on the curve minus beside it (±3–6 Hz). "
             ">+1 dB = ghost, <-1 dB = notch, ≈0 = gone. *smoothed* = 2 s time-averaged red-team view.",
             "- **SNR outside**: orig vs. orig−clean outside ±3 Hz corridors; **speech SNR**: 300–3400 Hz.",
             "- **AF** (anti-forensics, docs/research.md; `evaluate.anti_forensics`): **z** = significance of "
             "on-curve power over the per-bin floor integrated over every frame of the pass (null = same "
             "statistic on curves shifted ±8–14 Hz; z>3 = an integrating attacker sees it); **≤-10 dB?** = "
             "95% upper bound of that residual is ≥10 dB under the floor (n/a when the curve was not "
             "detectable before cleaning: then the bound only reflects measurement precision); **edge** = signed step across the "
             "±2 Hz corridor edges vs control curves (<0 notch, >0 over-fill); **var/kurt** = corridor-bin "
             "power variance vs flank (dB) and corridor excess kurtosis (noise ≈ 6); **leftover k·f** = "
             "harmonics of the true/detected fundamental with z>3 after cleaning.",
             "- **box-free** (no truth, no boxes, independent of dopplerscrub's detector; "
             "`scripts/boxfree.py`): a simple attacker-side ridge extractor (2 s smoothed view > 4 dB, "
             "≥ 6 s, sweeping ≥ 3 Hz) runs on the ORIGINAL 40–450 Hz view; **ridges z>3** = those an "
             "integrating attacker confirms, **still** = how many keep z>3 in the cleaned file, **(new)** = "
             "z>3 ridges found only in the cleaned file (mostly leftover harmonics that were merged into a "
             "larger component in the original, rarely artefacts). **persistent cells** = share of smoothed cells "
             "> 4 dB (white noise: 4.6%; 0 ridges).",
             "- **event-box p99**: 99th percentile of whitened dB inside the annotated event boxes "
             "(airshow, colsto); pure noise ≈ 7 dB, so → ~7 means the ridges are gone.",
             "- **visual verdict**: written after viewing each PNG (see `verdicts.json`).", "",
             *([f"_Verdict snapshot: {verdicts['_note']}_", ""] if "_note" in verdicts else []),
             *scope_tally(rows),
             "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        v = verdicts.get(r["clip"], "(not yet reviewed)")
        lines.append("| " + " | ".join([
            f"[{r['clip']}]({r['img_ba']}) ([rt]({r['img_rt']}))", r.get("split", "dev"), r["set"],
            "fixed-wing" if in_scope(r["clip"]) else "out of scope",
            f"{r['n_doppler']}/{r['n_detected']}", fnum(r["n_true"], 0) if r["n_true"] is not None else "–",
            fnum(r["n_matched"], 0) if r["n_matched"] is not None else "–",
            fnum(r["residual_db"]),
            f"{fnum(r['contrast_orig_db'])} → {fnum(r['contrast_clean_db'])}",
            fnum(r["contrast_clean_smoothed_db"]), fnum(r["snr_outside_db"]),
            fnum(r["speech_snr_db"]),
            (f"{fnum(r['box_p99_orig_db'])} → {fnum(r['box_p99_clean_db'])}" if "box_p99_orig_db" in r else "–"),
            *bf_cells(r.get("boxfree")),
            *af_cells(r.get("af")), fnum(r["runtime_s"], 0), v]) + " |")
    lines += ["", "## Images", ""]
    for r in rows:
        lines += [f"### {r['clip']}", "", f"![{r['clip']} before/after]({r['img_ba']})", "",
                  f"![{r['clip']} red team]({r['img_rt']})", ""]
    (RES / "benchmark.md").write_text("\n".join(lines))


# ------------------------------------------------------------------ box-free (post hoc)

def add_boxfree(rows, workdir, only=None):
    """Box-free checks for rows whose cleaned WAV is in workdir (<clip>_clean.wav)."""
    allc = {c["name"]: c for c in real_clips() + airshow_clips() + heldout_clips() + heldout_clips("heldout2")}
    for name, c in allc.items():
        if only and name not in only:
            continue
        out = workdir / f"{name}_clean.wav"
        if not out.exists():
            continue
        x, sr = load(c["path"]); y, _ = load(out)
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        if name == "colsto":
            for i, e in enumerate(c["events"], 1):
                k = f"colsto_ev{i}"
                if k in rows:
                    s0, s1 = int((e["start_s"] - 30) * sr), int((e["end_s"] + 30) * sr)
                    rows[k]["boxfree"] = bf.boxfree(x[s0:s1], y[s0:s1], sr)
            continue
        if name in rows:
            print("box-free", name, flush=True)
            rows[name]["boxfree"] = bf.boxfree(x, y, sr)


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-colsto", action="store_true")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--no-airshow", action="store_true")
    ap.add_argument("--split", choices=("dev", "heldout", "heldout2", "all"), default="dev",
                    help="dev = all clips used for tuning (default); heldout = data/real/heldout "
                         "(burned); heldout2 = data/real/heldout2 (fresh blind set); all = all three")
    ap.add_argument("--md-only", action="store_true")
    ap.add_argument("--blind", action="store_true", help="print no numbers (judge images first)")
    ap.add_argument("--boxfree-from", default=None,
                    help="add box-free checks to existing real/airshow/heldout rows from cleaned WAVs here")
    ap.add_argument("--workdir", default=None, help="where cleaned WAVs go (default: temp dir)")
    a = ap.parse_args(argv)

    IMG.mkdir(parents=True, exist_ok=True)
    jpath = RES / "benchmark.json"
    vpath = RES / "verdicts.json"
    verdicts = json.loads(vpath.read_text()) if vpath.exists() else {}
    prev = {r["clip"]: r for r in json.loads(jpath.read_text())} if jpath.exists() else {}
    for r in prev.values():   # rows from before the split existed were all dev/tuning clips
        r.setdefault("split", r.get("set") if r.get("set") in HELDOUT_SETS else "dev")
    if a.md_only:
        write_md(list(prev.values()), verdicts)
        return
    if a.boxfree_from:
        add_boxfree(prev, Path(a.boxfree_from), a.only)
        jpath.write_text(json.dumps(list(prev.values()), indent=1, default=float))
        write_md(list(prev.values()), verdicts)
        return

    dev = (synth_clips() + ([] if a.no_real else real_clips(a.no_colsto))
           + ([] if a.no_airshow else airshow_clips()))
    clips = {"dev": dev, "heldout": heldout_clips(), "heldout2": heldout_clips("heldout2"),
             "all": dev + heldout_clips() + heldout_clips("heldout2")}[a.split]
    if a.only:
        clips = [c for c in clips if c["name"] in a.only]
    work = Path(a.workdir or tempfile.mkdtemp(prefix="dsbench_"))
    work.mkdir(parents=True, exist_ok=True)
    rows = dict(prev)
    done: set[str] = set()
    failed: dict[str, str] = {}
    for c in clips:
        print(f"== {c['name']}", flush=True)
        x, sr = load(c["path"])
        try:
            y, det, rt_s, how = run_scrub(c["path"], work / f"{c['name']}_clean.wav")
        except Exception as e:  # keep benchmarking the other clips
            import traceback; traceback.print_exc()
            print(f"   FAILED: {type(e).__name__}: {e}", flush=True)
            failed[c["name"]] = f"{type(e).__name__}: {e}"
            continue
        n = min(len(x), len(y)); x, y = x[:n], y[:n]
        base = {"set": c["set"], "split": clip_split(c), "runtime_s": rt_s, "how": how}
        if c["name"] == "colsto":
            # numbers only inside the event windows (+-60 s); coarse full view + zooms
            doppler = [t for t in det if t.kind == "doppler"]
            m = metrics(x, y, sr, None, det)  # whole-file fidelity
            cboxes = [(e["start_s"], e["end_s"], e.get("fmin_hz", 60), e.get("fmax_hz", 300))
                      for e in c["events"]]
            r = evaluate_clip("colsto", x, y, sr, None, det, tag="_full", boxes=cboxes)
            r.update(base, **m); rows[r["clip"]] = r
            for i, e in enumerate(c["events"], 1):
                w = (e["start_s"] - 30, e["end_s"] + 30)
                r = evaluate_clip("colsto", x, y, sr, None, det,
                                  window=(w[0], w[1], 60, 300), tag=f"_ev{i}", boxes=[cboxes[i - 1]])
                s0, s1 = int(w[0] * sr), int(w[1] * sr)
                loc = [t for t in doppler if t.t1 > w[0] and t.t0 < w[1]]
                mm = metrics(x[s0:s1], y[s0:s1], sr, None,
                             [Track(t.t - w[0], t.f, t.amp, kind=t.kind) for t in loc])
                r.update(base, **mm); r["runtime_s"] = float("nan")
                rows[r["clip"]] = r
        elif c["set"] in ("airshow", *HELDOUT_SETS):
            m = metrics(x, y, sr, None, det)
            r = evaluate_clip(c["name"], x, y, sr, None, det, fmax=AIRSHOW_FMAX, boxes=c["boxes"] or None)
            r.update(base, **m); r["boxfree"] = bf.boxfree(x, y, sr); rows[r["clip"]] = r
        else:
            m = metrics(x, y, sr, c["true"], det)
            r = evaluate_clip(c["name"], x, y, sr, c["true"], det, wide=False)
            r.update(base, **m); rows[r["clip"]] = r
            if c["wide"]:
                r2 = evaluate_clip(c["name"], x, y, sr, c["true"], det, wide=True, tag="_wide")
                r2.update(base, **m); rows[r2["clip"]] = r2
        print(f"   {how} {rt_s:.1f}s" + ("" if a.blind else
              f"  detected {len(det)}  residual {rows[r['clip']]['residual_db']:.1f} dB"), flush=True)
        # merge with whatever is on disk now (other benchmark runs may write concurrently)
        disk = {r["clip"]: r for r in json.loads(jpath.read_text())} if jpath.exists() else {}
        disk.update({k: v for k, v in rows.items() if k in done or k not in disk})
        done.update(k for k in rows if k.startswith(c["name"]))
        disk.update({k: rows[k] for k in done})
        rows = disk
        jpath.write_text(json.dumps(list(rows.values()), indent=1, default=float))
    write_md(list(rows.values()), verdicts)
    if SHEET:
        tag = "colsto" if a.only == ["colsto"] else "all"
        write_sheet(IMG / f"sheet_{tag}.png", SHEET)
    if failed:
        print("FAILED clips:", failed)
    print(f"wrote {RES / 'benchmark.md'}")


if __name__ == "__main__":
    main()
