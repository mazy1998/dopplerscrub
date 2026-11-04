#!/usr/bin/env python
"""Hard synthetic fixed-wing suite with exact truth (data/synth_hard): generate and score.

  uv run python scripts/synth_hard.py generate [--only NAME ...]   # WAVs (gitignored) + truth.json
  uv run python scripts/synth_hard.py score [--only NAME ...] [--workers 4] [--img]
  uv run python scripts/synth_hard.py report                       # rewrite the md from the json

Scoring runs `pipeline.scrub_array` with default configs on every clip, then per clip and
truth harmonic (the part inside the attacker's 40-1950 Hz band, 4 kHz decode):

- visible s: frames where the 2 s-smoothed whitened attacker dB at f(t) is > 1.5 dB
  (scripts/diagnose_detector.py rule);
- cov 1 / 3 Hz: share of visible frames with a doppler-labelled detection within 1 / 3 Hz;
  time cov: visible frames inside the time span of the best-matching detection;
- z before -> after -> oracle: integrated on-curve power over the per-bin floor
  (`evaluate._curve_vs_floor`, null from +-8..14 Hz shifted copies); oracle = `remove_tracks`
  fed the truth itself (incl. truth width_hz). z > 3 = an integrating attacker detects it.
  Wide (FM-jittered) truth tracks also get z_wide: flanks and null shifts moved out past
  the tone's width (the stock +-3..6 Hz flanks sit inside a ~8 Hz wide tone and understate z);
- clip level: `evaluate.anti_forensics` residual z (all truth curves at once) before/after
  and for the oracle, leftover k*f harmonics; box-free ridges (scripts/boxfree.py) with the
  created / unmasked / seen split, at 40-450 Hz and 40-1950 Hz; spurious corridors =
  doppler detections that follow no truth curve (< 50% of their span within 3 Hz);
  null clips: every doppler corridor is a false alarm (count, seconds, TF share touched).

Clip verdict: PASS when every truth harmonic the attacker sees (z_before > 3) ends at z <= 3
and no ridge is created; null clips PASS with no doppler corridor and no created ridge.
`oracle-limited` marks curves the oracle cannot remove either (remover, not detector).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "data" / "synth_hard"
OUT_JSON = ROOT / "docs" / "results" / "synth_hard.json"
OUT_MD = ROOT / "docs" / "results" / "synth_hard.md"
IMG = ROOT / "docs" / "results" / "img"
FMAX = 1950.0
Z_DET = 3.0


def _band(tr, fmax=FMAX - 10, fmin=45.0):
    from dopplerscrub.types import Track
    keep = (tr.f < fmax) & (tr.f > fmin)
    if keep.sum() < 8:
        return None
    return Track(tr.t[keep], tr.f[keep], tr.amp[keep], 1.0, tr.fit, tr.harmonic, tr.group,
                 "doppler", tr.width_hz)


def _z(raw, t, f, tracks, flank=(3.0, 6.0), shifts=None):
    """evaluate._curve_vs_floor with optional custom null shifts (for wide tones)."""
    from dopplerscrub import evaluate as ev
    if shifts is None:
        return ev._curve_vs_floor(raw, t, f, tracks, flank)["z"]
    r_on, n = ev._ratio_to_floor(raw, t, f, tracks, 0.0, flank)
    null = np.array([ev._ratio_to_floor(raw, t, f, tracks, s, flank)[0] for s in shifts])
    null = null[np.isfinite(null)]
    if n < 8 or len(null) < 3 or not np.isfinite(r_on):
        return float("nan")
    sd = float(null.std(ddof=1))
    return float((r_on - null.mean()) / sd) if sd > 0 else float("nan")


def _follows(det, truth, tol=3.0, share=0.5):
    """True if >= share of det's frames lie within tol Hz of some truth curve."""
    hit = np.zeros(len(det.t), bool)
    for q in truth:
        inside = (det.t >= q.t[0]) & (det.t <= q.t[-1])
        hit |= inside & (np.abs(np.interp(det.t, q.t, q.f) - det.f) < tol)
    return hit.mean() >= share


def _fnum(v, nd=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:.{nd}f}"


def score_clip(rec, img=False):
    import soundfile as sf
    from dopplerscrub import evaluate as ev
    from dopplerscrub.pipeline import scrub_array
    from dopplerscrub.remove import remove_tracks
    from dopplerscrub.types import RemoveConfig, Track
    import boxfree as bf
    from diagnose_detector import coverage, visible_frames

    x, sr = sf.read(ROOT / rec["path"], dtype="float32")
    truth = [Track.from_dict(d) for d in rec["tracks"]]
    t0 = time.time()
    y, found = scrub_array(x, sr)
    runtime = time.time() - t0
    rcfg = RemoveConfig()
    dop = [q for q in found if q.kind in rcfg.remove_kinds]
    out = {"clip": rec["name"], "category": rec["category"], "null": rec["null"],
           "duration_s": rec["duration_s"], "runtime_s": runtime, "n_found": len(found),
           "n_doppler": len(dop), "doppler_s": float(sum(q.t1 - q.t0 for q in dop))}
    v = ev.attack_views(x, y, sr, fmax=FMAX)
    t, f = v["orig"].t, v["orig"].f
    tf = ev.tf_modified_fraction(x, y, sr)
    out["tf_modified"] = tf
    out["lsd_db"] = ev.log_spectral_distance(x, y, sr)
    out["boxfree_450"] = {k: w for k, w in bf.boxfree(x, y, sr).items() if k not in ("ridges", "new_ridges")}
    b19 = bf.boxfree(x, y, sr, fmax=FMAX, views=v)
    out["boxfree_1950"] = {k: w for k, w in b19.items() if k not in ("ridges",)}
    if not truth:
        out["fidelity"] = {k: w for k, w in ev.fidelity(x, y, sr).items() if k != "octave_change_db"}
        out["false_corridors"] = [{"t0": q.t0, "t1": q.t1, "f_min": float(q.f.min()), "f_max": float(q.f.max()),
                                   "score": float(q.doppler_score), "h": int(q.harmonic), "g": int(q.group)}
                                  for q in dop]
        if img:
            ev.render_before_after(x, y, sr, IMG / f"sh_{rec['name']}_ba.png",
                                   tracks=[Track(q.t, q.f, q.amp, kind="detected") for q in found],
                                   fmax=FMAX, views=v, title=f"{rec['name']} (null): before / after")
        return out
    yo = remove_tracks(x, sr, truth, rcfg)
    vo = ev.attack_views(x, yo, sr, fmax=FMAX)
    inband = [b for b in (_band(q) for q in truth) if b is not None]
    out["n_truth"] = len(truth); out["n_truth_inband"] = len(inband)
    spur = [q for q in dop if not _follows(q, truth)]
    out["n_spurious"] = len(spur)
    out["spurious_s"] = float(sum(q.t1 - q.t0 for q in spur))
    out["fidelity"] = {k: w for k, w in ev.fidelity(x, y, sr, exclude_tracks=truth + dop).items()
                       if k != "octave_change_db"}
    rows = []
    for q in inband:
        tt, ff, vis = visible_frames(v["orig"], q)
        c1, best = coverage(tt, ff, vis, dop, 1.0)
        c3, best3 = coverage(tt, ff, vis, dop, 3.0)
        wide = q.width_hz is not None and q.width_hz > 2.0
        row = {"group": int(q.group), "harmonic": int(q.harmonic),
               "f_lo": float(q.f.min()), "f_hi": float(q.f.max()), "width_hz": q.width_hz,
               "max_slope_hz_s": float(np.max(np.abs(np.diff(q.f) / np.diff(q.t)))),
               "visible_s": float(vis.sum() * 0.25),
               "cov1": c1, "cov3": c3,
               "time_cov": (best3 or {}).get("time_cov", 0.0),
               "best_rms_hz": (best3 or {}).get("rms_hz_overlap", float("nan")),
               "z_before": _z(v["raw_orig"], t, f, [q]),
               "z_after": _z(v["raw_clean"], t, f, [q]),
               "z_oracle": _z(vo["raw_clean"], t, f, [q])}
        if wide:
            w = q.width_hz
            fl = (w + 2.0, w + 8.0)
            sh = tuple(s * (2 * w + 10 + k) for s in (-1, 1) for k in (0, 3, 6, 9))
            row["z_wide_before"] = _z(v["raw_orig"], t, f, [q], fl, sh)
            row["z_wide_after"] = _z(v["raw_clean"], t, f, [q], fl, sh)
            row["z_wide_oracle"] = _z(vo["raw_clean"], t, f, [q], fl, sh)
        rows.append(row)
    out["curves"] = rows
    if inband:
        af = ev.anti_forensics(x, y, sr, inband, fmax=FMAX)
        afo = ev.anti_forensics(x, yo, sr, inband, fmax=FMAX)
        out["af"] = {"z_before": af["residual"]["orig"]["z"], "z_after": af["residual"]["clean"]["z"],
                     "z_oracle": afo["residual"]["clean"]["z"],
                     "upper95_after_db": af["residual"]["clean"]["upper95_db"],
                     "leftover_harmonics": len(af["leftover_harmonics"]),
                     "leftover_harmonics_oracle": len(afo["leftover_harmonics"]),
                     "edge_step_db": af["edge_step_db"], "pass_target": af["pass_target"]}
        bo = bf.boxfree(x, yo, sr, fmax=FMAX, views=vo)
        out["boxfree_1950_oracle"] = {k: w for k, w in bo.items() if k not in ("ridges", "new_ridges")}
    if img:
        overlay = list(truth) + [Track(q.t, q.f, q.amp, kind="detected") for q in found]
        ev.render_before_after(x, y, sr, IMG / f"sh_{rec['name']}_ba.png", tracks=overlay, fmax=FMAX,
                               views=v, overlay_label="cyan = truth, green = detected",
                               title=f"{rec['name']}: before / after (40-{FMAX:.0f} Hz)")
    return out


def verdict(r):
    created = max(r["boxfree_450"]["n_new_created_z3"], r["boxfree_1950"]["n_new_created_z3"])
    if r["null"]:
        ok = r["n_doppler"] == 0 and created == 0
        why = [] if ok else ([f"{r['n_doppler']} false corridor(s), {r['doppler_s']:.0f} s"] if r["n_doppler"] else []) + \
            ([f"{created} created ridge(s)"] if created else [])
        return ("PASS" if ok else "FAIL"), "; ".join(why)
    seen = [c for c in r["curves"] if np.isfinite(c["z_before"]) and c["z_before"] > Z_DET]
    left = [c for c in seen if not (np.isfinite(c["z_after"]) and c["z_after"] <= Z_DET)]
    if not seen:
        return "N/A", "attacker sees no curve (all z <= 3)"
    why = []
    if left:
        olim = [c for c in left if np.isfinite(c["z_oracle"]) and c["z_oracle"] > Z_DET]
        why.append(f"{len(left)}/{len(seen)} visible curves left at z>3"
                   + (f" ({len(olim)} oracle-limited)" if olim else ""))
    if created:
        why.append(f"{created} created ridge(s)")
    if not left and not created:
        return "PASS", f"{len(seen)}/{len(seen)} visible curves removed"
    return ("PARTIAL" if len(left) < len(seen) else "FAIL"), "; ".join(why)


def write_md(res):
    rows = res["clips"]
    L = ["# Hard synthetic fixed-wing suite (data/synth_hard): current detector",
         "",
         f"Generated by `scripts/synth_hard.py` ({res['time']}, commit {res.get('commit', '?')}). "
         "Clips: `data/synth_hard/truth.json` (WAVs gitignored; `uv run python scripts/synth_hard.py generate` "
         "rebuilds them bit-exactly). Generator: `dopplerscrub.synth.hard_specs` / `render_pass`. 48 kHz mono, "
         "30–120 s. Detector = `pipeline.scrub_array` with default configs; oracle = `remove_tracks` fed the truth.",
         "",
         "Truth is exact: received frequency = source frequency at the emission time × dte/dto of the direct "
         "path, on a 0.25 s grid (checked against the closed-form straight pass and dopplertrack's "
         "`curve_from_trajectory` in `tests/test_synth_hard.py`). Ground-reflected copies, FM jitter and "
         "backgrounds are in the audio, not the truth; jittered tones carry `width_hz` (1.75 × received FM std).",
         "",
         "**Metrics.** visible s = frames where the 2 s-smoothed attacker view at f(t) is > 1.5 dB. "
         "cov 1/3 Hz = share of visible frames with a doppler-labelled detection within 1/3 Hz; "
         "time cov = visible frames inside the best-matching detection's span. z = integrated on-curve "
         "power vs the per-bin floor (`_curve_vs_floor`, attacker 4 kHz STFT, 40–1950 Hz); z > 3 = detected by "
         "an integrating attacker. z_wide (FM-jittered tones) moves flanks/null past the tone width. "
         "AF z = all truth curves at once (`anti_forensics`). created = box-free ridges the processing put "
         "where the original had none (max of the 40–450 and 40–1950 Hz runs). spurious = doppler detections "
         "following no truth curve.",
         "",
         "## Verdicts", "",
         "| clip | category | verdict | why | AF z before → after (oracle) | leftover k·f (oracle) | doppler tracks (spurious) | created 450/1950 | LSD dB | runtime s |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        vd, why = verdict(r)
        af = r.get("af", {})
        b4, b19 = r["boxfree_450"], r["boxfree_1950"]
        sp = f"{r['n_doppler']} ({r['n_spurious']})" if not r["null"] else f"**{r['n_doppler']}** ({r['doppler_s']:.0f} s)"
        L.append(f"| {r['clip']} | {r['category']} | **{vd}** | {why} | "
                 + (f"{_fnum(af.get('z_before'))} → {_fnum(af.get('z_after'))} ({_fnum(af.get('z_oracle'))})" if af else "–")
                 + " | " + (f"{af.get('leftover_harmonics')} ({af.get('leftover_harmonics_oracle')})" if af else "–")
                 + f" | {sp} | {b4['n_new_created_z3']}/{b19['n_new_created_z3']} | {_fnum(r['lsd_db'], 2)} | {r['runtime_s']:.0f} |")
    tally = {}
    for r in rows:
        tally[verdict(r)[0]] = tally.get(verdict(r)[0], 0) + 1
    L += ["", "Tally: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())), ""]
    L += ["## Per curve (aircraft clips)", "",
          "Only curves with any part inside 45–1940 Hz. Bold z after = still detectable (> 3).", "",
          "| clip | g/h | f range Hz | max slope Hz/s | width Hz | visible s | cov 1 / 3 Hz | time cov | z before → after (oracle) | z_wide before → after (oracle) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for c in r.get("curves", []):
            za = c["z_after"]
            zs = f"**{_fnum(za)}**" if np.isfinite(za) and za > Z_DET and c["z_before"] > Z_DET else _fnum(za)
            zw = (f"{_fnum(c['z_wide_before'])} → {_fnum(c['z_wide_after'])} ({_fnum(c['z_wide_oracle'])})"
                  if "z_wide_before" in c else "")
            L.append(f"| {r['clip']} | {c['group']}/{c['harmonic']} | {c['f_lo']:.0f}–{c['f_hi']:.0f} | "
                     f"{c['max_slope_hz_s']:.0f} | {_fnum(c['width_hz'])} | {c['visible_s']:.0f} | "
                     f"{c['cov1']:.2f} / {c['cov3']:.2f} | {c['time_cov']:.2f} | "
                     f"{_fnum(c['z_before'])} → {zs} ({_fnum(c['z_oracle'])}) | {zw} |")
    L += ["", "## Null clips (no aircraft)", "",
          "| clip | doppler corridors | seconds | TF modified 40–450 / 450+ | speech-band SNR dB | attacker ridges z>3 in original (450/1950) | created 450/1950 |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not r["null"]:
            continue
        tf = r["tf_modified"]
        k = list(tf)
        L.append(f"| {r['clip']} | {r['n_doppler']} | {r['doppler_s']:.0f} | "
                 f"{100 * tf[k[1]]:.1f}% / {100 * tf[k[2]]:.1f}% | {_fnum(r['fidelity'].get('speech_snr_db'))} | "
                 f"{r['boxfree_450']['n_ridges_orig_z3']}/{r['boxfree_1950']['n_ridges_orig_z3']} | "
                 f"{r['boxfree_450']['n_new_created_z3']}/{r['boxfree_1950']['n_new_created_z3']} |")
    L += ["", "## Clip descriptions", ""]
    for d in res.get("descriptions", []):
        L.append(f"- **{d['name']}** ({d['category']}, {d['duration_s']} s): {d['description']}")
    if res.get("findings"):
        L += ["", "## Findings", "", res["findings"]]
    OUT_MD.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "score", "report"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img", action="store_true", help="write docs/results/img/sh_<clip>_ba.png")
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        from dopplerscrub.synth import make_hard_dataset
        import os
        os.chdir(ROOT)   # truth.json stores repo-relative paths
        recs = make_hard_dataset("data/synth_hard", only=a.only)
        print(f"{len(recs)} clips -> {DATA}/truth.json")
        return
    recs = json.loads((DATA / "truth.json").read_text())
    for r in recs:   # paths are stored relative to the repo root
        r["path"] = str(Path(r["path"]).relative_to(ROOT)) if Path(r["path"]).is_absolute() else r["path"]
    res = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"clips": []}
    if a.cmd == "score":
        todo = [r for r in recs if not a.only or r["name"] in a.only]
        if a.img:
            IMG.mkdir(parents=True, exist_ok=True)
        done = {r["clip"]: r for r in res["clips"]}
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(score_clip, r, a.img): r["name"] for r in todo}
            for fu in futs:
                r = fu.result()
                done[r["clip"]] = r
                print(f"{r['clip']}: {verdict(r)} ({r['runtime_s']:.0f} s)", flush=True)
        order = [r["name"] for r in recs]
        res["clips"] = [done[n] for n in order if n in done]
        res["time"] = time.strftime("%Y-%m-%d %H:%M")
        import subprocess
        res["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip()
    res["descriptions"] = [{k: r[k] for k in ("name", "category", "duration_s", "description")} for r in recs]
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
    write_md(res)


if __name__ == "__main__":
    main()
