#!/usr/bin/env python
"""Hard synthetic fixed-wing suite with exact truth (data/synth_hard): generate and score.

  uv run python scripts/synth_hard.py generate [--only NAME ...]   # WAVs (gitignored) + truth.json
  uv run python scripts/synth_hard.py score [--only NAME ...] [--workers 4] [--img]
  uv run python scripts/synth_hard.py report                       # rewrite the md from the json

Detector variants: `--detector mod:func` replaces `dopplerscrub.detect.detect_tracks`
(func(x, sr, cfg) -> list[Track]; the pipeline's 2 detect/remove passes + harden stay), or
`--scrub mod:func` replaces the whole `pipeline.scrub_array` (func(x, sr) -> (y, tracks)).
`mod` is a module name or a path to a .py file. `--tag NAME` writes
docs/results/synth_hard_NAME.{json,md} so variants never overwrite the default run.

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

Per pass (aircraft = truth group // 100; twins are one pass): z of all its in-band curves at
once, and "core" z on the steep part only (frames where |df/dt| >= 25% of the curve's max).
Core z is a DECHIRPED matched detector (heterodyne the attacker's 4 kHz signal by the true
core phase, tone band +-(width+1) Hz vs 20 offset bands, robust z): the 2 s attacker STFT
smears a 150-450 Hz/s core over tens of Hz, so spectrogram flank statistics fail there. It
is attacker-favourable (it knows the exact curve). A pass is USABLE (proxy for the
geolocation attacker, which needs the CPA shape to constrain v and d) when its combined core
z > 5 or one harmonic's core z > 8 (null calibration: aircraft cores laid on the ambience,
traffic and birds clips give pass z <= 4.1, single-harmonic p99 3.4-4.7). Headline per recording:
usable passes before -> after (oracle); the attacker needs >= 2 with an unknown clock, 1 with
a known clock. Faint plateau leftovers are visible (curve z > 3) but not usable.

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
CORE_FRAC = 0.25      # core = frames with |df/dt| >= this share of the curve's max slope
USABLE_Z_ALL, USABLE_Z_ONE = 5.0, 8.0   # calibrated: noise-like null clips give pass z <= 4.1
IMG = ROOT / "docs" / "results" / "img"
CACHE = ROOT / ".cache" / "synth_hard"   # scrubbed audio + tracks per detector tag (gitignored)
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


ATTACK_SR = 4000


def _dechirp_stat(x4, q, n_null=10):
    """Matched (dechirped) detector along one core segment, on the attacker's 4 kHz mono signal:
    heterodyne by the true phase 2*pi*int f dt, Hann-window the core span, and compare the mean
    baseband power within +-(width + 1 Hz) of 0 with the same band at 2*n_null offsets
    +-(2*hw + 6 + 3k) Hz. Immune to STFT smear (the core sweeps up to ~450 Hz/s here).
    Returns (on, null array) in linear power."""
    i0 = max(0, int(np.ceil(q.t0 * ATTACK_SR))); i1 = min(len(x4), int(q.t1 * ATTACK_SR))
    if i1 - i0 < ATTACK_SR:
        return float("nan"), np.array([])
    tt = np.arange(i0, i1) / ATTACK_SR
    ph = 2 * np.pi * np.cumsum(np.interp(tt, q.t, q.f)) / ATTACK_SR
    b = x4[i0:i1] * np.exp(-1j * ph) * np.hanning(i1 - i0)
    nf = 1 << int(np.ceil(np.log2(4 * len(b))))
    B = np.abs(np.fft.fft(b, nf)) ** 2
    fb = np.fft.fftfreq(nf, 1 / ATTACK_SR)
    hw = (q.width_hz or 0.0) + 1.0
    band = lambda c: float(B[np.abs(fb - c) <= hw].mean())
    offs = [s_ * (2 * hw + 6 + 3 * k) for s_ in (-1, 1) for k in range(n_null)]
    return band(0.0), np.array([band(o) for o in offs])


def _z_dechirp(x4, cores):
    """Robust z of the summed dechirped core power over its offset nulls (median / 1.4826 MAD)."""
    on, nulls = 0.0, None
    for q in cores:
        o, nl = _dechirp_stat(x4, q)
        if not np.isfinite(o) or not len(nl):
            continue
        on += o; nulls = nl if nulls is None else nulls + nl
    if nulls is None:
        return float("nan")
    med = float(np.median(nulls)); mad = 1.4826 * float(np.median(np.abs(nulls - med)))
    return float((on - med) / mad) if mad > 0 else float("nan")


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


def _load_func(spec):
    """'pkg.mod:func' or 'path/to/file.py:func' -> callable."""
    import importlib
    import importlib.util
    mod, _, name = spec.rpartition(":")
    if mod.endswith(".py"):
        sp = importlib.util.spec_from_file_location(Path(mod).stem, mod)
        m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, name)


def _run_scrub(x, sr, detector=None, scrub=None):
    from dopplerscrub import detect as det_mod
    from dopplerscrub.pipeline import scrub_array
    if scrub:
        return _load_func(scrub)(x, sr)
    if detector:
        det_mod.detect_tracks = _load_func(detector)   # scrub_array imports it at call time
    return scrub_array(x, sr)


def _core(q, frac=CORE_FRAC):
    """Contiguous steep segments of a truth curve (as Tracks)."""
    from dopplerscrub.types import Track
    sl = np.abs(np.gradient(q.f, q.t))
    m = sl >= frac * sl.max() if sl.max() > 0 else np.zeros(len(sl), bool)
    out, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if j - i >= 8:
                out.append(Track(q.t[i:j], q.f[i:j], q.amp[i:j], 1.0, None, q.harmonic, q.group, "doppler",
                                 q.width_hz))
            i = j
        else:
            i += 1
    return out


def score_clip(rec, img=False, detector=None, scrub=None, tag=None, reuse=False):
    import warnings
    warnings.filterwarnings("ignore")
    import soundfile as sf
    from dopplerscrub import evaluate as ev
    from dopplerscrub.remove import remove_tracks
    from dopplerscrub.types import RemoveConfig, Track
    import boxfree as bf
    from diagnose_detector import coverage, visible_frames

    x, sr = sf.read(ROOT / rec["path"], dtype="float32")
    truth = [Track.from_dict(d) for d in rec["tracks"]]
    from dopplerscrub.types import Track as _T
    cache = CACHE / (tag or "default")
    cache.mkdir(parents=True, exist_ok=True)
    cy, ct = cache / f"{rec['name']}_y.npy", cache / f"{rec['name']}_tracks.json"
    if reuse and cy.exists() and ct.exists():
        y = np.load(cy); c_ = json.loads(ct.read_text())
        found = [_T.from_dict(d) for d in c_["tracks"]]; runtime = c_["runtime_s"]
    else:
        t0 = time.time()
        y, found = _run_scrub(x, sr, detector, scrub)
        runtime = time.time() - t0
        np.save(cy, np.asarray(y, np.float32))
        ct.write_text(json.dumps({"runtime_s": runtime, "tracks": [q.to_dict() for q in found]}))
    rcfg = RemoveConfig()
    dop = [q for q in found if q.kind in rcfg.remove_kinds]
    out = {"clip": rec["name"], "category": rec["category"], "null": rec["null"],
           "duration_s": rec["duration_s"], "runtime_s": runtime, "n_found": len(found),
           "n_doppler": len(dop), "doppler_s": float(sum(q.t1 - q.t0 for q in dop))}
    v = ev.attack_views(x, y, sr, fmax=FMAX)
    t, f = v["orig"].t, v["orig"].f
    a4 = {"before": ev._attack_signal(x, sr).astype(np.float64), "after": ev._attack_signal(y, sr).astype(np.float64)}
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
    co = CACHE / "oracle" / f"{rec['name']}_yo.npy"
    if reuse and co.exists():
        yo = np.load(co)
    else:
        yo = remove_tracks(x, sr, truth, rcfg)
        co.parent.mkdir(parents=True, exist_ok=True); np.save(co, np.asarray(yo, np.float32))
    vo = ev.attack_views(x, yo, sr, fmax=FMAX)
    a4["oracle"] = ev._attack_signal(yo, sr).astype(np.float64)
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
        core = _core(q)
        incore = np.zeros(len(tt), bool)
        for c_ in core:
            incore |= (tt >= c_.t0) & (tt <= c_.t1)
        row["core_cov1"] = coverage(tt, ff, incore, dop, 1.0)[0] if incore.any() else float("nan")
        row["core_cov3"] = coverage(tt, ff, incore, dop, 3.0)[0] if incore.any() else float("nan")
        row["core_s"] = float(sum(c.t1 - c.t0 for c in core))
        for k_ in ("before", "after", "oracle"):
            row[f"zc_{k_}"] = _z_dechirp(a4[k_], core) if core else float("nan")
        if wide:
            w = q.width_hz
            fl = (w + 2.0, w + 8.0)
            sh = tuple(s * (2 * w + 10 + k) for s in (-1, 1) for k in (0, 3, 6, 9))
            row["z_wide_before"] = _z(v["raw_orig"], t, f, [q], fl, sh)
            row["z_wide_after"] = _z(v["raw_clean"], t, f, [q], fl, sh)
            row["z_wide_oracle"] = _z(vo["raw_clean"], t, f, [q], fl, sh)
        rows.append(row)
    out["curves"] = rows
    passes = []
    for p in rec.get("passes", []):
        mine = [q for q in inband if q.group // 100 == p["event"]]
        cr = [c for c in rows if c["group"] // 100 == p["event"]]
        core = [c for q in mine for c in _core(q)]
        e = {k: p[k] for k in ("event", "kind", "v", "d", "h", "f0", "straight")}
        e["n_curves"] = len(mine)
        for k_, raw in (("before", v["raw_orig"]), ("after", v["raw_clean"]), ("oracle", vo["raw_clean"])):
            e[f"z_{k_}"] = _z(raw, t, f, mine) if mine else float("nan")
            e[f"zc_{k_}"] = _z_dechirp(a4[k_], core) if core else float("nan")
            one = [c[f"zc_{k_}"] for c in cr if np.isfinite(c[f"zc_{k_}"])]
            e[f"zc_max_{k_}"] = float(max(one)) if one else float("nan")
            e[f"usable_{k_}"] = bool((np.isfinite(e[f"zc_{k_}"]) and e[f"zc_{k_}"] > USABLE_Z_ALL)
                                     or (one and max(one) > USABLE_Z_ONE))
        passes.append(e)
    out["passes"] = passes
    for k_ in ("before", "after", "oracle"):
        out[f"usable_{k_}"] = sum(e[f"usable_{k_}"] for e in passes)
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
    newly = [c for c in r["curves"] if c not in seen and np.isfinite(c["z_after"]) and c["z_after"] > Z_DET]
    if not seen:
        return "N/A", "attacker sees no curve (all z <= 3)"
    why = []
    if newly:
        why.append(f"{len(newly)} curve(s) z<=3 before rose above 3 (unmasked)")
    if left:
        olim = [c for c in left if np.isfinite(c["z_oracle"]) and c["z_oracle"] > Z_DET]
        why.append(f"{len(left)}/{len(seen)} visible curves left at z>3"
                   + (f" ({len(olim)} oracle-limited)" if olim else ""))
    if created:
        why.append(f"{created} created ridge(s)")
    if not left and not created and not newly:
        return "PASS", f"{len(seen)}/{len(seen)} visible curves removed"
    return ("PARTIAL" if len(left) < len(seen) else "FAIL"), "; ".join(why)


def write_md(res):
    rows = res["clips"]
    L = ["# Hard synthetic fixed-wing suite (data/synth_hard): current detector",
         "",
         f"Generated by `scripts/synth_hard.py` ({res['time']}, commit {res.get('commit', '?')}, "
         f"detector: {res.get('detector', 'default')}). "
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
         "cov 1/3 Hz = share of visible frames with a doppler-labelled detection within 1/3 Hz "
         "(core cov: same over the steep core frames only, visible or not); "
         "time cov = visible frames inside the best-matching detection's span. z = integrated on-curve "
         "power vs the per-bin floor (`_curve_vs_floor`, attacker 4 kHz STFT, 40–1950 Hz); z > 3 = detected by "
         "an integrating attacker. z_wide (FM-jittered tones) moves flanks/null past the tone width. "
         "AF z = all truth curves at once (`anti_forensics`). created = box-free ridges the processing put "
         "where the original had none (max of the 40–450 and 40–1950 Hz runs). spurious = doppler detections "
         "following no truth curve.",
         "",
         *(["## Findings", "", res["findings"], ""] if res.get("findings") else []),
         "## Verdicts", "",
         "| clip | category | usable passes before → after (oracle) | verdict | why | AF z before → after (oracle) | leftover k·f (oracle) | doppler tracks (spurious) | created 450/1950 | LSD dB | runtime s |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        vd, why = verdict(r)
        us = (f"{r['usable_before']} → **{r['usable_after']}** ({r['usable_oracle']})" if "usable_before" in r else "–")
        af = r.get("af", {})
        b4, b19 = r["boxfree_450"], r["boxfree_1950"]
        sp = f"{r['n_doppler']} ({r['n_spurious']})" if not r["null"] else f"**{r['n_doppler']}** ({r['doppler_s']:.0f} s)"
        L.append(f"| {r['clip']} | {r['category']} | {us} | **{vd}** | {why} | "
                 + (f"{_fnum(af.get('z_before'))} → {_fnum(af.get('z_after'))} ({_fnum(af.get('z_oracle'))})" if af else "–")
                 + " | " + (f"{af.get('leftover_harmonics')} ({af.get('leftover_harmonics_oracle')})" if af else "–")
                 + f" | {sp} | {b4['n_new_created_z3']}/{b19['n_new_created_z3']} | {_fnum(r['lsd_db'], 2)} | {r['runtime_s']:.0f} |")
    tally = {}
    for r in rows:
        tally[verdict(r)[0]] = tally.get(verdict(r)[0], 0) + 1
    L += ["", "Tally: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())), ""]
    L += ["## Per pass", "",
          "z = all in-band curves of the pass at once; core z = steep part only (|df/dt| >= 25% of max). "
          "Core z = dechirped matched detector on the steep part (attacker 4 kHz signal heterodyned by the "
          "true core phase; robust z vs 20 offset bands). usable = pass core z > 5 or one harmonic's core z > 8 "
          "(noise-like null clips: pass z <= 4.1). Proxy only; the real check is a (v, d) fit.", "",
          "| clip | pass | kind | v m/s | d m | h s | f0 Hz | z before → after (oracle) | core z before → after (oracle) | usable before → after (oracle) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "yes" if b else "no"
    for r in rows:
        for e in r.get("passes", []):
            L.append(f"| {r['clip']} | {e['event']} | {e['kind']}{'' if e['straight'] else ' (curved)'} | "
                     f"{e['v']:.0f} | {e['d']:.0f} | {e['h']:.1f} | {'/'.join(f'{x:g}' for x in e['f0'])} | "
                     f"{_fnum(e['z_before'])} → {_fnum(e['z_after'])} ({_fnum(e['z_oracle'])}) | "
                     f"{_fnum(e['zc_before'])} → {_fnum(e['zc_after'])} ({_fnum(e['zc_oracle'])}) | "
                     f"{yn(e['usable_before'])} → **{yn(e['usable_after'])}** ({yn(e['usable_oracle'])}) |")
    L += ["", "## Per curve (aircraft clips)", "",
          "Only curves with any part inside 45–1940 Hz. Bold z after = still detectable (> 3).", "",
          "| clip | g/h | f range Hz | max slope Hz/s | width Hz | visible s | cov 1 / 3 Hz | core cov 1 / 3 Hz | time cov | z before → after (oracle) | core z before → after (oracle) | z_wide before → after (oracle) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for c in r.get("curves", []):
            za = c["z_after"]
            zs = f"**{_fnum(za)}**" if np.isfinite(za) and za > Z_DET and c["z_before"] > Z_DET else _fnum(za)
            zw = (f"{_fnum(c['z_wide_before'])} → {_fnum(c['z_wide_after'])} ({_fnum(c['z_wide_oracle'])})"
                  if "z_wide_before" in c else "")
            L.append(f"| {r['clip']} | {c['group']}/{c['harmonic']} | {c['f_lo']:.0f}–{c['f_hi']:.0f} | "
                     f"{c['max_slope_hz_s']:.0f} | {_fnum(c['width_hz'])} | {c['visible_s']:.0f} | "
                     f"{c['cov1']:.2f} / {c['cov3']:.2f} | {_fnum(c.get('core_cov1'), 2)} / {_fnum(c.get('core_cov3'), 2)} | "
                     f"{c['time_cov']:.2f} | "
                     f"{_fnum(c['z_before'])} → {zs} ({_fnum(c['z_oracle'])}) | "
                     f"{_fnum(c.get('zc_before'))} → {_fnum(c.get('zc_after'))} ({_fnum(c.get('zc_oracle'))}) | {zw} |")
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
    OUT_MD.write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["generate", "score", "report"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img", action="store_true", help="write docs/results/img/sh_<clip>_ba.png")
    ap.add_argument("--detector", help="mod:func replacing detect.detect_tracks(x, sr, cfg)")
    ap.add_argument("--scrub", help="mod:func replacing pipeline.scrub_array(x, sr) -> (y, tracks)")
    ap.add_argument("--tag", help="write docs/results/synth_hard_<tag>.{json,md}")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cached scrubbed audio/tracks (.cache/synth_hard/<tag>) and oracle output")
    a = ap.parse_args(argv)
    if a.cmd == "generate":
        from dopplerscrub.synth import make_hard_dataset
        import os
        os.chdir(ROOT)   # truth.json stores repo-relative paths
        recs = make_hard_dataset("data/synth_hard", only=a.only)
        print(f"{len(recs)} clips -> {DATA}/truth.json")
        return
    global OUT_JSON, OUT_MD
    if a.tag:
        OUT_JSON = OUT_JSON.with_name(f"synth_hard_{a.tag}.json")
        OUT_MD = OUT_MD.with_name(f"synth_hard_{a.tag}.md")
    recs = json.loads((DATA / "truth.json").read_text())
    for r in recs:   # paths are stored relative to the repo root
        r["path"] = str(Path(r["path"]).relative_to(ROOT)) if Path(r["path"]).is_absolute() else r["path"]
    res = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"clips": []}
    if a.cmd == "score":
        todo = [r for r in recs if not a.only or r["name"] in a.only]
        todo.sort(key=lambda r: -r["duration_s"])   # longest first: better wall time
        if a.img:
            IMG.mkdir(parents=True, exist_ok=True)
        done = {r["clip"]: r for r in res["clips"]}
        with ProcessPoolExecutor(a.workers) as ex:
            futs = {ex.submit(score_clip, r, a.img, a.detector, a.scrub, a.tag, a.reuse): r["name"] for r in todo}
            from concurrent.futures import as_completed
            for fu in as_completed(futs):
                r = fu.result()
                done[r["clip"]] = r
                print(f"{r['clip']}: usable {r.get('usable_before')} -> {r.get('usable_after')} "
                      f"(oracle {r.get('usable_oracle')}); {verdict(r)} ({r['runtime_s']:.0f} s)", flush=True)
        order = [r["name"] for r in recs]
        res["clips"] = [done[n] for n in order if n in done]
        res["time"] = time.strftime("%Y-%m-%d %H:%M")
        res["detector"] = a.scrub or a.detector or "default (pipeline.scrub_array)"
        import subprocess
        res["commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip()
    res["descriptions"] = [{k: r[k] for k in ("name", "category", "duration_s", "description")} for r in recs]
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
    write_md(res)


if __name__ == "__main__":
    main()
