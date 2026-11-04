#!/usr/bin/env python
"""Diagnose the box-free "new ridges" (z>3 ridges found only in the cleaned file).

DEV clips only (never data/real/heldout*). For every method (dopplerscrub and the
blanket baselines) and clip, every ridge the box-free extractor finds in the cleaned
view that is not near an original ridge is measured on BOTH files along the same curve:

- z_clean / z_orig: integrated on-curve z (`evaluate._curve_vs_floor`) in the cleaned
  and in the ORIGINAL raw spectrogram. z_orig > 3 means an attacker integrating along
  that curve in the original already saw it, so cleaning did not create it.
- d_on_db / d_flank_db: mean raw power change (clean - orig, dB) on the curve and on its
  +-3..6 Hz flanks. d_on > 0: cleaning ADDED energy on the curve. d_flank << d_on:
  the flanks were notched, which raises z without adding anything.
- floor_shift_db: change of the per-bin whitening baseline at the curve's bins (clean -
  orig). Negative = the floor dropped, so the same content reads brighter.
- sm_orig_db: the original's 2 s smoothed whitened dB along the curve (4 dB = extractor
  threshold): 2-4 dB = sub-threshold original ridge.
- in_orig_mask: share of curve frames that were inside the original's persistent mask
  (i.e. the cells were part of an original component; the extractor only kept one path
  per component).
- corr_hz: median distance (Hz) to the nearest removed corridor (dopplerscrub's
  doppler tracks; nan for baselines).

Classes (first that applies):
  pre-existing  z_orig > 3 and d_on_db <= 1 dB          (measurement artefact)
  notch-raised  d_on_db <= 1, d_flank_db < d_on_db - 2   (flanks removed; content old)
  added         d_on_db > 1 dB                            (cleaning added energy: artefact)
  floor-shift   everything else                           (whitening/threshold effect)

Usage:
  uv run python scripts/new_ridges.py                      # all dev clips, all methods
  uv run python scripts/new_ridges.py --only prop_pass --methods dopplerscrub
Writes docs/results/new_ridges.json and docs/results/img/nr_<clip>__<method>.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dopplerscrub import evaluate as ev  # noqa: E402
import boxfree as bf  # noqa: E402
import baselines as bl  # noqa: E402
import benchmark as bm  # noqa: E402

RES = ROOT / "docs" / "results"
IMG = RES / "img"
METHODS = ["dopplerscrub", "bandstop_brick", "noise", "jitter"]


def dev_real_clips():
    d = ROOT / "data" / "real"
    out = [{"name": n, "path": d / f"{n}.wav", "true": None, "boxes": []}
           for n in ("prop_pass", "prop_flyover", "colsto_2050_2250", "colsto_2980_3170")]
    out += [{"name": c["name"], "path": c["path"], "true": None, "boxes": c["boxes"]}
            for c in bm.airshow_clips()]
    # fixed-wing aircraft only (research scope); helicopters are out of scope
    out = [c for c in out if "helicopter" not in c["name"]]
    for c in out:   # hard rule: never touch the held-out sets here
        assert "heldout" not in str(c["path"]), c
    return [c for c in out if Path(c["path"]).exists()]


def scrub_tracks(x, sr, name):
    """dopplerscrub's removed tracks (cached), to locate corridors."""
    p = bl.WORK / f"{name}__tracks.json"
    if p.exists():
        return [(np.array(a), np.array(b)) for a, b in json.loads(p.read_text())]
    from dopplerscrub.pipeline import scrub_array
    from dopplerscrub.remove import RemoveConfig
    cfg = RemoveConfig(harden=True)
    _, tracks = scrub_array(x, sr, remove_cfg=cfg)
    tracks = getattr(tracks, "tracks", tracks)
    tr = [(list(map(float, t.t)), list(map(float, t.f))) for t in tracks if t.kind in cfg.remove_kinds]
    p.write_text(json.dumps(tr))
    return [(np.array(a), np.array(b)) for a, b in tr]


def _at(arr, t, f, tt, ff):
    """arr (F x T) at nearest frame, max of the two nearest bins (as _lin_on_curve)."""
    ti = np.clip(np.round((tt - t[0]) / (t[1] - t[0])).astype(int), 0, arr.shape[1] - 1)
    fi = (ff - f[0]) / (f[1] - f[0])
    j0 = np.clip(np.floor(fi).astype(int), 0, arr.shape[0] - 2)
    return np.maximum(arr[j0, ti], arr[j0 + 1, ti]), ti, j0


def _db_mean(lin):
    lin = lin[np.isfinite(lin)]
    return float(10 * np.log10(np.mean(lin))) if len(lin) else float("nan")


def measure(v, r, sm_orig, mask_orig, corridors, Ba, Bb):
    t, f = v["orig"].t, v["orig"].f
    tt = np.arange(r.t0, r.t1, ev.HOP_S)
    ff = np.interp(tt, r.t, r.f)
    La, Lb = v["raw_orig"], v["raw_clean"]
    a_on, ti, j0 = _at(La, t, f, tt, ff)
    b_on, _, _ = _at(Lb, t, f, tt, ff)
    fl_a, fl_b = [], []
    for o in np.linspace(3, 6, 4):
        for s in (-1, 1):
            fl_a.append(_at(La, t, f, tt, ff + s * o)[0]); fl_b.append(_at(Lb, t, f, tt, ff + s * o)[0])
    lin = lambda d: 10 ** (np.asarray(d) / 10)
    d_on = _db_mean(lin(b_on)) - _db_mean(lin(a_on))
    d_fl = _db_mean(lin(np.concatenate(fl_b))) - _db_mean(lin(np.concatenate(fl_a)))
    shift = float(np.median(Bb[j0, ti] - Ba[j0, ti]))
    sm = float(np.median(_at(sm_orig, t, f, tt, ff)[0]))
    inm = float(np.mean(mask_orig[j0, ti] | mask_orig[j0 + 1, ti]))
    corr = float("nan")
    if corridors is not None:
        d = np.full(len(tt), np.inf)
        for ct, cf in corridors:
            ok = (tt >= ct[0]) & (tt <= ct[-1])
            if ok.any():
                d[ok] = np.minimum(d[ok], np.abs(np.interp(tt[ok], ct, cf) - ff[ok]))
        corr = float(np.median(d))  # inf = no corridor active over most of the ridge
    cc = ev._curve_vs_floor(Lb, t, f, [r]); co = ev._curve_vs_floor(La, t, f, [r])
    z_c, z_o = cc["z"], co["z"]
    return {"excess_db_orig": co["excess_db"], "excess_db_clean": cc["excess_db"],
            "null_sd_db_orig": co.get("null_sd_db", float("nan")),
            "null_sd_db_clean": cc.get("null_sd_db", float("nan")),"t0": float(r.t0), "t1": float(r.t1), "f_min": float(r.f.min()), "f_max": float(r.f.max()),
            "z_clean": z_c, "z_orig": z_o, "d_on_db": d_on, "d_flank_db": d_fl,
            "floor_shift_db": shift, "sm_orig_db": sm, "in_orig_mask": inm, "corr_hz": corr}


def classify(m):
    if np.isfinite(m["z_orig"]) and m["z_orig"] > 3 and m["d_on_db"] <= 1.0:
        return "pre-existing"
    if m["d_on_db"] <= 1.0 and m["d_flank_db"] < m["d_on_db"] - 2.0:
        return "notch-raised"
    if m["d_on_db"] > 1.0:
        return "added"
    return "floor-shift"


COLORS = {"pre-existing": "#7CFC00", "notch-raised": "#ffd400", "added": "#ff2a2a", "floor-shift": "#00e5ff"}


def render(name, method, v, r0, new, corridors, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(ev.FIG_W, 7.5), sharex=True)
    for ax, key, title in ((axes[0], "orig", "original (white = original ridges)"),
                           (axes[1], "clean", f"{method} (new z>3 ridges by class; dashed = removed corridors)")):
        sm = ev.smooth_along_time(v[key], bf.SMOOTH)
        ax.imshow(sm.S, origin="lower", aspect="auto", cmap="magma", vmin=-2, vmax=10,
                  extent=[sm.t[0], sm.t[-1], sm.f[0], sm.f[-1]], interpolation="antialiased")
        for r in r0:
            ax.plot(r.t, r.f, color="white", lw=0.8, alpha=0.8)
        ax.set_title(f"{name}: {title}", fontsize=9)
        ax.set_ylabel("Hz")
    for r, m in new:
        if m["z_clean"] > 3:
            for ax in axes:
                ax.plot(r.t, r.f, color=COLORS[m["class"]], lw=1.6, ls="-" if ax is axes[1] else ":")
    for ct, cf in corridors or []:
        axes[1].plot(ct, cf, color="#9ad0ff", lw=0.6, ls="--", alpha=0.7)
    axes[1].set_ylim(v["orig"].f[0], v["orig"].f[-1])
    axes[1].set_xlabel("s")
    handles = [plt.Line2D([], [], color=c, lw=2, label=k) for k, c in COLORS.items()]
    axes[1].legend(handles=handles, loc="upper right", fontsize=7)
    ev.save_png(fig, out)
    plt.close(fig)


def run_one(c, method, fmax):
    x, sr = bl.load(c["path"])
    y, _ = bl.cached_apply(method, x, sr, c)
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    corridors = scrub_tracks(x, sr, c["name"]) if method == "dopplerscrub" else None
    v = ev.attack_views(x, y, sr, fmax=fmax)
    t, f = v["orig"].t, v["orig"].f
    Ba, Bb = v["raw_orig"] - v["orig"].S, v["raw_clean"] - v["clean"].S
    sm_orig = ev.smooth_along_time(v["orig"], bf.SMOOTH).S
    mask_orig = sm_orig > bf.PERSIST_DB
    r0 = bf.ridge_curves(v["orig"]); r1 = bf.ridge_curves(v["clean"])
    # sub-threshold original ridges: same extractor at 2.5 dB
    r0_low = bf.ridge_curves(v["orig"], persist_db=2.5)
    new = [r for r in r1 if not bf._near(r, r0)]
    meas = []
    for r in new:
        m = measure(v, r, sm_orig, mask_orig, corridors, Ba, Bb)
        m["class"] = classify(m)
        m["near_orig_8hz"] = bool(bf._near(r, r0, tol_hz=8.0))
        m["near_orig_lowthr"] = bool(bf._near(r, r0_low))
        meas.append((r, m))
    z3 = [m for _, m in meas if np.isfinite(m["z_clean"]) and m["z_clean"] > 3]
    # original ridges that got STRONGER (z up by > 50% and > 3 after cleaning)
    stronger = []
    for r in r0:
        m = measure(v, r, sm_orig, mask_orig, corridors, Ba, Bb)
        if np.isfinite(m["z_clean"]) and m["z_clean"] > 3 and m["z_clean"] > 1.5 * max(m["z_orig"], 1.0):
            m["class"] = classify(m)
            stronger.append(m)
    render(c["name"], method, v, r0, meas, corridors, IMG / f"nr_{c['name']}__{method}.png")
    return {"clip": c["name"], "method": method, "n_orig": len(r0), "n_clean": len(r1),
            "n_new": len(new), "n_new_z3": len(z3),
            "classes": {k: sum(1 for m in z3 if m["class"] == k) for k in COLORS},
            "new_z3": z3, "stronger": stronger}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    a = ap.parse_args(argv)
    IMG.mkdir(parents=True, exist_ok=True)
    jp = RES / "new_ridges.json"
    rows = json.loads(jp.read_text()) if jp.exists() else []
    for c in dev_real_clips():
        if a.only and c["name"] not in a.only:
            continue
        fmax = ev.FMAX   # box-free runs on the stock 40-450 Hz view (as benchmark.py)
        for m in a.methods:
            r = run_one(c, m, fmax)
            print(f"{c['name']:28s} {m:15s} orig {r['n_orig']:3d} new {r['n_new']:3d} "
                  f"new z>3 {r['n_new_z3']:3d} {r['classes']} stronger {len(r['stronger'])}", flush=True)
            rows = [q for q in rows if not (q["clip"] == r["clip"] and q["method"] == m)] + [r]
            jp.write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
