"""Box-free, detector-independent visibility checks for clips without annotations.

The held-out clips have no truth tracks and no event boxes, and scoring them along
dopplerscrub's OWN detections would be circular (anything the detector missed would
never be measured). These checks look at the attacker view only:

- **hi cells**: share of whitened cells above +8 dB over the whole plane (noise ~0.5%).
- **persistent cells**: share of cells above PERSIST_DB in the 2 s time-smoothed view
  (a ridge that lasts; noise alone almost never passes).
- **attacker ridges**: an independent, deliberately simple ridge extractor (connected
  components of persistent cells, >= MIN_DUR_S long, sweeping >= MIN_SWEEP_HZ) run on
  the ORIGINAL. Each ridge becomes a curve f(t), and the integrated on-curve z
  (`evaluate._curve_vs_floor`, the same statistic as the AF column) is measured on the
  original and on the cleaned file. z > 3 after cleaning = still detectable by an
  integrating attacker. Ridges extracted from the CLEANED file that lie away from every
  original ridge are "new" (n_new_z3), and each new z>3 ridge is split by measuring the
  SAME curve in the original (docs/results/new_ridges.md has the evidence):
    seen      z > 3 along that curve in the original already (the extractor keeps one
              path per connected component, so it was hidden inside a bigger one);
    unmasked  not z > 3 before, but >= 50% of its frames were inside the original's
              persistent mask and its on-curve power did not rise (<= +1 dB): original
              content made separable by removing its neighbours (its flanks and the
              +-8..14 Hz null controls got quieter, which raises z);
    created   everything else: the processing put a ridge where the original had none
              (added energy, or a ridge in a band the original never lit up).
  Only `created` is an artefact of the processing; still + new = z>3 ridges left.
- **stronger**: original ridges whose excess over their +-3..6 Hz flanks rose by > 3 dB.
  (z itself is not compared: removing neighbours collapses the null spread, so z of an
  untouched ridge can rise 5-10x with its contrast unchanged.)

Thresholds were calibrated on white noise (0 ridges, persistent share ~0) and on the
synthetic set (`calibrate()`): at 4 dB / 6 s, 600 s of white noise yields 0 ridges, 4.6%
persistent cells and 1.3% hi cells. z_orig along an extracted ridge is biased upward
(the ridge was picked where the original is bright); z_clean is the number to read.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from dopplerscrub import evaluate as ev
from dopplerscrub.types import Track

PERSIST_DB = 4.0      # smoothed whitened dB; 3 dB lets noise-only ridges through (calibrated)
SMOOTH = 9            # frames (2 s)
MIN_DUR_S = 6.0
MIN_SWEEP_HZ = 3.0


def ridge_curves(spec, persist_db=PERSIST_DB, min_dur_s=MIN_DUR_S, min_sweep_hz=MIN_SWEEP_HZ):
    """Sweeping, persistent ridges in an attacker view, as Tracks (kind 'ridge')."""
    sm = ev.smooth_along_time(spec, SMOOTH)
    mask = sm.S > persist_db
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    out = []
    if n == 0:
        return out
    sl = ndimage.find_objects(lab)
    for k, s in enumerate(sl, 1):
        fs, ts = s
        dur = (ts.stop - ts.start) * spec.dt
        if dur < min_dur_s:
            continue
        sub = np.where(lab[fs, ts] == k, sm.S[fs, ts], -np.inf)
        cols = np.isfinite(sub.max(axis=0))
        j = np.argmax(sub, axis=0)
        t = spec.t[ts][cols]
        f = spec.f[fs][j[cols]]
        if len(t) < 8:
            continue
        f = ndimage.median_filter(f, size=9, mode="nearest")
        if np.ptp(f) < min_sweep_hz:
            continue
        out.append(Track(t=t.astype(float), f=f.astype(float), amp=np.ones(len(t)), kind="ridge"))
    return out


def _near(a: Track, others, tol_hz=4.0):
    for b in others:
        lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
        if hi - lo < 2.0:
            continue
        g = np.arange(lo, hi, 0.25)
        if np.median(np.abs(np.interp(g, a.t, a.f) - np.interp(g, b.t, b.f))) < tol_hz:
            return True
    return False


UNMASK_FRAC = 0.5     # share of frames inside the original's persistent mask
UNMASK_DON_DB = 1.0   # max on-curve raw power rise for "unmasked"
STRONGER_DB = 3.0     # excess rise (dB over flanks) that counts as "stronger"


def _curve_idx(t, f, r):
    tt = np.arange(r.t0, r.t1, ev.HOP_S)
    ff = np.interp(tt, r.t, r.f)
    ti = np.clip(np.round((tt - t[0]) / (t[1] - t[0])).astype(int), 0, len(t) - 1)
    j0 = np.clip(np.floor((ff - f[0]) / (f[1] - f[0])).astype(int), 0, len(f) - 2)
    return ti, j0


def on_curve_change_db(v, r):
    """Mean raw on-curve power change (clean - orig, dB; max of the two nearest bins)."""
    t, f = v["orig"].t, v["orig"].f
    ti, j0 = _curve_idx(t, f, r)
    p = lambda L: float(np.mean(10 ** (np.maximum(L[j0, ti], L[j0 + 1, ti]) / 10)))
    return float(10 * np.log10(p(v["raw_clean"]) / p(v["raw_orig"])))


def mask_fraction(mask, t, f, r):
    ti, j0 = _curve_idx(t, f, r)
    return float(np.mean(mask[j0, ti] | mask[j0 + 1, ti]))


def classify_new(v, r, mask_orig):
    """'seen' / 'unmasked' / 'created' for a ridge found only in the cleaned view."""
    t, f = v["orig"].t, v["orig"].f
    z_o = ev._curve_vs_floor(v["raw_orig"], t, f, [r])["z"]
    if np.isfinite(z_o) and z_o > 3:
        return "seen"
    if mask_fraction(mask_orig, t, f, r) >= UNMASK_FRAC and on_curve_change_db(v, r) <= UNMASK_DON_DB:
        return "unmasked"
    return "created"


def boxfree(x, y, sr, fmax=ev.FMAX, views=None):
    v = views or ev.attack_views(x, y, sr, fmax=fmax)
    t, f = v["orig"].t, v["orig"].f
    out = {"fmax": fmax}
    for key in ("orig", "clean"):
        S = v[key].S
        out[f"hi_{key}"] = float(np.mean(S > 8.0))
        out[f"p99_{key}"] = float(np.percentile(S, 99))
        out[f"persist_{key}"] = float(np.mean(ev.smooth_along_time(v[key], SMOOTH).S > PERSIST_DB))
    r0 = ridge_curves(v["orig"])
    r1 = ridge_curves(v["clean"])
    c0 = [ev._curve_vs_floor(v["raw_orig"], t, f, [r]) for r in r0]
    c1 = [ev._curve_vs_floor(v["raw_clean"], t, f, [r]) for r in r0]
    z0 = [c["z"] for c in c0]; z1 = [c["z"] for c in c1]
    new = [r for r in r1 if not _near(r, r0)]
    zn = [ev._curve_vs_floor(v["raw_clean"], t, f, [r])["z"] for r in new]
    mask_orig = ev.smooth_along_time(v["orig"], SMOOTH).S > PERSIST_DB
    kinds = [classify_new(v, r, mask_orig) if np.isfinite(z) and z > 3 else None
             for r, z in zip(new, zn)]
    stronger = sum(1 for a, b in zip(c0, c1) if np.isfinite(b["z"]) and b["z"] > 3
                   and np.isfinite(b["excess_db"]) and b["excess_db"] - a["excess_db"] > STRONGER_DB)
    det0 = [i for i, z in enumerate(z0) if np.isfinite(z) and z > 3]
    out.update({
        "n_ridges_orig": len(r0), "ridge_s_orig": float(sum(r.t1 - r.t0 for r in r0)),
        "n_ridges_orig_z3": len(det0),
        "n_still_z3": int(sum(1 for i in det0 if np.isfinite(z1[i]) and z1[i] > 3)),
        "z_orig_med": float(np.median([z0[i] for i in det0])) if det0 else float("nan"),
        "z_clean_med": float(np.median([z1[i] for i in det0])) if det0 else float("nan"),
        "n_ridges_clean": len(r1), "n_new_ridges": len(new),
        "n_new_z3": int(sum(1 for z in zn if np.isfinite(z) and z > 3)),
        "n_new_created_z3": kinds.count("created"),
        "n_new_unmasked_z3": kinds.count("unmasked"),
        "n_new_seen_z3": kinds.count("seen"),
        "n_stronger": int(stronger),
        "ridges": [{"t0": r.t0, "t1": r.t1, "f_min": float(r.f.min()), "f_max": float(r.f.max()),
                    "z_orig": z0[i], "z_clean": z1[i], "excess_db_orig": c0[i]["excess_db"],
                    "excess_db_clean": c1[i]["excess_db"]} for i, r in enumerate(r0)],
        "new_ridges": [{"t0": r.t0, "t1": r.t1, "f_min": float(r.f.min()), "f_max": float(r.f.max()),
                        "z_clean": z, "kind": k} for r, z, k in zip(new, zn, kinds) if k],
    })
    return out


def calibrate(sr=16000, dur=600, seed=0):
    """Noise-only reference values (printed by `python scripts/boxfree.py`)."""
    x = np.random.default_rng(seed).standard_normal(dur * sr).astype(np.float32) * 0.05
    return {k: v for k, v in boxfree(x, x, sr).items() if k != "ridges"}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    print(calibrate())
