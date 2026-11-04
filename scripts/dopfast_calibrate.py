#!/usr/bin/env python
"""Calibrate the dopfast EVENT attacker (evaluate.dopfast_event_list) on null clips and
synthetic passes only (never heldout*), then the thresholds are frozen in evaluate.py.

Null material (no Doppler pass):
  - synth_hard null_*.wav (music, birds, traffic, generator with RPM drift, ambience);
  - every fixed-wing DEV clip's spectrogram time-reversed (rising curves: not a pass) and
    frame-shuffled twice (per-frame spectra kept, curves destroyed).
Aircraft material:
  - synth_hard aircraft clips with exact truth (per event: d = sqrt(lateral^2 + alt^2),
    h = t_cpa + d / c), events matched within +-3 s of h;
  - plateau-only passes (ours): straight passes whose steep core |w| < 2 d / v is faded out,
    i.e. what a remover that only takes the CPA leaves. These must be detected-but-unusable or
    not detected.

Usage:
  uv run python scripts/dopfast_calibrate.py --synth-hard <dir with truth.json and WAVs>
Writes docs/results/dopfast_calibration.json and prints the FA curve and usability table.
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
import baselines as bl  # noqa: E402
import new_ridges as nr  # noqa: E402

RES = ROOT / "docs" / "results"


def events_all(E, P, t, lg):
    """Every grouped peak (tau = -inf) with its usability features."""
    return ev.dopfast_event_list(E, P, t, lg, tau=-1e9)


def plateau_pass(sr, dur, v, d, h, f0, amp, noise, seed):
    n = int(dur * sr)
    tt = np.arange(n) / sr
    g = ev.dopfast_g(tt - h, v, d)
    ph = 2 * np.pi * np.cumsum(f0 * g) / sr
    sig = sum(amp / j * np.cos(j * ph) for j in (1, 2, 3))
    w = tt - h
    core = 2.0 * d / v
    fade = np.clip((np.abs(w) - core) / 1.0, 0, 1)               # 0 inside the core, 1 s ramp
    rng = np.random.default_rng(seed)
    return sig * fade + noise * rng.standard_normal(n), sig + noise * rng.standard_normal(n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-hard", required=True)
    a = ap.parse_args(argv)
    sh = Path(a.synth_hard)
    truth = json.loads((sh / "truth.json").read_text())
    rng = np.random.default_rng(0)
    out = {"nulls": [], "aircraft": [], "plateau": []}

    def vol(x, sr):
        S, t, lg = ev._df_spectrogram(ev._mono(x).astype(np.float64), sr, ev.FMIN, 1000.0)
        return S, t, lg

    # ---- nulls
    for c in truth:
        if not c.get("null"):
            continue
        x, sr = bl.load(sh / Path(c["path"]).name)
        S, t, lg = vol(x, sr)
        E, P = ev.dopfast_evidence(S, lg)
        out["nulls"].append({"name": c["name"], "kind": "synth_null", "minutes": (t[-1] - t[0]) / 60,
                             "events": events_all(E, P, t, lg)})
        print("null", c["name"], flush=True)
    for c in nr.dev_real_clips():
        x, sr = bl.load(c["path"])
        S, t, lg = vol(x, sr)
        for kind, Sx in [("reversed", S[::-1])] + [(f"shuffled{k}", S[rng.permutation(S.shape[0])])
                                                   for k in range(2)]:
            E, P = ev.dopfast_evidence(Sx, lg)
            out["nulls"].append({"name": c["name"], "kind": kind, "minutes": (t[-1] - t[0]) / 60,
                                 "events": events_all(E, P, t, lg)})
        print("null dev", c["name"], flush=True)
    # ---- aircraft (synth_hard truth)
    for c in truth:
        if c.get("null"):
            continue
        x, sr = bl.load(sh / Path(c["path"]).name)
        S, t, lg = vol(x, sr)
        E, P = ev.dopfast_evidence(S, lg)
        evs = events_all(E, P, t, lg)
        tr = []
        for e in c.get("events", []):
            tj = e["traj"]
            if tj.get("kind") != "straight":
                tr.append({"kind": tj.get("kind")})
                continue
            d = float(np.hypot(tj["lateral"], tj["alt"]))
            tr.append({"kind": "straight", "v": tj["v"], "d": d, "h": tj["t_cpa"] + d / ev.DF_C, "f0": e.get("f0")})
        out["aircraft"].append({"name": c["name"], "truth": tr, "events": evs})
        print("aircraft", c["name"], flush=True)
    # ---- plateau-only passes (and their full versions)
    sr = 8000
    for k, (v, d) in enumerate([(50, 160), (70, 320), (100, 640), (140, 1280), (45, 80), (90, 2560)]):
        xp, xf = plateau_pass(sr, 90, v, d, 45.0, 120.0, 0.004, 0.01, k)
        for kind, x in (("plateau", xp), ("full", xf)):
            S, t, lg = vol(x, sr)
            E, P = ev.dopfast_evidence(S, lg)
            out["plateau"].append({"v": v, "d": d, "h": 45.0 + d / ev.DF_C, "kind": kind,
                                   "events": events_all(E, P, t, lg)})
        print("plateau", v, d, flush=True)
    (RES / "dopfast_calibration.json").write_text(json.dumps(out, indent=1, default=float))
    summarize(out)


def summarize(out, taus=(6, 7, 8, 9, 10, 11, 12, 14)):
    mins = sum(n["minutes"] for n in out["nulls"])
    print(f"\nnull material: {len(out['nulls'])} spectrograms, {mins:.1f} min")
    for tau in taus:
        fa = sum(1 for n in out["nulls"] for e in n["events"] if e["E"] >= tau)
        fu = sum(1 for n in out["nulls"] for e in n["events"] if e["E"] >= tau and e["usable"])
        det = []
        for c in out["aircraft"]:
            for tr in c["truth"]:
                if tr.get("kind") != "straight":
                    continue
                m = [e for e in c["events"] if abs(e["s"] - tr["h"]) <= 3 and e["E"] >= tau]
                det.append((bool(m), bool(m) and m[0]["usable"]))
        pl = [p for p in out["plateau"] if p["kind"] == "plateau"]
        fl = [p for p in out["plateau"] if p["kind"] == "full"]
        pu = sum(1 for p in pl if any(abs(e["s"] - p["h"]) <= 6 and e["E"] >= tau and e["usable"] for e in p["events"]))
        fu2 = sum(1 for p in fl if any(abs(e["s"] - p["h"]) <= 3 and e["E"] >= tau and e["usable"] for e in p["events"]))
        print(f"tau {tau:5.1f}: null events {fa} ({fa / mins:.3f}/min), usable {fu} ({fu / mins:.3f}/min) | "
              f"synth_hard straight passes detected {sum(d for d, _ in det)}/{len(det)}, usable "
              f"{sum(u for _, u in det)} | full synthetic usable {fu2}/{len(fl)}, plateau-only usable {pu}/{len(pl)}")


if __name__ == "__main__":
    main()
