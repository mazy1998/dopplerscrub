#!/usr/bin/env python
"""Why does dopplerscrub fail the integrated test? Per truth curve: detected? labelled
doppler? filled by harden? residual? time coverage vs frequency accuracy? Plus a
damage attribution for real prop_pass. Writes docs/results/detector_diagnosis.json;
the markdown write-up is docs/results/detector_diagnosis.md.

Usage: uv run python scripts/diagnose_detector.py [--clips speech stereo ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal as ss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dopplerscrub import evaluate as ev  # noqa: E402
from dopplerscrub.pipeline import scrub_array  # noqa: E402
from dopplerscrub.remove import remove_tracks  # noqa: E402
from dopplerscrub.ridges import ridge_spectrogram, track_ridges  # noqa: E402
from dopplerscrub.types import DetectConfig, RemoveConfig, Track  # noqa: E402

OUT = ROOT / "docs" / "results" / "detector_diagnosis.json"


def load(p):
    from dopplerscrub import io
    x, sr, _ = io.load(str(p))
    return np.asarray(x, np.float32), int(sr)


def visible_frames(spec, tr, thr_db=1.5, smooth=9):
    """Frames (0.25 s grid) where the true curve stands out in the ORIGINAL attacker view:
    2 s-smoothed whitened dB at f(t) (max of +-1 bin) > thr_db."""
    sm = ev.smooth_along_time(spec, smooth)
    tt = np.arange(tr.t[0], tr.t[-1], 0.25)
    ff = np.interp(tt, tr.t, tr.f)
    v = np.nanmax(np.stack([sm.sample(tt, ff + o) for o in (-spec.df, 0, spec.df)]), axis=0)
    return tt, ff, np.isfinite(v) & (v > thr_db)


def coverage(tt, ff, vis, tracks, tol):
    """Union coverage of visible frames by any track within tol Hz; also the best single
    track (most covered frames) and its RMS error over its own overlap."""
    if not vis.any():
        return 0.0, None
    cov = np.zeros(len(tt), bool)
    best, best_n = None, 0
    for q in tracks:
        if len(q.t) < 2:
            continue
        inside = (tt >= q.t[0]) & (tt <= q.t[-1])
        fq = np.interp(tt, q.t, q.f)
        hit = inside & (np.abs(fq - ff) < tol)
        cov |= hit
        n = int((hit & vis).sum())
        if n > best_n:
            ov = inside & vis
            rms = float(np.sqrt(np.mean((fq[ov] - ff[ov]) ** 2))) if ov.any() else float("nan")
            best_n, best = n, {"kind": q.kind, "score": float(q.doppler_score), "harmonic": int(q.harmonic),
                               "group": int(q.group), "t0": q.t0, "t1": q.t1,
                               "cov": n / vis.sum(), "rms_hz_overlap": rms,
                               "time_cov": float((inside & vis).sum() / vis.sum())}
    return float((cov & vis).sum() / vis.sum()), best


def curve_z(raw, t, f, tr):
    return ev._curve_vs_floor(raw, t, f, [tr])["z"]


def diagnose_synth(name, path, truth):
    x, sr = load(path)
    dcfg, rcfg = DetectConfig(), RemoveConfig()
    t0 = time.time()
    y, found = scrub_array(x, sr, dcfg, rcfg)
    rt = time.time() - t0
    S, rt_t, rt_f = ridge_spectrogram(x, sr, dcfg)
    raw_ridges = track_ridges(S, rt_t, rt_f, dcfg)
    truth_d = [Track(q.t, q.f, q.amp, 1.0, q.fit, q.harmonic, q.group, "doppler") for q in truth]
    yo = remove_tracks(x, sr, truth_d, RemoveConfig())
    v = ev.attack_views(x, y, sr, fmax=ev.FMAX_WIDE)
    vo = ev.attack_views(x, yo, sr, fmax=ev.FMAX_WIDE)
    t, f = v["orig"].t, v["orig"].f
    dop = [q for q in found if q.kind in rcfg.remove_kinds]
    filled = [q for q in dop if q.doppler_score >= rcfg.harden_min_score]
    rows = []
    for k, tr in enumerate(truth):
        z0 = curve_z(v["raw_orig"], t, f, tr)
        tt, ff, vis = visible_frames(v["orig"], tr)
        row = {"clip": name, "curve": k, "harmonic": int(tr.harmonic), "group": int(tr.group),
               "f_range": [float(np.min(tr.f)), float(np.max(tr.f))], "z_before": z0,
               "visible_s": float(vis.sum() * 0.25),
               "z_after": curve_z(v["raw_clean"], t, f, tr), "z_oracle": curve_z(vo["raw_clean"], t, f, tr)}
        res = ev.residual_details(x, y, sr, [tr])
        row["residual_db"] = res[0]["residual_db"] if res else float("nan")
        row["cov_ridges_1hz"], row["best_ridge"] = coverage(tt, ff, vis, raw_ridges, 1.0)
        row["cov_any_1hz"], row["best_any"] = coverage(tt, ff, vis, found, 1.0)
        row["cov_any_3hz"], _ = coverage(tt, ff, vis, found, 3.0)
        row["cov_doppler_1hz"], row["best_doppler"] = coverage(tt, ff, vis, dop, 1.0)
        row["cov_doppler_3hz"], _ = coverage(tt, ff, vis, dop, 3.0)
        row["cov_filled_1hz"], _ = coverage(tt, ff, vis, filled, 1.0)
        rows.append(row)
    summary = {"clip": name, "runtime_s": rt, "n_found": len(found), "n_doppler": len(dop),
               "n_filled": len(filled), "n_raw_ridges": len(raw_ridges),
               "raw_ridge_kinds": {k: sum(q.kind == k for q in raw_ridges) for k in {q.kind for q in raw_ridges}},
               "found_kinds": {k: sum(q.kind == k for q in found) for k in {q.kind for q in found}},
               "doppler_tracks": [{"t0": q.t0, "t1": q.t1, "f0": float(q.f[0]), "f1": float(q.f[-1]),
                                   "score": float(q.doppler_score), "h": int(q.harmonic), "g": int(q.group)}
                                  for q in dop]}
    return rows, summary


def diagnose_prop_pass():
    ann = json.loads((ROOT / "data/real/annotations.json").read_text())
    pp = next(c for c in ann["clips"] if c["path"].endswith("prop_pass.wav"))
    ev_box = pp["events"][0]
    x, sr = load(ROOT / pp["path"])
    rcfg = RemoveConfig()
    y, found = scrub_array(x, sr, DetectConfig(), rcfg)
    a, b = ev._align(ev._mono(x).astype(np.float64), ev._mono(y).astype(np.float64))
    d = a - b
    n = 1 << int(np.ceil(np.log2(0.5 * sr)))              # ~0.68 s, ~1.5 Hz bins
    fq, tq, A = ss.stft(a, sr, nperseg=n, noverlap=n * 3 // 4)
    _, _, D = ss.stft(d, sr, nperseg=n, noverlap=n * 3 // 4)
    PA, PD = np.abs(A) ** 2, np.abs(D) ** 2
    dop = [q for q in found if q.kind in rcfg.remove_kinds]
    pad, W = rcfg.pad_s, rcfg.width_hz
    union = np.zeros(PD.shape, bool)
    per = []
    for q in dop:
        m = np.zeros(PD.shape, bool)
        tm = (tq >= q.t0 - pad) & (tq <= q.t1 + pad)
        fc = np.interp(tq[tm], q.t, q.f)
        m[:, tm] = np.abs(fq[:, None] - fc[None]) <= W + 1.5   # corridor +- (W + 1 bin)
        union |= m
        in_box = (q.t1 > ev_box["start_s"]) and (q.t0 < ev_box["end_s"]) and \
                 (np.median(q.f) >= ev_box["fmin_hz"]) and (np.median(q.f) <= ev_box["fmax_hz"])
        sweep = float(np.max(q.f) - np.min(q.f))
        per.append({"t0": q.t0, "t1": q.t1, "dur": q.t1 - q.t0, "f_med": float(np.median(q.f)),
                    "sweep_hz": sweep, "score": float(q.doppler_score), "h": int(q.harmonic),
                    "g": int(q.group), "in_event": bool(in_box),
                    "removed_energy": float(PD[m].sum())})
    tot = PD.sum()
    sp = (fq >= 300) & (fq <= 3400)
    out = {
        "n_found": len(found), "n_doppler": len(dop),
        "found_kinds": {k: sum(q.kind == k for q in found) for k in {q.kind for q in found}},
        "removed_share_in_corridors": float(PD[union].sum() / tot),
        "removed_share_outside_corridors": float(PD[~union].sum() / tot),
        "speech_band_removed_share_outside_corridors": float(PD[sp][~union[sp]].sum() / max(PD[sp].sum(), 1e-30)),
        "speech_band_snr_db": float(10 * np.log10(PA[sp].sum() / PD[sp].sum())),
        "corridor_tf_share": float(union.mean()),
        "removed_vs_orig_db_outside": float(10 * np.log10(PD[~union].sum() / PA[~union].sum())),
        "removed_vs_orig_db_inside": float(10 * np.log10(PD[union].sum() / PA[union].sum())),
    }
    for p_ in per:
        p_["removed_share"] = p_["removed_energy"] / tot
    out["tracks"] = sorted(per, key=lambda r: -r["removed_share"])
    out["share_by_event"] = {"in_event": float(sum(p_["removed_share"] for p_ in per if p_["in_event"])),
                             "outside_event": float(sum(p_["removed_share"] for p_ in per if not p_["in_event"]))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*",
                    default=["speech", "stereo", "overlap", "prop_quiet", "prop_faint", "music", "long_hum60"])
    ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args(argv)
    truth = {Path(c["path"]).stem: c for c in json.loads((ROOT / "data/synth/truth.json").read_text())}
    res = {"time": time.strftime("%Y-%m-%d %H:%M"), "curves": [], "clips": []}
    for name in a.clips:
        c = truth[name]
        print("==", name, flush=True)
        rows, summ = diagnose_synth(name, ROOT / c["path"], [Track.from_dict(d) for d in c["tracks"]])
        res["curves"] += rows; res["clips"].append(summ)
        for r in rows:
            print(f"  h{r['harmonic']} g{r['group']} {r['f_range'][0]:.0f}-{r['f_range'][1]:.0f}Hz vis {r['visible_s']:.0f}s "
                  f"z {r['z_before']:.1f}->{r['z_after']:.1f} (oracle {r['z_oracle']:.1f}) "
                  f"ridges {r['cov_ridges_1hz']:.2f} any {r['cov_any_1hz']:.2f}/{r['cov_any_3hz']:.2f} "
                  f"dop {r['cov_doppler_1hz']:.2f}/{r['cov_doppler_3hz']:.2f} fill {r['cov_filled_1hz']:.2f} "
                  f"best {r['best_any'] and (r['best_any']['kind'], round(r['best_any']['score'], 2), round(r['best_any']['rms_hz_overlap'], 2), round(r['best_any']['time_cov'], 2))}",
                  flush=True)
        OUT.write_text(json.dumps(res, indent=1, default=float))
    if not a.no_real:
        print("== prop_pass", flush=True)
        res["prop_pass"] = diagnose_prop_pass()
        pp = res["prop_pass"]
        print({k: v for k, v in pp.items() if k != "tracks"})
        for q in pp["tracks"][:12]:
            print("  ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q.items()})
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
