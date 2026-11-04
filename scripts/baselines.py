#!/usr/bin/env python
"""Targeted removal (dopplerscrub) vs blanket baselines, on the same clips.

Methods: dopplerscrub (defaults, harden=True), oracle (remover with TRUE tracks;
synthetic only), bandstop 40-450 Hz, noise (<500 Hz at 0 dB band SNR), jitter
(+-1% wow, ~0.2 Hz). The claim under test: targeted removal hides the curves
AS WELL AS blanket methods while damaging far less audio.

Outputs: docs/results/baselines.json, docs/results/baselines.md and
docs/results/img/bl_<clip>.png (one comparison image per clip: the attacker's stock
view and a 2 s smoothed red-team view for every method). Visual verdicts come from
docs/results/baselines_verdicts.json ({clip: {method: text}} plus "_summary"), written
after viewing the PNGs; --md-only re-renders the markdown from the JSON.

Usage:
  uv run python scripts/baselines.py                       # synth + 2 real dev clips
  uv run python scripts/baselines.py --only speech prop_pass --methods bandstop jitter
  uv run python scripts/baselines.py --split heldout       # data/real/heldout (burned)
  uv run python scripts/baselines.py --split heldout2 --blind  # fresh blind set; no numbers on stdout
  uv run python scripts/baselines.py --md-only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dopplerscrub import baselines as bl  # noqa: E402
from dopplerscrub import evaluate as ev  # noqa: E402
from dopplerscrub.types import Track  # noqa: E402
import boxfree as bf  # noqa: E402

RES = ROOT / "docs" / "results"
IMG = RES / "img"
METHODS = ["dopplerscrub", "oracle", "bandstop", "bandstop_brick", "noise", "jitter"]
WORK = ROOT / ".cache" / "baselines"


def load(path):
    try:
        from dopplerscrub import io
        x, sr, _ = io.load(str(path))
        return np.asarray(x, np.float32), int(sr)
    except Exception:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype="float32")
        return x, sr


# ------------------------------------------------------------------ clips

def dev_clips():
    out = []
    truth = json.loads((ROOT / "data/synth/truth.json").read_text())
    for c in truth:
        out.append({"name": Path(c["path"]).stem, "path": ROOT / c["path"], "split": "dev", "set": "synth",
                    "true": [Track.from_dict(d) for d in c["tracks"]], "boxes": []})
    # two real dev clips with annotated event boxes (clip-relative seconds)
    ann = json.loads((ROOT / "data/real/annotations.json").read_text())
    pp = next(c for c in ann["clips"] if c["path"].endswith("prop_pass.wav"))
    out.append({"name": "prop_pass", "path": ROOT / pp["path"], "split": "dev", "set": "real", "true": None,
                "boxes": [(e["start_s"], e["end_s"], max(40, e["fmin_hz"]), min(450, e["fmax_hz"]))
                          for e in pp["events"]]})
    ce = json.loads((ROOT / "data/real/colsto_events.json").read_text())["events"][0]
    out.append({"name": "colsto_2050_2250", "path": ROOT / "data/real/colsto_2050_2250.wav", "split": "dev",
                "set": "real", "true": None,
                "boxes": [(ce["start_s"] - 2050, ce["end_s"] - 2050, ce["fmin_hz"], ce["fmax_hz"])]})
    return [c for c in out if Path(c["path"]).exists()]


def heldout(which: str = "heldout"):
    import benchmark as bm
    return [{"name": c["name"], "path": c["path"], "split": which, "set": which, "true": None,
             "boxes": c["boxes"]} for c in bm.heldout_clips(which)]


# ------------------------------------------------------------------ run methods

def apply(method, x, sr, clip):
    if method == "dopplerscrub":
        return bl.dopplerscrub(x, sr)
    if method == "oracle":
        return bl.oracle(x, sr, clip["true"]) if clip["true"] else None
    return bl.BLANKET[method](x, sr)


def cached_apply(method, x, sr, clip, force=False):
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f"{clip['name']}__{method}.npy"
    meta = WORK / f"{clip['name']}__{method}.json"
    if p.exists() and meta.exists() and not force:
        return np.load(p), json.loads(meta.read_text())["runtime_s"]
    t0 = time.time()
    y = apply(method, x, sr, clip)
    rt = time.time() - t0
    if y is None:
        return None, float("nan")
    # every method's output is written the way a delivered file would be: 16-bit PCM
    # (no dither). This matters for band-stop: a whitening attacker rescales each bin,
    # so only a real output noise floor can bury what a linear filter attenuated.
    y = (np.round(np.clip(y, -1, 1) * 32767) / 32767).astype(np.float32)
    np.save(p, y)
    meta.write_text(json.dumps({"runtime_s": rt, "time": time.strftime("%Y-%m-%d %H:%M:%S")}))
    return y, rt


# ------------------------------------------------------------------ metrics

def metrics(x, y, sr, clip, v_clean, v_orig):
    m = {}
    true = clip["true"]
    if true:
        tb = [t for t in true if np.nanmedian(t.f) <= ev.FMAX and np.nanmax(t.f) >= ev.FMIN]
        m["contrast_db"] = ev.track_contrast_db(v_clean, tb)
        m["ridge_excess_db"] = ev.ridge_excess_db(v_clean, tb)
        m["ridge_excess_orig_db"] = ev.ridge_excess_db(v_orig, tb)
        af = ev.anti_forensics(x, y, sr, true)
        m["af_z_orig"] = af["residual"]["orig"]["z"]
        m["af_z"] = af["residual"]["clean"]["z"]
        m["af_upper95_db"] = af["residual"]["clean"]["upper95_db"]
        m["af_leftover"] = len(af["leftover_harmonics"])
        fid = ev.fidelity(x, y, sr, exclude_tracks=true)
        m["snr_outside_db"] = fid["snr_outside_db"]
    elif not clip["boxes"]:      # held-out: no truth, no boxes -> box-free checks only
        fid = ev.fidelity(x, y, sr)
        m["snr_outside_db"] = fid["snr_db"]    # nothing to exclude: whole-signal SNR
    else:
        bo = [ev.box_stats(v_orig, b) for b in clip["boxes"]]
        bc = [ev.box_stats(v_clean, b) for b in clip["boxes"]]
        m["box_hi_orig"] = float(np.mean([b["frac_hi"] for b in bo]))
        m["box_hi"] = float(np.mean([b["frac_hi"] for b in bc]))
        m["box_p99_orig_db"] = float(np.mean([b["p99_db"] for b in bo]))
        m["box_p99_db"] = float(np.mean([b["p99_db"] for b in bc]))
        fid = ev.fidelity(x, y, sr)
        m["snr_outside_db"] = ev.fidelity_boxes(x, y, sr, clip["boxes"])
    if not true:
        b = bf.boxfree(x, y, sr)
        m.update({"bf_z3_orig": b["n_ridges_orig_z3"], "bf_still_z3": b["n_still_z3"],
                  "bf_new_z3": b["n_new_z3"], "bf_persist_orig": b["persist_orig"],
                  "bf_persist": b["persist_clean"], "bf_z_clean_med": b["z_clean_med"]})
    m["speech_snr_db"] = fid["speech_snr_db"]
    m["lsd_db"] = ev.log_spectral_distance(x, y, sr)
    m["lsd_above450_db"] = ev.log_spectral_distance(x, y, sr, fmin=450)
    m["stoi_like"] = ev.stoi_like(x, y, sr)
    tf = ev.tf_modified_fraction(x, y, sr)
    m["tf_mod_all"] = tf[next(k for k in tf if k.startswith("0-"))]
    m["tf_mod_40_450"] = tf["40-450"]
    m["tf_mod_above450"] = tf[next(k for k in tf if k.startswith("450-"))]
    return m


# ------------------------------------------------------------------ figure

def render(clip, views, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("original", views["original"])] + [(k, v) for k, v in views.items() if k != "original"]
    fig, axs = plt.subplots(len(rows), 2, figsize=(ev.FIG_W, 1.75 * len(rows)), constrained_layout=True,
                            squeeze=False, sharex=True)
    for (name, spec), (a, b) in zip(rows, axs):
        im1 = ev._imshow(a, spec, -3, 15, "magma")
        im2 = ev._imshow(b, ev.smooth_along_time(spec, 9), -2, 2, "RdBu_r")
        a.set_title(f"{name}: attacker stock view", fontsize=8, loc="left")
        b.set_title(f"{name}: 2 s smoothed (red = ridge, blue = hole)", fontsize=8, loc="left")
        for ax in (a, b):
            ax.tick_params(labelsize=7)
            for t0, t1, f0, f1 in clip["boxes"]:
                ax.add_patch(matplotlib.patches.Rectangle((t0, f0), t1 - t0, f1 - f0, fill=False,
                                                          ec="w" if ax is a else "k", lw=0.6, ls="--"))
    axs[-1, 0].set_xlabel("time (s)"); axs[-1, 1].set_xlabel("time (s)")
    fig.colorbar(im1, ax=axs[:, 0], shrink=0.4, label="dB")
    fig.colorbar(im2, ax=axs[:, 1], shrink=0.4, label="dB (smoothed)")
    fig.suptitle(f"{clip['name']}: methods compared in the attacker's view (40-450 Hz)", fontsize=10)
    ev.save_png(fig, out_png)
    plt.close(fig)


# ------------------------------------------------------------------ markdown

def fnum(v, nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, float) and np.isinf(v):
        return "∞" if v > 0 else "-∞"
    return f"{v:.{nd}f}"


def med(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(v)) if v else float("nan")


def is_visible(r):
    return ((r.get("af_z_orig") or 0) > 3 or (r.get("box_hi_orig") or 0) > 0.02
            or (r.get("bf_z3_orig") or 0) > 0)


def summary_table(rows, split):
    """Method x metric medians for one split, over in-scope (fixed-wing) clips with a curve
    visible before. Out-of-scope clips keep their per-clip sections but are not counted."""
    import benchmark as bm
    rows = [r for r in rows if r["split"] == split and bm.in_scope(r["clip"])]
    visible = {r["clip"] for r in rows if r["method"] == "dopplerscrub" and is_visible(r)}
    if not rows:
        return [f"_No {split} rows._", ""], visible
    L = [f"Clips with a curve visible before cleaning ({len(visible)}): {', '.join(sorted(visible)) or 'none'}.", "",
         "| method | AF z after (synth) ↓ | integrated residual ≤-10 dB (synth, n/N) | exact contrast dB ↓ | "
         "tolerant ridge excess dB ↓ | event-box ridge cells % (real) ↓ | box-free ridges still z>3 (Σ still / Σ before; new) ↓ | "
         "persistent cells % ↓ | SNR outside dB ↑ | speech SNR dB ↑ | "
         "LSD dB ↓ | LSD >450 Hz dB ↓ | STOI-like ↑ | TF modified (all / 40–450 / >450) ↓ | runtime s † |",
         "|" + "---|" * 16]
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        vis = [r for r in rs if r["clip"] in visible]
        syn = [r for r in vis if "af_z" in r]
        real = [r for r in vis if "box_hi" in r]
        bfr = [r for r in vis if "bf_z3_orig" in r]
        npass = sum(1 for r in syn if (r.get("af_upper95_db") or 99) <= -10)
        L.append("| " + " | ".join([
            f"**{m}**", fnum(med([r["af_z"] for r in syn])), f"{npass}/{len(syn)}" if syn else "–",
            fnum(med([r["contrast_db"] for r in syn])), fnum(med([r["ridge_excess_db"] for r in syn])),
            fnum(100 * med([r["box_hi"] for r in real])),
            (f"{sum(r['bf_still_z3'] for r in bfr)} / {sum(r['bf_z3_orig'] for r in bfr)}; "
             f"{sum(r['bf_new_z3'] for r in bfr)}") if bfr else "–",
            fnum(100 * med([r["bf_persist"] for r in bfr])),
            fnum(med([r["snr_outside_db"] for r in vis])), fnum(med([r["speech_snr_db"] for r in vis])),
            fnum(med([r["lsd_db"] for r in vis]), 2), fnum(med([r["lsd_above450_db"] for r in vis]), 2),
            fnum(med([r["stoi_like"] for r in vis]), 3),
            f"{fnum(100 * med([r['tf_mod_all'] for r in vis]))}% / {fnum(100 * med([r['tf_mod_40_450'] for r in vis]))}% / "
            f"{fnum(100 * med([r['tf_mod_above450'] for r in vis]))}%",
            fnum(med([r["runtime_s"] for r in rs]), 1)]) + " |")
    L.append("")
    return L, visible


def write_md(rows, verdicts):
    import benchmark as bm
    by_m = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    when = max((r.get("time", "") for r in rows), default="?")
    L = ["# Targeted removal vs blanket baselines", "",
         f"Generated by `scripts/baselines.py` at code tag **freeze-1** (last row written {when}). "
         "**dev** = synthetic clips plus two real clips used during tuning; **held-out** = "
         "data/real/heldout (9 clips, never used for tuning, no annotations).", "",
         "All outputs, the original reference included, are quantised to 16-bit PCM, as a delivered "
         "WAV would be. The attacker view is per-bin whitened, so it is blind to linear gain: "
         "only removal, masking or the output noise floor can hide a tone.", "",
         "† Runtimes were measured on a shared CPU while other jobs (benchmark runs, other agents) "
         "were running; treat them as rough upper bounds, not as a performance measurement.", "",
         "**Scope:** fixed-wing airplanes (propeller and jet). Medians below cover in-scope clips "
         "only; rotorcraft, drone and road-vehicle clips keep their per-clip sections, marked OUT OF SCOPE.", "",
         "**Claim under test:** targeted removal hides the Doppler curves as well as blanket "
         "methods while damaging far less audio.", ""]
    if "_summary" in verdicts:
        L += ["## Verdict", "", verdicts["_summary"], ""]
    L += ["## Method × metric (medians), dev", "",
          "Visibility and fidelity medians use only clips whose curve was detectable before cleaning "
          "(z > 3, box ridge cells > 2%, or ≥1 box-free ridge with z > 3). On clips with no visible "
          "curve dopplerscrub rightly does nothing, which would flatter its damage numbers.", ""]
    T, visible = summary_table(rows, "dev")
    L += T
    L += ["## Method × metric (medians), held-out", "",
          "No truth and no boxes: visibility is the box-free check only (`scripts/boxfree.py`) plus "
          "the visual verdicts below. SNR outside = whole-signal SNR (nothing to exclude), so it "
          "counts removing the target itself as damage. No oracle (no true tracks).", ""]
    T, vis_h = summary_table(rows, "heldout")
    L += T
    visible = visible | vis_h
    if any(r["split"] == "heldout2" for r in rows):
        L += ["## Method × metric (medians), held-out 2 (fresh blind set)", "",
              "data/real/heldout2: collected after the first held-out set was burned; clips and event "
              "boxes chosen from metadata and a broadband loudness envelope only (see its SOURCES.md). "
              "Same caveats as held-out.", ""]
        T, vis_h2 = summary_table(rows, "heldout2")
        L += T
        visible = visible | vis_h2
    L += ["", "Reference (no processing): exact contrast and ridge excess of the ORIGINAL, median over "
          f"visible synthetic clips: {fnum(med([r.get('contrast_orig_db') for r in by_m['bandstop'] if r['clip'] in visible and 'contrast_orig_db' in r]))} dB / "
          f"{fnum(med([r.get('ridge_excess_orig_db') for r in by_m['bandstop'] if r['clip'] in visible]))} dB; "
          f"AF z {fnum(med([r.get('af_z_orig') for r in by_m['bandstop'] if r['clip'] in visible]))}; "
          f"event-box ridge cells {fnum(100 * med([r.get('box_hi_orig') for r in by_m['bandstop'] if r['clip'] in visible and 'box_hi_orig' in r]))}%.", "",
          "### Metric definitions", "",
          "- **AF z**: significance of on-curve power over the per-bin floor, integrated over every "
          "frame of the TRUE curves (`evaluate.anti_forensics`). z > 3 means an attacker who integrates "
          "along the pass detects it. **≤-10 dB** = its 95% upper bound is ≥10 dB under the floor "
          "(the research target).",
          "- **exact contrast**: whitened dB on the true curve minus ±3–6 Hz beside it. A method that "
          "*moves* the curve (jitter) scores low here without hiding it, so also read:",
          "- **tolerant ridge excess**: per frame, the max whitened dB within ±8 Hz of the true curve, "
          "minus the same on control curves 24–30 Hz away (`evaluate.ridge_excess_db`). It still sees "
          "a wobbled or shifted ridge.",
          "- **box-free ridges** (real clips, no truth needed; `scripts/boxfree.py`): an independent "
          "attacker-side ridge extractor (2 s smoothed view > 4 dB, ≥ 6 s, sweeping ≥ 3 Hz) on the "
          "ORIGINAL; counted are ridges with integrated z > 3 before, those still z > 3 after, and "
          "z > 3 ridges that appear only after processing (new: mostly leftover harmonics that were part "
          "of a larger bright component in the original; still + new = z > 3 ridges left). White noise gives 0 ridges. "
          "**persistent cells**: share of smoothed cells > 4 dB (white noise 4.6%).",
          "- **event-box ridge cells** (real clips, no true tracks): share of attacker-view cells above "
          "+8 dB inside the annotated event box. Pure noise is about 0.5%.",
          "- **SNR outside**: orig vs orig−processed, excluding ±3 Hz around the true curves (synth) or "
          "the event boxes (real). **speech SNR**: 300–3400 Hz. Both are waveform measures, so they "
          "also penalise jitter's ms-scale timing changes.",
          "- **LSD**: log-spectral distance of 46 ms power spectra (80 dB range); **>450 Hz** variant "
          "covers the part of the spectrum a band-limited defence should not touch.",
          "- **STOI-like**: STOI-style envelope-correlation intelligibility proxy implemented from Taal "
          "et al. 2011 (`evaluate.stoi_like`; pystoi/PESQ/ViSQOL are not dependencies). It is only "
          "indicative on synthetic tones.",
          "- **TF modified**: share of active 46 ms TF cells whose change exceeds −20 dB of the cell's "
          "original power. At ~16 Hz bins a ±2 Hz corridor still marks a whole cell, so targeted "
          "methods read high in the 40–450 column. Compare methods, not absolute values.",
          "- **bandstop_brick** is not in the original brief: it was added because the specified "
          "Butterworth band-stop attenuates only ~10–20 dB near 40/450 Hz, which a per-bin-whitening "
          "attacker undoes, and the claim deserves the strongest blanket baseline.", "",
          "## Per clip", ""]
    clips = []
    for r in rows:
        if r["clip"] not in clips:
            clips.append(r["clip"])
    for c in clips:
        cr = [r for r in rows if r["clip"] == c]
        L += [f"### {c} ({cr[0]['set']}, {cr[0]['split']}{'' if bm.in_scope(c) else ', OUT OF SCOPE (not fixed-wing)'}{', curve visible before' if c in visible else ', curve NOT visible before'})",
              "", f"![{c}](img/bl_{c}.png)", "",
              "| method | AF z | ≤-10 dB | contrast | ridge excess | box ridge % | box-free still/before (new) | SNR out | speech SNR | LSD | STOI-like | TF mod all / 40–450 | visual verdict |",
              "|" + "---|" * 13]
        for r in cr:
            vd = verdicts.get(c, {}).get(r["method"], "(not yet reviewed)")
            L.append("| " + " | ".join([
                r["method"], fnum(r.get("af_z")),
                ("yes" if (r.get("af_upper95_db") or 99) <= -10 else "no") if "af_z" in r else "–",
                fnum(r.get("contrast_db")), fnum(r.get("ridge_excess_db")),
                fnum(100 * r["box_hi"]) if "box_hi" in r else "–",
                (f"{r['bf_still_z3']}/{r['bf_z3_orig']} ({r['bf_new_z3']})" if "bf_z3_orig" in r else "–"),
                fnum(r["snr_outside_db"]), fnum(r["speech_snr_db"]), fnum(r["lsd_db"], 2),
                fnum(r["stoi_like"], 3),
                f"{fnum(100 * r['tf_mod_all'])}% / {fnum(100 * r['tf_mod_40_450'])}%", vd]) + " |")
        L.append("")
    (RES / "baselines.md").write_text("\n".join(L))


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--split", choices=("dev", "heldout", "heldout2"), default="dev")
    ap.add_argument("--blind", action="store_true", help="print no numbers (judge images first)")
    ap.add_argument("--force", action="store_true", help="recompute cached method outputs")
    ap.add_argument("--md-only", action="store_true")
    a = ap.parse_args(argv)

    jpath = RES / "baselines.json"
    vpath = RES / "baselines_verdicts.json"
    verdicts = json.loads(vpath.read_text()) if vpath.exists() else {}
    rows = json.loads(jpath.read_text()) if jpath.exists() else []
    if a.md_only:
        write_md(rows, verdicts)
        return
    clips = dev_clips() if a.split == "dev" else heldout(a.split)
    if a.only:
        clips = [c for c in clips if c["name"] in a.only]
    for c in clips:
        print(f"== {c['name']}", flush=True)
        x, sr = load(c["path"])
        v_orig = ev.attack_views(x, x, sr)["orig"]
        views = {"original": v_orig}
        new = []
        for m in a.methods:
            y, rt = cached_apply(m, x, sr, c, a.force)
            if y is None:
                continue
            n = min(len(x), len(y))
            xx, y = x[:n], y[:n]
            v = ev.attack_views(xx, y, sr)
            views[m] = v["clean"]
            r = {"clip": c["name"], "method": m, "set": c["set"], "split": c["split"], "runtime_s": rt,
                 "time": time.strftime("%Y-%m-%d %H:%M")}
            r.update(metrics(xx, y, sr, c, v["clean"], v_orig))
            if c["true"]:
                tb = [t for t in c["true"] if np.nanmedian(t.f) <= ev.FMAX and np.nanmax(t.f) >= ev.FMIN]
                r["contrast_orig_db"] = ev.track_contrast_db(v_orig, tb)
            new.append(r)
            print(f"   {m:12s} {rt:6.1f}s  " + ("" if a.blind else "  ".join(
                f"{k}={fnum(r.get(k), 2)}" for k in ("af_z", "ridge_excess_db", "box_hi", "snr_outside_db",
                                                     "lsd_db", "stoi_like", "tf_mod_all"))), flush=True)
        render(c, views, IMG / f"bl_{c['name']}.png")
        rows = [r for r in rows if not (r["clip"] == c["name"] and r["method"] in {n["method"] for n in new})] + new
        jpath.write_text(json.dumps(rows, indent=1, default=float))
    write_md(rows, verdicts)
    print(f"wrote {RES / 'baselines.md'}")


if __name__ == "__main__":
    main()
