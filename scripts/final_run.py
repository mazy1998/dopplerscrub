#!/usr/bin/env python
"""Final-run harness: one blind, reproducible run of a split (built for heldout2).

Three commands; all outputs go to docs/results/final_<split>/ (OUT). Numbers stay sealed
under data/out/final_<split>/SEALED/ (gitignored, outside docs) until phase 2.

  verify  hash every input file and compare with the annotations' SHA-256 (prints counts only)
  phase1  verify, then scrub every clip and run box-free, the baselines and the usable-events
          attacker. Writes ONLY images and a manifest to OUT and prints no result numbers.
          All library output (warnings with percentages, tracebacks) goes to the sealed log.
          Also writes OUT/visual_verdicts_TEMPLATE.md.
  phase2  refuses to run unless OUT/visual_verdicts.md exists and has a verdict for every clip.
          Then renders OUT/results.md (fixed-wing headline per recording: n_usable before ->
          after, box-free still/created, speech SNR, LSD, verdict), OUT/benchmark.md,
          OUT/baselines.md and OUT/results.json.

Usage:
  uv run python scripts/final_run.py verify --split heldout2
  uv run python scripts/final_run.py phase1 --split heldout2
  #   ... view OUT/img/*.png, copy visual_verdicts_TEMPLATE.md to visual_verdicts.md, write verdicts ...
  uv run python scripts/final_run.py phase2 --split heldout2

  # harness test on dev clips (not blind-critical)
  uv run python scripts/final_run.py phase1 --split dev --only prop_quiet prop_pass air_p38

Rules enforced for heldout2: every clip must have a SHA-256 reference and match it, the git
tree must be clean (--allow-dirty overrides, and the manifest records it), and phase 1 refuses
to start over an existing sealed run (--resume continues the clips that are missing).

Usable-events attacker (fc): plug in a callable `usable_events(x, sr, boxes) -> dict` with
key "n_usable" (int) in module `dopplerscrub.usable`, or pass --usable MODULE:FUNC. Until then
the hook returns None and phase 2 prints "n/a (attacker not merged)".  TODO(fc): merge it.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark as bm  # noqa: E402
import baselines as bls  # noqa: E402
import boxfree as bf  # noqa: E402
from dopplerscrub import evaluate as ev  # noqa: E402
from dopplerscrub.types import DetectConfig, RemoveConfig, Track  # noqa: E402

SPLITS = ("dev", "heldout2")
REF_ANN = {"heldout2": [ROOT / "data/real/heldout2/annotations.json", ROOT / "docs/heldout2/annotations.json"]}
BL_METHODS = ["dopplerscrub", "oracle", "bandstop", "bandstop_brick", "noise", "jitter"]
VERDICT_TODO = "TODO"


def out_dir(split):
    return ROOT / "docs" / "results" / f"final_{split}"


def sealed_dir(split):
    return ROOT / "data" / "out" / f"final_{split}" / "SEALED"


# ------------------------------------------------------------------ clips + checksums

def clips_for(split, only=None):
    if split == "dev":
        cs = bm.synth_clips() + bm.real_clips(no_colsto=True) + bm.airshow_clips()
    else:
        cs = bm.heldout_clips(split)
    if only:
        missing = set(only) - {c["name"] for c in cs}
        if missing:
            raise SystemExit(f"unknown clip(s) for split {split}: {sorted(missing)}")
        cs = [c for c in cs if c["name"] in only]
    return cs


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def reference_hashes(split, ref_path=None):
    """{resolved path: sha256} from the split's annotations (all copies must agree)."""
    paths = [Path(ref_path)] if ref_path else REF_ANN.get(split, [])
    refs, ann_hashes = {}, {}
    for p in paths:
        if not p.exists():
            continue
        ann_hashes[str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)] = sha256_file(p)
        for c in json.loads(p.read_text()).get("clips", []):
            if c.get("sha256"):
                q = Path(c["path"]); q = q if q.is_absolute() else ROOT / q
                refs[str(q.resolve())] = c["sha256"]
    if len(set(ann_hashes.values())) > 1:
        raise SystemExit(f"annotation copies differ: {ann_hashes}")
    return refs, ann_hashes


def verify(split, clips, ref_path=None, strict=None):
    """Checksum every input; returns (records, ann_hashes). strict (default: heldout2) aborts
    on any missing reference or mismatch."""
    strict = (split != "dev") if strict is None else strict
    refs, ann_hashes = reference_hashes(split, ref_path)
    recs = []
    for c in clips:
        p = Path(c["path"]).resolve()
        got = sha256_file(p)
        ref = refs.get(str(p))
        recs.append({"clip": c["name"], "path": str(Path(c["path"]).relative_to(ROOT)) if Path(c["path"]).is_relative_to(ROOT) else str(c["path"]),
                     "sha256": got, "reference": ref,
                     "status": "no reference" if ref is None else ("match" if ref == got else "MISMATCH")})
    n_ok = sum(r["status"] == "match" for r in recs)
    bad = [r["clip"] for r in recs if r["status"] != "match"]
    if strict and (bad or not recs):
        raise SystemExit(f"checksum verification FAILED for {split}: {n_ok}/{len(recs)} match; "
                         f"not verified: {bad}. Nothing was run.")
    return recs, ann_hashes, n_ok


# ------------------------------------------------------------------ provenance

def git(*a):
    return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()


def code_status():
    """Tracked changes anywhere plus untracked files under src/ and scripts/ (code the run would import)."""
    tracked = git("status", "--porcelain", "--untracked-files=no").splitlines()
    untracked = [l for l in git("status", "--porcelain", "--untracked-files=all", "--", "src", "scripts").splitlines()
                 if l.startswith("??") and not l.endswith((".pyc", "/"))]
    return tracked + untracked


def manifest(split, args, recs, ann_hashes, n_ok):
    status = code_status()
    m = {
        "split": split, "started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "argv": sys.argv,
        "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": git("describe", "--always", "--tags", "--dirty"),
        "git_status_tracked": status,
        "diff_sha256": hashlib.sha256(code_diff().encode()).hexdigest() if status else None,
        "detect_config": dataclasses.asdict(DetectConfig()),
        "remove_config": dataclasses.asdict(RemoveConfig(harden=True)),
        "baseline_methods": args.methods,
        "usable_attacker": usable_name(args),
        "usable_attacker_sha256": (sha256_file(usable_name(args).rpartition(":")[0])
                                   if (usable_name(args) or "").rpartition(":")[0].endswith(".py") else None),
        "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
        "annotations_sha256": ann_hashes,
        "inputs": recs, "inputs_verified": f"{n_ok}/{len(recs)}",
    }
    return m


def write_manifest_md(m, path):
    L = [f"# Final run manifest: {m['split']}", "",
         f"- started: {m['started']}", f"- command: `{' '.join(m['argv'])}`",
         f"- commit: `{m['commit']}` ({m['describe']}, branch {m['branch']})",
         "- tracked changes: " + ("none (clean tree)" if not m["git_status_tracked"]
                                  else f"**DIRTY**: {m['git_status_tracked']}; diff sha256 `{m['diff_sha256']}` "
                                       "(full diff in SEALED/code.diff)"),
         f"- DetectConfig: `{json.dumps(m['detect_config'])}`",
         f"- RemoveConfig: `{json.dumps(m['remove_config'])}`",
         f"- baseline methods: {', '.join(m['baseline_methods'])}",
         f"- usable-events attacker: {m['usable_attacker'] or 'none (stub, TODO fc)'}"
         + (f" (file sha256 `{m['usable_attacker_sha256']}`)" if m.get("usable_attacker_sha256") else ""),
         f"- python {m['python']}, numpy {m['numpy']}, {m['platform']}",
         f"- annotations: " + (", ".join(f"`{k}` sha256 `{v}`" for k, v in m["annotations_sha256"].items()) or "none"),
         f"- inputs verified against annotation SHA-256: **{m['inputs_verified']}**", "",
         "| clip | file | sha256 | reference | status |", "|---|---|---|---|---|"]
    for r in m["inputs"]:
        L.append(f"| {r['clip']} | `{r['path']}` | `{r['sha256'][:16]}…` | "
                 f"{'`' + r['reference'][:16] + '…`' if r['reference'] else '–'} | {r['status']} |")
    path.write_text("\n".join(L) + "\n")


def code_diff():
    """git diff HEAD plus the full text of untracked src/scripts files."""
    parts = [git("diff", "HEAD")]
    for l in code_status():
        if l.startswith("??"):
            p = ROOT / l[3:].strip()
            if p.is_file():
                parts.append(f"--- untracked {l[3:].strip()}\n" + p.read_text(errors="replace"))
    return "\n".join(parts)


# ------------------------------------------------------------------ usable-events hook

def usable_name(args):
    if args.usable:
        return args.usable
    try:
        importlib.import_module("dopplerscrub.usable")
        return "dopplerscrub.usable:usable_events"
    except ImportError:
        return None


def load_usable(args):
    """TODO(fc): the usable-events (template-bank) attacker. Contract:
    usable_events(x: np.ndarray, sr: int, boxes: list[(t0, t1, f0, f1)] | None) -> {"n_usable": int, ...}.
    Called on the original and on the cleaned audio of every clip."""
    name = usable_name(args)
    if not name:
        return None
    mod, _, fn = name.rpartition(":") if name.count(":") else (name, "", "")
    if mod.endswith(".py"):                      # a file path: load it as a module
        import importlib.util
        spec = importlib.util.spec_from_file_location("usable_hook", mod)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod)
    return getattr(module, fn or "usable_events")


# ------------------------------------------------------------------ sealing stdout/stderr

class Sealed:
    """Redirect fds 1 and 2 (prints, warnings, native output) into the sealed log; say()
    writes a status line to the real terminal. Status lines must never carry results."""
    def __init__(self, log_path):
        self.log_path = log_path

    def __enter__(self):
        sys.stdout.flush(); sys.stderr.flush()
        self.o, self.e = os.dup(1), os.dup(2)
        self.f = open(self.log_path, "a")
        os.dup2(self.f.fileno(), 1); os.dup2(self.f.fileno(), 2)
        return self

    def say(self, msg):
        print(f"[status] {msg}", flush=True)       # also into the log
        os.write(self.o, (msg + "\n").encode())

    def __exit__(self, *exc):
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(self.o, 1); os.dup2(self.e, 2)
        os.close(self.o); os.close(self.e); self.f.close()
        return False


# ------------------------------------------------------------------ phase 1

def quantize16(y):
    return (np.round(np.clip(y, -1, 1) * 32767) / 32767).astype(np.float32)


def run_clip(c, split, img, sealed, methods, usable):
    """Everything for one clip. Returns (bench rows, baseline rows, extra dict). Renders PNGs."""
    name = c["name"]
    x, sr = bm.load(c["path"])
    y, det, rt_s, how = bm.run_scrub(c["path"], sealed / "wav" / f"{name}_clean.wav")
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    base = {"set": c["set"], "split": split, "runtime_s": rt_s, "how": how}
    rows = []
    if c["set"] == "synth":
        m = bm.metrics(x, y, sr, c["true"], det)
        r = bm.evaluate_clip(name, x, y, sr, c["true"], det, wide=False); r.update(base, **m); rows.append(r)
        if c["wide"]:
            r2 = bm.evaluate_clip(name, x, y, sr, c["true"], det, wide=True, tag="_wide"); r2.update(base, **m)
            rows.append(r2)
    else:
        wide = c["set"] in ("airshow", "heldout", "heldout2") or c.get("wide")
        m = bm.metrics(x, y, sr, None, det)
        r = bm.evaluate_clip(name, x, y, sr, None, det, fmax=bm.AIRSHOW_FMAX if wide else None,
                             boxes=c.get("boxes") or None)
        r.update(base, **m); r["boxfree"] = bf.boxfree(x, y, sr); rows.append(r)
    extra = {"lsd_db": ev.log_spectral_distance(x, y, sr),
             "lsd_above450_db": ev.log_spectral_distance(x, y, sr, fmin=450),
             "n_detected": len(det), "n_doppler": sum(t.kind == "doppler" for t in det)}
    if usable is not None:
        extra["usable_orig"] = usable(x, sr, c.get("boxes"))
        extra["usable_clean"] = usable(y, sr, c.get("boxes"))
    # baselines: same clip, the dopplerscrub output above (16-bit, as delivered), blanket methods fresh
    blc = {"name": name, "path": c["path"], "split": split, "set": c["set"], "true": c.get("true"),
           "boxes": c.get("boxes") or []}
    v_orig = ev.attack_views(x, x, sr)["orig"]
    views, blrows = {"original": v_orig}, []
    for meth in methods:
        if meth == "dopplerscrub":
            yb, rtb = quantize16(y), rt_s
        else:
            yb, rtb = bls.cached_apply(meth, x, sr, blc, force=True)
            if yb is None:
                continue
        nn = min(len(x), len(yb)); xx, yb = x[:nn], yb[:nn]
        v = ev.attack_views(xx, yb, sr)
        views[meth] = v["clean"]
        br = {"clip": name, "method": meth, "set": c["set"], "split": split, "runtime_s": rtb,
              "time": time.strftime("%Y-%m-%d %H:%M")}
        br.update(bls.metrics(xx, yb, sr, blc, v["clean"], v_orig))
        if blc["true"]:
            tb = [t for t in blc["true"] if np.nanmedian(t.f) <= ev.FMAX and np.nanmax(t.f) >= ev.FMIN]
            br["contrast_orig_db"] = ev.track_contrast_db(v_orig, tb)
        blrows.append(br)
    bls.render(blc, views, img / f"bl_{name}.png")
    return rows, blrows, extra


def template(clips, img_rel="img"):
    L = ["# Visual verdicts (write these BEFORE phase 2; phase 2 refuses to run without them)", "",
         "Copy this file to `visual_verdicts.md` and replace every TODO with your verdict, written from",
         "the images only (PASS / PARTIAL / FAIL / N/A + what you see). Keep the `## <clip>` headings.", ""]
    for c in clips:
        n = c["name"]
        imgs = [f"{img_rel}/{n}_ba.png", f"{img_rel}/{n}_redteam.png", f"{img_rel}/bl_{n}.png"]
        if c.get("wide") and c["set"] == "synth":
            imgs[2:2] = [f"{img_rel}/{n}_wide_ba.png", f"{img_rel}/{n}_wide_redteam.png"]
        L += [f"## {n}", "", "- images: " + ", ".join(f"[{Path(i).name}]({i})" for i in imgs),
              f"- verdict: {VERDICT_TODO}", ""]
    return "\n".join(L)


def phase1(args):
    split = args.split
    out, sealed = out_dir(split), sealed_dir(split)
    img = out / "img"
    status = code_status()      # checked before any input of the split is listed or read
    if split != "dev" and status and not args.allow_dirty:
        raise SystemExit(f"code is not committed ({status}); commit first (or --allow-dirty, recorded in the "
                         "manifest). Nothing was run and no input was read.")
    clips = clips_for(split, args.only)
    recs, ann_hashes, n_ok = verify(split, clips, args.ref_annotations, strict=None if not args.ref_annotations else True)
    rows_p = sealed / "bench_rows.json"
    if rows_p.exists() and not args.resume:
        raise SystemExit(f"a sealed run already exists in {sealed.relative_to(ROOT)}; use --resume to finish "
                         "missing clips (never delete it to re-run a blind split). Nothing was run.")
    for d in (img, sealed / "wav", sealed / "bl_cache"):
        d.mkdir(parents=True, exist_ok=True)
    (out / ".gitignore").write_text("img/\n")
    bm.IMG = img; bm.SHEET.clear()
    bls.WORK = sealed / "bl_cache"
    usable = load_usable(args)
    m = manifest(split, args, recs, ann_hashes, n_ok)
    if status:
        (sealed / "code.diff").write_text(code_diff())
    runs = json.loads((sealed / "runs.json").read_text()) if (sealed / "runs.json").exists() else []
    if runs and (runs[0]["commit"], runs[0]["diff_sha256"]) != (m["commit"], m["diff_sha256"]):
        raise SystemExit(f"--resume refused: code changed since the sealed run started "
                         f"({runs[0]['commit'][:10]} -> {m['commit'][:10]}). Nothing was run.")
    runs.append(m); (sealed / "runs.json").write_text(json.dumps(runs, indent=1, default=str))
    if len(runs) == 1:
        write_manifest_md(m, out / "MANIFEST.md")
        (out / "visual_verdicts_TEMPLATE.md").write_text(template(clips))
    else:
        with open(out / "MANIFEST.md", "a") as f:
            f.write(f"\n- resumed {m['started']}: `{' '.join(m['argv'])}` (same commit, inputs verified "
                    f"{m['inputs_verified']})\n")

    bench = json.loads(rows_p.read_text()) if rows_p.exists() else []
    blj = json.loads((sealed / "baseline_rows.json").read_text()) if (sealed / "baseline_rows.json").exists() else []
    extra = json.loads((sealed / "extra.json").read_text()) if (sealed / "extra.json").exists() else {}
    failed = []
    with Sealed(sealed / "phase1.log") as s:
        s.say(f"phase 1 on split {split}: inputs verified {m['inputs_verified']}; usable attacker: "
              f"{'yes' if usable else 'stub (TODO fc)'}; all numbers sealed")
        for c in clips:
            if c["name"] in extra:
                s.say(f"{c['name']}: already done (resume), skipped")
                continue
            s.say(f"{c['name']}: running")
            try:
                rows, blrows, ex = run_clip(c, split, img, sealed, args.methods, usable)
            except Exception:
                traceback.print_exc()
                failed.append(c["name"]); s.say(f"{c['name']}: FAILED (traceback in the sealed log)")
                continue
            bench += rows; blj += blrows; extra[c["name"]] = ex
            rows_p.write_text(json.dumps(bench, indent=1, default=float))
            (sealed / "baseline_rows.json").write_text(json.dumps(blj, indent=1, default=float))
            (sealed / "extra.json").write_text(json.dumps(extra, indent=1, default=float))
            s.say(f"{c['name']}: done (images written)")
        if bm.SHEET:
            bm.write_sheet(img / "sheet.png", bm.SHEET)
        s.say(("phase 1 finished with failures: " + ", ".join(failed)) if failed else "phase 1 finished")
    print(f"images: {img.relative_to(ROOT)}/   manifest: {(out / 'MANIFEST.md').relative_to(ROOT)}")
    print(f"next: copy {(out / 'visual_verdicts_TEMPLATE.md').relative_to(ROOT)} to visual_verdicts.md, write the "
          f"verdicts from the images, then run phase2. Do not open {sealed.relative_to(ROOT)}/.")
    return 1 if failed else 0


# ------------------------------------------------------------------ phase 2

def parse_verdicts(path):
    v, cur = {}, None
    for line in path.read_text().splitlines():
        h = re.match(r"^##\s+(\S+)\s*$", line)
        if h:
            cur = h.group(1); continue
        mv = re.match(r"^\s*-\s*verdict:\s*(.*)$", line)
        if mv and cur:
            v[cur] = mv.group(1).strip()
    return v


def fmt(v, nd=1):
    return bm.fnum(v, nd) if not isinstance(v, (int, np.integer)) else str(v)


def phase2(args):
    split = args.split
    out, sealed = out_dir(split), sealed_dir(split)
    vpath = out / "visual_verdicts.md"
    if not vpath.exists():
        raise SystemExit(f"refusing: {vpath.relative_to(ROOT)} is missing. Write the visual verdicts from the "
                         "phase-1 images first (template: visual_verdicts_TEMPLATE.md).")
    rows_p = sealed / "bench_rows.json"
    if not rows_p.exists():
        raise SystemExit(f"no sealed phase-1 results in {sealed.relative_to(ROOT)}; run phase1 first.")
    bench = json.loads(rows_p.read_text())
    blj = json.loads((sealed / "baseline_rows.json").read_text())
    extra = json.loads((sealed / "extra.json").read_text())
    verdicts = parse_verdicts(vpath)
    clips = list(extra)
    missing = [c for c in clips if not verdicts.get(c) or VERDICT_TODO in verdicts[c]]
    if missing:
        raise SystemExit(f"refusing: no verdict (or still TODO) for {missing} in {vpath.relative_to(ROOT)}.")
    runs = json.loads((sealed / "runs.json").read_text())
    run = dict(runs[0], resumes=len(runs) - 1)
    resumed = f" (resumed {run['resumes']}x, same code)" if run["resumes"] else ""

    # full tables via the existing writers, redirected to OUT
    vd = {"_note": f"Visual verdicts from {vpath.name}, written before phase 2 (numbers unsealed "
                   f"{time.strftime('%Y-%m-%d %H:%M')}); code {run['commit'][:10]}."}
    for r in bench:
        base = r["clip"][:-5] if r["clip"].endswith("_wide") else r["clip"]
        vd[r["clip"]] = verdicts.get(r["clip"], verdicts.get(base, ""))
    bm.RES = out; bm.write_md(bench, vd)
    bls.RES = out; bls.write_md(blj, {})

    # headline: fixed-wing, one row per recording
    by = {r["clip"]: r for r in bench}
    blds = {r["clip"]: r for r in blj if r["method"] == "dopplerscrub"}
    H, tally = [], {"n": 0, "le1": 0, "zero": 0, "usable_known": 0}
    for name in clips:
        if not bm.in_scope(name) or name not in by:
            continue
        r, ex = by[name], extra[name]
        uo, uc = ex.get("usable_orig"), ex.get("usable_clean")
        if uo is not None and uc is not None:
            nu_o, nu_c = uo.get("n_usable"), uc.get("n_usable")
            usable_s = f"{nu_o} → {nu_c}"
            tally["usable_known"] += 1; tally["le1"] += nu_c <= 1; tally["zero"] += nu_c == 0
        else:
            usable_s = "n/a (attacker not merged)"
        tally["n"] += 1
        b = r.get("boxfree")
        bf_s = (f"{b['n_ridges_orig_z3']} → {b['n_still_z3']} (created {b.get('n_new_created_z3', '–')})"
                if b else "–")
        H.append(f"| {name} | {usable_s} | {bf_s} | {fmt(r.get('speech_snr_db'))} | {fmt(ex.get('lsd_db'), 2)} | "
                 f"{fmt(ex.get('lsd_above450_db'), 2)} | {verdicts.get(name, '')} |")
    L = [f"# Final run: {split}", "",
         f"Code `{run['commit']}` ({run['describe']}); phase 1 started {run['started']}"
         f"{resumed}; inputs verified "
         f"{run['inputs_verified']} against the annotation SHA-256 (see MANIFEST.md). Visual verdicts were "
         f"written from the images before this file was generated.", "",
         "## Headline (fixed-wing recordings)", "",
         "Pass criterion (geolocation paper, docs memory): ≤1 usable pass left with an unknown recording clock, "
         "0 with a known clock.", ""]
    if tally["usable_known"]:
        L += [f"- usable passes left ≤1: **{tally['le1']}/{tally['usable_known']}** recordings; "
              f"0 left: **{tally['zero']}/{tally['usable_known']}**", ""]
    else:
        L += ["- usable-events attacker not merged: n_usable is n/a (TODO fc); judge by the verdicts and box-free.", ""]
    L += ["| recording | usable passes before → after | box-free ridges z>3 before → still (created) | "
          "speech SNR dB | LSD dB | LSD >450 Hz dB | visual verdict |",
          "|---|---|---|---|---|---|---|", *H, "",
          "Full per-clip table: [benchmark.md](benchmark.md); method comparison: [baselines.md](baselines.md); "
          "provenance: [MANIFEST.md](MANIFEST.md).", ""]
    (out / "results.md").write_text("\n".join(L))
    (out / "results.json").write_text(json.dumps({"run": run, "verdicts": verdicts, "bench": bench,
                                                  "baselines": blj, "extra": extra}, indent=1, default=float))
    print(f"wrote {(out / 'results.md').relative_to(ROOT)}, benchmark.md, baselines.md, results.json")
    return 0


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("verify", "phase1", "phase2"))
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--only", nargs="*", help="subset of clip names (harness tests)")
    ap.add_argument("--methods", nargs="*", default=BL_METHODS, help="baseline methods")
    ap.add_argument("--usable", default=None, help="MODULE:FUNC or path/file.py:FUNC of the usable-events attacker (default: "
                                                   "dopplerscrub.usable:usable_events if it exists)")
    ap.add_argument("--resume", action="store_true", help="finish missing clips of an existing sealed run")
    ap.add_argument("--allow-dirty", action="store_true", help="run with tracked changes (recorded)")
    ap.add_argument("--ref-annotations", default=None,
                    help="annotations JSON with clips[].sha256 to verify against (tests); strict when given")
    a = ap.parse_args(argv)
    if a.command == "verify":
        clips = clips_for(a.split, a.only)
        recs, _, n_ok = verify(a.split, clips, a.ref_annotations, strict=False)
        print(f"{a.split}: {n_ok}/{len(recs)} inputs match their annotation SHA-256")
        for r in recs:
            if r["status"] != "match":
                print(f"  {r['clip']}: {r['status']}")
        return 0 if n_ok == len(recs) else 1
    return phase1(a) if a.command == "phase1" else phase2(a)


if __name__ == "__main__":
    sys.exit(main())
