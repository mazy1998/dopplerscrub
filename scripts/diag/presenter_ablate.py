"""Presenter speech collateral: which part of the integration drops speech SNR?
Usage: uv run python scripts/diag/presenter_ablate.py <repo root> <clip name>   (SHOW=1 lists the doppler tracks)
Variants: as-is; width_hz stripped (no wide corridors / skirt); chirp tracks dropped."""
import sys, json, time
from dataclasses import replace
import numpy as np
root = sys.argv[1]; clip = sys.argv[2]
sys.path.insert(0, root + "/src"); sys.path.insert(0, root + "/scripts")
import benchmark as bm
from dopplerscrub import evaluate as ev, detect as D, pipeline as P
from dopplerscrub.types import RemoveConfig
c = next(c for c in bm.airshow_clips() + bm.real_clips(True) if c["name"] == clip)
x, sr = bm.load(c["path"])
orig_detect = D.detect_tracks
def run(name, hook):
    def det(y, sr, cfg=None):
        tr = orig_detect(y, sr, cfg)
        return hook(tr)
    D.detect_tracks = det
    t0 = time.time()
    y, found = P.scrub_array(x, sr, remove_cfg=RemoveConfig(harden=True))
    D.detect_tracks = orig_detect
    dop = [t for t in found if t.kind == "doppler"]
    n = min(len(x), len(y))
    fid = ev.fidelity(x[:n], y[:n], sr, exclude_tracks=dop)
    wide = [t for t in dop if getattr(t, "width_hz", None) and t.width_hz > 2.0]
    sp = [t for t in dop if 300 <= np.nanmedian(t.f) <= 3400]
    print(json.dumps({"variant": name, "n_doppler": len(dop), "n_wide": len(wide),
        "n_in_speech_band": len(sp), "wide_widths": sorted(round(t.width_hz, 1) for t in wide)[:30],
        "speech_snr_db": round(fid["speech_snr_db"], 1), "snr_outside_db": round(fid["snr_outside_db"], 1),
        "runtime_s": round(time.time() - t0)}), flush=True)
def strip(tr):
    for t in tr: t.width_hz = None
    return tr
src_attr = lambda t: getattr(t, "source", None)

def keep_low(tr):
    return [t for t in tr if not (t.kind=="doppler" and 300 <= np.nanmedian(t.f) <= 3400)]
def show(tr):
    d=[t for t in tr if t.kind=="doppler"]
    for t in sorted(d, key=lambda t: t.t0):
        print(f"  {t.t0:5.1f}-{t.t1:5.1f}s f {np.nanmin(t.f):6.0f}-{np.nanmax(t.f):6.0f} span {np.nanmax(t.f)-np.nanmin(t.f):5.1f} grp {t.group} h {t.harmonic} score {getattr(t,'doppler_score',None)}")
    return tr
import os
if os.environ.get("SHOW"): run("as-is", show); sys.exit()
run("drop speech-band doppler", keep_low)
