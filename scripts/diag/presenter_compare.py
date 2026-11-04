"""Usage: uv run python scripts/diag/presenter_compare.py <src tree, e.g. a git-archive of freeze-1> <repo root> <clip> <out.wav>
Run one airshow clip through a given src tree (pipeline.scrub_array, harden=True), write the
cleaned WAV and print the track counts. Score the WAVs with presenter_score.py (current
evaluate.py: speech SNR, SNR, LSD all / >450 Hz, whole clip and 0-20 s / 20 s-end)."""
import sys, json, importlib, time
src, cur, clip, out = sys.argv[1:5]
sys.path.insert(0, src)
from dopplerscrub import pipeline as P
from dopplerscrub.types import RemoveConfig
import numpy as np, soundfile as sf
sys.path.insert(0, cur + "/scripts")
import benchmark as bm
c = next(c for c in bm.airshow_clips() if c["name"] == clip)
x, sr = bm.load(c["path"])
t0 = time.time()
y, found = P.scrub_array(x, sr, remove_cfg=RemoveConfig(harden=True))
rt = time.time() - t0
sf.write(out, y, sr)
dop = [t for t in found if t.kind == "doppler"]
print(json.dumps({"src": src, "n_doppler": len(dop), "n_all": len(found), "runtime_s": round(rt, 1)}))
