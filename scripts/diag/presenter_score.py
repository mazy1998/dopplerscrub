"""Usage: uv run python scripts/diag/presenter_score.py <repo root> <clip> <clean.wav> [...]"""
import sys, json
cur, clip, *wavs = sys.argv[1:]
sys.path.insert(0, cur + "/src"); sys.path.insert(0, cur + "/scripts")
import benchmark as bm
from dopplerscrub import evaluate as ev
c = next(c for c in bm.airshow_clips() if c["name"] == clip)
x, sr = bm.load(c["path"])
for w in wavs:
    y, _ = bm.load(w); n = min(len(x), len(y)); xx, y = x[:n], y[:n]
    r = {"wav": w.split("/")[-1]}
    for a, b in [(0, None), (0, 20), (20, None)]:
        s0 = int(a * sr); s1 = None if b is None else int(b * sr)
        f = ev.fidelity(xx[s0:s1], y[s0:s1], sr)
        r[f"{a}-{b or 'end'}"] = dict(speech_snr=round(f["speech_snr_db"], 1), snr=round(f["snr_db"], 1),
            lsd=round(ev.log_spectral_distance(xx[s0:s1], y[s0:s1], sr), 2),
            lsd450=round(ev.log_spectral_distance(xx[s0:s1], y[s0:s1], sr, fmin=450), 2))
    print(json.dumps(r))
