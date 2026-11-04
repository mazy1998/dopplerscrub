"""Does the dopfast attacker fire on curve-shaped NOTCHES? Run it with S = deficit ("neg") and
|deviation| ("abs") on originals vs dopplerscrub outputs (cached), DEV clips only.
A notch shaped like the pass would show as new or stronger deficit events in the cleaned file."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
from dopplerscrub import evaluate as ev
import baselines as bl, new_ridges as nr

clips = {c["name"]: c for c in nr.dev_real_clips()}
import os
CLEAN_DIR = os.environ.get("CLEAN_DIR")     # e.g. a benchmark --workdir with <clip>_clean.wav
names = sys.argv[1:] or ["prop_pass", "colsto_2050_2250", "air_p51_mustang", "air_hercules_airshow", "air_p38"]
for name in names:
    c = clips[name]
    x, sr = bl.load(c["path"])
    if CLEAN_DIR:
        y, _ = bl.load(Path(CLEAN_DIR) / f"{name}_clean.wav")
    else:
        y = np.load(ROOT / ".cache" / "baselines" / f"{name}__dopplerscrub.npy")
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    for pol in ("pos", "neg", "abs"):
        ev.DF_POLARITY = pol
        res = []
        for key, sig in (("orig", x), ("clean", y)):
            E, P, t, lg = ev.dopfast_volume(sig, sr)
            evs = ev.dopfast_event_list(E, P, t, lg)
            res.append((key, float(np.max(E[np.isfinite(E)])) if np.isfinite(E).any() else float("nan"), len(evs), sum(e["usable"] for e in evs),
                        "; ".join(f"{e['s']:.0f}s v{e['v']:.0f} d{e['d']:.0f} E{e['E']:.0f}{'' if e['usable'] else ' (unus.)'}" for e in evs)))
        print(f"{name:22s} {pol}: " + " | ".join(f"{k}: max E {m:.1f}, {u}/{ne} usable/events [{s}]" for k, m, ne, u, s in res), flush=True)
    ev.DF_POLARITY = "pos"
