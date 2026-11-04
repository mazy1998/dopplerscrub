import sys, json
from pathlib import Path
W = Path(sys.argv[1]); sys.path[:0] = [str(W/'src'), str(W/'scripts')]
import numpy as np
from dopplerscrub.remove import _freq_curve
import baselines as bl
d = json.load(open(sys.argv[2])); t = np.array(d['t']); f = np.array(d['f'])
x, sr = bl.load(W/'data/real/airshow/p51_mustang.wav')
def prof(sig):
    m = sig if sig.ndim == 1 else sig.mean(1)
    a, b = int(t[0]*sr), int(t[-1]*sr); tq = np.arange(a, b)/sr
    z = m[a:b]*np.exp(-2j*np.pi*np.cumsum(_freq_curve(t, f, tq))/sr)
    # Welch-like: 1 s segments, 0.25 s hop
    n = sr; h = sr//4; win = np.hanning(n); P = 0
    for s0 in range(0, len(z)-n, h):
        P = P + np.abs(np.fft.fftshift(np.fft.fft(z[s0:s0+n]*win)))**2
    F = np.fft.fftshift(np.fft.fftfreq(n, 1/sr))
    return F, P
F, Px = prof(x)
out = {}
for k in sys.argv[3:]:
    y = np.load(Path(sys.argv[0]).parent.parent/'p51'/f'y_{k}.npy'); _, out[k] = prof(y)
for off in (0, 6, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200):
    sel = (np.abs(np.abs(F) - off) <= 1)
    print(f"{off:3d} Hz  orig {10*np.log10(Px[sel].mean()):6.1f}  " + "  ".join(f"w{k} {10*np.log10(P[sel].mean()):6.1f}" for k, P in out.items()))
