"""p51 wide-tone repro: remove f7's fine track with width None / W; report energy kept near the track."""
import sys, json
from pathlib import Path
W = Path(sys.argv[1]); sys.path[:0] = [str(W/'src'), str(W/'scripts')]
import numpy as np
from dopplerscrub.types import Track, RemoveConfig
from dopplerscrub.remove import remove_tracks, _freq_curve
from dopplerscrub import evaluate as ev
import baselines as bl, boxfree as bf
d = json.load(open(sys.argv[2]))
x, sr = bl.load(W/'data/real/airshow/p51_mustang.wav')
t = np.array(d['t']); f = np.array(d['f'])
print('track', t[0], t[-1], 'dt', np.median(np.diff(t)), 'f', f.min(), f.max())
def band_energy(sig, half):
    """dechirped energy within +-half Hz of the track over its span (all channels)."""
    from scipy.signal import stft
    m = sig if sig.ndim == 1 else sig.mean(1)
    a, b = int(t[0]*sr), int(t[-1]*sr)
    tq = np.arange(a, b)/sr
    ph = 2*np.pi*np.cumsum(_freq_curve(t, f, tq))/sr
    z = m[a:b]*np.exp(-1j*ph)
    Z = np.fft.fft(z); F = np.fft.fftfreq(len(z), 1/sr)
    return float(np.sum(np.abs(Z[np.abs(F) <= half])**2))
for wid in [None] + [float(v) for v in sys.argv[3:]]:
    tr = Track(t=t, f=f, amp=np.ones(len(t)), kind='doppler', doppler_score=0.8)
    if wid is not None:
        tr.width_hz = wid
    y, st = remove_tracks(x, sr, [tr], RemoveConfig(), return_stats=True)
    s = st['tracks'][0]
    e = {h: 10*np.log10(band_energy(y, h)/band_energy(x, h)) for h in (3, 10, 20)}
    b = bf.boxfree(x, y, sr, fmax=1950.0)
    v = ev.attack_views(x, y, sr, fmax=1950.0)
    af = ev._curve_vs_floor(v['raw_clean'], v['orig'].t, v['orig'].f, [Track(t=t, f=f, amp=np.ones(len(t)))])
    af0 = ev._curve_vs_floor(v['raw_orig'], v['orig'].t, v['orig'].f, [Track(t=t, f=f, amp=np.ones(len(t)))])
    print(f"width={wid}: dE(+-3/10/20 Hz) dB = {e[3]:.1f}/{e[10]:.1f}/{e[20]:.1f}  coherent {s.get('coherent_frac',float('nan')):.2f} reverted {s.get('reverted_frac',float('nan')):.2f} "
          f"filled {s.get('filled')} gate {s.get('fill_gate_mean',0):.2f} hot {s.get('fill_hot_frac',0):.2f} skirt {s.get('skirt_hz',float('nan')):.0f}  AF z {af0['z']:.1f}->{af['z']:.1f} exc {af0['excess_db']:.1f}->{af['excess_db']:.1f}  "
          f"boxfree still {b['n_still_z3']}/{b['n_ridges_orig_z3']} new {b['n_new_z3']} created {b['n_new_created_z3']}  tf {st['tf_fraction_40_1000']:.4f}", flush=True)
    np.save(Path(sys.argv[0]).parent.parent/'p51'/f'y_{wid}.npy', y)
