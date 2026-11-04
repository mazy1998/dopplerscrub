"""Zoomed original | cleaned | cleaned+corridors+new ridges views for docs/results/new_ridges.md.
Usage: uv run python scripts/diag/new_ridges_zoom.py <repo root> <out dir>  (dev clips only; needs
the cached outputs from scripts/new_ridges.py)."""
import sys, json
from pathlib import Path
W = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.path[:0] = [str(W/'src'), str(W/'scripts')]
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dopplerscrub import evaluate as ev
import baselines as bl, new_ridges as nr, boxfree as bf
cases = [('prop_pass', 15, 45, 250, 440), ('air_extra300', 0, 40, 150, 430), ('colsto_2050_2250', 60, 115, 100, 320), ('air_hercules_airshow', 35, 65, 100, 220), ('air_spitfire_mustang', 3, 20, 360, 450)]
clips = {c['name']: c for c in nr.dev_real_clips()}
for name, t0, t1, f0, f1 in cases:
    c = clips[name]; x, sr = bl.load(c['path'])
    y = np.load(bl.WORK/f'{name}__dopplerscrub.npy'); n = min(len(x), len(y)); x, y = x[:n], y[:n]
    cor = nr.scrub_tracks(x, sr, name)
    v = ev.attack_views(x, y, sr)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, key, ttl in ((axes[0], 'orig', 'original'), (axes[1], 'clean', 'dopplerscrub'), (axes[2], 'clean', 'dopplerscrub + corridors (cyan) + new z>3 (red)')):
        sm = ev.smooth_along_time(v[key], 9)
        ax.imshow(sm.S, origin='lower', aspect='auto', cmap='magma', vmin=-2, vmax=12, extent=[sm.t[0], sm.t[-1], sm.f[0], sm.f[-1]])
        ax.set_xlim(t0, t1); ax.set_ylim(f0, f1); ax.set_title(f'{name}: {ttl}', fontsize=9)
    for ct, cf in cor:
        axes[2].plot(ct, cf+2, color='cyan', lw=.5); axes[2].plot(ct, cf-2, color='cyan', lw=.5)
    clean_r = [r for r in bf.ridge_curves(v['clean']) if not bf._near(r, bf.ridge_curves(v['orig']))]
    for r in clean_r:
        axes[2].plot(r.t, r.f, color='red', lw=1.2)
    fig.tight_layout(); fig.savefig(out/f'{name}.png', dpi=80); plt.close(fig)
    print(name)
