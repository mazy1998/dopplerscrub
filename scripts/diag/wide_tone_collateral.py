"""Collateral check: remove each clip's cached dopplerscrub tracks with default width vs a
forced width_hz (worst case: every track 'measured wide'). Speech SNR, LSD, created ridges."""
import sys, json, time
from pathlib import Path
W = Path(sys.argv[1]); out = Path(sys.argv[2]); sys.path[:0] = [str(W/'src'), str(W/'scripts')]
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dopplerscrub.types import Track, RemoveConfig
from dopplerscrub.remove import remove_tracks
from dopplerscrub import evaluate as ev
import baselines as bl, boxfree as bf, new_ridges as nr
clips = {c['name']: c for c in nr.dev_real_clips()}
for name in sys.argv[3:]:
    c = clips[name]; x, sr = bl.load(c['path'])
    cor = nr.scrub_tracks(x, sr, name)
    res = {}
    fmax = 1950.0
    MODES = [None, '8', '8noskirt', 'steep8']
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharey=True)
    for col, wid in enumerate([None, 'orig', 8.0][::-1][::-1]):
        if wid == 'orig':
            continue
    import dopplerscrub.remove as RM
    _sk = RM._skirt_one
    for col, wid in enumerate(MODES):
        trs = []
        for t, f in cor:
            tr = Track(t=t, f=f, amp=np.ones(len(t)), kind='doppler', doppler_score=0.8)
            sl = np.abs(np.gradient(f, t)) if len(t) > 2 else np.zeros(1)
            if wid in ('8', '8noskirt') or (wid == 'steep8' and np.percentile(sl, 95) > 15):
                tr.width_hz = 8.0
            trs.append(tr)
        RM._skirt_one = (lambda *a, **k: None) if wid == '8noskirt' else _sk
        t0 = time.time()
        y, st = remove_tracks(x, sr, trs, RemoveConfig(), return_stats=True)
        rt = time.time() - t0
        y = (np.round(np.clip(y, -1, 1) * 32767) / 32767).astype(np.float32)
        fd = ev.fidelity(x, y, sr)
        lsd = ev.log_spectral_distance(x, y, sr)
        b = bf.boxfree(x, y, sr, fmax=fmax)
        b450 = bf.boxfree(x, y, sr)
        res[str(wid)] = dict(speech_snr_db=fd['speech_snr_db'], snr_db=fd['snr_db'], lsd_db=lsd,
                             tf=st['tf_fraction_40_1000'], runtime_s=rt,
                             still=b['n_still_z3'], orig=b['n_ridges_orig_z3'], new=b['n_new_z3'], created=b['n_new_created_z3'],
                             still450=b450['n_still_z3'], orig450=b450['n_ridges_orig_z3'], created450=b450['n_new_created_z3'],
                             skirt_tracks=sum(1 for s in st['tracks'] if 'skirt_hz' in s))
        print(name, wid, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res[str(wid)].items()}, flush=True)
        v = ev.attack_views(x, y, sr, fmax=fmax)
        sm = ev.smooth_along_time(v['clean'], 9)
        if col == 0:
            so = ev.smooth_along_time(v['orig'], 9)
            axes[0, 0].imshow(so.S, origin='lower', aspect='auto', cmap='magma', vmin=-2, vmax=12, extent=[so.t[0], so.t[-1], so.f[0], so.f[-1]]); axes[0, 0].set_title(f'{name} orig (2 s smoothed)')
            axes[1, 0].imshow(so.S, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-4, vmax=4, extent=[so.t[0], so.t[-1], so.f[0], so.f[-1]]); axes[1, 0].set_title('orig diverging (b)')
        axes[0, col+1].imshow(sm.S, origin='lower', aspect='auto', cmap='magma', vmin=-2, vmax=12, extent=[sm.t[0], sm.t[-1], sm.f[0], sm.f[-1]]); axes[0, col+1].set_title(f'width {wid}: smoothed')
        axes[1, col+1].imshow(sm.S, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-4, vmax=4, extent=[sm.t[0], sm.t[-1], sm.f[0], sm.f[-1]]); axes[1, col+1].set_title(f'width {wid}: diverging (b), blue = below floor')
    fig.tight_layout(); fig.savefig(out/f'coll_{name}.png', dpi=60); plt.close(fig)
    json.dump(res, open(out/f'coll_{name}.json', 'w'), indent=1)
