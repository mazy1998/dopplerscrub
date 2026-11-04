"""Source families: one shared f0(t) per moving source, every audible harmonic.

The upstream detectors return fragments: pieces of one fan with inconsistent
harmonic numbers, harmonics that were never proposed, spans that stop where a
detector's own support ended. Two lines of one source share the Doppler factor,
so their frequency RATIO is constant while both move. That is a much stronger
grouping test than any harmonic-number bookkeeping:

1. link moving doppler tracks whose log-frequency difference is constant over
   >= 3 s of overlap while they sweep; union the links into clusters;
2. assign integer orders with a harmonic sieve (largest f0 that puts every
   member on an integer multiple), measure the shared factor F(t) from all
   members, and extend it beyond the observed span with a straight-pass fit;
3. probe k*F(t) for every k in band, recentre locally, and keep only spans that
   pass the same fixed-curve local-evidence test as the other harmonic probes;
4. keep measured members (relabelled into one family with consistent orders)
   and add only the probe spans no doppler track already covers.

Nothing here searches blind: a family only exists where >= 2 moving tracks of
an already accepted source agree.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .types import Track


def _grid(q):
    return np.round(np.asarray(q.t) * 4) / 4


def _link(a, b, min_overlap=3., min_motion=.006):
    lo, hi = max(a.t0, b.t0), min(a.t1, b.t1)
    if hi - lo < min_overlap:
        return None
    g = np.arange(lo, hi + 1e-9, .25)
    la = np.log(np.interp(g, a.t, a.f)); lb = np.log(np.interp(g, b.t, b.f))
    sa = ndimage.gaussian_filter1d(la, 2); sb = ndimage.gaussian_filter1d(lb, 2)
    motion = min(np.ptp(sa), np.ptp(sb))
    if motion < min_motion:
        return None
    d = la - lb
    resid = float(np.std(d - np.median(d)))
    if resid > max(.0012, .07 * motion):
        return None
    return float(np.median(d)), len(g)


def clusters(tracks):
    """Union-find over constant-ratio links; returns (members, log offsets to root)."""
    mv = [i for i, q in enumerate(tracks) if q.kind == 'doppler' and len(q.t) >= 8]
    parent = {i: i for i in mv}; off = {i: 0. for i in mv}   # log f_i - log f_parent

    def find(i):
        if parent[i] == i:
            return i, 0.
        r, o = find(parent[i])
        parent[i] = r; off[i] = off[i] + o
        return r, off[i]

    order = sorted(mv, key=lambda i: tracks[i].t0)
    for n, i in enumerate(order):
        a = tracks[i]
        for j in order[:n]:
            b = tracks[j]
            if b.t1 < a.t0 + 3:
                continue
            r = _link(a, b)
            if r is None:
                continue
            ri, oi = find(i); rj, oj = find(j)
            if ri == rj:
                continue
            # log f_i - log f_j = r[0];  f_i rel root ri = oi; f_j rel rj = oj
            parent[ri] = rj; off[ri] = oj + r[0] - oi
    groups = {}
    for i in mv:
        r, o = find(i)
        groups.setdefault(r, []).append((i, o))
    return [g for g in groups.values() if len(g) >= 2]


def sieve(rel, weights, fmin=12., tol=.006, max_order=48):
    """rel: member frequencies (Hz) at a common reference. Largest f0 (smallest
    root order) that puts >= 85 % of the weight on integer multiples."""
    rel = np.asarray(rel, float); w = np.asarray(weights, float)
    base = rel.min()
    best = None
    for m in range(1, max_order + 1):
        f0 = base / m
        if f0 < fmin:
            break
        n = np.rint(rel / f0)
        ok = (n >= 1) & (abs(rel - n * f0) <= np.maximum(.8, tol * rel))
        frac = w[ok].sum() / w.sum()
        if frac >= .85:
            # refine f0 by least squares on the fitting members
            f0 = float(np.sum(n[ok] * rel[ok] * w[ok]) / np.sum(n[ok] ** 2 * w[ok]))
            return f0, n.astype(int), ok
        if best is None or frac > best[0]:
            best = (frac, f0, n.astype(int), ok)
    return None


def _split_families(tracks, cluster):
    """Sieve a constant-ratio cluster into integer-order families (iteratively)."""
    idx = [i for i, _ in cluster]; logoff = np.array([o for _, o in cluster])
    ref_f = float(np.median(tracks[idx[0]].f)) * np.exp(logoff - logoff[0])
    weight = np.array([max(tracks[i].t1 - tracks[i].t0, .25) for i in idx])
    remaining = np.arange(len(idx)); fams = []
    while len(remaining) >= 2:
        best = None
        for frac_needed in (.85, .5):
            s = sieve(ref_f[remaining], weight[remaining])
            if s is not None:
                best = s; break
            s = _partial_sieve(ref_f[remaining], weight[remaining], frac_needed)
            if s is not None:
                best = s; break
        if best is None:
            break
        f0, n, ok = best
        if ok.sum() < 2:
            break
        fams.append(dict(members=[idx[k] for k in remaining[ok]], orders=list(n[ok]),
                         f0_ref=f0, ref_f=ref_f[remaining[ok]], eps=0., parent=None))
        main = len(fams) - 1
        remaining = remaining[~ok]
        # Sidebands / a second prop at a constant offset: n*f0 + delta, Doppler
        # shifted with the source, i.e. a shared FRACTIONAL order k + eps.
        if len(remaining):
            frac = ref_f[remaining] / f0 - np.rint(ref_f[remaining] / f0)
            delta = frac * f0
            used = np.zeros(len(remaining), bool)
            for j in np.argsort(-weight[remaining]):
                if used[j] or abs(delta[j]) < 2.:
                    continue
                same = (~used) & (abs(frac - frac[j]) * f0 <= .6) & (abs(delta) >= 2.)
                if same.sum() < 2:
                    continue
                eps = float(np.average(frac[same], weights=weight[remaining][same]))
                fams.append(dict(members=[idx[k] for k in remaining[same]], orders=list(np.rint(ref_f[remaining[same]] / f0).astype(int)),
                                 f0_ref=f0, ref_f=ref_f[remaining[same]], eps=eps, parent=main))
                used |= same
            remaining = remaining[~used]
    return fams


def _partial_sieve(rel, w, frac_needed, fmin=12., tol=.006, max_order=48):
    base = rel.min(); best = None
    for m in range(1, max_order + 1):
        f0 = base / m
        if f0 < fmin:
            break
        n = np.rint(rel / f0)
        ok = (n >= 1) & (abs(rel - n * f0) <= np.maximum(.8, tol * rel))
        frac = w[ok].sum() / w.sum()
        if frac >= frac_needed and (best is None or frac > best[0] + .05):
            best = (frac, float(np.sum(n[ok] * rel[ok] * w[ok]) / np.sum(n[ok] ** 2 * w[ok])), n.astype(int), ok)
    return None if best is None else best[1:]


def shared_factor(tracks, members, orders, grid):
    """Weighted median of member f/n on a time grid (NaN where unobserved)."""
    bank = np.full((len(members), len(grid)), np.nan); wts = np.zeros(len(members))
    for r, (i, n) in enumerate(zip(members, orders)):
        q = tracks[i]
        inside = (grid >= q.t0) & (grid <= q.t1)
        bank[r, inside] = np.interp(grid[inside], q.t, q.f) / n
        wts[r] = n * min(q.t1 - q.t0, 30.)
    F = np.full(len(grid), np.nan)
    for j in range(len(grid)):
        col = bank[:, j]; ok = np.isfinite(col)
        if not ok.any():
            continue
        v = col[ok]; w = wts[ok]; o = np.argsort(v); c = np.cumsum(w[o])
        F[j] = v[o][np.searchsorted(c, .5 * c[-1])]
    # fill short holes, smooth lightly
    good = np.isfinite(F)
    if good.sum() >= 2:
        holes = np.flatnonzero(~good)
        for gap in np.split(holes, np.flatnonzero(np.diff(holes) > 1) + 1):
            if 0 < len(gap) <= 8 and gap[0] > 0 and gap[-1] < len(F) - 1 and good[gap[0] - 1] and good[gap[-1] + 1]:
                F[gap] = np.interp(grid[gap], grid[good], F[good])
    return F


def extend_factor(F, grid, horizon=60.):
    """Extend F beyond its observed span with a straight-pass fit, scaled to
    meet the observed factor at each end. Returns (F_ext, fit or None)."""
    from .detect import _fit, ideal_curve
    good = np.isfinite(F)
    if good.sum() < 16:
        return F, None
    tt = grid[good]; ff = F[good]
    fit = _fit(tt, ff)
    if not (fit['rms_hz'] <= max(.15, .004 * np.median(ff)) and 5 <= fit['v'] <= 200 and fit['d_cpa'] >= 50):
        return F, None
    p = [fit[k] for k in ('f0', 'v', 'd_cpa', 't_cpa')]
    model = ideal_curve(grid, *p)
    out = F.copy()
    first, last = np.flatnonzero(good)[[0, -1]]
    lead = grid < grid[first]; tail = grid > grid[last]
    near = lambda a, b: np.median(F[a:b] / model[a:b])
    if lead.any():
        out[lead] = model[lead] * near(first, min(first + 8, last + 1))
    if tail.any():
        out[tail] = model[tail] * near(max(last - 7, first), last + 1)
    keep = (grid >= grid[first] - horizon) & (grid <= grid[last] + horizon)
    out[~keep] = np.nan
    # interior holes too
    holes = ~np.isfinite(out) & keep
    out[holes] = model[holes] * np.nanmedian(F / model)
    return out, fit


def _recentre(view, f, curve, radius):
    """Smoothed per-frame offset to the local maximum within +-radius Hz."""
    df = f[1] - f[0]
    steps = np.arange(-int(np.ceil(radius / df)), int(np.ceil(radius / df)) + 1) * df
    if len(steps) < 3:
        return np.zeros(len(curve))
    from .detect import _sample
    vals = np.array([_sample(view, f, curve + d) for d in steps])
    off = steps[np.argmax(vals, axis=0)]
    off = ndimage.median_filter(off, 9, mode='nearest')
    return ndimage.gaussian_filter1d(off, 3)


def probe(F, t, f, s, amp, cfg, group, max_order=None, observed=(), eps=0., skip_multiples_of=None, unobserved_z=8., measured=None, backgrounds=None):
    """Evidence-gated harmonic tracks k*F(t) for every in-band k."""
    from .detect import _background_image, _curve_significance, _llr, _sample, _verified_curve_mask
    views = [np.asarray(s)]
    ridge = getattr(s, 'ridge_view', None)
    if ridge is not None:
        views.append(np.asarray(ridge))
    magnitude = getattr(s, 'ridge_amplitude', amp)
    ok = np.isfinite(F)
    if ok.sum() < 16:
        return []
    cols = np.flatnonzero(ok); left, right = cols[0], cols[-1] + 1
    Fc = F[left:right]
    if not np.isfinite(Fc).all():
        good = np.isfinite(Fc); Fc = np.interp(np.arange(len(Fc)), np.flatnonzero(good), Fc[good])
    tt = t[left:right]
    kmax = int(min(cfg.fmax, f[-1]) / max(np.nanmin(Fc), 1.))
    if max_order:
        kmax = min(kmax, max_order)
    out = []
    if backgrounds is None:
        backgrounds = [_background_image(v, f) for v in views]
    lik = [(v[:, left:right], np.clip(_llr(v[:, left:right]), -1, 20)) for v in views]
    bgs = [b[:, left:right] for b in backgrounds]
    for k in range(1, kmax + 1):
        if skip_multiples_of and k % skip_multiples_of == 0:
            continue
        curve = (k + eps) * Fc
        if np.max(curve) < cfg.fmin or np.min(curve) > cfg.fmax:
            continue
        # small recentring only: maximising over a wide corridor would also
        # maximise noise and bias the fixed-curve test below
        # cheap pre-filter, an exact superset of the test below: that test
        # needs a 9-frame mean on-curve level >= 0 dB somewhere
        bright = max(float(ndimage.uniform_filter1d(_sample(v, f, curve), 9, mode='nearest').max()) for v, _ in lik)
        if bright < 0.:
            continue
        radius = min(1., max(.6, .002 * k * float(np.median(Fc))))
        curve = curve + _recentre(lik[-1][0], f, curve, radius)
        verified = np.zeros(len(tt), bool)
        for (view, L), bg in zip(lik, bgs):
            verified |= _verified_curve_mask(view, f, curve, cfg, L, bg)
        idx = np.flatnonzero(verified)
        for part in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(part) < 2 or tt[part[-1]] - tt[part[0]] < cfg.min_duration_s:
                continue
            if k not in observed:
                # a new order needs the measured (not extrapolated) factor
                if measured is not None and np.mean(measured[left:right][part]) < .5:
                    continue
                if max(_curve_significance(v, f, curve, part, L) for v, L in lik) < unobserved_z:
                    continue
            g = np.arange(np.ceil(tt[part[0]] * 4), np.floor(tt[part[-1]] * 4) + 1) / 4
            if len(g) < 2:
                continue
            fr = np.interp(g, tt[part], curve[part])
            a = np.interp(g, tt[part], _sample(magnitude[:, left:right], f, curve)[part])
            q = Track(g, fr, a, max(.85, cfg.doppler_threshold), None, k, group, 'doppler')
            q._family = True; q._source_fragment = True; q._evidence_gated = True
            out.append(q)
    return out


def _coverage(q, others, tol=1.5):
    hit = np.zeros(len(q.t), bool)
    for o in others:
        if o.t1 < q.t0 or o.t0 > q.t1:
            continue
        inside = (q.t >= o.t0) & (q.t <= o.t1)
        if inside.any():
            hit[inside] |= abs(np.interp(q.t[inside], o.t, o.f) - q.f[inside]) <= tol
    return float(hit.mean()) if len(hit) else 1.


def apply_families(tracks, t, f, s, amp, cfg):
    """Consolidate doppler fragments into source families with complete,
    evidence-gated harmonic coverage. Non-doppler tracks pass through."""
    tracks = list(tracks)
    found = []
    for cl in clusters(tracks):
        for fam in _split_families(tracks, cl):
            found.append(fam)
    if not found:
        return tracks
    from .detect import _background_image
    views = [np.asarray(s)] + ([np.asarray(s.ridge_view)] if getattr(s, 'ridge_view', None) is not None else [])
    bgs = [_background_image(v, f) for v in views]
    next_group = max((q.group for q in tracks), default=-1) + 1
    new = []; relabel = {}; factors = {}; measured_span = {}
    # parents (eps == 0) before their offset families; larger sources first
    for fam in sorted(found, key=lambda fam: (fam['eps'] != 0, -sum(tracks[i].t1 - tracks[i].t0 for i in fam['members']))):
        members, orders = fam['members'], fam['orders']
        if fam['eps'] and id(found[fam['parent']]) in factors:
            F = factors[id(found[fam['parent']])]
        else:
            F = shared_factor(tracks, members, [n + fam['eps'] for n in orders], t)
            measured_span[id(fam)] = np.isfinite(F)
            F, _ = extend_factor(F, t)
        factors[id(fam)] = F
        span = measured_span.get(id(fam), measured_span.get(id(found[fam['parent']])) if fam['parent'] is not None else None)
        gid = next_group; next_group += 1
        probes = probe(F, t, f, s, amp, cfg, gid, observed=set(int(n) for n in orders), eps=fam['eps'], measured=span, backgrounds=bgs)
        if not fam['eps']:
            # The sieve takes the largest f0 consistent with the members; a
            # finer lattice (f0/2, f0/3) is accepted only on new evidence:
            # >= 2 verified orders that are not multiples of the divisor.
            for m in (2, 3):
                if np.nanmin(F) / m < 12.:
                    break
                extra = probe(F / m, t, f, s, amp, cfg, gid, skip_multiples_of=m, measured=span, backgrounds=bgs,
                              max_order=m * (max(int(n) for n in orders) + 2))
                if len({q.harmonic for q in extra}) >= 2:
                    for q in probes:
                        q.harmonic *= m
                    probes += extra
                    orders = [n * m for n in orders]
                    F = F / m; factors[id(fam)] = F
                    for other in found:
                        if other.get('parent') is not None and found[other['parent']] is fam:
                            other['eps'] *= m
                    break
        for i, n in zip(members, orders):
            relabel[i] = (gid, int(n))
        # keep the measured tracks; add only what the family newly covers
        for q in probes:
            for piece in _uncovered(q, [tracks[i] for i in range(len(tracks)) if tracks[i].kind == 'doppler'] + new, cfg):
                new.append(piece)
    out = []
    for i, q in enumerate(tracks):
        if i in relabel:
            q.group, q.harmonic = relabel[i]
        out.append(q)
    return out + new


def _uncovered(q, others, cfg, tol=1.5, overlap=.5):
    """Parts of q not already within tol Hz of another doppler track, padded by
    `overlap` s into the covered part so the corridors join."""
    hit = np.zeros(len(q.t), bool)
    for o in others:
        if o.t1 < q.t0 or o.t0 > q.t1:
            continue
        inside = (q.t >= o.t0) & (q.t <= o.t1)
        if inside.any():
            hit[inside] |= abs(np.interp(q.t[inside], o.t, o.f) - q.f[inside]) <= tol
    if not hit.any():
        return [q]
    free = ~hit
    pad = int(round(overlap / .25))
    free = ndimage.binary_dilation(free, np.ones(2 * pad + 1, bool)) if pad else free
    out = []
    idx = np.flatnonzero(free)
    for part in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(part) < 2 or q.t[part[-1]] - q.t[part[0]] < min(cfg.min_duration_s, 2.) or not (~hit[part]).sum() >= 8:
            continue
        p = Track(q.t[part], q.f[part], q.amp[part], q.doppler_score, None, q.harmonic, q.group, 'doppler')
        p._family = True; p._source_fragment = True; p._evidence_gated = True
        out.append(p)
    return out
