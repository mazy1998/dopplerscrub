import numpy as np
from dopplerscrub.detect import _spectrogram, ideal_curve
from dopplerscrub.families import apply_families, clusters
from dopplerscrub.types import DetectConfig, Track

P = (40., 70., 500., 30.)          # f0, v, d_cpa, t_cpa (emission)


def _audio(orders, sr=4000, dur=60., level=.02, seed=4):
    t = np.arange(int(sr * dur)) / sr
    x = np.random.default_rng(seed).normal(0, .05, len(t))
    phase = 2 * np.pi * np.cumsum(ideal_curve(t, *P)) / sr
    for k in orders:
        x += level * np.sin(k * phase)
    return x, sr


def _track(k, t0, t1, group):
    g = np.arange(t0, t1 + 1e-9, .25)
    return Track(g, k * ideal_curve(g, *P), np.zeros(len(g)), .85, None, 1, group, 'doppler')


def test_fragments_become_one_family_with_missing_orders():
    x, sr = _audio((3, 4, 5, 6, 7))
    cfg = DetectConfig()
    t, f, amp, s = _spectrogram(x, sr, cfg)
    given = [_track(4, 15, 45, 0), _track(6, 20, 40, 1)]
    out = apply_families(given, t, f, s, amp, cfg)
    groups = {q.group for q in out if q.kind == 'doppler'}
    assert len(groups) == 1
    orders = {q.harmonic for q in out}
    assert {3, 4, 5, 6, 7} <= orders
    for q in out:
        err = q.f - q.harmonic * ideal_curve(q.t, *P)
        assert np.median(abs(err)) < 1.
    # order 6 is extended beyond the 20-40 s fragment it came from
    six = [q for q in out if q.harmonic == 6]
    assert min(q.t0 for q in six) < 18 and max(q.t1 for q in six) > 42


import pytest


@pytest.mark.parametrize('seed', [5, 6, 7, 8, 9])
def test_no_evidence_no_new_tracks(seed):
    x = np.random.default_rng(seed).normal(0, .05, 4000 * 60)
    cfg = DetectConfig()
    t, f, amp, s = _spectrogram(x, 4000, cfg)
    given = [_track(4, 15, 45, 0), _track(6, 20, 40, 1)]
    assert len(clusters(given)) == 1
    out = apply_families(given, t, f, s, amp, cfg)
    assert len(out) == 2


def test_machinery_drift_is_not_expanded_into_a_family():
    """Generator harmonics drift together (constant ratio) but barely move."""
    sr, dur = 4000, 60.
    t = np.arange(int(sr * dur)) / sr
    f0 = 30 * (1 + .008 * np.sin(2 * np.pi * t / 20))
    x = np.random.default_rng(3).normal(0, .05, len(t))
    for k in range(3, 9):
        x += .02 * np.sin(k * 2 * np.pi * np.cumsum(f0) / sr)
    cfg = DetectConfig()
    tt, f, amp, s = _spectrogram(x, sr, cfg)
    g = np.arange(5, 55, .25)
    given = [Track(g, k * 30 * (1 + .008 * np.sin(2 * np.pi * g / 20)), np.zeros(len(g)), .85, None, 1, k, 'doppler')
             for k in (4, 6)]
    out = apply_families(given, tt, f, s, amp, cfg)
    assert len(out) == 2
