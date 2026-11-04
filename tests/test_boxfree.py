"""Box-free attacker check: 'new' ridges are split into created / unmasked / seen."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import boxfree as bf  # noqa: E402

SR = 8000
DUR = 60


def _sweep(f0, f1, t0, t1, amp, n):
    t = np.arange(n) / SR
    f = np.interp(t, [t0, t1], [f0, f1])
    on = ((t >= t0) & (t <= t1)).astype(float)
    return amp * on * np.sin(2 * np.pi * np.cumsum(f) / SR)


def _noise(seed=1):
    return (np.random.default_rng(seed).standard_normal(DUR * SR) * 0.05).astype(np.float32)


def test_added_ridge_is_created():
    x = _noise()
    y = x + _sweep(300, 260, 10, 40, 0.02, len(x)).astype(np.float32)
    b = bf.boxfree(x, y, SR)
    assert b["n_new_z3"] >= 1
    assert b["n_new_created_z3"] >= 1
    assert b["n_new_seen_z3"] == 0


def test_removal_creates_nothing():
    n = DUR * SR
    keep = _sweep(200, 170, 5, 45, 0.02, n)
    x = (_noise() + keep + _sweep(320, 280, 10, 40, 0.05, n)).astype(np.float32)
    y = (_noise() + keep).astype(np.float32)          # second tone removed exactly
    b = bf.boxfree(x, y, SR)
    assert b["n_new_created_z3"] == 0
    assert b["n_still_z3"] >= 1                      # the kept tone is still seen


def test_identity_has_no_new_ridges():
    x = (_noise() + _sweep(250, 220, 5, 45, 0.02, DUR * SR)).astype(np.float32)
    b = bf.boxfree(x, x, SR)
    assert b["n_new_z3"] == 0 and b["n_stronger"] == 0
