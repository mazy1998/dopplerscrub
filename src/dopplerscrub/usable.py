"""Adapter for scripts/final_run.py: the frozen dopfast (blind template-bank) attacker.

Returns the number of usable pass events (thresholds frozen in evaluate.DF_*).
Blind mode only: it errs toward missing passes, so a low count means "this attacker
cannot see it", not proof the pass is hidden (docs/results/dopfast.md).
"""
from __future__ import annotations

import numpy as np

from .evaluate import dopfast_file


def usable_events(x: np.ndarray, sr: int, boxes=None) -> dict:
    r = dopfast_file(x, sr)
    return {"n_usable": int(r["n_usable"]), "n_events": int(r["n_events"])}
