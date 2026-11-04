import numpy as np
import pytest

from dopplerscrub import baselines as bl
from dopplerscrub import evaluate as ev
from dopplerscrub.types import Track

SR = 8000


@pytest.fixture(scope="module")
def sig():
    rng = np.random.default_rng(0)
    t = np.arange(60 * SR) / SR
    f = 150 - 30 * np.tanh((t - 30) / 5)
    tone = 0.03 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    speech = 0.05 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    x = (tone + speech + 0.005 * rng.standard_normal(len(t))).astype(np.float32)
    tt = np.arange(0, 60, 0.25)
    return x, Track(tt, 150 - 30 * np.tanh((tt - 30) / 5), np.zeros(len(tt)), kind="doppler")


def test_shapes_dtypes(sig):
    x, _ = sig
    st = np.stack([x, x], 1)
    for fn in bl.BLANKET.values():
        for inp in (x, st):
            y = fn(inp, SR)
            assert y.shape == inp.shape and y.dtype == inp.dtype


def test_bandstop_kills_band_keeps_speech(sig):
    x, tr = sig
    y = bl.bandstop(x, SR)
    v = ev.attack_views(x, y, SR)
    assert ev.track_contrast_db(v["clean"], [tr]) < 1.0
    fr = ev.fidelity(x, y, SR)
    assert fr["octave_change_db"][125] < -30 and abs(fr["octave_change_db"][1000]) < 0.5


def test_noise_level(sig):
    x, _ = sig
    y = bl.noise_mask(x, SR, snr_db=0.0)
    lo = lambda z: np.mean(bl.ss.sosfiltfilt(bl.ss.butter(8, 500, fs=SR, output="sos"), z) ** 2)
    assert abs(10 * np.log10(lo(y - x) / lo(x))) < 0.5


def test_jitter_wobbles_but_ridge_stays(sig):
    x, tr = sig
    y = bl.jitter(x, SR)
    v = ev.attack_views(x, y, SR)
    # exact-track contrast drops, but the shift-tolerant ridge measure still sees it
    assert ev.ridge_excess_db(v["clean"], [tr]) > 4
    assert ev.stoi_like(x, y, SR) > 0.7   # ~0.8: gated-tone edges move by ms


def test_metrics_identity_and_damage(sig):
    x, _ = sig
    assert ev.log_spectral_distance(x, x, SR) == pytest.approx(0, abs=1e-9)
    assert ev.stoi_like(x, x, SR) == pytest.approx(1, abs=1e-6)
    assert ev.tf_modified_fraction(x, x, SR)["0-4000"] == 0
    y = bl.bandstop(x, SR)
    fr = ev.tf_modified_fraction(x, y, SR)
    assert fr["40-450"] > 0.8 and fr["450-4000"] < 0.2
    assert ev.fidelity_boxes(x, y, SR, [(0, 60, 0, 600)]) > 20


def test_bandstop_edge_weakness_and_brick(sig):
    """Butterworth band-stop leaves an edge tone fully visible to a whitening attacker;
    the brick-wall variant removes it."""
    rng = np.random.default_rng(3)
    t = np.arange(40 * SR) / SR
    f = 420 * (1 - 0.05 * np.tanh((t - 20) / 4))
    x = (0.01 * np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.003 * rng.standard_normal(len(t))).astype(np.float32)
    tt = np.arange(0, 40, 0.25)
    tr = Track(tt, 420 * (1 - 0.05 * np.tanh((tt - 20) / 4)), np.zeros(len(tt)), kind="doppler")
    q = lambda y: (np.round(y * 32767) / 32767).astype(np.float32)
    zb = ev.anti_forensics(x, q(bl.bandstop(x, SR)), SR, [tr], fmax=450)["residual"]["clean"]["z"]
    zk = ev.anti_forensics(x, q(bl.bandstop_brick(x, SR)), SR, [tr], fmax=450)["residual"]["clean"]["z"]
    assert zb > 10 and zk < 3
