import numpy as np
import pytest
from dopplerscrub.synth import flyover, mix, _kinematics
from dopplertrack.doppler import ideal_curve, curve_from_trajectory


def test_ideal_curve():
    c = 331.3 * np.sqrt(1 + 15 / 273.15)
    x, tracks = flyover("prop", 30, 4000, 12, 100, 70, 600, 3, -45, seed=1)
    for tr in tracks:
        expected = ideal_curve(tr.t, 100 * tr.harmonic, 70, 600, 12, c=c, n=100001)
        np.testing.assert_allclose(tr.f, expected, atol=0.0001)
    assert x.dtype == np.float32 and len(x) == 120000


def test_trajectory():
    c = 331.3 * np.sqrt(1 + 15 / 273.15)
    te = np.linspace(-20, 40, 100001)
    pos = np.column_stack((70 * (te - 12), np.full(len(te), 360), np.full(len(te), 480)))
    to, f, _ = curve_from_trajectory(te, pos, np.zeros(3), f0=100, c=c)
    _, _, ratio = _kinematics(to[1:-1], 70, 600, 12, c)
    np.testing.assert_allclose(ratio * 100, f[1:-1], atol=1e-6)


@pytest.mark.parametrize("kind", ["prop", "helicopter", "f1", "jet"])
def test_sources(kind):
    args = (kind, 5, 4000, 2, 100, 65, 100, 3, -45)
    x, tracks = flyover(*args, seed=12)
    y, _ = flyover(*args, seed=12)
    np.testing.assert_array_equal(x, y)
    assert tracks and all(t.kind == "doppler" for t in tracks)
    assert np.isfinite(x).all() and np.max(np.abs(x)) < 1


def test_stationary_hum_stereo():
    x = mix(np.zeros((16000, 2)), 4000, pink=0, wind=0, hum=0.01, seed=2)
    np.testing.assert_array_equal(x[:, 0], x[:, 1])
    spectrum = abs(np.fft.rfft(x[:, 0]))
    assert np.argmax(spectrum) == 200  # exactly 50 Hz for four seconds


def test_dataset(tmp_path):
    from dopplerscrub.synth import make_dataset
    import json

    records = make_dataset(tmp_path, sr=1000)
    assert len(records) == 10
    assert len(json.loads((tmp_path / "truth.json").read_text())) == 10
    assert all((tmp_path / (r["path"].split("/")[-1])).exists() for r in records)
