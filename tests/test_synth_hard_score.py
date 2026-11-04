"""scripts/synth_hard.py helpers: detector hook loading and the steep-core split."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import synth_hard as sh  # noqa: E402

from dopplerscrub.synth import render_pass, straight_track  # noqa: E402


def test_load_func(tmp_path):
    p = tmp_path / "mydet.py"
    p.write_text("def detect(x, sr, cfg=None):\n    return ['ok']\n")
    assert sh._load_func(f"{p}:detect")(None, 1) == ["ok"]
    assert sh._load_func("dopplerscrub.detect:detect_tracks").__name__ == "detect_tracks"


def test_core_is_the_steep_part():
    tone = [dict(f=lambda te: np.full(np.shape(te), 100.), harmonic=1, group=0)]
    _, (tr,) = render_pass(60, 4000, straight_track(70, 30, 200, 300), tone, level_db=-30, seed=0)
    core = sh._core(tr)
    assert len(core) == 1
    c = core[0]
    assert c.t0 < 31 < c.t1 and c.t1 - c.t0 < 20     # around the CPA (h ~ 31 s), not the plateaus


def test_follows():
    from dopplerscrub.types import Track
    t = np.arange(0, 10, .25)
    truth = [Track(t, 100 + t, np.ones(len(t)))]
    assert sh._follows(Track(t, 101 + t, np.ones(len(t))), truth)
    assert not sh._follows(Track(t, 110 + t, np.ones(len(t))), truth)


def test_dechirp_finds_steep_tone_and_ignores_noise():
    sr = 4000
    rng = np.random.default_rng(0)
    tone = [dict(f=lambda te: np.full(np.shape(te), 800.), harmonic=1, group=0)]
    sig, (tr,) = render_pass(40, sr, straight_track(120, 20, 60, 100), tone, level_db=-40, seed=1)
    core = sh._core(tr)
    noise = rng.normal(size=len(sig)) * 0.01
    assert sh._z_dechirp(sig + noise, core) > 20          # steep (~500 Hz/s) core, tone ~ noise level
    assert abs(sh._z_dechirp(noise, core)) < 5
