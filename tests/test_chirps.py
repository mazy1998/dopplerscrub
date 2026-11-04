import numpy as np
from dopplerscrub.chirps import detect_chirps


def _noise(sr, dur, seed=3):
    return np.random.default_rng(seed).normal(0, .05, int(sr * dur))


def _tone(sr, f):
    return np.sin(2 * np.pi * np.cumsum(f) / sr)


def test_fast_sweep_core_is_tracked_and_dechirp_refined():
    sr = 8000; dur = 20.; t = np.arange(int(sr * dur)) / sr
    # 1200 -> 600 Hz in about 6 s (up to ~150 Hz/s), tanh-shaped like a close pass core
    f = 900 - 300 * np.tanh((t - 10) / 2.5)
    x = _noise(sr, dur) + .05 * _tone(sr, f)
    found = detect_chirps(x, sr)
    assert found
    q = max(found, key=lambda q: q.t1 - q.t0)
    assert q.t1 - q.t0 > 5 and np.ptp(q.f) > 300
    # 1/16 s grid, accurate to ~1 Hz even at the steepest point
    assert np.median(np.diff(q.t)) <= .07
    err = q.f - (900 - 300 * np.tanh((q.t - 10) / 2.5))
    assert np.median(abs(err)) < 1.5
    # a pure tone needs only the default corridor
    assert q.width_hz is not None and q.width_hz <= 2.5


def test_oscillating_pitch_and_noise_are_rejected():
    sr = 8000; dur = 30.; t = np.arange(int(sr * dur)) / sr
    f0 = 125 + 22 * np.sin(2 * np.pi * .31 * t)          # voice-like pitch contour
    x = _noise(sr, dur)
    for h in range(1, 8):
        x += .03 / h * np.sin(h * 2 * np.pi * np.cumsum(f0) / sr)
    assert detect_chirps(x, sr) == []
    assert detect_chirps(_noise(sr, dur, 5), sr) == []


def test_steady_and_slow_lines_are_left_to_the_main_detector():
    sr = 8000; dur = 30.; t = np.arange(int(sr * dur)) / sr
    x = _noise(sr, dur) + .05 * np.sin(2 * np.pi * 440 * t) + .05 * _tone(sr, 300 - .5 * t)
    assert detect_chirps(x, sr) == []


def test_track_width_round_trips():
    from dopplerscrub.types import Track
    q = Track(np.arange(3.), np.ones(3), np.zeros(3), width_hz=6.5)
    assert Track.from_dict(q.to_dict()).width_hz == 6.5
    assert Track.from_dict({"t": [0, 1], "f": [1, 1]}).width_hz is None
