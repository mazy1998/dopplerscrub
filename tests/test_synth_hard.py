"""Exactness of the data/synth_hard generator (synth.render_pass and the clip specs)."""
import numpy as np
import pytest
from dopplertrack.doppler import curve_from_trajectory

from dopplerscrub.synth import (_kinematics, build_hard_clip, hard_specs, parabolic_track,
                                prop_components, render_pass, speed_of_sound, straight_track,
                                turn_track)

C = speed_of_sound(15)


def _tone(f0):
    return [dict(f=lambda te: np.full(np.shape(te), f0), harmonic=1, group=0)]


def test_straight_truth_matches_closed_form():
    traj = straight_track(70, 12, lateral=360, alt=481.5)        # slant 600 m from a 1.5 m mic
    _, tracks = render_pass(30, 4000, traj, _tone(100.), level_db=-40, seed=1)
    tr = tracks[0]
    _, _, ratio = _kinematics(tr.t, 70, 600, 12, C)
    np.testing.assert_allclose(tr.f, 100*ratio, atol=2e-3)


@pytest.mark.parametrize('traj', [parabolic_track(80, .55, 60, y0=150, alt=250),
                                  turn_track(65, 700, 50, center=(0, 500), alt=200)])
def test_curved_truth_matches_dopplertrack(traj):
    _, tracks = render_pass(100, 4000, traj, _tone(100.), level_db=-40, seed=2)
    tr = tracks[0]
    te = np.linspace(-60, 140, 400001)
    to, f, _ = curve_from_trajectory(te, traj(te), np.array([0, 0, 1.5]), f0=100., c=C)
    np.testing.assert_allclose(tr.f, np.interp(tr.t, to, f), atol=2e-3)


def test_audio_follows_truth():
    """Peak of a 2 s zero-padded FFT sits on the truth frequency (incl. a throttle step)."""
    sr = 8000
    comps = prop_components(150., 2, drift=lambda te: .03*.5*(1+np.tanh((np.asarray(te)-20)/3)))
    x, tracks = render_pass(40, sr, turn_track(60, 600, 20, center=(0, 450), alt=150), comps,
                            level_db=-20, seed=3)
    n = 2*sr
    win = np.hanning(n)
    for tr in tracks:
        for tc in (6., 15., 20., 27., 33.):
            seg = x[int((tc-1)*sr):int((tc-1)*sr)+n]*win
            spec = np.abs(np.fft.rfft(seg, 16*n))
            fr = np.fft.rfftfreq(16*n, 1/sr)
            f_true = np.interp(tc, tr.t, tr.f)
            band = np.abs(fr-f_true) < 15
            assert abs(fr[band][np.argmax(spec[band])]-f_true) < .5


def test_ground_reflection_is_phase_consistent():
    """With the mic on the ground the image path equals the direct path: R=1 doubles x."""
    traj = straight_track(100, 10, 50, 150)
    kw = dict(level_db=-30, seed=4, broadband=dict(level_db=-30, band=(60, 1500)))
    x0, _ = render_pass(20, 4000, traj, _tone(300.), mic_height=1e-6, **kw)
    x1, _ = render_pass(20, 4000, traj, _tone(300.), mic_height=1e-6, ground=1., **kw)
    np.testing.assert_allclose(x1, 2*x0, atol=1e-4*np.max(np.abs(x0)))


def test_jitter_widens_tone_not_truth():
    rng_j = lambda te: .005*np.sin(2*np.pi*3*np.asarray(te))
    comps = prop_components(200., 1, jitter=rng_j)
    traj = straight_track(60, 10, 200, 300)
    x, tracks = render_pass(20, 4000, traj, comps, level_db=-30, seed=5)
    _, plain = render_pass(20, 4000, traj, prop_components(200., 1), level_db=-30, seed=5)
    np.testing.assert_array_equal(tracks[0].f, plain[0].f)
    assert tracks[0].width_hz and 1.2 < tracks[0].width_hz < 2.2   # 1.75 * 0.707 Hz * ratio
    assert plain[0].width_hz is None


def test_hard_specs_build():
    specs = hard_specs()
    names = [s['name'] for s in specs]
    assert len(set(names)) == len(names) >= 19
    for i, s in enumerate(specs):
        s = dict(s, duration=min(s['duration'], 30))
        x, tracks, meta = build_hard_clip(s, sr=8000, seed=i)
        assert x.dtype == np.float32 and len(x) == 30*8000 and np.max(np.abs(x)) < .98
        assert meta['null'] == (not tracks)
        assert all(t.kind == 'doppler' and np.all(t.f < .48*8000) for t in tracks)
    x2, _, _ = build_hard_clip(dict(specs[2], duration=10), sr=8000, seed=1)
    x3, _, _ = build_hard_clip(dict(specs[2], duration=10), sr=8000, seed=1)
    np.testing.assert_array_equal(x2, x3)
