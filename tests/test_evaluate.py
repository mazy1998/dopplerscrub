import numpy as np
import pytest

from dopplerscrub import evaluate as ev
from dopplerscrub.types import Track

SR = 8000
C = 343.0


def doppler_f(t, f0=120.0, v=60.0, d=300.0, tcpa=60.0):
    r = np.sqrt(d ** 2 + (v * (t - tcpa)) ** 2)
    rr = v * v * (t - tcpa) / r
    return f0 * C / (C + rr)


@pytest.fixture(scope="module")
def clip():
    rng = np.random.default_rng(0)
    n = 120 * SR
    t = np.arange(n) / SR
    f = doppler_f(t)
    tone = 0.02 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    speech = 0.05 * np.sin(2 * np.pi * 1000 * t) * (np.sin(2 * np.pi * 0.3 * t) > 0)
    noise = 0.01 * rng.standard_normal(n)
    clean = (noise + speech).astype(np.float32)
    orig = (clean + tone).astype(np.float32)
    tt = np.arange(0, 120.25, 0.25)
    track = Track(tt, doppler_f(tt), np.full(len(tt), 0.02), kind="doppler")
    return orig, clean, track


def test_matches_dopplertrack(clip):
    pytest.importorskip("dopplertrack")
    orig, _, _ = clip
    ref = ev.attack_spectrogram(orig, SR)
    v = ev.attack_views(orig, orig, SR)
    assert ref.S.shape == v["orig"].S.shape
    np.testing.assert_allclose(v["orig"].S, ref.S, atol=1e-3)
    np.testing.assert_allclose(v["orig"].t, ref.t)


def test_wide_band(clip):
    s = ev.attack_spectrogram_wide(clip[0], SR)
    assert s.f.max() > 1400 and s.f.min() >= 40


def test_contrast_ghost_vs_clean(clip):
    orig, clean, tr = clip
    v = ev.attack_views(orig, clean, SR)
    assert ev.track_contrast_db(v["orig"], [tr]) > 6
    assert abs(ev.track_contrast_db(v["clean"], [tr])) < 1.5
    assert ev.track_contrast_db(v["diff"], [tr]) > 6


def test_residual_and_fidelity(clip):
    orig, clean, tr = clip
    assert ev.residual_db(orig, orig, SR, [tr]) == pytest.approx(0, abs=1e-6)
    r = ev.residual_db(orig, clean, SR, [tr])
    assert r < -10
    det = ev.residual_details(orig, clean, SR, [tr])[0]
    assert det["tone_snr_before_db"] > 10
    fid = ev.fidelity(orig, clean, SR, exclude_tracks=[tr])
    assert fid["snr_outside_db"] > 25        # only the tone was removed
    assert fid["speech_snr_db"] > 30
    assert fid["snr_db"] < fid["snr_outside_db"]
    assert abs(fid["octave_change_db"][1000]) < 0.1
    assert fid["octave_change_db"][125] < -1   # tone energy went away there
    ident = ev.fidelity(orig, orig, SR)
    assert ident["snr_db"] == float("inf")


def test_stereo_and_length_mismatch(clip):
    orig, clean, tr = clip
    st = np.stack([orig, orig], axis=1)
    assert ev.residual_db(st, clean[:-100], SR, [tr]) < -10


def test_render(tmp_path, clip):
    orig, clean, tr = clip
    p = tmp_path / "ba.png"
    views = ev.render_before_after(orig, clean, SR, p, tracks=[tr], window=(20, 100, 80, 160))
    assert p.stat().st_size > 10_000
    p2 = tmp_path / "ba2.png"
    ev.render_before_after(orig, clean, SR, p2, views=views)
    assert p2.exists()
    rt = ev.render_attacker_views(clean, SR, tmp_path / "rt.png", tracks=[tr], x_orig=orig)
    assert (tmp_path / "rt.png").exists()
    assert abs(rt["contrast_db"]["smoothed"]) < 1.0
    assert rt["contrast_db"]["orig_standard"] > 6


def test_curve_profile(clip):
    orig, clean, tr = clip
    v = ev.attack_views(orig, clean, SR)
    o, p_orig = ev.curve_profile(v["orig"], [tr])
    _, p_clean = ev.curve_profile(v["clean"], [tr])
    i0 = np.argmin(np.abs(o))
    assert p_orig[i0] - np.median(p_orig) > 6
    assert abs(p_clean[i0] - np.median(p_clean)) < 1.5


def test_notch_reads_negative(clip):
    """A curve-shaped hole (noise removed along the track) must read as negative contrast."""
    _, clean, tr = clip
    v = ev.attack_views(clean, clean, SR)
    notched = v["clean"]
    S = notched.S.copy()
    for tt, ff in ev._track_points([tr]):
        ti = np.clip(np.searchsorted(notched.t, tt), 0, len(notched.t) - 1)
        fi = np.clip(np.searchsorted(notched.f, ff), 0, len(notched.f) - 1)
        for d in (-1, 0, 1):
            S[np.clip(fi + d, 0, len(notched.f) - 1), ti] -= 10
    assert ev.track_contrast_db(ev.Spectrogram(notched.t, notched.f, S), [tr]) < -3


def test_anti_forensics(clip):
    orig, clean, tr = clip
    af = ev.anti_forensics(orig, clean, SR, [tr])
    assert af["residual"]["orig"]["z"] > 10
    assert af["residual"]["clean"]["z"] < 3
    assert af["pass_target"]
    assert abs(af["edge_step_db"]) < 1.0
    assert abs(af["stats"]["clean"]["var_ratio_db"]) < 1.5
    # a 10%-amplitude leftover (-20 dB) is NOT safe: integrated over the pass it shows
    bad = clean + 0.1 * (orig - clean)
    af2 = ev.anti_forensics(orig, bad, SR, [tr])
    assert af2["residual"]["clean"]["z"] > 3 and not af2["pass_target"]


def test_anti_forensics_harmonic_and_notch():
    rng = np.random.default_rng(1)
    n = 120 * SR
    t = np.arange(n) / SR
    ph = 2 * np.pi * np.cumsum(doppler_f(t, f0=100)) / SR
    noise = 0.01 * rng.standard_normal(n)
    orig = (noise + 0.02 * np.sin(ph) + 0.01 * np.sin(2 * ph)).astype(np.float32)
    clean = (noise + 0.01 * np.sin(2 * ph)).astype(np.float32)   # only k=1 removed
    tt = np.arange(0, 120, 0.25)
    tr = Track(tt, doppler_f(tt, f0=100), np.zeros(len(tt)), kind="doppler")
    af = ev.anti_forensics(orig, clean, SR, [tr])
    assert [k for _, k in af["leftover_harmonics"]] == [2]
    # brick-wall STFT notch (+-2 Hz zeroed, no fill) -> negative edge step, low corridor power
    import scipy.signal as ss
    fz, tz, Z = ss.stft(noise, SR, nperseg=4096)   # notch cut into pure noise
    fc = np.interp(tz, tt, doppler_f(tt, f0=100))
    for k in (1, 2):
        Z[np.abs(fz[:, None] - k * fc[None]) <= 2.0] = 0
    _, notched = ss.istft(Z, SR, nperseg=4096)
    af3 = ev.anti_forensics(noise.astype(np.float32), notched[:n].astype(np.float32), SR, [tr])
    assert af3["edge_step_db"] < -3
    assert af3["stats"]["clean"]["corridor"]["mean"] < 0.5 * af3["stats"]["clean"]["flank"]["mean"]


# ----------------------------------------------------------------------------- dopfast attacker

def _df_pass(n, sr, amp, v=70.0, d=160.0, h=30.0, f0=110.0, harmonics=(1, 2, 3)):
    from dopplerscrub import evaluate as ev
    tt = np.arange(n) / sr
    ph = 2 * np.pi * np.cumsum(f0 * ev.dopfast_g(tt - h, v, d)) / sr
    return sum(amp / j * np.cos(j * ph) for j in harmonics)


def test_dopfast_g_shape():
    from dopplerscrub import evaluate as ev
    w = np.linspace(-20, 20, 401)
    g = ev.dopfast_g(w, 100.0, 2500.0)
    assert abs(ev.dopfast_g(np.array([0.0]), 100.0, 2500.0)[0] - 1.0) < 1e-12
    assert np.all(np.diff(g) < 0)                         # falls through the pass
    assert g[0] > 1.0 > g[-1]


def test_dopfast_finds_faint_pass_and_its_shape():
    from dopplerscrub import evaluate as ev
    sr, n = 8000, 60 * 8000
    noise = 0.01 * np.random.default_rng(0).standard_normal(n)
    x = noise + _df_pass(n, sr, 0.004)
    r = ev.dopfast(x, noise, sr, nulls=3)
    assert r["z_orig"] > 10
    e = r["event_orig"]
    assert e["v"] == 70.0 and e["d"] == 160.0 and abs(e["s"] - 30.0) < 0.5 and abs(e["f0"] - 110) < 3
    assert r["z_clean"] < 3 and r["z_at_orig_clean"] < 3


def test_dopfast_events_usable_before_none_after():
    from dopplerscrub import evaluate as ev
    sr, n = 8000, 90 * 8000
    noise = 0.01 * np.random.default_rng(2).standard_normal(n)
    x = noise + _df_pass(n, sr, 0.004, h=45.0)
    r = ev.dopfast(x, noise, sr)
    assert r["n_usable_orig"] == 1 and r["n_events_orig"] == 1
    e = r["events_orig"][0]
    assert abs(e["s"] - 45.0) < 1.5 and e["usable"]
    assert r["n_usable_clean"] == 0 and r["pass_known_clock"] is True
    assert r["pass_unknown_clock"] is None              # only 1 usable before: n/a


def test_dopfast_plateau_only_is_not_usable():
    """A remover that takes only the steep CPA core leaves plateaus: not a usable event."""
    from dopplerscrub import evaluate as ev
    sr, n = 8000, 90 * 8000
    tt = np.arange(n) / sr
    noise = 0.01 * np.random.default_rng(3).standard_normal(n)
    full = _df_pass(n, sr, 0.004, h=45.0)
    fade = np.clip((np.abs(tt - 45.0 - 160 / ev.DF_C) - 2 * 160 / 70) / 1.0, 0, 1)
    r = ev.dopfast(noise + full, noise + full * fade, sr)
    assert r["n_usable_orig"] == 1 and r["n_usable_clean"] == 0


def test_dopfast_noise_and_reversed_pass_are_null():
    from dopplerscrub import evaluate as ev
    sr, n = 8000, 60 * 8000
    noise = 0.01 * np.random.default_rng(1).standard_normal(n)
    rising = noise + _df_pass(n, sr, 0.004)[::-1]           # time-reversed pass: not a Doppler curve
    r = ev.dopfast(noise, rising, sr, nulls=3)
    assert r["z_orig"] < 3
    assert r["peak_clean"] < r["null_mean_clean"] + 6 * r["null_sd_clean"]
