# Module contracts (read this first)

Project: **dopplerscrub**, a privacy filter. The attack it defeats is
[mjshiggins/dopplertrack](https://github.com/mjshiggins/dopplertrack), based on
colsto's video. That attack finds faint propeller tones (about 40–450 Hz,
blade-pass fundamental plus harmonics) whose frequency sweeps down as an
aircraft passes. It fits the Doppler curve and matches it against ADS-B
trajectories to locate the microphone. Its pipeline: decode to mono 4 kHz, run
an 8192-pt STFT with a 0.25 s hop over 40–450 Hz, whiten each bin by its running
median, fit `ideal_curve(f0, v, d_cpa, t_cpa)`, then score real trajectories.
Reference copy: `/private/tmp/claude-501/-Users-mazeyarmoeini-Desktop-dev-remove-doppler/5646ee71-81d4-44e6-9616-099cbae20b18/scratchpad/ref/dopplertrack`
(read `src/dopplertrack/{audio,doppler,score,pipeline}.py`). It is also a dev
dependency (`uv sync` → `import dopplertrack`).

Our goal: given any audio or video file, **find each Doppler sweep and remove
it**, leaving speech, music and the rest of the recording intact. Avoid blanket
high- or low-pass filtering. In the cleaned file, the curves must be visibly gone from the
attacker's whitened spectrogram. We verify by looking at before/after spectrograms, not
by running dopplertrack's slow fit/search.

Shared types are in `src/dopplerscrub/types.py` (`Track`, `DetectConfig`,
`RemoveConfig`, `ScrubReport`). Do not rename fields. If you need a new field,
add it with a default and say so in your final report.

Signals are `np.ndarray` float32 with shape `(n,)` (mono) or `(n, ch)`,
values in [-1, 1], plus an int `sr`.

## Module ownership (edit only your own files and your own tests)

| Module | Owner | Public API |
|---|---|---|
| `synth.py` + `data/` + `scripts/fetch_real.sh` | codex1 | `flyover(...)`, `make_dataset(outdir) -> list[dict]` (writes WAVs plus `truth.json` holding exact `Track` dicts) |
| `detect.py` | codex2 | `detect_tracks(x, sr, cfg=DetectConfig()) -> list[Track]` |
| `io.py` + `remove.py` | claude1 | `load(path) -> (x, sr, meta)`, `save(src_path, dst_path, x, sr, meta)` (remuxes video untouched), `remove_tracks(x, sr, tracks, cfg=RemoveConfig()) -> x_clean` |
| `evaluate.py` | claude2 | `attack_spectrogram(x, sr)`, `render_before_after(...)` (verification is VISUAL: before/after whitened spectrograms; we do NOT run dopplertrack's slow fit/search), `fidelity(...)`, `residual_db(...)` |
| `pipeline.py`, `cli.py`, README | orchestrator | `scrub(in_path, out_path, ...) -> ScrubReport` |

Tests go in `tests/test_<module>.py`. Run them with `uv run pytest tests/test_<module>.py -q`.
Keep tests fast (under 20 s each); mark slow ones with `@pytest.mark.slow`.

Do not `git commit`, `git push`, or touch other modules. The orchestrator integrates.
