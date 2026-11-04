<!--
DRAFT by claude2 for the orchestrator. Not README.md.
Fill every {{PLACEHOLDER}} from the frozen run:
  docs/results/benchmark.json, baselines.json (dev), and the heldout reruns.
Image paths are relative to the repo root; regenerate them from the frozen run.
-->

# dopplerscrub

Aircraft and vehicle engines leave faint tones in almost any outdoor recording. As the
source passes, those tones sweep down in pitch (the Doppler effect). Matched against
public ADS-B flight tracks, the shape of that sweep can reveal **where and when a
recording was made**. dopplerscrub finds each Doppler sweep in an audio or video file
and removes it, together with its harmonics. It then fills the gap with matched noise,
so the curve is gone from the attacker's own spectrogram while speech, music and the
rest of the soundtrack are left alone. Video streams are copied untouched and container
metadata (including GPS tags) is stripped.

![colsto recording, event 1: attacker's whitened spectrogram before (top), after (2nd), removed content (3rd), detected tracks (bottom)](docs/results/img/colsto_ev1_ba.png)

*Event 1 of the colsto recording (2060–2230 s) in the attacker's whitened spectrogram
view. The 180→85 Hz sweep and its 2nd harmonic disappear; the stepped engine tones stay.
{{IMAGE_NOTE: regenerate from the frozen run}}*

## The threat

- [colsto's video](https://youtu.be/4CxVxuMVkMc) and the open-source
  [dopplertrack](https://github.com/mjshiggins/dopplertrack) (mjshiggins) show that a
  propeller aircraft's blade-pass tone (about 40–450 Hz, plus harmonics) is audible in
  ordinary recordings.
- The attack decodes the audio to 4 kHz mono, computes an 8192-point STFT (0.25 s hop,
  40–450 Hz), and whitens each frequency bin by its running median.
- In that view a passing aircraft draws a falling curve. Fitting the ideal straight-pass
  model `f(t; f0, v, d_cpa, t_cpa)` gives speed, closest-approach distance and time.
- Scoring those parameters against ADS-B trajectories in the area picks out the aircraft,
  and with it the microphone's position and the recording time.
- Other location leaks are out of scope (see [Limitations](#limitations)); this tool
  removes the Doppler channel.

## Quick start

Requires Python ≥ 3.12 and `ffmpeg` on PATH.

```bash
# with uv
git clone {{REPO_URL}} && cd dopplerscrub
uv sync
uv run dopplerscrub clean video.mp4 --plot

# or with pip
pip install {{PIP_TARGET}}        # e.g. git+{{REPO_URL}}
dopplerscrub clean video.mp4 --plot
```

This writes `video.scrubbed.mp4` (video stream copied, audio replaced, metadata
stripped) and `video.scrubbed.png` (before/after spectrogram in the attacker's view).
Any ffmpeg-readable audio or video file works.

| Flag | Default | Meaning |
|---|---|---|
| `-o, --output PATH` | `<name>.scrubbed.<ext>` | output file (one input) or directory (several) |
| `--plot` | off | write a before/after attacker-view spectrogram PNG |
| `--report` | off | write `<output>.json` with every detected track |
| `--method {vk,subtract,mask}` | `vk` | removal: joint Vold-Kalman (default), track-locked subtraction, or STFT notch |
| `--no-harden` | harden on | skip the noise fill (leaves a curve-shaped notch; not recommended) |
| `--all-tones` | off | also remove stationary tones (mains hum, machinery) |
| `--passes N` | 2 | detect/remove rounds (a loud tone can mask a quieter one) |
| `--fmin / --fmax HZ` | 30 / 2000 | searched band |
| `--keep-metadata` | off | keep container metadata (default strips it, incl. GPS/location tags) |
| `-q, --quiet` | off | no progress messages |

`dopplerscrub detect FILE` lists detected Doppler and stationary tones without
modifying anything.

Runtime is roughly {{RUNTIME_PER_MIN}} s per minute of audio on {{HW}}. Long recordings
with many sources are slower; the one-hour colsto file took {{COLSTO_RUNTIME}}.

## How it works

```
input ─► decode (ffmpeg) ─► attacker-view spectrogram ─► ridge tracker ┐
                                                        + matched pursuit├─► tracks
                                                                         ┘
tracks ─► classify (moving Doppler vs stationary) ─► Vold-Kalman removal (do-no-harm)
       ─► matched noise fill along each corridor ─► remux (video copied, metadata stripped)
```

Design notes and citations are in [docs/research.md](docs/research.md).

1. **Attacker-view spectrogram.** Work in the same representation the attack uses: long
   (~2 s) windows for 0.5 Hz resolution, whitened per bin. A shorter view is fused in so
   steep sweeps near closest approach are not smeared.
2. **Detection** (`ridges.py`, `detect.py`). Two paths run together:
   - An HMM/Viterbi ridge tracker on whitened log-likelihood ratios, which groups ridges
     into harmonic families.
   - A parametric matched pursuit over the ideal-pass model, with harmonics combined.

   False alarms are calibrated against reversed-time controls and a Gumbel maximum model
   (research.md "Detection", D1/D2).
3. **Classification.** A track is labelled `doppler` when it is a coherent, monotonic
   sweep consistent with a moving source (`doppler_score` ≥ threshold), else
   `stationary`. Only Doppler tracks are removed by default.
4. **Removal** (`remove.py`). Joint multi-order **Vold-Kalman** filtering extracts each
   tracked tone's slowly varying envelope along f(t) and subtracts it (research.md R2).
   A **do-no-harm** pass guarantees no corridor ends up with more energy than the input,
   so a wrong track cannot inject a new curve.
5. **Noise fill.** Removal leaves a curve-shaped hole that an attacker can see as easily
   as the tone. Each corridor is refilled with noise shaped and scaled to the local
   flanking-bin spectrum, as in codec noise substitution (research.md "Noise fill" and
   the ENF anti-forensics lessons).
6. **Remux.** Audio is re-encoded in the source codec; video and other streams are copied
   bit-exact; global, per-stream and chapter metadata (location tags included) are
   dropped unless `--keep-metadata`.

## Evaluation methodology

We judge success the way an attacker would look, not by running dopplertrack's
(slow) fit and ADS-B search. All tools are in `src/dopplerscrub/evaluate.py`.

- **Attacker view.** `attack_spectrogram` calls dopplertrack's own `compute_spectrogram`
  (4 kHz, 8192 points, 0.25 s hop, 40–450 Hz, 600 s running-median whitening). Every
  clip is rendered before/after, plus red-team variants: low floor, a diverging scale
  (ghost vs notch), global whitening, 2 s smoothing, and a curve-aligned profile. Each
  PNG is inspected by eye and a verdict recorded.
- **Integrated z test.** An attacker who knows roughly where the curve is can sum power
  along all ~1000 frames of a pass. We compute on-curve power over the local per-bin
  floor, integrated over the whole curve. Significance is calibrated empirically against
  the same statistic on curves shifted ±8–14 Hz. **z > 3 = still detectable.** The
  research target is a 95% upper bound of the residual ≥ 10 dB below the floor.
- **Anti-forensic checks** (from ENF-removal literature): band-edge steps, corridor
  variance and kurtosis vs flanking bins, and leftover harmonics at k·f(t).
- **Fidelity.**
  - SNR outside the removed corridors, and speech-band (300–3400 Hz) SNR.
  - Log-spectral distance (LSD).
  - A STOI-style intelligibility proxy (implemented from Taal et al. 2011).
  - The fraction of the time-frequency plane modified.
- **Baselines** (`baselines.py`), applied to the same clips:
  - Butterworth band-stop 40–450 Hz.
  - **Brick-wall** band-stop (STFT bins 35–480 Hz zeroed; the strongest blanket method).
  - Low-band noise at 0 dB SNR.
  - ±1% pitch/time jitter.
  - An **oracle**: our remover fed the true tracks (synthetic only), which separates
    detector errors from remover errors.

  All outputs are quantised to 16-bit PCM before scoring.
- **Data splits.** **dev**: synthetic flyovers with exact truth (`data/synth`), the
  colsto recording and its event windows, short real clips, and an air-show set
  (`data/real`, `data/real/airshow`). All dev clips were used during tuning.
  **held-out**: `data/real/heldout`, collected after the detector was frozen and
  evaluated once. Results are reported per split.

## Results

From the frozen run ({{FREEZE_COMMIT}}, {{FREEZE_DATE}}). Full tables:
[benchmark.md](docs/results/benchmark.md), [baselines.md](docs/results/baselines.md).

### Hiding the curves

| split | clips with a curve visible before | curves hidden (z ≤ 3 after) | ≤ −10 dB target met | visual verdict: gone / faint ghost / visible |
|---|---|---|---|---|
| dev, synthetic | {{DEV_SYN_N}} | {{DEV_SYN_HIDDEN}} | {{DEV_SYN_TARGET}} | {{DEV_SYN_VERDICTS}} |
| dev, real (colsto, short clips) | {{DEV_REAL_N}} | – | – | {{DEV_REAL_VERDICTS}} |
| dev, air show | {{DEV_AIR_N}} | – | – | {{DEV_AIR_VERDICTS}} |
| **held-out** | {{HO_N}} | {{HO_HIDDEN}} | {{HO_TARGET}} | {{HO_VERDICTS}} |

### Targeted vs blanket (medians over clips with a visible curve)

| method | AF z after ↓ | ridge cells in event boxes ↓ | SNR outside dB ↑ | speech SNR dB ↑ | LSD dB ↓ | STOI-like ↑ | TF modified ↓ |
|---|---|---|---|---|---|---|---|
| original | {{Z_ORIG}} | {{BOX_ORIG}} | ∞ | ∞ | 0 | 1.00 | 0% |
| **dopplerscrub** | {{Z_DS}} | {{BOX_DS}} | {{SNR_DS}} | {{SP_DS}} | {{LSD_DS}} | {{STOI_DS}} | {{TF_DS}} |
| oracle (true tracks) | {{Z_OR}} | – | {{SNR_OR}} | {{SP_OR}} | {{LSD_OR}} | {{STOI_OR}} | {{TF_OR}} |
| brick-wall band-stop | {{Z_BW}} | {{BOX_BW}} | {{SNR_BW}} | {{SP_BW}} | {{LSD_BW}} | {{STOI_BW}} | {{TF_BW}} |
| Butterworth band-stop | {{Z_BS}} | {{BOX_BS}} | {{SNR_BS}} | {{SP_BS}} | {{LSD_BS}} | {{STOI_BS}} | {{TF_BS}} |
| 0 dB low-band noise | {{Z_NO}} | {{BOX_NO}} | {{SNR_NO}} | {{SP_NO}} | {{LSD_NO}} | {{STOI_NO}} | {{TF_NO}} |
| ±1% jitter | {{Z_JI}} | {{BOX_JI}} | {{SNR_JI}} | {{SP_JI}} | {{LSD_JI}} | {{STOI_JI}} | {{TF_JI}} |

Two findings held throughout development and are worth stating up front:

- **Blanket filtering mostly does not work against this attacker.** Because it whitens
  each bin, a band-stop's attenuation is undone wherever it is less than the tone's
  margin over the output noise floor. A Butterworth band-stop leaves harmonics near its
  edges fully visible. Low-band noise leaves the curves visible, and jitter only wobbles
  them. Only the brick-wall band-stop hides everything, by erasing the whole band.
- **Targeted removal with correct tracks hides as well as the brick wall at a small
  fraction of the damage** (oracle row). How close dopplerscrub gets to the oracle is
  a measure of the detector. {{ONE_SENTENCE_FROZEN_RESULT}}

### Limitations

What dopplerscrub does **not** defeat, or not reliably:

- **A better detector than ours.** We remove what we detect. Curves we miss (see
  [detector_diagnosis.md](docs/results/detector_diagnosis.md): incomplete time coverage
  of a pass, unproposed harmonics, dense harmonic fans) remain for an attacker with a
  more sensitive tracker, longer integration, or a known flight to search for.
- **Very faint curves.** Curves below our detection threshold are left alone. If they
  sit just above an integrating attacker's threshold, they can still be found. We test
  the stock attack and an integrating attacker, not every conceivable one.
- **Other attack settings.** We evaluate the stock 8192-point attack. An attacker using
  shorter windows, a different band (e.g. jet or rotor content outside 30–2000 Hz), or
  multi-microphone data may see things we did not test.
- **Imperfect concealment of the edit itself.** The noise fill is statistically matched
  but not perfect. Corridor variance and edge statistics can reveal *that* something
  was removed, and where, even when the curve is gone.
- **Collateral changes.** On recordings dominated by the aircraft itself, removing it
  changes the sound substantially; that is the point, but it is audible.
- **Other location and time leaks**, which this tool does not address:
  - background speech and announcements
  - electrical network frequency (ENF) in mains hum, which can date a recording
  - room acoustics and reverb
  - bird species, church bells, traffic patterns
  - visual content
  - metadata outside the container: file names, sidecar files, upload-platform
    records, and the original file you still have
- **Long files are slow** ({{COLSTO_RUNTIME}} for one hour on {{HW}}).

## Responsible use

dopplerscrub is a **privacy defence**: it removes one specific signal that can geolocate
the person who made a recording. It does not hide a recording's content, and it is not
evidence tampering protection. Do not use it to alter recordings you are obliged to keep
intact.

The attack is not ours. Credit to [colsto](https://youtu.be/4CxVxuMVkMc) for
demonstrating it and to [mjshiggins/dopplertrack](https://github.com/mjshiggins/dopplertrack)
for the open implementation we use as the reference attacker in every evaluation.
Publishing both the attack and a defence lets people who publish audio and video
judge the risk themselves.

## Development

```bash
uv sync                                          # dev deps include dopplertrack (reference attacker)
uv run pytest -q                                 # unit tests (fast; slow ones marked)
uv run python scripts/benchmark.py --no-colsto   # dev benchmark -> docs/results/benchmark.md
uv run python scripts/benchmark.py --split heldout
uv run python scripts/baselines.py               # targeted vs blanket -> docs/results/baselines.md
uv run python scripts/diagnose_detector.py       # per-curve detector diagnosis
```

- **Data.** `data/synth` is regenerated with `dopplerscrub.synth.make_dataset` and has
  exact truth tracks. Real clips are fetched with `scripts/fetch_real.sh` and are not
  committed (see `data/real/SOURCES.md` for provenance and licences).
- **Visual verdicts.** `docs/results/verdicts.json` and `baselines_verdicts.json` hold
  the by-eye verdicts behind the tables. Rerun the scripts, look at the PNGs, update the
  JSON, then regenerate with `--md-only`.
- **Layout.**

  | Module | Role |
  |---|---|
  | `io.py` | ffmpeg decode/remux and metadata stripping |
  | `ridges.py`, `detect.py` | detection |
  | `remove.py` | removal and noise fill |
  | `pipeline.py`, `cli.py` | orchestration and command line |
  | `evaluate.py`, `baselines.py` | evaluation |
  | `synth.py` | synthetic data |
  | `types.py` | shared interfaces |
- **How it was built.** dopplerscrub was developed by a team of coding agents (two Claude
  Code and two Codex agents, coordinated by a Claude Code orchestrator) working in
  parallel on separate modules against the contracts in `docs/CONTRACTS.md`, with a
  dedicated evaluation agent red-teaming every change visually. Humans set the goals,
  reviewed results and own the final decisions. {{OPTIONAL: link to a write-up}}

## License

MIT. dopplertrack is used only as a development dependency for evaluation; see its own
license.
