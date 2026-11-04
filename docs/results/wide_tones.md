# Wide (jittery) tones: per-track width and skirt removal

**Problem (f7, air_p51_mustang).** The steep core (24.8–31.9 s, 1263 → 587 Hz) stayed visible
even with an accurate fine-grid track (1/16 s). After dechirping, the tone is a ±7–10 Hz jittery
component with a slowly decaying skirt; the fixed ±2 Hz corridor caught 30–50% of it.

Repro: `scripts/diag/wide_tone_repro.py <repo> <track.json> [widths…]`,
`scripts/diag/wide_tone_profile.py` (dechirped power vs offset). Track: f7's
`p51_fine_track.json` (chirps.py, research/detect 5622387).

## Three causes, three fixes (remove.py)

1. **Fixed corridor width.** remove.py now reads `getattr(track, "width_hz", None)` (f7's
   measured core half-width; None = `cfg.width_hz`, floored at it, capped at 20 Hz). Subtract,
   VK, fill, do-no-harm and the TF statistics all use it. In mask mode, wide tracks take the VK +
   fill path. types.py is not touched here.
2. **The fill kept the original during the tone's bursts.** The crossing-partial rule (keep the
   input where the corridor is > 8 dB over its flanks for < 1.5 s; meant for speech crossing a
   corridor) read the jittery tone's own bursts as crossings. The kept 8–19% of the time
   dominated the corridor power (0.81·57 dB + 0.19·67 dB ≈ 61 dB). Measured-wide tracks get no
   crossing keep. Cost: a partial crossing a ≤ ±12 Hz core is filled for ≤ 1.5 s.
3. **The skirt.** Outside the core the dechirped profile decays slowly (p51: 55 dB at 16 Hz,
   51 at 25, background 48–49 dB from ~50 Hz). The fill takes its level from flanks still inside
   the skirt, and the attacker's 2 s STFT integrates the ±40 Hz skirt into a streak. The new
   `_skirt_one` step runs only for measured-wide tracks (`width_hz > cfg.width_hz`), before the
   fill, in the track's warped baseband at 400 Hz, 1 s STFT frames:
   - **background** per frame: median power of the far ring 55–80 Hz (other tracks excluded),
     smoothed over 2 s;
   - **extent** per frame: where the running minimum of the side-averaged profile (outward from
     the core) comes within 1 dB of the background, capped at 50 Hz. Frames that never decay to
     it, or whose core is < 6 dB over it, get no skirt (dense speech or harmonics: no shoulder);
   - **gain**: each ring scaled by sqrt(background / shoulder), where shoulder = that running
     minimum. Attenuation only, never below background. Content riding above the shoulder (a
     crossing speech harmonic) is attenuated only as much as the shoulder under it. Raised-cosine
     outer edge;
   - where the shoulder is still > 10 dB over background (too strong for an STFT gain, whose
     dynamic range is ~20–25 dB), the core width is raised to cover it (90th percentile over
     frames, ≤ 20 Hz). The exact long-FFT fill then removes the strong part.

Narrow tracks (`width_hz` None or ≤ `cfg.width_hz`) take exactly the old path. A test checks
that None, the default and a smaller value give bit-identical output.

## Results

p51, f7's track, dechirped power (dB) vs offset from f(t):

| offset Hz | 0 | 6 | 12 | 16 | 20 | 25 | 30 | 40 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| original | 68.1 | 65.7 | 59.7 | 55.0 | 54.9 | 51.5 | 52.0 | 50.6 | 48.4 | 48.8 |
| default width (before) | 64.6 | 65.7 | 59.7 | 55.0 | 54.9 | 51.5 | 52.0 | 50.6 | 48.4 | 48.8 |
| width_hz 6.8 (f7's value) | 49.3 | 48.9 | 49.5 | 48.9 | 51.0 | 49.8 | 50.6 | 50.5 | 48.4 | 48.8 |

| | ±3 / ±10 / ±20 Hz energy | coherent frac | AF z on the track (40–1950 Hz) | box-free created |
|---|---|---|---|---|
| default width | −2.6 / −0.8 / −0.7 dB | 0.38 | 6.1 → 5.1 | 0 |
| width_hz 6.8 | −17.1 / −16.6 / −14.0 dB | 0.86 | 6.1 → −0.3 | 0 |

Stock and 2 s smoothed views: `docs/results/img/wide_p51_view.png` (local). The steep core is
gone. What remains is a blob at ~1160 Hz, 24–25.5 s, just before the track starts (the pre-CPA
plateau, which this track does not cover). No blue notch streak in view (b).

Synthetic jittery tone ~22 dB over background (`test_measured_width_removes_jittery_tone_without_notch`):
with width_hz 10 the profile is within 1 dB of background from 0 to 45 Hz (default width:
+9 to +17 dB); never below background − 2 dB; created 0. **Limit:** at an extreme ~48 dB tone
SNR a 12–16 dB residual remains. The FM sidebands then reach past the 20 Hz core cap, and the
STFT gain's ~25 dB range cannot cover them.

## Collateral (worst case; `scripts/diag/wide_tone_collateral.py`)

Here every one of a clip's cached dopplerscrub tracks is forced to width_hz 8. These include
dozens of flat false corridors (60 on spitfire_presenter, 180 on prop_pass), so this is far
harsher than f7's chirp cores, which carry a width only where measured:

| clip | variant | speech SNR dB | LSD dB | TF modified 40–1000 | created (40–1950 / 40–450) |
|---|---|---|---|---|---|
| air_spitfire_presenter | default width | 10.6 | 1.18 | 10% | 1 / 1 |
| | all tracks 8 Hz, no skirt | 6.75 | 1.44 | 22% | 3 / 4 |
| | all tracks 8 Hz + skirt | 5.87 | 1.62 | 40% | 0 / 0 |
| | steep tracks only (>15 Hz/s) 8 Hz + skirt | 5.92 | 1.47 | 33% | 1 / 1 |
| prop_pass | default width | 0.62 | 2.06 | 24% | 1 / 1 |
| | all tracks 8 Hz + skirt | 0.19 | 2.60 | 72% | 2 / 2 |
| | steep only + skirt | 0.28 | 2.30 | 51% | 0 / 0 |

Most of the speech loss comes from the wider **core** on many tracks (10.6 → 6.75 dB with no
skirt). The skirt adds ~0.9 dB on top in this worst case. Created ridges stay at the default's
level (0–2). The realistic check must use f7's chirp tracks, where only measured steep cores
carry a width: re-run the presenter clip at integration and compare speech SNR and LSD with
the default.
