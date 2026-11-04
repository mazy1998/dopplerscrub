# Detector diagnosis: why dopplerscrub fails the integrated test

For codex2 (detect.py) and codex1 (ridges.py). Written by claude2; no detector code
was changed. Reproduce with `uv run python scripts/diagnose_detector.py` (about 2.5 min);
raw numbers are in `detector_diagnosis.json`.

**Snapshot:** 2026-11-04 ~02:05 (detect.py 01:57, ridges.py 02:00, remove.py 01:59).
The default `DetectConfig` / `RemoveConfig` (`vk`, `harden=True`, `harden_min_score=0.5`)
was run through `pipeline.scrub_array`, with `ridges.track_ridges` also run directly on
`ridge_spectrogram`. The oracle is `remove_tracks` fed the TRUE tracks.

**Method, per truth curve.** "Visible span" = frames where the 2 s-smoothed whitened
attacker dB at f(t) is above 1.5 dB in the original. Coverage = share of visible frames
where some track lies within 1 Hz (3 Hz in brackets). z = integrated on-curve power over
the per-bin floor (`evaluate._curve_vs_floor`, null from ±8–14 Hz shifted copies).
z > 3 means an attacker who integrates along the pass detects it.

## Per-curve table (synthetic)

| clip | curve (truth f range) | visible s | z before → after | oracle z | raw ridges cov | any track cov 1 Hz (3 Hz) | doppler cov | harden-filled cov | best track: score, RMS err, time cov | heterodyne residual dB | diagnosis |
|---|---|---|---|---|---|---|---|---|---|---|---|
| overlap | A h1 83–125 Hz | 101 | 32.7 → **10.0** | 0.6 | 0.31 | 0.33 (0.33) | 0.33 | 0.33 | 0.85, 0.03 Hz, 0.33 | – | **time coverage**: track spans 49–88 s + 88–101 s only; flat approach and recede parts missed |
| overlap | A h2 166–251 Hz | 75 | 13.4 → **4.3** | −1.4 | 0.00 | 0.39 (0.39) | 0.39 | 0.39 | 0.85, 0.05 Hz, 0.39 | – | time coverage (49–119 s) |
| overlap | A h3 250–376 Hz | 71 | 8.4 → **5.5** | −3.2 | 0.00 | 0.35 (0.36) | 0.35 | 0.35 | 0.85, 0.08 Hz, 0.35 | – | time coverage (66–84 s only) |
| overlap | B h1 117–193 Hz | 100 | 31.9 → **8.1** | −1.8 | 0.00 | 0.51 (0.51) | 0.51 | 0.51 | 0.85, 0.07 Hz, 0.38 | – | time coverage (32.5–65.5 s) |
| overlap | B h2 234–385 Hz | 79 | 6.0 → −0.5 | −1.7 | 0.00 | 0.70 (0.70) | 0.70 | 0.70 | 0.87, 0.10 Hz, 0.70 | – | OK |
| speech | h1 107–151 Hz | 70 | 17.8 → **6.7** | −0.3 | 0.00 | 0.35 (0.35) | 0.35 | 0.35 | 0.85, 0.16 Hz, 0.35 | −2.0 | **time coverage** (27.75–52 s); not from track_ridges (0 ridges) |
| speech | h2 214–302 Hz | 57 | 9.6 → **9.6** | −2.1 | 0.00 | 0.00 | 0 | 0 | none | 0.0 | **never detected**: no harmonic proposal from the detected h1 |
| speech | h3 320–453 Hz | 42 | 2.6 → 2.6 | −1.2 | 0.00 | 0.00 | 0 | 0 | none | 0.0 | not detectable anyway (z < 3) |
| stereo | h1 106–169 Hz | 64 | 5.8 → **5.8** | 0.6 | 0.00 | 0.00 | 0 | 0 | none | 0.0 | **never detected**: no ridge, no track at all on this clip |
| stereo | h2 212–338 Hz | 53 | 2.5 → 2.5 | −2.1 | 0.00 | 0 | 0 | 0 | none | 0.0 | not detectable anyway |
| stereo | h3 318–507 Hz | 44 | 1.1 → 1.1 | −1.8 | 0.00 | 0 | 0 | 0 | none | 0.0 | not detectable anyway |
| prop_quiet | h1 93–135 Hz | 84 | 46.4 → −0.2 | −0.4 | 0.42 | 1.00 | 1.00 | 1.00 | 0.85, 0.03 Hz, 1.0 | – | OK |
| prop_quiet | h2 186–271 Hz | 68 | 22.3 → −0.7 | −0.9 | 0.50 | 1.00 | 1.00 | 1.00 | 0.85, 0.08 Hz, 1.0 | – | OK |
| prop_quiet | h3 279–406 Hz | 70 | 29.2 → 1.3 | 2.9 | 0.40 | 0.98 | 0.98 | 0.98 | 0.90, 0.30 Hz, 1.0 | – | OK (the oracle is no better) |
| prop_faint, music, long_hum60 | all curves | – | ≤ 2.2 before | – | 0 | 0 | 0 | 0 | none | – | not detectable to the attacker (z < 3); correctly left alone |

Answers to the four questions for the failing curves:
- **(a) Detected?** Every curve with z_before > 3 is either found with ≥ 0.33 time
  coverage (overlap, speech h1) or not at all (speech h2, stereo h1). `track_ridges` alone
  returns **0 ridges** on speech, stereo, prop_faint and long_hum60 and **1** on overlap.
  The tracks that do exist on these clips come from detect.py's own search, not the
  ridge tracker.
- **(b) Labelled doppler / filled?** Yes for every detected curve: score 0.85–0.87, which
  is at least `harden_min_score`, and coverage by doppler tracks equals coverage by
  filled tracks. Labelling and the harden gate are **not** the problem.
- **(c) Residual: time or frequency?** **Time.** Where tracks exist, frequency error is
  tiny (RMS 0.03–0.16 Hz; 1 Hz and 3 Hz coverage are identical). But each track covers
  only 33–51% of the curve's visible span: the steep middle of the pass, not the long
  near-constant approach and recede plateaus. Those plateaus remain and carry the
  integrated z (overlap A h1: 10.0 after vs 0.6 for the oracle, at identical frequency
  accuracy). The remover is not the limit: on every curve the oracle reaches z ≤ 0.6,
  except prop_quiet h3 (2.9), where dopplerscrub is equal or better.
- **(d) prop_pass:** see below.

## prop_pass: why the damage matches the brick wall

127 tracks; 126 doppler-labelled (and filled), 1 stationary. Removed energy was
attributed with a 0.68 s / 1.5 Hz STFT at 48 kHz over each doppler corridor
(±(width_hz + 1 bin), ±pad_s):

| quantity | value |
|---|---|
| removed energy inside doppler corridors | **99.1%** |
| corridors' share of the TF plane | 1.0% |
| removed vs original, inside corridors | −0.06 dB (everything in them is removed) |
| removed vs original, outside corridors (true collateral) | −10.9 dB (SNR 10.9 dB outside corridors) |
| speech band (300–3400 Hz): removed energy outside corridors | 4.8% |
| removed energy from tracks outside the annotated event (15–38 s, 55–450 Hz) | about 7% |
| tracks with median f > 300 Hz | 96 of 127 (proposed harmonics up to h20 ≈ 1974 Hz) |
| near-duplicate track pairs | 3 (e.g. two h3 tracks at 226 Hz, 7.75/8.0–49.5 s) |

- **Most of the "damage" is the target itself.** prop_pass is a close Cessna pass whose
  harmonic stack *is* most of the clip's energy, including 300–3400 Hz. Removing those
  harmonics drops speech-band SNR to about 1.4 dB, and SNR/STOI count that as damage. The
  energy removed outside the corridors is small (−10.9 dB). For clips like this, waveform
  fidelity has to be measured outside the corridors (as `fidelity(..., exclude_tracks)`
  does), not over the whole band.
- **Not over-wide, but over-many and fragmented.** The source is split into about 60
  groups with inconsistent harmonic numbers (the same 343 Hz ridge is "h11", "h12" or
  "h13"), plus 3 duplicates and proposals out to 2 kHz.
- **Why the 300–450 Hz fan stays visible is coverage, not removal.** Along each of the
  ~38 detected fan tracks (18–32 s) z drops from 13–185 to about 0–1.5 (one exception:
  h11 g22 at 13.6). But in the cleaned attacker view, **70% of the bright cells left at
  300–450 Hz, 20–30 s (3057 of 4350) are more than 3 Hz from any doppler track**, and the
  strongest leftover ridge typically sits 4–6 Hz beside a removed one. Closely spaced
  secondary ridges (a second propeller or modulation sidebands, per the annotation) are
  never tracked.
- **Run-to-run variation:** group numbers differed between two runs a few minutes apart
  (g30 vs g37 for the main fan). Either the detector is nondeterministic or the code
  changed between runs; worth checking.

## Top 3 root causes, ranked by impact on z

Impact = summed z left by the cause across failing synthetic curves (dopplerscrub z
after minus oracle z, curves with z_before > 3).

1. **Track spans are clipped to the steep "moving" part of the pass (Σ Δz ≈ 41: overlap
   A h1 9.4, B h1 9.9, A h3 8.7, A h2 5.7; speech h1 7.0).** Frequency accuracy is
   excellent (≤ 0.16 Hz RMS) but coverage is only 33–51% of the visible span. The flat
   approach and recede plateaus, which an attacker integrates just as well, are left in
   place. ridges.py says so explicitly ("Parent spans are clipped before expansion, so no
   source is extended beyond its trusted moving span"). **Fix (codex1/codex2):** after
   confirming a moving source, extend each track forward and backward along its own
   ridge for as long as the per-frame evidence stays above the noise floor. Follow the
   fitted ideal-pass curve as a prior, gated by local evidence, like `_probe_known_harmonics`
   but in time. Stop when evidence ends, not at the end of the steep part. This alone
   brings overlap and speech h1 to the oracle's z.
2. **Missed harmonics of a detected source, and ridge-tracker recall at moderate SNR
   (Σ Δz ≈ 17: speech h2 11.7, stereo h1 5.2).** In speech, h1 is found but h2 at exactly
   2·f(t) (z 9.6, clearly visible) is never proposed: harmonic proposals exist only for
   `track_ridges` families, and this source came from detect.py's other path.
   `track_ridges` itself returns nothing on speech and stereo (single-curve z 5.8–17.8).
   **Fix:** (a) run the k·f(t) proposal and coherence gate for every accepted doppler
   source, whichever path found it. (b) Lower `track_ridges`' acceptance (`min_score` /
   LLR threshold in `_viterbi_lines`) or add a harmonic-sum detector. The stereo curve
   is detectable when its 3 harmonics are integrated jointly (the attacker effectively
   does this). Calibrate on the reversed-time null so false alarms stay bounded.
3. **Dense harmonic fans are under-covered and fragmented (prop_pass, real; no truth
   z).** 70% of the residual bright fan cells are more than 3 Hz from any track; about 60
   groups with inconsistent harmonic numbering; duplicates; proposals up to 2 kHz. This
   both leaves curves visible and inflates the number of filled corridors.
   **Fix:** consolidate one source's harmonics into a single family with a shared f0(t)
   (fit k·f0(t) jointly; dedupe tracks within 1 Hz and overlapping in time). Allow
   *two* families when ridges sit at a constant 4–6 Hz offset (twin props or sidebands).
   Cap proposals at the highest harmonic with local evidence instead of cfg.fmax.

**Not root causes:** doppler labelling and the harden gate (every detected curve scored
0.85–0.87 and was filled); frequency accuracy; and the remover (the oracle reaches z ≤
0.6 on every failing curve).

Secondary, for fidelity: proposed harmonics with no truth (overlap gets h7, h10, h13 at
800–2000 Hz) create false corridors. That is the source of dopplerscrub's 4.5%
modified cells above 450 Hz in `baselines.md`. Gate proposals on local evidence (see
cause 3).
