# Final run: heldout2

Code `b125b22542d7638d9feeaef83d0561bb66fc6e88` (freeze-1-29-gb125b22); phase 1 started 2026-11-04 23:06:05 +0100; inputs verified 11/11 against the annotation SHA-256 (see MANIFEST.md). Visual verdicts were written from the images before this file was generated.

## Headline (fixed-wing recordings)

> **Correction (coordinator, after reading the numbers): the 11/11 / 10/11 tally above is NOT evidence of success.** The frozen blind dopfast attacker found 0 usable passes in all 11 ORIGINAL recordings, so the criterion is vacuous (and its one 'usable' event, cub after cleaning, sits on a clip with no visible pass, i.e. a false alarm). Judge this run by the visual verdicts and box-free ridges: props (6) = 2 PASS (kingair tones, c172), 2 PARTIAL (c150, baron), 1 FAIL (pc12, tone never detected), 1 N/A (cub); jets (5) = nothing removed on any jet; 3 show no tonal pass to remove (a320, citation, harrier), alphajet's clear tones are missed, b737's curves are ground-reflection fringes. Box-free z>3 ridges: 86 before -> 55 still, 0 created.


Pass criterion (geolocation paper, docs memory): ≤1 usable pass left with an unknown recording clock, 0 with a known clock.

- usable passes left ≤1: **11/11** recordings; 0 left: **10/11**

| recording | usable passes before → after | box-free ridges z>3 before → still (created) | speech SNR dB | LSD dB | LSD >450 Hz dB | visual verdict |
|---|---|---|---|---|---|---|
| h2_c150_lowpass | 0 → 0 | 17 → 1 (created 0) | 5.3 | 1.40 | 0.82 | PARTIAL. Weak pass content to begin with: mostly flat lines (~500 and ~650 Hz, 0-30 s) and faint bending lines at 65-80 s. Cleaned: the ~650 Hz line and the late faint lines are removed; the ~500 Hz line (5-25 s) is still visible. Heavy collateral: dense corridors across 40-500 Hz for the whole clip. |
| h2_cub_lowpass | 0 → 1 | 1 → 0 (created 0) | 16.6 | 0.58 | 0.40 | N/A. No pass curve visible in the original. The detector still laid ~30 flat corridors (false removals, invisible as artefacts). |
| h2_kingair_b350 | 0 → 0 | 5 → 2 (created 0) | 5.8 | 1.02 | 0.75 | PASS (tones). The descending harmonic lines (~400->300, ~250->200, ~130 Hz, 7-16 s) are gone in the stock view; red-team profile flat at 0 dB, no ghost or notch. The diagonal ground-reflection fringe pattern across the whole band remains (not tonal, outside the tool's model, but it encodes pass geometry). |
| h2_pc12_lowpass | 0 → 0 | 4 → 4 (created 0) | ∞ | 0.00 | 0.00 | FAIL. One clear descending low tone (~150->100 Hz, 13-17 s) under the broadband swell; nothing detected, nothing removed. Unchanged. |
| h2_baron_flyby | 0 → 0 | 5 → 3 (created 0) | 2.5 | 1.17 | 0.89 | PARTIAL. Many descending harmonics removed (removed panel shows a fan 15-25 s), but curved descending stripes (harmonic remnants plus a V-shaped ground-reflection fringe) are still plainly visible at 15-28 s, 300-1600 Hz. |
| h2_c172_takeoff_talking | 0 → 0 | 18 → 9 (created 0) | 3.2 | 1.15 | 0.81 | PASS. The dense harmonic fan (20-30 s, 300-1800 Hz) and low lines are gone in the stock and red-team views (a)-(d); the curve profile shows only a small +1-2 dB bump. Talking after 37 s looks untouched in the images. |
| h2_a320_overhead | 0 → 0 | 7 → 7 (created 0) | ∞ | 0.00 | 0.00 | N/A. No Doppler curve visible: bursty broadband/voice-like content, steady lines, a 1 kHz burst at 9-10 s. Nothing removed. |
| h2_b737_overhead | 0 → 0 | 4 → 4 (created 0) | 38.8 | 0.17 | 0.02 | FAIL (not tonal). Descending curves converging at ~22 s (1000->100 Hz, 10-25 s) look like ground-reflection fringes, plus a ~1780 Hz tone at 0-5 s. Unchanged except a faint short ~880 Hz line and a low ~100 Hz track. |
| h2_citation_lowpass | 0 → 0 | 11 → 11 (created 0) | ∞ | 0.00 | 0.00 | N/A. Broadband bursts only; no tonal Doppler curve visible. Nothing removed. |
| h2_alphajet_flyby | 0 → 0 | 5 → 5 (created 0) | ∞ | 0.00 | 0.00 | FAIL. Clear descending tones at 0-7 s (~400->250 Hz and a lower one ~150->100 Hz) are unchanged; nothing detected. |
| h2_harrier_lowpass | 0 → 0 | 9 → 9 (created 0) | ∞ | 0.00 | 0.00 | N/A / weak FAIL. No clear pass curve; a faint low rising line near 100-150 Hz at 18-22 s remains. Nothing removed. |

Full per-clip table: [benchmark.md](benchmark.md); method comparison: [baselines.md](baselines.md); provenance: [MANIFEST.md](MANIFEST.md).
