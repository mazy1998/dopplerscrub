# Visual verdicts, heldout2 (written blind from the images, before phase 2; 2026-11-04, coordinator session)

Copy this file to `visual_verdicts.md` and replace every TODO with your verdict, written from
the images only (PASS / PARTIAL / FAIL / N/A + what you see). Keep the `## <clip>` headings.

## h2_c150_lowpass

- images: [h2_c150_lowpass_ba.png](img/h2_c150_lowpass_ba.png), [h2_c150_lowpass_redteam.png](img/h2_c150_lowpass_redteam.png), [bl_h2_c150_lowpass.png](img/bl_h2_c150_lowpass.png)
- verdict: PARTIAL. Weak pass content to begin with: mostly flat lines (~500 and ~650 Hz, 0-30 s) and faint bending lines at 65-80 s. Cleaned: the ~650 Hz line and the late faint lines are removed; the ~500 Hz line (5-25 s) is still visible. Heavy collateral: dense corridors across 40-500 Hz for the whole clip.

## h2_cub_lowpass

- images: [h2_cub_lowpass_ba.png](img/h2_cub_lowpass_ba.png), [h2_cub_lowpass_redteam.png](img/h2_cub_lowpass_redteam.png), [bl_h2_cub_lowpass.png](img/bl_h2_cub_lowpass.png)
- verdict: N/A. No pass curve visible in the original. The detector still laid ~30 flat corridors (false removals, invisible as artefacts).

## h2_kingair_b350

- images: [h2_kingair_b350_ba.png](img/h2_kingair_b350_ba.png), [h2_kingair_b350_redteam.png](img/h2_kingair_b350_redteam.png), [bl_h2_kingair_b350.png](img/bl_h2_kingair_b350.png)
- verdict: PASS (tones). The descending harmonic lines (~400->300, ~250->200, ~130 Hz, 7-16 s) are gone in the stock view; red-team profile flat at 0 dB, no ghost or notch. The diagonal ground-reflection fringe pattern across the whole band remains (not tonal, outside the tool's model, but it encodes pass geometry).

## h2_pc12_lowpass

- images: [h2_pc12_lowpass_ba.png](img/h2_pc12_lowpass_ba.png), [h2_pc12_lowpass_redteam.png](img/h2_pc12_lowpass_redteam.png), [bl_h2_pc12_lowpass.png](img/bl_h2_pc12_lowpass.png)
- verdict: FAIL. One clear descending low tone (~150->100 Hz, 13-17 s) under the broadband swell; nothing detected, nothing removed. Unchanged.

## h2_baron_flyby

- images: [h2_baron_flyby_ba.png](img/h2_baron_flyby_ba.png), [h2_baron_flyby_redteam.png](img/h2_baron_flyby_redteam.png), [bl_h2_baron_flyby.png](img/bl_h2_baron_flyby.png)
- verdict: PARTIAL. Many descending harmonics removed (removed panel shows a fan 15-25 s), but curved descending stripes (harmonic remnants plus a V-shaped ground-reflection fringe) are still plainly visible at 15-28 s, 300-1600 Hz.

## h2_c172_takeoff_talking

- images: [h2_c172_takeoff_talking_ba.png](img/h2_c172_takeoff_talking_ba.png), [h2_c172_takeoff_talking_redteam.png](img/h2_c172_takeoff_talking_redteam.png), [bl_h2_c172_takeoff_talking.png](img/bl_h2_c172_takeoff_talking.png)
- verdict: PASS. The dense harmonic fan (20-30 s, 300-1800 Hz) and low lines are gone in the stock and red-team views (a)-(d); the curve profile shows only a small +1-2 dB bump. Talking after 37 s looks untouched in the images.

## h2_a320_overhead

- images: [h2_a320_overhead_ba.png](img/h2_a320_overhead_ba.png), [h2_a320_overhead_redteam.png](img/h2_a320_overhead_redteam.png), [bl_h2_a320_overhead.png](img/bl_h2_a320_overhead.png)
- verdict: N/A. No Doppler curve visible: bursty broadband/voice-like content, steady lines, a 1 kHz burst at 9-10 s. Nothing removed.

## h2_b737_overhead

- images: [h2_b737_overhead_ba.png](img/h2_b737_overhead_ba.png), [h2_b737_overhead_redteam.png](img/h2_b737_overhead_redteam.png), [bl_h2_b737_overhead.png](img/bl_h2_b737_overhead.png)
- verdict: FAIL (not tonal). Descending curves converging at ~22 s (1000->100 Hz, 10-25 s) look like ground-reflection fringes, plus a ~1780 Hz tone at 0-5 s. Unchanged except a faint short ~880 Hz line and a low ~100 Hz track.

## h2_citation_lowpass

- images: [h2_citation_lowpass_ba.png](img/h2_citation_lowpass_ba.png), [h2_citation_lowpass_redteam.png](img/h2_citation_lowpass_redteam.png), [bl_h2_citation_lowpass.png](img/bl_h2_citation_lowpass.png)
- verdict: N/A. Broadband bursts only; no tonal Doppler curve visible. Nothing removed.

## h2_alphajet_flyby

- images: [h2_alphajet_flyby_ba.png](img/h2_alphajet_flyby_ba.png), [h2_alphajet_flyby_redteam.png](img/h2_alphajet_flyby_redteam.png), [bl_h2_alphajet_flyby.png](img/bl_h2_alphajet_flyby.png)
- verdict: FAIL. Clear descending tones at 0-7 s (~400->250 Hz and a lower one ~150->100 Hz) are unchanged; nothing detected.

## h2_harrier_lowpass

- images: [h2_harrier_lowpass_ba.png](img/h2_harrier_lowpass_ba.png), [h2_harrier_lowpass_redteam.png](img/h2_harrier_lowpass_redteam.png), [bl_h2_harrier_lowpass.png](img/bl_h2_harrier_lowpass.png)
- verdict: N/A / weak FAIL. No clear pass curve; a faint low rising line near 100-150 Hz at 18-22 s remains. Nothing removed.
