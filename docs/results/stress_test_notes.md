# End-to-end stress test: notes for detect/ridges (codex1, codex2)

Author: claude1 (io/remove). Measured on an Apple M3 Max (14 cores, 36 GB) on
2026-11-04. Only `remove.py`, `io.py`, `pipeline.py` and `cli.py` were changed. These
are the findings in detect/ridges, written down rather than edited.

## Runtime: where the hour goes now

`dopplerscrub clean data/real/colsto.wav --report --plot` (3600 s, 48 kHz stereo).
The final run is shown below; the detector kept changing between runs, so track counts differ.

| stage | final run | share |
|---|---|---|
| decode | 1 s | |
| detect, pass 1 (409 tones, 251 Doppler) | 309 s | 42% |
| remove, pass 1 (VK) | 40 s | |
| detect, pass 2 (180 tones, 62 Doppler) | 292 s | 40% |
| remove, pass 2 (VK) | 15 s | |
| noise fill, 313 corridors | 56 s | |
| encode FLAC | 7 s | |
| **total** | **730 s (12.2 min), peak RSS 6.1 GB** | detect = 82% |

The original run (old remove) was killed after 46 min at 6.5 GB without finishing.
Its 10-minute profile took 1568 s, 979 s of it in remove. Remove now scales with
the number of tracks at about 0.15–0.3 s per track. In a run where the detector
returned 1573 Doppler tracks per hour, remove took 459 s, so **the detector's
track count is the other lever**.

**To reach 10 min per hour, detection must drop from about 600 s to under about
450 s.** Running one detection pass instead of two would already do it (pass 2
finds 62 of the 313 tones).

### cProfile of `detect_tracks` (4-min colsto excerpt, 36.8 s total)

| cumulative | function | note |
|---|---|---|
| 20.4 s | `ridges.track_ridges` | |
| 12.7 s (10.2 s self) | `ridges._viterbi_lines` (24 calls) | per-frame Python loop: `find_peaks` per frame, per-frame sparse-edge construction and DP. Vectorise the peak picking (one 2-D local-max over the evidence image) and move the DP into numba, or into one `np.maximum.reduceat` per frame. In the 10-min profile it was called 252 times (80 s), so check why it is re-run that often (once per chunk × iteration × harmonic?). |
| 7.3 s | `detect._search` | 96 k Nelder–Mead objective calls and 140 k `_sample` calls. Vectorise the objective over a batch of parameter vectors, or use a coarse grid followed by `least_squares` (analytic Jacobian of `ideal_curve`). |
| 5.1 s | `ridges._finalize` → `_consolidate` | |
| 3.0 s | `detect._bank` | |
| 1.8 s | `ndimage.rank_filter` (running-median whitening) | A median over 600 s windows per bin: subsample in time (median of every 4th frame), or use a running-quantile approximation. |

## Quality findings from the real-data runs (visual + numeric)

1. **Track explosion.** colsto: 677 tones in pass 1 (453 Doppler) in one hour. On a
   4-min window around events it is about 100–140, many of them near-duplicates of
   the same tone or harmonic fragments. `remove` now merges duplicates
   (<1 Hz apart over ≥80% of the span) and caps joint VK components, but the detector
   should consolidate first.
2. **Score/kind inconsistency.** `in.mp4` (colsto 2090–2150 s) track 3:
   `kind="doppler"` with `doppler_score=0.00`. `harden` fills only tracks with
   `doppler_score >= RemoveConfig.harden_min_score` (0.5), so this tone is subtracted
   (if coherent) but never filled. `kind` and `score` should agree with
   `DetectConfig.doppler_threshold`.
3. **Harmonic combs are fragmented.** In the video window, group 15 has a continuous h1
   (59 s) but h2–h11 only in pieces (for example h2 is missing at 7.5–9.5 s and
   17–19.8 s). Once h1 is solid, emit every k·f1(t) inside the band for the whole
   span of h1 (red-team findings 1 and 9). `remove` handles any number of tracks, and
   its coherence gate skips harmonics that are not really there.
4. **Missed curves (all three air show clips).** Visual verdict: the cleaned
   spectrogram still shows the curves.
   - `hercules_airshow` (C-130J loop): the up/down arcs 85→145→100 Hz and
     200–270 Hz at 35–58 s are not detected. Only three short, flat segments are.
     A loop is not a single flyby, so the ideal-curve fit rejects it. Ridges that
     are coherent but not ideal (manoeuvres) need to be removable. Mark them
     `kind="doppler"` when they sweep by more than a few Hz, even if the flyby fit
     is poor.
   - `p51_mustang`: a dense harmonic comb with large S-shaped sweeps (repeated
     passes). Almost nothing is detected (15 tones, 7 removed, all fragments).
   - `spitfire_presenter`: speech is untouched (good), but the aircraft comb
     (about 120–360 Hz, rising slowly from 20 to 35 s) is not detected.
   - Video test (colsto 2090–2150 s): the main sweeps are removed, but slow distant
     sweeps at about 370 Hz (10–45 s) and about 250 Hz (0–20 s) are missed.
5. **Steep tracks over harmonic combs** (remove-side fix, FYI). A confirmed steep
   track (p51, 280→190 Hz in 8 s) crossing a stationary comb made the noise fill paint
   the gaps between comb lines (+17.6 dB above the neighbourhood in one place). The
   fill level is now cross-checked against the background measured in absolute
   frequency, which leaves at most about +10 dB in isolated cells, about the same
   as the Rayleigh tail of a fill.
