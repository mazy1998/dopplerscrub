# Detector v2: steep cores, source families, fewer false corridors, speech gate

Branch `research/detect`. Includes `research/detection-v2`: fc's width-aware remover
and skirt step. Tuned on **dev only**: synth, `data/real/*.wav`, airshow and the
colsto ranges. `data/real/heldout` and `heldout2` were never run, rendered or read
for this work. Commits, in order:

| commit | change |
|---|---|
| 5622387 | Fast-chirp tracker for steep pass cores. |
| a68e5bb | `Track.width_hz`. |
| b478666 | Source families; no unobserved-harmonic corridors. |
| 7f12bcf | Chirp continuation, including the p51 pre-CPA hook. |
| ad54350 | Speed-ups. |
| d9d08f1 | Excursion gate on families; chirp paths are judged piece by piece. |
| 886ea65 | Speech co-modulation gate. |
| 49d00e4 | Robust null Gumbel fit; hook for `jets.detect_jet_tones`. |

## What changed

1. **Fast-chirp tracker** (`chirps.py`).
   - **Why.** Steep pass cores (50–200 Hz/s) smear across every window the
     0.25 s-frame detectors use. In the attacker's 2 s view they still show as a
     bright diagonal.
   - **View.** 1/16 s hop and four short Hann windows (0.0625–0.5 s). Each window
     gives frequency-local contrast against a block-median background, and the
     view takes the per-cell maximum across windows.
   - **Oriented line integration.** The mean along ±0.5 s straight segments, over
     35 slopes up to ±220 Hz/s, plus the best slope per cell.
   - **Tracking.** Greedy and slope-predicted: the path slope and the cell's own
     slope must agree, with at most 2 coasting frames.
   - **Acceptance** requires all of:
     - ≥ 3 s long, with a continuous ≥ 2 s run steeper than 12 Hz/s;
     - a sweep of ≥ 25 Hz (or 3 % of f);
     - few turns (none under 6 s);
     - median contrast ≥ 6 dB;
     - integrated excess ≥ 5 dB and z ≥ 5 against copies shifted by ±24–48 Hz.

     A path that fails as a whole is split at frame-to-frame jumps and each piece
     is judged alone.
   - **Continuation.** Accepted cores are continued along their own ridge. At each
     frame the walk tries the current, flat and reversed slope (that last one
     catches the pre-CPA hook). It stops when the mean of the last 8 on-ridge
     values falls below 7 dB; the fused view sits near 4 dB on noise.
   - **Refinement.** Accepted cores are **dechirp-refined** (heterodyne, baseband
     offset in 0.25 s windows, 3 iterations) and emitted on a 1/16 s grid. Even
     an exact 0.25 s grid cannot carry a 150 Hz/s curve into the remover's ±3 Hz
     band.
   - **Width.** Each chirp track carries **`width_hz`**: the 90 %-energy
     half-width of the dechirped tone, with the window's own spread removed in
     quadrature, clipped to 2–12 Hz. A pure synthetic sweep gives 2.0 Hz; p51's
     main core gives 6.8 Hz.
2. **Source families** (`families.py`).
   - **Linking.** Two lines of one source share the Doppler factor, so their
     log-frequency difference stays constant while they move. Moving doppler
     tracks are linked when that difference has residual sd ≤ max(0.12 %, 7 % of
     the motion) over ≥ 3 s of overlap, and the links are union-found into
     clusters.
   - **Orders.** A harmonic sieve assigns integer orders: the largest f0 that puts
     ≥ 85 % (else ≥ 50 %) of the weight on integer multiples.
   - **Offset families.** Misfits that share a fractional order k + ε form an
     offset (sideband / twin) family.
   - **Shared factor.** F(t) is the weighted median of member f/n, extended beyond
     the observed span by a straight-pass fit.
   - **Expansion gate.** A family is expanded only if F swings by ≥ 3 % (5–95 %).
     A generator's ~1 % governor drift also gives constant-ratio harmonics.
   - **Probing.** Every order k·F(t) is probed and gated by the existing
     fixed-curve local-evidence test (`_verified_curve_mask`, its background now
     precomputed once per view).
   - **Unobserved orders** need z ≥ 8 inside the *measured* span.
   - **Sub-lattices.** A finer lattice (f0/2, f0/3) needs ≥ 2 new verified orders.
   - **Output.** Measured tracks are kept and relabelled into one group with
     consistent orders. Probes add only uncovered spans.
3. **No unobserved-harmonic corridors from the parametric source path.** It used
   to emit every predicted harmonic up to `fmax` over the full span; prop_quiet got
   h1–h21 at 1–89 s against 3 true harmonics. An order now needs ≥ 4 s of measured
   support.
4. **Speech co-modulation gate.**
   - **Problem.** air_spitfire_presenter had ~15 voice partials (300–1400 Hz)
     labelled doppler. The trackers had bridged syllables into smooth 5–13 s
     tracks, so a shape test cannot catch them.
   - **Test.** A voice partial's level follows the 250–3400 Hz band envelope at
     syllable rate; an engine tone under speech does not. Both envelopes are
     high-passed (1 s trend) and compared in 4 s windows where the band envelope
     is syllable-modulated (sd ≥ 2 dB). Frames with r ≥ 0.5 are cut (± 0.5 s) and
     the remaining pieces are kept.
   - **Placement.** Applied before chirps/families, so voice cannot seed them, and
     again after, because a family probe had re-extended a voice partial over
     1.2–35.2 s.
5. **Robustness.** The long-file Gumbel fit of reversed-time null maxima could
   overflow; it now falls back to the local threshold. `_jet_tones` is a guarded
   hook for `jets.detect_jet_tones` (remove-doppler-e7's module).

## Before / after on dev (fixed-wing, current remover)

- **Before** = detector at f0531a8; **after** = 886ea65. Both run with the same
  merged remover (`pipeline.scrub_array`, 2 passes, harden).
- Box-free ridges: `scripts/boxfree.py` attacker ridges with z > 3, given as
  original → still z > 3 after cleaning (+ split-off / new).
- Detect times are summed over both passes, with 5–6 clips running in parallel.
- Out-of-scope clips are omitted: f1, f1_pass and the helicopters.

| clip | 40–450 Hz before | after | wide before | after | doppler tracks | detect s | SNR outside dB |
|---|---|---|---|---|---|---|---|
| prop_pass | 13→3 (+11) | 13→2 (+9) | – | – | 224 → 281 | 16 → 18 | 7 → 7 |
| prop_flyover | 4→0 (+3) | 4→0 (+4) | – | – | 66 → 68 | 2 → 3 | 11 → 7 |
| colsto_2050_2250 | 36→27 (+6) | 36→24 (+9) | – | – | 183 → 203 | 52 → 58 | 25 → 24 |
| colsto_2980_3170 | 24→14 (+2) | 24→14 (+0) | – | – | 150 → 145 | 40 → 47 | 20 → 25 |
| air_p51_mustang | 41→21 (+18) | 41→18 (+21) | 90→54 (+37) | 90→49 (+37) | 291 → 368 | 28 → 32 | 12 → 9 |
| air_extra300 | 27→13 (+10) | 27→10 (+8) | 59→43 (+15) | 59→39 (+14) | 142 → 172 | 13 → 18 | 13 → 11 |
| air_hercules_airshow | 26→17 (+3) | 26→17 (+4) | 83→70 (+13) | 83→69 (+15) | 67 → 77 | 11 → 14 | 21 → 20 |
| air_spitfire_presenter | 13→6 (+3) | 13→8 (+2) | 38→26 (+12) | 38→31 (+5) | 76 → 69 | 10 → 11 | 17 → 20 |
| air_spitfire_mustang | 7→5 (+5) | 7→6 (+4) | 45→34 (+13) | 45→34 (+12) | 42 → 35 | 3 → 4 | 17 → 16 |
| air_p38 | 0→0 | 0→0 | 0→0 | 0→0 | 11 → 12 | 1 → 1 | 19 → 16 |
| air_star_wars_canyon (jet) | 4→4 | 4→4 | 17→17 | 17→17 | 8 → 6 | 7 → 9 | 34 → 34 |
| air_rainbow_f18 (jet) | 7→7 | 7→7 | 36→36 | 36→36 | 0 → 0 | 1 → 1 | – |

In-scope real/airshow totals over both views: still-z>3 ridges 397 → 385, new
ridges 151 → 144.

**Presenter voice-only section** (0–19 s, before the aircraft):

| | speech SNR |
|---|---|
| before | 9.9 dB |
| after | **26.7 dB** |

| synth clip | AF z per visible curve (orig → clean), before | after | doppler tracks |
|---|---|---|---|
| prop_quiet | 46→-0.1, 22→-0.7, 29→1.2 | same | **22 → 6** (false h4–h21 corridors gone) |
| overlap | 33→0.3, 13→-2.6, 8→-1.1, 32→-0.8, 6→-1.1 | 33→0.5, 13→-1.2, 8→-3.9, 32→-0.3, 6→-1.5 | 42 → 31 |
| speech | 18→0.7, 10→0.3 | 18→0.5, 10→-0.7 | 3 → 4 |
| stereo | 6→-1.0 | 6→0.2 | 11 → 12 |

**False-alarm guards:** music, long_hum60, prop_faint, jet, f1 and helicopter each
have 0 doppler tracks before and after.

### By eye (attacker view, before vs after)

- **p51 (wide):** with the width-aware remover, the 1200→650 Hz steep core
  (25–31 s, ~150 Hz/s, ~±7 Hz wide) is gone, as is most of the 200–500 Hz fan at
  25–60 s. Still visible:
  - a short blob at ~1150 Hz around 25 s;
  - the end of the 700→1000 Hz rise at 52–55 s, missed by the chirp tracker;
  - the broadband fringe arcs at 55–60 s, which are not tones.
- **prop_flyover:** the bright rising 150→220 Hz curve at 8–9 s, plainly left
  before, is gone. SNR outside drops 11 → 7 dB because the wide chirp corridors
  also take some broadband swell under the 2–6 s fan. It is not visible as an
  artefact, but it is real collateral.
- **colsto_2980:** the 175→110 Hz pass is removed in both versions. Stepped engine
  tones are kept.
- **presenter:** the voice section is now untouched apart from one thin line at
  ~1400 Hz (12–20 s). The pass at 22–35 s is still removed.

## Hard synthetic suite (synth_hard, exact truth)

Visible truth curves left at z > 3 after cleaning, with false corridors on the
null clips. Scored with a copy of e7's scorer at 7dd7ab5, pointed at research/detect.
The "after" column is d9d08f1, before the speech gate. fe has the re-score on
49d00e4 with the updated scorer (`--tag f7`, usable-pass metric).

| clip | before (f0531a8) | after (d9d08f1) |
|---|---|---|
| dense_fan_a | 12/20 | **3/20** |
| dense_fan_b | 5/18 | **1/18** |
| twin_props_a | 7/23 | 5/23 (2 created ridges) |
| twin_props_b | 7/27 | **2/27** |
| maneuver_turn | 3/4 | **1/4** |
| maneuver_parabola | 0/4 (PASS) | 0/4 (PASS) |
| wide_fm_tone | 0/6 (PASS) | 0/6 (PASS) |
| wide_fm_airshow | 5/7 | 5/7 (3 created ridges) |
| speech_prop | 3/4 | 3/4 |
| wind_traffic_prop, faint_long_a/b | all left | all left |
| jet_a, jet_b | 3/3 | 3/3 |
| null_generator | 57 false corridors | 57 (it was 283 before the excursion gate) |
| null_music | 38 | 40 |
| null_birds | 17 | 17 |
| null_traffic, null_ambience | 0 | 0 |

## Runtime and scale

- **Single pass, sequential**, on prop_pass, p51, extra300, colsto_2980, music and
  overlap: 62.3 s at f0531a8 vs 80.3 s at ad54350 (**+29 %**). The chirp tracker
  adds ~1.5–2.5 s per 75 s clip; the families add about the same on dense clips.
- **colsto full hour, single pass** (ad54350, before the gates):
  - detect time 798 → 916 s (+15 %);
  - doppler tracks 2132 → 2739;
  - doppler track-seconds 41.1k → 45.5k (+11 %).
- **Deterministic:** byte-identical output across runs and across processes
  (`PYTHONHASHSEED` 1 vs 2).

## Checked and not changed

- **fmax 2500 (dev evidence only).** Hercules' 52–58 s tone is already covered
  below 2 kHz, in pieces. fmax 2500 added 6 tracks above 2 kHz, none of them that
  tone, and Hercules' detect time went 11.3 → 20.1 s. music and prop_faint stayed
  clean. The attacker's wide view stops at 1950 Hz. **Kept fmax = 2000.**
- **Toneless jets** (star_wars_canyon, rainbow_f18): no resolvable tones up to
  12 kHz. Broadband swell plus ground-reflection fringes are out of reach for a
  tonal remover. Jet tones are e7's (`jets.py`, hooked).
- **Template bank** (log-frequency FFT2 per (v, d) comb, as the paper's
  attacker). It was prototyped, and the index mapping was verified exact on a
  synthetic image. It was not wired in:
  - On real evidence it gave weak and sub-harmonic proposals (prop_pass z 6.5 vs
    null 4.3, at the wrong f0; music/hum nulls ≈ 10).
  - A 6-tooth comb cannot represent the ≈ 33 Hz fan spacing below the log axis.
  - The remaining dev leftovers are unremoved harmonics of already-detected
    sources, which families target directly.
- **Path linking for chirps** (to recover p51's 52–55 s rise). Reverted: joining
  fragments let voice glides through (4 false chirps on synth speech).

## Honest failures and open items

- **Box-free counts improve only modestly** (397 → 385). On p51 the count swings
  by ±5–10 with small changes, because many ridges sit near z = 3. Visual
  inspection and the hard suite show larger gains than the counts.
- **More removal outside the ±3 Hz corridors** where chirp tracks are wide (p51
  12 → 9 dB, prop_flyover 11 → 7 dB SNR outside). It is not visible as an
  artefact.
- **Missed:**
  - p51's 52–55 s end of the 700→1000 Hz rise;
  - the faint passes (faint_long_a/b);
  - prop under heavy wind/traffic;
  - pass harmonics under speech (speech_prop 3/4);
  - the jets.
- **Created ridges** on twin_props_a (2) and wide_fm_airshow (3) need a look:
  they are either unmasked leftovers or remover artefacts.
- **Not done yet** (fe's queue):
  - chirp slope range beyond ±220 Hz/s (dense-fan cores up to ~450 Hz/s);
  - shaft-rate sidebands at k ± 1/3 and k ± 1/2;
  - probing the harmonic family of a wide chirp core with its width;
  - checking the speech gate against pass tones under speech.
