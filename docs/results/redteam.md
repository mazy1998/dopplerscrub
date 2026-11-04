# Red-team report: can a determined attacker still see the curves?

Owner: claude2 (evaluation). Tools: `dopplerscrub.evaluate` and `scripts/benchmark.py`.
We never run dopplertrack's fit/search. Instead, every cleaned file is inspected by eye
in the attacker's own whitened spectrogram. I then push harder than the stock
attack does, and back each look with a cheap numeric check.

## How I attacked the cleaned audio

| Look | What it catches | Function |
|---|---|---|
| Stock attacker view (4 kHz mono, 8192-pt, 0.25 s hop, 600 s running-median whitening, 40–450 Hz) | Anything dopplertrack sees | `attack_spectrogram`, `render_before_after` |
| Wide band 40–1500/1950 Hz | Vehicle, jet and warbird harmonics above 450 Hz | `attack_spectrogram_wide` |
| (a) low floor −2..8 dB | Faint ghosts hidden by the stock colour scale | `render_attacker_views` |
| (b) diverging map around 0 dB | Ghosts (red) **and** curve-shaped notches (blue) | 〃 |
| (c) global-median whitening | Artefacts that the 600 s running median hides or creates | 〃 |
| (d) 2 s time-smoothing, ±2 dB | Ghosts below single-frame noise (√8 noise reduction) | 〃 |
| (f) curve-aligned profile | Mean dB against offset from the curve over the **whole pass**: the most sensitive look | `curve_profile` |
| Anti-forensics (ENF literature, docs/research.md) | Band-edge steps; corridor variance/kurtosis; leftover k·f(t) harmonics; integrated residual against the per-bin floor | `anti_forensics` |

Calibration (unit tests, `tests/test_evaluate.py`):
- Leaving 10 % of the tone in place (−20 dB) is invisible in a naïve full-length
  render. With nearest-neighbour downsampling the ghost rows simply drop out, so all
  renders now use antialiased resampling.
- That same −20 dB ghost is visible in views (a)–(d). The profile shows a +5 dB bump,
  and the integrated statistic gives z ≫ 3. **A −20 dB per-frame residual is not
  safe.**
- The z statistic is calibrated empirically. Its null distribution is the same
  statistic on curves shifted by ±8–14 Hz. The naïve SE/√n gave false "leftover
  harmonics" on pure noise because the frames overlap 8×.
- A brick-wall ±2 Hz notch reads as edge step < −3 dB and corridor power < ½ of the flanks.

## Findings

_Snapshot findings from intermediate runs (23:10–23:30) while detect/remove were
still changing. Re-checked on the final run below._

1. **Missed harmonics (detector).** In an early run only the fundamental was
   detected on `prop_quiet`, and the 2nd and 3rd harmonics stayed fully visible
   (prop_quiet_ba). The anti-forensic harmonic check (`leftover k·f`) flags exactly
   this. **Fix:** once a fundamental f(t) is found, remove *every* k·f(t) inside the
   band, including harmonics too faint to detect alone (the ENF lesson). Also run
   a profile test at each k·f to decide whether that harmonic needs subtracting.
2. **Wrong tracks inject new curves (remove, `subtract`).** On `overlap` (23:20
   snapshot) the cleaned audio was *worse* than the original. Spurious tracks (an
   extrapolated near-vertical fit at 10–15 s and flat lines) were "subtracted" and
   **added** loud new tones along wrong paths. Energy rose by +1.4 to +4.2 dB in the
   62–500 Hz octaves, and speech-band SNR fell to 5 dB.
   **Fix:** never synthesise from `Track.amp`. Estimate each segment's complex
   amplitude by least squares, i.e. project the data onto exp(j·φ(t)). Accept a
   segment only if the corridor energy *drops*, so subtraction can never add
   energy. Clip fitted curves to the detected span; never extrapolate the
   ideal-pass fit.
3. **Residual ghost on removed corridors.** Where removal ran on the right track,
   contrast fell from 6–9 dB to +0.6–1.7 dB (synth) or +4.8 dB (prop_pass, 85 Hz).
   That is still a ridge in views (b)/(d), not a notch. **Fix:** track-locked
   subtraction needs sub-bin frequency refinement, since a 0.1 Hz error leaves
   about −20 dB at 2 s coherence. Use short segments (≤1 s) with overlap-add and
   per-segment LS amplitude and phase. After subtraction, check the integrated
   residual against the −10 dB-below-floor target, and if it fails apply the
   noise-fill harden on that corridor only.
4. **Over-removal on dense real recordings.** On `prop_pass`, `prop_flyover`
   and `f1_pass` the detector returned 180–510 tracks. The integrated residual met
   the target, but at a severe cost: speech-band SNR of 0.3–5 dB and SNR outside the
   corridors of 7–13 dB. That means audible damage to everything else.
   **Fix:** cap and merge tracks (one group per source; harmonics share one
   fundamental). Remove only kind="doppler" groups. Budget removal energy per
   octave, e.g. refuse to remove more than 1 dB from any octave without flagging.
5. **Fast passes are invisible to this attacker anyway.** Synth `f1`, `jet`,
   `helicopter` (fundamental 20–26 Hz, below the 40 Hz band edge) and
   `long_hum60` show no visible curve even in the original stock view: z ≈ 0.4–2.7
   before cleaning. The 2 s STFT window smears a sweep that changes faster than
   about 0.5 Hz per frame. These clips are low risk for *this* attack, but a
   shorter-window attacker would see them.

## Final run (snapshot 2026-11-04 ~00:20)

Detector and ridges as of 00:20, remove.py as of 00:04, defaults, `harden=True`.
Per-clip verdicts are in `benchmark.md` and `verdicts.json`. Summary:

| Set | Verdict |
|---|---|
| Synthetic, curves visible before (prop_quiet, overlap) | **Gone** in stock and red-team views. overlap meets the −10 dB integrated target; prop_quiet only reaches −6.8 dB. |
| Synthetic, curve under speech or in stereo | **Faint ghost remains** (z 7.6 and 4.5 after cleaning). Detection is incomplete (0–1 of 3 truth tracks matched). |
| Synthetic, invisible to this attacker anyway (prop_faint, f1, jet, helicopter, long_hum60, music) | Nothing visible before or after, and no collateral damage. |
| colsto event windows | Main event-1 sweep and its harmonic **gone**; event 2 **partially** removed. In the full-hour run event 1 meets the −10 dB target (−12.5 dB), but stationary tones are removed too, and event 2 leaves a visible dark patch. |
| Real short clips (prop_pass, prop_flyover, f1_pass) | **Fail**: the main sweeping harmonic stacks are untouched. |
| Air show (10 clips) | **Fail on all 10**: the big curves (often 300–1900 Hz) are untouched. Removal, when any, sits below about 250 Hz. |

### New findings from the final run

6. **Band coverage gap (detector config).** Airshow and warbird curves live at
   300–1900 Hz, but `DetectConfig.fmax = 1000` and the visible removals stop
   below about 250 Hz. The stock attacker band is 40–450 Hz, yet even inside it
   the 250–450 Hz harmonic fans of p51, prop_pass and extra300 remain.
   **Fix:** once a group's fundamental is known, extend every group to all k·f(t)
   up to about 0.45·work_sr. Use work_sr = 4 kHz (fmax 1.9 kHz) when the input
   allows. Run the detector at least over 40–2000 Hz.
7. **Fill is statistically detectable (ENF check 3).** On real clips the corridor
   power variance is **−5 to −17 dB** below the flanking bins (prop_pass −14,
   hercules −15, extra300 −17, colsto event 2 −14.5). The noise fill is far
   smoother than natural noise: a Rayleigh-power check would flag every
   corridor. **Fix:** fill with noise whose per-bin power follows an exponential
   distribution, matching the *local* flank level per frame. Equivalently,
   resample STFT magnitudes from the flanking bins of the same frame
   (copy-paste noise substitution) instead of a smoothed-PSD fill.
8. **Edge step +1.3 to +2.9 dB (ENF check 2).** Every real-clip corridor sits
   about 1.5–3 dB *above* its immediate surroundings: the fill level is set too
   high, or there is residual tone leakage at the corridor edge. It isn't visible
   at the stock colour scale, but the AF edge statistic sees it consistently.
   **Fix:** calibrate the fill to the *median* flank power (not the mean, which
   tones inflate). Taper the fill gain over the outer 0.5 Hz of the corridor.
9. **Leftover harmonics (ENF check 1).** After cleaning, the integrated k·f(t)
   test flags 12–136 harmonic curves per real clip (p51: 136, prop_pass: 53,
   extra300: 32, colsto event 2: 28). This is exactly the attack that survives
   per-track removal. **Fix:** same as finding 1. Subtract the whole harmonic comb
   for each group, sharing one phase-locked f(t).
10. **Integrated target not met on real data.** No real clip reaches "residual ≤
    −10 dB below the per-bin floor" at 95% confidence. The best are colsto event 1
    (−7.9 dB), f1_pass (−4.5 dB) and prop_pass (−0.7 dB). An attacker who sums
    about 1000 frames along a pass recovers what remains. **Fix:** findings 3, 7
    and 9. Also make the pipeline's last step a verify loop: compute
    `evaluate.anti_forensics` on each removed group and re-process any group whose
    `upper95_db` is above −10 dB.
11. **Track extrapolation.** In colsto, detected tracks extend across the whole
    200 s window, far beyond the visible sweeps. Each corridor is then
    noise-filled for minutes, which risks erasing unrelated content crossing
    those frequencies. **Fix:** clip each track to its supported span (plus
    `pad_s`).

12. **Over-detection and removal of stationary tones (colsto, run finished 00:58).**
    The 1-hour file yields 4195 tracks (821 doppler) and takes 51 min to scrub. The
    event-1 window's removed-content panel shows many *flat* engine and machinery
    tones removed alongside the sweep, and its speech-band SNR is 9.4 dB.
    The event-1 Doppler curves are now gone, with an integrated upper bound of
    −12.5 dB, the first real window to meet the target. That came at the cost of
    removing non-Doppler content the contract says to keep.
    **Fix:** remove only groups with kind="doppler" and doppler_score ≥ threshold.
    Never remove a track with < 0.5 Hz total sweep. Keep the harden noise fill to
    the doppler groups' corridors only.
13. **Region-shaped hole (colsto event 2).** Views (b) and (d) show a broad blue
    patch (3020–3110 s, 100–200 Hz). Many overlapping corridors were filled below the
    surrounding level, so the curve is gone but its *removal region* is visible
    (profile about −4 dB at every offset). **Fix:** same as finding 7 (fill level
    from per-frame flank median, exponential statistics). Also merge overlapping
    corridors before filling, so overlapping regions are not filled twice.

### What I'd check next by eye
- Re-run `scripts/benchmark.py` after fixes 6, 7 and 9, then review
  `img/sheet_all*.png` first (original | cleaned for every clip), then the per-clip
  `_redteam.png` panels (b), (d) and (f).
- A shorter-window attacker (1024–2048 pt) would see the fast passes (f1, jet)
  that the stock 8192-pt attack misses. That is out of scope for this attack, but
  worth noting as the next-strongest adversary.
