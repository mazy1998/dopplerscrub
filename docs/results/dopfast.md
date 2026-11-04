# dopfast attacker (matched-filter evidence volume)

## Headline: usable pass events per recording

The goal is to stop geolocation, not to make curves invisible. Per the paper (§7, §12.1, App. C):
- with an **unknown recording clock**, the attacker needs ≥ 2 usable pass events to intersect;
- with a **known or guessable clock** (video metadata, upload time), ONE usable event localizes
  to about two mirror points.

So `evaluate.dopfast` reports, before → after cleaning:

- **events**: grouped peaks of max_{v,d} E[v, d, s] over time with E ≥ τ = 12. Peaks closer than
  5 s are one event (non-maximum suppression; the paper's §6.3 grouping).
- **usable**: the event's (v, d) profile (max of E within ±1 s) has a near-peak region
  (E ≥ 0.75 · peak) spanning ≤ 2 d-grid steps (d known to ≤ ×4), and that region does not reach
  the largest d. A plateau-only leftover matches every distance about equally: it is detected
  but not usable.
- **pass criteria**: unknown clock = ≤ 1 usable event left; known clock = 0 usable left.

### Calibration (`scripts/dopfast_calibrate.py` → `dopfast_calibration.json`; frozen)

Only null clips and synthetic passes were used, never heldout*:

| material | what | result at τ = 12 |
|---|---|---|
| null: 5 synth_hard no-aircraft clips (music, birds, road traffic, generator with RPM drift, ambience) + the 12 fixed-wing dev clips time-reversed and frame-shuffled ×2 (41 spectrograms, 48.9 min) | false events | 14 (0.29/min), 12 of them "usable". Most come from synth road traffic (real Doppler sources at ~45 Hz) and bird chirps; reversed/shuffled dev give ~0.14/min |
| synth_hard aircraft, 32 straight passes with exact (v, d, h) truth (incl. the multi-pass clips) | detected within ±3 s of h | 8/32 detected, 5 usable |
| full synthetic straight passes (6 shapes, d 80–2560 m) | usable | 4/6 |
| the same passes with the steep core (\|w\| < 2d/v) faded out | usable | 0/6 |

**This attacker is weak before any cleaning.** It finds 8 of 32 synthetic straight passes; on the
12 real fixed-wing dev clips it finds 2 usable events in the originals, and on the 10-clip synth
set 3 (`dopfast_dev.md`). The pass criteria are therefore reported only where they apply:
- unknown clock: the original had ≥ 2 usable events;
- known clock: the original had ≥ 1.

Otherwise they are n/a, not a pass. Read n_usable before → after together with the
detection-rate caveat. A stronger attacker (lower τ = 9: 16/32 detected but 0.55 false
events/min; or longer recordings) would make the criteria bite more often.

τ trades false alarms against detection. At τ = 9 nulls give 0.55/min with the same synth_hard
detection; at τ = 14, 0.2/min and 7/14 detected. The front end was changed during calibration,
for null robustness only:
- each frame's median across the band is subtracted after per-bin whitening, because pink noise
  plus wind gusts had produced E = 18;
- b(w) has a minimum σ of 2 s, because an aircraft tone persists for seconds, unlike a bird chirp.

**Caveats.** The false-alarm rate (~0.3/min) makes "0 usable left" noisy on long recordings.
Usability is judged on a coarse 7 × 7 (v, d) grid, and d estimates on the synth_hard passes
were often off by ×2–4. Jets are largely outside the 40–1000 Hz band used here.

`evaluate.dopfast` / `evaluate.dopfast_file` implement the faster attacker of the Doppler
geolocation paper (§3.2, §4, Listing 3) as an evaluation tool. It does not look for visible
ridges. It matched-filters the whole pass:

1. **Spectrogram**: whitened log-frequency spectrogram S(s, ℓ).
   - 4 kHz mono, 0.512 s Hann STFT, 0.125 s hop.
   - Per-bin median whitening; S = positive excess dB clipped at 20 dB.
   - ℓ = log f on a 0.005 grid, 40–1000 Hz.
2. **Templates** T_{v,d}(w, ℓ) = b(w) Σ_j (1/j) K(ℓ − log j − log g(w; v, d)), j = 1..4:
   - g is the exact straight-pass received-time curve (eqs. 6 and 8, c = 343 m/s, g(0) = 1);
   - K is a Gaussian ridge, σ = 0.012 in log units;
   - b(w) is a Gaussian temporal weight, σ_w = d/v (see "Choices");
   - each template is zero-mean over its box and unit-norm.
   - Bank: v ∈ {30, 45, 70, 100, 140, 200, 280} m/s × d ∈ {40, …, 2560} m (7 × 7).
3. **Correlation**: L_{v,d}(s, ℓ0) = IFFT2[FFT2(S) · conj FFT2(T)] (eq. 13), with lag and anchor
   bookkeeping (eqs. 14–16).
   - A placement is valid only if the fundamental's whole curve is in band and the steep core
     (±σ_w) lies inside the file.
   - The score is the max over valid ℓ0 (eq. 17).
4. **Evidence** E[v, d, s] = (R − median_s R) / max(ε, 1.4826 MAD_s R) per shape (eq. 18). Its peak
   is the event the attacker would geolocate from: (v, d, s, f0).
5. **Null calibration** (ours; the paper notes robust standardization is not a false-alarm
   calibration):
   - the same peak statistic on the file's time-reversed spectrogram (falling templates cannot
     match rising curves) and on 6 frame-shuffled copies (per-frame spectra kept, curves
     destroyed);
   - z = (peak − null mean) / null sd.
   - **z at orig event** is the cleaned file's E at the original's peak shape and time (±1 s),
     scored on the cleaned file's null of peaks. That is conservative: a single value against
     maxima.

Reported in `benchmark.md` (column and fixed-wing headline tally), in `baselines.md` (summary
and per-clip columns, plus "visible before" now also counting dopfast z > 3), and per dev clip
by `scripts/dopfast_dev.py` (→ `dopfast_dev.md/json`).

## Validation (tests/test_evaluate.py)

- Events: a faint straight pass gives 1 usable event before and 0 after removal (both pass criteria met); a plateau-only leftover is not usable.

- g(0) = 1 and g falls monotonically through the pass.
- A faint 3-harmonic straight pass in white noise (60 s; amplitude 0.004/j against noise σ 0.01):
  z = 43. The exact shape and event are recovered (v 70, d 160, s 30.0, f0 110.4 for the true
  70 / 160 / 30 / 110). After exact removal: z_clean 0.4, z at orig event −5.
- Noise only: z ≈ 1. A time-reversed (rising) pass is not flagged.

## Choices made on dev clips, and a limitation

The paper fixes the template family but not the front end, ridge width, temporal support or
normalization. On real fixed-wing dev clips the plain version (0.256 s window, flat support up
to ±30 s, d ≤ 1280 m) was weak and unstable:

- colsto's obvious slow 165 → 105 Hz sweep was never the peak;
- prop_pass z ≈ 2;
- p51 "z 21" was an edge artifact (peak at s = 0, ±60 s template mostly outside the file).

Long harmonic plateaus and loud broadband CPA frames match the time-reversed and frame-shuffled
nulls about as well as the pass does. The chosen settings — 0.512 s window, b(w) concentrated
on the steep CPA part (where the paper's Fisher appendix puts the shape information), bank to
2560 m, core-inside-file validity — gave the strongest attacker among the variants tried:

| variant (z orig) | prop_pass | p51 | hercules | colsto_2050 | spitfire_mustang | extra300 |
|---|---|---|---|---|---|---|
| 0.256 s, flat support ≤ 30 s, d ≤ 1280 | 2.0 | 1.8 | 2.0 | −0.3 | 4.6 | 1.5 |
| 0.256 s, b(w) σ = 1.5 d/v | 1.6 | 0.6 | 1.1 | 0.6 | −0.2 | 10.3 |
| **0.512 s, b(w) σ = d/v (default)** | **3.1** | **4.0** | 2.8 | −0.1 | −1.0 | 3.6 |
| nonnegative unit-weight templates | −1.2 … 2.8 on all | | | | | |

Even so, this attacker is **less sensitive than the box-free ridge check on these real clips**,
and its z is sensitive to front-end choices. The frame-shuffle null has a heavy tail (loud CPA
frames), and the airshow passes are manoeuvres, not straight constant-speed passes. Treat
dopfast z as a complementary, shape-specific attacker:

- a high z after cleaning is a strong red flag;
- a low z does not prove a pass is hidden.

Tuning the attacker on dev makes the evaluation stricter, not laxer. heldout2 was not touched.

Possible strengthening (not done): a block-shuffle null that keeps loud stretches together; a
denser (v, d) grid with acceleration; data-driven harmonic weights.
