# Why cleaning seems to create new attacker-visible ridges

**Question.** On the (burned) held-out set the box-free check counted 42 z>3 ridges that were
present only in dopplerscrub's output. The band-stop brick wall gave 24, the noise baseline 41 and
jitter 23. Are these (a) a measurement artefact, or (b) real artefacts created by the cleaning
(notch edges, noise-fill texture, VK ghosts beside corridors)?

**Answer: (a) for dopplerscrub.** On the 12 fixed-wing dev clips, **none** of dopplerscrub's
61 "new" z>3 ridges was created by the cleaning:

- 34 were already z>3 along the same curve in the original;
- 27 were original content that became a separate ridge once its neighbours were removed.

The metric is fixed (`scripts/boxfree.py`): new ridges are now split into **created / unmasked /
seen**. No remove.py change is warranted, since nothing was created. The real problem behind
these ridges is coverage: original content the detector did not target (see "What it means").

Everything here was measured on DEV clips only: prop_pass, prop_flyover, colsto_2050_2250,
colsto_2980_3170 and the 8 fixed-wing airshow clips. The research scope is fixed-wing aircraft,
so the helicopter airshow clips were excluded. Nothing in data/real/heldout* was used.

## Method

`scripts/new_ridges.py` runs dopplerscrub (`harden=True`, defaults at f0531a8) and the blanket
baselines (bandstop_brick, noise, jitter). For every ridge that the box-free extractor finds in the
cleaned view away from all original ridges, it measures the **same curve in both files**:

| measure | meaning |
|---|---|
| z_orig / z_clean | integrated on-curve z (`evaluate._curve_vs_floor`) in the original / cleaned raw spectrogram |
| excess dB, null sd dB | the two parts of z: on-curve excess over the ±3–6 Hz flanks, and the spread of the null (same curve shifted ±8–14 Hz) |
| d_on / d_flank | raw power change (clean − orig) on the curve / on its flanks |
| floor shift | change of the per-bin whitening baseline at the curve's bins |
| in-mask | share of the curve's frames inside the original's persistent (>4 dB, 2 s smoothed) mask |
| corridor distance | median distance to the nearest removed corridor (dopplerscrub's doppler tracks) |

Raw per-ridge numbers: `docs/results/new_ridges.json`. Per-clip counts with the fixed box-free
check: `docs/results/new_ridges_boxfree_dev.json`. Images (local, gitignored with the other result
images): `docs/results/img/nr_<clip>__<method>.png` (full clip; new z>3 ridges coloured by class)
and `docs/results/img/nr_zoom_<clip>.png` (original | cleaned | cleaned + corridors + new
ridges; produced by `scripts/diag/new_ridges_zoom.py`).

## Findings

### 1. Most "new" ridges were in the original already (re-fragmentation)

The extractor keeps **one path per connected component** of the persistent mask. Original
harmonic fans, broadband blobs and stationary tones crossed by a sweep form one large component,
which yields one ridge. Once part of the component is removed, the rest splits into several
components, and each yields its own ridge. These ridges are not near the single original path, so
the old metric called them "new". For 34 of the 61, the original already had z>3 along exactly
that curve.

### 2. The rest are "unmasked" original content, not new energy

The other 27 had z ≤ 3 along that curve in the original. Every one of them:

- lies ≥ 50% inside the original's persistent mask (median 0.97), and
- had its **on-curve power unchanged or lowered** (d_on ≤ +1 dB; typically 0 to −10 dB).

dopplerscrub's do-no-harm cap held throughout: no new ridge had more on-curve power than the
original. These curves were inside bright original structure. Their z was low in the original
because their flanks and null controls were bright too. After cleaning, the flanks (often a removed
corridor 3–9 Hz away) and the ±8–14 Hz null controls are quieter, so the same content stands out.
The zooms show what they are:

- **prop_pass 270–420 Hz, 20–43 s**: fragments of the dense harmonic fan that the detector only
  partly covered;
- **colsto_2050_2250**: stationary tones at ~127 and ~180 Hz, visible in the original, now
  separable because the descending sweep that crossed them was removed;
- **air_hercules_airshow**: a 128 Hz stub at 53–55 s where the target curve's corridor ended
  early, plus a meandering path through the broadband blob at 185–210 Hz;
- **air_extra300 360–420 Hz** and **air_spitfire_mustang ~390 Hz**: the partially removed
  remainder of the original bright arch or blob.

### 3. No notch edges, fill texture or VK ghosts

There are no parallel lines along corridor edges and no textured fill in any zoom. Corridors are
seamless at this scale. Removal is power-complementary, the fill is level-matched, and do-no-harm
is enforced per block. The numbers agree: d_on > +1 dB never occurs for dopplerscrub.

### 4. "Stronger" ridges: a z-normalisation effect

f7 reported original ridges getting much stronger: air_p51_mustang 343 Hz z 11.6 → 97, 112 Hz
19 → 65, colsto_2980_3170 137–146 Hz 7 → 17.8. On the same curves:

| ridge | z orig → clean | excess over flanks dB | null sd dB | corridor distance |
|---|---|---|---|---|
| p51 343 Hz, 64–74 s | 11.6 → 97.0 | 14.9 → 15.0 | +4.3 → −4.9 | 16 Hz |
| p51 112 Hz, 16–23 s | 19.2 → 65.5 | 14.9 → 15.0 | +2.0 → −3.1 | 20 Hz |
| colsto_2980 138–146 Hz | 7.0 → 17.8 | 6.7 → 4.9 | −1.8 → −7.6 | 16 Hz |

The ridge's contrast did not change, and its on-curve power did not change either (d_on −0.0 to
−1.3 dB). What changed is the null spread: in the original, the curve shifted ±8–14 Hz landed on
other bright harmonics, and those were then removed. z = excess / null sd, so z explodes while the
ridge looks identical.

These ridges are **not inside or next to filled corridors**: the nearest corridor is 16–20 Hz away.
They are bright original harmonics the detector did not target, so this is a detector-coverage
issue, not a fill-level one. The per-bin whitening floor at their bins moved by only −0.2 to
−2 dB. Measured by contrast change (> 3 dB) instead of z, dopplerscrub has 3 stronger ridges on
dev, against 26 by the old z ratio.

### 5. The blanket baselines do create ridges

| method (12 dev clips) | ridges z>3 before | still z>3 | new z>3 | created | unmasked | seen |
|---|---|---|---|---|---|---|
| dopplerscrub | 202 | 117 | 61 | **0** | 27 | 34 |
| bandstop_brick | 202 | 8 | 18 | **8** | 9 | 1 |
| noise | 202 | 153 | 48 | **7** | 3 | 38 |
| jitter | 202 | 158 | 51 | **13** | 4 | 34 |

- **Noise and jitter** created ridges by adding energy along the curve (d_on +1 to +4 dB). Jitter
  smears tones into neighbouring bins; the noise baseline's fill has its own structure.
- **Brick wall**: its 8 created ridges sit in the zeroed band, where on-curve power fell 30–80 dB.
  The per-bin whitener rescales a near-silent band (16-bit quantisation floor) back up to 0 dB, so
  faint leftover structure crosses the 4 dB threshold. That is a whitening artefact of an empty
  band, and it shows how unreliable "new" counts are for blanket methods too.

## Metric fix (what changed in `scripts/boxfree.py`)

- `n_new_z3` is unchanged (all cleaned-only z>3 ridges). It is now split by measuring each one on
  the original:
  - `n_new_seen_z3`: z>3 along that curve in the original.
  - `n_new_unmasked_z3`: ≥50% inside the original's persistent mask and on-curve power not raised
    by more than 1 dB.
  - `n_new_created_z3`: everything else. **Only this one is an artefact of the processing.**
- `n_stronger` counts original ridges whose excess over their flanks rose by > 3 dB. z is not
  compared across files, because its null spread changes with the surroundings.
- Per-ridge `excess_db_orig/clean` and a `new_ridges` list with each ridge's class are stored.
- benchmark.md shows "(new; created c, unmasked u, seen s)", and its fixed-wing headline has a
  "created" column. baselines.md shows "new (c created)". Rows computed before this change show
  "split n/a" until box-free is re-run (`benchmark.py --boxfree-from <dir of cleaned WAVs>`,
  `baselines.py` re-run).
- `tests/test_boxfree.py`: an added sweep counts as created; exact removal of one tone creates
  nothing and keeps the other tone "still"; identity has 0 new and 0 stronger.

Why still count unmasked ridges at all? They **are** attacker-visible after cleaning. still + new
remains the honest count of z>3 ridges left; only the attribution changes. A cleaned file is not
more revealing than the original because of them, but it is not clean either.

## What it means for the next iteration

- No remove.py change. Do-no-harm, the power-complementary fill and the VK subtraction create no
  ridges on dev. Changing fill shaping to chase the "new" count would be tuning to a metric
  artefact.
- The visible leftovers (117 still + 27 unmasked + 34 seen ridges on dev) are **coverage**:
  - harmonics of a detected source that were not removed (prop_pass fan, p51 112/343 Hz);
  - corridors that end before the curve does (Hercules 128 Hz stub);
  - stationary tones (colsto).

  The first two belong to the detector (f7: harmonic completion around detected fundamentals,
  corridor span extension). Stationary tones are not Doppler evidence and should stay.
- For the final heldout2 evaluation, quote **created** as the artefact number and **still + new**
  as the residual-visibility number.
