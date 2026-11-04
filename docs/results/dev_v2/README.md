# Dev re-run at the detector + remover integration (37c5f34)

Branch `research/devrun`. Code under test: `research/detection-v2` at 37c5f34. That commit adds
the fast-chirp tracker (`chirps.py`, wired in as `detect._steep_cores`) and the per-track
`width_hz` + skirt handling in `remove.py`. Since freeze-1 (724e109) only `chirps.py`,
`detect.py` (+24 lines), `remove.py` and `types.py` have changed. **Dev split only.** Nothing in
`data/real/heldout*` was run, rendered or viewed. The full-hour `colsto` file was not re-run
(`--no-colsto`).

## Two references, and why

1. **The published table** (`docs/results/benchmark.md`, `verdicts.json`). **Caution: its real
   and airshow dev rows are stale.** They are identical in `benchmark.json` at 48c872c (01:48,
   before the freeze) and at f0531a8. Only the synth and held-out rows were re-run at freeze-1;
   `benchmark.py` keeps rows it did not re-run. The presenter's "speech SNR 35.2 dB" is from
   that pre-freeze code. The old real/airshow images in `docs/results/img/` are also pre-freeze.
2. **freeze-1 re-run today** on the 12 fixed-wing real/airshow dev clips. Code exported with
   `git archive freeze-1 src`, current `scripts/`, same machine, same metric code
   (`evaluate.py` is unchanged since freeze-1). Raw rows: `freeze1_rerun_benchmark.json`. This
   is the true "before" for this merge.

## How this was run

```
uv run python scripts/benchmark.py --split dev --no-colsto --blind --workdir data/out/dev_v2 --out docs/results/dev_v2
uv run python scripts/benchmark.py --split dev --boxfree-from data/out/dev_v2 --out docs/results/dev_v2
uv run python scripts/baselines.py --split dev --blind --out docs/results/dev_v2
```

`--out` is new in `benchmark.py` and `baselines.py`. It writes the JSON, markdown and images to
another results dir. Images (`dev_v2/img/`) are gitignored like the main `img/`.

**Visual verdicts were written blind.** Each was written from the before/after and red-team PNGs
as each clip finished, before any of this run's numbers were read (`verdicts.json` here; they are
also the verdict column of `benchmark.md`). One caveat: the old benchmark table had been read
beforehand as briefing.

## Verdicts

The middle column, *freeze-1 re-run*, comes from viewing the freeze-1 re-run images side by side
after the blind verdicts were written, and only for clips whose output changed.

| clip | published (stale for real/airshow) | freeze-1 re-run | 37c5f34 (blind) | due to this merge |
|---|---|---|---|---|
| prop_quiet, overlap, speech, stereo | PASS | = | PASS | none (bit-identical) |
| prop_faint, jet(_wide), long_hum60, music | N/A / no false removal | = | same | none (bit-identical) |
| colsto_2050_2250 | PASS | PASS | PASS | **small gain**: the steep 120→220 Hz chirp at ~90 s is now removed (+1 track) |
| colsto_2980_3170 | PARTIAL | identical output | PASS (main), minor fragments | none (the gain is pre-merge) |
| prop_flyover | FAIL | ≈ (66 vs 68 tracks, same numbers) | PARTIAL | ~none (gain is pre-merge) |
| prop_pass | FAIL | ≈ (224 vs 230 tracks) | PARTIAL | ~none (gain is pre-merge) |
| **air_p51_mustang** | FAIL | FAIL (steep core and rising curve fully visible) | PARTIAL | **clear gain**: steep 1200→700 Hz core mostly gone (+52 tracks) |
| **air_extra300** | FAIL | FAIL-ish (60 s fan visible) | PARTIAL | **gain**: 60 s fan reduced to faint (+9 tracks) |
| **air_p38** | FAIL | weak removal; 450/680 Hz lines visible | PARTIAL | **gain**: 680→600 and 460→380 Hz lines removed (+2 tracks) |
| air_hercules_airshow | FAIL | FAIL (1750→1350 Hz curve untouched) | FAIL, slightly dimmer | small gain (+4 tracks) |
| air_spitfire_presenter | FAIL | identical output | FAIL / unclear | none |
| air_spitfire_mustang, air_star_wars_canyon, air_rainbow_f18 | FAIL | identical output | FAIL | none |

Out of scope (f1, f1_pass, helicopter, air_helicopter_fast, air_military_helicopters): unchanged
by eye.

**Fixed-wing real/airshow (12 clips):**

- Published: 1 PASS / 1 PARTIAL / 10 FAIL.
- Now: 2 PASS / 5 PARTIAL / 5 FAIL.
- Of the four FAIL→PARTIAL moves, this merge accounts for p51, extra300 and p38. The prop_pass
  and prop_flyover gains already existed at freeze-1.

**No visual regressions.** Nothing previously hidden is visible again. No new ghosts or notch
walls appear, and box-free finds 0 "created" ridges on every clip.

## Collateral and the presenter speech check

### Presenter (fc's check)

| | speech SNR dB | LSD dB | LSD >450 Hz dB | speech SNR, 0–20 s (voice only, before the pass) |
|---|---|---|---|---|
| published (pre-freeze, 13/36 doppler) | 35.2 | – | – | – |
| freeze-1 re-run (76/79 doppler) | 10.4 | 1.18 | 0.74 | 9.9 |
| 37c5f34 (76/79 doppler) | 10.4 | 1.18 | 0.74 | 9.9 |

- **Against the published row: a 24.8 dB drop (FLAG, threshold 2 dB).** The voice itself is
  modified: 9.9 dB in the first 20 s, before the aircraft arrives.
- **This merge's own effect is 0.0 dB.** The output is bit-identical to freeze-1, because no
  presenter track is a chirp track or carries `width_hz`. The damage came in between 48c872c and
  freeze-1, from the frozen detector's doppler labelling.
- **Cause:** about 15 speech-period partials (1–20 s, 300–1400 Hz) are accepted as high harmonics
  (h 4–16) of low fundamentals. Examples: 1092–1378 Hz at 1.2–14.2 s, 1081–1312 Hz h13 at
  5.8–13.2 s, 423–526 Hz h6 at 9.2–20.2 s. They show up as short pieces in the voice on the
  removed-content panel. Ablation (`scripts/diag/presenter_ablate.py`):
  - dropping doppler tracks with median f in 300–3400 Hz brings speech SNR to 20.5 dB;
  - stripping `width_hz` or dropping chirp tracks changes nothing.
  - Old-vs-new runner: `scripts/diag/presenter_compare.py`, scored with `presenter_score.py`.
- By eye, the voice still looks intact in the spectrogram views. Nobody has listened to it.

### What this merge changed in fidelity (freeze-1 re-run → 37c5f34)

| clip | speech SNR dB | SNR outside dB | box-free still z>3 (new) | AF upper bound dB |
|---|---|---|---|---|
| air_p38 | 13.2 → 9.6 | **19.3 → 11.8** | 0 → 0 | -4.3 → -5.3 |
| air_hercules_airshow | 14.7 → 12.9 | **20.9 → 16.7** | 17 (3) → 16 (5) | -5.8 → -5.4 |
| air_extra300 | 5.6 → 4.6 | **13.5 → 10.7** | 13 (10) → 9 (13) | -6.2 → -4.3 |
| air_p51_mustang | 6.6 → 4.9 | **12.3 → 10.0** | 21 (18) → 16 (18) | -5.6 → -5.7 |
| colsto_2050_2250 | 8.5 → 8.5 | 25.3 → 24.1 | – | -13.7 → -13.7 |
| prop_pass, prop_flyover | ±0.1 | ±0.2 | – | ±0.3 |
| other 5 clips | identical | identical | identical | identical |

The chirp/wide removal costs 2–7.5 dB SNR outside the ±3 Hz corridors on the clips it touches.
That is expected for wider corridors, which the metric counts as outside. The speech-band
changes are at most 1.8 dB, except p38 at 3.6 dB (its 680/460 Hz chirps lie in the speech band).
The pipeline's >10% TF-area warning (40–1000 Hz) fires on 6 dev clips: prop_flyover 18.5%,
prop_pass 23.9%, p51 31.6%, spitfire_mustang 10.8%, extra300 13.0% and p38 10.4%. At freeze-1 it
fired on the first five, at 10.7–23.6%. This merge adds p38 and raises p51 from 21.1% to 31.6%.

## Runtime (s, one `pipeline.scrub` call)

Shared machine (load average 5–7 from other sessions), so treat these as rough. The synth rows
were 1.3–3× slower with bit-identical output, which is load, not code.

| clip | published | freeze-1 re-run | 37c5f34 |
|---|---|---|---|
| colsto_2050_2250 | 155 | 135 | 96 |
| colsto_2980_3170 | 85 | 59 | 69 |
| prop_pass | 46 | 48 | 50 |
| air_p51_mustang | 79 | 63 | 75 |
| air_extra300 | 32 | 24 | 31 |
| air_spitfire_presenter | 29 | 19 | 24 |
| air_hercules_airshow | 25 | 21 | 24 |
| air_star_wars_canyon | 36 | 9 | 16 |
| air_spitfire_mustang | 19 | 8 | 10 |
| air_p38 / rainbow / prop_flyover | 2 / 11 / 6 | 1 / 1 / 7 | 2 / 2 / 7 |

The chirp tracker adds roughly 10–30% on clips with chirps (p51 +12 s, extra300 +7 s) and a few
seconds elsewhere. Detection is still the bottleneck. Total dev scrub time, excluding the colsto
hour, is about 12 min.

## Box-free (40–450 Hz attacker-side extractor, 12 fixed-wing dev clips)

| | ridges z>3 before | still z>3 | new z>3 | created |
|---|---|---|---|---|
| f0531a8 (`new_ridges_boxfree_dev.json`) | 202 | 117 | 61 | 0 |
| 37c5f34 | 202 | 108 | 64 | 0 |

The gains are on p51 (21→16), extra300 (13→9) and hercules (17→16); prop_pass moved 3→4.

## Baselines (dev: synth + prop_pass + colsto_2050_2250)

dopplerscrub's dev medians are unchanged from the freeze-1 `baselines.md`. Examples: AF z after
-0.2, SNR outside 21.6, speech SNR 13.4, LSD 1.15 dB, STOI-like 0.933, TF modified 8.4%. The
only differences:

- box-free still/new: 30/17 → 31/14 (the old value predates the created/unmasked split);
- runtime median: 7.1 → 14.6 s (load).

None of the dev baseline clips carries chirp or wide tracks except colsto_2050_2250 (+1 track).
Full table: `baselines.md`.

## Things that look broken or worth a look

1. **The published benchmark table is stale for real/airshow dev rows**, from pre-freeze 48c872c.
   Anyone comparing against it, as this task's 35.2 dB presenter baseline did, compares against
   pre-freeze code. Use `freeze1_rerun_benchmark.json`, or re-run the real/airshow rows into the
   main table.
2. **Speech collateral from doppler labelling**, present since freeze-1 and not from this merge.
   About 97% of real-clip tracks are "doppler", including the presenter's voice partials (speech
   SNR 9.9 dB on the voice-only first 20 s). The fix belongs in the detector, e.g. a guard against
   isolated high harmonics in the speech band with no matching lower partials.
3. **prop_pass shows a "something was removed here" patch.** It is a broad dark region under the
   removed stack (20–35 s, 50–250 Hz) in red-team views (b)/(d), with a diagonal boundary that
   follows the pass.
4. **The wide-tone path has thin dev coverage.** It only engages where `chirps.py` measured a width
   (p51, extra300, p38, hercules). hercules' 1750→1350 Hz curve is detected but only dimmed, and
   p51's ~1000 Hz top of the rising curve (52–55 s) stays bright.
5. colsto_2980_3170: the bright non-aircraft beep blobs (95 s, 118 s, 320–430 Hz) are partly
   targeted. They remain visible, so this is pure collateral (pre-merge, identical output at
   freeze-1).
