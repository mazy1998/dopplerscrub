# Held-out test results (2026-11-04)

**Code under test:** commit `b125b22`. **Clips:** 11 fixed-wing recordings from YouTube
([docs/test_set.md](test_set.md)), collected after the code was frozen. Before the test, no
listening, no spectrograms and no tool runs were done on them. Inputs were checked against
their SHA-256 hashes before the run.

**Procedure:** (1) render before/after images only; (2) write a verdict per clip from the
images ([test_verdicts.md](test_verdicts.md), commit `4947306`); (3) only then compute the
numbers (commit `eb72010`).

**Columns:**
- **Ridges z>3**: curves in the attacker's whitened view (40–1950 Hz) that pass a
  statistical significance test, before → still present after. "Created" counts curves that
  appear only after cleaning.
- **Speech SNR**: original vs (original − cleaned) in 300–3400 Hz. It counts removed engine
  noise as error, so it's pessimistic. ∞ = nothing changed.
- **LSD**: log-spectral distance between original and cleaned audio.

## Propeller aircraft (what the tool is for)

| Recording | Verdict | Ridges z>3 before → after (created) | Speech SNR dB | LSD dB |
|---|---|---|---|---|
| King Air B350 | PASS (tones); ground-reflection fringes remain | 5 → 2 (0) | 5.8 | 1.02 |
| Cessna 172, people talking | PASS | 18 → 9 (0) | 3.2 | 1.15 |
| Cessna 150 | PARTIAL: ~500 Hz line remains | 17 → 1 (0) | 5.3 | 1.40 |
| Baron B55 | PARTIAL: curved stripes remain | 5 → 3 (0) | 2.5 | 1.17 |
| Pilatus PC-12 | FAIL: low tone not detected | 4 → 4 (0) | ∞ | 0.00 |
| Piper Cub | N/A: no pass visible; ~30 needless edits | 1 → 0 (0) | 16.6 | 0.58 |
| **Total** | 2 pass, 2 partial, 1 fail, 1 n/a | **50 → 19 (0)** | | |

## Jets (not supported; shown for completeness)

| Recording | Verdict | Ridges z>3 before → after (created) |
|---|---|---|
| A320 overhead | N/A: no Doppler curve visible; nothing removed | 7 → 7 (0) |
| Boeing 737 overhead | FAIL: ground-reflection fringes, not tones; unchanged | 4 → 4 (0) |
| Citation low pass | N/A: broadband only; nothing removed | 11 → 11 (0) |
| Alpha Jet flyby | FAIL: clear descending tones missed | 5 → 5 (0) |
| Harrier low pass | N/A / weak FAIL: faint low line remains | 9 → 9 (0) |

## A metric we dropped

We also ran an automated template-matching attacker ("usable passes left"). It found 0
usable passes even in the **original** recordings, so its "11/11 pass" result was
meaningless and is not reported as evidence. The verdicts and ridge counts above are what
count.
