# Blind held-out set 2 (heldout2)

Collected 2026-11-04 for the final evaluation after the heldout set was burned. Audio only, native sample rate, 24-bit PCM. Do not commit the audio.

No listening, spectrograms, detector/remover execution or tuning were done on these recordings. Selection used YouTube search results, titles, descriptions and durations only (descriptions screened for music/montage). Clip windows and pass windows come from a broadband loudness envelope (0.1 s RMS, 2 s moving average): clip = region within 20 dB of the envelope peak, plus 8 s each side (min 15 s); pass window = region within 10 dB of the peak, plus 2 s each side. These are not measured Doppler ground truth: the event boxes in annotations.json span 40-2000 Hz and are loudness-derived. Clips with small loudness range (c150_lowpass, cub_lowpass, fire_siren) have the whole clip as the pass window.

Freeze these files for the final blind evaluation: `uv run python scripts/benchmark.py --split heldout2 --blind`.

## c150_lowpass

- URL: https://www.youtube.com/watch?v=ncgTszJDc4I
- Title: Low Pass C150M @SCCV
- Uploader: Alejandro Gorigoitia
- Category: light piston single (Cessna 150) runway low pass
- Source clip: 0–82.8 s (source video length 83 s)
- Loudness peak: source 79.7 s; pass window clip-relative 0–82.8 s (unverified)
- File: `c150_lowpass.wav`; 82.800 s, 48000 Hz, 2 ch
- SHA-256: `c897d844f1f5d378df794e8ddffbe5990b9a0ed2bac1d7c11a6bffd9d630c0f1`

## cub_lowpass

- URL: https://www.youtube.com/watch?v=BBz6tzSoCnI
- Title: AVX RUNWAY PROJECT- JERRY’S CUB LOW PASS
- Uploader: Jerry Rootlieb
- Category: Piper Cub low pass at a private strip
- Source clip: 0–70.7 s (source video length 71 s)
- Loudness peak: source 13.4 s; pass window clip-relative 0–70.7 s (unverified)
- File: `cub_lowpass.wav`; 70.682 s, 48000 Hz, 2 ch
- SHA-256: `dfc8501d9726f12972fcff213dc28f4ce73882ee7d1cc1c44b9a5baed1ae49ff`

## kingair_b350

- URL: https://www.youtube.com/watch?v=BzUlY8YzarI
- Title: [4K] Raytheon/Beechcraft B350 King Air Flyby
- Uploader: HD Aviation Films
- Category: twin turboprop (King Air B350) flyby
- Source clip: 0–21.5 s (source video length 22 s)
- Loudness peak: source 13.2 s; pass window clip-relative 0–17.5 s (unverified)
- File: `kingair_b350.wav`; 21.500 s, 48000 Hz, 2 ch
- SHA-256: `a5eda37879611ebec3ffbeb0087012d26b7dab2c8b8d52569ed4b69b3b93307b`

## pc12_lowpass

- URL: https://www.youtube.com/watch?v=x1-21YrfPHA
- Title: Pilatus PC-12 NG SP-THC LOW PASS RUNWAY 29 EPKR
- Uploader: Pavlo EPKR
- Category: single turboprop (Pilatus PC-12) runway low pass
- Source clip: 3.0–24.5 s (source video length 25 s)
- Loudness peak: source 18.4 s; pass window clip-relative 11.2–19.2 s (unverified)
- File: `pc12_lowpass.wav`; 21.462 s, 48000 Hz, 2 ch
- SHA-256: `c703646cc015e19d2516e28cf499e73c6da041e5ca3410b964c01f589baedb99`

## r44_flyby

- URL: https://www.youtube.com/watch?v=2kNkgbfep58
- Title: Robison R44 Raven II Helicopter Fly By
- Uploader: Metra Force
- Category: light piston helicopter (Robinson R44) flyby
- Source clip: 0–40.6 s (source video length 41 s)
- Loudness peak: source 22.8 s; pass window clip-relative 18.7–40.6 s (unverified)
- File: `r44_flyby.wav`; 40.589 s, 48000 Hz, 2 ch
- SHA-256: `d84bb251292317ad5f25205d43262fe9c58be35ea2d6f9064ca1f6522525a28d`

## bell407_house

- URL: https://www.youtube.com/watch?v=7ebJM2kKpjA
- Title: Low flying Bell 407 (N765MT) over my house
- Uploader: Planes cars and weather
- Category: turbine helicopter (Bell 407) low over a house
- Source clip: 0–30.6 s (source video length 31 s)
- Loudness peak: source 12.4 s; pass window clip-relative 0–24.2 s (unverified)
- File: `bell407_house.wav`; 30.600 s, 48000 Hz, 2 ch
- SHA-256: `bd39715e13c2bdd7c8aa4bf16d700342005f83665e87c1cf67e307b136f5652c`

## train_horn

- URL: https://www.youtube.com/watch?v=3dzV8QCWJnk
- Title: BNSF 7102 P5 Train Horn At 70 MPH (Amazing Doppler Effect)
- Uploader: UPking
- Category: locomotive horn at 70 mph (tonal, non-engine)
- Source clip: 0–25.1 s (source video length 25 s)
- Loudness peak: source 18.0 s; pass window clip-relative 12.7–25.1 s (unverified)
- File: `train_horn.wav`; 25.100 s, 48000 Hz, 2 ch
- SHA-256: `5ea5e9bb621f2a9cb32568007d1ba9fd63455cdbe1ea353e757814b49a660535`

## fire_siren

- URL: https://www.youtube.com/watch?v=imoxDcn2Sgo
- Title: Fire Engine Siren Demonstrates The Doppler Effect
- Uploader: davidrobert2007
- Category: fire engine siren pass at speed (road vehicle, tonal siren)
- Source clip: 0–18.7 s (source video length 19 s)
- Loudness peak: source 0.0 s; pass window clip-relative 0–18.7 s (unverified)
- File: `fire_siren.wav`; 18.700 s, 48000 Hz, 2 ch
- SHA-256: `350a66d8e1e6c219be87c1dab6742fcfc714cf3aac7e43098cc8da7f4933e44f`

## tt_motorcycle

- URL: https://www.youtube.com/watch?v=_hf__rO7MD8
- Title: Isle of Man TT - original first time spectator reaction
- Uploader: Big Sky films
- Category: road-race motorcycle pass with spectator reaction (speech likely, unverified)
- Source clip: 0–15.0 s (source video length 15 s)
- Loudness peak: source 4.4 s; pass window clip-relative 1.2–15.0 s (unverified)
- File: `tt_motorcycle.wav`; 15.000 s, 48000 Hz, 2 ch
- SHA-256: `30ba014c6d888e9d135ca29a5c70b813ccbf7aabebf780198df7565557c57784`

## rally_car

- URL: https://www.youtube.com/watch?v=JqwYExVlXek
- Title: Rally monte Carlo 2022.  Sebastian Loeb flat out
- Uploader: Jose wa Liz💕
- Category: rally car flat-out pass (road vehicle)
- Source clip: 0.5–23.7 s (source video length 24 s)
- Loudness peak: source 13.4 s; pass window clip-relative 8.0–16.8 s (unverified)
- File: `rally_car.wav`; 23.196 s, 48000 Hz, 2 ch
- SHA-256: `40b304f02f6455c1d5113e39a7d6885898341c2f4373ff9e983e2bfb617ca381`

Tracked copy of data/real/heldout2/{SOURCES.md,annotations.json} (data/real is gitignored). Audio is not committed; re-fetch with the URLs and source clip times above (ffmpeg -ss/-t, pcm_s24le, native rate) and check SHA-256.
