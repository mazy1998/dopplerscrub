# Blind held-out set 2 (heldout2)

Collected 2026-11-04 for the final evaluation after the heldout set was burned. Audio only, native sample rate, 24-bit PCM. Do not commit the audio.

No listening, spectrograms, detector/remover execution or tuning were done on these recordings. Selection used YouTube search results, titles, descriptions and durations only (descriptions screened for music/montage). Clip windows and pass windows come from a broadband loudness envelope (0.1 s RMS, 2 s moving average): clip = region within 20 dB of the envelope peak, plus 8 s each side (min 15 s); pass window = region within 10 dB of the peak, plus 2 s each side. These are not measured Doppler ground truth: the event boxes in annotations.json span 40-2000 Hz and are loudness-derived. Clips with small loudness range (c150_lowpass, cub_lowpass) have the whole clip as the pass window.

Scope: fixed-wing airplanes only (propeller and jet). Six earlier picks (helicopters, road vehicles, train, siren) were moved to data/real/heldout2_excluded/ before any results existed; they are listed under `rejected` in annotations.json and are not part of --split heldout2.

Freeze these files for the final blind evaluation: `uv run python scripts/benchmark.py --split heldout2 --blind`.

## c150_lowpass

- URL: https://www.youtube.com/watch?v=ncgTszJDc4I
- Title: Low Pass C150M @SCCV
- Uploader: Alejandro Gorigoitia
- Category: prop: light piston single (Cessna 150) runway low pass
- Source clip: 0–82.8 s (source video length 83 s)
- Loudness peak: source 79.7 s; pass window clip-relative 0–82.8 s (unverified)
- File: `c150_lowpass.wav`; 82.800 s, 48000 Hz, 2 ch
- SHA-256: `c897d844f1f5d378df794e8ddffbe5990b9a0ed2bac1d7c11a6bffd9d630c0f1`

## cub_lowpass

- URL: https://www.youtube.com/watch?v=BBz6tzSoCnI
- Title: AVX RUNWAY PROJECT- JERRY’S CUB LOW PASS
- Uploader: Jerry Rootlieb
- Category: prop: Piper Cub low pass at a private strip
- Source clip: 0–70.7 s (source video length 71 s)
- Loudness peak: source 13.4 s; pass window clip-relative 0–70.7 s (unverified)
- File: `cub_lowpass.wav`; 70.682 s, 48000 Hz, 2 ch
- SHA-256: `dfc8501d9726f12972fcff213dc28f4ce73882ee7d1cc1c44b9a5baed1ae49ff`

## kingair_b350

- URL: https://www.youtube.com/watch?v=BzUlY8YzarI
- Title: [4K] Raytheon/Beechcraft B350 King Air Flyby
- Uploader: HD Aviation Films
- Category: prop: twin turboprop (King Air B350) flyby
- Source clip: 0–21.5 s (source video length 22 s)
- Loudness peak: source 13.2 s; pass window clip-relative 0–17.5 s (unverified)
- File: `kingair_b350.wav`; 21.500 s, 48000 Hz, 2 ch
- SHA-256: `a5eda37879611ebec3ffbeb0087012d26b7dab2c8b8d52569ed4b69b3b93307b`

## pc12_lowpass

- URL: https://www.youtube.com/watch?v=x1-21YrfPHA
- Title: Pilatus PC-12 NG SP-THC LOW PASS RUNWAY 29 EPKR
- Uploader: Pavlo EPKR
- Category: prop: single turboprop (Pilatus PC-12) runway low pass
- Source clip: 3.0–24.5 s (source video length 25 s)
- Loudness peak: source 18.4 s; pass window clip-relative 11.2–19.2 s (unverified)
- File: `pc12_lowpass.wav`; 21.462 s, 48000 Hz, 2 ch
- SHA-256: `c703646cc015e19d2516e28cf499e73c6da041e5ca3410b964c01f589baedb99`

## baron_flyby

- URL: https://www.youtube.com/watch?v=2XEm7aMBfGc
- Title: Colemill Beechcraft Baron B55 flyby
- Uploader: Warren Daniel
- Category: prop: twin piston (Beechcraft Baron B55) flyby
- Source clip: 0–31.3 s (source video length 31 s)
- Loudness peak: source 20.7 s; pass window clip-relative 12.9–28.5 s (unverified)
- File: `baron_flyby.wav`; 31.277 s, 48000 Hz, 2 ch
- SHA-256: `bd375b3b962c89afca2b8f79685e561d578fb906eb9ebb42d3885d1b2dc93ca7`

## c172_takeoff_talking

- URL: https://www.youtube.com/watch?v=Ixo3n4ZHqc8
- Title: Cessna 172M Skyhawk Take-Off at Doylestown Airport (DYL)
- Uploader: The Philly Buff
- Category: prop: Cessna 172 departure, people talking near the camera (per description)
- Source clip: 1.2–58.4 s (source video length 58 s)
- Loudness peak: source 23.9 s; pass window clip-relative 14.5–35.3 s (unverified)
- File: `c172_takeoff_talking.wav`; 57.175 s, 48000 Hz, 2 ch
- SHA-256: `99553a3c191d2ea6360eab699938fe2948e72b413232e8735e40e93ae8e2d691`

## a320_overhead

- URL: https://www.youtube.com/watch?v=Uwl8Yd6TN6g
- Title: easyJet A320 low pass overhead | Manchester Airport | Plane Hunter UK
- Uploader: Plane Hunter UK
- Category: jet: airliner (A320) low overhead on approach
- Source clip: 0–49.5 s (source video length 50 s)
- Loudness peak: source 7.0 s; pass window clip-relative 0–9.9 s (unverified)
- File: `a320_overhead.wav`; 49.451 s, 48000 Hz, 2 ch
- SHA-256: `c5a6c32a16bda110a6a86cfae47216864e381f3a292528074b55d50b408a0198`

## b737_overhead

- URL: https://www.youtube.com/watch?v=Pn_YfiEWygw
- Title: [4K] Spectacular Boeing 737 Overhead Approach to Corfu Airport | Ryanair EI-DWY | 14.08.2015
- Uploader: efbeVideo
- Category: jet: airliner (737-800) overhead approach
- Source clip: 8.8–76.3 s (source video length 77 s)
- Loudness peak: source 29.2 s; pass window clip-relative 13.7–26.5 s (unverified)
- File: `b737_overhead.wav`; 67.500 s, 48000 Hz, 2 ch
- SHA-256: `0558159397530facc5e6b2227cb0dfe775379422b35f68f538f8cd7e34154d1d`

## citation_lowpass

- URL: https://www.youtube.com/watch?v=l0trQ8gRJpE
- Title: Cessna Citation Jet Does a Low Pass!
- Uploader: FarmersRC Channel
- Category: jet: business jet (Cessna Citation) low pass
- Source clip: 0–74.1 s (source video length 86 s)
- Loudness peak: source 56.7 s; pass window clip-relative 49.1–67.9 s (unverified)
- File: `citation_lowpass.wav`; 74.100 s, 48000 Hz, 2 ch
- SHA-256: `0727317069c052b26bb0f681a7a80b6e4e28eb3c60daee206b98721a45230a20`

## alphajet_flyby

- URL: https://www.youtube.com/watch?v=byLplPe-_2k
- Title: Belgian Alphajets - Very Low Flyby
- Uploader: Frederic
- Category: jet: military trainers (3x Alpha Jet) very low flyby
- Source clip: 0–39.3 s (source video length 39 s)
- Loudness peak: source 25.7 s; pass window clip-relative 10.9–34.5 s (unverified)
- File: `alphajet_flyby.wav`; 39.288 s, 48000 Hz, 2 ch
- SHA-256: `272e2155b1f53acad2934e0204138d5c8f0f5c71934e29ac67491beb15a00408`

## harrier_lowpass

- URL: https://www.youtube.com/watch?v=WRfDsSnLtE4
- Title: VERY low Harrier ("I won't flinch..")
- Uploader: Dimpyz
- Category: jet: military (Harrier) low pass, speech before the pass (per description)
- Source clip: 0–25.1 s (source video length 25 s)
- Loudness peak: source 20.1 s; pass window clip-relative 15.2–23.2 s (unverified)
- File: `harrier_lowpass.wav`; 25.089 s, 48000 Hz, 2 ch
- SHA-256: `3fbbe9d63712c7622c5541e719e465960e369055b263bb05232dcfca7f80b2fc`

## Excluded (out of scope, not in --split heldout2)

- r44_flyby: https://www.youtube.com/watch?v=2kNkgbfep58 (Robison R44 Raven II Helicopter Fly By); source 0-40.6 s; file data/real/heldout2_excluded/r44_flyby.wav
- bell407_house: https://www.youtube.com/watch?v=7ebJM2kKpjA (Low flying Bell 407 (N765MT) over my house); source 0-30.6 s; file data/real/heldout2_excluded/bell407_house.wav
- train_horn: https://www.youtube.com/watch?v=3dzV8QCWJnk (BNSF 7102 P5 Train Horn At 70 MPH (Amazing Doppler Effect)); source 0-25.1 s; file data/real/heldout2_excluded/train_horn.wav
- fire_siren: https://www.youtube.com/watch?v=imoxDcn2Sgo (Fire Engine Siren Demonstrates The Doppler Effect); source 0-18.7 s; file data/real/heldout2_excluded/fire_siren.wav
- tt_motorcycle: https://www.youtube.com/watch?v=_hf__rO7MD8 (Isle of Man TT - original first time spectator reaction); source 0-15.0 s; file data/real/heldout2_excluded/tt_motorcycle.wav
- rally_car: https://www.youtube.com/watch?v=JqwYExVlXek (Rally monte Carlo 2022.  Sebastian Loeb flat out); source 0.5-23.7 s; file data/real/heldout2_excluded/rally_car.wav

Tracked copy of data/real/heldout2/{SOURCES.md,annotations.json} (data/real is gitignored). Audio is not committed; re-fetch with the URLs and source clip times above (ffmpeg -ss/-t, pcm_s24le, native rate) and check SHA-256.
