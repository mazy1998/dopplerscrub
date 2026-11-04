# Final run manifest: heldout2

- started: 2026-11-04 23:06:05 +0100
- command: `scripts/final_run.py phase1 --split heldout2`
- commit: `b125b22542d7638d9feeaef83d0561bb66fc6e88` (freeze-1-29-gb125b22, branch research/detection-v2)
- tracked changes: none (clean tree)
- DetectConfig: `{"fmin": 30.0, "fmax": 2000.0, "work_sr": 4000, "nperseg": 8192, "hop_s": 0.25, "min_duration_s": 4.0, "min_sweep_hz": 0.5, "snr_db": 3.0, "doppler_threshold": 0.5}`
- RemoveConfig: `{"method": "vk", "width_hz": 2.0, "pad_s": 2.0, "harden": true, "remove_kinds": ["doppler"], "harden_min_score": 0.5}`
- baseline methods: dopplerscrub, oracle, bandstop, bandstop_brick, noise, jitter
- usable-events attacker: dopplerscrub.usable:usable_events
- python 3.13.12, numpy 2.5.4, macOS-14.4-arm64-arm-64bit-Mach-O
- annotations: `data/real/heldout2/annotations.json` sha256 `061b2203b617c7b931cdea4fed52ba58c43754ab1c0ae5345559516a42ebe6ec`, `docs/heldout2/annotations.json` sha256 `061b2203b617c7b931cdea4fed52ba58c43754ab1c0ae5345559516a42ebe6ec`
- inputs verified against annotation SHA-256: **11/11**

| clip | file | sha256 | reference | status |
|---|---|---|---|---|
| h2_c150_lowpass | `data/real/heldout2/c150_lowpass.wav` | `c897d844f1f5d378…` | `c897d844f1f5d378…` | match |
| h2_cub_lowpass | `data/real/heldout2/cub_lowpass.wav` | `dfc8501d9726f129…` | `dfc8501d9726f129…` | match |
| h2_kingair_b350 | `data/real/heldout2/kingair_b350.wav` | `a5eda37879611ebe…` | `a5eda37879611ebe…` | match |
| h2_pc12_lowpass | `data/real/heldout2/pc12_lowpass.wav` | `c703646cc015e19d…` | `c703646cc015e19d…` | match |
| h2_baron_flyby | `data/real/heldout2/baron_flyby.wav` | `bd375b3b962c89af…` | `bd375b3b962c89af…` | match |
| h2_c172_takeoff_talking | `data/real/heldout2/c172_takeoff_talking.wav` | `99553a3c191d2ea6…` | `99553a3c191d2ea6…` | match |
| h2_a320_overhead | `data/real/heldout2/a320_overhead.wav` | `c5a6c32a16bda110…` | `c5a6c32a16bda110…` | match |
| h2_b737_overhead | `data/real/heldout2/b737_overhead.wav` | `0558159397530fac…` | `0558159397530fac…` | match |
| h2_citation_lowpass | `data/real/heldout2/citation_lowpass.wav` | `0727317069c052b2…` | `0727317069c052b2…` | match |
| h2_alphajet_flyby | `data/real/heldout2/alphajet_flyby.wav` | `272e2155b1f53aca…` | `272e2155b1f53aca…` | match |
| h2_harrier_lowpass | `data/real/heldout2/harrier_lowpass.wav` | `3fbbe9d63712c762…` | `3fbbe9d63712c762…` | match |
