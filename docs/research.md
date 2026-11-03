# Research notes: tracking and removing faint Doppler tones

Literature review compiled while designing dopplerscrub (2026-11-03). Formulas
marked **[derived]** are our own algebra, not quoted from a paper. Only three
papers were read in full (Chuang/Garg/Wu CCS'12, Ojowu et al. TIFS'12, Gunes &
Erdol EUSIPCO'06); the rest are confirmed from abstracts or catalogue records.

## Detection

**D1. Model-based coherent search over the whole pass, with harmonics.**
Lo & Ferguson and Ferguson & Quinn fit straight-line flight parameters plus
blade rate to the instantaneous frequency by nonlinear least squares. A
generalized Hough transform over the model parameters is equivalent to our
matched pursuit.
- Peak chirp rate **[derived]**: |df/dt| at CPA ≈ k·f0·v²/(c·d_cpa).
  Example: 100 Hz, 70 m/s, 500 m gives 2.9 Hz/s.
- A 2 s analysis window (0.5 Hz bins) smears that to about 6 Hz, losing SNR
  exactly near CPA, where the Doppler is most informative. Use T ≈ 1/√|ḟ| near
  CPA, or the **fan-chirp transform** (Weruaga & Képesi 2007), which makes a
  whole harmonic series under one Doppler factor stationary.
- Combine harmonics with SNR weights (ENF "spectrum combining", Hajj-Ahmad et
  al. 2013; Hua et al. 2021).
- Real flights turn and climb, so fit piecewise (30–60 s) and stitch with D2.

**D2. HMM/Viterbi line tracker on harmonic-summed log-likelihood ratios.**
States are frequency bins plus a null state for birth/death (Streit & Barrett
1990; Paris & Jauffret 2003; Gunes & Erdol 2006, with a Gaussian random-walk
prior and multitaper spectra).
- Prior: σ ≈ (k·f0·v_max²/(c·d_min))·hop / binwidth.
- Observation **[derived]**: whitened power z ~ Exp(1) under noise; for a tone
  with per-bin SNR s, LLR = z·s/(1+s) − ln(1+s). Sum over k·f(t) with one shared
  Doppler factor.

**False-alarm control.** Calibrate the maximum path or model score empirically
on aircraft-free recordings, or on circularly time-shifted harmonic sets: fit a
Gumbel to the per-file maxima and set the threshold for a target per-file false
alarm rate. Require ≥2 harmonics sharing one Doppler factor, or a minimum
duration.

## Removal

**R1. Doppler time-warp → remove → unwarp** (the computed-order-tracking
analogy). Resample to emission time τ(t) so every harmonic becomes a
near-constant line at k·f0, remove narrowband, then unwarp. Use polyphase or
sinc interpolation (Fyfe & Munck 1997).

**R2. Multi-order Vold-Kalman (2nd generation)** (Vold & Leuridan 1993; Tůma 2005).
- Envelope response: H(Δf) = 1/(1 + r²(2 − 2cos 2πΔf/fs)^p). Use p = 2–3.
- **[derived]** Suppression vs. frequency error: residual ≈ (δf/B)^(2p), i.e.
  about 20·2p·log10(B/δf) dB. With p = 2 and B = 3·δf_rms, that is about 38 dB.
- Refinement loop: demodulate with a wide B, fit a spline to the envelope's
  unwrapped phase as δf(t), then demodulate again with a narrow B.
- Turbulence and Lloyd's-mirror fading broaden the tone to about 0.1–0.5 Hz at
  baseband (Clair & Gabard).
- Subtraction also removes the background within ±B, which leaves a curve-shaped
  hole, so a **fill is mandatory**.

## Noise fill / inpainting

- Add complex Gaussian noise shaped by the same H and modulated onto the same
  f(t).
- Scale it frame by frame to the local PSD from flanking bins (±2B…±8 Hz,
  median), then apply envelope adjustment (Chuang/Garg/Wu):
  b̂ = |H{b_host}|/|H{b_fill}|·b_fill.
- Same principle as codec noise substitution (MPEG-4 PNS, MPEG-H IGF).
- Where speech or music harmonics cross the gap, interpolate in log-frequency or
  use TF-gap inpainting (Mokrý et al., Janssen 2.0).

## Anti-forensics lessons (ENF removal literature)

Chuang, Garg & Wu (CCS 2012; TIFS 2013) show that a ±1 Hz removal leaves a
visible notch unless it is filled. Even after filling, these checks still
detect the edit:
1. **Inter-harmonic consistency**: remove *all* harmonics at the predicted
   k·f(t), including ones too faint to detect on their own.
2. Band-edge discontinuities.
3. Variance/kurtosis statistics of the band.
4. Peak shifts introduced by the fill itself.

Nishimura (IEICE 2017) detects de-humming from residual statistics.

Target: an attacker integrates about 960 frames along a pass, so the residual
should sit ≥10 dB below the per-bin noise floor.

## Pitfalls

Reflections create extra Doppler curves (image sources). Also watch for
multi-engine beating, turbulence sidebands, and wind nonstationarity breaking
median whitening. Avoid over-subtracting the foreground where curves cross it.

## Citations

- Ferguson & Quinn, JASA 96(2):821–827, 1994
- Lo & Ferguson, IEEE JOE 2014
- Streit & Barrett, IEEE TASSP 38(4):586–598, 1990, doi:10.1109/29.52700
- Paris & Jauffret, IEEE TAES 39(2), 2003, doi:10.1109/TAES.2003.1207256
- Gunes & Erdol, EUSIPCO 2006
- Ojowu et al., IEEE TIFS 7(4), 2012, doi:10.1109/TIFS.2012.2197391
- Hajj-Ahmad, Garg & Wu, IEEE SPL 20(9):885–888, 2013
- Hua et al., IEEE TIFS 16:3874–3887, 2021, arXiv:2011.03414
- Barniv, IEEE TAES AES-21(1):144–156, 1985
- Weruaga & Képesi, Signal Processing 87(6), 2007, doi:10.1016/j.sigpro.2007.01.006
- Vold & Leuridan, SAE 931288; Vold, Mains & Blough, SAE 972007
- Tůma, VK passband width, 2005; US patent 9482644
- Pan & Lin, MSSP 20(5), 2006; Pan & Wu, ICASSP 2007
- Fyfe & Munck, MSSP 11(2):187–205, 1997
- McAulay & Quatieri, IEEE TASSP 34(4), 1986; Serra & Smith, CMJ 14(4), 1990
- Herre & Schulz, AES 4720, 1998 (PNS); Disch et al., AES 141, 2016 (IGF)
- Mokrý et al., arXiv:2001.02480
- Chuang, Garg & Wu, ACM CCS 2012; IEEE TIFS 2013, doi:10.1109/TIFS.2013.2285515
- Nishimura, IEICE EMM2017-8
- Clair & Gabard, turbulent spectral broadening (Southampton eprint 418892)
