"""Deterministic faint-source fixtures; CPA times refer to emission time.

``level_db`` is peak fundamental amplitude at CPA, in dBFS. Truth is
retained down to -85 dBFS peak (the nominal narrowband fixture noise floor).
``d_cpa`` is slant range: sqrt(altitude**2 + lateral_offset**2).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import signal
from .types import Track


def _kinematics(t, v, d, tc, c):
    q = np.asarray(t) - tc
    b = v / c
    u = (q - np.sqrt(b*b*q*q + (1-b*b)*(d/c)**2)) / (1-b*b)
    r = np.hypot(d, v*u)
    return tc + u, r, c / (c + v*v*u/r)


def flyover(kind, duration_s, sr, t_cpa, f0, v, d_cpa, harmonics,
            level_db, temp_c=15, seed=None):
    """Return continuous-phase audio and exact 0.25-second truth ridges.

    Kinds: prop, helicopter, road (or f1), jet. Road sources include a slow
    emission-time RPM drift. Helicopters include a separate tail rotor group.
    Jet broadband is range-modulated; truth describes its weak fan tones only.
    """
    aliases = {'aircraft': 'prop', 'propeller': 'prop', 'vehicle': 'road', 'f1': 'road'}
    kind = aliases.get(kind, kind)
    if kind not in ('prop', 'helicopter', 'road', 'jet'):
        raise ValueError('unknown source kind')
    c = 331.3*np.sqrt(1+temp_c/273.15)
    if duration_s <= 0 or sr <= 0 or d_cpa <= 0 or not 0 < v < c or f0 <= 0 or harmonics < 1:
        raise ValueError('require positive duration/rate/range/frequency/harmonics and subsonic speed')
    rng = np.random.default_rng(seed)
    t = np.arange(round(duration_s*sr))/sr
    te, r, ratio = _kinematics(t, v, d_cpa, t_cpa, c)
    jitter_phase = rng.uniform(0, 2*np.pi)
    def curves(tt, base, h, gain):
        ee, rr, dd = _kinematics(tt, v, d_cpa, t_cpa, c)
        rpm = 1 + .045*np.sin(2*np.pi*(ee-t_cpa)/9) if kind == 'road' else 1.
        ff = base*h*dd*rpm
        aa = 10**(level_db/20)*gain/h**.9 * d_cpa/rr
        aa *= np.exp(-2e-7*ff*(rr-d_cpa))
        aa *= 1+.08*np.sin(.7*ee+jitter_phase)+.04*np.sin(1.13*ee)
        return ff, aa
    components = [(f0, harmonics, 1., 0)]
    if kind == 'helicopter':
        components.append((np.clip(f0*4, 60, 120), 3, .35, 1))
    x = np.zeros(len(t), dtype=float)
    tracks = []
    grid = np.arange(0, duration_s, .25)
    for base, count, gain, group in components:
        for h in range(1, count+1):
            f, a = curves(t, base, h, gain*(.12 if kind == 'jet' else 1))
            # Trapezoidal integration puts f(t) at sample centers, no phase resets.
            phase = np.empty(len(t)); phase[0] = rng.uniform(0, 2*np.pi)
            phase[1:] = phase[0]+2*np.pi*np.cumsum((f[1:]+f[:-1])/(2*sr))
            valid = f < .48*sr
            x += a*np.sin(phase)*valid
            ft, at = curves(grid, base, h, gain*(.12 if kind == 'jet' else 1))
            keep = (at >= 10**(-85/20)) & (ft < .48*sr)
            if keep.any():
                tracks.append(Track(grid[keep], ft[keep], at[keep], 1.,
                    dict(f0=float(base*h), v=float(v), d_cpa=float(d_cpa),
                         t_cpa=float(t_cpa), rms_hz=0.) if kind != 'road' else None,
                    h, group, 'doppler'))
    if kind == 'jet':
        noise = signal.sosfilt(signal.butter(2, [80, min(3500, sr*.4)], fs=sr, btype='bandpass', output='sos'), rng.normal(size=len(t)))
        x += noise*10**(level_db/20)*d_cpa/r*3
    return np.clip(x, -1, 1).astype(np.float32), tracks


def mix(x, sr, *, pink=.035, wind=.012, hum=.004, mains_hz=50,
        speech=0., music=0., seed=None):
    """Add backgrounds (levels in linear RMS/approximate peak units).

    Accepts mono or samples-by-channels audio. Hum is strictly stationary.
    A common background is added to stereo channels, preserving source panning.
    """
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    n = len(x); t = np.arange(n)/sr
    white = rng.normal(size=n)
    freq = np.fft.rfftfreq(n, 1/sr)
    p = np.fft.irfft(np.fft.rfft(white)/np.sqrt(np.maximum(freq, .5)), n=n)
    p /= max(np.std(p), 1e-12)
    rumble = signal.sosfilt(signal.butter(2, min(70, sr*.1), fs=sr, output='sos'), white)
    rumble /= max(np.std(rumble), 1e-12)
    bg = pink*p + wind*rumble*(.6+.4*np.sin(.43*t)**2)
    for h in range(1, 5):
        bg += hum/h*np.sin(2*np.pi*mains_hz*h*t)
    if speech:
        # Moving resonances filter voiced harmonic excitation, mixed with breath.
        voiced = np.zeros(n)
        phase = 2*np.pi*np.cumsum(125+22*np.sin(2*np.pi*.31*t))/sr
        for h in range(1, 25):
            hz = h*(125+22*np.sin(2*np.pi*.31*t))
            weight = sum(np.exp(-.5*((hz-center)/width)**2) for center, width in
                         [(550+180*np.sin(.8*t), 120), (1500+350*np.sin(.53*t), 180), (2600, 250)])
            voiced += weight*np.sin(h*phase)/np.sqrt(h)
        envelope = (.2+.8*np.sin(2*np.pi*2.3*t)**2)*(.25+.75*np.sin(.51*t)**2)
        bg += speech*(voiced*envelope+.08*white)
    if music:
        for f in (130.81, 164.81, 196., 261.63, 329.63):
            bg += music/3*np.sin(2*np.pi*f*t)*(.7+.3*np.sin(.4*t))
    y = x + (bg[:, None] if x.ndim == 2 else bg)
    peak = np.max(np.abs(y))
    if peak > 1:
        raise ValueError('mix clips: reduce background/source levels')
    return y.astype(np.float32)


def make_dataset(outdir='data/synth', sr=16000):
    """Write ten seeded 60–180 second WAVs plus JSON truth; return records."""
    import soundfile as sf
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    specs = [
        ('prop_quiet', 90, [('prop', 40, 110, 65, 700, 3, -44)], {}),
        ('prop_faint', 120, [('prop', 55, 85, 75, 1100, 4, -61)], {}),
        ('overlap', 120, [('prop', 48, 100, 70, 800, 3, -46), ('prop', 65, 145, 85, 1000, 2, -48)], {}),
        ('speech', 90, [('prop', 43, 125, 60, 650, 3, -49)], {'speech': .07}),
        ('f1', 60, [('road', 30, 230, 75, 15, 5, -43)], {'pink': .025}),
        ('helicopter', 120, [('helicopter', 55, 23, 45, 450, 6, -46)], {}),
        ('jet', 90, [('jet', 40, 160, 150, 1700, 2, -44)], {}),
        ('music', 120, [('prop', 60, 95, 75, 900, 4, -54)], {'music': .04}),
        ('stereo', 90, [('prop', 42, 130, 80, 800, 3, -51)], {}),
        ('long_hum60', 180, [('prop', 95, 75, 55, 950, 5, -57)], {'mains_hz': 60}),
    ]
    records = []
    for i, (name, duration, events, bg) in enumerate(specs):
        audio = np.zeros(round(duration*sr), np.float32); tracks = []; metadata = []
        for j, (kind, tc, f0, v, d, h, level) in enumerate(events):
            kw = dict(kind=kind, duration_s=duration, sr=sr, t_cpa=tc, f0=f0, v=v,
                      d_cpa=d, harmonics=h, level_db=level, seed=100+i*10+j)
            y, truth = flyover(**kw); audio += y
            for track in truth:
                track.group += j*10
            tracks.extend(truth); metadata.append(kw)
        if name == 'stereo':
            audio = np.column_stack((audio, audio*.65))
        audio = mix(audio, sr, seed=500+i, **bg)
        path = outdir / (name+'.wav'); sf.write(path, audio, sr, subtype='PCM_24')
        records.append(dict(path=str(path), sr=sr, events=metadata, tracks=[t.to_dict() for t in tracks]))
    (outdir/'truth.json').write_text(json.dumps(records, indent=2)+'\n')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', nargs='?', default='data/synth')
    parser.add_argument('--sr', type=int, default=16000)
    args = parser.parse_args(); make_dataset(args.outdir, args.sr)

if __name__ == '__main__':
    main()


# ============================================================================
# Hard fixed-wing cases (data/synth_hard): arbitrary trajectories, source FM,
# multi-engine families, ground reflection, realistic backgrounds, null clips.
# Exact truth: received frequency = source frequency at the emission time,
# times the Doppler ratio dte/dto of the DIRECT path (closed form, no fitting).
# ============================================================================

def speed_of_sound(temp_c=15.):
    return 331.3*np.sqrt(1+temp_c/273.15)


def straight_track(v, t_cpa, lateral=0., alt=300.):
    """Level straight pass along x at speed v; CPA (emission time) at t_cpa."""
    def p(te):
        te = np.asarray(te, float)
        return np.column_stack((v*(te-t_cpa), np.full(len(te), lateral), np.full(len(te), alt)))
    return p


def parabolic_track(a, b, t0, y0=0., alt=300.):
    """Paper eq. 42 shape, p(u) = (a u, y0 + b u^2, alt), u = te - t0 (speed rises with |u|)."""
    def p(te):
        u = np.asarray(te, float)-t0
        return np.column_stack((a*u, y0+b*u*u, np.full(len(u), alt)))
    return p


def turn_track(v, radius, t0, center=(0., 0.), phase0=-np.pi/2, alt=250., climb=0.):
    """Constant-speed level (or climbing) circular turn; at te = t0 the aircraft is at angle phase0."""
    def p(te):
        u = np.asarray(te, float)-t0
        ang = phase0+v*u/radius
        return np.column_stack((center[0]+radius*np.cos(ang), center[1]+radius*np.sin(ang), alt+climb*u))
    return p


def smooth_noise(rng, duration_s, rate_hz, std, fs=200., t_start=0.):
    """Band-limited Gaussian process (std in output units), returned as a callable of time."""
    n = int(np.ceil(duration_s*fs))+8
    w = rng.normal(size=n+int(4*fs/max(rate_hz, .05)))
    sos = signal.butter(2, min(rate_hz, fs*.45), fs=fs, output='sos')
    s = signal.sosfiltfilt(sos, w)[-n:]
    s = s/max(np.std(s), 1e-12)*std
    tg = t_start+np.arange(n)/fs
    return lambda t: np.interp(t, tg, s)


def _path(traj, mic, t_out, te_lo, te_hi, c, dte=1e-3):
    """Emission time, range and Doppler ratio dte/dto at reception times t_out for one path."""
    te = np.arange(te_lo, te_hi, dte)
    pos = traj(te)-np.asarray(mic, float)
    r = np.linalg.norm(pos, axis=1)
    to = te+r/c
    if np.any(np.diff(to) <= 0):
        raise ValueError('supersonic or non-monotone path')
    te_o = np.interp(t_out, to, te)
    r_o = np.interp(t_out, to, r)
    rdot = np.gradient(r, dte)
    ratio = 1/(1+np.interp(t_out, to, rdot)/c)
    return te_o, r_o, ratio


def render_pass(duration_s, sr, traj, components, *, level_db, ref_range=300., mic_height=1.5,
                ground=0., broadband=None, temp_c=15., seed=None, hop=.25,
                truth_floor_db=-90.):
    """Render one aircraft along an arbitrary emission-time trajectory (metres).

    components: dicts with keys
      f      callable te -> smooth source frequency (Hz)  [truth follows this]
      jitter callable te -> extra source FM (Hz) or None [in the audio, not the truth;
             recorded as truth width_hz = 1.75 x received FM std]
      gain   linear amplitude relative to level_db (at ref_range)
      harmonic, group, base (truth f0 basis for fit dict)
    level_db: dBFS peak amplitude of a gain-1 component at slant range ref_range.
    ground: reflection coefficient (0 = none) of a flat ground; the microphone sits at
      mic_height, the image path gives a moving Lloyd's-mirror comb on broadband and fades
      on tones. Truth is the DIRECT path (the reflected tone differs by << 0.1 Hz).
    broadband: dict(level_db, band=(lo, hi), aft=gain behind the aircraft) for jet noise,
      emitted in source time (Doppler-warped, reflected like everything else).
    Returns float64 audio and truth Tracks (0.25 s grid, amp = direct-path peak amplitude).
    """
    rng = np.random.default_rng(seed)
    c = speed_of_sound(temp_c)
    n = round(duration_s*sr)
    t = np.arange(n)/sr
    pad = 30.
    te_lo, te_hi = -pad-ref_range/c-60, duration_s+pad
    mics = [(0., 0., mic_height)]
    if ground:
        mics.append((0., 0., -mic_height))   # image microphone == image source
    paths = [_path(traj, m, t, te_lo, te_hi, c) for m in mics]
    grid = np.arange(0, duration_s, hop)
    teg, rg, ratg = _path(traj, mics[0], grid, te_lo, te_hi, c)
    # uniform emission grid for phases / emitted noise (shared by both paths)
    tmin = min(p[0][0] for p in paths)-1; tmax = max(p[0][-1] for p in paths)+1
    ge = np.arange(tmin, tmax, 1/sr)
    wob_ph = rng.uniform(0, 2*np.pi, 2)
    wobble = lambda te: 1+.06*np.sin(.7*te+wob_ph[0])+.03*np.sin(1.13*te+wob_ph[1])
    A = 10**(level_db/20)
    x = np.zeros(n)
    tracks = []
    for comp in components:
        fe = comp['f'](ge)
        fj = fe+(comp['jitter'](ge) if comp.get('jitter') else 0.)
        ph = np.empty(len(ge)); ph[0] = rng.uniform(0, 2*np.pi)
        ph[1:] = ph[0]+2*np.pi*np.cumsum((fj[1:]+fj[:-1])/(2*sr))
        g = comp.get('gain', 1.)
        for k, (te_s, r_s, rat_s) in enumerate(paths):
            f_rx = np.interp(te_s, ge, fe)*rat_s
            amp = A*g*ref_range/r_s*np.exp(-2e-7*f_rx*(r_s-ref_range))*wobble(te_s)
            if k:
                amp *= ground
            x += amp*np.sin(np.interp(te_s, ge, ph))*(f_rx < .48*sr)
        ft = comp['f'](teg)*ratg
        at = A*g*ref_range/rg*np.exp(-2e-7*ft*(rg-ref_range))*wobble(teg)
        keep = (at >= 10**(truth_floor_db/20)) & (ft < .48*sr)
        if keep.any():
            width = None
            if comp.get('jitter'):
                width = float(round(1.75*np.std(comp['jitter'](ge))*np.max(ratg), 2))
            tracks.append(Track(grid[keep], ft[keep], at[keep], 1., comp.get('fit'),
                                int(comp.get('harmonic', 1)), int(comp.get('group', 0)),
                                'doppler', width))
    if broadband:
        lo, hi = broadband.get('band', (60, 4000))
        w = signal.sosfilt(signal.butter(2, [lo, min(hi, sr*.45)], fs=sr, btype='bandpass', output='sos'),
                           rng.normal(size=len(ge)))
        w /= np.std(w)
        # gentle tilt: more low-mid roar than hiss
        w = .7*w+.3*signal.sosfilt(signal.butter(1, 600, fs=sr, output='sos'), w)/.35
        B = 10**(broadband['level_db']/20)
        aft = broadband.get('aft', 1.5)
        for k, (te_s, r_s, _) in enumerate(paths):
            pos = traj(te_s)-np.asarray(mics[k])
            vel = (traj(te_s+.05)-traj(te_s-.05))/.1
            cosang = np.sum(pos*vel, axis=1)/np.maximum(np.linalg.norm(pos, axis=1)*np.linalg.norm(vel, axis=1), 1e-9)
            directivity = 1+aft*np.clip(-cosang, 0, 1)**2    # louder once it has passed (exhaust aft)
            x += (ground if k else 1.)*B*ref_range/r_s*directivity*np.interp(te_s, ge, w)
    return x, tracks


def prop_components(f0, harmonics, *, group=0, gains=None, jitter=None, drift=None,
                    sidebands=0., blades=3, v=None, d_cpa=None, t_cpa=None, sb_group=None):
    """Harmonic family of a blade-pass fundamental f0 (Hz, may be callable of te).

    gains: per-harmonic linear gains (default 1/h**.6); jitter: callable te -> fractional FM
    (applied to every harmonic, scaled by h*f0); drift: callable te -> fractional RPM change
    (part of the truth); sidebands: linear gain of shaft-rate sidebands h*f0 +- f0/blades
    (truth harmonics 3h+-1 of the shaft rate, group sb_group)."""
    base = f0 if callable(f0) else (lambda te, f0=f0: np.full(np.shape(te), float(f0)))
    rpm = drift or (lambda te: 0.)
    out = []
    for h in range(1, harmonics+1):
        g = gains[h-1] if gains is not None else h**-.6
        fit = None
        if v is not None and not callable(f0) and drift is None:
            fit = dict(f0=float(f0*h), v=float(v), d_cpa=float(d_cpa), t_cpa=float(t_cpa), rms_hz=0.)
        out.append(dict(f=lambda te, h=h: h*base(te)*(1+rpm(te)),
                        jitter=(lambda te, h=h: h*base(te)*jitter(te)) if jitter else None,
                        gain=g, harmonic=h, group=group, fit=fit))
        if sidebands:
            for s in (-1, 1):
                k = blades*h+s
                fit_s = None
                if fit:
                    fit_s = dict(fit, f0=float(f0*k/blades))
                out.append(dict(f=lambda te, k=k: k/blades*base(te)*(1+rpm(te)),
                                jitter=(lambda te, k=k: k/blades*base(te)*jitter(te)) if jitter else None,
                                gain=g*sidebands, harmonic=k,
                                group=sb_group if sb_group is not None else group+50, fit=fit_s))
    return out


# ------------------------------------------------------------------ backgrounds

def speech_like(n, sr, *, f0=120., level=.02, talkers=2, pause=.15, seed=None):
    """Continuous conversational speech stand-in: per talker a voiced source with random
    intonation (smooth pitch walk +-25%), syllable bursts at ~4 Hz, short pauses, three
    moving formants, plus unvoiced frication. Talkers overlap; peak ~ level*3."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/sr
    dur = n/sr
    out = np.zeros(n)
    for k in range(talkers):
        fk = f0*(1+.6*k)*rng.uniform(.9, 1.1)
        pitch = fk*np.exp(smooth_noise(rng, dur, 1.5, .12)(t))
        ph = 2*np.pi*np.cumsum(pitch)/sr
        # syllables: rectified smooth noise at ~4 Hz, phrases with pauses
        syl = np.maximum(smooth_noise(rng, dur, 5., 1.)(t), 0)**1.5
        phrase = (smooth_noise(rng, dur, .3, 1.)(t) > -pause*4).astype(float)
        phrase = np.convolve(phrase, np.ones(int(.05*sr))/int(.05*sr), 'same')
        env = syl*phrase
        fm = [smooth_noise(rng, dur, 2., s)(t) for s in (120, 250, 200)]
        centers = [(550+fm[0], 110), (1500+fm[1], 160), (2550+fm[2], 230)]
        voiced = np.zeros(n)
        for h in range(1, int(3800/fk)):
            hz = h*pitch
            wgt = sum(np.exp(-.5*((hz-cc)/bw)**2) for cc, bw in centers)+.02
            voiced += wgt*np.sin(h*ph)/np.sqrt(h)*(hz < .45*sr)
        fric = signal.sosfilt(signal.butter(2, [2500, min(7000, sr*.45)], fs=sr, btype='bandpass', output='sos'),
                              rng.normal(size=n))
        unv = np.maximum(smooth_noise(rng, dur, 6., 1.)(t)-1., 0)
        sig = voiced*env+.6*fric*unv*phrase
        out += sig/max(np.std(sig), 1e-12)
    return level*out/np.sqrt(talkers)


def wind_gusts(n, sr, *, level=.03, seed=None):
    """Microphone wind: 1/f^2-ish rumble below ~150 Hz with gust envelopes (lognormal)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/sr
    w = rng.normal(size=n)
    r = signal.sosfilt(signal.butter(2, 25, fs=sr, output='sos'), w)*4+signal.sosfilt(
        signal.butter(2, 150, fs=sr, output='sos'), w)
    r /= np.std(r)
    gust = np.exp(smooth_noise(rng, n/sr, .15, .8)(t))
    gust /= np.mean(gust)
    return level*r*gust


def traffic_rumble(n, sr, *, level=.02, idle_hz=(27., 41.), seed=None):
    """Distant road traffic: broadband 30-600 Hz roar with slow swells, plus faint
    near-stationary engine lines (idle firing rates, drifting < 1%)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/sr
    w = signal.sosfilt(signal.butter(2, [30, 600], fs=sr, btype='bandpass', output='sos'), rng.normal(size=n))
    w = w/np.std(w)*(1+.5*smooth_noise(rng, n/sr, .05, 1.)(t))
    out = w
    for f in idle_hz:
        drift = f*(1+.006*smooth_noise(rng, n/sr, .05, 1.)(t))
        ph = 2*np.pi*np.cumsum(drift)/sr
        for h in (1, 2, 3):
            out = out+.25/h*np.sin(h*ph+rng.uniform(0, 6.3))
    return level*out/np.std(out)


def music_like(n, sr, *, level=.03, bpm=96, seed=None):
    """Tonal music: chord pads, a melody with vibrato and portamento glides, bass, kick."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/sr
    beat = 60/bpm
    roots = [0, 5, 7, 3, 8, 10]
    midi = lambda m: 440*2**((m-69)/12)
    out = np.zeros(n)
    bar = 4*beat
    nb = int(np.ceil(n/sr/bar))
    prog = rng.choice(roots, nb)
    seg = np.minimum((t/bar).astype(int), nb-1)
    for iv, g in ((0, 1.), (4, .7), (7, .7), (12, .4)):
        f = midi(48+prog[seg]+iv)
        ph = 2*np.pi*np.cumsum(f)/sr
        out += g*(np.sin(ph)+.3*np.sin(2*ph)+.15*np.sin(3*ph))
    bass = midi(36+prog[seg]); out += 1.2*np.sin(2*np.pi*np.cumsum(bass)/sr)
    # melody: notes on beats, glide 60 ms, vibrato 5.5 Hz, +-0.4 semitone
    nn = int(np.ceil(n/sr/beat*2))
    notes = 72+np.cumsum(rng.choice([-3, -2, -1, 0, 1, 2, 3, 5], nn)) % 14
    idx = np.minimum((t/(beat/2)).astype(int), nn-1)
    m = np.asarray(notes, float)[idx]
    m = np.convolve(m, np.ones(int(.06*sr))/int(.06*sr), 'same')
    m += .4*np.sin(2*np.pi*5.5*t)*(t % (beat/2) > .15)
    f = midi(m)
    ph = 2*np.pi*np.cumsum(f)/sr
    env = np.exp(-((t % (beat/2))/.35))
    out += 1.5*env*(np.sin(ph)+.5*np.sin(2*ph)+.25*np.sin(3*ph))
    kick_env = np.exp(-(t % beat)/.08)
    out += 2*kick_env*np.sin(2*np.pi*(50+80*np.exp(-(t % beat)/.03))*(t % beat))
    return level*out/np.std(out)


def birdsong(n, sr, *, level=.02, seed=None):
    """Bird-like chirps: repeated down/up FM sweeps (mostly 1.5-6 kHz, some dipping to 900 Hz),
    trills, and a low dove coo (~450 Hz, 0.5 s notes). Each element < 1.5 s."""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    dur = n/sr
    tt = 0.
    while tt < dur-2:
        kind = rng.choice(['sweep', 'trill', 'coo'], p=[.55, .3, .15])
        if kind == 'sweep':
            reps = rng.integers(2, 6); f_hi = rng.uniform(1800, 6000); f_lo = rng.uniform(900, f_hi*.7)
            L = rng.uniform(.08, .35)
            for r in range(reps):
                s0 = int((tt+r*L*1.6)*sr); m = int(L*sr)
                if s0+m >= n: break
                u = np.arange(m)/m
                f = f_hi+(f_lo-f_hi)*u**rng.uniform(.5, 2)
                out[s0:s0+m] += np.sin(2*np.pi*np.cumsum(f)/sr)*np.sin(np.pi*u)**2
            tt += reps*L*1.6+rng.uniform(.5, 3)
        elif kind == 'trill':
            m = int(rng.uniform(.6, 1.4)*sr); s0 = int(tt*sr)
            if s0+m >= n: break
            u = np.arange(m)/sr
            fc = rng.uniform(2500, 5000); f = fc+rng.uniform(200, 800)*np.sin(2*np.pi*rng.uniform(15, 30)*u)
            out[s0:s0+m] += np.sin(2*np.pi*np.cumsum(f)/sr)*np.sin(np.pi*u/u[-1])
            tt += m/sr+rng.uniform(.5, 3)
        else:
            fc = rng.uniform(380, 520)
            for r in range(3):
                m = int(.5*sr); s0 = int((tt+r*.7)*sr)
                if s0+m >= n: break
                u = np.arange(m)/m
                f = fc*(1+.06*np.sin(np.pi*u))
                ph = 2*np.pi*np.cumsum(f)/sr
                out[s0:s0+m] += .7*(np.sin(ph)+.3*np.sin(2*ph))*np.sin(np.pi*u)
            tt += 2.1+rng.uniform(1, 4)
    return level*out/max(np.std(out), 1e-12)


def generator_hum(n, sr, *, rpm=1800., level=.02, drift=.02, cyl=4, seed=None):
    """Stationary engine-generator: firing-order harmonics of the shaft rate (rpm/60) up to
    ~1.5 kHz with governor hunting (slow drift, std `drift` fraction), plus 50 Hz-locked
    alternator lines. Slow drift at high harmonics sweeps several Hz: a classic false alarm."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/sr
    fr = rpm/60*(1+drift*smooth_noise(rng, n/sr, .04, 1.)(t)+drift*.3*smooth_noise(rng, n/sr, .3, 1.)(t))
    ph = 2*np.pi*np.cumsum(fr)/sr
    out = np.zeros(n)
    for h in range(1, int(1500/(rpm/60))):
        g = (1. if h % (cyl//2) == 0 else .35)/h**.5*np.exp(rng.normal(0, .4))
        out += g*np.sin(h*ph+rng.uniform(0, 6.3))
    for h in (1, 2, 3, 5):
        out += .3/h*np.sin(2*np.pi*50*h*t)
    out += .3*signal.sosfilt(signal.butter(2, [100, 2500], fs=sr, btype='bandpass', output='sos'), rng.normal(size=n))
    return level*out/np.std(out)


def ambience(n, sr, *, pink=.004, wind=.002, seed=None):
    """Quiet outdoor floor (pink + light low rumble); same recipe as mix() without hum."""
    return mix(np.zeros(n), sr, pink=pink, wind=wind, hum=0., seed=seed).astype(float)


def _lumpy(rng, count, slope=.5, spread_db=5.):
    return np.array([h**-slope*10**(rng.normal(0, spread_db)/20) for h in range(1, count+1)])


def _bg(n, sr, seed, **kw):
    """Background recipe: ambience always, plus optional speech/wind/traffic/music/birds/generator."""
    y = ambience(n, sr, pink=kw.pop('pink', .004), wind=kw.pop('wind_floor', .002), seed=seed)
    makers = dict(speech=speech_like, wind=wind_gusts, traffic=traffic_rumble, music=music_like,
                  birds=birdsong, generator=generator_hum)
    for i, (k, opts) in enumerate(kw.items()):
        opts = dict(opts); opts.setdefault('seed', seed+1+i)
        y += makers[k](n, sr, **opts)
    return y


def hard_specs():
    """The data/synth_hard clip list: (name, category, duration_s, builder, description).
    builder(sr, rng_seed) -> (aircraft_audio, truth_tracks, event_metadata, background_kwargs)."""
    specs = []

    def add(name, category, duration, description, bg, events):
        specs.append(dict(name=name, category=category, duration=duration,
                          description=description, bg=bg, events=events))

    S = lambda v, t_cpa, lateral, alt: dict(kind='straight', v=v, t_cpa=t_cpa, lateral=lateral, alt=alt)
    # 1. wide FM-jittered tone (airshow P-51 core: ~150 Hz/s, 1250 -> 600 Hz)
    add('wide_fm_tone', 'wide_fm', 60,
        'Warbird-like dive pass, 4th harmonic dominant (812 Hz source -> ~1245/605 Hz); RPM jitter '
        '0.4% std at ~4 Hz rate, so h4 is ~7 Hz wide with a skirt. Steepest ~170 Hz/s.',
        dict(), [dict(traj=S(118, 28, 120, 180), f0=203., harmonics=6, gains=[.08, .12, .2, 1., .3, .2],
                      jitter=dict(std=.004, rate=4.), level_db=-30)])
    add('wide_fm_airshow', 'wide_fm', 75,
        'Same physics, faster/closer (190 Hz base, h4 ~760 Hz source), 0.5% jitter, under an '
        'airshow announcer (continuous speech) and gusty wind.',
        dict(speech=dict(level=.012, f0=110, talkers=1), wind=dict(level=.02)),
        [dict(traj=S(125, 36, 80, 150), f0=190., harmonics=7, gains=[.3, .35, .4, 1., .45, .25, .2],
              jitter=dict(std=.005, rate=5.), level_db=-31)])
    # 2. twin props
    add('twin_props_a', 'twin', 90,
        'Twin-engine prop: two 3-blade families at 88 and 92.5 Hz (constant 4.5 Hz source offset), '
        'h1-6 each, shaft-rate sidebands (h*f0 +- f0/3) at -14 dB.',
        dict(), [dict(traj=S(75, 45, 250, 300), f0=88., harmonics=6, sidebands=.2, blades=3, level_db=-44, group=0, aircraft=0),
                 dict(traj=S(75, 45, 250, 300), f0=92.5, harmonics=6, sidebands=.2, blades=3, level_db=-46, group=1,
                      aircraft=0)])
    add('twin_props_b', 'twin', 75,
        'Twin turboprop, 4-blade families at 102 and 108 Hz (6 Hz offset), sidebands -12 dB, '
        'lower pass with a ground reflection (R=0.5).',
        dict(), [dict(traj=S(60, 38, 150, 200), f0=102., harmonics=6, sidebands=.25, blades=4, level_db=-46,
                      group=0, ground=.5, aircraft=0),
                 dict(traj=S(60, 38, 150, 200), f0=108., harmonics=6, sidebands=.25, blades=4, level_db=-47,
                      group=1, ground=.5, aircraft=0)])
    # 3. dense prop fan
    add('dense_fan_a', 'dense_fan', 60,
        'Crop-duster-like close low pass (60 m slant): f0 95 Hz, 20 lumpy harmonics to 1.9 kHz source; '
        'h20 sweeps ~450 Hz/s at CPA.',
        dict(), [dict(traj=S(70, 30, 40, 45), f0=95., harmonics=20, lumpy=dict(slope=.35, spread_db=5.),
                      level_db=-40)])
    add('dense_fan_b', 'dense_fan', 60,
        'f0 82 Hz, 18 harmonics, 92 m slant, ground reflection (R=0.6) fades the tones.',
        dict(wind=dict(level=.006)), [dict(traj=S(55, 26, 60, 70), f0=82., harmonics=18,
                                           lumpy=dict(slope=.4, spread_db=6.), level_db=-40, ground=.6)])
    # 4. jets
    add('jet_a', 'jet', 90,
        'Jet flyover (292 m slant, 110 m/s): fan tone 1180 Hz plus compressor tones 1720/2350 Hz '
        '(0.3% N1 drift, in the truth) under a strong broadband roar, louder aft; ground reflection '
        '(R=0.8, mic 1.7 m) gives a moving Lloyd\'s-mirror comb. Truth = tones only.',
        dict(), [dict(traj=S(110, 40, 150, 250), jet_tones=[(1180., 1.), (1720., .6), (2350., .5)],
                      level_db=-47, broadband=dict(level_db=-27, band=(60, 6000), aft=2.), ground=.8,
                      mic_height=1.7, drift=dict(std=.003, rate=.05))])
    add('jet_b', 'jet', 75,
        'Low fast jet (117 m slant, 150 m/s): tones 1450/1950/2300 Hz source, roar 20 dB over the '
        'tones, R=0.9, mic 1.5 m.',
        dict(wind=dict(level=.008)),
        [dict(traj=S(150, 34, 60, 100), jet_tones=[(1450., 1.), (1950., .7), (2300., .5)], level_db=-47,
              broadband=dict(level_db=-27, band=(60, 8000), aft=2.5), ground=.9, mic_height=1.5,
              drift=dict(std=.002, rate=.08))])
    # 5. maneuvering
    add('maneuver_parabola', 'maneuver', 120,
        'Curved track of the paper\'s eq. 42 shape, p(u) = (80u, 150 + 0.55u^2, 250) m: speed '
        '80 -> 115 m/s, heading turns ~100 deg. Prop f0 105 Hz, 4 harmonics.',
        dict(), [dict(traj=dict(kind='parabola', a=80., b=.55, t0=60., y0=150., alt=250.), f0=105.,
                      harmonics=4, level_db=-58)])
    add('maneuver_turn', 'maneuver', 100,
        'Constant-speed circling (r 700 m, 65 m/s, 68 s per orbit, nearest ~280 m) with a +3% '
        'throttle step at 70 s: approach/recede/approach, no straight template fits.',
        dict(), [dict(traj=dict(kind='turn', v=65., radius=700., t0=50., center=(0., 500.), alt=200.),
                      f0=96., harmonics=4, level_db=-54, throttle=dict(at=70., frac=.03, tau=3.))])
    # 6. interference
    add('speech_prop', 'interference', 90,
        'Prop pass (115 Hz, 4 harmonics, 400 m) under continuous two-talker speech whose voice '
        'harmonics cover the same band.',
        dict(speech=dict(level=.02, f0=118, talkers=2, pause=.05)),
        [dict(traj=S(65, 45, 230, 330), f0=115., harmonics=4, level_db=-60)])
    add('wind_traffic_prop', 'interference', 90,
        'Prop pass (72 Hz, 5 harmonics, 500 m) with heavy gusty mic wind and distant road traffic '
        '(broadband 30-600 Hz roar + idle lines at 27/41 Hz).',
        dict(wind=dict(level=.05), traffic=dict(level=.012)),
        [dict(traj=S(60, 48, 300, 400), f0=72., harmonics=5, level_db=-43)])
    # 7. faint
    add('faint_long_a', 'faint', 120,
        'Distant high pass (1.6 km slant): f0 98 Hz, 3 harmonics, level set so the attacker\'s '
        'integrated z is ~5.',
        dict(), [dict(traj=S(70, 60, 1500, 600), f0=98., harmonics=3, level_db=-61.)])
    add('faint_long_b', 'faint', 120,
        'Very distant pass (2.2 km), f0 130 Hz, 2 harmonics, with light speech; attacker z ~4.',
        dict(speech=dict(level=.004, f0=140, talkers=1)),
        [dict(traj=S(80, 55, 2000, 900), f0=130., harmonics=2, level_db=-64.)])
    # 8. nulls (no aircraft)
    add('null_music', 'null', 75, 'Tonal music: chords, bass, melody with vibrato and glides, kick. No aircraft.',
        dict(music=dict(level=.04)), [])
    add('null_birds', 'null', 75, 'Birdsong: FM sweeps (some down to 900 Hz), trills, a 450 Hz dove coo. No aircraft.',
        dict(birds=dict(level=.03)), [])
    add('null_traffic', 'null', 90, 'Distant road-traffic roar with idle engine lines, light wind. No aircraft.',
        dict(traffic=dict(level=.03), wind=dict(level=.01)), [])
    add('null_generator', 'null', 90,
        'Stationary generator, 1800 rpm (30 Hz shaft) harmonics to 1.5 kHz with 0.8% governor drift '
        '(h45 wanders ~+-11 Hz), 50 Hz alternator lines. No aircraft.',
        dict(generator=dict(level=.03, drift=.008)), [])
    add('null_ambience', 'null', 60, 'Outdoor ambience only (pink floor + light wind). No aircraft.',
        dict(wind=dict(level=.008)), [])
    # 9. multi-pass recordings (what the geolocation attacker actually needs: >= 2 usable passes)
    W = lambda tc, half=100: (tc-half, tc+half)
    twin = lambda tc, v, lat, alt, f0, off, lvl, air, **kw: [
        dict(traj=S(v, tc, lat, alt), f0=f0, harmonics=6, sidebands=.2, blades=3, level_db=lvl, group=0,
             aircraft=air, window=W(tc), **kw),
        dict(traj=S(v, tc, lat, alt), f0=f0+off, harmonics=6, sidebands=.2, blades=3, level_db=lvl-2, group=1,
             aircraft=air, window=W(tc), **kw)]
    add('multipass_mixed', 'multipass', 480,
        'Five aircraft in 8 min: single prop (350 m), twin (600 m, 5 Hz offset), jet (400 m, roar + '
        'ground comb) overlapped 20 s later by a second prop, close dense-fan prop (100 m), distant '
        'prop (1.2 km).',
        dict(wind=dict(level=.006)),
        [dict(traj=S(65, 55, 200, 290), f0=110., harmonics=4, level_db=-50, aircraft=0, window=W(55)),
         *twin(150, 75, 450, 400, 86., 5., -50, 1),
         dict(traj=S(115, 235, 250, 315), jet_tones=[(1250., 1.), (1700., .6)], level_db=-50,
              broadband=dict(level_db=-30, band=(60, 6000), aft=2.), ground=.8, mic_height=1.5,
              drift=dict(std=.003, rate=.05), aircraft=2, window=W(235, 110)),
         dict(traj=S(60, 255, 500, 500), f0=98., harmonics=3, level_db=-52, aircraft=3, window=W(255)),
         dict(traj=S(70, 330, 60, 80), f0=92., harmonics=16, lumpy=dict(slope=.4, spread_db=5.),
              level_db=-46, aircraft=4, window=W(330)),
         dict(traj=S(75, 420, 1100, 500), f0=120., harmonics=3, level_db=-56, aircraft=5, window=W(420))])
    add('multipass_speech', 'multipass', 420,
        'Four aircraft under continuous two-talker speech for 7 min: prop (400 m), twin (500 m, 4 Hz '
        'offset), jet (350 m), and a circling prop (orbit r 600 m).',
        dict(speech=dict(level=.018, f0=115, talkers=2, pause=.05)),
        [dict(traj=S(70, 60, 250, 310), f0=105., harmonics=4, level_db=-54, aircraft=0, window=W(60)),
         *twin(160, 70, 350, 360, 90., 4., -54, 1),
         dict(traj=S(120, 255, 200, 280), jet_tones=[(1150., 1.), (1600., .7)], level_db=-50,
              broadband=dict(level_db=-29, band=(60, 6000), aft=2.), ground=.8, mic_height=1.5,
              drift=dict(std=.003, rate=.05), aircraft=2, window=W(255, 110)),
         dict(traj=dict(kind='turn', v=60., radius=600., t0=350., center=(0., 450.), alt=200.), f0=100.,
              harmonics=4, level_db=-54, aircraft=3, window=(300, 420))])
    add('multipass_faint', 'multipass', 600,
        'Six passes in 10 min near the faint limit (attacker z ~3-10 each): distant props at 0.9-2.5 km, '
        'one twin, two of them overlapping, under ambience and light traffic.',
        dict(traffic=dict(level=.004)),
        [dict(traj=S(70, 60, 1400, 600), f0=98., harmonics=3, level_db=-60, aircraft=0, window=W(60)),
         dict(traj=S(80, 160, 2000, 900), f0=125., harmonics=2, level_db=-56, aircraft=1, window=W(160)),
         *twin(260, 75, 1000, 700, 84., 5., -57, 2),
         dict(traj=S(65, 350, 800, 500), f0=110., harmonics=3, level_db=-60, aircraft=3, window=W(350)),
         dict(traj=S(85, 365, 1800, 800), f0=140., harmonics=2, level_db=-55, aircraft=4, window=W(365)),
         dict(traj=S(70, 500, 2300, 1000), f0=105., harmonics=3, level_db=-53, aircraft=5, window=W(500))])
    return specs


def _trajectory(spec):
    s = dict(spec)
    kind = s.pop('kind')
    if kind == 'straight':
        return straight_track(s['v'], s['t_cpa'], s['lateral'], s['alt'])
    if kind == 'parabola':
        return parabolic_track(**s)
    if kind == 'turn':
        return turn_track(**s)
    raise ValueError(kind)


def _event_components(ev, rng):
    tr = ev['traj']
    st = tr['kind'] == 'straight'
    geo = dict(v=tr['v'], d_cpa=float(np.hypot(tr['lateral'], tr['alt']-ev.get('mic_height', 1.5))),
               t_cpa=tr['t_cpa']) if st else {}
    group = ev.get('group', 0)
    if 'jet_tones' in ev:
        d = ev.get('drift')
        drift = smooth_noise(rng, 400, d['rate'], d['std'], t_start=-200) if d else (lambda te: 0.)
        out = []
        for k, (f, g) in enumerate(ev['jet_tones']):
            out.append(dict(f=lambda te, f=f: f*(1+drift(te)), gain=g, harmonic=1, group=group+k))
        return out
    jit = None
    if ev.get('jitter'):
        j = ev['jitter']; jit = smooth_noise(rng, 400, j['rate'], j['std'], fs=400., t_start=-200)
    drift = None
    if ev.get('throttle'):
        th = ev['throttle']
        drift = lambda te, th=th: th['frac']*.5*(1+np.tanh((np.asarray(te)-th['at'])/th['tau']))
    gains = ev.get('gains')
    if ev.get('lumpy'):
        gains = _lumpy(rng, ev['harmonics'], **ev['lumpy'])
    return prop_components(ev['f0'], ev['harmonics'], group=group, gains=gains, jitter=jit, drift=drift,
                           sidebands=ev.get('sidebands', 0.), blades=ev.get('blades', 3),
                           sb_group=group+50, **geo)


def _shift_comps(comps, dt):
    """Components re-expressed in a window's local emission time (te_local = te - dt)."""
    out = []
    for c in comps:
        c = dict(c)
        for k in ('f', 'jitter'):
            if c.get(k):
                c[k] = (lambda te, g=c[k]: g(np.asarray(te)+dt))
        out.append(c)
    return out


def pass_truth(ev, mic_height=1.5, temp_c=15.):
    """Geometry truth of one event: CPA emission time, slant distance d, speed v at CPA,
    reception time h of the CPA (paper's h = t_cpa + d/c), and source fundamentals."""
    traj = _trajectory(ev['traj'])
    c = speed_of_sound(temp_c)
    tr = ev['traj']
    if tr['kind'] == 'straight':
        te_cpa = float(tr['t_cpa'])
    else:
        te = np.arange(-300, 1200, .01)
        r = np.linalg.norm(traj(te)-np.array([0, 0, mic_height]), axis=1)
        te_cpa = float(te[np.argmin(r)])
    p = traj(np.array([te_cpa-.01, te_cpa, te_cpa+.01]))
    d = float(np.linalg.norm(p[1]-np.array([0, 0, mic_height])))
    v = float(np.linalg.norm(p[2]-p[0])/.02)
    if 'jet_tones' in ev:
        f0 = [float(f) for f, _ in ev['jet_tones']]
        kind = 'jet'
    else:
        f0 = [float(ev['f0'])]
        kind = 'prop'
    return dict(v=v, d=d, t_cpa_emission=te_cpa, h=te_cpa+d/c, f0=f0, kind=kind,
                straight=tr['kind'] == 'straight', level_db=ev['level_db'])


def build_hard_clip(spec, sr=48000, seed=0):
    """Render one hard clip -> (audio float32, tracks, metadata).

    Each event is one engine family; events sharing an `aircraft` id (twins) form one pass.
    Track groups are offset by 100 * aircraft (so group // 100 = aircraft/pass id; sidebands
    sit at +50). An event `window` (t0, t1) renders the pass only there, with 4 s raised-cosine
    fades (applied to the truth amplitudes too) — used by the long multi-pass clips.
    """
    n = round(spec['duration']*sr)
    x = np.zeros(n)
    tracks = []
    rng = np.random.default_rng(seed)
    passes = {}
    for j, ev in enumerate(spec['events']):
        comps = _event_components(ev, rng)
        air = ev.get('aircraft', j)
        for c_ in comps:
            c_['group'] = c_.get('group', 0)+100*air
        traj = _trajectory(ev['traj'])
        t0, t1 = ev.get('window', (0, spec['duration']))
        t0, t1 = max(0., float(t0)), min(float(spec['duration']), float(t1))
        if t1-t0 < 1:
            continue
        if t0 > 0 or t1 < spec['duration']:
            comps = _shift_comps(comps, t0)
            traj = (lambda te, g=traj, dt=t0: g(np.asarray(te, float)+dt))
        y, tr = render_pass(t1-t0, sr, traj, comps,
                            level_db=ev['level_db'], ref_range=300., mic_height=ev.get('mic_height', 1.5),
                            ground=ev.get('ground', 0.), broadband=ev.get('broadband'),
                            seed=seed+100+j)
        if t0 > 0 or t1 < spec['duration']:
            ramp = lambda tt: np.clip(np.minimum((tt-t0)/4 if t0 > 0 else 1, (t1-tt)/4 if t1 < spec['duration'] else 1), 0, 1)
            fade = lambda tt: np.sin(np.pi/2*ramp(tt))**2
            i0 = round(t0*sr); y = y[:n-i0]
            x[i0:i0+len(y)] += y*fade(t0+np.arange(len(y))/sr)
            for q in tr:
                q.t = q.t+t0
                q.amp = q.amp*fade(q.t)
                keep = q.amp >= 10**(-90/20)
                q.t, q.f, q.amp = q.t[keep], q.f[keep], q.amp[keep]
            tr = [q for q in tr if len(q.t) >= 2]
        else:
            x += y
        tracks += tr
        pt = pass_truth(ev, ev.get('mic_height', 1.5))
        if air in passes:     # twin: second engine of the same aircraft
            passes[air]['f0'] += pt['f0']
            passes[air]['kind'] = 'twin'
            passes[air]['groups'] += sorted({q.group for q in tr})
        else:
            passes[air] = dict(event=air, **pt, groups=sorted({q.group for q in tr}))
    x += _bg(n, sr, seed+1000, **spec['bg'])
    peak = float(np.max(np.abs(x)))
    if peak > .98:
        raise ValueError(f"{spec['name']} clips (peak {peak:.2f}): lower levels")
    meta = dict(name=spec['name'], category=spec['category'], duration_s=spec['duration'], sr=sr,
                seed=seed, null=not spec['events'], description=spec['description'],
                passes=list(passes.values()), events=spec['events'], background=spec['bg'], peak=peak)
    return x.astype(np.float32), tracks, meta


def _compact(d):
    d['t'] = [round(v, 3) for v in d['t']]
    d['f'] = [round(v, 5) for v in d['f']]
    d['amp'] = [float(f'{v:.5g}') for v in d['amp']]
    return d


def make_hard_dataset(outdir='data/synth_hard', sr=48000, only=None, write_audio=True):
    """Write the hard clips (24-bit WAV) and truth.json; return records.

    Truth per clip: every tone of every aircraft on a 0.25 s grid (received frequency of the
    direct path, direct-path peak amplitude, width_hz for FM-jittered tones), harmonic and
    group ids (sidebands: group+50, harmonic of the shaft rate). Null clips have no tracks.
    """
    import soundfile as sf
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    tp = outdir/'truth.json'
    old = {r['name']: r for r in json.loads(tp.read_text())} if (only and tp.exists()) else {}
    records = []
    for i, spec in enumerate(hard_specs()):
        if only and spec['name'] not in only:
            if spec['name'] in old:
                records.append(old[spec['name']])
            continue
        x, tracks, meta = build_hard_clip(spec, sr, seed=7000+37*i)
        path = outdir/(spec['name']+'.wav')
        if write_audio:
            sf.write(path, x, sr, subtype='PCM_24')
        records.append(dict(meta, path=str(path), tracks=[_compact(t.to_dict()) for t in tracks]))
    tp.write_text(json.dumps(records, indent=1, default=float)+'\n')
    return records
