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
