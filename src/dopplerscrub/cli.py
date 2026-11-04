"""dopplerscrub command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .types import DetectConfig, RemoveConfig


def _log(msg: str) -> None:
    print(f"[dopplerscrub] {msg}", file=sys.stderr, flush=True)


def _default_out(p: Path) -> Path:
    return p.with_name(f"{p.stem}.scrubbed{p.suffix}")


def cmd_clean(a) -> int:
    from .pipeline import scrub

    dcfg = DetectConfig(fmin=a.fmin, fmax=a.fmax)
    kinds = ("doppler", "stationary") if a.all_tones else ("doppler",)
    rcfg = RemoveConfig(method=a.method, harden=a.harden, remove_kinds=kinds)
    for src in a.inputs:
        src = Path(src)
        dst = Path(a.output) if a.output and len(a.inputs) == 1 else (
            Path(a.output) / src.name if a.output else _default_out(src))
        if not src.exists():
            print(f"error: {src}: no such file", file=sys.stderr)
            return 2
        if dst.resolve() == src.resolve():
            print(f"error: refusing to overwrite the input {src}", file=sys.stderr)
            return 2
        dst.parent.mkdir(parents=True, exist_ok=True)
        log = _log if not a.quiet else (lambda *_: None)
        rep, x, y, sr = scrub(src, dst, dcfg, rcfg, passes=a.passes,
                              strip_metadata=not a.keep_metadata,
                              report_path=dst.with_suffix(dst.suffix + ".json") if a.report else None,
                              log=log, return_audio=True)
        print(f"{src} -> {dst}: {rep.removed} Doppler tone(s) removed "
              f"({len(rep.tracks)} tones seen, {rep.duration_s:.0f}s audio; {'; '.join(rep.notes)})")
        if a.report:
            print(f"  report: {dst.with_suffix(dst.suffix + '.json')}")
        if a.plot:
            from .evaluate import render_before_after
            png = dst.with_suffix(".png")
            log("rendering before/after spectrogram ...")
            from .evaluate import attack_views
            views = attack_views(x, y, sr)
            st = views["orig"].t
            dt = float(st[1] - st[0]) if len(st) > 1 else 0.25
            # pin the time axis to the spectrogram's own extent (otherwise the
            # track overlay's autoscale margin leaves an empty strip at the end)
            render_before_after(x, y, sr, png, tracks=rep.tracks, title=src.name, views=views,
                                window=(float(st[0]) - dt / 2, float(st[-1]) + dt / 2))
            print(f"  spectrogram: {png}")
    return 0


def cmd_detect(a) -> int:
    from . import io
    from .detect import detect_tracks

    x, sr, _ = io.load(a.input)
    tracks = detect_tracks(x, sr, DetectConfig(fmin=a.fmin, fmax=a.fmax))
    for i, t in enumerate(tracks):
        fit = t.fit or {}
        print(f"#{i:02d} {t.kind:10s} score={t.doppler_score:.2f} {t.t0:8.1f}-{t.t1:8.1f}s "
              f"{t.f.max():7.1f}->{t.f.min():7.1f} Hz  h{t.harmonic} g{t.group}  "
              + " ".join(f"{k}={v:.3g}" for k, v in fit.items() if isinstance(v, (int, float))))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="dopplerscrub", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("clean", help="remove Doppler tones from audio/video files")
    c.add_argument("inputs", nargs="+")
    c.add_argument("-o", "--output", help="output file (single input) or directory")
    c.add_argument("--method", choices=["vk", "subtract", "mask"], default=RemoveConfig.method,
                   help="removal method (default: %(default)s, joint Vold-Kalman)")
    c.add_argument("--no-harden", dest="harden", action="store_false",
                   help="skip the notch + noise-fill step (leaves a curve-shaped notch; not recommended)")
    c.add_argument("--all-tones", action="store_true", help="also remove stationary tones (hum etc.)")
    c.add_argument("--passes", type=int, default=2)
    c.add_argument("--fmin", type=float, default=DetectConfig.fmin)
    c.add_argument("--fmax", type=float, default=DetectConfig.fmax)
    c.add_argument("--keep-metadata", action="store_true",
                   help="keep container metadata (default strips it, incl. GPS tags)")
    c.add_argument("--report", action="store_true", help="write <output>.json with detected tracks")
    c.add_argument("--plot", action="store_true", help="write a before/after spectrogram PNG")
    c.add_argument("-q", "--quiet", action="store_true", help="no progress messages")
    c.set_defaults(func=cmd_clean)

    d = sub.add_parser("detect", help="list Doppler and stationary tones without modifying anything")
    d.add_argument("input")
    d.add_argument("--fmin", type=float, default=DetectConfig.fmin)
    d.add_argument("--fmax", type=float, default=DetectConfig.fmax)
    d.set_defaults(func=cmd_detect)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
