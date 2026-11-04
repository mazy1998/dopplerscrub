#!/usr/bin/env python3
"""Build deterministic, local-only media fixtures: python scripts/make_fixtures.py.

The signal and encoder arguments are seeded/fixed. Encoded bytes can vary with
ffmpeg version. Existing fixtures are overwritten only when this script is run.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from dopplerscrub.synth import flyover, mix

FORMATS = ("mp4", "mov", "mkv", "mp3", "m4a", "flac", "ogg", "wav")
LOCATION = "+40.8764-073.5249/"


def make_fixtures(outdir="data/fixtures"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sr = 48000
    duration = 36
    event = dict(
        kind="prop",
        duration_s=duration,
        sr=sr,
        t_cpa=16.0,
        f0=105.0,
        v=65.0,
        d_cpa=350.0,
        harmonics=3,
        level_db=-25.0,
        seed=721,
    )
    tone, tracks = flyover(**event)
    mono = mix(tone, sr, pink=0.0008, wind=0.0003, hum=0.0002, speech=0.022, seed=722)
    stereo = np.column_stack((mono, 0.85 * mono)).astype(np.float32)
    records = []
    with tempfile.TemporaryDirectory(prefix="dopplerscrub-fixtures-") as tmp:
        source = Path(tmp) / "source.wav"
        sf.write(source, stereo, sr, subtype="PCM_24")
        for ext in FORMATS:
            dest = outdir / f"prop_speech.{ext}"
            cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source)]
            video = ext in ("mp4", "mov", "mkv")
            if video:
                cmd += [
                    "-f",
                    "lavfi",
                    "-i",
                    f"testsrc=size=160x90:rate=10:duration={duration}",
                    "-map",
                    "1:v:0",
                    "-map",
                    "0:a:0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "1",
                ]
            else:
                cmd += ["-map", "0:a:0"]
            codecs = {
                "mp4": ["aac", "-b:a", "192k"],
                "mov": ["aac", "-b:a", "192k"],
                "mkv": ["libopus", "-b:a", "160k"],
                "mp3": ["libmp3lame", "-b:a", "192k"],
                "m4a": ["aac", "-b:a", "192k"],
                "flac": ["flac"],
                "ogg": ["libopus", "-b:a", "160k"],
                "wav": ["pcm_s24le"],
            }
            cmd += [
                "-c:a",
                *codecs[ext],
                "-ar",
                str(sr),
                "-ac",
                "2",
                "-metadata",
                f"location={LOCATION}",
                "-metadata",
                f"com.apple.quicktime.location.ISO6709={LOCATION}",
            ]
            if ext in ("mp4", "mov", "m4a"):
                cmd += ["-movflags", "use_metadata_tags"]
            cmd += ["-t", str(duration), str(dest)]
            subprocess.run(cmd, check=True, capture_output=True)
            records.append(
                dict(
                    path=dest.name,
                    format=ext,
                    has_video=video,
                    duration_s=duration,
                    sr=sr,
                    channels=2,
                )
            )
    manifest = dict(
        seed=721,
        event=event,
        channel_source_gains=[1.0, 0.85],
        tracks=[t.to_dict() for t in tracks],
        fixtures=records,
    )
    (outdir / "fixtures.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", default="data/fixtures")
    make_fixtures(parser.parse_args().outdir)
