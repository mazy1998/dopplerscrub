"""Real end-to-end media checks; run with pytest tests/test_pipeline.py -v.

No detector/remover mocks: every encoded fixture passes through scrub().
Fixtures are built in pytest's temporary directory, independent of local data.
"""

from __future__ import annotations
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import numpy as np
import pytest

FORMATS = ("mp4", "mov", "mkv", "mp3", "m4a", "flac", "ogg", "wav")


def _probe(path):
    return json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
        )
    )


def _video_hashes(path):
    # Compare payloads only: muxers may legitimately change packet timestamps.
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-show_entries",
                "packet=data_hash",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    return [p["data_hash"] for p in info["packets"]]


def _location_tags(info):
    scopes = [info.get("format", {}), *info.get("streams", [])]
    return {
        k: v
        for scope in scopes
        for k, v in scope.get("tags", {}).items()
        if any(term in k.lower() for term in ("location", "gps", "iso6709"))
    }


@pytest.fixture(scope="module")
def fixture_bundle(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("media integration requires ffmpeg and ffprobe")
    try:
        from dopplerscrub.pipeline import scrub
        from dopplerscrub.io import load
        from dopplerscrub.evaluate import residual_db
    except ImportError as exc:
        pytest.skip(f"pipeline integration dependencies have not landed: {exc}")
    root = Path(__file__).resolve().parents[1]
    make = runpy.run_path(str(root / "scripts/make_fixtures.py"))["make_fixtures"]
    directory = tmp_path_factory.mktemp("media-fixtures")
    manifest = make(directory)
    return directory, manifest, scrub, load, residual_db


@pytest.mark.slow
@pytest.mark.parametrize("extension", FORMATS)
def test_scrub_media_format(extension, fixture_bundle, tmp_path):
    from dopplerscrub.types import Track

    directory, manifest, scrub, load, residual_db = fixture_bundle
    source = directory / f"prop_speech.{extension}"
    dest = tmp_path / f"clean.{extension}"
    before_info = _probe(source)
    # Prove the GPS-removal check is non-vacuous for supported Apple containers.
    if extension in ("mp4", "mov", "m4a"):
        assert _location_tags(before_info), "fixture must carry GPS metadata"
    report = scrub(source, dest)
    assert dest.exists() and report.removed > 0
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-i", str(dest), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    after_info = _probe(dest)
    errors = []
    drift = abs(float(before_info["format"]["duration"]) - float(after_info["format"]["duration"]))
    if drift > 0.050:
        errors.append(f"duration drift {drift:.6f} s exceeds 50 ms")
    if _location_tags(after_info):
        errors.append(f"GPS metadata remains: {_location_tags(after_info)}")
    if extension in ("mp4", "mov", "mkv"):
        hashes = _video_hashes(source)
        assert hashes, "video fixture must have packets"
        if hashes != _video_hashes(dest):
            errors.append("video packet payloads changed")
    a, sr, _ = load(source, sr=16000)
    b, out_sr, _ = load(dest, sr=16000)
    assert out_sr == sr
    assert np.isfinite(b).all()
    tracks = [Track.from_dict(d) for d in manifest["tracks"]]
    reduction = residual_db(a, b, sr, tracks)
    if not np.isfinite(reduction) or reduction > -15.0:
        errors.append(f"truth-track tone energy changed {reduction:.2f} dB; require <= -15 dB")
    assert not errors, "\n".join(errors)
