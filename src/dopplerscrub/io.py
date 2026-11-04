"""Audio/video I/O via ffmpeg/ffprobe.

- ``probe(path)``: container/stream metadata (ffprobe JSON, summarized).
- ``load(path, sr=None, stream=0)``: decode one audio stream to float32 ``(n, ch)``
  at native sample rate (or ``sr`` if given), keeping every channel.
- ``load_all(path, sr=None)``: decode every audio stream -> list of ``(x, sr)``.
- ``save(src_path, dst_path, x, sr, meta, strip_metadata=True)``: write cleaned
  audio. If the source has video, the video/subtitle (and attached-picture)
  streams are stream-copied untouched and only the audio is replaced,
  re-encoded with the source codec family and similar bitrate where the output
  container allows it. ``x`` can be one array (replaces the first audio stream)
  or a list of arrays (one per audio stream, in order).

Privacy: with ``strip_metadata=True`` all global, per-stream and chapter
metadata is dropped (that includes ``location``,
``com.apple.quicktime.location.ISO6709``, ``creation_time``, device make/model,
etc.), data streams (e.g. GoPro ``gpmd`` GPS telemetry, timecode tracks) are not
copied, and ``-fflags +bitexact`` suppresses the encoder version tag. Only the
harmless per-stream ``language`` and ``handler_name`` tags are carried over
(a handler name that names a device brand, e.g. "GoPro AVC", is dropped too).
``sensitive_tags(meta)`` lists what a strip removes, for reporting.
Audio streams that were not given cleaned audio are DROPPED (with a warning)
rather than copied, because they would still contain the Doppler tones.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"

# source codec_name -> (ffmpeg encoder, extra args)
_ENCODERS: dict[str, tuple[str, list[str]]] = {
    "aac": ("aac", []),
    "mp3": ("libmp3lame", []),
    "opus": ("libopus", []),
    "vorbis": ("vorbis", ["-strict", "-2"]),  # native (experimental); no libvorbis here
    "flac": ("flac", []),
    "alac": ("alac", []),
}
_LOSSY = {"aac", "mp3", "opus", "vorbis"}
_DEFAULT_BITRATE = {"aac": 192_000, "mp3": 192_000, "opus": 128_000, "vorbis": 160_000}

# output extension -> (codecs the container accepts, default codec)
_CONTAINERS: dict[str, tuple[set[str], str]] = {
    ".wav": ({"pcm"}, "pcm_s16le"),
    ".flac": ({"flac"}, "flac"),
    ".mp3": ({"mp3"}, "mp3"),
    ".m4a": ({"aac", "alac"}, "aac"),
    ".aac": ({"aac"}, "aac"),
    ".m4b": ({"aac", "alac"}, "aac"),
    ".opus": ({"opus"}, "opus"),
    ".ogg": ({"vorbis", "opus", "flac"}, "opus"),
    ".oga": ({"vorbis", "opus", "flac"}, "opus"),
    ".mp4": ({"aac", "mp3", "alac", "opus", "flac"}, "aac"),
    ".m4v": ({"aac", "mp3", "alac"}, "aac"),
    ".mov": ({"aac", "mp3", "alac", "pcm"}, "aac"),
    ".mkv": ({"aac", "mp3", "opus", "vorbis", "flac", "alac", "pcm"}, "opus"),
    ".mka": ({"aac", "mp3", "opus", "vorbis", "flac", "alac", "pcm"}, "flac"),
    ".webm": ({"opus", "vorbis"}, "opus"),
    ".avi": ({"mp3", "pcm", "aac"}, "mp3"),
    ".aiff": ({"pcm"}, "pcm_s16be"),
    ".aif": ({"pcm"}, "pcm_s16be"),
    ".caf": ({"pcm", "aac", "alac", "flac", "opus"}, "alac"),
}
_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)
_MP3_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, **kw)
    if p.returncode != 0:
        err = p.stderr.decode(errors="replace")[-3000:]
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{err}")
    return p


def _int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def probe(path) -> dict:
    """Summarize a media file with ffprobe.

    Returns a dict with keys: path, container, format_long_name, duration,
    bit_rate, tags (global), has_video, audio_streams (list), video_streams,
    subtitle_streams, other_streams, chapters (count), and for convenience
    ``audio`` (first audio stream dict or None), ``sr`` and ``channels``.
    Each audio stream dict: index (absolute), codec, profile, bit_rate,
    sample_rate, channels, channel_layout, sample_fmt, bits_per_raw_sample,
    tags, attached_pic.
    """
    path = str(path)
    p = _run(
        [
            FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            path,
        ]
    )
    info = json.loads(p.stdout or b"{}")
    fmt = info.get("format", {})
    audio, video, subs, other = [], [], [], []
    for s in info.get("streams", []):
        ctype = s.get("codec_type")
        attached = bool(s.get("disposition", {}).get("attached_pic", 0))
        d = {
            "index": s.get("index"),
            "codec_type": ctype,
            "codec": s.get("codec_name"),
            "profile": s.get("profile"),
            "bit_rate": _int(s.get("bit_rate")),
            "tags": dict(s.get("tags", {})),
            "attached_pic": attached,
        }
        if ctype == "audio":
            d.update(
                sample_rate=_int(s.get("sample_rate")),
                channels=_int(s.get("channels")),
                channel_layout=s.get("channel_layout"),
                sample_fmt=s.get("sample_fmt"),
                bits_per_raw_sample=_int(s.get("bits_per_raw_sample")),
                bits_per_sample=_int(s.get("bits_per_sample")),
            )
            audio.append(d)
        elif ctype == "video":
            d.update(width=s.get("width"), height=s.get("height"))
            video.append(d)
        elif ctype == "subtitle":
            subs.append(d)
        else:
            other.append(d)
    a0 = audio[0] if audio else None
    return {
        "path": path,
        "container": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
        "duration": _float(fmt.get("duration")),
        "bit_rate": _int(fmt.get("bit_rate")),
        "tags": dict(fmt.get("tags", {})),
        "has_video": any(not v["attached_pic"] for v in video),
        "audio_streams": audio,
        "video_streams": video,
        "subtitle_streams": subs,
        "other_streams": other,
        "chapters": len(info.get("chapters", [])),
        "audio": a0,
        "sr": a0["sample_rate"] if a0 else None,
        "channels": a0["channels"] if a0 else None,
    }


def _decode(path: str, a_idx: int, channels: int, sr: int | None) -> np.ndarray:
    cmd = [
        FFMPEG,
        "-nostdin",
        "-v",
        "error",
        "-i",
        path,
        "-map",
        f"0:a:{a_idx}",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        str(channels),
    ]
    if sr:
        cmd += ["-ar", str(sr)]
    cmd += ["-f", "f32le", "-c:a", "pcm_f32le", "-"]
    p = _run(cmd)
    x = np.frombuffer(p.stdout, dtype="<f4").astype(np.float32, copy=False)
    return x[: len(x) // channels * channels].reshape(-1, channels).copy()


def load(path, sr: int | None = None, stream: int = 0):
    """Decode audio stream ``stream`` (0-based among audio streams).

    Returns ``(x, sr, meta)`` with ``x`` float32 of shape ``(n, ch)`` (always 2-D,
    all channels kept) at the native sample rate unless ``sr`` is given.
    """
    meta = probe(path)
    if not meta["audio_streams"]:
        raise ValueError(f"{path}: no audio stream")
    if len(meta["audio_streams"]) > 1 and stream == 0:
        warnings.warn(
            f"{path}: {len(meta['audio_streams'])} audio streams; load() returns the "
            "first. Use load_all() and pass a list to save() to clean all of them "
            "(otherwise save() drops the extra streams).",
            stacklevel=2,
        )
    a = meta["audio_streams"][stream]
    ch = a["channels"] or 1
    out_sr = int(sr or a["sample_rate"] or 48000)
    x = _decode(str(path), stream, ch, sr)
    meta["sr"], meta["channels"], meta["audio"] = out_sr, ch, a
    return x, out_sr, meta


def load_all(path, sr: int | None = None):
    """Decode every audio stream. Returns ``(list[(x, sr)], meta)``."""
    meta = probe(path)
    out = []
    for i, a in enumerate(meta["audio_streams"]):
        ch = a["channels"] or 1
        out.append((_decode(str(path), i, ch, sr), int(sr or a["sample_rate"] or 48000)))
    return out, meta


def _pcm_codec(src: dict | None, ext: str) -> str:
    be = ext in (".aiff", ".aif")
    c = (src or {}).get("codec") or ""
    if c.startswith("pcm_"):
        if be and c.endswith("le"):
            c = c[:-2] + "be"
        if not be and c.endswith("be"):
            c = c[:-2] + "le"
        return c
    bits = (src or {}).get("bits_per_raw_sample") or 16
    depth = 24 if bits > 16 else 16
    return f"pcm_s{depth}{'be' if be else 'le'}"


def _audio_codec_args(src: dict | None, ext: str, sr: int, out_idx: int) -> list[str]:
    """Encoder args for output audio stream ``out_idx``."""
    allowed, default = _CONTAINERS.get(ext, ({"aac", "mp3", "opus", "flac", "pcm"}, "aac"))
    src_codec = (src or {}).get("codec") or ""
    family = "pcm" if src_codec.startswith("pcm_") else src_codec
    if family not in allowed:
        family = "pcm" if default.startswith("pcm_") else default
    s = f":a:{out_idx}"
    if family == "pcm":
        args = [f"-c{s}", _pcm_codec(src if src_codec.startswith("pcm_") else None, ext)]
    else:
        enc, extra = _ENCODERS[family]
        args = [f"-c{s}", enc] + extra
        if family in _LOSSY:
            br = (src or {}).get("bit_rate") if src_codec == family else None
            br = br or _DEFAULT_BITRATE[family]
            args += [f"-b{s}", str(int(br))]
        if family == "flac" and ((src or {}).get("bits_per_raw_sample") or 16) > 16:
            args += [f"-sample_fmt{s}", "s32"]
        if family == "alac" and ((src or {}).get("bits_per_raw_sample") or 16) > 16:
            args += [f"-sample_fmt{s}", "s32p"]
    if family == "opus" and sr not in _OPUS_RATES:
        args += [f"-ar{s}", "48000"]
    if family == "mp3" and sr not in _MP3_RATES:
        args += [f"-ar{s}", "48000" if sr > 48000 else "44100"]
    if family == "aac" and sr > 96000:
        args += [f"-ar{s}", "96000"]
    return args


KEEP_STREAM_TAGS = ("language", "handler_name")  # harmless; everything else is stripped
_STRUCTURAL = {"major_brand", "minor_version", "compatible_brands", "encoder", "vendor_id"}
_LOCATION_RE = re.compile(r"location|iso6709|gps|xyz|latitude|longitude|altitude", re.I)
_COORD_RE = re.compile(r"[+-]\d{1,3}\.\d+[+-]\d{1,3}\.\d+")
_BRAND_RE = re.compile(
    r"gopro|apple|iphone|ipad|samsung|galaxy|sony|canon|nikon|dji|osmo|"
    r"panasonic|lumix|google|pixel|huawei|xiaomi|oneplus|insta360|garmin|"
    r"fuji|olympus|blackmagic|zoom|tascam",
    re.I,
)


def _harmless(key: str, value) -> bool:
    v = str(value)
    return (
        len(v) <= 64
        and v.isprintable()
        and not _COORD_RE.search(v)
        and not _LOCATION_RE.search(v)
        and not (key == "handler_name" and _BRAND_RE.search(v))
    )


def sensitive_tags(meta: dict) -> dict:
    """What a metadata strip removes from this file: {"location": [...tag names],
    "other": [...tag names], "chapters": n, "data_streams": [...codec names]}.
    Container-structural tags (brands, encoder) and the kept per-stream
    language/handler tags are not listed."""
    loc, other = [], []

    def scan(tags, prefix=""):
        for k, v in (tags or {}).items():
            if k in _STRUCTURAL or (prefix and k in KEEP_STREAM_TAGS and _harmless(k, v)):
                continue
            name = prefix + k
            (loc if (_LOCATION_RE.search(k) or _COORD_RE.search(str(v))) else other).append(name)

    scan(meta.get("tags"))
    for grp in ("video_streams", "audio_streams", "subtitle_streams", "other_streams"):
        for st in meta.get(grp, []):
            scan(st.get("tags"), f"stream{st.get('index')}:")
    return {
        "location": sorted(set(loc)),
        "other": sorted(set(other)),
        "chapters": int(meta.get("chapters") or 0),
        "data_streams": [st.get("codec") or "?" for st in meta.get("other_streams", [])],
    }


def _keep_tag_args(meta: dict, keep_visual: bool, subs_mapped: bool, n_audio: int) -> list[str]:
    """-metadata args re-adding the harmless per-stream tags after -map_metadata -1."""
    args = []
    groups = [("a", meta.get("audio_streams", [])[:n_audio])]
    if keep_visual:
        groups.append(("v", meta.get("video_streams", [])))
        if subs_mapped:
            groups.append(("s", meta.get("subtitle_streams", [])))
    for t, streams in groups:
        for j, st in enumerate(streams):
            for k in KEEP_STREAM_TAGS:
                v = (st.get("tags") or {}).get(k)
                if v is not None and _harmless(k, v):
                    args += [f"-metadata:s:{t}:{j}", f"{k}={v}"]
    return args


def save(src_path, dst_path, x, sr: int, meta: dict | None = None, strip_metadata: bool = True):
    """Write cleaned audio ``x`` to ``dst_path``.

    ``x``: float array ``(n,)``/``(n, ch)``, or a list of such arrays (one per
    source audio stream, in order). If ``src_path`` has video, video/subtitle
    streams are copied bit-exact (``-c:v copy -c:s copy``) and only audio is
    re-encoded. Output format follows ``dst_path``'s extension. Returns the
    probe() of the written file.
    """
    src_path, dst_path = str(src_path), str(dst_path)
    meta = meta if meta is not None and "audio_streams" in meta else probe(src_path)
    xs = list(x) if isinstance(x, (list, tuple)) else [x]
    srs = [sr] * len(xs)
    if xs and isinstance(xs[0], tuple):  # list of (x, sr) from load_all
        xs, srs = [a for a, _ in xs], [int(b) for _, b in xs]
    n_src_audio = len(meta["audio_streams"])
    if n_src_audio > len(xs):
        warnings.warn(
            f"{src_path}: {n_src_audio} audio streams but only {len(xs)} cleaned "
            f"array(s) given; dropping the {n_src_audio - len(xs)} uncleaned audio "
            "stream(s) so they cannot leak the tones.",
            stacklevel=2,
        )
    ext = Path(dst_path).suffix.lower()
    has_video = bool(meta["has_video"])

    keep_visual = has_video or any(v["attached_pic"] for v in meta["video_streams"])
    # input 0 = source when we copy streams from it or keep its metadata
    use_src = keep_visual or not strip_metadata
    off = 1 if use_src else 0
    tmpdir = tempfile.mkdtemp(prefix="dopplerscrub_")
    try:
        cmd = [FFMPEG, "-nostdin", "-v", "error", "-y"]
        if use_src:
            cmd += ["-i", src_path]
        for i, (xi, si) in enumerate(zip(xs, srs)):
            wp = os.path.join(tmpdir, f"a{i}.wav")
            sf.write(wp, np.asarray(xi, dtype=np.float32), int(si), subtype="FLOAT", format="WAV")
            cmd += ["-i", wp]
        subs_mapped = bool(
            keep_visual
            and meta["subtitle_streams"]
            and ext in (".mkv", ".mp4", ".mov", ".m4v", ".webm")
        )
        if keep_visual:
            cmd += ["-map", "0:v?"]
            if subs_mapped:
                cmd += ["-map", "0:s?"]
        for i in range(len(xs)):
            cmd += ["-map", f"{i + off}:a:0"]
        if keep_visual:
            cmd += ["-c:v", "copy"]
            subs = meta["subtitle_streams"]
            if (
                subs
                and ext in (".mp4", ".m4v", ".mov")
                and any(s["codec"] != "mov_text" for s in subs)
            ):
                cmd += ["-c:s", "mov_text"]  # text subs in mp4/mov must be mov_text
            else:
                cmd += ["-c:s", "copy"]
        for i in range(len(xs)):
            src_a = meta["audio_streams"][i] if i < n_src_audio else None
            cmd += _audio_codec_args(src_a, ext, srs[i], i)
        if strip_metadata:
            cmd += [
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-fflags",
                "+bitexact",
                "-flags:a",
                "+bitexact",
            ]
            cmd += _keep_tag_args(meta, keep_visual, subs_mapped, min(len(xs), n_src_audio))
        else:
            cmd += ["-map_metadata", "0"]
            for i in range(min(len(xs), n_src_audio)):
                cmd += [f"-map_metadata:s:a:{i}", f"0:s:a:{i}"]
        cmd.append(dst_path)
        _run(cmd)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return probe(dst_path)
