import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from dopplerscrub import io

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

LOC = "+37.7749-122.4194+010.000/"


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def all_tags(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                          "-show_streams", "-show_chapters", str(path)],
                         capture_output=True, check=True).stdout
    info = json.loads(out)
    tags = dict(info["format"].get("tags", {}))
    for s in info["streams"]:
        for k, v in s.get("tags", {}).items():
            tags[f"s{s['index']}:{k}"] = v
    return tags, info


def stream_md5(path, sel):
    out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", sel,
                          "-c", "copy", "-f", "md5", "-"], capture_output=True, check=True)
    return out.stdout.strip()


@pytest.fixture
def tone_wav(tmp_path):
    sr = 44100
    t = np.arange(sr * 2) / sr
    x = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 330 * t)], 1)
    p = tmp_path / "tone.wav"
    sf.write(p, x.astype(np.float32), sr, subtype="PCM_24")
    return p, x, sr


@pytest.fixture
def video_mp4(tmp_path, tone_wav):
    """H.264 + AAC (stereo 44.1k) mp4 with location tags, creation time,
    title and a chapter."""
    wav, _, _ = tone_wav
    chap = tmp_path / "chap.txt"
    chap.write_text(";FFMETADATA1\ntitle=Secret trip\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\n"
                    "END=1000\ntitle=Home\n")
    p = tmp_path / "in.mp4"
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2", "-i", str(wav),
       "-i", str(chap), "-map", "0:v", "-map", "1:a", "-map_metadata", "2", "-map_chapters", "2",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
       "-metadata", f"location={LOC}", "-metadata", f"location-eng={LOC}",
       "-metadata", "creation_time=2026-01-02T03:04:05Z", "-metadata", "make=Apple",
       "-metadata:s:a:0", "language=eng", str(p))
    return p


@pytest.fixture
def video_mov_iso6709(tmp_path, tone_wav):
    wav, _, _ = tone_wav
    p = tmp_path / "in.mov"
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2", "-i", str(wav),
       "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
       "-movflags", "use_metadata_tags",
       "-metadata", f"com.apple.quicktime.location.ISO6709={LOC}",
       "-metadata", "com.apple.quicktime.make=Apple",
       "-metadata", "com.apple.quicktime.model=iPhone 15", str(p))
    return p


def test_probe_and_load_wav(tone_wav):
    p, x, sr = tone_wav
    meta = io.probe(p)
    assert meta["sr"] == sr and meta["channels"] == 2 and not meta["has_video"]
    assert meta["audio"]["codec"] == "pcm_s24le"
    y, sr2, meta2 = io.load(p)
    assert sr2 == sr and y.dtype == np.float32 and y.shape == x.shape
    np.testing.assert_allclose(y, x, atol=1e-6)
    y3, sr3, _ = io.load(p, sr=16000)
    assert sr3 == 16000 and y3.shape[1] == 2 and abs(len(y3) - 32000) < 64


def test_mp4_location_and_metadata_stripped(tmp_path, video_mp4):
    tags_in, info_in = all_tags(video_mp4)
    assert any("37.7749" in v for v in tags_in.values()), tags_in   # fixture really has it
    assert info_in.get("chapters")
    x, sr, meta = io.load(video_mp4)
    assert meta["has_video"] and x.shape[1] == 2 and sr == 44100
    out = tmp_path / "out.mp4"
    io.save(video_mp4, out, x * 0.5, sr, meta)
    tags, info = all_tags(out)
    blob = json.dumps(tags).lower()
    assert "location" not in blob and LOC.lower() not in blob and "37.77" not in blob
    assert "creation_time" not in blob and "apple" not in blob and "secret" not in blob
    assert not info.get("chapters")
    # also check the raw bytes (in case a tag ffprobe doesn't surface survived)
    raw = out.read_bytes()
    assert b"37.7749" not in raw and b"\xa9xyz" not in raw
    # video stream is bit-identical, audio replaced and re-encoded as AAC
    assert stream_md5(out, "0:v") == stream_md5(video_mp4, "0:v")
    m2 = io.probe(out)
    assert m2["has_video"] and m2["audio"]["codec"] == "aac" and m2["audio"]["channels"] == 2
    assert m2["audio"]["sample_rate"] == 44100
    assert 64_000 < (m2["audio"]["bit_rate"] or 128_000) < 200_000
    y, _, _ = io.load(out)
    n = min(len(y), len(x)) - 4096
    # AAC round trip of the halved signal (allow encoder delay alignment slack)
    r = np.sqrt(np.mean(y[2048:n] ** 2)) / np.sqrt(np.mean((0.5 * x[2048:n]) ** 2))
    assert 0.9 < r < 1.1


def test_mov_iso6709_stripped(tmp_path, video_mov_iso6709):
    tags_in, _ = all_tags(video_mov_iso6709)
    assert "com.apple.quicktime.location.ISO6709" in tags_in
    x, sr, meta = io.load(video_mov_iso6709)
    out = tmp_path / "out.mov"
    io.save(video_mov_iso6709, out, x, sr, meta)
    tags, _ = all_tags(out)
    blob = json.dumps(tags).lower()
    assert "iso6709" not in blob and "location" not in blob and "iphone" not in blob
    assert b"37.7749" not in out.read_bytes()
    assert stream_md5(out, "0:v") == stream_md5(video_mov_iso6709, "0:v")


def test_keep_metadata_when_asked(tmp_path, video_mp4):
    x, sr, meta = io.load(video_mp4)
    out = tmp_path / "keep.mp4"
    io.save(video_mp4, out, x, sr, meta, strip_metadata=False)
    tags, _ = all_tags(out)
    assert any("37.7749" in v for v in tags.values())


@pytest.mark.parametrize("ext,codec", [(".wav", "pcm_s24le"), (".flac", "flac"),
                                       (".mp3", "mp3"), (".m4a", "aac"), (".opus", "opus"),
                                       (".ogg", "opus"), (".aiff", "pcm_s24be")])
def test_audio_only_formats(tmp_path, tone_wav, ext, codec):
    p, x, sr = tone_wav
    meta = io.probe(p)
    out = tmp_path / f"out{ext}"
    io.save(p, out, x, sr, meta)
    m = io.probe(out)
    assert m["audio"]["codec"] == codec and m["channels"] == 2 and not m["has_video"]
    y, sr2, _ = io.load(out)
    assert y.shape[1] == 2
    if ext in (".wav", ".flac", ".aiff"):
        assert sr2 == sr
        np.testing.assert_allclose(y[: len(x)], x, atol=1e-5)


def test_codec_family_follows_source(tmp_path, tone_wav):
    p, x, sr = tone_wav
    src = tmp_path / "src.mp3"
    ff("-i", str(p), "-c:a", "libmp3lame", "-b:a", "96k", str(src))
    xs, sr2, meta = io.load(src)
    out = tmp_path / "out.mkv"           # mkv accepts mp3 -> keep the family
    io.save(src, out, xs, sr2, meta)
    m = io.probe(out)
    assert m["audio"]["codec"] == "mp3"
    assert 80_000 <= (m["audio"]["bit_rate"] or m["bit_rate"]) <= 130_000


def test_mono_input_vector(tmp_path, tone_wav):
    p, x, sr = tone_wav
    mono = tmp_path / "mono.wav"
    sf.write(mono, x[:, 0].astype(np.float32), sr)
    y, _, meta = io.load(mono)
    assert y.shape == (len(x), 1)
    out = tmp_path / "mono_out.flac"
    io.save(mono, out, y[:, 0], sr, meta)   # 1-D accepted
    assert io.probe(out)["channels"] == 1


def test_multiple_audio_streams_and_subtitles(tmp_path, tone_wav):
    wav, x, sr = tone_wav
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    src = tmp_path / "multi.mkv"
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=2", "-i", str(wav),
       "-f", "lavfi", "-i", "sine=f=500:d=2:sample_rate=48000", "-i", str(srt),
       "-map", "0:v", "-map", "1:a", "-map", "2:a", "-map", "3:s",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a:0", "aac", "-c:a:1", "libopus",
       "-c:s", "srt", "-metadata", f"location={LOC}", str(src))
    meta = io.probe(src)
    assert len(meta["audio_streams"]) == 2 and meta["subtitle_streams"]
    streams, meta = io.load_all(src)
    assert [s[1] for s in streams] == [44100, 48000]
    # all streams cleaned -> both kept, codecs per stream
    out = tmp_path / "out.mkv"
    io.save(src, out, streams, sr, meta)
    m = io.probe(out)
    assert [a["codec"] for a in m["audio_streams"]] == ["aac", "opus"]
    assert m["subtitle_streams"] and stream_md5(out, "0:v") == stream_md5(src, "0:v")
    assert "37.77" not in json.dumps(all_tags(out)[0])
    # only the first stream cleaned -> warn and drop the uncleaned one
    with pytest.warns(UserWarning):
        x0, sr0, meta0 = io.load(src)
    with pytest.warns(UserWarning, match="dropping"):
        io.save(src, tmp_path / "one.mkv", x0, sr0, meta0)
    assert len(io.probe(tmp_path / "one.mkv")["audio_streams"]) == 1
    # mp4 output: srt subtitles converted to mov_text
    io.save(src, tmp_path / "out.mp4", streams, sr, meta)
    m4 = io.probe(tmp_path / "out.mp4")
    assert m4["subtitle_streams"][0]["codec"] == "mov_text"
    assert [a["codec"] for a in m4["audio_streams"]] == ["aac", "opus"]


def test_harmless_stream_tags_kept_everything_else_stripped(tmp_path, video_mp4):
    """language/handler_name survive the strip; a handler naming a device brand
    does not; location/creation_time/make never do."""
    src = tmp_path / "tags.mov"
    ff("-i", str(video_mp4), "-map", "0:v", "-map", "0:a", "-c", "copy",
       "-metadata:s:v:0", "handler_name=GoPro AVC", "-metadata:s:v:0", "language=fra",
       "-metadata:s:a:0", "handler_name=Core Media Audio", "-metadata:s:a:0", "language=eng",
       "-metadata", f"location={LOC}", "-metadata", "make=GoPro", str(src))
    x, sr, meta = io.load(src)
    out = tmp_path / "out.mov"
    io.save(src, out, x, sr, meta)
    m = io.probe(out)
    v, a = m["video_streams"][0]["tags"], m["audio_streams"][0]["tags"]
    assert v.get("language") == "fra" and "gopro" not in v.get("handler_name", "").lower()
    assert a.get("language") == "eng" and a.get("handler_name") == "Core Media Audio"
    blob = json.dumps(all_tags(out)[0]).lower()
    assert "37.77" not in blob and "creation_time" not in blob and "gopro" not in blob
    assert b"37.7749" not in out.read_bytes()


def test_sensitive_tags_report(video_mp4, tone_wav):
    st = io.sensitive_tags(io.probe(video_mp4))
    assert "location" in st["location"] and "location-eng" in st["location"]
    assert "creation_time" in st["other"] and "title" in st["other"]
    assert st["chapters"] == 1
    assert not any(k.endswith(":language") for k in st["other"])      # kept, not reported
    wav = io.sensitive_tags(io.probe(tone_wav[0]))
    assert wav == {"location": [], "other": [], "chapters": 0, "data_streams": []}
