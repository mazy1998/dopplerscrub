#!/usr/bin/env bash
# Download the public YouTube clips used for the README figures and the held-out test.
# Audio only, saved to data/real/ (gitignored; we do not redistribute the audio).
# Needs yt-dlp and ffmpeg. Then: uv run python scripts/make_figures.py
#
# YouTube may serve a different encode than the one we tested, so your WAVs can differ
# slightly from ours (the expected SHA-256 values are in docs/test_set.md).
set -uo pipefail
cd "$(dirname "$0")/.."
command -v yt-dlp >/dev/null || { echo "install yt-dlp first (e.g. uv tool install yt-dlp)" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "install ffmpeg first" >&2; exit 1; }

# dir name url start_s end_s
CLIPS="
heldout2 kingair_b350         https://www.youtube.com/watch?v=BzUlY8YzarI 0   21.5
heldout2 c172_takeoff_talking https://www.youtube.com/watch?v=Ixo3n4ZHqc8 1.2 58.4
heldout2 c150_lowpass         https://www.youtube.com/watch?v=ncgTszJDc4I 0   82.8
heldout2 baron_flyby          https://www.youtube.com/watch?v=2XEm7aMBfGc 0   31.3
heldout2 pc12_lowpass         https://www.youtube.com/watch?v=x1-21YrfPHA 3.0 24.5
heldout2 cub_lowpass          https://www.youtube.com/watch?v=BBz6tzSoCnI 0   70.7
props_dev c182p_overhead      https://www.youtube.com/watch?v=19l2lZ7fZO8 0   58
props_dev caravan_flyby       https://www.youtube.com/watch?v=rdku7MD_R6Y 0   60
props_dev seneca_v_flyby      https://www.youtube.com/watch?v=QgmD2ny3PNk 0   30
props_dev pa28_cherokee_flyby https://www.youtube.com/watch?v=ba2Tucnws8I 5   62
"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "$CLIPS" | while read -r dir name url t0 t1; do
  [ -z "${dir:-}" ] && continue
  out="data/real/$dir/$name.wav"
  if [ -f "$out" ]; then echo "have $out"; continue; fi
  mkdir -p "data/real/$dir"
  if yt-dlp -q --no-playlist -f bestaudio -o "$tmp/$name.%(ext)s" "$url"; then
    ffmpeg -v error -y -i "$tmp/$name".* -ss "$t0" -to "$t1" -c:a pcm_s24le "$out" \
      && echo "wrote $out" || echo "ffmpeg failed: $name" >&2
  else
    echo "download failed (video removed or region-locked?): $name $url" >&2
  fi
done
