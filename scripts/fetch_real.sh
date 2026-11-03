#!/usr/bin/env bash
# Local-only reference audio. Failed sources do not prevent other downloads.
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p data/real
if ! command -v yt-dlp >/dev/null 2>&1; then
  uv tool install yt-dlp || true
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v yt-dlp >/dev/null 2>&1; then
  echo 'yt-dlp unavailable; downloads skipped' >&2
  exit 1
fi
if [ "${1:-}" != --airshow-only ]; then
cat > data/real/colsto_events.json <<'JSON'
{
  "source": "https://youtu.be/4CxVxuMVkMc",
  "time_units": "seconds from recording start",
  "events": [
    {"start_s": 2090, "end_s": 2200, "fmin_hz": 95, "fmax_hz": 200, "harmonics": 2},
    {"start_s": 3020, "end_s": 3130, "fmin_hz": 95, "fmax_hz": 195}
  ]
}
JSON
printf '# Local audio sources\n\nDownloaded audio is for local testing and must not be committed.\n' > data/real/SOURCES.md
fetch() {
  local label="$1" query="$2" short="$3"
  local extra=()
  if [ "$short" = yes ]; then extra=(--match-filter 'duration < 120'); fi
  printf '\n## %s\n\nQuery/source: %s\n' "$label" "$query" >> data/real/SOURCES.md
  if yt-dlp --no-playlist --socket-timeout 15 --retries 1 --extractor-retries 1 \
      "${extra[@]}" -f bestaudio -x --audio-format wav \
      --print-to-file 'after_move:Source: %(webpage_url)s | %(title)s' data/real/SOURCES.md \
      -o "data/real/${label}.%(ext)s" "$query"; then
    printf '\nDownload command completed.\n' >> data/real/SOURCES.md
  else
    printf '\nDownload failed (network, unavailable source, or duration filter).\n' >> data/real/SOURCES.md
  fi
}
fetch colsto 'https://youtu.be/4CxVxuMVkMc' no
fetch prop_flyover 'ytsearch1:propeller plane flyover sound short' yes
fetch prop_pass 'ytsearch1:Cessna flyover sound' yes
fetch f1_pass 'ytsearch1:F1 car pass by doppler' yes
shopt -s nullglob
for file in data/real/*.wav; do
  if ffmpeg -v error -i "$file" -f null -; then
    printf '\nDecoded successfully: %s\n' "$file" >> data/real/SOURCES.md
  else
    printf '\nDECODE FAILED: %s\n' "$file" >> data/real/SOURCES.md
  fi
done

fi

# Curated airshow sources. --airshow-only avoids fetching the baseline set;
# --base-only skips this section. All downloaded streams are AUDIO ONLY.
if [ "${1:-}" != --base-only ]; then
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
"$PYTHON_BIN" - <<'PY'
import concurrent.futures
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

out = Path('data/real/airshow'); out.mkdir(parents=True, exist_ok=True)
# Source windows, seconds in the original video; never more than 90 seconds.
sources = [
 ('p51_mustang', 'V0MAv1CHDy8', 15, 90, 'warbird propeller'),
 ('spitfire_presenter', 'brd44OS0Ueo', 0, 41, 'warbird propeller'),
 ('spitfire_mustang', 'o0Ff5E6Y_m8', 0, 22, 'warbird propeller pair'),
 ('star_wars_canyon', 'xg4G6XIkrw8', 0, 75, 'fighter jet canyon pass'),
 ('rainbow_f18', 'tgEi2xMK-P4', 0, 32, 'F/A-18 fighter jet canyon pass'),
 ('hercules_airshow', 'xbwM-nBKVaI', 150, 225, 'LM-100J / C-130-family turboprop'),
 ('helicopter_fast', '0P-byaAa_Xo', 0, 27, 'helicopter'),
 ('military_helicopters', 'cDIM2IhFnHU', 0, 63, 'military helicopters'),
 ('extra300', 'Zi59sVzp3cM', 0, 75, 'Extra 300 aerobatic propeller'),
 ('p38', 'duJP-wA0FKE', 0, 8, 'P-38 warbird twin propeller'),
]
ytdlp = shutil.which('yt-dlp')
def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True)
def fetch(spec):
    name, vid, start, end, kind = spec
    url = 'https://www.youtube.com/watch?v='+vid
    row = dict(name=name, url=url, source_type=kind, source_start_s=start, source_end_s=end)
    try:
        metadata = json.loads(run([ytdlp, '--no-warnings', '--no-playlist', '--skip-download',
                                   '--dump-single-json', '--socket-timeout', '15', url]).stdout)
        row.update(title=metadata['title'], description=metadata.get('description',''))
        text = (row['title']+'\n'+row['description']).lower()
        # Ignore explicit assurances of natural sound; flag positive music credits.
        text = re.sub(r'no music|without music|music-free', '', text)
        if re.search(r'\bmusic\b|soundtrack|background song|remix|montage', text):
            row['status'] = 'rejected: title/description suggests music or montage'
            return row
        path = out/(name+'.wav')
        if not path.exists():
            with tempfile.TemporaryDirectory(prefix='airshow-audio-') as tmp:
                run([ytdlp, '--no-playlist', '--no-warnings', '--socket-timeout', '15',
                     '--retries', '1', '--extractor-retries', '1', '-f', 'bestaudio',
                     '--download-sections', f'*{start}-{end}', '-x', '--audio-format', 'wav',
                     '-o', str(Path(tmp)/'audio.%(ext)s'), url])
                audio = next(Path(tmp).glob('*.wav'))
                # No -ar: preserve the downloaded stream's native sample rate.
                run(['ffmpeg','-nostdin','-v','error','-y','-i',str(audio),
                     '-t',str(end-start),'-c:a','pcm_s24le',str(path)])
        run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(path),'-f','null','-'])
        probe = json.loads(run(['ffprobe','-v','error','-show_streams','-show_format',
                               '-of','json',str(path)]).stdout)
        row.update(path=str(path), sr=int(probe['streams'][0]['sample_rate']),
                   duration_s=float(probe['format']['duration']), status='downloaded; see annotations.json for visual review decisions')
    except Exception as exc:
        row['status'] = 'failed: '+str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            row['error'] = exc.stderr[-1500:]
    return row
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    rows = list(pool.map(fetch, sources))
(out/'downloads.json').write_text(json.dumps(rows, indent=2)+'\n')
with (out/'SOURCES.md').open('w') as f:
    f.write('# Local airshow audio sources\n\nAudio only; never commit media. Source windows below are in the original video. Annotations use clip-relative times. Native sample rate is preserved.\n\n')
    for r in rows:
        f.write(f"## {r['name']}\n\n- URL: {r['url']}\n- Title: {r.get('title', 'unavailable')}\n- Source window: {r['source_start_s']}–{r['source_end_s']} s\n- Status: {r['status']}\n- Sample rate: {r.get('sr', 'unknown')} Hz\n\n")
if (out/'annotations.json').exists():
    reviewed = json.loads((out/'annotations.json').read_text())
    with (out/'SOURCES.md').open('a') as f:
        f.write('## Manual visual review\n\nTimes below are clip-relative; add the source-window start for original-video times. Screening used descriptions and spectrograms, not a listening-based guarantee.\n\n')
        for clip in reviewed['clips']:
            windows = ', '.join(f"clip {e['start_s']}–{e['end_s']} s / source {clip['source_start_s']+e['start_s']}–{clip['source_start_s']+e['end_s']} s ({e['fmin_hz']}–{e['fmax_hz']} Hz; clean={e['clean_sweep']})" for e in clip['events'])
            f.write(f"- {clip['name']}: {windows}\n")
        for clip in reviewed.get('rejected', []):
            f.write(f"- Pruned {clip['name']}: {clip['url']}; source {clip['source_start_s']}–{clip['source_end_s']} s. {clip['reason']}\n")
print('\n'.join(f"{r['name']}: {r['status']}" for r in rows), flush=True)
# Both views use dopplertrack's exact STFT and whitening implementation.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dopplertrack.audio import compute_spectrogram
Path('docs/img').mkdir(parents=True, exist_ok=True)
for r in rows:
    if 'path' not in r:
        continue
    raw = subprocess.check_output(['ffmpeg','-v','error','-i',r['path'],
                                   '-ac','1','-ar','4000','-f','f32le','-'])
    x = np.frombuffer(raw, dtype='<f4')
    if len(x) < 8192:
        continue
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.subplots_adjust(left=.08, right=.88, bottom=.07, top=.90, hspace=.30)
    for ax, hi in zip(axes, [450, 2000]):
        s = compute_spectrogram(x, 4000, fmin=40, fmax=hi)
        im = ax.imshow(s.S, origin='lower', aspect='auto', interpolation='nearest',
                       extent=[s.t[0]-.125, s.t[-1]+.125, s.f[0], s.f[-1]],
                       vmin=-5, vmax=18, cmap='magma')
        ax.set(xlabel='Clip time (s)', ylabel='Frequency (Hz)', title=f'40–{hi} Hz')
        fig.colorbar(im, ax=ax, label='dB above per-bin median', pad=.02)
    fig.suptitle(r['name']+' — '+r['title'], fontsize=10)
    fig.savefig('docs/img/airshow_'+r['name']+'.png', dpi=120)
    plt.close(fig)
PY
fi
