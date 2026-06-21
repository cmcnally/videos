# condor-clipper

Batch-triage a folder of videos into condor-only clips. Claude vision looks at
sampled frames, confirms whether a California condor is present, scores how
striking each shot is, and `ffmpeg` cuts the keepers into clips with a ranked
manifest. You curate the survivors.

```
videos/ ──▶ sample frames ──▶ confirm condor + score ──▶ merge ──▶ cut clips ──▶ output/clips/
            (ffmpeg)          (Claude vision)            hits      (ffmpeg)      + manifest.csv
```

## Quick start (one command)

Raw videos in → finished, graded, follow-the-bird reel out (~2-3 min, no clicking):

```bash
./reel.sh /path/to/folder_of_videos
# mix in still photos (Ken Burns pan/zoom), punchier look:
./reel.sh ~/Downloads/trip --photos ~/Downloads/trip_photos --pop 1.4
```

It runs the two steps below (clip → reel) and opens `output/reel.mp4` when done.
First run does the Claude analysis; re-runs reuse cached tracking and are instant.

## One-time setup

```bash
# 1. System tool (you don't have ffmpeg yet)
brew install ffmpeg

# 2. Python deps (system Python 3.9 is fine; a venv keeps it isolated)
cd condor-clipper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Anthropic API key — get one at https://console.anthropic.com
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# A whole folder of videos
python condor_clipper.py /path/to/videos --out ./output

# A single file, sampling every 1.5s, padding clips by 3s
python condor_clipper.py condor_footage.mp4 --interval 1.5 --pad 3

# See what it would find without cutting anything (cheap-ish dry run)
python condor_clipper.py /path/to/videos --dry-run --keep-frames
```

Output:
- `output/clips/*.mp4` — one clip per condor event
- `output/manifest.csv` — every clip with `max_interesting`, `best_species`,
  `best_confidence`, timecodes, and Claude's one-line reason. **Sort by
  `max_interesting` to surface your best shots.**

## Tuning

| Flag | Default | What it does |
|------|---------|--------------|
| `--interval` | `2.0` | Seconds between sampled frames. Lower = finer (catches brief fly-throughs) but more API calls. |
| `--pad` | `2.0` | Seconds added before/after each clip. |
| `--gap` | `3.0` | Hits closer than this merge into one clip. |
| `--min-confidence` | `0.6` | Raise toward `0.8` to cut false positives (vultures/eagles); lower to catch more. |
| `--min-interesting` | `1` | Raise to `5+` to keep only striking shots. |
| `--model` | `claude-opus-4-8` | Most accurate. `--model claude-haiku-4-5` is ~5x cheaper for high volume. |
| `--workers` | `8` | Parallel API calls. |
| `--copy` | off | Fast stream-copy cutting (clips may start on a keyframe). Default re-encodes for accuracy. |
| `--max-frames` | `0` | Cap frames per video — useful for a cheap test pass. |
| `--keep-frames` | off | Keep sampled JPEGs for inspection/debugging. |
| `--dry-run` | off | Classify + report, write no clips. |

## Make a highlight reel

After clipping, stitch the keepers into one marketing reel with a Claude-tracked
zoom that follows the condor (so a tiny bird in a wide shot reads on screen):

```bash
python make_reel.py                          # vertical 9:16, best-first, cross-faded
python make_reel.py --zoom 1.0               # no punch-in (full native resolution)
python make_reel.py --music ~/song.mp3       # lay a music bed under it
python make_reel.py --canvas 1920x1080       # landscape instead
```

Output: `output/reel.mp4` — vertical **1080×1920** by default, clips ordered by
interest, cross-faded with fade in/out to black.

Tracking is **cached** in `output/_cache`, so after the first run you can re-edit
framing, zoom, transitions, or music instantly with **no API cost**. Use
`--retrack` to force fresh tracking.

Flags: `--zoom` punch-in (1.0 = full frame, higher = tighter but softer),
`--canvas WxH`, `--transition` cross-fade seconds, `--music`, `--step` tracking
sample rate, `--out`.

## Cost note

Cost scales with frames classified = total video seconds ÷ `--interval`. To
estimate before a big run: do a `--dry-run --max-frames 50` pass first, or drop
to `--model claude-haiku-4-5`. A cheap pre-filter (a local YOLO "is there a
bird" pass) could cut the frame count before Claude ever sees them — not built
in yet, but the natural next optimization if your library gets large.

## Trip reels — house style (reusable workflow)

The locked-in look for trip recaps (Catharine's preset), now the defaults:

- **Vertical 9:16, ~50s**, breathing pace (`--photo-len 1.6`, `--transition 0.35`)
- **Title cover slide** first (`--title`/`--subtitle` over a `--cover` hero photo) — also the thumbnail, so no black preview
- **Chronological, a bit of every day** (`--coverage`); photos limited to the trip window
- **Best moments only**: video scored, weak clips dropped (`--min-score 7`); photos scored & best-per-day kept (`--max-photos`)
- **Varied magazine collages** (1/2/3/4-up, orientation-matched so they fill cleanly) with **white frames** (`--gutter 14`), each **sliding/fading in**
- **Varied transitions** (slides/wipes/splits/dissolves), reshuffle with `--seed`
- **Tasteful vibrance** (natural, not oversaturated)
- **Upbeat music**, beat-paced cuts. Pick a track by mood/BPM; analyze a file's tempo with librosa if unsure (Catharine is deaf — verify audio technically, never by ear)

### One command per trip

```bash
./trip-reel.sh --videos ~/trip/videos --photos ~/trip/photos \
  --title "SOUTHERN PATAGONIA" --subtitle "NOVEMBER 2025" \
  --cover ~/trip/photos/fitzroy.JPG --music music/track.mp3 \
  --opener arrival_clip.mov            # optional: feature an arrival shot after the cover
```

Tips:
- **Curated photos** (e.g. from a Blurb book): export the placed photos with capture dates in the
  filename (`YYYY-MM-DDTHH-MM-SS_*.jpg`) so chronological order + the trip-window filter work.
- **Screenshots sneak in** (video stills with a playhead/timestamp): spot them by odd pixel size
  (`ffprobe`), and crop the UI off before using.
- Re-edits are instant & free once scored (cached in `output/_cache`); change `--seed`, `--photo-len`,
  `--title`, `--music`, etc. without re-scoring.
