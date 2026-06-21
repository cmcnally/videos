# condor-clipper

Batch-triage a folder of videos into condor-only clips. Claude vision looks at
sampled frames, confirms whether a California condor is present, scores how
striking each shot is, and `ffmpeg` cuts the keepers into clips with a ranked
manifest. You curate the survivors.

```
videos/ ──▶ sample frames ──▶ confirm condor + score ──▶ merge ──▶ cut clips ──▶ output/clips/
            (ffmpeg)          (Claude vision)            hits      (ffmpeg)      + manifest.csv
```

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
