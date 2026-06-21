#!/usr/bin/env bash
#
# House-style trip reel in one command (Catharine's preset).
#
#   ./trip-reel.sh --videos <dir> --photos <dir> \
#       --title "SOUTHERN PATAGONIA" --subtitle "NOVEMBER 2025" \
#       --cover <hero_photo> --music <track.mp3> [--out file.mp4] [--opener <clip|name>]
#
# The look (locked in): vertical 9:16, ~50s, title cover -> chronological trip,
# a bit of every day, varied magazine collages that fill cleanly (white frames),
# varied transitions, tasteful vibrance, photos scored & best kept, beat-paced music.
#
# Notes:
#   - Scoring NEW video/photos uses the Anthropic API (cheap haiku). Top up credits first.
#   - --photos: a folder of photos (a dump → they get scored & the best per day kept).
#     For already-curated photos, the defaults still work; raise --max-photos to keep more.
#   - --opener (optional): a clip file, or text matching a video name, to feature right
#     after the cover (e.g. an arrival/plane shot).

set -euo pipefail
cd "$(dirname "$0")"

VIDEOS="" PHOTOS="" TITLE="" SUBTITLE="" COVER="" MUSIC="" OUT="output/trip_reel.mp4" OPENER="" MAXPHOTOS="40"
while [ $# -gt 0 ]; do
  case "$1" in
    --videos) VIDEOS="$2"; shift 2;;
    --photos) PHOTOS="$2"; shift 2;;
    --title) TITLE="$2"; shift 2;;
    --subtitle) SUBTITLE="$2"; shift 2;;
    --cover) COVER="$2"; shift 2;;
    --music) MUSIC="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --opener) OPENER="$2"; shift 2;;
    --max-photos) MAXPHOTOS="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done
: "${VIDEOS:?--videos required}" "${PHOTOS:?--photos required}" "${TITLE:?--title required}"
: "${COVER:?--cover required}" "${MUSIC:?--music required}"

source .venv/bin/activate

FEATURE=()
if [ -n "$OPENER" ]; then
  if [ -f "$OPENER" ]; then FEATURE=(--intro "$OPENER"); else FEATURE=(--feature "$OPENER" --feature-at 2 --feature-len 1.2); fi
fi

python make_reel.py "$VIDEOS" \
  --photos "$PHOTOS" --max-photos "$MAXPHOTOS" \
  --coverage --min-score 7 \
  --title "$TITLE" --subtitle "$SUBTITLE" --cover "$COVER" \
  --subject "people laughing, candid joyful moments, great landscapes and scenery, wildlife, the energy and happiness of the trip" \
  --model claude-haiku-4-5 --step 2.0 \
  --music "$MUSIC" \
  "${FEATURE[@]}" \
  --out "$OUT"

open "$OUT"
