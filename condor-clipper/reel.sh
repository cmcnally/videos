#!/usr/bin/env bash
#
# One command: a folder of raw videos -> a finished condor highlight reel.
#
#   ./reel.sh /path/to/folder_of_videos
#   ./reel.sh ~/Downloads/new_condors --pop 1.4 --spotlight 0.8
#
# Step 1 finds & trims the condor moments (Claude vision).
# Step 2 builds the graded, follow-the-bird reel (cached after first run).
# Any extra args after the folder are passed to the reel builder.

set -euo pipefail
cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
  echo "Usage: ./reel.sh /path/to/folder_of_videos [--pop N] [--spotlight N] [--zoom N] ..."
  exit 1
fi

INPUT="$1"; shift

source .venv/bin/activate

echo "==> [1/2] Finding & trimming condor moments in: $INPUT"
python condor_clipper.py "$INPUT" --out ./output

echo "==> [2/2] Building graded, follow-the-bird reel"
python make_reel.py "$@"

echo "==> Done. Opening reel."
open output/reel.mp4
