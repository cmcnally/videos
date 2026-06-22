#!/usr/bin/env bash
#
# One-time setup. Run once after cloning the repo:  ./setup.sh
# Installs system tools + Python deps and scaffolds your API key file.
# (macOS. Needs Homebrew: https://brew.sh)

set -euo pipefail
cd "$(dirname "$0")"

echo "1/4  System tools (ffmpeg, librsvg)…"
if ! command -v brew >/dev/null 2>&1; then
  echo "    ! Homebrew not found. Install it from https://brew.sh then re-run."; exit 1
fi
brew list ffmpeg  >/dev/null 2>&1 || brew install ffmpeg
brew list librsvg >/dev/null 2>&1 || brew install librsvg

echo "2/4  Python virtual env + libraries…"
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

echo "3/4  API key…"
[ -f .env ] || cp .env.example .env
if grep -q 'sk-ant-api' .env 2>/dev/null; then
  echo "    ✅ API key already set in .env"
else
  echo "    -> Add YOUR OWN Anthropic API key to:  $(pwd)/.env"
  echo "       Get one (with a little billing credit) at https://console.anthropic.com"
  echo "       Never use someone else's key — it bills them."
fi

echo "4/4  Done!"
echo
echo "Make a reel (see README for details):"
echo "  ./trip-reel.sh --album \"My Trip\" --videos ~/trip/videos \\"
echo "      --title \"MY TRIP\" --subtitle \"MONTH 2026\" --cover ~/trip/cover.jpg --music music/track.mp3"
