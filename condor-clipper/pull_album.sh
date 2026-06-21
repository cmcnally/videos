#!/usr/bin/env bash
#
# Export a Photos.app album's originals to a folder, renamed with capture dates
# (YYYY-MM-DDTHH-MM-SS_*) so chronological ordering + the trip-window filter work.
#
#   ./pull_album.sh "Album Name" /dest/folder
#
# Note: with iCloud "Optimize Mac Storage", originals download first — a big album
# can take several minutes on the first export. Requires Photos automation access
# (you'll get a one-time approval prompt).

set -euo pipefail
ALBUM="${1:?album name required}"
DEST="${2:?dest folder required}"
mkdir -p "$DEST"
find "$DEST" -maxdepth 1 -type f -delete 2>/dev/null || true

echo "Exporting album '$ALBUM' from Photos (this can take a while if it downloads from iCloud)…"
osascript - "$ALBUM" "$DEST" <<'OSA'
on run argv
  set albumName to item 1 of argv
  set destPath to item 2 of argv
  tell application "Photos"
    set ml to media items of album albumName
    with timeout of 3600 seconds
      export ml to (POSIX file destPath) with using originals
    end timeout
  end tell
end run
OSA

# Prefix each file with its capture date for chronological order.
shopt -s nullglob
for f in "$DEST"/*; do
  [ -f "$f" ] || continue
  dt=$(mdls -raw -name kMDItemContentCreationDate "$f" 2>/dev/null)
  if [ -n "$dt" ] && [ "$dt" != "(null)" ]; then
    pre=$(printf '%s' "$dt" | cut -c1-19 | tr ' :' 'T-')
    mv -f "$f" "$(dirname "$f")/${pre}_$(basename "$f")"
  fi
done
echo "Pulled $(ls "$DEST" | wc -l | tr -d ' ') photo(s) from album '$ALBUM' into $DEST"
