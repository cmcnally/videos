#!/usr/bin/env python3
"""Stock the music shelf. Drop royalty-free mp3s (e.g. Pixabay) into a folder and run:

    .venv/bin/python music_intake.py ../reel-web/music

It analyzes each track's tempo (BPM) and energy with librosa — no listening needed —
and writes `library.json` next to the tracks. The web picker reads that file, so adding
music is drop-in. Mood + theme-fit are auto-suggested; edit library.json to taste.
"""
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np


def pretty_name(stem: str) -> str:
    """'alexguz-travel-341602' -> 'Travel (Alexguz)'. Best-effort; not load-bearing."""
    parts = [p for p in re.split(r"[-_]+", stem) if p and not p.isdigit()]
    if not parts:
        return stem
    artist = parts[0].title()
    words = [w.title() for w in parts[1:]] or [artist]
    title = " ".join(words)
    return f"{title} ({artist})" if len(parts) > 1 else title


def mood_and_themes(bpm: float, energy: float):
    """Rough mood label + which visual themes the track suits. Heuristic, editable."""
    if bpm >= 108 and energy >= 0.6:
        return "Upbeat · driving", ["bold", "bright"]
    if bpm <= 92 and energy < 0.55:
        return "Warm · mellow", ["film", "clean"]
    return "Bright · cinematic", ["bright", "cool", "film"]


def analyze(path: Path) -> dict:
    # Cap to 90s for speed; tracks loop in the reel anyway.
    y, sr = librosa.load(str(path), mono=True, duration=90)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo)[0])))
    rms = float(np.sqrt(np.mean(y ** 2)))
    energy = max(0.25, min(1.0, rms * 4))  # same scaling as the browser uploader
    mood, themes = mood_and_themes(bpm, energy)
    return {
        "id": path.stem,
        "file": f"music/{path.name}",
        "nm": pretty_name(path.stem),
        "bpm": bpm,
        "energy": round(energy, 2),
        "mood": mood,
        "forThemes": themes,
    }


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "../reel-web/music").resolve()
    mp3s = sorted(folder.glob("*.mp3"))
    if not mp3s:
        print(f"No .mp3 files in {folder}", file=sys.stderr)
        sys.exit(1)
    library = []
    for p in mp3s:
        print(f"  analyzing {p.name} …", file=sys.stderr)
        try:
            library.append(analyze(p))
        except Exception as e:
            print(f"  ! skipped {p.name}: {e}", file=sys.stderr)
    out = folder / "library.json"
    out.write_text(json.dumps(library, indent=2))
    print(f"\nWrote {len(library)} tracks → {out}")
    for t in library:
        print(f"  {t['bpm']:>3} BPM  energy {t['energy']:.2f}  {t['mood']:<18}  {t['nm']}")


if __name__ == "__main__":
    main()
