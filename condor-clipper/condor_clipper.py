#!/usr/bin/env python3
"""
condor_clipper.py — batch-triage a folder of videos into condor-only clips.

Pipeline:
    videos/ -> sample frames (ffmpeg) -> classify each frame (Claude vision)
            -> merge positive timestamps -> cut clips (ffmpeg) -> clips/ + manifest.csv

You curate the survivors. Claude does the finding and the species/interest scoring.

See README.md for setup. Requires: ffmpeg, ffprobe, the `anthropic` package,
and ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import anthropic

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts", ".m2ts"}

# Per the project's model guidance, default to the latest Opus. For high-volume
# triage, claude-haiku-4-5 is ~5x cheaper and usually plenty for "is this a
# condor" — switch with --model claude-haiku-4-5.
DEFAULT_MODEL = "claude-opus-4-8"

# Structured-output schema: forces Claude to return exactly these fields.
FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "is_condor": {
            "type": "boolean",
            "description": "True if any condor (Andean or California) is clearly present.",
        },
        "species": {
            "type": "string",
            "description": "Which condor, e.g. 'Andean condor' or 'California condor'; "
                           "or the bird's species; or 'none' if no bird.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0 confidence that the bird is a condor (any species).",
        },
        "interesting": {
            "type": "integer",
            "description": "1-10 how visually striking/usable this shot is (close, in flight, sharp = high).",
        },
        "reason": {
            "type": "string",
            "description": "One short phrase explaining the call.",
        },
    },
    "required": ["is_condor", "species", "confidence", "interesting", "reason"],
    "additionalProperties": False,
}

PROMPT = (
    "You are reviewing a single frame from wildlife footage to find condors.\n"
    "Condors are very large soaring vultures with broad fingered wings (~3m span):\n"
    "- Andean condor: black body, white upperwing panels, males have a white neck ruff and "
    "head comb; often shown over Andean/Patagonian mountains.\n"
    "- California condor: black with a white triangular patch UNDER each wing, bald pink/orange "
    "head, often numbered wing (patagial) tags.\n"
    "Distinguish condors from turkey vultures (smaller, silvery trailing wing edge), "
    "golden/bald eagles, and ravens.\n"
    "Set is_condor true for EITHER condor species and name which in 'species'. Set confidence "
    "to how sure you are it is a condor (any species). Be conservative: don't flag just any "
    "large bird as a condor."
)


@dataclass
class FrameResult:
    index: int
    timestamp: float
    is_condor: bool
    species: str
    confidence: float
    interesting: int
    reason: str


def load_env_file() -> None:
    """Load KEY=VALUE lines from a local .env (next to this script) into the
    environment, without overriding anything already set. No dependency needed."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        sys.exit(
            f"error: missing required tool(s): {', '.join(missing)}\n"
            "Install with:  brew install ffmpeg"
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "error: ANTHROPIC_API_KEY is not set.\n"
            "Get a key at https://console.anthropic.com, then either:\n"
            "  - put it in condor-clipper/.env  ->  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  - or export it:                   ->  export ANTHROPIC_API_KEY=sk-ant-..."
        )


def probe_duration(video: Path) -> float:
    cp = run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ]
    )
    try:
        return float(cp.stdout.strip())
    except ValueError:
        return 0.0


def extract_frames(video: Path, frames_dir: Path, interval: float, max_width: int) -> list[Path]:
    """Single-pass sampling: one frame every `interval` seconds, scaled down."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps=1/{interval},scale='min({max_width},iw)':-2"
    cp = run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vf", vf, "-q:v", "3", str(frames_dir / "f_%06d.jpg"),
        ]
    )
    if cp.returncode != 0:
        print(f"  ! ffmpeg sampling failed: {cp.stderr.strip()[:200]}", file=sys.stderr)
        return []
    return sorted(frames_dir.glob("f_*.jpg"))


def classify_frame(client: anthropic.Anthropic, model: str, frame: Path, index: int,
                   interval: float) -> FrameResult | None:
    b64 = base64.standard_b64encode(frame.read_bytes()).decode("utf-8")
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }],
            output_config={"format": {"type": "json_schema", "schema": FRAME_SCHEMA}},
        )
    except anthropic.APIError as e:
        print(f"  ! API error on frame {index}: {e}", file=sys.stderr)
        return None

    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return FrameResult(
        index=index,
        timestamp=(index - 1) * interval,  # f_000001 -> t=0
        is_condor=bool(data.get("is_condor")),
        species=str(data.get("species", "")),
        confidence=float(data.get("confidence", 0.0)),
        interesting=int(data.get("interesting", 0)),
        reason=str(data.get("reason", "")),
    )


def merge_ranges(hits: list[FrameResult], interval: float, gap: float, pad: float,
                 duration: float) -> list[dict]:
    """Merge nearby positive frames into padded clip ranges."""
    if not hits:
        return []
    hits = sorted(hits, key=lambda h: h.timestamp)
    ranges: list[dict] = []
    cur = [hits[0]]
    for h in hits[1:]:
        if h.timestamp - cur[-1].timestamp <= interval + gap:
            cur.append(h)
        else:
            ranges.append(_finalize(cur, pad, duration))
            cur = [h]
    ranges.append(_finalize(cur, pad, duration))
    return ranges


def _finalize(group: list[FrameResult], pad: float, duration: float) -> dict:
    start = max(0.0, group[0].timestamp - pad)
    end = group[-1].timestamp + pad
    if duration:
        end = min(end, duration)
    best = max(group, key=lambda h: (h.interesting, h.confidence))
    return {
        "start": start,
        "end": end,
        "num_frames": len(group),
        "max_interesting": best.interesting,
        "best_species": best.species,
        "best_confidence": round(best.confidence, 2),
        "reason": best.reason,
    }


def cut_clip(video: Path, out: Path, start: float, end: float, copy: bool) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}",
               "-i", str(video), "-to", f"{end - start:.2f}", "-c", "copy", str(out)]
    else:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.2f}",
               "-i", str(video), "-to", f"{end - start:.2f}",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-c:a", "aac", str(out)]
    return run(cmd).returncode == 0


def gather_videos(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    return sorted(p for p in inp.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def process_video(client, args, video: Path, manifest_writer, frames_root: Path) -> int:
    print(f"\n== {video.name}")
    duration = probe_duration(video)
    frames_dir = frames_root / video.stem
    frames = extract_frames(video, frames_dir, args.interval, args.max_width)
    if not frames:
        print("  (no frames extracted)")
        return 0
    if args.max_frames and len(frames) > args.max_frames:
        print(f"  capping {len(frames)} -> {args.max_frames} frames (--max-frames)")
        frames = frames[: args.max_frames]
    print(f"  {len(frames)} frames @ {args.interval}s; classifying with {args.model} ...")

    results: list[FrameResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(classify_frame, client, args.model, f, i + 1, args.interval): i
            for i, f in enumerate(frames)
        }
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r:
                results.append(r)
            if done % 25 == 0 or done == len(frames):
                print(f"    {done}/{len(frames)} classified", end="\r")
    print()

    # Per-frame detections CSV (always written — useful for tuning thresholds).
    results.sort(key=lambda r: r.timestamp)
    det_path = args.out / f"detections_{video.stem}.csv"
    with det_path.open("w", newline="") as fh:
        dw = csv.writer(fh)
        dw.writerow(["timestamp_sec", "is_condor", "species", "confidence", "interesting", "reason"])
        for r in results:
            dw.writerow([round(r.timestamp, 1), r.is_condor, r.species,
                         round(r.confidence, 2), r.interesting, r.reason])

    if args.verbose:
        for r in results:
            flag = "CONDOR" if r.is_condor else "      "
            print(f"    [{r.timestamp:6.1f}s] {flag} conf={r.confidence:.2f} "
                  f"int={r.interesting:>2} {r.species} — {r.reason}")

    def species_ok(r: FrameResult) -> bool:
        return args.species == "any" or args.species in r.species.lower()

    hits = [
        r for r in results
        if r.is_condor
        and r.confidence >= args.min_confidence
        and r.interesting >= args.min_interesting
        and species_ok(r)
    ]
    sp = "" if args.species == "any" else f", species~={args.species}"
    print(f"  {len(hits)} condor frames pass thresholds "
          f"(conf>={args.min_confidence}, interesting>={args.min_interesting}{sp})")

    ranges = merge_ranges(hits, args.interval, args.gap, args.pad, duration)
    print(f"  -> {len(ranges)} clip(s)")

    clips_dir = args.out / "clips"
    written = 0
    for n, rng in enumerate(ranges, 1):
        clip_name = f"{video.stem}_clip{n:03d}_{int(rng['start'])}s.mp4"
        clip_path = clips_dir / clip_name
        ok = args.dry_run or cut_clip(video, clip_path, rng["start"], rng["end"], args.copy)
        if not ok:
            print(f"  ! failed to cut {clip_name}", file=sys.stderr)
            continue
        written += 1
        manifest_writer.writerow({
            "source_file": str(video),
            "clip_file": "" if args.dry_run else str(clip_path),
            "start_sec": round(rng["start"], 2),
            "end_sec": round(rng["end"], 2),
            "duration_sec": round(rng["end"] - rng["start"], 2),
            "num_frames": rng["num_frames"],
            "max_interesting": rng["max_interesting"],
            "best_species": rng["best_species"],
            "best_confidence": rng["best_confidence"],
            "reason": rng["reason"],
        })

    if not args.keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-triage videos into condor-only clips.")
    ap.add_argument("input", type=Path, help="Video file or folder of videos.")
    ap.add_argument("--out", type=Path, default=Path("./output"), help="Output dir (default: ./output)")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between sampled frames (default: 2).")
    ap.add_argument("--pad", type=float, default=2.0, help="Seconds of padding around each clip (default: 2).")
    ap.add_argument("--gap", type=float, default=3.0, help="Merge hits within this gap, seconds (default: 3).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL}).")
    ap.add_argument("--workers", type=int, default=8, help="Parallel API calls (default: 8).")
    ap.add_argument("--min-confidence", type=float, default=0.6, help="Min condor confidence 0-1 (default: 0.6).")
    ap.add_argument("--min-interesting", type=int, default=1, help="Min interesting score 1-10 (default: 1).")
    ap.add_argument("--species", choices=["any", "california", "andean"], default="any",
                    help="Restrict to a condor species (default: any).")
    ap.add_argument("--max-width", type=int, default=1024, help="Downscale frames to this width (default: 1024).")
    ap.add_argument("--max-frames", type=int, default=0, help="Cap frames per video (0 = no cap).")
    ap.add_argument("--copy", action="store_true", help="Fast stream-copy clips (may start on a keyframe).")
    ap.add_argument("--keep-frames", action="store_true", help="Keep sampled frames for inspection.")
    ap.add_argument("--dry-run", action="store_true", help="Classify and report, but don't write clips.")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print every frame's classification.")
    args = ap.parse_args()

    load_env_file()
    require_tools()

    videos = gather_videos(args.input)
    if not videos:
        sys.exit(f"No videos found in {args.input}")
    print(f"Found {len(videos)} video(s).")

    args.out.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()

    frames_root = Path(tempfile.mkdtemp(prefix="condor_frames_"))
    manifest_path = args.out / "manifest.csv"
    fields = ["source_file", "clip_file", "start_sec", "end_sec", "duration_sec",
              "num_frames", "max_interesting", "best_species", "best_confidence", "reason"]

    total = 0
    try:
        with manifest_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for v in videos:
                total += process_video(client, args, v, w, frames_root)
    finally:
        if not args.keep_frames:
            shutil.rmtree(frames_root, ignore_errors=True)

    print(f"\nDone. {total} clip(s) written to {args.out / 'clips'}")
    print(f"Manifest: {manifest_path}  (sort by max_interesting to find the best shots)")


if __name__ == "__main__":
    main()
