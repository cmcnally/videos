#!/usr/bin/env python3
"""
make_reel.py — stitch condor clips into one highlight reel with a tracking zoom.

For each clip, Claude vision locates the condor over time; we build an ffmpeg
crop that follows the bird (punching in so it's clearly visible), normalize every
clip to a common 16:9 canvas, and concatenate them best-first into reel.mp4.

Usage:
    python make_reel.py                       # uses ./output/clips + ./output/manifest.csv
    python make_reel.py --zoom 2.5 --step 1.0
    python make_reel.py /path/to/clips --out reel.mp4
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
from pathlib import Path

import anthropic

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
DEFAULT_MODEL = "claude-opus-4-8"

BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean", "description": "Is a condor visible in this frame?"},
        "cx": {"type": "number", "description": "Condor center X as a fraction of width, 0.0-1.0."},
        "cy": {"type": "number", "description": "Condor center Y as a fraction of height, 0.0-1.0."},
    },
    "required": ["present", "cx", "cy"],
    "additionalProperties": False,
}

BBOX_PROMPT = (
    "Locate the condor (large soaring vulture) in this frame. Return present=true and the "
    "center of the bird as fractions of the image width (cx) and height (cy), where 0,0 is "
    "top-left and 1,1 is bottom-right. If no condor is visible, present=false."
)


def load_env_file() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def require_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        sys.exit(f"error: missing tool(s): {', '.join(missing)}  (brew install ffmpeg)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("error: ANTHROPIC_API_KEY not set (put it in condor-clipper/.env)")


def probe_dims(video: Path) -> tuple[int, int, float]:
    cp = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height", "-show_entries", "format=duration",
              "-of", "json", str(video)])
    data = json.loads(cp.stdout or "{}")
    st = data.get("streams", [{}])[0]
    dur = float(data.get("format", {}).get("duration", 0.0))
    return int(st.get("width", 0)), int(st.get("height", 0)), dur


def sample_centers(client, model: str, video: Path, step: float, tmp: Path) -> list[tuple[float, float, float]]:
    """Return [(t, cx, cy)] for the condor across the clip; gaps filled by hold."""
    fdir = tmp / (video.stem + "_frames")
    fdir.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
         "-vf", f"fps=1/{step},scale=640:-2", "-q:v", "4", str(fdir / "f_%04d.jpg")])
    frames = sorted(fdir.glob("f_*.jpg"))
    raw: list[tuple[float, float | None, float | None]] = []
    for i, f in enumerate(frames):
        b64 = base64.standard_b64encode(f.read_bytes()).decode()
        try:
            r = client.messages.create(
                model=model, max_tokens=256,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": BBOX_PROMPT}]}],
                output_config={"format": {"type": "json_schema", "schema": BBOX_SCHEMA}})
            txt = next((b.text for b in r.content if b.type == "text"), "{}")
            d = json.loads(txt)
            if d.get("present"):
                raw.append((i * step, float(d["cx"]), float(d["cy"])))
            else:
                raw.append((i * step, None, None))
        except (anthropic.APIError, json.JSONDecodeError, KeyError, ValueError):
            raw.append((i * step, None, None))

    # Fill gaps by carrying the nearest known center (forward then back).
    known = [(t, x, y) for t, x, y in raw if x is not None]
    if not known:
        return [(0.0, 0.5, 0.5)]
    filled: list[tuple[float, float, float]] = []
    last = (known[0][1], known[0][2])
    for t, x, y in raw:
        if x is not None:
            last = (x, y)
        filled.append((t, last[0], last[1]))
    # light smoothing (moving average of 3)
    sm = []
    for i, (t, x, y) in enumerate(filled):
        lo, hi = max(0, i - 1), min(len(filled), i + 2)
        win = filled[lo:hi]
        sm.append((t, sum(p[1] for p in win) / len(win), sum(p[2] for p in win) / len(win)))
    return sm


def even(n: float) -> int:
    return int(n) // 2 * 2


def piecewise(times: list[float], vals: list[float]) -> str:
    """ffmpeg expr: piecewise-linear value over time t."""
    expr = f"{vals[-1]:.1f}"
    for i in range(len(times) - 2, -1, -1):
        t0, t1, v0, v1 = times[i], times[i + 1], vals[i], vals[i + 1]
        seg = f"({v0:.1f}+({v1 - v0:.1f})*(t-{t0:.3f})/{max(t1 - t0, 0.001):.3f})"
        expr = f"if(lt(t,{t1:.3f}),{seg},{expr})"
    return f"if(lt(t,{times[0]:.3f}),{vals[0]:.1f},{expr})"


def build_zoom_clip(video: Path, centers, src_w: int, src_h: int, zoom: float,
                    cw_out: int, ch_out: int, out: Path) -> bool:
    """Crop a moving window that follows the condor, scale to the output canvas."""
    target_ar = cw_out / ch_out
    if src_w / src_h >= target_ar:          # source wider than canvas -> limit by height
        max_h, max_w = src_h, src_h * target_ar
    else:                                    # taller -> limit by width
        max_w, max_h = src_w, src_w / target_ar
    win_w = even(min(src_w, max_w / zoom))
    win_h = even(min(src_h, max_h / zoom))

    times = [t for t, _, _ in centers]
    xs = [max(0.0, min(src_w - win_w, cx * src_w - win_w / 2)) for _, cx, _ in centers]
    ys = [max(0.0, min(src_h - win_h, cy * src_h - win_h / 2)) for _, _, cy in centers]
    xexpr = f"max(0,min({src_w - win_w},{piecewise(times, xs)}))"
    yexpr = f"max(0,min({src_h - win_h},{piecewise(times, ys)}))"
    # Escape commas so ffmpeg treats the exprs as single crop options.
    xexpr, yexpr = xexpr.replace(",", "\\,"), yexpr.replace(",", "\\,")

    vf = (f"crop={win_w}:{win_h}:x={xexpr}:y={yexpr},"
          f"scale={cw_out}:{ch_out}:flags=lanczos,setsar=1,fps=30,format=yuv420p")
    cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
              "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-movflags", "+faststart", str(out)])
    if cp.returncode != 0:
        print(f"  ! zoom failed for {video.name}: {cp.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def order_by_manifest(clips: list[Path], manifest: Path) -> list[Path]:
    if not manifest.exists():
        return sorted(clips)
    rank: dict[str, int] = {}
    with manifest.open() as fh:
        for row in csv.DictReader(fh):
            name = Path(row.get("clip_file", "")).name
            if name:
                rank[name] = int(row.get("max_interesting", 0))
    return sorted(clips, key=lambda c: rank.get(c.name, 0), reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stitch condor clips into a tracking-zoom highlight reel.")
    ap.add_argument("clips", nargs="?", type=Path, default=Path("./output/clips"),
                    help="Folder of clips (default: ./output/clips).")
    ap.add_argument("--manifest", type=Path, default=Path("./output/manifest.csv"),
                    help="Manifest for best-first ordering (default: ./output/manifest.csv).")
    ap.add_argument("--out", type=Path, default=Path("./output/reel.mp4"), help="Output reel path.")
    ap.add_argument("--zoom", type=float, default=2.2, help="Punch-in factor (default: 2.2).")
    ap.add_argument("--step", type=float, default=1.0, help="Seconds between tracking samples (default: 1).")
    ap.add_argument("--canvas", default="1920x1080", help="Output WxH (default: 1920x1080).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL}).")
    args = ap.parse_args()

    load_env_file()
    require_tools()

    cw_out, ch_out = (even(int(x)) for x in args.canvas.lower().split("x"))
    clips = [p for p in args.clips.iterdir() if p.suffix.lower() in VIDEO_EXTS] \
        if args.clips.is_dir() else [args.clips]
    if not clips:
        sys.exit(f"No clips found in {args.clips}")
    clips = order_by_manifest(clips, args.manifest)
    print(f"Building reel from {len(clips)} clip(s), best-first:")
    for c in clips:
        print(f"  - {c.name}")

    client = anthropic.Anthropic()
    tmp = Path(tempfile.mkdtemp(prefix="reel_"))
    normalized: list[Path] = []
    try:
        for c in clips:
            print(f"\n== {c.name}: locating condor + zooming ...")
            w, h, _ = probe_dims(c)
            if not w or not h:
                print("  ! couldn't probe dimensions, skipping")
                continue
            centers = sample_centers(client, args.model, c, args.step, tmp)
            outn = tmp / f"norm_{c.stem}.mp4"
            if build_zoom_clip(c, centers, w, h, args.zoom, cw_out, ch_out, outn):
                normalized.append(outn)
                print(f"  ok ({len(centers)} tracking points)")

        if not normalized:
            sys.exit("No clips processed.")

        listfile = tmp / "concat.txt"
        listfile.write_text("".join(f"file '{p}'\n" for p in normalized))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat",
                  "-safe", "0", "-i", str(listfile), "-c", "copy", str(args.out)])
        if cp.returncode != 0:
            sys.exit(f"concat failed: {cp.stderr.strip()[:300]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    dur = probe_dims(args.out)[2]
    print(f"\nDone. Reel: {args.out}  ({len(normalized)} clips, {dur:.1f}s, {args.canvas})")


if __name__ == "__main__":
    main()
