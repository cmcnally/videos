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
                    cw_out: int, ch_out: int, out: Path, pop: float = 1.0,
                    spotlight: float = 0.6) -> bool:
    """Crop a moving window that follows the condor, scale to the output canvas.

    Because the crop is centered on the bird, the condor sits near the middle of
    the output frame — so a centered vignette acts as a spotlight that follows it.
    `pop` scales a punchy grade (contrast/saturation/sharpness); `spotlight`
    (0-1) sets the vignette strength.
    """
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

    filters = [f"crop={win_w}:{win_h}:x={xexpr}:y={yexpr}",
               f"scale={cw_out}:{ch_out}:flags=lanczos", "setsar=1"]
    if pop > 0:  # the "pop" grade: punchier contrast/saturation + sharpen the bird
        filters.append(f"eq=contrast={1 + 0.14 * pop:.3f}:saturation={1 + 0.22 * pop:.3f}:"
                       f"brightness={0.01 * pop:.3f}")
        filters.append(f"unsharp=5:5:{0.9 * pop:.3f}:5:5:0.0")
    if spotlight > 0:  # centered vignette = spotlight that follows the (centered) bird
        filters.append(f"vignette=angle={0.6 + 0.6 * spotlight:.3f}")
    filters += ["fps=30", "format=yuv420p"]
    vf = ",".join(filters)
    cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
              "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-pix_fmt", "yuv420p", "-color_range", "tv", "-movflags", "+faststart", str(out)])
    if cp.returncode != 0:
        print(f"  ! zoom failed for {video.name}: {cp.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def build_reel(normalized: list[Path], out: Path, transition: float, music: Path | None) -> bool:
    """Stitch normalized clips with cross-fades, fade in/out to black, optional music."""
    durs = [probe_dims(p)[2] for p in normalized]
    n = len(normalized)
    fade = max(0.0, transition)
    inputs: list[str] = []
    for p in normalized:
        inputs += ["-i", str(p)]

    fc: list[str] = []
    if n > 1 and fade > 0:
        cur, total = "[0:v]", durs[0]
        for j in range(1, n):
            off = max(0.0, total - fade)
            lab = f"[vx{j}]"
            fc.append(f"{cur}[{j}:v]xfade=transition=fade:duration={fade:.3f}:offset={off:.3f}{lab}")
            cur, total = lab, total + durs[j] - fade
    else:
        joined = "".join(f"[{i}:v]" for i in range(n))
        fc.append(f"{joined}concat=n={n}:v=1:a=0[vc]")
        cur, total = "[vc]", sum(durs)

    fdur = fade if fade > 0 else 0.5
    fo_st = max(0.0, total - fdur)
    fc.append(f"{cur}fade=t=in:st=0:d={fdur:.2f},fade=t=out:st={fo_st:.3f}:d={fdur:.2f},"
              f"format=yuv420p[vout]")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs]
    maps = ["-map", "[vout]"]
    if music and music.exists():
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        fc.append(f"[{n}:a]afade=t=in:st=0:d=1.5,afade=t=out:st={fo_st:.3f}:d={fdur:.2f}[aout]")
        maps += ["-map", "[aout]", "-shortest"]
    else:
        maps += ["-an"]

    cmd += ["-filter_complex", ";".join(fc), *maps,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-color_range", "tv", "-movflags", "+faststart", str(out)]
    cp = run(cmd)
    if cp.returncode != 0:
        print(f"stitch failed: {cp.stderr.strip()[:300]}", file=sys.stderr)
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
    ap.add_argument("--zoom", type=float, default=1.4,
                    help="Punch-in factor (default: 1.4; 1.0 = full frame, higher = tighter but softer).")
    ap.add_argument("--step", type=float, default=1.0, help="Seconds between tracking samples (default: 1).")
    ap.add_argument("--canvas", default="1080x1920", help="Output WxH (default: 1080x1920, vertical 9:16).")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model (default: {DEFAULT_MODEL}).")
    ap.add_argument("--transition", type=float, default=0.7,
                    help="Cross-fade seconds between clips, and fade in/out (0 = hard cuts). Default 0.7.")
    ap.add_argument("--music", type=Path, help="Audio file to lay under the reel (looped, faded).")
    ap.add_argument("--pop", type=float, default=1.0,
                    help="Grade intensity: contrast/saturation/sharpness (0 = off, 1 = default, 2 = bold).")
    ap.add_argument("--spotlight", type=float, default=0.6,
                    help="Spotlight/vignette on the (centered) condor, 0-1 (0 = off). Default 0.6.")
    ap.add_argument("--retrack", action="store_true",
                    help="Force re-running condor tracking (ignore cached tracking in output/_cache).")
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out.parent / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames_tmp = Path(tempfile.mkdtemp(prefix="reel_frames_"))
    norm_tmp = Path(tempfile.mkdtemp(prefix="reel_norm_"))

    def cached_centers(clip: Path):
        cf = cache_dir / f"{clip.stem}.centers.json"
        if cf.exists() and not args.retrack:
            d = json.loads(cf.read_text())
            if abs(d.get("step", -1) - args.step) < 1e-6:
                return [(t, cx, cy) for t, cx, cy in d["centers"]]
        return None

    need_track = any(cached_centers(c) is None for c in clips)
    client = anthropic.Anthropic() if need_track else None
    if not need_track:
        print("\n(using cached condor tracking — no API calls)")

    normalized: list[Path] = []
    try:
        for c in clips:
            w, h, _ = probe_dims(c)
            if not w or not h:
                print(f"  ! {c.name}: couldn't probe dimensions, skipping")
                continue
            centers = cached_centers(c)
            if centers is None:
                print(f"\n== {c.name}: locating condor ...")
                centers = sample_centers(client, args.model, c, args.step, frames_tmp)
                (cache_dir / f"{c.stem}.centers.json").write_text(json.dumps(
                    {"step": args.step,
                     "centers": [[round(t, 3), round(cx, 4), round(cy, 4)] for t, cx, cy in centers]}))
            else:
                print(f"== {c.name}: cached tracking")
            outn = norm_tmp / f"norm_{c.stem}.mp4"
            if build_zoom_clip(c, centers, w, h, args.zoom, cw_out, ch_out, outn,
                               args.pop, args.spotlight):
                normalized.append(outn)
        if not normalized:
            sys.exit("No clips processed.")

        print(f"\nStitching reel ({args.canvas}, zoom {args.zoom}, transition {args.transition}s"
              f"{', music' if args.music else ''}) ...")
        if not build_reel(normalized, args.out, args.transition, args.music):
            sys.exit("Reel build failed.")
    finally:
        shutil.rmtree(frames_tmp, ignore_errors=True)
        shutil.rmtree(norm_tmp, ignore_errors=True)

    dur = probe_dims(args.out)[2]
    print(f"\nDone. Reel: {args.out}  ({len(normalized)} clips, {dur:.1f}s, {args.canvas})")
    print("Tracking cached in output/_cache — re-edit framing/zoom/music instantly (no API cost).")


if __name__ == "__main__":
    main()
