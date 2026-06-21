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
import random
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".heic"}
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


SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "flight": {"type": "integer",
                   "description": "1-10 how striking this frame is as a condor in flight against "
                                  "scenery (mountains/sky/lake). 10 = dramatic soaring + beautiful "
                                  "landscape; 5 = condor present but ordinary; 1 = no condor or dull."}
    },
    "required": ["flight"],
    "additionalProperties": False,
}

SCORE_PROMPT = (
    "Rate this video frame 1-10 as a highlight of a CONDOR IN FLIGHT against scenery (mountains, "
    "sky, lake). 10 = the condor is clearly soaring/gliding with wings spread in a beautiful "
    "landscape; 5 = a condor is present but the shot is ordinary; 1 = no condor visible, or the "
    "frame is dull, blurry, or just empty landscape. Reward dramatic flight and great backdrops."
)


def criteria_prompt(subject: str, medium: str) -> str:
    """Build a scoring prompt from a free-text 'look for' brief (or fall back to condors)."""
    if not subject:
        return SCORE_PROMPT
    return (f"Rate this {medium} 1-10 for a fun, upbeat trip highlight reel. We especially want: "
            f"{subject}. 10 = a vivid, joyful, share-worthy moment that captures that energy; "
            f"5 = fine but ordinary; 1 = dull, blurry, empty, or off-topic.")


def score_clip(client, model: str, clip: Path, step: float, tmp: Path, workers: int = 8,
               prompt: str = None):
    """Return [(t, score 1-10)] sampled across the clip."""
    prompt = prompt or SCORE_PROMPT
    fdir = tmp / (clip.stem + "_score")
    fdir.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(clip),
         "-vf", f"fps=1/{step},scale=768:-2", "-q:v", "4", str(fdir / "f_%04d.jpg")])
    frames = sorted(fdir.glob("f_*.jpg"))

    def one(i: int, f: Path):
        b64 = base64.standard_b64encode(f.read_bytes()).decode()
        try:
            r = client.messages.create(
                model=model, max_tokens=120,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt}]}],
                output_config={"format": {"type": "json_schema", "schema": SCORE_SCHEMA}})
            txt = next((b.text for b in r.content if b.type == "text"), "{}")
            return (i * step, int(json.loads(txt)["flight"]))
        except (anthropic.APIError, json.JSONDecodeError, KeyError, ValueError, TypeError):
            return (i * step, 0)

    scores: list = [None] * len(frames)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, i, f): i for i, f in enumerate(frames)}
        for fut in as_completed(futs):
            scores[futs[fut]] = fut.result()
    return [s for s in scores if s]


def best_window(centers, clip_dur: float, win_len: float) -> tuple[float, float]:
    """Pick the most dynamic win_len-second window (most condor motion = liveliest).
    Falls back to the middle of the clip when tracking is flat/missing."""
    if clip_dur <= win_len:
        return 0.0, clip_dur
    middle = (max(0.0, (clip_dur - win_len) / 2.0), win_len)
    if len(centers) < 2:
        return middle
    best_start, best_motion = 0.0, -1.0
    s = 0.0
    while s <= clip_dur - win_len + 1e-6:
        pts = [(t, cx, cy) for t, cx, cy in centers if s - 1e-6 <= t <= s + win_len + 1e-6]
        motion = sum(abs(pts[i][1] - pts[i - 1][1]) + abs(pts[i][2] - pts[i - 1][2])
                     for i in range(1, len(pts)))
        if motion > best_motion:
            best_start, best_motion = s, motion
        s += 0.5
    return (best_start, win_len) if best_motion > 1e-6 else middle


def grade_filters(pop: float, spotlight: float) -> list[str]:
    """The shared bright/vibrant 'pop' grade (+ optional vignette), for video and photos."""
    f = []
    if pop > 0:
        f.append(f"eq=contrast={1 + 0.12 * pop:.3f}:brightness={0.05 * pop:.3f}:"
                 f"saturation={1 + 0.42 * pop:.3f}:gamma={1 + 0.05 * pop:.3f}")
        # vibrance lifts the muted colors more than the already-saturated ones (Animoto/Quik pop)
        f.append(f"vibrance=intensity={min(1.0, 0.35 * pop):.3f}")
        f.append(f"unsharp=5:5:{0.8 * pop:.3f}:5:5:0.0")
    if spotlight > 0:
        f.append(f"vignette=angle={0.6 + 0.6 * spotlight:.3f}")
    return f


def build_kenburns_clip(image: Path, out: Path, dur: float, cw: int, ch: int,
                        pop: float, spotlight: float, idx: int) -> bool:
    """Turn a still photo into a moving clip: slow Ken Burns zoom (alternating in/out),
    filling the canvas, with the same grade as the video clips."""
    frames = max(2, int(round(dur * 30)))
    big_w, big_h = cw * 2, ch * 2  # oversample so the zoom stays smooth
    if idx % 2 == 0:
        z = "min(zoom+0.0012,1.15)"           # slow zoom in
    else:
        z = "if(lte(on,1),1.15,max(zoom-0.0012,1.0))"  # start tight, ease out
    filters = [
        f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase",
        f"crop={big_w}:{big_h}",
        f"zoompan=z='{z}':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={cw}x{ch}:fps=30",
    ]
    filters += grade_filters(pop, spotlight)
    filters += ["setsar=1", "format=yuv420p"]
    vf = ",".join(filters)
    cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
              "-loop", "1", "-i", str(image), "-vf", vf, "-t", f"{dur:.3f}",
              "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-pix_fmt", "yuv420p", "-color_range", "tv", "-movflags", "+faststart", str(out)])
    if cp.returncode != 0:
        print(f"  ! Ken Burns failed for {image.name}: {cp.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


# Layout library: each option is a list of (x, y, w, h) cells as fractions of the canvas.
# Multiple arrangements per photo-count so collages don't feel templated.
T = 1 / 3
LAYOUTS = {
    2: [
        [(0, 0, 1, .5), (0, .5, 1, .5)],            # stacked
        [(0, 0, .5, 1), (.5, 0, .5, 1)],            # side by side
    ],
    3: [
        [(0, 0, 1, T), (0, T, 1, T), (0, 2 * T, 1, T)],            # three rows
        [(0, 0, 1, .58), (0, .58, .5, .42), (.5, .58, .5, .42)],   # big top + 2
        [(0, 0, .5, .42), (.5, 0, .5, .42), (0, .42, 1, .58)],     # 2 + big bottom
        [(0, 0, .62, 1), (.62, 0, .38, .5), (.62, .5, .38, .5)],   # big left + 2 right
    ],
    4: [
        [(0, 0, .5, .5), (.5, 0, .5, .5), (0, .5, .5, .5), (.5, .5, .5, .5)],         # 2x2
        [(0, 0, .62, 1), (.62, 0, .38, T), (.62, T, .38, T), (.62, 2 * T, .38, T)],   # big left + 3
        [(0, 0, 1, .55), (0, .55, T, .45), (T, .55, T, .45), (2 * T, .55, T, .45)],   # big top + 3
    ],
}


def pick_layout(n: int, rng) -> list:
    """A cell-rectangle layout for n photos, chosen with variety."""
    if n <= 1:
        return [(0.0, 0.0, 1.0, 1.0)]
    return rng.choice(LAYOUTS.get(min(n, 4), LAYOUTS[4]))


def build_grid_clip(images: list, cells: list, out: Path, dur: float, cw: int, ch: int,
                    pop: float, spotlight: float, gutter: int = 12, border: str = "white") -> bool:
    """Place photos into arbitrary cell rectangles (magazine-style). Each gets a `gutter`-px
    `border` frame and slides + fades in, staggered, for energetic motion."""
    n = min(len(images), len(cells))
    images, cells = images[:n], cells[:n]
    g = max(0, gutter)
    inputs = []
    for im in images:
        inputs += ["-loop", "1", "-i", str(im)]
    fc = [f"color=c={border}:s={cw}x{ch}:r=30:d={dur:.3f}[bg]"]
    rects = []
    for i, (fx, fy, fw, fh) in enumerate(cells):
        cx, cy = even(fx * cw), even(fy * ch)
        cwi, chi = even(fw * cw), even(fh * ch)
        iw, ih = max(2, even(cwi - 2 * g)), max(2, even(chi - 2 * g))
        delay = round(0.10 + i * 0.15, 3)
        fc.append(f"[{i}]scale={iw}:{ih}:force_original_aspect_ratio=increase,crop={iw}:{ih},"
                  f"pad={cwi}:{chi}:{(cwi - iw) // 2}:{(chi - ih) // 2}:color={border},setsar=1,"
                  f"format=yuva420p,fade=t=in:st={delay}:d=0.35:alpha=1[c{i}]")
        rects.append((cx, cy, delay))
    dirs = [(0, 70), (70, 0), (0, -70), (-70, 0)]  # slide from bottom / right / top / left
    cur = "[bg]"
    for i, (cx, cy, delay) in enumerate(rects):
        dx, dy = dirs[i % 4]
        prog = f"min(1,max(0,(t-{delay})/0.35))"
        fc.append(f"{cur}[c{i}]overlay=x='{cx}+({dx})*(1-{prog})':"
                  f"y='{cy}+({dy})*(1-{prog})'[o{i}]")
        cur = f"[o{i}]"
    grade = ",".join(grade_filters(pop, spotlight))
    fc.append(f"{cur}{grade + ',' if grade else ''}format=yuv420p[v]")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
           "-filter_complex", ";".join(fc), "-map", "[v]", "-t", f"{dur:.3f}", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-color_range", "tv", "-movflags", "+faststart", str(out)]
    cp = run(cmd)
    if cp.returncode != 0:
        print(f"  ! grid failed: {cp.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


def build_zoom_clip(video: Path, centers, src_w: int, src_h: int, zoom: float,
                    cw_out: int, ch_out: int, out: Path, pop: float = 1.0,
                    spotlight: float = 0.6, start: float = 0.0,
                    dur: float | None = None) -> bool:
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
    filters += grade_filters(pop, spotlight)
    filters += ["fps=30", "format=yuv420p"]
    vf = ",".join(filters)
    trim = ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}"] if dur else []
    cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *trim, "-i", str(video),
              "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
              "-pix_fmt", "yuv420p", "-color_range", "tv", "-movflags", "+faststart", str(out)])
    if cp.returncode != 0:
        print(f"  ! zoom failed for {video.name}: {cp.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


# Varied xfade transitions for a kinetic Quik-style feel (cross, slides, wipes, splits).
TRANSITIONS = ["fade", "dissolve", "slideleft", "slideright", "slideup", "slidedown",
               "wipeleft", "wiperight", "wipeup", "wipedown", "smoothleft", "smoothright",
               "smoothup", "smoothdown", "circleopen", "horzopen", "vertopen", "fadeblack"]


def build_reel(normalized: list[Path], out: Path, transition: float, music: Path | None,
               seed: int = 0) -> bool:
    """Stitch clips with varied transitions (slides/wipes/splits/dissolves), fade in/out, music."""
    durs = [probe_dims(p)[2] for p in normalized]
    n = len(normalized)
    fade = max(0.0, transition)
    rng = random.Random(seed)
    inputs: list[str] = []
    for p in normalized:
        inputs += ["-i", str(p)]

    fc: list[str] = []
    if n > 1 and fade > 0:
        cur, total, prev = "[0:v]", durs[0], None
        for j in range(1, n):
            off = max(0.0, total - fade)
            lab = f"[vx{j}]"
            trans = rng.choice([t for t in TRANSITIONS if t != prev])
            prev = trans
            fc.append(f"{cur}[{j}:v]xfade=transition={trans}:duration={fade:.3f}:offset={off:.3f}{lab}")
            cur, total = lab, total + durs[j] - fade
    else:
        joined = "".join(f"[{i}:v]" for i in range(n))
        fc.append(f"{joined}concat=n={n}:v=1:a=0[vc]")
        cur, total = "[vc]", sum(durs)

    fdur = fade if fade > 0 else 0.5
    fo_st = max(0.0, total - fdur)
    fc.append(f"{cur}fade=t=out:st={fo_st:.3f}:d={fdur:.2f},format=yuv420p[vout]")

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


PHOTO_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "integer",
                             "description": "1-10 how striking/share-worthy this photo is."}},
    "required": ["score"],
    "additionalProperties": False,
}

PHOTO_PROMPT = (
    "Rate this travel photo 1-10 for a highlight reel. 10 = striking and share-worthy "
    "(dramatic scenery, wildlife, a beautiful or emotional moment, great light/composition); "
    "5 = fine but ordinary; 1 = dull, blurry, cluttered, a duplicate, or a throwaway shot."
)


def score_photos(client, model: str, images: list, cache_dir: Path, tmp: Path,
                 workers: int = 8, prompt: str = None) -> dict:
    """Score photos 1-10 (cached by filename). Downscales each before sending."""
    prompt = prompt or PHOTO_PROMPT
    cf = cache_dir / "photo_scores.json"
    cache = json.loads(cf.read_text()) if cf.exists() else {}
    todo = [im for im in images if im.name not in cache]

    def one(im: Path):
        small = tmp / f"ps_{im.stem}.jpg"
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(im),
             "-vf", "scale=768:-2", "-frames:v", "1", "-q:v", "4", str(small)])
        try:
            b64 = base64.standard_b64encode(small.read_bytes()).decode()
            r = client.messages.create(
                model=model, max_tokens=80,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": prompt}]}],
                output_config={"format": {"type": "json_schema", "schema": PHOTO_SCHEMA}})
            txt = next((b.text for b in r.content if b.type == "text"), "{}")
            return im.name, int(json.loads(txt)["score"])
        except Exception:
            return im.name, 0

    if todo:
        print(f"Scoring {len(todo)} photo(s) with {model} ...")
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for name, sc in ex.map(one, todo):
                cache[name] = sc
                done += 1
                if done % 25 == 0 or done == len(todo):
                    print(f"   {done}/{len(todo)} scored", end="\r")
        print()
        cf.write_text(json.dumps(cache))
    return cache


def best_score_window(scores, dur: float, length: float) -> tuple[float, float]:
    """Best (start, mean-score) window of `length` seconds from per-frame scores."""
    if dur <= length:
        return 0.0, (sum(v for _, v in scores) / len(scores) if scores else 0.0)
    best_start, best_sc, s = 0.0, -1.0, 0.0
    while s <= dur - length + 1e-6:
        pts = [v for t, v in scores if s - 1e-6 <= t <= s + length + 1e-6]
        sc = sum(pts) / len(pts) if pts else 0.0
        if sc > best_sc:
            best_start, best_sc = s, sc
        s += 0.5
    return best_start, best_sc


def interleave(vids: list, photos: list) -> list:
    """Mix photos in among the video highlights so they're spread through the reel."""
    out, i, j = [], 0, 0
    while i < len(vids) or j < len(photos):
        if i < len(vids):
            out.append(vids[i]); i += 1
        if j < len(photos):
            out.append(photos[j]); j += 1
    return out


def capture_dt(path: Path) -> str:
    """Capture timestamp 'YYYY-MM-DD HH:MM:SS'. Prefers a date prefix in the filename
    (e.g. extracted book photos), else falls back to Spotlight metadata; '' if unknown."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})[T ](\d{2})[-:](\d{2})[-:](\d{2})", path.name)
    if m:
        return f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}"
    cp = run(["mdls", "-raw", "-name", "kMDItemContentCreationDate", str(path)])
    s = (cp.stdout or "").strip()
    return s[:19] if s and s != "(null)" else ""


def scores_from_detections(clip_name: str, manifest: Path, out_dir: Path, clip_dur: float):
    """Reuse the clipper's per-frame 'interesting' scores (no API). Maps the clip back to
    its source via the manifest and reads detections_<source>.csv. Non-condor frames -> 0."""
    if not manifest.exists():
        return None
    info = None
    with manifest.open() as fh:
        for row in csv.DictReader(fh):
            if Path(row.get("clip_file", "")).name == clip_name:
                info = row
                break
    if not info:
        return None
    det = out_dir / f"detections_{Path(info['source_file']).stem}.csv"
    if not det.exists():
        return None
    start = float(info.get("start_sec", 0) or 0)
    scores = []
    with det.open() as fh:
        for row in csv.DictReader(fh):
            t = float(row["timestamp_sec"])
            rel = t - start
            if -1e-6 <= rel <= clip_dur + 2.0:
                scores.append((rel, int(row["interesting"]) if row.get("is_condor") == "True" else 0))
    return scores or None


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
    ap.add_argument("--spotlight", type=float, default=0.0,
                    help="Spotlight/vignette strength, 0-1 (0 = off, brighter). Default 0.")
    ap.add_argument("--clip-len", type=float, default=3.0,
                    help="Length of each highlight window, seconds (default: 3.0).")
    ap.add_argument("--target", type=float, default=21.0,
                    help="Target total seconds of highlights before transitions (default: 21 -> ~17s reel).")
    ap.add_argument("--max-clips", type=int, default=0,
                    help="Only consider this many source clips, best-first (0 = all).")
    ap.add_argument("--per-clip-max", type=int, default=2,
                    help="Max highlights taken from any one clip, for variety (default: 2).")
    ap.add_argument("--feature", type=str,
                    help="Lead the reel with the clip whose name contains this text (the hero shot).")
    ap.add_argument("--feature-len", type=float, default=5.0,
                    help="Seconds of screen time for the featured clip (default: 5).")
    ap.add_argument("--feature-at", type=float, default=None,
                    help="Start the featured window at this second (e.g. 0 for a fly-in). "
                         "Default: auto-pick the highest-scoring window.")
    ap.add_argument("--photos", type=Path,
                    help="A photo file or folder of photos to mix in (Ken Burns motion).")
    ap.add_argument("--photo-len", type=float, default=2.5,
                    help="Seconds per still photo (default: 2.5).")
    ap.add_argument("--max-photos", type=int, default=0,
                    help="Score photos and keep only the best N (0 = use all, no scoring).")
    ap.add_argument("--seed", type=int, default=7,
                    help="Random seed for layout/size variety (change it to reshuffle the look).")
    ap.add_argument("--gutter", type=int, default=12,
                    help="Border/gutter px between photos in grid collages (0 = none). Default 12.")
    ap.add_argument("--gutter-color", type=str, default="white",
                    help="Color of the grid border/gutter (default: white).")
    ap.add_argument("--add-clips", type=Path, nargs="*", default=[],
                    help="Extra video files to include directly (trimmed to a middle window).")
    ap.add_argument("--subject", type=str, default=None,
                    help="Free-text 'look for' brief for scoring (e.g. 'people laughing, "
                         "landscapes, animals, the energy of the trip'). Default: condors in flight.")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="Drop video moments scoring below this (0-10) so only strong clips show; "
                         "photos still cover every day.")
    ap.add_argument("--coverage", action="store_true",
                    help="Spread picks across every capture day (a bit of each day) and order the "
                         "reel chronologically, instead of just the globally top-scoring moments.")
    ap.add_argument("--rescore", action="store_true",
                    help="Force re-scoring moments (ignore cached scores in output/_cache).")
    args = ap.parse_args()

    load_env_file()
    require_tools()

    cw_out, ch_out = (even(int(x)) for x in args.canvas.lower().split("x"))
    clips = []
    if args.clips and args.clips.exists():
        clips = [p for p in args.clips.iterdir() if p.suffix.lower() in VIDEO_EXTS] \
            if args.clips.is_dir() else [args.clips]
    if not clips and not args.add_clips and not args.photos:
        sys.exit(f"No clips found in {args.clips}")
    clips = order_by_manifest(clips, args.manifest)
    if args.max_clips > 0:
        clips = clips[: args.max_clips]
    print(f"Scanning {len(clips)} clip(s) for the best ~{args.clip_len}s flight moments "
          f"(target ~{args.target:.0f}s):")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out.parent / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames_tmp = Path(tempfile.mkdtemp(prefix="reel_frames_"))
    norm_tmp = Path(tempfile.mkdtemp(prefix="reel_norm_"))

    WIN = args.clip_len

    def cached_scores(clip: Path):
        cf = cache_dir / f"{clip.stem}.scores.json"
        if cf.exists() and not args.rescore:
            d = json.loads(cf.read_text())
            if abs(d.get("step", -1) - args.step) < 1e-6:
                return [(t, v) for t, v in d["scores"]]
        return None

    client = None

    nseg = 0
    try:
        # Score every clip (reusing the clipper's per-frame scores when available, so no
        # new API calls), then build candidate highlight windows across all of them.
        candidates = []  # (score, clip, start, dur, w, h)
        clip_meta = {}   # name -> (path, scores, w, h, dur)
        for c in clips:
            w, h, dur = probe_dims(c)
            if not w or not h:
                print(f"  ! {c.name}: couldn't probe dimensions, skipping")
                continue
            scores = cached_scores(c)
            source = "cached scores"
            if scores is None:
                scores = scores_from_detections(c.name, args.manifest, cache_dir.parent, dur)
                source = "clipper scores (no API)"
            if scores is None:
                if client is None:
                    client = anthropic.Anthropic()
                scores = score_clip(client, args.model, c, args.step, frames_tmp,
                                    prompt=criteria_prompt(args.subject, "video frame"))
                (cache_dir / f"{c.stem}.scores.json").write_text(json.dumps(
                    {"step": args.step, "scores": [[round(t, 3), v] for t, v in scores]}))
                source = "scored via API"
            print(f"== {c.name}: {source}")
            if not scores:
                continue
            clip_meta[c.name] = (c, scores, w, h, dur)
            if dur <= WIN:
                candidates.append((sum(v for _, v in scores) / len(scores), c, 0.0, dur, w, h))
            else:
                s = 0.0
                while s <= dur - WIN + 1e-6:
                    pts = [v for t, v in scores if s - 1e-6 <= t <= s + WIN + 1e-6]
                    candidates.append((sum(pts) / len(pts) if pts else 0.0, c, s, WIN, w, h))
                    s += 0.5

        if args.min_score > 0:
            before = len(candidates)
            candidates = [c for c in candidates if c[0] >= args.min_score]
            print(f"Quality floor: kept {len(candidates)}/{before} video windows "
                  f"scoring >= {args.min_score}")

        if not candidates and not args.add_clips and not args.photos:
            sys.exit("Nothing to build.")

        cdt = {n: capture_dt(clip_meta[n][0]) for n in clip_meta}
        opener_dt = ""
        if args.coverage:
            # Best moment from each capture day, round-robin, until we hit the target.
            by_day = {}
            for cand in candidates:
                by_day.setdefault(cdt.get(cand[1].name, "")[:10], []).append(cand)
            for d in by_day:
                by_day[d].sort(key=lambda x: x[0], reverse=True)
            days = sorted(by_day)
            selected, used, total, r = [], {}, 0.0, 0
            # Optional forced opener (e.g. the plane landing); chronological sort keeps it first.
            if args.feature:
                fname = next((n for n in clip_meta if args.feature.lower() in n.lower()), None)
                if fname:
                    fc, fscores, fw, fh, fdur = clip_meta[fname]
                    flen = min(args.feature_len, fdur)
                    fst = (max(0.0, min(args.feature_at, fdur - flen)) if args.feature_at is not None
                           else best_score_window(fscores, fdur, flen)[0])
                    selected.append((10.0, fc, fst, flen, fw, fh))
                    used.setdefault(fc.name, []).append((fst, fst + flen))
                    total += flen
                    opener_dt = cdt.get(fc.name, "")
                    print(f"Opening with {fname} @ {fst:.1f}-{fst + flen:.1f}s ({opener_dt[:10]})")
                else:
                    print(f"  ! --feature '{args.feature}' matched no clip; ignoring.")
            maxlen = max((len(v) for v in by_day.values()), default=0)
            while total < args.target and r < maxlen:
                for d in days:
                    if total >= args.target:
                        break
                    if r >= len(by_day[d]):
                        continue
                    sc, c, st, du, w, h = by_day[d][r]
                    if len(used.get(c.name, [])) >= args.per_clip_max:
                        continue
                    if any(not (st + du <= a or st >= b) for a, b in used.get(c.name, [])):
                        continue
                    selected.append(by_day[d][r])
                    used.setdefault(c.name, []).append((st, st + du))
                    total += du
                r += 1
            print(f"\nCoverage: {len(selected)} clip moment(s) (~{total:.0f}s) across "
                  f"{len([d for d in days if d])} day(s).")
        else:
            # Optionally feature a hero clip: lead with its best window, given more time.
            featured = None
            if args.feature:
                fname = next((n for n in clip_meta if args.feature.lower() in n.lower()), None)
                if fname:
                    fc, fscores, fw, fh, fdur = clip_meta[fname]
                    flen = min(args.feature_len, fdur)
                    if args.feature_at is not None:
                        fst, fscore = max(0.0, min(args.feature_at, fdur - flen)), 10.0
                    else:
                        fst, fscore = best_score_window(fscores, fdur, flen)
                    featured = (fscore, fc, fst, flen, fw, fh)
                    print(f"Featuring {fname} @ {fst:.1f}-{fst + flen:.1f}s as the lead shot.")
                else:
                    print(f"  ! --feature '{args.feature}' matched no clip; ignoring.")

            # Greedily pick the best non-overlapping windows (multiple per clip allowed).
            candidates.sort(key=lambda x: x[0], reverse=True)
            selected, used, total = [], {}, 0.0
            if featured:
                _, fc, fst, flen, _, _ = featured
                selected.append(featured)
                used.setdefault(fc.name, []).append((fst, fst + flen))
                total += flen
            for sc, c, st, du, w, h in candidates:
                if total >= args.target:
                    break
                if len(used.get(c.name, [])) >= args.per_clip_max:
                    continue
                if any(not (st + du <= a or st >= b) for a, b in used.get(c.name, [])):
                    continue
                selected.append((sc, c, st, du, w, h))
                used.setdefault(c.name, []).append((st, st + du))
                total += du
            rest = sorted([s for s in selected if s is not featured], key=lambda x: x[0], reverse=True)
            selected = ([featured] if featured else []) + rest
            print(f"\nSelected {len(selected)} highlight(s) (~{total:.0f}s) from {len(used)} clip(s):")
        for sc, c, st, du, _, _ in selected:
            print(f"  - {c.name} @ {st:.1f}-{st + du:.1f}s  (score {sc:.1f}, {cdt.get(c.name, '?')[:10]})")

        video_items = []  # (capture_dt, path)
        for i, (sc, c, st, du, w, h) in enumerate(selected):
            outn = norm_tmp / f"norm_{i:02d}_{c.stem}.mp4"
            if build_zoom_clip(c, [(0.0, 0.5, 0.5)], w, h, args.zoom, cw_out, ch_out, outn,
                               args.pop, args.spotlight, st, du):
                video_items.append((cdt.get(c.name, ""), outn))

        # Extra videos included directly (no scoring), trimmed to a middle window.
        for i, extra in enumerate(args.add_clips):
            ew, eh, edur = probe_dims(extra)
            if not ew:
                print(f"  ! {extra.name}: can't read, skipping")
                continue
            elen = min(args.clip_len, edur)
            est = max(0.0, (edur - elen) / 2.0)
            outn = norm_tmp / f"add_{i:02d}.mp4"
            if build_zoom_clip(extra, [(0.0, 0.5, 0.5)], ew, eh, args.zoom, cw_out, ch_out, outn,
                               args.pop, args.spotlight, est, elen):
                video_items.append((capture_dt(extra), outn))
                print(f"+ added video {extra.name}")

        # Still photos -> Ken Burns clips. Score & keep the best (coverage spreads across days).
        photo_items = []  # (capture_dt, path)
        if args.photos:
            photos = ([args.photos] if args.photos.is_file()
                      else sorted(p for p in args.photos.iterdir() if p.suffix.lower() in IMAGE_EXTS))
            if args.coverage and opener_dt:  # keep only photos from the trip window onward
                photos = [p for p in photos if not (capture_dt(p) and capture_dt(p) < opener_dt)]
            if args.max_photos and len(photos) > args.max_photos:
                if client is None:
                    client = anthropic.Anthropic()
                pscores = score_photos(client, args.model, photos, cache_dir, frames_tmp,
                                       prompt=criteria_prompt(args.subject, "photo"))
                if args.coverage:
                    pdt = {p.name: capture_dt(p) for p in photos}
                    if opener_dt:  # only photos from the trip window (on/after the opener)
                        photos = [p for p in photos if not (pdt[p.name] and pdt[p.name] < opener_dt)]
                    by_day = {}
                    for p in photos:
                        by_day.setdefault(pdt[p.name][:10], []).append(p)
                    for d in by_day:
                        by_day[d].sort(key=lambda p: pscores.get(p.name, 0), reverse=True)
                    chosen, r = [], 0
                    maxlen = max((len(v) for v in by_day.values()), default=0)
                    while len(chosen) < args.max_photos and r < maxlen:
                        for d in sorted(by_day):
                            if len(chosen) >= args.max_photos:
                                break
                            if r < len(by_day[d]):
                                chosen.append(by_day[d][r])
                        r += 1
                    photos = chosen
                    print(f"Coverage: kept {len(photos)} photo(s) across {len(by_day)} day(s).")
                else:
                    photos = sorted(photos, key=lambda p: pscores.get(p.name, 0), reverse=True)[: args.max_photos]
                    print(f"Kept top {len(photos)} photo(s) by score.")
            # Fast montage: walk photos into a varying mix of 1-up / 2-up / 3-up / 2x2 grids,
            # each with a Ken Burns zoom.
            rng = random.Random(args.seed)
            i = seg = 0
            while i < len(photos):
                k = min(rng.choice([1, 1, 2, 2, 3, 4, 4]), len(photos) - i)
                group = photos[i:i + k]
                outp = norm_tmp / f"ph_{seg:02d}.mp4"
                d = args.photo_len if k == 1 else args.photo_len * 1.5
                if k == 1:
                    ok = build_kenburns_clip(group[0], outp, d, cw_out, ch_out,
                                             args.pop, args.spotlight, seg)
                else:
                    cells = pick_layout(k, rng)
                    ok = build_grid_clip(group, cells, outp, d, cw_out, ch_out,
                                         args.pop, args.spotlight, args.gutter, args.gutter_color)
                if ok:
                    photo_items.append((capture_dt(group[0]), outp))
                i += k
                seg += 1
            print(f"Added {len(photo_items)} photo segment(s) (varied layouts).")

        if args.coverage:  # chronological: tell the trip in order, photos & video interwoven
            items = video_items + photo_items
            if args.feature and video_items:
                # Forced opener: pin it first and drop anything captured before it.
                opener_dt, opener_path = video_items[0]
                rest = [(dt, p) for dt, p in items
                        if p != opener_path and not (dt and opener_dt and dt < opener_dt)]
                rest.sort(key=lambda x: x[0] or "9999")
                dropped = len(items) - 1 - len(rest)
                if dropped > 0:
                    print(f"Dropped {dropped} segment(s) captured before the opener.")
                sequence = [opener_path] + [p for _, p in rest]
            else:
                items.sort(key=lambda x: x[0] or "9999")
                sequence = [p for _, p in items]
        else:
            sequence = interleave([p for _, p in video_items], [p for _, p in photo_items])
        if not sequence:
            sys.exit("Nothing to build (no usable clips or photos).")
        nseg = len(sequence)

        print(f"\nStitching reel ({args.canvas}, zoom {args.zoom}, transition {args.transition}s"
              f"{', music' if args.music else ''}) ...")
        if not build_reel(sequence, args.out, args.transition, args.music, args.seed):
            sys.exit("Reel build failed.")
    finally:
        shutil.rmtree(frames_tmp, ignore_errors=True)
        shutil.rmtree(norm_tmp, ignore_errors=True)

    dur = probe_dims(args.out)[2]
    print(f"\nDone. Reel: {args.out}  ({nseg} segments, {dur:.1f}s, {args.canvas})")
    print("Tracking cached in output/_cache — re-edit framing/zoom/music instantly (no API cost).")


if __name__ == "__main__":
    main()
