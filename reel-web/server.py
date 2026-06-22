#!/usr/bin/env python3
"""Reel web server. Serves the picker UI and renders a reel from the user's chosen
photos — in their exact order, with their title/theme/music — by calling the
make_reel.py building blocks directly. No Claude: the user does the selecting, so
the render is pure ffmpeg (no API cost, no key needed by friends).

Renders run in a background thread with a progress counter the UI polls.

Run with the condor-clipper venv (has ffmpeg helpers + flask):
    ../condor-clipper/.venv/bin/python server.py
"""
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
CLIPPER = HERE.parent / "condor-clipper"
sys.path.insert(0, str(CLIPPER))
import make_reel as mr  # building blocks: build_title_cover / build_kenburns_clip / build_grid_clip / build_reel

OUTPUT = HERE / "output"; OUTPUT.mkdir(exist_ok=True)
MUSIC = HERE / "music"

CW, CH = 720, 1280   # 9:16 vertical; 720p fits the 512MB Starter instance (1080 OOMs there)
PHOTO_LEN, TITLE_LEN, TRANSITION = 1.6, 2.8, 0.35
MIN_LEN, MAX_LEN = 1.1, 2.2          # clamp per-clip on-screen time when fitting a target

# Theme → grade strength ("pop") + a per-theme color "look" (ffmpeg filters).
THEME_POP = {"bright": 1.2, "film": 0.9, "clean": 0.7, "bold": 1.4, "cool": 1.0}
THEME_LOOK = {
    "bright": "",                                                              # pop already covers vibrant
    "film":   "colorbalance=rm=0.06:gm=0.02:bm=-0.06:rs=0.03:bs=-0.04,eq=gamma=1.04",  # warm, lifted
    "clean":  "eq=saturation=0.97:contrast=1.02",                              # neutral, crisp
    "bold":   "eq=contrast=1.12:saturation=1.18",                             # contrasty punch
    "cool":   "colorbalance=rm=-0.04:bm=0.06:rs=-0.03:bs=0.05",               # cool cast
}

jobs: dict[str, dict] = {}   # job_id -> {state, done, total, url, error}
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "1"))   # renders at once (1 = safe on small instances); the rest queue
OUTPUT_TTL_MIN = int(os.environ.get("OUTPUT_TTL_MIN", "180")) # delete finished reels after this
render_sem = threading.BoundedSemaphore(MAX_CONCURRENT)

app = Flask(__name__, static_folder=None)


def prune_outputs():
    """Delete reels older than OUTPUT_TTL_MIN so the disk doesn't fill."""
    cutoff = time.time() - OUTPUT_TTL_MIN * 60
    for p in OUTPUT.glob("reel-*.mp4"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass


def plan_clips(imgs, target):
    """Group photos into clips and pick a per-clip on-screen time to hit `target` seconds.
    Few photos → each lingers (up to MAX_LEN). More photos than fit → extras are grouped
    into 2–4-up collages so the pace stays right. target=None → Auto (all singles, default pace)."""
    n = len(imgs)
    if target is None:
        return [[im] for im in imgs], PHOTO_LEN
    ideal = max(1, round((target - TITLE_LEN) / (PHOTO_LEN - TRANSITION)))
    if n <= ideal:
        groups = [[im] for im in imgs]
    else:
        clips = ideal
        while -(-n // clips) > 4:        # raise clip count until ≤4 photos per collage
            clips += 1
        groups, i = [], 0
        base, extra = divmod(n, clips)
        for k in range(clips):
            size = base + (1 if k < extra else 0)
            groups.append(imgs[i:i + size]); i += size
    L = (target - TITLE_LEN) / len(groups) + TRANSITION
    return groups, max(MIN_LEN, min(MAX_LEN, L))


def render_job(job_id, work, imgs, title, subtitle, pop, look, target, music_path):
    j = jobs[job_id]
    with render_sem:   # cap concurrent renders; extra jobs wait here, still "queued"
        try:
            groups, clip_len = plan_clips(imgs, target)
            j.update(total=(1 if title else 0) + len(groups) + 1, done=0, state="rendering")
            normalized = []
            tmp = work / "tmp"; tmp.mkdir(exist_ok=True)
            if title:
                cover = work / "cover.mp4"
                if mr.build_title_cover(imgs[0], title, subtitle, cover, TITLE_LEN, CW, CH, pop, 0.0, tmp, look):
                    normalized.append(cover)
                j["done"] += 1
            rng = random.Random(7)
            for idx, group in enumerate(groups):
                clip = work / f"clip{idx:04d}.mp4"
                if len(group) == 1:
                    ok = mr.build_kenburns_clip(group[0], clip, clip_len, CW, CH, pop, 0.0, idx, look)
                else:   # collage: orientation-matched cells, white frames, slide/fade-in
                    ordered, cells = mr.fit_layout_and_order(group, CW, CH, rng)
                    ok = mr.build_grid_clip(ordered, cells, clip, clip_len, CW, CH, pop, 0.0,
                                            gutter=14, border="white", fit="cover", look=look)
                if ok:
                    normalized.append(clip)
                j["done"] += 1
            if not normalized:
                j.update(state="error", error="Could not build any clips."); return
            out_name = f"reel-{job_id}.mp4"
            if not mr.build_reel(normalized, OUTPUT / out_name, TRANSITION, music_path, seed=7):
                j.update(state="error", error="Final stitch failed."); return
            j["done"] += 1
            j.update(state="done", url=f"/output/{out_name}")
        except Exception as e:
            j.update(state="error", error=str(e))
        finally:
            shutil.rmtree(work, ignore_errors=True)


@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/music/<path:f>")
def music(f):
    return send_from_directory(MUSIC, f)


@app.route("/output/<path:f>")
def output(f):
    return send_from_directory(OUTPUT, f)


@app.route("/api/render", methods=["POST"])
def render():
    prune_outputs()
    photos = request.files.getlist("photos")
    if not photos:
        return jsonify(error="No photos received."), 400
    title = request.form.get("title", "").strip()
    subtitle = request.form.get("subtitle", "").strip()
    theme = request.form.get("theme", "bright")
    pop, look = THEME_POP.get(theme, 1.0), THEME_LOOK.get(theme, "")
    t = request.form.get("target", "auto")
    target = None if t == "auto" else float(t)

    work = Path(tempfile.mkdtemp(prefix="reel_"))
    imgs = []
    for i, f in enumerate(photos):              # save synchronously, preserve order
        p = work / f"{i:04d}.jpg"
        f.save(str(p))
        imgs.append(p)

    music_path = None
    mf = request.files.get("music_file")
    if mf and mf.filename:
        music_path = work / ("upload_" + Path(mf.filename).name)
        mf.save(str(music_path))
    else:
        tid = request.form.get("music_track", "")
        libf = MUSIC / "library.json"
        lib = json.loads(libf.read_text()) if libf.exists() else []
        entry = next((x for x in lib if x.get("id") == tid), None)
        if entry and (HERE / entry["file"]).exists():
            music_path = HERE / entry["file"]

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {"state": "queued", "done": 0, "total": 0, "url": None, "error": None}
    threading.Thread(target=render_job, daemon=True,
                     args=(job_id, work, imgs, title, subtitle, pop, look, target, music_path)).start()
    return jsonify(job=job_id)


@app.route("/api/progress/<job_id>")
def progress(job_id):
    j = jobs.get(job_id)
    if not j:
        return jsonify(error="unknown job"), 404
    return jsonify(j)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "4321")), threaded=True)
