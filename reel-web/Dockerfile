# Reel web app — Flask UI + ffmpeg render engine. Build context is the repo root.
FROM python:3.12-slim

# System tools the reel engine needs: ffmpeg (video) + rsvg (title cards).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg librsvg2-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY reel-web/requirements.txt /app/reel-web/requirements.txt
RUN pip install --no-cache-dir -r /app/reel-web/requirements.txt

# The engine (building blocks) + the web app (UI, server, music shelf).
COPY condor-clipper/make_reel.py /app/condor-clipper/make_reel.py
COPY reel-web/ /app/reel-web/

WORKDIR /app/reel-web
ENV PORT=8000
EXPOSE 8000
# One worker (the job/progress state is in-process), threaded for concurrent polls.
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 8 -b 0.0.0.0:${PORT:-8000} --timeout 120 server:app"]
