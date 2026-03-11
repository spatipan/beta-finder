# ── Base stage ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for OpenCV + torch + ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY scripts/ ./scripts/

# Data directories (mounted as volumes in production)
RUN mkdir -p data/cache/thumb data/cache/frames data/vectordb

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── API service ──────────────────────────────────────────────────────────────────
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Scheduler service ────────────────────────────────────────────────────────────
FROM base AS scheduler
CMD ["python", "-m", "scripts.scheduler.cron"]
