# BetaFinder CNX

Visual beta search engine for Chiang Mai bouldering gyms. Upload a wall photo — get back matching Instagram Reels.

**Version 1.1.0 · Branch: `dev`**

---

## How it works

**Indexing** (every 6h)
```
Scrape Reels (Instaloader)
→ Startup sweep: re-index or delete any orphaned data/raw/ dirs from prior crashes
→ Stage 1: Extract sparse frames (5s interval) → CLIP wall filter [batch, ~11 frames]
           not a wall → skip reel entirely
→ Stage 2: Extract dense frames (0.5s interval) → score & select top 4 [numpy only]
→ Cache thumb + keyframes
→ DINOv2-base embed top 4 frames (batch → avg → 768-dim vector)
→ ChromaDB
```

**Search**
```
Upload photo → CLIP wall filter → DINOv2 embed
→ ChromaDB cosine search (oversample 4×)
→ Recency boost (+0.02 if < 30 days)
→ Return top-k results
```

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Vite + React 18 + TypeScript + Tailwind (mobile-first, 440px) |
| Backend | FastAPI + Uvicorn |
| ML — embed | DINOv2-base (`facebook/dinov2-base`), 768-dim CLS token |
| ML — filter | CLIP ViT-B/32, 5 pos + 4 neg prompts, threshold 0.20 |
| Vector DB | ChromaDB (cosine similarity, persistent) |
| Scraper | Instaloader (incremental, official + tagged posts) |
| Scheduler | APScheduler every 6h |

---

## Gyms

| Gym | Handle | Status |
|---|---|---|
| Alpine | official + tagged | 94 Reels indexed |
| Main Wall | tagged only | 8 Reels indexed |
| Progression | — | 0 (not yet scraped) |

---

## Quick start

### Local dev

```bash
# Frontend (mock mode)
cd frontend && npm run dev          # http://localhost:5173

# Backend
uvicorn backend.app.main:app --reload --port 8000

# Index a gym
python -m scripts.indexer.build_index --gym alpine
python -m scripts.indexer.build_index --gym all

# Scheduler
python -m scripts.scheduler.cron
```

### Docker

```bash
cp .env.example .env               # fill in Instagram credentials
docker compose -f infra/docker-compose.yml up --build
# frontend → http://localhost
# api      → http://localhost:8000
```

---

## Project structure

```
backend/         FastAPI app (routes: search, thumb, frames, stats, health, gyms)
frontend/        Vite + React UI (SearchPage, StatsPage, SettingsPage, AboutPage)
ml/              DINOv2 embedder, CLIP wall filter, frame extractor, ChromaDB store
scripts/         Instaloader scraper, build_index pipeline, APScheduler cron
data/            ChromaDB index + image cache (gitignored, Docker volume)
infra/           docker-compose.yml, nginx.conf
tests/           28 passing tests
```

---

## Environment

Copy `.env.example` to `.env`:

```
INSTALOADER_USER=your_instagram_username
INSTALOADER_SESSION_FILE=/path/to/session/file
```

---

## Key config

| Setting | File | Key |
|---|---|---|
| CLIP wall threshold | `scripts/scraper/accounts.yaml` | `ml.wall_filter_threshold` |
| Max posts per gym | `scripts/scraper/accounts.yaml` | `gyms[].max_posts` |
| Contributor accounts | `scripts/scraper/accounts.yaml` | `contributor_scraping.accounts` |
| Recency boost | `ml/vectordb/search.py` | `RECENT_DAYS`, boost `+0.02` |
| CLIP sparse interval | `scripts/indexer/build_index.py` | `WALL_CHECK_INTERVAL_SEC = 5.0` |
| Dense frame interval | `ml/video/extractor.py` | `interval_sec=0.5` (default) |

---

## Indexing pipeline design notes

**Two-stage frame extraction** (CPU-first):
- Stage 1 uses sparse sampling (every 5s, ~11 frames) for CLIP wall classification. A climbing wall doesn't change second-to-second — a few frames are enough to answer the binary question.
- Stage 2 uses dense sampling (every 0.5s, ~118 frames) only for confirmed wall reels, to find the sharpest + best-composed 4 keyframes via numpy scoring (no ML).
- Non-wall reels are rejected after Stage 1, saving the dense extraction entirely.

**GPU path (future)**: When GPU is available, restructure `index_gym()` to batch CLIP across all reels at once (Stage 1), then batch DINOv2 across all wall reels at once (Stage 3). See `PROJECT_STATUS.md` Issue 1 for details.
