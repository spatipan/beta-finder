# BetaFinder CNX — Operator Manual

> Version 1.1.0 · Conda env: `beta-finder` · Python 3.12

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Running the Frontend](#3-running-the-frontend)
4. [Running the Backend](#4-running-the-backend)
5. [Indexing Gyms](#5-indexing-gyms)
6. [Running the Scheduler](#6-running-the-scheduler)
7. [Running Tests](#7-running-tests)
8. [Tuning the Wall Filter](#8-tuning-the-wall-filter)
9. [Adding a New Gym](#9-adding-a-new-gym)
10. [Adding a Contributor Account](#10-adding-a-contributor-account)
11. [Data Directories](#11-data-directories)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Conda env `beta-finder` | Python 3.12 | All commands use this env |
| Node.js | ≥ 20 | For frontend |
| Instaloader CLI session | — | Created via `instaloader --load-cookies safari --login climb.with.poom` |

**Always prefix Python commands with the conda env:**
```bash
~/miniconda3/envs/beta-finder/bin/python  ...
~/miniconda3/envs/beta-finder/bin/pytest  ...
```

Or activate the env first:
```bash
conda activate beta-finder
```

---

## 2. Environment Setup

### `.env` file (project root)
```
INSTALOADER_USER=climb.with.poom
SCRAPING_DELAY=2.0
```

- `SCRAPING_DELAY` — seconds between post downloads (increase to 4–6 if getting rate-limited)
- Session is loaded from `~/.config/instaloader/session-climb.with.poom` automatically (created by CLI)

### Refresh session (when expired)
```bash
instaloader --load-cookies safari --login climb.with.poom
```
This updates `~/.config/instaloader/session-climb.with.poom` with fresh Safari cookies.

### Verify session is valid
```bash
~/miniconda3/envs/beta-finder/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from scripts.scraper.ig_scraper import _loader
L = _loader()
print('Logged in as:', L.context.username)
"
# Expected: Logged in as: climb.with.poom
```

---

## 3. Running the Frontend

```bash
cd frontend && npm run dev
```

Opens at **http://localhost:5173**.

- Real backend mode: `VITE_USE_MOCK=false` in `frontend/.env.development` (default)
- Mock mode: set `VITE_USE_MOCK=true` for offline UI development

---

## 4. Running the Backend

```bash
~/miniconda3/envs/beta-finder/bin/uvicorn backend.app.main:app --reload --port 8000
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check → `{"status":"ok","version":"1.1.0"}` |
| `GET` | `/api/gyms` | List of gyms with id, label, color |
| `GET` | `/api/stats` | Total indexed, by gym, by source |
| `POST` | `/api/search` | Upload wall photo → matching Reels |
| `GET` | `/api/thumb/{reel_id}` | Cached thumbnail (400×711 JPEG) |
| `GET` | `/api/frames/{reel_id}/{n}` | Cached keyframe n=0–3 (chronological order) |

### Quick API checks
```bash
curl http://localhost:8000/api/health
curl http://localhost:8000/api/stats | python -m json.tool

# Test search with a cached frame
curl -X POST http://localhost:8000/api/search \
  -F "file=@data/cache/frames/<reel_id>/0.jpg" \
  -F "gym=alpine" -F "top_k=5" | python -m json.tool
```

---

## 5. Indexing Gyms

The indexer runs the full pipeline: **Scrape → Extract → Filter → Embed → Cache → Store**

### Index one gym
```bash
~/miniconda3/envs/beta-finder/bin/python -m scripts.indexer.build_index --gym alpine
~/miniconda3/envs/beta-finder/bin/python -m scripts.indexer.build_index --gym mainwall
~/miniconda3/envs/beta-finder/bin/python -m scripts.indexer.build_index --gym progression
```

### Index all gyms at once
```bash
~/miniconda3/envs/beta-finder/bin/python -m scripts.indexer.build_index --gym all
```

### What the indexer does
Per gym (official + tagged):
1. **Official posts** — fetches up to `max_posts` (100) Reels from the gym's account, stopping early at already-indexed shortcodes
2. **Tagged posts** — fetches community Reels via CLI (`instaloader --tagged`), capped at `max_tagged_posts` (50)

After all gyms (`--gym all` only):
3. **Contributor accounts** — fetches Reels from personal climber accounts (see §10). Each post's gym is auto-detected from caption; runs once, not per gym.

For each Reel:
- Extract frames every 0.5s (skip first 1s)
- Filter with CLIP wall classifier (threshold from `accounts.yaml` → `ml.wall_filter_threshold`)
- Select top 4 quality frames, **re-sorted chronologically** (keyframe 0 = earliest in video)
- Thumbnail = highest-quality frame (scale-to-fill + center-crop to 400×711, no squeezing)
- Embed with DINOv2 → 768-dim vector → store in ChromaDB
- Delete raw video to save disk space

### Reading indexer output
```
[index] alpine: 47 already indexed               ← skipped (already in DB)
Indexing alpine (official): 100%|██| 8/8         ← progress
  [skip] ABC123: no wall frames detected (max score=0.187)  ← below threshold
[scraper] CLI tagged: ...                         ← tagged via CLI
Indexing alpine (tagged): 100%|██| 50/50
[index] alpine: indexed 47 new Reels
[index] contributor @patipan_poty (default_gym=alpine)
  [alpine] DEF456                                 ← detected from caption
  [mainwall] GHI789                               ← detected from @mention
[index] contributors: indexed 12 new Reels
```

### Re-index from scratch
```bash
rm -rf data/vectordb/ data/cache/
~/miniconda3/envs/beta-finder/bin/python -m scripts.indexer.build_index --gym all
```

---

## 6. Running the Scheduler

Runs the full indexer for all gyms every 6 hours:

```bash
~/miniconda3/envs/beta-finder/bin/python -m scripts.scheduler.cron
```

Background with logging:
```bash
~/miniconda3/envs/beta-finder/bin/python -m scripts.scheduler.cron >> logs/scheduler.log 2>&1 &
```

---

## 7. Running Tests

```bash
~/miniconda3/envs/beta-finder/bin/pytest -v
```

### Test suite (28 tests)

| File | Tests | Notes |
|---|---|---|
| `tests/test_search.py` | 10 | ChromaDB only, no ML |
| `tests/test_api.py` | 8 | FastAPI + CLIP (mock DINOv2) |
| `tests/test_filter.py` | 5 | CLIP wall classifier |
| `tests/test_embedder.py` | 5 | DINOv2 embedder |

---

## 8. Tuning the Wall Filter

The CLIP wall classifier decides which video frames contain a climbing wall.

### Current threshold: `0.20`

Set in `scripts/scraper/accounts.yaml`:
```yaml
ml:
  wall_filter_threshold: 0.20
```

### Score reference (from alpine gym data)
| Score range | Interpretation |
|---|---|
| < 0.15 | Not a wall (faces, food, outdoor) |
| 0.15 – 0.19 | Ambiguous gym/lifestyle content |
| 0.19 – 0.25 | Indoor climbing content ✓ |
| > 0.25 | Strong wall signal ✓ |

### Test a single image
```bash
~/miniconda3/envs/beta-finder/bin/python -c "
from ml.embedder.wall_filter import is_climbing_wall
from PIL import Image
img = Image.open('data/cache/frames/<reel_id>/0.jpg')
is_wall, score = is_climbing_wall(img)
print(f'is_wall={is_wall}, score={score:.3f}')
"
```

---

## 9. Adding a New Gym

### Step 1 — `scripts/scraper/accounts.yaml`
```yaml
gyms:
  - id: newgym
    ig_handle: newgym_ig
    max_posts: 100

tagged_scraping:
  sources:
    - gym_id: newgym
      tagged_account: newgym_ig
```

### Step 2 — `backend/app/api/routes/gym.py`
```python
{"id": "newgym", "label": "New Gym Name", "handle": "newgym_ig", "color": "#HEXCOLOR"},
```

### Step 3 — `frontend/src/types.ts`
```typescript
{ id: "newgym", label: "New Gym Name", handle: "newgym_ig", color: "#HEXCOLOR" },
```

### Step 4 — Index
```bash
~/miniconda3/envs/beta-finder/bin/python -m scripts.indexer.build_index --gym newgym
```

---

## 10. Adding a Contributor Account

Contributor accounts are personal climber accounts (e.g. `patipan_poty`) who climb at multiple gyms. Each post is automatically assigned to the correct gym based on caption hashtags / @mentions. If no gym can be detected, the post falls back to `default_gym`.

### Step 1 — Add `gym_hints` to each gym in `scripts/scraper/accounts.yaml`

Already done for the three current gyms. Add hints for any new gym (§9):
```yaml
gyms:
  - id: alpine
    ig_handle: the_alpine_outpost
    max_posts: 100
    gym_hints:
      handles: [the_alpine_outpost, alpineoutpost]
      hashtags: [alpine, alpineoutpost, thealpineoutpost]
```

### Step 2 — Add the contributor account
```yaml
contributor_scraping:
  enabled: true
  accounts:
    - ig_handle: patipan_poty
      default_gym: alpine   # fallback when no gym detected from caption
      max_posts: 50
    # Add more:
    # - ig_handle: another_climber
    #   default_gym: mainwall
    #   max_posts: 30
```

### Step 3 — Index (always run with `--gym all`)
```bash
~/miniconda3/envs/beta-finder/bin/python -m scripts.indexer.build_index --gym all
```

Contributors are indexed **once per run** (not per gym) so each account is scraped exactly once regardless of how many gyms their posts cover. They appear in search results with `source_type: "contributor"` and `account` set to their handle.

**Detection logic (first match wins):**
1. `@mention` of any gym handle in caption → that gym
2. `#hashtag` matching any gym hashtag in caption → that gym
3. No match → `default_gym`

---

## 11. Data Directories

```
data/
├── raw/                    # Temporary — deleted after indexing
│   └── {shortcode}/
│       └── {shortcode}.mp4
├── cache/
│   ├── thumb/              # /api/thumb/:id
│   │   └── {reel_id}.jpg   # 400×711 JPEG (scale-to-fill, center-crop)
│   └── frames/             # /api/frames/:id/:n
│       └── {reel_id}/
│           ├── 0.jpg       # Earliest selected frame
│           ├── 1.jpg
│           ├── 2.jpg
│           └── 3.jpg       # Latest selected frame
└── vectordb/               # ChromaDB persistent store
    └── chroma.sqlite3
```

**Disk estimates:** ~500 KB/Reel · 100 Reels ≈ 50 MB cache

---

## 12. Troubleshooting

### Instagram 400 / 401 / 403 during scraping
Rate-limiting. Wait 15–30 min, or:
```
SCRAPING_DELAY=4.0  # in .env
```
Official posts scraping recovers automatically when the next indexer run starts.

### Tagged posts: "No tagged dir found" or slow
Tagged scraping shells out to the CLI (`instaloader --tagged`). If it fails:
1. Verify session: `instaloader --login climb.with.poom --tagged the_alpine_outpost` (should list posts)
2. Refresh session: `instaloader --load-cookies safari --login climb.with.poom`
3. The CLI process is killed once `max_tagged_posts` mp4 files are downloaded — normal behavior

### All posts skipped ("no wall frames detected")
Check the `max score=X.XXX` values in the output.
- If scores cluster 0.17–0.19, lower `wall_filter_threshold` to `0.15` in `accounts.yaml`
- Then re-index: `rm -rf data/vectordb/ data/cache/` → `build_index --gym all`

### Session expired / "Logged in as: None"
```bash
instaloader --load-cookies safari --login climb.with.poom
```
Must be run while logged into Instagram in Safari.

### Search returns `query_valid: false`
The uploaded image failed the wall filter. The threshold (0.20) may be too strict for the query image. This is expected for non-wall photos (faces, text, outdoors).

### Frontend "No results" with real backend
1. `curl http://localhost:8000/api/stats` — if `total: 0`, run the indexer
2. Check `VITE_USE_MOCK=false` in `frontend/.env.development`
3. Verify backend is running on port 8000

### pytest: "No module named 'chromadb'"
Wrong Python. Use:
```bash
~/miniconda3/envs/beta-finder/bin/pytest -v
```

### Port already in use
```bash
lsof -ti:8000 | xargs kill -9   # backend
lsof -ti:5173 | xargs kill -9   # frontend
```
