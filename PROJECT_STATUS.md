# BetaFinder CNX — Project Status

> Last updated: 2026-03-11 · Version 1.1.0 · Branch: `dev`

---

## Current Index Stats

| Metric | Value |
|---|---|
| Total indexed | 102 Reels |
| Alpine | 94 (official 55 + tagged 39) |
| Main Wall | 8 (tagged only) |
| Progression | 0 |
| Contributor (`patipan_poty`) | Not yet run |

---

## PRD v3 Feature Checklist

### ✅ Fully Implemented

| Feature | File(s) |
|---|---|
| Frame extraction (0.5s interval, skip 1s) | `ml/video/extractor.py` |
| CLIP wall filter — 5 pos + 4 neg prompts | `ml/embedder/wall_filter.py` |
| CLIP scoring: `pos_mean - 0.5 × neg_mean` | `ml/embedder/wall_filter.py` |
| Frame scoring: 60% sharpness + 40% coverage | `ml/video/frame_scorer.py` |
| Top-4 frames selected, returned **chronologically** | `ml/video/frame_scorer.py` |
| Thumbnail = best-quality frame (scale-to-fill, center-crop) | `ml/cache/image_cache.py` |
| DINOv2-base 768-dim embedding (avg of 4 frames) | `ml/embedder/dino.py` |
| ChromaDB cosine similarity search | `ml/vectordb/search.py` |
| Recency boost: +0.02 for posts < 30 days | `ml/vectordb/search.py` |
| Oversample 4× before re-ranking to top_k | `ml/vectordb/search.py` |
| Priority upsert: official > tagged > contributor | `ml/vectordb/store.py` |
| Official gym post scraping (incremental) | `scripts/scraper/ig_scraper.py` |
| Tagged post scraping (CLI shell-out workaround) | `scripts/scraper/ig_scraper.py` |
| Contributor post scraping with gym auto-detect | `scripts/scraper/ig_scraper.py` |
| Gym detection from caption (@mentions / #hashtags) | `scripts/scraper/ig_scraper.py` |
| gym_hints config per gym | `scripts/scraper/accounts.yaml` |
| Scheduler: every 6h via APScheduler | `scripts/scheduler/cron.py` |
| All API endpoints: search / thumb / frames / stats / health / gyms | `backend/app/api/routes/` |
| `processing_ms` in SearchResponse | `backend/app/api/routes/search.py` |
| top_k max=20 enforcement | `backend/app/api/routes/search.py` |
| Score tiers (0.85/0.70/0.55) with emoji labels | `frontend/src/pages/SearchPage.tsx` |
| Source badges: 🏟️ official / 🏷️ community / 👤 contributor | `frontend/src/components/ResultCard.tsx` |
| Stats page: by_gym + by_source breakdown | `frontend/src/pages/StatsPage.tsx` |
| Flipbook hover animation (4 keyframes) | `frontend/src/components/MediaThumb.tsx` |
| Dark mode toggle | `frontend/src/App.tsx` |
| Settings: top_k slider (1–15) | `frontend/src/pages/SettingsPage.tsx` |
| Search state machine (idle/uploading/searching/results/not_wall/error/no_results) | `frontend/src/pages/SearchPage.tsx` |
| Test suite: 28 tests passing | `tests/` |

---

### 🐛 Known Bugs (to fix next)

#### Bug 1 — Scheduler crashes with `TypeError` (HIGH)
**File**: `scripts/scheduler/cron.py` line 29

`index_gym(gym["id"], embedder, config)` — missing 4th arg `ml_cfg`.

Current `index_gym()` signature: `(gym_id, embedder, config, ml_cfg)` → TypeError at runtime.

**Fix**:
```python
# cron.py — replace scheduled_scrape():
from scripts.indexer.build_index import load_accounts, index_gym, index_contributors

def scheduled_scrape() -> None:
    config = load_accounts()
    ml_cfg = config.get("ml", {})
    embedder = DinoEmbedder()
    total = 0
    for gym in config["gyms"]:
        try:
            n = index_gym(gym["id"], embedder, config, ml_cfg)
            total += n
        except Exception as e:
            log.error(f"Error indexing {gym['id']}: {e}")
    try:
        total += index_contributors(embedder, config, ml_cfg)
    except Exception as e:
        log.error(f"Error indexing contributors: {e}")
    log.info(f"Scheduled scrape done. {total} new Reels indexed.")
```

#### Bug 2 — Scheduler never calls `index_contributors()` (HIGH)
**File**: `scripts/scheduler/cron.py`

`index_contributors()` was added to `build_index.py` but is never imported or called in `cron.py`. Contributor posts (`patipan_poty`) are only indexed during manual `--gym all` runs, not on the 6h schedule.

**Fix**: See Bug 1 — same fix adds the `index_contributors()` call.

---

### ❌ PRD Features Not Yet Implemented

#### `find_optimal_threshold()` — CLIP calibration (PRD §3.2)
**PRD requirement**: "Calibration: `find_optimal_threshold(labeled_frames)` function available. Requires 200–500 manually labeled frames. Uses sklearn `precision_recall_curve` to find optimal F1 threshold."

**Current state**: Threshold is hardcoded to 0.20 in `ml/embedder/wall_filter.py` (lowered empirically from PRD's 0.272). No calibration function exists.

**Implementation plan**:

Add to `ml/embedder/wall_filter.py` after `is_climbing_wall()`:

```python
def find_optimal_threshold(
    labeled_frames: list[tuple["np.ndarray | Image.Image", bool]],
) -> float:
    """Find F1-optimal CLIP threshold from labeled frames (PRD §3.2).

    Args:
        labeled_frames: List of (frame, is_wall) pairs.
                        Recommended: 200–500 frames from real gym footage.

    Returns:
        Optimal threshold that maximises F1 on the provided labels.

    Usage:
        from ml.embedder.wall_filter import find_optimal_threshold
        from PIL import Image
        import os

        labeled = []
        for f in os.listdir("data/labeled/wall"):
            labeled.append((Image.open(f"data/labeled/wall/{f}"), True))
        for f in os.listdir("data/labeled/not_wall"):
            labeled.append((Image.open(f"data/labeled/not_wall/{f}"), False))

        threshold = find_optimal_threshold(labeled)
        print(f"Optimal threshold: {threshold:.3f}")
        # Then update: ml.wall_filter_threshold in accounts.yaml
    """
    from sklearn.metrics import precision_recall_curve

    scores, labels = [], []
    for frame, is_wall in labeled_frames:
        _, score = is_climbing_wall(frame, threshold=-999)  # raw score only
        scores.append(score)
        labels.append(int(is_wall))

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(f1.argmax())
    return float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.20
```

**To use**: Collect labeled frames (wall images → `data/labeled/wall/`, non-wall → `data/labeled/not_wall/`), run the function, update `ml.wall_filter_threshold` in `accounts.yaml`.

---

## Immediate Next Actions

| Priority | Action |
|---|---|
| 🔴 HIGH | Fix `scripts/scheduler/cron.py` (Bug 1 + 2) |
| 🔴 HIGH | Index mainwall and progression: `build_index --gym all` |
| 🟡 MED | Implement `find_optimal_threshold()` in `wall_filter.py` |
| 🟡 MED | Run contributor scraping: `build_index --gym all` (includes `patipan_poty`) |
| 🟢 LOW | Collect labeled frames → calibrate CLIP threshold |
| 🟢 LOW | Start scheduler for automated 6h indexing |

---

## Architecture Quick Reference

```
Upload wall photo
    │
    ▼
POST /api/search
    │
    ├── CLIP wall filter (threshold 0.20 from accounts.yaml)
    │   └── query_valid=false if not a wall
    │
    ├── DINOv2 embed → 768-dim vector
    │
    ├── ChromaDB cosine search (oversample 4×)
    │
    ├── Recency boost (+0.02 if < 30 days)
    │
    └── Return top_k results with rank, score, gym, source_type
```

```
build_index --gym all (every 6h via scheduler)
    │
    ├── Official posts (instaloader Python API, incremental)
    ├── Tagged posts (instaloader CLI shell-out, iPhone API)
    └── Contributors (gym detected from caption per-post)
            │
            └── For each Reel:
                    extract frames → CLIP filter → top-4 select
                    → DINOv2 embed → cache thumb+keyframes → ChromaDB
```

---

## Key Config Locations

| Setting | File | Key |
|---|---|---|
| CLIP wall threshold | `scripts/scraper/accounts.yaml` | `ml.wall_filter_threshold` |
| Thumbnail dimensions | `scripts/scraper/accounts.yaml` | `ml.thumb_width / ml.thumb_height` |
| Max posts per gym | `scripts/scraper/accounts.yaml` | `gyms[].max_posts` |
| Contributor accounts | `scripts/scraper/accounts.yaml` | `contributor_scraping.accounts` |
| Gym color / handle | `backend/app/api/routes/gym.py` + `frontend/src/types.ts` | `GYMS[]` |
| Recency boost window | `ml/vectordb/search.py` | `RECENT_DAYS = 30` |
| Recency boost amount | `ml/vectordb/search.py` | `return 0.02` |
