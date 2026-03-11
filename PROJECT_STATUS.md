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

### 🐳 Docker Issues (fixed)

#### Docker Fix 1 — `nginx.conf` not found in frontend prod stage (FIXED)
**Error**: `COPY infra/nginx.conf ... not found` during `docker compose up --build`.

**Root cause**: The `prod` stage (`FROM nginx:alpine`) is a fresh image — it cannot directly `COPY` files from the build context. Only the `build` stage (which shares the context) can access the repo root.

**Fix**: Copy `nginx.conf` into the `build` stage first, then use `COPY --from=build` in the `prod` stage (`frontend/Dockerfile`).

#### Docker Fix 2 — Obsolete `version` field in docker-compose.yml (FIXED)
**Warning**: `the attribute 'version' is obsolete, it will be ignored`.

**Fix**: Removed `version: "3.9"` from `infra/docker-compose.yml` (not required in Compose v2+).

---

### ⚠️ Known Issues / To Investigate

#### Issue 1 — `data/raw/` bloats + indexing is slow (HIGH) → PLANNED

**Symptoms**:
- `data/raw/{shortcode}/` directories accumulate on disk even after indexing completes
- Full indexing run is very slow — bottleneck not yet profiled

**Root cause — cleanup (confirmed by code review)**:

`build_index.py` does call `shutil.rmtree(raw_dir)` in every `finally` block after `_index_reel()`. However, the `data/raw/` dir still bloats because:

1. **Tagged scraping copies then leaves**: `scrape_tagged_reels()` (line 332) does `shutil.copy2(mp4_tmp, dest_mp4)` — the temp dir is cleaned up automatically, but `dest_mp4` at `data/raw/{shortcode}/{shortcode}.mp4` persists until `_index_reel()` is called. If `_index_reel()` raises before the `finally`, the raw dir is orphaned (no outer cleanup).

2. **Filter rejects leave raw files**: When CLIP rejects all frames (`wall_frames = []`), `_index_reel()` returns `0` before reaching the `finally` block's `shutil.rmtree` — the `finally` is on the caller loop in `index_gym()`, so the `raw_dir` is cleaned. But if `extract_frames()` itself raises an exception, the `finally` still runs — **this path is fine**. Re-check: the `finally` is in `index_gym()` loop, not in `_index_reel()`, so it always runs. ✓

3. **Crash / keyboard interrupt**: If the process is killed mid-run, all in-flight `data/raw/` dirs are orphaned permanently — no startup sweep exists.

4. **Official scraping downloads extra files**: `_download_reel()` calls `loader.download_post()` which also saves `.json.xz`, `.jpg` (thumbnail), and other sidecar files alongside the `.mp4`. Only `*.mp4` is found by `glob(f"{shortcode}*.mp4")` — the rest stay in `data/raw/{shortcode}/` until the `shutil.rmtree` runs. If rmtree is skipped, all sidecars leak.

**Likely bottlenecks (to profile)**:
- `extract_frames()` with `interval=0.5s` on a 60s Reel = ~120 frames decoded per video — likely the biggest per-reel cost
- `is_climbing_wall()` runs CLIP on every single frame — lazy-loaded but inference is serial; no batching
- `SCRAPE_DELAY = 2.0s` between every official post download — intentional rate-limit but compounds with large `max_posts`
- Tagged scraping polls every 2s with no early exit if `tagged_dir` doesn't exist yet, wastes up to 10 min deadline

**Implemented fixes**:

Two-stage extraction + startup sweep + crash-resume. Implemented in:
- `ml/embedder/wall_filter.py` → `filter_wall_frames()` (batch CLIP)
- `ml/embedder/dino.py` → `embed_frames()` single batched call
- `scripts/indexer/build_index.py` → `sweep_raw_dir()` + two-stage `_index_reel()`
- `scripts/scraper/ig_scraper.py` → `.reel.json` sidecar + crash-resume in `scrape_new_reels()`

```
Stage 1 — Sparse wall check (CPU-friendly):
  extract_frames(interval=5s)      → ~11 frames (was 118)
  filter_wall_frames([11 frames])  → 1 small CLIP batch
  if not wall → early exit (Stage 2 never runs) ✅

Stage 2 — Dense frame selection (numpy only, no ML):
  extract_frames(interval=0.5s)    → ~118 frames (only for confirmed walls)
  select_top_frames(n=4)           → sharpness + Canny edge score
  cache_reel_images()

Stage 3 — Batch embed:
  embed_frames(top_4)              → 1 DINOv2 batch call (was 4 serial)

Startup sweep (before scraping):
  for dir in data/raw/*/:
    already in ChromaDB or no .mp4 → delete
    .mp4 + not in ChromaDB + sidecar → re-index using sidecar metadata (no IG API)
    .mp4 + not in ChromaDB + no sidecar → fetch IG API → mtime fallback

Crash-resume (.reel.json sidecar):
  After each .mp4 download, write data/raw/{shortcode}/.reel.json with full metadata.
  On restart, scrape_new_reels() detects existing dir + sidecar → skips re-download,
  reuses existing .mp4. sweep_raw_dir() prefers sidecar over IG API call.

Tagged stage dir (data/raw/_tagged_stage/):
  CLI writes here (persistent — survives Ctrl+C). On restart, existing .mp4s are
  counted and CLI continues from where it left off. Cleaned up after parsing.
```

**GPU future path**: When `BETAFINDER_DEVICE=cuda`, restructure `index_gym()` into
cross-reel stages — batch CLIP across all reels at once, then batch DINOv2 across
all wall reels at once. Not implemented yet.

---

#### Issue 2 — No selectable CPU/GPU mode for embedding (LOW)
**Symptom**: `DinoEmbedder` and CLIP wall filter auto-select device at init time. There is no runtime flag to force CPU or GPU, making it hard to run on machines without CUDA or to benchmark performance.

**Impact**: On CPU-only machines (e.g. CI, cheap VPS), torch silently falls back to CPU but startup is slow and there is no visibility. On GPU machines, there is no way to pin to CPU for testing.

**To investigate**:
- `ml/embedder/dino.py` — check current device selection logic
- `ml/embedder/wall_filter.py` — same

**Likely fix**: Accept `device: str = "auto"` param in `DinoEmbedder.__init__()` and in CLIP loader. Expose as `BETAFINDER_DEVICE=cpu|cuda|auto` env var, read in `backend/app/api/dependencies.py` and `scripts/scheduler/cron.py`.

---

#### Issue 3 — No region field on gyms (LOW / future)
**Symptom**: Gym metadata (in `backend/app/api/routes/gym.py` and `frontend/src/types.ts`) has no `region` field. Currently all gyms are in Chiang Mai, but the system has no concept of geography.

**Impact**: If gyms from other cities are added later, the frontend has no way to group or filter by region. The API also cannot scope searches to a city.

**To add (when needed)**:
- Add `region: str` to `GymConfig` in `backend/app/api/routes/gym.py` and `GymId`/gym type in `frontend/src/types.ts`
- Populate `GYMS[]` with `region: "chiang_mai"` for existing entries
- Add optional `region` query param to `GET /api/gyms` for filtering
- Frontend: region grouping in `GymFilter` component

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

> **Future note**: May need to collect **negative examples** (non-wall frames from real scraped reels — e.g. people's faces, gym entrance, transition shots) specifically to fine-tune the CLIP wall filter model, not just calibrate its threshold. A threshold-only calibration assumes the CLIP feature space already separates walls from non-walls well; if false positive rates remain high on real data, fine-tuning on domain-specific negatives will be necessary.

---

## Immediate Next Actions

| Priority | Action |
|---|---|
| 🔴 HIGH | Fix `scripts/scheduler/cron.py` (Bug 1 + 2) |
| 🔴 HIGH | Index mainwall and progression: `build_index --gym all` |
| 🔴 HIGH | Implement two-stage pipeline + startup sweep in `build_index.py`, `wall_filter.py`, `dino.py` (Issue 1) |
| 🟡 MED | Implement `find_optimal_threshold()` in `wall_filter.py` |
| 🟡 MED | Run contributor scraping: `build_index --gym all` (includes `patipan_poty`) |
| 🟢 LOW | Add selectable CPU/GPU mode via `BETAFINDER_DEVICE` env var (Issue 2) |
| 🟢 LOW | Collect labeled frames → calibrate CLIP threshold |
| 🟢 LOW | Start scheduler for automated 6h indexing |
| 🔵 FUTURE | Add `region` field to gym metadata for multi-city support (Issue 3) |

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
