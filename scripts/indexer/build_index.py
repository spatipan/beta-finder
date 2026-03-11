"""build_index.py — Scrape → Extract → Filter → Embed → Cache → VectorDB

Usage:
    python -m scripts.indexer.build_index [--gym all|alpine|mainwall|progression]
    python -m scripts.indexer.build_index --gym alpine
"""
from __future__ import annotations

import argparse
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

load_dotenv()

from ml.embedder.dino import DinoEmbedder
from ml.embedder.wall_filter import filter_wall_frames
from ml.video.extractor import extract_frames
from ml.video.frame_scorer import select_top_frames, score_frame
from ml.cache.image_cache import cache_reel_images
from ml.vectordb.store import add_reel, get_all_ids
from scripts.scraper.ig_scraper import Reel, scrape_new_reels, scrape_tagged_reels, fetch_post_meta, read_sidecar

ACCOUNTS_FILE = Path("scripts/scraper/accounts.yaml")
WALL_CHECK_INTERVAL_SEC = 5.0   # sparse interval for Stage 1 CLIP wall check

# ── Logging helpers ───────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _log(tag: str, msg: str) -> None:
    tqdm.write(f"{_ts()}  {tag:<12} {msg}")

def _section(title: str) -> None:
    tqdm.write(f"\n{_ts()}  {'─' * 60}")
    tqdm.write(f"{_ts()}  {title}")
    tqdm.write(f"{_ts()}  {'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────

def load_accounts() -> dict:
    with open(ACCOUNTS_FILE) as f:
        return yaml.safe_load(f)


def index_gym(gym_id: str, embedder: DinoEmbedder, config: dict, ml_cfg: dict) -> int:
    """Scrape + index one gym (official + tagged posts). Returns new Reels indexed."""
    gym_cfg = next((g for g in config["gyms"] if g["id"] == gym_id), None)
    if not gym_cfg:
        _log("[index]", f"Unknown gym: {gym_id}")
        return 0

    _section(f"GYM: {gym_id}  (@{gym_cfg['ig_handle']})")

    # Sweep orphaned data/raw/ dirs from any previous crash before scraping.
    sweep_raw_dir(gym_id, embedder, ml_cfg)

    existing_ids = get_all_ids(gym=gym_id)
    _log("[index]", f"{gym_id}: {len(existing_ids)} already in index")

    # 1. Official posts — scrape and index first
    _log("[scrape]", f"Fetching official posts for @{gym_cfg['ig_handle']} …")
    t0 = time.monotonic()
    official_reels = scrape_new_reels(
        gym_id=gym_id,
        ig_handle=gym_cfg["ig_handle"],
        existing_ids=existing_ids,
        max_posts=gym_cfg.get("max_posts", 100),
    )
    _log("[scrape]", f"Found {len(official_reels)} new official reel(s)  ({time.monotonic()-t0:.1f}s)")

    new_reels: list = list(official_reels)

    # Index official posts before fetching tagged, so the refreshed
    # existing_ids set below includes them and avoids re-downloading.
    official_indexed = 0
    if official_reels:
        for reel in tqdm(official_reels, desc=f"  Indexing {gym_id} (official)", unit="reel", leave=False):
            try:
                official_indexed += _index_reel(reel, embedder, ml_cfg)
            except Exception as e:
                _log("[error]", f"{reel.shortcode}: {e}")
            finally:
                raw_dir = Path("data/raw") / reel.shortcode
                if raw_dir.exists():
                    shutil.rmtree(raw_dir, ignore_errors=True)
        _log("[index]", f"Official: {official_indexed}/{len(official_reels)} indexed")

    # 2. Tagged (community) posts — refresh existing_ids first so tagged
    #    scraper skips reels just indexed from official posts.
    tagged_cfg = config.get("tagged_scraping", {})
    if tagged_cfg.get("enabled", False):
        # Refresh to include newly indexed official reels
        existing_ids = get_all_ids(gym=gym_id)
        for src in tagged_cfg.get("sources", []):
            if src["gym_id"] == gym_id:
                _log("[scrape]", f"Fetching tagged posts for @{src['tagged_account']} …")
                t0 = time.monotonic()
                tagged = scrape_tagged_reels(
                    gym_id=gym_id,
                    tagged_account=src["tagged_account"],
                    existing_ids=existing_ids,
                    max_posts=tagged_cfg.get("max_tagged_posts", 50),
                )
                _log("[scrape]", f"Found {len(tagged)} new tagged reel(s)  ({time.monotonic()-t0:.1f}s)")
                new_reels.extend(tagged)

    extra_reels = new_reels[len(official_reels):]  # tagged only
    if not extra_reels and official_indexed == 0:
        _log("[index]", f"{gym_id}: nothing new to index")
        return official_indexed

    tagged_indexed = 0
    if extra_reels:
        for reel in tqdm(extra_reels, desc=f"  Indexing {gym_id} (tagged)", unit="reel", leave=False):
            try:
                tagged_indexed += _index_reel(reel, embedder, ml_cfg)
            except Exception as e:
                _log("[error]", f"{reel.shortcode}: {e}")
            finally:
                raw_dir = Path("data/raw") / reel.shortcode
                if raw_dir.exists():
                    shutil.rmtree(raw_dir, ignore_errors=True)
        _log("[index]", f"Tagged:   {tagged_indexed}/{len(extra_reels)} indexed")

    indexed = official_indexed + tagged_indexed
    _log("[index]", f"{gym_id}: total {indexed} new reel(s) indexed  ✓")
    return indexed


def index_contributors(embedder: DinoEmbedder, config: dict, ml_cfg: dict) -> int:
    """Scrape + index all contributor accounts. Returns new Reels indexed.

    Runs once across all gyms (not per-gym) because a contributor can post
    from multiple gyms. Gym assignment is auto-detected per post from caption
    hashtags / @mentions using gym_hints; falls back to default_gym.
    """
    contrib_cfg = config.get("contributor_scraping", {})
    if not contrib_cfg.get("enabled", False):
        return 0

    _section("CONTRIBUTORS")

    # Build gym_hints from all gyms
    gym_hints = [
        {
            "gym_id": g["id"],
            "handles": g.get("gym_hints", {}).get("handles", []),
            "hashtags": g.get("gym_hints", {}).get("hashtags", []),
        }
        for g in config["gyms"]
    ]

    # Use a global existing_ids set — contributors can post at any gym,
    # so we skip by shortcode across the entire index.
    existing_ids_all = get_all_ids()
    _log("[index]", f"{len(existing_ids_all)} total reels already indexed (global)")

    total = 0
    accounts = contrib_cfg.get("accounts", [])
    for acct in accounts:
        ig_handle = acct["ig_handle"]
        default_gym = acct.get("default_gym", config["gyms"][0]["id"])
        max_posts = acct.get("max_posts", 50)

        _log("[scrape]", f"@{ig_handle}  (default_gym={default_gym}) …")
        t0 = time.monotonic()
        reels = scrape_new_reels(
            gym_id=default_gym,
            ig_handle=ig_handle,
            existing_ids=existing_ids_all,
            max_posts=max_posts,
            gym_hints=gym_hints,
            default_gym=default_gym,
            source="contributor",
        )
        _log("[scrape]", f"@{ig_handle}: {len(reels)} new reel(s)  ({time.monotonic()-t0:.1f}s)")

        acct_indexed = 0
        for reel in tqdm(reels, desc=f"  Indexing @{ig_handle}", unit="reel", leave=False):
            try:
                n = _index_reel(reel, embedder, ml_cfg)
                acct_indexed += n
                total += n
                if n:
                    existing_ids_all.add(reel.shortcode)
            except Exception as e:
                _log("[error]", f"{reel.shortcode}: {e}")
            finally:
                raw_dir = Path("data/raw") / reel.shortcode
                if raw_dir.exists():
                    shutil.rmtree(raw_dir, ignore_errors=True)

        if reels:
            _log("[index]", f"@{ig_handle}: {acct_indexed}/{len(reels)} indexed")

    _log("[index]", f"Contributors: {total} total new reel(s) indexed  ✓")
    return total


def sweep_raw_dir(gym_id: str, embedder: DinoEmbedder, ml_cfg: dict) -> int:
    """Re-index or delete orphaned data/raw/ directories from prior crashes.

    Called once at the start of index_gym() before any scraping begins.
    Handles three cases:

    1. Already in ChromaDB (or no .mp4 found):
       → Delete the directory — nothing useful to recover.

    2. Has an .mp4 + not in ChromaDB:
       → Attempt to re-index with partial metadata:
           gym_id    = caller's gym_id (best available guess)
           ig_url    = constructed from shortcode
           posted_at = .mp4 file mtime (fallback if IG fetch fails)
           source    = "contributor" (lowest priority — gets overwritten by a
                       proper scraping run via add_reel()'s priority-upsert logic)
           account   = fetched from Instagram; falls back to "unknown"
       → Delete the directory afterwards (success or failure).

    Returns:
        Number of orphan reels successfully recovered.
    """
    raw_root = Path("data/raw")
    if not raw_root.exists():
        return 0

    # Use global (no gym filter) so a shortcode already indexed under any gym
    # is treated as present and skipped — avoids duplicate re-indexing.
    existing = get_all_ids()
    orphans = [
        d for d in raw_root.iterdir()
        if d.is_dir() and d.name not in existing and next(d.glob("*.mp4"), None)
    ]
    stale = [
        d for d in raw_root.iterdir()
        if d.is_dir() and (d.name in existing or not next(d.glob("*.mp4"), None))
    ]

    # Clean up stale dirs (already indexed or no video)
    for d in stale:
        shutil.rmtree(d, ignore_errors=True)

    if not orphans and not stale:
        return 0

    if stale:
        _log("[sweep]", f"Cleaned {len(stale)} stale raw dir(s)")

    if not orphans:
        return 0

    _log("[sweep]", f"Found {len(orphans)} orphaned raw dir(s) — attempting recovery …")
    recovered = 0

    for raw_dir in orphans:
        shortcode = raw_dir.name
        mp4 = next(raw_dir.glob("*.mp4"))

        # Recover metadata: sidecar (free, offline) → IG API → mtime fallback
        sidecar = read_sidecar(raw_dir)
        mtime = datetime.fromtimestamp(mp4.stat().st_mtime, tz=timezone.utc)

        if sidecar:
            _log("[sweep]", f"  {shortcode}: sidecar → @{sidecar['account']}  {sidecar['posted_at']}")
            reel = Reel(
                shortcode=shortcode,
                video_path=str(mp4),
                ig_url=sidecar["ig_url"],
                caption=sidecar["caption"],
                posted_at=sidecar["posted_at"],
                account=sidecar["account"],
                gym_id=sidecar.get("gym_id", gym_id),
                source=sidecar.get("source", "contributor"),
            )
        else:
            _log("[sweep]", f"  {shortcode}: no sidecar — fetching IG metadata …")
            meta = fetch_post_meta(shortcode)
            if meta:
                _log("[sweep]", f"  {shortcode}: @{meta['account']}  {meta['posted_at']}")
            else:
                _log("[sweep]", f"  {shortcode}: IG fetch failed — using mtime fallback ({mtime.date()})")
            reel = Reel(
                shortcode=shortcode,
                video_path=str(mp4),
                ig_url=f"https://www.instagram.com/reel/{shortcode}",
                caption=meta["caption"] if meta else "",
                posted_at=meta["posted_at"] if meta else mtime.date().isoformat(),
                account=meta["account"] if meta else "unknown",
                gym_id=gym_id,
                source="contributor",  # lowest priority; overwritten on next proper scrape
            )
        try:
            n = _index_reel(reel, embedder, ml_cfg)
            recovered += n
            status = "✓ recovered" if n else "✗ rejected (not a wall)"
            _log("[sweep]", f"  {shortcode}: {status}")
        except Exception as e:
            _log("[sweep]", f"  {shortcode}: could not recover — {e}")
        finally:
            shutil.rmtree(raw_dir, ignore_errors=True)

    _log("[sweep]", f"Recovery complete: {recovered}/{len(orphans)} reel(s) indexed")
    return recovered


def _index_reel(reel, embedder: DinoEmbedder, ml_cfg: dict) -> int:
    """Process a single Reel: two-stage extract → filter → score → embed → cache → store.

    Stage 1 — Sparse wall check (~11 frames at 5s interval, 1 small CLIP batch):
        Answers the binary reel-level question "is this a climbing wall?"
        10× fewer frames than dense extraction; rejects non-wall reels cheaply.

    Stage 2 — Dense frame selection (~118 frames at 0.5s interval, pure numpy):
        Only runs for confirmed wall reels. Finds the 4 sharpest / best-composed
        frames via Laplacian sharpness + Canny edge density scoring. No ML.

    Stage 3 — Batch embed (1 DINOv2 forward pass for top-4 frames):
        embed_frames() now processes all frames in a single batched call.
    """
    wall_threshold = ml_cfg.get("wall_filter_threshold", 0.20)
    thumb_w = ml_cfg.get("thumb_width", 400)
    thumb_h = ml_cfg.get("thumb_height", 711)

    # --- Stage 1: Sparse wall check (CLIP, 1 small batch) --------------------
    sparse_frames = extract_frames(reel.video_path, interval_sec=WALL_CHECK_INTERVAL_SEC)
    if not sparse_frames:
        _log("[skip]", f"{reel.shortcode}: no frames extracted")
        return 0

    wall_results = filter_wall_frames(sparse_frames, threshold=wall_threshold)
    max_score = max((s for _, s in wall_results), default=0.0)
    is_wall = any(ok for ok, _ in wall_results)
    del sparse_frames, wall_results  # free ~11 frame buffers before dense extract

    if not is_wall:
        _log("[skip]", f"{reel.shortcode}: not a wall (CLIP max={max_score:.3f} < {wall_threshold})")
        return 0

    # --- Stage 2: Dense frame selection (numpy only, no ML) ------------------
    frames = extract_frames(reel.video_path, interval_sec=0.5)
    if not frames:
        _log("[skip]", f"{reel.shortcode}: no frames on dense pass")
        return 0

    best_frame = max(frames, key=score_frame)           # single best for thumbnail
    top_frames = select_top_frames(frames, n=4)         # chronological after selection
    if not top_frames:
        _log("[skip]", f"{reel.shortcode}: select_top_frames returned empty")
        return 0

    # Cache thumbnail (best-quality frame) + keyframes (chronological)
    cache_meta = cache_reel_images(
        reel.shortcode, top_frames,
        thumb_width=thumb_w, thumb_height=thumb_h,
        best_frame=best_frame,
    )

    # --- Stage 3: Batch embed (single DINOv2 forward pass) -------------------
    pil_frames = [Image.fromarray(f[..., ::-1]) for f in top_frames]  # BGR → RGB
    vec = embedder.embed_frames(pil_frames).squeeze().tolist()

    # Store in ChromaDB
    add_reel(
        reel_id=reel.shortcode,
        vector=vec,
        gym=reel.gym_id,
        source_type=reel.source,
        account=reel.account,
        ig_url=reel.ig_url,
        thumbnail_url=cache_meta["thumbnail_url"],
        keyframe_urls=cache_meta["keyframe_urls"],
        caption=reel.caption,
        posted_at=reel.posted_at,
    )

    _log("[indexed]", f"{reel.shortcode}  @{reel.account}  [{reel.gym_id}]  CLIP={max_score:.3f}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BetaFinder vector index")
    parser.add_argument("--gym", default="all", help="Gym to index (all|alpine|mainwall|progression)")
    args = parser.parse_args()

    config = load_accounts()
    ml_cfg = config.get("ml", {})

    gym_ids = [g["id"] for g in config["gyms"]] if args.gym == "all" else [args.gym]

    _section(f"BetaFinder Indexer  —  gyms: {', '.join(gym_ids)}")
    _log("[init]", "Loading DINOv2 embedder …")
    t_start = time.monotonic()
    embedder = DinoEmbedder()
    _log("[init]", f"Embedder ready  ({time.monotonic()-t_start:.1f}s)")

    total = 0
    for gid in gym_ids:
        total += index_gym(gid, embedder, config, ml_cfg)

    # Contributor scraping runs once after all gyms (not per-gym) so each
    # contributor account is scraped exactly once regardless of --gym flag.
    if args.gym == "all":
        total += index_contributors(embedder, config, ml_cfg)

    elapsed = time.monotonic() - t_start
    _section(f"Done — {total} new reel(s) indexed  ({elapsed:.0f}s total)")


if __name__ == "__main__":
    main()
