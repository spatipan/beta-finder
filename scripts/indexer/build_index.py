"""build_index.py — Scrape → Extract → Filter → Embed → Cache → VectorDB

Usage:
    python -m scripts.indexer.build_index [--gym all|alpine|mainwall|progression]
    python -m scripts.indexer.build_index --gym alpine
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from ml.embedder.dino import DinoEmbedder
from ml.embedder.wall_filter import is_climbing_wall
from ml.video.extractor import extract_frames
from ml.video.frame_scorer import select_top_frames
from ml.cache.image_cache import cache_reel_images
from ml.vectordb.store import add_reel, get_all_ids
from scripts.scraper.ig_scraper import scrape_new_reels, scrape_tagged_reels

ACCOUNTS_FILE = Path("scripts/scraper/accounts.yaml")


def load_accounts() -> dict:
    with open(ACCOUNTS_FILE) as f:
        return yaml.safe_load(f)


def index_gym(gym_id: str, embedder: DinoEmbedder, config: dict, ml_cfg: dict) -> int:
    """Scrape + index one gym (official + tagged posts). Returns new Reels indexed."""
    gym_cfg = next((g for g in config["gyms"] if g["id"] == gym_id), None)
    if not gym_cfg:
        print(f"[index] Unknown gym: {gym_id}")
        return 0

    existing_ids = get_all_ids(gym=gym_id)
    print(f"[index] {gym_id}: {len(existing_ids)} already indexed")

    # 1. Official posts — scrape and index first
    official_reels = scrape_new_reels(
        gym_id=gym_id,
        ig_handle=gym_cfg["ig_handle"],
        existing_ids=existing_ids,
        max_posts=gym_cfg.get("max_posts", 100),
    )

    new_reels: list = list(official_reels)

    # Index official posts before fetching tagged, so the refreshed
    # existing_ids set below includes them and avoids re-downloading.
    official_indexed = 0
    for reel in tqdm(official_reels, desc=f"Indexing {gym_id} (official)"):
        try:
            official_indexed += _index_reel(reel, embedder, ml_cfg)
        except Exception as e:
            print(f"[index] Error indexing {reel.shortcode}: {e}")
        finally:
            raw_dir = Path("data/raw") / reel.shortcode
            if raw_dir.exists():
                shutil.rmtree(raw_dir, ignore_errors=True)

    # 2. Tagged (community) posts — refresh existing_ids first so tagged
    #    scraper skips reels just indexed from official posts.
    tagged_cfg = config.get("tagged_scraping", {})
    if tagged_cfg.get("enabled", False):
        # Refresh to include newly indexed official reels
        existing_ids = get_all_ids(gym=gym_id)
        for src in tagged_cfg.get("sources", []):
            if src["gym_id"] == gym_id:
                tagged = scrape_tagged_reels(
                    gym_id=gym_id,
                    tagged_account=src["tagged_account"],
                    existing_ids=existing_ids,
                    max_posts=tagged_cfg.get("max_tagged_posts", 50),
                )
                new_reels.extend(tagged)

    extra_reels = new_reels[len(official_reels):]  # tagged only
    if not extra_reels and official_indexed == 0:
        print(f"[index] {gym_id}: no new Reels to index")
        return official_indexed

    indexed = official_indexed
    for reel in tqdm(extra_reels, desc=f"Indexing {gym_id} (tagged)"):
        try:
            indexed += _index_reel(reel, embedder, ml_cfg)
        except Exception as e:
            print(f"[index] Error indexing {reel.shortcode}: {e}")
        finally:
            raw_dir = Path("data/raw") / reel.shortcode
            if raw_dir.exists():
                shutil.rmtree(raw_dir, ignore_errors=True)

    print(f"[index] {gym_id}: indexed {indexed} new Reels")
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

    total = 0
    for acct in contrib_cfg.get("accounts", []):
        ig_handle = acct["ig_handle"]
        default_gym = acct.get("default_gym", config["gyms"][0]["id"])
        max_posts = acct.get("max_posts", 50)

        print(f"[index] contributor @{ig_handle} (default_gym={default_gym})")
        reels = scrape_new_reels(
            gym_id=default_gym,
            ig_handle=ig_handle,
            existing_ids=existing_ids_all,
            max_posts=max_posts,
            gym_hints=gym_hints,
            default_gym=default_gym,
            source="contributor",
        )

        for reel in tqdm(reels, desc=f"Indexing contributor @{ig_handle}"):
            try:
                n = _index_reel(reel, embedder, ml_cfg)
                total += n
                if n:
                    # Add to local set so subsequent contributors don't re-download
                    existing_ids_all.add(reel.shortcode)
                    print(f"  [{reel.gym_id}] {reel.shortcode}")
            except Exception as e:
                print(f"[index] Error indexing {reel.shortcode}: {e}")
            finally:
                raw_dir = Path("data/raw") / reel.shortcode
                if raw_dir.exists():
                    shutil.rmtree(raw_dir, ignore_errors=True)

    print(f"[index] contributors: indexed {total} new Reels")
    return total


def _index_reel(reel, embedder: DinoEmbedder, ml_cfg: dict) -> int:
    """Process a single Reel: extract → filter → score → embed → cache → store."""
    from PIL import Image

    wall_threshold = ml_cfg.get("wall_filter_threshold", 0.20)
    thumb_w = ml_cfg.get("thumb_width", 400)
    thumb_h = ml_cfg.get("thumb_height", 711)

    frames = extract_frames(reel.video_path)
    if not frames:
        return 0

    # Filter to climbing wall frames — log scores for diagnostics
    scored = [(f, is_climbing_wall(f, threshold=wall_threshold)) for f in frames]
    max_score = max((score for _, (_, score) in scored), default=0.0)
    wall_frames = [f for f, (is_wall, _) in scored if is_wall]
    if not wall_frames:
        print(f"  [skip] {reel.shortcode}: no wall frames detected (max score={max_score:.3f})")
        return 0

    # Select top 4 frames (chronological order) + track the single best for thumbnail
    from ml.video.frame_scorer import score_frame
    best_frame = max(wall_frames, key=score_frame)
    top_frames = select_top_frames(wall_frames, n=4)  # chronological after selection

    # Cache thumbnail (best quality) + keyframes (chronological)
    cache_meta = cache_reel_images(
        reel.shortcode, top_frames,
        thumb_width=thumb_w, thumb_height=thumb_h,
        best_frame=best_frame,
    )

    # Embed (convert BGR → PIL for DinoEmbedder)
    pil_frames = [Image.fromarray(f[..., ::-1]) for f in top_frames]
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
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BetaFinder vector index")
    parser.add_argument("--gym", default="all", help="Gym to index (all|alpine|mainwall|progression)")
    args = parser.parse_args()

    config = load_accounts()
    ml_cfg = config.get("ml", {})
    embedder = DinoEmbedder()

    gym_ids = [g["id"] for g in config["gyms"]] if args.gym == "all" else [args.gym]
    total = 0
    for gid in gym_ids:
        total += index_gym(gid, embedder, config, ml_cfg)

    # Contributor scraping runs once after all gyms (not per-gym) so each
    # contributor account is scraped exactly once regardless of --gym flag.
    if args.gym == "all":
        total += index_contributors(embedder, config, ml_cfg)

    print(f"[index] Done. Total new Reels indexed: {total}")


if __name__ == "__main__":
    main()
