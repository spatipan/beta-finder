"""
update.py — Auto-update pipeline for BetaFinder

Scheduled scraper that updates the dataset incrementally. Can run as:
  - Cron job (background scheduler)
  - Manual CLI trigger
  - Part of CI/CD pipeline

Pipeline:
  1. Scrape new posts from configured sources
  2. Filter images (wall classification)
  3. Embed images (CLIP vectors)
  4. Rebuild FAISS index
  5. Log statistics and errors
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

from src.config import load_config, get_path, get_nested
from src.logger import setup_logger
from src.scrape import scrape_gyms, scrape_contributors, load_contributors
from src.filter import (
    load_clip_model,
    get_text_embeddings,
    filter_images,
    load_filter_cache,
    save_filter_cache,
)
from src.embed import build_faiss_index, load_gym_index_safe

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Update Statistics
# ---------------------------------------------------------------------------


class UpdateStats:
    """Track update pipeline statistics"""

    def __init__(self):
        self.timestamp = datetime.now()
        self.total_new_images = 0
        self.total_filtered = 0
        self.total_embedded = 0
        self.total_indexed = 0
        self.walls_detected = 0
        self.non_walls_detected = 0
        self.scrape_time = 0
        self.filter_time = 0
        self.embed_time = 0
        self.errors = []

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_new_images": self.total_new_images,
            "total_filtered": self.total_filtered,
            "walls_detected": self.walls_detected,
            "non_walls_detected": self.non_walls_detected,
            "total_embedded": self.total_embedded,
            "total_indexed": self.total_indexed,
            "scrape_time_seconds": self.scrape_time,
            "filter_time_seconds": self.filter_time,
            "embed_time_seconds": self.embed_time,
            "total_time_seconds": self.scrape_time + self.filter_time + self.embed_time,
            "errors": self.errors,
        }

    def log_summary(self):
        """Log update summary"""
        total_time = self.scrape_time + self.filter_time + self.embed_time
        log.info("=" * 60)
        log.info("  Update Complete")
        log.info("=" * 60)
        log.info(f"New images scraped: {self.total_new_images}")
        log.info(f"Filtered: {self.total_filtered}")
        log.info(f"  Walls: {self.walls_detected}")
        log.info(f"  Non-walls: {self.non_walls_detected}")
        log.info(f"Embedded: {self.total_embedded}")
        log.info(f"Indexed: {self.total_indexed}")
        log.info(f"Total time: {total_time:.1f}s")
        if self.errors:
            log.warning(f"Errors: {len(self.errors)}")
            for err in self.errors[:5]:
                log.warning(f"  - {err}")


def save_update_log(stats: UpdateStats):
    """Save update statistics to log file"""
    log_dir = get_path("data_dir").parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"update_{stats.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, "w") as f:
        json.dump(stats.to_dict(), f, indent=2)

    log.info(f"Update log saved: {log_file}")


# ---------------------------------------------------------------------------
# Update Pipeline Steps
# ---------------------------------------------------------------------------


def step_scrape(stats: UpdateStats, args) -> int:
    """Step 1: Scrape Instagram sources"""
    log.info("=" * 60)
    log.info("  Step 1: Scraping Instagram")
    log.info("=" * 60)

    start_time = time.time()
    cfg = load_config()

    try:
        if args.contributors_only:
            log.info("Scraping contributors only...")
            contributors = load_contributors()
            if not contributors:
                log.warning("No contributors found")
                return 0

            new_count = 0
            for contrib in contributors:
                username = contrib.get("username")
                limit = args.limit or get_nested("scraping.default_limit")
                delay = args.delay or get_nested("scraping.recommended_delay")
                log.info(f"  Scraping @{username} (limit={limit})...")
                try:
                    scrape_contributors([username], limit=limit, delay=delay)
                    new_count += 1
                except Exception as e:
                    stats.errors.append(f"Scrape contributor {username}: {str(e)}")
                    log.error(f"Failed to scrape @{username}: {e}")
        else:
            log.info("Scraping official gym accounts...")
            gyms = list(cfg.get("gyms", {}).keys())
            for gym in gyms:
                limit = args.limit or get_nested("scraping.default_limit")
                delay = args.delay or get_nested("scraping.recommended_delay")
                log.info(f"  Scraping {gym} (limit={limit})...")
                try:
                    scrape_gyms([gym], limit=limit, delay=delay)
                except Exception as e:
                    stats.errors.append(f"Scrape gym {gym}: {str(e)}")
                    log.error(f"Failed to scrape {gym}: {e}")

        # Count new images
        gym_index_before = len(load_gym_index_safe())
        stats.total_new_images = len(load_gym_index_safe()) - gym_index_before

        stats.scrape_time = time.time() - start_time
        log.info(f"Scrape complete: {stats.total_new_images} new images, {stats.scrape_time:.1f}s")
        return stats.total_new_images

    except Exception as e:
        stats.errors.append(f"Scrape step failed: {str(e)}")
        log.error(f"Scrape step failed: {e}")
        return 0


def step_filter(stats: UpdateStats, skip_non_walls: bool = True) -> int:
    """Step 2: Classify images as walls vs non-walls"""
    log.info("=" * 60)
    log.info("  Step 2: Wall Classification Filter")
    log.info("=" * 60)

    start_time = time.time()

    try:
        cfg = load_config()
        wall_cfg = cfg.get("wall_filter", {})

        if not wall_cfg.get("enabled"):
            log.warning("Wall filter disabled in config")
            return 0

        # Load CLIP model
        model_name = wall_cfg.get("model", "ViT-B-32")
        threshold = wall_cfg.get("threshold", 0.05)
        device = "cpu"  # Use CPU for compatibility

        log.info(f"Loading CLIP model {model_name}...")
        model, preprocess, device = load_clip_model(model_name, device)

        log.info("Creating text embeddings...")
        text_embeds = get_text_embeddings(model, device)

        log.info("Loading cache...")
        cache = load_filter_cache()

        log.info("Loading gym index...")
        gym_index = load_gym_index_safe()
        log.info(f"Total images: {len(gym_index)}")

        log.info(f"Filtering (threshold={threshold})...")
        updated_index = filter_images(
            gym_index, cache, model, preprocess, device, text_embeds, threshold, rebuild=False
        )

        # Save cache and index
        log.info("Saving cache...")
        save_filter_cache(cache)

        log.info("Updating gym_index.json...")
        index_file = get_path("index_file")
        with open(index_file, "w") as f:
            json.dump(updated_index, f, ensure_ascii=False, indent=2)

        # Count results
        stats.total_filtered = len(updated_index)
        stats.walls_detected = sum(1 for m in updated_index if m.get("is_wall"))
        stats.non_walls_detected = stats.total_filtered - stats.walls_detected

        stats.filter_time = time.time() - start_time
        log.info(
            f"Filter complete: {stats.walls_detected} walls, {stats.non_walls_detected} non-walls, {stats.filter_time:.1f}s"
        )
        return stats.total_filtered

    except Exception as e:
        stats.errors.append(f"Filter step failed: {str(e)}")
        log.error(f"Filter step failed: {e}")
        return 0


def step_embed(stats: UpdateStats) -> int:
    """Step 3: Embed images with CLIP and build FAISS index"""
    log.info("=" * 60)
    log.info("  Step 3: Embedding & FAISS Indexing")
    log.info("=" * 60)

    start_time = time.time()

    try:
        log.info("Loading gym index...")
        gym_index = load_gym_index_safe()

        # Only embed wall images if filtering is enabled
        cfg = load_config()
        wall_cfg = cfg.get("wall_filter", {})
        if wall_cfg.get("exclude_non_walls"):
            wall_images = [m for m in gym_index if m.get("is_wall", True)]
            log.info(f"Embedding walls only: {len(wall_images)} images")
        else:
            wall_images = gym_index
            log.info(f"Embedding all images: {len(wall_images)}")

        log.info("Building FAISS index...")
        build_faiss_index(model_name=get_nested("embedding.model_name"), force_rebuild=False)

        stats.total_embedded = len(wall_images)
        stats.total_indexed = len(wall_images)

        stats.embed_time = time.time() - start_time
        log.info(f"Embedding complete: {stats.total_indexed} indexed, {stats.embed_time:.1f}s")
        return stats.total_indexed

    except Exception as e:
        stats.errors.append(f"Embed step failed: {str(e)}")
        log.error(f"Embed step failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# Main Update Function
# ---------------------------------------------------------------------------


def update_dataset(args):
    """Run full update pipeline"""
    log.info("=" * 60)
    log.info("  BetaFinder Auto-Update Pipeline")
    log.info("=" * 60)

    stats = UpdateStats()

    # Step 1: Scrape
    if not args.skip_scrape:
        step_scrape(stats, args)
    else:
        log.info("Skipping scrape (--skip-scrape)")

    # Step 2: Filter (wall classification)
    if not args.skip_filter and stats.total_new_images > 0 or args.force_filter:
        step_filter(stats)
    else:
        log.info("Skipping filter (no new images or --skip-filter)")

    # Step 3: Embed & Index
    if not args.skip_embed and (stats.total_filtered > 0 or args.force_embed):
        step_embed(stats)
    else:
        log.info("Skipping embed (no new images or --skip-embed)")

    # Summary
    stats.log_summary()
    save_update_log(stats)

    return 0 if not stats.errors else 1


def main():
    parser = argparse.ArgumentParser(
        description="BetaFinder Auto-Update Pipeline",
        epilog="""
Examples:
  python update.py                          # Full update (scrape, filter, embed)
  python update.py --contributors-only      # Update contributors only
  python update.py --skip-scrape --force-embed  # Re-embed without scraping
  python update.py --dry-run                # Show what would happen
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Scrape options
    parser.add_argument("--skip-scrape", action="store_true", help="Skip scraping step")
    parser.add_argument("--contributors-only", action="store_true", help="Scrape only contributors")
    parser.add_argument("--limit", type=int, default=None, help="Max posts per account")
    parser.add_argument("--delay", type=float, default=None, help="Delay between requests (seconds)")

    # Filter options
    parser.add_argument("--skip-filter", action="store_true", help="Skip wall filter step")
    parser.add_argument("--force-filter", action="store_true", help="Force re-filtering")

    # Embed options
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding step")
    parser.add_argument("--force-embed", action="store_true", help="Force rebuild of FAISS index")

    # General options
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN: Would execute:")
        log.info(f"  Scrape: {not args.skip_scrape}")
        log.info(f"  Filter: {not args.skip_filter or args.force_filter}")
        log.info(f"  Embed: {not args.skip_embed or args.force_embed}")
        return 0

    return update_dataset(args)


if __name__ == "__main__":
    sys.exit(main())
