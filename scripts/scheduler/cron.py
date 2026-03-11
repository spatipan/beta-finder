"""APScheduler cron — runs build_index every 6 hours (PRD §2.5).

Usage:
    python -m scripts.scheduler.cron
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from scripts.indexer.build_index import load_accounts, index_gym, index_contributors
from ml.embedder.dino import DinoEmbedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

scheduler = BlockingScheduler()


@scheduler.scheduled_job("interval", hours=6, id="scrape_and_index")
def scheduled_scrape() -> None:
    log.info("Scheduled scrape started")
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


def main() -> None:
    log.info("Starting BetaFinder scheduler (every 6 hours)")
    # Run immediately on startup, then every 6 hours
    scheduled_scrape()
    scheduler.start()


if __name__ == "__main__":
    main()
