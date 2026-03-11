"""ChromaDB vector store for BetaFinder Reels.

Schema (PRD §3.6):
    vector        float32[768]   DINOv2 embedding (avg of top 4 frames)
    reel_id       string         Instagram shortcode (unique per post)
    gym           string         alpine / mainwall / progression
    source_type   string         official / tagged / contributor
    account       string         Source IG account handle
    ig_url        string         Direct URL to Instagram Reel
    thumbnail_url string         Cached local path → /api/thumb/:id
    keyframe_urls string         JSON-encoded list of /api/frames/:id/:n paths
    caption       string         Post caption
    posted_at     string         ISO date string
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

COLLECTION_NAME = "betafinder_reels"
PERSIST_DIR = str(Path("data/vectordb"))

# Module-level singleton — avoids re-opening the SQLite backing store and
# re-loading the HNSW index on every store operation (add_reel, get_all_ids, etc.)
_collection: "chromadb.Collection | None" = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(
            path=PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# Priority order: lower number = higher priority
# Official posts always win; first-write-wins for equal priority (prevents
# cross-gym tagged-feed overwrites of the `gym` field).
SOURCE_PRIORITY: dict[str, int] = {
    "official": 0,
    "tagged": 1,
    "contributor": 2,
}


def add_reel(
    reel_id: str,
    vector: list[float],
    gym: str,
    source_type: str,
    account: str,
    ig_url: str,
    thumbnail_url: str,
    keyframe_urls: list[str],
    caption: str,
    posted_at: str,
) -> None:
    """Priority-based upsert — protects both `gym` and `source_type` fields.

    Rules (lower SOURCE_PRIORITY number = higher priority):
      new_priority > cur_priority  →  skip  (keep existing higher-priority entry)
      new_priority == cur_priority →  skip  (first-write-wins; prevents gym overwrite
                                             when two gyms' tagged feeds find same post)
      new_priority < cur_priority  →  upsert (e.g. official beats tagged; correct gym
                                              is known from the official account)
      no existing entry            →  upsert (first write)
    """
    collection = get_collection()
    new_priority = SOURCE_PRIORITY.get(source_type, 99)

    # Official posts (priority 0) always win — skip the read-before-write for
    # the highest-volume scrape path.  All other sources check for an existing
    # higher-priority entry before deciding whether to upsert.
    if new_priority > 0:
        existing = collection.get(ids=[reel_id], include=["metadatas"])
        if existing["ids"]:
            cur_priority = SOURCE_PRIORITY.get(
                existing["metadatas"][0].get("source_type", "contributor"), 99
            )
            if new_priority >= cur_priority:
                return  # keep the existing higher-priority entry unchanged

    collection.upsert(
        ids=[reel_id],
        embeddings=[vector],
        metadatas=[{
            "reel_id": reel_id,
            "gym": gym,
            "source_type": source_type,
            "account": account,
            "ig_url": ig_url,
            "thumbnail_url": thumbnail_url,
            "keyframe_urls": json.dumps(keyframe_urls),
            "caption": caption,
            "posted_at": posted_at,
        }],
    )


def get_all_ids(gym: str | None = None) -> set[str]:
    """Return all stored reel_ids, optionally filtered by gym."""
    collection = get_collection()
    where: dict[str, Any] | None = {"gym": gym} if gym else None
    result = collection.get(where=where, include=[])
    return set(result["ids"])


def count(gym: str | None = None) -> int:
    """Return total number of indexed Reels, optionally filtered by gym.

    Note: ChromaDB's collection.count() does not support a `where` filter, so
    per-gym counts are computed via get_all_ids() which does support `where`.
    """
    if gym:
        return len(get_all_ids(gym=gym))
    return get_collection().count()


def stats() -> dict:
    """Return total count and per-gym / per-source breakdowns."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return {"total": 0, "by_gym": {}, "by_source": {}}

    all_meta = collection.get(include=["metadatas"])["metadatas"]

    by_gym: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for m in all_meta:
        g = m.get("gym", "unknown")
        s = m.get("source_type", "unknown")
        by_gym[g] = by_gym.get(g, 0) + 1
        by_source[s] = by_source.get(s, 0) + 1

    return {"total": total, "by_gym": by_gym, "by_source": by_source}
