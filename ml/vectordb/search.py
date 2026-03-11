"""Cosine similarity search against the ChromaDB collection."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from ml.vectordb.store import get_collection

RECENT_DAYS = 30  # PRD §5.2: boost posts < 30 days old


def _recency_boost(posted_at: str) -> float:
    """Return a small additive boost for recent posts (< 30 days)."""
    try:
        posted = datetime.fromisoformat(posted_at)
        if datetime.utcnow() - posted < timedelta(days=RECENT_DAYS):
            return 0.02
    except Exception:
        pass
    return 0.0


def search(
    query_vector: list[float],
    gym: str = "all",
    top_k: int = 5,
    oversample: int = 4,
) -> list[dict]:
    """Search the collection and return top_k results.

    Args:
        query_vector: 768-dim DINOv2 embedding of the query image.
        gym: "all" or one of alpine / mainwall / progression.
        top_k: Number of results to return.
        oversample: Fetch this many × top_k candidates before re-ranking.

    Returns:
        List of result dicts sorted by adjusted score descending.
    """
    collection = get_collection()
    n_candidates = top_k * oversample

    where = {"gym": gym} if gym != "all" else None

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=min(n_candidates, collection.count() or 1),
        where=where,
        include=["metadatas", "distances"],
    )

    ids = result["ids"][0]
    distances = result["distances"][0]  # ChromaDB cosine: distance = 1 - similarity
    metadatas = result["metadatas"][0]

    results = []
    for reel_id, dist, meta in zip(ids, distances, metadatas):
        similarity = 1.0 - dist  # convert distance → similarity score
        boosted = similarity + _recency_boost(meta.get("posted_at", ""))
        results.append({
            "reel_id": reel_id,
            "score": round(min(boosted, 1.0), 4),
            "url": meta.get("ig_url", f"https://www.instagram.com/reel/{reel_id}"),
            "thumbnail_url": meta.get("thumbnail_url", ""),
            "keyframe_urls": json.loads(meta.get("keyframe_urls", "[]")),
            "gym": meta.get("gym", ""),
            "source_type": meta.get("source_type", "official"),
            "username": meta.get("account", ""),
            "caption": meta.get("caption", ""),
            "date": meta.get("posted_at", ""),
        })

    # Re-rank by boosted score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Add rank and media_type fields
    for i, r in enumerate(results[:top_k], start=1):
        r["rank"] = i
        r["media_type"] = "keyframe"

    return results[:top_k]
