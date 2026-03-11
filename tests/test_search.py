"""Tests for ml/vectordb/store.py and ml/vectordb/search.py.

Uses tmp_chroma_dir fixture to isolate each test in a fresh ChromaDB directory.
No ML models are loaded — only random 768-dim vectors are used.
"""
from __future__ import annotations

import pytest
from tests.helpers import make_vector
from ml.vectordb import store as store_mod
from ml.vectordb.store import add_reel, get_all_ids
from ml.vectordb.search import search


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add(reel_id: str, gym: str, source_type: str, meta: dict, seed: int = 0) -> None:
    add_reel(
        reel_id=reel_id,
        vector=make_vector(seed),
        gym=gym,
        source_type=source_type,
        **meta,
    )


# ---------------------------------------------------------------------------
# Store tests
# ---------------------------------------------------------------------------

class TestAddReel:
    def test_add_reel_stores_entry(self, tmp_chroma_dir, fake_reel_meta):
        _add("abc123", "alpine", "official", fake_reel_meta)
        assert "abc123" in get_all_ids()

    def test_official_beats_tagged(self, tmp_chroma_dir, fake_reel_meta):
        """Re-adding as official after tagged should upgrade source_type."""
        _add("abc123", "alpine", "tagged", fake_reel_meta, seed=1)
        _add("abc123", "alpine", "official", fake_reel_meta, seed=2)

        col = store_mod.get_collection()
        meta = col.get(ids=["abc123"], include=["metadatas"])["metadatas"][0]
        assert meta["source_type"] == "official"

    def test_tagged_does_not_overwrite_official(self, tmp_chroma_dir, fake_reel_meta):
        """Official entry must not be downgraded by a later tagged write."""
        _add("abc123", "alpine", "official", fake_reel_meta, seed=1)
        _add("abc123", "alpine", "tagged", fake_reel_meta, seed=2)

        col = store_mod.get_collection()
        meta = col.get(ids=["abc123"], include=["metadatas"])["metadatas"][0]
        assert meta["source_type"] == "official"

    def test_first_gym_wins_on_equal_priority(self, tmp_chroma_dir, fake_reel_meta):
        """Two tagged writes with different gyms — first gym must be preserved."""
        _add("shared1", "alpine", "tagged", fake_reel_meta, seed=1)
        _add("shared1", "mainwall", "tagged", fake_reel_meta, seed=2)

        col = store_mod.get_collection()
        meta = col.get(ids=["shared1"], include=["metadatas"])["metadatas"][0]
        assert meta["gym"] == "alpine"

    def test_get_all_ids_filtered_by_gym(self, tmp_chroma_dir, fake_reel_meta):
        _add("a1", "alpine", "official", fake_reel_meta, seed=1)
        _add("m1", "mainwall", "official", fake_reel_meta, seed=2)
        _add("m2", "mainwall", "tagged", fake_reel_meta, seed=3)

        alpine_ids = get_all_ids(gym="alpine")
        mainwall_ids = get_all_ids(gym="mainwall")

        assert alpine_ids == {"a1"}
        assert mainwall_ids == {"m1", "m2"}


# ---------------------------------------------------------------------------
# Search tests
# ---------------------------------------------------------------------------

class TestSearch:
    def _populate(self, meta: dict, n: int = 5, gym: str = "alpine") -> list[str]:
        ids = [f"reel{i}" for i in range(n)]
        for i, rid in enumerate(ids):
            _add(rid, gym, "official", meta, seed=i)
        return ids

    def test_search_returns_top_k(self, tmp_chroma_dir, fake_reel_meta):
        self._populate(fake_reel_meta, n=5)
        results = search(make_vector(seed=99), top_k=3)
        assert len(results) == 3

    def test_search_gym_filter(self, tmp_chroma_dir, fake_reel_meta):
        """Only alpine reels should come back when filtering by gym='alpine'."""
        self._populate(fake_reel_meta, n=3, gym="alpine")
        for i in range(3, 6):
            _add(f"reel{i}", "mainwall", "official", fake_reel_meta, seed=i)

        results = search(make_vector(seed=99), gym="alpine", top_k=10)
        assert all(r["gym"] == "alpine" for r in results)

    def test_search_result_fields(self, tmp_chroma_dir, fake_reel_meta):
        self._populate(fake_reel_meta, n=1)
        results = search(make_vector(seed=99), top_k=1)
        assert len(results) == 1
        r = results[0]
        for field in ("rank", "score", "reel_id", "gym", "source_type", "url"):
            assert field in r, f"Missing field: {field}"

    def test_search_score_descending(self, tmp_chroma_dir, fake_reel_meta):
        self._populate(fake_reel_meta, n=5)
        results = search(make_vector(seed=99), top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_db_returns_empty(self, tmp_chroma_dir, fake_reel_meta):
        results = search(make_vector(seed=0), top_k=5)
        assert results == []
