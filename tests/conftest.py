"""Shared pytest fixtures for BetaFinder tests."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Tiny image fixtures — no real photos needed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tiny_rgb_image() -> Image.Image:
    """64×64 solid white PIL RGB image."""
    return Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8), mode="RGB")


@pytest.fixture(scope="session")
def tiny_bgr_frame() -> np.ndarray:
    """64×64 solid white numpy BGR frame (OpenCV convention)."""
    return np.full((64, 64, 3), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# ChromaDB fixture — isolated tmp collection per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_chroma_dir(tmp_path, monkeypatch):
    """Patch store.PERSIST_DIR to an isolated tmp directory per test.

    Also resets the module-level _collection singleton so each test gets a
    fresh ChromaDB client pointing at the new tmp path.
    """
    import ml.vectordb.store as store_mod

    monkeypatch.setattr(store_mod, "PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(store_mod, "_collection", None)
    yield tmp_path / "chroma"
    # Reset after the test so the singleton doesn't leak into the next test
    # (monkeypatch already restores PERSIST_DIR, but we also need _collection gone)
    store_mod._collection = None


# ---------------------------------------------------------------------------
# Fake reel metadata helper
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_reel_meta() -> dict:
    """Return a base metadata dict for add_reel() calls."""
    return {
        "account": "test_account",
        "ig_url": "https://www.instagram.com/reel/TEST123",
        "thumbnail_url": "/api/thumb/TEST123",
        "keyframe_urls": ["/api/frames/TEST123/0"],
        "caption": "Test caption",
        "posted_at": "2024-01-01",
    }


