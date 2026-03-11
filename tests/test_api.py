"""FastAPI integration tests using httpx AsyncClient + ASGITransport.

The DINOv2 embedder is replaced with a mock that returns a random vector so
the full ML model is never loaded during API tests. CLIP wall filter runs
normally — blank images reliably return query_valid=False.
"""
from __future__ import annotations

import io
import numpy as np
import pytest
import torch
from httpx import AsyncClient, ASGITransport
from PIL import Image

from backend.app.main import app
from backend.app.api.dependencies import get_embedder


# ---------------------------------------------------------------------------
# Mock embedder — returns a fixed random vector without loading DINOv2
# ---------------------------------------------------------------------------

class _MockEmbedder:
    def embed(self, image):
        v = np.random.default_rng(0).standard_normal(768).astype(np.float32)
        v /= np.linalg.norm(v)
        return torch.tensor(v).unsqueeze(0)  # shape (1, 768)


def _mock_embedder():
    return _MockEmbedder()


# ---------------------------------------------------------------------------
# Async test client fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
async def client():
    """Async httpx client with mocked embedder dependency."""
    app.dependency_overrides[get_embedder] = _mock_embedder
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helper — create a minimal PNG in memory
# ---------------------------------------------------------------------------

def _blank_png(width: int = 64, height: int = 64) -> bytes:
    img = Image.fromarray(np.full((height, width, 3), 255, dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealth:
    async def test_health_returns_ok(self, client):
        r = await client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.1.0"


class TestGyms:
    async def test_gyms_returns_list(self, client):
        r = await client.get("/api/gyms")
        assert r.status_code == 200
        ids = [g["id"] for g in r.json()]
        assert "alpine" in ids
        assert "mainwall" in ids
        assert "progression" in ids

    async def test_gyms_have_required_fields(self, client):
        r = await client.get("/api/gyms")
        for gym in r.json():
            for field in ("id", "label", "handle", "color"):
                assert field in gym


class TestStats:
    async def test_stats_returns_structure(self, client):
        r = await client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "by_gym" in data
        assert "by_source" in data


class TestSearch:
    async def test_search_with_blank_image_not_a_wall(self, client):
        """Blank white image must fail the CLIP wall filter → query_valid=False."""
        r = await client.post(
            "/api/search",
            files={"file": ("wall.png", _blank_png(), "image/png")},
            data={"gym": "all", "top_k": "5"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["query_valid"] is False
        assert data["results"] == []

    async def test_search_rejects_non_image(self, client):
        """Uploading a text file must return 400."""
        r = await client.post(
            "/api/search",
            files={"file": ("data.txt", b"not an image", "text/plain")},
            data={"gym": "all", "top_k": "5"},
        )
        assert r.status_code == 400


class TestMedia:
    async def test_thumb_404_for_unknown_id(self, client):
        r = await client.get("/api/thumb/doesnotexist_xyz")
        assert r.status_code == 404

    async def test_frames_404_for_unknown_id(self, client):
        r = await client.get("/api/frames/doesnotexist_xyz/0")
        assert r.status_code == 404
