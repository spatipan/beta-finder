"""Tests for ml/embedder/dino.py — DinoEmbedder.

Strategy: use tiny 64×64 images — DINOv2 handles any valid PIL image.
Tests run on CPU (no GPU required). The embedder is session-scoped to
avoid loading the model multiple times.
"""
from __future__ import annotations

import math
import numpy as np
import pytest
from PIL import Image

from ml.embedder.dino import DinoEmbedder


@pytest.fixture(scope="session")
def embedder() -> DinoEmbedder:
    """Load DINOv2 once for the entire test session."""
    return DinoEmbedder()


class TestDinoEmbedder:
    def test_embed_returns_768_dim(self, embedder, tiny_rgb_image):
        out = embedder.embed(tiny_rgb_image)
        assert out.shape == (1, 768)

    def test_embed_frames_returns_768_dim(self, embedder, tiny_rgb_image):
        out = embedder.embed_frames([tiny_rgb_image, tiny_rgb_image])
        assert out.shape == (1, 768)

    def test_embed_numpy_returns_768_dim(self, embedder, tiny_bgr_frame):
        out = embedder.embed_numpy(tiny_bgr_frame)
        assert out.shape == (1, 768)

    def test_embed_output_is_finite(self, embedder, tiny_rgb_image):
        out = embedder.embed(tiny_rgb_image)
        assert out.isfinite().all().item()

    def test_embed_single_and_batch_same_shape(self, embedder, tiny_rgb_image):
        single = embedder.embed(tiny_rgb_image)
        batch = embedder.embed_frames([tiny_rgb_image])
        assert single.shape == batch.shape
