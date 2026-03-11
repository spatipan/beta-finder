"""Tests for ml/embedder/wall_filter.py — is_climbing_wall().

Strategy: blank/solid-colour images reliably score below the 0.272 threshold.
No real climbing photos are needed. CLIP is lazy-loaded on first call and
reused for the whole session (session-scoped fixtures in conftest).
"""
from __future__ import annotations

import math
import numpy as np
import pytest
from PIL import Image

from ml.embedder.wall_filter import is_climbing_wall


class TestIsClimbingWall:
    def test_returns_bool_and_float(self, tiny_rgb_image):
        result = is_climbing_wall(tiny_rgb_image)
        assert isinstance(result, tuple) and len(result) == 2
        is_wall, score = result
        assert isinstance(is_wall, bool)
        assert isinstance(score, float)

    def test_score_is_finite(self, tiny_rgb_image):
        _, score = is_climbing_wall(tiny_rgb_image)
        assert math.isfinite(score)

    def test_accepts_bgr_array(self, tiny_bgr_frame):
        """BGR numpy array (OpenCV convention) should not raise."""
        is_wall, score = is_climbing_wall(tiny_bgr_frame)
        assert isinstance(is_wall, bool)
        assert math.isfinite(score)

    def test_accepts_pil_image(self, tiny_rgb_image):
        """PIL RGB Image should not raise."""
        is_wall, score = is_climbing_wall(tiny_rgb_image)
        assert isinstance(is_wall, bool)

    def test_blank_image_is_not_a_wall(self, tiny_rgb_image):
        """A plain white image must not be classified as a climbing wall."""
        is_wall, _ = is_climbing_wall(tiny_rgb_image, threshold=0.272)
        assert is_wall is False
