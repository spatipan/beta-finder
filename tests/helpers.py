"""Standalone test helpers — importable as a regular module (not conftest)."""
from __future__ import annotations

import numpy as np


def make_vector(seed: int = 0) -> list[float]:
    """Return a reproducible 768-dim unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(768).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()
