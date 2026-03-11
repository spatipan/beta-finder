"""Shared FastAPI dependencies: singleton embedder instance."""
from functools import lru_cache
from ml.embedder.dino import DinoEmbedder


@lru_cache(maxsize=1)
def get_embedder() -> DinoEmbedder:
    """Load DINOv2 once and reuse across requests."""
    return DinoEmbedder()
