"""Thin wrapper that exposes is_climbing_wall at the video module level."""
from ml.embedder.wall_filter import is_climbing_wall

__all__ = ["is_climbing_wall"]
