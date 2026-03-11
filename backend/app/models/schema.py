"""Pydantic response schemas (mirrors frontend src/types.ts)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class SearchResult(BaseModel):
    rank: int
    reel_id: str
    url: str
    thumbnail_url: str
    keyframe_urls: list[str]
    media_type: Literal["keyframe", "image"] = "keyframe"
    gym: str
    source_type: Literal["official", "tagged", "contributor"]
    score: float
    username: str
    caption: str
    date: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_valid: bool
    processing_ms: int


class StatsResponse(BaseModel):
    total: int
    by_gym: dict[str, int]
    by_source: dict[str, int]
    last_updated: str | None = None


class GymInfo(BaseModel):
    id: str
    label: str
    handle: str
    color: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.1.0"
