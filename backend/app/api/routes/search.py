"""POST /api/search — upload a wall photo, get matching Reels."""
from __future__ import annotations

import io
import time

from fastapi import APIRouter, Depends, Form, UploadFile, HTTPException
from PIL import Image

from backend.app.api.dependencies import get_embedder
from backend.app.models.schema import SearchResponse, SearchResult
from ml.embedder.wall_filter import is_climbing_wall
from ml.vectordb.search import search as vector_search

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/search", response_model=SearchResponse)
async def search_beta(
    file: UploadFile,
    gym: str = Form(default="all"),
    top_k: int = Form(default=5, ge=1, le=20),
    embedder=Depends(get_embedder),
) -> SearchResponse:
    t0 = time.monotonic()

    # --- Validate upload ---
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # --- CLIP wall filter ---
    is_wall, _ = is_climbing_wall(pil)
    if not is_wall:
        ms = int((time.monotonic() - t0) * 1000)
        return SearchResponse(results=[], query_valid=False, processing_ms=ms)

    # --- DINOv2 embed ---
    vec = embedder.embed(pil).squeeze().tolist()

    # --- Vector search ---
    raw = vector_search(query_vector=vec, gym=gym, top_k=top_k)

    results = [SearchResult(**r) for r in raw]
    ms = int((time.monotonic() - t0) * 1000)
    return SearchResponse(results=results, query_valid=True, processing_ms=ms)
