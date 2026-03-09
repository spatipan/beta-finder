"""
api.py — BetaFinder CNX FastAPI backend

Wraps src/search.py for the React frontend (app.jsx).

Endpoints:
    POST /api/search   — upload image → ranked results
    GET  /api/stats    — index statistics
    GET  /api/thumb/{encoded_path} — serve local image thumbnail

Usage:
    uvicorn api:app --reload --port 8000
"""

import base64
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import tempfile

from src.config import load_config, get_path, get_gym_names
from src.search import search as _search
from src.logger import setup_logger

log = setup_logger(__name__)

app = FastAPI(title="BetaFinder CNX API", version="1.0.0")

# Allow the React dev server (Vite default: 5173) and any local origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# /api/search
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/search")
async def api_search(
    file: UploadFile = File(...),
    gym:  str        = Form(default="all"),
    topK: int        = Form(default=5),
    model: str       = Form(default="ViT-B-32"),
):
    """
    Upload a wall photo → get ranked beta results.

    Returns list of result objects matching app.jsx ResultCard expectations:
        rank, score, gym, url, caption, date,
        sourceType, username, mediaType, keyframeUrls, thumbnailUrl
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save upload to temp file
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        gym_filter = None if gym == "all" else gym
        pretrained = "openai"

        results = _search(
            query_path=tmp_path,
            top_k=topK,
            gym_filter=gym_filter,
            model_name=model,
            pretrained=pretrained,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Load full metadata to enrich results with fields app.jsx needs
    index_file = get_path("index_file")
    meta_by_file: dict = {}
    if index_file.exists():
        raw = json.loads(index_file.read_text())
        for m in raw:
            meta_by_file[m["filename"]] = m

    enriched = []
    for r in results:
        meta = meta_by_file.get(r["filename"], {})

        # Build keyframe URLs if this is a video keyframe set
        # Keyframes share a shortcode: {shortcode}_kf0.jpg .. _kf3.jpg
        keyframe_urls = None
        media_type = meta.get("media_type", "image")
        if media_type == "keyframe":
            shortcode = meta.get("video_shortcode") or meta.get("shortcode", "")
            parent = Path(meta.get("filename", "")).parent
            frames = []
            for i in range(4):
                kf_path = parent / f"{shortcode}_kf{i}.jpg"
                if kf_path.exists():
                    frames.append(_thumb_url(str(kf_path)))
                else:
                    frames.append(None)
            # Only return non-None if at least one frame found
            if any(frames):
                keyframe_urls = frames

        # Thumbnail URL (first keyframe for reels, image file for photos)
        if keyframe_urls:
            thumbnail_url = keyframe_urls[0]
        elif Path(r["filename"]).exists():
            thumbnail_url = _thumb_url(r["filename"])
        else:
            thumbnail_url = None

        enriched.append({
            "rank":          r["rank"],
            "score":         round(r["score"], 4),
            "gym":           r["gym"],
            "url":           r["url"],
            "caption":       r["caption"],
            "date":          _friendly_date(r["date"]),
            "sourceType":    meta.get("source_type", "official"),
            "username":      meta.get("username") or meta.get("tagger_username") or "",
            "mediaType":     media_type,
            "keyframeUrls":  keyframe_urls,
            "thumbnailUrl":  thumbnail_url,
        })

    return JSONResponse(content={"results": enriched})


# ─────────────────────────────────────────────────────────────────────────────
# /api/stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def api_stats():
    """Return index statistics for the Stats tab."""
    index_file = get_path("index_file")
    if not index_file.exists():
        return JSONResponse(content={
            "total": 0, "walls": 0, "contributors": 0,
            "alpine": 0, "mainwall": 0, "progression": 0,
        })

    data = json.loads(index_file.read_text())

    gym_counts: dict = {}
    contributors: set = set()
    for m in data:
        gym = m.get("gym") or m.get("source_key") or "unknown"
        gym_counts[gym] = gym_counts.get(gym, 0) + 1
        username = m.get("tagger_username") or m.get("username") or ""
        if username and m.get("source_type") in ("tagged", "contributor"):
            contributors.add(username)

    return JSONResponse(content={
        "total":        len(data),
        "walls":        len(data),               # all indexed are wall images (post-filter)
        "contributors": len(contributors),
        "alpine":       gym_counts.get("alpine", 0),
        "mainwall":     gym_counts.get("mainwall", 0),
        "progression":  gym_counts.get("progression", 0),
    })


# ─────────────────────────────────────────────────────────────────────────────
# /api/thumb/{encoded_path} — serve local image files to the browser
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/thumb/{encoded_path:path}")
def api_thumb(encoded_path: str):
    """
    Serve a local image file by base64url-encoded path.
    app.jsx calls this URL for thumbnails.
    """
    try:
        file_path = Path(base64.urlsafe_b64decode(encoded_path.encode()).decode())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path encoding")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    # Safety: only serve files inside the data directory
    data_dir = get_path("data_dir").parent  # data/
    try:
        file_path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(str(file_path), media_type="image/jpeg")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _thumb_url(file_path: str) -> str:
    """Encode a local file path as a /api/thumb/ URL."""
    encoded = base64.urlsafe_b64encode(file_path.encode()).decode()
    return f"/api/thumb/{encoded}"


def _friendly_date(iso_date: str) -> str:
    """Convert YYYY-MM-DD to 'Mar 1' style."""
    if not iso_date:
        return ""
    try:
        from datetime import date
        d = date.fromisoformat(iso_date[:10])
        return d.strftime("%b %-d")
    except Exception:
        return iso_date[:10]
