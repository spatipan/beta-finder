"""Cache thumbnail and keyframe JPEGs for a Reel after indexing.

PRD §4: thumbnail → data/cache/thumb/{reel_id}.jpg (400×711, 9:16)
        keyframes  → data/cache/frames/{reel_id}/{0-3}.jpg
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

CACHE_DIR = Path("data/cache")
THUMB_WIDTH = 400
THUMB_HEIGHT = 711  # 9:16 ratio
THUMB_QUALITY = 85
FRAME_QUALITY = 80


def _scale_and_crop(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale-to-fill then center-crop to w×h (no squeeze/stretch)."""
    fh, fw = frame.shape[:2]
    scale = max(w / fw, h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    return resized[y0 : y0 + h, x0 : x0 + w]


def cache_reel_images(
    reel_id: str,
    top_frames: list[np.ndarray],
    thumb_width: int = THUMB_WIDTH,
    thumb_height: int = THUMB_HEIGHT,
    best_frame: "np.ndarray | None" = None,
) -> dict:
    """Save thumbnail + top-4 keyframes to disk.

    Args:
        reel_id: Instagram shortcode used as filename stem.
        top_frames: BGR numpy arrays in chronological order.
        thumb_width: Thumbnail width in pixels (default from config).
        thumb_height: Thumbnail height in pixels (default from config).
        best_frame: Highest-quality frame to use as thumbnail. If None,
                    falls back to top_frames[0].

    Returns:
        Dict with thumbnail_url and keyframe_urls (API paths).
    """
    if not top_frames:
        raise ValueError("top_frames is empty")

    thumb_dir = CACHE_DIR / "thumb"
    frames_dir = CACHE_DIR / "frames" / reel_id
    thumb_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Thumbnail = best-quality frame, scale-to-fill then center-crop
    thumb_src = best_frame if best_frame is not None else top_frames[0]
    thumb_path = thumb_dir / f"{reel_id}.jpg"
    thumb = _scale_and_crop(thumb_src, thumb_width, thumb_height)
    cv2.imwrite(str(thumb_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])

    # Keyframes = all top frames
    frame_paths: list[str] = []
    for i, frame in enumerate(top_frames):
        path = frames_dir / f"{i}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
        frame_paths.append(str(path))

    return {
        "thumbnail_url": f"/api/thumb/{reel_id}",
        "keyframe_urls": [f"/api/frames/{reel_id}/{i}" for i in range(len(top_frames))],
    }


def thumb_path(reel_id: str) -> Path:
    return CACHE_DIR / "thumb" / f"{reel_id}.jpg"


def frame_path(reel_id: str, n: int) -> Path:
    return CACHE_DIR / "frames" / reel_id / f"{n}.jpg"
