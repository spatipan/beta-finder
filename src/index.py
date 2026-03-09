"""
Index helpers for BetaFinder — load, save, and join post/frame indexes.

Supports both old (gym_index.json) and new (posts_index + frames_index) schemas.
"""

import json
from pathlib import Path
from src.config import get_path
from src.logger import setup_logger

log = setup_logger(__name__)


def load_gym_index() -> list[dict]:
    """Load the old single-file index (gym_index.json).

    Returns:
        List of metadata records (file-level, one per keyframe/image)

    Raises:
        FileNotFoundError: If gym_index.json doesn't exist
    """
    index_file = get_path("index_file")
    with open(index_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _data_root() -> Path:
    """Derive the data/ root dir from the index_file path (data/gym_index.json → data/)."""
    return get_path("index_file").parent


def load_posts_index() -> list[dict]:
    """Load posts index (new schema).

    Returns:
        List of post-level records (1 per Instagram post)
        Empty list if file doesn't exist yet (before migration)
    """
    posts_file = _data_root() / "posts_index.json"
    if not posts_file.exists():
        return []
    with open(posts_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frames_index() -> list[dict]:
    """Load frames index (new schema).

    Returns:
        List of frame-level records (N per post, 1 per image/keyframe)
        Empty list if file doesn't exist yet (before migration)
    """
    frames_file = _data_root() / "frames_index.json"
    if not frames_file.exists():
        return []
    with open(frames_file, "r", encoding="utf-8") as f:
        return json.load(f)


def join_post_frame(posts: list[dict], frames: list[dict]) -> dict[str, dict]:
    """
    Join posts_index and frames_index by shortcode.

    Enables access to post-level info (caption, url, gyms, etc.) when working
    with frame-level records.

    Args:
        posts: List of post records (from load_posts_index)
        frames: List of frame records (from load_frames_index)

    Returns:
        Dict mapping shortcode → (post_record, [frame_records])
        Example:
        {
            "ABC123": {
                "post": {...post fields...},
                "frames": [{...frame0...}, {...frame1...}, ...]
            }
        }
    """
    # Build post lookup
    posts_by_shortcode = {p["shortcode"]: p for p in posts}

    # Group frames by shortcode
    result = {}
    for frame in frames:
        shortcode = frame["shortcode"]
        post = posts_by_shortcode.get(shortcode)

        if shortcode not in result:
            result[shortcode] = {
                "post": post,
                "frames": []
            }

        result[shortcode]["frames"].append(frame)

    return result


def save_frames_index(frames: list[dict]) -> Path:
    """
    Save frames_index.json to disk.

    Centralizes the JSON write pattern used by embed.py and filter.py.

    Args:
        frames: List of frame records to save

    Returns:
        Path to saved file
    """
    frames_file = _data_root() / "frames_index.json"
    with open(frames_file, "w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False, indent=2)
    return frames_file
