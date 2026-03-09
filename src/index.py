"""
Index helpers for BetaFinder — load and join post/frame indexes.

Supports both old (gym_index.json) and new (posts_index + frames_index) schemas.
All functions are read-only.
"""

import json
from pathlib import Path
from src.config import get_path


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


def load_posts_index() -> list[dict]:
    """Load posts index (new schema).

    Returns:
        List of post-level records (1 per Instagram post)
        Empty list if file doesn't exist yet (before migration)

    Returns empty list if not migrated yet.
    """
    posts_file = get_path("data_dir") / "posts_index.json"
    if not posts_file.exists():
        return []
    with open(posts_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frames_index() -> list[dict]:
    """Load frames index (new schema).

    Returns:
        List of frame-level records (N per post, 1 per image/keyframe)
        Empty list if file doesn't exist yet (before migration)

    Returns empty list if not migrated yet.
    """
    frames_file = get_path("data_dir") / "frames_index.json"
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


def get_frame_post_context(shortcode: str, posts: list[dict]) -> dict | None:
    """
    Get post-level context for a specific frame.

    Helper for when you have a frame's shortcode and want post metadata.

    Args:
        shortcode: Instagram post ID
        posts: List of post records (from load_posts_index)

    Returns:
        Post record if found, None otherwise
    """
    for post in posts:
        if post["shortcode"] == shortcode:
            return post
    return None
