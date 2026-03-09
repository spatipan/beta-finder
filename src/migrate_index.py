"""
migrate_index.py — One-time migration from flat gym_index.json to two-level schema.

Reads:   data/gym_index.json    (1,223 flat file-level records)
Writes:  data/posts_index.json  (1 record per Instagram post)
         data/frames_index.json (1 record per frame/image, FK=shortcode)

Non-destructive: does NOT modify gym_index.json.
Safe to re-run: overwrites output files each time.

Usage:
    python -c "from src.migrate_index import migrate; migrate()"
    python migrate_index.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, get_path, get_gym_names
from src.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Gym detection from caption (backfill for existing records)
# ---------------------------------------------------------------------------

def _build_gym_signals() -> dict[str, list[str]]:
    """Build gym signal patterns from config (Instagram handles + common hashtags)."""
    cfg = load_config()
    gyms = cfg.get("gyms", {})
    signals = {}

    for gym_key, gym_info in gyms.items():
        handle = gym_info.get("instagram", "")
        signals[gym_key] = [
            f"@{handle}",
        ]

    # Add well-known hashtags per gym (extend as needed)
    _extra = {
        "alpine":      ["#alpineoutpost", "#alpinecnx", "alpine outpost"],
        "mainwall":    ["#mainwall", "#mainwallcnx", "main wall"],
        "progression": ["#progressionvertical", "#progressioncnx", "progression vertical"],
    }
    for gym_key, extras in _extra.items():
        if gym_key in signals:
            signals[gym_key].extend(extras)

    return signals


GYM_SIGNALS = _build_gym_signals()


def detect_gyms_from_caption(caption: str, source_type: str, source_key: str) -> tuple[list[str], str]:
    """
    Detect gym keys from caption text using signal matching.

    For official accounts: use source_key directly (authoritative).
    For contributors: parse caption for mentions/hashtags.

    Returns:
        (gyms, gyms_detected_from) — e.g. (["alpine"], "caption_mention")
    """
    if source_type == "official":
        return [source_key], "official_account"

    text = (caption or "").lower()

    found = []
    for gym_key, signals in GYM_SIGNALS.items():
        if any(sig.lower() in text for sig in signals):
            found.append(gym_key)

    if found:
        return found, "caption_mention"

    return [], "none"


# ---------------------------------------------------------------------------
# Migration core
# ---------------------------------------------------------------------------

def migrate(gym_index_path: Path | None = None) -> tuple[int, int]:
    """
    Migrate gym_index.json → posts_index.json + frames_index.json.

    Args:
        gym_index_path: Override path to gym_index.json (default: from config)

    Returns:
        (n_posts, n_frames) tuple
    """
    load_config()

    # Paths
    index_file  = gym_index_path or get_path("index_file")
    data_root   = index_file.parent
    posts_file  = data_root / "posts_index.json"
    frames_file = data_root / "frames_index.json"

    if not index_file.exists():
        log.error(f"❌ {index_file} not found. Run scrape.py first.")
        return 0, 0

    log.info(f"📂 Loading {index_file}...")
    with open(index_file, encoding="utf-8") as f:
        flat_records = json.load(f)

    log.info(f"   {len(flat_records)} flat records found")

    # Group records by shortcode to build post-level view
    by_shortcode: dict[str, list[dict]] = {}
    for rec in flat_records:
        sc = rec.get("shortcode")
        if sc:
            by_shortcode.setdefault(sc, []).append(rec)

    log.info(f"   {len(by_shortcode)} unique posts detected")

    migrated_at = datetime.now(timezone.utc).isoformat()

    posts_index  = []
    frames_index = []

    for shortcode, recs in by_shortcode.items():
        # Use first record as source of post-level truth
        r0 = recs[0]

        caption     = r0.get("caption", "")
        source_type = r0.get("source_type", "")
        source_key  = r0.get("source_key", "")

        gyms, gyms_detected_from = detect_gyms_from_caption(caption, source_type, source_key)

        # Determine media type at post level
        frame_count = len(recs)
        is_video    = any(rec.get("media_type") == "keyframe" for rec in recs)
        post_media  = "video" if is_video else "image"

        post_rec = {
            "shortcode":           shortcode,
            "url":                 r0.get("url", f"https://www.instagram.com/p/{shortcode}/"),
            "username":            r0.get("username", ""),
            "source_type":         source_type,
            "source_key":          source_key,
            "date":                r0.get("date", ""),
            "likes":               r0.get("likes", 0),
            "caption":             caption,
            "is_relevant":         r0.get("is_relevant", False),
            "scrape_mode":         r0.get("scrape_mode", "posts"),
            "tagger_username":     r0.get("tagger_username", None),
            # Gym detection (backfilled from caption)
            "gyms":                gyms,
            "gyms_detected_from":  gyms_detected_from,
            # Media summary
            "media_type":          post_media,
            "frame_count":         frame_count,
            "best_frame_index":    None,         # filled by filter.py later
            # Tracking
            "scraped_at":          migrated_at,
            "last_updated":        migrated_at,
        }
        posts_index.append(post_rec)

        # One frame record per original flat record
        for rec in recs:
            frame_rec = {
                "filename":      rec.get("filename", ""),
                "shortcode":     shortcode,
                "frame_index":   rec.get("frame_index", None),
                "media_type":    rec.get("media_type", "image"),
                # Filter fields (populated by filter.py)
                "is_wall":       None,
                "wall_score":    None,
                "occlusion_score": None,
                "is_best_frame": False,
                # Embedding fields (populated by embed.py)
                "clip_embedded": False,
                "faiss_id":      None,
                "skip_embed":    False,
            }
            frames_index.append(frame_rec)

    # Save outputs
    with open(posts_file, "w", encoding="utf-8") as f:
        json.dump(posts_index, f, ensure_ascii=False, indent=2)
    log.info(f"✅ posts_index.json → {len(posts_index)} posts")

    with open(frames_file, "w", encoding="utf-8") as f:
        json.dump(frames_index, f, ensure_ascii=False, indent=2)
    log.info(f"✅ frames_index.json → {len(frames_index)} frames")

    # Summary
    gyms_detected = sum(1 for p in posts_index if p["gyms"])
    gyms_empty    = len(posts_index) - gyms_detected
    log.info(f"\n📊 Migration Summary:")
    log.info(f"   Posts with gym detected: {gyms_detected}")
    log.info(f"   Posts with no gym (empty): {gyms_empty}")
    log.info(f"   Total posts:  {len(posts_index)}")
    log.info(f"   Total frames: {len(frames_index)}")
    log.info(f"   Avg frames/post: {len(frames_index)/len(posts_index):.1f}")

    return len(posts_index), len(frames_index)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n_posts, n_frames = migrate()
    print(f"\n✅ Done: {n_posts} posts, {n_frames} frames")
