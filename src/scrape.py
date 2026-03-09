"""
scrape.py - ดึงรูปจาก Instagram ของยิมปีนผาในเชียงใหม่
ใช้ instaloader (ไม่ต้องการ API key)

Sources:
  1. Official gym accounts  (from config.yaml)
  2. Beta contributors      (contributors.json — community-maintained)

Usage:
    python scrape.py                             # scrape ทุก source
    python scrape.py --gym alpine                # scrape เฉพาะ gym นั้น
    python scrape.py --gym alpine --mode tagged  # scrape only community-tagged posts
    python scrape.py --mode both                 # own posts + community tagged (all)
    python scrape.py --contributors-only         # scrape เฉพาะ contributor accounts
    python scrape.py --add-contributor username  # เพิ่ม contributor แล้ว scrape เลย
    python scrape.py --list-contributors         # ดูรายชื่อ contributors ทั้งหมด
    python scrape.py --limit 50                  # จำกัดจำนวนรูปต่อ account
"""

import instaloader
import argparse
import json
import time
import itertools
from pathlib import Path

from src.config import load_config, get_path, get_gym_names, get_instagram_handle, get_nested
from src.logger import setup_logger

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Video keyframe extraction (Phase 2)
# ---------------------------------------------------------------------------

def extract_keyframes(video_path: Path, n_frames: int = 4, skip_pct: float = 0.05) -> list:
    """
    Extract N evenly-spaced keyframes from a video file.

    Skips the first and last skip_pct of the video to avoid intros/outros.
    Returns list of Paths for saved JPEG keyframes.
    Deletes the source video if configured.
    """
    try:
        import cv2
    except ImportError:
        log.warning("opencv-python not installed — skipping keyframe extraction")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_frames:
        cap.release()
        return []

    start = int(total * skip_pct)
    end = int(total * (1 - skip_pct))
    positions = [start + i * (end - start) // max(n_frames - 1, 1) for i in range(n_frames)]

    frames = []
    stem = video_path.stem  # e.g. "ABC123_0"
    for i, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        out_path = video_path.parent / f"{stem}_kf{i}.jpg"
        cv2.imwrite(str(out_path), frame)
        frames.append(out_path)
        log.debug(f"   Keyframe {i}: {out_path.name}")

    cap.release()
    log.info(f"   Extracted {len(frames)} keyframes from {video_path.name}")
    return frames

# ---------------------------------------------------------------------------
# Contributors management
# ---------------------------------------------------------------------------

def load_contributors() -> list:
    contrib_file = get_path("contributors_file")
    if not contrib_file.exists():
        return []
    with open(contrib_file, encoding="utf-8") as f:
        return json.load(f)


def save_contributors(contributors: list):
    contrib_file = get_path("contributors_file")
    contrib_file.parent.mkdir(parents=True, exist_ok=True)
    with open(contrib_file, "w", encoding="utf-8") as f:
        json.dump(contributors, f, ensure_ascii=False, indent=2)


def add_contributor(username: str, note: str = "", gyms: list = None):
    """
    เพิ่ม contributor เข้า list
    - username: IG handle (ไม่ต้องใส่ @)
    - note:     หมายเหตุ เช่น "posts alpine beta mostly"
    - gyms:     ยิมที่คนนี้ปีนบ่อย เช่น ["alpine", "mainwall"]
    """
    contributors = load_contributors()
    existing = {c["username"] for c in contributors}
    username = username.lstrip("@").strip()

    if username in existing:
        log.info(f"ℹ️  @{username} already in contributors list")
        return False

    contributors.append({
        "username": username,
        "note":     note,
        "gyms":     gyms or [],
        "active":   True,
        "added":    time.strftime("%Y-%m-%d"),
    })
    save_contributors(contributors)
    log.info(f"✅ Added contributor: @{username}")
    return True


def list_contributors():
    contributors = load_contributors()
    if not contributors:
        print("\n  (no contributors yet — use --add-contributor to add)\n")
        return

    active = [c for c in contributors if c.get("active", True)]
    print(f"\n{'─'*60}")
    print(f"  {'USERNAME':<25} {'GYMS':<20} NOTE")
    print(f"{'─'*60}")
    for c in active:
        gyms_str = ", ".join(c.get("gyms", [])) or "—"
        print(f"  @{c['username']:<24} {gyms_str:<20} {c.get('note','')}")
    print(f"{'─'*60}")
    print(f"  Total: {len(active)} active contributors\n")


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

def get_loader():
    """Create instaloader with config-driven settings"""
    cfg = load_config()
    insta_cfg = cfg.get("scraping", {}).get("instaloader", {})

    return instaloader.Instaloader(
        download_videos=insta_cfg.get("download_videos", False),
        download_video_thumbnails=insta_cfg.get("download_video_thumbnails", True),
        download_geotags=insta_cfg.get("download_geotags", False),
        download_comments=insta_cfg.get("download_comments", False),
        save_metadata=insta_cfg.get("save_metadata", True),
        compress_json=insta_cfg.get("compress_json", False),
        quiet=insta_cfg.get("quiet", True),
    )


def scrape_account(username, source_type, source_key,
                   limit=100, delay=2.0, loader=None, scrape_mode="posts"):
    """
    Scrape รูปจาก 1 Instagram account
    บันทึกลง data/images/{source_type}/{source_key}/

    source_type: "official" | "contributor"
    source_key:  gym key (เช่น "alpine") หรือ username (สำหรับ contributor)
    scrape_mode: "posts" (account's own) | "tagged" (posts tagged by others) | "both"
    """
    base_dir = get_path("data_dir")
    out_dir = base_dir / source_type / source_key
    out_dir.mkdir(parents=True, exist_ok=True)

    icon = "🏟️" if source_type == "official" else "👤"
    log.info(f"{icon} [{source_type.upper()}] Scraping @{username} → {out_dir}")

    L = loader or get_loader()
    metadata = []
    count = 0

    # Load beta keywords from config
    cfg = load_config()
    keywords_config = cfg.get("scraping", {}).get("beta_keywords", {})
    BETA_KEYWORDS = (
        keywords_config.get("english", []) +
        keywords_config.get("thai", []) +
        keywords_config.get("grades", [])
    )

    # Load keyframe extraction config
    kf_cfg = cfg.get("scraping", {}).get("keyframe_extraction", {})
    kf_enabled = kf_cfg.get("enabled", False)
    kf_n_frames = kf_cfg.get("n_frames", 4)
    kf_skip_pct = kf_cfg.get("skip_pct", 0.05)
    kf_delete_video = kf_cfg.get("delete_video", True)

    try:
        profile = instaloader.Profile.from_username(L.context, username)
        log.info(f"   Found: {profile.full_name} | {profile.mediacount} posts")

        # Build post iterator based on scrape_mode (Phase 2.1)
        if scrape_mode == "posts":
            posts_iter = profile.get_posts()
        elif scrape_mode == "tagged":
            posts_iter = profile.get_tagged_posts()
        elif scrape_mode == "both":
            posts_iter = itertools.chain(profile.get_posts(), profile.get_tagged_posts())
        else:
            log.warning(f"Unknown scrape_mode: {scrape_mode}, defaulting to 'posts'")
            posts_iter = profile.get_posts()

        for post in posts_iter:
            if count >= limit:
                break

            caption = (post.caption or "").lower()
            is_relevant = any(kw in caption for kw in BETA_KEYWORDS)
            base_meta = {
                "source_type": source_type,
                "source_key":  source_key,
                "gym":         source_key if source_type == "official" else None,
                "username":    username,
                "shortcode":   post.shortcode,
                "url":         f"https://www.instagram.com/p/{post.shortcode}/",
                "caption":     post.caption or "",
                "date":        post.date_utc.isoformat(),
                "likes":       post.likes,
                "is_relevant": is_relevant,
                "scrape_mode": scrape_mode,
                "tagger_username": post.owner_username if scrape_mode in ("tagged", "both") else None,
            }

            is_video = post.is_video

            if is_video and kf_enabled:
                # Download video then extract keyframes
                video_path = out_dir / f"{post.shortcode}_0.mp4"
                if not video_path.exists():
                    try:
                        L.download_post(post, target=str(out_dir))
                        # instaloader saves as {shortcode}.mp4 — find it
                        mp4_files = list(out_dir.glob(f"{post.shortcode}*.mp4"))
                        if mp4_files:
                            video_path = mp4_files[0]
                    except Exception as e:
                        log.warning(f"   ⚠️  Video download failed {post.shortcode}: {e}")
                        time.sleep(delay)
                        continue

                if video_path.exists():
                    keyframes = extract_keyframes(video_path, kf_n_frames, kf_skip_pct)
                    if kf_delete_video:
                        video_path.unlink(missing_ok=True)

                    for kf_idx, kf_path in enumerate(keyframes):
                        metadata.append({
                            **base_meta,
                            "filename":       str(kf_path),
                            "media_type":     "keyframe",
                            "frame_index":    kf_idx,
                            "video_shortcode": post.shortcode,
                        })
                        count += 1
            else:
                # Image post (or video with keyframes disabled — fall back to image)
                nodes = list(post.get_sidecar_nodes()) if post.typename == "GraphSidecar" else [post]

                for idx, node in enumerate(nodes):
                    filename = f"{post.shortcode}_{idx}.jpg"
                    filepath = out_dir / filename

                    if not filepath.exists():
                        try:
                            node_url = getattr(node, 'display_url', getattr(node, 'url', None))
                            L.download_pic(str(filepath), node_url, post.date_utc)
                            log.info(f"   ✅ {filename}")
                        except Exception as e:
                            log.warning(f"   ⚠️  Failed {filename}: {e}")
                            continue
                    else:
                        log.debug(f"   Skip (exists): {filename}")

                    metadata.append({
                        **base_meta,
                        "filename":   str(filepath),
                        "media_type": "image",
                        "frame_index": None,
                        "video_shortcode": None,
                    })
                    count += 1

            time.sleep(delay)

    except instaloader.exceptions.ProfileNotExistsException:
        log.error(f"❌ @{username} not found")
    except instaloader.exceptions.LoginRequiredException:
        log.error(f"❌ Login required for @{username}")
    except Exception as e:
        log.error(f"❌ Error scraping @{username}: {e}")

    log.info(f"   Done: {count} images from @{username}")
    return metadata


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def load_index():
    index_file = get_path("index_file")
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"📂 Loaded existing index: {len(data)} entries")
        return data, {m["filename"] for m in data}
    return [], set()


def save_index(all_metadata):
    index_file = get_path("index_file")
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)


def merge_and_save(new_meta: list):
    all_metadata, existing_files = load_index()
    added = 0
    for m in new_meta:
        if m["filename"] not in existing_files:
            all_metadata.append(m)
            existing_files.add(m["filename"])
            added += 1
    save_index(all_metadata)
    log.info(f"✅ Saved {len(all_metadata)} entries (+{added} new) → {INDEX_FILE}")
    return all_metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BetaFinder CNX - Instagram Scraper")

    # Source selection (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group()
    gym_choices = get_gym_names() + ["all"]
    source_group.add_argument(
        "--gym", choices=gym_choices, default=None,
        help="scrape เฉพาะ official gym account",
    )
    source_group.add_argument(
        "--contributors-only", action="store_true",
        help="scrape เฉพาะ contributor accounts",
    )
    source_group.add_argument(
        "--contributor", metavar="USERNAME",
        help="scrape contributor account เดียว",
    )

    # Contributor management
    parser.add_argument("--add-contributor", metavar="USERNAME",
                        help="เพิ่ม contributor (ใช้ร่วมกับ --note และ --gyms-tag)")
    parser.add_argument("--note",      default="",
                        help="หมายเหตุสำหรับ contributor ใหม่")
    parser.add_argument("--gyms-tag",  nargs="+", choices=list(GYMS.keys()),
                        help="ยิมที่ contributor นี้ปีน")
    parser.add_argument("--list-contributors", action="store_true",
                        help="แสดงรายชื่อ contributors ทั้งหมด")
    parser.add_argument("--no-scrape", action="store_true",
                        help="แค่ add/list contributor ไม่ต้อง scrape")

    # Scrape options - use config defaults
    scrape_cfg = get_nested("scraping")
    default_limit = scrape_cfg.get("default_limit", 100)
    default_delay = scrape_cfg.get("default_delay", 2.0)

    parser.add_argument("--limit", type=int,   default=default_limit,
                        help=f"จำนวนรูปสูงสุดต่อ account (default: {default_limit})")
    parser.add_argument("--delay", type=float, default=default_delay,
                        help=f"delay ระหว่าง request วินาที (default: {default_delay})")
    parser.add_argument("--mode", choices=["posts", "tagged", "both"], default=None,
                        help="scrape mode: 'posts' (own), 'tagged' (community), 'both' (default: from config)")

    args = parser.parse_args()

    # ── List contributors ──────────────────────────────────────────────────
    if args.list_contributors:
        list_contributors()
        return

    # ── Add contributor ────────────────────────────────────────────────────
    if args.add_contributor:
        add_contributor(args.add_contributor, note=args.note, gyms=args.gyms_tag)
        if args.no_scrape:
            return
        username = args.add_contributor.lstrip("@")
        cfg = load_config()
        scrape_modes = cfg.get("scraping", {}).get("scrape_modes", {})
        mode = args.mode or scrape_modes.get("contributors", "posts")
        new_meta = scrape_account(username, "contributor", username,
                                  limit=args.limit, delay=args.delay, scrape_mode=mode)
        merge_and_save(new_meta)
        return

    # ── Build task list ────────────────────────────────────────────────────
    # tasks: list of (username, source_type, source_key)
    tasks = []

    if args.contributor:
        username = args.contributor.lstrip("@")
        tasks.append((username, "contributor", username))

    elif args.contributors_only:
        for c in load_contributors():
            if c.get("active", True):
                tasks.append((c["username"], "contributor", c["username"]))

    else:
        # default: official gyms + all contributors
        cfg = load_config()
        all_gyms = cfg.get("gyms", {})

        if args.gym is None or args.gym == "all":
            gyms = all_gyms
        else:
            gyms = {args.gym: all_gyms[args.gym]}

        for key, gym_info in gyms.items():
            uname = gym_info["instagram"]
            tasks.append((uname, "official", key))

        # contributors ด้วย ถ้าไม่ได้ระบุ --gym เดียว
        if args.gym is None:
            for c in load_contributors():
                if c.get("active", True):
                    tasks.append((c["username"], "contributor", c["username"]))

    if not tasks:
        log.info("Nothing to scrape. Use --add-contributor USERNAME to add beta contributors.")
        return

    log.info(f"📋 Scraping {len(tasks)} accounts...")

    # ── Scrape all ────────────────────────────────────────────────────────
    loader       = get_loader()
    all_metadata, existing_files = load_index()
    new_this_run = []
    scrape_modes = cfg.get("scraping", {}).get("scrape_modes", {})

    for username, source_type, source_key in tasks:
        # Determine scrape mode (Phase 2.1)
        if args.mode:
            mode = args.mode
        else:
            if source_type == "official":
                mode = scrape_modes.get("official_gyms", "both")
            elif source_type == "contributor":
                mode = scrape_modes.get("contributors", "posts")
            else:
                mode = scrape_modes.get("default", "tagged")

        new_meta = scrape_account(
            username, source_type, source_key,
            limit=args.limit, delay=args.delay, loader=loader,
            scrape_mode=mode,
        )
        for m in new_meta:
            if m["filename"] not in existing_files:
                all_metadata.append(m)
                existing_files.add(m["filename"])
                new_this_run.append(m)

    save_index(all_metadata)

    official_n    = sum(1 for m in all_metadata if m["source_type"] == "official")
    contributor_n = sum(1 for m in all_metadata if m["source_type"] == "contributor")

    log.info(f"\n✅ Index updated → {INDEX_FILE}")
    log.info(f"   Official accounts:  {official_n} images")
    log.info(f"   Contributor accounts: {contributor_n} images")
    log.info(f"   New this run:       {len(new_this_run)} images")
    log.info(f"   Total:              {len(all_metadata)} images")


if __name__ == "__main__":
    main()