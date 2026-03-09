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
import os
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
    """Create instaloader with config-driven settings and optional credential login"""
    cfg = load_config()
    insta_cfg = cfg.get("scraping", {}).get("instaloader", {})

    L = instaloader.Instaloader(
        download_videos=insta_cfg.get("download_videos", False),
        download_video_thumbnails=insta_cfg.get("download_video_thumbnails", True),
        download_geotags=insta_cfg.get("download_geotags", False),
        download_comments=insta_cfg.get("download_comments", False),
        save_metadata=insta_cfg.get("save_metadata", True),
        compress_json=insta_cfg.get("compress_json", False),
        quiet=insta_cfg.get("quiet", True),
    )

    username = os.environ.get("INSTALOADER_USER")
    password = os.environ.get("INSTALOADER_PASS")
    
    if username:
        try:
            L.load_session_from_file(username)
            log.info(f"🔐 Loaded session for @{username} from file")
        except FileNotFoundError:
            log.info(f"ℹ️  No session file found for @{username}, attempting password login...")
            if password:
                try:
                    L.login(username, password)
                    log.info(f"🔐 Logged in as @{username}")
                    L.save_session_to_file()
                except Exception as e:
                    log.warning(f"⚠️  Login failed ({e}) — continuing without auth")
            else:
                log.warning("⚠️  No session file and no password provided — continuing without auth")
        except Exception as e:
            log.warning(f"⚠️  Failed to load session ({e}) — continuing without auth")
    else:
        log.info("ℹ️  No Instagram credentials set — running unauthenticated")

    return L


import os
from PIL import Image

def scrape_account(username, source_type, source_key,
                   limit=100, delay=2.0, loader=None, scrape_mode="posts",
                   gym_tags=None, scraped_shortcodes: set = None):
    """
    Scrape รูปจาก 1 Instagram account
    บันทึกลง data/images/{source_type}/{source_key}/

    source_type:       "official" | "contributor"
    source_key:        gym key (เช่น "alpine") หรือ username (สำหรับ contributor)
    scrape_mode:       "posts" (account's own) | "tagged" (posts tagged by others) | "both"
    scraped_shortcodes: set of post shortcodes already in the index — skip them entirely
    """
    if source_type == "official":
        out_dir = get_path("official_dir") / source_key
    else:
        out_dir = get_path("contributors_dir") / source_key

    if not out_dir.exists():
        log.info(f"ℹ️  Creating directory: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        log.info(f"ℹ️  Directory already exists: {out_dir}")

    icon = "🔖" if source_type == "official" else "👤"
    log.info(f"{icon} [{source_type.upper()}] Scraping @{username} → {out_dir}")

    L = loader or get_loader()
    metadata = []
    count = 0
    skipped = 0
    already_scraped = scraped_shortcodes or set()

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

    def is_valid_image(filepath):
        """ตรวจสอบว่าไฟล์รูปภาพ valid และมีขนาดมากกว่า 0"""
        try:
            if not filepath.exists():
                return False
            if filepath.stat().st_size == 0:
                log.warning(f"   ⚠️  Empty file detected: {filepath.name}")
                return False
            # ลอง open ด้วย PIL เพื่อ verify
            with Image.open(filepath) as img:
                img.verify()
            return True
        except Exception as e:
            log.warning(f"   ⚠️  Invalid image {filepath.name}: {e}")
            return False

    def _download_and_process_video(post, base_meta):
        nonlocal count
        video_path = out_dir / f"{post.shortcode}.mp4"
                
        if not video_path.exists():
            try:
                old_dirname = L.dirname_pattern
                old_filename = getattr(L, 'filename_pattern', '{date_utc}_UTC')
                # bypass instaloader target path sanitization (which replaces / with ∕)
                L.dirname_pattern = str(out_dir).replace("{", "{{").replace("}", "}}")
                L.filename_pattern = "{shortcode}"
                L.download_post(post, target="")
                L.dirname_pattern = old_dirname
                L.filename_pattern = old_filename

                # instaloader saves as {shortcode}.mp4 — find it
                mp4_files = list(out_dir.glob(f"{post.shortcode}*.mp4"))
                if mp4_files:
                    video_path = mp4_files[0]
                else:
                    log.warning(f"   ⚠️  Video file not found after download: {post.shortcode}")
                    return
            except Exception as e:
                log.warning(f"   ⚠️  Video download failed {post.shortcode}: {e}")
                return

        if video_path.exists():
            keyframes = extract_keyframes(video_path, kf_n_frames, kf_skip_pct)
                    
            if kf_delete_video:
                video_path.unlink(missing_ok=True)
                # Clean up instaloader's metadata JSON files
                for json_file in out_dir.glob(f"{post.shortcode}*.json"):
                    json_file.unlink(missing_ok=True)

            for kf_idx, kf_path in enumerate(keyframes):
                metadata.append({
                    **base_meta,
                    "filename":       str(kf_path),
                    "media_type":     "keyframe",
                    "frame_index":    kf_idx,
                    "video_shortcode": post.shortcode,
                })
                count += 1

    def _download_and_process_images(post, base_meta):
        nonlocal count
        # Image post (or video with keyframes disabled — fall back to image)
        nodes = list(post.get_sidecar_nodes()) if post.typename == "GraphSidecar" else [post]

        for idx, node in enumerate(nodes):
            filename = f"{post.shortcode}_{idx}.jpg"
            filepath = out_dir / filename
                    
            if is_valid_image(filepath):
                log.debug(f"   ⏭️  Skip (exists & valid): {filename}")
            else:
                # Remove corrupt file if it exists
                if filepath.exists():
                    log.warning(f"   🗑️  Removing invalid file: {filename}")
                    filepath.unlink()
                        
                # Download new
                try:
                    node_url = getattr(node, 'display_url', getattr(node, 'url', None))
                    # download_pic uses path WITHOUT extension
                    filepath_no_ext = out_dir / f"{post.shortcode}_{idx}"
                    L.download_pic(str(filepath_no_ext), node_url, post.date_utc)
                            
                    if not filepath.exists():
                        log.warning(f"   ⚠️  File not created after download: {filename}")
                        continue
                            
                    if not is_valid_image(filepath):
                        log.warning(f"   ⚠️  Downloaded file is invalid: {filename}")
                        filepath.unlink(missing_ok=True)
                        continue
                            
                    log.info(f"   ✅ {filename}")
                            
                except Exception as e:
                    log.warning(f"   ⚠️  Failed {filename}: {e}")
                    continue

            # Add to metadata only if valid
            if filepath.exists() and is_valid_image(filepath):
                metadata.append({
                    **base_meta,
                    "filename":   str(filepath),
                    "media_type": "image",
                    "frame_index": None,
                    "video_shortcode": None,
                })
                count += 1

    try:
        profile = instaloader.Profile.from_username(L.context, username)
        log.info(f"   Found: {profile.full_name} | {profile.mediacount} posts")

        if scrape_mode == "posts":
            posts_iter = profile.get_posts()
            log.info(f"   Scraping {profile.mediacount} posts")
        elif scrape_mode == "tagged":
            posts_iter = profile.get_tagged_posts()
            log.info(f"   Scraping tagged posts")
        elif scrape_mode == "both":
            posts_iter = itertools.chain(profile.get_posts(), profile.get_tagged_posts())
            log.info(f"   Scraping posts and tagged posts")
        else:
            log.warning(f"Unknown scrape_mode: {scrape_mode}, defaulting to 'posts'")
            posts_iter = profile.get_posts()

        for post in posts_iter:
            if count >= limit:
                break

            if post.shortcode in already_scraped:
                log.debug(f"   ⏭️  Already scraped: {post.shortcode}")
                skipped += 1
                continue

            caption = (post.caption or "").lower()
            base_meta = {
                "source_type": source_type,
                "source_key":  source_key,
                "gym":         source_key if source_type == "official" else (gym_tags[0] if gym_tags else None),
                "gyms":        [source_key] if source_type == "official" else (gym_tags or []),
                "username":    username,
                "shortcode":   post.shortcode,
                "url":         f"https://www.instagram.com/p/{post.shortcode}/",
                "caption":     post.caption or "",
                "date":        post.date_utc.isoformat(),
                "likes":       post.likes,
                "is_relevant": any(kw in caption for kw in BETA_KEYWORDS),
                "scrape_mode": scrape_mode,
                "tagger_username": post.owner_username if scrape_mode in ("tagged", "both") else None,
            }

            if post.is_video and kf_enabled:
                _download_and_process_video(post, base_meta)
            else:
                _download_and_process_images(post, base_meta)

            time.sleep(delay)

    except instaloader.exceptions.ProfileNotExistsException:
        log.error(f"❌ @{username} not found")
    except instaloader.exceptions.LoginRequiredException:
        log.error(f"❌ Login required for @{username}")
    except Exception as e:
        log.error(f"❌ Error scraping @{username}: {e}")

    log.info(f"   Done: {count} new images, {skipped} posts already scraped (skipped) from @{username}")
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
    log.info(f"✅ Saved {len(all_metadata)} entries (+{added} new) → {get_path('index_file')}")
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
    parser.add_argument("--gyms-tag",  nargs="+", choices=get_gym_names(),
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
    # tasks: list of (username, source_type, source_key, gym_tags)
    tasks = []

    if args.contributor:
        username = args.contributor.lstrip("@")
        # look up gym_tags from contributors.json if available
        contribs = {c["username"]: c for c in load_contributors()}
        gym_tags = contribs.get(username, {}).get("gyms", [])
        tasks.append((username, "contributor", username, gym_tags))

    elif args.contributors_only:
        for c in load_contributors():
            if c.get("active", True):
                tasks.append((c["username"], "contributor", c["username"], c.get("gyms", [])))

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
            tasks.append((uname, "official", key, []))

        # contributors ด้วย ถ้าไม่ได้ระบุ --gym เดียว
        if args.gym is None:
            for c in load_contributors():
                if c.get("active", True):
                    tasks.append((c["username"], "contributor", c["username"], c.get("gyms", [])))

    if not tasks:
        log.info("Nothing to scrape. Use --add-contributor USERNAME to add beta contributors.")
        return

    log.info(f"📋 Scraping {len(tasks)} accounts...")

    # ── Scrape all ────────────────────────────────────────────────────────
    cfg          = load_config()
    loader       = get_loader()
    all_metadata, existing_files = load_index()
    new_this_run = []
    scrape_modes = cfg.get("scraping", {}).get("scrape_modes", {})

    # Build per-account set of already-scraped shortcodes from the index
    scraped_by_account: dict[str, set] = {}
    for m in all_metadata:
        uname = m.get("username", "")
        scraped_by_account.setdefault(uname, set()).add(m["shortcode"])

    for username, source_type, source_key, gym_tags in tasks:
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
            scrape_mode=mode, gym_tags=gym_tags,
            scraped_shortcodes=scraped_by_account.get(username, set()),
        )
        for m in new_meta:
            if m["filename"] not in existing_files:
                all_metadata.append(m)
                existing_files.add(m["filename"])
                new_this_run.append(m)
                # Keep per-account set in sync so later tasks see the update
                scraped_by_account.setdefault(m.get("username", ""), set()).add(m["shortcode"])

    save_index(all_metadata)

    official_n    = sum(1 for m in all_metadata if m["source_type"] == "official")
    contributor_n = sum(1 for m in all_metadata if m["source_type"] == "contributor")

    log.info(f"\n✅ Index updated → {get_path('index_file')}")
    log.info(f"   Official accounts:  {official_n} images")
    log.info(f"   Contributor accounts: {contributor_n} images")
    log.info(f"   New this run:       {len(new_this_run)} images")
    log.info(f"   Total:              {len(all_metadata)} images")


if __name__ == "__main__":
    main()