"""
scrape.py - ดึงรูปจาก Instagram ของยิมปีนผาในเชียงใหม่
ใช้ instaloader (ไม่ต้องการ API key)

Sources:
  1. Official gym accounts  (GYMS dict)
  2. Beta contributors      (contributors.json — community-maintained)

Usage:
    python scrape.py                             # scrape ทุก source
    python scrape.py --gym alpine                # scrape เฉพาะ gym นั้น
    python scrape.py --contributors-only         # scrape เฉพาะ contributor accounts
    python scrape.py --add-contributor username  # เพิ่ม contributor แล้ว scrape เลย
    python scrape.py --list-contributors         # ดูรายชื่อ contributors ทั้งหมด
    python scrape.py --limit 50                  # จำกัดจำนวนรูปต่อ account
"""

import instaloader
import argparse
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---- Config ----------------------------------------------------------------

GYMS = {
    "alpine":      "the_alpine_outpost",
    "mainwall":    "mainwallcnx",
    "progression": "progressionvertical",
}

DATA_DIR            = Path("data/images")
INDEX_FILE          = Path("data/gym_index.json")
CONTRIBUTORS_FILE   = Path("data/contributors.json")

# ---------------------------------------------------------------------------
# Contributors management
# ---------------------------------------------------------------------------

def load_contributors() -> list:
    if not CONTRIBUTORS_FILE.exists():
        return []
    with open(CONTRIBUTORS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_contributors(contributors: list):
    CONTRIBUTORS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONTRIBUTORS_FILE, "w", encoding="utf-8") as f:
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
    return instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=True,
        download_geotags=False,
        download_comments=False,
        save_metadata=True,
        compress_json=False,
        quiet=True,
    )


def scrape_account(username, source_type, source_key,
                   limit=100, delay=2.0, loader=None):
    """
    Scrape รูปจาก 1 Instagram account
    บันทึกลง data/images/{source_type}/{source_key}/

    source_type: "official" | "contributor"
    source_key:  gym key (เช่น "alpine") หรือ username (สำหรับ contributor)
    """
    out_dir = DATA_DIR / source_type / source_key
    out_dir.mkdir(parents=True, exist_ok=True)

    icon = "🏟️" if source_type == "official" else "👤"
    log.info(f"{icon} [{source_type.upper()}] Scraping @{username} → {out_dir}")

    L = loader or get_loader()
    metadata = []
    count = 0

    BETA_KEYWORDS = [
        "climb", "boulder", "route", "beta", "wall", "grade",
        "ปีน", "บอลเดอร์", "เส้น",
        "v0","v1","v2","v3","v4","v5","v6","v7","v8","v9","v10",
    ]

    try:
        profile = instaloader.Profile.from_username(L.context, username)
        log.info(f"   Found: {profile.full_name} | {profile.mediacount} posts")

        for post in profile.get_posts():
            if count >= limit:
                break

            caption = (post.caption or "").lower()
            is_relevant = any(kw in caption for kw in BETA_KEYWORDS)

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
                    "source_type":  source_type,
                    "source_key":   source_key,
                    "gym":          source_key if source_type == "official" else None,
                    "username":     username,
                    "shortcode":    post.shortcode,
                    "filename":     str(filepath),
                    "url":          f"https://www.instagram.com/p/{post.shortcode}/",
                    "caption":      post.caption or "",
                    "date":         post.date_utc.isoformat(),
                    "likes":        post.likes,
                    "is_relevant":  is_relevant,
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
    if INDEX_FILE.exists():
        with open(INDEX_FILE, encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"📂 Loaded existing index: {len(data)} entries")
        return data, {m["filename"] for m in data}
    return [], set()


def save_index(all_metadata):
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
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
    source_group.add_argument(
        "--gym", choices=list(GYMS.keys()) + ["all"], default=None,
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

    # Scrape options
    parser.add_argument("--limit", type=int,   default=100,
                        help="จำนวนรูปสูงสุดต่อ account (default: 100)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="delay ระหว่าง request วินาที (default: 2.0)")

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
        new_meta = scrape_account(username, "contributor", username,
                                  limit=args.limit, delay=args.delay)
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
        gyms = GYMS if (args.gym is None or args.gym == "all") \
               else {args.gym: GYMS[args.gym]}

        for key, uname in gyms.items():
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

    for username, source_type, source_key in tasks:
        new_meta = scrape_account(
            username, source_type, source_key,
            limit=args.limit, delay=args.delay, loader=loader,
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