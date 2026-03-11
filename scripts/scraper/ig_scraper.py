"""Instagram Reel scraper using Instaloader.

PRD §2.4: Incremental scraping — stops when reaching already-indexed shortcodes.

Tagged scraping note
--------------------
Instagram's graphql/query?query_hash=e31a871f... endpoint (used by
Profile.get_tagged_posts()) returns 401 in Python but works via the CLI.
The CLI (`instaloader --tagged`) routes through the iPhone API instead.
We work around this by shelling out to the system instaloader CLI binary
for tagged post scraping, then parsing the downloaded files.
"""
from __future__ import annotations

import json
import lzma
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import instaloader

RAW_DIR = Path("data/raw")
SCRAPE_DELAY = float(os.getenv("SCRAPING_DELAY", "2.0"))


def detect_gym_from_caption(
    caption: str,
    gym_hints: list[dict],
    default_gym: str,
) -> str:
    """Detect which gym a post belongs to by scanning caption for hints.

    Args:
        caption: Post caption text (may be empty).
        gym_hints: List of dicts from accounts.yaml, each with keys:
                   ``gym_id``, ``handles`` (list[str]), ``hashtags`` (list[str]).
        default_gym: Fallback gym_id when no hint matches.

    Returns:
        gym_id string — one of the known gym IDs or the default.

    Detection logic (first match wins):
      1. @mention of any known gym handle in caption
      2. #hashtag matching any known gym hashtag in caption
    """
    if not caption or not gym_hints:
        return default_gym

    lower = caption.lower()

    for hint in gym_hints:
        gym_id = hint.get("gym_id", "")
        handles = hint.get("handles", [])
        hashtags = hint.get("hashtags", [])

        # Check @mentions
        for handle in handles:
            if f"@{handle.lower()}" in lower:
                return gym_id

        # Check #hashtags
        for tag in hashtags:
            if f"#{tag.lower()}" in lower:
                return gym_id

    return default_gym

# System CLI binary — must be instaloader 4.10 (supports iPhone API for tagged).
# The beta-finder venv's `instaloader` command is the same version.
_CLI_BIN = shutil.which("instaloader") or "instaloader"


@dataclass
class Reel:
    shortcode: str
    video_path: str
    ig_url: str
    caption: str
    posted_at: str
    account: str
    gym_id: str
    source: str  # official | tagged


def _loader() -> instaloader.Instaloader:
    loader = instaloader.Instaloader(
        download_videos=True,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        quiet=True,
        # iphone_support=True is the default and must remain so — the CLI uses
        # the iPhone API endpoint for tagged posts which works around the
        # GraphQL query_hash 401 issues on business accounts.
    )
    user = os.getenv("INSTALOADER_USER")

    # 1. Try the default Instaloader CLI session location first:
    #    ~/.config/instaloader/session-{username}
    #    This is created by: instaloader --load-cookies safari --login {user}
    #    and produces a richer session that works for tagged post scraping.
    if user:
        try:
            loader.load_session_from_file(user)
            print(f"[scraper] Loaded CLI session for @{user}")
            return loader
        except Exception:
            pass  # no saved session — fall through

    # 2. Try explicit session file path (INSTALOADER_SESSION_FILE env var).
    #    Falls back to the Safari-exported pickle if no CLI session exists.
    session_file = os.getenv("INSTALOADER_SESSION_FILE")
    if user and session_file:
        try:
            loader.load_session_from_file(user, session_file)
            print(f"[scraper] Loaded pickle session for @{user}")
            return loader
        except Exception as e:
            print(f"[scraper] Session file load failed ({session_file}): {e}")

    # 3. Fall back to username/password (likely 401 on modern Instagram).
    pwd = os.getenv("INSTALOADER_PASS")
    if user and pwd:
        try:
            loader.login(user, pwd)
        except Exception as e:
            print(f"[scraper] Password login failed: {e}")

    return loader


def scrape_new_reels(
    gym_id: str,
    ig_handle: str,
    existing_ids: set[str],
    max_posts: int = 100,
    gym_hints: list[dict] | None = None,
    default_gym: str | None = None,
    source: str = "official",
) -> list[Reel]:
    """Scrape posts from an IG account.

    For official gym accounts, gym_id is fixed and known.
    For contributor accounts, pass gym_hints + default_gym so each post's
    gym is auto-detected from its caption (@mentions / #hashtags).

    Args:
        gym_id: Fixed gym to assign to all posts. Ignored when gym_hints is set.
        ig_handle: Instagram handle to scrape.
        existing_ids: Shortcodes already in the index — stop early when hit.
        max_posts: Maximum number of Reels to download.
        gym_hints: If provided, detect gym per-post from caption. Each item:
                   {gym_id, handles: [...], hashtags: [...]}.
        default_gym: Fallback gym_id used when gym_hints is set but no hint
                     matches. Defaults to gym_id if not given.
        source: "official" or "contributor".
    """
    loader = _loader()
    try:
        profile = instaloader.Profile.from_username(loader.context, ig_handle)
    except Exception as e:
        print(f"[scraper] Could not load profile @{ig_handle}: {e}")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reels: list[Reel] = []
    _default = default_gym or gym_id

    try:
        for post in profile.get_posts():
            if len(reels) >= max_posts:
                break
            if post.shortcode in existing_ids:
                break  # stop early — older posts already indexed
            if not post.is_video:
                continue

            # Detect gym per-post when scraping contributor accounts
            if gym_hints:
                assigned_gym = detect_gym_from_caption(
                    post.caption or "", gym_hints, _default
                )
            else:
                assigned_gym = gym_id

            try:
                reel = _download_reel(loader, post, gym_id=assigned_gym, source=source)
                if reel:
                    reels.append(reel)
            except Exception as e:
                print(f"[scraper] Skip {post.shortcode}: {e}")

            time.sleep(SCRAPE_DELAY)
    except Exception as e:
        print(f"[scraper] Official posts iteration stopped for @{ig_handle}: {e}")

    return reels


def scrape_tagged_reels(
    gym_id: str,
    tagged_account: str,
    existing_ids: set[str],
    max_posts: int = 50,
) -> list[Reel]:
    """Scrape Reels posted by climbers who tagged the gym.

    Uses the instaloader CLI (--tagged) instead of the Python API because
    Instagram's graphql tagged-posts endpoint returns 401 from Python while
    the CLI successfully routes through the iPhone API.
    """
    user = os.getenv("INSTALOADER_USER")
    if not user:
        print("[scraper] INSTALOADER_USER not set — skipping tagged scraping")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reels: list[Reel] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        session_file = os.path.expanduser(f"~/.config/instaloader/session-{user}")
        tagged_dir = Path(tmpdir) / tagged_account / ":tagged"
        cmd = [
            _CLI_BIN,
            "--login", user,
            "--sessionfile", session_file,
            "--no-posts",          # skip official posts
            "--tagged",            # only tagged posts
            "--no-profile-pic",
            "--no-captions",       # skip .txt files
            "--no-compress-json",  # plain JSON for easy parsing
            tagged_account,
        ]
        print(f"[scraper] CLI tagged: {' '.join(cmd)}")

        # --count does not apply to --tagged (only works for hashtag/feed/saved).
        # We use Popen + polling: kill the process once we have enough .mp4 files.
        proc = subprocess.Popen(
            cmd,
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 600  # 10 min absolute ceiling
        while proc.poll() is None and time.time() < deadline:
            time.sleep(2)
            mp4_count = len(list(tagged_dir.glob("*.mp4"))) if tagged_dir.exists() else 0
            if mp4_count >= max_posts:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
        else:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        if not tagged_dir.exists():
            print(f"[scraper] No tagged dir found — CLI may have failed for @{tagged_account}")
            return []

        # Collect (timestamp, shortcode, json_path, mp4_path) tuples
        entries: list[tuple[int, str, Path, Path | None]] = []
        for json_file in tagged_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                node = data.get("node", data)
                shortcode = node.get("shortcode", "")
                is_video = node.get("is_video", False)
                ts = node.get("taken_at_timestamp", 0)
                owner = node.get("owner", {}).get("username", tagged_account)
                caption_edges = node.get("edge_media_to_caption", {}).get("edges", [])
                caption = caption_edges[0]["node"]["text"] if caption_edges else ""

                if not shortcode or not is_video:
                    continue
                if shortcode in existing_ids:
                    continue

                mp4 = json_file.with_suffix(".mp4")
                entries.append((ts, shortcode, json_file, mp4 if mp4.exists() else None))
            except Exception as e:
                print(f"[scraper] Could not parse {json_file.name}: {e}")

        # Also handle .json.xz (in case --no-compress-json didn't apply)
        for json_file in tagged_dir.glob("*.json.xz"):
            try:
                with lzma.open(json_file) as f:
                    data = json.load(f)
                node = data.get("node", data)
                shortcode = node.get("shortcode", "")
                is_video = node.get("is_video", False)
                ts = node.get("taken_at_timestamp", 0)
                caption_edges = node.get("edge_media_to_caption", {}).get("edges", [])
                caption = caption_edges[0]["node"]["text"] if caption_edges else ""

                if not shortcode or not is_video:
                    continue
                if shortcode in existing_ids:
                    continue

                mp4 = tagged_dir / f"{json_file.stem.replace('.json', '')}.mp4"
                entries.append((ts, shortcode, json_file, mp4 if mp4.exists() else None))
            except Exception as e:
                print(f"[scraper] Could not parse {json_file.name}: {e}")

        # Sort by newest first, cap at max_posts
        entries.sort(key=lambda x: x[0], reverse=True)
        entries = entries[:max_posts]

        for ts, shortcode, json_file, mp4_tmp in entries:
            if not mp4_tmp:
                print(f"  [skip] {shortcode}: no .mp4 downloaded")
                continue

            # Move mp4 into RAW_DIR/{shortcode}/
            dest_dir = RAW_DIR / shortcode
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_mp4 = dest_dir / f"{shortcode}.mp4"
            shutil.copy2(mp4_tmp, dest_mp4)

            posted_at = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() if ts else ""
            reels.append(Reel(
                shortcode=shortcode,
                video_path=str(dest_mp4),
                ig_url=f"https://www.instagram.com/reel/{shortcode}",
                caption=caption,
                posted_at=posted_at,
                account=tagged_account,
                gym_id=gym_id,
                source="tagged",
            ))

    return reels


def _download_reel(
    loader: instaloader.Instaloader,
    post: instaloader.Post,
    gym_id: str,
    source: str,
) -> Reel | None:
    """Download a single video post and return a Reel dataclass.

    Uses dirname_pattern + filename_pattern override to bypass Instaloader's
    path sanitization, which replaces '/' with '∕' (U+2215) and creates
    mangled directory names when the target path contains slashes.
    """
    out_dir = RAW_DIR / post.shortcode
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bypass Instaloader path sanitization: set patterns directly so the
    # target="" call writes to exactly out_dir/{shortcode}.mp4
    old_dirname = loader.dirname_pattern
    old_filename = getattr(loader, "filename_pattern", "{date_utc}_UTC")
    try:
        loader.dirname_pattern = str(out_dir).replace("{", "{{").replace("}", "}}")
        loader.filename_pattern = "{shortcode}"
        loader.download_post(post, target="")
    finally:
        loader.dirname_pattern = old_dirname
        loader.filename_pattern = old_filename

    # Find the downloaded .mp4 (named {shortcode}.mp4)
    video_files = list(out_dir.glob(f"{post.shortcode}*.mp4"))
    if not video_files:
        return None

    return Reel(
        shortcode=post.shortcode,
        video_path=str(video_files[0]),
        ig_url=f"https://www.instagram.com/reel/{post.shortcode}",
        caption=post.caption or "",
        posted_at=post.date_utc.date().isoformat(),
        account=post.owner_username,
        gym_id=gym_id,
        source=source,
    )
