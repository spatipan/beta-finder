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
import logging
import lzma
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import instaloader
from tqdm import tqdm

RAW_DIR = Path("data/raw")
SCRAPE_DELAY = float(os.getenv("SCRAPING_DELAY", "2.0"))

# Sidecar filename written alongside each downloaded .mp4.
# Contains full post metadata so crash-recovery doesn't need to re-hit the IG API.
SIDECAR_NAME = ".reel.json"

log = logging.getLogger(__name__)

# ── Logging helpers (tqdm-safe) ───────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _log(tag: str, msg: str) -> None:
    tqdm.write(f"{_ts()}  {tag:<12} {msg}")


# ─────────────────────────────────────────────────────────────────────────────

def read_sidecar(raw_dir: Path) -> dict | None:
    """Read the .reel.json sidecar from a raw dir, if it exists.

    Returns the metadata dict (shortcode, account, caption, posted_at, gym_id,
    source, ig_url) or None if missing or corrupt.
    """
    sidecar = raw_dir / SIDECAR_NAME
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return None


def _write_sidecar(raw_dir: Path, reel: "Reel") -> None:
    """Write metadata to .reel.json alongside the downloaded .mp4.

    Called immediately after a successful download so the metadata survives
    a crash and can be used by sweep_raw_dir() without re-hitting the IG API.
    """
    sidecar = raw_dir / SIDECAR_NAME
    sidecar.write_text(json.dumps({
        "shortcode": reel.shortcode,
        "account": reel.account,
        "caption": reel.caption,
        "posted_at": reel.posted_at,
        "gym_id": reel.gym_id,
        "source": reel.source,
        "ig_url": reel.ig_url,
    }))


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
    source: str  # official | tagged | contributor


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
            _log("[session]", f"Loaded CLI session for @{user}")
            return loader
        except Exception:
            pass  # no saved session — fall through

    # 2. Try explicit session file path (INSTALOADER_SESSION_FILE env var).
    #    Falls back to the Safari-exported pickle if no CLI session exists.
    session_file = os.getenv("INSTALOADER_SESSION_FILE")
    if user and session_file:
        try:
            loader.load_session_from_file(user, session_file)
            _log("[session]", f"Loaded pickle session for @{user}")
            return loader
        except Exception as e:
            _log("[session]", f"Session file load failed ({session_file}): {e}")

    # 3. Fall back to username/password (likely 401 on modern Instagram).
    pwd = os.getenv("INSTALOADER_PASS")
    if user and pwd:
        try:
            loader.login(user, pwd)
            _log("[session]", f"Password login succeeded for @{user}")
        except Exception as e:
            _log("[session]", f"Password login failed for @{user}: {e}")

    if not user:
        _log("[session]", "No INSTALOADER_USER set — scraping anonymously (rate limits apply)")

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
        _log("[scraper]", f"Could not load profile @{ig_handle}: {e}")
        return []

    _log("[scraper]", f"@{ig_handle}: {profile.mediacount} total post(s) on profile")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reels: list[Reel] = []
    _default = default_gym or gym_id
    skipped_non_video = 0
    skipped_existing = 0
    resumed = 0

    try:
        for post in profile.get_posts():
            if len(reels) >= max_posts:
                _log("[scraper]", f"@{ig_handle}: reached max_posts={max_posts}, stopping")
                break
            if post.shortcode in existing_ids:
                skipped_existing += 1
                _log("[scraper]", f"@{ig_handle}: hit existing shortcode {post.shortcode} — stopping early")
                break
            if not post.is_video:
                skipped_non_video += 1
                continue

            # Detect gym per-post when scraping contributor accounts
            if gym_hints:
                assigned_gym = detect_gym_from_caption(
                    post.caption or "", gym_hints, _default
                )
                if assigned_gym != _default:
                    _log("[scraper]", f"  {post.shortcode}: gym detected from caption → {assigned_gym}")
            else:
                assigned_gym = gym_id

            # Resume: if raw dir + sidecar already exist, the .mp4 was downloaded
            # in a previous (crashed) run — reuse it without re-downloading.
            raw_dir = RAW_DIR / post.shortcode
            sidecar = read_sidecar(raw_dir)
            mp4_exists = bool(next(raw_dir.glob("*.mp4"), None)) if raw_dir.exists() else False
            if sidecar and mp4_exists:
                mp4 = next(raw_dir.glob("*.mp4"))
                reels.append(Reel(
                    shortcode=post.shortcode,
                    video_path=str(mp4),
                    ig_url=sidecar["ig_url"],
                    caption=sidecar["caption"],
                    posted_at=sidecar["posted_at"],
                    account=sidecar["account"],
                    gym_id=sidecar.get("gym_id", assigned_gym),
                    source=sidecar.get("source", source),
                ))
                resumed += 1
                _log("[scraper]", f"  {post.shortcode}: resumed from previous run (skipping re-download)")
                continue

            try:
                _log("[scraper]", f"  Downloading {post.shortcode}  ({post.date_utc.date()})  @{post.owner_username} …")
                t0 = time.monotonic()
                reel = _download_reel(loader, post, gym_id=assigned_gym, source=source)
                if reel:
                    _write_sidecar(RAW_DIR / post.shortcode, reel)
                    reels.append(reel)
                    _log("[scraper]", f"  {post.shortcode}: downloaded  ({time.monotonic()-t0:.1f}s)")
                else:
                    _log("[scraper]", f"  {post.shortcode}: no .mp4 found after download")
            except Exception as e:
                _log("[scraper]", f"  {post.shortcode}: download failed — {e}")

            time.sleep(SCRAPE_DELAY)
    except Exception as e:
        _log("[scraper]", f"@{ig_handle}: post iteration stopped — {e}")

    _log("[scraper]", (
        f"@{ig_handle}: {len(reels)} reel(s) ready"
        + (f"  ({resumed} resumed, {len(reels)-resumed} freshly downloaded)" if resumed else "")
        + f"  (skipped {skipped_non_video} non-video, {skipped_existing} already indexed)"
    ))
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
        _log("[scraper]", "INSTALOADER_USER not set — skipping tagged scraping")
        return []

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reels: list[Reel] = []

    # Use a persistent staging dir so downloads survive a Ctrl+C interrupt.
    # The CLI writes files here; we move .mp4s to data/raw/ after parsing.
    # The dir is cleaned up only after successful parsing, not on interrupt.
    stage_root = RAW_DIR / "_tagged_stage"
    stage_root.mkdir(parents=True, exist_ok=True)
    tagged_dir = stage_root / tagged_account / ":tagged"

    session_file = os.path.expanduser(f"~/.config/instaloader/session-{user}")
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
    _log("[scraper]", f"CLI tagged: {_CLI_BIN} --tagged @{tagged_account} (max={max_posts})")
    _log("[scraper]", f"  Stage dir: {stage_root / tagged_account}")

    existing_mp4s = len(list(tagged_dir.glob("*.mp4"))) if tagged_dir.exists() else 0
    if existing_mp4s:
        _log("[scraper]", f"  Resuming — {existing_mp4s} .mp4 file(s) already staged from previous run")

    t0 = time.monotonic()
    # --count does not apply to --tagged (only works for hashtag/feed/saved).
    # We use Popen + polling: kill the process once we have enough .mp4 files.
    proc = subprocess.Popen(
        cmd,
        cwd=str(stage_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 600  # 10 min absolute ceiling
    last_count = existing_mp4s
    while proc.poll() is None and time.time() < deadline:
        time.sleep(2)
        mp4_count = len(list(tagged_dir.glob("*.mp4"))) if tagged_dir.exists() else 0
        if mp4_count != last_count:
            _log("[scraper]", f"  @{tagged_account}: {mp4_count} .mp4 file(s) staged …")
            last_count = mp4_count
        if mp4_count >= max_posts:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            _log("[scraper]", f"  @{tagged_account}: reached max_posts={max_posts}, CLI terminated")
            break
    else:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    cli_elapsed = time.monotonic() - t0
    rc = proc.returncode
    _log("[scraper]", f"  @{tagged_account}: CLI finished  rc={rc}  ({cli_elapsed:.1f}s)")

    if not tagged_dir.exists():
        _log("[scraper]", f"  @{tagged_account}: tagged dir not found — CLI likely failed")
        return []

    # Collect (timestamp, shortcode, json_path, mp4_path) tuples
    entries: list[tuple[int, str, Path, Path | None]] = []
    skipped_existing = 0
    skipped_non_video = 0
    parse_errors = 0

    for json_file in tagged_dir.glob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            node = data.get("node", data)
            shortcode = node.get("shortcode", "")
            is_video = node.get("is_video", False)
            ts = node.get("taken_at_timestamp", 0)
            caption_edges = node.get("edge_media_to_caption", {}).get("edges", [])
            caption = caption_edges[0]["node"]["text"] if caption_edges else ""

            if not shortcode or not is_video:
                skipped_non_video += 1
                continue
            if shortcode in existing_ids:
                skipped_existing += 1
                continue

            mp4 = json_file.with_suffix(".mp4")
            entries.append((ts, shortcode, json_file, mp4 if mp4.exists() else None))
        except Exception as e:
            parse_errors += 1
            _log("[scraper]", f"  Could not parse {json_file.name}: {e}")

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
                skipped_non_video += 1
                continue
            if shortcode in existing_ids:
                skipped_existing += 1
                continue

            mp4 = tagged_dir / f"{json_file.stem.replace('.json', '')}.mp4"
            entries.append((ts, shortcode, json_file, mp4 if mp4.exists() else None))
        except Exception as e:
            parse_errors += 1
            _log("[scraper]", f"  Could not parse {json_file.name}: {e}")

    _log("[scraper]", (
        f"  @{tagged_account}: parsed {len(entries)} valid video(s)  "
        f"(skipped {skipped_non_video} non-video, {skipped_existing} already indexed"
        + (f", {parse_errors} parse error(s)" if parse_errors else "") + ")"
    ))

    # Sort by newest first, cap at max_posts
    entries.sort(key=lambda x: x[0], reverse=True)
    entries = entries[:max_posts]

    no_mp4 = 0
    for ts, shortcode, json_file, mp4_tmp in entries:
        if not mp4_tmp:
            no_mp4 += 1
            _log("[scraper]", f"  {shortcode}: no .mp4 — skipping")
            continue

        # Move .mp4 from stage into data/raw/{shortcode}/ (no copy — same filesystem)
        dest_dir = RAW_DIR / shortcode
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_mp4 = dest_dir / f"{shortcode}.mp4"
        shutil.move(str(mp4_tmp), dest_mp4)

        posted_at = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() if ts else ""
        reel = Reel(
            shortcode=shortcode,
            video_path=str(dest_mp4),
            ig_url=f"https://www.instagram.com/reel/{shortcode}",
            caption=caption,
            posted_at=posted_at,
            account=tagged_account,
            gym_id=gym_id,
            source="tagged",
        )
        _write_sidecar(dest_dir, reel)
        reels.append(reel)

    if no_mp4:
        _log("[scraper]", f"  @{tagged_account}: {no_mp4} entry/entries had no .mp4")
    _log("[scraper]", f"  @{tagged_account}: {len(reels)} reel(s) ready for indexing")

    # Clean up the stage dir now that all .mp4s have been moved out
    shutil.rmtree(stage_root / tagged_account, ignore_errors=True)
    _log("[scraper]", f"  Stage cleaned up: {stage_root / tagged_account}")

    return reels


def fetch_post_meta(shortcode: str) -> dict | None:
    """Fetch metadata for a single post by shortcode without downloading the video.

    Used by sweep_raw_dir() to recover the real account name, caption, and
    posted_at for orphaned .mp4 files instead of defaulting to "unknown".

    Returns a dict with keys: account, caption, posted_at (ISO date string).
    Returns None if the post cannot be fetched (private, deleted, rate-limited).
    """
    loader = _loader()
    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
        return {
            "account": post.owner_username,
            "caption": post.caption or "",
            "posted_at": post.date_utc.date().isoformat(),
        }
    except Exception as e:
        _log("[scraper]", f"fetch_post_meta({shortcode}): {e}")
        return None


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
