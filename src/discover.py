"""
discover.py — Snowball account discovery for BetaFinder (Phase 2)

Traverses the Instagram social graph starting from seed accounts (official gyms
+ manual contributors) to automatically discover new beta creators.

Discovery strategy:
  1. Extract @mentions from captions of known accounts
  2. Score each mentioned account by climbing keyword relevance
  3. Add high-scoring accounts to suggestion list
  4. BFS up to max_depth hops from seeds

Usage:
    python discover.py                         # discover from all seeds
    python discover.py --depth 1               # 1 hop only (faster)
    python discover.py --min-score 0.5         # stricter relevance filter
    python discover.py --add-suggested         # add all suggestions to contributors
    python discover.py --dry-run               # show suggestions without saving
"""

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

from src.config import load_config, get_path, get_gym_names, get_instagram_handle, get_nested
from src.logger import setup_logger
from src.scrape import load_contributors

log = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Mention extraction
# ---------------------------------------------------------------------------

def extract_mentions(text: str) -> list:
    """Extract all @username mentions from a caption or bio."""
    if not text:
        return []
    return [m.lower() for m in re.findall(r"@(\w+)", text)]


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def score_account(username: str, captions: list, keywords: list) -> float:
    """
    Score an account's beta relevance based on keyword frequency in captions.

    Returns float [0.0, 1.0] — ratio of posts containing climbing keywords.
    """
    if not captions:
        return 0.0

    matching = sum(
        1 for cap in captions
        if any(kw in (cap or "").lower() for kw in keywords)
    )
    return round(matching / len(captions), 4)


# ---------------------------------------------------------------------------
# Graph persistence
# ---------------------------------------------------------------------------

def load_graph() -> dict:
    """Load existing adjacency graph from data/graph.json."""
    cfg = load_config()
    graph_file = Path(cfg.get("discovery", {}).get("graph_file", "data/graph.json"))
    if graph_file.exists():
        with open(graph_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_graph(graph: dict) -> None:
    """Save adjacency graph to data/graph.json."""
    cfg = load_config()
    graph_file = Path(cfg.get("discovery", {}).get("graph_file", "data/graph.json"))
    graph_file.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    log.info(f"Graph saved: {len(graph)} nodes → {graph_file}")


# ---------------------------------------------------------------------------
# Snowball discovery
# ---------------------------------------------------------------------------

def discover_from_seed(
    seed_accounts: list,
    max_depth: int = 2,
    min_score: float = 0.3,
    keywords: list = None,
    delay: float = 2.0,
    dry_run: bool = False,
) -> list:
    """
    BFS from seed accounts to discover new beta creators.

    For each account in the queue:
      - Load their recent posts via instaloader (or from existing gym_index)
      - Extract @mentions from captions
      - Score each mentioned account by keyword relevance
      - If score >= min_score, add to suggestions and enqueue for next depth

    Returns list of dicts: {username, score, discovered_via, depth}
    """
    try:
        import instaloader
    except ImportError:
        log.error("instaloader not installed")
        return []

    if keywords is None:
        cfg = load_config()
        keywords = cfg.get("discovery", {}).get("relevance_keywords", [
            "climb", "boulder", "beta", "route", "ปีน", "บอลเดอร์"
        ])

    # Load known accounts (seeds + existing contributors) to avoid re-adding
    known = set(seed_accounts)
    for contrib in load_contributors():
        known.add(contrib.get("username", "").lower())

    # Also check gym_index for any accounts already tracked
    index_file = get_path("index_file")
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            gym_index = json.load(f)
        for entry in gym_index:
            uname = entry.get("username", "").lower()
            if uname:
                known.add(uname)

    graph = load_graph()
    suggestions = []
    visited = set(known)

    # BFS queue: (username, depth, discovered_via)
    queue = deque([(acc, 0, "seed") for acc in seed_accounts])

    L = instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        save_metadata=False,
        quiet=True,
    )

    while queue:
        username, depth, via = queue.popleft()

        if depth > max_depth:
            continue

        log.info(f"  [depth={depth}] Scanning @{username} (via {via})")

        try:
            profile = instaloader.Profile.from_username(L.context, username)
            posts = list(profile.get_posts())[:30]  # limit to recent 30
            captions = [p.caption or "" for p in posts]

            # Build graph edges
            mentions_in_captions: set = set()
            for cap in captions:
                for mention in extract_mentions(cap):
                    mentions_in_captions.add(mention)

            graph[username] = list(mentions_in_captions)

            # Score and enqueue unvisited mentions
            for mention in mentions_in_captions:
                if mention in visited:
                    continue
                visited.add(mention)

                # Quick score based on mention context (captions that contain this mention)
                mention_captions = [
                    c for c in captions if f"@{mention}" in c.lower()
                ]
                score = score_account(mention, mention_captions or captions, keywords)

                if score >= min_score:
                    suggestions.append({
                        "username": mention,
                        "score": score,
                        "discovered_via": username,
                        "depth": depth + 1,
                    })
                    log.info(f"    ✅ @{mention} (score={score:.2f})")

                    if depth + 1 <= max_depth:
                        queue.append((mention, depth + 1, username))

            time.sleep(delay)

        except Exception as e:
            log.warning(f"  ⚠️  Could not scan @{username}: {e}")
            continue

    # Deduplicate and sort by score descending
    seen = set()
    unique = []
    for s in suggestions:
        if s["username"] not in seen:
            seen.add(s["username"])
            unique.append(s)
    unique.sort(key=lambda x: x["score"], reverse=True)

    if not dry_run:
        save_graph(graph)

    log.info(f"Discovery complete: {len(unique)} suggestions found")
    return unique


# ---------------------------------------------------------------------------
# Suggestions management
# ---------------------------------------------------------------------------

def suggest_contributors(min_score: float = 0.3) -> list:
    """Return discovered accounts not yet in contributors.json."""
    existing = {c.get("username", "").lower() for c in load_contributors()}
    graph = load_graph()

    suggestions = []
    for node, neighbors in graph.items():
        if node not in existing:
            suggestions.append({"username": node, "neighbors": len(neighbors)})

    return suggestions


def add_suggestions_to_contributors(suggestions: list) -> None:
    """Append suggested accounts to contributors.json."""
    from src.scrape import save_contributors
    existing = load_contributors()
    existing_names = {c.get("username", "").lower() for c in existing}

    added = 0
    for s in suggestions:
        uname = s["username"]
        if uname.lower() not in existing_names:
            existing.append({
                "username": uname,
                "note": f"auto-discovered (score={s.get('score', 0):.2f})",
                "gyms": [],
                "auto_discovered": True,
            })
            existing_names.add(uname.lower())
            added += 1

    save_contributors(existing)
    log.info(f"Added {added} new contributors from suggestions")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BetaFinder — Snowball Account Discovery",
        epilog="""
Examples:
  python discover.py                     # discover from all seeds (depth=2)
  python discover.py --depth 1           # 1 hop, faster
  python discover.py --min-score 0.5    # stricter filter
  python discover.py --add-suggested    # auto-add suggestions to contributors
  python discover.py --dry-run          # print suggestions without saving
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--add-suggested", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config()
    disc_cfg = cfg.get("discovery", {})

    if not disc_cfg.get("enabled", True):
        log.warning("Discovery disabled in config.yaml")
        return 1

    max_depth = args.depth or disc_cfg.get("max_depth", 2)
    min_score = args.min_score or disc_cfg.get("min_relevance_score", 0.3)
    delay = args.delay or get_nested("scraping.recommended_delay")
    keywords = disc_cfg.get("relevance_keywords", [])

    # Build seed accounts: official gyms + manual contributors
    seeds = [get_instagram_handle(g) for g in get_gym_names()]
    seeds += [c["username"] for c in load_contributors()]
    seeds = list(dict.fromkeys(seeds))  # deduplicate, preserve order

    log.info(f"Seeds: {seeds}")
    log.info(f"Config: depth={max_depth}, min_score={min_score}, dry_run={args.dry_run}")

    suggestions = discover_from_seed(
        seed_accounts=seeds,
        max_depth=max_depth,
        min_score=min_score,
        keywords=keywords,
        delay=delay,
        dry_run=args.dry_run,
    )

    # Print results table
    print(f"\n{'─'*60}")
    print(f"  🔍 Discovered {len(suggestions)} potential beta creators")
    print(f"{'─'*60}")
    for s in suggestions[:20]:  # show top 20
        print(f"  @{s['username']:<25} score={s['score']:.2f}  via=@{s['discovered_via']}")
    if len(suggestions) > 20:
        print(f"  ... and {len(suggestions) - 20} more")
    print(f"{'─'*60}\n")

    if args.add_suggested and not args.dry_run:
        add_suggestions_to_contributors(suggestions)
        print(f"✅ Added {len(suggestions)} accounts to contributors.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
