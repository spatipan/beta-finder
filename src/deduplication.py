"""
Deduplication logic for BetaFinder — detect already-scraped posts.

Module: Reusable, pure functions for shortcode-based deduplication.
No external dependencies. Works across all indexes (gym_index, posts_index, etc.)
"""


def load_post_shortcodes(index_dict: list[dict]) -> set[str]:
    """
    Extract all shortcodes from an index.

    Args:
        index_dict: List of metadata records from gym_index.json or posts_index.json

    Returns:
        Set of shortcodes (e.g., {"ABC123", "XYZ789", ...})
        Shortcodes are unique per Instagram post.

    Example:
        >>> records = [
        ...     {"shortcode": "ABC123", "username": "user1"},
        ...     {"shortcode": "DEF456", "username": "user2"},
        ...     {"shortcode": None, "username": "user3"},  # skip this
        ... ]
        >>> load_post_shortcodes(records)
        {'ABC123', 'DEF456'}
    """
    return {record.get("shortcode") for record in index_dict if record.get("shortcode")}


def build_shortcode_set_per_account(index_dict: list[dict]) -> dict[str, set[str]]:
    """
    Group shortcodes by username for per-account deduplication.

    Useful when scraping the same username multiple times with different modes
    (e.g., "posts" mode vs "tagged" mode). Ensures we skip already-seen posts.

    Args:
        index_dict: List of metadata records (from any index file)

    Returns:
        Dict mapping username → set of their shortcodes
        Example:
        {
            "the_alpine_outpost": {"ABC123", "DEF456"},
            "climb.with.poom": {"GHI789", "JKL012"}
        }

    Example:
        >>> records = [
        ...     {"username": "user_a", "shortcode": "ABC123"},
        ...     {"username": "user_a", "shortcode": "DEF456"},
        ...     {"username": "user_b", "shortcode": "GHI789"},
        ... ]
        >>> build_shortcode_set_per_account(records)
        {'user_a': {'ABC123', 'DEF456'}, 'user_b': {'GHI789'}}
    """
    result = {}
    for record in index_dict:
        username = record.get("username")
        shortcode = record.get("shortcode")
        if username and shortcode:
            result.setdefault(username, set()).add(shortcode)
    return result


def is_already_scraped(shortcode: str, scraped_set: set[str]) -> bool:
    """
    Check if a post has already been scraped.

    Args:
        shortcode: Instagram post ID
        scraped_set: Set of known shortcodes (from load_post_shortcodes or
                     build_shortcode_set_per_account(...).get(username, set()))

    Returns:
        True if shortcode is in the scraped_set, False otherwise

    Example:
        >>> scraped = {"ABC123", "DEF456"}
        >>> is_already_scraped("ABC123", scraped)
        True
        >>> is_already_scraped("NEW789", scraped)
        False
    """
    return shortcode in scraped_set
