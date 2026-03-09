"""
Configuration loader for BetaFinder CNX
Loads YAML config, handles path resolution, provides typed access
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file with caching

    Args:
        config_path: Path to config.yaml (relative or absolute)

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        _config_cache = yaml.safe_load(f)

    return _config_cache


def get_config() -> Dict[str, Any]:
    """Get current config (must call load_config() first)"""
    if _config_cache is None:
        load_config()
    return _config_cache


def get_gym_names() -> List[str]:
    """Get list of gym keys from config"""
    config = get_config()
    return list(config.get("gyms", {}).keys())


def get_gym_info(gym_key: str) -> Dict[str, str]:
    """Get full gym info (name, instagram handle)"""
    config = get_config()
    gyms = config.get("gyms", {})

    if gym_key not in gyms:
        raise KeyError(f"Gym '{gym_key}' not found in config")

    return gyms[gym_key]


def get_instagram_handle(gym_key: str) -> str:
    """Get Instagram handle for a gym"""
    return get_gym_info(gym_key)["instagram"]


def get_path(key: str) -> Path:
    """
    Resolve a path from config, creating parent directories if they don't exist

    Args:
        key: Path key in config (e.g., "data_dir", "index_file")

    Returns:
        Resolved Path object
    """
    config = get_config()
    paths = config.get("paths", {})

    if key not in paths:
        raise KeyError(f"Path '{key}' not found in config")

    path_str = paths[key]
    path_obj = Path(path_str)

    # Create parent directories if it's a file path and parent doesn't exist
    if path_obj.suffix:  # has file extension
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    else:  # is a directory
        path_obj.mkdir(parents=True, exist_ok=True)

    return path_obj


def get_nested(keys: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation

    Example:
        get_nested("scraping.default_limit") -> 100
        get_nested("embedding.model_name") -> "ViT-B-32"
    """
    config = get_config()
    keys_list = keys.split(".")
    value = config

    try:
        for key in keys_list:
            value = value[key]
        return value
    except (KeyError, TypeError):
        if default is not None:
            return default
        raise KeyError(f"Config key '{keys}' not found")


def get_scraping_config() -> Dict[str, Any]:
    """Get scraping configuration section"""
    return get_config().get("scraping", {})


def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration section"""
    return get_config().get("embedding", {})


def get_search_config() -> Dict[str, Any]:
    """Get search configuration section"""
    return get_config().get("search", {})


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration section"""
    return get_config().get("logging", {})


def reload_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Force reload config, clearing cache"""
    global _config_cache
    _config_cache = None
    return load_config(config_path)


if __name__ == "__main__":
    # Test config loading
    cfg = load_config()
    print("✅ Config loaded successfully!")
    print(f"Gyms: {get_gym_names()}")
    print(f"Paths: {cfg['paths']}")
    print(f"Models: {get_nested('embedding.model_name')}")
