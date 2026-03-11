"""
Configuration manager for BetaFinder CNX backend.

Loads YAML configuration from project root and supports environment variable overrides.
Uses Pydantic for settings validation.

Usage:
    from app.core.config import settings
    settings.DATABASE_URL
    settings.get_nested("embedding.model_name")
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

# Legacy YAML config cache
_config_cache: Optional[Dict[str, Any]] = None


class Settings(BaseSettings):
    """FastAPI application settings with environment variable support"""

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    ALLOWED_HOSTS: List[str] = ["*"]

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/betafinder.db")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Paths (relative to project root)
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent.parent  # backend/../..
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    CONFIG_FILE: Path = Path(os.getenv("CONFIG_FILE", "./config/config.yaml"))
    INDEX_FILE: Path = Path(os.getenv("INDEX_FILE", "./data/gym_index.json"))
    FAISS_FILE: Path = Path(os.getenv("FAISS_FILE", "./data/faiss.index"))
    CONTRIBUTORS_FILE: Path = Path(os.getenv("CONTRIBUTORS_FILE", "./data/contributors.json"))

    # ML/Embedding
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "ViT-B-32")
    EMBEDDING_PRETRAINED: str = os.getenv("EMBEDDING_PRETRAINED", "openai")
    DEVICE: str = os.getenv("DEVICE", "auto")  # auto | cuda | cpu

    # Scraping
    INSTALOADER_USER: Optional[str] = os.getenv("INSTALOADER_USER")
    INSTALOADER_PASS: Optional[str] = os.getenv("INSTALOADER_PASS")
    SCRAPING_DELAY: float = float(os.getenv("SCRAPING_DELAY", "2.0"))

    # File uploads
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()


# ── Legacy YAML Config Compatibility ────────────────────────────────────────

def load_yaml_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file with caching.

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


def get_yaml_config() -> Dict[str, Any]:
    """Get current YAML config (loads if not already cached)"""
    if _config_cache is None:
        load_yaml_config(str(settings.CONFIG_FILE))
    return _config_cache


def get_gym_names() -> List[str]:
    """Get list of gym keys from config"""
    config = get_yaml_config()
    return list(config.get("gyms", {}).keys())


def get_gym_info(gym_key: str) -> Dict[str, str]:
    """Get full gym info (name, instagram handle)"""
    config = get_yaml_config()
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
    config = get_yaml_config()
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
    config = get_yaml_config()
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
    return get_yaml_config().get("scraping", {})


def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration section"""
    return get_yaml_config().get("embedding", {})


def get_faiss_config() -> Dict[str, Any]:
    """Get FAISS configuration section"""
    return get_yaml_config().get("faiss", {})


def get_search_config() -> Dict[str, Any]:
    """Get search configuration section"""
    return get_yaml_config().get("search", {})


def get_filter_config() -> Dict[str, Any]:
    """Get wall filter configuration section"""
    return get_yaml_config().get("wall_filter", {})


def get_discovery_config() -> Dict[str, Any]:
    """Get account discovery configuration section"""
    return get_yaml_config().get("discovery", {})
