"""
Centralized logging setup for BetaFinder CNX
Loads logging configuration from config.yaml
"""

import logging
from typing import Optional

from src.config import get_logging_config

_loggers = {}


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with config-based level and format

    Args:
        name: Logger name (typically __name__)
        level: Optional override for logging level (e.g., "DEBUG", "INFO")

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    # Load config
    try:
        log_config = get_logging_config()
        config_level = log_config.get("level", "INFO")
        config_format = log_config.get("format", "%(asctime)s %(levelname)s %(message)s")
    except Exception:
        # Fallback if config loading fails
        config_level = "INFO"
        config_format = "%(asctime)s %(levelname)s %(message)s"

    # Override with provided level if given
    if level:
        config_level = level

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, config_level.upper(), logging.INFO))

    # Create formatter
    formatter = logging.Formatter(config_format)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)
    logger.propagate = False

    _loggers[name] = logger
    return logger


if __name__ == "__main__":
    # Test logger setup
    log = setup_logger(__name__)
    log.info("✅ Logger initialized successfully!")
    log.debug("This is a debug message")
    log.warning("This is a warning message")
