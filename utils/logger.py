# utils/logger.py
"""
Centralized logger for the entire project.

WHY: Using Python's built-in `logging` module instead of print() gives us:
- Timestamped log lines
- Log level filtering (DEBUG vs INFO vs ERROR)
- Easy redirection to files later without changing any other code
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with consistent formatting.

    Args:
        name: typically __name__ of the calling module

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    # Only add handler if none exist (prevents duplicate logs in long sessions)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger