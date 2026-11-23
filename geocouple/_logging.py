"""Logging setup shared by the command-line entry points (library modules only create loggers)."""

from __future__ import annotations

import logging
import os

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup(level: str | int | None = None) -> None:
    """Configure the root logger once (level from the argument, $GEOCOUPLE_LOG or INFO)."""
    level = level or os.environ.get("GEOCOUPLE_LOG", "INFO")
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=FORMAT, datefmt="%H:%M:%S")
    else:
        root.setLevel(level)
