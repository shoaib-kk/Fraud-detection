import logging
import sys
from pathlib import Path


def setup_logger(name, log_file: str | Path = "logs/testing.log"):
    """Create or reuse a named logger with console and file handlers."""
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this logger was already configured.
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s |-| %(name)s |-| %(levelname)s |-| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:

        # if log file is specified, appends to it, creating the file and parent directories if they don't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger