import logging
import logging.handlers
from pathlib import Path

_COLORS = {
    "DEBUG": "\033[36m",     # cyan
    "INFO": "\033[32m",      # green
    "WARNING": "\033[33m",   # yellow
    "ERROR": "\033[31m",     # red
    "CRITICAL": "\033[35m",  # magenta
    "RESET": "\033[0m",
}


class _ColorConsoleFormatter(logging.Formatter):
    """Console formatter with ANSI color per level."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        reset = _COLORS["RESET"]
        record.levelname_colored = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logger(
    name: str,
    log_dir: str = "outputs/",
    level: str = "INFO",
) -> logging.Logger:
    """Create (or retrieve) a named logger with console + rotating file output.

    Args:
        name:    Logger name, e.g. ``"biolit.crop_inference"``.
        log_dir: Directory where the log file will be written.
                 The file is named after the last path component of *log_dir*
                 (typically the run folder: ``run_YYYYMMDD_HHMMSS``).
        level:   Minimum log level for the console handler ("DEBUG", "INFO", …).

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    # Idempotent: do not add handlers twice (e.g. if main() is called again).
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(logging.DEBUG)  # handlers filter individually

    # ------------------------------------------------------------------
    # Console handler — short colored format
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(
        _ColorConsoleFormatter(
            fmt="[%(asctime)s] %(levelname_colored)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Rotating file handler — full format
    # ------------------------------------------------------------------
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    run_name = log_path.name or "crop_inference"
    log_file = log_path / f"{run_name}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s — %(module)s:%(lineno)d — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    # Redirect Python warnings (warnings.warn) through the logging system.
    logging.captureWarnings(True)

    return logger
