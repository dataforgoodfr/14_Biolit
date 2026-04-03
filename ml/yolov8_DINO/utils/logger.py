import logging
import sys
from pathlib import Path
from datetime import datetime

# FICHIER DES LOGS
def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%H:%M:%S")
    )

    _file_handler = logging.FileHandler(
        Path(log_dir) / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        mode="w",
        encoding="utf-8",
    )
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", "%Y-%m-%d %H:%M:%S")
    )

    logger.addHandler(_stream_handler)
    logger.addHandler(_file_handler)

    return logger
