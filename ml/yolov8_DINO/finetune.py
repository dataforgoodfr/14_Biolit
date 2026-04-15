import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

torch.use_deterministic_algorithms(False)

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from ultralytics import YOLO  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import get_logger  # noqa: E402

log = get_logger("finetune")


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _validate_path(path_str: str, label: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = _device()
    log.info("Fine-tuning YOLOv8 | device=%s", device)

    weights_path = _validate_path(cfg["model"], "Model weights")
    data_yaml_path = _validate_path(cfg["data_yaml"], "data.yaml")

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_yaml_path),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        device=device,
        project=cfg["project"],
        name=cfg["name"],
        exist_ok=True,
    )

    log.info("Fine-tuning terminé.")


if __name__ == "__main__":
    main()
