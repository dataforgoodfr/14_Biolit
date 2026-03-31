import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# compat —> ajustement version
torch.use_deterministic_algorithms(False)
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **kw, weights_only=False)

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

from autodistill_yolov8 import YOLOv8  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from utils.logger import get_logger  # noqa: E402

log = get_logger("autodistill_train")


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/autodistill_boostrap.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = _device()
    log.info("Training YOLOv8 | device=%s", device)

    model = YOLOv8(cfg["model"])
    model.yolo.train(
        data=cfg["data_yaml"],
        epochs=cfg["epochs"],
        imgsz=cfg.get("imgsz", 640),
        device=device,
        project=cfg["project"],
        name=cfg["name"],
        batch=cfg.get("batch", 16),
        exist_ok=True,
    )

    log.info("Entraînement terminé.")


if __name__ == "__main__":
    main()
