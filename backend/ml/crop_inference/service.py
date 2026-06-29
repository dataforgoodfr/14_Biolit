"""Service d'inférence YOLOv8 utilisé par le worker de production."""

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import torch
import ultralytics.nn.modules as ultralytics_modules
import yaml
from PIL import Image
from ultralytics import YOLO

from .model_loader import load_model_weights

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

# Compatibilité avec les poids YOLO sérialisés par une ancienne version.
for module_name in ("conv", "block", "head", "transformer"):
    sys.modules[f"ultralytics.nn.modules.{module_name}"] = ultralytics_modules


@dataclass(frozen=True)
class CropResult:
    image_id: str
    original: Image.Image
    crop: Image.Image | None
    label: str | None
    confidence: float | None


class CropService:
    """Télécharge un lot d'images et conserve la meilleure détection par image."""

    def __init__(self, config_path: Path = DEFAULT_CONFIG):
        self.config = yaml.safe_load(config_path.read_text())
        self.model = self._load_model()

    def process(self, images: list[dict[str, Any]]) -> list[CropResult]:
        if not images:
            return []

        with tempfile.TemporaryDirectory(prefix="biolit-crop-") as temp_dir:
            paths = self._download(images, Path(temp_dir))
            inference = self.config["inference"]
            results = self.model.predict(
                source=[str(path) for path in paths],
                conf=inference["conf"],
                iou=inference["iou"],
                imgsz=inference["imgsz"],
                device=inference["device"],
                save=False,
                save_crop=False,
                max_det=1,
            )

            output = []
            for metadata, path, result in zip(images, paths, results, strict=True):
                original = Image.open(path).convert("RGB")
                if len(result.boxes) == 0:
                    output.append(CropResult(metadata["id_image"], original, None, None, None))
                    continue

                box = result.boxes[result.boxes.conf.argmax()]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls)
                output.append(
                    CropResult(
                        image_id=metadata["id_image"],
                        original=original,
                        crop=original.crop((x1, y1, x2, y2)).convert("RGB"),
                        label=result.names[class_id],
                        confidence=round(float(box.conf), 4),
                    )
                )
            return output

    def _load_model(self) -> YOLO:
        original_torch_load = torch.load

        def compatible_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        try:
            torch.load = compatible_load
            return YOLO(load_model_weights(self.config))
        finally:
            torch.load = original_torch_load

    @staticmethod
    def _download(images: list[dict[str, Any]], directory: Path) -> list[Path]:
        paths = []
        for index, image in enumerate(images):
            path = directory / f"{index:04d}.jpg"
            with requests.get(image["source_url"], stream=True, timeout=30) as response:
                response.raise_for_status()
                with path.open("wb") as target:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            target.write(chunk)
            paths.append(path)
        return paths
