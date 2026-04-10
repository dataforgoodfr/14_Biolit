import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml
import torch
from PIL import Image
from ultralytics import YOLO
from model_loader import load_model_weights
from utils.logger import setup_logger

import ultralytics.nn.modules as modules
import sys

# Hardfix module
sys.modules["ultralytics.nn.modules.conv"] = modules
sys.modules["ultralytics.nn.modules.block"] = modules
sys.modules["ultralytics.nn.modules.head"] = modules
sys.modules["ultralytics.nn.modules.transformer"] = modules

torch.use_deterministic_algorithms(False)
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **kw, weights_only=False)

_LOGGER_NAME = "biolit.crop_inference"


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config introuvable : {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg: dict) -> YOLO:
    return YOLO(load_model_weights(cfg))


def run_inference(model: YOLO, input_path: str, cfg: dict, run_name: str) -> list:
    infer = cfg["inference"]
    results = model.predict(
        source=input_path,
        conf=infer["conf"],
        iou=infer["iou"],
        imgsz=infer["imgsz"],
        device=infer["device"],
        save=False,
        max_det=1,
        save_crop=False,
        project=infer["save_dir"],
        name=run_name,
    )
    return results


def build_manifest(results: list, run_name: str, output_dir: str) -> list:
    logger = logging.getLogger(_LOGGER_NAME)
    run_path = Path(output_dir) / run_name / "crops"
    run_path.mkdir(parents=True, exist_ok=True)

    manifest = []

    for r in results:
        if len(r.boxes) == 0:
            continue

        best_idx = r.boxes.conf.argmax()
        box = r.boxes[best_idx]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls)
        cls_name = r.names[cls_id]
        conf = float(box.conf)
        source_name = Path(r.path).stem

        img = Image.open(r.path)
        crop = img.crop((x1, y1, x2, y2))

        crop_filename = f"{source_name}_{cls_name}_{conf:.2f}.jpg"
        crop_path = run_path / crop_filename
        crop.save(crop_path)

        logger.debug(
            "Crop sauvegardé : %s (classe=%s, conf=%.4f, bbox=[%.1f, %.1f, %.1f, %.1f])",
            crop_path, cls_name, conf, x1, y1, x2, y2,
        )

        manifest.append({
            "source_image": r.path,
            "crop_path": str(crop_path),
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(conf, 4),
            "bbox_xyxy": [round(c, 1) for c in [x1, y1, x2, y2]],
            "orig_shape": list(r.orig_shape),
        })

    manifest_path = run_path.parent / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("%d crops sauvegardés → %s", len(manifest), run_path)
    logger.info("Manifeste → %s", manifest_path)

    return manifest


def print_results(model: YOLO, results: list) -> None:
    logger = logging.getLogger(_LOGGER_NAME)
    for r in results:
        logger.info("[%s] — %d détection(s)", r.path, len(r.boxes))
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            logger.info("  → %s : %.2f", label, conf)


def main():
    parser = argparse.ArgumentParser(description="Inférence YOLOv8 Biolit — crop + manifeste")
    parser.add_argument("--source", required=True, help="image, dossier ou vidéo")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de log pour la console",
    )
    args = parser.parse_args()

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        logging.basicConfig()
        logging.getLogger(_LOGGER_NAME).error(
            "Fichier de config introuvable : %s", args.config, exc_info=True
        )
        raise

    log_dir = Path(cfg["inference"]["save_dir"]) / run_name
    logger = setup_logger(_LOGGER_NAME, log_dir=str(log_dir), level=args.log_level)

    # Redirect ultralytics internal logs to our file handler so warnings land
    # in the run log alongside inference output.
    ul_logger = logging.getLogger("ultralytics")
    ul_logger.handlers = list(logger.handlers)
    ul_logger.setLevel(logging.WARNING)
    ul_logger.propagate = False

    infer = cfg["inference"]
    logger.info(
        "Démarrage run=%s | source=%s | device=%s | conf=%.2f | iou=%.2f | imgsz=%d | max_det=1",
        run_name, args.source, infer["device"], infer["conf"], infer["iou"], infer["imgsz"],
    )

    t0 = time.perf_counter()

    try:
        model = load_model(cfg)
    except Exception:
        logger.error("Impossible de charger le modèle", exc_info=True)
        raise

    try:
        results = run_inference(model, args.source, cfg, run_name)
    except Exception:
        logger.error("Erreur pendant l'inférence sur '%s'", args.source, exc_info=True)
        raise

    print_results(model, results)

    try:
        manifest = build_manifest(results, run_name, cfg["inference"]["save_dir"])
    except Exception:
        logger.error("Erreur lors de la construction du manifeste", exc_info=True)
        raise

    elapsed = time.perf_counter() - t0
    logger.info(
        "Run terminé | images=%d | crops=%d | durée=%.2fs",
        len(results), len(manifest), elapsed,
    )


if __name__ == "__main__":
    main()
