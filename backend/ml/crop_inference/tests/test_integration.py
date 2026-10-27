"""
Test d'intégration : vrai modèle de détection ONNX (téléchargé depuis Hugging Face
via config.yaml) sur les images d'exemple, ou des images aléatoires à défaut.

    RUN_ONNX_REGRESSION=1 uv run pytest ml/crop_inference/tests/test_integration.py -vv
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ONNX_REGRESSION") != "1",
    reason="test lourd : RUN_ONNX_REGRESSION=1 (télécharge le modèle depuis Hugging Face)",
)

CROP_DIR = Path(__file__).parents[1]


def _images() -> list[Image.Image]:
    folder = CROP_DIR / "images"
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if folder.exists() else []
    images = [Image.open(p).convert("RGB") for p in files]
    rng = np.random.default_rng(0)
    images += [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
               for w, h in [(640, 640), (1280, 720), (97, 61)]]
    return images


def test_end_to_end_detect():
    from ml.crop_inference.detector import class_names, detect
    from ml.crop_inference.predict import load_config, load_model

    session = load_model(load_config(CROP_DIR / "config.yaml"))
    assert class_names(session) == {0: "plant", 1: "animal"}

    images = _images()
    dets = [detect(session, img, conf=0.0) for img in images]
    for img, det in zip(images, dets):
        assert det is not None and det.class_name in {"plant", "animal"}
        assert 0.0 <= det.confidence <= 1.0
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0 <= x1 <= x2 <= img.width and 0 <= y1 <= y2 <= img.height
    # déterminisme : deux passages identiques
    assert dets == [detect(session, img, conf=0.0) for img in images]
