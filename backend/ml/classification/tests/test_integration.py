"""
Test d'intégration : image → encodeur ONNX → têtes ONNX → décision, sur les vrais
fichiers du modèle (results/, produits par training/ ou téléchargés depuis Hugging Face).

L'équivalence ONNX ↔ PyTorch est vérifiée à chaque export (training/export_onnx.py
échoue au-delà de la tolérance) ; ce test vérifie la chaîne de production complète.

    RUN_ONNX_REGRESSION=1 uv run pytest ml/classification/tests/test_integration.py -vv
"""

import os

import numpy as np
import pytest
from PIL import Image

from ml.classification.config import ENCODER_ONNX_FILE, HEADS_ONNX_FILE, META_FILE, ROOT_DIR

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ONNX_REGRESSION") != "1"
    or not all(p.exists() for p in [ENCODER_ONNX_FILE, HEADS_ONNX_FILE, META_FILE]),
    reason="test lourd : RUN_ONNX_REGRESSION=1 et fichiers modèle dans results/",
)


def _images() -> list[Image.Image]:
    folder = ROOT_DIR / "data/sample_crops"
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
        if folder.exists() else []
    images = [Image.open(p).convert("RGB") for p in files]
    rng = np.random.default_rng(0)
    images += [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
               for w, h in [(224, 224), (640, 480), (97, 61)]]
    return images


def test_end_to_end_predict_batch():
    from ml.classification.classifier_infer import load_model, predict_batch

    model = load_model(source="local")
    images = _images()
    preds = predict_batch(images, model)
    assert len(preds) == len(images)
    for p in preds:
        assert p["path"].startswith("hier_")
        if p["best_level"] is not None:
            assert p[p["best_level"]] == p["best_label"]
            assert p["best_score"] >= model.thresholds[p["best_level"]]
    assert predict_batch([], model) == []
    # déterminisme : deux passages identiques
    assert preds == predict_batch(images, model)
