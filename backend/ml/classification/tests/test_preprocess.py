"""
Prétraitement numpy de l'inférence (runtime ONNX, sans PyTorch).
"""

import numpy as np
from PIL import Image

from ml.classification.classifier_bioclip import preprocess_batch


def _images() -> list[Image.Image]:
    rng = np.random.default_rng(0)
    sizes = [(224, 224), (640, 480), (300, 800), (97, 61), (1, 1)]
    images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for w, h in sizes]
    grad = np.linspace(0, 255, 256 * 256 * 3).reshape(256, 256, 3).astype(np.uint8)
    images.append(Image.fromarray(grad))
    images.append(Image.fromarray(grad[..., 0]))                 # niveaux de gris
    images.append(Image.fromarray(grad).convert("RGBA"))        # canal alpha
    images.append(Image.fromarray(grad).convert("P"))           # palette
    return images


def test_preprocess_batch_shape_and_bad_image():
    out = preprocess_batch(_images()[:2] + [None])
    assert out.shape == (3, 3, 224, 224) and out.dtype == np.float32
    assert not out[2].any()          # image illisible → tenseur nul (comportement historique)
