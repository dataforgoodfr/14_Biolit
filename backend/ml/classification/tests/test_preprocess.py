"""
Non-régression du prétraitement : la version numpy (runtime ONNX, sans PyTorch)
doit être identique à l'ancienne chaîne torchvision.
"""

import numpy as np
from PIL import Image

from ml.classification.classifier_bioclip import CLIP_MEAN, CLIP_STD, preprocess_batch, preprocess_image


def _torchvision_reference(img: Image.Image) -> np.ndarray:
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN.tolist(), std=CLIP_STD.tolist()),
    ])
    return tf(img.convert("RGB")).numpy()


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


def test_preprocess_matches_torchvision():
    for img in _images():
        np.testing.assert_allclose(preprocess_image(img), _torchvision_reference(img),
                                   rtol=0, atol=1e-6)


def test_preprocess_batch_shape_and_bad_image():
    out = preprocess_batch(_images()[:2] + [None])
    assert out.shape == (3, 3, 224, 224) and out.dtype == np.float32
    assert not out[2].any()          # image illisible → tenseur nul (comportement historique)
