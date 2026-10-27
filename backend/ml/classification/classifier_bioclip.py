"""
classifier_bioclip.py — Extraction des features BioCLIP-2 (ONNX Runtime)
=========================================================================
- preprocess_image / preprocess_batch
- BioCLIPOnnxExtractor : encodeur visuel exporté en ONNX (training/export_onnx.py).

Usage :
    from ml.classification.classifier_bioclip import BioCLIPOnnxExtractor

    extractor = BioCLIPOnnxExtractor(model_path)
    features = extractor.extract_batch(images)
"""

import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import BATCH_SIZE, FEATURE_DIM, IMG_SIZE, ONNX_THREADS

# ════════════════════════════════════════════════════════════════════════════
# PRÉTRAITEMENT (normalisation CLIP)
# ════════════════════════════════════════════════════════════════════════════

CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """Image PIL → tableau (3, 224, 224) float32 normalisé CLIP."""
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_batch(images: list[Image.Image]) -> np.ndarray:
    """Liste d'images → (N, 3, 224, 224). Une image illisible devient un tenseur nul."""
    out = np.zeros((len(images), 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, img in enumerate(images):
        try:
            out[i] = preprocess_image(img)
        except Exception:
            pass
    return out


# ════════════════════════════════════════════════════════════════════════════
# EXTRACTEUR ONNX
# ════════════════════════════════════════════════════════════════════════════

def make_onnx_session(path: Path, threads: int = ONNX_THREADS):
    """Session onnxruntime CPU avec optimisations de graphe complètes."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class BioCLIPOnnxExtractor:
    """
    Encodeur visuel BioCLIP-2 exécuté avec onnxruntime.

    Entrée du graphe : "pixel_values" (N, 3, 224, 224) float32
    Sortie du graphe : "features"     (N, 768) float32, L2-normalisé
    """

    def __init__(self, model_path: Path, batch_size: int = BATCH_SIZE):
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        t0 = time.time()
        self.session = make_onnx_session(self.model_path)
        print(f"  BioCLIP-2 ONNX prêt en {time.time() - t0:.1f}s ({self.model_path.name})")

    def extract_pixels(self, pixels: np.ndarray) -> np.ndarray:
        feats = [self.session.run(["features"], {"pixel_values": pixels[i:i + self.batch_size]})[0]
                 for i in range(0, len(pixels), self.batch_size)]
        if not feats:
            return np.zeros((0, FEATURE_DIM), np.float32)
        x = np.concatenate(feats)
        return (x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)).astype(np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self.extract_pixels(preprocess_batch(images))
