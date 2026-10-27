"""Détection de l'espèce principale avec le modèle ONNX (sans PyTorch ni Ultralytics)."""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass
class ImageResult:
    path: str
    orig_shape: tuple[int, int]  # (hauteur, largeur)
    detection: Detection | None  # None : rien au-dessus du seuil


def load_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def class_names(session: ort.InferenceSession) -> dict[int, str]:
    # Ultralytics écrit les classes dans les métadonnées du .onnx : "{0: 'plant', 1: 'animal'}"
    return ast.literal_eval(session.get_modelmeta().custom_metadata_map["names"])


def list_images(source: str) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def detect(session: ort.InferenceSession, img: Image.Image, conf: float) -> Detection | None:
    """Détection la plus confiante (max_det=1), ou None si elle est sous `conf`."""
    imgsz = session.get_inputs()[0].shape[2]  # entrée fixe (1, 3, 640, 640)
    w, h = img.size

    # Letterbox : redimensionne sans déformer, complète en gris jusqu'à imgsz x imgsz
    r = min(imgsz / w, imgsz / h)
    nw, nh = round(w * r), round(h * r)
    left, top = round((imgsz - nw) / 2 - 0.1), round((imgsz - nh) / 2 - 0.1)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0

    # Sortie (1, 4 + nb_classes, 8400) : cx, cy, w, h puis un score par classe.
    # Une seule détection gardée : pas besoin de NMS, la meilleure boîte suffit.
    preds = session.run(None, {session.get_inputs()[0].name: x})[0][0].T
    scores = preds[:, 4:]
    best = scores.max(axis=1).argmax()
    confidence = float(scores[best].max())
    if confidence < conf:
        return None

    # Boîte dans l'image d'origine : retire le padding puis l'échelle
    cx, cy, bw, bh = preds[best, :4]
    box = [(cx - bw / 2 - left) / r, (cy - bh / 2 - top) / r,
           (cx + bw / 2 - left) / r, (cy + bh / 2 - top) / r]
    box = [float(np.clip(v, 0, lim)) for v, lim in zip(box, (w, h, w, h))]
    class_id = int(scores[best].argmax())
    return Detection(class_id, class_names(session)[class_id], confidence, box)


def detect_images(session: ort.InferenceSession, source: str, conf: float) -> list[ImageResult]:
    results = []
    for path in list_images(source):
        img = Image.open(path).convert("RGB")
        results.append(ImageResult(str(path), (img.height, img.width), detect(session, img, conf)))
    return results
