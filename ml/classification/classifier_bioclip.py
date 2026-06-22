"""
classifier_bioclip.py — Extraction des features BioCLIP
=========================================================
Ce module gère l'extraction des features visuelles avec BioCLIP2.
Utilisé à la fois pour l'entraînement ET l'inférence.

Usage :
    from classifier_bioclip import BioCLIPExtractor, extract_features

    extractor = BioCLIPExtractor(device="cuda")
    features = extractor.extract_batch(images)
"""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    import open_clip
    BIOCLIP_AVAILABLE = True
except ImportError:
    BIOCLIP_AVAILABLE = False
    print("⚠ open_clip manquant — pip install open-clip-torch")

from config import (
    DEVICE,
    BIOCLIP_MODEL_ID as BIOCLIP_MODEL,
)


# ════════════════════════════════════════════════════════════════════════════
# TRANSFORMEUR D'IMAGE (normalisation BioCLIP)
# ════════════════════════════════════════════════════════════════════════════

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# ════════════════════════════════════════════════════════════════════════════
# EXTRACTEUR BIOCLIP
# ════════════════════════════════════════════════════════════════════════════

class BioCLIPExtractor:
    """
    Extracteur de features BioCLIP2.

    Usage :
        extractor = BioCLIPExtractor(device="cuda")
        features = extractor.extract_batch([img1, img2, img3])
    """

    def __init__(
        self,
        model_id: str = BIOCLIP_MODEL,
        device: Optional[torch.device] = None,
    ):
        if not BIOCLIP_AVAILABLE:
            raise ImportError("BioCLIP2 non disponible. Installez open-clip-torch")

        self.model_id = model_id
        self.device = device or DEVICE
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Charge le modèle BioCLIP (lazy loading)."""
        if self._model is not None:
            return

        print(f"  Chargement BioCLIP2 sur {self.device}...")
        t0 = time.time()
        self._model, _, _ = open_clip.create_model_and_transforms(self.model_id)
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_id)
        print(f"  BioCLIP2 prêt en {time.time() - t0:.1f}s")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def transform_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Transforme une image PIL en tensor normalisé."""
        return IMG_TRANSFORM(pil_img.convert("RGB"))

    def extract(self, pil_img: Image.Image) -> np.ndarray:
        """
        Extrait les features d'une seule image.

        Args:
            pil_img: Image PIL

        Returns:
            Vecteur de features normalisé (512d)
        """
        tensor = self.transform_image(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = F.normalize(feat, dim=-1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        """
        Extrait les features d'un batch d'images.

        Args:
            images: Liste d'images PIL

        Returns:
            Matrice de features (N x 512)
        """
        tensors = []
        for img in images:
            try:
                tensors.append(self.transform_image(img))
            except Exception:
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = F.normalize(features, dim=-1)

        return features.cpu().numpy().astype(np.float32)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode des descriptions textuelles.

        Args:
            texts: Liste de descriptions

        Returns:
            Matrice de features textuelles (N x 512)
        """
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = F.normalize(features, dim=-1)
        return features.cpu().numpy().astype(np.float32)
