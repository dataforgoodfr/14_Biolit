"""Inférence taxonomique BioCLIP2, sans code d'entraînement."""

import math
import os
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as functional
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

DEVICE = torch.device(os.getenv("ML_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
BIOCLIP_MODEL_ID = os.getenv("BIOCLIP_MODEL_ID", "hf-hub:imageomics/bioclip-2")
HF_REPO_ID = os.getenv("CLASSIFICATION_MODEL_REPO", "salima2509001/bioclip-classif")
CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFICATION_THRESHOLD", "0.35"))
MARGIN_MIN = float(os.getenv("CLASSIFICATION_MARGIN", "0.05"))
SUPERVISED_LEVELS = ("regne", "phylum", "classe", "ordre", "famille")

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


class LevelMLP(nn.Module):
    """Architecture minimale requise pour recharger les poids existants."""

    def __init__(self, input_dim: int, class_count: int, dropout: float = 0.3):
        super().__init__()
        hidden = max(256, class_count * 8)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, class_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


@dataclass(frozen=True)
class Prediction:
    name: str | None
    rank: str | None
    score: float
    margin: float
    details: dict[str, Any]


@dataclass
class ModelArtifacts:
    prototypes: dict[str, np.ndarray]
    taxonomy: dict[str, dict[str, Any]]
    whitening: tuple[np.ndarray, np.ndarray, np.ndarray]
    temperature: float
    mlps: dict[str, tuple[LevelMLP, Any]]


class ClassificationService:
    """Charge BioCLIP2 et les artefacts une fois, puis classe des lots de crops."""

    def __init__(self):
        self.artifacts = self._load_artifacts()
        self.encoder, _, _ = open_clip.create_model_and_transforms(BIOCLIP_MODEL_ID)
        self.encoder = self.encoder.to(DEVICE).eval()

    def process(self, images: list[Image.Image]) -> list[Prediction]:
        if not images:
            return []
        tensors = torch.stack([IMAGE_TRANSFORM(image.convert("RGB")) for image in images]).to(
            DEVICE
        )
        with torch.no_grad():
            features = functional.normalize(self.encoder.encode_image(tensors), dim=-1)
        whitened = self._whiten(features.cpu().numpy().astype(np.float32))
        return [self._predict(feature) for feature in whitened]

    def _predict(self, feature: np.ndarray) -> Prediction:
        species = list(self.artifacts.prototypes)
        prototype_matrix = np.vstack([self.artifacts.prototypes[name] for name in species])
        similarities = feature @ prototype_matrix.T
        logits = (similarities - similarities.max()) * self.artifacts.temperature
        probabilities = np.exp(logits) / np.exp(logits).sum()
        indexes = probabilities.argsort()[::-1]
        species_name = species[indexes[0]]
        species_score = float(probabilities[indexes[0]])
        species_margin = (
            species_score - float(probabilities[indexes[1]]) if len(indexes) > 1 else species_score
        )

        mlp_scores = self._mlp_scores(feature)
        if species_score >= CONFIDENCE_THRESHOLD and species_margin >= MARGIN_MIN:
            rank = "species"
            name = species_name
            score = species_score
            taxonomy = self.artifacts.taxonomy.get(species_name, {})
        else:
            rank, name, score = self._best_parent(mlp_scores)
            taxonomy = self._taxonomy_for(rank, name)

        return Prediction(
            name=name,
            rank=rank,
            score=round(score, 6),
            margin=round(species_margin, 6),
            details={
                "species_top3": [
                    {"name": species[index], "score": round(float(probabilities[index]), 6)}
                    for index in indexes[:3]
                ],
                "levels": mlp_scores,
                "taxonomy": taxonomy,
            },
        )

    def _mlp_scores(self, feature: np.ndarray) -> dict[str, dict[str, Any]]:
        tensor = torch.tensor(feature[None], dtype=torch.float32, device=DEVICE)
        scores = {}
        with torch.no_grad():
            for level, (model, encoder) in self.artifacts.mlps.items():
                probabilities = functional.softmax(model(tensor).squeeze(0), dim=-1).cpu().numpy()
                indexes = probabilities.argsort()[::-1]
                top_score = float(probabilities[indexes[0]])
                scores[level] = {
                    "name": str(encoder.classes_[indexes[0]]),
                    "score": round(top_score, 6),
                    "margin": round(
                        top_score - float(probabilities[indexes[1]])
                        if len(indexes) > 1
                        else top_score,
                        6,
                    ),
                }
        return scores

    @staticmethod
    def _best_parent(scores: dict[str, dict[str, Any]]) -> tuple[str | None, str | None, float]:
        for level in ("famille", "ordre", "classe", "phylum", "regne"):
            candidate = scores.get(level)
            if (
                candidate
                and candidate["score"] >= CONFIDENCE_THRESHOLD
                and candidate["margin"] >= MARGIN_MIN
            ):
                return level, candidate["name"], candidate["score"]
        return None, None, 0.0

    def _taxonomy_for(self, rank: str | None, name: str | None) -> dict[str, Any]:
        if not rank or not name:
            return {}
        for taxonomy in self.artifacts.taxonomy.values():
            value = taxonomy.get(rank)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                if str(value) == name:
                    return {
                        key: str(item)
                        for key, item in taxonomy.items()
                        if item is not None and not (isinstance(item, float) and math.isnan(item))
                    }
        return {}

    def _whiten(self, features: np.ndarray) -> np.ndarray:
        mean, components, variance = self.artifacts.whitening
        projected = (features - mean) @ components.T / np.sqrt(variance + 1e-8)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return (projected / np.clip(norms, 1e-8, None)).astype(np.float32)

    @staticmethod
    def _load_artifacts() -> ModelArtifacts:
        prototype_path = hf_hub_download(HF_REPO_ID, "proto_model.npz")
        taxonomy_path = hf_hub_download(HF_REPO_ID, "tax_lookup.pkl")
        mlp_path = hf_hub_download(HF_REPO_ID, "mlp_model.pt")

        data = np.load(prototype_path, allow_pickle=True)
        species = [str(name) for name in data["species_list"]]
        prototypes = {name: data["prototypes"][index] for index, name in enumerate(species)}
        with open(taxonomy_path, "rb") as stream:
            taxonomy = pickle.load(stream)

        mlps = {}
        checkpoints = torch.load(mlp_path, map_location=DEVICE, weights_only=False)
        for level, checkpoint in checkpoints.items():
            model = LevelMLP(checkpoint["input_dim"], checkpoint["n_classes"]).to(DEVICE)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            mlps[level] = (model, checkpoint["encoder"])

        return ModelArtifacts(
            prototypes=prototypes,
            taxonomy=taxonomy,
            whitening=(data["wh_mean"], data["wh_components"], data["wh_variance"]),
            temperature=float(data["temperature"][0]),
            mlps=mlps,
        )
