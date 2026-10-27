"""
classifier_infer.py — Inférence (ONNX Runtime, sans PyTorch)
=============================================================
Chaîne :
    image ──BioCLIP-2 (ONNX, gelé)──▶ features 768d
          ──têtes (ONNX)──▶ P(espèce) = w × régression logistique + (1 − w) × prototypes
                            P(famille), P(ordre)… = somme des P(espèce) du taxon

Décision hiérarchique : on retient le niveau le PLUS FIN (espèce → famille → ordre →
classe → phylum → règne) dont la meilleure probabilité atteint le seuil de ce niveau ;
sinon rejet. Les seuils sont calibrés à l'entraînement (training/train.py) et stockés
dans classifier_meta.json ; ils peuvent être surchargés via `thresholds=`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .classifier_bioclip import BioCLIPOnnxExtractor, make_onnx_session
from .config import (
    DECISION_LEVELS,
    HEADS_ONNX_FILE,
    HF_REPO_ID,
    HF_SUBFOLDER,
    META_FILE,
    MODEL_SOURCE,
    RESULTS_DIR,
    SUPERVISED_LEVELS,
)


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BioModel:
    encoder: Optional[BioCLIPOnnxExtractor]   # None si on ne travaille que sur des features
    heads: object                             # onnxruntime.InferenceSession
    meta: dict

    @property
    def thresholds(self) -> dict:
        return dict(self.meta["thresholds"])


def _resolve(filename: str, source: str) -> Path:
    if source == "local":
        path = RESULTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Fichier modèle introuvable : {path}")
        return path
    if source == "huggingface":
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=filename,
                                    subfolder=HF_SUBFOLDER or None))
    raise ValueError(f"Source inconnue : {source!r} (huggingface | local)")


def load_model(source: str = MODEL_SOURCE, with_encoder: bool = True) -> BioModel:
    """Charge méta-données, têtes ONNX et (optionnellement) l'encodeur BioCLIP-2 ONNX."""
    print(f"Chargement du modèle de classification (source={source})")
    with open(_resolve(META_FILE.name, source), encoding="utf-8") as f:
        meta = json.load(f)
    heads = make_onnx_session(_resolve(meta.get("heads_file", HEADS_ONNX_FILE.name), source))
    encoder = (BioCLIPOnnxExtractor(_resolve(meta["encoder_file"], source))
               if with_encoder else None)
    print(f"Modèle {meta.get('version')} : {len(meta['classes']['species_name'])} espèces | "
          f"seuils {meta['thresholds']}")
    return BioModel(encoder=encoder, heads=heads, meta=meta)


# ════════════════════════════════════════════════════════════════════════════
# PROBABILITÉS (têtes ONNX)
# ════════════════════════════════════════════════════════════════════════════

def predict_proba(features: np.ndarray, model: BioModel) -> dict[str, np.ndarray]:
    """features (N, 768) → {"species_name": (N, S), "famille": (N, F), …}."""
    features = np.ascontiguousarray(np.atleast_2d(features), dtype=np.float32)
    names = [o.name for o in model.heads.get_outputs()]
    outs = model.heads.run(names, {"features": features})
    return {name.removeprefix("p_"): out for name, out in zip(names, outs)}


# ════════════════════════════════════════════════════════════════════════════
# DÉCISION HIÉRARCHIQUE
# ════════════════════════════════════════════════════════════════════════════

def _top(p: np.ndarray, labels: list[str], k: int = 3):
    idx = np.argsort(-p)[:k]
    top = [{"label": labels[i], "score": round(float(p[i]), 4)} for i in idx]
    margin = float(p[idx[0]] - p[idx[1]]) if len(idx) > 1 else float(p[idx[0]])
    return labels[idx[0]], float(p[idx[0]]), margin, top


def decide(probs: dict[str, np.ndarray], meta: dict, thresholds: dict) -> dict:
    """Décision pour UNE image : le niveau le plus fin dont la probabilité atteint son seuil."""
    tops = {lvl: _top(probs[lvl], meta["classes"][lvl]) for lvl in DECISION_LEVELS}
    best_level = best_label = None
    best_score = margin = 0.0
    for lvl in DECISION_LEVELS:
        label, score, m, _ = tops[lvl]
        if score >= thresholds[lvl]:
            best_level, best_label, best_score, margin = lvl, label, score, m
            break

    # Taxonomie cohérente : niveaux ≥ best_level depuis le référentiel, plus fins = None
    tax = {lvl: None for lvl in SUPERVISED_LEVELS + ["species_name"]}
    if best_level == "species_name":
        tax.update(meta["taxonomy"][best_label])
        tax["species_name"] = best_label
    elif best_level is not None:
        tax.update(meta["taxon_parents"][best_level].get(best_label, {}))
        tax[best_level] = best_label

    return {
        "best_level": best_level,
        "best_label": best_label,
        "best_score": round(best_score, 4),
        "margin": round(margin, 4),
        "path": f"hier_{best_level or 'rejet'}",
        "all_scores": {lvl: t[3] for lvl, t in tops.items()},
        **tax,
    }


def predict_features(features: np.ndarray, model: BioModel,
                     thresholds: Optional[dict] = None) -> list[dict]:
    """Prédiction depuis des features BioCLIP-2 (N, 768) déjà extraites."""
    th = {**model.thresholds, **(thresholds or {})}
    probs = predict_proba(features, model)
    n = len(probs["species_name"])
    return [decide({k: v[i] for k, v in probs.items()}, model.meta, th) for i in range(n)]


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION SUR IMAGES
# ════════════════════════════════════════════════════════════════════════════

def predict_batch(images: list[Image.Image], model: BioModel,
                  thresholds: Optional[dict] = None) -> list[dict]:
    """Prédit sur un batch d'images PIL."""
    if not images:
        return []
    return predict_features(model.encoder.extract_batch(images), model, thresholds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inférence classification")
    parser.add_argument("--images", type=str, required=True, help="Dossier ou fichier image")
    parser.add_argument("--source", type=str, default=MODEL_SOURCE)
    args = parser.parse_args()

    model = load_model(args.source)
    path = Path(args.images)
    files = (sorted(f for f in path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
             if path.is_dir() else [path])
    results = predict_batch([Image.open(f).convert("RGB") for f in files], model)
    for f, r in zip(files, results):
        print(f"{f.name}: {r['best_level']} / {r['best_label']} ({r['best_score']:.2f})")
