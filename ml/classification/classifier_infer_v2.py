"""
Inference du modèle de classification
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    DEVICE,
    CONFIDENCE_THRESHOLD as THRESHOLD,
    PROTO_MODEL_FILE as MODEL_FILE,
    PROTO_TAX_FILE as TAX_LOOKUP_FILE,
    MARGIN_MIN,
    MLP_MODEL_FILE,
    HF_REPO_ID,
)
from classifier_bioclip import BioCLIPExtractor

# Niveaux taxonomiques (du plus fin au plus large)
ALL_LEVELS = ["species_name", "famille", "ordre", "classe", "phylum", "regne"]
SUPERVISED_LEVELS = ["regne", "phylum", "classe", "ordre", "famille"]


# ════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE MLP
# ════════════════════════════════════════════════════════════════════════════

from classifier_mlp import LevelMLP


# ════════════════════════════════════════════════════════════════════════════
# CONTENEUR DU MODÈLE
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BioModel:
    """
    Conteneur pour tous les artefacts du modèle
    """
    prototypes: dict  # {espèce → vecteur 256d}
    tax_lookup: dict  # {espèce → {famille, ordre, classe, phylum, règne}}
    whitening: tuple  # (mean, components, variance) pour apply_whitening
    temperature: float  # température calibrée pour Proto-CLIP
    mlp_dict: dict  # {niveau → (LevelMLP, LabelEncoder)}


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ════════════════════════════════════════════════════════════════════════════

def _hf_download(filename: str) -> Path:
    """Télécharge un fichier depuis Hugging Face Hub """
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=filename))


def load_model(device: torch.device = DEVICE) -> BioModel:
    """
    Charge le modèle depuis Hugging Face Hub.
    """
    print(f"Chargement depuis Hugging Face : {HF_REPO_ID}")
    model_file = _hf_download(MODEL_FILE.name)
    tax_file   = _hf_download(TAX_LOOKUP_FILE.name)
    mlp_file   = _hf_download(MLP_MODEL_FILE.name)
    
    # === Proto-CLIP ===
    data = np.load(model_file, allow_pickle=True)
    species_list = list(data["species_list"])
    proto_mat = data["prototypes"]
    prototypes = {sp: proto_mat[i] for i, sp in enumerate(species_list)}
    whitening = (data["wh_mean"], data["wh_components"], data["wh_variance"])
    temperature = float(data["temperature"][0])
    
    # === Taxonomie ===
    with open(tax_file, "rb") as f:
        tax_lookup = pickle.load(f)
    
    # === MLP par niveau ===
    mlp_dict = {}
    if mlp_file.exists():
        mlp_save = torch.load(mlp_file, map_location=device, weights_only=False)
        for level, ckpt in mlp_save.items():
            mlp = LevelMLP(ckpt["input_dim"], ckpt["n_classes"]).to(device)
            mlp.load_state_dict(ckpt["state_dict"])
            mlp.eval()
            mlp_dict[level] = (mlp, ckpt["encoder"])
    
    print(f"Modèle chargé : {len(prototypes)} espèces | "
          f"MLP : {list(mlp_dict.keys())} | T={temperature:.2f}")
    
    return BioModel(
        prototypes=prototypes,
        tax_lookup=tax_lookup,
        whitening=whitening,
        temperature=temperature,
        mlp_dict=mlp_dict,
    )


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _lookup_parents(level: str, label: str, tax_lookup: dict) -> dict:
    """
    Trouve les niveaux supérieurs cohérents depuis tax_lookup.
    Cherche la première espèce qui a level=label et retourne sa taxo complète.
    """
    hierarchy = ["regne", "phylum", "classe", "ordre", "famille"]
    level_rank = hierarchy.index(level) if level in hierarchy else -1

    for tax in tax_lookup.values():
        if str(tax.get(level, "")) == label:
            # Retourne uniquement les niveaux AU-DESSUS du niveau confiant
            return {
                lvl: str(tax[lvl])
                for lvl in hierarchy
                if hierarchy.index(lvl) < level_rank and tax.get(lvl) and pd.notna(tax.get(lvl))
            }
    return {}


# ════════════════════════════════════════════════════════════════════════════
# WHITENING
# ════════════════════════════════════════════════════════════════════════════

def apply_whitening(
    features: np.ndarray,
    mean,
    components,
    variance
) -> np.ndarray:
    """Applique le whitening PCA aux features extraites."""
    x = (features - mean) @ components.T / np.sqrt(variance + 1e-8)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-8, None)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION INDIVIDUELLE
# ════════════════════════════════════════════════════════════════════════════

def predict(
    feat_w: np.ndarray,
    model: BioModel,
    threshold: float = THRESHOLD,
    margin_min: float = MARGIN_MIN
) -> dict:
    """
    Prédit le niveau taxonomique le plus fin possible depuis un vecteur de features.
    
    Args:
        feat_w: Vecteur numpy 256d whitened
        model: BioModel chargé avec load_model()
        threshold: Seuil de confiance
        margin_min: Garde-fou faux positifs (score_top1 - score_top2)
        
    Retourne:
        {
            "best_level": "famille",
            "best_label": "Fucaceae",
            "best_score": 0.87,
            "margin": 0.34,
            "path": "proto_clip" | "mlp",
            "all_scores": {...}
        }
    """
    feat_w = np.squeeze(feat_w)  # garantit shape (256,) même si (1,256) est passé
    feat_t = torch.tensor(feat_w[None], dtype=torch.float32).to(DEVICE)

    # === Proto-CLIP (espèce) ===
    species_list = list(model.prototypes.keys())
    proto_mat = np.vstack([model.prototypes[sp] for sp in species_list])
    
    sims = feat_w @ proto_mat.T
    logits = sims * model.temperature - sims.max() * model.temperature
    proto_proba = np.exp(logits) / np.exp(logits).sum()
    
    top_idx = proto_proba.argsort()[::-1]
    sp_top1 = species_list[top_idx[0]]
    sp_score = float(proto_proba[top_idx[0]])
    sp_margin = sp_score - float(proto_proba[top_idx[1]]) if len(top_idx) > 1 else sp_score
    
    # === MLP (niveaux supérieurs) ===
    mlp_scores = {}
    with torch.no_grad():
        for level, (mlp, enc) in model.mlp_dict.items():
            logits_l = mlp(feat_t).squeeze(0)
            proba_l = F.softmax(logits_l, dim=-1).detach().cpu().numpy()
            top_l = proba_l.argsort()[::-1]
            
            mlp_scores[level] = {
                "label": enc.classes_[top_l[0]],
                "score": float(proba_l[top_l[0]]),
                "margin": float(proba_l[top_l[0]]) - float(proba_l[top_l[1]])
                          if len(top_l) > 1 else float(proba_l[top_l[0]]),
                "top3": [
                    {"label": enc.classes_[i], "score": round(float(proba_l[i]), 4)}
                    for i in top_l[:3]
                ],
            }
    
    # === all_scores ===
    all_scores = {
        "species_name": [
            {"label": species_list[i], "score": round(float(proto_proba[i]), 4)}
            for i in top_idx[:3]
        ]
    }
    for level, ms in mlp_scores.items():
        all_scores[level] = ms["top3"]
    
    # === Décision hiérarchique ===
    sp_confident = (sp_score >= threshold and sp_margin >= margin_min)
    
    # Taxonomie complète (remplie selon le chemin proto_clip ou mlp)
    taxonomy = {level: None for level in ["regne", "phylum", "classe", "ordre", "famille", "species_name"]}

    if sp_confident:
        # Espèce confiante → lookup taxo
        tax = model.tax_lookup.get(sp_top1, {})
        best_level = "species_name"
        best_label = sp_top1
        best_score = sp_score
        path = "proto_clip"

        taxonomy["species_name"] = sp_top1
        for level in SUPERVISED_LEVELS:
            true_val = tax.get(level)
            if true_val and pd.notna(true_val):
                taxonomy[level] = str(true_val)
                existing = [e for e in all_scores.get(level, [])
                            if e["label"] != str(true_val)]
                all_scores[level] = [
                    {"label": str(true_val), "score": round(best_score, 4), "inherited": True}
                ] + existing[:2]
    else:
        # Pas assez confiant → MLP du niveau le plus fin confiant
        path = "mlp"
        best_level = None
        best_label = None
        best_score = 0.0

        for level in ["famille", "ordre", "classe", "phylum", "regne"]:
            if level not in mlp_scores:
                continue
            ms = mlp_scores[level]
            taxonomy[level] = ms["label"]
            if best_level is None and ms["score"] >= threshold and ms["margin"] >= margin_min:
                best_level = level
                best_label = ms["label"]
                best_score = ms["score"]

        # Niveaux supérieurs au best_level : lookup taxonomique pour cohérence
        if best_level and best_level != "regne":
            parents = _lookup_parents(best_level, best_label, model.tax_lookup)
            taxonomy.update(parents)

    return {
        "best_level": best_level,
        "best_label": best_label,
        "best_score": best_score,
        "margin": round(sp_margin, 4),
        "path": path,
        "all_scores": all_scores,
        # Taxonomie complète aplatie
        "regne": taxonomy["regne"],
        "phylum": taxonomy["phylum"],
        "classe": taxonomy["classe"],
        "ordre": taxonomy["ordre"],
        "famille": taxonomy["famille"],
        "species_name": taxonomy["species_name"],
    }


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION SUR IMAGE
# ════════════════════════════════════════════════════════════════════════════

def predict_image(
    image_path: str,
    model: BioModel,
    bioclip: Optional[BioCLIPExtractor] = None,
    threshold: float = THRESHOLD,
    margin_min: float = MARGIN_MIN
) -> dict:
    """
    Prédit directement depuis un chemin d'image.
    
    Args:
        image_path: Chemin vers l'image
        model: BioModel chargé
        bioclip: Extracteur BioCLIP (créé si None)
        threshold: Seuil de confiance
        margin_min: Marge minimum
        
    Retourne:
        Résultat de prédiction
    """
    if bioclip is None:
        bioclip = BioCLIPExtractor()
    
    # Charger et transformer l'image
    img = Image.open(image_path).convert("RGB")
    
    # Extraire features BioCLIP
    feat_512 = bioclip.extract(img)
    
    # Appliquer whitening
    wh_mean, wh_comp, wh_var = model.whitening
    feat_w = apply_whitening(feat_512, wh_mean, wh_comp, wh_var)
    
    # Prédire
    return predict(feat_w, model, threshold, margin_min)


def predict_pil_image(
    pil_img: Image.Image,
    model: BioModel,
    bioclip: Optional[BioCLIPExtractor] = None,
    threshold: float = THRESHOLD,
    margin_min: float = MARGIN_MIN
) -> dict:
    """
    Prédit depuis une image PIL (depuis S3 par exemple).
    """
    if bioclip is None:
        bioclip = BioCLIPExtractor()
    
    feat_512 = bioclip.extract(pil_img)
    wh_mean, wh_comp, wh_var = model.whitening
    feat_w = apply_whitening(feat_512, wh_mean, wh_comp, wh_var)
    
    return predict(feat_w, model, threshold, margin_min)


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION PAR BATCH
# ════════════════════════════════════════════════════════════════════════════

def predict_batch(
    images: list[Image.Image],
    model: BioModel,
    bioclip: Optional[BioCLIPExtractor] = None,
    threshold: float = THRESHOLD,
    margin_min: float = MARGIN_MIN
) -> list[dict]:
    """
    Prédit sur un batch d'images
    
    Args:
        images: Liste d'images PIL
        model: BioModel chargé
        bioclip: Extracteur BioCLIP
        threshold: Seuil de confiance
        margin_min: Marge minimum
        
    Retourne:
        Liste de résultats de prédiction
    """
    if bioclip is None:
        bioclip = BioCLIPExtractor()
    
    # Extraction batch
    features_512 = bioclip.extract_batch(images)
    
    # Whitening
    wh_mean, wh_comp, wh_var = model.whitening
    features_w = apply_whitening(features_512, wh_mean, wh_comp, wh_var)
    
    # Prédiction individuelle
    results = []
    for feat_w in features_w:
        results.append(predict(feat_w[None], model, threshold, margin_min))
    
    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN (test rapide)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference classification")
    parser.add_argument("--images", type=str, help="Dossier ou fichier image")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--margin", type=float, default=MARGIN_MIN)
    args = parser.parse_args()
    
    # Charger le modèle
    model = load_model()
    
    if args.images:
        from pathlib import Path
        path = Path(args.images)
        
        if path.is_dir():
            # Dossier d'images
            from classifier_bioclip import IMG_TRANSFORM
            images = []
            for f in path.glob("*"):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    images.append(Image.open(f).convert("RGB"))
            
            results = predict_batch(images, model, threshold=args.threshold, margin_min=args.margin)
            for img_path, result in zip(path.glob("*"), results):
                print(f"{img_path.name}: {result['best_level']} / {result['best_label']} ({result['best_score']:.2f})")
        else:
            # Fichier unique
            result = predict_image(str(path), model, threshold=args.threshold, margin_min=args.margin)
            print(f"Résultat: {result['best_level']} / {result['best_label']} ({result['best_score']:.2f})")
            print(f"  Path: {result['path']}")
            print(f"  Margin: {result['margin']}")
    else:
        print("Usage: python classifier_infer.py --images <dossier|fichier>")