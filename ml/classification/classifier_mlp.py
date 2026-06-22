"""
 Entraînement des MLP par niveau taxonomique
===============================================================
Classificateurs MLP pour les niveaux
taxonomiques supérieurs (règne → famille).

"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    DEVICE,
    PROTO_WHITENING_DIM as WHITENING_DIM,
    MLP_EPOCHS,
    MLP_DROPOUT,
    MLP_MODEL_FILE,
    SUPERVISED_LEVELS,
)


# ════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE MLP
# ════════════════════════════════════════════════════════════════════════════

class LevelMLP(nn.Module):
    """
    MLP pour un niveau taxonomique.

    Architecture :
        Linear(input_dim, hidden) → BatchNorm → ReLU → Dropout → Linear(hidden, n_classes)
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        dropout: float = MLP_DROPOUT
    ):
        super().__init__()
        hidden = max(256, n_classes * 8)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class FeaturesDataset(Dataset):
    """Dataset sur features pré-calculées."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ════════════════════════════════════════════════════════════════════════════

def train_level_mlps(
    features_w: np.ndarray,
    df: pd.DataFrame,
    levels: list[str] = None,
    epochs: int = MLP_EPOCHS,
    dropout: float = MLP_DROPOUT,
    verbose: bool = True
) -> dict:
    """
    Entraîne un MLP par niveau taxonomique.

    Args:
        features_w: Features whitened (N x 256)
        df: DataFrame avec les labels taxonomiques
        levels: Niveaux à entraîner (défaut: SUPERVISED_LEVELS)
        epochs: Nombre d'epochs
        dropout: Taux de dropout
        verbose: Afficher la progression

    Retourne:
        Dict {niveau → (LevelMLP, LabelEncoder)}
    """
    levels = levels or SUPERVISED_LEVELS

    # Filtrer les données avec species_name
    df_lab = df[df["species_name"].notna()].reset_index(drop=True)

    if len(df_lab) == 0:
        raise ValueError("Aucune donnée labellisée trouvée")

    mlp_dict = {}

    for level in levels:
        if level not in df_lab.columns:
            if verbose:
                print(f"  Colonne '{level}' absente — skip")
            continue

        # Filtrer les données pour ce niveau
        df_lvl = df_lab[df_lab[level].notna()].reset_index(drop=True)

        if len(df_lvl) == 0:
            if verbose:
                print(f"  Aucune donnée pour '{level}' — skip")
            continue

        # Encoder les labels
        enc = LabelEncoder()
        labels = enc.fit_transform(df_lvl[level])
        n_cls = len(enc.classes_)

        # Features correspondantes
        feats_l = features_w[df_lvl.index]

        # Weighted sampler pour équilibrer les classes
        counts = df_lvl[level].value_counts()
        weights = df_lvl[level].map(lambda x: 1.0 / max(counts[x], 1)).values
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float32),
            num_samples=len(df_lvl), replacement=True
        )

        # DataLoader
        ds = FeaturesDataset(feats_l, labels)
        loader = DataLoader(ds, batch_size=64, sampler=sampler)

        # Modèle
        mlp = LevelMLP(WHITENING_DIM, n_cls, dropout).to(DEVICE)
        opt = AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = CosineAnnealingLR(opt, T_max=epochs)

        # Entraînement
        mlp.train()
        for epoch in range(epochs):
            total, n = 0.0, 0
            for feats_b, labels_b in loader:
                feats_b = feats_b.to(DEVICE)
                labels_b = labels_b.to(DEVICE)

                loss = F.cross_entropy(mlp(feats_b), labels_b)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                opt.step()

                total += loss.item()
                n += 1

            sch.step()

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                print(f"    [{level} — {n_cls} cls] "
                      f"epoch {epoch+1}/{epochs} loss={total/max(n,1):.4f}")

        mlp.eval()
        mlp_dict[level] = (mlp, enc)

        if verbose:
            print(f"  ✓ {level} ({n_cls} classes)")

    return mlp_dict


# ════════════════════════════════════════════════════════════════════════════
# SAUVEGARDE / CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════

def save_mlp_model(
    mlp_dict: dict,
    path: Path = None
) -> Path:
    """
    Sauvegarde les MLP entraînés.

    Args:
        mlp_dict: Dict {niveau → (LevelMLP, LabelEncoder)}
        path: Chemin de sauvegarde (défaut: MLP_MODEL_FILE)

    Retourne:
        Chemin du fichier sauvegardé
    """
    path = path or MLP_MODEL_FILE

    save_data = {}
    for level, (mlp, enc) in mlp_dict.items():
        save_data[level] = {
            "input_dim": WHITENING_DIM,
            "n_classes": len(enc.classes_),
            "state_dict": mlp.state_dict(),
            "encoder": enc,
        }

    torch.save(save_data, path)
    print(f"  MLP sauvegardés → {path}")
    return path


def load_mlp_model(
    path: Path = None,
    device: torch.device = DEVICE
) -> dict:
    """
    Charge les MLP entraînés.

    Args:
        path: Chemin du fichier (défaut: MLP_MODEL_FILE)
        device: Device PyTorch

    Retourne:
        Dict {niveau → (LevelMLP, LabelEncoder)}
    """
    path = path or MLP_MODEL_FILE

    if not path.exists():
        raise FileNotFoundError(f"Modèle MLP introuvable: {path}")

    save_data = torch.load(path, map_location=device, weights_only=False)

    mlp_dict = {}
    for level, ckpt in save_data.items():
        n_classes = ckpt["n_classes"]
        mlp = LevelMLP(ckpt["input_dim"], n_classes).to(device)
        mlp.load_state_dict(ckpt["state_dict"])
        mlp.eval()
        mlp_dict[level] = (mlp, ckpt["encoder"])

    print(f"  MLP chargés: {list(mlp_dict.keys())}")
    return mlp_dict


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION MLP (inference)
# ════════════════════════════════════════════════════════════════════════════

def predict_level(
    features_w: np.ndarray,
    mlp_dict: dict,
    level: str,
    threshold: float = 0.5
) -> dict:
    """
    Prédit un niveau taxonomique avec le MLP correspondant.

    Args:
        features_w: Features whitened (N x 256)
        mlp_dict: Dict des MLP chargés
        level: Niveau taxonomique
        threshold: Seuil de confiance

    Retourne:
        Dict avec 'label', 'score', 'margin', 'top3'
    """
    if level not in mlp_dict:
        return {"label": None, "score": 0.0, "margin": 0.0, "top3": []}

    mlp, enc = mlp_dict[level]
    feat_t = torch.tensor(features_w, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = mlp(feat_t)
        proba = F.softmax(logits, dim=-1).cpu().numpy()

    # Top predictions
    top_idx = proba[0].argsort()[::-1]
    top1_idx = top_idx[0]

    score = float(proba[0][top1_idx])
    margin = score - float(proba[0][top_idx[1]]) if len(top_idx) > 1 else score

    return {
        "label": enc.classes_[top1_idx],
        "score": score,
        "margin": margin,
        "top3": [
            {"label": enc.classes_[i], "score": round(float(proba[0][i]), 4)}
            for i in top_idx[:3]
        ]
    }


def predict_all_levels(
    features_w: np.ndarray,
    mlp_dict: dict,
    threshold: float = 0.5,
    margin_min: float = 0.1
) -> dict:
    """
    Prédit tous les niveaux taxonomiques et retourne le meilleur.

    Args:
        features_w: Features whitened (N x 256)
        mlp_dict: Dict des MLP chargés
        threshold: Seuil de confiance
        margin_min: Marge minimum (score_top1 - score_top2)

    Retourne:
        Dict avec 'best_level', 'best_label', 'best_score', 'all_scores'
    """
    mlp_scores = {}

    for level in SUPERVISED_LEVELS:
        if level in mlp_dict:
            result = predict_level(features_w, mlp_dict, level)
            mlp_scores[level] = result

    # Trouver le meilleur niveau confident
    best_level = None
    best_label = None
    best_score = 0.0

    for level in ["famille", "ordre", "classe", "phylum", "regne"]:
        if level not in mlp_scores:
            continue
        ms = mlp_scores[level]
        if ms["score"] >= threshold and ms["margin"] >= margin_min:
            best_level = level
            best_label = ms["label"]
            best_score = ms["score"]
            break

    return {
        "best_level": best_level,
        "best_label": best_label,
        "best_score": best_score,
        "all_scores": mlp_scores,
    }