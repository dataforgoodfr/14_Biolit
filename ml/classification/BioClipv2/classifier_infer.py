"""
classifier_infer.py — Inférence uniquement
==========================================
Ce fichier est le seul nécessaire en production.
Il charge le modèle sauvegardé et prédit sur de nouvelles images.

Dépendances minimales :
    pip install torch numpy pandas pillow

BioCLIP2 n'est PAS nécessaire ici — les prototypes et le whitening
sont déjà calculés et sauvegardés dans results/.

Fichiers requis dans results/ :
    proto_model.npz   → prototypes + whitening PCA
    tax_lookup.pkl    → hiérarchie taxonomique
    mlp_model.pt      → MLP par niveau (règne → famille)

Usage :
    from classifier_infer import load_model, predict_image

    model = load_model()
    result = predict_image("mon_image.jpg", model)
    print(result["best_level"], result["best_label"], result["best_score"])

    # Ou en ligne de commande
    python classifier_infer.py --images mon_dossier/
    python classifier_infer.py --images mon_dossier/ --threshold 0.6
"""

import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torchvision import transforms

from config import (
    DEVICE,
    RESULTS_DIR,
    CONFIDENCE_THRESHOLD as THRESHOLD,
    PROTO_WHITENING_DIM  as WHITENING_DIM,
    PROTO_MODEL_FILE     as MODEL_FILE,
    PROTO_TAX_FILE       as TAX_LOOKUP_FILE,
    MARGIN_MIN,
    MLP_MODEL_FILE,
)

# Niveaux du plus fin au plus large
ALL_LEVELS        = ["species_name", "famille", "ordre", "classe", "phylum", "regne"]
SUPERVISED_LEVELS = ["regne", "phylum", "classe", "ordre", "famille"]

# Normalisation BioCLIP
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


# ── Architecture MLP (doit être identique à classifier_train.py) ──────────────

class LevelMLP(nn.Module):
    """MLP léger pour un niveau taxonomique. Doit être identique à classifier_train."""
    def __init__(self, input_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        hidden = max(256, n_classes * 8)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── Structure du modèle chargé ────────────────────────────────────────────────

@dataclass
class BioModel:
    """
    Conteneur pour tous les artefacts du modèle.
    Passé tel quel à predict_image() — pas besoin de déstructurer.
    """
    prototypes:  dict    # {espèce → vecteur 256d}
    tax_lookup:  dict    # {espèce → {famille, ordre, classe, phylum, règne}}
    whitening:   tuple   # (mean, components, variance) pour apply_whitening
    temperature: float   # température calibrée pour Proto-CLIP
    mlp_dict:    dict    # {niveau → (LevelMLP, LabelEncoder)}


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU MODÈLE
# ════════════════════════════════════════════════════════════════════════════

def load_model(results_dir: Path = RESULTS_DIR) -> BioModel:
    """
    Charge le modèle depuis results/.
    BioCLIP2 n'est PAS nécessaire.

    Paramètre :
        results_dir : dossier contenant les fichiers sauvegardés
                      (défaut : results/ depuis config.py)

    Retourne un BioModel prêt pour predict_image().
    """
    model_file    = results_dir / MODEL_FILE.name
    tax_file      = results_dir / TAX_LOOKUP_FILE.name
    mlp_file      = results_dir / MLP_MODEL_FILE.name

    if not model_file.exists():
        raise FileNotFoundError(
            f"{model_file} introuvable.\n"
            "Entraînez d'abord : python classifier_train.py --fit --images ..."
        )

    # Proto-CLIP
    data         = np.load(model_file, allow_pickle=True)
    species_list = list(data["species_list"])
    proto_mat    = data["prototypes"]
    prototypes   = {sp: proto_mat[i] for i, sp in enumerate(species_list)}
    whitening    = (data["wh_mean"], data["wh_components"], data["wh_variance"])
    temperature  = float(data["temperature"][0])

    with open(tax_file, "rb") as f:
        tax_lookup = pickle.load(f)

    # MLP par niveau
    mlp_dict = {}
    if mlp_file.exists():
        mlp_save = torch.load(mlp_file, map_location=DEVICE, weights_only=False)
        for level, ckpt in mlp_save.items():
            mlp = LevelMLP(ckpt["input_dim"], ckpt["n_classes"]).to(DEVICE)
            mlp.load_state_dict(ckpt["state_dict"])
            mlp.eval()
            mlp_dict[level] = (mlp, ckpt["encoder"])

    print(f"Modèle chargé : {len(prototypes)} espèces | "
          f"MLP : {list(mlp_dict.keys())} | T={temperature:.2f}")

    return BioModel(
        prototypes  = prototypes,
        tax_lookup  = tax_lookup,
        whitening   = whitening,
        temperature = temperature,
        mlp_dict    = mlp_dict,
    )


# ════════════════════════════════════════════════════════════════════════════
# WHITENING (appliqué aux features extraites)
# ════════════════════════════════════════════════════════════════════════════

def apply_whitening(features: np.ndarray, mean, components, variance) -> np.ndarray:
    x     = (features - mean) @ components.T / np.sqrt(variance + 1e-8)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-8, None)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION
# ════════════════════════════════════════════════════════════════════════════

def predict(feat_w:     np.ndarray,
            model:      BioModel,
            threshold:  float = THRESHOLD,
            margin_min: float = MARGIN_MIN) -> dict:
    """
    Prédit le niveau taxonomique le plus fin possible depuis un vecteur de features.

    Paramètre d'entrée :
        feat_w     : vecteur numpy 256d whitened (sortie de apply_whitening)
        model      : BioModel chargé avec load_model()
        threshold  : seuil de confiance (défaut config: CONFIDENCE_THRESHOLD)
        margin_min : garde-fou faux positifs (défaut config: MARGIN_MIN)

    Retourne :
        {
            "best_level":  "famille",      ← niveau le plus fin accepté
            "best_label":  "Fucaceae",     ← label prédit
            "best_score":  0.87,           ← score de confiance
            "margin":      0.34,           ← écart top1/top2 (qualité)
            "path":        "proto_clip",   ← chemin utilisé
            "all_scores":  {               ← scores à tous les niveaux
                "species_name": [...],
                "famille": [...],
                ...
            }
        }
    """
    feat_t = torch.tensor(feat_w[None], dtype=torch.float32).to(DEVICE)

    # ── Proto-CLIP espèce ──────────────────────────────────────────────────────
    species_list = list(model.prototypes.keys())
    proto_mat    = np.vstack([model.prototypes[sp] for sp in species_list])
    sims         = feat_w @ proto_mat.T
    logits       = sims * model.temperature - sims.max() * model.temperature
    proto_proba  = np.exp(logits) / np.exp(logits).sum()

    top_idx   = proto_proba.argsort()[::-1]
    sp_top1   = species_list[top_idx[0]]
    sp_score  = float(proto_proba[top_idx[0]])
    sp_margin = sp_score - float(proto_proba[top_idx[1]]) if len(top_idx) > 1 else sp_score

    # ── MLP niveaux supérieurs ─────────────────────────────────────────────────
    mlp_scores = {}
    with torch.no_grad():
        for level, (mlp, enc) in model.mlp_dict.items():
            logits_l = mlp(feat_t).squeeze(0)
            proba_l  = F.softmax(logits_l, dim=-1).detach().cpu().numpy()
            top_l    = proba_l.argsort()[::-1]
            mlp_scores[level] = {
                "label":  enc.classes_[top_l[0]],
                "score":  float(proba_l[top_l[0]]),
                "margin": float(proba_l[top_l[0]]) - float(proba_l[top_l[1]])
                          if len(top_l) > 1 else float(proba_l[top_l[0]]),
                "top3":   [{"label": enc.classes_[i], "score": round(float(proba_l[i]), 4)}
                           for i in top_l[:3]],
            }

    # ── Construire all_scores ──────────────────────────────────────────────────
    all_scores = {
        "species_name": [
            {"label": species_list[i], "score": round(float(proto_proba[i]), 4)}
            for i in top_idx[:3]
        ]
    }
    for level, ms in mlp_scores.items():
        all_scores[level] = ms["top3"]

    # ── Décision hiérarchique cohérente ───────────────────────────────────────
    sp_confident = (sp_score >= threshold and sp_margin >= margin_min)

    if sp_confident:
        # Espèce confiante → lookup taxo (cohérence garantie)
        tax        = model.tax_lookup.get(sp_top1, {})
        best_level = "species_name"
        best_label = sp_top1
        best_score = sp_score
        path       = "proto_clip"
        for level in SUPERVISED_LEVELS:
            true_val = tax.get(level)
            if true_val and pd.notna(true_val):
                existing = [e for e in all_scores.get(level, [])
                            if e["label"] != str(true_val)]
                all_scores[level] = [
                    {"label": str(true_val), "score": round(best_score, 4), "inherited": True}
                ] + existing[:2]
    else:
        # Pas assez confiant → MLP du niveau le plus fin confiant
        path       = "mlp"
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
        "margin":     round(sp_margin, 4),
        "path":       path,
        "all_scores": all_scores,
    }


def predict_image(image_path: str,
                  model:      BioModel,
                  bioclip               = None,
                  threshold:  float     = THRESHOLD,
                  margin_min: float     = MARGIN_MIN) -> dict:
    """
    Prédit directement depuis un chemin d'image.

    Si bioclip est None, importe BioCLIP2 à la volée (un peu plus lent).
    Pour traiter beaucoup d'images, passer bioclip explicitement pour
    ne le charger qu'une fois.

    Exemple :
        model  = load_model()
        result = predict_image("photo.jpg", model)

        # Ou pour un lot d'images (plus efficace) :
        from classifier_train import load_bioclip
        bioclip = load_bioclip()
        for path in images:
            result = predict_image(path, model, bioclip)
    """
    if bioclip is None:
        try:
            import open_clip
            bc, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:imageomics/bioclip-2"
            )
            bc = bc.to(DEVICE).eval()
            bioclip = bc
        except ImportError:
            raise ImportError(
                "BioCLIP2 requis pour lire les images.\n"
                "pip install open-clip-torch\n"
                "Ou pré-extrayez les features avec classifier_train.py."
            )

    with torch.no_grad():
        img  = Image.open(image_path).convert("RGB")
        x    = IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        feat = bioclip.encode_image(x)
        feat = F.normalize(feat, dim=-1).squeeze(0).cpu().numpy()

    feat_w = apply_whitening(feat[None], *model.whitening)[0]
    return predict(feat_w, model, threshold, margin_min)


# ════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE — PRÉDICTION EN BATCH
# ════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Inférence — classifier Biolit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python classifier_infer.py --images mon_dossier/
  python classifier_infer.py --images mon_dossier/ --threshold 0.6 --margin 0.15

Paramètres dans config.py :
  CONFIDENCE_THRESHOLD = 0.5   (seuil d'acceptation)
  MARGIN_MIN           = 0.10  (garde-fou faux positifs)
        """
    )
    parser.add_argument("--images",    type=str, required=True,
                        help="Dossier contenant les images à prédire")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Seuil de confiance (défaut {THRESHOLD})")
    parser.add_argument("--margin",    type=float, default=MARGIN_MIN,
                        help=f"Margin min top1/top2 (défaut {MARGIN_MIN})")
    parser.add_argument("--output",    type=str,   default=None,
                        help="Chemin CSV de sortie (défaut results/predictions.csv)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"Dossier introuvable : {images_dir}")
        return

    output = Path(args.output) if args.output else RESULTS_DIR / "predictions.csv"

    print("Chargement du modèle...")
    model = load_model()

    print("Chargement BioCLIP2 pour l'extraction features...")
    try:
        import open_clip
        bc, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
        bioclip  = bc.to(DEVICE).eval()
    except ImportError:
        print("⚠  open_clip manquant — pip install open-clip-torch")
        return

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    images     = [p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    print(f"\n{len(images)} images à prédire...")

    rows = []
    t0   = time.time()
    for img_path in images:
        try:
            result = predict_image(str(img_path), model, bioclip,
                                   args.threshold, args.margin)
            sp = result["all_scores"].get("species_name", [{}])
            rows.append({
                "image":       img_path.name,
                "best_level":  result["best_level"] or "rejeté",
                "best_label":  result["best_label"] or "—",
                "best_score":  f"{result['best_score']:.0%}",
                "margin":      f"{result['margin']:.3f}",
                "path":        result["path"],
                "top1_espece": sp[0].get("label", "—") if sp else "—",
                "conf_espece": f"{sp[0].get('score', 0):.0%}" if sp else "0%",
                "top2_espece": sp[1].get("label", "—") if len(sp) > 1 else "—",
            })
        except Exception as e:
            rows.append({"image": img_path.name, "best_level": "erreur", "erreur": str(e)})

    elapsed = time.time() - t0
    df_res  = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output, index=False)

    accepted = df_res[df_res["best_level"] != "rejeté"]
    print(f"\n{len(images)} images en {elapsed:.1f}s "
          f"({elapsed/max(len(images),1)*1000:.0f}ms/image)")
    print(f"Acceptées  : {len(accepted)}/{len(images)} "
          f"({len(accepted)/max(len(images),1):.0%})")
    print(f"Niveaux    : {df_res['best_level'].value_counts().to_dict()}")
    print(f"Résultats  → {output}")


if __name__ == "__main__":
    main()