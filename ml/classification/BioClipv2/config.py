"""
config.py — Configuration centrale de la pipeline
"""

import os
import torch
from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent
DATA_DIR      = ROOT_DIR / "data/dataset_biolit"
IMAGES_DIR    = DATA_DIR / "images"
LABELS_DIR    = DATA_DIR / "labels"
TAXONOMY_FILE = DATA_DIR / "taxref.parquet"
RESULTS_DIR   = ROOT_DIR / "results"
CROPS_DIR     = RESULTS_DIR / "crops"
REJECTED_DIR  = RESULTS_DIR / "rejected"
REPORTS_DIR   = RESULTS_DIR / "reports"


# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Niveaux taxonomiques ───────────────────────────────────────────────────────
TAXONOMY_LEVELS = ["regne", "phylum", "classe", "ordre", "famille", "sous_famille", "species_name"]

# ── Proto-CLIP classifier (classifier.py) ─────────────────────────────────────
BIOCLIP_MODEL_ID     = "hf-hub:imageomics/bioclip-2"
CONFIDENCE_THRESHOLD = 0.5    # seuil arrêt hiérarchique
PROTO_ALPHA          = 0.6    # fusion visuel/textuel Proto-CLIP
PROTO_WHITENING_DIM  = 256    # 512d → 256d après PCA whitening
PROTO_BATCH_SIZE     = 32     # batch extraction features BioCLIP2

# ── MLP par niveau taxonomique ────────────────────────────────────────────────
MLP_EPOCHS     = 40     # epochs par MLP (règne, phylum, classe, ordre, famille)
MLP_DROPOUT    = 0.3    # régularisation MLP

# ── margin score ─────────────────────────────────────────────────────
# Rejette une prédiction si score_top1 - score_top2 < MARGIN_MIN
# même si score_top1 ≥ CONFIDENCE_THRESHOLD
# Évite les faux positifs à haute confiance (ex: 91% mais faux)
# Recommandé : 0.05 (permissif) à 0.20 (strict)
MARGIN_MIN     = 0.10

# ── Fichiers de sortie ─────────────────────────────────────────────────────────
PROTO_CACHE_FILE  = RESULTS_DIR / "bioclip_features.npz"
PROTO_MODEL_FILE  = RESULTS_DIR / "proto_model.npz"
PROTO_TAX_FILE    = RESULTS_DIR / "tax_lookup.pkl"
MLP_MODEL_FILE    = RESULTS_DIR / "mlp_model.pt"

# ── Rapport HTML ───────────────────────────────────────────────────────────────
CLASSIFY_IMAGES_DIR  = IMAGES_DIR / "identifiable"
PROTO_REPORT_FILE    = RESULTS_DIR / "report_proto_clip.html"
PROTO_REPORT_GALLERY = 60