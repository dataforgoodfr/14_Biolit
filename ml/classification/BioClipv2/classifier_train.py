"""
classifier_train.py — Entraînement du modèle
=============================================
Ce fichier gère tout ce qui touche à l'entraînement.
Une fois le modèle sauvegardé, seul classifier_infer.py est nécessaire.

Ce qui est fait ici :
    - Extraction des features BioCLIP2 (avec cache sur disque)
    - Whitening PCA (512d → 256d)
    - Entraînement d'un MLP par niveau taxonomique (règne → famille)
    - Calcul des prototypes Proto-CLIP (espèce, few-shot)
    - Apprentissage de la température
    - Sauvegarde de tous les artefacts dans results/
    - Évaluation propre (split 80/20 sans data leakage)
    - Mise à jour du modèle pour de nouvelles espèces

Dépendances :
    pip install open-clip-torch torch torchvision scikit-learn pandas pillow

Usage :
    python classifier_train.py --fit --eval --images data/images/identifiable
    python classifier_train.py --fit         --images data/images/identifiable
    python classifier_train.py --eval        --images data/images/identifiable
    python classifier_train.py --update      --images nouvelles_images/
"""

import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    import open_clip
    BIOCLIP_AVAILABLE = True
except ImportError:
    BIOCLIP_AVAILABLE = False
    print("⚠  open_clip manquant — pip install open-clip-torch")

from config import (
    DEVICE,
    RESULTS_DIR,
    TAXONOMY_LEVELS,
    BIOCLIP_MODEL_ID     as BIOCLIP_MODEL,
    CONFIDENCE_THRESHOLD as THRESHOLD,
    PROTO_ALPHA          as ALPHA,
    PROTO_WHITENING_DIM  as WHITENING_DIM,
    PROTO_BATCH_SIZE     as BATCH_SIZE,
    PROTO_CACHE_FILE     as CACHE_FILE,
    MARGIN_MIN,
    MLP_EPOCHS,
    MLP_DROPOUT,
    PROTO_MODEL_FILE  as MODEL_FILE,
    PROTO_TAX_FILE    as TAX_LOOKUP_FILE,
    MLP_MODEL_FILE,
)
from classifier_infer import (
    LevelMLP, BioModel, apply_whitening,
    predict, ALL_LEVELS, SUPERVISED_LEVELS,
    IMG_TRANSFORM,
)
from build_classify_dataset import build_dataset

import pickle


# ── Descriptions textuelles pour Proto-CLIP ───────────────────────────────────
# Pour toute espèce absente → template automatique depuis taxonomy.parquet

SPECIES_DESCRIPTIONS = {
    "Actinia equina":           "a photo of Actinia equina, beadlet anemone, red or green column with blue acrorhagi beads, rocky intertidal shore",
    "Anemonia viridis":         "a photo of Anemonia viridis, snakelocks anemone, long green tentacles with purple tips, shallow rocky coast",
    "Asterias rubens":          "a photo of Asterias rubens, common starfish, five-armed orange brown seastar on mussel beds",
    "Carcinus maenas":          "a photo of Carcinus maenas, European green shore crab, five frontal teeth, olive to dark green carapace",
    "Eriphia verrucosa":        "a photo of Eriphia verrucosa, warty crab, robust dark brown crab with unequal claws and knobbly carapace",
    "Fucus serratus":           "a photo of Fucus serratus, serrated wrack, brown seaweed with finely serrated frond edges no bladders",
    "Fucus spiralis":           "a photo of Fucus spiralis, spiral wrack, twisted narrow brown fronds at top of shore without bladders",
    "Fucus vesiculosus":        "a photo of Fucus vesiculosus, bladder wrack, brown seaweed with paired spherical air bladders on rocky shore",
    "Ascophyllum nodosum":      "a photo of Ascophyllum nodosum, egg wrack, long yellowish-brown seaweed with single egg-shaped air bladders",
    "Mytilus edulis":           "a photo of Mytilus edulis, blue mussel, elongated blue-black bivalve shell in dense clusters on rocks",
    "Mytilus galloprovincialis":"a photo of Mytilus galloprovincialis, Mediterranean mussel, elongated blue-black bivalve in dense clusters on rocks",
    "Nucella lapillus":         "a photo of Nucella lapillus, dog whelk, robust spiral shell banded white grey brown near barnacles",
    "Octopus vulgaris":         "a photo of Octopus vulgaris, common octopus, eight-armed mollusc with large rounded head and suckers",
    "Pachygrapsus marmoratus":  "a photo of Pachygrapsus marmoratus, marbled rock crab, square dark carapace with white marbled pattern on rocks",
    "Palaemon elegans":         "a photo of Palaemon elegans, rock pool prawn, small translucent slim prawn in tide pools",
    "Palaemon serratus":        "a photo of Palaemon serratus, common prawn, transparent shrimp with red-brown bands on body and claws",
    "Paracentrotus lividus":    "a photo of Paracentrotus lividus, purple sea urchin, dark purple-brown spiny echinoderm on rocky shores",
    "Patella vulgata":          "a photo of Patella vulgata, common limpet, oval ribbed conical shell clamped on rock, intertidal gastropod",
    "Patella ulyssiponensis":   "a photo of Patella ulyssiponensis, chinaman hat limpet, flat ribbed limpet with orange foot on Atlantic rocks",
    "Pecten maximus":           "a photo of Pecten maximus, great scallop, large fan-shaped ribbed bivalve with equal ears on sandy seabed",
    "Corallina officinalis":    "a photo of Corallina officinalis, coral weed, jointed pink calcified alga forming tufts in rock pools",
    "Ulva lactuca":             "a photo of Ulva lactuca, sea lettuce, bright green flat translucent alga sheets on rocky shore",
    "Semibalanus balanoides":   "a photo of Semibalanus balanoides, acorn barnacle, white volcano-shaped encrusting shell in dense colonies on rock",
    "Littorina littorea":       "a photo of Littorina littorea, common periwinkle, small dark rounded snail shell on rocks or seaweed",
    "Cancer pagurus":           "a photo of Cancer pagurus, edible crab, large brown oval crab with pie-crust edge carapace",
    "Necora puber":             "a photo of Necora puber, velvet swimming crab, dark blue-brown crab with red eyes and velvety hairy legs",
    "Maja brachydactyla":       "a photo of Maja brachydactyla, spiny spider crab, triangular spiny carapace with long thin legs",
    "Marthasterias glacialis":  "a photo of Marthasterias glacialis, spiny starfish, large pale starfish with prominent spines on each arm",
    "Sepia officinalis":        "a photo of Sepia officinalis, common cuttlefish, broad flat cephalopod with brown zebra-striped pattern",
    "Himanthalia elongata":     "a photo of Himanthalia elongata, thongweed, long strap-like brown seaweed from button-shaped holdfast on lower shore",
    "Pelvetia canaliculata":    "a photo of Pelvetia canaliculata, channelled wrack, yellowish-brown seaweed with inrolled channel-shaped fronds above tide",
    "Sabellaria alveolata":     "a photo of Sabellaria alveolata, honeycomb worm, reef-building polychaete forming honeycomb-textured sand tubes on rock",
    "Posidonia oceanica":       "a photo of Posidonia oceanica, Neptune seagrass, long ribbon-like green leaves in Mediterranean underwater meadows",
    "Gibbula umbilicalis":      "a photo of Gibbula umbilicalis, purple top shell, flattened spiral shell with purple streaks under rocks",
    "Galathea squamifera":      "a photo of Galathea squamifera, squat lobster, flattened brown crustacean with broad abdomen under rocks",
    "Codium tomentosum":        "a photo of Codium tomentosum, velvet horn, dark green spongy dichotomously branching alga on rocks",
    "Cerastoderma edule":       "a photo of Cerastoderma edule, common cockle, ribbed rounded bivalve with 22 to 28 ribs on sandy intertidal sediment",
    "Magallana gigas":          "a photo of Magallana gigas, Pacific oyster, large irregularly shaped craggy cupped bivalve in intertidal clusters",
    "Haliotis tuberculata tuberculata": "a photo of Haliotis tuberculata, green ormer, flat ear-shaped shell with row of holes and iridescent nacre",
    "Ligia oceanica":           "a photo of Ligia oceanica, sea slater, large grey isopod crustacean running on rocky supralittoral",
    "Padina pavonica":          "a photo of Padina pavonica, peacock tail, fan-shaped brown alga with concentric white bands Mediterranean",
    "Sargassum muticum":        "a photo of Sargassum muticum, wireweed, bushy brown invasive alga with long main axis and small oval air bladders on rock",
    "Phorcus lineatus":         "a photo of Phorcus lineatus, toothed top shell, small conical snail with zebra-striped shell on wet rocks",
    "Aplysia punctata":         "a photo of Aplysia punctata, spotted sea hare, large soft purple-brown mollusc with wing-like parapodia",
    "Zostera marina":           "a photo of Zostera marina, common eelgrass, narrow ribbon-like green seagrass in sheltered bays and estuaries",
}


# ════════════════════════════════════════════════════════════════════════════
# 1. BIOCLIP2 — EXTRACTION FEATURES
# ════════════════════════════════════════════════════════════════════════════

def load_bioclip():
    """Charge BioCLIP2. Nécessaire uniquement pour l'entraînement."""
    print(f"  Chargement BioCLIP2 sur {DEVICE}...")
    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_MODEL)
    model = model.to(DEVICE).eval()
    print(f"  BioCLIP2 prêt en {time.time() - t0:.1f}s")
    return model


@torch.no_grad()
def extract_features(df: pd.DataFrame, bioclip, use_cache: bool = True) -> np.ndarray:
    """
    Extrait les features BioCLIP2 et les cache sur disque.
    Empreinte : ~0.5 kWh pour 2000 images sur GPU — fait UNE SEULE FOIS.
    """
    if use_cache and CACHE_FILE.exists():
        cache = np.load(CACHE_FILE, allow_pickle=True)
        if list(cache["image_paths"]) == list(df["image_path"]):
            print("  Features chargées depuis le cache")
            return cache["features"]
        print("  Cache obsolète — réextraction...")

    print(f"  Extraction BioCLIP2 ({len(df)} images) "
          f"— ~{len(df) * 0.00025:.3f} kWh estimés...")
    t0, all_feats = time.time(), []

    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE]
        imgs  = []
        for _, row in batch.iterrows():
            try:
                imgs.append(IMG_TRANSFORM(Image.open(row["image_path"]).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        feats = bioclip.encode_image(torch.stack(imgs).to(DEVICE))
        all_feats.append(F.normalize(feats, dim=-1).cpu().numpy())
        if (start // BATCH_SIZE + 1) % 10 == 0:
            print(f"    {start + len(batch)}/{len(df)}...")

    features = np.vstack(all_feats).astype(np.float32)
    print(f"  Terminé en {time.time() - t0:.1f}s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_FILE, features=features,
             image_paths=np.array(list(df["image_path"])))
    print(f"  Cache → {CACHE_FILE}")
    return features


# ════════════════════════════════════════════════════════════════════════════
# 2. WHITENING PCA
# ════════════════════════════════════════════════════════════════════════════

def fit_whitening(features: np.ndarray) -> tuple:
    """Whitening PCA : 512d → 256d. Ajusté sur le train uniquement."""
    print(f"  Whitening PCA : {features.shape[1]}d → {WHITENING_DIM}d...")
    pca = PCA(n_components=WHITENING_DIM, whiten=True)
    pca.fit(features)
    print(f"  Variance expliquée : {pca.explained_variance_ratio_.sum():.1%}")
    return pca.mean_, pca.components_, pca.explained_variance_


# ════════════════════════════════════════════════════════════════════════════
# 3. MLP PAR NIVEAU TAXONOMIQUE
# ════════════════════════════════════════════════════════════════════════════

class FeaturesDataset(Dataset):
    """Dataset sur features pré-calculées."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_level_mlps(features_w: np.ndarray, df: pd.DataFrame) -> dict:
    """
    Entraîne un MLP par niveau taxonomique (règne → famille).
    Retourne {niveau → (LevelMLP, LabelEncoder)}.
    """
    print(f"\n  Entraînement MLP par niveau taxonomique...")
    df_lab   = df[df["species_name"].notna()].reset_index(drop=True)
    mlp_dict = {}

    for level in SUPERVISED_LEVELS:
        if level not in df_lab.columns:
            continue

        df_lvl = df_lab[df_lab[level].notna()].reset_index(drop=True)
        if len(df_lvl) == 0:
            continue

        enc     = LabelEncoder()
        labels  = enc.fit_transform(df_lvl[level])
        n_cls   = len(enc.classes_)
        feats_l = features_w[df_lvl.index]

        counts  = df_lvl[level].value_counts()
        weights = df_lvl[level].map(lambda x: 1.0 / max(counts[x], 1)).values
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.float32),
            num_samples=len(df_lvl), replacement=True
        )

        ds     = FeaturesDataset(feats_l, labels)
        loader = DataLoader(ds, batch_size=64, sampler=sampler)

        mlp = LevelMLP(WHITENING_DIM, n_cls).to(DEVICE)
        opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)

        mlp.train()
        for epoch in range(MLP_EPOCHS):
            total, n = 0.0, 0
            for feats_b, labels_b in loader:
                feats_b, labels_b = feats_b.to(DEVICE), labels_b.to(DEVICE)
                loss = F.cross_entropy(mlp(feats_b), labels_b)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                opt.step(); total += loss.item(); n += 1
            sch.step()
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"    [{level} — {n_cls} cls] "
                      f"epoch {epoch+1}/{MLP_EPOCHS} loss={total/max(n,1):.4f}")

        mlp.eval()
        mlp_dict[level] = (mlp, enc)
        print(f"    {level} ✓ ({n_cls} classes)")

    return mlp_dict


# ════════════════════════════════════════════════════════════════════════════
# 4. PROTO-CLIP — NIVEAU ESPÈCE
# ════════════════════════════════════════════════════════════════════════════

def get_description(species: str, tax_info: dict = None) -> str:
    if species in SPECIES_DESCRIPTIONS:
        return SPECIES_DESCRIPTIONS[species]
    if tax_info:
        parts = [f"a photo of {species}"]
        for lvl in ["famille", "ordre", "classe"]:
            v = tax_info.get(lvl)
            if v and pd.notna(v):
                parts.append(str(v))
        parts.append("marine coastal species")
        return ", ".join(parts)
    return f"a photo of {species}, marine species"


@torch.no_grad()
def encode_text_descriptions(species_list, tax_lookup, bioclip) -> np.ndarray:
    tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL)
    feats     = []
    for sp in species_list:
        desc   = get_description(sp, tax_lookup.get(sp, {}))
        tokens = tokenizer([desc]).to(DEVICE)
        feat   = bioclip.encode_text(tokens)
        feats.append(F.normalize(feat, dim=-1).squeeze(0).cpu().numpy())
    return np.vstack(feats).astype(np.float32)


def compute_prototypes(df, features_w, bioclip, alpha=ALPHA) -> tuple:
    """Calcule les prototypes Proto-CLIP = α × visuel + (1-α) × textuel."""
    print(f"\n  Prototypes Proto-CLIP (α={alpha})...")
    df_lab       = df[df["species_name"].notna()].reset_index(drop=True)
    species_list = sorted(df_lab["species_name"].unique())

    tax_cols   = [l for l in TAXONOMY_LEVELS if l != "species_name" and l in df_lab.columns]
    tax_lookup = {}
    for _, row in df_lab.drop_duplicates("species_name").iterrows():
        tax_lookup[row["species_name"]] = {lvl: row.get(lvl) for lvl in tax_cols}

    # Prototypes visuels : centroïde pondéré
    visual_protos = {}
    for sp in species_list:
        mask  = (df_lab["species_name"] == sp).values
        feats = features_w[mask]
        if len(feats) == 0:
            continue
        if len(feats) == 1:
            visual_protos[sp] = feats[0] / (np.linalg.norm(feats[0]) + 1e-8)
        else:
            centroid = feats.mean(0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            weights  = np.exp(feats @ centroid / 0.1)
            weights /= weights.sum()
            proto    = (weights[:, None] * feats).sum(0)
            visual_protos[sp] = proto / (np.linalg.norm(proto) + 1e-8)

    # Prototypes textuels : projetés dans l'espace whitened
    print("  Encodage descriptions textuelles...")
    text_feats = encode_text_descriptions(species_list, tax_lookup, bioclip)
    vis_mat    = np.vstack([visual_protos[sp] for sp in species_list if sp in visual_protos])
    W          = np.linalg.lstsq(text_feats, vis_mat, rcond=None)[0]
    text_proj  = text_feats @ W
    norms      = np.linalg.norm(text_proj, axis=1, keepdims=True)
    text_proj  = text_proj / np.clip(norms, 1e-8, None)

    prototypes = {}
    for i, sp in enumerate(species_list):
        if sp not in visual_protos:
            continue
        fused = alpha * visual_protos[sp] + (1 - alpha) * text_proj[i]
        prototypes[sp] = fused / (np.linalg.norm(fused) + 1e-8)

    n_manual = sum(1 for sp in species_list if sp in SPECIES_DESCRIPTIONS)
    print(f"  {len(prototypes)} prototypes "
          f"({n_manual} manuelles, {len(species_list)-n_manual} templates)")
    return prototypes, tax_lookup


def learn_temperature(prototypes, features_w, df) -> float:
    """Calibre la température T pour des scores de confiance honnêtes."""
    print("  Apprentissage température...")
    df_lab       = df[df["species_name"].notna()].reset_index(drop=True)
    species_list = list(prototypes.keys())
    proto_mat    = np.vstack([prototypes[sp] for sp in species_list])
    proto_t      = torch.tensor(proto_mat, dtype=torch.float32)

    feats, labels = [], []
    for i, (_, row) in enumerate(df_lab.iterrows()):
        sp = row.get("species_name")
        if sp in prototypes:
            feats.append(features_w[i])
            labels.append(species_list.index(sp))

    if not feats:
        return 10.0

    feats_t  = torch.tensor(np.vstack(feats), dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_T    = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))
    opt      = torch.optim.Adam([log_T], lr=0.05)

    for _ in range(300):
        loss = F.cross_entropy(feats_t @ proto_t.T * log_T.exp(), labels_t)
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            log_T.clamp_(np.log(1.0), np.log(100.0))

    T = float(log_T.exp())
    print(f"  Température : T={T:.2f}")
    return T


# ════════════════════════════════════════════════════════════════════════════
# 5. SAUVEGARDE
# ════════════════════════════════════════════════════════════════════════════

def save_model(prototypes, tax_lookup, whitening, temperature, alpha, mlp_dict):
    """Sauvegarde tous les artefacts dans results/."""
    mean, components, variance = whitening
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(MODEL_FILE,
             prototypes    = np.vstack([prototypes[sp] for sp in prototypes]),
             species_list  = np.array(list(prototypes.keys())),
             wh_mean       = mean,
             wh_components = components,
             wh_variance   = variance,
             temperature   = np.array([temperature]),
             alpha         = np.array([alpha]))
    with open(TAX_LOOKUP_FILE, "wb") as f:
        pickle.dump(tax_lookup, f)

    mlp_save = {}
    for level, (mlp, enc) in mlp_dict.items():
        mlp_save[level] = {
            "state_dict": mlp.state_dict(),
            "encoder":    enc,
            "input_dim":  WHITENING_DIM,
            "n_classes":  len(enc.classes_),
        }
    torch.save(mlp_save, MLP_MODEL_FILE)

    print(f"\n  Sauvegardé :")
    print(f"    Proto-CLIP  → {MODEL_FILE}")
    print(f"    Taxo lookup → {TAX_LOOKUP_FILE}")
    print(f"    MLP niveaux → {MLP_MODEL_FILE} ({list(mlp_dict.keys())})")


# ════════════════════════════════════════════════════════════════════════════
# 6. ÉVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate(features_w, df, prototypes, tax_lookup, temperature,
             mlp_dict, threshold=THRESHOLD, margin_min=MARGIN_MIN) -> dict:
    """Évalue sur df. Retourne dict de résultats par niveau."""
    print(f"\n  Évaluation ({len(df)} images, seuil={threshold:.0%}, margin≥{margin_min:.2f})...")
    df_lab      = df[df["species_name"].notna()].reset_index(drop=True)
    records     = {lvl: [] for lvl in ALL_LEVELS}
    path_counts = {"proto_clip": 0, "mlp": 0}

    model = BioModel(prototypes, tax_lookup,
                         (features_w,), temperature, mlp_dict)
    # On reconstruit un BioModel léger juste pour passer à predict()
    # (whitening déjà appliqué donc on le bypasse)

    for i, (_, row) in enumerate(df_lab.iterrows()):
        # predict() attend un feat_w déjà whitened — c'est le cas ici
        result = predict(features_w[i], BioModel(
            prototypes, tax_lookup,
            (np.zeros(WHITENING_DIM), np.eye(WHITENING_DIM, features_w.shape[1]),
             np.ones(features_w.shape[1])),
            temperature, mlp_dict
        ), threshold, margin_min)
        path_counts[result["path"]] += 1

        for level in ALL_LEVELS:
            true_val = row.get(level)
            if not true_val or (isinstance(true_val, float) and pd.isna(true_val)):
                continue
            preds = result["all_scores"].get(level, [])
            if not preds:
                continue
            records[level].append({
                "true":     str(true_val),
                "pred":     preds[0]["label"],
                "score":    preds[0]["score"],
                "accepted": preds[0]["score"] >= threshold,
                "path":     result["path"],
            })

    n_total = len(df_lab)
    print(f"\n  proto_clip : {path_counts['proto_clip']} ({path_counts['proto_clip']/max(n_total,1):.0%})")
    print(f"  mlp        : {path_counts['mlp']} ({path_counts['mlp']/max(n_total,1):.0%})")
    print(f"\n  {'Niveau':<15} {'Accuracy':>10} {'Coverage':>10} {'F-cov':>8} {'N':>12}")
    print("  " + "─" * 60)

    results = {}
    for level in ALL_LEVELS:
        recs = records.get(level, [])
        if not recs:
            continue
        accepted = [r for r in recs if r["accepted"]]
        n_tot    = len(recs)
        n_acc    = len(accepted)
        coverage = n_acc / n_tot if n_tot else 0
        acc      = accuracy_score(
            [r["true"] for r in accepted],
            [r["pred"] for r in accepted],
        ) if n_acc else 0
        fcov = round(acc * coverage, 3)
        print(f"  {level:<15} {acc:>10.1%} {coverage:>10.1%} {fcov:>8.3f} {n_acc:>5}/{n_tot}")
        results[level] = {"accuracy": acc, "coverage": coverage,
                          "n_accepted": n_acc, "n_total": n_tot, "records": recs}

    results["_paths"] = path_counts
    return results


# ════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE PRINCIPALE
# ════════════════════════════════════════════════════════════════════════════

def run_fit(images_dir: Path, with_eval: bool = False, no_cache: bool = False):
    """Pipeline complète d'entraînement."""
    print("\n" + "=" * 60)
    print("ENTRAÎNEMENT — MLP par niveau + Proto-CLIP espèce")
    print("=" * 60)

    df     = build_dataset(images_dir=images_dir)
    df_lab = df[df["species_name"].notna()].reset_index(drop=True)

    if len(df_lab) == 0:
        print("⚠  Aucune image labellisée.")
        return

    bioclip = load_bioclip()

    if with_eval:
        try:
            train_df, test_df = train_test_split(
                df_lab, test_size=0.2, random_state=42,
                stratify=df_lab["species_name"]
            )
        except ValueError:
            train_df, test_df = train_test_split(df_lab, test_size=0.2, random_state=42)
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        print(f"\n  Split : {len(train_df)} train | {len(test_df)} test")
    else:
        train_df, test_df = df_lab, None

    feats_train = extract_features(train_df, bioclip, use_cache=not no_cache)
    wh          = fit_whitening(feats_train)
    feats_w     = apply_whitening(feats_train, *wh)

    mlp_dict    = train_level_mlps(feats_w, train_df)
    protos, tax = compute_prototypes(train_df, feats_w, bioclip)
    temp        = learn_temperature(protos, feats_w, train_df)

    save_model(protos, tax, wh, temp, ALPHA, mlp_dict)

    if with_eval and test_df is not None:
        print("\n  Extraction features test set...")
        feats_test   = extract_features(test_df, bioclip, use_cache=False)
        feats_test_w = apply_whitening(feats_test, *wh)
        evaluate(feats_test_w, test_df, protos, tax, temp, mlp_dict)

    print("\n  Entraînement terminé ✓")
    return protos, tax, wh, temp, mlp_dict


def run_update(images_dir: Path):
    """Ajoute de nouvelles espèces Proto-CLIP sans réentraîner les MLP."""
    print("\n" + "=" * 60)
    print("MISE À JOUR — nouvelles espèces")
    print("=" * 60)

    from classifier_infer import load_model
    model   = load_model()
    bioclip = load_bioclip()

    df      = build_dataset(images_dir=images_dir)
    df_new  = df[df["species_name"].notna()].reset_index(drop=True)
    new_sp  = set(df_new["species_name"]) - set(model.prototypes.keys())

    if not new_sp:
        print("  Aucune nouvelle espèce.")
        return

    print(f"  {len(new_sp)} nouvelles espèces détectées")
    df_only      = df_new[df_new["species_name"].isin(new_sp)].reset_index(drop=True)
    feats        = extract_features(df_only, bioclip, use_cache=False)
    feats_w      = apply_whitening(feats, *model.whitening)
    new_p, new_t = compute_prototypes(df_only, feats_w, bioclip)

    model.prototypes.update(new_p)
    model.tax_lookup.update(new_t)
    save_model(model.prototypes, model.tax_lookup, model.whitening,
               model.temperature, ALPHA, model.mlp_dict)
    print(f"  {len(model.prototypes)} espèces au total ✓")


# ════════════════════════════════════════════════════════════════════════════
# 8. POINT D'ENTRÉE
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Entraînement — classifier Biolit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tous les paramètres sont dans config.py.

Exemples :
  python classifier_train.py --fit --eval --images data/images/identifiable
  python classifier_train.py --fit         --images data/images/identifiable
  python classifier_train.py --eval        --images data/images/identifiable
  python classifier_train.py --update      --images nouvelles_images/
        """
    )
    parser.add_argument("--fit",      action="store_true", help="Entraîner")
    parser.add_argument("--eval",     action="store_true", help="Évaluer (split 80/20)")
    parser.add_argument("--update",   action="store_true", help="Ajouter nouvelles espèces")
    parser.add_argument("--images",   type=str, required=True)
    parser.add_argument("--no-cache", action="store_true",
                        help="Forcer réextraction features")
    args = parser.parse_args()

    if not BIOCLIP_AVAILABLE:
        print("pip install open-clip-torch")
        return

    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"Dossier introuvable : {images_dir}")
        return

    if args.fit:
        run_fit(images_dir, with_eval=args.eval, no_cache=args.no_cache)
    elif args.eval:
        from classifier_infer import load_model
        model        = load_model()
        bioclip      = load_bioclip()
        df           = build_dataset(images_dir=images_dir)
        df_lab       = df[df["species_name"].notna()].reset_index(drop=True)
        feats        = extract_features(df_lab, bioclip, use_cache=not args.no_cache)
        feats_w      = apply_whitening(feats, *model.whitening)
        evaluate(feats_w, df_lab, model.prototypes, model.tax_lookup,
                 model.temperature, model.mlp_dict)
    elif args.update:
        run_update(images_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()