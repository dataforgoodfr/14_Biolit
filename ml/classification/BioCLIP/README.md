# 🌊 Identification automatique des espèces avec système hybride BioCLIP + enrichissement taxonomique GBIF

**84.1% accuracy · Top 100 espèces · 88% de couverture**

---

## Approche

Le système repose sur [BioCLIP](https://huggingface.co/imageomics/bioclip), un modèle CLIP entraîné sur 10 millions d'images d'organismes vivants (TreeOfLife-10M). BioCLIP est utilisé comme **feature extractor gelé** (vecteurs 512d) — seule la tête de classification est adaptée aux données BioLit.

Les données BioLit ont été enrichies via l'API GBIF pour récupérer la hiérarchie taxonomique complète (règne, embranchement, classe, ordre, famille, genre) de chaque espèce, utilisée notamment pour construire les descriptions textuelles des prototypes Proto-CLIP (`export_biolit_enriched.csv`).

L'architecture finale est un **système hybride à deux composantes** :

- Un **classifier MLP** entraîné sur les 50 espèces les plus représentées (>50 images), opérant sur les features BioCLIP 512d originales
- Un **réseau prototypique** pour les 50 espèces rares (10–50 images), avec fusion visuel/textuel Proto-CLIP, opérant sur les features **whitened 256d**

Le routing entre les deux est déterminé par des seuils de confiance calibrés sur le jeu de validation.

Cette approche résout le déséquilibre fort des données BioLit : le classifier gère les espèces communes avec haute précision (89.7%), tandis que les prototypes permettent une classification few-shot des espèces rares sans surapprentissage (51.1%).

**Références scientifiques :**
- Prototypes pondérés par similarité : [arXiv:2110.11553](https://arxiv.org/abs/2110.11553)
- Température apprise par descente de gradient : [arXiv:2108.00340](https://arxiv.org/abs/2108.00340)
- Proto-CLIP (fusion visuel + textuel) : [arXiv:2307.03073](https://arxiv.org/abs/2307.03073)
- Whitening + PCA sur features BioCLIP : Wu et al. ECCV2020 

---

## Résultats

|                | Espèces communes (top 50) | Espèces rares (51–100) | Global     |
|----------------|--------------------------|------------------------|------------|
| Accuracy       | 89.7%                    | 51.1%                  | **84.1%**  |
| Couverture     | —                        | —                      | 88.0%      |

Le taux de rejet (12%) correspond aux images sous le seuil de confiance minimal — le modèle préfère ne pas répondre plutôt que de prédire avec une faible certitude.

**Test terrain (40 images .webp)** : 72.5% classifiées, 27.5% rejetées — l'écart avec le benchmark est attendu vu la variabilité des conditions de prise de vue terrain.

---

## Modèles pré-entraînés

| Fichier | Description |
|---|---|
| `best_model_top50.pth` | Classifier MLP — 50 espèces communes (features 512d) |
| `prototypes_v4.pt` | Prototypes Proto-CLIP + transformation whitening — 100 espèces |

---

## Installation

```bash
pip install open-clip-torch torch torchvision pandas tqdm pillow
```

---

## Inférence sur nouvelles images

```bash
python scripts/infer_local_v4.py \
  --images     /chemin/vers/dossier/ \
  --prototypes models/prototypes_v4.pt \
  --classifier models/best_model_top50.pth \
  --output     resultats_inference_biolit.csv \
  --ext        webp        # ou jpg, png
```

**Colonnes du CSV de sortie :**

| Colonne | Description |
|---|---|
| `espece_pred` | Espèce prédite (`?` si rejetée) |
| `confiance` | Score de confiance [0–1] |
| `methode` | `classifier_top50` / `classifier_top50_low` / `prototypical_rare` / `rejected` |
| `top1_common` … `top3_proto` | Top 3 alternatives pour chaque composante |

---

## Structure des fichiers

```
models/
├── best_model_top50.pth          # Classifier MLP 
└── prototypes_v4.pt              # Prototypes + whitening transform
└── README.md

results/
├── predictions_sample_test.csv   # Prédictions sur 40 images terrain
└── export_biolit_enriched.csv    # Dataset BioLit enrichi GBIF
└── README.md

scripts/
└── infer_local_v4.py         # Script d'inférence hybride v4 (CPU/GPU)

README.md
```

---

## Espèces couvertes

Les 100 espèces correspondent aux espèces les plus observées dans la base BioLit au moment de l'entraînement (février 2026). La liste complète est disponible dans [`scripts/inference/infer_local_v4.py`](scripts/inference/infer_local_v4.py) (dictionnaire `SPECIES_DESCRIPTIONS`).
