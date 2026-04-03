# BioModel — Classification hiérarchique d'espèces marines côtières

Le modèle prédit le **niveau taxonomique le plus fin possible** — espèce, famille, ordre, classe, phylum ou règne — en s'arrêtant dès qu'il est suffisamment confiant

---

## Architecture

```
Image
  ↓
BioCLIP2 (frozen)              backbone pré-entraîné sur 200M images biologiques
  ↓
Whitening PCA (512d → 256d)    rend les distances cosine plus fiables
  ↓
MLP supervisé par niveau       un MLP par niveau taxonomique
  ├── MLP règne   (3 classes)  
  ├── MLP phylum  (~6 classes)
  ├── MLP classe  (~9 classes) 
  ├── MLP ordre  (~12 classes) 
  └── MLP famille(~50 classes)  
  +
Proto-CLIP espèce (232 classes) few-shot : prototype = α × visuel + (1-α) × texte
  ↓
margin score         rejette si score_top1 - score_top2 < MARGIN_MIN
  ↓
Décision hiérarchique cohérente
  ├── Espèce confiante → espèce + lookup taxref pour niveaux supérieurs
  └── Sinon → MLP du niveau le plus fin confiant
```

---

## Fichiers du projet

```
classifier_train.py         Entraînement — extraction features, MLP, Proto-CLIP
classifier_infer.py         Inférence — charge le modèle et prédit
build_classify_dataset.py   Charge les images et construit le DataFrame labellisé
config.py                   Tous les paramètres — source de vérité unique
README.md
.gitignore
```

---

## Données d'entrée

### Structure attendue

```
data/
├── export_biolit.csv          métadonnées terrain
├── taxonomy.parquet          référentiel taxonomique
└── images/
    └── identifiable/
        ├── 1234_fucus-spiralis_42.jpg
        └── ...
```

### Format des noms de fichiers images

Les images doivent respecter le format établi :

```
{id_n1}_{nom_commun}_{index}.{ext}
```

| Segment | Description | Exemple |
|---------|-------------|---------|
| `id_n1` | Identifiant de l'observation (correspond à `ID - N1` dans observations.csv) | `1234` |
| `nom_commun` | Nom commun normalisé (espaces → `_`, `/` → `-`) | `fucus-spiralis` |
| `index` | Numéro de l'image pour cette observation | `42` |
| `ext` | Extension : jpg, jpeg, png ou webp | `jpg` |




## Requirements

```bash
pip install open-clip-torch torch torchvision scikit-learn pandas pillow pyarrow
```

---

## Usage

### 1. Entraîner le modèle

```bash
# Entraîner + évaluer sur split 80/20 
python classifier_train.py --fit --eval --images data/images/identifiable

# Entraîner sur tout le dataset 
python classifier_train.py --fit --images data/images/identifiable
```

### 2. Prédire sur de nouvelles images

```bash
python classifier_infer.py --images mon_dossier/
```

Résultats dans `results/predictions.csv`.

### 4. Ajouter de nouvelles espèces sans réentraîner

```bash
python classifier_train.py --update --images nouvelles_images/
```

Seuls les prototypes Proto-CLIP des nouvelles espèces sont recalculés.  
Les MLP par niveau et le whitening restent inchangés.

---

## Utiliser le modèle depuis Python

```python
from classifier_infer import load_model, predict_image

# Charger le modèle (une seule fois)
model = load_model()

# Prédire sur une image
result = predict_image("ma_photo.jpg", model)

# Scores à tous les niveaux
for level, preds in result["all_scores"].items():
    print(f"{level}: {preds[0]['label']} ({preds[0]['score']:.0%})")
```

## Fichiers sauvegardés après l'entraînement

```
results/
├── proto_model.npz          prototypes Proto-CLIP + whitening PCA    
├── tax_lookup.pkl           hiérarchie taxonomique espèce → niveaux  
├── mlp_model.pt             poids MLP par niveau (règne → famille)   
└── bioclip_features.npz     cache features BioCLIP2                  (~1 GB, non versionné)
```
---

## Paramètres — config.py

Tous les paramètres sont centralisés dans `config.py`. *
`CONFIDENCE_THRESHOLD` Seuil d'acceptation. Baisser → plus de prédictions, plus d'erreurs. Monter → moins de prédictions, plus fiables.
 `MARGIN_MIN` Seuil faux positifs. Si top1=91% et top2=89%, margin=0.02 → rejeté. Baisser pour plus de couverture. Plage : 0.05–0.20 |
 `PROTO_ALPHA` Poids visuel dans Proto-CLIP. `0.7` si images de qualité. `0.4` si espèces très rares. |
 `MLP_EPOCHS`  Epochs par MLP. Monter si underfitting, baisser si overfitting. 
`MLP_DROPOUT` Régularisation MLP. Monter si overfitting


## Dépendances

```
open-clip-torch    BioCLIP2 backbone
torch              PyTorch
torchvision        transforms
scikit-learn       PCA, LabelEncoder, metrics
pandas             manipulation données
pillow             lecture images
pyarrow            lecture taxonomy.parquet
```