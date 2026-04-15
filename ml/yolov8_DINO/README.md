# ML - YOLOv8 / GroundingDINO

Approche en deux temps : un premier entraînement sur ~10 000 images annotées automatiquement (quantitatif)
pour bootstrapper le modèle, suivi d'un re-entraînement sur ~1 400 images cropées et annotées manuellement (qualitatif)
pour affiner les performances. L'annotation manuelle ne porte que sur la deuxième partie.

## Partie 1 — Bootstrap autodistill

Annotation automatique via GroundingDINO + fine-tuning YOLOv8. Pas d'annotation manuelle.
L'ontologie est à affiner dans `configs/autodistill_boostrap.yaml`.

```bash
python build_dataset.py      # téléchargement + nettoyage images
python check_dataset.py      # vérification qualité (résolutions, espèces, corrompues)
python autodistill_label.py  # pseudo-labels GroundingDINO
python autodistill_train.py  # fine-tuning YOLOv8
```

`--limit N` sur `build_dataset.py` pour tester sur un sous-ensemble.

**Structure de sortie :**

```text
dataset_biolit/
├── images/
└── labeled-images/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

Poids entraînés → `runs/biolit_v2_bootstrap/weights/`

## Partie 2 — Fine-tuning

Cette étape consiste à améliorer le modèle YOLOv8 obtenu lors du bootstrap en utilisant un dataset plus petit (~1,400 images) mais de meilleure qualité, annoté manuellement.
Le dataset est augmenté artificiellement ×3 sur le train set uniquement (pas sur le validation set) via flips, rotations (±15° et 90°), ajustements de luminosité (±10 %) et léger flou.

### Données

````text
14_Biolit/
├── data/
│   └── manual_annotations/
│       ├── train/...
│       ├── valid/...
│       └── data.yaml

### Modèle de départ

Le fine-tuning part du modèle entraîné lors de la partie 1 :

```text
runs/biolit_v2_bootstrap/weights/best.pt
````

### Lancer le fine-tuning

```bash
python finetune.py
```

### Configuration

Le fichier `configs/finetune.yaml` contient les paramètres principaux :

- chemin vers le modèle bootstrap (`best.pt`)
- chemin vers le dataset manuel (`data.yaml`)
- hyperparamètres d'entraînement (`epochs`, `batch`, `learning rate`)

### Résultat

Les nouveaux poids sont sauvegardés dans :

```text
runs/biolit_v2_finetuned/weights/
```

Ce modèle est ensuite utilisé pour générer les crops finaux ou pour l’inférence.
