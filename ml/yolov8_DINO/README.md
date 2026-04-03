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

## Partie 2 — Fine-tuning (à venir)

Pris en charge par un autre membre de l'équipe.
Entraînement sur des images cropées et annotées manuellement pour améliorer les performances du modèle bootstrap.
