# Inférence ML

Ce dossier contient uniquement deux services de production :

- `crop_inference/` : YOLOv8, une détection maximale par image ;
- `classification/` : BioCLIP2 + Proto-CLIP + MLP hiérarchiques.

Les poids sont téléchargés depuis Hugging Face et conservés dans `HF_HOME`.
L'entraînement, les benchmarks, les datasets et les anciens modèles ont été
retirés du dépôt.
