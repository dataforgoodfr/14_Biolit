# Crop Inference — Biolit V2

## Description

Module d'inférence ONNX (YOLOv8 exporté) qui détecte et croppe l'espèce principale (animal/plant) sur des photos
littorales. Pour chaque image, une seule détection est conservée (la plus confiante). Les crops sont
sauvegardés sur disque et un manifeste JSON est produit pour l'étape suivante du pipeline
(classification taxonomique).

## Structure

```
crop_inference/
├── predict.py        # Point d'entrée CLI — orchestre config, modèle, inférence, manifeste
├── detector.py       # Détection ONNX : letterbox, meilleure boîte, remise à l'échelle
├── model_loader.py   # Téléchargement/cache du modèle depuis HuggingFace
├── config.yaml       # Paramètres d'inférence (conf, save_dir)
└── utils/
    └── logger.py     # Logging console coloré + fichier rotatif par run
```

## Quickstart

À lancer depuis la racine du dépôt :

```bash
uv run python -m ml.crop_inference.predict \
  --config ml/crop_inference/config.yaml \
  --source ml/crop_inference/images/
```

## Configuration

```yaml
# config.yaml
model:
  source: "huggingface"                          # huggingface (dataforgood) | local
  repo_id: "DataForGood/biolit-species-identification"
  filename: "detection/model.onnx"

inference:
  conf: 0.4             # Seuil de confiance minimum pour garder une détection
  save_dir: "outputs/"  # Répertoire racine des sorties
```

Une seule détection est gardée par image (la plus confiante) : pas besoin de NMS. La taille
d'entrée (640) et les classes sont lues dans le modèle ONNX. L'inférence tourne sur CPU avec
`onnxruntime`, sans PyTorch ni Ultralytics.

## Outputs

```
outputs/run_YYYYMMDD_HHMMSS/
├── crops/
│   ├── img_001_animal_0.95.jpg
│   └── img_002_plant_0.87.jpg
├── manifest.json
└── run_YYYYMMDD_HHMMSS.log
```

**Exemple `manifest.json` :**

```json
[
  {
    "source_image": "/data/images/img_001.jpg",
    "crop_path": "outputs/run_20240410_143012/crops/img_001_animal_0.95.jpg",
    "class_id": 1,
    "class_name": "animal",
    "confidence": 0.9502,
    "bbox_xyxy": [124.3, 87.1, 512.8, 498.6],
    "orig_shape": [640, 640]
  }
]
```

## Modèle

- **Source :** [DataForGood/biolit-species-identification](https://huggingface.co/DataForGood/biolit-species-identification) (`detection/model.onnx`) — téléchargé automatiquement via `hf_hub_download` et mis en cache localement
- **Architecture :** YOLOv8s finetuné sur images littorales Biolit, exporté en ONNX (opset 18)
- **Classes :** `{0: plant, 1: animal}`

## Pipeline

```
images/
  └──> crop_inference (ce module)
            │  conf=0.4, max_det=1
            ▼
       manifest.json + crops/
            │
            ▼
       ml/classification
```
