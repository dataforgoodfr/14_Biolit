# Crop Inference — Biolit V2

## Description

Module d'inférence YOLOv8 qui détecte et croppe l'espèce principale (animal/plant) sur des photos
littorales. Pour chaque image, une seule détection est conservée (la plus confiante). Les crops sont
sauvegardés sur disque et un manifeste JSON est produit pour l'étape suivante du pipeline
(classification).

## Structure

```
crop_inference/
├── predict.py        # Point d'entrée CLI — orchestre config, modèle, inférence, manifeste
├── model_loader.py   # Téléchargement/cache du modèle depuis HuggingFace
├── config.yaml       # Paramètres d'inférence (conf, iou, device…)
└── utils/
    └── logger.py     # Logging console coloré + fichier rotatif par run
```

## Quickstart


```bash
# Image unique
uv run predict.py --source photo.jpg

# Dossier entier -> à télécharger au préalable 
uv run predict.py --source ./images/ # TODO: cf. dataeng pour cadrer ceci


```

## Configuration

```yaml
# config.yaml
model:
  source: "huggingface"                          # huggingface | local
  repo_id: "mandresyandri/yolov8_biolit_crop"
  filename: "runs/biolit_v2_yolo_finetuned/best.pt"

inference:
  conf: 0.4       # Seuil de confiance minimum pour garder une détection
  iou: 0.45       # Seuil IoU pour la suppression non-maximale (NMS)
  imgsz: 640      # Taille d'entrée du modèle (pixels)
  device: "cpu"   # cpu | cuda | mps
  save: true      # Sauvegarde des crops (toujours true en pratique)
  save_dir: "outputs/"  # Répertoire racine des sorties
```

`max_det` est fixé à `1` dans le code — une seule détection par image, la plus confiante.

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

- **Source :** [mandresyandri/yolov8_biolit_crop](https://huggingface.co/mandresyandri/yolov8_biolit_crop) — téléchargé automatiquement via `hf_hub_download` et mis en cache localement
- **Architecture :** YOLOv8 finetuné sur images littorales Biolit
- **Classes :** `{0: plant, 1: animal}`

**Note — hardfix `ultralytics.nn.modules` :** `predict.py` réexpose manuellement les sous-modules
`conv`, `block`, `head`, `transformer` pour contourner un mismatch entre la version du modèle
sérialisé et la version installée d'ultralytics. Si ultralytics est mis à jour et que le chargement
échoue, vérifier en priorité ce bloc.

## Pipeline

```
images/
  └──> crop_inference (ce module)
            │  conf=0.4, max_det=1
            ▼
       manifest.json + crops/
            │
            ▼
       classification (à venir)
```
