# ML - YOLOv8 (détection + crop)

Objectif : détecter l'espèce (végétal/animal) et générer un crop centré sur l'objet.

## Entrées
Structure de données généré à partir du fichier `export_biolit.csv`.

### Récupération propre des données
build_dataset.py
Pipeline de constitution du dataset Biolit pour inférence YOLO / Grounding DINO.

**Structure de sortie :**

```text
    dataset_biolit/
    ├── images/
    │   ├── identifiable/
    │   └── non_identifiable/ # à valider
    ├── labels/
    │   ├── identifiable/
    │   └── non_identifiable/ # à valider
    ├── metadata.csv # GroundingDINO
    └── data.yaml #YOLO
```


## Sorties

- Bboxes + classes : `dataset_biolit/exports/yolov8_detections.csv`
- Images crops : `dataset_biolit/crops/images/`

## Routage

- si détection forte → **Classification**
- si détection faible → **Label Studio (CROP)**
- si pas de détection animal ou végétal → stop
