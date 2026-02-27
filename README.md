# Biolit — workflow de classification d'espèces (images)

Ce dépôt fournit un **template infra + data** pour lancer le pipeline de
classification et guider les bénévoles sur les 3 tâches ML :
1. **YOLOv8/DINO** (détection + crop)
2. **Classification hiérarchique** (règne → espèce)

Les pipelines existants d'export sont conservés et intégrés.

## Architecture (résumé)

1. **Entrée** : API quotidienne (à venir) ou CSV (`data/raw/observations.csv`).
2. **Détection DINO/YOLOv8** (ML) → bboxes + crops.
3. **Classification hiérarchique** (ML) → taxonomie.
4. **Label Studio** : boucle d'annotation/correction si besoin.
5. **Dataviz** : CSV compatible Metabase (puis dashboard).
6. **Exports** : CSV d'annotations (base de données plus tard).

## Structure du repo

```
biolit/                # Lib Python (taxref, observations, dataviz)
pipelines/             # Orchestration
ml/                    # Dossiers des 2 tâches ML
dataviz/               # Docs dataviz
infra/                 # Docker Compose (Label Studio)
data/                  # Workspace local (non versionné)
```

### Dossiers data (proposés)

- `data/raw/` : CSV brut + images du jour (dump API)
- `data/crops/` : crops issus de YOLOv8
- `data/label-studio/files/` : images à annoter
- `data/exports/` : sorties CSV (annotations, qualité, etc.)
- `data/dataviz/` : CSV pour Metabase

## Installation

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances.

```bash
uv sync
```

Si besoin :

```bash
source .venv/bin/activate
```

## Flux quotidien (API → ML → Label Studio)

1. **Récupération quotidienne** depuis l'API (à venir) ou CSV local.

2. **DINO/YOLOv8** : détection + crop.
   - si détection forte → **Classification**
   - si détection faible → **Label Studio (CROP)**
   - si pas de détection animal ou végétal → stop
3. **Classification** : prédiction + probabilité.
   - certitude faible → **Label Studio (pré-annotations + probas)**
   - certitude forte → export direct
4. **Export CSV** : `data/exports/annotations.csv`
5. **Dataviz** : `data/dataviz/observations.csv` (Metabase)

## Enrichissement des données

Cette pipeline a pour but de télécharger l'ensemble des images biolit et de les enrichir pour créer une base dédiées au machine learning ou à la data visualisation.

Pour lancer la pipeline :
- Télécharger sur le site TaxRef le fichier `TAXREF_v18_2025` et le décompresser dans le dossier `data/`.
- Télécharger le fichier excel d'export test et le placer dans `data/export_biolit.csv`.
- Lancer la pipeline suivante
```bash
uv run pipelines/export_inpn.py
```

La pipeline propose différents logs pour alerter sur des problèmes de qualité.

La pipeline crée plusieurs fichier:

- `data/biolit_valid_observations.parquet` : fichier final avec l'ensemble des images annotées et enrichies.
- `data/observations_missing_taxref.csv` : fichier des images dont l'expèce n'est pas présente dans TaxRef.
- `data/biolit_observation_missing_nom.csv` : observations validées mais sans nom d'espèce.
- `data/biolit_observation_validated_non_identifiable.csv`
- `data/distribution_images.html` : Visualisation de la répartition des images en fonction des espèces.

## Label Studio (annotation)

```bash
docker compose -f infra/docker-compose.yml up
```

UI : http://localhost:8080

Les images à annoter sont montées depuis `data/label-studio/files`.

## Déploiement local

Il est possible de lancer l'ensemble en local pour les premiers tests.
L'objectif est d'étudier les sorties de chaque modèle avant d'automatiser
le workflow complet.


## Contribution

### Pre-commit

```bash
pre-commit run --all-files
```

### Tests

```bash
tox -vv
```
