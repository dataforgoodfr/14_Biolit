# Revue de code - Biolit

> Analyse apres preparation de la branche Clever Cloud.
> Objectif: lisibilite, productionisation et cout d'exploitation reduit.

## Architecture retenue

- `backend/pipelines/run.py`: pipeline production Clever Cloud.
- `labelstudio/`: image Label Studio pinnee pour migration.
- `metabase/`: image Metabase pinnee pour migration.
- `backend/biolit/settings.py`: point central pour les variables d'environnement.

Le backend ML tourne comme Task hebdomadaire. Label Studio et Metabase restent
en apps web continues.

## Nettoyage deja applique

- Suppression des samples et artefacts versionnes:
  `sample_data/`, anciens resultats BioCLIP/BioClipv2, datasets DINO/prompt YOLO.
- Suppression des restes de template non utilises:
  `d4g-utils/`, workflow associe, `tox.ini`.
- Suppression de l'ancien `docker-compose` et de `.clever.json` versionne.
- Suppression du code mort `biolit/minio.py` et
  `biolit/label_studio_postprocessing.py`.
- Suppression de la dataviz HTML locale (`biolit/visualisation`), remplacee en
  production par Metabase.
- Runtime resserre: plus de groupe R&D deploye, Docker installe `uv sync --no-dev`.
- Regroupement final en 3 dossiers d'apps: `backend/`, `labelstudio/`,
  `metabase/`.

## Robustesse deja amelioree

- S3/Cellar centralise dans `backend/biolit/s3.py`.
- SSL S3 actif par defaut, desactivable seulement via `S3_VERIFY_SSL=false`.
- Bucket S3, cle DORIS et projets Label Studio configurables.
- `.clever.json`, `data/`, `outputs/`, `runs/`, `sample_data/` ignores.
- Workflow GitHub d'ingestion passe en planification hebdomadaire.

## Reste a surveiller

- Les inserts PostgreSQL ligne par ligne dans `backend/biolit/create_table.py` peuvent devenir
  lents; a remplacer plus tard par `executemany`, `COPY` ou chargement batch.
- `observations.py` et `lien_doris.py` gardent encore quelques sorties console a
  convertir en logs structures.
- Les plages de dates Label Studio dans `backend/pipelines/run.py` sont encore en dur.
- Faire un run complet avec les vrais credentials apres restauration des dumps.

## Validation conseillee avant PR

1. `cd backend && uv lock --check`
2. `cd backend && uv sync --dry-run --locked --no-dev`
3. `cd backend && python -m py_compile` sur `biolit`, `pipelines` et `ml`
4. Run Clever Cloud sur donnees de validation apres migration PostgreSQL/S3
