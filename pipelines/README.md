# Pipelines

Ce dossier regroupe les scripts d'orchestration et documente le **flux quotidien**.
Les pipelines utilisent le dossier `data/` (non versionné) comme espace de travail.

## Flux quotidien (API → ML → Label Studio)

1. **Ingestion & préparation des données** depuis l'API.
   1. **Objectifs**
      - Récupérer les données depuis l’API Biolit
      - Standardiser et nettoyer les données
      - Les stocker dans une base PostgreSQL
      - Mettre les données à disposition des autres systèmes (ML, dataviz)
   2. **Étapes du pipeline**
      1. Ingestion
      - appel à l’API Biolit
      - récupération des observations

      2. Transformation
      - normalisation des noms de colonnes
      - typage des champs
      - nettoyage des données (dates, coordonnées, identifiants)

      3. Chargement
      - insertion dans PostgreSQL
      - gestion des doublons via un mécanisme d’UPSERT (ON CONFLICT DO NOTHING)
   3. **Variables d'environnement**
      - POSTGRES_URL=postgresql://user:password@host:port/dbname
      - BIOLIT_API_URL=https://biolit.fr/wp-json/biolit/v1/observations?token=XXX
   4. **Lancer la pipeline**
      - uv run python -m pipelines.run
2. **Qualité** : si l'image est mauvaise → stop.
3. **YOLOv8** : détection + crop.
   - si aucune détection → **Label Studio (CROP)**
4. **Classification** : prédiction + probabilité.
   - certitude faible → **Label Studio (pré-annotations + probas)**
   - certitude forte → export direct
5. **Export CSV** : `data/exports/annotations.csv`
6. **Dataviz** : `data/dataviz/observations.csv` (Metabase)
