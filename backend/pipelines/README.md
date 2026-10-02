# Pipelines

`pipelines/run.py` est le flux de production deploye sur Clever Cloud.

## Flux

1. Charger les observations depuis `BIOLIT_API_URL`.
2. Normaliser les donnees et les inserer dans PostgreSQL.
3. Enrichir les observations avec la geolocalisation.
4. Filtrer les observations deja traitees.
5. Lancer le crop YOLO et stocker les crops dans S3/Cellar.
6. Envoyer les observations sans crop vers Label Studio.
7. Classifier les crops, enrichir avec le fichier DORIS S3 et envoyer les
   pre-annotations vers Label Studio.
8. Recuperer les annotations Label Studio et alimenter les tables finales.

## Lancement

```bash
uv run python -m pipelines.run
```

Les variables attendues sont documentees dans `.env.example` et
`../docs/clevercloud-apps.md`.

`pipelines/export_inpn.py` reste un flux utilitaire separe pour les exports INPN;
il n'est pas appele par le deploiement Clever Cloud.
