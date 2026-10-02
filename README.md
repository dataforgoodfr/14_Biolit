# Biolit

Depot organise autour des 3 apps Clever Cloud de production.

## Structure

```text
backend/       # pipeline ingestion + ML, code Python, tests, Dockerfile
labelstudio/   # app Label Studio, Dockerfile et notes de migration
metabase/      # app Metabase, Dockerfile et notes de migration
docs/          # passation Clever Cloud et revue technique
```

Cette separation garde la lecture simple et colle au deploiement cible:

- `backend`: Task Docker hebdomadaire, puis arretee;
- `labelstudio`: app web continue;
- `metabase`: app web continue.

## Pourquoi 3 apps

Metabase et Label Studio doivent rester accessibles en continu. Le backend
embarque les dependances ML et n'a besoin de tourner qu'une fois par semaine.
Le separer evite de payer une grosse image ML allumee en permanence.

## Backend

```bash
cd backend
uv sync --group dev
cp .env.example .env
uv run python -m pipelines.run
```

Sur Clever Cloud:

```text
CC_DOCKERFILE=backend/Dockerfile
CC_RUN_COMMAND=uv run python -m pipelines.run
```

Le schema SQL d'initialisation est dans `backend/init.sql`.

## Label Studio

```text
CC_DOCKERFILE=labelstudio/Dockerfile
```

Voir `labelstudio/README.md`. Les dumps et fichiers de migration se preparent
dans `labelstudio/migration/` mais ne sont pas versionnes.

## Metabase

```text
CC_DOCKERFILE=metabase/Dockerfile
```

Voir `metabase/README.md`. Le dump de migration se prepare dans
`metabase/migration/` mais n'est pas versionne.

## Deploiement

Clever Cloud ne lance pas `docker compose`: chaque dossier correspond a une app
Docker separee. Les donnees persistantes passent par PostgreSQL et Cellar/S3,
pas par le filesystem du conteneur.

Voir `docs/clevercloud-apps.md` pour les variables, commandes de creation et
restaurations Metabase/Label Studio.

## Tests backend

```bash
cd backend
uv run pytest
```
