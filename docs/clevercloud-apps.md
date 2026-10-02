# Deploiement Clever Cloud

Objectif: deployer Biolit sur Clever Cloud sans `docker compose`, avec une app
Clever Cloud par service.

## Architecture cible

| Service | Type Clever Cloud | Dockerfile | Persistance |
|---|---|---|---|
| Ingestion Biolit | Docker Task | `backend/Dockerfile` | PostgreSQL + Cellar |
| Label Studio | Docker web app | `labelstudio/Dockerfile` | PostgreSQL dedie ou partage |
| Metabase | Docker web app | `metabase/Dockerfile` | PostgreSQL dedie Metabase |

Points importants:

- Les apps Docker Clever Cloud ne supportent pas Docker Compose.
- Les apps Docker Clever Cloud ne supportent pas les FS Buckets montes.
- Les donnees a recuperer depuis les anciens volumes doivent donc etre restaurees
  vers des services persistants: PostgreSQL pour Metabase/Label Studio, Cellar/S3
  pour les fichiers.
- Les apps web doivent ecouter sur le port `8080`. Metabase est force via
  `MB_JETTY_PORT=8080`; Label Studio ecoute deja sur `8080`.

## Pourquoi 3 apps

Une app Docker unique serait plus simple a lancer au debut, mais elle oblige a
garder le backend ML allume en continu pour servir aussi Metabase et Label
Studio. C'est plus cher et plus fragile.

La cible recommandee est donc:

- **Metabase** et **Label Studio** en apps web continues, avec petites instances
  ajustees a leur charge;
- **Ingestion/ML** en Task relancee une fois par semaine, puis arretee.

Cette separation permet aussi de mettre a jour ou redemarrer un service sans
couper les deux autres.

## Versions pinnees pour la migration

| App | Image |
|---|---|
| Label Studio | `heartexlabs/label-studio:1.22.0` |
| Metabase | `metabase/metabase:v0.60.1.3` |

Ces tags correspondent aux versions identifiees dans les anciens deploiements a
restaurer. Ne pas les remplacer par `latest` pendant la migration.

## Variables d'environnement

Voir aussi `backend/.env.example`.

### Ingestion

| Variable | Description |
|---|---|
| `POSTGRES_URL` ou `POSTGRESQL_ADDON_URI` | base Biolit |
| `BIOLIT_API_URL` | URL API Biolit avec token |
| `BIOLIT_S3_BUCKET` | bucket images/crops/cache, defaut `biolit-uploads` |
| `DORIS_S3_KEY` | chemin du CSV DORIS, defaut `doris_data.csv` |
| `CELLAR_ADDON_HOST` / `CELLAR_ADDON_KEY_ID` / `CELLAR_ADDON_KEY_SECRET` | injectees par Cellar |
| `S3_ENDPOINT_URL` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` | fallback Scaleway/local |
| `LABEL_STUDIO_URL` / `LABEL_STUDIO_API_KEY_DATAFORGOOD` | connexion Label Studio |
| `LABEL_STUDIO_CROPS_PROJECT` / `LABEL_STUDIO_NO_CROPS_PROJECT` | noms des projets LS |
| `CC_DOCKERFILE` | `backend/Dockerfile` |
| `CC_RUN_COMMAND` | `uv run python -m pipelines.run` |

L'image d'ingestion installe uniquement les dependances runtime avec:

```bash
uv sync --no-dev
```

### Label Studio

| Variable | Description |
|---|---|
| `CC_DOCKERFILE` | `labelstudio/Dockerfile` |
| `LABEL_STUDIO_HOST` | URL publique Clever Cloud |
| `DJANGO_DB` | `default` |
| `POSTGRE_NAME` / `POSTGRE_USER` / `POSTGRE_PASSWORD` | base PostgreSQL Label Studio |
| `POSTGRE_HOST` / `POSTGRE_PORT` | host et port PostgreSQL |

### Metabase

| Variable | Description |
|---|---|
| `CC_DOCKERFILE` | `metabase/Dockerfile` |
| `MB_DB_TYPE` | `postgres` |
| `MB_DB_HOST` / `MB_DB_PORT` | host et port PostgreSQL Metabase |
| `MB_DB_DBNAME` / `MB_DB_USER` / `MB_DB_PASS` | base applicative Metabase |
| `MB_JETTY_PORT` | `8080` |

## Creation des apps

Adapter `--org` et les plans avant execution.

```bash
# Ingestion Task
clever create --type docker biolit-ingestion --region par
clever env set CC_DOCKERFILE "backend/Dockerfile" --alias biolit-ingestion
clever env set CC_RUN_COMMAND "uv run python -m pipelines.run" --alias biolit-ingestion

# Label Studio
clever create --type docker biolit-label-studio --region par
clever env set CC_DOCKERFILE "labelstudio/Dockerfile" --alias biolit-label-studio

# Metabase
clever create --type docker biolit-metabase --region par
clever env set CC_DOCKERFILE "metabase/Dockerfile" --alias biolit-metabase
```

## Add-ons

Prevoir au minimum:

- PostgreSQL Biolit: observations, tables ML, eventuellement Label Studio.
- PostgreSQL Metabase dedie: base applicative Metabase.
- Cellar: images, crops, caches geoloc, `doris_data.csv`, dumps de migration.

Le schema Biolit doit etre initialise une fois:

```bash
psql "$POSTGRESQL_ADDON_URI" -f backend/init.sql
```

## Migration des anciennes persistances

Les dumps et fichiers de migration peuvent etre poses localement dans:

```text
metabase/migration/
labelstudio/migration/
```

Ces dossiers sont volontairement ignores par Git. Ils servent de zone de travail
avant restauration vers PostgreSQL et Cellar/S3.

### Metabase

1. Creer et lier un PostgreSQL dedie a l'app Metabase.
2. Restaurer le dump avant de demarrer l'app Metabase.
3. Garder l'image `metabase/metabase:v0.60.1.3` pendant la premiere reprise.

```bash
pg_restore --clean --if-exists --no-owner --no-acl \
  --dbname "$METABASE_POSTGRES_URL" \
  metabase/migration/pg-dump-metabase-1785271271.dmp
```

Ensuite seulement, configurer l'app:

```bash
clever env set MB_DB_TYPE postgres --alias biolit-metabase
clever env set MB_DB_HOST "<host>" --alias biolit-metabase
clever env set MB_DB_PORT "5432" --alias biolit-metabase
clever env set MB_DB_DBNAME "<dbname>" --alias biolit-metabase
clever env set MB_DB_USER "<user>" --alias biolit-metabase
clever env set MB_DB_PASS "<password>" --alias biolit-metabase
```

### Label Studio

1. Faire un `pg_dump` de l'ancienne base Label Studio.
2. Restaurer ce dump dans le PostgreSQL cible Label Studio.
3. Demarrer l'app avec `heartexlabs/label-studio:1.22.0`.
4. Verifier les deux projets attendus par l'ingestion:
   `Biolit Crops` et `Biolit No Crops`, ou ajuster
   `LABEL_STUDIO_CROPS_PROJECT` / `LABEL_STUDIO_NO_CROPS_PROJECT`.

```bash
pg_restore --clean --if-exists --no-owner --no-acl \
  --dbname "$LABEL_STUDIO_POSTGRES_URL" \
  labelstudio/migration/pg-dump-labelstudio.dmp
```

Si l'ancien deploiement contenait aussi des fichiers locaux, ne pas essayer de
les remonter en FS Bucket dans l'app Docker. Les migrer vers Cellar/S3 ou garder
des URL publiques externes, puis configurer les projets Label Studio en
consequence.

## Deploiement

Le deploiement Clever Cloud pousse le commit courant vers le Git Clever Cloud de
l'app cible.

```bash
clever deploy --alias biolit-ingestion
clever deploy --alias biolit-label-studio
clever deploy --alias biolit-metabase
```

Pour declencher l'ingestion:

```bash
clever restart --alias biolit-ingestion
```

## Cron

Les Docker Tasks n'ont pas de cron natif. Le workflow
`.github/workflows/clevercloud-ingestion.yml` declenche `clever restart` chaque
lundi a 02:00 UTC, avec declenchement manuel possible via `workflow_dispatch`.

Secrets GitHub attendus:

- `CLEVER_TOKEN`
- `CLEVER_SECRET`
- `CLEVER_INGESTION_APP_ID`
