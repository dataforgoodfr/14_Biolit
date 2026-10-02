# Migration Metabase

Ce dossier sert uniquement de zone locale pour preparer les artefacts de
migration Metabase. Son contenu est ignore par Git.

## A placer ici

```text
metabase/migration/
└── pg-dump-metabase-*.dmp
```

Le dump contient l'interface Metabase: utilisateurs, collections, questions,
dashboards, connexions aux sources, permissions et parametrage applicatif.

## Restauration cible

1. Creer un PostgreSQL dedie a Metabase sur Clever Cloud.
2. Restaurer le dump avant le premier demarrage de l'app Metabase.
3. Demarrer l'app avec `metabase/metabase:v0.60.1.3`.

```bash
pg_restore --clean --if-exists --no-owner --no-acl \
  --dbname "$METABASE_POSTGRES_URL" \
  metabase/migration/pg-dump-metabase-*.dmp
```

Ensuite configurer l'app:

```text
CC_DOCKERFILE=metabase/Dockerfile
MB_DB_TYPE=postgres
MB_DB_HOST=<host>
MB_DB_PORT=5432
MB_DB_DBNAME=<dbname>
MB_DB_USER=<user>
MB_DB_PASS=<password>
```

Le volume Docker d'origine ne doit pas etre monte dans l'app Clever Cloud.
L'etat Metabase doit etre restaure dans PostgreSQL.
