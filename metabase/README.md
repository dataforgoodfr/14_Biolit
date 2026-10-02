# Metabase

Metabase est deploye comme app Docker separee sur Clever Cloud.

## Image

`metabase/Dockerfile` pinne:

```text
metabase/metabase:v0.60.1.3
```

Cette version correspond au dump Coolify a restaurer. Eviter `latest` pendant la
migration.

## Persistance

Metabase doit utiliser une base PostgreSQL applicative dediee:

```text
MB_DB_TYPE=postgres
MB_DB_HOST=<host>
MB_DB_PORT=5432
MB_DB_DBNAME=<dbname>
MB_DB_USER=<user>
MB_DB_PASS=<password>
MB_JETTY_PORT=8080
```

Restaurer le dump dans ce PostgreSQL avant le premier demarrage de l'app.
