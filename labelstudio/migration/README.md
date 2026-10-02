# Migration Label Studio

Ce dossier sert uniquement de zone locale pour preparer les artefacts de
migration Label Studio. Son contenu est ignore par Git.

## A placer ici

```text
labelstudio/migration/
├── pg-dump-labelstudio-*.dmp
└── files/
    └── ... fichiers issus de l'ancien volume si necessaire
```

Le dump PostgreSQL contient l'interface Label Studio: utilisateurs, projets,
taches, annotations, predictions, templates et parametrage applicatif.

Les fichiers de l'ancien volume ne doivent pas etre remis dans le filesystem du
conteneur Clever Cloud. S'ils sont encore necessaires, les uploader dans
Cellar/S3 puis garder leurs URLs ou chemins S3 dans les taches Label Studio.

## Restauration cible

1. Creer ou choisir le PostgreSQL cible pour Label Studio.
2. Restaurer le dump avant le premier demarrage de l'app Label Studio.
3. Demarrer l'app avec `heartexlabs/label-studio:1.22.0`.

```bash
pg_restore --clean --if-exists --no-owner --no-acl \
  --dbname "$LABEL_STUDIO_POSTGRES_URL" \
  labelstudio/migration/pg-dump-labelstudio-*.dmp
```

Ensuite configurer l'app:

```text
CC_DOCKERFILE=labelstudio/Dockerfile
DJANGO_DB=default
POSTGRE_NAME=<dbname>
POSTGRE_USER=<user>
POSTGRE_PASSWORD=<password>
POSTGRE_HOST=<host>
POSTGRE_PORT=5432
LABEL_STUDIO_HOST=<url publique Clever Cloud>
```

Verifier apres restauration que les projets attendus existent:

```text
Biolit Crops
Biolit No Crops
```
