# Label Studio

Label Studio est deploye comme app Docker separee sur Clever Cloud.

## Image

`labelstudio/Dockerfile` pinne:

```text
heartexlabs/label-studio:1.22.0
```

Cette version correspond a l'ancien deploiement a restaurer.

## Persistance

Ne pas compter sur un volume Docker ou un FS Bucket avec cette app Docker Clever
Cloud. La persistance doit passer par PostgreSQL:

```text
DJANGO_DB=default
POSTGRE_NAME=<dbname>
POSTGRE_USER=<user>
POSTGRE_PASSWORD=<password>
POSTGRE_HOST=<host>
POSTGRE_PORT=5432
```

Les fichiers issus d'anciens volumes doivent etre migres vers Cellar/S3 ou rester
accessibles par URL publique. Ne pas les remettre dans le filesystem du
conteneur Docker: ils seraient perdus au rebuild/restart. L'ingestion pousse les
images a Label Studio via des URL S3 presignees.

## Projets attendus

Par defaut, le pipeline cherche:

- `Biolit Crops`
- `Biolit No Crops`

Ces noms peuvent etre ajustes avec:

```text
LABEL_STUDIO_CROPS_PROJECT
LABEL_STUDIO_NO_CROPS_PROJECT
```
