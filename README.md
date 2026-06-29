# BioLit — production

Le dépôt est organisé par composant et démarre depuis un unique
[`docker-compose.yml`](docker-compose.yml) à la racine.

```text
.
├── docker-compose.yml       # orchestration complète
├── .env.example             # variables à renseigner
├── backend/                 # API → PostgreSQL → crop → classification
├── label-studio/            # interfaces de crop et de validation
└── metabase/                # raccordement du volume Metabase existant
```

Le flux de production est le suivant :

```text
API BioLit → PostgreSQL → YOLOv8
                            ├─ crop trouvé ───────────────┐
                            └─ sinon → crop Label Studio ─┤
                                                         ▼
                                                     BioCLIP2
                                                         ▼
                                             validation Label Studio
                                                         ▼
                    PostgreSQL ← taxon validé + id_image → suppression du crop
```

MinIO ne conserve que les originaux et crops nécessaires à une validation en
cours. PostgreSQL reste la source de vérité durable, et Metabase lit la vue
`metabase_observations`.

## Démarrage

```bash
cp .env.example .env

# Premier démarrage : services de données et interfaces
docker-compose --env-file .env up -d \
  postgres minio minio-init label-studio metabase
```

Ouvrir Label Studio sur <http://localhost:8089>, récupérer le jeton API du
compte, puis renseigner `LABEL_STUDIO_API_KEY` dans `.env`.

```bash
docker-compose --env-file .env up -d --build backend
```

Interfaces locales : Label Studio `:8089`, Metabase `:3000`, MinIO `:9001` et
PostgreSQL `:5432`.

## Documentation par composant

- [backend](backend/README.md) : états, tables, stockage temporaire et erreurs ;
- [Label Studio](label-studio/README.md) : projets et configurations ;
- [Metabase](metabase/README.md) : réutilisation d'un volume existant.

## Vérification du backend

```bash
cd backend
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Les datasets, notebooks, benchmarks, poids et scripts d'entraînement ont été
retirés. Les modèles sont téléchargés au premier lancement dans le volume
`model_cache`; ils ne sont pas intégrés à l'image Docker.
