# Template DataForGood

This file will become your README and also the index of your
documentation.

# Contributing


## Installation

- [Installation de Python](#installation-de-python)

Ce projet utilise [uv](https://docs.astral.sh/uv/) pour la gestion des dépendances Python. Il est préréquis pour l'installation de ce projet.

Une fois installé, il suffit de lancer la commande suivante pour installer la version de Python adéquate, créer un environnement virtuel et installer les dépendances du projet.

```bash
uv sync
```

### Git Submodules

Ce projet utilise des submodules Git. Après avoir cloné le repo, initialisez-les :

```bash
git submodule update --init --recursive
```

Ou clonez directement avec les submodules :

```bash
git clone --recurse-submodules <repo-url>
```

A l'usage, si vous utilisez VSCode, l'environnement virtuel sera automatiquement activé lorsque vous ouvrirez le projet. Sinon, il suffit de l'activer manuellement avec la commande suivante :

```bash
source .venv/bin/activate
```

Ou alors, utilisez la commande `uv run ...` (au lieu de `python ...`) pour lancer un script Python. Par exemple:

```bash
uv run pipelines/run.py run build_database
```

## Préparer les données pour Label Studio

Télécharge les images, crée le JSON et démarre le serveur HTTP :

```bash
# Traiter toutes les images
uv run cmd/prepare_labelstudio.py

# Limiter à un nombre spécifique
uv run cmd/prepare_labelstudio.py --limit 10

# Forcer le re-téléchargement des images existantes
uv run cmd/prepare_labelstudio.py --force
```

Cela va :
1. Télécharger les images dans le répertoire `images/`
2. Créer `labelstudio_tasks.json` avec les annotations
3. Démarrer un serveur HTTP avec CORS activé sur `http://localhost:8000`

## Lancer Label Studio

Dans un autre terminal :

```bash
uv run label-studio
```

Ensuite, importez `labelstudio_tasks.json` dans l'interface de Label Studio.

## ML Backend YOLO (pré-annotations automatiques)

Le ML backend utilise YOLOv8 pour générer des pré-annotations automatiques.

### Configuration

1. Récupérez votre token API Label Studio :
   - Ouvrez Label Studio → Account & Settings → Access Token
   - Copiez le token

2. Créez un fichier `.env` à la racine du projet :
   ```bash
   cp .env.example .env
   # Editez .env avec votre token
   ```

3. Créez un lien symbolique pour que docker-compose puisse lire le `.env` :
   ```bash
   ln -s $(pwd)/.env label-studio-ml-backend/label_studio_ml/examples/yolo/.env
   ```

4. Lancez le ML backend :
   ```bash
   cd label-studio-ml-backend/label_studio_ml/examples/yolo
   docker compose up
   ```

   Pour relancer sans rebuild (plus rapide) :
   ```bash
   docker compose up --no-build
   ```

   Pour arrêter le backend :
   ```bash
   docker compose down
   ```

5. Connectez le backend dans Label Studio :
   - Project Settings → Model → Add Model
   - URL: `http://localhost:9090`

## Lancer les precommit-hook localement

[Installer les precommit](https://pre-commit.com/)

    pre-commit run --all-files

## Utiliser Tox pour tester votre code

    tox -vv
