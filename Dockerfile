FROM python:3.12-slim

WORKDIR /app

# Installer uv
RUN pip install uv

# Copier les fichiers
COPY pyproject.toml uv.lock ./

# Installer dépendances
RUN uv sync

COPY . .

CMD ["uv", "run", "python", "-m", "pipelines.run"]