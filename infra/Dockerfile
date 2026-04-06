FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app

RUN uv sync

CMD ["uv", "run", "pipelines/run.py"]