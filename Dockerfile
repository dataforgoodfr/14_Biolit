FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libxcb1 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync

COPY . .