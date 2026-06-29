FROM python:3.12-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends binutils \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.25

WORKDIR /app/backend
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra ml --no-dev --no-install-project --no-cache \
    && rm -rf \
        .venv/lib/python3.12/site-packages/torch/include \
        .venv/lib/python3.12/site-packages/torch/test \
    && find .venv/lib/python3.12/site-packages/torch/bin -type f \
        ! -name torch_shm_manager -delete \
    && find .venv/lib/python3.12/site-packages -type d \
        \( -name tests -o -name test \) -prune -exec rm -rf '{}' + \
    && find .venv/lib/python3.12/site-packages -type f -name '*.so*' \
        -exec strip --strip-unneeded '{}' +

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PATH="/app/backend/.venv/bin:$PATH"

WORKDIR /app/backend
COPY --from=builder /app/backend/.venv ./.venv

COPY biolit ./biolit
COPY pipelines ./pipelines
COPY ml ./ml

CMD ["python", "-m", "pipelines.worker"]
