# syntax=docker/dockerfile:1.7
# One image for api and workers. Models are baked in, so containers run offline.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/app/.venv PATH=/app/.venv/bin:$PATH \
    HF_HOME=/models/hf
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /bin/uv
WORKDIR /app

# --- dependencies (cached layer) ---
FROM base AS deps
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ml --no-install-project
ARG SPACY_DE=https://github.com/explosion/spacy-models/releases/download/de_core_news_lg-3.8.0/de_core_news_lg-3.8.0-py3-none-any.whl
ARG SPACY_EN=https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "de_core_news_lg @ ${SPACY_DE}" "en_core_web_lg @ ${SPACY_EN}"

# --- Hugging Face models (cached layer, only re-run when the config's model names change) ---
FROM deps AS models
COPY app/__init__.py app/config.py ./app/
COPY config/config.yaml ./config/
COPY scripts/download_models.py ./scripts/
RUN python scripts/download_models.py && rm -rf /models/hf/xet

# --- runtime ---
FROM deps AS runtime
COPY --from=models /models/hf /models/hf
COPY app ./app
COPY config ./config
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --extra ml --inexact \
    && useradd --system --uid 10001 --no-create-home pii \
    && python -c "import app.api.main"  # fail the build on import errors
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    PII_ROOT=/app TOKENIZERS_PARALLELISM=false
USER 10001
EXPOSE 8000
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
