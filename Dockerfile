# RAG config search: pipeline (CLI) + Gradio GUI in one image.
#
#   docker compose build
#   docker compose up gui                                  # http://localhost:7860
#   docker compose run --rm pipeline run -p bm25_512 --max-questions 50
#
# Data, model cache and results are mounted as volumes
# (see docker-compose.yml), nothing from them is baked into the image.

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache


FROM python:3.12-slim-bookworm

ARG UID=1000
ARG GID=1000
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${GID} app \
    && useradd -m -u ${UID} -g ${GID} app

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY config.py ./
COPY configs ./configs
COPY evalkit ./evalkit
# Mount points; /app itself must be writable (deepeval keeps .deepeval/ in cwd)
RUN mkdir -p data runs dataset data_bpmsoft \
    && chown app:app /app /app/data /app/runs /app/dataset /app/data_bpmsoft

USER app
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False

EXPOSE 7860
ENTRYPOINT ["python", "-m", "evalkit.cli"]
CMD ["gui"]
