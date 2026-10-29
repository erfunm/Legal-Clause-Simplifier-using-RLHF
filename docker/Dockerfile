# syntax=docker/dockerfile:1
# CPU image for `lcs serve` (and the Gradio demo): multi-stage, slim, non-root.
#   docker build -f docker/Dockerfile -t legal-simplifier:cpu .      (or: make docker-build)
# The model is not baked in: mount a merged checkpoint at /app/checkpoints (see docker-compose.yml).

FROM python:3.12-slim AS builder
RUN pip install --no-cache-dir uv==0.13.1
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached unless the lockfile changes); CPU torch comes from the lockfile.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra infer --extra serve --extra otel --extra review
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable --extra infer --extra serve --extra otel --extra review

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
COPY space ./space
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["lcs", "serve", "--config", "configs/serve.yaml", "--host", "0.0.0.0", "--port", "8000"]
