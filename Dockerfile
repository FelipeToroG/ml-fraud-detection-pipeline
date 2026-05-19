# Dockerfile
# ==========
# Multi-stage build for the fraud detection inference API.
#
# Stage 1 (builder): installs Python dependencies into a virtualenv.
# Stage 2 (runtime): slim base, copies the venv and project source only.
#
# The split keeps the final image small (no compilers, no build deps) and
# pushes most of the cost of `docker build` into a cacheable layer that
# invalidates only when requirements-api.txt changes.

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is required by XGBoost and LightGBM. scikit-learn and shap link
# against it transitively. Installing here, not in the runtime stage, keeps
# the runtime image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements-api.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements-api.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# libgomp1 is also required at runtime (not just build time) because the
# loaded model executes against the same compiled extensions.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user. Containers running as root are a routine finding
# in security review, and there is no reason the API needs root privileges.
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app configs/ ./configs/
COPY --chown=app:app models/ ./models/
COPY --chown=app:app README.md ./README.md

EXPOSE 8000

# Healthcheck so orchestrators can detect a stuck process. Curl is chosen
# over a Python script to keep the runtime layer small.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
