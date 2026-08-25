# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Single resolution pass with the CPU-only index so torch+torchvision pins
# resolve there and every transitive dep sees them as already satisfied.
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libegl1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    XDG_CACHE_HOME=/app/.cache \
    OMP_NUM_THREADS=2 \
    PORT=8000

# MediaPipe Face Landmarker (float16, ~6 MB)
RUN mkdir -p /app/models && \
    wget -q -O /app/models/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

COPY app/ ./app/
COPY pyproject.toml .
COPY scripts/prefetch_models.py ./scripts/prefetch_models.py

# Bake every model weight into the image so the container never needs
# network access to serve a request. Fail the build if any model is missing.
RUN PYTHONPATH=/app python scripts/prefetch_models.py

# Run unprivileged. Cache dirs must be writable by the app user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/.cache /data \
    && chown -R appuser:appuser /app /data
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]