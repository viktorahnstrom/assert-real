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

RUN pip install --upgrade pip && \
    # Install CPU-only torch first to avoid the 2.4GB CUDA wheel
    pip install --prefix=/install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu && \
    # Install everything else (torch already satisfied, pip skips it)
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY pyproject.toml .

# Download the MediaPipe Face Landmarker model (float16, ~6 MB)
RUN mkdir -p /app/models && \
    wget -q -O /app/models/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

ENV HF_HOME=/app/.cache/huggingface
# Railway injects PORT; default to 8000 for local docker run
ENV PORT=8000

EXPOSE ${PORT}

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]