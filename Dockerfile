# ═══════════════════════════════════════════════════════════════════════
#  Dockerfile — LLM Playground API
# ═══════════════════════════════════════════════════════════════════════
#
#  Design decisions (senior-engineer perspective):
#
#  1. Base image: We use the CUDA devel image (not runtime) because
#     flash-attn requires nvcc to compile from source. If you don't need
#     flash-attn, switch to nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
#     to save ~2 GB on image size.
#
#  2. Non-root user: The container runs as user 'app' (UID 1000). Running
#     as root is a security risk — if the app has an RCE vulnerability,
#     the attacker gets root inside the container. Kubernetes Pod Security
#     Standards also require non-root containers.
#
#  3. Health check: Docker uses the /health endpoint to know if the API
#     is responsive. Without it, Docker can't detect hung processes.
#
#  4. Layer caching: requirements.txt is copied and installed BEFORE the
#     source code. This means dependency installation is cached — changing
#     a .py file doesn't trigger a full pip reinstall.
#
#  5. .dockerignore: Should be used to exclude .git, __pycache__, etc.
#     (Added in a separate commit.)
#

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# ── Environment variables ────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3.11 as the default python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

# ── Create non-root user ────────────────────────────────────────────────
# UID 1000 is the standard first non-root user on most Linux systems.
# This matches the host user UID in most dev environments, which avoids
# file permission issues when mounting volumes.
RUN groupadd -g 1000 app && \
    useradd -m -u 1000 -g app -s /bin/bash app

WORKDIR /app

# ── Install Python dependencies (cached layer) ─────────────────────────
# Copy only requirements first so this layer is cached.
COPY --chown=app:app requirements.txt .

# Install PyTorch with CUDA support, then other deps
# We run pip install as root (needs system write access), then switch to app
RUN pip install --upgrade pip && \
    pip install torch==2.1.0 torchvision \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────
COPY --chown=app:app . .
RUN pip install -e .

# ── Switch to non-root user ─────────────────────────────────────────────
USER app

# ── Expose ports ────────────────────────────────────────────────────────
# 8000 = FastAPI, 7860 = Gradio UI (if run in same container)
EXPOSE 8000

# ── Health check ────────────────────────────────────────────────────────
# Docker will call /health every 30s. If it fails 3 times in a row,
# the container is marked unhealthy and orchestrators (k8s, swarm)
# will restart it.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start command ───────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
