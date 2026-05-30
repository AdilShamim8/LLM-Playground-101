FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install torch==2.1.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 8000 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]