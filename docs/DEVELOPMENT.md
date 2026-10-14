# Development Guide

## Quick Start

```bash
# Clone the repo
git clone https://github.com/AdilShamim8/LLM-Playground-101.git
cd LLM-Playground-101

# Set up environment
cp .env.example .env
# Edit .env: set DEV_MODE=true, generate JWT_SECRET_KEY

# Install dependencies (choose one)
make dev-install          # Full install with dev tools
# OR
make install              # Production deps only

# Verify installation
make verify               # Run 40-test verification suite

# Run tests
make test                 # Full test suite
make test-core            # Core tests only (skip data tests needing trafilatura)

# Start services (choose one)
make docker-up            # Docker (recommended)
# OR
uvicorn api.main:app --reload  # API only, for development
```

## Project Structure

```
LLM-Playground-101/
├── api/                    # FastAPI backend
│   ├── main.py             # App factory + middleware stack
│   ├── routes/             # API endpoints (chat, models, health, eval)
│   ├── middleware/          # Auth, CORS, security headers, metrics, request ID
│   └── utils/              # Pagination, helpers
├── config/                 # Settings, logging
├── data/                   # Data pipeline (crawling, cleaning, tokenization)
├── db/                     # SQLAlchemy models
├── docs/                   # Architecture, security, deployment docs
├── evaluation/             # Metrics (BLEU, ROUGE), benchmarks (MMLU)
├── generation/             # Text generation (greedy, beam, top-k, top-p)
├── models/                 # Model architectures (GPT, DeepSeek, Qwen, Gemma)
├── nginx/                  # Reverse proxy config
├── scripts/                # Pipeline runner
├── tests/                  # Test suite
├── training/               # Training pipelines (pretrain, SFT, RLHF)
├── ui/                     # Gradio frontend
├── Makefile                # Common commands
├── docker-compose.yml      # Dev environment
├── docker-compose.prod.yml # Production overrides
└── pyproject.toml          # Tool config (ruff, mypy, pytest)
```

## Common Tasks

### Run the API server
```bash
uvicorn api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Run the UI
```bash
python ui/app.py
# UI: http://localhost:7860
```

### Run a specific test
```bash
pytest tests/test_models.py::test_gpt_forward -v
```

### Run lint and format
```bash
make lint      # Check for issues
make format    # Auto-fix issues
```

### Add a new API route
1. Create `api/routes/your_route.py`
2. Define `router = APIRouter()`
3. Add endpoints
4. Register in `api/main.py`: `app.include_router(router, prefix="/api/v1")`

### Add a new model architecture
1. Create `models/architectures/your_model.py`
2. Inherit from `nn.Module`, implement `forward()`
3. Return `GPTOutput` (loss, logits)
4. Register in `models/__init__.py`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | development / staging / production |
| `DEV_MODE` | false | Bypasses auth (dev only!) |
| `JWT_SECRET_KEY` | (required) | 32+ char random string |
| `DATABASE_URL` | (computed) | PostgreSQL connection string |
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection |
| `CORS_ORIGINS` | localhost | Comma-separated allowed origins |
| `LOG_LEVEL` | INFO | DEBUG / INFO / WARNING / ERROR |

## Debugging Tips

### Model won't load
```bash
python -c "from models import GPTModel; m = GPTModel.from_pretrained('./checkpoints/pretrain/best'); print('OK', m.num_parameters())"
```

### Check GPU memory
```python
import torch
print(torch.cuda.memory_allocated() / 1e9, "GB used")
print(torch.cuda.memory_reserved() / 1e9, "GB reserved")
```

### View logs
```bash
make docker-logs    # Docker logs
tail -f logs/app.log # File logs
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`
Install PyTorch separately (it's not in requirements.txt for CUDA flexibility):
```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### Tests fail with `Numpy is not available`
numpy 2.x is incompatible with torch 2.1.0. Pin numpy < 2.0:
```bash
pip install "numpy<2.0"
```

### `RuntimeError: JWT_SECRET_KEY must be set`
Generate a key and set it in `.env`:
```bash
echo "JWT_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
```
