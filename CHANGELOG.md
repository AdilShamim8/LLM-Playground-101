# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/),
versioning follows [Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-10-14

### Major: Production Infrastructure Overhaul

This release transforms the project from a tutorial into a production-ready
system with full infrastructure, observability, and deployment tooling.

### Added — Infrastructure
- **System Architecture document** (`docs/ARCHITECTURE.md`) with 100-200M MAU
  scaling design, capacity planning, and 8-phase migration path
- **Kubernetes manifests** (`k8s/`) with Deployment, HPA, Ingress, ConfigMap, Secret
- **Nginx reverse proxy** config with load balancing, rate limiting, TLS, SSE support
- **Production docker-compose** (`docker-compose.prod.yml`) with resource limits, 2 replicas
- **Prometheus + Grafana** monitoring stack with auto-provisioned dashboards
- **PostgreSQL backup script** with S3 upload and 30-day retention

### Added — API & Backend
- **Pydantic Settings** (`config/settings.py`) — typed env vars with fail-fast validation
- **Structured logging** with rotation (100MB), JSON output for production, loguru context
- **Global exception handler** — catches unhandled errors, logs with error ID, clean JSON response
- **Security headers middleware** — X-Frame-Options, X-Content-Type-Options, HSTS
- **Request ID middleware** — correlation IDs for distributed tracing across services
- **Prometheus metrics** at `/metrics` — request counts, latency histograms, generation stats
- **Health endpoints** — `/health` (liveness), `/ready` (readiness), `/health/detailed` (components)
- **Database models** (SQLAlchemy) — User, APIKey, Conversation, Message, GenerationLog
- **Alembic migrations** — version-controlled schema evolution
- **Pagination utilities** — in-memory and database-level (LIMIT/OFFSET)
- **Thread-safe ModelRegistry** with RLock — prevents concurrent load/unload races
- **Input sanitization** — control char removal, path traversal prevention, prompt injection detection
- **Graceful shutdown** handler with cleanup task registry

### Added — DevOps & Tooling
- **Makefile** with 14 common commands (install, test, lint, format, docker-up, etc.)
- **Pre-commit hooks** — ruff lint + format on every commit
- **CONTRIBUTING.md** with setup, workflow, commit conventions
- **CHANGELOG.md** (this file)
- **CODE_OF_CONDUCT.md**
- **.dockerignore** — prevents image bloat and secret leaks

### Added — Testing
- **conftest.py** with shared fixtures (device, small_config, small_model, mock_tokenizer)
- **API integration tests** (9 tests) using FastAPI TestClient
- **Locust load testing** script with 4 user behaviors (chat, streaming, models, health)

### Added — Documentation
- `docs/ARCHITECTURE.md` — full system design for 100-200M MAU
- `docs/SECURITY.md` — vulnerability reporting + production hardening checklist
- `docs/DEVELOPMENT.md` — dev setup, project structure, troubleshooting
- `docs/API_REFERENCE.md` — all endpoints with request/response examples
- `docs/PRODUCTION_DEPLOYMENT.md` — cost analysis + migration phases
- README updated with "Production Infrastructure" section

### Changed
- Version bumped from 1.1.0 → 2.0.0 (major: production infrastructure added)
- API `__version__` updated to 2.0.0

## [1.2.0] — 2026-10-14

### Added
- Production deployment guide (`docs/PRODUCTION_DEPLOYMENT.md`)
- GitHub Actions CI pipeline (lint, type-check, tests)
- `pyproject.toml` with ruff, mypy, pytest configuration
- `.gitignore` and `.dockerignore` for safety
- `scripts/run_pipeline.sh` end-to-end pipeline runner
- Makefile with common dev commands
- Pre-commit hooks for automated code quality
- CONTRIBUTING.md and CODE_OF_CONDUCT.md

### Security
- Removed hardcoded JWT secret (fail-fast if not set in production)
- Fixed DEV_MODE auth bypass (defaults to false)
- Fixed CORS wildcard + credentials vulnerability
- Secured `torch.load` with `weights_only=True` (prevents RCE)
- Dockerfile runs as non-root user
- docker-compose reads passwords from env, not hardcoded

### Changed
- Pinned exact dependency versions in `requirements.txt`
- Migrated to modern PyTorch AMP API (`torch.amp`)
- Migrated `datetime.utcnow()` → `datetime.now(timezone.utc)`
- Fixed event-loop blocking in async chat routes
- Reformatted entire codebase with ruff (88-char line length)

### Fixed
- 310 ruff lint violations → 0
- Undefined class references (DeepSeekConfig, QwenConfig, GPTModel)
- Broken README quick-start (`cd llm-playground` → `cd LLM-Playground-101`)
- numpy 2.x incompatibility with torch 2.1.0

## [1.1.0] — 2026-10-14

### Added
- Comprehensive Tutorial-101.md updated with October 2026 LLM landscape
  - GPT-5, Claude 4.5, Gemini 3, DeepSeek V4, Qwen3, Gemma 4
  - 2026 post-training stack (GRPO, DAPO, RLVR)
  - New benchmarks (MMLU-Pro, GPQA, Humanity's Last Exam, SWE-bench)
  - ChatGPT 1B MAU milestone (June 2026)

## [1.0.0] — 2025

### Added
- Complete LLM pipeline: data collection → cleaning → tokenization →
  pre-training → SFT → RLHF (PPO) → evaluation → deployment
- Model architectures: GPT, DeepSeek (MLA + MoE), Qwen, Gemma
- BPE tokenizer implementation from scratch
- FastAPI backend with OpenAI-compatible API
- Gradio UI for interactive exploration
- Docker deployment support
