# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/),
versioning follows [Semantic Versioning](https://semver.org/).

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
