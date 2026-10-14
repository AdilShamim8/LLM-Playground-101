# ═══════════════════════════════════════════════════════════════════════
#  Makefile — Common development commands
# ═══════════════════════════════════════════════════════════════════════
#
#  Usage:
#    make help          — show all available commands
#    make install       — install dependencies
#    make test          — run test suite
#    make lint          — run linter
#    make format        — format code
#    make docker-up     — start all services
#    make docker-down   — stop all services
#

.PHONY: help install dev-install test lint format type-check docker-up docker-down clean pipeline verify

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install --upgrade pip
	pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
	pip install -r requirements.txt
	pip install -e .

dev-install: ## Install development dependencies (includes test tools)
	$(MAKE) install
	pip install -e ".[dev]"
	pre-commit install

test: ## Run the test suite
	pytest tests/ -v --tb=short

test-core: ## Run only core tests (skip data tests needing trafilatura)
	pytest tests/test_models.py tests/test_generation.py tests/test_training.py -v --tb=short

lint: ## Run linter
	ruff check .

format: ## Format code
	ruff format .
	ruff check . --fix

type-check: ## Run type checker
	mypy .

docker-up: ## Start all services with Docker
	docker-compose up --build -d

docker-down: ## Stop all Docker services
	docker-compose down

docker-logs: ## Tail Docker logs
	docker-compose logs -f

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/

verify: ## Run runtime verification
	python run_verification.py

pipeline: ## Run the full training pipeline
	./scripts/run_pipeline.sh

pipeline-verify: ## Run only verification step
	./scripts/run_pipeline.sh --verify
