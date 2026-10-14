# Contributing to LLM Playground 101

Thank you for your interest in contributing! This document describes how to
set up your environment and submit changes.

## Development Setup

```bash
# 1. Fork and clone the repo
git clone https://github.com/YOUR_USERNAME/LLM-Playground-101.git
cd LLM-Playground-101

# 2. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install with dev dependencies
make dev-install

# 4. Install pre-commit hooks
pre-commit install

# 5. Copy and configure environment
cp .env.example .env
# Edit .env: set JWT_SECRET_KEY, DEV_MODE=true for local dev
```

## Workflow

1. **Create a branch**: `git checkout -b feature/your-feature-name`
2. **Make changes**: Write code + tests
3. **Run checks locally**:
   ```bash
   make lint      # ruff check
   make format    # ruff format
   make test      # pytest
   ```
4. **Commit**: Pre-commit hooks will run automatically
5. **Push and open a PR**: Describe what changed and why

## Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

Types:
  feat     — new feature
  fix      — bug fix
  docs     — documentation only
  style    — formatting, no code change
  refactor — code change that neither fixes a bug nor adds a feature
  test     — adding or correcting tests
  chore    — build process, tooling, dependencies
  perf     — performance improvement
  security — security fix
```

**Examples:**
```
feat(api): add streaming chat endpoint
fix(auth): correct JWT expiry calculation
docs: update README quick-start instructions
security: remove hardcoded API key from auth module
```

## Code Style

- **Line length**: 88 characters (enforced by ruff)
- **Imports**: Sorted automatically by ruff (isort compatible)
- **Types**: Use `X | None` not `Optional[X]` (Python 3.10+ syntax)
- **Docstrings**: Required on all public functions and classes

## Testing

- All new features must include tests
- Tests live in `tests/` mirroring the source structure
- Run `make test` before pushing
- Tests that need a GPU should be marked: `@pytest.mark.gpu`

## Pull Request Checklist

- [ ] Code passes `make lint` and `make format`
- [ ] All tests pass (`make test`)
- [ ] New features have tests
- [ ] Commit messages follow Conventional Commits
- [ ] No secrets or credentials in the diff
- [ ] Documentation updated if needed

## Reporting Issues

When reporting a bug, please include:
1. Python version and OS
2. Steps to reproduce
3. Expected vs actual behavior
4. Error logs (if any)
5. Whether you're using CPU or GPU

## License

By contributing, you agree that your contributions are licensed under the
MIT License.
