#!/usr/bin/env just --justfile

set dotenv-load

@_:
    just --list

# List available recipes
help:
    just --list

# Sync the environment and install pre-commit hooks
[group('lifecycle')]
install:
    uv sync
    uv run pre-commit install

# Run all code quality and Python compatibility checks
[group('qa')]
check:
    uv sync --locked
    uv run pre-commit run --all-files
    just typing

# Run Ruff lint checks
[group('qa')]
lint *args:
    uv run ruff check {{ args }}

# Check types for Python 3.14 and 3.15
[group('qa')]
typing *args:
    uv run ty check {{ args }}
    uv run ty check --python-version 3.15 {{ args }}

# Run tests with coverage
[group('qa')]
test *args:
    uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml {{ args }}

# Build source and wheel distributions
[group('release')]
build: clean-build
    uvx --from build pyproject-build --installer uv

# Remove distribution artifacts
[group('release')]
clean-build:
    uv run python -c "import shutil; from pathlib import Path; shutil.rmtree('dist') if Path('dist').exists() else None"

# Build and upload a release to PyPI
[group('release')]
publish: build
    uvx twine upload --repository-url https://upload.pypi.org/legacy/ dist/*

# Build and upload a release to PyPI
[group('release')]
build-and-publish: publish

# Build documentation with strict checks
[group('docs')]
docs-test *args:
    uv run mkdocs build --strict {{ args }}

# Serve documentation locally
[group('docs')]
docs *args:
    uv run mkdocs serve {{ args }}

# Serve documentation locally
[group('docs')]
docs-serve *args:
    just docs {{ args }}
