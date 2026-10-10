#! /usr/bin/env bash

set -euo pipefail

# Install uv
curl -LsSf https://astral.sh/uv/0.13.0/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install Dependencies
uv sync

# Install pre-commit hooks
uv run pre-commit install --install-hooks
