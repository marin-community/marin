#!/usr/bin/env bash
set -o errexit
set -o nounset

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Install Levanter in editable mode.
# the system flags are only OK because we are using a dedicated container.
uv pip install --system --break-system-packages --editable ".[test]"
