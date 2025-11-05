#!/bin/bash
# Wrapper script for running pyrefly in pre-commit
# Tries multiple methods to find and run pyrefly

set -e

# Try uv tool run first (preferred for local dev)
if command -v uv &> /dev/null; then
    exec uv tool run pyrefly "$@"
fi

# Try system pyrefly
if command -v pyrefly &> /dev/null; then
    exec pyrefly "$@"
fi

# Try python -m pyrefly (if installed in current Python environment)
if python -m pyrefly --version &> /dev/null; then
    exec python -m pyrefly "$@"
fi

# If nothing works, give helpful error
echo "Error: pyrefly not found. Install with one of:"
echo "  - uv tool install pyrefly"
echo "  - pip install pyrefly"
echo "  - uv pip install pyrefly (in dev dependencies)"
exit 1
