#!/bin/bash
#
# Setup helper for Marin on Linux/macOS.
# Checks for prerequisites (uv, Python) and validates platform compatibility.
#

set -e

echo "🌊 Marin Setup Helper"
echo "====================="

# 1. Check for Python
echo -n "[1/3] Checking Python... "
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1)
    echo "OK ($PY_VERSION)"
else
    echo "MISSING"
    echo "Error: Python 3 is not found in PATH."
    exit 1
fi

# 2. Check for uv
echo -n "[2/3] Checking uv... "
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version 2>&1)
    echo "OK ($UV_VERSION)"
else
    echo "MISSING"
    echo "Error: 'uv' is not installed. Install with: pip install uv"
    exit 1
fi

# 3. Check Platform Compatibility
echo -n "[3/3] Checking Platform Compatibility... "
OS="$(uname -s)"
case "$OS" in
    Linux|Darwin)
        echo "OK ($OS)"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        echo "WARNING"
        echo ""
        echo "[!] CRITICAL COMPATIBILITY NOTICE:"
        echo "    Marin currently depends on 'vortex-data', which does not have pre-built binaries for Windows."
        echo "    You may encounter installation errors."
        echo "    RECOMMENDATION: Use WSL2 (Windows Subsystem for Linux) or Docker for a smooth experience."
        echo "    See: https://learn.microsoft.com/en-us/windows/wsl/install"
        ;;
    *)
        echo "UNKNOWN ($OS)"
        echo "Warning: Untested platform. Proceed with caution."
        ;;
esac

echo ""
echo "Setup check complete."
