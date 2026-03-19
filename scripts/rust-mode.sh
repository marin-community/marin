#!/usr/bin/env bash
# Toggle dupekit between source build (dev) and pre-built wheel (user) install
set -euo pipefail
MODE="${1:-status}"
ROOT="$(git rev-parse --show-toplevel)"
PYPROJECT="$ROOT/pyproject.toml"

case "$MODE" in
  dev)
    perl -i -pe 's|^dupekit = .*|dupekit = { path = "rust/dupekit", editable = true }|' "$PYPROJECT"
    touch "$ROOT/.rust-dev-mode"
    echo "Switched to dev mode (source build). Run: uv sync"
    ;;
  user)
    perl -i -pe 's|^dupekit = .*|dupekit = { version = ">=0.1.0" }|' "$PYPROJECT"
    rm -f "$ROOT/.rust-dev-mode"
    echo "Switched to user mode (pre-built wheel). Run: uv sync"
    ;;
  status)
    if [ -f "$ROOT/.rust-dev-mode" ]; then echo "dev"; else echo "user"; fi
    ;;
  *)
    echo "Usage: $0 {dev|user|status}" >&2
    exit 1
    ;;
esac
