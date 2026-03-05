#!/usr/bin/env bash
# Blow away worktree, recreate, run migration, uv sync, run tests.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WT_DIR="$REPO_ROOT/.claude/worktrees/flatten-test"
SCRIPT_SRC="$REPO_ROOT/scripts/migrations/flatten_monorepo.py"

echo "=== Cleaning up old worktree ==="
git -C "$REPO_ROOT" worktree remove "$WT_DIR" --force 2>/dev/null || true
rm -rf "$WT_DIR"

echo "=== Creating fresh worktree ==="
git -C "$REPO_ROOT" worktree add "$WT_DIR" HEAD

echo "=== Copying migration script ==="
cp "$SCRIPT_SRC" "$WT_DIR/scripts/migrations/flatten_monorepo.py"

echo "=== Running migration ==="
cd "$WT_DIR"
"$REPO_ROOT/.venv/bin/python" scripts/migrations/flatten_monorepo.py --execute 2>&1

echo "=== Running uv sync ==="
uv sync --extra train --extra cpu --extra dedup 2>&1

echo "=== Running tests ==="
uv run pytest tests/ -n4 -x --timeout=120 \
    -m "not tpu_ci and not slow and not tpu and not docker and not e2e and not ray" \
    --ignore=tests/levanter \
    --ignore=tests/haliax \
    --deselect=tests/rl/test_weight_transfer.py::test_multiple_weight_updates \
    -q 2>&1
