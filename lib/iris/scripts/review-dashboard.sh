#!/usr/bin/env bash
# Capture dashboard screenshots via e2e tests then ask Claude Code to review them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="${1:-$(mktemp -d)}"

echo "=== Capturing dashboard screenshots to $OUTPUT_DIR ==="
cd "$REPO_ROOT"
IRIS_SCREENSHOT_DIR="$OUTPUT_DIR" uv run pytest lib/iris/tests/e2e/test_dashboard.py -x -o "addopts="

echo ""
echo "=== Reviewing screenshots ==="

# Build a prompt listing all screenshot paths
file_list=""
for png in "$OUTPUT_DIR"/*.png; do
  file_list="$file_list $png"
done

verdict=$(claude --dangerously-skip-permissions --model sonnet -p \
  "Read all of the following dashboard screenshots and review each one for obvious visual bugs. Be generous — only flag true brokenness like blank pages, giant error messages, completely missing content, or rendering failures. Minor cosmetic issues are fine.

Screenshots to review:
$file_list

For each screenshot, output exactly one line in this format:
  <filename> OK
or
  <filename> NOT_OK <brief reason why>

After all lines, output a final line:
  OVERALL OK
or
  OVERALL NOT_OK" 2>/dev/null)

echo "$verdict"

# Check overall result
if echo "$verdict" | grep -q "OVERALL OK"; then
  exit 0
else
  exit 1
fi
