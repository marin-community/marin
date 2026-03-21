#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_run_exp2039.sh" "us-east5-a" "v5p-8" "v5p_east5a" "marin-us-east5"
