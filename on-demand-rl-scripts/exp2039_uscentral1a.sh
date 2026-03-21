#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_run_exp2039.sh" "us-central1-a" "v5p-8" "v5p_central1a" "marin-us-central1"
