#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_run_exp2039.sh" "us-east1-d" "v6e-8" "v6e_east1d" "marin-us-east1"
