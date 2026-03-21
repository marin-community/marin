#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/_run_exp2039.sh" "europe-west4-a" "v6e-8" "v6e_euw4a" "marin-eu-west4"
