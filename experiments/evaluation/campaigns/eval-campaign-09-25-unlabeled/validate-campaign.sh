#!/usr/bin/env bash
set -euo pipefail

. "$(dirname "$0")/common.sh"

validate_marin_checkout
validate_campaign_configs

echo "validated repository pins and campaign configs"
