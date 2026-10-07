#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/../evaluation/campaigns/eval-campaign-09-25-unlabeled/common.sh"
model=$1
shift
submit=1
if [ "${1:-}" = --dry-run ]; then submit=0; shift; fi
[ "$#" -eq 0 ] || die "usage: launch-nupa200.sh MODEL [--dry-run]"
resolve_hf_token
resolve_coreweave_credentials
arguments=(
  --model-config "$MODEL_CONFIG_DIR/$model.yaml"
  --evalchemy-config "$MARIN_DIR/experiments/weight_merging/nupa200.yaml"
  --evalchemy-config "$EVALCHEMY_CONFIG_DIR/math500.yaml"
  --evalchemy-config "$EVALCHEMY_CONFIG_DIR/gpqa-diamond.yaml"
  --evalchemy-config "$EVALCHEMY_CONFIG_DIR/ifbench.yaml"
  --federated_cluster "$FEDERATED_CLUSTER"
  --priority "$PRIORITY"
)
run_campaign_launch "$submit" "${arguments[@]}"
