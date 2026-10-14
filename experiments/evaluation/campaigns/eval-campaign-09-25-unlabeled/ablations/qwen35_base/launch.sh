#!/usr/bin/env bash
set -euo pipefail

ABLATION_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$ABLATION_DIR/../../common.sh"

usage() {
  cat <<'EOF'
Usage: ./launch.sh --phase cache|remaining [--submit] [--wait]

The cache phase runs PIQA alone. Marin mirrors the pinned model revision and writes
its completion marker before starting vLLM. Run the remaining phase only after the
cache phase has passed serving startup.
EOF
}

phase=
submit=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --phase)
      [ "$#" -ge 2 ] || die "--phase requires a value"
      phase=$2
      shift 2
      ;;
    --submit)
      submit=1
      shift
      ;;
    --wait)
      WAIT_FOR_RESULTS=1
      export WAIT_FOR_RESULTS
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "$phase" in
  cache|remaining) ;;
  *) die "--phase must be cache or remaining" ;;
esac

model=Qwen-Qwen3.5-35B-A3B-Base
model_config="$ABLATION_DIR/model.yaml"
tau3_model_config="$ABLATION_DIR/model-tau3.yaml"

validate_marin_checkout
validate_campaign_configs
prepare_judge_environment

(cd "$MARIN_DIR" && \
  STANDARD_MODEL_CONFIG="$model_config" \
  TAU3_MODEL_CONFIG="$tau3_model_config" \
  uv run python - <<'PY'
import os
from pathlib import Path

from marin.evaluation.model_config import load_model_config

revision = "0f0813072d2358973511097385626f21fcb6d422"
standard = load_model_config(Path(os.environ["STANDARD_MODEL_CONFIG"]))
tau3 = load_model_config(Path(os.environ["TAU3_MODEL_CONFIG"]))
assert standard.location == tau3.location == "Qwen/Qwen3.5-35B-A3B-Base"
assert standard.revision == tau3.revision == revision
assert standard.serve.max_model_len == 73728
assert tau3.serve.max_model_len == 131072
assert tau3.generation.max_gen_toks == 32768
assert standard.serve.tensor_parallel_size == tau3.serve.tensor_parallel_size == 2
assert standard.serve.data_parallel_size == tau3.serve.data_parallel_size == 2
PY
)

launch_evalchemy() {
  local config=$1
  reset_staging_root
  stage_model_config "$model" "$model_config"
  run_campaign_launch "$submit" \
    --model-config "$STAGED_MODEL_CONFIG" \
    --federated_cluster "$FEDERATED_CLUSTER" \
    --priority "$PRIORITY" \
    --evalchemy-config "$EVALCHEMY_CONFIG_DIR/$config.yaml"
}

launch_harbor() {
  local config=$1
  local selected_model_config=$model_config
  [ "$config" != tau3-pi ] || selected_model_config=$tau3_model_config
  reset_staging_root
  stage_model_config "$model" "$selected_model_config"
  stage_harbor_configs "$model"
  local arguments=(
    --model-config "$STAGED_MODEL_CONFIG"
    --federated_cluster "$FEDERATED_CLUSTER"
    --priority "$PRIORITY"
    --harbor-config "$STAGED_HARBOR_DIR/$config.yaml"
  )
  if [ "$config" = bixbench-pi ]; then
    arguments+=(--judge-model-config "$JUDGE_MODEL_CONFIG" --judge-accelerator H100x8)
  fi
  run_campaign_launch "$submit" "${arguments[@]}"
}

if [ "$phase" = cache ]; then
  launch_evalchemy piqa
  exit 0
fi

reset_staging_root
stage_model_config "$model" "$model_config"
evalchemy_names=(
  math500 humanevalplus mbppplus olympiadbench gsm8k-0shot winogrande boolq
  truthfulqa triviaqa aime24 mmlu-pro gpqa-diamond cruxeval financebench ifbench
  mrcr nupa
)
arguments=(
  --model-config "$STAGED_MODEL_CONFIG"
  --federated_cluster "$FEDERATED_CLUSTER"
  --priority "$PRIORITY"
)
for name in "${evalchemy_names[@]}"; do
  arguments+=(--evalchemy-config "$EVALCHEMY_CONFIG_DIR/$name.yaml")
done
run_campaign_launch "$submit" "${arguments[@]}"

harbor_names=(
  swebench-verified ot-tblite-recovery tb2-recovery ds-1000-local
  bfclparity-pi bixbench-pi tau3-pi sotopia-hard
)
for name in "${harbor_names[@]}"; do
  launch_harbor "$name"
done
