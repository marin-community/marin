#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage: ./launch-campaign.sh [options]

By default, validates the full tracker-backed campaign without submitting jobs. Use
--submit to launch separate shared-serving roots for the nonagentic and agentic suites.

Options:
  --model NAME       Launch one model-config basename; repeatable. Default: all.
  --suite NAME       all, nonagentic, agentic, or terminal-bench. Default: all.
  --eval NAME        Select one evaluations.yaml entry; repeatable. Default: the whole suite.
  --limit COUNT      Cap selected benchmark items for a smoke reproduction.
  --version LABEL    Attach a submitter-controlled cohort label. Not policy attestation.
  --run-id ID        Resume the launch with this ID, reusing its finished attempts. Requires one
                     --model and a selection within one suite.
  --submit           Submit jobs. Without this flag every launch is a dry run.
  --wait             Wait for each submitted model before launching the next.
  --help             Show this help.

Environment:
  MARIN_DIR, FEDERATED_CLUSTER, PRIORITY, HF_TOKEN
EOF
}

submit=0
suite=all
version=
limit=
run_id=
models=()
selected_names=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      [ "$#" -ge 2 ] || die "--model requires a value"
      models+=("$2")
      shift 2
      ;;
    --eval)
      [ "$#" -ge 2 ] || die "--eval requires a value"
      selected_names+=("$2")
      shift 2
      ;;
    --suite)
      [ "$#" -ge 2 ] || die "--suite requires a value"
      suite=$2
      shift 2
      ;;
    --version)
      [ "$#" -ge 2 ] || die "--version requires a value"
      [ -n "$2" ] || die "--version requires a non-empty value"
      version=$2
      shift 2
      ;;
    --run-id)
      [ "$#" -ge 2 ] || die "--run-id requires a value"
      run_id=$2
      shift 2
      ;;
    --limit)
      [ "$#" -ge 2 ] || die "--limit requires a value"
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "--limit must be a positive integer"
      limit=$2
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
    *)
      die "unknown argument: $1"
      ;;
  esac
done

nonagentic_names=(
  math500 humanevalplus mbppplus olympiadbench gsm8k-0shot piqa winogrande
  boolq truthfulqa triviaqa aime24 mmlu-pro gpqa-diamond cruxeval
  financebench ifbench mrcr nupa
)
agentic_names=(
  swebench-verified ot-tblite-recovery tb2-recovery ds-1000-local
  bfclparity-pi bixbench-pi
)
case "$suite" in
  all) ;;
  nonagentic) agentic_names=() ;;
  agentic) nonagentic_names=() ;;
  terminal-bench)
    nonagentic_names=()
    agentic_names=(tb2-recovery)
    ;;
  *) die "--suite must be all, nonagentic, agentic, or terminal-bench" ;;
esac

# Keep the selected names of a suite, in selection order.
selected_in() {
  local name
  for name in "${selected_names[@]}"; do
    [[ " $* " == *" $name "* ]] && echo "$name"
  done
  return 0
}

if [ "${#selected_names[@]}" -gt 0 ]; then
  for name in "${selected_names[@]}"; do
    [[ " ${nonagentic_names[*]} ${agentic_names[*]} " == *" $name "* ]] \
      || die "--eval $name is not an evaluation of suite $suite"
  done
  mapfile -t nonagentic_names < <(selected_in "${nonagentic_names[@]}")
  mapfile -t agentic_names < <(selected_in "${agentic_names[@]}")
fi

if [ "${#models[@]}" -eq 0 ]; then
  while IFS= read -r config; do
    models+=("${config##*/}")
  done < <(find "$MODEL_CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
  for index in "${!models[@]}"; do
    models[$index]="${models[$index]%.yaml}"
  done
fi

# The nonagentic and agentic suites launch separately, so one launch ID names one model's suite.
if [ -n "$run_id" ]; then
  [ "${#models[@]}" -eq 1 ] || die "--run-id resumes one model's launch; pass one --model"
  [ "${#nonagentic_names[@]}" -eq 0 ] || [ "${#agentic_names[@]}" -eq 0 ] \
    || die "--run-id resumes one suite's launch; select evaluations of one suite"
fi

validate_marin_checkout
validate_campaign_configs
prepare_credentials
if [ "${WAIT_FOR_RESULTS:-0}" = "1" ]; then
  resolve_coreweave_credentials
fi

for model in "${models[@]}"; do
  model_config="$MODEL_CONFIG_DIR/$model.yaml"
  [ -f "$model_config" ] || die "unknown model config: $model"
  override_eval=
  if [ "$suite" = terminal-bench ]; then
    override_eval=tb2-recovery
  elif [ "${#selected_names[@]}" -eq 1 ]; then
    override_eval="${selected_names[0]}"
  fi
  if [ -n "$override_eval" ]; then
    override=$(cd "$MARIN_DIR" && CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" MODEL_NAME="$model" EVAL_NAME="$override_eval" uv run python - <<'PY'
import os
from pathlib import Path

import yaml

campaign = yaml.safe_load(Path(os.environ["CAMPAIGN_CONFIG"]).read_text())
print(campaign.get("model_config_overrides", {}).get(os.environ["MODEL_NAME"], {}).get(os.environ["EVAL_NAME"], ""))
PY
)
    if [ -n "$override" ]; then
      model_config="$CANONICAL_CONFIG_ROOT/$override"
      [ -f "$model_config" ] || die "missing canonical model override: $model_config"
    fi
  fi
  reset_staging_root
  stage_model_config "$model" "$model_config"
  stage_evaluation_config "$model"

  base_arguments=(
    --policy-finalization-config "$CANONICAL_CONFIG_ROOT/finalization.yaml"
    --model-config "$STAGED_MODEL_CONFIG"
    --evaluation-config "$STAGED_EVALUATION_CONFIG"
    --federated_cluster "$FEDERATED_CLUSTER"
    --priority "$PRIORITY"
  )
  if [ -n "$version" ]; then
    base_arguments+=(--version "$version")
  fi
  if [ -n "$limit" ]; then
    base_arguments+=(--limit "$limit")
  fi
  if [ -n "$run_id" ]; then
    base_arguments+=(--run-id "$run_id")
  fi
  # Agentic runs serve the model's resolved serving config, so the suites launch separately and
  # nonagentic run identities do not depend on which agentic evaluations share their launch.
  for suite_name in nonagentic agentic; do
    if [ "$suite_name" = nonagentic ]; then
      names=("${nonagentic_names[@]}")
    else
      names=("${agentic_names[@]}")
    fi
    [ "${#names[@]}" -gt 0 ] || continue
    selection=$(IFS=,; echo "${names[*]}")
    echo "== $model ($suite_name; $([ "$submit" = 1 ] && echo submit || echo validate))"
    run_campaign_launch "$submit" "${base_arguments[@]}" --evals "$selection"
  done
done

echo "completed ${#models[@]} model launch plan(s)"
