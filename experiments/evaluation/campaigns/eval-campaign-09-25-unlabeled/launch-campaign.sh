#!/usr/bin/env bash
set -euo pipefail
. "$(dirname "$0")/common.sh"

usage() {
  cat <<'EOF'
Usage: ./launch-campaign.sh [options]

By default, validates the full tracker-backed campaign without submitting jobs. Use
--submit to launch separate shared-serving roots for the Evalchemy and Harbor suites.

Options:
  --model NAME       Launch one model-config basename; repeatable. Default: all.
  --suite NAME       all, nonagentic, agentic, or terminal-bench. Default: all.
  --evalchemy NAME   Select one Evalchemy config basename; repeatable. Default: all.
  --harbor NAME      Select one Harbor config basename; repeatable. Default: all.
  --version LABEL    Attach a submitter-controlled cohort label. Not policy attestation.
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
models=()
selected_evalchemy_names=()
selected_harbor_names=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      [ "$#" -ge 2 ] || die "--model requires a value"
      models+=("$2")
      shift 2
      ;;
    --evalchemy)
      [ "$#" -ge 2 ] || die "--evalchemy requires a value"
      selected_evalchemy_names+=("$2")
      shift 2
      ;;
    --harbor)
      [ "$#" -ge 2 ] || die "--harbor requires a value"
      selected_harbor_names+=("$2")
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

case "$suite" in
  all|nonagentic|agentic|terminal-bench) ;;
  *) die "--suite must be all, nonagentic, agentic, or terminal-bench" ;;
esac

if [ "${#models[@]}" -eq 0 ]; then
  while IFS= read -r config; do
    models+=("${config##*/}")
  done < <(find "$MODEL_CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
  for index in "${!models[@]}"; do
    models[$index]="${models[$index]%.yaml}"
  done
fi

validate_marin_checkout
validate_campaign_configs
prepare_judge_environment
if [ "${WAIT_FOR_RESULTS:-0}" = "1" ]; then
  resolve_coreweave_credentials
fi

evalchemy_names=(
  math500 humanevalplus mbppplus olympiadbench gsm8k-0shot piqa winogrande
  boolq truthfulqa triviaqa aime24 mmlu-pro gpqa-diamond cruxeval
  financebench ifbench mrcr nupa
)
if [ "${#selected_evalchemy_names[@]}" -gt 0 ]; then
  for name in "${selected_evalchemy_names[@]}"; do
    [ -f "$EVALCHEMY_CONFIG_DIR/$name.yaml" ] || die "unknown Evalchemy config: $name"
  done
  evalchemy_names=("${selected_evalchemy_names[@]}")
fi
harbor_names=(
  swebench-verified ot-tblite-recovery tb2-recovery ds-1000-local
  bfclparity-pi bixbench-pi tau3-pi sotopia-hard
)
if [ "${#selected_harbor_names[@]}" -gt 0 ]; then
  [ "$suite" = all ] || [ "$suite" = agentic ] \
    || die "--harbor requires --suite all or --suite agentic"
  for name in "${selected_harbor_names[@]}"; do
    [ -f "$HARBOR_CONFIG_DIR/$name.yaml" ] || die "unknown Harbor config: $name"
  done
fi

for model in "${models[@]}"; do
  model_config="$MODEL_CONFIG_DIR/$model.yaml"
  [ -f "$model_config" ] || die "unknown model config: $model"
  reset_staging_root
  stage_model_config "$model"
  stage_harbor_configs "$model"

  base_arguments=(
    --model-config "$STAGED_MODEL_CONFIG"
    --federated_cluster "$FEDERATED_CLUSTER"
    --priority "$PRIORITY"
  )
  if [ -n "$version" ]; then
    base_arguments+=(--version "$version")
  fi
  if [ "$suite" = all ] || [ "$suite" = nonagentic ]; then
    arguments=("${base_arguments[@]}")
    for name in "${evalchemy_names[@]}"; do
      arguments+=(--evalchemy-config "$EVALCHEMY_CONFIG_DIR/$name.yaml")
    done
    echo "== $model (nonagentic; $([ "$submit" = 1 ] && echo submit || echo validate))"
    run_campaign_launch "$submit" "${arguments[@]}"
  fi
  if [ "$suite" = all ] || [ "$suite" = agentic ] || [ "$suite" = terminal-bench ]; then
    arguments=("${base_arguments[@]}")
    launch_harbor_names=("${harbor_names[@]}")
    if [ "$suite" = terminal-bench ]; then
      launch_harbor_names=(tb2-recovery)
    elif [ "${#selected_harbor_names[@]}" -gt 0 ]; then
      launch_harbor_names=("${selected_harbor_names[@]}")
    fi
    for name in "${launch_harbor_names[@]}"; do
      arguments+=(--harbor-config "$STAGED_HARBOR_DIR/$name.yaml")
    done
    for name in "${launch_harbor_names[@]}"; do
      if [ "$name" = bixbench-pi ]; then
        arguments+=(--judge-model-config "$JUDGE_MODEL_CONFIG" --judge-accelerator H100x8)
        break
      fi
    done
    echo "== $model (agentic; $([ "$submit" = 1 ] && echo submit || echo validate))"
    run_campaign_launch "$submit" "${arguments[@]}"
  fi
done

echo "completed ${#models[@]} model launch plan(s)"
