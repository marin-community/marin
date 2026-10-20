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
  --limit COUNT      Cap items for smoke testing; requires an experimental --version.
  --version LABEL    Override the default verified cohort label.
  --submit           Submit jobs. Without this flag every launch is a dry run.
  --wait             Wait for each submitted model before launching the next.
  --help             Show this help.

Environment:
  MARIN_DIR, FEDERATED_CLUSTER, PRIORITY, HF_TOKEN
EOF
}

submit=0
suite=all
version=$CAMPAIGN_VERIFIED_VERSION
limit=
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

if [ -n "$limit" ] && [[ "$version" == eval-policy-* ]]; then
  die "--limit requires an experimental cohort, for example --version smoke"
fi

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
  [ -f "$MODEL_CONFIG_DIR/$model.yaml" ] || die "unknown model config: $model"
  reset_staging_root
  stage_harbor_configs "$model"
  for mechanism in nonagentic agentic; do
    if [ "$mechanism" = nonagentic ]; then
      [ "$suite" = all ] || [ "$suite" = nonagentic ] || continue
      launch_names=("${evalchemy_names[@]}")
      config_flag=--evalchemy-config
      config_directory=$EVALCHEMY_CONFIG_DIR
    else
      [ "$suite" = all ] || [ "$suite" = agentic ] || [ "$suite" = terminal-bench ] || continue
      launch_names=("${harbor_names[@]}")
      if [ "$suite" = terminal-bench ]; then
        launch_names=(tb2-recovery)
      elif [ "${#selected_harbor_names[@]}" -gt 0 ]; then
        launch_names=("${selected_harbor_names[@]}")
      fi
      config_flag=--harbor-config
      config_directory=$STAGED_HARBOR_DIR
    fi
    groups=$(campaign_model_groups "$model" "${launch_names[@]}")
    while IFS=$'\t' read -r config names; do
      stage_model_config "$model" "$CANONICAL_CONFIG_ROOT/$config"
      arguments=(
        --model-config "$STAGED_MODEL_CONFIG"
        --federated_cluster "$FEDERATED_CLUSTER"
        --priority "$PRIORITY"
        --max-concurrent 4
        --version "$version"
      )
      if [ -n "$limit" ]; then
        arguments+=(--limit "$limit")
      fi
      read -r -a group_names <<< "$names"
      for name in "${group_names[@]}"; do
        arguments+=("$config_flag" "$config_directory/$name.yaml")
        if [ "$mechanism" = agentic ] && [ "$name" = bixbench-pi ]; then
          arguments+=(--judge-model-config "$JUDGE_MODEL_CONFIG" --judge-accelerator H100x8 --judge-eval "$name")
        fi
      done
      echo "== $model ($mechanism; $config; $([ "$submit" = 1 ] && echo submit || echo validate))"
      run_campaign_launch "$submit" "${arguments[@]}"
    done <<< "$groups"
  done
done

echo "completed ${#models[@]} model launch plan(s)"
