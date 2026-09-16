#!/usr/bin/env bash
set -euo pipefail
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
  --enable-extra-resources --cpu 1 --memory 4GB --disk 32GB --timeout 172800 \
  --max-retries 0 --max-preemption-retries 0 --priority interactive \
  --extra cpu --no-preemptible --region us-central1 --zone us-central1-a \
  --job-name tpp10-finemath-target-completion \
  --bundle-include .agents/projects/starcoder_tpp10/domain_sweeps/target_completion_20260912/plan.json \
  --bundle-include 'experiments/domain_phase_mix/tpp10_domain_sweeps_assets/*' \
  --bundle-include 'experiments/domain_phase_mix/starcoder_tpp10_assets/*' \
  --exclude '^(checkpoints|cache|logs|wandb|tmp)/' \
  --exclude '^experiments/domain_phase_mix/exploratory/.*(?<!\.py)$' \
  --exclude '^\.agents/(?!projects/starcoder_tpp10/domain_sweeps/target_completion_20260912/plan\.json$).*' \
  --exclude '^(docs|tests)/' \
  --exclude '^lib/[^/]+/tests/' \
  --exclude '^experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/' \
  -e MARIN_PREFIX gs://marin-us-central1 \
  -e WANDB_API_KEY "${WANDB_API_KEY:?WANDB_API_KEY must be available for the original tracker}" \
  -- python -m experiments.domain_phase_mix.complete_tpp10_finemath_targets \
  --plan .agents/projects/starcoder_tpp10/domain_sweeps/target_completion_20260912/plan.json --submit
