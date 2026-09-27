#!/bin/sh
set -eu
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
  --priority interactive --job-name dm-delphi-3e18-regmix-reference-20260913 \
  --region us-east5 --zone us-east5-a --enable-extra-resources \
  --cpu 1 --memory 4GB --disk 20GB --timeout 172800 --no-preemptible \
  --max-retries 0 --max-preemption-retries 0 --extra marin-core:tpu \
  --bundle-include experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/regmix_official_rerun_20260913/candidate_weights.csv \
  --exclude '^(checkpoints|logs|wandb)/' --exclude '^\.experiments\.zip$' \
  --exclude '^\.agents/' --exclude '^(docs|tests|tmp|scratch)/' \
  --exclude '^lib/[^/]+/tests/' \
  --exclude '^experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/(?!regmix_official_rerun_20260913/candidate_weights\.csv$)' \
  --exclude '^experiments/domain_phase_mix/exploratory/starcoder_generic_selector_outputs/' \
  --exclude '^experiments/domain_phase_mix/exploratory/two_phase_many/dsre_ceq_debug/' \
  --exclude '^experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many\.csv$' \
  --exclude '\.pdf$' \
  -e MARIN_PREFIX gs://marin-us-east5 \
  -e WANDB_API_KEY "${WANDB_API_KEY:?WANDB_API_KEY is required}" \
  -- python -m experiments.domain_phase_mix.launch_delphi_regmix_reference_3e18 \
  --tpu-type v6e-8 --tpu-region us-east5 --tpu-zone us-east5-b \
  --max-concurrent 4 --prefix gs://marin-us-east5
