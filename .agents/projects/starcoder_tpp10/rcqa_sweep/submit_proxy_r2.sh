#!/bin/zsh
# Resubmission 2 (two truncated shards repinned; plan rebuilt): submit the Wikipedia-as-QA (wiki_to_rcqa) proxy sweep coordinator to Iris (us-central1-a). Secrets are
# sourced into this shell only; the caller must pipe stdout/stderr through the redaction sed.
set -euo pipefail
cd /Users/calvinxu/Projects/Work/Marin/marin
set -a; source ~/.zshrc.secrets; set +a
export MARIN_PREFIX=gs://marin-us-central1
uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
  --enable-extra-resources --cpu 2 --memory 8GB --disk 32GB --timeout 172800 \
  --max-retries 0 --max-preemption-retries 0 --priority interactive \
  --extra cpu --no-preemptible --region us-central1 --zone us-central1-a \
  --job-name tpp10-rcqa-proxy-sweep-r2 \
  --exclude '^(checkpoints|logs|wandb)/' \
  --exclude '^experiments/domain_phase_mix/exploratory/.*(?<!\.py)$' \
  --exclude '^\.agents/(?!projects/starcoder_tpp10/rcqa_sweep/(plan_proxy_r2|release_proxy_r2)\.json$).*' \
  --exclude '^(docs|tests)/' \
  --exclude '^lib/[^/]+/tests/' \
  --exclude '^experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/' \
  --exclude '\.(png|pkl|npz|parquet|pdf)$' \
  --exclude '^third_party/wheels/.*macosx' \
  --exclude '^infra/grafana/' \
  -e MARIN_PREFIX gs://marin-us-central1 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.domain_phase_mix.launch_tpp10_rcqa_sweep \
  --plan .agents/projects/starcoder_tpp10/rcqa_sweep/plan_proxy_r2.json \
  --release .agents/projects/starcoder_tpp10/rcqa_sweep/release_proxy_r2.json --stage proxy --submit
