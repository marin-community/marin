#!/bin/bash
# Spaced acceptance re-runs: one batch of 4 jobs per 65-minute window, with a
# stale-snapshot purge between windows so the earlier windows' snapshots age
# into the purge threshold and free their quota slots. The 42 cohort contexts
# plus the standing provider-owned snapshots exceed the 40-snapshot org quota
# when built concurrently; spacing makes the arithmetic fit.
set -uo pipefail
cd /Users/benfeuer/Documents/marin-worktrees/atlas-ci-green

REV=86478373a21a30c73e1be85f52ff461c6e5f8af2

set +x
source /Users/benfeuer/Documents/secrets.env
export DAYTONA_RL_API_KEY="$(gcloud secrets versions access latest --project=hai-gcp-models --secret=DAYTONA_RL_API_KEY)"
export DAYTONA_TARGET=us
set -x

for batch_file in /tmp/spaced_batch_*.txt; do
  batch=$(basename "$batch_file" .txt)
  echo "=== $batch $(date -u +%H:%M) ==="
  while IFS=$'\t' read -r jname sid; do
    slug=$(printf '%s' "$sid" | sed -e 's/^[^:]*://' -e 's#[/@]#-#g' | tr '[:upper:]' '[:lower:]' | cut -c1-63 | sed 's/-$//')
    hash8=$(printf '%s' "$sid" | shasum -a 256 | cut -c1-8)
    set +x
    source /Users/benfeuer/Documents/secrets.env
    export DAYTONA_RL_API_KEY="$(gcloud secrets versions access latest --project=hai-gcp-models --secret=DAYTONA_RL_API_KEY)"
    set -x
    uv run --no-sync python -m experiments.post_training.task_curation.launch_acceptance \
      --revision "$REV" --source-id "$sid" \
      --release-repo "open-athena/${slug}-${hash8}-taskspec-20261124g" \
      --job-name "$jname" --sample 12 >/dev/null 2>&1 && echo "submitted $jname" || echo "SUBMIT-FAIL $jname"
  done < "$batch_file"
  echo "=== $batch submitted; purging stale + waiting 65 minutes ==="
  uv run --project lib/shellbox --frozen --extra daytona shellbox-daytona-snapshots \
    --api-key-env DAYTONA_RL_API_KEY purge --apply >> "/tmp/spaced_purge_$batch.log" 2>&1 || true
  sleep 3900
done
echo "=== spaced re-runs complete $(date -u +%H:%M) ==="
