#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# One #212 generation lane: run its shards one after another, rebuild each shard's traces and check them
# for capability tokens. Each step skips work already finished (--skip-if-done), so a lane can be resubmitted.
#
#   plain: k attempts per task                          -> <out>/shard-NN
#   relay: round 1 is 2 attempts per task; each later   -> <out>/shard-NN/round-R
#          round is 1 attempt for every non-held-out task with fewer than 4 relay episodes (a pass with
#          a GLM turn) and fewer than MAX_RUNS runs (default 5; relay_followup.py), up to round MAX_ROUND
#          (default 4). Before each new round, a lane stops once <out>/STOP exists (the family's relay target
#          is met) and moves to its next shard once <out>/shard-NN/STOP exists (that shard's target is met).
#
# Needs GLM_BULK_TOKEN and DAYTONA_API_KEY in the environment, and for relay STUDENT_BASE_URL (serve_student.py).
# Optional: STUDENT (harbor_glm --student, default v0; recorded in every run) and REASONING_EFFORT (low or high; unset
# sends none, so GLM's template uses Max), BUDGET_FRACTION (relay handoff trigger as a fraction of the window;
# unset keeps harbor_glm's default 0.75) and WINDOW_TOKENS (SWE-smith generation window; unset keeps harbor_glm's
# 30,000).
#   bash experiments/post_training/sft_shootout/run_lane.sh <plain|relay> <family> <agent> <k> <concurrency> \
#     <tasks parquet> <package revision> <shards dir> <out> <shard numbers...>
set -uo pipefail
mode=$1 family=$2 agent=$3 k=$4 concurrency=$5 tasks=$6 revision=$7 shards=$8 out=$9
shift 9
dir=experiments/post_training/sft_shootout
max_round=${MAX_ROUND:-4}
max_runs=${MAX_RUNS:-5}

exists() {  # <remote path>
  uv run python -c "import sys; from rigging.filesystem.storage_path import StoragePath; sys.exit(not StoragePath('$1').exists())"
}

# Which credentials this lane holds, as fingerprints (sha256 prefix), so a secret rotation can be checked from the log.
fingerprint() { { set +x; } 2>/dev/null; printf %s "${!1:-}" | sha256sum | cut -c1-12; }
echo "credential fingerprints: GLM_BULK_TOKEN=$(fingerprint GLM_BULK_TOKEN) DAYTONA_API_KEY=$(fingerprint DAYTONA_API_KEY)"

generate() {  # <task list> <name> <attempts> <output> [--relay]
  uv run python $dir/harbor_glm.py --agent "$agent" --family "$family" --package-revision "$revision" --tasks "$tasks" \
    --task-list "$1" --name "$2" --n-attempts "$3" --n-concurrent "$concurrency" --skip-if-done --output "$4" \
    --student "${STUDENT:-v0}" ${REASONING_EFFORT:+--reasoning-effort "$REASONING_EFFORT"} \
    ${BUDGET_FRACTION:+--budget-fraction "$BUDGET_FRACTION"} ${WINDOW_TOKENS:+--window-tokens "$WINDOW_TOKENS"} "${@:5}"
}

finish() {  # <output>
  uv run python $dir/rebuild_traces.py "$1" "$1/rebuilt.jsonl" | tail -1 || echo "REBUILD_FAILED $1"
  uv run python $dir/scrub_run.py "$1" --check | tail -1 || echo "TOKENS_FOUND $1"
}

for i in "$@"; do
  shard="$shards/$family-shard-$i.json"
  if [ "$mode" = plain ]; then
    generate "$shard" "$family-shard-$i" "$k" "$out/shard-$i" && finish "$out/shard-$i" || echo "SHARD_FAILED $i"
    continue
  fi
  rounds=()
  for r in $(seq 1 "$max_round"); do
    round="$out/shard-$i/round-$r"
    if exists "$round/_SUCCESS"; then  # finished earlier (its task list stays as it was)
      exists "$round/rebuilt.jsonl" || finish "$round"
      rounds+=("$round")
      continue
    fi
    if exists "$out/STOP"; then echo "STOP marker: $out/STOP"; break 2; fi
    if exists "$out/shard-$i/STOP"; then echo "STOP marker: $out/shard-$i/STOP"; break; fi
    if [ "$r" = 1 ]; then
      generate "$shard" "$family-relay-$i-r1" 2 "$round" --relay || { echo "SHARD_FAILED $i round-1"; break; }
    else
      remaining=$(uv run python $dir/relay_followup.py --shard "$shard" --rounds "${rounds[@]}" --need 4 --max-runs "$max_runs" \
        --out "$round-tasks.json" | tail -1) || { echo "SHARD_FAILED $i followup $r"; break; }
      echo "shard $i: $remaining tasks for round $r"
      [ "$remaining" -gt 0 ] || break
      generate "$round-tasks.json" "$family-relay-$i-r$r" 1 "$round" --relay || { echo "SHARD_FAILED $i round-$r"; break; }
    fi
    finish "$round"
    rounds+=("$round")
  done
done
