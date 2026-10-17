#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# One #212 generation lane: run its shards one after another, rebuild each shard's traces and check them
# for capability tokens. Each step skips work already finished (--skip-if-done), so a lane can be resubmitted.
#
#   plain: k attempts per task                          -> <out>/shard-NN
#   relay: 2 attempts per task, then 1 more for tasks   -> <out>/shard-NN/round-1, round-2
#          with fewer than 2 relay episodes (a pass with a GLM turn)
#
# Needs GLM_BULK_TOKEN and DAYTONA_API_KEY in the environment.
#   bash experiments/post_training/sft_shootout/run_lane.sh <plain|relay> <family> <agent> <k> <concurrency> \
#     <tasks parquet> <package revision> <shards dir> <out> <shard numbers...>
set -u
mode=$1 family=$2 agent=$3 k=$4 concurrency=$5 tasks=$6 revision=$7 shards=$8 out=$9
shift 9
dir=experiments/post_training/sft_shootout

generate() {  # <task list> <name> <attempts> <output> [--relay]
  uv run python $dir/harbor_glm.py --agent "$agent" --family "$family" --package-revision "$revision" --tasks "$tasks" \
    --task-list "$1" --name "$2" --n-attempts "$3" --n-concurrent "$concurrency" --skip-if-done --output "$4" "${@:5}"
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
  first="$out/shard-$i/round-1" second="$out/shard-$i/round-2"
  if ! generate "$shard" "$family-relay-$i-r1" 2 "$first" --relay; then echo "SHARD_FAILED $i round-1"; continue; fi
  finish "$first"
  remaining=$(uv run python $dir/relay_followup.py --shard "$shard" --rounds "$first" --need 2 \
    --out "$out/shard-$i/round-2-tasks.json" | tail -1) || { echo "SHARD_FAILED $i followup"; continue; }
  echo "shard $i: $remaining tasks need round 2"
  if [ "$remaining" -gt 0 ]; then
    generate "$out/shard-$i/round-2-tasks.json" "$family-relay-$i-r2" 1 "$second" --relay && finish "$second" \
      || echo "SHARD_FAILED $i round-2"
  fi
done
