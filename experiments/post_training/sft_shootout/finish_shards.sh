#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Companion to lanes that do not rebuild (the first #212 TMax plain lanes): every few minutes, rebuild and
# scrub each finished shard (_SUCCESS present) that has no rebuilt.jsonl yet; exit once every listed shard
# is rebuilt.
#   bash experiments/post_training/sft_shootout/finish_shards.sh <out> <shard numbers...>
set -uo pipefail
out=$1
shift
dir=experiments/post_training/sft_shootout
pending=("$@")
while [ ${#pending[@]} -gt 0 ]; do
  left=()
  for i in "${pending[@]}"; do
    shard="$out/shard-$i"
    state=$(uv run python -c "from rigging.filesystem.storage_path import StoragePath as P; \
      print('done' if P('$shard/rebuilt.jsonl').exists() else 'ready' if P('$shard/_SUCCESS').exists() else 'wait')")
    case $state in
      done) ;;
      ready)
        uv run python $dir/rebuild_traces.py "$shard" "$shard/rebuilt.jsonl" | tail -1 || echo "REBUILD_FAILED $shard"
        uv run python $dir/scrub_run.py "$shard" | tail -1 && uv run python $dir/scrub_run.py "$shard" --check | tail -1 || echo "TOKENS_LEFT $shard" ;;
      *) left+=("$i") ;;
    esac
  done
  pending=("${left[@]}")
  [ ${#pending[@]} -gt 0 ] && sleep 600
done
echo "ALL_REBUILT $out"
