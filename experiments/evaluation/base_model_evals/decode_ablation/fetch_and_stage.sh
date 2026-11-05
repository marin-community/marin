#!/usr/bin/env bash
# Pull generation outputs (small: traces, summaries, self-scores, stores) to ../data and
# stage the combined trace file for the scorer jobs.  usage: fetch_and_stage.sh [gen|smoke] MODEL...
set -euo pipefail
kind=$1; shift
set -a; source /Users/benfeuer/Documents/secrets.env; set +a
FS=/Users/benfeuer/Documents/marin-worktrees/chat-aligned-base-evals/.venv/bin/fsutil
ROOT=s3://marin-rno2a/tmp/ttl=30d/benfeuer/decode-ablation/2026.11.04/v2
D=/Users/benfeuer/Documents/experiments/active/base-model-evals/artifacts/decode-ablation/data
[[ $kind == smoke ]] && D=$D-smoke
mkdir -p $D/traces $D/summaries $D/selfscores $D/stores $D/logs
for m in "$@"; do
  for f in $($FS ls $ROOT/$kind/$m/ | grep -v '/$'); do
    b=$(basename "$f")
    case "$b" in
      traces-*) dir=traces ;; summary-*) dir=summaries ;; selfscore-*.gz) dir=selfscores ;; *) dir=logs/$m ;;
    esac
    mkdir -p $D/$dir; $FS cp "$ROOT/$kind/$m/$b" "$D/$dir/$b" >/dev/null
  done
  for s in $($FS ls $ROOT/$kind/$m/ | grep 'store-'); do
    c=$(basename "$s"); $FS rsync "$ROOT/$kind/$m/$c" "$D/stores/$m/$c" >/dev/null
  done
done
cat $D/traces/traces-*.jsonl > $D/all-cells.jsonl
wc -l $D/all-cells.jsonl
$FS cp $D/all-cells.jsonl $ROOT/traces/$kind-all-cells.jsonl
