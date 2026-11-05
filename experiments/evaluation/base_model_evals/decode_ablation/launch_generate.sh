#!/usr/bin/env bash
# usage: launch_generate.sh JOB_NAME MODEL LIMIT CELL...
# Submits one whole-node H100x8 generation job (decode_ablation_generate.sh) to cw-rno2a.
set -euo pipefail
name=$1; model=$2; limit=$3; shift 3
cd /Users/benfeuer/Documents/marin-worktrees/decode-ablation
set -a; source /Users/benfeuer/Documents/secrets.env; set +a
ROOT=s3://marin-rno2a/tmp/ttl=30d/benfeuer/decode-ablation/2026.11.04/v2
DEST=$ROOT/gen
[[ "$limit" == full || "$limit" == selfscore ]] || DEST=$ROOT/smoke
IMG=ghcr.io/marin-community/iris-task@sha256:13519c59442bd70ed5ec2902869f15094992d6849d7e34e353f1338dec7ff5f8
/Users/benfeuer/Documents/marin-worktrees/chat-aligned-base-evals/.venv/bin/iris --cluster marin job run \
  --target-cluster cw-rno2a --enable-extra-resources --priority interactive --no-wait \
  --gpu H100x8 --cpu 64 --memory 512GB --disk 512GB --timeout 14400 \
  -e MARIN_PREFIX s3://marin-rno2a/marin -e MARIN_TEMP_PREFIX s3://marin-rno2a \
  --job-name "$name" --task-image "$IMG" -- \
  bash experiments/evaluation/mini_policy_runtime/decode_ablation_generate.sh "$model" "$DEST" "$limit" "$@"
