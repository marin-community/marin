#!/usr/bin/env bash
# usage: launch_score.sh JOB_NAME SCORER TRACES_NAME
# Scores $ROOT/traces/TRACES_NAME under one scorer, the trace_launch_v3.sh pattern
# (generational_perplexity_serve.sh, TP1 x DP4/EP4 vLLM, whole H100x8 node), with
# --enable-thinking false for every scorer.
set -euo pipefail
name=$1; scorer=$2; traces=$3
cd /Users/benfeuer/Documents/marin-worktrees/decode-ablation
set -a; source /Users/benfeuer/Documents/secrets.env; set +a
ROOT=s3://marin-rno2a/tmp/ttl=30d/benfeuer/decode-ablation/2026.11.04/v2
DEST=$ROOT/scores
IMG=ghcr.io/marin-community/iris-task@sha256:13519c59442bd70ed5ec2902869f15094992d6849d7e34e353f1338dec7ff5f8
SERVE=experiments/evaluation/mini_policy_runtime/generational_perplexity_serve.sh
TPL=experiments/evaluation/mini_policy_runtime/chat-templates/grug-67b-a2b-datakit-sft.jinja
Q=s3://marin-rno2a/tmp/ttl=30d/quick-serve-models
TR=/tmp/traces.jsonl
fetch=""
case "$scorer" in
  qwen3.5-35b-a3b-base) src=$Q/Qwen_Qwen3.5-35B-A3B-Base_0f0813072d2358973511097385626f21fcb6d422-05871e4233a75e43; extra="" ;;
  qwen3.5-35b-a3b) src=/tmp/pp/hf; extra=""
    fetch="uv run --frozen --with huggingface_hub hf download Qwen/Qwen3.5-35B-A3B --revision 59d61f3ce65a6d9863b86d2e96597125219dc754 --local-dir /tmp/pp/hf &&" ;;
  qwen3.6-35b-a3b) src=$Q/Qwen_Qwen3.6-35B-A3B_995ad96eacd98c81ed38be0c5b274b04031597b0-567d286455fd6825; extra="" ;;
  snowball-67b-a2b-base-skew8) src=$Q/open-athena_snowball-67b-a2b-base-262k-qk175-skew8_3bbf729f0555965c8667600ee91c3c109ac21b19-722f147d8996ed84; extra="--chat-template-file $TPL" ;;
  grug-67b-a2b-datakit-sft) src=$Q/open-athena_Grug-67B-A2B-Datakit-SFT-Repro-262K-Step471-2026.10.25_51669ee19b8972fc2e74c10b39eaef2eb244c44a-cc6512b344b08123; extra="--chat-template-file $TPL" ;;
  *) echo "unknown scorer $scorer" >&2; exit 2 ;;
esac
out=${traces%.jsonl}
/Users/benfeuer/Documents/marin-worktrees/chat-aligned-base-evals/.venv/bin/iris --cluster marin job run \
  --target-cluster cw-rno2a --enable-extra-resources --priority interactive --no-wait \
  --gpu H100x8 --cpu 64 --memory 512GB --disk 512GB --timeout 14400 \
  -e MARIN_PREFIX s3://marin-rno2a/marin -e MARIN_TEMP_PREFIX s3://marin-rno2a \
  --job-name "$name" --task-image "$IMG" -- \
  bash -lc "uv run --frozen fsutil cp $ROOT/traces/$traces $TR && $fetch bash $SERVE $src $scorer --template-family template --enable-thinking false $extra --traces $TR --trace-output /tmp/$scorer.jsonl.gz && uv run --frozen fsutil cp /tmp/$scorer.jsonl.gz $DEST/$out/$scorer.jsonl.gz"
