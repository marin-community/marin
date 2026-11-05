#!/usr/bin/env bash
# Run generational_perplexity.py against a GrugMoE checkpoint served the way the
# Mini840 eval runtime serves it: MarinSkyRL's frozen runtime environment (pinned
# vLLM fork, wheel CUDA toolkit wired for FlashInfer JIT) and one TP1 engine with
# DP4/EP4 expert parallelism on four GPUs.
#
# usage: generational_perplexity_serve.sh MIRROR_URI LABEL [probe args...]
set -euo pipefail

mirror_uri="$1"
label="$2"
shift 2

workspace="$PWD"
model_dir=/tmp/pp/model
environment=/tmp/skyrl-env
runtime_file=/tmp/skyrl-runtime.env

uv run --frozen fsutil rsync "$mirror_uri" "$model_dir"

(cd "$workspace/vendor/skyrl" && bash cloud/iris/bootstrap_runtime.sh "$PWD" "$environment" "$runtime_file" megatron production)
source "$runtime_file"
# Tasks can share a host network namespace, and vLLM's API servers bind with
# SO_REUSEPORT, so a fixed port would silently load-balance across jobs.
port=$("$environment/bin/python" -c "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")

vllm serve "$model_dir" \
  --served-model-name "$label" \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port "$port" &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

until "$environment/bin/python" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/health')" 2>/dev/null; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before becoming healthy" >&2
    wait "$server_pid"
    exit 1
  fi
  sleep 10
done
echo "vLLM server healthy"

# Run as a module: a script path would put mini_policy_runtime/ first on sys.path,
# where profile.py shadows the stdlib module torch._dynamo imports.
"$environment/bin/python" -m experiments.evaluation.mini_policy_runtime.generational_perplexity \
  --model "$model_dir" \
  --label "$label" \
  --engine vllm-server \
  --server-url "http://127.0.0.1:$port" \
  --served-model "$label" \
  "$@"
