#!/usr/bin/env bash
# Grid sweep: visual-token budget x in-flight sizing, to find max throughput
# and the CPU:GPU ratio at each budget.
#
# Axes:
#   - budget (in-job): 512..8192 visual tokens per page. 16384 is omitted: the
#     300-DPI upscale cap already binds at 8192 (mean page ~8.1MP vs the 8.4MP
#     budget), so a 16384 arm renders near-identical payloads.
#   - concurrency (in-job): the client-side in-flight sweep.
#   - pod sizing (one job per config; a vLLM restart is needed): lean/base/rich
#     vary API-side CPU, pod RAM, and API-server count together — the API side
#     (multimodal preprocessing) is the resource consumer that scales with both
#     page size and in-flight count.
#   - a high-parallelism arm for the cheap budgets, where saturation needs more
#     in-flight requests than the base grid's 768 (and more engine slots).
#
# GPU jobs run strictly sequentially: co-scheduled bench jobs can land on the
# same GB200 node and contaminate each other through node CPU/RAM contention.
# Each job rewrites its durable results to $RESULTS_BASE after every point
# (--results-jsonl), so nothing depends on job-log retention.
#
# Usage: ./experiments/b200_ocr/launch_budget_sweep.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

RESULTS_BASE=s3://marin-us-east-02a/marin/scratch/mark/ocr-budget-sweep
BUDGETS=512,1024,2048,4096,8192

SUBMIT=(uv run iris --cluster=marin job run --target-cluster cw-us-east-08a
        --cpu 32 --memory 48GB --disk 50GB --enable-extra-resources
        --extra datakit --priority interactive --user mark --no-wait)
BENCH=(python -m experiments.b200_ocr.bench_infinity_parser
       --gdn-backend flashinfer --prefix-caching
       --max-model-len 24576 --max-tokens 4096
       --max-num-seqs 1024 --max-num-batched-tokens 131072
       --num-pdfs 64 --max-pages-per-pdf 6 --num-requests 2048 --warmup 8
       --client-processes 16
       --max-visual-tokens "$BUDGETS")

run_and_wait() {
  local name=$1
  shift
  "${SUBMIT[@]}" --job-name "$name" -- "$@" --results-jsonl "$RESULTS_BASE/$name.jsonl"
  echo "waiting on /mark/$name ..."
  # A failed arm (e.g. the lean pod collapsing at the big budgets) is itself a
  # data point; keep going so the remaining arms still run.
  uv run iris --cluster=marin job wait "/mark/$name" || echo "arm $name did not succeed; continuing"
}

# Render cost per budget (pages/core/s), for the CPU:GPU ratio. No GPU, so it
# may run while GPU arms queue; it finishes long before they matter.
"${SUBMIT[@]}" --job-name ocr-budget-cpubench -- "${BENCH[@]}" \
    --cpu-bench-only --results-jsonl "$RESULTS_BASE/ocr-budget-cpubench.jsonl"

run_and_wait ocr-budget-base "${BENCH[@]}" \
    --gpu-worker-cpu 64 --gpu-worker-ram-gb 220 --api-server-count 4 \
    --concurrency 256,512,768

run_and_wait ocr-budget-lean "${BENCH[@]}" \
    --gpu-worker-cpu 32 --gpu-worker-ram-gb 160 --api-server-count 2 \
    --concurrency 256,512,768

run_and_wait ocr-budget-rich "${BENCH[@]}" \
    --gpu-worker-cpu 96 --gpu-worker-ram-gb 280 --api-server-count 8 \
    --concurrency 256,512,768

# Cheap pages saturate the GPU only at much higher in-flight counts; open up
# the engine (seqs/btok) and the client accordingly. num-requests rises so the
# largest concurrency still gets a steady-state window (>2x in-flight).
run_and_wait ocr-budget-highpar "${BENCH[@]}" \
    --gpu-worker-cpu 64 --gpu-worker-ram-gb 220 --api-server-count 4 \
    --max-visual-tokens 512,1024,2048 \
    --max-num-seqs 2048 --max-num-batched-tokens 262144 \
    --num-requests 4096 --concurrency 512,1024,1536

echo "budget sweep complete; results under $RESULTS_BASE/"
