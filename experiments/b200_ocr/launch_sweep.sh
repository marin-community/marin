#!/usr/bin/env bash
# Launch the B200 OCR throughput sweep: one iris job per *server* config (a vLLM
# restart is needed to change server knobs); client-side knobs (concurrency,
# max_visual_tokens, longest_side) sweep inside each job so the fleet startup is
# paid once.
#
# Usage: ./experiments/b200_ocr/launch_sweep.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SUBMIT=(uv run iris --cluster=marin job run --target-cluster cw-us-east-08a
        --cpu 8 --memory 24GB --disk 50GB --enable-extra-resources
        --extra datakit --priority interactive --user mark --no-wait)
# Attention backend is left at vLLM's Blackwell default (FLASHINFER): the
# prebuilt cubin/jit-cache artifacts the harness ships make it start on the
# nvcc-less CoreWeave images, and the backend A/B measured it ~4-6% faster
# than FLASH_ATTN at saturation.
BENCH=(python -m experiments.b200_ocr.bench_infinity_parser
       --num-pdfs 64 --max-pages-per-pdf 6 --num-requests 192 --warmup 8)

# S1: batching knobs on a single GPU (client concurrency swept in-job).
"${SUBMIT[@]}" --job-name ocr-sweep-batch8k  -- "${BENCH[@]}" \
    --concurrency 8,16,32,64,128
"${SUBMIT[@]}" --job-name ocr-sweep-batch32k -- "${BENCH[@]}" \
    --max-num-batched-tokens 32768 --concurrency 8,16,32,64,128
"${SUBMIT[@]}" --job-name ocr-sweep-batch64k -- "${BENCH[@]}" \
    --max-num-batched-tokens 65536 --concurrency 8,16,32,64,128
"${SUBMIT[@]}" --job-name ocr-sweep-seq128   -- "${BENCH[@]}" \
    --max-num-seqs 128 --max-num-batched-tokens 32768 --concurrency 32,64,128

# S2: cost/quality trade (server at defaults). The budget is the control
# variable; read frac_below_floor in the emitted cpu stats to see what each
# budget does to the large-format tail before reading its pages_per_second.
"${SUBMIT[@]}" --job-name ocr-sweep-res -- "${BENCH[@]}" \
    --max-visual-tokens 1024,2048,4096,8192 --concurrency 32,64

# S3: one instance spanning the node (TP=4) vs S4: four 1-GPU instances brokered.
"${SUBMIT[@]}" --job-name ocr-sweep-tp4 -- "${BENCH[@]}" \
    --tensor-parallel 4 --concurrency 32,64,128
"${SUBMIT[@]}" --job-name ocr-sweep-4x1 -- "${BENCH[@]}" \
    --instances 4 --max-in-flight 64 --concurrency 64,128,256

echo "submitted 7 sweep jobs"
