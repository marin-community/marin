#!/usr/bin/env bash
# S5 H100 — controlled Triton FP8 tuning sweep (GFP8-018 "B-lite": mixed-f8 dtype fix via
# lax.dot_general + f32 accumulator, plus block_k/scheduling retune). Single job, identical
# real Grug shape and step count across every arm, so fp8-vs-bf16 and tuning deltas are
# apples-to-apples. Backend forced to Triton.
#
#   Q1: does the dtype fix unblock the fp8 backward on Triton? (it crashed in pl.dot before)
#   Q2: does tuning block_k / warps / stages lift fp8 above the bf16 production ~455 TFLOP/s?
#   Q3: how much of any fp8 win is f8 MMA vs just smaller operands? RAGGED_DOT_F8_COMPUTE=bf16
#       upcasts inside the kernel (f8 memory traffic, bf16 tensor-core math).
#
# Smem note: at block_m=block_n=128 the f8 tiles cost ~64KB/pipeline-stage; the 232KB H100
# smem budget caps block_k=128 at num_stages<=3 and rules out block_k=256 at these tiles
# (a prior run OOM'd block_k>=128 at the default 4 stages). Each arm is paired accordingly.
set +e
B=lib/levanter/scripts/bench/bench_ragged_fp8.py
COMMON="--implementation triton --tokens 8192 --hidden 2048 --intermediate 5632 --experts 8 --steps 20 --warmup 5 --no-print-hlo"

echo "===== BASELINES (fwd+bwd, real shape) ====="
echo "### B0 bf16 fwd+bwd — production baseline (~455)"
python $B $COMMON --path bf16
echo "### B1 fp8 fwd+bwd — DEFAULT block_k=32 (does the backward run now?)"
python $B $COMMON --path fp8

echo "===== block_k / scheduling sweep (fp8 fwd+bwd) ====="
echo "### K64 fp8 fwd+bwd block_k=64"
RAGGED_DOT_BLOCK_K=64 python $B $COMMON --path fp8
echo "### K64-W8 fp8 fwd+bwd block_k=64 warps=8"
RAGGED_DOT_BLOCK_K=64 RAGGED_DOT_NUM_WARPS=8 python $B $COMMON --path fp8
echo "### K128-S3 fp8 fwd+bwd block_k=128 stages=3"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_NUM_STAGES=3 python $B $COMMON --path fp8
echo "### K128-S2 fp8 fwd+bwd block_k=128 stages=2"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_NUM_STAGES=2 python $B $COMMON --path fp8
echo "### K128-S2-W8 fp8 fwd+bwd block_k=128 stages=2 warps=8"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_NUM_STAGES=2 RAGGED_DOT_NUM_WARPS=8 python $B $COMMON --path fp8

echo "===== f8-MMA attribution control (Q3): same f8 operands, bf16 MMA inside the kernel ====="
echo "### C-bf16 fp8 fwd+bwd block_k=64 F8_COMPUTE=bf16"
RAGGED_DOT_BLOCK_K=64 RAGGED_DOT_F8_COMPUTE=bf16 python $B $COMMON --path fp8

echo "===== forward-only isolation (cleanest single-GEMM f8 signal) ====="
echo "### F-bf16 bf16 fwd-only (~417)"
python $B $COMMON --path bf16 --forward-only
echo "### F-fp8-32 fp8 fwd-only block_k=32 (prior: 240)"
python $B $COMMON --path fp8 --forward-only
echo "### F-fp8-64 fp8 fwd-only block_k=64"
RAGGED_DOT_BLOCK_K=64 python $B $COMMON --path fp8 --forward-only
echo "### F-fp8-128-S3 fp8 fwd-only block_k=128 stages=3"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_NUM_STAGES=3 python $B $COMMON --path fp8 --forward-only
echo "### DONE"
