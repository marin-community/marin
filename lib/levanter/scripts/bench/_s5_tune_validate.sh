#!/usr/bin/env bash
# S5 H100 — controlled Triton FP8 tuning sweep (the GFP8-018 "B-lite": mixed-f8 dtype fix
# + block_k/scheduling retune). Single job, identical real Grug shape and step count across
# every arm, so the fp8-vs-bf16 and tuning deltas are apples-to-apples (the prior bf16/fp8
# numbers came from separate jobs). Backend forced to Triton throughout.
#
#   Q1: does the dtype fix unblock the fp8 backward on Triton? (GFP8-017 crashed here)
#   Q2: does tuning block_k / warps / stages lift fp8 above the bf16 production 447 TFLOP/s?
#   Q3: how much of any fp8 win is f8 MMA vs just smaller operands? RAGGED_DOT_F8_COMPUTE=bf16
#       upcasts inside the kernel (f8 memory traffic, bf16 tensor-core math).
#   Q4: does pl.dot(f8,f8) actually lower to an f8 tensor-core dot? (Triton-IR dump, ARM T)
set +e
B=lib/levanter/scripts/bench/bench_ragged_fp8.py
SHAPE="--tokens 8192 --hidden 2048 --intermediate 5632 --experts 8"
COMMON="--implementation triton $SHAPE --steps 20 --warmup 5 --no-print-hlo"

echo "===== BASELINES (fwd+bwd, real shape) ====="
echo "### B0 bf16 fwd+bwd — production baseline (~447)"
python $B $COMMON --path bf16
echo "### B1 fp8 fwd+bwd — DEFAULT (block_k=32, passthrough mixed-f8): does the backward run now?"
python $B $COMMON --path fp8

echo "===== block_k sweep (fp8 fwd+bwd, passthrough mixed-f8) ====="
echo "### K64 fp8 fwd+bwd block_k=64"
RAGGED_DOT_BLOCK_K=64 python $B $COMMON --path fp8
echo "### K128 fp8 fwd+bwd block_k=128"
RAGGED_DOT_BLOCK_K=128 python $B $COMMON --path fp8
echo "### K256 fp8 fwd+bwd block_k=256"
RAGGED_DOT_BLOCK_K=256 python $B $COMMON --path fp8

echo "===== scheduling sweep (fp8 fwd+bwd, block_k=128) ====="
echo "### S3 fp8 fwd+bwd block_k=128 stages=3"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_NUM_STAGES=3 python $B $COMMON --path fp8
echo "### W8 fp8 fwd+bwd block_k=128 warps=8"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_NUM_WARPS=8 python $B $COMMON --path fp8
echo "### N256 fp8 fwd+bwd block_k=128 block_n=256"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_BLOCK_N=256 python $B $COMMON --path fp8

echo "===== f8-MMA attribution control (Q3): same f8 operands, bf16 MMA inside the kernel ====="
echo "### C-bf16 fp8 fwd+bwd block_k=128 F8_COMPUTE=bf16"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_F8_COMPUTE=bf16 python $B $COMMON --path fp8

echo "===== forward-only isolation (cleanest single-GEMM f8 signal) ====="
echo "### F-bf16 bf16 fwd-only (~411)"
python $B $COMMON --path bf16 --forward-only
echo "### F-fp8-32 fp8 fwd-only block_k=32"
python $B $COMMON --path fp8 --forward-only
echo "### F-fp8-128 fp8 fwd-only block_k=128"
RAGGED_DOT_BLOCK_K=128 python $B $COMMON --path fp8 --forward-only
echo "### F-fp8-256 fp8 fwd-only block_k=256"
RAGGED_DOT_BLOCK_K=256 python $B $COMMON --path fp8 --forward-only

echo "===== ARM T: Triton-IR dump — confirm pl.dot(f8,f8) emits an f8 tensor-core dot ====="
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR="$PWD/_s5_ttir"
rm -rf "$TRITON_DUMP_DIR"; mkdir -p "$TRITON_DUMP_DIR"
RAGGED_DOT_BLOCK_K=128 python $B $COMMON --path fp8 --forward-only
echo "--- Triton-IR f8/dot grep (.ttir/.ttgir) ---"
grep -rhoE "tt\.dot|warp_group_dot|nvgpu.wgmma|wgmma|f8E4M3[A-Za-z]*|f8E5M2" "$TRITON_DUMP_DIR" 2>/dev/null | sort | uniq -c | sort -rn | head -40
echo "--- ttgir files present ---"
find "$TRITON_DUMP_DIR" -name '*.ttgir' 2>/dev/null | head
echo "### DONE"
