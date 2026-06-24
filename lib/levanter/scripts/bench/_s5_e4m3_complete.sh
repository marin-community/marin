#!/usr/bin/env bash
# S5 H100 — completes the Triton f8 table. The Pallas-Triton dot_general lowering rejects
# MIXED f8 (e5m2 x e4m3), so the only f8 fwd+bwd that lowers is same-type e4m3 (output grad
# also quantized to e4m3 via RAGGED_DOT_F8_COMPUTE=e4m3). This is the best-case Triton f8
# fwd+bwd number to weigh against bf16 ~454 and the bf16-upcast control (272).
set +e
B=lib/levanter/scripts/bench/bench_ragged_fp8.py
COMMON="--implementation triton --tokens 8192 --hidden 2048 --intermediate 5632 --experts 8 --steps 20 --warmup 5 --no-print-hlo"

echo "### E4M3-K32 fp8 fwd+bwd block_k=32 F8_COMPUTE=e4m3 (best-case same-type f8 fwd+bwd)"
RAGGED_DOT_F8_COMPUTE=e4m3 python $B $COMMON --path fp8
echo "### E4M3-K64 fp8 fwd+bwd block_k=64 F8_COMPUTE=e4m3"
RAGGED_DOT_BLOCK_K=64 RAGGED_DOT_F8_COMPUTE=e4m3 python $B $COMMON --path fp8
echo "### E4M3-K128-S3 fp8 fwd+bwd block_k=128 stages=3 F8_COMPUTE=e4m3"
RAGGED_DOT_BLOCK_K=128 RAGGED_DOT_NUM_STAGES=3 RAGGED_DOT_F8_COMPUTE=e4m3 python $B $COMMON --path fp8
echo "### E4M3-F64 fp8 fwd-only block_k=64 F8_COMPUTE=e4m3 (sanity vs passthrough 261)"
RAGGED_DOT_BLOCK_K=64 RAGGED_DOT_F8_COMPUTE=e4m3 python $B $COMMON --path fp8 --forward-only
echo "### DONE"
