#!/usr/bin/env bash
# S5 H100 — FP8 ragged_dot on the XLA backend (jax.lax.ragged_dot_general + manual q/dq).
# This is the "simplest thing": no custom kernel. Goal here is CORRECTNESS, not speed —
# confirm the f8 fwd+bwd runs end-to-end on GPU (the Triton backward crashed on mixed f8;
# XLA's ragged_dot_general accepts mixed f8), numerics track bf16, and f8 operands reach the
# GEMM. XLA ragged on GPU is known slow / compile-memory-heavy at full shapes, so the first
# arm uses a moderate shape that is guaranteed to complete; later arms attempt the real shape.
set +e
B=lib/levanter/scripts/bench/bench_ragged_fp8.py

echo "### ARM 1: fp8 XLA fwd+bwd (MODERATE shape) — guaranteed backward correctness + numerics"
python "$B" --path fp8 --implementation xla --tokens 2048 --hidden 1024 --intermediate 2048 --experts 8 \
  --steps 10 --warmup 3 --no-print-hlo

echo "### ARM 2: bf16 XLA fwd+bwd (MODERATE shape) — same-backend baseline"
python "$B" --path bf16 --implementation xla --tokens 2048 --hidden 1024 --intermediate 2048 --experts 8 \
  --steps 10 --warmup 3 --no-print-hlo

echo "### ARM 3: fp8 XLA fwd-only (small) WITH full HLO — confirm f8 operands into the XLA ragged GEMM"
python "$B" --path fp8 --implementation xla --tokens 1024 --hidden 512 --intermediate 1024 --experts 8 \
  --steps 5 --warmup 2

echo "### ARM 4: fp8 XLA fwd+bwd (REAL Grug shape) — may compile-OOM; informational"
python "$B" --path fp8 --implementation xla --tokens 8192 --hidden 2048 --intermediate 5632 --experts 8 \
  --steps 5 --warmup 2 --no-print-hlo

echo "### ARM 5: bf16 XLA fwd+bwd (REAL Grug shape) — baseline; may be slow/OOM"
python "$B" --path bf16 --implementation xla --tokens 8192 --hidden 2048 --intermediate 5632 --experts 8 \
  --steps 5 --warmup 2 --no-print-hlo

echo "### DONE"
