# Session Directive: Macro Move H (Matmul Batching)

## Objective
Reduce dot invocation count in hot train kernels by stacking left operands that
share the same right operand.

## Priority opportunities
1. `QK` and `KKT` share `k^T`:
   - compute `concat([q, k_beta], axis=0) @ k^T`, then split.
2. `inter` and `v_prime` share `S`:
   - compute `concat([q_scaled, k_cumdecay], axis=0) @ S`, then split.

## Implementation constraints
- Keep using TPU-safe fused transpose matmul helper (`dot_general`-based).
- Preserve BF16 input + FP32 accumulation policy.
- Avoid introducing new trailing singleton lanes in intermediate tensors.
- Avoid increasing peak VMEM enough to trigger compile/runtime OOM.

## Measurement requirements
- Compare before/after train-path `shard_map/pallas_call` wall times.
- Track top callsite delta for both forward and backward closed-call paths.

## Guardrails
- If batching increases compute duplication or hurts numerics, revert and log.
- If gain is <3% and dominant hotspot unchanged, escalate to Macro Move I or E.
