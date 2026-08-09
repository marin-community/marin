# Generated normalized-exp Contract reverse on H100

This artifact records the first physical execution of Shuttle's bounded generic
normalized-exp reverse family through the Torch-free JAX CUDA typed-FFI
boundary. The generated one-CTA body contains a BF16 score Contract, identity
score Map, restricted normalized-exp Fold reverse, indexed selected-coordinate
correction, and two BF16 reverse Contracts.

This is proof of component execution, not acceptance of the full Grug training
step or a general attention-backward performance claim. The shape is the small
`[8,32] @ [32,128]` output-head reverse recovered from the preserved natural
Grug HLO fixture. No physical schedule tuning was performed.

## CPU preflight

A CPU-only Iris job compiled, linked, and loaded both generated `sm_90a`
programs before the physical reservation:

- identity score Map;
- `6 * tanh(raw_score / 6)` mutation and its generated derivative.

Both handler symbols resolved. The mutation retained rows, reduction extent,
Fold extent, thread count, shared-memory size, and the generic physical family,
while changing semantic and source digests. It was not executed on a GPU.

Job: `/dlwh/shuttle-normalized-exp-reverse-preflight-20260809`.

## H100 result

The single physical run used one NVIDIA H100 80GB HBM3, driver `595.71.05`,
JAX/JAXlib `0.10.1`, and generated `sm_90a` code. The batch reservation
requested one H100, one CPU, 16 GB host memory, and 50 GB ephemeral disk. The
post-measurement telemetry sample reported a 700 W power limit, 1830 MHz SM
clock, and 2619 MHz memory clock.

Thirty samples alternated both process orders exactly fifteen times. Each sample
is the average of 1,000 executions.

| Path | Median latency |
| --- | ---: |
| Generated JAX typed FFI | 0.029879 ms |
| Matched natural JAX reverse | 0.037568 ms |
| Generated / matched JAX | 0.795326x |

The generated component is `20.47%` faster for this tiny fixed shape. This is
mostly evidence that the fused family executes and avoids XLA's multi-operation
component overhead; it does not predict long-sequence attention throughput.
`h100/result.json` preserves all raw samples and launch orders.

Job: `/dlwh/shuttle-normalized-exp-reverse-h100-20260809`.

## Semantic fixture and correctness

The identity fixture exercises:

- 99 valid and 29 invalid Fold coordinates;
- nontrivial selected indices `[1,17,31,47,61,79,101,127]`;
- three invalid rows;
- nonzero random row cotangents and BF16 Contract inputs.

All selected indices address valid Fold positions. For invalid rows, the
selected-coordinate correction is disabled while the normalized-exp term
remains active, matching the recovered generic semantics.

An ordinary JAX row objective independently computes the BF16 score Contract,
restricted log-sum-exp, and indexed selected score. JAX owns its VJP. A second
explicit JAX reverse accepts the same saved-state boundary used by the generated
handler. Both generated outputs are bitwise identical to both references:
maximum and mean absolute error are zero for input and operand cotangents.

Three generated executions produced identical output hashes. The handler count
was exactly `30,013`: three correctness/determinism calls, ten warmups, and
30,000 timed calls.

## Numerical and runtime boundary

The score Contract uses FP32 accumulation and rounds to BF16 RNE before the
normalized-exp Map/Fold. The FP32 score cotangent rounds to BF16 RNE before both
reverse Contracts, whose FP32 accumulators round to BF16 outputs. A deliberately
relaxed JAX reference without the intermediate BF16 boundaries produces
different hashes, so the boundaries are observable rather than documentary.

The generated CUDA uses ordered FP32 loops. XLA may choose a different FP32
reduction tree; this fixture happened to round identically.

Compilation, input generation, and saved-state forward computation are excluded
from both timed reverse paths. Timing begins before enqueueing 1,000 calls and
ends after `jax.block_until_ready` on the final result, then divides by 1,000.
The generated timing therefore includes the JAX typed-FFI and kernel-launch
boundary. The matched JAX timing accepts the same saved state and excludes the
same forward work. The independently differentiated natural JAX program is a
correctness reference, not the timed baseline.

The later compact normalized-exp forward family at revision `554e4ecc65` was
not available before this authorized H100 invocation completed. It was not
added retroactively and no second physical run was launched.

## Files

- `h100/result.json`: raw timings, order, fixture hashes, numerical audit,
  correctness, determinism, and handler count.
- `h100/generated/identity/generated_normalized_exp_contract_reverse.cu`:
  exact executed generated source.
- `h100/generated/source-natural-reverse-stablehlo.mlir.bc`: frozen ordinary
  JAX VJP reference.
- `h100/generated/matched-natural-reverse-optimized-hlo.txt`: optimized timed
  JAX reverse.
- `preflight/preflight.json`: CPU compile/load audit.
- `preflight/generated/*/*.cu`: identity and tanh-softcap generated sources.
- `reproduction.txt`: exact bounded commands.
- `SHA256SUMS`: checksums for every artifact file other than itself.

