# Generated normalized-exp Contract forward on H100

This artifact records one physical execution of Shuttle's bounded generic
normalized-exp forward family through a Torch-free JAX CUDA typed-FFI boundary.
The generated one-CTA body contains a BF16 score Contract, scalar score Map,
Fold-domain restriction, source-ordered max/sum-exp Folds, indexed selection,
and saved-state finalization.

This is a component proof at the `[8,32] @ [32,128]` output-head shape recovered
from natural Grug HLO. It is not a full-Grug or general attention-forward
performance claim. No schedule tuning or second physical invocation was run.

## Compile preflight

Before the generated GPU invocation, a compile/link/load-only preflight built
the identity score Map and the `6 * tanh(raw_score / 6)` mutation for `sm_90a`.
Both typed-FFI symbols resolved. The mutation changed semantic and source
digests while retaining rows, reduction extent, Fold extent, thread count,
shared-memory size, and the same physical generator. The mutation did not run
on the GPU.

## H100 result

The single physical run used one NVIDIA H100 80GB HBM3 with driver `595.71.05`,
JAX/JAXlib `0.10.1`, CUDA compiler `13.2.78`, and a 700 W power limit. Thirty
samples alternated both path orders exactly fifteen times. Each sample averages
1,000 executions.

| Path | Median latency |
| --- | ---: |
| Generated JAX typed FFI | 0.055004 ms |
| Matched explicit JAX forward | 0.058033 ms |
| Generated / matched JAX | 0.947791x |

The raw samples have two clock regimes. The first and last fifteen generated
medians are `0.055174 ms` and `0.038805 ms`; matched JAX changes from
`0.060217 ms` to `0.044890 ms`. The median paired ratio is `0.900688x`.
`analysis.json` preserves these derived statistics, and `h100/result.json`
preserves every raw sample and execution order. The clock transition makes the
absolute medians unsuitable as a broader performance claim, but both paths were
counterbalanced through the same transition.

## Semantic fixture and correctness

The fixture uses 99 valid and 29 invalid Fold positions. Selected indices are
`[1,-1,31,47,0,79,101,127]`; `-1` and the restricted coordinate `0` create two
invalid rows. Invalid rows produce zero loss and zero saved state.

The matched explicit JAX formulation and an independent natural JAX
log-sum-exp formulation both preserve the exported FP32-accumulate-to-BF16
score boundary. Maximum error from the generated loss and saved state is
`4.76837158203125e-07`; mean error is `1.1920928955078125e-07` for both
references. The matched and natural references are bitwise identical on this
fixture.

Three generated executions produced the same two hashes. The handler count was
exactly `30,013`: three correctness/determinism calls, ten warmups, and 30,000
timed calls. A relaxed reference without the BF16 score boundary produces
different hashes, so the cast boundary is observable.

## Boundary

Compilation and input generation are outside both timings. Timing begins
before enqueueing 1,000 calls and ends after `jax.block_until_ready`, then
divides by 1,000. The generated path includes JAX typed FFI and kernel launch.
The matched path runs the same BF16 score Contract, restriction, indexed
selection, and two outputs through ordinary JAX/XLA.

The generated CUDA uses ordered FP32 Contract and Fold loops. XLA may use a
different FP32 reduction tree. This fixture differs by less than one FP32 ulp
after both references preserve the BF16 score boundary.

## Files

- `h100/result.json`: raw timing, order, fixture hashes, correctness,
  determinism, and handler count.
- `h100/generated/identity/generated_normalized_exp_contract_forward.cu`:
  exact executed generated source.
- `h100/generated/source-natural-forward-stablehlo.mlir.bc`: frozen ordinary
  JAX semantic reference.
- `h100/generated/matched-natural-forward-optimized-hlo.txt`: optimized timed
  JAX comparison.
- `preflight/preflight.json`: compile/link/load audit.
- `preflight/generated/*/*.cu`: generated identity and tanh-softcap sources.
- `analysis.json`: derived distribution statistics.
- `environment.txt`: hardware, toolchain, package, and linkage record.
- `reproduction.txt`: exact bounded commands.
- `SHA256SUMS`: checksums for every artifact file other than itself.
