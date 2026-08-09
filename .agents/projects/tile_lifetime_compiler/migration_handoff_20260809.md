# Shuttle migration handoff

## Canonical branch

`research/shuttle-clean-helper-boundaries`

The branch is pushed to `origin`. All GPU allocations used in this session were
explicitly released and verified inactive. Do not infer GB200 evidence from the
available secondary Blackwell cluster: it provides B200 only.

## Last accepted Grug training-region result

Commit `17bf026034` preserves the accepted ten-call physical-H100 replay from
ordinary JAX/Grug after JAX-owned AD.

- Ten exact generated targets occur once in transformed HLO.
- Every handler executed 35 times.
- Generated output is bitwise stable across 30 repetitions.
- Ordered-FP maximum absolute error is `9.760261e-7` over 53 output leaves.
- XLA median: `0.585042 ms`.
- Shuttle median: `0.689586 ms`.
- Ratio: `1.178695x`, inside the `1.20x` target.

Artifact:
`lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_992a7467_v0/`

The ten generated calls own routed forward, two input-adjoint Contracts, a
shared Contract with generated forward/reverse scalar Maps, a deterministic
source Fold, two expert-weight Contracts, streaming-attention reverse, and two
row Folds.

## Weighted RelationProgram reverse

Commits `1eeeea1091` and `2732ef51a9` add and integrate a generic weighted
RelationProgram reverse:

```text
rank-two Contract
-> scalar edge Map
-> hidden Fold
-> deterministic source-slot Fold
```

The generator is recovered structurally from post-SPMD HLO. It does not dispatch
on MoE or Grug names. Scalar-Map mutation regenerates the same generic physical
family. The placement all-reduce and normalized router VJP remain explicit XLA
boundaries.

The first twelve-call H100 replay is correct and generated-output deterministic,
but unaccepted:

- XLA median: `0.527480 ms`.
- Shuttle median: `0.658288 ms`.
- Ratio: `1.247988x`.
- XLA produced two hashes; only
  `[0].params.blocks[0].mlp_gated_norm.w_down` varied, in the final sample.
- Shuttle produced one hash across all 30 repetitions.

Artifact:
`lib/tile_lifetime/benchmarks/artifacts/xla_grug_shared_map_h100_unaccepted_2732ef51_v0/`

Before this region, ten calls owned 84.1% of physical pre-scheduler dot FLOPs in
the captured padded fixture. Owning the 4,194,304-FLOP weighted payload Contract
raises static dot ownership to approximately 91.8%. This percentage describes
the captured HLO, not a production-shape throughput estimate.

## Current stopping point: demand-driven Contract domain

Commit `1ee45b825d` completes a clean CPU-verified checkpoint for generic
consumer-domain narrowing. The weighted reverse previously recomputed a
`[512,32]` payload although the nested Fold consumes only 16 logical
relation-edge rows. The compiler now traces a single-user contiguous row
slice/view relation backward from the consumer, moves the Contract custom-call
site to the demanded `[16,32]` value, and records an explicit LHS row offset.

The implementation:

- contains no instruction-name, MoE, or Grug dispatch;
- rejects noncontiguous slices and competing consumers;
- preserves the full LHS/RHS operand ABI while narrowing the produced M domain;
- handles nonzero row offsets;
- verifies that the original full Contract and slice become dead;
- keeps Contract and Fold as two generated custom calls.

For the captured shape, the narrowed Contract reduces work from 4,194,304 to
131,072 FLOPs and estimated ideal BF16 traffic from 172,032 to 13,312 bytes.
These are static estimates, not a GPU performance claim. Canonical verification
after integration passed 26 focused tests, Pyrefly, scoped pre-commit, and
`git diff --check`. No GPU replay was launched before migration.

The first restart action is a single bounded H100 replay of the unchanged
twelve-call natural Grug harness at commit `1ee45b825d` or its descendant. Do
not tune before measuring. If narrowing is not enough to recover the `1.20x`
target, the next bounded candidate is a generic composed
Contract-plus-nested-Fold handler. Do not replace it with a workload-specific
MoE reverse kernel.

## RMSNorm and LayerNorm backward

The generic JAX-owned VJP path is correct and deterministic on H100 but remains
slow:

- generated RMS reverse: approximately `0.1041 ms`;
- matched XLA: approximately `0.0705 ms`;
- ratio: approximately `1.477x` in the latest replay.

The three-stage generated pipeline is:

```text
K0 row sum-square -> FP32 inverse scratch
K1 row correlation + BF16 input cotangent
K2 column feature-scale cotangent Fold
```

The checked-in component profile is explicitly non-attributable: its isolated
programs change input algebra, dtypes, output dtypes, and scratch interfaces.
Do not use their timings to choose an optimization.

Artifact:
`lib/tile_lifetime/benchmarks/artifacts/jax_row_normalization_backward_h100_components_non_attributable_fdd838/`

An exact profiler-delimited full-call capture executed successfully, but Nsight
postprocessing failed before the raw report was transferred. The next attempt
must:

1. capture one unchanged full typed-FFI handler execution;
2. preserve/upload the raw `.nsys-rep` before any optional postprocessing;
3. run `nsys stats --force-export=true` for GPU trace, kernel summary, and CUDA
   API summary;
4. request one CPU and one H100 with zero retries;
5. make no optimization decision unless exact K0/K1/K2 timings are recovered.

Likely generic candidates after exact attribution are same-domain row-Fold
coalescing, BF16x2 loads with warp reductions, a better column-Fold geometry, or
emitting deterministic column partials from the final Map followed by a small
Fold. The centered LayerNorm mutation already demonstrates that the semantic
algebra generalizes by adding mean and backward-centering Folds.

## Remaining arithmetic ownership order

After the weighted relation reverse, continue in this order unless profiling
changes it:

1. normalized router-weight VJP and router Contract around the retained
   placement collective;
2. output-head Contract plus normalized-exponential loss Fold and their VJPs;
3. repeated GatedNorm/RMS training regions;
4. remaining attention forward/rematerialization and projection Contracts;
5. DenseMLP Contracts through the same generic Contract/Map machinery.

Keep JAX responsible for AD and the natural frontend. Keep the default compiler
Torch-free. Library/runtime primitives may implement generic contractions and
transport, but accepted execution must not call opaque workload-semantic kernels.

## Collectives and EventTensor

Current collectives remain JAX/XLA-owned. EventTensor already has:

- GPU attention attachment evidence;
- grouped-Contract synchronization ABI evidence on a physical GB200;
- a two-H100 JAX collective proof.

After arithmetic ownership is stable, connect tile-visible Contract completion
to a coarsened collective-launch event while keeping transport generic. Do not
confuse B200 portability checks with actual GB200 acceptance evidence.

## Verification commands

Focused weighted-reverse suite:

```bash
uv run pytest -q --tb=short \
  lib/tile_lifetime/tests/test_xla_shared_contract_multimap.py \
  lib/tile_lifetime/tests/test_xla_weighted_relation_reverse_ffi.py \
  lib/tile_lifetime/tests/test_xla_routed_remainder_ffi.py
```

Required source checks before publishing:

```bash
uv run pyrefly check <changed-python-files>
./infra/pre-commit.py <changed-files>
git diff --check
```

The migration checkpoint was verified with:

```text
26 focused tests passed
Pyrefly: 0 errors
scoped pre-commit: passed
```

## Hardware policy

- H100 results require a physically verified H100.
- B200 results are portability evidence only unless the target explicitly names
  B200.
- GB200 claims require a physically verified GB200.
- Request one CPU for bounded GPU work; record if the scheduler normalizes it to
  a larger platform minimum.
- Preserve raw repeated-run distributions, exact revisions, clocks/power,
  output hashes, generated sources, and checksums.
