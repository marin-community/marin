# Generic tiled Fold finalization on GB200

This artifact records the generic tiled Fold checkpoint for the clean MSA
path. One backend-neutral `TiledFoldFinalizeProgram` and one SM100 emitter are
instantiated unchanged for:

- dense attention partial-state merge: 16 normalized-exponential partials,
  FP32 maximum/denominator state, and a BF16 value accumulator;
- indexed deterministic weighted merge: six non-prefix-valid indexed partials
  and explicit source-ordered FP32 multiply-then-add.

The physical skeleton uses 128-bit global-to-shared copies, four-stage shared
staging, warp-distributed scalar-state reduction, vectorized feature work, and
deterministic stores. The semantic contribution, state update, and finalizer
are emitted from scalar ASTs; there is no MSA or MoE dispatch in the generator.

## Primary result

Configuration: `Q=K=16384`, `Hq/Hkv=64/4`, `D=128`, block 128, top-k 16,
causal, BF16 payload, and FP32 projection and Fold accumulation on one GB200.
The matched natural boundary starts from BF16 hidden inputs and includes index
projection, selection, relation scheduling, sparse QK/normalized-exp/PV, and
deterministic Fold finalization. Main QKV and output projections are excluded
on both sides.

Two isolated captures preserve 40 raw samples per implementation:

| Implementation | 10-run median | 30-run median | Pooled 40-run median |
| --- | ---: | ---: | ---: |
| Generated Shuttle | 3.809584 ms | 3.831632 ms | 3.823488 ms |
| Pinned MSA oracle | 3.187072 ms | 3.192960 ms | 3.191376 ms |

The pooled-median ratio is `1.198069x`. Shuttle therefore improves the prior
4.431920-ms generated path by 13.7%, meets the explicit `<3.88 ms` objective,
and is just inside the clean-synthesis `1.20x` gate. The 30-run medians alone
give `1.200025x`, so the result should be described as lying at the gate, not
comfortably below it.

The selector and oracle must run in isolated processes. Loading both physical
implementations together triggers CUDA error 719. Running the generated
compiler immediately after the oracle in the same shared cache also exposed a
CUTLASS DSL dynamic-Boolean compilation failure. The two fixture hashes and
reference-route hash establish that the isolated captures use identical
inputs. These failures are process/cache integration defects, not silently
discarded performance samples.

## Selection and numerical contract

Selection now exposes descending score order, lower-right-index tie breaking,
validity, and underfilled capacity in compiler IR. Causally invalid output
slots are set to `-1` and excluded from `RelationPlan`; the earlier 61,446
reference mismatches fall to six entries in one row. That row has an exactly
zero top-k cutoff margin, so the generated physical top-k deterministically
selects a different tied block from the source reference.

Repeated routes and outputs are bitwise stable. The remaining tie changes the
selected set, producing maximum/mean output error `0.0536499`/`6.8702e-5`.
This exceeds the frozen `0.01` maximum-error threshold despite the small mean.
The performance result therefore uses the explicit `real_algebra_equivalent`
selection policy. Exact source-order tie behavior remains open and must not be
reported as passing.

## Reuse and mutation checks

The indexed non-attention instantiation uses 4,096 output rows, 8,192 source
rows, six route slots, feature size 128, and arbitrary non-prefix validity.
It matches a source-ordered reference exactly, repeats bitwise, and has a
0.018128-ms median over ten samples. The partial count deliberately exercises
an incomplete four-stage pipeline tail.

A smaller dense-attention binding (`Q=128`, `K=512`, `Hq/Hkv=8/2`, `D=128`,
four partials) is finite and deterministic, with maximum/mean error
`0.00134033`/`0.000149893` against the semantic reference.

## Fold schedule ablation

At Shuttle revision `5dd21768c15501fe5cd10966b4c3737789bbd0d7`, one
low-priority GB200 run compared exactly three generic normalized-exp Fold
schedules at 4,096 rows, 16 partials, and 128 features. Each candidate has 100
raw samples and produces the same deterministic output hash.

| Feature tile x shared buffers | Median | Change from 64x1 |
| --- | ---: | ---: |
| 64x1, no overlap | 0.015728 ms | baseline |
| 64x2, ping-pong | 0.014896 ms | -5.29% |
| 128x2, ping-pong | 0.013824 ms | -12.11% |

One full generated-only MSA confirmation used the winning 128x2 schedule at
the primary 16K shape. Its natural projection/selection/payload median was
3.310736 ms, 13.59% below the preserved 64x2 result of 3.831632 ms. The
selection-only median changed from 0.766000 to 0.773888 ms and the projected
selection core changed from 0.637680 to 0.635952 ms, so the full-path gain is
localized to the Fold/finalization path rather than selection. No 64x1 full
candidate was added.

The confirmation remains under the artifact's `real_algebra_equivalent`
selection policy. It is deterministic, but six selected slots differ from the
materialized reference and the maximum payload error remains 0.0536499; it is
not a source-order numerical acceptance result.

Commands:

```text
TORCH_CUDA_ARCH_LIST=10.0a uv run --frozen --package marin-tile-lifetime \
  python lib/tile_lifetime/backends/sm100/smoke_tiled_fold_finalize.py \
  --semantics normalized_exp --rows 4096 --partials 16 --features 128 \
  --warmups 20 --repeats 100 \
  --build-directory /tmp/shuttle-fold-ablation-build \
  --output /tmp/shuttle-fold-ablation-gb200.json

TORCH_CUDA_ARCH_LIST=10.0a python \
  lib/tile_lifetime/benchmarks/sm100_fused_projected_selection.py \
  --msa-root /tmp/shuttle-msa \
  --build-directory /tmp/shuttle-fold-full-128x2-build-441 \
  --query-count 16384 --right-count 16384 \
  --query-heads 64 --key-value-heads 4 \
  --warmups 3 --repeats 10 --execution-mode generated_only \
  --output /tmp/shuttle-fold-full-128x2-gb200.json
```

The full confirmation used MSA `80434d7f67877c6570ca19cac444b84bc9855dac`,
CUTLASS `eb61c911471867a5fd2466bfd8f29306cea6ebf8`, CUTLASS DSL 4.4.1,
QuACK 0.2.10, Torch 2.10.0+cu130, CUDA 13.0.88, and driver 595.71.05.
The first compile attempt accidentally used CUTLASS DSL 4.6.2 and failed on an
`nvvm.fmax` signature incompatibility. It produced no timing sample; the one
documented rerun restored the pinned 4.4.1 environment.

## Performance interpretation

The old generic scalar merge measured 1.831552 ms. The bounded direct
combine-only experiment now measures the winning 128x2 Fold at 0.013824 ms,
and the full natural path improves by about 0.52 ms while selection changes by
only a few thousandths of a millisecond. Both measurements attribute the gain
to Fold finalization rather than selection.

Pinned MSA still has a tighter workload-integrated implementation: its KV-major
pipeline, partial layout, dependent launch, and combine were designed together.
Shuttle intentionally keeps relation scheduling, attention tile work, and Fold
finalization as generic compiler-owned components. Comparing the one-sided
128x2 confirmation with the preserved oracle median suggests a 3.7% full-path
gap, but it is not a fresh counterbalanced acceptance capture. The stricter
accepted comparison remains the pooled 1.198069x checkpoint above.

For scale, the preserved naive selected-attention reference takes
220.194427 ms at the same 16K shape. It launches 256 eager group/chunk bodies,
gathers selected K/V, and materializes FP32 QK scores and probabilities. It has
no resident K/V reuse, online Fold state, QK-softmax-PV fusion, or asynchronous
pipeline. On its exact-relation payload boundary it is 59.48x slower than the
prior generated streaming payload and 83.26x slower than MSA.

## Pins and files

- Shuttle checkpoint: `932db2193f30d38b391f839945918d06c8727c3e`
- MSA: `80434d7f67877c6570ca19cac444b84bc9855dac`
- CUTLASS gitlink: `eb61c911471867a5fd2466bfd8f29306cea6ebf8`
- Hardware and software details: `device.txt`
- Generated captures: `shuttle-fold-primary-generated*.json`
- Isolated oracle captures: `shuttle-fold-primary-oracle*.json`
- Non-attention reuse: `shuttle-fold-indexed.json`
- Attention binding: `shuttle-fold-attention.json`
- Schedule smoke: `fold-schedule-smoke-gb200.json`
- Three-candidate schedule ablation: `fold-schedule-ablation-gb200.json`
- 128x2 full confirmation: `shuttle-fold-full-128x2-gb200.json`
