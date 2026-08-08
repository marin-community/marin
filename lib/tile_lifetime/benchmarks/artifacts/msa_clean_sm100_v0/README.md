# Clean MSA synthesis checkpoint

This artifact freezes the first clean MiniMax Sparse Attention (MSA) synthesis
comparison on one NVIDIA GB200. The generated path starts from BF16 hidden
inputs and uses generic Shuttle `Contract`, maximum `Fold`, `Selection`,
`RelationPlan`, normalized-exponential `Fold`, and output `Contract` machinery.
It does not call the public MSA score, attention, or combine entry points.

The physical score and selected-attention templates retain low-level CuTe
layout, copy, MMA, and pipeline machinery extracted from pinned MSA. Shuttle
owns the semantic score body, causal domain restriction, online state,
relation scheduling, and deterministic state merge. Candidate choice was kept
deliberately small: block size, query tile, partial-state dtype, and one of two
generic row-merge schedules. The rejected BF16x2 merge is preserved because it
regressed rather than being hidden.

## Result

Configuration: `Q=K=16384`, `Hq/Hkv=64/4`, `D=128`, block 128, top-k 16,
causal, BF16 payload, FP32 index projections and accumulation.

| Matched boundary | Shuttle | MSA oracle | Ratio |
|---|---:|---:|---:|
| Score Contract + block-max Fold + top-k | 0.637888 ms | 0.707600 ms | 0.9015x |
| Natural index projections + selection | 0.785760 ms | 0.837360 ms | 0.9384x |
| Natural projection + selection + selected payload | 4.431920 ms | 3.234160 ms | 1.37035x |

The full result does not meet the written `1.20x` completion gate. It is kept
as partial progress because closing the remaining gap with an MSA-specific
combine would weaken the clean-synthesis result. With the exact official
relation, the generic payload is 1.39992x the oracle; 1.83155 ms of its
3.70227-ms median is the generic deterministic merge.

The generated and isolated-oracle selectors produce the identical route hash
`5d669570...` and repeat bitwise. Both differ from a materialized Torch
reference in 61,446 slots across 7,681 rows. Of those rows, 7,680 are early
causal rows with fewer than top-k finite competitors; the remaining row has a
zero cutoff margin. This is recorded under the declared
`real_algebra_equivalent` routing policy. Consequently, the natural-program
output differs from the materialized reference by maximum `0.0536499` and mean
`6.8702e-5`, exceeding the existing `0.01` maximum-error threshold. The
exact-relation payload independently agrees with official MSA to maximum
`0.0009765625` and mean `3.75e-9` and is deterministic.

## Performance anatomy

The generated and official payloads use the same high-level physical strategy:
invert the selected relation into a KV-major work schedule, stage a selected KV
block, run tiled QK, update normalized-exponential state, run PV, and write one
BF16 value partial plus one FP32 log-normalizer per selected block. Both
therefore materialize the same 4-GiB BF16 value-partial buffer and 64-MiB
log-normalizer buffer at this shape.

The measured difference is predominantly finalization. Shuttle's generic
`warp_rows` Fold merge assigns one output row to each warp. Lane zero computes
the 16 normalized-exponential weights serially, then each lane gathers four
feature coordinates across all partials with scalar BF16 loads and stores. It
has no staged shared-memory value pipeline, vectorized 128-bit copies, or
compile-time specialization of the partial axis.

Pinned MSA's combine is a four-stage 8-row by 64-feature tiled kernel. It uses
128-bit asynchronous global-to-shared copies, shared-memory staging for values
and log-normalizers, warp-distributed max/sum reductions, compile-time top-k,
packed output stores, and programmatic dependent launch. The generated merge
measures 1.831552 ms. Subtracting the common physical remainder from the
official payload implies approximately 0.773904 ms for MSA's combine; that
number is an inference, not an independently timed pinned measurement. On the
4.3125-GiB minimum merge traffic, the corresponding lower-bound effective
bandwidths are about 2.53 and 5.98 TB/s. This accounts for essentially all of
the 1.057648-ms exact-relation payload gap.

The next clean improvement is a generic tiled Fold-finalization skeleton with
layout-aware vector loads, shared staging, a symbolic/compile-time partial
axis, pipeline stages, and optional dependency launch. Copying MSA's semantic
combine or adding MSA-specific cases is intentionally out of scope.

## Naive semantic reference

A deliberately direct eager reference was measured to establish the cost of
executing the natural selected-attention algebra without a streaming skeleton.
It excludes routing/index projection/top-k and includes selected K/V gather,
materialized FP32 QK scores, causal masking, materialized FP32 softmax, FP32 PV,
and the BF16 output cast.

At the full 16K shape, one warmup and three repetitions measure
`220.386566`, `220.194427`, and `220.188004` ms, with a 220.194427-ms median
and a 2.708-GiB peak allocated-memory delta. The relation and output hashes
match the preserved materialized semantic reference and repeated outputs are
bitwise identical. This is 59.48 times Shuttle's 3.702272-ms exact-relation
payload and 83.26 times official MSA's 2.644624-ms payload.

The naive run uses the materialized-reference route hash `18886bcb...`, not the
shared generated/oracle selector hash `5d669570...`; the difference is confined
to the underfilled causal rows and cutoff tie described above. The ratios are
therefore scale comparisons at the same shape and selected-count boundary, not
matched acceptance ratios.

The implementation runs four KV groups by 64 query chunks, hence 256 eager
loop bodies. Each chunk gathers 2,048 selected tokens per query and separately
materializes scores, probabilities, and PV work. It has no KV tile residency,
online Fold state, QK-softmax-PV fusion, or producer-consumer pipeline. A fully
vectorized selected implementation would require roughly 144 GiB just for
selected K/V, scores, and probabilities before allocator/workspace overhead;
it is impractical on this 184.3-GiB GB200 and does not fit an 80-GiB H100.

The replacement low-priority pod resolved Torch `2.13.0+cu130`, rather than
the pinned oracle's Torch `2.10.0+cu130`. The naive number is therefore an
order-of-magnitude semantic control, not an acceptance-ratio input. A 1K-query
pilot is also retained and measures 14.032576 ms median.

`invalid-pre-causal-fix.json` records a run that accidentally omitted the
causal `DomainRestriction`; it is invalid evidence and is retained to make the
correction auditable.

## Pins

- MSA: `80434d7f67877c6570ca19cac444b84bc9855dac`
- CUTLASS gitlink: `eb61c911471867a5fd2466bfd8f29306cea6ebf8`
- CUDA/NVCC: `13.0.88`
- Torch: `2.10.0+cu130`
- CUTLASS DSL: `4.4.1`
- QuACK: `0.2.10`
- Driver: `595.71.05`
- GPU: NVIDIA GB200, 1200-W policy, clocks unpinned

The generated and oracle runners execute in separate processes because loading
the direct instantiation and the identical private MSA variant in one process
causes CUDA error 719 in the second runner. The isolated route hashes establish
that both processes used the same seeded fixture and selected relation.

## Files

- `generated-natural-16k.json`: all generated-path raw samples and audits.
- `oracle-natural-16k.json`: isolated oracle raw samples and route diagnostics.
- `generated-route-diagnostic-16k.json`: isolated generated route hash.
- `exact-relation-payload.json`: paired payload correctness and timing.
- `rejected-bf16x2-merge.json`: negative generic merge candidate.
- `invalid-pre-causal-fix.json`: invalidated pre-fix run.
- `oracle_only_natural_16k_capture.py`: exact isolated oracle harness.
- `naive-q16384-k16384.json`: full-shape naive raw samples and hashes.
- `naive-q1024-k16384.json`: smaller naive pilot.
- `naive_selected_attention_benchmark.py`: naive reference harness.
- `naive-gb200-nvidia-smi-q.txt`: naive-run device telemetry.
- `*-command.txt`: reproduction commands.
