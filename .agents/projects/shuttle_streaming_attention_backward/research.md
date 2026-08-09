# Background Research Brief

- Effort: medium
- Stop rule: stop when source inspection and the existing negative experiment identify one bounded, clean schedule change with a measurable falsifier
- Date: 2026-08-09

## Question

What is the smallest reusable physical-schedule improvement for Shuttle's
generic streamed normalized-exp backward program that preserves deterministic
Fold semantics and does not call or import an opaque attention kernel?

## Current Marin Context

The generated GB200 component measures 0.864582 ms at sequence length 2,048,
versus 0.148534 ms for matched Torch SDPA. The physical implementation uses a
query-major dQ traversal and a K/V-major dK/dV traversal. Both are generated
from visible Contract, Fold, Map, and DomainRestriction semantics.

Before this change, dK/dV already packed the four query heads mapped to each
K/V head. The dQ traversal launched one program per query head and loaded the
same K/V tile independently for all four mapped heads.

The completed H100 control measures 1.297498 ms for scalar-head dQ and
0.584992 ms for packed dQ under the same 32-by-32, eight-warps, three-stage
configuration. The packed result is 1.258x matched SDPA and remains outside the
1.20 performance gate.

## Internal Prior Work

The reverted query-partition experiment is the key negative result. Four
partitions improved generated latency by only 1.17%, from 0.864582 ms to
0.854435 ms, while adding 64 MiB of FP32 partial gradients, four times as many
dK/dV tasks, and 512 finalizers. This falsified lack of task parallelism as the
primary bottleneck.

`StreamingAttentionBackwardTileSchedule` already derives the GQA grouping
factor from the QK Contract's input index map. Reusing the same relation in the
dQ traversal requires no new semantic primitive or workload-name dispatch.

## External Prior Art

The official FlashAttention Hopper backward implementation uses K/V-major work
tiles, retains FP32 dK/dV accumulators, stages Q and dO through asynchronous
pipelines, and computes the reverse contraction family inside a pipelined tile
loop. These are useful physical-schedule references. Its imported attention
mask and softmax helpers are not acceptable Shuttle dependencies and are not
used by this change.

The FlashAttention documentation also records that deterministic backward is
slower and uses more memory. This supports treating deterministic accumulation
order as a first-class schedule constraint rather than hiding atomics or an
unordered reduction behind the backend.

## Negative / Failed Leads

- Query partitioning increased parallelism but did not materially improve
  latency; it also added large partial storage.
- Materializing one dQ partial per K/V tile would require roughly 2 GiB of FP32
  partials for the primary shape.
- Fusing both reverse traversals in a single K/V-major task avoids duplicated
  score algebra, but deterministic dQ ownership then requires ordered global
  accumulation or a bounded wavefront design. That is not the smallest safe
  change.
- Tile-size search alone does not answer the compiler-design question and risks
  replacing a generic schedule improvement with shape tuning.

## Evidence Map

### Claim: GQA row packing is a clean reusable dQ transformation

- Support:
  - Marin schedule IR: the QK Contract index map already identifies every query head mapped to one K/V head.
  - Marin dK/dV emitter: the same packed-row representation is already executable for the reverse K/V Fold.
- Contradictions:
  - Packed dQ uses 114,688 bytes of shared memory, versus 45,568 bytes for scalar-head dQ; register and spill counts were unavailable.
- Directness to Marin: exact primary shape and current emitter.
- Confidence: replicated on one H100 with 30 counterbalanced samples per variant.
- Action: retain packing and profile the residual 1.258x gap before adding another candidate.

### Claim: the dominant residual gap is an expert physical pipeline, not task count

- Support:
  - Query-partition negative result: 4x task partitioning changed latency by only 1.17%.
  - Official FlashAttention source: backward uses staged asynchronous movement and persistent accumulator ownership.
- Contradictions:
  - No profiler counters have yet decomposed the 0.864582 ms result.
- Directness to Marin: one direct negative experiment plus source-level schedule comparison.
- Confidence: exploratory.
- Action: after dQ packing, profile contraction utilization, register spills, and memory traffic before adding another schedule dimension.

## Recommended Next Experiments

### 1. Pack mapped query heads in the dQ traversal

- Minimum experiment: run the existing S=2,048, 32x32 benchmark with the packed dQ emitter and the same numerical thresholds.
- Baseline/control: scalar-head parent commit on the same H100, 1.297498 ms.
- Expected signal: at least 10% generated latency improvement, with unchanged determinism and error thresholds.
- Result: generated latency fell 54.91%, from 1.297498 ms to 0.584992 ms on H100; correctness and the deterministic hash were unchanged.
- Cost/risk: one H100 schedule ablation; no tuning sweep.
- Sources: current Marin schedule/emitter and official FlashAttention physical schedule.

### 2. Profile one packed run before designing a fused reverse pipeline

- Minimum experiment: collect kernel-level duration, achieved tensor-core utilization, memory throughput, registers, and spills for dQ and dK/dV.
- Baseline/control: Torch SDPA under the same boundary.
- Expected signal: identify whether the next gap is duplicated reverse algebra, lack of asynchronous movement, or resource pressure.
- Falsifier: counters do not isolate a dominant component; then instrument per-kernel timings before changing the schedule.
- Cost/risk: one profiler run.
- Sources: query-partition negative result and official Hopper schedule.

### 3. Only then consider a deterministic fused reverse wavefront

- Minimum experiment: model memory and ordering for one bounded wavefront that shares recomputed score state across dQ/dK/dV.
- Baseline/control: packed two-traversal implementation.
- Expected signal: reduce reverse Contract invocations from seven to five per logical tile pair without sequence-squared storage.
- Falsifier: deterministic ownership requires unbounded partials or serializes below the existing two-traversal latency.
- Cost/risk: medium/high; likely needs a CuTe/TMA/WGMMA skeleton rather than a local Triton edit.
- Sources: official FlashAttention Hopper backward schedule.

## Hypothesis Queue Update

- Add: profile the residual packed reverse pipeline on H100.
- Revise: move a fused reverse wavefront behind a profiler gate.
- Falsify / stop: query-domain partitioning as the primary fix.
- Promote: mapped-head packing and explicit Contract index-map reuse across both reverse orientations.

## Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Shuttle logbook | logbook | `.agents/logbooks/tile_lifetime_compiler.md` | measured partition negative result | high | exact recorded timings and storage cost |
| Shuttle backward IR | Marin code | `lib/tile_lifetime/src/tile_lifetime/streaming_attention_backward.py` | GQA relation derives from Contract index map | high | current branch source |
| Shuttle backward emitter | Marin code | `lib/tile_lifetime/benchmarks/h100_generated_streaming_attention_backward.py` | duplicate scalar-head K/V loads in dQ | high | current branch source |
| Packed H100 artifact | benchmark | `lib/tile_lifetime/benchmarks/artifacts/streaming_attention_backward_packed_h100/` | same-H100 schedule ablation, correctness, determinism, and resource metadata | high | 30 counterbalanced samples per path |
| FlashAttention | external code | `https://github.com/Dao-AILab/flash-attention/tree/a369df707e1980fb328abcc1733e3457ec10155f/hopper` | Hopper backward pipeline and accumulator ownership | high | physical reference only |
| FlashAttention README | external code | `https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/README.md` | deterministic backward cost caveat | medium | documentation-level statement |
| FlashAttention-3 | paper | `https://arxiv.org/abs/2407.08608` | Hopper asynchronous pipeline context | high | forward emphasis; schedule context only |

## Handoff

- Suggested issue `Prior work` block: The official Hopper backward schedule confirms that persistent operand ownership and pipelined movement matter, while the reverted partition experiment rules out task count as the primary current bottleneck. Contract-index-map packing reduced generated H100 latency by 54.91%; the remaining 1.258x SDPA gap now needs component profiling.
- Suggested logbook entry: Packed dQ reduced physical reverse Contract invocations by 56.25% and measured latency by 54.91% on H100 without changing the deterministic output hash. The 0.584992 ms result remains 1.258x matched SDPA.
- Open questions: Does the 128-row dQ tile spill? How much of the residual gap is duplicated score algebra versus missing TMA/WGMMA overlap?
- Stop reason: the bounded H100 ablation completed; broader search requires profiler evidence.
