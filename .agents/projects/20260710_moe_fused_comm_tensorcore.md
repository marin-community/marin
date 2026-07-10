# Fused Communication and Tensor-Core MoE

Logbook: `.agents/logbooks/6597-moe-fused-comm-tensorcore.md`

## TL;DR

Build the Hopper expert-parallel MoE MLP around persistent producer/consumer
kernels that overlap communication-heavy routing with tensor-core-heavy expert
matmuls.

The semantic source-push plan defines slot-free work items and their partial
order. The lowering groups those work items into large rectangular sends and
WGMMA-compatible compute tiles. Queue slots, semaphores, CTA roles, and pipeline
depth are implementation details below the semantic plan.

The primary deliverables are:

1. fused source-push permute + W13;
2. fused W2 + source return/combine;
3. fused source dcombine/dy-route + W2 backward;
4. fused W13 backward + source dX return/combine.

A split `pack; barrier; matmul` or `route; barrier; matmul` implementation is a
correctness baseline, not completion of this project. The target remains 250
useful TFLOP/s/rank for the aggregate forward and backward MLP at the target
shape.

## Motivation

The current semantic implementation establishes correct metadata, JAX
references, expert-major layouts, and isolated Pallas/WGMMA kernels. Its honest
custom-VJP boundary takes `63.038497 ms`, or `122.702286` useful TFLOP/s/rank,
on the target shape. The manual split graph takes `51.465903 ms`, or
`150.293052` useful TFLOP/s/rank.

The split stages are already too slow even under optimistic host-level overlap.
Pairing their isolated medians gives a floor of about `35.85 ms`, while 250
useful TFLOP/s/rank requires about `30.94 ms`. This is not a ceiling on a fused
design. It shows that separately materializing and synchronizing communication
stages cannot reach the target.

The stable source-push inbox experiment reached `216.949` TFLOP/s/rank for its
fused W13 slice. That result is the implementation starting point. This project
adapts that producer/consumer structure to the JAX-native semantic routing
contract and extends the same structure to the communication-heavy backward
phases.

## Goal

For every communication-heavy MoE phase adjacent to an expert matmul, arrange
work so that:

- producer CTAs move large source-owned chunks into destination-visible staging;
- consumer warpgroups operate on WGMMA-compatible subtiles of ready chunks;
- producers refill released slots while consumers compute older slots;
- no whole-stage barrier separates communication from tensor-core work;
- no dense replicated route tensor or all-gather is introduced at the MLP
  boundary;
- expert-major outputs are written directly in the layout consumed by the next
  phase or saved by the custom VJP.

Communication is less shape-sensitive than WGMMA, provided each send is large,
aligned, and mostly live. Compute shape is the hard constraint. The lowering
must therefore choose compute tiles first and pack one or more compute tiles
into each send chunk.

Use:

```text
send_m = compute_m * compute_blocks_per_send
```

`compute_m` must be a legal WGMMA row tile. `send_m` should be large enough to
amortize queue, semaphore, and peer-memory overhead. A send chunk may cover an
expert group, but each compute work item must resolve to one local expert and a
contiguous expert-major row interval.

The semantic plan deliberately does not require send and compute row blocks to
have the same size. Transport owns `send_m`; tensor-core work owns `compute_m`.
The lowering only requires `send_m % compute_m == 0` and a stable mapping from
each send chunk to its contiguous expert-local compute blocks. The initial
Hopper profile uses B256 sends feeding four B64 compute blocks, but that ratio
is a physical tuning choice rather than part of route semantics.

## Two Physical Templates

The old source-push inbox kernel established a useful physical baseline: two
whole-chunk transfer owners, a fixed 32-program peer-local worker grid, 12
rolling slots, one publication per ready unit, fixed consumer completion
fan-in, and one release. The semantic design replaces its host/Python queue
construction and queue-ordered output layout; it does not replace this proven
producer/consumer discipline without measured evidence.

The four fused stages should be implemented as two reusable physical shapes:

1. **send + compute**: forward permute/W13 and backward dcombine/dy-route/W2.
   Source producers send semantic expert-grouped chunks; destination consumers
   run WGMMA on `compute_m` subtiles. W2 backward additionally accumulates dW2
   as an expert-local reduction side output.
2. **compute + return/combine**: forward W2 and backward W13. Destination
   consumers run WGMMA and return route-local tiles to source-owned storage;
   source consumers combine top-k routes. W13 backward additionally accumulates
   dW13 as an expert-local reduction side output.

Do not maintain unrelated queue/semaphore protocols for the four stages. Any
stage-specific divergence from these templates needs a concrete data dependency
or measured performance justification.

## Non-Goals

- Do not optimize split x-pack, source-expand, or return kernels as though they
  were the production architecture.
- Do not count host-side concurrent launches as the required fusion.
- Do not add broad mode matrices to production APIs.
- Do not require every isolated matmul to reach 250 TFLOP/s. The target is the
  aggregate forward and backward MLP.
- Do not expose physical queue slots in semantic routing metadata.
- Do not resume destination-pull ring or `lax.switch` experiments without new
  evidence that they solve a required lowering problem.

Split implementations remain useful for correctness, copy-only/compute-only
tax measurements, and fallback comparisons.

## Target Shape

Per rank:

```text
tokens:                 32768
top-k:                  4
assignments:            131072
EP ranks:               8
global experts:         256
local experts:          32
hidden dimension:       2560
intermediate dimension: 1280
input/weight dtype:     bf16
accumulator dtype:      fp32
```

Roughly balanced routing is the production benchmark. Exact balanced routing is
only a diagnostic.

## Semantic Routing Contract

The existing JAX-native semantic plan is the canonical description of route
identity and expert-major placement:

```text
assignment_id[s, d, pair_row] = token * topk + route_slot
valid[s, d, pair_row]
xcounts[s, d, e]
pair_expert_base[s, d, e]
expert_base[d, e]
src_base_by_expert[d, s, e]
```

Rows within each `(source, destination)` pair are grouped by local expert. For a
live expert-local row:

```text
pair_row = pair_expert_base[s, d, e] + local_row
expert_row = expert_base[d, e] + src_base_by_expert[d, s, e] + local_row
token = assignment_id[s, d, pair_row] // topk
route_slot = assignment_id[s, d, pair_row] % topk
```

This contract does three things for fused kernels:

1. gives producers source token identities in expert-grouped order;
2. gives consumers contiguous destination expert-major output rows;
3. gives return paths the source token and route slot without constructing an
   inverse `argwhere` tensor.

Metadata construction stays inside the JIT/custom-VJP boundary. Capacity is
static, but useful counts, overflow accounting, queue work, and reverse routes
are data-dependent JAX values.

## Semantic Work Items

The semantic DAG does not contain slots or semaphores. It contains data
movement and compute operations over logical chunks.

### Forward Permute + W13

```text
FSend(s, d, e_group, send_chunk, k_copy_tile)
FW13(d, s, e, compute_chunk, n_tile)
```

`FSend` gathers source tokens from semantic pair rows and copies a large
`[send_m, hidden_dim]` chunk into destination-visible staging. Physical copy
tiles may subdivide the hidden dimension.

`FW13` consumes one `[compute_m, hidden_dim]` subtile, loops over WGMMA K tiles,
and writes gate/up preactivation directly to destination expert-major rows.

Semantic dependency:

```text
all FSend(..., k_copy_tile) for a send chunk
  < every FW13(...) whose compute rows are inside that send chunk
```

There is no dependency between compute from chunk `q` and sends for chunk
`q+1`.

### Forward W2 + Return/Combine

```text
FW2(d, e, compute_chunk, n_tile)
FReturn(d, s, e, compute_chunk, n_tile)
FCombine(s, token_block, n_tile)
```

`FW2` reads saved gate/up preactivation, computes SwiGLU in the load/compute
prologue, applies route weights at the chosen algebraically equivalent point,
and runs W2 WGMMA.

`FReturn` writes route outputs to source-owned route storage using
`assignment_id`. `FCombine` accumulates top-k route slots in fp32 and stores the
source-sharded bf16 output.

Semantic dependencies:

```text
FW2(route, n_tile) < FReturn(route, n_tile)
all FReturn(token, route_slot, n_tile) < FCombine(token, n_tile)
```

The preferred lowering streams W2 output tiles directly into return buffers and
allows source combine consumers to process ready token blocks. It must not
materialize `partial_y[S,T,H]` or perform a dense `psum`.

### Backward DCombine/DY Route + W2

```text
BDCombine(s, token_block, route_slot, n_tile)
BYSend(s, d, e_group, send_chunk, n_copy_tile)
BW2dH(d, s, e, compute_chunk, i_tile)
BW2dW(d, s, e, compute_chunk, i_tile, n_tile)
```

The source producer reads source-sharded `dy[token, hidden]` and returned route
outputs. It computes:

```text
d_route_weight = dot(dy, y_route)
dy_route = dy * route_weight
```

It then sends large expert-grouped `dy_route` chunks to destination-visible
staging. The destination consumer uses ready WGMMA-compatible subtiles to
compute:

```text
dA_weighted = dy_route @ W2.T
dW2 += A_weighted.T @ dy_route
dA = route_weight * dA_weighted
dZ13 = d_swiglu(dA, Z13)
```

Depending on the selected algebra, route weighting may move between the source
producer and destination epilogue. The implementation must preserve both
`d_route_weight` and weight-gradient semantics.

Semantic dependencies:

```text
BDCombine(route) < BYSend(route)
all BYSend(..., n_copy_tile) for a send chunk
  < BW2dH/BW2dW for compute rows inside that send chunk
all BW2dW partials for an expert < final dW2 reduction/store
```

This fused stage must consume source-sharded `dy` directly. A replicated `dy`,
`all_gather(dy)`, or separately materialized full expert-major `dy_route` is not
an acceptable production boundary.

### Backward W13 + DX Return/Combine

```text
BXSend(s, d, e_group, send_chunk, k_copy_tile)
BW13dX(d, s, e, compute_chunk, hidden_tile)
BW13dW(d, s, e, compute_chunk, hidden_tile, n_tile)
BXReturn(d, s, e, compute_chunk, hidden_tile)
BXCombine(s, token_block, hidden_tile)
```

The source producer rematerializes expert-grouped x chunks using the same
semantic assignment rows used by forward. Consumers compute:

```text
dW13 += x.T @ dZ13
dX_route = dZ13 @ W13.T
```

`dX_route` is returned to source route slots and reduced across top-k into
source-owned `dX[token, hidden]`.

Semantic dependencies:

```text
all BXSend(..., k_copy_tile) for a send chunk < BW13dW for that chunk
BW13dX(route, hidden_tile) < BXReturn(route, hidden_tile)
all BXReturn(token, route_slot, hidden_tile) < BXCombine(token, hidden_tile)
all BW13dW partials for an expert < final dW13 reduction/store
```

## Physical Queue Lowering

Slots implement bounded storage for the semantic DAG. A slot has a generation
and transitions through:

```text
released -> filling -> ready -> consuming -> released
```

For generation `g` of slot `q`:

```text
Release(q, g - 1)
  < every CopySubtile(q, g)
  < PublishReady(q, g)
  < every ComputeTile(q, g)
  < Release(q, g)
```

Required semaphore properties:

- A producer cannot overwrite a slot until the previous generation is
  released.
- Ready is published only after every physical copy subtile is globally visible
  to the destination consumer.
- A consumer cannot read a slot before observing the matching ready generation.
- Release occurs only after every consumer using that generation has finished.
- Generation values prevent an old ready signal from satisfying a reused slot.
- Different slots and generations have no ordering unless the semantic DAG
  requires it.

The implementation should use separate producer and consumer CTA sets in one
persistent kernel. A producer that executes a blocking copy and only then calls
compute in the same sequential CTA is not overlap.

Remote peer-id GMEM access requires Lane lowering on the current Hopper stack.
Local WGMMA compute may require a Warpgroup-compatible lowering path. The
implementation may use separate kernel roles or a custom Mosaic/CUDA lowering,
but the resulting runtime must keep producer copies and consumer WGMMA in
flight concurrently.

## Chunk and Tile Shape

Choose tensor-core tiles first:

```text
compute_m: legal WGMMA row tile, initially 64 or 128
block_n:   legal WGMMA output tile, initially 64 or 128
block_k:   legal WGMMA reduction tile, initially 64, 128, or 256
```

Then choose communication aggregation:

```text
send_m = compute_m * {1, 2, 4}
expert_group_size in {1, 2, 4, 8}
send bytes should normally be multiple MiB, not a single compute tile fragment
```

An expert group is a scheduling and aggregation unit. It does not permit a
single WGMMA tile to cross expert boundaries. The send header identifies the
expert-local intervals contained in the chunk, and consumers dispatch separate
compute work items for those intervals.

Prefer an expert-group outer, rotating-peer inner traversal initially:

```text
for expert_group:
  for peer_phase:
    peer = (rank + peer_phase) % ep_size
    enqueue live chunks for (expert_group, peer)
```

Keep peer-major/expert-group-inner as a diagnostic queue-order comparison, not
an ordinary production knob.

## Production Configuration

Keep only parameters that change legal tile shape or real pipeline occupancy:

```text
send_m
compute_m
block_n
block_k
expert_group_size
pipeline_depth
num_producer_ctas
num_consumer_ctas
```

Use one named Hopper target profile. Diagnostic binaries may expose queue order,
copy-only, compute-only, semaphore-only, and debug output modes, but those modes
must not branch through the production hot path or change its return type.

## Implementation Sequence

### Milestone 1: Fused Permute + W13

Adapt the proven source-push inbox producer/consumer kernel to consume the
semantic plan directly.

Required outcome:

- source token rows are packed in producer CTAs;
- W13 consumer warpgroups operate on ready slots concurrently;
- output is compact expert-major gate/up preactivation;
- no full x-pack tensor is an input or output of the production call;
- rough-balanced target correctness matches the semantic JAX reference;
- full fused time beats `x_pack + W13`, not merely either isolated stage.

### Milestone 2: Fused DCombine/DY Route + W2 Backward

Reuse the producer/consumer queue mechanics with source-sharded `dy` producers
and W2 backward consumers.

Required outcome:

- no `dy` all-gather;
- no standalone source-expand in the production path;
- `d_route_weight`, `dW2`, and `dZ13` match the semantic JAX reference;
- full fused time beats `source-expand + W2 backward` and materially reduces
  the current `11.572594 ms` API-boundary tax.

### Milestone 3: Stream W2 Return/Combine

Write W2 outputs directly into source-owned route slots and overlap source
combine with destination W2 work. Remove dense partial output buffers and
destination-side atomics from the production path.

### Milestone 4: Fuse W13 Backward With DX Return

Rematerialize x through producer slots, compute `dW13`/`dX_route`, and stream
`dX_route` back to source combine consumers.

### Milestone 5: Integrate the MLP Custom VJP

Replace split semantic stages only after each fused stage passes isolated and
integrated target-shape correctness. Preserve gate/up preactivation as the
intentional residual. Benchmark the honest public boundary, including metadata
and any required synchronization.

## Benchmark and Tax Decomposition

Every fused stage needs the following target-shape rows, using repeat medians:

```text
semantic JAX reference
split production baseline
semaphore-only
copy-only
compute-only from prefilled local staging
full fused kernel
full fused kernel with pipeline_depth = 1, 2, 3, 4
```

Also report:

```text
useful rows and padded rows
useful and rounded TFLOP/s/rank
bytes sent per rank
effective peer-memory bandwidth
producer slot wait cycles or equivalent counter
consumer ready wait cycles or equivalent counter
slot occupancy/high-water mark
live and masked send chunks
live and masked compute tiles
```

Evidence of overlap requires more than a source-level prefetch loop. At least
one of the following must hold:

- full fused time is materially below copy-only plus compute-only;
- increasing pipeline depth improves full fused time until occupancy saturates;
- profiler traces show producer memory operations concurrent with consumer
  WGMMA;
- producer/consumer wait counters demonstrate that both roles make progress on
  different slots concurrently.

If pipeline depth does not help and full time is approximately additive, call
the implementation blocking staging and do not promote it.

## Correctness Gates

For tiny interpreted/JAX tests and target H100 tests:

- compare every fused stage with the semantic JAX reference;
- use roughly balanced random routing, not only exact balanced routing;
- cover pair-capacity and expert-capacity overflow independently;
- report router drops and metadata overflow separately;
- verify route-weight gradients and duplicate top-k token accumulation;
- verify invalid and padded rows contribute exact zero;
- verify queue generations across repeated slot reuse;
- run at least 48 target repeats for a promoted profile;
- reject any path with dropped live routes, semaphore timeout, or nondeterministic
  corruption.

## Success Criteria

The project is complete only when:

1. `permute + W13` is one genuinely overlapped producer/consumer runtime stage;
2. source `dcombine/dy-route + W2 backward` is one genuinely overlapped stage
   consuming source-sharded `dy` without all-gather;
3. forward return and dX return avoid dense replicated partial buffers;
4. the integrated custom-VJP path is correct at the target shape;
5. aggregate forward+backward reaches 250 useful TFLOP/s/rank, or a new measured
   hardware/compiler limit is established for the fused implementation itself.

The existing split-stage overlap floor does not satisfy the final condition. A
ceiling claim must come from the fused producer/consumer design, with copy,
compute, semaphore, occupancy, and masking taxes measured separately.

## Prior Art and Starting Points

- `.agents/projects/20260707_source_push_jax_semantic_plan.md` defines the
  current semantic metadata and reference implementation.
- `.agents/projects/20260703_moe_inbox.md` defines the MLP custom-VJP and saved
  preactivation boundary.
- `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py` contains the proven
  Hopper source-push inbox producer/consumer work.
- `lib/levanter/scripts/bench/bench_source_push_semantic_plan.py` contains the
  per-stage and integrated correctness/performance harness.
- PR #6841 and issue #6597 contain the current implementation and benchmark
  record.
