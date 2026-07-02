
## Current Status - 2026-07-02

The active forward candidate is the source-push inbox path, not the earlier
destination-pull/static-workqueue line below.

Code boundary:

- `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py`
  contains the package-private source-push inbox prototype: `PushInboxConfig`,
  Pallas kernel factory, sharded wrapper, input generation, validation, timing,
  parser, and `main()`.
- `lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py`
  contains the measured profile defaults.
- `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  is now only a thin CLI wrapper.

Best measured forward profile:

- `hopper_queue_ngroups2_210`
- H100 job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-123853`
- Target uniform all-to-all shape, median `0.008288984719s`,
  `219.747234 TFLOP/s/rank`, `38/48 >=210`, `5/48 <160`.
- Checked smoke `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-124258`
  had metadata mismatches `0` and `max_abs_diff=0.04443550109863281`.

Target-like diagnostic profile:

- `hopper_queue_roughly_balanced_ngroups2_210`
- Multi-seed rough-balanced job
  `/dlwh/repro-source-push-roughly-balanced-ngpj2-seeds-20260702-055716`
- Overall median `0.008391041s`, `217.325 TFLOP/s/rank`,
  `51/72 >=210`, `11/72 <160`.

Current bottleneck read:

- Stage 1 median target is met by repeated target timing, but slow-tail outliers
  remain.
- Receiver schedule variants, ready-scan, slot-group, direct self compute,
  emit-pipeline, slot order, CTA split, buffering, and N grouping have all been
  tested enough to stop blind tuning.
- Remaining work for Stage 2 is PR cleanup around the package-private boundary,
  plus deciding whether to keep this as an isolated prototype or wire it into the
  `pallas_mgpu` forward path.

---

Historical starting strategy:
# Spec: Static Workqueue `permute_up` Pallas Mosaic MGPU Kernel

## Goal

Implement a Hopper Pallas Mosaic MGPU forward kernel for the MoE “up” half:

```text
permute routed token assignments over NVLink
  -> compute local expert W_gate/W_up + SiLU
  -> materialize hidden in destination expert-major layout
```

This kernel replaces the staged path:

```text
permute_mgpu(...)
  -> local fused ragged_w13_swiglu_mgpu(...)
```

with a fused static-phase kernel that can eventually overlap NVLink remote writes with local WGMMA compute.

The target is **single-node EP over NVLink**, not NIC/RDMA/NVSHMEM/NCCL EP.

## Non-goals

Do not implement:

```text
down projection
unpermute/combine
backward
FP8
NIC/RDMA
EP > 8
dynamic runtime work queues
remote atomics
top-k same-destination dedup
```

Do not try to beat NCCL by reimplementing a staged all-to-all. The point of this kernel is to exploit MoE-specific layout and eventually overlap communication with expert compute.

## Hardware and shape target

Target hardware:

```text
H100 / Hopper
single-node NVLink
EP <= 8
initial target EP = 8
```

Target model shape per rank:

```text
x_local:          [T, D]
selected_experts: [T, K]
w_gate_up_local:  [E_local, D, 2I]
hidden_out:       [capacity, I]
```

Concrete intended values:

```text
T = 32768
K = 4
D = 2560
I = 1280
E_total = 256
EP = 8
E_local = 32
dtype = bfloat16
```

Balanced routing expectation:

```text
assignments per source rank = T * K = 131072
assignments per source/global expert ≈ T*K/E_total = 512
```

These expectations are just that, expectaitons. do not rely on them. assume some imbalance. we allow for drop on overflow, but with a generous buffer that should be applied per-device, not per-expert.

## Key abstraction: `expert_group`

The core scheduling unit is not a single expert. It is an **expert group**.

```text
expert_group = a contiguous group of local experts on the destination rank
```

For `E_local = 32`, recommended initial values:

```text
expert_group_size = 16
num_expert_groups = E_local / expert_group_size = 2
```

Benchmark later:

```text
expert_group_size = 8, 16, 32
```

Why expert groups matter:

1. They give each phase enough WGMMA work to fill the GPU.
2. They improve weight locality by processing several adjacent local experts in sequence.
3. They reduce semaphore/phase overhead versus one-expert phases.

Approximate tile count per `(src_rank, expert_group)` with balanced routing:

```text
rows per source/expert ≈ 512
block_m = 64
block_n = 128
I = 1280

one expert:
    ceil(512/64) * ceil(1280/128) = 8 * 10 = 80 WGMMA output tiles

8 experts:
    640 tiles

16 experts:
    1280 tiles

32 experts:
    2560 tiles
```

One expert at a time is too skinny. Start with 16 experts per phase.

## Static workqueue discipline

Use a **static phase schedule**, not a dynamic queue.

At every phase `t`, every rank knows exactly:

```text
expert_group(t)
peer_phase(t)
dst_rank(t)
src_rank(t)
which local rows it must send
which remote rows it will receive
which local expert rows it can compute after wait
```

The schedule itself is the workqueue.

There should be no dynamic operation like:

```python
sender, unit, tokens = receive_work_unit()
```

Instead:

```python
src = deterministic_src_for_phase(rank, t)
expert_group = deterministic_expert_group_for_phase(rank, t)

wait_ready(src, expert_group, t)
compute_known_rows(src, expert_group)
```

This matters because dynamic GPU queues are expensive, hard to make deterministic, and unnecessary here. Counts and offsets already define the full schedule.

## Phase schedule

For initial implementation, use simple nested phases:

```text
for expert_group_phase in 0 .. num_expert_groups-1:
    expert_group = expert_group_phase

    for peer_phase in 0 .. EP-1:
        dst = (rank + peer_phase) % EP
        src = (rank - peer_phase + EP) % EP
```

This gives:

```text
num_phases = num_expert_groups * EP
```

For `expert_group_size = 16` and `EP = 8`:

```text
num_phases = 2 * 8 = 16
```

Include `peer_phase = 0` initially for uniformity. That phase is the local/self case. It can later be optimized into a local copy or skipped/special-cased.

### Swizzling

Swizzling is a later optimization, not required for v0.

Potential peer swizzle:

```python
peer_phase_order = [0, 1, 3, 5, 7, 2, 4, 6]
```

or any deterministic order with good spread over the NVLink topology.

Potential expert group swizzle:

```python
expert_group = (expert_group_phase + rank) % num_expert_groups
```

Do not start with this. Start boring:

```python
expert_group = expert_group_phase
peer_phase = peer_phase_phase
```

Once correctness and performance harnesses are stable, add configurable swizzle knobs.

## Inputs and metadata

The kernel should receive precomputed routing metadata. Do not compute stable sort inside the Pallas kernel.

Flatten assignments:

```text
assignment_id = token_id * K + route_slot
```

Stable-sort by global expert id:

```text
assignment_ids_sorted = stable_argsort(expert_ids_flat)
```

Given deterministic token-major/route-slot-major flattening, stable sort by global expert id induces order:

```text
global_expert_id, assignment_id
```

Since:

```text
global_expert_id = dst_rank * E_local + local_expert
```

this gives deterministic ordering by:

```text
dst_rank, local_expert, token_id, route_slot
```

Required metadata arrays:

```python
assignment_ids_sorted:  [T*K]      # source-local assignment id
token_ids_sorted:       [T*K]      # assignment_ids_sorted // K
expert_ids_sorted:      [T*K]
dst_ranks_sorted:       [T*K]
local_experts_sorted:   [T*K]
local_pos_sorted:       [T*K]      # position within source bucket (dst_rank, local_expert)
send_counts:            [EP, E_local]
clipped_counts:         [EP, EP, E_local]
```

The `clipped_counts` tensor is indexed:

```text
clipped_counts[src_rank, dst_rank, local_expert]
```

The source rank also needs:

```python
source_expert_offsets: [EP * E_local]
```

where:

```text
source_expert_offsets[dst * E_local + expert]
```

points to the first sorted assignment for that destination/global expert in the source’s sorted assignment list.

Destination layout helpers:

```python
rows_per_dst_expert:       [EP, E_local]
expert_base_by_dst:        [EP, E_local]
src_base_by_dst_expert:    [EP, EP, E_local]
```

where:

```text
rows_per_dst_expert[dst, expert] =
    sum_src clipped_counts[src, dst, expert]

expert_base_by_dst[dst, expert] =
    prefix_sum over expert of rows_per_dst_expert[dst, expert]

src_base_by_dst_expert[src, dst, expert] =
    prefix_sum over src of clipped_counts[src, dst, expert]
```

Remote row formula:

```python
remote_row(dst, src, expert, local_pos) =
    expert_base_by_dst[dst, expert]
  + src_base_by_dst_expert[src, dst, expert]
  + local_pos
```

This row is unique and deterministic.

## Outputs

The kernel should output:

```python
hidden:              [capacity, I]
metadata sufficient to reconstruct source assignment for each row in hidden
```


Do not store:

```text
recv_weight
recv_token_id
recv_route_slot
recv_expert
```

They are either source-local or derivable.

## Core algorithm

At phase `t`, each rank does two conceptual jobs:

```text
producer job:
    send local assignments for (dst_rank(t), expert_group(t))
    into dst's final recv_x layout
    signal dst

compute job:
    wait for source src_rank(t) to finish sending
    compute W_gate/W_up + SiLU for rows from src_rank(t)
    for local experts in expert_group(t)
    store hidden into final local hidden layout
```

Important: receiving is not an explicit copy. The source remote-writes into the destination’s local GMEM. The compute side only waits and then reads local GMEM.

## GMEM scratch and output buffers

Inside the kernel, use:

```python
recv_x:              [capacity, D]   # destination receive scratch
hidden:              [capacity, I]   # final output of permute_up
```

The source remote-writes into peer `recv_x`.

The destination computes from its local `recv_x` into local `hidden`.

Initially, keep `recv_x` as a debug/temporary output if useful. Long term, it can be internal scratch.

## Phase-level pseudocode

Conceptual non-overlapped static phased version:

```python
for phase in range(num_phases):
    expert_group = expert_group_for_phase(rank, phase)
    peer_phase = peer_phase_for_phase(rank, phase)

    dst = (rank + peer_phase) % EP
    src = (rank - peer_phase + EP) % EP

    send_phase(dst, expert_group)
    signal_ready(dst, phase)

    wait_ready(src, phase)
    compute_phase(src, expert_group)
```

Overlapped version:

```python
# producer WG
for phase in range(num_phases):
    expert_group = expert_group_for_phase(rank, phase)
    peer_phase = peer_phase_for_phase(rank, phase)
    dst = (rank + peer_phase) % EP

    send_phase(dst, expert_group)
    signal_ready(dst, phase)

# compute WG
for phase in range(num_phases):
    expert_group = expert_group_for_phase(rank, phase)
    peer_phase = peer_phase_for_phase(rank, phase)
    src = (rank - peer_phase + EP) % EP

    wait_ready(src, phase)
    compute_phase(src, expert_group)
```

This is the important distinction:

* non-overlapped: one phase loop does send then compute;
* overlapped: producer and compute run independent phase loops connected by semaphores.

## Role structure

### V0: single-WG serial

Use one role. No `num_threads=2`.

```python
for phase:
    send_phase
    signal
    wait
    compute_phase
```

This is for correctness and debugging only.

### V1: two-WG lockstep

Use:

```python
num_threads = 2
thread_name = "wg"
```

Roles:

```text
WG0: producer
WG1: compute
```

But keep them in the same phase loop:

```python
for phase:
    if producer: send + signal
    if compute: wait + compute
```

This mainly validates role partitioning and semaphore protocol. It may not improve performance much.

### V2: two-WG overlapped

Use two independent loops:

```python
if producer:
    for phase:
        send + signal

if compute:
    for phase:
        wait + compute
```

This is the actual outer software pipeline.

There is no separate receiver WG. Receiving is just remote writes landing in local GMEM plus a semaphore. The compute WG is the consumer.

## Semaphore protocol

The semaphore protocol only needs to signal **data readiness**. The sender does not need to remote-write per-row source metadata in the production `permute_up` path.

Producer on source rank `s`, phase `t`, destination rank `d`:

```python
remote_write(d.recv_x[remote_rows_for_phase, :], x_local[token_rows, :])
signal(d.ready[s, t])
```

Compute on destination rank `d`:

```python
wait(local.ready[src, t])
compute rows for (src, expert_group(t))
```

The source rank and source-local assignment for each row are reconstructed later from the deterministic routing metadata:

```text
row = expert_base[expert] + src_base[src, dst, expert] + local_pos

global_expert = dst * E_local + expert
offset = source_expert_offsets_by_src[src, global_expert] + local_pos
source_assignment = assignment_ids_sorted_by_src[src, offset]
```

So the production `permute_up` hot path should not write:

```text
recv_src_rank
recv_src_assignment
recv_weight
recv_token_id
recv_route_slot
```

Optional debug mode may materialize `recv_src_rank` and `recv_src_assignment` and compare them against reconstructed values, but this should not be the default path.

### Readiness granularity

The readiness unit is:

```text
(src_rank, dst_rank, phase)
```

where `phase` uniquely determines:

```text
expert_group
peer_phase
src_rank
dst_rank
the row ranges being sent
```

A phase is ready on destination `d` when all producer workers on source `s` have completed the remote writes for that phase.

### Preferred semaphore shape

Prefer a phase/source-indexed readiness protocol:

```text
ready[src_rank, phase]
```

Producer:

```python
signal(remote_ready[src_rank, phase])
```

Compute:

```python
wait(local_ready[src_rank, phase])
```

If Pallas/Mosaic makes explicit semaphore arrays awkward, use an equivalent cumulative protocol, but keep the same logical contract: `compute_phase(src, phase)` must not read `recv_x` until all writes from `src` for that phase are complete.

### Worker-level signaling

Avoid relying on a grid-wide producer barrier before one elected worker signals. A cross-SM barrier is exactly the kind of thing that can become fragile.

Instead, the simplest robust protocol is:

```text
each producer worker signals once after finishing its slice of the phase
destination waits for num_producer_workers arrivals
```

For single-WG serial V0:

```text
num_producer_workers = num_sms
```

For two-WG V1/V2 with one producer WG per SM:

```text
num_producer_workers = num_sms
```

If the implementation later uses multiple producer WGs per SM, update the expected arrival count accordingly:

```text
num_producer_workers = num_sms * producer_wgs_per_sm
```

### V0 serial protocol

Single role, no `wg` axis:

```python
for phase in phases:
    send_phase(phase)
    signal_ready(phase)      # each SM/program signals after its send slice
    wait_ready(src, phase)   # wait for num_producer_workers arrivals
    compute_phase(src, phase)
```

This is for correctness bring-up.

### V1 two-WG lockstep protocol

Two roles, same phase loop:

```python
for phase in phases:
    if is_producer:
        send_phase(phase)
        signal_ready(phase)

    if is_compute:
        wait_ready(src, phase)
        compute_phase(src, phase)
```

This validates role partitioning and semaphore accounting.

### V2 overlapped protocol

Two independent loops:

```python
if is_producer:
    for phase in phases:
        send_phase(phase)
        signal_ready(phase)

if is_compute:
    for phase in phases:
        wait_ready(src, phase)
        compute_phase(src, phase)
```

This is the actual outer software pipeline.

### Semaphore accounting requirements

For every `(src, dst, phase)`:

```text
signal count == number of producer workers that write a slice of that phase
wait count   == same number of producer workers
```

Do not use `decrement=True` unless the accounting is very clearly phase-local. For the overlapped version, a monotonic cumulative scheme is usually safer than consuming/decrementing a shared semaphore in a way that can race later phase arrivals.

The correctness condition is:

```text
All remote writes to recv_x for (src, dst, phase)
happen-before
compute_phase(src, phase) reads those recv_x rows.
```



## `send_phase(dst, expert_group)` details

The send side sends all assignments from this source rank whose destination is `dst` and whose local expert is in `expert_group`.

The production `permute_up` hot path should remote-write only token payloads into the destination’s `recv_x` buffer, then signal readiness for the phase.

It should not remote-write per-row metadata:

```text
do not write recv_src_rank
do not write recv_src_assignment
do not write recv_weight
do not write recv_token_id
do not write recv_route_slot
```

Those values are reconstructible later from the deterministic schedule, counts, and source routing metadata.

For each expert `e` in the expert group:

```python
count = clipped_counts[rank, dst, e]
```

For each local position:

```python
local_pos in [0, count)
```

Find the source-side sorted assignment:

```python
global_expert = dst * E_local + e
offset = source_expert_offsets[global_expert] + local_pos
token_id = token_ids_sorted[offset]
```

Compute the destination row:

```python
rr = remote_row(dst, rank, e, local_pos)
```

Remote-write only the token row:

```python
remote_recv_x[rr, :] = x_local[token_id, :]
```

After all producer workers finish their slice of the phase, signal readiness:

```python
signal_ready(dst, phase)
```

The destination compute side already knows, from `phase`, which source and expert group this corresponds to:

```python
src = deterministic_src_for_phase(rank, phase)
expert_group = deterministic_expert_group_for_phase(rank, phase)
```

and later kernels can reconstruct the original source assignment for any row using:

```python
global_expert = dst * E_local + expert
offset = source_expert_offsets_by_src[src, global_expert] + local_pos
source_assignment = assignment_ids_sorted_by_src[src, offset]
```

### Copy tiling

Copy token rows in D-tiles:

```python
copy_tile_n = 256 or 512 initially
```

Avoid tiny copy tiles. Existing 128-wide copies are likely too small. Start with:

```text
dispatch_chunk_copy_tile = 256
```

and benchmark 512.

A conceptual copy loop:

```python
for expert in experts_in_group:
    count = clipped_counts[rank, dst, expert]
    global_expert = dst * E_local + expert
    source_base = source_expert_offsets[global_expert]

    for local_pos in static_range_with_mask:
        if local_pos < count:
            offset = source_base + local_pos
            token_id = token_ids_sorted[offset]
            rr = remote_row(dst, rank, expert, local_pos)

            for d_tile in D_tiles:
                remote_recv_x[rr, d_tile] = x_local[token_id, d_tile]
```

### Variable counts

Do not assume balanced routing.

The Pallas loop bound must be static, but all token loads and remote stores must be guarded by:

```python
local_pos < count
```

Options for the static loop bound:

1. Use a conservative capacity-derived upper bound per `(src, dst, expert)`.
2. Use a precomputed static upper bound per expert group.
3. Use a padded layout per `(src, dst, expert)` to a fixed `block_m` multiple.

The current balanced synthetic path that assumes:

```python
rows_per_source_expert = assignments // (EP * E_local)
```

is not valid for real routing unless all `send_counts` buckets are exactly equal. Disable that path for real routing or assert the equality before using it.

### Local/self phase

For `peer_phase = 0`, `dst == rank`.

For v0, it is acceptable to treat this uniformly as a remote/local write into `recv_x`. Later, special-case it as a local copy or direct compute if that improves performance.

### Correctness invariant

For each valid routed assignment, exactly one producer writes exactly one destination row:

```text
(src_rank, dst_rank, expert, local_pos) -> unique remote_row
```

For each phase, the readiness signal must happen after all token payload writes for that phase are complete.



## `compute_phase(src, expert_group)` details

The compute side consumes token rows that have already been remote-written into this rank’s local `recv_x` buffer.

There is no explicit receive copy and no per-row metadata read in the production `permute_up` compute phase. The phase schedule determines the source rank and expert group:

```python
src = deterministic_src_for_phase(rank, phase)
expert_group = deterministic_expert_group_for_phase(rank, phase)
```

For each local expert `e` in the expert group:

```python
row_count = clipped_counts[src, rank, e]
row_start =
    expert_base_by_dst[rank, e]
  + src_base_by_dst_expert[src, rank, e]
```

The rows for this `(src, rank, e)` subrange are contiguous:

```python
recv_x[row_start : row_start + row_count, :]
```

Compute:

```python
hidden[row_start : row_start + row_count, :] =
    SiLU(recv_x[...] @ W_gate[e]) * (recv_x[...] @ W_up[e])
```

The inner compute body should conceptually delegate to the existing local fused W13/SwiGLU kernel. Do not hand-roll a new K pipeline inside the outer phase scheduler.

Use the known-good pattern:

```text
mgpu.emit_pipeline
  LHS tile: recv_x
  RHS gate tile: w_gate_up[e, :, n]
  RHS up tile:   w_gate_up[e, :, I+n]
  WGMMA into two accumulators
  hidden = SiLU(gate_acc) * up_acc
  store hidden
```

### Static loop structure

The Pallas loop bounds must be static, but `row_count` is dynamic. Use a static upper bound with masks, or use a padded layout.

Conceptually:

```python
for expert in experts_in_group:
    row_count = clipped_counts[src, rank, expert]
    row_start = expert_base_by_dst[rank, expert] + src_base_by_dst_expert[src, rank, expert]

    for m_block in static_m_blocks:
        local_row = m_block * block_m
        actual_rows = min(block_m, row_count - local_row)

        if actual_rows > 0:
            compute W13/SwiGLU tile(s)
            store hidden[row_start + local_row : ...]
```

For v0, it is acceptable to reuse the existing ragged-tail/log2 store ladder from the local fused W13/SwiGLU kernel. If this becomes too awkward, add an explicit padded mode where each `(src, dst, expert)` subrange is padded to a `block_m` multiple.

### Metadata reconstruction

The compute phase does not need:

```text
recv_src_rank
recv_src_assignment
recv_token_id
recv_route_slot
recv_weight
```

Later kernels can reconstruct return destinations from deterministic routing metadata. For any computed row:

```python
local_pos = row - expert_base_by_dst[rank, expert] - src_base_by_dst_expert[src, rank, expert]
global_expert = rank * E_local + expert
offset = source_expert_offsets_by_src[src, global_expert] + local_pos
source_assignment = assignment_ids_sorted_by_src[src, offset]
```

permute_up can return/cache this to make down_unpermute and backward easier, but it is not required for correctness.


### Zero-count experts

If:

```python
row_count == 0
```

the compute phase should do no reads, no WGMMA, and no stores for that expert/source subrange.

Add explicit tests for:

```text
zero rows for one expert in the group
zero rows for all experts in the group from one source
many empty source/expert subranges
```

### Correctness invariant

For phase `(src, dst=rank, expert_group)`, the compute phase may read only rows whose token payload writes have been completed and signaled by `src`.

The required happens-before relationship is:

```text
source remote-writes recv_x rows for (src, rank, expert_group, phase)
happen-before
destination compute_phase(src, expert_group, phase) reads those rows
```


### Important implementation constraint

The existing local fused kernel is ragged over all experts. For this phase kernel, adapt the same body to operate on a known expert range:

```text
expert_group_start : expert_group_start + expert_group_size
```

and a known source subrange within each expert:

```text
row_start = expert_base[e] + src_base[src, e]
row_count = counts[src, rank, e]
```

This is a phase-restricted local fused W13/SwiGLU, not a new matmul kernel.

## Occupancy and expert group size

Outer scheduling does not itself keep all SMs busy. The inner WGMMA tile scheduler does.

Each phase must provide enough WGMMA tiles:

```text
tiles_per_phase =
    sum_e_in_expert_group ceil(count[src, rank, e] / block_m)
  * ceil(I / block_n)
```

For H100, target at least several hundred output tiles per phase.

Start with:

```text
expert_group_size = 16
```

Benchmark:

```text
expert_group_size = 8
expert_group_size = 16
expert_group_size = 32
```

Do not start with expert_group_size = 1.

## Padding strategy

For v0, compact expert-major layout is okay.

But for phase compute, tails are annoying. Consider padding each `(src, dst, expert)` range to `block_m` if it simplifies the compute body.

Options:

### Compact layout

Pros:

```text
less memory
matches existing ragged layout
```

Cons:

```text
needs tail handling/log2 ladder
dynamic row_count per source/expert
```

### Per-source/expert block-m padded layout

Pros:

```text
full block_m stores
simpler phase compute
simpler WGMMA tile loops
```

Cons:

```text
more memory
more padded compute
slightly different layout from current staged reference
```

Initial recommendation:

* Keep compact layout for correctness parity with staged path.
* If implementation complexity explodes, add an explicit padded layout mode for the fused phased kernel.


## Pallas boilerplate sketch

This is schematic. Use the real Mosaic API syntax from the existing examples and current local fused W13/SwiGLU kernel. The important changes from the earlier sketch are:

```text
permute_up does not write recv_src_rank
permute_up does not write recv_src_assignment
permute_up only writes token payloads into recv_x and signals readiness
```

The source/local assignment metadata is reconstructed later by `down_unpermute` or backward from deterministic routing metadata.

```python
def permute_up_static_workqueue_mgpu(
    x_local,                    # [T, D]
    token_ids_sorted,            # [T*K], sorted by global expert id
    source_expert_offsets,       # [EP * E_local]
    clipped_counts,              # [EP, EP, E_local]
    expert_base_by_dst,          # [EP, E_local]
    src_base_by_dst_expert,      # [EP, EP, E_local]
    w_gate_up_local,             # [E_local, D, 2I]
    *,
    capacity: int,
    ep_size: int,
    local_experts: int,
    expert_group_size: int,
    expert_axis: str,
    config: MoeMgpuConfig,
):
    T, D = x_local.shape
    E_local, D2, I2 = w_gate_up_local.shape
    I = I2 // 2

    assert D == D2
    assert local_experts == E_local
    assert local_experts % expert_group_size == 0
    assert D % config.dispatch_chunk_copy_tile == 0

    num_expert_groups = local_experts // expert_group_size
    num_phases = num_expert_groups * ep_size
    num_sms = config.num_sms or jax.devices()[0].core_count

    copy_tile_n = config.dispatch_chunk_copy_tile
    d_tiles = D // copy_tile_n

    def body(
        x_ref,
        token_ids_ref,
        source_expert_offsets_ref,
        clipped_counts_ref,
        expert_base_by_dst_ref,
        src_base_by_dst_expert_ref,
        w13_ref,
        hidden_ref,      # [capacity, I]
        recv_x_ref,      # [capacity, D], scratch/debug output
    ):
        rank = lax.axis_index(expert_axis)
        sm_id = lax.axis_index("sm")

        # V1/V2 role split will add:
        #
        #   wg_id = lax.axis_index("wg")
        #   is_producer = wg_id == 0
        #   is_compute = wg_id == 1
        #
        # V0 has no wg axis.

        ready_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)

        def phase_to_expert_group(phase):
            eg_phase = phase // ep_size

            # Later swizzle option:
            # return (eg_phase + rank) % num_expert_groups

            return eg_phase

        def phase_to_peer_phase(phase):
            peer_phase = phase % ep_size

            # Later swizzle option:
            # return peer_phase_order[peer_phase]

            return peer_phase

        def remote_row(dst, src, expert, local_pos):
            return (
                expert_base_by_dst_ref[dst, expert]
                + src_base_by_dst_expert_ref[src, dst, expert]
                + local_pos
            )

        def signal_phase_ready(dst, phase):
            # Logical contract:
            #
            #   signal dst.ready[rank, phase]
            #
            # Implementation may use a phase-indexed semaphore or a cumulative
            # monotonic protocol. Each producer worker should signal once after
            # finishing its slice of this phase. Destination waits for the matching
            # number of producer-worker arrivals.
            pl.semaphore_signal(
                ready_sem,
                device_id=dst,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

        def wait_phase_ready(src, phase):
            # Logical contract:
            #
            #   wait local.ready[src, phase]
            #
            # The actual wait value must match the chosen semaphore accounting.
            # For a phase-local semaphore, wait for num_producer_workers arrivals.
            pl.semaphore_wait(
                ready_sem,
                value=num_sms,
                decrement=True,
            )

        def send_phase(phase):
            eg = phase_to_expert_group(phase)
            peer_phase = phase_to_peer_phase(phase)
            dst = (rank + peer_phase) % ep_size
            expert_start = eg * expert_group_size

            # Send all rows for this destination and expert group.
            for e_in_group in range(expert_group_size):
                expert = expert_start + e_in_group
                count = clipped_counts_ref[rank, dst, expert]

                global_expert = dst * local_experts + expert
                source_base = source_expert_offsets_ref[global_expert]

                # Static loop bound. This is intentionally schematic.
                #
                # Options:
                #   - capacity-derived conservative max_count
                #   - precomputed static max per (dst, expert_group)
                #   - padded per-(src,dst,expert) layout
                #
                # Every real load/store must be masked by local_pos < count.
                max_count = STATIC_MAX_COUNT_FOR_ONE_SRC_DST_EXPERT
                total_tiles = max_count * d_tiles
                steps = ceil_div(total_tiles, num_sms)

                @pl.loop(0, steps)
                def _send_step(step):
                    linear = step * num_sms + sm_id
                    local_pos = linear // d_tiles
                    d_tile = linear - local_pos * d_tiles

                    @pl.when(local_pos < count)
                    def _copy_payload_tile():
                        d_start = d_tile * copy_tile_n

                        offset = source_base + local_pos
                        token_id = token_ids_ref[offset]

                        rr = remote_row(dst, rank, expert, local_pos)

                        remote_recv_x = mgpu.remote_ref(
                            recv_x_ref,
                            dst,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )

                        remote_recv_x[
                            rr,
                            pl.ds(d_start, copy_tile_n),
                        ] = x_ref[
                            token_id,
                            pl.ds(d_start, copy_tile_n),
                        ]

            signal_phase_ready(dst, phase)

        def compute_phase(phase):
            eg = phase_to_expert_group(phase)
            peer_phase = phase_to_peer_phase(phase)
            src = (rank - peer_phase + ep_size) % ep_size
            expert_start = eg * expert_group_size

            wait_phase_ready(src, phase)

            for e_in_group in range(expert_group_size):
                expert = expert_start + e_in_group

                row_count = clipped_counts_ref[src, rank, expert]
                row_start = (
                    expert_base_by_dst_ref[rank, expert]
                    + src_base_by_dst_expert_ref[src, rank, expert]
                )

                emit_w13_swiglu_for_expert_range(
                    recv_x_ref,
                    w13_ref,
                    hidden_ref,
                    expert=expert,
                    row_start=row_start,
                    row_count=row_count,
                    config=config,
                )

        # ------------------------------------------------------------------
        # V0: single-WG serial correctness path.
        # No wg axis. No overlap. Simplest bring-up.
        # ------------------------------------------------------------------
        @pl.loop(0, num_phases)
        def _phase_loop(phase):
            send_phase(phase)
            compute_phase(phase)

        # ------------------------------------------------------------------
        # V1: two-WG lockstep.
        # Add num_threads=2/thread_name="wg" and role guards.
        #
        # for phase:
        #     if is_producer:
        #         send_phase(phase)
        #     if is_compute:
        #         compute_phase(phase)
        #
        # This validates role partitioning but may not overlap much.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # V2: two-WG overlapped.
        # Add num_threads=2/thread_name="wg" and independent loops.
        #
        # if is_producer:
        #     for phase:
        #         send_phase(phase)
        #
        # if is_compute:
        #     for phase:
        #         compute_phase(phase)
        #
        # This is the actual outer software pipeline.
        # ------------------------------------------------------------------

    kernel = mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct((capacity, I), x_local.dtype),
            # Keep recv_x as a debug/scratch output initially.
            # Long term this can become internal scratch if/when the API allows.
            jax.ShapeDtypeStruct((capacity, D), x_local.dtype),
        ],
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(
            lowering_semantics=mgpu.LoweringSemantics.Warpgroup,
        ),
    )

    hidden, recv_x = kernel(
        x_local,
        token_ids_sorted,
        source_expert_offsets,
        clipped_counts,
        expert_base_by_dst,
        src_base_by_dst_expert,
        w_gate_up_local,
    )

    return hidden
```

Important notes:

1. `assignment_ids_sorted` is not needed by this hot kernel. It is needed later by `down_unpermute`/backward to reconstruct return destinations.
2. `recv_src_rank` and `recv_src_assignment` are not outputs of production `permute_up`.
3. `recv_x` is shown as an output only because Pallas kernels often use output refs for GMEM scratch/debug. The public API should return only `hidden` plus any small layout/count metadata already available outside the kernel.
4. The semaphore calls above are schematic. The implementation must use either phase-indexed semaphores or a carefully documented cumulative monotonic protocol.


## Milestones

### Milestone 0: stop optimizing the balanced chunked fused path

The current balanced chunked fused path is not the production design.

Tasks:

```text
- Disable it by default.
- Add a hard assertion that it is only used when every send_counts bucket equals the assumed rows_per_source_expert.
- Mark it synthetic-only.
```

Acceptance:

```text
Real routing does not silently enter the balanced chunked path.
```

### Milestone 1: refactor local W13/SwiGLU into a reusable compute body

Create a helper or internal structure that can compute W13/SwiGLU for:

```text
one expert or one contiguous expert_group
one known row range per expert
```

without changing the existing local fused W13/SwiGLU numerics.

Acceptance:

```text
Helper matches current local fused up kernel for already-local expert-major inputs.
Empty experts and ragged tails tested.
```

### Milestone 2: implement static phased V0, single-WG serial

Implement:

```text
for phase:
    send_phase
    signal
    wait
    compute_phase
```

No warp specialization yet.

Use:

```text
expert_group_size = 16
peer_phase_order = [0,1,2,3,4,5,6,7]
expert_group_order = [0,1,...]
```

Acceptance:

```text
Correct vs staged permute + local fused up.
EP=1, EP=2, EP=8 correctness.
Random routing.
Skewed routing.
Many empty experts.
No balanced-routing assumption.
```

Do not optimize performance here. This is correctness/protocol bring-up.

### Milestone 3: measure phase granularity

Benchmark V0 with:

```text
expert_group_size = 8
expert_group_size = 16
expert_group_size = 32
```

Record:

```text
runtime
tiles per phase
estimated dispatch bytes/s
effective W13/SwiGLU TFLOP/s
```

Acceptance:

```text
Choose default expert_group_size for next milestones.
Expected initial default: 16.
```

### Milestone 4: implement V1 two-WG lockstep

Use:

```text
num_threads = 2
thread_name = "wg"
WG0 = producer
WG1 = compute
```

But keep same phase loop:

```text
for phase:
    producer sends/signals
    compute waits/computes
```

Acceptance:

```text
Correctness identical to V0.
Semaphore protocol works with role split.
No deadlocks.
Performance does not need to win yet.
```

### Milestone 5: implement V2 two-WG overlapped loops

Use independent loops:

```text
producer WG:
    for phase:
        send/signals

compute WG:
    for phase:
        wait/compute
```

Acceptance:

```text
Correct vs V0/staged reference.
No deadlocks.
Measured overlap: V2 faster than V1 for at least one target shape/config.
```

If V2 is not faster, collect profiling data before making it more complex.

### Milestone 6: add swizzling

Add configurable swizzle options:

```text
peer_phase_order
expert_group_order
```

Start with:

```text
no swizzle
```

Then test:

```text
expert_group = (expert_group_phase + rank) % num_expert_groups
peer_phase_order = [0, 1, 3, 5, 7, 2, 4, 6]
```

Acceptance:

```text
Correctness unchanged.
Performance data logged.
Keep only swizzles that help.
```

### Milestone 7: remove unnecessary initialization

Avoid full `recv_x` zeroing if all valid rows are uniquely written.

Initialize only:

```text
metadata padding rows
hidden padding rows if needed
```

Acceptance:

```text
Correctness unchanged.
Less dispatch overhead.
```

### Milestone 8: tune copy granularity

Test:

```text
dispatch_chunk_copy_tile = 256, 512
```

Optional:

```text
dispatch_chunk_copy_rows > 1
vectorized multi-row copy
```

Acceptance:

```text
Best copy tile selected based on target-shape benchmark.
```

### Milestone 9: tune all config knobs

```text
expert_group_size = 8, 16, 32
dispatch_chunk_copy_tile = 256, 512
existing local w13_silu config knobs
```

### Milestone 10: compare against staged/NCCL baseline

Compare:

```text
A. NCCL/staged reference
B. staged Pallas permute + local fused up
C. static phased V0
D. static phased V2 overlapped
```

Acceptance:

```text
If V2 is not meaningfully faster than staged Pallas, identify bottleneck.
If V2 is not competitive with NCCL, decide whether to continue based on whether the fused path provides an end-to-end advantage.
```

## Correctness test matrix

Test:

```text
EP = 1, 2, 8
K = 1, 2, 4
E_local = 1, 4, 32
expert_group_size = 1, 8, 16, 32 where valid
D/I small debug shapes
D=2560, I=1280 target shape
T small, medium, target
```

Routing cases:

```text
uniform random
balanced synthetic
all to one expert
all to local experts
all to one remote rank
zero-token experts
many zero-token experts
capacity overflow/clipping
```

Numerics:

```text
bfloat16 primary
compare to staged reference
use fused-semantics tolerance from local W13/SwiGLU tests
no NaNs/Infs
```

## Performance notes

Do not expect this kernel to win merely by replacing NCCL with remote writes. It must win by:

```text
directly writing destination expert-major layout
avoiding unnecessary materialization/reformatting
eventually overlapping phase sends with W13/SwiGLU compute
processing sufficiently large expert_groups per phase
```

Critical measurement:

```text
tiles_per_phase =
    sum_e_in_group ceil(clipped_counts[src, rank, e] / block_m)
  * ceil(I / block_n)
```

If this is too close to or below `num_sms`, the phase is too small.

Initial target:

```text
expert_group_size = 16
```

Do not tune small expert groups until the fused schedule is correct.

## Guidance to Codex

1. Do not use a dynamic receive queue.
2. Do not assume balanced routing.
3. Do not build a scratch-ring path unless it supports variable counts.
4. Do not rewrite the inner WGMMA K pipeline.
5. Treat `expert_group` as the key scheduling abstraction.
6. Use the static phase schedule as the workqueue.
7. Start with single-WG serial correctness.
8. Add two-WG role split only after correctness.
9. Add independent producer/compute loops only after role split works.
10. Swizzling is a later optimization, not part of initial correctness.
