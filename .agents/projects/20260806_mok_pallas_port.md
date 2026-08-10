# Port Mixture-of-Kittens to Pallas Mosaic GPU

Target: one NVL72 rack (GB200, SM100/SM103), EP64. Forward and backward.
Reference: `cursor/mixture-of-kittens` commit
`3e1cf43ab93ad040afed52a45ab03cb490ffe4be` (Apache 2.0), read but not copied.

## Why this shape

MoK replaces the EP all-to-all with one-sided TMA copies against peer memory,
driven by a precomputed destination-order address table. Two properties carry
the design:

- The scheduler emits, for every row of the local expert-sorted buffer, the
  `(peer_rank, peer_token_idx)` that fills it. There is no runtime offset
  exchange and no capacity negotiation, so the transport is deterministic.
- Dispatch gathers many remote rows into one staging buffer against a single
  mbarrier, so an irregular gather costs one wait rather than one per peer.

Both are expressible in Pallas: `plgpu.remote_ref` supplies a dynamic peer id
per copy, and `plgpu.Barrier(num_arrivals=N)` accounts N outstanding copies.

Handoff between dispatch and the expert GEMM goes through GMEM, not shared
memory. MoK does the same — it stages into an untiled SMEM buffer, transforms,
stores to GMEM, and the GEMM reads tiles back. This sidesteps any need to
gather remote rows directly into MMA-tiled shared memory.

## Static shapes

`schedule_capacity` is a static host argument in MoK as well; `schedule_kernel`
traps when the padded token count exceeds it. Ported behavior: size the table
statically, mark unfilled rows with `peer_rank = -1`, and zero-fill them during
dispatch. Group sizes stay dynamic values feeding `ragged_dot`, which does not
constrain shapes.

## Phases

0. **Scheduler.** Pure JAX port of `scheduler.cuh` count/pad/schedule. No GPU
   needed; validated on CPU against a direct transcription of the CUDA loops.
   Every later phase consumes its output.
1. **Dispatch.** Pallas MGPU. Per destination row, `copy_gmem_to_smem` from
   `remote_ref(x_send, peer_rank)` into an untiled staging tile, N copies on one
   barrier, then `copy_smem_to_gmem` to the local expert-major buffer. Validate
   against a `jnp` gather.
2. **Combine.** Mirror of dispatch: local load, then `copy_smem_to_gmem` into
   `remote_ref(y_recv, peer_rank)`. Completion via `semaphore_signal(device_id=)`
   in place of MoK's remote red/add counters.
3. **Forward integration.** dispatch → `haliax.nn.ragged_dot` → SwiGLU →
   combine. No hand-written GEMM. First landable artifact, directly comparable
   to the existing EP backends.
4. **Backward.** MoK reuses both kernels with roles swapped: dispatch with row
   scaling on `d_y`, combine on `d_x`, router-weight gradient as a remote scalar
   store. Wire through `jax.custom_vjp`.
5. **Fusion.** Persistent grid, comm CTAs split from compute CTAs by
   `pl.program_id`, macrobatch pipelining walked backwards so combine for
   macrobatch i overlaps dispatch for i-1. This is where MoK's speedup lives.
6. **MXFP8.** `scales_layout` / `async_copy_scales_to_tmem`. Note MoK sends BF16
   on the wire because its transposed quantize needs 128 gathered rows to
   coexist; sending FP8 instead halves dispatch bytes and is worth measuring.

## Measured Mosaic constraints (jaxlib 0.11.0, GB200)

Established by running the probe on `cw-us-east-08a`. Each is a hard constraint,
not a tuning knob.

- Remote GMEM refs lower only under `LoweringSemantics.Lane`. Warpgroup lowering
  rejects any GMEM ref carrying a peer id (`primitives.py:931`).
- Every TMA copy moves a whole multiple of 128 bytes, and every vectorised SMEM
  read covers a whole multiple of 128 elements (`fragmented_array.py:1050`).
  Together these force a 128-row dispatch tile, which is also MoK's `DISPATCH_Mb`.
- `cp_async` needs at least 4 bytes per lane, so a gathered row must be at least
  512 bytes. Benign: real hidden sizes are far larger.
- Broadcasting a fragmented vector across the row axis is unimplemented, so the
  padding mask cannot yet run inside the kernel.

**The blocking one: a peer id cannot depend on data.**

- Under TMA, Mosaic replays the peer-id computation on the host to build the
  tensormap (`launch_context.py:879`, `_recompute_peer_id`). A peer id loaded
  from the schedule is an SMEM load and fails with `ReplicationError:
  Unrecognized op can't be recomputed on the host`.
- Under `cp_async`, `gmem_peer_id` is accepted and then never applied — the only
  device-side peer resolution (`to_remote(..., on_host=False)`,
  `launch_context.py:1737`) sits behind `implementation == TMA`. The copy
  silently reads local memory. Measured: the gather returned valid token rows
  from the wrong device, with the padding mask exactly right.

This is precisely MoK's dispatch primitive — 128 rows from up to 128 distinct
peers, addresses taken from a table — and it is not expressible today. It is
also a sharper statement of the blocker dlwh recorded on #6597, which attributed
the problem to WGMMA-tiled shared memory rather than to the peer id.

The silent `cp_async` drop is worth an upstream report: accepting a peer id and
ignoring it produces wrong numerics with no diagnostic.

Two workarounds failed before the working one. Predicating an individual copy
makes barrier arrivals conditional, which the barrier's fixed transaction count
cannot express, and unrolling `block_rows * axis_size` copies fails the MLIR
pass pipeline at both two and four peers.

## What works

`mok_peer_schedule.py` + `mok_peer_gather.py`, green on four GB200s:

```
[mok-peer] PASS: 2048 rows exact across 4 peers (1024 carrying tokens)
```

Three properties together clear every constraint above:

- The receive buffer is segmented by source rank, so a grid block belongs to one
  rank and the peer is a Python literal. TMA then resolves it on device.
- Predication is at block granularity. Every row inside a taken branch copies
  unconditionally, so barrier arrivals stay exact.
- The row loop is a `fori_loop`, not 128 unrolled copies, which keeps the pass
  pipeline small.

The cost is MoK's round-robin interleave across source ranks within an expert
segment, which spreads a destination tile's reads over many peers. Recovering it
needs a peer id Mosaic cannot express, so it waits on upstream.

## Forward pass

`mok_ep_schedule.py` + `mok_ep_mlp.py` run a whole expert-parallel MoE MLP
forward pass, green on four GB200s against a dense reference:

```
dtype=float32  relative_norm_err=2.7e-05
```

Segmenting the receive buffer by (local expert, source rank) buys two things at
once. The rank stays a Python literal, as above, and every expert holds exactly
`world_size * rows_per_pair` rows, so the expert MLP is a dense batched einsum
instead of a ragged dot. Fixed capacity per (expert, rank) pair replaces MoK's
256-row expert padding.

Combine pushes each expert result straight into slot `token * topk + k` of its
owner's buffer, so contributions never collide and the weighted top-k sum is a
local epilogue with no atomics. The kernel ends with a rendezvous so a peer's
writes are visible before the epilogue reads them.

Known gap: the combine buffer is not zero-initialised, so a schedule that drops
assignments would leave those slots undefined. `ep_moe_mlp_forward` documents
this and the probe fails fast when `dropped` is non-zero. Fix before any run
where capacity can overflow.

## Backward pass

`ep_moe_mlp_backward` runs both transports again with their roles swapped, which
is how MoK reuses dispatch and combine rather than writing new kernels:

- the transpose of combine is a gather of each contribution's cotangent from its
  owner, so it is `dispatch` indexed by `dest_slot`;
- the transpose of dispatch is a push back into per-`(token, k)` slots, so it is
  `combine` indexed by `dest_slot`.

Because every contribution owns a distinct slot, neither direction needs a remote
accumulation and both top-k sums stay local.

Checked against `jax.vjp` of the dense reference on four GB200s:

| quantity | float32 | bfloat16 |
| --- | --- | --- |
| forward | 2.7e-05 | 5.5e-03 |
| `d_x` | 1.8e-04 | 6.2e-03 |
| `d_router_weights` | 2.2e-05 | 6.0e-03 |
| `d_w_gate` | 1.8e-04 | 6.3e-03 |
| `d_w_up` | 1.8e-04 | 5.6e-03 |
| `d_w_down` | 1.3e-04 | 5.7e-03 |

Every bfloat16 figure sits at the same 0.6% the forward pass costs, so the
backward adds no error of its own beyond precision.

Reproduce with `lib/levanter/scripts/bench/probe_mok_ep_mlp.py`, optionally
`--dtype bfloat16`.

## Risks

- Phase 1 is the crux: N copies with dynamic per-row peer ids against one
  barrier is unproven in Pallas. If it does not hold, the gather degrades to
  per-peer grouped copies and loses the single-wait property.
- Peer refs require NVSHMEM and multi-process. Single-process multi-device sets
  `num_peers = 0`, so the debug loop needs real multi-process from the start.
- The device mesh must use row-major device ids.
- Phase 5 may not be expressible in one Pallas grid. If not, the port caps at
  transport parity rather than MoK's overlap gain.

## Baselines to beat

- `#7891`: exact MoonEP on one NVL72, 9.94% MFU, ragged all-to-all 55.7% of step.
- `#7670`: ECHO two-chunk EP64, 22.64% p50 MFU at 2.51% drop.

## Differentiable wrapper and training equivalence

`ep_moe_mlp` wraps the forward and backward in `jax.custom_vjp`, so `jax.grad`
drives the peer transports directly. The schedule is integer-valued and returns
`float0` cotangents.

Twenty SGD steps against the dense reference from one initialisation, learning
router weights and all three expert tensors:

```
loss fell 0.526702 -> 0.427798;  worst per-step gap 2.45e-06
router_weights  relative_norm_err 1.3e-07   disagreement/move 9.7e-06
w_gate          relative_norm_err 7.2e-07   disagreement/move 5.0e-05
w_up            relative_norm_err 6.8e-07   disagreement/move 5.0e-05
w_down          relative_norm_err 3.4e-07   disagreement/move 4.6e-05
```

`disagreement/move` is the honest ratio: how far the two runs diverge against how
far training actually moved the parameter. Agreeing to 1e-4 means nothing if
neither run moved.

## CUDA graph capture corrupts peer transports

**Callers must set `--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN` before
importing JAX.** Pallas lowers to custom calls, so omitting `CUSTOM_CALL` is
enough; fusions and GEMMs keep their command buffers. Measured equivalent to
disabling capture outright over twenty training steps, which matters because
these kernels should not cost the whole model its launch-overhead wins.

XLA captures a computation into a CUDA graph after roughly ten executions and
replays it with the buffer addresses recorded at capture time. It cannot know
these kernels write into peer buffers, so a replayed graph targets stale peer
addresses.

The symptom is badly misleading. Every single-shot probe passes. In a training
loop the first eight steps are exact, step nine returns a gradient 2000x too
large, step ten is fine again, step eleven is wrong. With the learning rate set
to zero -- every step then an identical computation -- steps 9 and 11 are still
wrong and the rest exact, which is what finally ruled out any value-dependent
explanation. Disabling command buffers makes all twenty steps exact.

Worth reporting upstream alongside the silent `cp_async` peer-id drop: both fail
without any diagnostic.

Two dead ends recorded so they are not retried. A `pl.get_global` semaphore
persists across launches, so an absolute `semaphore_wait` threshold only
synchronises on the first launch. And a barrier built from an arithmetically-zero
marker is folded away by XLA; the collective must be tied to the value with
`optimization_barrier` to survive, which is why the first two barrier attempts
produced bit-identical wrong answers.

## Capacity overflow

`outgoing_keep_mask` closes the unwritten-slot gap. A slot index is the running
count of a rank's own earlier assignments to the same global expert, so whether
an assignment is dropped depends only on that rank's routing and not on which
rank owns the expert. Each rank derives its own mask with no extra
communication, and forward and backward mask their unwritten combine slots.

Validated on four GB200s with capacity deliberately overflowed -- 1024 tokens per
rank into one expert per rank at 128 rows per pair, so only 2048 of 8192
assignments survive:

```
forward 2.6e-05   d_x 2.4e-04   d_router_weights 2.3e-05
d_w_gate 2.4e-04  d_w_up 2.4e-04  d_w_down 1.7e-04
```

Two CPU tests pin the invariant the fix rests on: the locally-derived mask must
agree exactly with what every receiver's schedule actually kept, across three
capacity settings.

## Endpoint concentration

Found by an independent review, not by me. The grid walked segments in the same
order on every rank, so all ranks read source rank 0 first and stepped in
lockstep -- one peer's memory saturated at a time while the rest of the fabric
idled. `_rotated_segment` offsets the peer index by `lax.axis_index`, which is a
permutation of the peer axis and so leaves the block-to-segment map a bijection.

Scheduling model at EP64 (one block per SM, in-order dispatch), peak ranks
reading one peer concurrently:

```
before   1024   6.92x a perfectly spread schedule
after     148   1.00x
```

That is a model, not a measurement, and #7279 says the regime is latency-bound
with bandwidth headroom, so relieving concentration need not convert into
throughput one-for-one. Revalidated on four GB200s: forward/backward parity,
capacity overflow, and twenty training steps all unchanged to within noise,
which is the expected signature for a pure reordering.

## Column chunking (fixes a hard blocker)

The kernel could not compile at production hidden size. A whole-row staging
buffer at H=5120 asks for 2.62 MB against 227 KB:

```
ValueError: Mosaic GPU kernel exceeds available shared memory:
            smem_bytes=2622480 > max_smem_bytes=232448
```

Every probe until now used H=256, which fit in 65 KB and hid this completely.

The fix is MoK's design, not a workaround: MoK never copies whole rows.
`DISPATCH_Nb = 512` (`mok_megakernel.cuh:44`) makes a tile 128 rows x 512
*columns* and walks the hidden dimension as separate tasks (`:379-387`). Both
kernels now do the same.

One further Pallas constraint, tighter than MoK's: a TMA box is capped at 256
elements per dimension (`launch_context.py:1778`), so where MoK uses 512-column
chunks we use 256 and issue twice as many copies. Measured at H=5120 / I=2048,
bfloat16, forward and all five gradients:

```
forward 5.4e-03   d_x 6.2e-03   d_router_weights 5.7e-03
d_w_gate 6.2e-03  d_w_up 5.6e-03  d_w_down 5.6e-03
```

## Open: intermittent corruption, roughly one run in four

**The transport is not correctness-safe for training yet.** Four 20-step runs of
identical code and configuration: three clean, one corrupted a single step.

```
step 3: gap=1.65e-06   |g_k|=0.075807  |g_v|=0.075807
step 4: gap=9.13e-02   |g_k|=0.065491  |g_v|=0.073569   <- one bad gradient
step 5: gap=2.48e-03   ... tracks in parallel thereafter, permanently offset
```

Sporadic rather than deterministic, which rules out a logic error in the peer
rotation or the column chunking and points at a residual race. The parameter
checks still passed, so only the loss-gap check caught it -- a weaker test would
have called this run green.

This corrects an earlier claim in this file. The CUDA-graph capture hazard is
real and the scoped `--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUDNN`
setting was declared equivalent to disabling capture on the strength of a
*single* clean 20-step run. One sample is weak evidence against an intermittent
fault. A run with capture disabled outright also passed here, which likewise
proves little at n=1.

This outranks parallel issue and macrobatching. A transport that silently
corrupts one step in four cannot be used, and the failure is invisible to
per-tensor parameter checks.

## One-hour MoK-fidelity pass (2026-08-07)

Three gaps closed, each verified against the dense reference on four GB200s.

**Shared expert (MoK's `w_shared_*`).** We were missing an entire architectural
component. MoK's epilogue is `output = y_shared + sum_k w[k] * combine[k]`
(`utils.cuh:100-110`): a dense SwiGLU over every local token, no transport,
added unweighted. Now in forward, backward and the custom VJP. All eight
gradients match dense autodiff, the three new ones at 6.4e-06 / 7.8e-06 /
2.1e-07.

**Parallel copy issue.** MoK issues one copy per thread -- `is_worker = tid <
DISPATCH_Mb` with 128 threads each issuing their own (`mok_megakernel.cuh:376,
416-419`) -- where we issued serially from one thread. Pallas issues from a
warpgroup leader, so the closest analogue is four warpgroups each driving an
independent slice of the tile: its own staging rows, schedule slice and barrier,
so no cross-warpgroup synchronisation is needed. Column chunking made this more
urgent, since a 256-element cap means 10-20x more copies than MoK issues.

**Replicated-gradient handling.** Not a kernel bug but a real one: the shared
expert is replicated per rank, so `jax.grad` yields each rank's local gradient
and SGD walks every replica in a different direction. Training now all-reduces
it, as real training would. Caught only because the training probe compares
against a globally-differentiated reference.

Verified after all three: gradient probe 1.8e-05, production H=5120/I=2048
bfloat16 5.3e-03, capacity overflow 3.1e-05.

## Corruption rate

Before the 2026-08-07 fidelity pass: ten 20-step runs, nine clean, one corrupted
a single step -- **roughly 1 in 10**, not the 1 in 4 the first four samples
suggested.

Afterwards: **19 runs, 0 failures**. That is suggestive but not conclusive -- at a
10 percent rate, 19 consecutive clean runs occur about 13 percent of the time by
chance. Nothing in the fidelity pass targeted this bug, so any improvement would
be incidental. Do not record it as resolved on a clean streak; prefer finding the
mechanism. Note also that the failure only ever showed up in the *training*
probe: single-shot correctness probes cannot see it, because one corrupted step
still leaves per-tensor parity within tolerance.

## Dynamic `expect_bytes` is closer than it looked

A review concluded MoK's runtime transaction count is inexpressible. The dialect
is more capable than that under the semantics we already use
(`primitives.py:4510-4531`): under Warpgroup lowering `arrive_expect_tx` takes
tx_bytes as an attribute and is static-only (TODO b/415721295), but under **Lane**
lowering -- which remote refs force anyway -- it is passed an `ir.Value`
produced by a `select`, i.e. genuinely dynamic. So the gap is Pallas API
exposure, not dialect capability. Reaching it needs a new Pallas primitive or
dropping to raw Mosaic GPU; it is not blocked by hardware or by MLIR.

## Correctness sweep after the fidelity pass

All on four GB200s, against the dense autodiff reference:

| case | relative norm error |
| --- | --- |
| forward + 8 gradients, float32 | 1.8e-05 |
| forward + 8 gradients, bfloat16 | 5.4e-03 |
| top-k 4, different seed | 1.7e-05 |
| 2 blocks/pair, multiple column chunks | 1.5e-06 |
| production H=5120 / I=2048, bfloat16 | 5.3e-03 |
| capacity overflow, 75 percent dropped | 3.1e-05 |
| 20 SGD steps vs dense autodiff | 1.8e-06 worst per-step |

`test_mok_block_cols.py` pins the two tile constraints that cost a GPU
round-trip each to discover: a whole row does not fit in shared memory at
production hidden sizes, and a TMA box is capped at 256 elements per dimension.

## Tensor cores under Lane semantics: demonstrated, not a blocker

An adversarial review named this the kill switch. Remote refs only lower under
`LoweringSemantics.Lane`, so if MMA required Warpgroup then no Pallas kernel
could hold both peer transport and tensor-core compute, N2 would be unbuildable,
and the ceiling would be transport-only parity -- worth nothing, since #7670
shows ECHO-ragged already matches NCCL on throughput.

Measured on a GB200 using JAX's shipped `blackwell_matmul_mgpu`, changing only
the lowering semantics:

```
default: RUNS rel_err=0.00165
Lane:    RUNS rel_err=0.00165
```

Identical numerics, so Lane is not merely accepted. **Peer transport and tensor
cores can share one kernel.** Reproduce with
`lib/levanter/scripts/bench/probe_lane_mma.py`.

Three earlier probes said the opposite and were all wrong, which is worth
recording because the failure mode is seductive:

- the first used WGMMA operands without swizzle/tiling transforms, and printed
  "MMA does NOT lower under Lane" for what was a usage error;
- the second fixed the transforms and failed identically under *Warpgroup*, the
  shipped configuration, and printed the same verdict anyway;
- the third flipped semantics on `hopper_matmul_mgpu` -- but WGMMA is an sm90
  instruction and cannot compile for sm100 at all, so the whole family of
  attempts was testing the wrong instruction for this hardware.

A self-authored verdict line agreeing with the hypothesis under test is worth
less than a shipped baseline. Always check that the known-good control passes
before believing the experimental arm's failure.

Remaining blockers for N2 are unchanged: no dynamic `expect_bytes` exposed
through Pallas (the dialect supports it under Lane), and the peer id having to be
host-recomputable.

## Next lever: `blackwell_ragged_dot_mgpu`

JAX ships a Blackwell grouped GEMM at
`jax/experimental/pallas/ops/gpu/blackwell_ragged_dot_mgpu.py`, entry point
`ragged_dot_kernel(a, b, group_sizes, config)`. That is MoK's expert path, and a
direct fix for the one place a review called this port *structurally* behind:
our dense einsum runs over the whole padded receive buffer, so capacity headroom
costs real FLOPs, where MoK's grouped GEMM walks `tokens_per_expert` and pays
nothing for it. Adopting a tested kernel beats hand-writing MMA, and it does not
depend on N2 resolving favourably.

## Resolution of the three open issues (2026-08-07)

### 1. Intermittent corruption -- no longer reproducible, not provably fixed

Could not be reproduced on current code. **1,380 consecutive clean steps**
(19 x 20-step runs, then 5 x 200-step runs), against one observed failure in
roughly 80 steps of the older state. Under the rate that failure implies, 1,380
clean has probability well under one percent, so the code genuinely changed --
most plausibly as an incidental effect of the shared-expert or warpgroup work,
since neither targeted it.

A system-scope release fence was added to combine regardless, because the
argument stands on its own: those stores land in *peer* memory, and the
collective barrier orders execution, not visibility. Pallas exposes no fence
primitive, but `semaphore_signal` emits `fence_release_sys` after its atomic, so
signalling a throwaway semaphore is the only route to one. The semaphore is
never waited on, so its value accumulating across launches is harmless.

**Do not record this as fixed.** There is no failing case to demonstrate the
fence repairs anything, and absence of a repro is not a root cause. If it
returns, the useful facts are: it only ever appeared in the *training* probe
(single-shot correctness cannot see it, because one bad step still leaves
per-tensor parity in tolerance), and 200-step runs are the cheapest exposure.

### 2. Grouped GEMM -- measured, and not worth doing

A review called the dense einsum the one place this port is *structurally*
behind MoK: capacity headroom costs real FLOPs here, and nothing in MoK. That is
true at small token counts and nearly vanishes at production scale.

| tokens/rank | rows_per_pair | padded rows | real rows | FLOP waste |
| --- | --- | --- | --- | --- |
| 1,024 | 128 | 32,768 | 8,192 | 4.00x |
| 8,192 | 384 | 98,304 | 65,536 | 1.50x |
| 65,536 | 2,176 | 557,056 | 524,288 | **1.06x** |

At 4.19M tokens/rack with no microbatching the waste is six percent. Capturing it
with `blackwell_ragged_dot_mgpu` additionally needs a compaction pass, because
padding is interleaved *per (expert, source rank) cell*: a grouped GEMM keyed on
true per-expert counts would read padding and miss real rows. Spending bandwidth
to recover six percent of MLP FLOPs is not worth it. Revisit only if token counts
per forward drop by an order of magnitude.

### 3. Megakernel (N2) -- unblocked, out of scope for now

The feasibility question is answered: tcgen05 runs under Lane lowering, measured
on the shipped `blackwell_matmul_mgpu` with identical numerics either way, so
peer transport and tensor cores can share one kernel.

The two remaining constraints -- no dynamic `expect_bytes` through Pallas, and a
host-recomputable peer id -- both already have working block-level workarounds in
this kernel. What N2 additionally needs is a persistent grid with SM
partitioning by `pl.program_id` plus the backwards macrobatch loop, and N2 is not
separable from N1. It is now a scoping decision, not a feasibility one.

An earlier draft of this section cited #6841 -- 130k lines, stalled -- as evidence
that N2 is a multi-week slog. That comparison was unfair and is withdrawn. #6597
opened 2026-06-23 and #6841 on 2026-07-02, last active 2026-07-11; MoK was
published in August 2026. **dlwh had no MoK reference.** He was searching the
design space, which his logbook shows directly: rolling-slot protocols redesigned
repeatedly, schedules reverted, and an explicit correction that the 35.85 ms
floor had been mistaken for the architecture's ceiling.

Implementing a published design is a different and cheaper problem than finding
one. MoK gives exact tile sizes, the destination-order schedule, the comm/compute
split and the macrobatch loop. What #6841 still legitimately indicates is that
the *implementation* is substantial regardless, and that he would have hit the
same Mosaic constraints recorded above.

## N1 macrobatching -- implemented and validated

MoK's macrobatch ring buffer, ported. Chunking is by expert, so each chunk's rows
stay contiguous and its expert weights are disjoint; only one chunk's routed
activations are live at a time, so peak memory falls by `num_chunks`. That is the
lever for the 16.9 GB/layer the production token count implies.

The blocker recorded earlier -- a chunk writes only its own experts' combine
slots, the rest are never written, and the buffer is not zero-initialised -- is
resolved by `slot_chunk_index`. The owning chunk follows from a rank's own
routing (`global_expert % num_local_experts`), so it needs no communication, the
same insight as [[outgoing_keep_mask]].

Backward mirrors the loop, masking each chunk's cotangent and concatenating the
disjoint per-chunk weight gradients. `num_chunks=1` is the default and is
bit-identical to the unchunked path.

| | forward | d_x |
| --- | --- | --- |
| `num_chunks=1` | 1.7935e-05 | 1.1863e-04 |
| `num_chunks=2` | 1.7814e-05 | 1.1822e-04 |

Twenty SGD steps at two chunks track the dense reference to 1.58e-06. Chunking is
numerically equivalent, not an approximation.

### What N2 still needs

With N1 landed the remaining gap is a single fused kernel:

1. one launch with `grid = comm_ctas + compute_ctas`, branching on
   `pl.program_id`, mirroring MoK's `cluster_idx < comm_clusters`
   (`mok_megakernel.cuh:1593`);
2. cross-CTA sequencing inside that kernel -- MoK uses GMEM counters
   (`barrier_arrive`/`barrier_wait`); Pallas semaphores provide the equivalent;
3. the expert GEMM callable from inside that kernel -- **resolved**, see below.

## The expert GEMM is vendored, not written

An earlier draft of this file claimed `blackwell_ragged_dot_mgpu` is "a whole
kernel, not a callable fragment", so a fused megakernel would need `tcgen05`
written from scratch. That was wrong. `do_matmul` is a reusable fragment: it
emits one output block's matmul given refs, grid indices, scratch and barriers,
which is exactly how `ragged_dot_kernel` calls it. It is warp-specialised the
same way MoK is -- TMA warp, MMA warp, compute and store warpgroups -- with TMEM
double-buffered accumulators.

Vendored to `_moe/mok_expert_gemm.py` (Apache 2.0, dual attribution). Three
deviations from upstream, all forced:

- `GroupInfo` inlined from `ragged_dot_mgpu` so this does not depend on a
  benchmark file;
- **two upstream bugs on the non-collective path**, which apparently is untested:
  the body reads `lax.axis_index("x")` unconditionally while the launch only
  declares that axis under `collective`, and the call site always passes three
  grid indices while `do_matmul` unpacks two unless collective. A fused
  megakernel wants the non-collective path, so both are fixed here;
- lowering semantics parameterised rather than hardcoded to Warpgroup, since a
  fused kernel needs one setting for transport and compute.

Measured on a GB200, all four configurations identical at `rel_err=0.00169`:

```
non-collective default: RUNS      non-collective Lane: RUNS
collective     default: RUNS      collective     Lane: RUNS
```

**The expert GEMM runs under Lane, so it can share a kernel with peer transport.**
Reproduce with `lib/levanter/scripts/bench/probe_mok_expert_gemm.py`.

`GroupInfo` also handles genuinely ragged groups, so feeding it true per-expert
counts closes D5 without the separate compaction pass costed earlier.

What N2 now needs is integration, not new kernel authorship: one launch with
`grid = comm_ctas + compute_ctas`, branching on `pl.program_id`; semaphore
sequencing between them; and `plgpu.nd_loop(..., collective_axes="sm")` for the
persistent grid, which the reference already uses.

## EP64 validation -- XLA requests an unused multicast pointer

The conventional-collective path is a correctness control, not the solution.
It changes MoK's sparse peer traffic into a global `all_gather` followed by
`psum_scatter`, and cannot provide MoK-like scaling or overlap.

`metadata_only` in `lib/levanter/scripts/bench/probe_mok_remote_pointer.py`
makes a multiprocess Mosaic parameter participate in symmetric-memory setup but
never dereferences it. The traced EP64 run
(`/held/mok-metadata-trace-ep64-1830`) initialises the 64-rank NCCL communicator
and registers a 188 GB symmetric VA window, then fails before peer-address
lookup or kernel launch:

```
ncclGetLsaMultimemDevicePointer(win_, 0, &multimem): unhandled system error
```

JAX 0.11 marks every multiprocess Mosaic kernel parameter as a symmetric-memory
argument. Its Mosaic custom call then asks each registered parameter for both a
multicast address and every peer address. The multicast lookup is unconditional
even when the compiled kernel uses only `remote_ref`; `remote_ref` needs peer
addresses but does not set `kernel->is_multimem_used`.

That differs from MoK's explicit split in `mok/functional.py:208-251`: ordinary
activation and combine buffers expose per-rank `buffer_ptrs`, while only the
routing all-gather and barrier buffers request `multicast_ptr`. The minimal XLA
fix is therefore to call `multimem_addr()` only when
`kernel->is_multimem_used`, while retaining peer-address lookup for
`remote_ref`.

The first patched build produced only `jax-cuda13-plugin`. Installing it in
`/held/mok-metadata-peeronly-ep64-2244` left the failure unchanged because that
wheel contains Mosaic's compiler extension, while `MosaicGpuInitialize` is
linked into `jax-cuda13-pjrt`. The runtime wheel must carry the patch as well.
A CUDA 13 PJRT build from JAX commit
`a1521744c6dc074443fe549f19f48d7197abf759` is running on the GB200 cluster;
the EP64 probe will install both patched wheels before validation.

The patched compiler and PJRT wheels were both installed successfully in
`/held/mok-metadata-pjrt-patched-ep64-0035`. The unconditional multicast lookup
is gone: initialization now advances to peer-address enumeration and fails at
`ncclGetPeerDevicePointer`. The runtime identifies itself as NCCL
`2.28.9+cuda13.0`, so this run does not establish anything about NCCL 2.30.7.
The next transport test must locate the failing rank-count boundary and repeat
on 2.30.7 before changing the Pallas schedule or accepting a collective path.

NCCL 2.30.7 resolves that peer-address enumeration failure. In
`/held/mok-metadata-pjrt-nccl2307-ep64-0831`, the runtime reports
`2.30.7+cuda13.3`, and the metadata-only kernel reaches execution. Devices 0--31
complete while devices 32--63 raise `CUDA_ERROR_ILLEGAL_ADDRESS`. The boundary
comes from XLA, not NCCL or Pallas: XLA commit
`131bf41acb4650e4391a640c3f1859c1c86ad74b` hard-codes both multi-GPU barrier
variants to `kMaxPeers = 32`. The NCCL-backed launch omits the participant-count
check present in the raw-pointer variant, launches 32 threads with
`num_ranks = 64`, and `SyncRemoteBlocks` indexes its 32-entry signal-pointer
array by the global rank. Ranks 32--63 therefore access the array and the
32-slot symmetric signal allocation out of bounds. The cluster-only PJRT
rebuild raises both barrier capacities to the physical NVL72 size and adds the
missing bound check.

That rebuild succeeded as `jax-cuda13-pjrt` SHA-256
`e0325f8a81482055286b19e65834f00511f26a46dbd36409a636497d778dd4e7`.
With NCCL `2.30.7+cuda13.3`, `/held/mok-metadata-pjrt-nvl72-ep64-0903`
passes on all 16 processes / 64 GPUs. The direct transport probe
`/held/mok-remote-pjrt-nvl72-ep64-0905` also passes remote loads and remote TMA
from every rank to peers 0, 31, 32, and 63. The 32-rank collective fallback is
therefore removed; the full EP64 MoE validation now exercises peer transport.

The full direct routed path passes as well. In
`/held/mok-direct-ep64-f32-0908`, all 16 processes complete forward and every
backward gradient for 8,192 tokens over 64 ranks; the largest rank-0 relative
error is `4.191e-05`. `/held/mok-direct-ep64-bf16-0912` also passes, with the
largest rank-0 relative error `0.0064563`. Both runs retain all
16,384 assignments. EP64 correctness is closed; schedule density and launch
overlap are now the performance-critical work.

The existing collective checks remain useful controls: the routed forward and
every backward gradient match dense autodiff at EP64 in float32
(`/held/mok-collective-fallback-ep64-1525`) and bfloat16
(`/held/mok-collective-fallback-ep64-bf16-1530`), and the direct EP8 path is
exact (`/held/mok-direct-regression-ep8-1528`). They do not close the EP64 or
performance work. Incident record: https://echo.oa.dev/wiki/94.

## Pinned MoK parity audit

Three current differences are performance-significant:

1. MoK's `scheduler.cuh:34-145` counts assignments per expert and peer, pads
   each expert's aggregate count to 256 rows, and writes one variable-length
   expert-major schedule. `mok_ep_schedule.py` instead reserves a fixed number
   of rows for every `(expert, peer)` pair, can drop overflow, and makes dense
   GEMMs pay for all reserved rows.
2. MoK's forward communication clusters interleave combine for macrobatch `i`
   with dispatch for `i-1` inside one persistent kernel
   (`mok_megakernel.cuh:1593-1633`). The current Pallas path launches dispatch,
   expert computation, and combine sequentially.
3. MoK's backward communication clusters interleave reverse-dispatch, router
   preload, reverse-combine, and activation replay
   (`mok_megakernel.cuh:2162-2209`). The current custom VJP performs the same
   mathematical transposes but not that device-side pipeline.

Restoring EP64 peer pointers removes the immediate launch blocker. Matching
MoK performance still requires the aggregate expert schedule, bounded ring
buffers, and persistent communication/compute overlap.

## Dynamic bulk transport and bounded workspace (2026-08-08)

The earlier peer-id limitation applies to host-built TMA tensor maps, not to
Mosaic's simple contiguous bulk-copy path. With
`OOBFillMode.PROMISE_IN_BOUNDS`, `copy_gmem_to_smem` resolves both a scheduled
peer id and row index on the device and emits one bulk copy. No JAX patch is
needed. `/held/mok-dynamic-peer-bulk-ep64-0947` passed scheduled peers 0, 31,
32, and 63 on all 64 ranks. The experimental `cp_async` patch was removed;
lane copies were correct but emitted roughly 128 small instructions per
512-byte row and are not a performance path.

The aggregate dispatch and combine now use that dynamic bulk path. Workspace
capacity is static, while `num_routed_tokens` bounds the active row prefix.
The grouped GEMM independently derives its persistent iteration count from
`sum(tokens_per_expert)`, so neither transport nor compute has to execute the
unused capacity. `/held/mok-ragged-dynamic-grid-smoke4-1005` passed with a
2,048-row allocation and 1,024 active rows.

The first bounded combine corrupted 13--14 output slots per device. The root
cause was inactive CTAs issuing the system-fence semaphore signal without
joining a completion rendezvous. Moving the fence into the active branch fixed
the four-rank case, but the outer `psum` used as the device rendezvous then
failed twice at EP64 during NCCL communicator initialization, before Pallas
kernel entry (`/held/mok-active-fence-ep64-1025` and `-1028`).

The final transport follows MoK's peer-visible barrier structure without that
collective. Every CTA signals each peer's regular semaphore and waits until all
peer CTAs have arrived. Dispatch does this before remote reads; combine does it
after remote TMA stores. Results:

- `/held/mok-p2p-barrier-smoke4-1032`: exact gather/combine parity for
  4 x 4,096 rows on four GB200s.
- `/held/mok-p2p-barrier-ep64-1035`: all 16 processes succeeded, with exact
  gather/combine parity for 64 x 4,096 rows across the 64-rank NVL domain.

EP64 jobs require 96 GB host memory per four-GPU task. At 64 GB, PJRT symmetric
memory initialization reaches roughly 66 GB RSS and is cgroup-OOM-killed.

This closes correctness for MoK's aggregate, runtime-addressed transport over
the full EP64 domain. It is not yet performance parity: the current unicast
barrier costs `axis_size * grid_blocks` signals per rank, whereas MoK's
`barrier_all` uses one multicast reduction (`utils.cuh:262-310`), and the
dispatch, grouped GEMMs, and combine are still separate launches rather than
the persistent overlapped schedule in `mok_megakernel.cuh:1593-1633`.

The complete aggregate forward path also passes at EP64. The first attempt,
`/held/mok-aggregate-mlp-ep64-1045`, stopped before compilation because the
gather wrapper incorrectly required the local source-token count to be a
multiple of its 128-row destination tile. Removing that unrelated constraint
allowed 16-token shards while retaining the 4,096-row padded workspace.
`/held/mok-aggregate-mlp-ep64-1047` then passed on all 16 processes / 64 GB200s
with relative norm error `0.0053126` against the dense BF16 reference. This
validates dynamic aggregate dispatch, three expert-major ragged GEMMs,
aggregate combine, router weighting, and the shared expert together across the
full NVL domain.

## Multicast rendezvous and production transport (2026-08-08)

Pallas exposes the same primitive as MoK's multicast barrier:
`semaphore_signal_multicast` lowers to
`multimem.red.release.sys.global.add.u32`, matching `barrier_all` in
`utils.cuh:262-310`. The earlier conclusion that Pallas could only signal every
peer separately was wrong.

The first multicast probe exposed two independent runtime problems. JAX's
default preallocated device arena made NCCL try to map another roughly 148 GB
symmetric window for even a tiny multicast semaphore, which failed VA placement
under NCCL 2.30.7. Setting `XLA_PYTHON_CLIENT_PREALLOCATE=false` makes the
allocation small and `/held/mok-multicast-no-prealloc-smoke4-1065` passes.

The original rendezvous used one scalar semaphore and made every CTA wait for
`axis_size * grid_blocks` arrivals. A production tile grid has 1,680 CTAs per
device, so resident CTAs waited before the remaining CTAs could launch and the
kernel deadlocked. The replacement allocates one semaphore slot per tile. Each
CTA multicast-signals its own slot and waits only for 64 devices, so it has no
whole-grid residency requirement. `/held/mok-multicast-slots-grid448-1070`
passes with a grid larger than SM residency, and
`/held/mok-multicast-slots-ep64-1075` passes exact gather and scatter across all
64 ranks.

This resolves the suspected NCCL 2.30.7 EP64 LSA failure in the tested
environment: peer pointers and multicast mappings both work. The concrete
failures were XLA's unconditional multicast lookup, its 32-peer launch barrier,
JAX preallocation exhausting symmetric VA space, and the transport kernel's
whole-grid barrier.

At MoK's default communication shape -- 2,048 tokens per rank, hidden size
7,168, top-k 6, 384 experts, and six local experts --
`/held/mok-multicast-prod-localdata-ep64-1085` completes on all 64 GB200s but
measures 3.561 ms for gather and 4.218 ms for scatter. This is not a viable MoK
performance result. The current Pallas kernel emits 128 row copies from one
warpgroup leader in a Python loop, while MoK's dispatch workers issue one row
copy from each of 128 threads concurrently (`mok_megakernel.cuh:344-480`). The
next transport change must recover that issuing parallelism before launch
fusion or expert overlap can matter.

## MoK worker-thread transport (2026-08-08)

The missing issuing parallelism was in JAX's Lane lowering, not the hardware.
Omitting `predicate` from `LaunchContext.async_copy` selects the warpgroup
leader, so the first thread-parallel prototype still issued only one TMA while
its barrier expected 128. `/held/held-mok-thread-one-tx-smoke4` completed when
the barrier expected one row and exposed that mismatch. Passing
`predicate=None` explicitly makes all 128 workers issue their own TMA, matching
MoK dispatch at `mok_megakernel.cuh:392-430`.

Pallas Lane barriers also multiply `num_arrivals` by 128 physical lanes. The
working lowering therefore keeps `Barrier(num_arrivals=1)`, has one lane retire
the other 127 physical arrivals without completing the phase, and registers
the aggregate transaction bytes. For predicated rows it uses NVVM's
`barrier.reduction popc`, the direct lowering of MoK's
`__syncthreads_count(peer_rank >= 0)`, and expects only the valid workers'
bytes. Each valid worker then loads its scheduled remote row and stores its own
output row. Combine uses the same 128-worker ownership in the reverse
direction, matching `mok_megakernel.cuh:480-620`.

Correctness controls progressed in order:

- `/held/held-mok-thread-dynamic-correct4`: exact dynamic peer and token gather
  across four GB200s.
- `/held/held-mok-thread-combine-correct4`: exact 128-worker gather and combine.
- `/held/held-mok-thread-predicated-correct4`: exact predicated aggregate byte
  count and per-worker transfers with 75% padding rows.
- `/held/held-mok-predicated-correct64-64g`: exact gather and combine for
  `64 x 1,024` scheduled rows across all 16 processes / 64 GB200s.

The old EP64 correctness probe's task-0 exit 137 was not a transport failure.
The same 48 GB task limit still kills task 0 before kernel entry, while the
otherwise identical 64 GB run completes exact gather and combine on every
rank. The probe now builds only each process's four schedules and computes the
expected result from deterministic peer/token encodings, avoiding replicated
all-rank reference materialization. The remaining threshold is PJRT/XLA host
memory during full-domain setup and compilation.

At the production transport shape, the unpredicated 128-worker path measures
0.864 ms gather / 0.806 ms scatter on four GB200s
(`/held/held-mok-thread-transport-prod4`) and 3.622 ms / 2.097 ms on 64 GB200s
(`/held/held-mok-thread-transport-prod64`). Predicating padding transfers, as
MoK does, improves the four-GPU gather to 0.797 ms while scatter remains
0.801 ms (`/held/held-mok-thread-predicated-prod4`). The corresponding EP64
run measures 2.128 ms gather / 2.111 ms scatter and succeeds on all 16
processes (`/held/held-mok-predicated-prod64`). This closes the serial-issuer gap, but not MoK's
`COMBINE_PIPE_DEPTH` staging or persistent dispatch/compute/combine overlap at
`mok_megakernel.cuh:1593-1633`.

## MoK combine tiling and pipeline (2026-08-08)

Combine now uses MoK's exact `COMBINE_Mb=16`, `COMBINE_Nb=1024`, and
`COMBINE_PIPE_DEPTH=7`. Each communication CTA launches all seven local
GMEM-to-SMEM stages, then waits and performs the scheduled remote stores one
stage at a time, matching `mok_megakernel.cuh:480-620`. The schedule remains a
direct GMEM scalar load per worker as in MoK. Copying the 16-row int32 schedule
through Lane TMA was both unnecessary and invalid: the initial
`/held/held-mok-combine16-correct4` compile rejected its 64-byte transfer
because generic Lane TMA requires a byte count divisible by the 128-worker
warpgroup. This was a probe-design error, not a transport limitation.

`/held/held-mok-combine16-directschedule4` passes exact gather/combine with one
stage, and `/held/held-mok-combine-pipe7-correct4` passes exact gather/combine
with all seven stages and hidden size 7,168. At the production transport shape,
the seven-stage path measures 0.805 ms gather / 0.767 ms combine on four GB200s
(`/held/held-mok-combine-pipe7-prod4`) and 2.118 ms / 1.987 ms across all 64
GB200s (`/held/held-mok-combine-pipe7-prod64`). Relative to the predicated
16-row-unaware EP64 path, MoK's combine tiling and staging reduce combine from
2.111 ms to 1.987 ms.

The aggregate expert MLP remains correct after the transport change.
`/held/held-mok-aggregate-combine16-correct4` passes on four GB200s with
relative norm error `0.0053609`. The full-domain
`/held/held-mok-aggregate-combine16-correct64` passes on all 16 processes / 64
GB200s with relative norm error `0.0053126`. This validates MoK-shaped
dispatch, three expert-major ragged GEMMs, MoK-shaped pipelined combine, router
weighting, and the shared expert together. The remaining performance gap is
the same one visible in MoK's forward main loop at
`mok_megakernel.cuh:1593-1633`: the Pallas implementation still launches
communication and expert compute separately instead of reserving communication
CTAs and overlapping combine of macrobatch i with dispatch of i-1 while compute
CTAs process minibatches.

## Production aggregate baseline (2026-08-08)

The aggregate probe now has a process-local benchmark path, so every host
materializes only its four devices' expert weights and schedules. This makes it
possible to run the exact BF16 shape from MoK's pinned
`benchmarks/bench_mok.py:10-18`: 2,048 tokens/rank, H=7,168, I=3,072, 384
experts, top-k 6, and six local experts at EP64. The schedule workspace uses
15,360 rows per rank; random routing produces 12,288-13,824 padded active rows,
so no route is dropped.

`/held/held-mok-aggregate-prod4` measures 4.745 ms for the complete unfused
forward on one GB200 tray. `/held/held-mok-aggregate-prod64` measures 7.155 ms
for the same per-device workload across all 64 GB200s. These measurements
include dispatch, three BF16 expert ragged GEMMs, SwiGLU, combine, router
weighting, and the shared expert. They do not include schedule construction,
matching MoK's `benchmark_fwd` boundary.

The 7.155 ms number is a baseline, not the performance target. Isolated EP64
transport accounts for about 4.105 ms (2.118 ms dispatch + 1.987 ms combine),
and the current JAX program serializes those launches around the expert GEMMs.
MoK instead launches 256-thread CTAs, reserves communication CTAs inside the
same persistent kernel, and overlaps communication with compute at minibatch
granularity (`mok_megakernel.cuh:55-59, 1593-1633`). The next implementation
step is therefore the direct MoK CTA-role split, not tuning the serialized
fallback.

## Persistent dispatch/TMEM interaction (2026-08-08)

The persistent kernel reproduces a deferred CUDA fault only when a
thread-parallel remote TMA gather and TMEM allocation coexist. Schedule-only,
store-only, zero-route, shared-memory-only, and standalone TMEM allocation
controls exit cleanly. Gather-only and complete dispatch finish their device
work, but CUDA reports `CUDA_ERROR_LAUNCH_FAILED` or
`CUDA_ERROR_ILLEGAL_INSTRUCTION` during the kernel's TMEM teardown.

Comparing the lowering directly with MoK's dispatch at
`csrc/mok_megakernel.cuh:385-421` found a real translation bug: NVVM's
`barrier.reduction popc`, used for MoK's `__syncthreads_count`, ran only in the
first 128-thread warpgroup of a 256-thread CTA. The dispatch now makes all 256
threads participate in the count and wait, predicates the second warpgroup out
of the 128 worker loads, registers aggregate bytes from thread 0, and executes
the CTA-wide synchronization that MoK places between `expect_bytes` and the
worker TMA loads. `/held/held-mok-cta-count-wait4` and subsequent full-dispatch
controls complete rather than hanging, but the deferred teardown fault remains.

The remaining allocator differences were tested independently against
ThunderKittens' `tensor_allocator<1, 2>` at
`include/types/tensor/tensor.cuh:36-149`: retaining the allocation address,
direct i32 `tcgen05.dealloc.cta_group::2`, post-allocation cluster
release/acquire sync, and warp-0 cluster release/acquire immediately before
deallocation. None removed the fault, including the combined exact lifecycle
in `/held/held-mok-full-exact-lifecycle4`. The next diagnostic must inspect the
generated Mosaic module/PTX around the remote `cp.async.bulk`, mbarrier phase,
and deallocation rather than changing allocator instructions again.

## Persistent root cause and cluster-2 compute (2026-08-08)

The teardown fault was caused by a second dispatch lowering bug, not TMEM
allocation. The thread-parallel gather registered its aggregate byte count
under JAX's `single_lane_predicate`, which selects one lane per 128-thread
warpgroup. A 256-thread MoK CTA therefore executed `arrive_expect_tx` twice,
while MoK executes it once under `threadIdx.x == 0`
(`csrc/mok_megakernel.cuh:385-410`). The generated SASS showed both warpgroup
leaders reaching `SYNCS.ARRIVE.TRANS64`. Using a block-scoped leader produces
one CTA-level arrival. `/held/held-mok-block-leader4` exits without a deferred
CUDA error, `/held/held-mok-dispatch-correct4` matches every BF16 gathered value
exactly, and `/held/held-mok-block-leader-ep64` passes exact local-shard checks
on all 16 processes / 64 GB200s with clean TMEM teardown.

The first full persistent compile then exposed the expected architectural
mismatch directly: ptxas rejected single-CTA MMA (`cta_group::1`) in the same
kernel as CTA-pair TMEM (`cta_group::2`). MoK uses cluster-2 MLP tasks
throughout. The persistent scheduler now assigns compute by cluster rather
than SM, passes the CTA rank into the collective GEMM, waits on the matching
128-row dispatch block from each CTA, and uses the same logical 256x256 MLP
tile as MoK. `/held/held-mok-cluster2-full4` passes the complete forward against
the dense reference. At the pinned production shape,
`/held/held-mok-cluster2-prod4` measures 3.457 ms, down from the previous
4.745 ms x4 aggregate baseline. The EP64 production timing is queued as
`/held/held-mok-cluster2-prod-ep64`.

## MoK CLC scheduler and fused routed forward (2026-08-08)

`/held/held-mok-cluster2-prod-ep64` completed on all 16 processes / 64 GB200s
at 6.481 ms. This confirmed that static cluster-2 compute removed little of
the EP64 serialization cost relative to the 7.155 ms unfused baseline.

JAX 0.11 exposes Blackwell Cluster Launch Control through
`plgpu.dynamic_scheduling_loop`, including cluster cancellation and the
logical task grid that MoK implements in `csrc/scheduler.cuh:34-145`. The
persistent compute CTAs now use that scheduler directly. Routed MLP tasks use
MoK's 256x256 logical tiles, K=64, six-stage operand pipeline, 32-column
epilogue, cluster size two, and supergroup-eight snake mapping. The mapping is
Pallas `planar_snake(..., tile_width=8)`, which is equivalent to ThunderKittens
`get_swizzled_2d_idx<8>`.

`/held/held-mok-clc-gateup4` passes the four-GPU dense reference with relative
norm error `0.0053393`; `/held/held-mok-clc-prod4` measures 2.383 ms for the
aggregate forward while SwiGLU, down projection, and combine are still outside
the CLC task chain. Porting MoK's 128x128 three-stage SwiGLU task and the down
projection into the same scheduler passes with relative norm error `0.0048945`
(`/held/held-mok-clc-down4`) and measures 2.597 ms
(`/held/held-mok-clc-down-prod4`).

The communication CTAs now execute MoK's 16x1024, seven-stage combine after
dispatch. Each combine task waits for the down-projection minibatches it reads,
uses explicit per-stage mbarrier parity, launches valid remote stores as their
stages complete, waits for SMEM store readers, and then reuses the aliased
staging region. The explicit-parity Pallas primitive was extended to accept the
same indexed barrier references as JAX's standard barrier operations; the first
attempt failed at compile time because this transform handling was absent.
`/held/held-mok-clc-combine4c` passes four-GPU dense parity at relative norm
error `0.0048945`, and `/held/held-mok-clc-combine-prod4` measures 2.503 ms at
the pinned production shape. The unchanged kernel is running at EP64 as
`/held/held-mok-clc-combine-ep64`.

## MoK BF16 backward (2026-08-08)

The backward port now follows MoK's scheduler order at
`csrc/mok_megakernel.cuh:2039-2460`: shared down dgrad, two-stage 128x128
SwiGLU backward, fused gate/up dgrad, three shared wgrads, then the equivalent
routed stages with reverse-combine and reverse-dispatch on 28 communication
SMs. Dgrad and wgrad use separate normal and K-major BF16 SMEM aliases, as in
MoK, while reusing the same barriers and collective TMEM. This removed the
transposed-destination TMA lowering failure.

`/held/held-mok-bwd-correct4b` passes a staged BF16 reference for `d_x`,
`d_router`, all routed weight gradients, and all shared weight gradients. The
largest relative norm error is `0.0066425`. The same small shape measures
`0.925 ms` for fused forward plus backward in `/held/held-mok-bwd-prod4`.

The initial backward rejected capacity above the 4,096-row minibatch. MoK only
replays after its 131,072-row macrobatch; minibatches are readiness and task
scheduling units inside that buffer (`globals_bwd::grid` and
`mok_bwd_kernel`). The guard now uses MoK's macrobatch boundary. A 5,120-5,632
active-row reproducer exposed that reverse-combine readiness was incorrectly a
single semaphore even though dispatch signals by minibatch. Readiness is now
allocated and waited per minibatch, including MoK-style wgrad waits for every
overlapping minibatch.

The multi-minibatch reproducer still stalls in the reverse-combine/routed-dgrad
region after that fix. Stage-cut diagnostics narrowed the fault below routed
SwiGLU and wgrad, but comm-only variants are not conclusive because removing
the compute clusters changes the patched collective TMEM lifecycle. The next
implementation step is to port MoK's exact per-minibatch routed task mapping
and per-minibatch reverse-dispatch readiness instead of extending the current
global-block approximation. The pinned EP64 workload remains below one
macrobatch and therefore does not require replay.

The queued fused-forward production run
`/held/held-mok-clc-shared-ep64` completed on all 16 processes / 64 GB200s.
Every rank finished at the target 12,288-13,824 active routed rows within the
15,360-row workspace, and the complete aggregate forward measured 5.809 ms.
This validates the full NVL72 LSA mapping and multi-minibatch forward path. It
is 18.8% faster than the 7.155 ms unfused EP64 baseline and 10.4% faster than
the 6.481 ms static cluster-2 implementation.

The routed backward scheduler now ports MoK's saved-macrobatch mapping at
`csrc/mok_megakernel.cuh:2361-2397` directly: each 4,096-row minibatch runs
down dgrad, two-stage SwiGLU backward, and fused gate/up dgrad before the next
minibatch, followed by the three routed wgrad matrices. Reverse-dispatch
readiness is also indexed and waited per minibatch, matching MoK's combine
overlap instead of using one global completion barrier.
`/held/held-mok-bwd-permini-correct4` preserves the small-shape staged-reference
errors, with `d_router` again the maximum at `0.0066425`.

The H=512/I=256 multi-minibatch smoke
`/held/held-mok-bwd-permini-multi4` stalls, but the matching forward-only job
`/held/held-mok-fwd-small-multi4` stalls before backward as well. That shape is
an undersized scheduler probe, not evidence about the production backward.
The exact MoK workload succeeds on four GB200s in
`/held/held-mok-bwd-permini-target4`: H=7,168, I=3,072, and 12,800-13,568
active routed rows/rank complete the fused forward and backward. The EP64
timing run is `/held/held-mok-bwd-permini-ep64`.

The first EP64 attempt reached the 64-device banner and was then preempted.
Iris atomically rescheduled all 16 tasks, but the retry stalled inside JAX
distributed initialization before any application banner or kernel launch;
all ranks remained sleeping at about 420 MB RSS. The job was terminated rather
than recording the retry as a kernel failure.

The benchmark boundary now follows MoK's `benchmarks/utils.py:92-111`: it runs
forward to produce the saved context before starting the backward timer, then
times only backward. The forward context uses MoK's 24 communication SMs and
backward uses 28. `/held/held-mok-bwd-boundary4` validates that split boundary
at the exact workload and measures backward-only at 5.212 ms. The clean EP64
run is `/held/held-mok-bwd-boundary-ep64`.

`/held/held-mok-bwd-boundary-ep64` subsequently completed on all 16 processes
and 64 GB200s with NCCL 2.30.7. Every rank completed fused forward and
backward at 12,288-13,824 active routed rows, and the MoK-style backward-only
boundary measured 10.775 ms. This timing still includes Pallas's external
`d_x` reduction and router-gradient einsum and uses the probe's zero-valued
performance inputs, so it is a functional full-NVL72 result rather than a
native-MoK parity claim.

Cluster-side Ruff 0.14.3 formatting and lint pass under Levanter's own Ruff
configuration in `/held/held-mok-cluster-lint5`. No Pallas or JAX compilation
was run on the local host.

Pinned native MoK commit `3e1cf43ab93ad040afed52a45ab03cb490ffe4be`
with ThunderKittens commit `1c3920d993404dd49a6d4c7267ea11d583bd5c68`
builds for SM100 with PyTorch 2.10.0+cu130 in the CUDA 13.0.2 devel image on
the cluster. `/held/mok-native-sm100-artifacts3-0808` serves the resulting
wheel and a source archive containing MoK's unmodified benchmark. Native
validation runs as a separate four-GPU process in
`/held/held-mok-native-sm100-x4e`; it is not linked into the JAX/Pallas path.
That run passes MoK's BF16 and MXFP8 correctness checks and reports BF16
forward at 1.876 ms and backward at 3.574 ms for EP4 with six local experts.
The corresponding 16-tray native run is
`/held/held-mok-native-sm100-ep64`; it uses the same MoK commits and benchmark
with NCCL 2.30.7.

The Pallas performance probe now mirrors MoK's `tests/utils.py:40-100` input
distribution: unique top-k routes from Gaussian logits, softmax router
weights, random BF16 activations and scaled weights, and an independent scaled
`d_output`. `/held/held-mok-bwd-random4` passes at the exact EP4 workload and
measures 5.382 ms over 100 timed iterations after 100 warmups. The comparable
pinned native MoK BF16 result is 3.574 ms, leaving a 50.6% gap. The next port
target is therefore MoK's in-kernel router path at
`csrc/mok_megakernel.cuh:308-455, 768-1125, 2162-2209`, replacing the external
Pallas `d_slots` materialization and router-gradient einsum.

The pinned native MoK EP64 run `/held/held-mok-native-sm100-ep64b` completes
on all 16 tasks / 64 GB200s with NCCL 2.30.7. BF16 forward is 1.947 ms and
backward is 3.659 ms; MXFP8 forward is 1.492 ms and backward is 2.522 ms. BF16
correctness passes, including `d_router` mean absolute error `1.782839e-03`,
maximum absolute error `1.150912e-02`, and relative norm error
`3.747770e-03`. This rules out NCCL 2.30.7 or EP64 LSA mapping as the cause of
the Pallas failures and establishes the full-domain performance target.

Router weights and router-gradient partials now use separate buffers, matching
MoK's `router_weights` and `d_router_weight_partials`. The previous combined
buffer had an overlapping GMEM race: the router preload wrote columns 0:3
while routed SwiGLU wrote gradient partials beginning at column 1. The split
buffer path `/held/held-mok-drouter-split4ap` restores small-shape correctness,
but exact-shape jobs still stalled in every tile-wide Pallas reduction variant.
Transposed contiguous stores, explicit shuffles, local-only accumulation,
fused product/reduction, saved-hidden algebra, and MoK's 256-register setting
did not remove the stall. The exact transposed store-only control
`/held/held-mok-bwd-drouter-transstore4au` succeeds at 5.661 ms, isolating the
remaining failure to the tile-wide router-gradient computation rather than the
output store.

The routed BF16 SwiGLU backward now ports MoK's literal scalar implementation
at `csrc/mok_megakernel.cuh:1040-1115`. Its 256 threads map as one thread per
row and two 64-column halves, perform the same scalar BF16 gate/up/d-hidden
loads and SiLU derivatives, accumulate one FP32 router partial per half, and
join the halves through MoK's `(2, 128)` FP32 shared scratch and CTA barrier.
`/held/held-mok-drouter-scalar4be` passes small-shape correctness: relative
norm errors are `0.0046019` for `d_x`, `0.0072234` for `d_router`, `0.0044239`
for routed gate wgrad, `0.0044067` for routed up wgrad, and `0.003435` for
routed down wgrad; shared wgrad errors are about `1e-4`. The exact EP4 run
`/held/held-mok-bwd-drouter-scalar4bf` succeeds at 6.504 ms. A stability run
with 100 warmups and 100 timed iterations,
`/held/held-mok-bwd-drouter-scalar4bg`, succeeds at 6.438 ms.

The remaining router-gradient transport is intentionally isolated in
`_scatter_router_gradients`: it sums the SwiGLU column partials, scatters them
into owner/token slots, and uses XLA `psum_scatter`. This is not performance
equivalent to MoK's `combine_kernel` at `csrc/mok_megakernel.cuh:500-610`,
which sums the same partials and writes the scalar directly to peer memory
inside reverse-dispatch while the token chunk is already in flight. The
unchanged literal-scalar kernel is queued at EP64 as
`/held/held-mok-bwd-drouter-scalar-ep64`; its result must be compared with the
3.659 ms native MoK BF16 backward rather than described as parity.

Four combine-side probes isolate why that final MoK operation cannot yet move
into the Pallas megakernel. `/held/held-mok-drouter-combine4bj` adds MoK's
partial sum and direct peer scalar store beside the working remote `d_x` TMA;
it stalls before the first aggregate call. The local-sum control
`/held/held-mok-drouter-combine-local4bk` also stalls, as does the zero local
scalar-store control `/held/held-mok-drouter-combine-zero4bl`. Routing the
result through a padded 16-byte remote TMA in
`/held/held-mok-drouter-combine-tma4bm` stalls at the same boundary. The
failure therefore does not depend on the sum arithmetic, peer address, or
4-byte store width; adding the side effect to Mosaic's pipelined combine CTA is
the common trigger. All four jobs were terminated and the source was restored
to the correctness- and stability-proven external `psum_scatter` boundary.

Cluster-side Ruff 0.14.3 formatting and lint pass for the restored megakernel
and probe in `/held/held-mok-cluster-lint-scalar3`. No compilation, JAX execution, or
lint ran on the local host.

JAX commit `7cf5d762ab088506f9087ce4574ebac4f3923063` extends Mosaic GPU's
inline and atomic-store lowering paths to accept peer references. The matching
distributed test zeros the destination, synchronizes peers, performs a remote
`plgpu.atomic_add`, and synchronizes completion. The JAX 0.11 source used by
this port already contains the equivalent `allow_peer_refs=True` lowering.
`/held/held-mok-remote-atomic4` confirms that public vector remote atomic add
works across all four GPUs with no mismatched local devices.

Public scalar `plgpu.atomic_add` does not lower because the scalar carries a
splat layout (`NotImplementedError: Atomic stores not supported for splat
layout`). The production path therefore uses `plgpu.inline_mgpu` to issue the
same internal atomic-store operation used by JAX's public lowering, with the
peer-reference support above. This is the direct Pallas encoding of MoK's
per-thread peer scalar addition, not a different collective or scheduler.

Reverse combine now follows pinned MoK
`csrc/mok_megakernel.cuh:470-610`: column block zero sums the two routed
SwiGLU partials, maps each valid route back to its owner rank and token, and
atomically adds the scalar router gradient to peer memory while reverse
dispatch is active. As in MoK's `peer_rank >= 0` branch, invalid rows perform
no atomic operation. Communication SM zeroing, a peer rendezvous before use,
and the existing combine rendezvous provide the initialization and completion
ordering. The output is padded only to make the communication-CTA zeroing
layout legal and is sliced back to `num_slots + 1` at the wrapper boundary.

`/held/held-mok-drouter-atomic4j` passes the small-shape correctness check:
relative norm errors are `0.0052068` for `d_x`, `0.0075442` for `d_router`,
`0.0041606` for routed gate wgrad, `0.0041822` for routed up wgrad,
`0.0034374` for routed down wgrad, about `0.00356` for shared gate/up wgrad,
and `0.00010939` for shared down wgrad. At the exact EP4 workload,
`/held/held-mok-bwd-drouter-atomic4k` measures 6.438 ms over 100 warmups and
100 timed iterations, equal to the prior external-`psum_scatter` baseline.
Avoiding atomics for invalid rows removed the regression in the first version
and matches MoK's branch behavior.

The production EP64 validation is
`/held/held-mok-bwd-drouter-atomic-ep64`, submitted as 16 coscheduled x4 GB200
tasks with the exact 2,048-token, H7168, I3072, top-k-6, six-local-expert
workload and MoK's 24/28 forward/backward communication-SM split. It replaces
the old scalar-`psum_scatter` and standalone atomic queues. It is currently
scheduling-gated while Iris waits for a coherent 16-node NVLink domain; no
EP64 success or performance result is claimed yet.

Cluster job `/held/held-mok-cluster-lint-atomic3` passes Black and Ruff for the
megakernel and probes. The full cluster pre-commit pass still reports the
pre-existing pyrefly errors in the untracked megakernel; no local compilation,
JAX execution, formatting, or lint was run.

Pinned MoK's BF16 routed SwiGLU backward at
`csrc/mok_megakernel.cuh:1040-1115` performs 16 iterations per thread, loading
and storing four contiguous BF16 values from one 64-column half on each
iteration. The first Pallas vector port materialized an entire 128-by-64 half
per warpgroup. Its initial noncanonical layout failed compilation in
`/held/held-mok-swiglu-vector4a`; the canonical whole-half layout passed the
small probe in `/held/held-mok-swiglu-vector4b` but stalled all exact-shape
runs. This rules out whole-half register materialization despite its nominal
vector width because it does not match MoK's per-thread register footprint.

The production path now streams the same `(row, half, iteration)` ownership as
MoK. Its `(128, 4)` custom register layout gives each of the 256 threads one
row and one four-BF16 vector, and the loop repeats exactly 16 times. The first
streaming version passed small correctness and measured 5.854 ms in a one-shot
exact run, but array-form shared/global router-partial stores caused a
nondeterministic single-GPU stall during long runs
(`/held/held-mok-swiglu-vector4d` through `4h`). The stable version extracts
the one-element reduction to a per-thread scalar, stores the second half to
MoK's 128-float shared scratch, synchronizes the CTA, and has the first half
store the summed scalar partial. This matches MoK's scalar transport as well as
its vector access pattern.

`/held/held-mok-swiglu-vector4i` passes the small backward correctness probe
with relative norm errors `0.0052068` for `d_x`, `0.0075442` for `d_router`,
`0.0041606`, `0.0041822`, and `0.0034374` for routed gate/up/down wgrads,
`0.0035672` and `0.0035528` for shared gate/up wgrads, and `0.00010939` for
shared down wgrad. `/held/held-mok-swiglu-vector4j` completes 100 warmups and
100 timed exact-shape iterations at 5.706 ms. This is 0.732 ms or 11.4% faster
than the 6.438 ms scalar baseline, but remains 2.132 ms or 59.7% slower than
pinned native MoK's 3.574 ms EP4 BF16 backward.

The earlier EP64 atomic snapshot was replaced by
`/held/held-mok-swiglu-vector-ep64`, which contains the stable MoK-width
streaming path. It remains scheduling-gated with all 16 tasks waiting for a
coherent NVLink domain; no EP64 result is claimed. No compilation, execution,
formatting, or lint ran on the local host.

The nondeterministic backward stalls after the streaming SwiGLU port came from
two remaining departures from pinned MoK and ThunderKittens. First, Pallas had
two alternating TMEM accumulators while MoK's GEMM at
`csrc/mok_megakernel.cuh:1127-1500` owns one collective accumulator and lets
only the lead block wait for its store completion. The Pallas GEMM now uses the
same single-accumulator lifecycle and lead-block barrier ownership. Second,
the backward epilogue did not reproduce ThunderKittens' vector-TMA protocol.
MoK's seven `sv_bf<1024>` inputs are each four 256-element TMA operations, but
ThunderKittens registers their total byte count once on a one-arrival barrier
and commits each output vector's four stores as one TMA group. The old Pallas
translation registered 28 arrivals and issued one 1024-element store.

`jax_grouped_tma_expect.patch` adds static `arrive` and `expect_bytes`
arguments to Mosaic GPU's GMEM-to-SMEM copy primitive so the epilogue can
encode the exact ThunderKittens contract: the first of 28 loads registers the
combined byte count, the other loads do not arrive, and the four 256-element
stores share one commit group. A direct 1024-element load is not a substitute;
the cluster compiler rejects it because Mosaic TMA copies support at most 256
elements per dimension. Grouped loads alone progressed farther but still
stalled at backward iteration 304 in
`/held/held-mok-bwd-grouped-expect-trace500-4`. Grouped loads and stores pass
500 iterations in `/held/held-mok-bwd-grouped-io-trace500-4` and an independent
500-iteration repeat in `/held/held-mok-bwd-grouped-io-repeat500-4`.

The wrapper-level rendezvous now ports MoK's `barrier_all` from
`csrc/utils.cuh:262-310` as a one-block Pallas kernel using the same system-scope
multimem reduction, alias-proxy fence, and acquire wait. The exact barrier and
grouped epilogue pass 500 backward iterations in
`/held/held-mok-bwd-native-barrier-grouped500-4`. The final small-shape
correctness run `/held/held-mok-final-correctness-4` passes all outputs:
relative norm errors are `0.0045542` for `d_x`, `0.0072234` for `d_router`,
`0.0044239`, `0.0044067`, and `0.003435` for routed gate/up/down wgrads, and
`0.0001219`, `0.00012351`, and `0.00010414` for shared gate/up/down wgrads.

The reverse-combine outgoing peer `d_x` copies retain one deliberate Pallas
lowering difference: they use a full GMEM wait rather than MoK's read-only
wait. Repeated XLA launches otherwise reuse the source before the remote store
has completed. All other scheduler, readiness-count, barrier, atomic router
publication, and communication ordering follows the pinned MoK source.

At the exact EP4 workload, the current grouped-I/O implementation with MoK's
multimem wrapper barrier measures 5.715 ms over 100 warmups and 100 timed
iterations in `/held/held-mok-native-barrier-perf100-4`. Pinned native MoK is
3.574 ms, so the Pallas port remains 59.9% slower; replacing the barrier with a
scalar XLA collective measured 5.665 ms and does not explain the gap. The
current source is submitted unchanged for a 16-node / 64-rank NVL72 run as
`/held/held-mok-exact-ep64-perf20`. No compilation, JAX execution, formatting,
or lint ran on the local host.

`/held/held-mok-exact-ep64-perf20` acquired the full 16-node / 64-GPU domain
after a 4.5-hour capacity wait, initialized all 16 JAX processes, and then
failed during tracing before compiling the megakernel. GNU patch refused every
JAX target because the Iris environment had installed the Python package as
symlinks (`File ... is not a regular file -- refusing to patch`). The launch
shell lacked `set -e`, so it continued with stock JAX and failed on the first
patched API call: `copy_gmem_to_smem(..., predicate=...)`. This result says
nothing about EP64 kernel correctness or performance. A retry must install the
pure-Python `jax` package with copy link mode and abort on any failed patch
before initializing the distributed runtime.
