# Expert-Chunked EP Pipeline Investigation

## TL;DR

Test whether expert-level chunking can overlap communication for chunk `n + 1`
with grouped GEMMs for chunk `n`. The first implementation is a disposable
standalone benchmark, not a production backend. It compares a serialized
schedule with a two-slot schedule, inspects the compiled HLO and GPU trace, and
measures forward and forward-plus-backward performance at high expert counts.

The experiment starts with `jax.lax.ragged_all_to_all` because it is the current
scalable EP baseline. A result counts as positive only when a trace shows NCCL
send/receive work overlapping QuACK or another grouped GEMM implementation. If
the primitive cannot expose that overlap cleanly, the next design will use a
transport with explicit asynchronous ownership. Existing SonicMoE, QuACK, and
EP branches are evidence and possible source material; none constrain the
production design.

## Context

Marin's E512/top-8 standalone benchmark scaled from 6.05% MFU at EP1 to
14.39% at EP64 with ragged all-to-all and QuACK under uniform routing. This is
the strongest result so far for large expert count and EP, but it covers one
rack and synthetic routing. See
[#7332](https://github.com/marin-community/marin/issues/7332#issuecomment-5007041962).

The current ragged all-to-all lowering has no public start/wait interface. Its
NCCL path also performs count exchange and host synchronization for each
invocation. Splitting one exchange into several calls may expose overlap, or it
may multiply launch and synchronization overhead. See
[#7012](https://github.com/marin-community/marin/issues/7012#issuecomment-4994519151)
and the
[forced-NCCL measurements](https://github.com/marin-community/marin/issues/7012#issuecomment-4995418725).

The chunked NCCL_EP experiment in #7331 established two constraints. Chunking
can bound an individual communication buffer, and a scanned chunk body must be
rematerialized to avoid retaining residuals for every chunk. That experiment
serialized dispatch, compute, and combine; it did not test overlap. See
[#7331](https://github.com/marin-community/marin/issues/7331#issuecomment-5009585521).

Ordinary NCCL experiments did not show a benefit from limiting communication to
8 or 16 CTAs without a working overlap schedule. The investigation therefore
does not hard-code a 16-SM carveout. SM or CTA limits become sweep parameters
only after the trace proves overlap. See
[#7012](https://github.com/marin-community/marin/issues/7012#issuecomment-4985697941)
and the
[EP8 ring measurements](https://github.com/marin-community/marin/issues/7012#issuecomment-4987885345).

## Goal

Answer these questions in order:

1. Can expert-level chunking make the current ragged all-to-all lowering overlap
   communication and grouped GEMMs on the target GPU topology?
2. Does the overlap survive forward and backward execution and offset the cost
   of additional collectives, count exchanges, launches, and buffers?
3. If ragged all-to-all cannot support the schedule, which synchronization in
   the HLO or trace prevents it?

The investigation produces a benchmark, correctness checks, compiled artifacts,
and traces. A production EP backend is a separate follow-up design selected from
the result.

## Non-goals

- Preserve or upstream an existing experimental EP or SonicMoE branch.
- Make the first benchmark path available from the Grug model configuration.
- Implement a new persistent transport kernel before testing the JAX primitive.
- Tune a communication SM budget before demonstrating communication/compute
  overlap.
- Generalize the initial schedule beyond depths 1 and 2.
- Change routing, capacity, drop, or combine-weight semantics.

## Expert partition and routing semantics

Each EP rank owns `local_experts = num_experts / ep_size` experts. The benchmark
partitions the local expert index range into equal contiguous chunks. A chunk
contains the same local expert indices on every rank. For example, with eight
local experts and two chunks, chunk 0 contains local experts 0--3 on every rank
and chunk 1 contains local experts 4--7.

Routing is computed once for the full token batch. The benchmark gathers global
expert counts, applies receiver-capacity clipping once, and determines the
accepted routes before forming chunks. Chunking partitions those accepted
routes; it does not run capacity clipping independently per chunk. This preserves
the unchunked drop set, sender priority, route order, combine weights, and
`dropped_total`.

The initial benchmark requires `num_expert_chunks` to divide `local_experts`.
Each chunk's ragged communication buffer uses the unchunked receiver capacity as
its static upper bound. This is conservative under skew but preserves semantics
without data-dependent shapes. The trace gate comes before attempts to reduce
that bound. A follow-up production design may choose a different capacity model
only with an explicit numerical and training-quality comparison.

## Schedules

The benchmark exposes two compile-time settings:

```text
num_expert_chunks: number of equal local-expert slices
pipeline_depth:    maximum dispatched-input chunks retained for compute
```

`pipeline_depth=1` is the serialized control:

```text
dispatch(0) -> compute(0) -> combine(0)
dispatch(1) -> compute(1) -> combine(1)
```

`pipeline_depth=2` permits the target schedule:

```text
dispatch(0)
dispatch(1) || compute(0)
combine(0)  || compute(1)
combine(1)
```

The JAX graph uses a two-slot window and data dependencies between windows so
the setting controls the maximum live dispatched-input buffers. It does not
claim overlap from Python operation order. XLA may split and schedule a ragged
collective asynchronously, or preserve a synchronization that serializes the
graph. The compiled HLO and GPU trace decide which occurred.

Depths greater than 2 are rejected in the first benchmark. They add live buffers
and collectives without answering a different architectural question. A later
production configuration may accept larger values if depth 2 is positive and a
deeper schedule improves a measured workload.

## Compute boundary

The schedule consumes a grouped expert-MLP operation with this conceptual
contract:

```text
expert_mlp(
    dispatched_rows,
    local_group_sizes,
    valid_rows,
    w13_expert_slice,
    w2_expert_slice,
) -> output_rows
```

The first GPU benchmark may use the QuACK kernels ported from SonicMoE when they
support the required dtype, architecture, gradients, and expert slice. A plain
ragged-dot implementation remains the correctness oracle. Kernel selection does
not leak into routing or scheduling, and no existing wrapper is preserved solely
because it already exists.

`dispatched_rows` has the static chunk-buffer capacity, while `valid_rows` is
the number of accepted routes in that chunk. The grouped operation must not
treat trailing buffer padding as expert input. This avoids multiplying GEMM work
when a chunk uses the conservative full-receiver-capacity buffer bound.

The production compute boundary will be chosen after the trace gate. If the
cleanest result requires replacing the current QuACK bindings, changing their
custom VJP, or importing a different SonicMoE kernel subset, that work belongs in
the follow-up design.

## Communication boundary and pivot rule

The ragged-all-to-all experiment remains local to the standalone benchmark. It
does not introduce a generic asynchronous transport interface into Levanter.
Such an interface would be premature because `jax.lax.ragged_all_to_all` does
not expose an asynchronous handle.

The ragged path is rejected for pipelining if any of these conditions holds:

- HLO has no asynchronous collective start/done decomposition that can span the
  grouped GEMM.
- The GPU trace shows no NCCL send/receive overlap with grouped GEMMs.
- Forward-plus-backward time does not improve by at least 5% over the best
  equivalent unchunked or serialized configuration after a small chunk-count
  sweep.
- Peak device memory grows with `num_expert_chunks` instead of remaining bounded
  by the configured pipeline depth.

If rejected, the benchmark and trace document the blocking synchronization. The
next design uses the same expert partition and global routing plan with an
explicit asynchronous transport, starting from NCCL_EP evidence in #7331. It
will define ownership of communication buffers, stream/event ordering, and
custom-VJP residuals directly. It will not wrap the synchronous ragged primitive
in additional scheduling abstractions.

## Autodiff and memory

Forward correctness is insufficient because weight-gradient kernels and saved
activations can change the schedule. The benchmark measures both forward and
`value_and_grad` paths.

Each chunk body is rematerialized with `jax.checkpoint(prevent_cse=False)` or an
equivalent boundary demonstrated by the compiled program. Saved state contains
the full-batch routing plan plus at most `pipeline_depth` dispatched-input
buffers and their corresponding compute residuals. It does not retain
dispatched activations for every expert chunk. Transient return buffers may add
another constant-size slot, but memory must remain independent of
`num_expert_chunks` at fixed depth. Peak device memory is recorded for depths 1
and 2 and for every chunk-count setting.

Numerical comparisons cover the output and gradients for input activations,
`w13`, and `w2`. The QuACK comparison uses tolerances justified by its BF16 or
FP8 accumulation behavior; the routing and drop-set comparisons are exact.

## Correctness coverage

CPU-capable routing tests exercise the partitioning independently from ragged
all-to-all and GPU kernels. They cover:

- balanced routing;
- one hot expert;
- an empty expert chunk;
- counts exactly at receiver capacity;
- overflow spanning a chunk boundary;
- nontrivial combine weights; and
- zero accepted rows for one sender-to-receiver pair.

For each case, concatenating chunk plans must reproduce the unchunked accepted
route indices, group sizes, dropped count, and combined output order.

GPU correctness compares depth 1 and depth 2 with the unchunked ragged EP path
on a small multi-device shape. It checks forward values and `dx`, `dw13`, and
`dw2`. Performance runs do not replace these checks.

## Performance experiment

The first target is a one-rack high-expert workload derived from #7332: E512,
top-8, BF16, and EP16 or EP64. The sweep includes:

```text
num_expert_chunks = 1, 2, 4
pipeline_depth    = 1, 2 where num_expert_chunks >= 2
routing           = uniform and deterministic skew
```

Every comparison uses the same model shape, token count, routing assignments,
capacity factor, compiler flags, and device placement. Report compile time,
forward latency, forward-plus-backward latency, tokens/s, MFU, dropped routes,
peak device memory, and per-collective payload sizes. Repeat the performance
measurement across at least three process starts or placements because earlier
all-to-all results varied with placement.

The depth-2 trace must show a time interval where communication for chunk
`n + 1` and grouped GEMMs for chunk `n` execute concurrently. A shorter wall
time without that evidence is recorded as a chunking result, not a pipelining
result.

After overlap is visible, sweep the backend's default communication allocation
against 8 and 16 CTAs or SMs if the selected backend exposes that control. Keep
the default unless a limited allocation improves forward-plus-backward time and
does not reduce achieved transport bandwidth materially.

## Deliverables and decision

The investigation ends with:

1. A standalone benchmark with serialized and depth-2 schedules.
2. Routing and numerical correctness tests.
3. HLO snippets identifying collective start/done and dependency boundaries.
4. One forward and one forward-plus-backward GPU trace for each schedule.
5. A result table for the high-expert sweep and a recommendation recorded on
   #7279.

A positive result leads to a production design for an expert-chunked ragged EP
backend. A negative result leads to a production design for explicit
asynchronous NCCL_EP transport. Neither outcome requires preserving the
standalone prototype or the existing SonicMoE-derived wrappers.
