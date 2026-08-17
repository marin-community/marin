# mixture-of-kittens: reaching the real model shape, then scale

Every mok_like measurement so far was taken on a shape nobody trains. The backend rejects
`latent_dim`, and both the current production model and the proposed 8/384 architecture are latent
MoE, so the matched comparison has been pinned to eight experts with no latent projection. The
numbers from that shape are real but they answer a question no one asked: 23.69% MFU dropless on one
node, and 21.87% MFU dropless at EP64 on one rack, both against a 26.2835% reference measured on a
different model.

Latent support therefore comes first. It is not a feature request behind the performance work; it is
the precondition for the performance work meaning anything.

## Why latent support also relieves two other blockers

LatentMoE narrows only the routed path, from `hidden_dim` 6144 to `latent_dim` 3072. That halves the
bytes on the expert-parallel wire and shrinks the routed activations and the arena by the same
factor. Two current blockers are downstream of that width:

- Forty-eight layers at EP64 does not fit: rematerialization reports 240.81 GiB against a 158.13 GiB
  budget. A narrower routed path is the largest single lever available on that number.
- The 8/384 proposal doubles the active experts and therefore the comms. Larry's position in
  the Slack thread is that it is only viable if the comms hide under shared-expert compute, which is
  exactly what the megakernel fuses.

So the sequence below is not arbitrary: the width change plausibly unblocks the depth gate, and the
fusion it must preserve is what makes the target architecture worth attempting at all.

## Phase 0 -- land what is already done

The working tree holds two schedule regression tests, a CPU drop probe, and three launcher
tolerance overrides (`--max-retries-preemption`, `--max-retries-failure`, `--max-task-failures`).
Lint is clean and 51 tests pass. The overrides are what made the one-rack measurement reachable at
all, so they gate every later phase that needs a rack.

## Phase 1 -- express the architecture

Stand the parity harness up first. `mok_like_correctness.py` and `mok_like_stateful_parity.py` need
a GPU session with the training environment; a bare `iris job run` fails because the image carries a
CPU-only jaxlib. Nothing in this phase is trustworthy without them, because the change lands in
megakernel task-index arithmetic where an off-by-one corrupts silently rather than raising.

Take the two-width shape, not de-fusing. De-fusing the shared expert would express latent MoE with
no kernel change, by mirroring what the non-mok path already does, but it forfeits the
shared/routed overlap that is the reason to use this backend. Keep the fusion:

- The FFI takes separate routed and shared activations with separate widths. Today it takes one `x`
  and derives a single `hidden_dim` for both (`mok_forward_ffi.cu` around the `y_shared`/`y_routed`
  globals).
- The backward grid derives routed task counts from `d_y_shared.cols()`
  (`mok_megakernel.cuh:281`, mirrored in the device dispatch near 2106). Split
  `hidden_dim_col_blocks` into shared and routed. The forward grid already derives every routed
  quantity from routed tensors and needs no change.
- Move the shared-plus-routed sum out of the FFI epilogue into JAX, so the routed result is expanded
  by `w_latent_up` before it is added to the shared output.
- Drop the `latent_dim` rejection in `model.py` and pass the narrowed routed input.

The shared tensors appear at roughly 112 sites across `api.py` and the FFI. The work is wide and
mechanical rather than deep, and the reference implementation in `api.py` must move with it so the
parity gate keeps comparing like with like.

### What is already done

Three steps landed, each verified before the next began, and all of them are inert until a caller
passes two widths:

- `mok_like_reference` takes `routed_x` and `latent_up`, runs the experts at the latent width, and
  expands before adding the full-width shared output. Two CPU tests.
- `validate_mok_like_inputs` checks the routed weights against the routed input rather than against
  `x`, and requires `latent_up` to span the two widths. That check was what made latent MoE
  unrepresentable. Three CPU tests.
- `ForwardEpilogueKernel` is width-agnostic and takes `y_shared == nullptr` to mean "the caller
  composes", so the two widths never meet inside it. The parity gate reproduced its pre-change
  numbers to the last decimal.

### How the two widths were separated

The design question was where the shared expert's full-width input lives once the routed path stops
sharing its buffer. The answer is that it needs no home at all. `globals_fwd.x_shared` is only ever
a rank-local GEMM operand -- the megakernel reads it at `expert_grouped_gemm_kernel` for the shared
gate, up and wgrad, and never through a peer mapping, which is `x_routed_send_buffer`'s job. So
`x_shared` now points straight at the XLA input, and the arena carries only routed tensors. Neither
a second staging buffer nor a second peer region is required, and the same argument applies to
`d_y_shared` in the backward.

That makes the arena strictly smaller than the earlier estimate. `ComputeArenaLayout` is dominated
by `x` and `d_y` at `tokens * hidden` and by `combine` and `d_x_routed` at `tokens * top_k * hidden`,
so it is `4 * tokens * routed * (1 + top_k)` bytes. At the production EP64 shape -- 65,536 local
tokens, hidden 6144, top-4 -- that is 7.50 GiB per rank today and 3.75 GiB at a 3072 latent:

| | today | with latent |
| --- | --- | --- |
| routed terms | 7.50 GiB | 3.75 GiB |
| shared staging | none | none |
| total | 7.50 GiB | 3.75 GiB |

The arena halves rather than growing, so latent support relieves the depth gate's memory pressure
instead of adding to it -- the same argument that put this phase first.

### What landed

- **FFI arity.** `ForwardBf16` takes `routed_x` beside `x`; `BackwardBf16` takes `routed_x` and
  `grad_routed_output`, and returns `d_routed_x` beside `d_x`. Every routed global, weight shape and
  peer registration follows `routed_dim`; the shared ones keep `hidden_dim`.
- **Composition moved to JAX.** The forward's `output` is now the routed combine alone, at the
  routed width, and `y_shared` is returned beside it. `mok_like_mlp` expands the first through
  `latent_up` and adds the second, which is what lets the widths differ. The cost is one extra bf16
  rounding of the routed term and one elementwise add per layer.
- **Backward task decomposition.** `hidden_dim_col_blocks` served both paths. Six `build.py` patches
  add `routed_dim_col_blocks` from `d_x_routed.cols()` and move the routed dgrad gate/up and all
  three routed wgrads onto it, in both the host `grid()` and the device prologue. The forward needed
  no equivalent edit -- it already derives every routed count from the routed weights.
- **`model.py`.** The `latent_dim` rejection is gone. The down-projection and its norm now run
  before the backend split, so both paths share one projection, and the fused call receives
  `routed_x` and `w_latent_up` directly.
- **Gates.** A CPU test pins the routed and shared widths through both traced FFI ABIs, where a
  mistake would otherwise land as a silent out-of-bounds read on device.
  `mok_like_correctness` gained `--latent-dim`, which puts `routed_x` and `latent_up` under the same
  gradient comparison as every other leaf.

The verification loop is a parity-gate run of roughly ten minutes:

```bash
uv run iris --config lib/iris/config/cw-us-east-08a.yaml job run --no-wait \
  --enable-extra-resources --priority interactive --cpu 32 --memory 200GB --disk 64GB \
  --gpu GB200x4 --extra gpu --timeout 5400 --max-retries 0 --job-name <name> -- \
  bash -lc "uv pip install --python .venv/bin/python nvidia-cuda-cccl==13.0.85 \
    nvidia-cuda-crt==13.0.88 nvidia-cuda-nvcc==13.0.88 nvidia-cuda-runtime==13.0.88 \
    nvidia-nvvm==13.0.88 && .venv/bin/python -m experiments.grug.moe_hero_ep.mok_like_correctness \
    --forward-repeats 4 --forward-chained"
```

The GPU spec must be `GB200x4` rather than a bare count, `--extra gpu` is what supplies the CUDA
jaxlib, and the five build packages are what the training path injects and a bare job does not.
Append `--hidden-dim 512 --latent-dim 256` for the two-width arm.

Baseline for the single-width arm: `dropped_assignments 0`,
`ffi_call_counts {forward: 4, backward: 4}`, `schedule_capacity 2816`. `loss_absolute_error` was
11.6529541015625 before the composition moved into JAX and should shift slightly rather than match,
because the routed term now rounds to bf16 once more; judge it against the reference tolerance.

### Gate results

| gate | single width | latent 512/256 |
| --- | --- | --- |
| forward parity, 4 repeats, chained | pass | pass |
| forward+backward, every leaf | pass | pass |
| back-to-back and concurrent slots | pass | pass |
| skewed routing | -- | pass |
| failure `forward_before_input_ready` | -- | pass |
| failure `backward_before_completion` | fail | fail, identically |

Two things the gates settled beyond the numbers.

Staged bytes per rank went 524,288/1,064,960 to 262,144/540,672. The activation terms halve exactly
and the router terms do not move, which is the wire traffic narrowing and is what the whole change
is for.

Two harness faults surfaced, neither in the kernel. `--back-to-back` passed a substituted activation
into `reference_output`'s routing parameter, so it had been raising a `TypeError` at HEAD; the helper
now takes the differentiable leaves and closes over the routing, matching `fused_output`. And the
gradient tolerance was a fixed `atol` of 0.5, which is one bf16 ULP at magnitude 64 and only held
because the single-width shapes happened to peak there. Every leaf in both arms sits at one to two
ULP; `routed_down` and `latent_up` failed only because their peaks are 156 and 230, where one ULP is
1.0. The tolerance now scales with the tensor's peak, and a relative-L2 bound is required alongside
it so a systematic error cannot hide under the wider floor. Relative L2 is 0.0044 to 0.0072 across
every leaf of both arms.

`backward_before_completion` fails the same way on both widths, with the same
`max_active_slots=(2,2,2,2)` against an expected one. It is pre-existing and untouched by this work.

### The gate shape that hid a deadlock

All of the above passed, and the first training run still spun forever with all four GPUs at 100%
and no step. The cause was a third class of site the width split had to reach, and the parity shape
could not see it.

`expert_grouped_gemm_kernel`'s non-wgrad path derives a barrier's required arrival count from its
own A operand, `a_gmem.cols()`, which is already the routed width for routed tasks. The wgrad path
cannot: wgrad iterates K over token rows, so `a_gmem.cols()` is the intermediate dim. It takes the
transferred width as a parameter instead -- and all three routed wgrads were handed
`g.d_y_shared.cols()`. The waiter then demands `ceil(hidden/DISPATCH_Nb)` arrivals where only
`ceil(routed/DISPATCH_Nb)` are ever published. `num_dispatch_tasks_of` and `num_combine_tasks_of`
read the same shared width for the same reason.

`DISPATCH_Nb` is 512. At the parity shape -- hidden 512, latent 256 -- both round to one block and
agree by accident. At the hero widths, 6144 against 3072, it is exactly a factor of two and the
rank waits on a count that can never arrive. **A latent gate shape is only meaningful if the two
widths differ in `DISPATCH_Nb` blocks**: hidden 1024 against latent 512 is the smallest that does,
and it now runs alongside the hero widths.

Five sites moved to `g.d_x_routed.cols()`. A CPU test pins the result -- `d_y_shared.cols()` may
appear in the generated backward only in the two definitions of the shared column-block count --
so this cannot regress on a gate shape that happens not to expose it.

The lesson generalises past this bug: a compression ratio that divides evenly into a tile constant
makes a whole class of width errors invisible. Choose gate shapes that straddle `MLP_Nb` (256),
`DISPATCH_Nb` (512), and `COMBINE_Nb`, not merely shapes that are "small but different".

## Phase 2 -- re-measure on the shape that matters

The launcher pinned `latent_dim=None` and both intermediate dims to 3072 for a reason it stated:
two hero asymmetries were inexpressible. One of them is now expressible, so the launcher stops
overriding `latent_dim` and the hero's 3072 reaches both arms.

The other remains. The megakernel derives every intermediate-width count from one value
(`intermediate_dim_col_blocks = hidden_shared.cols() / MLP_Nb`), so the hero's widened routed
experts -- intermediate 6144 against a 3072 shared width -- still cannot be expressed. That is the
same kind of split as the one just landed, on a different axis, and it is what remains before the
comparison runs on the true hero shape. Until then `intermediate_dim` and
`shared_expert_intermediate_dim` stay pinned together on both arms.

Re-run the matched comparison: one node first, then one rack. Report MFU, not raw tokens/s.

### Measured, 48 layers, latent 3072, capacity factor 1.1

| | MFU | tokens/s | drops |
| --- | --- | --- | --- |
| one node (4 GPUs) | 23.83% | 23,797 | 0 |
| two nodes | 23.53% | 46,981 | 4.7e-05 |
| one rack (16 nodes) | 21.87% final, 22.42% p90 | 349,382 | 0 |

Two-node weak-scale efficiency is 98.7%. The rack loses about two points of MFU against a single
node to data-parallel reduction, and still beats the sealed v15 two-rack run's 20.25%.

Tuning is exhausted at this shape. Capacity factor 1.1 beats the promoted 4.0 by 0.7 points and
stays dropless -- the strict factor buys nothing once drops are known to be a router transient --
while 1.0 is worse on both axes because drops return. The comm-SM split, the minibatch and
macrobatch sizes, `save_moe`, and a higher device-memory fraction were each measured and each lost
to the defaults.

### The routed intermediate split, and what it bought

The second asymmetry is now expressible too. `intermediate_dim_col_blocks` served both paths in the
backward exactly as `hidden_dim_col_blocks` did, and the same three classes of site needed
splitting: task counts, the wgrad barrier widths, and -- new here -- the two
`*_row_block_ready_required_count` values, which count SwiGLU tiles across a path's intermediate
width and were single values passed to both shared and routed call sites. The forward again needed
no task-count edit, only the routed branch of that required count.

Measured at 16 layers on one node, matched but for the routed intermediate:

| routed intermediate | MFU | drops |
| --- | --- | --- |
| 3072 (the old pin) | 23.85% | 0 |
| 6144 (hero) | 24.43%, and 25.35 / 25.10 / 24.63 / 25.34 / 25.13 across five repeats | 0 |

Five matched repeats of the hero width average 25.11%. The spread is 0.7 points, so a single sample
cannot establish this number; the earlier 24.43% reading is the same shape on a busier machine.

Widening the *shared* expert to 6144 as well -- the hero's two shared experts of 3072 folded into
one, which the summation makes equivalent -- is much worse, 13.25% with drops. That direction is
closed.

### Closing the rack gap: batch amortises the data-parallel reduction

Both hero asymmetries are expressible and the kernel-side tuning space is measured out. What
remained was a gap between a node and a rack, and it was not in the megakernel.

Data-parallel reduction costs the same per step whatever the batch, so tokens per node per step set
how much compute that fixed cost is amortised over. At 64 sequences per node the rack gave up about
a point to a single node; at 128 it gives up none. That also puts the global batch at 2,048 on one
rack, which is what the sealed two-rack runs already used.

One rack, 16 layers, hero widths, capacity factor 1.1, 128 sequences per node:

| | MFU | tokens/s | drops |
| --- | --- | --- | --- |
| sample 1 | 25.28% | 853,302 | 0 |
| sample 2 | 25.42% | 857,839 | 0 |
| sample 3 | 25.14% | 848,445 | 0 |

Mean 25.28%, and every sample clears 25%. At 64 sequences per node the same shape reads 24.19%.

Read tokens/s with the depth in mind. Sixteen layers is a third of the hero depth, so 853,000 is
about 284,000 normalised to forty-eight -- above the 250,000 bar, but the raw number is not a
rack-at-hero-depth figure and should not be quoted as one. MFU is the depth-normalised metric and
is the honest headline: at a fixed macrobatch, depth does not move it, 23.85% at 16 layers against
23.83% at 48 on the pinned width.

The remaining depth constraint is memory, not efficiency. At the hero routed width the saved
context does not fit at 48 layers -- 163 GiB on device under `save_moe`, and the pinned-host
allocator overflows under `offload_moe` -- so 48 layers needs the routed macrobatch cut from 32,768
to 8,192, and that costs about 1.3 points. Restoring the wide macrobatch at depth is the next
throughput lever, and it is an activation-memory problem rather than a Mixture-of-Kittens one.

### Expert count still requires EP64

The rack figure above uses eight experts at an expert axis of four. Per-token FLOPs are unchanged
by expert count at top-4 routing, so MFU and tokens/s carry over, but the hero's 192 experts do not
fit: expert weights shard only on the expert axis, so 192 experts over four ranks is 48 local
experts and the optimizer state OOMs. Sharding them needs EP64, EP64 needs the fabric transport,
and that is the deadlock above. The chain is 192 experts -> EP64 -> fabric -> unfixed bug.

Report MFU, not raw tokens/s. Depth and width both change tokens/s by factors unrelated to the
backend, and a raw figure invited exactly one wrong conclusion already.

## The fabric deadlock is what bounds expert-parallel width

Measured today, and it changes how the scale ladder should be read.

A matched A/B settles attribution: two 2-node runs under `fabric_symmetric`, identical but for
`--latent-dim 3072` against `--latent-dim 0`, both hang. The deadlock is pre-existing and has
nothing to do with the two-width path. The same shape on one node under `in_process_peer` trains
twenty of twenty updates.

It is also far more likely than the 29%-at-four-processes figure recorded earlier. Six of six
attempts at eight processes hung today, against a busy cluster -- consistent with the documented
contention correlation, but at eight processes it is effectively deterministic rather than
occasional. Neither `--mok-like-workspace-slots 2` nor the capacity-limited preset changes it.

`MOK_TRACE_INVOCATIONS` narrows where it sits. Every rank enters the same handler at the same
generation and stops inside it, so this is not a rank that failed to arrive; it is a device-side
deadlock in the backward with the full group present. That rules out the lease and generation
protocol on the host side and puts the fault inside the megakernel's backward comms.

**The consequence for scale: `mok_like_num_devices` above four requires the fabric transport, so an
EP64 expert axis is gated on this bug.** An expert axis of four is not: one process owns a node's
four GPUs, peers stay process-local, and the rack scales by data parallelism instead. That is the
topology the sealed v12 rack runs used, and it is the one to measure on until the fabric path is
fixed. Reading "EP64 on one rack" as the only rack configuration is what made this bug look like a
hard blocker on rack measurement; it blocks the expert-axis width, not the rack.

## Phase 3 -- reliability, in parallel from the start

An intermittent device-side deadlock in the cross-process fabric path remains unfixed: all GPUs
pinned at full utilisation, ranks spinning in `WaitCompletionKernel`, no progress. It reproduces
around 29% of the time on four processes, more often on eight, and correlates with cluster
contention. Six hypotheses have been wrong and one `atomicMax` patch was reverted after a matched
A/B showed no effect.

Stop hypothesising and instrument. Capture device state automatically when a run stalls -- the
debug counters, and a `cuda-gdb` or thread dump of a spinning rank -- so the next occurrence yields
evidence without anyone watching. This is independent of Phase 1 and should start immediately,
because occurrences are opportunistic and the current cost is a wasted rack-hour each time.

Repro: `--num-nodes 1 --num-layers 4 --num-experts 8 --mok-like-num-devices 4
--mok-like-workspace-transport fabric_symmetric`. Control: the same with `in_process_peer`.

## Phase 4 -- depth, then racks

Re-test forty-eight layers at EP64 once the routed path is narrow. If it still does not fit,
the remaining levers are the schedule capacity factor, now that drops are known to be a transient
rather than a steady-state property, and the remat mode.

Multi-rack is blocked on infrastructure, not on this backend. Roughly eight attempts never reached a
first step; every failure was the JAX coordination service becoming unreachable during distributed
init across 128 processes. That wants its own issue against Iris rather than more retries.

## Phase 5 -- the proposed architecture

With latent support in place, 8/384 becomes measurable rather than a feasibility question. Compare
it against 4/192 on the same backend, and against the ragged-a2a arm, which already runs the shape
at 256k tokens/s and 24% MFU on one rack.

## Phase 6 -- the 8/384 shape is arena-bound, not transport-bound

First EP64 run at the proposed architecture (`mok-goal-ep64-16l-a-20260816`: d6144, latent 3072,
routed intermediate 6144, 8-of-384, two shared experts, 16 layers, one rack) compiled and reached
the first `jit_train_step`, where every one of the sixty-four ranks failed the same way:

```
ncclCuMemAlloc ... Cuda failure 2 'out of memory'
INTERNAL: NCCL operation ncclAlltoAll(send_contiguous, recv_contiguous, ...) failed
```

The arena is sized `4 * tokens_per_rank * routed_dim * (1 + top_k) * workspace_slots` and allocated
outside XLA through CUDA VMM, so it competes with NCCL for whatever the device-memory fraction
leaves. At one rack, `tokens_per_rank` is 131,072:

| routing | routed width | arena per rank |
| --- | --- | --- |
| top-4 | latent 3072 | 7.50 GiB |
| top-8 | latent 3072 | 13.50 GiB |
| top-8 | hidden 6144 | 27.00 GiB |

Top-8 is what the proposed architecture buys, and it costs 6 GiB per rank on top of the sealed
figure. The preset leaves 20% of the device for everything outside XLA; that was sized against a
7.50 GiB arena.

Note what the table also shows: without the latent split this shape would need 27 GiB of arena per
rank and would not be reachable at all. The width work is what makes 8/384 a memory-fraction
question rather than an impossibility.

The arena follows tokens per rank, not the routed macrobatch, so shrinking the macrobatch does not
relieve it -- which matters because the goal forbids that trade. The lever is the device-memory
fraction.

### What the ragged-a2a profile implies about the MoK ceiling

The `marin_ep` arm profiled its EP64 hero step (MEP-036, MEP-044) and attributed the busy time.
Two of its buckets are what the megakernel exists to remove, and the rest it cannot touch:

| bucket | s/step | does MoK remove it? |
| --- | --- | --- |
| attention + dense GEMMs | 4.95 | no |
| XLA fusions | 3.55 | partly -- the dispatch/combine fusions only |
| FSDP one-shot collectives | 2.85 | no |
| ragged a2a + barriers (cute arm) | 2.30 | yes, this is the fused transport |
| expert GEMMs | ~2.4 | no, but it fuses them with the transport |

That arm sits at 17.0 s/step and 22.6% MFU. Removing the 2.3 s transport bucket outright would put
it near 14.7 s/step, which is about 26% MFU on the same calibration. So the goal is not obviously
out of reach for a backend that fuses exactly that bucket -- but it also means almost the entire
margin comes from one bucket, and nothing in the fused design attacks attention, dense GEMMs, or
the FSDP collectives. Those are the floor.

### A proxy that matched the wrong quantity

A two-node EP8 run was submitted as a cheap stand-in for the rack while the cluster was full. The
reasoning was that tokens per rank is 131,072 either way, so the symmetric arena is the same 13.50
GiB and the memory fix could be tested at an eighth of the capacity.

That is true of the arena and false of everything else. Expert weights and their optimizer state
shard on the expert axis, so 384 experts over eight ranks is forty-eight local experts against the
rack's six. The run died in `cuMemAllocAsync` on a single 145.3 GB request with the pool already at
its 147.44 GiB limit -- a shape that does not exist at EP64.

The proxy is only valid for quantities that follow tokens per rank. Anything that follows experts
per rank -- weights, optimizer state, the expert GEMM working set -- is eight times too large in it.
Wider expert parallelism makes this configuration *easier*, not harder, which is the opposite of the
usual scaling intuition and is why the substitution looked reasonable.

## Phase 7 -- the shape moved under us

`origin/main` changed the hero on 2026-08-16 in #8289 and #8348, hours into this work. Every branch
in flight predates it:

| | routed experts | top-k | routed intermediate | capacity factor | transport |
| --- | --- | --- | --- | --- | --- |
| `origin/main` | 384 | 8 | 3072 | 1.15 | pooled-wave |
| this branch | 192 | 4 | 6144 | 1.33 | fixed all-to-all |
| `marin_ep` branch | 192 | 4 | 6272 | 1.33 | ragged / fused mosaic |

Three consequences.

The routed and shared intermediates are equal on main, so the intermediate-width split this branch
implements is no longer load-bearing for the current architecture. Only the token-width split, which
`latent_dim` drives, is. The intermediate split remains correct and tested and should still land,
but it is no longer what unblocks the hero.

The `marin_ep` arm's 22.6% MFU and 247k tokens/s are measured at the old 192/top-4 shape. They are
not a like-for-like reference for the 384/top-8 target, and no backend has yet measured main's
current hero.

Per-token cost at main's hero is 49.91 GFLOP, so on sixty-four GB200s 25% MFU is 267,000 tokens/s
and 275,000 tokens/s is 25.7% MFU. The goal's two figures are the same bar. An earlier note in this
document derived them from this branch's stale 6144 routed intermediate and concluded they were
inconsistent; that conclusion was wrong and is withdrawn.

## Blocker ledger

| blocker | state | resolution |
| --- | --- | --- |
| `latent_dim` rejected by the fused kernel | closed | two-width FFI, this branch |
| top-k and shared-expert count not expressible | closed | `--num-experts-per-token`, `--num-shared-experts` |
| numerical gate pinned to top-4 | closed | `--top-k`, `TOP_K` was a module constant |
| `ncclCuMemAlloc` OOM at 64 ranks | closed | `NCCL_BUFFSIZE=1048576`, from ra2a RA2A-003 |
| NCCL window VA exhaustion | closed | `--xla_gpu_enable_allocator_spatial_partitioning=false` |
| collective params escaping XLA coloring | closed | `--xla_gpu_enable_dynamic_slice_fusion=false` |
| cross-process clique-init deadlock | mitigated | dev20260809 nightly, behind `--jax-nightly` |
| MoK fabric device deadlock | open, intermittent | retry budget; not a hard block per #8244 |
| forty-eight layers at EP64 | unmeasured | the 240.81 GiB report predates latent and the 3072 intermediate |
| rack capacity | open | Kueue-gated behind production |

### Top-8 is numerically sound at the hero widths

`mok-gate-topk8-c-20260816`, one GB200 tray, four ranks in one process, hidden 6144, latent 3072,
intermediate 3072, top-8:

| quantity | value |
| --- | --- |
| forward max absolute error | 0.125 against a peak magnitude of 13.25 |
| forward relative L2 | 0.0062 |
| forward mismatch fraction | 0 |
| gradients within tolerance | 11 of 11, including `routed_x` and `latent_up` |
| dropped assignments | 0 |
| FFI calls per rank | one forward, one backward |

The call counts matter as much as the errors: one forward per rank means the remat policy is still
naming the combine before `latent_up` expands it. When that name is missing the whole fused call is
recomputed and the forward runs twice per layer, which is a silent throughput halving rather than a
failure.

The gate's bank is eight experts at top-8, so every token reaches every expert -- denser than
8-of-384 and a harder capacity case, not an easier one. It does not exercise the fabric transport,
which is single-process here.

### The 8/384 shape reaches the gate at node scale

`mok-hero384-ep4-16l-20260816`: one node, EP4, in-process peers, hidden 6144, latent 3072, routed
and shared intermediate 3072, top-8, two shared experts, sixteen layers, batch 128 per node so
tokens per rank is 131,072 and the arena matches the rack's 13.50 GiB. Thirty-two experts, eight
local per rank. 25/25 steps, loss 6.23 and falling.

| steps 11-24 | value |
| --- | --- |
| MFU | 25.08% low, 25.56% high, ~25.3% mean |
| tokens/s on four GPUs | 47,419 - 48,355 |
| step duration | 10.84 - 11.06 s |
| drop fraction | zero from step 19 onward |

Expert count does not enter per-token FLOPs at fixed top-k, and MFU is depth-neutral at fixed
macrobatch, so this figure should carry to 384 experts and to forty-eight layers. It does not carry
across EP4 to EP64: that step adds the cross-node transport, and the previous EP64 reading on the
old shape gave up between three and six points against its EP4 sibling.

The drop trace repeats the router transient seen before -- 0.93% at step 11 falling to zero by step
19 -- rather than a steady-state capacity shortfall, at capacity factor 1.1 against main's 1.15.

So the shape is not the problem, and neither is top-8. What remains between this and the goal is
entirely the EP4-to-EP64 transport gap.

### The NCCL and XLA fixes do not touch the fabric deadlock

`mok-hero384-ep8-16l-b-20260816`: two nodes, EP8 over `fabric_symmetric`, sixty-four experts so
local experts per rank is eight and matches the rack's density, hero widths, top-8, sixteen layers.
Carrying `NCCL_BUFFSIZE=1048576` and both marin_ep XLA workarounds.

XLA logged `Can't reduce memory use below 163.23GiB by rematerialization` at 04:41:45 and the run
then went silent: no further log line, no W&B row, twenty minutes later. Killed at 05:03.

This is the established hang signature and it says the three borrowed fixes are orthogonal to it,
which is what their mechanisms predict. All three act on NCCL's allocator or XLA's collective
buffers; the megakernel's transport is its own CUDA VMM code and issues neither. They were worth
carrying because they cleared a real OOM at sixty-four ranks -- they were never going to clear this.

One observation, at eight processes, against a prior of seven hangs in seven attempts at that width.
It does not add much to that prior and it is not a controlled A/B. What it does establish is that
nothing acquired from the sibling arms today changes the picture, so the EP64 attempt still rests on
the retry budget rather than on a fix.

The rematerialization warning in that log is not evidence of anything. The EP4 run that completed at
25.3% MFU reported a *larger* figure, 165.88 GiB against this one's 163.23 GiB, so
`Can't reduce memory use below N` under `cuda_async` is a note about the remat pass rather than a
prediction of failure. It should not be read as a memory verdict, and the forty-eight-layer question
is still open rather than already lost.

## Phase 8 -- what the marin_ep event simulator says about the EP width gap

`experiments/marin_ep/perfmodel/eventsim.py` is a discrete-event model of one MoE layer over
explicit per-device NVLink ingress/egress and compute resources, with tile-granular pipelining,
incast and a real `[S, E]` count matrix. Run at main's hero widths (routed path 3072 wide, top-8,
131,072 tokens per rank, capacity factor 1.1), forward plus backward per layer:

| expert axis | routing | fully fused | bulk-synchronous |
| --- | --- | --- | --- |
| EP4 | balanced | 102.19 ms | 129.65 ms |
| EP64 | balanced | 103.53 ms | 143.41 ms |
| EP4 | one expert at 4x mean | 112.49 ms | 142.09 ms |
| EP64 | one expert at 4x mean | 112.41 ms | 162.24 ms |

The fused transport is essentially invariant to expert-axis width: +1.3% from EP4 to EP64 balanced,
and -0.1% under a 4x hot expert. The bulk-synchronous shape degrades over the same range, +10.6% and
+14.2%. Routing skew costs about 10 ms and costs it equally at both widths.

If that holds for the megakernel, the EP64 MFU deficit measured earlier -- 18.80% to 21.87% against
25.3% at EP4 -- is not paid for in transport bytes or link contention. It is somewhere the model
does not look: synchronization and launch structure, the fabric stalls, or the non-MoE mass that
grows with rank count, principally the FSDP one-shot collectives. That is a different and more
tractable problem than "wide expert parallelism is inherently slower", and it points at the same
subsystem the deadlock lives in.

Five reasons not to lean on this too hard:

- It models the `marin_ep` transport, not the megakernel. Both are fused put-plus-arrival designs
  with tile granularity, so the shape is a fair structural proxy, but it is not the same code.
- Its own fidelity notes exclude SM contention between transport and MMA warps. That is precisely
  the megakernel's design point, so the one thing most specific to this backend is unmodelled.
- No NVLink switch contention beyond per-device ingress/egress serialization, and no HBM bandwidth.
- One MoE layer only: no attention, no FSDP collectives, no XLA fusion mass.
- The file is under active edit in the sibling worktree. Its API changed from `pipelined: bool` to
  `mode: PipelineMode` between two runs of mine, so these numbers come from a moving snapshot
  (`560dfd6305` plus uncommitted changes) rather than a pinned commit.

It is a hypothesis generator, not evidence. What it is good for is telling us where *not* to spend
the next rack: widening the wire or shrinking bytes is predicted to buy nothing.

### The node-scale 25.3% is an optimistic predictor, and here is why

The Grug mesh is `(replica_dcn, data, expert, model)` and `_FSDP_AXES` is `("data", "expert")`, so
non-expert parameters shard across the product of the data and expert axes.

The single-node run that read 25.3% had one process, four local devices, `expert_axis_size` 4 and
`replica_axis_size` 1: FSDP over four devices, entirely inside one NVLink domain. Any rack
configuration shards those same parameters over sixty-four -- EP64 because the expert axis takes
every device, and EP4-plus-data-parallel because the data axis takes what the expert axis does not.
Both put the parameter all-gathers across sixteen nodes.

So 25.3% is not a like-for-like predictor of a rack number; it omits a cost every rack run pays. The
honest comparator for it is the earlier one-rack EP4 reading of 25.28%, which did carry sixty-four
way FSDP -- though that was a different shape (top-4, eight experts, routed intermediate 6144), so
the agreement between the two is suggestive rather than conclusive.

Taken with the event-simulator result, the two together say: transport width is predicted not to
matter, and FSDP width apparently did not matter much at sixteen layers on the old shape. Neither
observation explains the earlier EP64 deficit, which leaves synchronization and launch structure --
the fabric path -- as the remaining candidate, and that is the subsystem with the open deadlock.

## Phase 9 -- the EP64 deficit is staging, not the wire

The runtime counters from the node-scale run locate the synchronization cost precisely. Per process,
25 steps, 16 layers, four macrobatches per layer:

| phase | peer-wait cycles | share | events | mean cycles |
| --- | --- | --- | --- | --- |
| forward_pre | 14,909,802,282 | 4.8% | 83,863 | 177,788 |
| forward_post | 16,670,215,628 | 5.4% | 2,008 | 8,301,900 |
| backward_pre | 229,568,883,406 | 73.9% | 1,729,379 | 132,746 |
| backward_post | 49,477,377,082 | 15.9% | 1,944 | 25,451,326 |

`backward_pre` is three quarters of all peer waiting and has a million and a half more events than
anything else. Alongside it, `forward_staging_bytes` is zero and `backward_staging_bytes` is 2.59 TB
per process, about 103 GB per step.

The asymmetry is not incidental. The promoted preset runs `forward_x_storage` as
`XLA_PEER_EXPERIMENTAL`, which reads a peer's XLA buffer in place, and `backward_peer_storage` as
`RUNTIME_STAGED`, which copies through the symmetric arena. The forward path stages nothing because
it does not have to.

`MokLikeConfig.__post_init__` then rejects both direct-read modes whenever the transport crosses
processes, and its comment gives the reason: those modes use space-0 peer mappings that stay
process-local however wide the group is. So EP64, which requires `FABRIC_SYMMETRIC`, cannot use the
forward path that the node-scale run used. It must stage the forward as well.

That means the 25.3% node-scale reading is optimistic for a second and larger reason than the FSDP
width noted above: it runs a forward path that is structurally unavailable at the target
configuration. The honest expectation for EP64 is 25.3% minus whatever forward staging costs, and
backward staging already accounts for three quarters of peer-wait at a width where it is cheapest.

This also reconciles with the event simulator rather than contradicting it. That model puts bytes on
the wire and finds the fused transport width-invariant, which is probably right. The cost that grows
is not the wire -- it is the extra copy into the arena that a cross-process transport forces, and
the simulator has no notion of staging at all.

The lever this identifies is a real change rather than a flag: let the fabric path map XLA buffers
directly instead of copying them into the arena, so the forward keeps its in-place read at EP64.
That is the difference between a preset choice and a kernel/runtime change, and it is where the next
engineering effort belongs.

### Staging is not the EP64 tax -- the hypothesis is falsified

`mok-hero384-ep4-16l-stage-20260816` repeats the 25.3% node-scale run with one field changed,
`forward_x_storage` from `XLA_PEER_EXPERIMENTAL` to `RUNTIME_STAGED`, which is what a cross-process
transport forces.

| | forward staging bytes | MFU, steps 17-24 | drops |
| --- | --- | --- | --- |
| `XLA_PEER_EXPERIMENTAL` | 0 | 25.08 - 25.56% | zero from step 19 |
| `RUNTIME_STAGED` | 1.29 TB | 25.17 - 25.68% | zero from step 19 |

The staging happened -- 1.29 TB of it per process, where the other run staged nothing -- and it cost
nothing measurable. The forced arm is marginally *faster*, which is run-to-run noise, not a result.
`backward_pre` peer-wait rose from 229.6 to 284.3 billion cycles, a 24% increase in waiting that
still did not move the step.

So the Phase 9 conclusion is wrong and is withdrawn. The reasoning was that EP64 must stage the
forward while the node-scale run did not, and that this explained the deficit. The premise is
correct and the consequence does not follow: the staging copy is arena-local HBM traffic at both
widths, and it is cheap relative to the expert GEMMs.

Two things this is worth.

It kills a change before it was written. The lever Phase 9 proposed -- mapping XLA buffers into the
fabric so the forward keeps its in-place read -- would have been substantial kernel and runtime work
against a cost that measures as zero.

It also says peer-wait cycles are a poor proxy for time. A 24% rise in `backward_pre` waiting moved
MFU by less than the noise band, so those counters record how long warps sit on flags, not exposed
wall-clock. They should not be read as a cost attribution again without a profile alongside them.

What remains unexplained is the earlier EP64 reading of 18.80% to 21.87%. That was measured on the
old shape -- no latent projection, routed intermediate 6144 -- so it is not a like-for-like
comparison with anything here, and the EP64 number for main's hero is still simply unknown.

## Phase 10 -- first EP64 numbers, and the deadlock at rack width

Two EP64 attempts on main's hero shape, sixteen layers, 384 experts at top-8.

`mok-hero384-ep64-16l-b-20260816`, capacity factor 1.1, reached three steps and then raised
`Non-finite loss (nan) at step 4`:

| step | MFU | tokens/s | drop fraction | loss | router z-loss | routing entropy |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 7.76% | 234,103 | 4.1% | 11.81 | 35.8 | 5.861 |
| 1 | 17.83% | 537,565 | 18.1% | 9.90 | 78.0 | 5.314 |
| 2 | 24.20% | 729,677 | 52.5% | 9.05 | 286.2 | 3.807 |

24.20% MFU at step two, still inside the warm-up ramp, on the configuration the goal names. The
node-scale run only settled at 25.3% around step eleven, so this is a floor rather than an estimate,
and the EP4-to-EP64 gap looks far smaller than the 18.80% to 21.87% readings taken on the old shape.

The failure is a router collapse, not a capacity shortfall. Entropy falls to 0.64 of the uniform
bound `ln 384 = 5.951` by step two while z-loss rises eightfold and twenty-four of 384 experts stop
receiving tokens entirely. Drops are downstream of that. Per the user, transient drops in the first
few hundred steps are not a fidelity concern at all, so the drop column is not the problem here --
the diverging logits are.

`mok-hero384-ep64-16l-cf4-20260817`, capacity factor 4.0, never reached a step. It logged the
rematerialization warning at 16:21:01 and went silent for fifty-four minutes; killed. That is the
fabric hang signature, now observed at rack width, and the third occurrence today after the
eight-process run.

So the two attempts fail differently and neither is a measurement: one diverged, one hung. Taken
together they say the transport reaches EP64 and runs, intermittently, and that the router is the
open modelling question at 384 experts.

`qb_beta_per_layer` is train state that the logger filters out, so nothing in either trace shows
what the load-balancing controller was doing. Scalar summaries are now emitted under
`train/router/qb_beta_*`, including a non-finite fraction, because the next step's router bias is
exactly the negation of that estimate: a degenerate value there appears as a routing collapse one
step later with no other trace.
