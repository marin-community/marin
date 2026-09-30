# mfu30 agent A: host-offload exposure

Branch `research/mcwitt/mfu30-offload` (worktree `~/projects/marin.mfu30-offload`), based on
`research/mcwitt/mfu30` (main `f38da1173d`). Parent logbook: `.agents/logbooks/mfu30.md`. Runs `m30a-*`,
ports 33100-33149.

## M30A-001 Memory picture from the baseline trace (2026-09-30)

Source: `overlap-remeasure-main-144k` xplane (main `8d70d9cbb0`). The xplane's train-step `HloProto`
carries the full `BufferAssignmentProto` (field 3), so the arena layout is readable offline
(`bufassign.py`, `instr_ids.py`, `liveness2.py` in the session scratchpad).

Per-device entry I/O: device inputs 35.09 GiB (fp32 params), pinned-host inputs 38.55 GiB (optimizer
state). The host state is three f32[48,6,3072,3072] expert momentum shards (3 x 10.125 GiB), Adam m/v of
the replicated f32[128256,6144] token embedding (2 x 2.94 GiB), router Adam m/v (2 x 0.42 GiB), and ~1.5
GiB of small leaves.

Allocations: device temp arena 67.49 GiB (color 0), host temp 36.0 GiB (color 5, the offloaded layer
carry stack bf16[48,16,4096,6144]), collective buffers 18.9 GiB (color 1). Peak 103.09 GiB = 35.6
persistent + 67.5 arena, which matches W&B `memory/peak_gib`.

Arena usage over the schedule (buffer offsets + HLO liveness): forward loop ~35 GiB, backward loop ~63.5
GiB (bf16 expert weight casts 15.2 + bf16 expert grad accumulators 15.2 + body temps), optimizer phase
peak 67.49 GiB when the three momentum H2D copies have landed (copy-start.44 -> .43 -> .42 at 38 -> 48 ->
58 -> 67.5 GiB).

H-A1 (all optimizer state resident) estimate: persistent 35.6 + 38.55 = 74.2 GiB; the optimizer phase
loses its 30 GiB of momentum copies, so the arena becomes the backward peak (~63.5 GiB, more once remat
is relaxed). Peak ~137-142 GiB against the 138.2 GiB release threshold, and the scheduler limit
`(0.8 x 184.3 - device_io) x 0.85` drops from ~95.5 to ~62.7 GiB. Does not fit at 0.75 without
raising both the fraction and the slop factor. Parked behind the cheaper finding below.

Schedule: the three momentum D2H copies (`copy-start.97/.98/.99`) sit 10 instructions before their
`copy-done` at the very end of the step, although the new momentum is ready ~2,700 instructions earlier.
The GPU LHS falls back to the T-shirt `GpuLatencyEstimator` (the SOL estimator rejects ragged
all-to-all), which gives every async copy 5,000 units regardless of 10 GiB size. PGLE (another session)
may move them; code-level ties are the fallback.

## M30A-002 XLA remat counts the host carry stack as device memory (2026-09-30)

`HloRematerialization` (post-scheduling) zeroes host-memory-space buffers only when
`--xla_gpu_enable_host_memory_offloading` is set (`AllocatedSize` in hlo_rematerialization.cc, checked at
the pinned PJRT commit `708c3a4ec79c`). The hero leaves it false, so remat sees the 36 GiB pinned-host
carry stack as device temps in main and charges it against the backward body's limit
(`limit - caller usage at the while`).

Evidence in the baseline HLO: 143 `*.remat*` instructions, 84 of them in the backward body
`region_81.191_spmd` (10.6 GiB of recomputed values), including a synchronous `all-gather.127.remat`
per layer. Trace cost: **0.633 s/step on the compute stream** (0.630 in the backward body; 0.227 of it is
the sync all-gather remat, mostly rank-skew wait; 0.11 overlaps collectives).

The LHS is unaffected by the flag: its `MemoryPressureTracker` skips non-default memory spaces and entry
parameters (`ShouldSkipBufferAllocations`), so its temp budget is `(147.4 - 35.1) x 0.85 = 95.5 GiB`
against a 67.5 GiB arena and does not bind. Only the remat clones should go.

H-A4: `XLA_FLAGS=--xla_gpu_enable_host_memory_offloading=true`. Prediction: remat count -> ~0, compute
stream -0.4 s, step -0.3 to -0.5 s, arena +5-10 GiB (peak ~110 GiB). Numerics unchanged (remat recomputes
identical values). Engagement: TF_CPP remat logs (dispatch.py now forwards `TF_CPP_*`) and the profile's
`*.remat*` kernel count.

## M30A-003 Rack arm m30a-hmo-01 submitted (2026-09-30 11:03 PT)

`/mwittmann/m30a-hmo-01-coord`, port 33100, `XLA_FLAGS=--xla_gpu_enable_host_memory_offloading=true`,
`TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=hlo_rematerialization=1`, profiled 180021-180023, seed 0, stop
180060. Branch commit `f600de5d8e` (program identical to main apart from the flag). Gated in Kueue at
submission. Reads: first-step loss vs `mhep-ctx4k-s0-20260930`, remat log lines (limit, peak, count), the
profile's `*.remat*` kernels, median MFU over 180005-180059 minus the profiled steps, `memory/peak_gib`.

## M30A-004 PGLE does not hide the optimizer D2H (2026-09-30)

Re-read the PR #9481 PGLE trace `overlap80-t21-pgle-rep2-144k` (another session's scratchpad) with the
toolkit. Under PGLE the first momentum H2D moves into the backward (copy-start.44 at 3.6 s) and hides,
but the three momentum D2H copies stay at the end of the step and stay exposed (~0.2 s), and the forward
carry D2H becomes exposed (~0.1-0.17 s) where main hides it. XLA remat still costs 0.35 s/step there.
So PGLE and H-A4 are complementary, and the end-of-step D2H needs removal (H-A1) or a code-level
reorder, not a better latency estimate.

## M30A-005 H-A1 knob

`launch_diagnostics --no-offload-opt-state` (default unchanged). Memory plan waits for the H-A4 remat
logs, which print the real scheduler/remat limit. With `base = 0.8 x 184.3 = 147.4 GiB`, H-A1 raises
device I/O to 73.65 GiB, so keeping the LHS/remat limit at the baseline ~95.5 GiB needs
`--xla_gpu_memory_limit_slop_factor=129`, and the ~137.7 GiB peak needs
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.78` (143.8 GiB pool, ~12 GiB outside for NCCL/cuBLAS/context).
H-A1 and H-A4 compete for the same headroom: stacked, the peak lands near 145-148 GiB.

## M30A-006 Adjusted predictions after agent C's power-cap finding (2026-09-30)

Agent C: tensor-core GEMMs are power-capped (1200 W, clocks 1200-1600 MHz), so removing exposed or
idle time next to GEMMs may return ~10-15% of it as slower GEMMs. H-A4's removed work is mostly
memory-bound remat fusions (0.40 s) plus one sync all-gather per layer (0.23 s, rank-skew wait), not
tensor-core recompute, so the power-cap penalty applies to the step-time gain, not to the removed kernels
themselves. Revised H-A4 prediction: -0.25 to -0.45 s/step (+0.5 to +0.9 MFU). H-A1 (0.33 s exposed
copies) scales to ~0.28-0.30 s. Headroom budget: H-A4 (+5-10 GiB arena) and H-A1 (+38.55 GiB
persistent) compete for the same ~35 GiB below the 0.75 threshold; any remat-reducing lever from other
agents also draws on it.

## M30A-007 H-A4 correctness review, prebuilt arms, headroom budget (2026-09-30)

Flag scope at `708c3a4ec79c`: `xla_gpu_enable_host_memory_offloading` is read only in
`gpu_compiler.cc` `CreateHloAnalysisOpts` / `CreateRematOpts`, both consumed only by the post-schedule
`HloRematerialization` in `RunPostSchedulingPipelines` (after LHS and copy insertion). HostOffloader,
memory-space propagation, the host-transfer asyncifier and the LHS never read it. Remat only inserts clones
before uses; it cannot move the existing carry `dynamic-slice-done` relative to its consumer, and the
collective overlap limit stays forced to 1. One side effect: the flag also enables remat's host-offload
mode, whose cost model uses HBM bandwidth for host transfers, so if remat still binds in the device-only
view it would prefer inserting post-schedule host copies. Engagement gate: zero `Remat via offload` lines
in the logs; if any appear, treat the arm as a different treatment.

Fidelity gate vs `mhep-ctx4k-s0-20260930`: loss at 180000 equal (restore and forward are untouched by
remat); 180001-180003 within the ~1e-4 C-C band from the August loop; `moe/drop_fraction` ~3.3e-5 and
`train/router/load_balancing_loss` in family through 180059.

Prebuilt (scratchpad `mfu30a/`): `arm2_hmo_pipe.sh` (H-A4 + `--xla_gpu_enable_pipelined_host_offloading`,
collective_pipeliner VLOG) and `arm3_resident.sh` (`--no-offload-opt-state`, fraction, slop, optional
H-A4). The LHS skips entry parameters, so H-A1's slop only has to restore the 95.5 GiB temp budget:
`(147.4 - 73.65) x s = 95.5` gives s = 129. Slop 85 would cut the budget to 62.7 GiB, under the 63.5 GiB
backward need, and make remat (host-counted) cut ~33 GiB more.

Headroom: the pool can reach ~0.81 x 184.3 = 149 GiB while leaving ~6.5 GiB outside it (collective
buffers 18.9 + NCCL/cuBLAS/context ~9.6 GiB). Peak today 103.1, so ~46 GiB exists at 0.81 and ~35 at
0.75. Costs: H-A4 +5-10 GiB for ~0.35 s (~50 ms/GiB); H-A1 +34.6 GiB at the backward peak (38.55
persistent minus the 4 GiB optimizer-phase arena excess) for ~0.29 s (~8 ms/GiB); carry prefetch ~+9 GiB
(August) for <=0.2 s. Reservation: 10 GiB for H-A4 first. Anything from C that reduces remat should beat
~8 ms/GiB to outrank H-A1; H-A1 goes only if >=35 GiB remains after H-A4 and C, at fraction 0.80-0.81.

## M30A-008 Budget order from the orchestrator (2026-09-30)

Order: H-A4 10 GiB, then B's saved latent MoE output ~19 GiB (~25-30 ms/GiB), then H-A1 only if room
remains. After H-A4 + B (~29 GiB) there is ~6 GiB left at 0.75 and ~17 GiB at 0.81, so full H-A1
(34.6 GiB) is out. Whole-leaf partial residency is the August C3' configuration (k expert momentum leaves
resident, donated), which NaN'd from in-step clobbering. The donation-excluded variant was clean but
measured -0.35. The only leaf set that fits afterwards is the dense one (embedding/router Adam + small
leaves, 8.2 GiB, ~0.08 s exposed, ~10 ms/GiB), in the same hazard class and near the noise bar, so it
is not planned unless the C3' root cause is found. Fidelity ruling recorded: rounding and reassociation
changes are allowed within the same-code loss band; training-semantics changes are not.

## M30A-009 hmo-01 timed out in the queue; resubmitted as m30a-hmo-02 (2026-09-30 12:05 PT)

`m30a-hmo-01` hit the coordinator's `--timeout 3540` while still gated in Kueue (the timeout counts queue
time) and never ran. Resubmitted the identical treatment without `--timeout`:
`/mwittmann/m30a-hmo-02-coord`, port 33101, branch tip `f14d34b279` (program identical to main
apart from the flag; the `--offload-opt-state` knob defaults to the hero value). Protocol note: same-code
gcab runs drift ~0.4 MFU down over 330 steps, so main reads ~28.5 in the 55-step screening window; score
against `mhep-ctx4k-s0-20260930` over the same steps.

## M30A-010 Seed-0 control landed; scoring window 180011-180059 (2026-09-30)

`mhep-ctx4k-s0-20260930`: median 28.255 MFU (13.892 s) over 180005-180059, peak 103.09 GiB; loss
180000 1.261413, 180001 1.234596, 180002 1.200221, 180003 1.256785. Steps 180000-180010 are restore
warmup with no later drift, so arms score 180011-180059 (profiled 180021-180023 excluded). Gap to 30%:
-0.81 s. Stack note: PR #9481's "re-gather hero attention weights in the backward pass" targets the same
XLA-remat sync all-gathers H-A4 removes; the hmo-02 analysis must list which `all-gather*.remat` clones
disappear so the stack knows whether it still needs that commit.

## M30A-011 Carry prefetch prepared on top of the stack (2026-09-30)

Prebuilt `mfu30a/arm2_on_stack.sh <NN> <port> <fraction> "<stack flags>" -- <stack switches>` (scratchpad).
It checks the current `origin/research/mcwitt/mfu30-stack` tip out into A's own worktree
(`~/projects/marin.mfu30-offload-stack`, branch `research/mcwitt/mfu30-offload-stack`) and submits through the
stack's `arm.sh` with the stack flags plus `--xla_gpu_enable_pipelined_host_offloading=true`, a trace, and
`hlo_rematerialization,collective_pipeliner` VLOGs (the stack's `dispatch.py` already forwards `TF_CPP_*`).
Dry run at `8eff7b8ec4` builds the expected command.

Pairing and order: the control is the stack arm with the same commit, flags and switches minus the
pipelining flag. The flag renames instructions, so it must be decided before the PGLE build. Memory: the
stack peak is ~103 + 18 (D's saved routed output) + H-A4 5-10 - E/unfilled savings; August measured +9 GiB
for pipelined offload, so read the stack arm's `memory/peak_gib` first and set the fraction so that
peak + 9 stays >= 4 GiB under fraction x 184.3 (0.78 gives 143.8 GiB). Risks: the pass also sinks the
forward carry D2H by an iteration (hidden on main today); it may fail to pipeline the reload through D's
barriers (engagement: `collective_pipeliner` "Transforming"/pipelined-while lines plus the reload's
position in the trace); #8317 family is covered by the forced overlap limit 1 plus the loss check against
the paired stack arm.
