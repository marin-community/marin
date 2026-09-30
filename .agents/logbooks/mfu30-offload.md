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

The LHS is unaffected by the flag: its `MemoryPressureTracker` already skips non-default memory spaces,
but it does count the live device entry parameters against a limit that has already subtracted them, so
its temp budget is `(147.4 - 35.1) x 0.85 - 35.1 = 60.4 GiB` against a 67.5 GiB arena. Only the remat
clones should go.

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
