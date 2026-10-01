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

## M30A-012 m30a-hmo-02 result: H-A4 engaged, -0.157 s/step on one draw (2026-10-01)

Score (orchestrator, 180011-180059 minus profiled): 28.580 MFU / 13.734 s vs `mhep-ctx4k-s0` 28.258 /
13.891 (+0.32 MFU, -0.157 s). Peak 104.07 GiB (+0.98). Loss 180000 = 1.2614134550 exactly; then
-2.4e-6, +1.7e-5, +3.6e-5; max |d| 4.4e-4; late mean +1.58e-4, positive on 48/49 late steps.

Remat logs (rank 0, `jit_train_step`): `HloRematerialization() with memory limit of 161.31GiB`, adjusted
for output (73.65 GiB, host included) to **87.66 GiB**. That equals `(138.22 - 35.09) x 0.85`, so the
XLA base is the allocator pool (`XLA_PYTHON_CLIENT_MEM_FRACTION x 184.3`), not 0.8 x device memory:
raising the fraction raises the scheduler/remat limit too. Per-computation peaks: backward body
`region_81` 48.89 GiB, forward body 16.11 GiB, main 89.94 GiB. Remat ran only in main: 7 instructions,
89.94 -> 87.45 GiB (`Remat via recomputation` x7, `Remat via offload` x0, compression x0). Baseline had 143
instructions (84 in the backward body).

Trace (`hmo02/`, steps 180021-23) vs the Sep 24 baseline trace (different main, step 144k; a same-code
main trace does not exist yet):
- XLA remat kernels 0.633 -> 0.0007 s/step; `all-gather.127.remat` is gone, and the backward body now has
  no synchronous all-gather at all (baseline: the original plus the clone).
- Arena 67.49 -> 68.47 GiB. The remat had been cutting phantom (host) bytes, so H-A4 costs ~1 GiB.
- Where the 0.633 s went: 0.227 s was the sync AG clone, mostly rank-skew wait; it moved to the
  backward latent reduce-scatter (`symk_ReduceScatter` moe_latent_proj exposed 0.051 -> 0.255). Of the
  0.400 s of fusion clones, 0.111 s ran under collectives, so ~0.29 s was on the critical path; the power
  cap gives some of that back. Expected ~0.25 s vs -0.157 measured on one draw (seed controls spread
  0.25 MFU, ~0.12 s). Compute-stream busy 11.79 -> 11.63 s cross-version.
- Exposed host copies unchanged (0.618 s), as expected.

#9481 overlap: with the flag there is no XLA-remat re-gather left. #9481's "re-gather attention weights
in the backward" then adds a deliberate second (async) AG per layer in the backward, which holds the
overlap-1 collective slot. It looks redundant with the flag; A/B the stack with and without
`ebde49980b`.

Stack implication: the remat limit (87.66 GiB at slop 85, fraction 0.75) already binds by 2.3 GiB on
main+flag. B's D adds ~18 GiB live across the backward, so remat (device-only view) would cut ~18 GiB
again and the LHS would sit at its limit. The stack needs `--xla_gpu_memory_limit_slop_factor` ~110
(limit 113.4 GiB) or a higher fraction; physical peak ~104 + 18 - E/unfilled savings ~ 120 GiB fits under
138.2.

My score (`mfu30a/score.py`, 180011-180059 minus profiled): treatment median 28.580 (sd 0.41, n=46),
13.734 s; control 28.261 (sd 0.23, n=49), 13.889 s. Step 180000 loss, drop fraction and load-balancing
loss are identical. From 180001, drops and load-balancing loss differ at the 1e-5 level in both
directions; drop median 1.75e-4 vs control 1.92e-4. The loss difference is small and mostly positive.
Read: the program change is placement-only (remat clones removed; buffer offsets and alignment change), so
rounding-level differences from step 180001 are expected (allowed by the fidelity ruling). They also
match B's attention-backward nondeterminism, and a one-signed difference within one pair is what a
chaotic but unbiased divergence looks like: once weights differ, consecutive steps share the offset.
Verdict waits on the same-code repeat `m30-ctl-s0-r2`. If its |dloss| vs s0 is of the same order (max
~4e-4), H-A4 is clean.

## M30A-013 Memory settings for D-containing arms (2026-10-01)

Model, fitted to hmo-02: pool `P = fraction x 184.3`; scheduler/remat limit `L = (P - 35.09) x slop/100`
(87.66 GiB logged = (138.22 - 35.09) x 0.85). Remat's view counts the 18.9 GiB of collective-memory (S(1))
buffers (post-remat 87.45 GiB = arena 68.47 + 18.9); the LHS view excludes S(1) and entry params. So remat
caps the arena at `L - 18.9`, and the worst-case pool use is `35.6 + L - 18.9`. hmo-02's remat peak is the
backward phase (main live ~38.8 + backward body 48.89 = ~89.9 GiB); D adds ~18 GiB live across the
backward, minus ~1.4 from unfilled buffers, giving ~104-106 GiB, so D needs `L >= ~112`.

Recommended pair: fraction 0.78, slop 105. P 143.76, L 114.1 (8-10 GiB above D's view), arena cap 95.2,
worst-case pool use 130.8 (13 under), expected peak ~123, outside reserve 40.5 vs 28.5 needed. Passing
alternatives: (0.80, 100) L 112.4, outside 8.4 GiB free; (0.75, 110) worst case 130.1 vs 138.2. D arms pair
only with stack arms at the same pair. Verification per arm: remat count in `jit_train_step`, `Peak memory
for main`, `memory/peak_gib` < ~139, `memory/limit_gib` = 143.76. Fallback if D's view exceeds ~125:
(0.80, 110), else D's output to pinned host (~0.2 s of C2C, eats most of D's gain).

## M30A-014 End-of-step optimizer D2H: priced, PGLE-profile patch built (2026-10-01)

hmo-02 step 1 timeline (s from launch): backward ends 12.78; momentum H2D .44 12.782-12.850 (exposed, the
first momentum update waits on it), .43/.42 hidden under leaf NS. New momentum ready at 12.855 / 12.991 /
13.143 (leaf 1/2/3 `input_add_reduce_fusion.37/39/41`); expert NS and updates run to 13.33; embedding
scatter-add, Adam H2D (.21/.23 13.382-13.419, mostly exposed) and updates end at 13.448. Then the compute
stream is idle while the writebacks drain serially: .97 13.448-13.520, .98 -13.589, .99 -13.662 (70 ms
each, 155 GB/s, no overlap between them), router .66/.78, embedding .60/.72 13.673-13.708. Tail: 0.261 s.

Price: the three momentum shards have 0.59 / 0.46 / 0.31 s of compute after them, enough to hide all
0.21 s; the embedding m/v (ready at 13.42) has ~0.03 s. Removable ~0.25 s, ~0.21-0.23 s after the
power-cap give-back. No HBM: D2H device buffers live until copy-done at the end either way.
"Overlap with the next step" is ruled out by the no-state-across-steps constraint.

Why PGLE alone misses it: the profile does carry each copy's real cost (copy-start.97/98/99 ~70 ms), but the
GPU LHS models async memcpy with unlimited concurrency (`kNumAsyncMemcpy = INT_MAX`), so it covers ~70 ms
for the whole batch. `autoresearch/loop-260930-mfu30/a/pgle_patch_d2h.py <rows.pkl> <in> <out>` sets every
D2H copy-start of >= 1 ms/step to the serialized batch total x 1.1 (276 ms on hmo-02's profile; 7 copies).
It applies after `pgle_build.sh`, before the commit. Without PGLE, the fallback is a JAX
optimization_barrier tie (allowed on the host path), which with T-shirt latencies hides ~half (~0.1 s).

Also seen in the #9481 PGLE trace t21: every forward carry D2H (4.4 ms, 48/step) is fully exposed under
PGLE (0.19 s/step vs 0.006 on main and hmo-02); compute stops at copy-start and resumes at copy-done. The
stack's PGLE arm should check this; it may eat a large part of PGLE's gain. Also under PGLE, t21 hoisted the
first momentum H2D (copy-start.44) to the start of the backward, holding 10 GiB through the backward peak;
on the D stack at 0.78/105 that fits (~133 vs 143.76 GiB) but shows up in memory/peak_gib.

## M30A-015 Carry-copy stream collision: mechanism and fix (2026-10-01)

Mechanism, from XLA source at the pinned `708c3a4ec79c`:
- `DynamicSliceCopyFusionAsyncWrapper` (pre-scheduling) wraps every dynamic-slice / DUS copy fusion in an
  async pair: the per-layer weight slices from the scan's stacked params, and the carry DUS-to-host /
  DS-from-host. Its only off switch, `--xla_gpu_experimental_dynamic_slice_fusion_verify_offsets`,
  de-asyncs all of them and adds runtime offset checks. Unusable.
- `ExecutionStreamAssignment` gives every compute-scope async start (those plus `copy-start`)
  `ComputationStreamId(n mod kDefaultNumComputeStreams=4)`, with one counter over each computation's
  post-order (BFS over computations). That modulus is a constant;
  `--xla_gpu_executable_num_compute_streams` only raises the allocated-stream floor. Within a loop body,
  which slices share the carry copy's stream depends only on their relative post-order positions mod 4.
  A global phase shift therefore cannot fix it, and any change to the body's async ops re-draws it.
- An explicit `_xla_stream_annotation` frontend attribute overrides the assignment, but
  `CreateAsyncInstructions` copies metadata and backend config, not frontend attributes, so JAX cannot
  put one on these async starts.
- The LHS models async memcpys with unlimited concurrency, so it never sees the shared queue.
Conclusion: no flag and no clean JAX-level annotation.

Deterministic fix, written but not pushed: XLA branch `mcwitt/adhoc-host-transfer-streams` in worktree
`~/projects/xla.host-transfer-streams` (commit `283d5b6d98` on `708c3a4ec79c`, +74 lines; patch copied to
`autoresearch/loop-260930-mfu30/a/xla-host-transfer-streams.patch`). With `XLA_GPU_HOST_TRANSFER_STREAMS=1`,
async starts touching host memory get dedicated streams: compute stream 4 for H2D, compute stream 5 for
D2H. That covers `copy-start` with an S(5) source or destination, and async slice fusions reading or
writing S(5). Everything else stays round-robin over 0-3. Slice copies can never queue behind a host
transfer; the H2D and D2H batches can overlap each other. Zero runtime cost, same ordering semantics
(async start/done events), bit-identical when the env var is unset. Needs a branch push to
marin-community/xla plus a `marin-pjrt.yaml` dispatch (candidate prerelease, ~3.5 h cold build), then a
`--pjrt-wheel` sideload as in August's H11. That is an external write beyond this agent's permissions.

JAX-level fallback (not built): make the offloaded carry depend on every layer weight slice through a
non-foldable integer zero. Use `z = OR_k((u16(slice_k[0]) >> 15) & (~u16(slice_k[0]) >> 15))`,
`x2 = bitcast(bitcast(x, u16) | z)`, and route every use through `name(x2)`. The forward D2H then cannot
issue before the layer's slices finish. The backward uses the offloaded `x2`, so no extra residual.
Bitwise exact. It costs ~25 ms/step always (an extra carry pass ~14 ms, plus the layer start waiting for
all slices ~10 ms), it covers only the forward, and XLA must be checked not to fold `z`.

## M30A-016 Stream patch approved: branch pushed, cluster build running (2026-10-01)

Orchestrator approved a branch push only: no PR, no release, and no `marin-pjrt.yaml` dispatch. Pushed
`mcwitt/adhoc-host-transfer-streams` (`283d5b6d98`) to marin-community/xla; `marin-pjrt.yaml` triggers on
main only, so the push built nothing. Built on the cluster instead:
`experiments/grug/moe_hero_ep/pjrt_build_job.sh` (copied from `research/mcwitt/xla-ragged-dot`, then changed):
- `HERMETIC_NCCL_VERSION=2.30.7`, matching the fork's production `marin/build_pjrt.sh`; the copied script
  left NCCL at jax's default, and a 2.29.7-header wheel hung the ragged hero before.
- `CUDNN_VERSION` is now required (uv.lock: `nvidia-cudnn-cu13` 9.19.0.56).
- `S3_DEST` is required.
- A `SENTINEL_FILE`/`SENTINEL` check that the delta is present.
Version check: config.json at the ref gives jax `2d66622450e2` / 0.11.1, and the suffix is
`+marin.<sha12>` (carries its own `+`), so the wheel is `0.11.1+marin.283d5b6d98xx` and
`verify_ragged_pjrt` accepts it. Job `/mwittmann/m30a-pjrt-build-01` (GB200x1, 64 CPU, 400 GB) uploads to
`s3://marin-us-east-02a/marin/research/mcwitt-mfu30/pjrt/<sha12>/`.

Consumption: the base had no wheel option, so this branch adds the ragged-dot campaign's `--pip-package`
(`GrugRunConfig.pip_packages` -> dispatch -> fray `EnvironmentSpec.pip_packages`, installed after the
sync). Pass a presigned URL from `rclone link --expire 168h`.

Smoke (`autoresearch/loop-260930-mfu30/a/stream_smoke.{py,sh}`, GB200x1): an 8-layer rematted scan with
six stacked weights and the carry offloaded to pinned host, run with and without
`XLA_GPU_HOST_TRANSFER_STREAMS=1` under `TF_CPP_VMODULE=execution_stream_assignment=3`. It prints each
async start's stream and checks the gradients are bitwise equal across modes.

## M30A-017 Wheel built; GB200 smoke passes (2026-10-01)

Wheel `jax_cuda13_pjrt-0.11.1+marin.283d5b6d98cd-py3-none-manylinux_2_27_aarch64.whl` (sha256
`1ab96eb43cfeb6f5b14b79c797c0dd951c41a31eefd3b36b3f86b0c77e35f881`, 126 MB), built in 15 min by
`/mwittmann/m30a-pjrt-build-01`: cuDNN 9.19.0 headers, hermetic NCCL 2.30.7, jax `2d66622450e2`. At
`s3://marin-us-east-02a/marin/research/mcwitt-mfu30/pjrt/283d5b6d98cd/`. Presigned URL (168 h) in scratchpad
`mfu30a/pjrt_283d5b6d98cd.url`; a ranged GET returns 206.

Smoke `stream_smoke.sh` on GB200x1 (8-layer rematted scan, 6 stacked weights, carry offloaded to pinned
host, `TF_CPP_VMODULE=execution_stream_assignment=3`):
- Production wheel (`m30a-stream-smoke-prod-01`): carry D2H `dynamic-update-slice-start.1` on compute
  stream 2, shared with weight slices `fusion-start.2/.6/.10/.15`; carry H2D `dynamic-slice-start.1` on
  stream 3, shared with `.3/.7/.11/.14`. This reproduces C's collision; the env var is a no-op there.
- New wheel, env unset (`m30a-stream-smoke-hts-01`): assignment identical to production.
- New wheel, `XLA_GPU_HOST_TRANSFER_STREAMS=1`: carry H2D on stream 4, carry D2H on stream 5, all weight
  slices on 0-3. Gradients are bitwise equal to the env-unset run, and step time is unchanged
  (84.7 vs 84.8 ms; the toy does not stall).

Rack pairing (next): the final program on the new wheel via `--pip-package <url>`, with and without
`--env XLA_GPU_HOST_TRANSFER_STREAMS=1`, gated on `carry_stall.py` < 10 ms/step. The stack needs this
branch's `--pip-package` plumbing (commit `cf5bc74409`: dispatch, train, launch_diagnostics).

## M30A-018 Ownership and loss calibration (2026-10-01)

C owns the final-program arms: C cherry-picks `cf5bc74409` (`--pip-package`) onto the stack and runs the
F1 (`XLA_GPU_HOST_TRANSFER_STREAMS=1`) / F0 pair on the 283d5b6d98cd wheel. Carry prefetch (arm2) stays a
candidate add-on, as its own paired comparison against the final program. `arm2_on_stack.sh` now takes
`EXTRA_ENV` and passes `--pip-package` through the stack switches.

Same-code repeat `m30-ctl-s0-r2` vs `mhep-ctx4k-s0`: MFU 28.235 vs 28.261; |dloss| max 3.3e-4, late mean
-3.9e-5, 35/89 positive. hmo-02's drift (max 4.4e-4, late mean +1.58e-4, 48/49 positive) is near that max
but more one-signed. The final program gets replicated loss pairs to settle it.

## M30A-019 Custom PJRT wheel ON HOLD (2026-10-01)

The orchestrator withdrew approval for campaign rack jobs that install the 283d5b6d98cd wheel, pending the
user's decision. Do not submit any job that installs it, and do not pass the presigned URL to other
agents. The pushed XLA branch stays as-is. The two single-node smokes (`m30a-stream-smoke-prod-01`,
`-hts-01`) ran before the hold. If the user declines, the fallback is the JAX-only integer-zero
dependency (M30A-015), built only if `carry_stall.py` shows the stall on the final program.

## M30A-020 Wheel hold released (2026-10-01)

The user approved the custom wheel for campaign rack jobs and approved keeping the XLA branch. C owns the
final-program arms: C cherry-picks `cf5bc74409`, adds `--wheel` to `stack/arm.sh`, and runs F1 (env on,
traced) / F0 (env off) on the 283d5b6d98cd wheel after the -03 lineage pick.

A's analysis checklist for F1/F0:
- F1: carry H2D/D2H memcpys on streams 4/5, no weight slices on those streams.
- F1: `carry_stall.py` < 10 ms/step.
- F0 vs production-wheel arms: same stream layout, no stream-related regression (cuDNN headers differ).
Later, for the PGLE arm: `copy-start.44` placement, the D2H tail after `pgle_patch_d2h.py`, and peak memory.

## M30A-021 F1/F0 analysis pipeline ready (2026-10-01)

B found stackpipe-03 lost the draw (carry_stall 147 ms/step vs 2.8 in sonic-02, +0.15 s/step). Queued on
`research/mcwitt/mfu30-final-seq` @ `d4234c88e7`: `m30-f1-seq-02` (env on) and `m30-f0-seq-01` (env off),
same custom wheel. `autoresearch/loop-260930-mfu30/a/analyze_arm.sh <run> <dir>` downloads the rank-0
xplane, then runs `stream_check.py` (per-stream memcpy kinds; whether the carry streams share anything),
C's `carry_stall.py`, and the anatomy summary. On hmo-02 (production wheel) the carry D2H/H2D share all
four memcpy streams with ~740 D2D slice copies per step each, and carry_stall reads 6.0 ms/step there
(a won draw).

## M30A-022 F1 (m30-f1-seq-02) trace: stream fix works (2026-10-01)

Orchestrator score: 30.219 MFU / 12.990 s (steady 30.0-30.34), peak 123.50 of 143.75 GiB; loss max |d|
3.8e-4, late mean +9e-5, 44/49 positive.
- Wheel loaded: all 16 tasks logged `- jax-cuda13-pjrt==0.11.1+marin.708c3a4ec79c` / `+
  jax-cuda13-pjrt==0.11.1+marin.283d5b6d98cd` (from the presigned URL). The job succeeded, so
  `verify_ragged_pjrt` accepted the wheel.
- Streams: XLA borrows its async streams from a pool each execution, so the xprof stream behind an
  execution-stream id rotates between steps. Aggregated over steps, the slices appear to share the carry
  streams; per step they never do. Every step has one stream with only carry H2D (48) plus optimizer H2D
  (54), one with only carry D2H (48) plus optimizer D2H (56), and the D2D slice copies on the other four
  (`stream_check.py` now groups by step).
- `carry_stall.py`: 3.7 ms/step (stackseq-03 on the production wheel: 140.9 ms/step, a lost draw).

Exposed copies, s/step (`copy_exposure.py`):

| class | stackseq-03 | F1 | hmo-02 |
|---|---|---|---|
| opt-state D2H (end-of-step tail) | 0.261 | 0.261 | 0.260 |
| carry H2D (backward reload) | 0.219 | 0.218 | 0.219 |
| carry D2H | 0.141 | 0.004 | 0.006 |
| opt-state H2D | 0.100 | 0.101 | 0.099 |
| other D2D | 0.050 | 0.056 | 0.038 |
| total | 0.773 | 0.641 | 0.624 |

What is left: the end-of-step D2H tail (0.26; target of `pgle_patch_d2h.py` under PGLE), the backward carry
reload (0.22; arm2 carry prefetch), and the first momentum and embedding H2D (0.10; PGLE hoists it).

## M30A-023 F0 (m30-f0-seq-02) reused F1's cached executable; not a valid env-off control (2026-10-01)

Per-step `stream_check.py`: in F0 the carry copies sit alone on two direction-split streams, sharing only
with optimizer-state copies, and there are six async memcpy streams. That is exactly F1's patched layout.
stackseq-03 (production wheel, stalled) shows carry D2H and H2D sharing one stream with ~720 slice copies in
every step. The GB200x1 smoke on the same wheel with the env unset produced the production layout (four
streams, collision). So F0 did not win a draw; it ran the patched assignment.

Why: F0 never compiled `jit_train_step`. With the remat VLOG on, F0 logged 20,992 `hlo_rematerialization`
lines for other modules but zero `Rematerialized ... in module jit_train_step` lines; F1 logged 64 (one
per process). F0 loaded F1's executable from the persistent compilation cache. The cache key covers the
program, compile options and XLA flags, but not the `XLA_GPU_HOST_TRANSFER_STREAMS` env var, and stream
assignment is fixed at compile time.

Consequences:
- F0 is not a control. "The wheel alone removes the stall" is unsupported; the smoke says the wheel without
  the env behaves like production.
- The env-gated switch is cache-unsafe both ways: whichever setting compiles a program first is what every
  later run of the same program and flags gets. Any env-on/off pair needs a forced compile, e.g.
  `JAX_ENABLE_COMPILATION_CACHE=false` (forwarded `JAX_` prefix), or a flag difference in the key.
- For deployment, set the env on every run and never mix settings within a cache, or turn the switch into
  a DebugOptions flag (part of the key) or default-on in a promoted wheel.

## M30A-024 Close-out: result and reusable lessons (2026-10-01)

Result: the campaign goal was met. The final program (`research/mcwitt/mfu30-final-seq` @ `d4234c88e7`,
custom wheel `0.11.1+marin.283d5b6d98cd` with `XLA_GPU_HOST_TRANSFER_STREAMS=1`, H-A4 at fraction 0.78 /
slop 105) confirmed at 30.218 / 30.206 / 30.206 MFU on seeds 0/1/2, against main at 28.260 / 28.235 /
28.266. Loss divergence was inside the same-code range (seed-2 main-vs-main reached max 9.1e-4). This
direction contributed H-A4 (remat accounting, -0.157 s/step on its own), the memory settings for D, and the
stream patch, which removes a lost-draw stall worth ~0.14 s/step.

Reusable lessons:
1. **XLA's post-schedule remat counts pinned-host buffers as device memory** unless
   `--xla_gpu_enable_host_memory_offloading=true` is set (`AllocatedSize` in hlo_rematerialization.cc). With
   a 36 GiB host carry stack, the hero ran 143 phantom remat clones (0.63 s/step of kernels, including
   synchronous attention-weight re-gathers). The flag removes them for ~1 GiB of arena. Engagement check:
   rank-0 "Rematerialized N instructions in module jit_train_step" and zero "Remat via offload".
2. **XLA's scheduler/remat limit is `(pool - device params) x slop`**, where pool =
   `XLA_PYTHON_CLIENT_MEM_FRACTION x device memory`, not 0.8 x device memory. Remat's view includes the
   collective (S(1)) buffers, so the arena is capped at `limit - S(1)`. Size slop and fraction together
   when a change adds live memory (D needed 0.78 / 105).
3. **Async copies share 4 round-robin compute streams.** Assignment follows post-order (`kDefaultNumComputeStreams=4`,
   no flag), so a µs weight-slice copy can queue behind a 4.4 ms host carry transfer. Whether it does is a
   lottery that any program change or PGLE profile re-draws (stackseq-03 lost: 0.141 s/step). Frontend
   stream annotations can't reach these async starts because `CreateAsyncInstructions` drops frontend
   attributes. Fix: XLA patch `mcwitt/adhoc-host-transfer-streams` (`283d5b6d98`), which gives host
   transfers dedicated streams under `XLA_GPU_HOST_TRANSFER_STREAMS=1`. Built on-cluster in 15 min
   (`pjrt_build_job.sh`, pin `HERMETIC_NCCL_VERSION=2.30.7`); validate per step, since xprof's pooled stream
   numbers rotate across steps.
4. **The persistent compilation cache ignores env-gated XLA behavior.** The cache key covers program, options
   and XLA flags, not arbitrary env vars, so an env-off run silently reused the env-on executable (F0). Any
   env-gated on/off pair needs `JAX_ENABLE_COMPILATION_CACHE=false` or a key-changing flag, and deployments
   must not mix settings within one cache. A proper DebugOptions flag avoids this.
5. **Untested lever: `pgle_patch_d2h.py`.** The GPU LHS models async memcpys with unlimited concurrency, so
   even with PGLE the ~0.26 s end-of-step optimizer D2H tail stays exposed. Costing every large D2H
   copy-start at the serialized batch total should hide ~0.2 s with no HBM cost. A patched profile exists
   from m30-f1-seq-02 (`pgle/m30-f1-seq-02-d2h.pbtxt` on the stack branch) but no arm ran it. Watch for
   PGLE hoisting the first momentum H2D into the backward (+10 GiB at the peak).
6. Placement and offload changes still need a GPU smoke on a real restore. Full optimizer residency (H-A1) did
   not fit next to D; partial residency remains unsafe (August C3').

Remaining exposed copies on the final program (F1): end-of-step optimizer D2H 0.26 s, backward carry reload
0.22 s, first momentum and embedding H2D 0.10 s. Carry prefetch (`arm2_on_stack.sh`) was never run.
