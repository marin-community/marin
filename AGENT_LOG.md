# AGENT_LOG — ep25-d1 (custom scatter-add adjoint for gather-dispatch backward)

Worktree: `/home/marin/projects/marin/.worktrees/ep25-d1-adjoint`, branch `agent/ep25-d1-adjoint`, base `rav/ep-2` @ fe21ea495.

## Design (established from comments 5073017396, 5074952738, EVIDENCE-slack)

The fixed-a2a MoE has two structured gathers whose XLA-autodiff backward is a
generic scatter-add — the two costs the XProf trace flags (gather-dispatch backward
+ combine/all-gather backward, the latter called out by David Hall as
"pathologically bad ... worth custom vjp work"):

- **Dispatch** (gather mode): `send_x = padded_x[token_sources]`. Fan-out topk, so its
  transpose is a genuine segment-sum: `d_x[t] = sum_k cotangent[linear_indices[t*topk+k]]`
  over kept assignments. Replaces scatter-add with gather+reduce.
- **Combine**: `gathered = send_output[linear_indices]`. Injective on kept assignments,
  so its transpose is a pure gather along the slot->assignment inverse
  (`assignment_sources`), not a scatter-add: `d_send_output[j] = cotangent[assignment_sources[j]]`.

Both reuse the forward's int32 index composition. Exact transposes (no approximation).

Flags: `SCALE_A2A_GATHER_DISPATCH=1` selects gather dispatch (the 20.558% control);
`SCALE_A2A_CUSTOM_ADJOINT=1` (requires gather dispatch) wraps both gathers in custom_vjp.

## Check-in (start)
Findings so far:
- Reconstructed gather-dispatch patch matches comment 5073017396 snippet exactly; commit 1.
- custom_vjp math derived and verified on paper for both dispatch (segment-sum) and combine
  (injective gather); both are exact transposes reusing linear_indices / assignment_sources.
- Combine adjoint added to scope per David Hall's Slack note (prime target).
Confidence: 6/10 that this direction is a significant step toward 25% MFU (backward is 1 of 2
top costs; combine-backward being pathological is the upside; A/B not yet measured).
Next: CPU numerics test (gradient parity rtol=atol=1e-5, identical drop counts), then 1-replica smoke, then rack A/B.

## Check-in 23:28 UTC
Findings so far (numbers first):
- IMPLEMENTATION DONE + VALIDATED. Two commits: gather-dispatch reconstruction
  (SCALE_A2A_GATHER_DISPATCH) + structured custom_vjp for BOTH gathers
  (SCALE_A2A_CUSTOM_ADJOINT). Dispatch backward = segment-sum over topk; combine
  backward = injective gather along assignment_sources. Exact transposes.
- NUMERICS: new CPU tests pass. Gradient parity vs autodiff at rtol=atol=1e-5 for
  x/combine_weights/w13/w2; identical nonzero drop counts. (test_custom_adjoint_matches_autodiff_gradients,
  test_gather_dispatch_matches_scatter_forward_and_drops)
- HLO: backward drops from scatter=544 -> scatter=0 (gathers only) with custom adjoint
  at d5120/8-of-256. The pathological scatter-add transpose is fully eliminated.
- CLUSTER OVERLAP (critical): rav is LIVE on this exact direction right now.
  /rav/ep64-dispatch-grad-only-30-v1 (running) shows **p50 MFU 25.43%** (p10 25.20 / p90 25.63,
  415.8K tok/s, ~10.09s/step) at the EP64 operating point — past the 25% goal vs 20.558% baseline.
  His custom-combine XProf (profile-ep64-custom-combine-nocg): backward scatter gone, comm now
  29.5%, top op ncclDevKernel_SendRecv 22.4% — consistent with the adjoint removing the backward cost.
  Caveat: "grad-only" may be a partial-step bench; treat 25.43% as directional not a locked matched A/B.
  rav's 120-step ep64-custom-combine-stability was KILLED (no completed matched MFU A/B yet).
Confidence: 8/10 that this direction is a significant step toward 25% MFU (rav's live 25.4% + my
  scatter->0 HLO + 1e-5 parity all agree; the only gap is a clean completed 120-step matched A/B).
Next: coordinate before burning a shared rack (etiquette: avoid duplicating rav's in-flight work,
  cluster heavily contended). Messaging coordinator with numbers + recommendation.

## Handoff artifact (commits on agent/ep25-d1-adjoint)
- 45ce02d20  gather-dispatch reconstruction (SCALE_A2A_GATHER_DISPATCH) — the 20.558% control path.
- c9e30f848  structured custom_vjp for BOTH gathers (SCALE_A2A_CUSTOM_ADJOINT) — the treatment. THIS is the deliverable.
- f2cbc192c  black formatting. Tests live in lib/levanter/tests/grug/test_grugformer_moe.py
  (test_gather_dispatch_matches_scatter_forward_and_drops, test_custom_adjoint_matches_autodiff_gradients).
Do NOT hand to rav directly (coordinator's call).

## Prepared rack A/B (one keystroke when a window opens; coordinator deferred the rack for now)
Control (gather-dispatch, XLA-autodiff backward) — DATE=$(date +%Y%m%d-%H%M):
  IRIS_USER=mwittmann .venv/bin/iris --cluster=marin job run --no-wait \
    --target-cluster cw-us-east-08a --priority interactive \
    --job-name ep25d1-adj-control-120-vDATE -e RUN_ID ep25d1-adj-control-120-vDATE \
    -e SCALE_ATTN_IMPL gpu_fa4_cute -e SCALE_WATCH_INTERVAL 0 -e SCALE_CHECKPOINTS local \
    -e SCALE_A2A_FIXED 1 -e SCALE_A2A_CHUNKS 1 -e SCALE_A2A_NO_BARRIER 1 -e SCALE_A2A_GATHER_DISPATCH 1 \
    -e SCALE_GPUS_PER_NODE 4 -e SCALE_GPU_TYPE GB200 -e SCALE_GPU_REPLICAS 16 \
    -e SCALE_EXPERT_AXIS 64 -e SCALE_NUM_EXPERTS 256 -e SCALE_TOP_K 8 \
    -e SCALE_HIDDEN_DIM 5120 -e SCALE_NUM_LAYERS 48 -e SCALE_INTERMEDIATE 1280 -e SCALE_SHARED_INTERMEDIATE 5120 \
    -e SCALE_SEQ_LEN 4096 -e SCALE_BATCH 1024 -e SCALE_SLIDING_WINDOW 2048 \
    -e SCALE_STEPS 120 -e SCALE_MOE_IMPL ragged_all_to_all -e SCALE_OPTIMIZER muonh -e SCALE_MUON_SYRK 1 \
    -e SCALE_SCAN_LAYERS 1 -e SCALE_REMAT recompute_all \
    -e SCALE_TRACKER json_logger -e SCALE_JSON_LOGGER ep25d1-adj-control-120-vDATE.metrics \
    -e SCALE_DISABLE_CHECKPOINT 1 \
    -- python -m experiments.grug.moe.launch_cw_scale --version ep25d1-adj-dev --run

Treatment (adds custom adjoint): identical EXCEPT job-name/RUN_ID -> ep25d1-adj-custom-120-vDATE, add
    -e SCALE_A2A_CUSTOM_ADJOINT 1 , and --version ep25d1-adj-dev .
Report p10/p50/p90 MFU (2.5 PF/s denom), tok/s, step time, drop fraction (dropped/assignments), loss trajectory for both.

## Check-in 23:37 UTC — profile-lead assessment + smoke status
Profile leads from EVIDENCE-slack (HLO on 8-CPU-device shard_map repro of the real fixed-a2a path, grad HLO):
- **unstack (David's lead 2)**: 18 unstack ops, ALL in op_name .../combine/unstack, and they lower to
  `slice`+`bitcast` (views), not kernels. Origin: jnp.stack(output_parts) in the combine scope + its
  backward transpose (stack->unstack) and the per-local-expert loop indexing. VERDICT: cosmetic, ~free;
  not worth a dedicated fix. Ranks LOW.
- **reduce-scatter.10 (David's lead 3)**: NOT reproducible without full FSDP weight sharding — my repE
  shows all-reduce (shard_map psum for dropped_total + grad psum), reduce-scatter=0. On the rack it is the
  FSDP gradient reduce-scatter across the data axis; overlapping it with next-layer compute is blocked by
  the layer scan (ties to #7507 "scan blocks weight-gather overlap"). VERDICT: real but it's a
  scheduling/overlap item belonging to the pipelined-a2a / overlap direction (peer), NOT an adjoint
  sub-direction; LHS/auto-PGLE sealed null here so it needs manual structural overlap. Ranks MEDIUM,
  owned by the overlap thread.
- Post-adjoint the bottleneck is comm: rav's profile shows 29.5% comm / SendRecv 22.4%. So the a2a
  SendRecv legs (fixed-A2A) and overlap are now the top remaining costs — raises the value of the
  overlap/leg-batching directions over the two leads above.

Smoke A/B (4-GPU EP4, remat+scan+shard_map e2e; confirms adjoint survives e2e, directional delta):
- ep25d1-smoke-ep4-custom-0724-1633: RUNNING (past setup -> bundle+code ship OK; sentinel/custom path active).
- ep25d1-smoke-ep4-control-0724-1636: resubmitted (prior two control attempts hit transient [iris setup] step 1/2).
Confidence: 8/10 (unchanged). Next: collect smoke MFU delta + drop fractions; tripwire-poll rav.

## SMOKE A/B RESULT (4-GPU EP4, hidden3072 L12 e64 top8 batch16 seq4096, remat_all + scan, muonh, 40 steps)
| arm | p10 | p50 | p90 | tok/s | loss@39 | samples |
|---|---|---|---|---|---|---|
| control (gather-dispatch, autodiff backward) | 11.99 | 12.146 | 12.27 | 81.5K | 6.819 | 39 |
| treatment (+SCALE_A2A_CUSTOM_ADJOINT) | 12.16 | 12.392 | 12.56 | 82.2K | 6.857 | 39 |
Delta: p50 +0.246pp (+2.0% rel tok/s), medians cleanly separated (control p90 12.27 < treatment p50 12.39).
Both ran 40 steps clean on real GB200 (no OOM/NaN), loss descending 9->6.8. CONFIRMS the custom adjoint
survives remat_all + layer scan + shard_map/EP end-to-end and gives a consistent throughput win.
CAVEAT: EP4 tiny model -> backward-scatter is a small step fraction, so +0.25pp here understates the
operating point; rav's live EP64 signal (~20.6% gather-dispatch -> 25.4% custom, +~4.9pp) is the scale-up.
Loss 0.04 apart = known independent-run RNG divergence (baseline saw 0.064 gather-vs-scatter); kernel test
proves grad parity at 1e-5, so this is not a numerics bug.

## Check-in 23:42 UTC — RANKING of candidate pool + tripwire
Tripwire: rav ACTIVE, not idle — moved off the adjoint to /rav/ep64-batched-expert-gemms-30-v2 (running),
i.e. he's now on the per-local-expert GEMM/leg batching (the secondary item I was told to leave to a peer).
No trigger to launch my rack A/B (not idle >=45m; his runs are progressing, not just dying). Defer stands.

RANKING (contribution >=1pp toward LOCKED 25%, preserving fidelity; post-adjoint a2a legs = 26-29% of step):
1. **1a Lock the adjoint (9/10)** — the win is already demonstrated (rav live 25.4% grad-only; my HLO scatter 544->0,
   1e-5 grad parity, EP4 smoke +0.25pp e2e). A matched 120-step A/B + drop fractions is the highest-value,
   most-measurable, lowest-risk action; it converts a strong signal into the locked result the record lacks.
2. **4 Rotation ppermute overlap (5/10)** — structurally overlaps the now-dominant SendRecv a2a legs (scheduler-
   independent, the only kind that works here since LHS/PGLE are sealed null); uncovered by rav's own rotation work;
   CPU parity passing. High ceiling (~+3-4pp if 50% of the 29% comm overlaps) but risky to land under remat+scan.
3. **2 Transport bake-off (5/10)** — settles fixed+gather vs ragged(one-shot off) vs ring_cute at the real
   e256/EP64 shape and gates MXFP8(5); post-adjoint the transport IS the bottleneck so the choice is worth pp,
   but its own delta may return parity and d2 is submission-blocked (coordinator-relayed).
4. **5 MXFP8 (4/10)** — attacks the ~70% compute majority (measured 1.308x within-EP8, smaller arena eases EP OOM);
   discounted by the microbench-overstates-e2e lesson, +0.11-0.21% held-out loss (fidelity trade), and #7079 unmerged.
5. **4b Token-chunk pipelining (4/10)** — the ONLY overlap mechanism with a landed win here (FSDP expert chunk-2
   21.8->22.7, +0.9pp); lower ceiling than rotation but more proven; fallback if 4 fails.
6. **6 fa4-lse primal output / #7507 (4/10)** — est +~1pp, independent of the a2a budget so it stacks; unstarted,
   moderate effort; competes on EV with the comm levers.
7. **1c Overlap reduce-scatter.10 (3/10)** — REAL but my HLO check shows it needs full FSDP weight sharding to even
   appear; it's the grad RS whose next-layer overlap is blocked by the layer scan (#7507), needs manual structural
   overlap (LHS/PGLE null). Belongs to the overlap/#7507 thread, not the adjoint; uncertain standalone pp.
8. **1b Eliminate stray unstack (1/10)** — my HLO check: all 18 unstack ops are in combine/ and lower to slice+bitcast
   (views, ~free), from jnp.stack(output_parts) + its backward transpose. Cosmetic; not worth a fix.
9. **3 TE-at-tip NCCL_EP (1/10)** — d3 trending confident-negative: #3231's collective-stream pin deterministically
   crashes 64-GPU first exec; stripped, tip==old wheel ~17% vs 18% anchor. Not a path to 25%; value is the NVIDIA report.

## Check-in 23:46 UTC — watcher + tripwire (per coordinator)
- Tripwire ARMED, NOT triggered. rav active (not idle). Prepared 120-step matched A/B commands stand as
  the fallback to lock 1a if rav goes idle >=45m or his runs keep dying. No matched 120-step adjoint A/B
  exists on the record yet.
- WATCH /rav/ep64-batched-expert-gemms-30-v2-20260724-2339 (leg-batching = the batching-vs-rotation race vs d4):
  - Config harvested = the REAL operating point: d5120, 256e/top8, 48L, EP64, batch1024, seq4096, muonh,
    remat recompute_all, scan (split_scan_layers), 30 steps. Same denom (2.5 PF/s, theoretical_flops 1e16).
  - v1 (2338) FAILED at [iris setup]; v2 (2339) RUNNING, still in compile/warmup — 96 watch(grad/param) summaries
    logged, ZERO throughput/mfu samples so far. No step-time/MFU number to harvest yet. Re-poll next round.
- No new build work (per coordinator). Holding for reassignment (new sub-direction or watch/lock role).

## Check-in 00:08 UTC — drop metric landed + 1a A/B control submitted
- DROP METRIC: SCALE_REPORT_DROPS=1 threads a per-layer dropped-assignment count up the qb_beta aux
  channel (scan stacks [L]); train.py logs moe/dropped_assignments + moe/drop_fraction (denom
  B*S*topk*num_layers). Threaded unconditionally as cheap int32; only the overflow compute is gated.
  Validated CPU e2e (reference attn, EP1 mesh): forward/loss/grad-norm IDENTICAL with flag on/off and
  custom-vs-autodiff; report-off path unchanged. pyrefly clean; test_model.py + adjoint kernel tests pass.
  Committed 4fbc89152.
- 1a A/B CONTROL SUBMITTED (one rack, one-in-flight per rule): /mwittmann/ep25d1-adj-control-120-0724-1707
  = gather-dispatch + autodiff backward + SCALE_REPORT_DROPS=1, 120 steps, SCALE_DISABLE_CHECKPOINT=1,
  json_logger, operating point (d5120/256e-top8/48L/EP64/b1024). Will babysit to terminal, THEN submit
  treatment (+SCALE_A2A_CUSTOM_ADJOINT=1) back-to-back. NOT resubmitting on PENDING (rack at capacity).
- JOB-STATE MUTATIONS BY ME THIS CHECK-IN: none (only my own submission).

## Watcher harvest 00:14 UTC — rav leg-batching stability-120 (d4-rotation comparator)
/rav/ep64-batched-expert-stability-120-v1-20260724-2353 (RUNNING, step 59/120, real 120-step, not grad-only):
  p50 MFU 25.39% (p10 25.02 / p90 26.00, 57 samples), loss 6.823 descending. Operating point
  (d5120/256e-top8/48L/EP64/b1024). => leg-batching (on top of the custom adjoint) clears 25% on a REAL
  run; this is the empirical bar d4's rotation must beat/match. Both restructure the a2a loop once (the
  ranked-comment "do it once" decision now has a batching number: 25.39% p50).
My 1a A/B control (/mwittmann/ep25d1-adj-control-120-0724-1707) still compiling; background monitor armed
for drop_fraction + MFU + terminal, then I submit the treatment leg.

## Check-in 00:20 UTC — fidelity framing + handoff candidate
- 1a A/B REFRAMED (coordinator): raw speed to 25% is ~settled by rav's leg-batching stability-120 (25.39%
  p50, real run). My A/B's UNIQUE contribution is the FIDELITY verdict — the FIRST drop_fraction ever
  measured at the fixed path's 64x256 bucket granularity (Larry's explicitly-untested territory; rav's
  runs have report_capacity_overflow OFF so they log no drops).
- DECISION RULE for the record: drop_fraction <= ~3% (Larry's known-acceptable ref at 8 buckets)
  => fixed path fidelity-CLEARED, goal likely met by adjoint(+batching) pending loss parity.
  drop_fraction materially >3% => goal-relevant NEGATIVE for the fixed transport, elevates d2's
  ragged/ring arms from "decision quality" to "necessary" (coordinator re-ranks immediately).
- HANDOFF CANDIDATE: commit 4fbc89152 (SCALE_REPORT_DROPS drop-fraction metric) is exactly what rav
  will want to gate drops on his fixed-path runs. Flagged for the coordinator to relay; NO direct contact
  from me.
- STATUS: control still compiling (monitor iter 2, running). Will report BOTH legs' drop_fraction + p50
  MFU + loss trajectory side-by-side the moment treatment terminates.

## Check-in 05:20 UTC — CONTROL results + DROP-METRIC BUG + reconciliation
CONTROL /mwittmann/ep25d1-adj-control-120-0724-1707 SUCCEEDED (34 min):
  p50 MFU 20.61% (p10 20.51 / p90 20.69, 119 samples) — reproduces the 20.558% gather-dispatch baseline.
  loss descending 5.836 -> 5.738 over 120 steps. Healthy.
  BUT moe/drop_fraction logged NOTHING -> found the bug (below).

DROP-METRIC LOGGING BUG (FIXED, commit 2d4a87395): the grug loop logs train/loss via state_callbacks,
NOT the returned `metrics` dict, so my moe/ keys were never emitted. Fix: explicit levanter.tracker.log
of moe/dropped_assignments + moe/drop_fraction in the loop. pyrefly clean, formatted.

RECONCILIATION (d3's 65-68% "drop fraction"; rav cf1.2 capacity_overflow_rate_mean 0.77-0.92):
- My metric is dimensionally CORRECT: dropped_total = psum over (replica_dcn,data,expert) => GLOBAL per
  layer; denominator B*S*topk*num_layers = GLOBAL. So mine reads the true token-drop-fraction.
- Hand estimate at the operating point: tokens_per_shard=65,536 (4.19M tokens / 64 batch-shards),
  assignments_per_shard=524,288, capacity=ceil(1.0*524288/256)=2048 == MEAN load per (shard,expert).
  Capacity==mean => at ~random routing each expert bucket ~ N(2048, ~45), E[(X-2048)+] ~= 45*0.399 ~= 18
  per expert => ~256*18/524288 ~= 0.9% TOKEN-drop-fraction. NOT 65%.
- Key insight: capacity==mean means ~half of experts/buckets overflow their tail -> a "fraction of
  buckets that overflow" metric reads ~50-90% (matches rav's 0.77-0.92 and layer_0 0.22-0.52), while the
  actual TOKEN-drop-fraction (dropped tail / total) is single-digit %. d3's 65% is most consistent with
  either that overflow-rate semantics OR a global-numerator/per-shard-denominator (64x) scope mismatch.
- VERDICT PENDING MEASUREMENT: treat the fixed path as likely fidelity-OK (~1-3% true drops) but MUST
  confirm with my fixed metric. My in-flight treatment (adj-custom-120-0724-2216) has the OLD broken
  logging -> will not produce the number. Need a run with commit 2d4a87395.

## 1a MATCHED A/B LOCKED (speed half) — 05:22 UTC
Operating point (d5120/256e-top8/48L/EP64/b1024), 120 steps, back-to-back, DISABLE_CHECKPOINT:
| leg | p10 | p50 | p90 | loss@119 |
|---|---|---|---|---|
| control  gather-dispatch + autodiff backward   | 20.51 | 20.61 | 20.69 | 5.738 |
| treatment gather-dispatch + CUSTOM ADJOINT      | 23.73 | 24.04 | 24.75 | 5.711 |
=> custom adjoint = +3.43pp p50 (+16.6% rel), p10/p90 bands FULLY separated. Clean, decisive win from the
adjoint alone (before leg-batching/rotation). Loss parity: 5.711 vs 5.738 (0.027 apart = known independent-run
RNG divergence; kernel test proves 1e-5 grad parity). Speed half of 1a is LOCKED.

DROP JOB submitted: /mwittmann/ep25d1-drops-30-0724-2318 (custom adjoint + SCALE_REPORT_DROPS, 30 steps,
commit 2d4a87395). d4's fp8-wire leg already exercised the metric: 0.755 early -> 0.172 late; hypothesizes
64x expert-shard overcount => /64 gives 1.18% -> 0.27%, matching my ~0.9% multinomial estimate. Auditing
the psum scope now.

## PSUM AUDIT RESULT 05:25 UTC — metric is CORRECT (disproves 64x hypothesis)
Controlled CPU test (_fixed_a2a_core under a REAL 2-expert-shard shard_map, cf=0.5 to force drops):
  metric dropped_total = 87, numpy reference (true global drops, per-shard capacity) = 87, ratio = 1.000.
=> The psum over (replica_dcn,data,expert) sums DISTINCT per-shard drop counts to the correct GLOBAL total;
   out_spec P() replicates that single global scalar. NO expert-axis double-count. denominator (global
   B*S*topk*num_layers) matches numerator scope. d4's 64x-overcount hypothesis is DISPROVEN for this code.
Implication: the true token-drop-fraction is whatever MY drop job reads (commit 2d4a87395), taken at face
value. Expectation still ~0.9% at the operating point (cf=1.0, capacity==mean per (shard,expert) bucket of
2048, near-uniform init routing). d4's 0.755 must be fp8-wire-config-specific (different bucket size/capacity)
or a different metric build, NOT a 64x bug in this metric. Awaiting drop-job numbers to state the final value.

## 1a FULL PACKAGE 05:40 UTC — A/B + drop verdict
### Matched 120-step A/B (operating point, back-to-back)
| leg | p10 | p50 | p90 | loss@119 |
| control  (autodiff backward)  | 20.51 | 20.61 | 20.69 | 5.738 |
| treatment (CUSTOM ADJOINT)    | 23.73 | 24.04 | 24.75 | 5.711 |
custom adjoint = +3.43pp p50 (+16.6% rel), bands fully separated. Loss parity within 0.027 (RNG). LOCKED.

### Drop measurement (/mwittmann/ep25d1-drops-30-0724-2318, custom adjoint + SCALE_REPORT_DROPS, commit 2d4a87395)
Per-step moe/drop_fraction (window steps 23-29): 0.893, 0.881, 0.870, 0.862, 0.855, 0.850, 0.846 (monotone down).
dropped_assignments @ step23 = 1,438,043,460. Integer cross-check: global assignments = B*S*topk*L =
1024*4096*8*48 = 1,610,612,736; 1.438043460e9 / 1.610612736e9 = 0.8929 == logged drop_fraction. Self-consistent.

### Metric validation (THE audit)
Controlled CPU tests of _fixed_a2a_core under REAL shard_map, metric dropped_total vs numpy global-drop reference:
  nshard=2 -> metric 96 = ref 96 (ratio 1.000);  nshard=4 -> metric 192 = ref 192 (ratio 1.000).
=> metric is EXACT and does NOT scale/inflate with shard count. d4's 64x-overcount hypothesis DISPROVEN.
Real untrained-router toy (expert_axis=2, cf=1.0): ne=64 -> 7.2%, ne=256 -> 14-16% (small 64-128 buckets).

### VERDICT
The metric is mechanically correct, so ~85-89% in this window is the TRUE routed-token-drop-fraction at the
barebones fixed path (qb_routing OFF => NO load-balancing loss). d4's independent bf16 AND fp8 runs show the
same trajectory oscillating 0.17-0.79 over a full run. This is >> Larry's 3% bar => the fixed path's routed
experts drop the MAJORITY of assignments during early training (classic router collapse without load balancing).
CAVEATS: (1) the always-on SHARED expert processes every token, so loss still descends normally (5.84->5.71);
(2) drops decrease with training (d4 late ~0.17) - steady-state unknown from 30 steps; (3) my small-scale toy
maxed at ~16% with untrained random-token routing, so the jump to ~89% is real-scale routing collapse, best
CONFIRMED by a QB-load-balancing run (expected to crush drops) - that's d4's cf/QB sweep territory.
=> FIDELITY NOT CLEARED for the barebones fixed path as-configured. Mitigation required (QB load-balancing /
cf sweep / ragged). Goal-relevant negative for fixed transport fidelity; speed half (24.04% lock) stands.

## QB-on drop A/B submitted 00:28 UTC
QB load-balancing knob = SCALE_MOE_QB=1 (model.py:150 qb_routing; NOT SCALE_MOE_SKIP_QB which would disable the
router_bias update). Job /mwittmann/ep25d1-qbon-drops-30-0725-0027: custom adjoint + SCALE_REPORT_DROPS +
SCALE_MOE_QB=1, cf1.0, 30 steps, operating point, DISABLE_CHECKPOINT. Comparator: QB-OFF series 0.893->0.846.
Division of labor: I own THIS quick 30-step confirmation; d4 owns the 120-step production QB-on legs (cf1.0 then
cf1.15) - not duplicating. Also to report: step-time QB-on vs QB-off (fixed path GEMMs run on capacity-sized
buffers regardless of drops, so headline MFU should be drop-insensitive; only the router aux-loss adds cost).

## QB-on drop A/B result 00:40 UTC — fidelity section closer
Matched steps 23-29, drop_fraction (custom adjoint + SCALE_REPORT_DROPS, cf1.0, operating point):
| step | QB-OFF | QB-ON |
|  23  | 0.893  | 0.492 |
|  24  | 0.881  | 0.442 |
|  25  | 0.870  | 0.397 |
|  26  | 0.862  | 0.362 |
|  27  | 0.855  | 0.327 |
|  28  | 0.850  | 0.289 |
|  29  | 0.846  | 0.249 |
=> QB load-balancing cuts drops ~3.4x AND accelerates the decrease (slope -0.035/step vs -0.008/step QB-off).
Clearly the correct lever, but 30 steps only reaches 0.249 (25%) - still above Larry's 3% bar; whether QB-on
crosses <3% at steady state needs the longer trajectory (d4's 120-step production QB-on legs).
STEP-TIME: QB-on p50 24.11% vs QB-off drop-job p50 24.52% = -0.41pp (within +-2-4pp placement noise) => headline
MFU is DROP-INSENSITIVE (fixed-path GEMMs run on capacity-sized buffers regardless of drops); router aux-loss
cost is negligible. So QB-on buys a >3x drop reduction essentially for free on MFU.

FIDELITY SECTION SYNTHESIS (mine + peers):
- QB is the drops lever, NOT capacity factor: d4 cf1.15 QB-OFF = 22.13% p50 (-1.91pp for +15% cap) with drops
  STILL 0.86 peak / 0.65 late => cf-without-QB is pure MFU cost, zero fidelity gain.
- My matched A/B: QB-off 0.89->0.85; QB-on 0.49->0.25 (steeply down) at ~free MFU cost.
- d4 full-run oscillation 0.17-0.79 (QB-off).
- Metric proven mechanically exact (ratio 1.0 at 2 and 4 shards) - the numbers are real.
=> The fixed path needs QB-on for fidelity; QB-on is nearly free on speed and drives drops down >3x fast, but
the <3% crossing is unconfirmed in 30 steps and rests on d4's longer QB-on legs. d2's ragged arm (imminent) is
the cross-transport fidelity comparator.

## Milestone draft 00:52 UTC
Wrote EP25_MILESTONE_DRAFT.md (worktree root, internal - NOT pushed/posted). Structure: TL;DR, speed A/B
(+3.43pp -> 24.04, composes with leg-batching 25.39), fidelity (drops 0.85-0.89 QB-off -> 0.25 QB-on 3.4x at
~zero MFU cost; metric proven exact; <3% steady-state TODO=d4 120-step QB-on), sealed negatives (rotation
-9.46, prefetch null, token-chunk -1.96, fp8 wire -2.02, TE-at-tip crash), transport TODO=d2 ragged/ring,
goal ledger, repro pointers (branch/commits/env/jobs; PR from agent/ep25-d1-adjoint still needed - no pushes).
Style per .agents/skills/writing-style (reports.md + ai-writing-donts.md): numbers first, one table/section,
mixed results plain. Awaiting human review before any posting.

## Milestone draft final fills 01:10 UTC
Filled d4's frontier table (fidelity §2): QB-off cf1.0 24.04 (collapsed) / QB-off cf1.15 22.13 (drops 0.649@119)
/ QB-on cf1.0 22.60 (drops 0.083@119, loss 5.767) / QB-on cf1.15 20.85 (drops 0.037@119, loss 5.788). Prices:
QB -1.44pp, cf1.15-under-QB -1.75pp, combined -3.19pp. Transport table (§4): ragged 12.38 mean/drops 0.433,
ring_cute EP64 DNF (OOM 141.79 GiB in jit_train_step); direction-2 = transport leaves no pp. Goal ledger (§5):
honest production config (QB-on cf1.0) = 22.60%; >=25% honest needs leg-batching (~+1.3pp) + fa4-lse; QB-off
24.04/25.39 are bench artifacts. Added rav emission-bug note (drops computed, never logged; fix 2d4a87395).
TL;DR updated to honest framing. ai-writing-donts pass done. TWO TODO slots remain: d4 300-step steady-state
drop series (in flight), d3 fa4-lse A/B. Draft committed locally, NOT pushed/posted.

## Milestone draft: fidelity resolved 01:30 UTC
Filled d4's 350-step steady-state (/mwittmann/ep25d4-qb-cf100-drops-350-v1): p50 22.00% (~22.6 within draw
variance), loss to 3.335, drops 0.885(5)->0.271(60)->0.175(119)->0.089(250)->0.064(349), tail-100 mean 7.3%,
halving time growing past ~150 steps. VERDICT: cf1.0 QB-on levels ~6%, does NOT cross 3% - the 120-step <3%
extrapolation is FALSIFIED. Step-119 drop is draw-variable (0.175 vs 0.083). Strict-3% compliant config =
QB+cf1.15 @ 20.85%. Ledger: throughput frontier 24.04 vs strict-fidelity 20.85 = ~3.2pp gap; closing it
(faster/tuned QB balancing) flagged as the top open fidelity direction alongside leg-batching + fa4-lse. TL;DR,
§2, §5 updated; coherence fix on the 30-step cross-reference; final scrub done. ONE TODO remains: d3 fa4-lse.
Draft committed locally, NOT pushed/posted.

## Milestone draft: QB-gain probe + MFU caveat 01:50 UTC
Added (1) measurement-methodology caveat near top: heavy-drop runs read HIGHER MFU (dropped assignments gather
the zero pad row = less real work at same step accounting); g=2 posted 23.4% while dropping 68%; MFU comparable
only within a matched drop regime. Protected the §1 adjoint A/B (matched: both QB-off, identical drops -> +3.43pp
valid) and flagged QB-off-vs-QB-on -1.44pp as a cross-regime UPPER BOUND. (2) QB-gain probe in §2: grug QB =
implicit proportional controller (1x residual/step, not DeepSeek integral); g=2 (SCALE_QB_GAIN, 58c9a19eb)
diverges (drops 0.67-0.72 x350, loss +0.091) => residual ~6% is NOT global-bias under-correction; leading
hypothesis = sender-local bucket hotspots (64x256, invisible to global bias; explains cf1.15 halving). Follow-ups
listed: damped g<1, DeepSeek integral, per-sender bias (kernel-level, only one aimed at the cause). §5 updated:
gap-closing targets sender-local balancing (global gain tuning falsified). §6 repro: added d4 fidelity + gain
jobs. Scrubbed (removed emphasis asterisks). ONLY fa4-lse TODO remains; whole-doc final pass on its landing.

## Milestone draft FINAL 02:10 UTC — fa4-lse filled, whole-doc pass done
fa4-lse (d3, agent/ep25-d3-fa4lse) into §3: control 20.465% (3 draws) vs fa4-lse+host-offload 20.648% (2 draws)
= +0.18pp, below 0.5pp bar; on-device DEAD (+32.7 GiB won't fit), host-offload over Grace C2C saves ~70ms/13.2s
step; d2560 ~1pp estimate doesn't transfer to d5120 EP64; behind-flag ~0.2pp if robust at d6144, NOT in the 25%
bridge. §5 goal ledger rewritten to FINAL: goal NOT met at honest fidelity; best honest 22.60% (QB-on cf1.0, ~6%
drops), strict-3% 20.85% (QB+cf1.15), matched-regime QB-off frontier 24.04/+batching 25.39; ranked follow-ups:
sender-local balancing (kernel), leg-batching+QB composition, DeepSeek-integral/damped QB, MXFP8 (speed/quality),
fused fp8 epilogues. Whole-doc ai-writing-donts pass: reconciled the "drop-insensitive"/"−1.44pp not drops"
claims with the measurement caveat (now: MFU not comparable across drop regimes, −1.44pp = upper bound). Number
verification: ALL my own measured numbers confirmed against this log; peer numbers (frontier/350-step/gain/
ragged/ring/fa4-lse/negatives/leg-batching) are coordinator-relayed and not independently verifiable by me.
Draft committed locally, NOT pushed/posted. Ready for human.

## R6-1 check-in 18:30 UTC — leg-batching reconstructed
- Bundle extraction (route 1) FAILED: controller get-job-state only returns the state enum, get-task-status
  no bundle_id; bundles are content-addressed (thousands) with no name->hash map exposed. Pivoted to route 2.
- RECONSTRUCTED leg-batching (commit 65e3ca50d, SCALE_A2A_BATCH_EXPERTS=1): one dispatch a2a + one grouped
  up/down einsum + one combine a2a over all local experts, replacing the 4-collective/4-GEMM Python loop.
  Local-expert axis is a batch dim the a2a passes through (split/concat on the expert-shard axis).
- PARITY: bit-exact vs the loop at expert_axis=2 (out/grad max abs diff 0.0, drops identical=49>0);
  single-device kernel test passes at 1e-5. Composes with the custom adjoint (wraps send_x/send_output).
- Confidence: 6/10 that batching's +1.35pp transfers to QB-on. It's a real launch-overhead/GEMM-efficiency
  win independent of routing, so it SHOULD transfer, but QB-on changes the drop regime and MFU is only
  comparable within a matched regime; the rack A/B settles it.
- Next: EP4 smoke (batching + QB-on) to confirm real multi-GPU a2a, then rack A/B control vs +batching.

## R6-1 check-in 14:54 local — smoke green, rack A/B control submitted
- EP4 batched smoke /mwittmann/ep25d1-batch-smoke-ep4-0725-1350 SUCCEEDED (clean 40-step run on GB200 -> the
  batched a2a/GEMM path works on real multi-GPU; CPU parity already bit-exact so numerics confirmed).
- RACK A/B CONTROL submitted: /mwittmann/ep25d1-qbon-adj-control-120-0725-1454 = QB-on cf1.0 + adjoint,
  NO batching, 120 steps, SCALE_REPORT_DROPS, DISABLE_CHECKPOINT, operating point. Comparator band = d4's
  QB-on cf1.0 draws 22.595 / 22.002. Will submit +batching treatment back-to-back after control terminates
  (one rack in flight). Both legs report the drop series for matched-regime comparison.
- Confidence: 6/10 (unchanged) on batching's +1.35pp transferring to QB-on.

## R6-1 INFRA HOLD 16:00Z — control leg data LOST, treatment held
- Log-shipper sidecar amd64-only since #7583 rollout (16:07Z): no metrics ship for GB200 jobs.
- My control /mwittmann/ep25d1-qbon-adj-control-120-0725-1454 = JOB_STATE_SUCCEEDED (completed, NOT still
  running). Data LOST: iris logs returns 0 metric rows; salvage agent has NO dir for it (find for
  qbon-adj-control / 1454 empty), though it IS actively capturing peers (ep25d2-mxfp8, ep25d3-qbint01).
  My control's pods completed+deleted without capture. => goes on the RERUN list.
- HOLDING per standing orders: NOT submitting the +batching treatment leg until the multi-arch fix ships,
  controller restarts, and the canary flips (fresh GB200 pod 2/2, new grug-train rows in finelog).
- Treatment STAGED (exact cmd below): rerun BOTH legs back-to-back for same-draw comparison —
  control = QB-on cf1.0 adjoint (SCALE_MOE_QB=1 SCALE_REPORT_DROPS=1 SCALE_A2A_GATHER_DISPATCH=1
  SCALE_A2A_CUSTOM_ADJOINT=1), treatment = + SCALE_A2A_BATCH_EXPERTS=1. 120 steps, cf1.0, DISABLE_CHECKPOINT.
- Code is committed and parity-proven (65e3ca50d bit-exact); no code work blocked, only the rack measurement.

## R6-1 rerun 18:23 local — hold lifted, control resubmitted
Canary flipped (multi-arch log-shipper fixed, controller restarted). Resubmitted control
/mwittmann/ep25d1-qbon-adj-control-120-0725-1823 = QB-on cf1.0 adjoint + drops, 120 steps. Poller bs63gv65n
does an EARLY log-flow probe once running (>=3 metric lines around step ~5) so we catch any loss before harvest,
then harvests at terminal. Treatment (+SCALE_A2A_BATCH_EXPERTS=1) goes back-to-back after control terminates.

## R6-1 control harvested + treatment submitted 20:16
CONTROL /mwittmann/ep25d1-qbon-adj-control-120-0725-1823: p50 22.66% (p10 22.34 / p90 24.23, 119 samples),
drops 113-119 = 0.094/0.087/0.092/0.080/0.080/0.088, loss 5.614. In d4's QB-on cf1.0 band (22.595/22.002). VALID.
TREATMENT submitted: /mwittmann/ep25d1-qbon-batch-120-0725-2016 = control + SCALE_A2A_BATCH_EXPERTS=1. Polling inline to terminal.

## R6-1 treatment v1 FAILED (transient), resubmitted v2 20:26
/mwittmann/ep25d1-qbon-batch-120-0725-2016 FAILED during init: "ABORTED: 7 unexpectedly tried to connect with a
different incarnation. It has likely restarted" = JAX distributed gang-abort from one worker (task 7) restarting
at startup. NOT a batched-code bug: my sentinel loaded on task 13, no OOM/ResourceExhausted in logs, no step ran,
and rav's batched run succeeded at EP64. Transient node/preemption flake. Resubmitted as /mwittmann/ep25d1-qbon-batch-120-0725-2026-v2.
Control /mwittmann/ep25d1-qbon-adj-control-120-0725-1823 = p50 22.66% stands as the matched control.

## R6-1 treatment WEDGED 21:13 local — batched path not completing at EP64
- CONTROL /mwittmann/ep25d1-qbon-adj-control-120-0725-1823 = p50 22.66% (p10 22.34/p90 24.23), drops ~0.088@119,
  loss 5.614. SOLID matched control.
- TREATMENT (+SCALE_A2A_BATCH_EXPERTS=1) has NOT produced a single step in two attempts:
  v1 (2016): gang-abort at init (incarnation mismatch), ~7 min.
  v2 (2026): compiled ~23 min (03:28->03:51), then "ABORTED: another task died / incarnation mismatch" fatal at
  03:51:40; job now JOB_STATE_RUNNING but logs FROZEN at 03:51:53 for 21 min = WEDGED (post-fatal clique hang,
  matches the known GB200 fast-restart deadlock).
- No explicit OOM/ResourceExhausted string, but incarnation-mismatch = a worker process died (OS OOM-kill or
  crash shows this way). Two batched failures vs zero non-batched (control succeeded on same cluster) points at
  the batched code. Two hypotheses: (a) memory regression — my full a2a+GEMM batching raises the peak (one 5.4GB
  send_x + received + grouped-GEMM intermediates simultaneously) past the 0.75 HBM wall at d5120 EP64; (b) the
  batched einsum/split_axis=1 a2a triggers a ~23-min pathological XLA compile that widens the transient-flake
  window. rav's batched 25.39% run predates the infra incident, so his success doesn't disprove either.
- Kernel parity is bit-exact (65e3ca50d) so the FORWARD/BACKWARD math is right; the problem is runtime/memory at
  scale, not correctness.
- NOT killing the wedged rack job without coordinator approval (rule). Asking for the call + whether to try a
  memory-lighter variant (batch GEMM only, keep per-expert a2a) or defer.

## R6-1 grouped-batching prepared (while awaiting kill/direction decision) 21:20 local
Committed grouped variant: SCALE_A2A_BATCH_GROUP=G processes local experts in groups of G (default full).
G=local_experts = prior full-batch (bit-exact, tests pass); G=2 halves the extra memory peak AND shrinks each
compiled graph (mitigating BOTH failure hypotheses - OOM peak and 23-min pathological compile). Bit-exact vs
the loop at G=full and G=2 (CPU, expert_axis=2, diff 0.0, drops identical). Ready to fire a grouped rack A/B
(SCALE_A2A_BATCH_EXPERTS=1 + SCALE_A2A_BATCH_GROUP=2) the instant the coordinator approves killing the wedged
v2 and pursuing option (a). Control 22.66% stands regardless.

## R6-1 RESULT 00:16 — leg-batching x QB-on: NEGATIVE (matched pair complete)
| leg | p10 | p50 | p90 | drops@119 | loss@119 | samples |
|---|---|---|---|---|---|---|
| control: QB-on cf1.0 + adjoint            | 22.34 | 22.66 | 24.23 | 0.088 | 5.614 | 119 |
| treatment: + batching G=2 (SCALE_A2A_BATCH_GROUP=2) | 18.72 | 19.00 | 20.27 | 0.092 | 5.643 | 119 |
=> batching G=2 = -3.66pp p50 vs matched control. Bands do NOT overlap (control p10 22.34 > treatment p90 20.27).
FIDELITY PARITY CONFIRMED: drops 0.092 vs 0.088 (same regime, so MFU IS comparable here), loss 5.643 vs 5.614
(0.029 = RNG-scale). So this is a pure THROUGHPUT regression, not a numerics or drop artifact.
VERDICT: rav's +1.35pp batching win does NOT transfer to QB-on in my implementation. Note this is MY grouped
reconstruction (G=2, forced by two full-batch gang-aborts), not rav's exact uncommitted patch: full batching
(G=4) never produced a step so it is UNMEASURED. Two readings: (a) the batching mechanism genuinely doesn't help
at this shape once QB is on, or (b) my grouped variant's extra concatenate + 2x-per-group a2a structure costs
more than the launch-overhead it saves, and only rav's exact full-batch patch would show the win. Cannot separate
without rav's patch running at G=4.
ROUND-6 NUMBER: 22.66% (control) stands as my honest QB-on cf1.0 datapoint; it reproduces d4's 22.595/22.002 band.
Confidence that leg-batching contributes >=1pp toward honest 25%: 2/10 (was 6/10) - measured negative at G=2,
unmeasured at G=4.

## R6-NEW: SAME-STEP SPILL implemented 00:25 — design + fidelity argument
MECHANISM (commit 1224ccb02, SCALE_A2A_SPILL=m, 0=off byte-identical): after the baseline placement, each
still-unplaced assignment is re-offered to the next-ranked expert THE SAME TOKEN SELECTED, taking only that
bucket's remaining headroom. m bounded attempts (each = one extra segment-rank/argsort over T*K).
Bucket layout + capacity UNCHANGED (Larry's granularity constraint preserved).

WHY THE ADJOINT COMPOSES (the delicate part, and it fell out clean): spill is expressed PURELY as a rewrite of
(linear_indices, keep). Both custom VJPs are generic in those: _dispatch_gather_bwd is a segment-sum over the
token's topk send slots indexed by linear_indices+keep; _combine_gather_bwd is a gather along assignment_sources
(the inverse of linear_indices). Neither hard-codes WHICH expert a slot belongs to. So spill required ZERO
adjoint changes. VERIFIED: custom-adjoint vs autodiff at 1e-5 with spill m=1,2 on a real 2-shard shard_map
(grad diff 2.3e-10..4.7e-10, drops identical 61/40/20). Also survives scan+recompute_all by construction (pure
index math, no new residuals).

WHY SPILL IS A FIDELITY IMPROVEMENT OVER DROPPING (the semantics DO change, so arguing it explicitly):
A dropped assignment (t,k) contributes EXACTLY ZERO to token t's output — the router asked for expert e_k with
weight w_k and got nothing, so the token's effective combine weights silently don't sum to their intended total
and the token is under-computed. A spilled assignment instead computes w_k * E_{e_j}(x_t) where e_j is ANOTHER
EXPERT THE ROUTER ITSELF SELECTED for that token (rank k+1..k+m of its own top-k). So: (i) the token gets real
expert signal instead of nothing; (ii) the expert used is one the router endorsed, not an arbitrary substitute;
(iii) the combine weight is the token's own w_k, unchanged; (iv) total expert FLOPs and the capacity invariant
are identical — we fill idle capacity that would otherwise compute padding. The approximation is substituting a
lower-ranked selected expert for a higher-ranked full one, which is strictly closer to the true MoE output than
substituting zero. This is why LOSS PARITY IS THE VERDICT: if the argument holds, loss should be equal or BETTER.

CPU MODEL AT THE OPERATING-POINT SHAPE (ne=256, topk=8, cf=1.0, capacity==mean):
  uniform     drops 0.0758 -> m1 0.0482 -> m2 0.0366 -> m3 0.0304
  mild burst  drops 0.1062 -> m1 0.0688 -> m2 0.0536 -> m3 0.0441
  strong burst drops 0.2321 -> m1 0.1816 -> m2 0.1498 -> m3 0.1284
The uniform m0 (7.58%) MATCHES the observed real steady 6-8%, which supports the burstiness diagnosis and
predicts m=2 lands ~3.5-5% and m=3 ~3-4.5% live. PRE-REGISTERED: m=2 may not clear 3% alone; m=3 is the follow-up.
Invariants tested: drops strictly fall with m, no bucket exceeds capacity, slots unique, every placed assignment
targets an expert the token actually selected. 20/20 kernel tests pass (spill-off path unchanged).
EP4 smoke m=2 submitted: /mwittmann/ep25d1-spill-smoke-ep4-0726-0023. Then 350-step QB-on cf1.0 rack leg.
Confidence: 6/10 that spill takes cf1.0 under 3% at <=0.5pp MFU cost (mechanism is sound and directly targets
the diagnosed cause; risk is the extra argsorts' cost and whether live burstiness is worse than my model).

## CORRECTION 01:00 — my 7.58% CPU number does NOT support the "statistical floor" reframing
I checked the coordinator's proposed reframing before writing it up, and it does not survive. Correcting it here
because it was headed for the milestone report.
THE ERROR IS MINE: my CPU model ran ne=256/topk=8 at T=1024, giving per-bucket mean load 32. The REAL operating
point has per-(sender,expert) bucket mean 2048 (4.19M tokens / 64 batch shards = 65,536 tokens/shard x topk 8 =
524,288 assignments / 256 experts = 2048 = capacity at cf1.0). My 7.58% matching the observed 6-8% was a
COINCIDENCE OF SCALE, not evidence.
SIMULATED uniform-random routing, capacity==mean, drop fraction vs per-bucket mean (matches 0.3989/sqrt(mu),
the normal-approx E[(X-mu)+]/mu):
  mean 32   -> 0.0754 (formula 0.0705)   <- my toy model lived here
  mean 128  -> 0.0335 (0.0353)
  mean 512  -> 0.0176 (0.0176)
  mean 2048 -> 0.0091 (0.0088)           <- THE REAL OPERATING POINT
=> The statistical floor of the no-spill policy at real scale is ~0.9%, NOT 6-8%. The coordinator's ORIGINAL
~0.9% per-bucket normal approximation was CORRECT and should NOT be superseded; my number does not supersede it.
=> Therefore the observed 6-8% residual is ~8x the statistical floor, so it IS dominated by non-uniform (bursty,
document-correlated) routing — d4's burstiness diagnosis STANDS rather than being replaced.
QUANTIFYING THE BURSTINESS: inverting drop ~= 0.3989*sigma/mu at mu=2048 gives implied effective sigma 329-411
vs Poisson 45.3, i.e. routing is 7-9x more clustered than independent-uniform (53-82x the variance). That is a
concrete measure of the within-batch burstiness and is the thing spill must reclaim.
WHY THIS IS GOOD NEWS FOR SPILL: in my toy (mean 32) nearly the whole 7.58% WAS irreducible floor, so spill was
fighting the floor. At real scale ~88% of the residual is non-uniform excess sitting next to genuinely underfull
buckets, which is exactly what spill reclaims. Note the 0.9% is only a floor for the NO-SPILL policy: spill can
go below it, since it moves overflow into underfull buckets whenever the token selected one.
This does NOT retro-explain d3/d4's null results as "nothing to correct" — the controllers plateau at ~8x the
floor, so there IS non-uniformity present; it is just invisible to any one-step-delayed controller because it
decorrelates step to step (the original diagnosis).
