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

## Calibrated prior at REAL scale 01:10 (replaces the discarded toy predictions)
Rebuilt the CPU model at the true per-shard scale: T=65,536 tokens, ne=256, topk=8, capacity=2048, with
document-block clustered routing (16 blocks x 4096 = seq_len structure) instead of uniform draws.
  burst=0.00 (uniform) -> m0 drop 0.0096   <- independently reproduces the 0.9% analytic floor
  burst=0.15           -> m0 drop 0.0329
  burst=0.30           -> m0 drop 0.0692   <- CALIBRATED: matches the observed 6-8% band
  burst=0.50           -> m0 drop 0.1187
At the calibrated burst=0.30, spill sweep AT REAL SCALE:
  m=0 0.0692 | m=1 0.0420 (-39%) | m=2 0.0304 (-56%) | m=3 0.0237 (-66%)
READ: m=2 lands right AT the 3% bar (borderline, could fall either side live); m=3 clears it (~2.4%).
This raises the prior that the pre-approved m=3 follow-up will be required. Treating these as a calibrated
PRIOR, not a prediction - the live legs decide. The toy (mu=32) m=2/m=3 numbers are discarded entirely.
Independent corroboration: a plain document-block structure at seq_len granularity reproduces the observed
6-8% naturally at burst=0.30, supporting the document-correlation account of the burstiness.

## SPILL m=2 RESULT 01:49 — drops HALVED at ~zero MFU cost, loss BETTER
/mwittmann/ep25d1-spill2-cf100-350-0726-0028 SUCCEEDED (350 steps, QB-on cf1.0):
| metric | baseline (d4 global-QB) | spill m=2 | delta |
| p50 MFU        | 22.002 | 21.872 | -0.13pp |
| drops @349     | 0.064  | 0.0369 | -42%    |
| drops tail-100 | 0.073  | 0.0373 | -49%    |
| loss @349      | 3.335  | 3.3225 | -0.0125 (BETTER) |
READS:
- MFU cost is -0.13pp for 2 attempts = ~0.065pp per attempt. Far inside the <=0.5pp budget. Spill does real
  work dropping skipped (it fills buckets that would have computed padding), so a small cost is legitimate and
  expected - this is cheaper than expected because the expert GEMMs run on capacity-sized buffers either way,
  so the only new cost is the extra segment-rank/argsort per attempt.
- FIDELITY ARGUMENT CONFIRMED: loss is BETTER (3.3225 vs 3.335), not merely at parity. Predicted: a spilled
  token gets w_k * E_{e_j}(x) from an expert the router itself selected instead of contributing zero. The
  measurement matches the prediction, so the semantic change is an improvement, not a regression.
- BAR: 3.73% tail-100 mean does NOT clear the 3% bar. Calibrated prior said m=2 ~3.04% (borderline) - it landed
  slightly worse but in the predicted neighbourhood, and the prior's m=3 ~2.37% now looks like the clearing shot.
FIRED the pre-approved m=3 leg: /mwittmann/ep25d1-spill3-cf100-350-0726-0149.
PROJECTED COMPLIANT FRONTIER (to confirm with m=3): if m=3 clears 3% near ~21.8%, that beats the current only
3%-compliant config (QB+cf1.15 at 20.85%) by ~+1.0pp of compliant MFU. If m=3 lands ~2.4% it may also permit
cf BELOW 1.0 or compose with cf1.15 for deeper compliance.
Confidence: 8/10 that spill is the right mechanism (halved drops, free, loss better); 6/10 that m=3 alone clears 3%.

## Conservation bound (closes "lower cf and spill it back" permanently) 02:00
Total capacity = num_shards * num_experts * ceil(cf * assignments_per_shard / num_experts) ~= cf * total
assignments. So for cf < 1, drop_fraction >= 1 - cf REGARDLESS of routing, balancing, or spill:
  cf 0.85 -> >=15% drops | 0.90 -> >=10% | 0.95 -> >=5% | 0.97 -> >=3% | 1.00 -> >=0%
Under a 3% bar only cf >= ~0.97 is even feasible, which buys no meaningful speed. Spill can only reclaim drops
caused by MISPLACEMENT (overflow sitting next to underfull buckets); it can never manufacture capacity that the
global budget does not contain. NOT running the spill x cf<1.0 composition - it is dead by conservation.
=> COROLLARY (the headline framing): cf1.0 is the fastest feasible operating point for the fixed-capacity policy
under any drop bar. Spill's contribution is precisely that it makes cf1.0 COMPLIANT, saving the -1.75pp that
cf1.15 costs. Net ~+1.0pp of compliant MFU after spill's own -0.13pp, not the naive +1.75pp.
=> WHAT SPILL IS: a mechanism that recovers the ~8x burstiness excess ABOVE the statistical floor. Bounded below
by the floor (0.9% here) and by conservation (1-cf). Not a way to buy capacity.

## Note for the d6144 4-of-256 peer shape
  d5120 8-of-256 (here): bucket mean 2048, no-spill floor ~0.88%, max spill attempts = topk-1 = 7
  d6144 4-of-256 (peer): bucket mean 1024, no-spill floor ~1.25%, max spill attempts = topk-1 = 3
Double caution for that shape: the statistical floor is HIGHER (1.25%) while spill has LESS headroom (m capped
at 3, and each attempt chooses among fewer alternatives). Spill may be needed there just to reach where we are
here, and m=3 is the ceiling rather than a tuning choice.

## FIRST-CLASS FINDING: top-k is the budget for drop recovery (architecture input for the hero run)
Spill's maximum strength is structurally capped at m_max = topk - 1. Lowering top-k degrades fidelity headroom
FOUR ways at once, all compounding:
| | 8-of-256 (proxy, measured here) | 4-of-256 (hero candidate) |
| per-(sender,expert) bucket mean | 2048 | 1024 |
| statistical floor (no-spill)    | 0.88% | 1.25%  (halved mean -> floor x sqrt2) |
| m_max = topk-1                  | 7 | 3 |
| alternatives per spilled token  | 7 | 3 |
| share of a token's routed signal lost per dropped assignment | ~12.5% | ~25% |
So at 4-of-256: the floor is HIGHER, the mechanism's ceiling is LOWER, each attempt chooses among FEWER
candidates, and each residual drop costs the token TWICE as much of its routed signal. A peer model (validated
out-of-sample against my live m=2 to within 0.3pp) puts 4-of-256 at ~2.88% at m=3 - just inside the 3% bar with
the mechanism already AT ITS CEILING and zero headroom left; the proxy 8-of-256 models ~2.7% at m=3 with four
more attempts still in reserve.
CONSEQUENCE FOR THE HERO-RUN DECISION: at 4-of-256, m=3 is the CEILING, not a tuning choice. If a top-4
architecture is chosen and the drop bar later tightens below 3%, the ONLY remaining lever is capacity factor, at
the measured -1.75pp per +0.15 cf. A top-8 architecture still has spill attempts in reserve. This is an
architecture consideration absent from the tracker's current candidate comparison (total params / active params
/ TPS). Holds regardless of how my own m=3 leg lands.
NOTE the last row is analytical (weight-share arithmetic), not measured; the first four rows are
measured/derived.

## SPILL FINAL FRONTIER 03:12 — m=3 lands 3.29%, bar NOT cleared but 91% of excess removed
/mwittmann/ep25d1-spill3-cf100-350-0726-0149: 349 measured samples, then failed at TEARDOWN after step 349
(same completed-training-then-coordination-teardown pattern as the original baseline A/B; all 349 samples valid,
mfu_sample_count=349, no incarnation/OOM string in the window).
| m | p50 MFU | p10/p90 | steady drops | loss@349 |
| 0 (d4 baseline) | 22.002 | - | 0.064@349 | 3.335 |
| 2 | 21.872 | 21.74/22.71 | 0.0373 | 3.3225 |
| 3 | 21.849 | 21.73/22.81 | 0.0329 | 3.3196 |
(steady drops for m=2/3 are the mean over the last 7 logged steps, 343-349 - the iris log window is ~1000 lines
so a true tail-100 is not recoverable from logs; correcting my earlier "tail-100" label for m=2, which was in
fact this same 7-step window.)
MARGINAL DECOMPOSITION: m0->m2 MFU -0.130pp, drops -42%, loss -0.0125 | m2->m3 MFU -0.023pp, drops -12%,
loss -0.0029 | total m0->m3 MFU -0.153pp, drops -49%, loss -0.0154 (loss BETTER at every step).
Per-attempt MFU cost: ~0.065pp each for the first two, 0.023pp for the third - cost falls because later attempts
process fewer surviving assignments. Benefit also falls (-42% then -12%), so returns diminish faster than cost.
VERDICT: spill does NOT clear the 3% bar at cf1.0 (3.29% vs 3.0%). But it removes 91% of the baseline's excess
over the bar (3.40pp of excess -> 0.29pp) at -0.153pp MFU with loss BETTER at every m. Extrapolating the -12%
per-attempt trend, m=4/5 would land ~2.9-3.0% - i.e. the bar is reachable but only just, and 8-of-256 has the
attempts in reserve to try (m_max=7) whereas 4-of-256 does not (m_max=3).
COMPLIANT-FRONTIER CAVEAT I CANNOT RESOLVE: QB+cf1.15's steady-state drop was never measured at 350 steps (only
0.037@119), so a strict compliant-vs-compliant comparison against spill's 21.85%/3.29% is not on the record. The
honest claim is: spill buys ~+1.0pp over cf1.15's 20.85% at comparable (not proven-better) fidelity. Closing that
needs one 350-step QB+cf1.15 leg.
Confidence: 9/10 spill is a real fidelity mechanism (drops halved, loss better at every m, ~free); 3/10 that
m<=3 alone clears a strict 3% bar at cf1.0.

## THREE CORRECTIONS 03:30

### 1. STRIKE my "no scale run could ever have varied capacity factor" inference (it was wrong)
I generalized from my branch to the fleet. Agent d4 added the SAME knob on ITS branch (commit 3e149490f;
`agent/ep25-d4-pipelined:experiments/grug/moe/model.py` reads capacity_factor from SCALE_CAPACITY_FACTOR). So
d4's cf1.15 = 20.848% and cf1.15-QB-off = 22.127% were produced with a WORKING knob and are of sound provenance.
The real situation was BRANCH DIVERGENCE: the knob exists on d4's branch, not mine, and my env var hit a branch
where the constant was still hard-wired. My diagnosis of my own symptom was right; the fleet-wide inference was not.
MERGE HAZARD: my fix (595958b83) and d4's (3e149490f) are now INDEPENDENT implementations of the same knob under
the same env name. Whoever merges these branches must RECONCILE, not double-apply. Mine is a GrugModelConfig
field + env_float in the launcher; d4's reads os.environ inline at the construction site.

### 2. TAIL-100 IS RECOVERABLE — and my 7-step numbers were biased LOW
`iris job logs --max-lines N` (default 1000 = the truncation that bit me) fetches the full history. Refetched
both legs at --max-lines 400000 and recovered 348 of 350 steps each. So we do NOT need the labelled-window
fallback: true tail-100 is available for every leg, and I will use it from here.
| leg | TRUE tail-100 (n=100, steps 250-349) | the 7-step window I reported | bias |
| m=2 | 0.0414 | 0.0373 | -0.0041 |
| m=3 | 0.0366 | 0.0329 | -0.0037 |
The window was biased LOW by ~0.004 (~10% relative) in both legs, because drops are still declining at 350 steps
so the last 7 steps are the lowest ones. The bias is systematic, not noise.

### 3. CORRECTED SPILL VERDICT (supersedes the numbers in my final frontier report)
Comparing tail-100 to tail-100 against d4's baseline tail-100 of 0.073:
| m | p50 MFU | TRUE tail-100 drops | vs baseline |
| 0 | 22.002 | 0.073 | - |
| 2 | 21.872 | 0.0414 | -43% |
| 3 | 21.849 | 0.0366 | -50% |
Excess over the 3% bar: baseline 4.3pp -> m=3 0.66pp = 85% of the excess removed (I previously said 91%).
m=3 steady state is 3.66%, NOT the 3.29% I reported - further from the bar than stated. Extrapolating -12% per
attempt from the corrected base: m=4 ~3.2%, m=5 ~2.9%. So m=5 could still just reach the bar, but it is tighter
than my earlier extrapolation implied and I would not bet above ~50% on it.
MATCHED-STEP series (m=2 vs m=3): 60: 0.2418/0.1835 | 119: 0.0900/0.0922 | 250: 0.0524/0.0489 |
300: 0.0371/0.0353 | 349: 0.0369/0.0326. Note m=2 and m=3 are indistinguishable at step 119 - spill's third
attempt only separates late, which is consistent with it acting on the residual burstiness rather than on the
early collapse.
UNCHANGED by this correction: the MFU costs (-0.130pp / -0.023pp), the loss improvements (better at every m),
the structural explanation of why spill is cheap, and the top-k-is-the-recovery-budget finding.

## LEG RANKING via d5's model, recalibrated against my TRUE tail-100 (not extrapolation) 04:00
Ran d5's approach through my own shipping `_assign_with_spill` kernel at real scale (ne=256, topk=8,
mu=2048, document-block routing) to get what d5 never published: m=4/m=5 and a capacity sweep.

### The model over-predicts spill's benefit — measured against my corrected numbers
| m | model reclaim | MY measured reclaim (vs B=0.073) | measured/model ABS drop ratio |
| 2 | 56.1% | 43.3% | 1.36x |
| 3 | 65.7% | 49.9% | 1.54x |
d5's "within 0.3pp" validation was against my 7-step-biased 3.7%; against my TRUE tail-100 of 4.14% the
model is 0.70pp optimistic at m=2. The error GROWS with m (+0.18 ratio per attempt), which makes sense:
an idealized model over-estimates how many free buckets later attempts find, because a token's alternative
experts are correlated with its first choice.

### Capacity sweep (burst 0.30, model absolutes) — spill x cf is strongly MULTIPLICATIVE
| cf | m=0 | m=2 | m=3 | m=5 |
| 1.00 | 0.0692 | 0.0304 | 0.0237 | 0.0169 |
| 1.05 | 0.0484 | 0.0127 | 0.0078 | 0.0031 |
| 1.10 | 0.0336 | 0.0045 | 0.0020 | 0.0004 |
| 1.15 | 0.0230 | 0.0013 | 0.0004 | 0.0000 |
The coordinator's COMBINATION hypothesis is CONFIRMED and is the strongest result here: a small capacity
bump plus modest spill beats either mechanism pushed to its limit.

### Ranking by expected compliant MFU (capacity -0.583pp per +0.05 cf; spill -0.130/-0.153/-0.178pp at m=2/3/5)
| config | MFU | predicted drops (K-corrected) | clears? |
| m=5 @ cf1.00 | 21.82 | 2.6% (const-K) / 3.2% (growing-K) | ~50/50 |
| m=2 @ cf1.05 | 21.29 | 1.73% | YES, 1.27pp margin |
| m=3 @ cf1.05 | 21.27 | 1.20% | YES, 1.80pp margin |
| m=3 @ cf1.10 | 20.68 | 0.31% | YES but MFU worse than cf1.15-beating candidates |
Checked the coordinator's prior: their m=5@cf1.0 ~21.83%/~50% and m=3@cf1.05 ~21.27% are both CONFIRMED
by my independent arithmetic. Expected compliant MFU: m=5@cf1.0 = 0.5x21.82 = 10.9; m=2@cf1.05 = 0.85x21.29
= 18.1. The combination wins on expected value; m=5@cf1.0 wins only if you are willing to take the coin flip
for +0.53pp.
CRITICAL: m=5's K is EXTRAPOLATED (never measured at m>3), while m=2/m=3's K is MEASURED at that same m. That
asymmetry, not the point estimates, is what decides the bet.

### The cf1.15 leg does double duty (why the ordering is right)
The cf AXIS has never been validated against any live run of mine - every combination number above rests on
the model's untested capacity response. The cf1.15 leg IS that validation.
PRE-REGISTERED: model m=0 @ cf1.15 = 0.0230, so I predict measured tail-100 ~2.3%. Near 2.3% => cf axis
sound and the combination ranking stands. Materially higher => the cf axis is optimistic too and every
combination candidate needs the same discount, which would push the pick toward higher cf.
RECOMMENDATION: run cf1.15 next as planned, then pick the combination leg with BOTH axes calibrated -
most likely m=2 @ cf1.05 (~21.29%, beats cf1.15's 20.85% by +0.44pp with a 1.3pp compliance margin).

## DE-CONFOUNDED FRONTIER 04:35 — same-draw m=0 lands; spill cost is HIGHER than I published
/mwittmann/ep25d1-qbon-cf115-350-0726-0313 SUCCEEDED (it ran cf1.0 because its bundle predates my knob fix;
confirmed by the ABSENCE of capacity_factor in its logged hparams). So it is a clean same-draw m=0 baseline.
| m | p50 MFU | TRUE tail-100 (n=100) | vs m=0 |
| 0 | 22.062 | 0.0710 | - |
| 2 | 21.872 | 0.0414 | -41.7% |
| 3 | 21.849 | 0.0366 | -48.4% |
CORRECTION TO MY MOST-QUOTED NUMBER: spill's MFU cost is -0.190pp (m=2) and -0.213pp (m=3) against a SAME-DRAW
baseline, not the -0.130/-0.153pp I published against d4's cross-draw 22.002. The true cost is ~0.06pp HIGHER,
i.e. spill is slightly more expensive than I reported. Still far inside the 0.5pp budget, and the qualitative
claim (drops halved for a fifth of a point) is unchanged.
WHAT THE DE-CONFOUNDING ALSO SHOWED: d4's baseline was sound. Their tail-100 0.073 vs my same-draw 0.0710
(0.002 apart) and their p50 22.002 vs my 22.062 (0.06pp apart). Draw variance here was much smaller than the
brief's +-2-4pp worst case - but it was still the same order as the effect I was reporting, which is exactly
why the cross-draw comparison was not safe.
RANKING IS ROBUST TO THE CORRECTION: baseline rose +0.060 and spill cost rose +0.060, so every candidate's
absolute MFU is within 0.01pp of my earlier ranking. m=5@cf1.0 ~21.82, m=2@cf1.05 ~21.29, m=3@cf1.05 ~21.27.
The pick does not change.
MODEL over-prediction ALSO unchanged: measured reclaim 41.7%/48.4% vs model 56.1%/65.7%; abs ratio 1.36x/1.54x.

## cf1.15 leg submitted 04:34 (record repair + first live validation of the model's capacity axis)
/mwittmann/ep25d1-cf115-m0-350-0726-0434 with SCALE_CAPACITY_FACTOR=1.15 through my new knob.
RUNTIME VERIFICATION METHOD (not trusting the env var): capacity_factor is now a GrugModelConfig field, so it
is serialized into the logged hparams. I will grep '"capacity_factor": 1.15' in the leg's hparams before
trusting any number from it. The m=0 leg above demonstrates the negative control - the field is ABSENT there
because that bundle predates the fix.
PRE-REGISTERED (restated): model m=0 @ cf1.15 = 0.0230 -> predict tail-100 ~2.3%.

## METHODOLOGICAL FINDING 05:20 — step-N drop readings are NOT comparable across runs of different length
My cf1.15 leg reads 0.1487 at step 119; d4's published cf1.15 reads 0.037 at step 119. A 4x gap that is not
draw variance. Cause: the LR schedule is defined over num_train_steps, so step 119 sits at a completely
different schedule position in a 120-step run than in a 350-step run.
  120-step run at step 119: 99% through the schedule, LR ~6% of peak (fully annealed)
  350-step run at step 119: 34% through the schedule, LR ~68% of peak  (measured in my log: 0.0265 at 119,
                            decaying to 0.0021 by 349)
A high LR churns the router, which drives drops up; an annealed LR lets the router settle, which drives them
down. So a step-119 drop from a 120-step run is an END-OF-ANNEAL number and a step-119 drop from a 350-step
run is a MID-SCHEDULE number, and the two must never be compared.
THIS RE-ATTRIBUTES AN EARLIER CLAIM. The record says "step-119 drop is draw-variable (0.175 here vs 0.083 in
the frontier leg)". That was not draw variance: 0.175 is the 350-step run mid-schedule and 0.083 is the
120-step frontier run at end-of-anneal, the same config at two different schedule positions. The apparent
"draw variance" was a systematic schedule effect, and attributing it to draws made the numbers look noisier
and less trustworthy than they actually are.
CONSEQUENCES:
1. The 120-step frontier table (QB-on cf1.0 0.083, QB-on cf1.15 0.037) reports END-OF-ANNEAL drops, not
   steady-state drops, and is not comparable to any 350-step tail-100 in this work.
2. My own three legs (m=0/2/3) are ALL 350 steps and all reported as tail-100, so the spill comparison is
   internally consistent and unaffected.
3. The "cf1.15 = 3.7%" figure now being used for the public record repair is an end-of-anneal 120-step number.
   My running 350-step leg is the first measurement of cf1.15 that is comparable to the spill legs. It is at
   0.063 at step 154 and still declining, so I expect its tail-100 to come in BELOW 0.037 (more total training
   = better-balanced router), not above.
4. General rule for this work: only compare drop fractions at the same fraction of the LR schedule, and prefer
   a tail window over any single step.

## cf1.15 LANDED 06:00 — it IS compliant at steady state, and the published claim was right for the wrong reason
/mwittmann/ep25d1-cf115-m0-350-0726-0434, provenance verified in-log ("capacity_factor": 1.15), 349 samples.
| config | p50 MFU | TRUE tail-100 | clears 3%? |
| cf1.0  m=0 | 22.062 | 0.0710 | no |
| cf1.0  m=2 | 21.872 | 0.0414 | no |
| cf1.0  m=3 | 21.849 | 0.0366 | no |
| cf1.15 m=0 | 20.416 | 0.0260 | YES |
THE RECORD REPAIR RESOLVES AS "CONFIRM THE CONCLUSION, REPLACE THE EVIDENCE": QB+cf1.15 IS 3%-compliant -
2.60% at steady state. But the number the published comment cited for that claim (0.037, i.e. 3.7%) is ABOVE
the bar and was an end-of-anneal 120-step reading. So the claim was accidentally correct: the cited evidence
never supported it, and the true supporting evidence did not exist until this leg. Both the original error and
its sibling (the "draw variance" misattribution) stand corrected, and the headline conclusion survives.
SECOND CORRECTION: cf1.15's MFU is 20.416%, not the 20.85% on the record - it is 0.43pp MORE expensive than
believed. So the price of compliance via capacity is higher than the ledger says.
PRE-REGISTERED TEST, stated either way as promised: predicted 2.3%, measured 2.60%, error +0.30pp (+13%).
The model is OPTIMISTIC on the capacity axis but only mildly.
MODEL CALIBRATION SPLITS CLEANLY BY AXIS - the useful diagnostic:
  capacity axis: 1.03x (cf1.0 m0), 1.13x (cf1.15 m0)  -> well calibrated
  spill axis:    1.36x (m=2), 1.54x (m=3)             -> optimistic, and grows with m
So the model's weakness is specifically its idealization of how many free buckets later spill attempts find;
its capacity response is trustworthy. That is exactly the split that makes the combination prediction usable.
COMBINATION CANDIDATES (spill-axis K applied, capacity interpolated ~1.03):
  m=2 @ cf1.05: MFU 21.29%, drops ~1.78%  -> +0.87pp over measured cf1.15
  m=3 @ cf1.05: MFU 21.27%, drops ~1.24%  -> +0.85pp over measured cf1.15
Both are predicted compliant with real margin, and both beat the only measured compliant config by ~0.9pp
(better than the +0.44pp I estimated when cf1.15 was believed to be 20.85%).
RECOMMENDATION for the final leg: m=3 @ cf1.05. Same MFU as m=2 within 0.02pp but 0.54pp more compliance
margin, which is worth having given the spill axis is the optimistic one. If it lands at ~1.2% that is a
signal there is headroom to lower cf further, and cf1.02 (est. 21.62%, ~2.3%) becomes the follow-up.

## RETRACTION 06:05 — cf1.15's "20.85% is wrong" was itself the error class we have been fixing
I wrote that cf1.15's MFU is 20.416% "not the 20.85% on the record". That was wrong and I retract it.
20.85% came from a 120-STEP leg; my 20.416% is a 350-STEP leg. By the LR-schedule finding I established
myself, those sample different schedule positions; and by the drop-artifact rule also established this
session, they sit at different DROP REGIMES (3.7% end-of-anneal vs 2.60% steady state). Fewer drops means
more real work at the same step accounting, which reads as LOWER MFU. So:
  20.85%  @ 3.7% drops, 120-step leg
  20.416% @ 2.60% drops, 350-step leg
are CONSISTENT measurements of one configuration in two regimes, not a correction. Calling the older one
wrong would have repeated exactly the mistake this whole repair exists to fix - reading a right number in the
wrong frame. Rule going forward: quote cf1.15 with its run length and drop level attached, and use the
350-step number when comparing against the 350-step spill legs.

## SAME-REGIME CAPACITY PRICE and the re-derived prediction
Both endpoints now measured at 350 steps on my own draws:
  cf1.0 m=0 22.062 -> cf1.15 m=0 20.416 = -1.646pp per +0.15 cf = -0.5487pp per +0.05
  (the 120-step pair implied -0.583pp per +0.05; the same-regime price is 0.034pp cheaper)
PRE-REGISTERED PREDICTION for /mwittmann/ep25d1-spill3-cf105-350-0726-0603 (m=3 @ cf1.05, 350 steps),
every term now from the same regime:
  MFU   = 22.062 - 0.5487 (capacity) - 0.213 (spill m=3) = 21.30%
  drops = 0.0078 (model) x 1.590 (axis-split correction: 1.50 spill at m=3, 1.03 capacity interp at cf1.05)
        = 1.24%
  => predicted +0.88pp over cf1.15 m=0 (20.416% @ 2.60%) at comparable-or-better fidelity, same run length.
Stated before the result, and I will report it against the measurement either way.

## FINAL COMBINATION LEG 07:30 — clears the bar, beats cf1.15, but MFU missed my prediction by 0.63pp
/mwittmann/ep25d1-spill3-cf105-350-0726-0603 (provenance "capacity_factor": 1.05), 349 samples, then failed at
teardown as before; data valid.
PRE-REGISTERED TEST, stated before the run:
  MFU:   predicted 21.30%  measured 20.670%  error -0.63pp   <- MISSED
  drops: predicted 1.24%   measured 1.72%    error +0.48pp   <- close, right side of the bar
FULL FRONTIER (all my draws, all 350-step tail-100):
| config | p50 MFU | tail-100 | clears 3%? |
| cf1.0  m=0 | 22.062 | 0.0710 | no |
| cf1.0  m=2 | 21.872 | 0.0414 | no |
| cf1.0  m=3 | 21.849 | 0.0366 | no |
| cf1.05 m=3 | 20.670 | 0.0172 | YES |
| cf1.15 m=0 | 20.416 | 0.0260 | YES |
RESULT: the combination IS the best compliant config - 20.670% at 1.72% drops vs cf1.15's 20.416% at 2.60%.
Better on BOTH axes. But the margin is +0.25pp, not the +0.88pp I predicted.

## WHY THE PREDICTION MISSED: the capacity price is a CLIFF, not a slope
Three same-length points now pin the capacity curve:
  cf1.00 -> cf1.05: 1.179pp for +0.05 (net of the 0.213pp spill cost)   EXPENSIVE
  cf1.05 -> cf1.15: 0.254pp for +0.10 = 0.127pp per +0.05               NEARLY FREE
I had assumed a linear -0.5487pp per +0.05 from the two endpoints. The truth is strongly front-loaded.
MECHANISM (capacity alignment): capacity = ceil(cf * assignments_per_shard / num_experts).
  cf1.00   -> 2048 = 16.00 x 128  ALIGNED
  cf1.05   -> 2151 = 16.80 x 128  unaligned
  cf1.15   -> 2356 = 18.41 x 128  unaligned
cf1.0's capacity is exactly 16 x 128. The cost is paid on LEAVING that alignment, and once paid, growing
capacity further is almost free. That is a tiling effect on the expert GEMM's M dimension, not the cost of
the extra compute - which is consistent with the collective-bound picture, where a bit more padded compute
is nearly free but a badly shaped GEMM is not.
TWO CONSEQUENCES:
1. RETRACT my proposed cf1.02 follow-up. Capacity would be 2089 (unaligned), so it pays the SAME cliff and
   lands near 20.6%, not the 21.62% I estimated from the linear price. The estimate was an artifact of
   assuming a slope where there is a step.
2. NEW TESTABLE CANDIDATE: cf 1.0625 -> capacity 2176 = 17 x 128, REALIGNED. If the cliff is alignment, this
   should recover most of the 1.18pp while keeping cf1.05-or-better fidelity (more capacity than cf1.05).
   Predicted ~21.6-21.8% at drops below 1.72%. That would be a compliant config ~1.2pp above cf1.15 and
   would make the combination clearly worth it rather than marginally.
This is the third time a smooth-looking extrapolation has failed against a structural effect in this work
(schedule position, log truncation, now capacity alignment). Pattern: on this stack the interpolations that
break are the ones crossing a discretization boundary.
