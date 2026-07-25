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
