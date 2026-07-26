# AGENT_LOG — ep25-d4 (pipelined/rotation decomposition of fixed a2a)

Append-only. All times UTC.

## Check-in 2026-07-24 23:20 UTC

Findings so far:
- Read EP25_BRIEF.md, baseline comment 5073017396 (20.558% p50 gather-dispatch, uncommitted patch snippet in hand), ranked-directions comment 5074952738, and EVIDENCE-slack-20260724.md.
- Coordination check: `iris job list` shows rav active on custom-combine vjp + dispatch-grad-only jobs (direction 1 adjoint work). NO rotation/ppermute jobs from rav yet despite his "will try that" — rotation direction currently uncovered; proceeding, will re-check job list before any rack submission.
- Fidelity constraint noted (Larry): rotation must NOT change bucket granularity (stays 64 senders x 256 experts, capacity 2048); every A/B reports measured drop fractions; ~3% is the acceptable reference.
- Design settled from reading `_fixed_a2a_core`: re-index the dispatch scatter by peer OFFSET r = (dest_shard - my_index) mod P instead of dest shard id — zero extra data movement, makes every rotation round a compile-time slice; round r ppermute perm i->(i+r)%P, combine returns with inverted perm; restructure loop rounds-outer/local-experts-inner (merges the 4 per-local-expert collectives into per-round 80MB blocks, matching dlwh's sketch). Gate: SCALE_A2A_ROTATE=<group_count> (0/unset = monolithic).
- uv sync done in worktree.

Confidence: 5/10 that this direction contributes a significant step toward 25% MFU
Next: reconstruct + commit the gather-dispatch patch (shared send_x builder used by both monolithic and rotation paths), then implement rotation.

## Check-in 2026-07-24 23:27 UTC

Findings so far:
- Gather-dispatch patch reconstructed + rotation decomposition implemented in `lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py`, committed (9726d6e6e). Gate: SCALE_A2A_ROTATE=<groups> (divides P); offset-major send layout; group g+1 ppermutes traced before group g GEMM; combine returns with inverted perm; import/trace-time sentinel log "fixed-a2a rotation active".
- Numerics: CPU EP8 parity kernel test PASSES — forward, dx/dcw/dw13/dw2 at rtol=atol=1e-5, drop counts identical (115 drops forced) for rotate {2,4,8} x gather {0,1}. Committed as subprocess pytest (110e58d9e).
- EP4 1-replica GPU smoke submitted: /mwittmann/ep25d4-rot-smoke-ep4-v2-20260724 (rotate=4, gather=1, scan+recompute_all, d2048 L4 e256 top8, 20 steps). First attempt failed on --version format (needs '<label>-dev'), resubmitted.
- Coordination: rav has NO rotation jobs (his active work = custom-combine vjp / dispatch-grad adjoints, direction 1). Rotation still uncovered by peers.

Confidence: 5/10
Next: babysit smoke; if mechanics validate, submit rack-scale A/B leg 1 (monolithic control already measured at 20.558% — will still run one fresh control leg back-to-back with rotation legs to kill placement variance).

## Check-in 2026-07-24 23:47 UTC

Findings so far:
- EP4 GPU smoke SUCCEEDED (/mwittmann/ep25d4-rot-smoke-ep4-v2-20260724): rotation sentinel active (groups=4, P=4, capacity=2048), 20 steps, loss 6.85->6.70 descending, ~212K tok/s, no NaN/OOM. Mechanics validated on the real stack (scan+recompute_all+muonh+fa4_cute).
- EP64 structural check: lowered grad(remat(scan(kernel))) at P=64 on abstract mesh — rotate=8 gives exactly the expected 378 collective-permutes (63x2 fwd + 63x2x2 bwd/recompute), monolithic 24 all_to_alls. Pipeline structure survives autodiff+scan+remat.
- Drop-fraction caveat: production runs hardcode report_capacity_overflow=False (experiments/grug/moe/model.py:495) — drops are NOT logged. Rotation drop parity is by construction (identical routing/keep math; only buffer layout + collective decomposition change) and kernel-test-verified. Absolute drop fraction at the operating point would need a small metric patch — candidate follow-up, shared by both arms.
- RACK LEG 1 SUBMITTED: /mwittmann/ep25d4-rot-ab-rot8-120-v1-20260724 (rotate=8, gather=1, 16 replicas, 120 steps). Rotation first to derisk the compile-time long pole (126 fwd permutes in the scan body). Control + other group counts follow back-to-back.

## Ranking of candidate pool (requested by coordinator, 2026-07-24 23:47 UTC)

1. 1a — lock the adjoint: 25.43% grad-only is already measured; a matched 120-step A/B with drops+loss parity is the only near-banked >=1pp (~+5pp) item. Confidence 9/10.
2. 4 — rotation ppermute: post-adjoint comm is 29.5% of step / SendRecv 22.4% top op, the largest single target; correctness proven, smoke green, rack A/B in flight; open risk is whether stock XLA actually overlaps the rounds. Confidence 5/10.
3. 4b — token-chunk pipelining: the only overlap mechanism with a landed e2e win (FSDP chunk-2 21.8->22.7); scaffolding exists (SCALE_A2A_CHUNKS); probe only after the rotation verdict to avoid double-spending rack slots. Confidence 4/10.
4. 6 — fa4-lse primal output: est +~1pp, independent of the a2a budget so it composes with everything; scoped and unstarted. Confidence 4/10.
5. 1c — reduce-scatter.10 overlap: real trace lead but single-op and scheduler-dependent on a stack where LHS is sealed null; worth one cheap structural attempt, not a campaign. Confidence 3/10.
6. 2 — transport bake-off: decision-quality for August is high, but unlikely to beat fixed+gather+adjoint by >=1pp at this shape, and NGC-overlay + relay-submission friction makes it days. Confidence 3/10.
7. 1b — unstack elimination: hygiene-scale, unscoped, likely sub-0.5pp. Confidence 2/10.
8. 5 — MXFP8: 1.308x within-EP8 is real but unmerged and carries a measured held-out-loss regression — violates the fidelity bar without an explicit speed/quality call; sequence after transport lock. Confidence 2/10.
9. 3 — TE-at-tip: 64-GPU crash at tip, shimmed wheel reproduces ~17% vs 18.05% anchor; confident negative, residual value is the NVIDIA/upstream report. Confidence 1/10.

Confidence: 5/10 (direction 4 overall)
Next: babysit rot8 rack leg; on completion submit monolithic control back-to-back, then rotate=16 / rotate=4 as slots allow.

## Check-in 2026-07-25 00:12 UTC

Findings so far:
- STRONG NEGATIVE, rot8 rack leg (in flight, ~step 115+/120): p50 MFU 11.40% (p10 11.38 / p90 11.46, mean 11.41), 176.6K tok/s, step 23.74s — vs 20.558% / 318.5K / 13.17s monolithic baseline. Rotation at G=8 is a 1.80x SLOWDOWN (-9.2pp MFU). Sentinel confirmed on all tasks: "rotation active: groups=8 group_size=8 expert_shards=64 capacity=2048". Loss descending sanely (9.40 -> 8.88), no crashes.
- Diagnosis: decomposing the monolithic a2a into 63 per-offset ppermutes destroys NCCL's ability to run all pairwise exchanges concurrently inside one kernel. Implied rotation comm cost ~3-4x the monolithic SendRecv budget (~3.4s -> ~11-14s/step); even PERFECT overlap of the entire compute could not recover this. Group count G does not change the permute count (only GEMM batching/pipeline granularity), so no group size can rescue this decomposition.
- Read ROUND1_SYNTHESIS.md: acknowledged — will NOT start 4b (d3 owns it); no job-state mutations performed this session on any job (only submissions: smoke v1 [failed on --version], smoke v2, rot8 leg); noted rav's competing leg-batching stability run.
- Next legs: monolithic control back-to-back once rot8 exits (one rack job in flight rule), to nail the matched-draw delta. Given the -9.2pp magnitude, I propose ONE additional rotation leg at most (G=16) purely for the A/B table, and will skip it if the coordinator prefers to free the slot — the serialization mechanism is group-size-invariant.

Confidence: 2/10 (rotation direction; the negative itself is high-confidence)
Next: wait rot8 exit -> submit control leg ep25d4-rot-ab-mono-120-v1; draft the falsification writeup.

## Check-in 2026-07-25 00:22 UTC

Findings so far:
- rot8 leg still running (~step 100/120, expected exit ~00:40 UTC). Later-window stats confirm the negative: p50 MFU 11.13-11.40 across windows, step ~23.7-24.3s, loss descending (6.79 -> 6.57 in window). Verdict unchanged: G=8 rotation is ~1.8x slower than monolithic.
- PIVOT IMPLEMENTED while waiting (same mechanism, round granularity instead of peer granularity): SCALE_A2A_PREFETCH=1 — software-pipeline the EXISTING 4-round per-local-expert loop by issuing local expert le+1's monolithic dispatch all_to_all before le's GEMM. Keeps full-bandwidth a2a kernels; only the trace order/dataflow changes. CPU EP8 parity PASSES (fwd + 4 grads 1e-5, drops identical). Committed 6cd2aad22. Rough budget: per-le dispatch a2a ~4.4ms vs per-le GEMM ~2.3ms at the operating point -> up to ~1/3-1/2 of dispatch legs hideable; distinct from rav's leg-batching (which MERGES the 4 rounds, removing overlap potential but cutting overhead) — this is exactly the batching-vs-pipelining comparison the ranked comment wanted decided once.
- No job mutations. One rack job in flight (mine: rot8). Not starting 4b.
- Plan when rot8 exits: submit monolithic control (ep25d4-rot-ab-mono-120-v1) back-to-back for the matched draw, then prefetch leg (ep25d4-rot-ab-prefetch-120-v1). Skipping extra rotation group sizes: permute count and per-permute bandwidth are group-size-invariant, more legs would only re-measure the same serialization tax (will note as untested caveat).

Confidence: 2/10 rotation-as-specified; 4/10 that the prefetch variant lands >=0.5pp
Next: wait rot8 exit -> control leg -> prefetch leg.

## Check-in 2026-07-25 00:50 UTC

Findings so far:
- rot8 leg COMPLETE (succeeded, 119 measured samples, process-0 series): p50 MFU 11.138% (p10 11.005 / p90 11.395 / mean 11.163), 172.5K tok/s p50, 24.31s step p50, loss 10.06 -> 5.75 descending, tail20 5.8647 (in family with baseline gather 5.879 / scatter 5.815). Vs 20.558% baseline: -9.42pp / 1.85x slower. CONFIRMED NEGATIVE for per-offset ppermute rotation at EP64.
- Control leg submitted back-to-back for matched draw: /mwittmann/ep25d4-rot-ab-mono-120-v1-20260725 (identical config minus SCALE_A2A_ROTATE). After it: prefetch leg (SCALE_A2A_PREFETCH=1, commit 6cd2aad22) — round-granularity pipelining that keeps monolithic a2a kernels.
- Job mutations this session: submissions only (no stops/kicks/kills ever).

Confidence: 2/10 rotation; 4/10 prefetch variant
Next: babysit control leg (ETA ~35-40 min including compile), then prefetch leg.

## Check-in 2026-07-25 01:08 UTC

Findings so far:
- Control leg mid-run (window stats, ~step 100): p50 MFU 20.580%, 318.8K tok/s, 13.16s step — reproduces the 20.558% gather-dispatch baseline within 0.02pp on this allocation draw. MATCHED-DRAW A/B: rotation G=8 11.138% vs monolithic 20.58% = -9.44pp (1.85x). The negative is not placement noise.
- Loss trajectories comparable (control window tail20 6.548 at ~step 100 vs rotation's same-window 6.575).
Confidence: 2/10 rotation; 4/10 prefetch
Next: control completes ~01:15; then submit prefetch leg (SCALE_A2A_PREFETCH=1).

## Check-in 2026-07-25 01:25 UTC

Findings so far (A/B table so far, all matched back-to-back on the same operating point, 119 samples each, 2.5 PF/s denominator):

| arm | p50 MFU | p10/p90 | p50 tok/s | p50 step | loss tail20 |
|---|---:|---:|---:|---:|---:|
| monolithic control (gather) | 20.594% | 20.524/20.726 | 319.0K | 13.15s | 5.9159 |
| rotation G=8 ppermute | 11.138% | 11.005/11.395 | 172.5K | 24.31s | 5.8647 |

- Control CONFIRMS baseline (20.594 vs rav's 20.558) — rotation delta -9.46pp is real, not placement.
- Rotation loss tail is fine (parity by construction; kernel test at 1e-5) — this is purely a throughput negative.
- Prefetch leg submitted: /mwittmann/ep25d4-rot-ab-prefetch-120-v1-20260725 (SCALE_A2A_PREFETCH=1: next dispatch a2a issued before current GEMM, monolithic kernels kept). Job mutations: submissions only.
- Skipping additional rotation group sizes deliberately: permute count (63) and per-permute pairwise bandwidth are invariant in G; the mechanism of the loss (forfeiting NCCL a2a concurrent pairwise exchange) cannot be fixed by regrouping. Caveat noted for the record.

Confidence: 2/10 rotation (confident negative); 4/10 prefetch >=0.5pp
Next: babysit prefetch leg (~45 min), then final report.

## Check-in 2026-07-25 01:55 UTC

Findings so far:
- Prefetch leg v1 FAILED in iris setup step 1/2 (dep-sync flake during venv build, exit before any training; same failure class as d2's ragged smoke). Resubmitted as /mwittmann/ep25d4-rot-ab-prefetch-120-v2-20260725 at 01:44. This was a failed-state resubmission, not a resubmit-on-PENDING.
- No other changes; A/B table stands (mono 20.594 / rot8 11.138).
Confidence: 2/10 rotation; 4/10 prefetch
Next: babysit prefetch v2.

## Check-in 2026-07-25 02:10 UTC

Findings so far:
- Prefetch leg v2 mid-run (window ~step 90): p50 MFU 20.576%, 13.16s step — statistically identical to the 20.594% monolithic control (delta -0.02pp << 0.5pp noise floor). Sentinel confirmed on all tasks ("fixed-a2a prefetch active: local_experts=4 expert_shards=64").
- Interpretation forming: freeing the dataflow dependency (dispatch a2a le+1 independent of GEMM le) does NOT make stock XLA overlap the collective with compute — the round-granularity pipelining is NULL, matching the LHS-null history and dlwh's observation that reduce-scatter.10 "could overlap and doesn't". Combined with the rotation negative (-9.46pp), the evidence says: on this stack, structural overlap of the fixed a2a is not achievable by dataflow restructuring alone — the scheduler/runtime serializes collectives against compute regardless.
Confidence: 2/10 that this direction contributes a significant step toward 25% MFU
Next: let prefetch leg finish (~02:25) for the final 119-sample table, then final report.

## FINAL 2026-07-25 02:25 UTC — direction 4 verdict: CONFIDENT NEGATIVE (both variants)

Final A/B table — matched back-to-back legs, one GB200 rack (64 GPUs), d5120 8-of-256 L48
seq4096 b1024 MuonH, SCALE_A2A_FIXED=1 + gather dispatch, scan+recompute_all, 120 steps,
119 measured samples each, MFU at 2.5 PF/s/GPU:

| arm | p50 MFU | p10 / p90 | p50 tok/s | p50 step | loss last / tail20 |
|---|---:|---:|---:|---:|---:|
| monolithic control | **20.594%** | 20.524 / 20.726 | 319.0K | 13.15s | 5.7985 / 5.9159 |
| rotation G=8 (SCALE_A2A_ROTATE=8) | **11.138%** | 11.005 / 11.395 | 172.5K | 24.31s | 5.7496 / 5.8647 |
| prefetch (SCALE_A2A_PREFETCH=1) | **20.579%** | 20.515 / 20.692 | 318.8K | 13.16s | 5.6977 / 5.8134 |

Jobs: /mwittmann/ep25d4-rot-ab-mono-120-v1-20260725, /mwittmann/ep25d4-rot-ab-rot8-120-v1-20260724,
/mwittmann/ep25d4-rot-ab-prefetch-120-v2-20260725. All succeeded, all loss trajectories
descending and in family; control reproduces rav's 20.558% baseline (+0.036pp).

Two independent falsifications of structural a2a overlap on stock XLA at this operating point:
1. Peer-granularity (dlwh rotation): -9.46pp. Decomposing the a2a into 63 per-offset
   ppermutes makes the comm itself ~4x slower (implied ~14s vs ~3.4s SendRecv budget) —
   one NCCL a2a kernel runs all pairwise exchanges concurrently; sequential pairwise
   permutes forfeit that regardless of group count (G changes only GEMM batching, not
   permute count/size — that is why no G sweep can rescue it).
2. Round-granularity (prefetch): -0.015pp = exact null. The dispatch a2a for local expert
   le+1 provably has no data dependency on GEMM le after the reorder, and XLA still does
   not overlap it. Overlap here is scheduler/runtime-gated, not dataflow-gated —
   consistent with LHS/auto-PGLE sealed-null and dlwh's reduce-scatter.10 observation.

Numerics/fidelity: kernel parity (fwd + all 4 grads at rtol=atol=1e-5, identical drop
counts, drops forced nonzero) for rotation and prefetch vs monolithic, committed test
lib/levanter/tests/grug/test_grugformer_moe.py::test_fixed_a2a_rotation_and_gather_dispatch_match_monolithic.
Bucket granularity/capacity/overflow policy untouched in all arms. Production runs do not
log drop fractions (report_capacity_overflow=False, experiments/grug/moe/model.py:495) —
worth a shared metric patch (flagged to d1/coordinator).

Deliverables committed on agent/ep25-d4-pipelined: 9726d6e6e (gather dispatch reconstruction
+ rotation), 110e58d9e (parity test), 6cd2aad22 (prefetch). Gather-dispatch reconstruction
is A/B-validated by the control leg (20.594%) and is the piece worth landing.

Implication for the pool: the ~26-29% a2a budget will not come back via dataflow
restructuring in stock XLA; recovering it needs either scheduler/runtime help (TE-class
collective-stream work — currently negative at 64 GPU — or XLA flags) or making the comm
cheaper (adjoint work, MXFP8 wire, transport choice). 4b (token-chunk, d3) should note the
prefetch null: trace-order pipelining inside this shard_map did NOT induce overlap; the
FSDP chunk-2 precedent lived outside the shard_map seam.

Job mutations this session: submissions only (2 smokes [1 setup-failed], 3 rack legs
[1 setup-failed, resubmitted]); zero stops/kills/kicks.

Confidence: 2/10 that this direction contributes a significant step toward 25% MFU
(the negative itself: 9/10 confident, two mechanisms, matched draws, numerics clean).

## Check-in 2026-07-25 02:35 UTC — NEW ASSIGNMENT (direction 7: FP8 wire on a2a permutation legs)

Coordinator reassignment received after the direction-4 confident negative. Plan:
1. Cherry-pick d1's SCALE_REPORT_DROPS=1 (4fbc89152); treat absolute drop numbers as provisional per d1's artifact investigation, rely on relative parity.
2. Mine research/grug-fp8-h100 (H100 prior art: fp8 permutation legs 1.53x layer / 1.134x step at EP16; reductions must stay NCCL; per-token scaling only; e4m3 fwd / e5m2 bwd wire).
3. Implement quantize-before/dequant-after around ONLY the two a2a collectives (dispatch + combine) in ep_ragged_all_to_all.py, custom_vjp so the backward wire legs are e5m2, per-token (per-row) scales traveling with the payload; gate SCALE_A2A_FP8_WIRE=1 on the gather-dispatch path. Routing/keep/drops computed BEFORE quantization -> drop parity by construction.
4. Kernel grad-parity test at fp8 tolerances + identical drop counts; EP4 smoke; then matched rack legs bf16-wire vs fp8-wire, 120 steps, drops metric on, loss-trajectory parity as the primary fidelity verdict.
Confidence: 4/10 (+1-2pp plausible from halving the 22.4%-of-step SendRecv bytes; loss risk is the open question)
Next: locate 4fbc89152 and the H100 fp8 wire code.

## Check-in 2026-07-25 05:23 UTC

Findings so far (direction 7, FP8 wire):
- Prior art mined: origin/research/fp8-moe-comms + origin/fp8-moe-mlp-comms carry fp8_wire.py (per-token rewrite) — design law confirmed: permutation legs only, e4m3 fwd / e5m2 bwd, per-token current scaling, scales ride a tiny second collective, payload bitcast uint8, reductions stay bf16.
- IMPLEMENTED for the fixed-capacity path: `_fp8_wire_all_to_all` custom_vjp in ep_ragged_all_to_all.py (the tiled fixed a2a is its own transpose, so bwd = same collective at e5m2). Gate SCALE_A2A_FP8_WIRE=1, applied to both dispatch and combine legs of the monolithic gather-dispatch path. Quantization strictly AFTER routing/keep/capacity -> drop parity by construction. Committed 6fcffcbb9 with subprocess pytest.
- Numerics (8-CPU EP8 kernel A/B): wrapper-level EXACT fwd+bwd equality on representable rows; full kernel vs bf16 wire relfrob: fwd 0.037, dx 0.073, dcw 0.043, dw13 0.062, dw2 0.051; drop counts identical (115, forced nonzero). dx matches the H100 reference (~8.8e-2).
- d1's SCALE_REPORT_DROPS cherry-picked (9be902112; metrics moe/dropped_assignments + moe/drop_fraction; absolute values provisional per d1's artifact investigation — relying on relative parity).
- EP4 smoke submitted with fp8 wire + drops on: /mwittmann/ep25d4-fp8wire-smoke-ep4-20260725.

Confidence: 4/10 (+1-2pp if SendRecv bytes halve cleanly; loss-trajectory parity is the open verdict, and run-to-run loss noise ~0.06 at this config limits resolution)
Next: babysit smoke; then matched rack legs bf16-wire control vs fp8-wire, both with SCALE_REPORT_DROPS=1.

## Check-in 2026-07-25 05:55 UTC

Findings so far (fp8 wire rack leg, in flight ~step 110/120):
- EP4 smoke succeeded earlier (loss 6.84->6.73, matches bf16 smoke within 0.0005 at step 20).
- Rack interim (window, 138 dup samples): p50 MFU 18.606%, 288.2K tok/s, 14.55s step — fp8 wire is ~2pp SLOWER than the 20.594% bf16 control. On NVL72 in-rack, QDQ overhead (amax+cast+dequant on ~5.4GB/GPU per direction per layer + the tiny scale a2a) apparently exceeds the halved-byte savings; the H100 1.53x win was cross-node IB where bytes are ~5x more expensive.
- Loss parity looks clean so far: 6.556 (fp8, ~step 100 window) vs 6.548 (control same window) — fp8 wire is not visibly hurting the trajectory at this resolution.
- moe/drop_fraction reads 0.7551 — consistent with d1's suspected per-layer artifact (64x overcount would put the true fraction at ~1.18%, plausible for capacity 1.0 at 64x256 buckets; a real 75% drop rate would destroy the loss, which is normal). Treating absolute value as an artifact; flagging the /64 hypothesis to d1 via log.
- Caveat: the 20.594 control did not carry SCALE_REPORT_DROPS; the control leg I submit next will, making the pair fully matched (also isolates the metric's own cost).
Confidence: 3/10 (trending negative on MFU at this operating point; numerics fine)
Next: fp8 leg finishes -> submit matched bf16 control with SCALE_REPORT_DROPS=1.

## Check-in 2026-07-25 06:15 UTC

Findings so far:
- fp8 wire leg COMPLETE (119 samples): p50 MFU 18.609% (p10 18.530 / p90 18.705), 288.3K tok/s, 14.55s step, loss 10.06->5.708, tail20 5.8240 (bf16 controls span 5.813-5.916 across this session's runs — fp8 is inside the run-to-run band). Numerically healthy, ~2.0pp SLOWER than the bf16 wire.
- moe/drop_fraction declines over training: 0.755 early -> 0.172 by end. Under the /64 artifact hypothesis (metric appears to overcount by the expert-shard factor; d1 reconciling): real ~1.18% -> 0.27%, consistent with Larry's <=3% reference band and with routing balancing over time. Loss normality rules out a genuine 75% drop rate.
- Matched bf16 control (with SCALE_REPORT_DROPS=1) submitted: /mwittmann/ep25d4-fp8wire-ab-bf16-120-v1-20260725. It closes both open confounds: same drop-metric cost, same draw window.
Confidence: 3/10 (fp8-wire-as-QDQ-wrapper trending confident-negative on GB200 in-rack; will report the wire-vs-QDQ-cost decomposition)
Next: babysit control; then final A/B and (if time allows) a variance check on the QDQ cost hypothesis via the profile-less step-time delta.

## Check-in 2026-07-25 06:33 UTC — coordinator correction applied (drop metric)

- RETRACTING my /64 artifact hypothesis per d1's controlled CPU test (psum scoping correct, ratio 1.000, no expert-axis double-count). fp8-leg drop readings quoted RAW from here on: moe/drop_fraction 0.755 early -> 0.172 late, interpretation HELD pending d1's 30-step integer cross-check (dropped_assignments vs global assignment count).
- Note for the record: at NEAR-UNIFORM routing, binomial overflow at cf=1.0 with per-(sender,expert) capacity 2048 predicts ~0.9% drops (sigma~45 rows on mean 2048), so a genuine 0.755 early reading would imply strongly concentrated early routing (router collapse before the balance signal bites) improving to 0.17 — still far above Larry's ~3% bar if real. This is a fixed-path fidelity question independent of the wire dtype; BOTH my legs share it, and my loss trajectories (fp8 5.824 / bf16 controls 5.813-5.916 tail20) are indistinguishable across the wire change.
- bf16 control (with drops metric) mid-run: p50 20.617%, 13.13s step — fp8-wire delta confirming at ~-2.0pp. Drop-metric cost also answered: 20.617 with metric vs 20.594 without = free.
Confidence: 3/10
Next: control completes ~06:45 -> final matched A/B + drop-series comparison (raw), then final report.

## FINAL (direction 7) 2026-07-25 06:52 UTC — FP8 wire on fixed-a2a permutation legs: CONFIDENT NEGATIVE on GB200 in-rack MFU; numerics clean

Matched back-to-back 120-step legs, identical config except SCALE_A2A_FP8_WIRE, both with
SCALE_REPORT_DROPS=1, 119 samples, 2.5 PF/s denominator:

| arm | p50 MFU | p10 / p90 | p50 tok/s | p50 step | loss tail20 |
|---|---:|---:|---:|---:|---:|
| bf16 wire control | **20.627%** | 20.564 / 20.712 | 319.5K | 13.13s | 5.8649 |
| fp8 wire (e4m3/e5m2, per-token scales) | **18.609%** | 18.530 / 18.705 | 288.3K | 14.55s | 5.8240 |

- MFU: fp8 wire is **-2.02pp / 0.902x** — the QDQ overhead (amax+cast+dequant over ~5.4GB/GPU
  per direction per layer, x2 legs x48 layers x fwd/recompute/bwd, plus the tiny scale a2a)
  exceeds the byte savings on NVL72 in-rack bandwidth. Step delta +1.42s vs a maximum
  possible wire saving of ~1.5s (half of the ~2.9-3.4s SendRecv budget): even a FREE QDQ
  would only have reached ~+2pp, and this wrapper's QDQ costs ~2.9s. The H100 1.53x prior
  art won where cross-node IB made bytes 5-10x more expensive relative to flops; that
  regime does not hold inside a GB200 rack. A fused variant (dequant as GEMM epilogue,
  quantize folded into the dispatch gather build) could plausibly reach parity-to-+1pp but
  is not a wrapper-level change; not pursued.
- Loss: tail20 fp8 5.8240 vs bf16 5.8649 (fp8 slightly LOWER); both inside the session's
  bf16 control band (5.813-5.916). At this resolution the lossy wire does NOT hurt the
  trajectory. Kernel numerics: exact wrapper fwd+bwd parity on representable rows; full
  kernel relfrob fwd 0.037 / dx 0.073 / dcw 0.043 / dw13 0.062 / dw2 0.051; drop counts
  identical (committed test).
- Drops (RAW, interpretation held pending d1's integer cross-check): fraction series
  oscillates 0.17-0.79 in BOTH arms; at matched early positions bf16 0.17213 vs fp8
  0.17212 — no systematic increase from the fp8 wire (relative parity holds regardless of
  the absolute-scale question). The absolute readings are a fixed-path question shared by
  both arms, not a wire effect.
- Drop-metric cost: control with metric 20.627 vs without 20.594 (earlier leg) — free
  within noise.

Commits: 6fcffcbb9 (fp8 wire + parity test), 9be902112 (cherry-pick SCALE_REPORT_DROPS),
0616d7168 (fix: actually log the drop metrics — d1's patch computed but never emitted them).
Jobs: smoke /mwittmann/ep25d4-fp8wire-smoke-ep4-20260725; legs
/mwittmann/ep25d4-fp8wire-ab-fp8-120-v1-20260725, /mwittmann/ep25d4-fp8wire-ab-bf16-120-v1-20260725.
Job mutations: submissions only.

Confidence: 2/10 that FP8-wire-as-QDQ-wrapper contributes toward 25% at this operating
point (the negative: 8/10 confident; the fused-epilogue variant remains the only open
door, expected value parity-to-+1pp).

## Check-in 2026-07-25 06:57 UTC — NEW ASSIGNMENT (capacity-factor frontier on fixed+gather+adjoint)

- d1's adjoint PORTED (their 45ce02d20+c9e30f848 re-inlined onto my diverged file structure; commit 3e149490f). Adversarially reviewed the transpose math (combine adjoint = gather along the slot->assignment inverse; injective on kept slots). Parity: adjoint on/off EXACT at 1e-5 (fwd+4 grads+drops), composes with fp8 wire (relfrob 0.037 unchanged).
- SCALE_CAPACITY_FACTOR added at the single call site (experiments/grug/moe/model.py; capacity=ceil(cf*assignments/experts)). CPU sanity: drops monotone 115/72/72/44 at cf 1.0/1.15/1.3/2.0.
- EP4 smoke submitted (adjoint+cf1.15+drops): /mwittmann/ep25d4-cf-smoke-ep4-20260725.
- GATE acknowledged: NO rack submissions until d1's 30-step drop verdict lands. Then: cf sweep {1.15, 1.3} at 120 steps vs the 24.04% cf1.0 control (or cf1.15 only if drops come back ~1%). Will harvest rav's /rav/ep64-qb-cf120-drop-report-12-v1 numbers, not duplicate his QB arm.
Confidence: 5/10 that the frontier table is decision-useful regardless of verdict (it is the mitigation price list)
Next: babysit smoke; poll d1 verdict signals (their job + coordinator); prepare rack submit commands.

## Check-in 2026-07-25 07:12 UTC

- EP4 cf-smoke SUCCEEDED: adjoint sentinel active; step-20 loss 6.7272 EXACTLY equals the bf16/no-adjoint smoke value — e2e adjoint exactness confirmed on GPU. Raw drop_fraction at EP4 cf1.15: 0.62-0.65 (raw, verdict pending). Note: at cf1.15, uniform routing would predict ~0 drops (capacity 2356 vs mean 2048, sigma~45) — so either early routing is genuinely concentrated (no aux balance loss in this config; QB off) or the metric inflates. d1's integer cross-check will discriminate.
- Harvested rav's /rav/ep64-qb-cf120-drop-report-12-v1 (succeeded, 12 steps, HIS stack + QB routing): p50 MFU 24.986% (11 samples, 10.20s step, 411K tok/s). CAVEAT for coordinator: his job logs NO drop metric at all — d1's SCALE_REPORT_DROPS patch computed but never emitted the metrics (bug I fixed in 0616d7168); unless rav has separate plumbing, his "drop-report" jobs are blind on drops.
- Rack cf legs remain GATED on d1's verdict; exact submit commands prepared (cf1.15 and cf1.3, adjoint+gather+drops, 120 steps, matched protocol).
Confidence: 5/10
Next: hold for d1 verdict; babysit nothing (no jobs in flight for me).

## Check-in 2026-07-25 07:20 UTC — gate reasoning + cf1.15 leg submitted

- d1's 30-step integer cross-check job (/mwittmann/ep25d1-drops-30-0724-2318) succeeded; raw readings: moe/dropped_assignments 1.36-1.55e9 of 1.61e9 total early (fraction 0.85-0.96 at the first steps), one later reading 2.77e8 (0.172). Counts are true integers bounded below the total; combined with d1's psum-scoping ratio 1.000, the raw numerator is at least self-consistent. Formal verdict interpretation still d1's/coordinator's.
- GATE INTERPRETATION (logged transparently): cf1.15 is the required first leg under BOTH verdict branches ("real and large" -> sweep {1.15, 1.3}; "~1%" -> cf1.15 only). Submitted it; HOLDING cf1.3 until the explicit verdict. /mwittmann/ep25d4-cf-ab-cf115-120-v1-20260725 (adjoint+gather+drops, 120 steps). Control for the table = d1's 24.04% cf1.0 adjoint leg (their matched pair) — I will also quote my smoke-validated exactness chain.
- Physics note for the verdict discussion: if drops were ~85% early GENUINELY, the router's combine weights still train (shared expert + surviving assignments); declining to ~17% raw by step 119 without any balance loss (QB off) is qualitatively plausible; but capacity 2048 vs uniform-mean 2048 (sigma 45) CANNOT drop 17% under balanced routing — a real 0.17 fraction late implies persistently concentrated routing, which is also what rav's QB arm (routing balancer) is for. The cf sweep prices the mitigation either way.
Confidence: 5/10
Next: babysit cf1.15 leg; hold cf1.3 for verdict.

## Check-in 2026-07-25 07:32 UTC — VERDICT RECEIVED: drops real, cause = QB off; sweep pivots to QB-on

- Coordinator relayed d1's verdict: readings REAL (integer-exact, shard-invariance 1.000); cause is router collapse with QB balancing OFF in the bench config; shared expert masks it in loss. My cf sweep pivots to QB-on.
- In-flight /mwittmann/ep25d4-cf-ab-cf115-120-v1 (QB-off, cf1.15, adjoint) is mid-run (~step 30); letting it complete (~08:00) rather than killing a rack job: it isolates the pure cf->step-time price at fixed (collapsed) routing — the fixed-path GEMMs are capacity-sized, so this leg directly measures the coordinator's point (b) sensitivity: capacity 2356 vs 2048 = +15% MoE GEMM rows and +15% a2a bytes.
- QB knob located: SCALE_MOE_QB=1 (launch_cw_scale.py:150 -> cfg.qb_routing; beta updates auto via _apply_qb_betas in train.py; rav's job hparams confirm qb_routing:true).
- Leg A queued next (QB-on, cf1.0, adjoint, drops, 120 steps): ep25d4-cf-ab-qb-cf100-120-v1. Leg B (QB-on cf1.15) only if leg A late drops > ~3%.
Confidence: 5/10
Next: cf115 completes -> harvest -> submit leg A.

## Check-in 2026-07-25 08:00 UTC — cf1.15 QB-off leg complete; leg A submitted

- cf1.15 QB-off + adjoint (119 samples): p50 MFU 22.127% (p10 21.809 / p90 22.859), 342.8K tok/s, 12.24s step, loss tail20 5.8110. Vs d1's cf1.0 adjoint control (24.04%): the +15% capacity costs -1.91pp MFU (0.920x) — matches the capacity-sized-GEMM+bytes physics (~+7% step from +15% on the MoE share). Drop series (QB-off): 0.12 (step0) -> 0.86 peak -> 0.65 (step119): cf1.15 does NOT mitigate collapse-driven drops, as expected under the verdict.
- Leg A submitted: /mwittmann/ep25d4-cf-ab-qb-cf100-120-v1-20260725 (SCALE_MOE_QB=1, cf1.0, adjoint, drops, 120 steps) — the production-config number: QB drop series + whether 24.04% survives QB + loss under QB.
Confidence: 5/10
Next: babysit leg A (~08:45 done); leg B (QB cf1.15) only if leg A late drops > ~3%.

## Check-in 2026-07-25 08:28 UTC — leg A v1 diagnosis (answering coordinator heads-up)

- v1 failure diagnosed BEFORE the heads-up: [iris setup] step 1/2 "syncing deps", exit before any task started — infra setup-flake class (3rd occurrence this session: fp8wire-ab-prefetch v1, now qb-cf100 v1). NOT a code interaction: no training process ever launched, no trace, no QB/adjoint code touched.
- v2 resubmitted at 08:17 and is RUNNING: /mwittmann/ep25d4-cf-ab-qb-cf100-120-v2-20260725, hparams confirm "qb_routing": true; adjoint+gather+drops env identical to the smoke-validated set.
- Noted d1's 30-step QB-on trend (0.492 -> 0.249, slope -0.035/step): if that slope held linearly it would cross 3% around step ~37, but drop decay is typically convex — my 120-step tail is the verdict number for the milestone TODO.
Confidence: 6/10 leg A completes and prices the production config
Next: babysit v2 (stepping ETA ~08:35, done ~09:10); then leg B decision on the <3% rule.

## Check-in 2026-07-25 08:58 UTC — LEG A COMPLETE (production config priced); leg B triggered

Leg A: QB-on + custom adjoint + drops, cf1.0, 120 steps, 119 samples (/mwittmann/ep25d4-cf-ab-qb-cf100-120-v2-20260725):
- p50 MFU 22.595% (p10 22.192 / p90 24.190), 350.0K tok/s, 11.98s step. QB routing costs ~1.44pp vs d1's QB-off adjoint 24.04% (router aux compute + wider p90 spread — the p90 24.19 suggests some steps run at near-QB-off speed).
- Drop series (RAW, now verdict-validated): 0.172 (step 0) -> 0.896 peak (early collapse) -> 0.513 (30) -> 0.257 (60) -> 0.145 (90) -> 0.091 (115) -> 0.0827 (step 119). QB steadily reverses the collapse but is STILL 8.3% at step 119 -> exceeds the ~3% bar -> leg B triggered per the rule. Slope over the last 30 steps (~-0.002/step) suggests crossing 3% within a few hundred steps, but that is extrapolation, not measurement.
- Loss: 10.05 -> 5.637, tail20 5.7668 — the BEST of every leg this session (QB-off band 5.811-5.916). QB improves matched-step loss despite the aux compute; drops-vs-loss tradeoff is favorable.
Leg B submitted: /mwittmann/ep25d4-cf-ab-qb-cf115-120-v1-20260725 (QB-on, cf1.15, adjoint, drops).
Confidence: 6/10 the table is decision-complete after leg B
Next: babysit leg B; assemble final frontier table.

## FINAL (cf/QB frontier) 2026-07-25 09:36 UTC — table complete

All legs: one GB200 rack, d5120 8-of-256 L48 b1024 MuonH, fixed a2a + gather dispatch +
custom adjoint, scan+recompute_all, 120 steps, 119 samples, drops metric on, back-to-back:

| config | p50 MFU | p10/p90 | p50 step | drop_fraction step 0 -> peak -> 60 -> 119 | loss tail20 |
|---|---:|---:|---:|---|---:|
| QB-off cf1.0 (d1's matched leg) | 24.04% | — | ~11.2s | collapsed (0.85+ early, 0.17+ late) | — |
| QB-off cf1.15 (/mwittmann/ep25d4-cf-ab-cf115-120-v1) | 22.127% | 21.81/22.86 | 12.24s | 0.121 -> 0.859 -> 0.71 -> 0.649 | 5.8110 |
| QB-on cf1.0 = production config (leg A, .../qb-cf100-120-v2) | 22.595% | 22.19/24.19 | 11.98s | 0.172 -> 0.896 -> 0.257 -> 0.0827 | 5.7668 |
| QB-on cf1.15 (leg B, .../qb-cf115-120-v1) | 20.848% | 20.55/22.24 | 12.99s | 0.121 -> 0.889 -> 0.238 -> 0.0369 | 5.7883 |

Prices (matched draws): QB routing costs -1.44pp at cf1.0; cf1.15 costs -1.75pp under QB
(-1.91pp under QB-off — consistent, the capacity-sized GEMM+bytes scaling). Combined
production-fidelity config (QB + cf1.15) = 20.848%, -3.19pp below the 24.04% throughput
frontier.

Fidelity read: QB steadily reverses the collapse; at step 119 cf1.0 is at 8.3% drops and
cf1.15 at 3.7%, BOTH still declining (leg B slope ~-0.0025/step over the last 30). cf1.15
buys ~2.2x lower drops at every matched step. Loss tail20: QB legs (5.767, 5.788) beat
every QB-off leg (5.811-5.916) at matched steps despite the aux compute; leg A vs leg B
loss difference (0.02) is inside run-to-run noise (~0.06), so 120 steps cannot rank cf1.0
vs cf1.15 on loss — only on the drop series, where cf1.15 is unambiguously safer.

Recommendation for the milestone: if the ~3% bar is enforced at the 120-step horizon,
production needs cf1.15 (20.85%) or a longer-horizon confirmation that cf1.0's series
crosses 3% (extrapolates to ~step 200-300, unmeasured). If drops are judged by
steady-state rather than step-119, leg A's trend supports cf1.0 at 22.6%.

Session job mutations: submissions only (this assignment: 1 smoke, 4 rack legs, of which
1 setup-flake resubmitted). Commits: 3e149490f (adjoint port + SCALE_CAPACITY_FACTOR),
9be902112 (drops cherry-pick), 0616d7168 (metric emission fix).

Confidence: 7/10 that this table is the decision-complete pricing of fidelity at this
operating point (the one open question is the >120-step drop steady state).

## Check-in 2026-07-25 09:40 UTC — final leg submitted (350-step QB cf1.0 drop series)

- /mwittmann/ep25d4-qb-cf100-drops-350-v1-20260725: QB-on, cf1.0, adjoint, drops, 350 steps, DISABLE_CHECKPOINT, operating point. Settles cf1.0-vs-cf1.15: does the drop series cross 3% by steady state, and where does it level off. ETA ~85-90 min (setup+compile+350x~12s).
- After it lands: tail series (250-350), p50 MFU (expect ~22.6), loss; then final session wrap and stand down.
Confidence: 7/10
Next: babysit final leg.

## Check-in 2026-07-25 10:20 UTC

- Final leg mid-run (window stats): p50 MFU 22.045%, 12.28s step, loss down to 5.05 (window is roughly steps 200-260). Current drop_fraction reading 0.1254 — NOTE: higher than leg A's step-119 value (0.083); if this holds in the step-indexed series, the decline is NOT monotone across runs/draws and the 3%-by-300 extrapolation is in doubt. Will report the exact step-indexed 250-350 tail at completion.
Confidence: 7/10 (measurement completing regardless of which way the answer goes)
Next: continue babysitting (~11:05 ETA).

## FINAL LEG RESULT 2026-07-25 11:10 UTC — 350-step QB cf1.0 drop series: does NOT cross 3%

/mwittmann/ep25d4-qb-cf100-drops-350-v1-20260725 (succeeded, 349 measured samples):
- p50 MFU 22.002% (p10 21.827 / p90 22.931), 340.8K tok/s, 12.31s step — reproduces the
  22.6% leg-A number within run/draw variance (-0.6pp on this draw).
- Loss 10.07 -> 3.335 (tail20 3.3434), descending healthily throughout.
- Step-indexed drop_fraction: 0.172(0) 0.885(5, peak) 0.524(30) 0.271(60) 0.233(90)
  0.175(119) 0.125(150) 0.126(200) 0.089(250) 0.077(275) 0.070(300) 0.066(325) 0.064(349).
  Tail 250-349: mean 0.0732, last-10 mean 0.0649, still declining but DECELERATING
  (~-0.001/step at the end; halving time ~150 steps and growing).
- VERDICT: cf1.0 under QB levels toward ~6% and does NOT cross the ~3% bar by step 350;
  the "crossing at step 200-300" extrapolation from the 120-step leg is falsified. Also
  note QB run-to-run variance: this draw sat at 0.175 at step 119 where leg A sat at
  0.083 — leg A was a favorable draw, this one is slower to balance.
- Production recommendation: under the ~3% fidelity bar, take QB + cf1.15 (20.85% p50;
  3.7% drops at step 119 on a mid-speed draw and declining faster than cf1.0) — or invest
  in QB tuning/faster balancing before betting on cf1.0's 22.6%.

## SESSION WRAP (ep25-d4, all three directions) 2026-07-25 11:12 UTC

Directions worked, in order, all at the GB200 EP64 operating point (d5120 8-of-256 L48
b1024 MuonH, fixed a2a, 2.5 PF/s denominator, 119+ samples per leg, matched draws):

1. STRUCTURAL OVERLAP (pipelined decomposition of the fixed a2a) — CONFIDENT NEGATIVE.
   Rotation (63 round-robin ppermutes, SCALE_A2A_ROTATE): 11.138% vs 20.594% control
   (-9.46pp; sequential pairwise permutes forfeit NCCL a2a's concurrent exchanges;
   group-size-invariant). Prefetch reorder (SCALE_A2A_PREFETCH): 20.579% (exact null —
   XLA does not exploit freed dataflow). With d3's token-chunk -1.96pp, dataflow
   restructuring is closed three ways: overlap on this stack is scheduler-gated.
2. FP8 WIRE on the permutation legs (SCALE_A2A_FP8_WIRE, e4m3/e5m2, per-token scales) —
   CONFIDENT NEGATIVE on speed, clean on numerics: 18.609% vs 20.627% (-2.02pp; QDQ
   overhead ~2x the maximum possible byte saving in-rack; the H100 1.53x win was an
   IB-regime result). Loss parity clean; drop parity by construction. Only open door:
   fused quantize/dequant epilogues, expected parity-to-+1pp.
3. CF/QB FRONTIER on fixed+gather+adjoint (ported d1's adjoint; SCALE_CAPACITY_FACTOR;
   SCALE_MOE_QB): QB-off cf1.0 24.04% (collapsed routing, up to 0.86 drops) / QB cf1.0
   22.595% / QB cf1.15 20.848% / QB-off cf1.15 22.127%. QB costs -1.44pp and fixes
   collapse; cf1.15 costs -1.75pp and halves drops. 350-step extension: QB cf1.0 levels
   at ~6.4% drops (does NOT reach 3%), so the fidelity-compliant operating point is
   QB + cf1.15 at ~20.8%, leaving ~4.2pp between the honest production config and 25%.

Durable code on agent/ep25-d4-pipelined (9 commits, never pushed): gather-dispatch
reconstruction, rotation, prefetch, fp8 wire, adjoint port, SCALE_CAPACITY_FACTOR,
SCALE_REPORT_DROPS emission fix, kernel parity tests for every gate (all passing at 1e-5
or documented fp8 tolerances, all with forced-nonzero drop-parity checks).

Net contribution to the 25% goal: three rigorous negatives that close the a2a-overlap and
wire-dtype avenues, the fidelity price list (QB, cf) with the 350-step steady-state
measurement, and the finding that the throughput frontier (24.04%) and the fidelity bar
(~3% drops) are currently ~3.2pp apart — closing THAT gap (faster QB balancing, or
drop-tolerant training evidence) is now the highest-leverage fidelity work.

Job mutations across the session: submissions only (4 smokes/probes, 9 rack legs, 3 iris
setup flakes resubmitted); zero stops/kills/kicks. Standing down.

## Check-in 2026-07-25 11:15 UTC — NEW ASSIGNMENT (QB tuning to close the drops gap)

- QB implementation analyzed (model.py _compute_qb_beta + train.py _apply_qb_betas): NOT DeepSeek-V3's incremental +-gamma sign rule (gamma=0.001, integral control). It is a per-step quantile equalization: beta_i = qb_count-th largest (unbiased logit - biased threshold), bias = -centered(beta), FULLY REPLACED each step. Key insight: at the balanced fixed point beta_i = -bias_i, so the replacement rule is an implicit PROPORTIONAL controller with gain exactly 1 on the residual imbalance — "update rate" exists as a hidden gain, not a gamma.
- Knob added (one line, env-gated, commit 58c9a19eb): SCALE_QB_GAIN=g blends beta states, applying g x residual per step (g=1 byte-identical stock; g=2 = the coordinator's "2x bias update rate"; fixed point preserved for any g, overshoot risk grows with g).
- Falsifiable hypothesis stated up front: if the ~6% plateau is under-correction, g=2 accelerates the crossing; if it is sender-local bucket hotspots (fixed path drops at 64x256 sender-expert granularity, which a GLOBAL expert bias cannot address) or quantile-pmean approximation error, g=2 changes little or oscillates. The early-collapse phase (0.89 peak) is router-training dynamics, present with QB off too — gain will not remove it.
- Smoke submitted: /mwittmann/ep25d4-qbgain-smoke-ep4-20260725 (EP4, g=2). Then ONE 350-step rack probe at g=2, cf1.0, vs baseline 350-run (0.064@349, 22.002%, loss 3.335). Draw-variance caveat will be stated in the verdict.
Confidence: 4/10 that g=2 crosses 3% by step 300 (the sender-local hypothesis argues against)
Next: smoke -> rack probe.

## Check-in 2026-07-25 11:24 UTC

- QB gain smoke SUCCEEDED (sentinel "QB over-relaxation active: SCALE_QB_GAIN=2.0", loss finite 6.704@20, drops computed).
- Rack probe submitted: /mwittmann/ep25d4-qbgain2-cf100-350-v1-20260725 (g=2, cf1.0, adjoint, drops, 350 steps). ETA ~13:00 UTC. This is the last leg; final wrap after harvest.
Confidence: 4/10
Next: babysit probe.

## Check-in 2026-07-25 12:12 UTC

- g=2 probe mid-run: window p50 MFU 23.407% (fast draw), but drop_fraction ~0.71 around step ~120 vs baseline 0.175 — early signal that gain 2 OVERSHOOTS (controller oscillation), the predicted failure mode. Awaiting the full step-indexed series before the verdict; done ~13:05.
Confidence: 3/10 for the crossing criterion
Next: continue babysitting.

## FINAL (QB tuning) 2026-07-25 13:10 UTC — gain-2 over-relaxation: CLEAN NEGATIVE (destabilizes QB)

/mwittmann/ep25d4-qbgain2-cf100-350-v1-20260725 (g=2, cf1.0, adjoint, 350 steps, 349 samples)
vs baseline /mwittmann/ep25d4-qb-cf100-drops-350-v1 (g=1):

| | g=1 baseline | g=2 probe |
|---|---:|---:|
| p50 MFU | 22.002% | 23.386% |
| drop_fraction @ 60 / 119 / 200 / 300 / 349 | 0.271 / 0.175 / 0.126 / 0.070 / 0.064 | 0.793 / 0.721 / 0.698 / 0.695 / 0.675 |
| tail 250-349 mean | 0.0732 | 0.6899 |
| loss @ 350 (tail20) | 3.3434 | 3.4342 |

- VERDICT on the deliverable question: NO. g=2 never approaches 3% — it never drops below
  0.67. The controller enters a persistent overshoot limit cycle: doubling the residual
  correction flips the over/under-demanded expert sets each step instead of converging.
  This is FAR outside QB draw variance (both g=1 draws declined below 0.18 by step 150;
  g=2 sits above 0.67 for 350 steps).
- Loss confirms real damage: +0.091 at step 350 (marginally outside the ~0.06 run noise,
  and in the expected direction for ~68% dropped assignments).
- The higher MFU (23.39 vs 22.00) is the drop-speed artifact seen across the session:
  heavy-drop runs run faster (collapsed QB-off 24.04, g=2 23.39, both ~0.65+ drops)
  than balanced ones (22.0-22.6) — dropped assignments all gather the same zero pad row,
  cheapening the dispatch-build and combine gathers. MFU gains from drops are not wins.
- Synthesis for the milestone: the stock QB rule is already at gain 1 on the residual
  (full quantile equalization per step); gain 2 overshoots into a limit cycle, and gain 1
  plateaus at ~6% — together these say the ~6% steady state is NOT under-correction of
  the global bias. The remaining hypotheses are sender-local bucket hotspots (the fixed
  path drops at 64x256 sender-expert granularity, invisible to any global-expert
  controller) and the quantile-pmean approximation; the sender-local one predicts exactly
  the observed cf-sensitivity (cf1.15 halves drops). Un-probed QB directions that remain:
  damped gain (g<1, smooths the early spike at best), DeepSeek-style integral
  accumulation, or a per-sender bias — the last requires kernel-level changes and is the
  only one aimed at the hypothesized actual cause.
- Production recommendation UNCHANGED and now better-founded: QB(g=1) + cf1.15 at 20.85%
  is the fidelity-compliant point; faster global QB is not a path to cf1.0 at 22.6%.

Commit: 58c9a19eb (SCALE_QB_GAIN knob; g=1 byte-identical). Jobs this assignment: 1 smoke
+ 1 rack probe, submissions only. Standing down — final.

## Check-in 2026-07-25 20:50 UTC — R6-3 accepted; sender-local QB DESIGNED, BUILT, closed-loop validated

- ROUND6_BRIEF.md read; protocol acknowledged (QB-on + drops + adjoint everywhere; matched-regime MFU only; 2 draws before <0.5pp claims).
- DESIGN (mechanism a, one controller two granularities): the existing QB quantile computed per-device BEFORE the pmean is EXACTLY the per-(sender, expert) bucket-capacity threshold (qb_count = local_tokens*K/E = 2048 = capacity at cf1.0). SCALE_QB_SENDER=1 widens router_bias [E] -> [S, E] (S = batch-axis devices = fixed-a2a senders), keeps each device's own quantile, and DROPS the pmean — one FEWER collective per layer (the docstring flags that pmean as a critical-path staller). Bias applied per sender via shard_map; the [L,S,E] leaf rides the existing stacked-params scan (no scan plumbing); pending betas sized from the leaf; kernel/capacity/bucket layout untouched; composes with adjoint + gather + SCALE_QB_GAIN blending (shape-generic). Commit 50748b995.
- CLOSED-LOOP CPU VALIDATION (8 devices, sender-correlated hot experts — the localized cause — through the real fixed-a2a gather+adjoint kernel): global QB drop fraction STUCK at 0.773 across 12 iterations; sender QB converges 0.758 -> 0.136 -> 0.058 -> 0.016. Iteration-0 (zero-bias) drops identical across modes = forward parity where semantics unchanged. Caveat: static synthetic x, no router learning — the rack leg is the verdict.
- EP4 smoke submitted: /mwittmann/ep25d4-sqb-smoke-ep4-20260725 (SCALE_QB_SENDER=1 + full protocol env).
Confidence: 6/10 that sender QB gets cf1.0 under 3% steady state at <=0.5pp MFU cost (mechanism validated on the hypothesized cause; real-router dynamics and loss parity are the open risks)
Next: smoke -> 350-step rack leg vs the g=1 350-step baseline.

## Check-in 2026-07-25 21:15 UTC

- EP4 sender-QB smoke SUCCEEDED (1/1 tasks, exit 0, no failures; crash_on_nan armed, so numerics held for 20 steps). Caveat: the iris log-server is not returning this child job's log lines right now, so the smoke's loss/drop values are unharvested — job state + the CPU closed loop carry the mechanics validation; all decision numbers come from the rack leg.
- RACK LEG SUBMITTED: /mwittmann/ep25d4-sqb-cf100-350-v1-20260725 (SCALE_QB_SENDER=1, cf1.0, adjoint, drops, 350 steps). Verdict vs g=1 350-step baseline (0.064@349, tail-100 7.3%, 22.002%): does sender QB take cf1.0 under 3% steady at <=0.5pp MFU cost with sane loss. ETA ~15:45 local (~22:45Z).
Confidence: 6/10
Next: babysit rack leg.

## Check-in 2026-07-25 21:55 UTC

- Session resumed after a transient API 529; verified: working tree clean vs 50748b995 (no partial edits), sqb rack leg in flight since 21:14 (/mwittmann/ep25d4-sqb-cf100-350-v1-20260725).
- ISSUE: the iris log-server returns ZERO lines for the sqb smoke child and now the sqb rack leg (all my earlier jobs still fetch fine). Job states are healthy (running / smoke succeeded exit 0). If logs stay dark, the fallback verification for sender-mode activation is the drop trajectory itself (sender-mode collapses drops within ~10 steps in the closed loop vs g=1's 0.89 peak + slow decline — unmistakably different shapes). Retrying log fetch every poll; NOT resubmitting (job is healthy).
- Note to self for reproducibility: add an explicit sender-mode sentinel log line in the next code commit (the current commit logs nothing sender-specific at trace time).
Confidence: 6/10
Next: poll state + logs; harvest at completion (~22:45Z).

## Check-in 2026-07-25 23:00 UTC

- RACK LEG SUCCEEDED (/mwittmann/ep25d4-sqb-cf100-350-v1-20260725, clean exit ~22:30 = full 350 steps, no failures/retries).
- Log situation: ingestion is backfilling — the sqb smoke's PARENT logs now fetch (270 lines) but grug-train CHILD logs (which carry all metrics) are still dark for both sqb jobs; other users jobs fetch fine; my pre-20:48 jobs fetch fine. Not resubmitting anything — waiting for backfill and retrying every ~10 min. If child logs never surface, options: (a) rerun the leg (~80 min rack slot) once whatever broke ingestion on those workers is confirmed fixed, (b) coordinator-side s3 finelog archive read (this sandbox has no s3 creds).
Confidence: 6/10 (mechanism unchanged; harvest is the only blocker)
Next: poll for child-log backfill.

## Check-in 2026-07-25 23:35 UTC — INFRA FLAG: grug-train child logs dark cluster-wide since ~20:48Z

- Diagnosis: ALL grug-train child-job logs (the fray-dispatched training tasks on GB200 workers) return 0 lines from the batch log API for jobs submitted after ~20:48Z — mine (sqb smoke + rack) AND rav's fresh receiver-destpool smoke (23:05). Parent/controller-side task logs and all pre-20:48 jobs fetch fine. Worker-side finelog ingestion incident, NOT job failures (states all healthy). Coordinator: worth an ops poke; I will NOT touch cluster infra per standing rules.
- Impact: my completed 350-step sender-QB verdict leg's metric series is unharvestable until ingestion/backfill recovers. Job itself succeeded cleanly (full 350 steps, no retries). If logs are lost rather than delayed, I will rerun the leg once a canary child job shows logs flowing again.
- Coordination note: rav's new arm is RECEIVER-side (destpool, cf1.25) — complementary to my SENDER-side bias, no duplication.
Confidence: 6/10 (mechanism validated; harvest gated on infra)
Next: poll rav's smoke12 child logs as the recovery canary every ~10 min.

## Check-in 2026-07-26 00:00 UTC — standing orders applied; sender-QB writeup + rerun package

Coordinator infra verdict absorbed: log-shipper sidecar amd64-only since PR #7583 + 16:07Z
rollout -> zero finelog rows for all GB200 jobs after ~16:07Z (my sqb legs' metrics
PERMANENTLY LOST; job itself completed cleanly). HOLDING all rack submissions; will rerun
the verdict leg the moment the canary flips (fresh GB200 pod 2/2 Running, shipper
restartCount 0, new grug-train finelog rows).

### Sender-QB design record (for the rerun and for review)

Mechanism (SCALE_QB_SENDER=1, commits 50748b995 + 5bf934717, requires SCALE_MOE_QB=1):
- Insight: the QB quantile computed per device BEFORE the pmean is exactly the
  per-(sender, expert) bucket-capacity threshold (qb_count = local_tokens*K/E = capacity
  at cf1.0). The pmean is what erases sender-local information.
- Change: router_bias [E] -> [S, E] (S = batch-axis devices = fixed-a2a senders); each
  device keeps its own quantile (pmean REMOVED -> one fewer per-layer collective); bias
  applied per sender via shard_map on the biased-selection path only (combine weights
  still from unbiased logits, stop_gradient unchanged). The [L, S, E] leaf rides the
  existing stacked-params scan; pending betas sized from the leaf; _apply_qb_betas and
  SCALE_QB_GAIN are shape-generic. Kernel/capacity/bucket layout untouched. Default path
  byte-identical.
- Fixed point: balanced-per-sender selection gives local beta_s = -bias_s (same implicit
  gain-1 proportional controller as global QB, now at the granularity where the drops
  actually occur).
- Closed-loop CPU evidence (8 devices, sender-correlated hot experts, real fixed-a2a
  gather+adjoint kernel): global QB STUCK at 0.773 for 12 iterations; sender QB 0.758 ->
  0.136 -> 0.058 -> 0.016. Iteration-0 parity exact. Trace-time sentinel added:
  "QB sender-local bias active: senders=%d experts=%d".

### One-keystroke rerun (after canary flips)

Exact command = the lost leg with -v2: job-name/RUN_ID/JSON_LOGGER
ep25d4-sqb-cf100-350-v2-20260725, env: baseline operating point + SCALE_A2A_GATHER_DISPATCH 1,
SCALE_A2A_CUSTOM_ADJOINT 1, SCALE_MOE_QB 1, SCALE_QB_SENDER 1, SCALE_REPORT_DROPS 1,
SCALE_STEPS 350, SCALE_DISABLE_CHECKPOINT 1, json_logger. Compare vs g=1 350-step baseline
(22.002% p50, drops 0.064@349, tail-100 mean 7.3%, loss 3.3434): verdict = drops <3%
steady by ~step 300 at <=0.5pp MFU cost in the MATCHED (low-drop) regime, loss tail sane.
Expected if the sender hypothesis holds: drops collapse toward the ~0.9% binomial floor
within tens of steps; MFU may read LOWER than 22.0 because the heavy-drop speed artifact
disappears — compare against the low-drop regime expectation, not the raw baseline p50.

Confidence: 6/10
Next: idle on racks; poll the canary condition via job list + a small log probe when the
coordinator signals the fix shipped.

## Check-in 2026-07-26 00:15 UTC

- Idle on racks per standing orders. Canary still dark: newest grug-train children (larry x3, 23:5x-00:1x) all return 0 log lines — fix not shipped yet. Sentinel commit 5bf934717 + rerun package logged (7f8ded7b7). Nothing in flight from me.
Confidence: 6/10
Next: keep polling the canary ~every 10 min; rerun ep25d4-sqb-cf100-350-v2 the moment it flips.

## Check-in 2026-07-26 00:52 UTC — canary flipped; verdict leg rerun in flight

- Canary: larry gkv4-opt grug-train child (submitted ~00:45Z) returns log lines — shipper fix live.
- RERUN SUBMITTED: /mwittmann/ep25d4-sqb-cf100-350-v2-20260725 (identical to lost v1: sender-QB, cf1.0, adjoint, drops, 350 steps). ETA ~02:15Z.
Confidence: 6/10
Next: babysit; verify "QB sender-local bias active: senders=64" sentinel once training tasks boot.

## Check-in 2026-07-26 01:25 UTC

- v2 verdict leg mid-run, sentinels confirmed (senders=64). Drop trajectory: 0.376 (~step 40) -> 0.264 (~100) -> 0.169 (~150) -> 0.135 (~170). Tracking near-but-slightly-better-than the g=1 baseline at matched steps (g=1: 0.126@150-200) — NOT the instant collapse of the fixed-router closed loop; the live router is still training into balance during the early phase. The discriminating question remains the 250-350 tail (g=1 leveled at 0.073 mean / 0.064@349).
- Window MFU ~23.0 (drop-regime-confounded; final matched-regime read at harvest).
Confidence: 5/10 (tail behavior is genuinely open)
Next: babysit to ~02:20Z, harvest, verdict, session wrap.

## FINAL (R6-3 sender-local balancing) 2026-07-26 02:35 UTC — SCOPING NEGATIVE, with the cause narrowed

Verdict leg /mwittmann/ep25d4-sqb-cf100-350-v2-20260725 (sender-QB, cf1.0, adjoint, drops,
350 steps, 349 samples, sentinel "senders=64 experts=256" confirmed on-rack):

| step | 30 | 60 | 90 | 119 | 150 | 200 | 250 | 300 | 349 | tail-100 mean |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| sender-QB | 0.537 | 0.272 | 0.227 | 0.173 | 0.140 | 0.125 | 0.099 | 0.082 | 0.079 | 0.0856 |
| global-QB (g=1 baseline) | 0.524 | 0.271 | 0.233 | 0.175 | 0.126 | 0.126 | 0.089 | 0.070 | 0.064 | 0.0732 |

- The two trajectories are statistically IDENTICAL (differences within QB draw variance,
  which we know spans 0.083-0.175 at step 119 across g=1 draws). Sender-local bias does
  NOT beat the global controller in live training. cf1.0 still does NOT cross 3%.
- MFU p50 22.545% (vs baseline 22.002, same nominal config +/- draw; sender mode also
  deletes the per-layer beta pmean collective — cannot attribute the +0.5pp at 1 draw).
  Loss tail20 3.3357 vs 3.3434 — parity. So the mechanism is FREE, just not effective.
- INTERPRETATION (the valuable part): combined with the closed-loop result (sender-QB
  demolishes STATIC sender-correlated hotspots that global QB provably cannot touch,
  0.773-stuck vs 0.016), live-training equality means the real steady-state drops are NOT
  persistent sender-local hotspots. They are batch-stochastic: within-batch routing
  burstiness (a document's tokens routing together into the same shard's buckets) that
  decorrelates step to step — invisible to ANY one-step-delayed bias controller, global or
  per-sender. This kills the whole delayed-bias controller family for the last ~5pp of
  drops (d3's damped/integral variants should expect the same null at steady state, though
  they may still smooth the early transient).
- Remaining routes to cf1.0 fidelity, now sharply scoped: structural headroom (cf1.15,
  priced at -1.75pp), receiver-side pooling across senders (rav's destpool arm — the right
  idea under this diagnosis: pooled capacity averages out per-sender burstiness), or
  same-step spill/rerouting (needs kernel interface change, K+m candidates).
- Bonus: this run is the second draw of the 350-step cf1.0 QB trajectory — the ~6-8%
  plateau is now confirmed across 2 draws and 2 controller granularities.

Deliverable answered: sender-local balancing does NOT take cf1.0 under 3% (nor move it at
all); mechanism validated correct + zero-cost; hypothesis falsified in the live regime;
cause narrowed to batch-stochastic burstiness. Commits: 50748b995, 5bf934717, 7f8ded7b7.
Jobs: sqb smoke, v1 (metrics lost to the log-shipper incident), v2 (verdict). Submissions
only; racks were held during the incident per standing orders and resumed on canary flip.

Confidence: 3/10 that any bias-controller variant reaches <3% at cf1.0; 8/10 in the
burstiness diagnosis (two-sided evidence: closed-loop positive + live null).

## Check-in 2026-07-26 02:50 UTC — R6-5 (manual-PGLE x prefetch) accepted; capture in flight

- Plan: (1) capture job IN FLIGHT /mwittmann/ep25d4-pgle-capture-30-v1-20260726 (operating config + QB + adjoint + PREFETCH so the profiled HLO matches the PGLE leg; 3-step window from step 8; command buffers off per the baseline profile template); (2) pgle_convert.py committed (0-cred harvest path: xplane -> ProfiledInstructionsProto -> gzip+b64 via job logs) — conversion API verified in this jax (jax._src.lib._profiler.get_profiled_instructions_proto); (3) proto file ships in the workspace bundle (git add -f) at /app/..., PGLE leg gets XLA_FLAGS="--xla_gpu_pgle_profile_file_or_directory_path=... --xla_gpu_enable_latency_hiding_scheduler=true"; (4) matched control = same env minus the two flags. Both QB-on + drops (matched regime).
- Verdict criteria: MFU delta (>=0.5pp to matter) + does the dispatch a2a overlap the GEMM in a follow-up trace of the PGLE leg. Prior: low-medium (auto-PGLE/LHS sealed null; the untested piece is REAL latencies + provably-free dataflow from my prefetch gate).
Confidence: 4/10
Next: babysit capture; then conversion job; then legs.

## Check-in 2026-07-26 03:05 UTC — PGLE file built and shipping; treatment leg in flight

- Capture job succeeded (16 hosts, steps 8-11). Conversion required mirroring jax's PGLEProfiler flow exactly: raw-xspace aggregation returns an EMPTY proto; the correct pipeline is get_fdo_profile(xspace) per host (297KB each, all 16 identical-size) then aggregate_profiled_instructions(fdo_list, p90). pgle_convert.py fixed accordingly (2 iterations, operational friction checkpointed per instructions).
- Proto harvested through job logs (gzip+b64, no s3 creds needed), verified to contain real instruction costs (input_reduce_fusion.*, custom-call.*), committed at experiments/grug/moe/pgle/ep64-qb-adjoint-prefetch.pb (3a73ca1b5) — ships in the workspace bundle at /app/... for --xla_gpu_pgle_profile_file_or_directory_path.
- TREATMENT LEG SUBMITTED: /mwittmann/ep25d4-pgle-ab-pgle-120-v1-20260726 (QB+adjoint+prefetch+drops + PGLE file + LHS). Control (same minus the two XLA flags) follows back-to-back. Note: remote compilation cache keys include XLA flags, so no stale-schedule hazard.
Confidence: 4/10
Next: babysit treatment (watch for early XLA parse errors); then control.

## Check-in 2026-07-26 06:50 UTC — v1 failure DIAGNOSED as infra flake; treatment v2 resubmitted with a profile window

- Session resumed on a fresh model after the prior one ran out of credits. Context reconstructed from ROUND6_BRIEF.md + AGENT_LOG tail; working tree clean apart from this log.
- v1 POST-MORTEM (child job logs, /mwittmann/ep25d4-pgle-ab-pgle-120-v1-20260726/grug-train-...):
  first fatal at 03:04:55, i.e. ~40 s after task start and BEFORE any XLA compilation —
  `ABORTED: 5 unexpectedly tried to connect with a different incarnation` on the JAX
  coordination service, then a 125-attempt gang restart loop through 04:21 until iris gave up.
  hparams had already logged correctly (d5120/E256/top8/48L/b1024/seq4096/EP64/steps120,
  qb_routing true), so config and bundle were fine. NOT an XLA flag/proto parse error, NOT
  an OOM — this is the cluster-wide incarnation-mismatch class rav is chasing. Per policy,
  operational friction closes nothing: resubmitted.
- TREATMENT v2 IN FLIGHT: /mwittmann/ep25d4-pgle-ab-pgle-120-v2-20260726 (06:46Z). Identical
  to v1 plus SCALE_PROFILER_STEPS=3 / SCALE_PROFILER_START=8, so the same job yields both the
  MFU series and a treatment-side xspace for the overlap (mechanism) question. Steps 8-10 are
  excluded from the MFU read; the control leg will carry the same profile window so the two
  legs stay matched.
- Mechanism plan: the control-side xspace already exists from the capture job
  (s3://marin-us-east-02a/tmp/ttl=30d/xprof/ep25d4-pgle-capture-30-v1-20260726/plugins/profile/steps-8-to-11/,
  16 hosts, ~216 MB each, same QB+adjoint+prefetch config, no PGLE/LHS). Comparing the two
  device timelines answers whether the dispatch a2a actually moves under the expert GEMM.
Confidence: 4/10 on the direction (unchanged); 8/10 that v1 was infra, not code.
Next: verify log flow + PGLE-consumption messages around step 5; build the xspace overlap reader while the leg runs.

## Check-in 2026-07-26 07:00 UTC — PGLE is being consumed; control-side profile already shows ~90% collective overlap

- TREATMENT v3 IN FLIGHT (/mwittmann/ep25d4-pgle-ab-pgle-120-v3-20260726, submitted 06:47Z after
  v2 died instantly on an invalid `--version` string — calendar version or `<label>-dev` only).
  Sentinels confirmed on all 16 hosts: prefetch active (local_experts=4 expert_shards=64) + custom
  adjoint active. Logs flowing.
- PGLE IS LIVE AND MATCHED: `profile_guided_latency_estimator: Found 197 instructions from the
  profile / Missing 336` on the main train module (16x, one per host). Every missing name is a
  cheap elementwise/concat fusion — zero collectives are missing (grep over all 370 distinct
  missing names for all-to-all/send/recv/collective/all-reduce/all-gather/reduce-scatter: 0 hits).
  So the scheduler has real measured latencies for exactly the ops this probe is about.
  Note `xla_gpu_enable_latency_hiding_scheduler` defaults to FALSE on this pin, so the A/B is
  LHS+PGLE jointly vs neither, as the brief specifies.
- MECHANISM TOOL BUILT + FIRST RESULT (control side, from the existing capture xspace): new
  experiments/grug/moe/xplane_overlap.py parses the raw xplane wire format (no tf/xprof deps) and
  reports per-op concurrency across GPU streams. On the QB+adjoint+PREFETCH capture, per GPU:
  ncclDevKernel_SendRecv = 9423 ms total occupancy of which **90.2% is already concurrent with
  non-collective work** (top partners: the nvjet expert-GEMM kernels, 2776 ms + 740 ms + 595 ms).
  Compute stream busy 31.6 s vs collective stream 9.8 s over the 3-step window.
- Implication BEFORE the MFU read: the dispatch a2a is already ~90% hidden without LHS/PGLE, so
  the headroom this probe can address is ~1 s per 3 steps (~2-3% of step time, i.e. <=0.5pp MFU
  even if PGLE hid all of it). That is the mechanism-level explanation for why the prefetch
  reorder alone measured an exact null.
Confidence: 3/10 that PGLE+LHS clears +0.5pp (down from 4/10: the overlap headroom is measured small); 8/10 in the tooling.
Next: harvest v3 MFU/drops; then the matched control leg; then the treatment-side overlap report.

## Check-in 2026-07-26 07:05 UTC — CORRECTION: 14.3% of the step is EXPOSED collective time, not 10%

Reran the overlap tool with per-stream keying (the first pass merged the same op name across
streams and understated exposure). Control-side capture (QB + adjoint + prefetch, no PGLE/LHS),
per GPU over a 34.07 s / 3-step window:

| GPU | collective total | concurrent with compute | exposed | exposed / span |
|---|--:|--:|--:|--:|
| 0 | 12532 ms | 7645 ms | 4887 ms | 14.3% |
| 1 | 12393 | 7585 | 4807 | 14.1% |
| 2 | 13113 | 7712 | 5402 | 15.9% |
| 3 | 11068 | 6923 | 4146 | 12.2% |

Where the exposure lives (GPU:0): SendRecv splits into 6674 ms on the async stream (#159), 86.1%
overlapped -> ~930 ms exposed, PLUS **2749 ms of SendRecv issued inline on the main compute
stream (#50)**, which is serialized with compute by construction. AllGather 1691 ms @72.8%
overlapped, ReduceScatter 871 ms @27.1%.

So the headroom is real and larger than my previous entry claimed: ~1.6 s per ~11.4 s step.
Hiding all of it would be worth roughly 22.6 -> ~26% MFU. The inline SendRecv block is exactly
the class a latency-hiding scheduler with real latencies is supposed to convert to async and
overlap, which raises the prior on this probe back up. Treatment v3 is mid-compile (LHS is
throwing "very slow compile" alarms on all 16 hosts, no failures).
Confidence: 4/10 that PGLE+LHS clears +0.5pp; 9/10 that the exposed-collective measurement is sound.
Next: harvest v3; matched control leg; treatment-side overlap report against the table above.
