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
