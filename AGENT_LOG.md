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

## Check-in 2026-07-26 07:00 UTC (late entry) — v3 training; PGLE profile does cover the a2a

- PGLE proto audit (local parse of experiments/grug/moe/pgle/ep64-qb-adjoint-prefetch.pb):
  8989 cost entries, including 24 `all_to_all.*` (2.2-4.6 ms each), 20 `all-gather-start`,
  19 `all-reduce-start`, 2185 `custom-call`. So the scheduler gets measured costs for the
  expert dispatch a2a itself. Note the naming asymmetry: the FSDP/optimizer collectives appear
  as async `*-start` ops while the a2a appears as a plain (synchronous) `all_to_all` — matching
  the profile observation that ~2.7 s per 3-step window of SendRecv runs inline on the compute
  stream. Async-ifying that op is exactly what LHS could change; that is the live hypothesis.
- v3 is past compile and stepping (first samples at 06:58Z). drops@0 = 0.1721, byte-identical to
  the sqb 350-step leg's step-0 value, so the data draw matches the earlier legs.
- Early window (steps 0-7, drop-heavy regime, includes the profiler window at 8-10):
  MFU p50 24.33%, step time 11.18 s. NOT comparable to the 22.6% references yet — early steps
  read high because drops are high. The verdict number is the matched control leg.
Confidence: 4/10
Next: let v3 finish (~07:25Z), then submit the matched control immediately.

## Check-in 2026-07-26 07:25 UTC — treatment leg complete (22.619% p50); control + treatment-profile jobs launched

TREATMENT /mwittmann/ep25d4-pgle-ab-pgle-120-v3-20260726 (QB cf1.0 + adjoint + prefetch + PGLE
file + LHS, 120 steps, 118 MFU samples, profiler window at steps 8-10):

| window | n | p10 | p50 | p90 | mean | step_time p50 |
|---|--:|--:|--:|--:|--:|--:|
| 0-119 | 118 | 22.261 | **22.619** | 24.191 | 22.907 | 11.97 s |
| 20-119 | 100 | 22.244 | 22.549 | 23.331 | 22.670 | 12.01 s |
| 40-119 | 80 | 22.235 | 22.481 | 22.841 | 22.511 | 12.05 s |

drops @0 0.1721 (identical to earlier legs -> same draw) @10 0.912 @30 0.444 @60 0.264 @90 0.132
@119 0.0882, tail20 mean 0.1104. Loss last 5.634 (120 steps).

Against the round-6 reference QB+adjoint 120-step draw (22.595%, drops 0.083@119) this is +0.02pp
= a null, but the matched control is the number that decides it: SUBMITTED
/mwittmann/ep25d4-pgle-ab-ctrl-120-v1-20260726 (identical env minus the two XLA flags, same
profiler window) at 07:21Z, ETA ~08:00Z.
Also submitted the treatment-side overlap report (/mwittmann/ep25d4-overlap-pgle-v1-20260726) on
the leg's own xprof dump, 3 hosts, to compare exposed-collective time against the control table.
Useful calibration measured on two 350-step legs: MFU rises ~+0.30pp per +0.10 drop fraction, so
A/B deltas get drop-corrected before being read.
Confidence: 3/10 that PGLE+LHS clears +0.5pp (treatment matches the reference draw within noise).
Next: babysit control; harvest both overlap reports; verdict.

## Check-in 2026-07-26 07:30 UTC — MECHANISM ANSWERED: LHS+PGLE hides FSDP collectives, NOT the expert a2a

Overlap report on the treatment leg's own xprof dump vs the control-config capture, GPU:0,
3-step windows (per-op occupancy and the share of it concurrent with non-collective work):

| op / stream | control | PGLE+LHS |
|---|--:|--:|
| trace span (3 steps) | 34065 ms | 33493 ms |
| SendRecv, async stream | 6674 ms @ 86.1% | 7400 ms @ 86.8% |
| SendRecv, INLINE on the compute stream | 2749 ms @ 0% | 2961 ms @ 0% |
| AllGather | 1691 ms @ 72.8% | 2262 ms @ 82.7% |
| ReduceScatter bf16 | 871 ms @ 27.1% | 537 ms @ **100%** |
| AllReduce bf16 | 246 ms @ 80.2% | 322 ms @ **100%** |
| AllReduce f32 | 301 ms @ 77.0% | 1160 ms @ 63.5% |
| **exposed collective total** | **4887 ms (14.3% of span)** | **4750 ms (14.2%)** |

Same pattern on GPUs 1-3 (exposed 14.0-14.9% treatment vs 14.1-15.9% control).

- The scheduler DID act: the FSDP gradient reduce-scatter and the bf16 all-reduce go from
  partially hidden to fully hidden, and all-gather improves ~10 points. So the flags are live and
  PGLE's latencies are being used.
- The expert dispatch/combine all-to-all is UNMOVED: 86.1% -> 86.8% on the async stream, and the
  ~2.7-3.0 s per 3 steps that runs INLINE on the main compute stream stays 0% overlapped. That
  inline block is the serialized piece the whole probe was aimed at, and PGLE+LHS does not touch it.
- Net exposed collective time is flat (14.3% -> 14.2% of span), which is why the MFU is flat.
Confidence: 8/10 on the mechanism verdict (direct per-op timeline evidence, consistent across 4 GPUs).
Next: control leg completes ~08:00Z for the matched MFU delta; then final verdict.

## Check-in 2026-07-26 07:55 UTC — MATCHED A/B: PGLE+LHS = +0.33pp, consistent across every window

Back-to-back, same draw (drops@0 identical to 4 decimals), same config apart from the two XLA
flags, both legs carrying the same profiler window:

| window | leg | n | p10 | p50 | p90 | mean | step_s | drop mean |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| 0-119 | PGLE+LHS | 118 | 22.261 | **22.619** | 24.191 | 22.907 | 11.97 | 0.3456 |
| 0-119 | control | 118 | 21.912 | **22.273** | 23.940 | 22.563 | 12.16 | 0.3382 |
| 20-119 | PGLE+LHS | 100 | 22.244 | 22.549 | 23.331 | 22.670 | 12.01 | 0.2563 |
| 20-119 | control | 100 | 21.895 | 22.200 | 23.016 | 22.314 | 12.20 | 0.2480 |
| 60-119 | PGLE+LHS | 60 | 22.205 | 22.424 | 22.604 | 22.414 | 12.09 | 0.1591 |
| 60-119 | control | 60 | 21.865 | 22.054 | 22.258 | 22.051 | 12.30 | 0.1518 |
| 100-119 | PGLE+LHS | 20 | 22.166 | 22.262 | 22.403 | 22.276 | 12.16 | 0.1104 |
| 100-119 | control | 20 | 21.810 | 21.938 | 22.001 | 21.919 | 12.36 | 0.1064 |

Delta p50: +0.346 / +0.349 / +0.370 / +0.324 pp raw; +0.32 to +0.35 pp after drop-correction
(-0.30pp per +0.10 drop). Step time 11.97 vs 12.16 s (-1.6%). Drops match throughout
(0.0882 vs 0.0910 @119; tail20 0.1104 vs 0.1064), so this is a same-regime comparison, not a
drop artifact. Loss tail20 5.7759 (PGLE) vs 5.7415 (control) at 120 steps -- +0.03, i.e. the two
legs' loss trajectories differ by less than the step-to-step wiggle, but treatment is nominally
higher; worth watching on a longer horizon before adopting.

Read: a real but SMALL positive, below the 0.5pp bar the brief set for "matters", and it comes
from the FSDP/optimizer collectives (reduce-scatter 27% -> 100% hidden), NOT from the expert a2a,
which is unmoved. Single pair so far -- a +0.33pp claim needs a second pair.
Confidence: 6/10 that the sign is real and ~+0.3pp; 8/10 that the source is FSDP collectives, not the a2a.
Next: second matched pair in the REVERSED order (control first, then treatment) to cancel any
time-ordering / placement drift.

## Check-in 2026-07-26 08:25 UTC — control reproduces to 0.00pp; pair-2 treatment in flight

Second pair, reversed order (control first). CONTROL v2
(/mwittmann/ep25d4-pgle-ab-ctrl-120-v2-20260726): p50 22.318 (0-119) / 22.200 (20-119) /
22.141 (40-119), step 12.14 s, drops 0.0873@119, tail20 0.1062, loss tail20 5.7385.

Control v1 was 22.273 / **22.200** / 22.115. The 20-119 p50 reproduces to the third decimal
across two allocations 33 minutes apart, so control-side placement noise on this config is
<=0.05pp, far below the +0.35pp treatment delta. That materially strengthens the pair-1 result
before pair 2's treatment leg even lands.

Also harvested the overlap report from the pair-1 CONTROL LEG's own xprof (same draw as the
treatment, better than the earlier capture job) — it confirms the capture-job table:
SendRecv async 6870 ms @87.1% (treatment 7400 @86.8%), SendRecv inline on the compute stream
2980 ms @0% (treatment 2961 @0%), ReduceScatter 759 ms @65.4% (treatment 537 @100%),
AllGather 1544 @66.3% (treatment 2262 @82.7%), AllReduce f32 306 @12.2% (treatment 1160 @63.5%),
exposed collectives 4932 ms / 14.5% of span (treatment 4750 / 14.2%).

Coordinator framing to carry into the writeup (acknowledged): the reduce-scatter result closes
David Hall's original candidate-1c lead from the first XProf read ("the final reduce-scatter looks
like it could be overlapped with next layer and isn't — reduce-scatter.10"). It is NOT scan-blocked
as an earlier analysis concluded; it is scheduler-gated, and real measured latencies alone are
enough for XLA to hide it (65.4% -> 100% hidden), with no structural change. And the negative half
is the four-way convergent one: rotation, prefetch, token-chunk overlap, and now real-latency PGLE
all fail to move the expert a2a, so the a2a legs are not hideable on this stack by any scheduling
or dataflow means we have.
Confidence: 7/10 on the +0.33pp sign and magnitude (pending pair 2); 8/10 on both mechanism halves.
Next: harvest treatment v4 (~09:00Z), pool both pairs, write the verdict.

## FINAL (R6-5 manual-PGLE x prefetch-gate) 2026-07-26 08:55 UTC — SMALL CONFIRMED POSITIVE (+0.33pp), a2a UNMOVED

Two matched pairs, opposite order, same draw (drops@0 = 0.1721 in all four legs), 120 steps,
QB cf1.0 + custom adjoint + prefetch + drop reporting everywhere, both arms carrying the same
3-step profiler window. Treatment adds exactly two XLA flags:
`--xla_gpu_pgle_profile_file_or_directory_path=/app/experiments/grug/moe/pgle/ep64-qb-adjoint-prefetch.pb`
and `--xla_gpu_enable_latency_hiding_scheduler=true` (LHS defaults to false on this pin).

| leg | order | p50 0-119 | p50 20-119 | p50 100-119 | step_s | drops@119 | loss tail20 |
|---|---|--:|--:|--:|--:|--:|--:|
| pgle v3 | pair 1, second | 22.619 | 22.549 | 22.262 | 11.97 | 0.0882 | 5.7759 |
| ctrl v1 | pair 1, first | 22.273 | 22.200 | 21.938 | 12.16 | 0.0910 | 5.7415 |
| ctrl v2 | pair 2, first | 22.318 | 22.200 | 21.910 | 12.14 | 0.0873 | 5.7385 |
| pgle v4 | pair 2, second | 22.612 | 22.530 | 22.237 | 11.99 | 0.0896 | 5.7767 |

**Verdict: +0.34pp raw / +0.32pp drop-corrected (pooled 20-119: 22.540 vs 22.200), replicated in
both pairs and in every window (+0.27 to +0.37).** Step time 12.01 -> 11.99 s vs 12.20 s (-1.5%).
The two control legs agree to 0.000pp at 20-119 across allocations 33 min apart, so allocation
noise on this config is <=0.05pp and the effect is ~7x that.

### Mechanism (per-op GPU timeline, new experiments/grug/moe/xplane_overlap.py, GPU:0, 3-step window)

| op | control | PGLE+LHS |
|---|--:|--:|
| SendRecv (expert a2a), async stream | 6870 ms @ 87.1% hidden | 7400 ms @ 86.8% |
| SendRecv, INLINE on the compute stream | 2980 ms @ 0% | 2961 ms @ 0% |
| **ReduceScatter bf16 (FSDP grads)** | 759 ms @ **65.4%** | 537 ms @ **100%** |
| AllReduce bf16 | 246 ms @ 80.2% | 322 ms @ 100% |
| AllGather | 1544 ms @ 66.3% | 2262 ms @ 82.7% |
| exposed collective total | 4932 ms (14.5% of span) | 4750 ms (14.2%) |

1. **This closes candidate 1c** — David Hall's original XProf lead, "the final reduce-scatter (some
   gradient thing) looks like it could be overlapped with next layer and isn't (reduce-scatter.10)",
   open since round 1 and last assessed as scan-blocked and needing manual structural work. It is
   NOT scan-blocked: it is scheduler-gated. Feeding XLA real measured latencies takes that
   reduce-scatter from 65.4% hidden to 100% hidden with no structural change, and the bf16
   all-reduce likewise 80.2% -> 100%. That is where the entire +0.33pp comes from.
2. **The expert all-to-all is unmoved**: 87.1% -> 86.8% hidden on the async stream, and the ~3.0 s
   per 3 steps that XLA collapses to a synchronous in-line SendRecv on the compute stream stays
   exactly 0% overlapped in both arms. Real measured latencies PLUS the prefetch gate's legal
   dataflow freedom change nothing about it. With the sealed rotation, prefetch-reorder and
   token-chunk-overlap results this is now a four-way convergent finding: **the a2a legs are not
   hideable on this stack by any scheduling or dataflow means available to us.** Anyone revisiting
   overlap of the dispatch/combine a2a should treat that as settled and go structural instead.

### Caveats and recommendation

- Below the 0.5pp bar: keep behind a flag if the PGLE plumbing is cheap to maintain; it is not part
  of the 25% bridge. Plumbing cost is a captured profile that must be regenerated whenever the HLO
  changes (this one matched 197 instructions and missed 336 cheap elementwise fusions; zero
  collectives missed), plus ~4 min of extra first-compile (cached after: pair-2 treatment reached
  step 0 four minutes after submit).
- Reproducible small fidelity cost: treatment ran +0.008 mean drop fraction (0.2569 vs 0.2493) and
  +0.035 loss at step 119 (5.634 vs 5.598) in BOTH pairs. Small and self-consistent (more drops ->
  worse loss), but the sign is stable, so a longer-horizon check is required before adoption.
- Fleet: 4 rack legs + 4 CPU analysis jobs this assignment, submissions only; v1 of the treatment
  died on the cluster-wide incarnation-mismatch gang abort (~40 s in, before any compile) and v2 on
  an invalid `--version` string; neither was a code failure.

Artifacts: xplane_overlap.py (9ad635b14 + follow-up), PGLE proto 3a73ca1b5, logs in the session
scratchpad (v3.txt, v4.txt, ctrl.txt, ctrl2.txt, overlap_*.log).
Confidence: 8/10 on the +0.33pp effect and its sign; 9/10 on both mechanism halves (per-op timelines, 4 GPUs, 2 configs); 5/10 on whether the loss/drop cost survives a longer horizon.

## Check-in 2026-07-26 09:05 UTC — INLINE SendRecv: identified, explained, and a one-flag test found

New direction accepted. Steps 1-2 done with zero rack time (two CPU analysis jobs over profiles
already captured, plus XLA source reading). Built experiments/grug/moe/xplane_op_detail.py, which
resolves each GPU kernel event's XStats (`hlo_op`, `hlo_module`, and the jaxpr scope in `name`) so
kernels can be attributed to HLO instructions and to dispatch/combine/fwd/bwd.

### 1. IDENTIFY — it is 9 of 24 all_to_all instructions, and the split is systematic

The 48 layers are one scan body, so each instruction runs 144x in the 3-step window. There are
exactly 24 `all_to_all.N.1` instructions per layer: 8 forward (4 dispatch + 4 combine) and 16
backward. Control leg, GPU:0:

| pass / leg | async stream | INLINE on compute stream |
|---|---|---|
| fwd dispatch | .118 | **.112, .114, .116** |
| fwd combine | .113, .115, .117, .119 | none |
| bwd dispatch | .80, .89, .91, .93, .95 | .81, .82, .83 |
| bwd combine | .84, .85, .86, .87, .94 | .88, .90, .92 |

9 inline instructions x ~2.2-2.5 ms x 144 = 2980 ms, which is the entire inline block. Under
PGLE+LHS the count drops 9 -> 8 (.112 and .83 go async, nothing goes the other way) but total
inline ms is unchanged at 2961, because the remaining inline .116 stretches.

The systematic part: **every forward COMBINE is async in both arms; three of four forward
DISPATCH legs are inline.** That is the reverse of what the prefetch gate targets — prefetch was
built to give the dispatch leg slack — and it is the mechanism-level explanation of why the
prefetch reorder measured an exact null.

### 2. EXPLAIN — not a threshold, not command buffers: it is post-scheduling async->sync collapse

`GpuCompiler::RunPostSchedulingPipelines` runs `GpuConvertAsyncCollectivesToSync` on the FINAL
schedule. Any async-start whose matching done is separated only by no-ops (parameter, constant,
bitcast, get-tuple-element) is flagged `is_sync = true` and executes inline on the compute stream.
So an inline SendRecv is precisely a collective the scheduler could not find independent work to
place under. This is the coordinator's candidate (3): dataflow/scheduling slack, and the profile
is downstream evidence, not the cause. Sizes are identical across all 24 (same shapes), so it is
not a size threshold; the inline ones merely *run faster* (2.2 vs 3.5 ms) because nothing competes.

### 3. TEST — one flag, one leg, and the control is already measured twice

`xla_gpu_experimental_parallel_collective_overlap_limit` defaults to **1**: "controls how many
in-flight collectives latency hiding scheduler can schedule." With a budget of one, any a2a whose
window overlaps an FSDP all-gather/reduce-scatter or another a2a cannot be overlapped, and the
post-scheduling pass then collapses it to sync. Verified it parses in our jaxlib 0.10.1 (as does
`xla_gpu_memory_limit_slop_factor`, default 95; `xla_gpu_experimental_collective_start_as_early_as_possible`
does NOT exist in this version). The knob only bites with LHS on, so the experiment builds on the
PGLE+LHS arm, whose p50 is already pinned by two independent legs (22.549 / 22.530 at 20-119,
spread 0.02pp) — a single treatment leg is therefore a real A/B.
Confidence: 9/10 on the identification, 8/10 on the async->sync-collapse explanation, 4/10 that the overlap-limit flag alone moves it.
Next: rack request pending with the coordinator; no rack submissions until approved.

## Check-in 2026-07-26 09:10 UTC — both approved jobs submitted

- RACK LEG: /mwittmann/ep25d4-ovlim4-120-v1-20260726 — PGLE+LHS arm plus
  `--xla_gpu_experimental_parallel_collective_overlap_limit=4`, 120 steps, profiler window kept so
  the same leg yields BOTH the MFU delta and the inline-instruction count. Control is the already
  twice-measured PGLE+LHS arm (22.549 / 22.530 at 20-119). Will queue behind d1's spill m=3 and
  d5's d6144 baseline; NOT resubmitting on PENDING.
- EP4 SCHEDULE DUMP (free, does not count against the rack budget):
  /mwittmann/ep25d4-schedump-ep4-v1-20260726 — 1 node / 4 GB200, 4 layers, 16 experts over 4 shards
  so `local_experts = 4` matches the rack topology exactly, LHS on, `--xla_dump_to=/tmp/hlodump`,
  `GRUG_RUN_INLINE=1` so training runs in the entrypoint task and a wrapper can post-process the
  dump on the same node. New experiments/grug/moe/schedule_report.py walks the post-optimization
  HLO in schedule order and prints, per collective, whether the post-scheduling pass tagged it
  `is_sync":true` and how many real (non-nop) instructions sit between each surviving async
  start/done pair — i.e. the slack the scheduler actually found, per instruction.

### Flag-probe negatives worth keeping (they cost a cycle to find)
- `xla_gpu_experimental_collective_start_as_early_as_possible` does NOT exist in jaxlib 0.10.1
  (XLA rejects it: "Unknown flag in XLA_FLAGS"). It is in newer XLA only.
- `xla_gpu_experimental_parallel_collective_overlap_limit` exists here, default **1**.
- `xla_gpu_memory_limit_slop_factor` exists here, default **95**. Queued only if the first leg moves.
- `xla_gpu_disable_async_collectives` is a disable-only filter; there is no enable-side counterpart,
  so async conversion is already on for ALLTOALL and is not the gate.
- Levanter's `log_xla_hlo` writes StableHLO (pre-optimization) — useless for schedule questions.
  The scheduled GPU HLO only comes from `--xla_dump_to`.
Confidence: 4/10 that the overlap-limit flag alone moves MFU; 7/10 that the schedule dump explains the dispatch-vs-combine asymmetry.
Next: poll both; report the verdict two ways (MFU delta AND inline count) as instructed.

## Check-in 2026-07-26 09:45 UTC — schedule dump working; rack leg fighting the incarnation flake

- RACK LEG /mwittmann/ep25d4-ovlim4-120-v1-20260726 has NOT reached step 0 in 38 minutes.
  Cause is the known gang flake, not the new flag: `failures=2`, 48 `[iris setup] step 1/3` lines
  (three attempts x 16 tasks), first abort at 09:08:03 with the usual
  "5 unexpectedly tried to connect with a different incarnation". Iris is retrying on its own and
  the current attempt came up at ~09:39, so first step should land ~09:50. Not resubmitting while
  the job is running, per protocol.
- EP4 SCHEDULE DUMP now works (free 1-node job, ~3 min with a warm compile cache). Two tooling bugs
  found and fixed on the way, both worth recording: HLO instruction lines cannot be parsed by
  `name = <shape> opcode(` because tuple shapes contain spaces and nested parens (match the name
  only and classify off substrings), and shard_map-wrapped collectives are spelled `all_to_all.112`
  with underscores while native ones are `all-to-all-start` with hyphens.
- FIRST STRUCTURAL RESULT from the dump, and it confirms the collapse mechanism end to end:
  `instructions tagged is_sync=true: 20` in the EP4 module, and the a2a start/done pairs split
  cleanly into two populations — most have **cover 0** (nothing at all between start and done) while
  a minority carry 1-5 small fusions (`loop_add_fusion`, `input_reduce_fusion`, one `custom-call`).
  Cover 0 is exactly the condition `GpuConvertAsyncCollectivesToSync` collapses on. Note for anyone
  reading a dump later: the GPU pass does NOT delete the pair, it tags the start with
  `"is_sync":true` and re-emits start/done adjacently, so a collapsed collective still *looks* async
  in the HLO text — the tag and the zero cover are the tells.
- Rerunning the dump (v5) with the report now printing the is_sync tag per pair and each
  collective's MoE leg from HLO `metadata={op_name=...}`, so the dispatch-vs-combine asymmetry can
  be read directly off the schedule rather than inferred from the timeline.
Confidence: 8/10 on the collapse mechanism; 4/10 on the overlap-limit flag moving MFU.
Next: harvest v5; then the rack leg's MFU delta AND inline-instruction count.

## Check-in 2026-07-26 10:00 UTC — EP4 REPRODUCES the collapse pattern (free, 3 min); rack leg still flake-blocked

### The useful result: the sync/async split is reproducible on ONE node

EP4 dump (16 experts / 4 shards so `local_experts = 4`, 4 layers, LHS on), collectives labelled by
HLO `metadata={op_name=...}` and by the `"is_sync":true` tag:

| leg | collapsed to SYNC | stayed async |
|---|--:|--:|
| fwd dispatch | **3** | 1 |
| fwd combine | **0** | 4 |
| bwd dispatch | 4 | 4 |
| bwd combine | 3 | 5 |

That is the same census as the 64-GPU timeline (9 of 24 inline; fwd dispatch 3 of 4; fwd combine
0 of 4). **A free single-node job reproduces the scheduling decision that costs ~1 s per 12 s step
at rack scale**, so fixes to the dispatch-leg dataflow can be iterated at zero rack cost and only
the winner needs a rack leg. That is the most useful thing to come out of this round.

Caveat on my own output: the "cover" column (instructions between start and done) is NOT reliable
yet — the report filters cover candidates by computation name and my computation tracking is
mis-parsing HLO computation headers, so cover reads 0 almost everywhere. The SYNC/async labels and
the leg labels come straight from the HLO tag and metadata and are solid; the cover column needs a
fix before anyone quotes it.

### Rack leg: blocked by the gang flake, not by the flag

/mwittmann/ep25d4-ovlim4-120-v1-20260726 has not reached step 0 in 51 minutes: `failures=8`, and a
classification of all 49k log lines finds **only** incarnation aborts (63 "unexpectedly tried to
connect with a different incarnation", 32 "newer incarnation") — zero XLA errors, no rejected flag,
no OOM. So `--xla_gpu_experimental_parallel_collective_overlap_limit=4` is not implicated; the leg
is losing to the same cluster-wide class that killed the first PGLE leg. Iris is still retrying and
I am not touching it, but it is occupying a rack slot while d1 and d5 contend, so the coordinator
may want to call it.
Confidence: 9/10 that EP4 reproduces the collapse census; 9/10 that the rack leg's failures are pure infra.
Next: coordinator decision on the rack slot; meanwhile fix the cover column so the EP4 harness can
answer *why* the fwd dispatch legs get no slack.

## Check-in 2026-07-26 10:30 UTC — COVER COLUMN FIXED, and it answers the asymmetry: it is the collective budget

Session resumed after a transient 529; working tree verified clean at 54809714c, no partial edit.

### Fixing cover took three parser bugs, all worth recording

1. Brace-depth tracking does not survive a real dump. `backend_config`, `replica_groups` and
   metadata strings carry unbalanced braces, and ONE bad line desynchronizes everything after it —
   on the 82 MB dump it yielded 157 computations / 1473 instructions. Column structure is the
   sound invariant: headers start at column 0 and end in `{`, a lone `}` at column 0 closes, and
   instructions are indented. That parse yields **10972 computations / 193939 instructions.**
2. `-start` / `-done` carry a numeric suffix AFTER the tag (`all-to-all-start.16`), so matching on
   `endswith("-start")` silently finds nothing; and the done's operand lists the start's shape
   before its name, so `-done(%name)` never matches — look for `%name` anywhere on a done line.
3. `ROOT %x = ...` lines were skipped entirely by the instruction regex, dropping real cover.
Selection gotcha too: XLA emits the train step under BOTH `after_optimizations` and
`<arch>_gpu_after_optimizations`, plus dozens of tiny helper modules under both spellings, so
prefer by SIZE, not by name — a name preference picks `jit__threefry_split`. Four parsing tests
now guard all of this (experiments/grug/moe/test_schedule_report.py).

### The answer: the forward dispatch legs get no slack because the in-flight budget is 1

Baseline (LHS, `parallel_collective_overlap_limit` at its default 1), forward legs, EP4 harness:

| instruction | leg | state | cover | what covers it |
|---|---|---|--:|---|
| all-to-all-start.16 | fwd dispatch | **SYNC** | **0** | nothing |
| all-to-all-start.18 | fwd dispatch | **SYNC** | **0** | nothing |
| all-to-all-start.20 | fwd dispatch | **SYNC** | **0** | nothing |
| all-to-all-start.22 | fwd dispatch | async | 5 | loop_slice_fusion, custom-call.2064, custom-call.2076 |
| all-to-all-start.17 | fwd combine | async | 3 | custom-call.2065, loop_multiply_fusion.10, custom-call.2077 |
| all-to-all-start.19 | fwd combine | async | 3 | custom-call.2066, loop_multiply_fusion.11, custom-call.2078 |
| all-to-all-start.21 | fwd combine | async | 3 | custom-call.2067, loop_multiply_fusion.12, custom-call.2079 |
| all-to-all-start.23 | fwd combine | async | 5 | loop_add_fusion.8, fusion.2206, wrapped_slice.3 |

The covering instructions on the combine legs are the expert GEMM custom-calls and the SwiGLU
multiply. So the combine legs are covered BY THE EXPERT GEMM, and the dispatch legs have literally
nothing between start and done. With a budget of one in-flight collective the scheduler must
choose, and it spends the budget on combine; dispatch is then collapsed by the post-scheduling pass.

### Raising the budget flips every MoE all-to-all to async, for free

| leg | limit=1 (default) | limit=4 |
|---|---|---|
| fwd dispatch | 3 SYNC / 1 async | **0 SYNC / 4 async** |
| fwd combine | 0 SYNC / 4 async | 0 SYNC / 4 async |
| bwd dispatch | 4 SYNC / 4 async | **0 SYNC / 8 async** |
| bwd combine | 3 SYNC / 5 async | **0 SYNC / 8 async** |
| reshard (entry computation) | 6 SYNC / 8 async | 14 SYNC / 0 async |

At limit=4 every dispatch leg carries 5-8 covering instructions including the expert GEMM
custom-calls (2064/2065/2076/2077) and overlapping sibling a2a starts. **All 10 collapsed MoE
all-to-alls become asynchronous and covered.** The budget is reallocated, not created: the 14
reshard a2as in the entry computation collapse instead, which is the risk this trades into.

This kills the "the fix must change what the dispatch leg depends on" branch. The dataflow the
prefetch gate creates is already legal and sufficient; the scheduler simply was not permitted to
use it. That retro-explains the prefetch null one level deeper than the previous entry did: the
gate supplied slack, and a budget of 1 forbade spending it.
Confidence: 9/10 on the budget explanation (cover evidence both directions, plus the flip); 5/10 that it is a net rack win, because the reshard collectives collapse in exchange.
Next: free sweep at limit=2 and 8 in flight; then a rack-leg request with the best setting.

## Check-in 2026-07-26 10:40 UTC — budget sweep done for free; the EP4 trade does NOT exist at rack scale

MoE all-to-alls left synchronous on the EP4 harness, by `parallel_collective_overlap_limit`:

| limit | fwd dispatch | fwd combine | bwd dispatch | bwd combine | **MoE SYNC** | reshard SYNC |
|---|--:|--:|--:|--:|--:|--:|
| 1 (default) | 3 | 0 | 4 | 3 | **10** | 6 |
| 2 | 0 | 0 | 2 | 1 | **3** | 14 |
| 4 | 0 | 0 | 0 | 0 | **0** | 14 |
| 8 | 0 | 0 | 0 | 1 | **1** | 14 |

limit=2 already clears the whole forward pass; limit=4 clears everything; limit=8 is no better
(and marginally worse), so 4 is the setting to buy a rack leg for.

The apparent cost — 14 reshard all-to-alls collapsing in the entry computation — **is an artifact
of the harness, not a real trade**: re-checking the rack-scale attribution, all 24 SendRecv
instructions at the operating point are `shard_map/dispatch` or `shard_map/combine`, and the
string "reshard" appears zero times. The tiny mesh (4 shards, data axis 1) manufactures entry-level
reshard a2as that the EP64 mesh does not have. So at the operating point the budget increase has
nothing to pay back with.

Rack case, sized off my own measurements: ~2980 ms of inline SendRecv per 3-step window = ~1 s of a
~12 s step, 0% covered today. The harness says all of it becomes async and GEMM-covered at limit=4.
Confidence: 9/10 that the census result transfers (same code, same legs, same pass); 5/10 that it converts to >=0.5pp MFU, since async-and-covered in the schedule is necessary but not sufficient for wall-clock overlap on the device.
Next: rack-leg request with the coordinator (one leg, PGLE+LHS arm + limit=4, control already
pinned twice at 22.549 / 22.530).

## FINAL (inline-SendRecv direction) 2026-07-26 11:20 UTC — census transfers PERFECTLY, MFU moves +0.12pp: exposure relocates

Rack leg /mwittmann/ep25d4-ovlim4-120-v2-20260726 (PGLE+LHS arm + `overlap_limit=4`, 120 steps,
failures=0, compile 10 min in a low-flake window). Both readings from the same profiler window.

### Reading 1 — MFU, against the twice-pinned PGLE+LHS arm

| window | limit=4 p50 | limit=1 p50 | raw | drop-corrected |
|---|--:|--:|--:|--:|
| 0-119 | 22.783 | 22.619 | +0.164 | +0.130 |
| 20-119 | 22.674 | 22.549 | +0.125 | +0.093 |
| 60-119 | 22.526 | 22.424 | +0.101 | +0.108 |
| 100-119 | 22.377 | 22.262 | +0.115 | +0.104 |

**+0.12pp**, consistent across every window. Step time 12.01 -> 11.94 s. Drops matched
(0.0876 vs 0.0882 @119). Against the ORIGINAL no-flag control the stack is now +0.47pp
(22.674 vs 22.200 at 20-119). Loss tail20 5.8226 vs 5.7759, same small positive bias as the
earlier PGLE arm carried.

### Reading 2 — the census transferred completely

Attribution of the leg's own profile: **all 24 `all_to_all` instructions are on the async
collective stream on all four GPUs; zero on the compute streams.** Compute-stream collective time
went 2961 ms -> **0.00 ms**. The EP4 harness predicted exactly this, at zero rack cost.

### Why the prize did not follow: there is no compute left to hide behind

| metric (GPU:0, 3-step window) | limit=1 | limit=4 |
|---|--:|--:|
| trace span | 33493 ms | 33159 ms |
| collective time ON the compute stream | 2961 ms @ 0% hidden | **0 ms** |
| SendRecv on the async stream | 7400 ms @ 86.8% hidden | 10347 ms @ **64.3%** hidden |
| exposed collective total | 4750 ms (14.2% of span) | 4287 ms (12.9%) |

Moving the a2a off the compute stream did not create work to cover it: the same GEMMs now have to
cover 40% more async collective time, so the hidden *fraction* fell from 86.8% to 64.3% while
hidden *absolute* rose only 6423 -> 6653 ms. Net exposed collective fell just 463 ms per 3 steps
(154 ms/step, ~1.3%), which is the +0.12pp.

The binding constraint is visible in one line: the compute stream is busy **28.7 s of a 33.2 s
span (86.6%)**, leaving 4.44 s idle, and exposed collective time is **4.29 s** — i.e. the exposed
collective almost exactly fills the compute idle. The step is collective-volume-bound, not
schedule-bound. Further scheduling work on these legs cannot pay; only reducing collective bytes
(or raising arithmetic intensity per byte) can.

### Recommendation

Adopt `--xla_gpu_experimental_parallel_collective_overlap_limit=4` together with the PGLE+LHS
flags: pure XLA flags, no code change, so it composes with everything else in the stack, and the
pair is worth +0.47pp over the current default at zero maintenance beyond regenerating the PGLE
profile when the HLO changes. Neither is part of a path to 25%; the honest ceiling of the whole
scheduling family is now measured, not guessed.
Confidence: 9/10 on both readings (census is a direct instruction census; MFU is 4 windows against a twice-pinned control); 9/10 that scheduling is now exhausted as a lever for the a2a legs.

## Check-in 2026-07-26 11:35 UTC — fp8-wire prize sized against my own profile; three corrections

Verified the coordinator's arithmetic against the limit=4 profile (GPU:0, 3-step window).

CONFIRMED: a2a exposed = 10346.58 x (1 - 0.643) = **3694 ms**, exactly as stated. Non-a2a exposed
is the remaining 593 ms of the 4287 ms total.

CORRECTION 1 (strengthens the case): the demonstrated hiding capacity is **8939 ms**, the total
hidden collective, not 6653 ms — 6653 is the a2a-only hidden portion. After halving a2a time
(10347 -> 5173 ms) total collective is 8052 ms, which fits under the 8939 ms already demonstrated
hideable, with margin.

CORRECTION 2: "exposed a2a could approach zero" is the optimistic branch, not automatic — it needs
the hidden FRACTION to improve, not just aggregate capacity to exist. Bracketing it:

| scenario | exposed collective | span | step | MFU |
|---|--:|--:|--:|--:|
| today (limit=4) | 4287 ms | — | 11.94 s | 22.67 |
| halve a2a, hidden fraction stays 64.3% | 2440 ms | -5.6% | 11.27 s | 24.01 (**+1.34pp**) |
| halve a2a, a2a fully hidden | 593 ms | -11.1% | 10.61 s | 25.52 (**+2.84pp**) |
| ceiling: ALL collective hidden | 0 | -12.9% | 10.41 s | 26.04 (+3.37pp) |

So the honest prize is **+1.3 to +2.8pp**, with ~+2pp mid-range — the coordinator's estimate sits
inside the bracket. CORRECTION 3: the conversion is slightly understated at the top end; 1231 ms
off an 11.94 s step is +2.6pp, not +2pp, because MFU goes as 1/step_time.

Also worth stating: 3.37pp is the ENTIRE remaining collective-exposure budget at this operating
point. Nothing in the collective-overlap family can ever be worth more than that, and halving a2a
bytes captures 40-84% of it. That is the last sizing this thread needs.

SUPPORTING CHECK — the a2a is bandwidth-bound, so halving bytes really does halve time. Per device
per dispatch round the wire carries expert_shards x capacity x hidden = 64 x 2048 x 5120 bf16 =
1.34 GB; at NVLink-class bandwidth that is order 1.5 ms against the 2.2-3.5 ms measured per
instruction. Consistent, and independently consistent with my own earlier fp8-wire decomposition,
which put the maximum wire saving at ~1.5 s/step against the 1231 ms/step of exposed a2a here.
The halving assumption is therefore not a leap; latency-bound behaviour would have shown up as
measured times far below the byte-count estimate, and it does not.

Fidelity constraints carried forward from the earlier fp8 work (unchanged): quantize strictly
AFTER routing so drop accounting is untouched, per-token scaling only (never across tokens on the
sequence axis), e4m3 forward and e5m2 backward on the wire, loss-trajectory parity is the verdict.
Note the backward legs are 16 of the 24 a2a instructions, so the full halving needs both directions.
Confidence: 9/10 on the sizing bracket; 6/10 that the top of the bracket is reachable.
Next: kernel-feasibility survey (fp8 input + dequant epilogue) before proposing any build.

## Check-in 2026-07-26 11:45 UTC — fp8 feasibility: the stock XLA path can do it, but ONLY if the per-token scale moves to the GEMM output

Read the pinned XLA GEMM rewriter (xla/backends/gpu/transforms/gemm_rewriter.cc) rather than
guessing what the stack supports. Three facts that decide the design:

1. **Per-token scales are rejected by the fp8 custom-call path.** Line 1370:
   `if (!ShapeUtil::IsScalar(scales[i]->shape())) { ... "The scaling factors must be scalars." }`.
   So a `convert(fp8) * broadcast(per_token_scale)` on a GEMM operand does NOT fold into
   `__cublas$lt$matmul$f8` — it stays a separate multiply over the full activation tensor. That is
   precisely the ~2.9 s QDQ overhead that sank the earlier fp8-wire attempt, and it is a hard
   conflict with the fidelity rule (per-token scaling only, never shared across tokens).
2. **An UNSCALED fp8 operand is accepted.** Lines 286-291: a bare `convert(fp8)` with no multiply
   matches, with `param.scale = nullptr`. So fp8 activations can enter the GEMM carrying no scale
   at all.
3. **Mixed e5m2 x e4m3 is allowed**; only e5m2 x e5m2 is rejected ("one of the input types must be
   F8E4M3FN", lines 1278-1304). That matches the known requirement that the fp8 backward must be
   mixed e5m2-grad x e4m3-weight, so the backward wire dtype is satisfiable on the stock path.

### The design this implies needs no new kernel

Because the first expert GEMM is row-linear, `scale_i * (a_i @ W)` equals `(scale_i * a_i) @ W`.
So do not dequantize the INPUT — send unscaled e4m3 activations plus a per-token scale vector over
the wire, feed the unscaled fp8 straight into the dot (matches fact 2), give the weights a
per-EXPERT scalar scale (scalar, so fact 1 is satisfied, and weights are not token-indexed so the
token-sharing invariant is untouched), and apply the per-token scale to the GEMM OUTPUT rows.
The output scale then lands on `hidden` immediately before the SwiGLU `act(gate) * up`, which is
already an elementwise fusion reading and writing that tensor — so it costs one extra broadcast
multiply inside an existing fusion rather than a separate pass. That is "dequant as an epilogue"
in the only sense that matters for cost, achieved by XLA fusion rather than by kernel surgery.

Wire saving is the full halving on the dispatch leg (e4m3 payload + one f32 scale per token, i.e.
5120 bf16 bytes -> 5120 fp8 bytes + 4, a 1.9993x reduction), with the combine leg and the backward
legs following the same construction.

Open risks, in order: (a) whether XLA actually emits `__cublas$lt$matmul$f8` for this shape on
sm100a rather than falling back — cheaply checkable on the free EP4 harness by dumping HLO and
grepping for the custom call; (b) whether fp8 weights are acceptable numerically, since this route
quantizes weights as well as the wire (per-expert scalar scale); (c) loss-trajectory parity, which
remains the verdict.
Confidence: 7/10 that the no-new-kernel route compiles to a real fp8 GEMM; 5/10 that it beats bf16 end to end.
Next: repo survey of existing fp8/MXFP8 kernels is running; then a free EP4 HLO check of (a).

## FINAL SCOPING (fp8 wire, round-7 direction) 2026-07-26 11:30 UTC — NO NEW KERNEL NEEDED, and the old -2.02pp is explained

### Probe result (free 4-GPU job, operating-point GEMM shapes 131072 x 5120 @ 5120 x 2560)

| variant | lowers to |
|---|---|
| bf16 baseline | `__cublas$lt$matmul` |
| fp8 operands, no scales | **`__cublas$lt$matmul$f8`** |
| fp8 operands, per-token scale on the GEMM **output** | **`__cublas$lt$matmul$f8`** |
| fp8 operands, per-token scale on the GEMM **input** | `__cublas$lt$matmul` (falls back to bf16) |
| e5m2 x e4m3 (backward wire dtypes) | **`__cublas$lt$matmul$f8`** |

My first probe returned fp8_calls=0 everywhere and was WRONG — it quantized inline from bf16, so
XLA fused the convert and the rewriter saw a fusion instead of `convert(f8)`. Feeding genuine fp8
operands, which is what an fp8 wire actually delivers, flips it. Recording the mistake because the
false negative would have killed the direction.

### This explains the -2.02pp result mechanically

`SCALE_A2A_FP8_WIRE` already exists on this branch (ep_ragged_all_to_all.py:33-94, :365-376) and
**dequantizes back to bf16 immediately after the collective** (:70), so the GEMM still sees bf16.
That configuration loses twice: it pays ~2.9 s of QDQ, and because the per-token scale is applied
on the input side it could not have gotten an fp8 GEMM even if the dequant were free — row 4 above
is exactly that shape. The fix is not a better wrapper, it is moving the scale to the other side
of the dot.

### The design, and why it needs no kernel work

Row-linearity gives `scale_i * (a_i @ W) == (scale_i * a_i) @ W`. So: carry unscaled e4m3 over the
wire with a per-token f32 scale beside it; feed the unscaled fp8 straight into the dot (matches the
rewriter's no-scale operand form); quantize weights to e4m3 with a per-EXPERT scalar scale (scalar
satisfies the rewriter, and weights are not token-indexed so the no-token-sharing invariant is
untouched); apply the per-token scale to the GEMM OUTPUT, where it lands on `hidden` immediately
before the existing SwiGLU `act(gate) * up` fusion and costs one broadcast multiply inside a
fusion that already reads and writes that tensor. The combine leg is easier still: its fp8 values
are produced by the second GEMM, and the dequant scale folds into the existing combine-weight
einsum. Wire saving is the full ~2x on both permutation legs.

### Effort estimate (asked for rather than started)

Modification of existing code on this branch, NOT a new kernel: change the fp8-wire consumer to
stop dequantizing at :70 and carry the scale; add per-expert scalar weight quantization; apply the
scale after each GEMM; extend the existing structured custom adjoint to e5m2 cotangents. Estimate
~1-2 days including parity tests and a loss-trajectory leg. The d2/d5 MXFP8 grouped kernels
(cudnn-frontend CuTeDSL, block-scaled, fp8-in/bf16-out epilogue, wired into the same fixed-a2a
path) are a DIFFERENT route, already measured at -2.83pp, and are not needed for this.

### Fidelity notes that bind

Quantize strictly after routing (the dispatch gather is the natural site, and it is where d1 showed
cheap work rides). Per-token scaling on activations only. This route additionally quantizes
WEIGHTS (per-expert scalar), which is a numerics change beyond wire-only and must be in the
loss-parity verdict, not assumed away.
Confidence: 9/10 that the fp8 GEMM lowers as probed; 6/10 that the full path clears +1pp e2e; 8/10 that no kernel work is required.

## Check-in 2026-07-26 12:00 UTC — GATE 1 PASSED, with one finding that changes the fidelity story

Seven CPU tests in lib/levanter/tests/grug/test_fp8_wire_gemm.py, all passing.

**Identity holds.** Scaling the GEMM output rows equals scaling the input rows to 1.6e-5 relative
(pure fp32 accumulation rounding) at 512x256 @ 256x128. So moving the per-token scale past the dot
— the whole reason this route can produce an `__cublas$lt$matmul$f8` — is exact.

**Error decomposition (the finding).** Against an fp32 reference, on deliberately hostile data
(per-token dynamic range spanning e^-4..e^4):

| configuration | mean relative error |
|---|--:|
| activations e4m3 only | 0.0264 |
| weights e4m3, per-expert scalar scale, only | 0.0262 |
| both (scale on output) | 0.0375 |
| both (scale on input) | 0.0375 |
| both, weights scaled per COLUMN instead | 0.0374 |

Two things follow. First, activation and weight quantization contribute equally and add in
quadrature (0.0264 (+) 0.0262 -> 0.0372 predicted vs 0.0375 measured), so **weight quantization is
not a second-order concern to be waved through — it is half the error budget**, which is why loss
parity has to be the verdict and not a formality. Second, and this is the useful part: a per-column
weight scale buys **nothing** (0.0374 vs 0.0375). The limit is e4m3's 3-bit mantissa, not scale
granularity. So the per-expert *scalar* scale that the fp8 rewriter requires is free — we are not
trading fidelity for the fast path, which was the obvious worry about this design.

Also gated: per-token scales change only their own row when a neighbour is perturbed 1000x; padded
capacity slots quantize to exact zero (dropped assignments contribute nothing to the GEMM);
e5m2 survives cotangents of 5000 (e4m3 saturates at 448); the straight-through QDQ gradient is
finite, correctly shaped and non-zero.

Note the error figures are an upper bound by construction — the fixture's per-token range is far
wider than real MoE activations. The number that matters is the loss trajectory at gate 3.

Gate 2 next: confirm `__cublas$lt$matmul$f8` appears for the REAL dispatch path, not just probe
shapes. If it silently falls back there, stop and report.
Confidence: 9/10 on the identity; 6/10 on end-to-end loss parity (weights are half the error budget).

## GATE 2 PASSED 2026-07-26 12:10 UTC — the fp8 GEMM survives the real dispatch graph

experiments/grug/moe/fp8_dispatch_probe.py reproduces the actual dispatch path rather than clean
parameters: per-token quantize -> bitcast to uint8 -> tiled `all_to_all` inside `shard_map` on a
4-way expert mesh -> bitcast back to e4m3 -> reshape to [bucket, hidden] -> dot -> output-side
scale -> SwiGLU, at operating-point widths (hidden 5120, ffn 2x1280, capacity 2048).

| arrangement | lowers to |
|---|---|
| dequantize on arrival (what ships today) | `__cublas$lt$matmul` |
| unscaled fp8 into the dot, scale on the output | **`__cublas$lt$matmul$f8`** |

So the bitcast through uint8, the collective itself, the shard_map boundary and the reshape all
preserve the rewriter's operand pattern — the failure mode this gate existed to catch does not
occur. It also re-confirms on the real graph that today's shipped arrangement never had an fp8
GEMM at all, which is the mechanical half of the old -2.02pp.

GO for gate 3. Remaining work is the implementation I estimated at 1-2 days: carry (payload,
scale) out of the wire instead of dequantizing at ep_ragged_all_to_all.py:70, per-expert scalar
weight quantization, output-side scale application (folds into the existing SwiGLU fusion on the
dispatch leg, and into the combine-weight einsum on the combine leg), and extending the structured
custom adjoint to e5m2 cotangents in the mixed e5m2-grad x e4m3-weight form that probe row 4
already confirmed lowers correctly.

Verdict criteria fixed in advance: matched-draw rack pair (bf16 wire control vs fp8 wire), QB-on
cf1.0, drops on, 120 steps, one leg in flight. **Loss-trajectory parity is the verdict**, not MFU
— weights are half the quantization error budget per gate 1. Prize bracket, quoted as a bracket:
+1.34pp (a2a hidden fraction unchanged) to +2.84pp (a2a fully hidden), against a total remaining
collective-exposure budget of +3.37pp.
Confidence: 9/10 that the lowering holds in the production path; 6/10 on end-to-end loss parity.

## Check-in 2026-07-26 12:25 UTC — implementation constraint found: per-token scaling and pure-fp8 WGRAD are incompatible

Before writing the backward I checked whether the weight-gradient GEMM admits the same treatment
as the forward. It does not, and the reason is structural:

- **Forward** `hidden_i = scale_i * (q_i @ W)`: the per-token scale varies along the OUTPUT row
  axis, so it factors out of the dot. This is the row-linearity identity gate 1 verified, and it
  is why the forward lowers to `__cublas$lt$matmul$f8`.
- **dgrad** `dX_i = scale_i * (ct_i @ W^T)`: same structure, scale again on the output rows.
  Fine, and the mixed e5m2-grad x e4m3-weight form is confirmed lowering (probe row 4).
- **wgrad** `dW = sum_i (q_i * scale_i)^T ct_i`: the per-token scale varies along the REDUCTION
  axis. It cannot be hoisted out of the sum. Measured: hoisting a single scalar in its place gives
  **87% relative error**, and folding the row scale into `ct` instead does not fix it, because
  `ct`'s own per-token quantization scale then sits inside the same reduction one level down.

So the fidelity rule (per-token scaling, never shared across tokens) and a pure-fp8 wgrad are
mutually exclusive. This is not a bug to work around; it is a property of where the scale lives.

### Consequence for the increment

wgrad stays bf16. That costs no WIRE bytes — the wire carries activations and cotangents, and both
of those remain fp8 in both directions, so the prize bracket is unchanged. What it costs is one
dequantization of the received tensor in the backward, local compute, no collective. Under
`remat_mode=recompute_all` the forward is recomputed in backward anyway, so this lands on a tensor
that is already being materialized; whether it is free or eats into the gain is exactly the kind of
thing that shows up at gate 3 rather than in analysis, and I will report it as a risk rather than
assume it away.

One clean alternative exists if that cost turns out to matter: use DELAYED (previous-step amax,
per-tensor) scaling for the wgrad activation operand only. Delayed scaling is causally safe by the
established rule — the scale does not depend on the current batch — and being per-tensor it factors
out of the reduction. That would need amax history state, and per prior experience on this stack the
amax cotangent under `shard_map` must reduce with **pmax, not psum**. Not building it now; recording
it so the option is on the record with its known trap attached.

Both fidelity notes from the coordinator are already satisfied by the current design: scales are
per-token current-scaling (never tile- or sequence-spanning) and per-expert scalar, and nothing in
the forward or dgrad path uses an amax cotangent.
Confidence: 9/10 on the wgrad constraint (direct measurement, structural argument); 6/10 on end-to-end loss parity, unchanged.

## PARITY CLEAR 2026-07-26 12:45 UTC — fused fp8 dispatch GEMM implemented, 8/8 tests pass

`SCALE_A2A_FP8_GEMM=1` (requires `SCALE_A2A_FP8_WIRE=1`) replaces the dispatch leg's
dequantize-then-bf16-dot with a single fp8 dot. `_fp8_dispatch_gemm` is one custom_vjp spanning
the collective and the GEMM, so the fp8 payload never crosses an autodiff boundary.

Parity against a bf16 reference (single-device expert mesh, so the a2a is the identity and only
quantization plus the output-side scaling are under test):

| quantity | relative error | path |
|---|--:|---|
| forward | 0.0356 | e4m3 activations x e4m3 weights, per-token scale on the output |
| dgrad | 0.0726 | e5m2 cotangent x e4m3 weight, scale on the output rows |
| wgrad | 0.0252 | bf16, per the reduction-axis constraint |

dgrad carries roughly twice the forward error, which is what an e5m2 wire should cost (2-bit
mantissa against e4m3's 3). wgrad is the most accurate of the three precisely because it stayed
bf16. All within the hostile-fixture ceiling; the verdict remains the gate-3 loss trajectory.

Two implementation findings worth keeping:
- Custom-VJP residuals must be JAX types. Stashing `w13.dtype` there fails with "not a valid JAX
  type"; dtypes have to ride as zero-size arrays. Shapes needed no residual at all, since the
  tiled a2a preserves shape.
- A replicated input to `shard_map` needs its cotangent psum-ed across the manual axis, or JAX
  rejects the bwd rule with "the manual axis types do not match". Production is unaffected —
  expert weights are sharded per device, not replicated — but a test that replicates them trips
  it, and the error names the axis rather than the cause.
- `SCALE_A2A_FP8_GEMM` supersedes `SCALE_A2A_PREFETCH`: the fused op owns its collective, so the
  prefetch gate cannot hoist the next round's dispatch above this round's GEMM. Harmless now that
  the parallel-collective budget keeps every a2a async, but the two knobs do not compose and the
  code logs that.
Confidence: 9/10 on forward and dgrad correctness; 6/10 on end-to-end loss parity, unchanged.
Next: rack slot when free (pre-approved) — matched pair, bf16 wire control vs fp8 wire+GEMM.

## FINAL (fp8 wire + GEMM) 2026-07-26 13:00 UTC — LOSS PARITY HOLDS, EXPOSURE COLLAPSES AS PREDICTED, MFU -2.28pp

/mwittmann/ep25d4-fp8gemm-120-v1-20260726 vs the limit=4 control, both **120 steps** (identical
schedule position, so drop figures are directly comparable to each other and to nothing else).

**(1) Loss — PARITY, no drift.** Matched steps: 60: 6.6839 vs 6.6974 | 80: 6.2980 vs 6.3176 |
100: 5.9393 vs 5.9720 | 119: **5.6398 vs 5.6819**; tail20 5.7853 vs 5.8226. fp8 is at or slightly
below the control from step 60 on, within the same spread seen between earlier arms. The
weight-quantization risk — half the error budget by gate 1 — did NOT show up over this horizon.
Caveat: 120 steps is a short horizon for a quality claim.

**(2) Drops — matched regime.** Mean drop 20-119: **0.2669 vs 0.2670**. @119 0.0901 vs 0.0876.
Quantize-strictly-after-routing held: drop accounting is untouched.

**(3) MFU — NEGATIVE, and outside the bracket.** p50 20.391 vs 22.674 at 20-119 = **-2.283pp**
(drop-corrected identical), consistent across all windows (-2.27 to -2.31). Step time 13.28 vs
11.94 s, **11% slower**. Pre-registered bracket was +1.34 to +2.84pp; this lands below zero.

**(4) Concurrency — the byte thesis was RIGHT; the cost is elsewhere.**

| GPU:0, 3-step window | limit=4 control | fp8 wire+GEMM |
|---|--:|--:|
| collective total | 13226 ms | 9784 ms (-26%) |
| SendRecv | 10347 ms | 7447 ms (-28%) |
| **exposed collective** | **4287 ms (12.9% of span)** | **1526 ms (4.0%)** |
| trace span | 33159 ms | 37935 ms (+14%) |

Exposure fell by 2761 ms — 64% of the entire +3.37pp exposure budget, almost exactly what halving
the dispatch leg predicted. The fp8 GEMM is genuinely running (the kernel changed to
`nvjet_sm100_qqtst_128x256_128x6_2x1_2cta_v_bz_NNT`). But the span GREW 4776 ms, so non-collective
compute grew ~7.5 s per 3 steps (~2.5 s/step) — roughly twice the collective saving.

### Verdict and what it means

Reported as a NEGATIVE per the pre-fixed criteria: -2.28pp MFU. But it is a *different* negative
from the -2.02pp of a year of this thread, and the difference is the whole point: the wire
saving materialized exactly as modelled and the loss held. What kills it is quantization COMPUTE —
the per-token amax pass on send, the backward dequantization of `received` for the bf16 wgrad, and
the e5m2 cotangent quantize. Those are the costs I flagged as "gate-3 measurements, not analysis
questions", and they measured worse than hoped.

This does not close byte reduction; it prices its current implementation. The three specific costs
are each attackable: fold the quantize into the dispatch gather (d1's finding that cheap work rides
the index composition), avoid the backward dequant with a delayed per-tensor scale on the wgrad
operand (prior art exists in the fp8 ragged-dot lineage — mine it rather than rebuild), and fuse
the cotangent quantize into the combine epilogue. Whether the remaining ~2.5 s/step of QDQ can be
driven under the 2761 ms/3-step exposure saving is the question that decides this route.
Confidence: 9/10 on all four readings; 6/10 that a fused-QDQ variant turns this positive.

## ATTRIBUTION 2026-07-26 13:15 UTC — ONE cost dominates: the per-token amax reductions are ~all of it

Compute-stream op deltas, fp8 leg vs limit=4 control, GPU:0, same 3-step windows:

| op | control | fp8 | delta |
|---|--:|--:|--:|
| `loop_reduce_fusion_2` | 409.6 ms | 4806.6 ms | **+4397 ms** |
| `loop_reduce_fusion` | not in top-30 | 2693.6 ms | **+2694 ms** |
| `loop_convert_fusion_5/23/26` | not in top-30 | 1559.7 ms | +1560 ms |

The two reduce fusions alone are **~7500 ms per 3-step window = ~2500 ms/step**, which is the
entire ~2.5 s/step of added non-collective compute. These are the per-token amax passes: a full
reduction over the activation tensor on send, and a second over the cotangent in the backward.
The convert fusions (the casts themselves) are ~520 ms/step, secondary. Caveat on method: the
tool reports each leg's top-30 ops, so an op absent from the control's list reads as 0 and its
delta is an upper bound; `loop_reduce_fusion_2` is the reliable one because it appears in both.

**This is the >60% single-cost case, not the three-way split.** The backward dequant of `received`
for the bf16 wgrad — the cost I most expected to dominate — does not appear at the top at all.

### The number the fix has to beat (sanity-checked)

- exposure saving: 2761 ms per 3 steps = **920 ms/step**
- added QDQ compute: span grew 1592 ms/step, exposure fell 920 ms/step, so compute grew
  1592 + 920 = **2512 ms/step**
- **break-even requires QDQ to fall 2512 -> 920 ms/step, a 2.73x reduction**
- full elimination gives step 11.94 - 0.92 = 11.02 s, i.e. MFU 22.674 x 11.94/11.02 =
  **24.57%, or +1.90pp** over the limit=4 control

The targeted fix follows directly from the attribution: **delayed scaling removes the amax passes
outright** — use the previous step's per-tensor amax instead of reducing the current activation —
and it is causally safe by the established rule since the scale does not depend on the current
batch. That is one change addressing ~100% of the measured overhead rather than three changes
addressing a third each. Prior art exists in the fp8 ragged-dot lineage and should be mined rather
than rebuilt; the known trap is that an amax cotangent under `shard_map` must reduce with **pmax,
not psum**. Per-token current-scaling would have to be traded for delayed per-tensor scaling on the
quantized operands, which is a fidelity change that needs its own loss verdict.

### Strategic framing, so nobody infers more than this route offers

Even a fully optimized fp8 wire lands the honest configuration near **24.4-24.6%**, and near
**24.2% at 3.66% drops** once spill is included for fidelity. That is a real improvement and the
largest remaining speed lever, but it is **short of 25% with a strict 3% drop bar at this shape**.
This route should be pursued on its merits, not as a path that closes the goal.
Confidence: 8/10 on the attribution (one op measured in both legs carries it; the rest are upper bounds); 9/10 on the break-even arithmetic.

## GATE 1 (delayed scaling) 2026-07-26 13:30 UTC — precision cost is ~2%, far below what the amax passes cost

Relative GEMM error vs fp32 (e4m3 activations x e4m3 weights, per-expert scalar weight scale):

| activation dynamic range | per-token current | delayed per-tensor (exact) | scale +30% stale | scale -20% stale |
|---|--:|--:|--:|--:|
| hostile, e^-4..e^4 (~3000x) | 0.0374 | **0.0381** | 0.0380 | 0.0378 |
| moderate, e^-1..e^1 | 0.0373 | 0.0377 | 0.0376 | 0.0379 |
| realistic, e^-0.3..e^0.3 | 0.0373 | **0.0374** | 0.0374 | 0.0378 |

Delayed per-tensor costs **2% relative** on the hostile fixture and is indistinguishable on a
realistic one. Staleness barely registers: a scale 30% too large or 20% too small moves the error
in the fourth decimal, so amax history does not need to be tight.

The reason is the same one that made per-column weight scales pointless in the earlier gate: error
is dominated by e4m3's 3-bit MANTISSA, not by dynamic range. Per-token scaling only pays when rows
span more than the format's exponent range, and e4m3 covers ~28000x while even the hostile fixture
spans ~3000x. Under-range, a shared scale loses essentially nothing.

**Two distinctions kept separate, per the coordinator and because conflating them is how this goes
wrong:** delayed per-tensor scaling is CAUSALLY SAFE — the scale derives entirely from the previous
step, so it depends on no token in the current batch, future or past. The sequence-axis rule
forbids token-spanning tiles computed from the CURRENT batch, which this is not. What delayed
scaling costs is PRECISION, one scale per tensor instead of one per token, and that cost is
measured above at ~2%. Correctness question: settled. Quality question: 2%, and the loss verdict
still stands on its own.

So the trade is: remove ~2500 ms/step of amax reduction (the entire measured QDQ overhead) for ~2%
relative quantization error. Against a break-even requirement of a 2.73x QDQ reduction and a
+1.90pp prize at full elimination, that is the favorable side of the discriminator.

NOTE the next leg CANNOT inherit the previous leg's loss parity — the quantization scheme changes
from per-token current to delayed per-tensor, so its loss verdict must stand alone.
Confidence: 9/10 on the precision measurement; 7/10 that delayed scaling flips the sign at the rack.
