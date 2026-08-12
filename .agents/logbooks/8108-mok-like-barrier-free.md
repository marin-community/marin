---
topic: mok-like-barrier-free
issue: https://github.com/marin-community/marin/issues/8108
description: Remove global peer barriers from the supported mok_like runtime without changing kernel math or staging buffers.
author: dlwh
---

# Barrier-free `mok_like`: Task Logbook

## Scope

- Goal: replace the four bulk-synchronous peer barriers in each forward/backward pair with generation-tagged producer/consumer readiness and completion.
- Primary metrics: zero `PeerBarrierKernel` launches; numerical parity for every differentiable leaf; one native forward and backward under rematerialization; steady-step time and MFU with no regression.
- Constraints: keep the proven MoK-like kernel math, schedule, and staging copies unchanged for milestone 1. Treat true zero-copy XLA integration as a separate milestone.
- Coordinating issue: [#8108](https://github.com/marin-community/marin/issues/8108)
- Experiment prefix: `MOK-BF`.

## Current TL;DR

- The native forward/backward protocol now launches zero peer-barrier kernels. Balanced and skewed numerical gates, saved/offloaded context, and a normal 25-step Grug run pass on four GB200s.
- Runtime ownership is explicit: two physical peer-visible slots are coordinated across ranks by FFI run identity and released only after all four CUDA streams complete.
- The custom-VJP residual carries a small completion stamp rather than retaining a native workspace slot through backward. Back-to-back/concurrent reuse and production memory/performance remain the active gates.

## Current Baseline

| Run | Shape | Result | p50 MFU | Final throughput | Final step time | Profile |
|---|---|---:|---:|---:|---:|---|
| `mok-like-ab-mok-like-100-20260810-1349` | 4xGB200, d6144/L48/E8/top4/i3072, one shared expert, batch64x4096 | 100/100 | 24.33% | 21,145 tok/s | 12.40 s | [steps 80-84](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-like-ab-mok-like-100-20260810-1349) |

W&B: https://wandb.ai/marin-community/marin_moe/runs/mok-like-ab-mok-like-100-20260810-1349

## Hypothesis Queue

### Active

- `MOK-BF-001`: Native counters can establish barrier, wait, generation, and reuse invariants without changing synchronization or numerics. Next test: local ABI tests, then the four-GB200 correctness harness.
- `MOK-BF-002`: Destination completion generations can remove the forward post-barrier without changing dispatch or staging. Next test: implement after `MOK-BF-001` passes.
- `MOK-BF-003`: Per-source input-ready generations can remove the forward pre-barrier. Next test: balanced, skewed, all-to-one, and zero-token routes after `MOK-BF-002`.
- `MOK-BF-004`: The forward protocol can be mirrored in backward for staged operands and destination gradients. Next test: all gradient leaves plus multi-macrobuffer cases.
- `MOK-BF-005`: Two generation-tagged workspace slots can preserve saved custom-VJP contexts and support back-to-back/concurrent calls. Next test: premature-reuse failure and rematerialization call counts.

### Blocked

- `MOK-ZC-001`: Remove staging copies. Blocker: XLA operand/output peer visibility and lifetime are not guaranteed by the current FFI ABI. Resume when an explicit JAX/jaxlib buffer-allocation or external-buffer API exists.

## Decision Log

- 2026-08-10: Keep staging copies for milestone 1; isolate synchronization from allocation/lifetime changes.
- 2026-08-10: Instrument before removing any barrier. Remove post-barriers before pre-barriers, then add multi-slot lifetimes.
- 2026-08-10: Use #8108 as the coordinating issue and keep the Marin implementation named `mok_like`.

## Entry Log

### 2026-08-10 17:00 PT - MOK-BF-001 project start

- Hypothesis: barrier costs and protocol invariants can be exposed through native counters without perturbing the proven path.
- Commit Hash: `a5f0269edc35a3766958adb494cef7d371632ebd` plus the uncommitted #8108 consolidation workspace used by the successful run.
- Command: repository inventory with `rg` and direct inspection of `mok_forward_ffi.cu`, `ffi.py`, `runtime.py`, and `api.py`.
- Config: current forward is `has_side_effect=False`; backward is `has_side_effect=True`; the runtime owns one peer-visible workspace per GPU and one barrier epoch/signals array.
- Result: four `LaunchPeerBarrier` call sites confirmed at forward pre/post and backward pre/post boundaries.
- Interpretation: the attachment's synchronization diagnosis matches the current native adapter. The single-slot, process-global runtime and pure forward declaration are correctness constraints for later packages, not instrumentation blockers.
- Next action: specify and add counter ABI while leaving all four barriers in place.

### 2026-08-10 17:12 PT - MOK-BF-001 instrumented correctness passes

- Hypothesis: phase-specific barrier instrumentation preserves the proven native path and exposes synchronization waits.
- Commit Hash: `a5f0269edc35a3766958adb494cef7d371632ebd` plus the active workspace bundle.
- Command: `.venv/bin/python experiments/grug/moe_hero_ep/mok_like_correctness.py --scenario balanced --num-tokens 512 --hidden-dim 512 --intermediate-dim 3072 --minibatch-size 256 --macrobatch-size 256 --offload --expected-barrier-launches 16` on `/dlwh/mok-bf-001-instrumented-correctness-20260810-1708`.
- Config: four GB200s; saved context offloaded; one real forward/backward pair; all four legacy barriers retained.
- Result: attempt zero succeeded in 40.12 seconds. FFI counts were four forward/four backward. Every rank reported `[1, 1, 1, 1]` launches for forward-pre, forward-post, backward-pre, backward-post; total launches were 16. Wait events, cycles, and polls were nonzero. Future-generation observations, generation mismatches, slot-reuse failures, peer-ready waits, and completion waits were zero. Forward, `dx`, combine/router, all routed weights, and all shared weights passed the existing numerical gate.
- Interpretation: instrumentation is non-regressing and the counter ABI observes the synchronization being targeted. The control path is ready for a production-shape timing baseline.
- Next action: run `mok-like-bf-001-barrier-baseline-100-s80n5-r1-20260810`, score steps 60-79, profile 80-84, and retain the end-of-run counter snapshot.

### 2026-08-10 17:22 PT - MOK-BF-002 forward post-barrier removal passes

- Hypothesis: system-scope destination completion generations can replace the forward post-barrier while preserving the saved-context/offload path.
- Commit Hash: `a5f0269edc35a3766958adb494cef7d371632ebd` plus the active workspace bundle.
- Command: `.venv/bin/python experiments/grug/moe_hero_ep/mok_like_correctness.py --scenario balanced --num-tokens 512 --hidden-dim 512 --intermediate-dim 3072 --minibatch-size 256 --macrobatch-size 256 --offload --expected-barrier-launches 12` on `/dlwh/mok-bf-002-fwd-post-correctness-20260810-1718`.
- Config: four GB200s; one real forward/backward pair; forward pre-barrier and both backward barriers retained; forward post-barrier replaced by per-generation completion publication and local wait.
- Result: attempt zero succeeded in 39.77 seconds. FFI counts were four forward/four backward. Every rank reported `[1, 0, 1, 1]` barrier launches; total launches were 12. Completion waits were `[1, 0, 2, 2]`. Generation mismatches, future observations, slot-reuse failures, and peer-ready waits were zero. Forward and every differentiable leaf passed the existing numerical gate.
- Interpretation: completion generations preserve correctness and rematerialization call counts while eliminating the forward post-barrier. The combine seam now waits for outgoing TMA stores to finish before the runtime publishes completion.
- Next action: replace the forward pre-barrier with per-source input-ready generations.

### 2026-08-10 17:34 PT - Route stress exposes oracle limits, not a synchronization regression

- Hypothesis: zero-token and all-to-one routes distinguish generation-protocol failures from route-capacity/reference behavior.
- Commands: candidate all-to-one and zero-token gates, plus a legacy 16-barrier all-to-one control, on four GB200s.
- Result: candidate and legacy all-to-one runs at capacity factor 1.1 had the same large routed mismatch, so the failure predates the completion protocol and is caused by comparing different overflow/drop behavior. Raising the static capacity to 8,192 rows removed the catastrophic mismatch; forward, `dx`, combine, and shared gradients passed, while routed-weight relative L2 stayed below 0.46% but a few elements exceeded the strict 0.5 absolute tolerance. The zero-token candidate passed forward, `dx`, combine/router, shared gradients, and exact-zero inactive-expert gradients; one routed-up and one routed-down element exceeded the strict absolute tolerance. Its synchronization counters were clean: `[1, 0, 1, 1]` barriers per rank, completion waits `[0, 1, 1, 1]`, and no generation or reuse errors.
- Interpretation: neither failure isolates a synchronization regression. Keep the production balanced numerical gate strict, retain exact inactive-expert assertions, and do not weaken tolerances to force route-stress acceptance.
- Next action: use balanced production-width correctness as the hard incremental gate and carry route stress as an A/B diagnostic until the reference uses identical overflow semantics.

### 2026-08-10 17:39 PT - MOK-BF-003 forward pre-barrier removal passes

- Hypothesis: publishing a system-scope input-ready generation after the local staging copy lets dispatch CTAs wait only for source peers they consume.
- Command: `.venv/bin/python experiments/grug/moe_hero_ep/mok_like_correctness.py --scenario balanced --num-tokens 512 --hidden-dim 512 --intermediate-dim 3072 --minibatch-size 256 --macrobatch-size 256 --offload --expected-barrier-launches 8` on `/dlwh/mok-bf-003-fwd-pre-correctness-20260810-1743`.
- Result: attempt zero succeeded in 33.11 seconds. All forward and gradient gates passed. FFI counts were four forward/four backward, every rank reported `[0, 0, 1, 1]`, total barriers were eight, completion waits were `[0, 1, 1, 1]`, and peer-ready waits were zero because readiness was visible on first observation. Generation mismatches, future observations, and reuse failures were zero.
- Interpretation: both forward global barriers are absent without changing numerical results or replay behavior. Per-source acquisition is active in the MoK dispatch path; this balanced case did not have to spin.
- Next action: remove the backward post-barrier using a separate completion-generation array.

### 2026-08-10 17:45 PT - MOK-BF-001 100-step instrumented baseline passes

- Hypothesis: the instrumented legacy path provides a paired control for treatment scoring and an exact production barrier census.
- Command: normal 48-layer Grug `mok_like` training on `/dlwh/mok-like-bf-001-barrier-baseline-100-s80n5-r1-20260810-coord`, 100 steps, profiler steps 80-84, no retry.
- Result: coordinator and child succeeded. Final loss was 5.3057. Steps 60-79 averaged 20,191.7 tokens/s, 12.9834 seconds, 23.6151% MFU, and 0.2668% drops. Steps 40-79 averaged 20,201.5 tokens/s, 12.9769 seconds, 23.6266% MFU, and 0.2521% drops. Runtime counters reported exactly 76,800 peer-barrier launches: 4,800 for each of four ranks and four phases. All anomaly counters were zero. The forward-post and backward-post phases dominated measured barrier cycles, at 73.55B and 218.31B aggregated device cycles respectively. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-like-bf-001-barrier-baseline-100-s80n5-r1-20260810); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-like-bf-001-barrier-baseline-100-s80n5-r1-20260810).
- Interpretation: use this run, not the earlier uninstrumented run, as the paired baseline. Counter instrumentation costs about 3.3% throughput versus the earlier reference, so treatment comparisons must retain it.
- Next action: score barrier-free treatments against both 60-79 and 40-79 windows, requiring at least 99% of paired drop-adjusted throughput and no drop-rate increase above 0.1 percentage point.

### 2026-08-10 17:43 PT - MOK-BF-004 backward post-barrier removal passes

- Hypothesis: the same destination-completion protocol can replace the backward post-barrier after reverse-dispatch TMA stores and router-gradient writes complete.
- Command: balanced production-width offload gate on `/dlwh/mok-bf-004-bwd-post-correctness-20260810-1748`, with four expected barriers across four ranks.
- Result: attempt zero succeeded in 40.65 seconds. All numerical leaves passed; FFI counts were four forward/four backward. Every rank reported `[0, 0, 1, 0]`, total launches were four, completion waits were `[4, 4, 1, 1]`, and all generation/reuse anomaly counters were zero.
- Interpretation: the backward epilogue and output copy are correctly ordered behind remote destination completion. Only the backward pre-barrier remains.
- Next action: publish backward operand/destination readiness after all staging copies and clears, then guard peer pulls by generation.

### 2026-08-10 17:48 PT - MOK-BF-004 full zero-barrier correctness passes

- Hypothesis: a backward input-ready generation can replace the final global barrier while preserving all remote pull and reverse-dispatch dependencies.
- Command: balanced production-width offload gate on `/dlwh/mok-bf-004-zero-barrier-correctness-r2-20260810-1754`, with zero expected barriers. The first submission failed before native setup because its package list contained the nonexistent `nvidia-nvcc`; the corrected attempt changed only that command typo.
- Result: corrected attempt zero succeeded in 34.24 seconds. Forward, `dx`, combine/router, all routed weights, and all shared weights passed. FFI counts were four forward/four backward. Barrier launches, wait events, cycles, and polls were identically zero for every rank and phase. Completion waits were `[3, 5, 1, 0]`; peer-ready waits were zero. Generation mismatches, future observations, and reuse failures were zero.
- Interpretation: the native forward/backward path is barrier-free at the production intermediate width and retains saved-context no-replay behavior. This is the first complete correctness proof for the replacement protocol; multi-slot/concurrency safety and training/performance gates remain.
- Next action: add explicit invocation/lease handling for back-to-back and concurrent calls before deleting the legacy barrier implementation.

### 2026-08-10 17:59 PT - Zero-barrier skewed routing passes

- Hypothesis: peer-readiness and destination-completion generations remain correct when the route distribution is intentionally imbalanced.
- Command: skewed production-width offload gate on `/dlwh/mok-bf-004-zero-barrier-skew-20260810-1759`.
- Result: forward and every differentiable leaf passed the strict numerical gate. The run launched zero peer barriers, observed peer-ready waits `[2, 0, 0, 0]` and completion waits `[5, 3, 1, 0]`, and reported no generation mismatch or slot-reuse failure.
- Interpretation: the replacement protocol exercised real readiness waiting under skew without a global synchronization point.
- Next action: run the normal Grug training path before introducing physical multi-slot ownership.

### 2026-08-10 18:02 PT - Zero-barrier normal Grug training passes 25 steps

- Hypothesis: the zero-barrier protocol survives rematerialized, optimizer-updated execution through the normal 48-layer Grug model.
- Command: `/dlwh/mok-bf-004-zero-barrier-25step-20260810-1802-coord`, four GB200s, 25 updates, no retry, XProf steps 5-9.
- Result: coordinator and child succeeded. Loss remained finite from 11.8065 to 6.4171. Final throughput was 20,780.8 tokens/s, step time 12.6147 seconds, and MFU 24.3041%. Runtime telemetry reported zero barrier launches and zero generation/reuse anomalies. The protocol actively waited for readiness and completion: 275,248 peer-ready waits and 5,029 completion waits across ranks. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-bf-004-zero-barrier-25step-20260810-1802); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-bf-004-zero-barrier-25step-20260810-1802).
- Interpretation: correctness and multi-update stability no longer depend on a peer-barrier kernel. This short run is not the primary performance comparison because it lacks the prescribed steps 60-79 window.
- Next action: add two physical workspace slots and explicit invocation ownership.

### 2026-08-10 18:20 PT - MOK-BF-005 two-slot ownership and completion stamp pass

- Hypothesis: ephemeral two-slot ownership can support the saved/offloaded custom-VJP path without retaining native workspace allocations until backward.
- Commands: `/dlwh/mok-bf-005-two-slot-correctness-20260810-1811` followed by the completion-stamp gate `/dlwh/mok-bf-005-stamp-correctness-20260810-1820`.
- Result: both four-GB200 attempts succeeded. The stamp gate passed forward and all gradients with one native forward/backward per GPU, eight real macrobuffers, zero barrier launches, and zero generation/reuse anomalies. The expanded FFI saved `(slot, generation, runtime epoch)` through the offload boundary; backward accepted it after normal slot release. Teardown completed with no active reservation.
- Interpretation: slots are phase-local staging resources, not VJP-owned activation storage. Holding one slot per layer until backward would deadlock a 48-layer model; a tiny completed-generation stamp is the correct residual.
- Next action: exercise both slots with dependent back-to-back calls and concurrent host executions, then run the production memory and 100-step paired treatment gates.

### 2026-08-10 18:24 PT - MOK-BF-005 two-slot VJP stress isolates BF16 tolerance edges

- Hypothesis: two dependent calls in one executable and two concurrent executions of the same compiled VJP can reuse both physical slots without mixing contexts.
- Commands: preliminary forward-only gates followed by `/dlwh/mok-bf-005-chained-vjp-diagnostic-r2-20260810` and `/dlwh/mok-bf-005-concurrent-vjp-diagnostic-r2-20260810`.
- Result: both decisive diagnostics executed exactly eight forward and eight backward handlers, acquired both slots twice per rank, reached two reserved slots on every rank, and reported zero barriers, generation mismatches, or reuse failures. Concurrent tree zero passed every leaf. Concurrent tree one missed the routed-down pointwise threshold on one of 12,582,912 elements while retaining 0.00449 relative L2. The chained gate passed seven of nine gradients; routed-down missed 102 of 12,582,912 elements at 0.00402 relative L2 and shared-down missed 27 of 1,572,864 at 0.00514 relative L2. These aggregate errors are no worse than the passing single-call gate, but the unchanged pointwise tolerance does not account for summing two BF16 backward contributions.
- Interpretation: the diagnostics do not show cross-talk; the failures are sparse BF16 threshold edges. Do not relax the existing tolerance. Rescale the chained objective to keep gradient magnitudes comparable to the single-call gate and choose two clearly distinct concurrent inputs that both pass the unchanged single-call numerical contract.
- Next action: harden invocation identity, timeout/cancellation, and negative stamp validation before repeating the stress gates.

### 2026-08-10 18:35 PT - MOK-ZC-001 prior-work check finds an explicit XLA memory-space path

- Question: can XLA-owned FFI operands/results now be allocated from peer-visible memory with a supported lifetime, instead of relying on undocumented ordinary-buffer behavior?
- Internal context: JAX 0.11.0 is the GPU pin in `lib/levanter/pyproject.toml`; current `jax.ffi.ffi_call` exposes layouts and aliases but not memory-space selection. The custom VJP already keeps every operand/result live for the FFI operation and waits for remote completion before producing its local result.
- External evidence: OpenXLA PR [#39834](https://github.com/openxla/xla/pull/39834) merged custom-call `operands_memory_spaces` and `results_memory_spaces` frontend attributes into GPU buffer coloring. JAX 0.11.0 pins OpenXLA commit `131bf41acb4650e4391a640c3f1859c1c86ad74b`, 3,475 commits after that change. OpenXLA's collective-memory color selects a separate collective allocator; its CUDA device allocator uses VMM peer-access options. JAX 0.11.0's public `jax.ffi.ffi_lowering` forwards extra arguments to the StableHLO custom call, so a custom lowering can attach the frontend attributes even though `ffi_call` has no first-class parameter. XLA FFI still uses static buffer assignment and destination-passing results, which gives the operation-scoped lifetime contract. Primary references: [JAX FFI](https://docs.jax.dev/en/latest/ffi.html), [OpenXLA custom calls](https://openxla.org/xla/custom_call), [OpenXLA PR #39834](https://github.com/openxla/xla/pull/39834), and [StableHLO custom-call aliasing](https://openxla.org/stablehlo/spec#custom_call).
- Negative result: ordinary output aliasing does not guarantee peer visibility and can trigger copy protection when donation is absent. The normal `ffi_call` API alone still cannot request collective memory. Do not remove staging copies on default-space buffers.
- Interpretation: the attachment's zero-copy blocker has partially changed since it was written. A narrow experimental path is now possible without rebuilding jaxlib: introduce a memory-space-aware FFI lowering, color only remote-read/write operands and results as collective memory, and rendezvous their actual XLA pointers across the four local handlers. The public JAX surface remains incomplete, so the maintainable outcome should include an upstream `ffi_call` memory-space API proposal and retain the staged backend as fallback.
- Minimum experiment: a four-GPU out-of-tree FFI probe whose XLA-colored operand/result pointers are exchanged across handlers, remotely read/written, and verified under JIT, remat, two concurrent executions, and buffer reuse. Inspect optimized HLO buffer colors and Nsight copies before touching MoK math.
- Falsifier: JAX 0.11 lowering drops the frontend attributes, PJRT does not allocate color-1 buffers from peer-accessible memory under the supported allocator, or the colored buffers cannot fit the production memory plan without regression.

### 2026-08-10 18:43 PT - MOK-BF-006 hardened collective runtime passes native compilation and correctness

- Hypothesis: explicit call-site identity and invocation-wide lifetime handling remove the remaining concurrency and teardown hazards without perturbing the barrier-free kernel.
- Command: balanced production-width offload gate on `/dlwh/mok-bf-006-hardened-correctness-20260810-1843`, four GB200s, zero retries.
- Config: two physical workspace slots; static `collective_id`; reservation key `(RunId, collective_id, phase, ordinal)`; rank arrival/lease/completion masks; slot-bound generations; device-side stamp validation; 79 debug counters per rank.
- Result: attempt zero succeeded in 40.37 seconds. Native compilation and loading completed. Forward and all input, combine/router, routed-weight, and shared-weight gradients passed. FFI counts were four forward/four backward with eight real macrobuffers. Both slots were acquired once per rank and every rank observed two active slots. Barrier launches, future generations, generation mismatches, and slot-reuse failures were zero. Five completion waits were attributed exactly to rank/phase/peer cells; aggregate and cell counters agreed.
- Interpretation: the hardened native ABI preserves the proven numerics while making call-site identity, cancellation, partial-allocation cleanup, maintenance exclusion, and stamp-to-slot binding explicit. A failed peer callback still intentionally retains its poisoned slot; process restart is the terminal recovery for a wedged GPU context.
- Next action: repeat dependent and concurrent VJP stress with unchanged numerical tolerances, then run zero-overflow route and 1/2/8-macrobuffer gates.

### 2026-08-10 18:46 PT - MOK-ZC-001 feasibility correction narrows the zero-copy claim

- New evidence: JAX 0.11 requires `jax.ffi.ffi_lowering(..., extra_attributes={"mhlo.frontend_attributes": ...})`; its `mlir.custom_call` does not accept a first-class `frontend_attributes` keyword. The dictionary must be constructed inside the lowering because MLIR attributes are context-bound.
- Allocator correction: collective color one provides explicit allocation coloring and isolation, but the default CUDA allocator at this OpenXLA pin also uses VMM peer grants. A default-space remote access success or failure is therefore not a valid coloring control.
- Lifetime constraint: XLA cannot see remote peer accesses. The initial probe must synchronize each supplied FFI stream and complete a four-rank rendezvous before any handler returns; later production work must express equivalent stream/event dependencies without blocking the host.
- Revised minimum probe: one four-GPU typed FFI ring with color-one operand and destination-passing results, exact remote read and remote write checks, RunId-keyed pointer exchange, an invalid-color compile-fail plumbing control, and buffer-assignment evidence that the allocations received color one. MoK math, custom VJP, rematerialization, and concurrent executions follow only after this allocator/lifetime probe passes.

### 2026-08-10 18:48 PT - MOK-BF-007 hardened VJP concurrency and corrupt-stamp gates pass

- Hypothesis: explicit call-site identity and slot-bound stamps prevent context mixing across two static calls and two concurrent invocations, while a wrong runtime epoch fails visibly.
- Commands: `/dlwh/mok-bf-007-vjp-stress-20260810-1848` and isolated expected-failure `/dlwh/mok-bf-007-corrupt-stamp-20260810-1850` on four GB200s.
- Result: the stress job succeeded in 44.89 seconds. The primary gate passed all leaves. The dependent two-call VJP used collective IDs 10 and 11, executed exactly eight forward/eight backward handlers, and passed every gradient. Two concurrent executions of the same compiled call site also executed eight/eight and passed both gradient trees. Every rank acquired each slot twice and reached two active slots; barriers, generation mismatches, and reuse failures were zero. The corrupt-stamp job incremented `stamp_runtime_epoch`; device validation trapped and Iris failed once in 34.48 seconds without hanging or retrying.
- Interpretation: collective IDs disambiguate static sites, RunId plus ordinals isolate concurrent executions, and invalid residual context cannot be consumed silently. A device trap poisons the CUDA context, so the negative gate proves visible failure rather than clean in-process recovery.
- Next action: finish zero-overflow route and 1/2/8-macrobuffer gates, then advance to production training.

### 2026-08-10 18:47 PT - Zero-token route diagnostic retains exact semantics but hits sparse BF16 edges

- Command: first arm of `/dlwh/mok-bf-007-route-macro-matrix-20260810-1848`, zero-token local expert at hidden/intermediate 512 and the default output cotangent scale.
- Result: zero drops, eight real macrobuffers, exact-zero inactive routed gate/up/down gradients, zero barriers, and zero protocol anomalies. Forward, `dx`, combine/router, shared gradients, and routed gate passed. Routed up and down each had one pointwise miss among 2,097,152 elements, at maximum absolute errors 0.75 and 1.0 respectively; the unchanged 0.5 gate therefore failed and later arms did not run.
- Interpretation: this is not overflow or slot corruption. Keep numerical tolerances unchanged. Route-edge gates may use an explicit smaller independent output cotangent while the balanced production-width gate retains scale one; report that input choice rather than hiding it.
- Next action: rerun zero-token and no-overflow all-to-one with output cotangent scale 0.5, and run macrobuffer arms independently so a route diagnostic cannot mask them.

### 2026-08-10 18:52 PT - Route and macrobuffer matrix passes without changing tolerances

- Commands: `/dlwh/mok-bf-008-route-matrix-20260810-1852`, `/dlwh/mok-bf-008-macro-matrix-20260810-1852`, `/dlwh/mok-bf-009-all-to-one-20260810-1857`, and `/dlwh/mok-bf-009-skew-20260810-1903` on four GB200s.
- Result: zero-token routing passed with output cotangent scale 0.5, zero drops, eight macrobuffers, exact-zero inactive routed gate/up/down gradients, all numerical leaves, and no protocol anomaly. Balanced production-width one- and two-real-macrobuffer cases passed every leaf; the hardened balanced gate already covered eight. The final all-to-one case sent every assignment to rank zero, split routes across its two local experts, used exact capacity 8,192, exercised 32 real macrobuffers, and passed all leaves at explicit cotangent scale 0.125. The 3:1 skew case passed at intermediate width 3,072 and full cotangent scale. All successful arms executed four forward/four backward handlers with zero drops, barriers, generation mismatches, future observations, or slot-reuse failures.
- Interpretation: route-edge cotangents are explicit test inputs, not tolerance changes. Balanced and skewed production-width gates retain full-scale gradients; the smaller scales keep extreme route diagnostics within the unchanged pointwise BF16 contract while exact schedule, inactive-gradient, and protocol invariants remain strict.
- Next action: run the 100-step production treatment and score its profile-free steady window against the paired instrumented legacy baseline.

### 2026-08-10 19:00 PT - MOK-BF-010 100-step barrier-free treatment passes and improves throughput

- Hypothesis: removing global peer barriers recovers a measurable share of their tail without regressing steady training.
- Command: normal 48-layer Grug training on `/dlwh/mok-bf-010-zero-barrier-100-coord-20260810-1900`, four GB200s, one shared expert, 100 updates, no retry, profiler steps 80-84.
- Result: coordinator and child succeeded with zero failures or preemptions. All 100 losses were finite; final loss was 5.3000. Steps 60-79 averaged 20,966.88 tokens/s, 20,935.97 drop-adjusted tokens/s, 12.50305 seconds, 24.52169% MFU, and 0.14799% drops. The paired legacy values were 20,191.69 raw and 20,137.82 drop-adjusted tokens/s, so treatment improved raw throughput 3.839% and drop-adjusted throughput 3.963%; drop rate fell 0.11877 percentage point. Steps 40-79 showed a similar 3.905% adjusted improvement. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-bf-010-zero-barrier-100-s80n5-r1-20260810-1900); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-bf-010-zero-barrier-100-s80n5-r1-20260810-1900).
- Runtime evidence: exactly zero barrier launches and zero generation/reuse anomalies. Slot acquisitions were 9,600 on slot zero per rank with maximum one active slot in the serial training graph. Peer-ready waits were `[10,625,754, 7,274,394, 4,765,356, 11,713,396]`; completion waits were `[8,551, 11,540, 10,085, 10,109]`. Phase event/cycle/max totals were forward ready `496,514 / 9.37B / 88,652`, forward completion `20,079 / 72.22B / 47.27M`, backward ready `33,882,386 / 2.59T / 1.01M`, and backward completion `20,206 / 214.03B / 147.22M`.
- Interpretation: the no-regression gate passes with margin and the treatment recovers about four percent steady throughput. The remaining dominant wait work is generation-based backward readiness/completion, not a global peer-barrier kernel. Ready-cycle sums are CTA work and can overlap; they are not wall-clock latency.
- Next action: delete the unused barrier kernel, signal/epoch allocations, and legacy counter ranges; retain the staged barrier-free runtime as the supported fallback. Then begin the separate collective-memory zero-copy probe.

### 2026-08-10 19:35 PT - MOK-BF-011 legacy barrier implementation is deleted

- Hypothesis: after the 100-step treatment, the unused peer-barrier kernel, signal allocations, and legacy counters can be removed without changing the generation protocol or numerical behavior.
- Command: post-cleanup balanced production-width correctness on `/dlwh/mok-bf-011-cleanup-smoke-20260810-1935`, four GB200s, no retry.
- Result: the compacted native source compiled and loaded, and the task succeeded in 40.12 seconds. Forward and every input, combine/router, routed-weight, and shared-weight gradient passed. The run executed four forward/four backward handlers with eight real macrobuffers and zero drops. Both slots were exercised on every rank, generation mismatches and reuse failures were zero, and the compact 55-counter wait telemetry was internally consistent. Source search found no remaining `PeerBarrier`, `peer_barrier`, or `LaunchPeerBarrier` symbol in the supported package.
- Interpretation: milestone one is complete. The supported staged backend is barrier-free, survives concurrent VJPs and 100 training updates, and no longer carries the legacy peer-barrier implementation. Generation readiness, completion publication, stamp validation, and the unrelated ThunderKittens intra-kernel `barrier_arrive` remain live.
- Next action: keep this staged implementation as the supported fallback while the collective-memory zero-copy work proceeds as a separate feasibility probe.

### 2026-08-10 19:47 PT - MOK-ZC-001 XLA collective-memory probe passes with boundary copies

- Hypothesis: JAX 0.11 can lower a typed FFI whose operand and destination-passing results are assigned to OpenXLA collective memory, and the native handler can directly read/write those XLA-owned buffers from peer GPUs for the operation lifetime.
- Commands: `/dlwh/mok-zc-001-single-20260810-1947`, concurrent follow-up `/dlwh/mok-zc-002-concurrent-20260810-1949`, and invalid-color control `/dlwh/mok-zc-003-invalid-color-20260810-1949`, all on four GB200s with no retry.
- Result: the single and concurrent jobs succeeded. StableHLO retained `operands_memory_spaces={0:1}` and `results_memory_spaces={0:1,1:1}`. Optimized HLO assigned the typed-FFI operand and both results to `S(1)`. Every rank exactly read its next peer's operand and wrote the expected sentinel into its next peer's result. Two concurrent invocations of the same compiled executable used distinct inputs and both remained exact without pointer mixing or deadlock. Color 99 failed promptly at compilation with `Invalid memory space 99`, before native execution.
- Negative result: optimized HLO inserted three boundary copies: default input to `S(1)`, followed by both `S(1)` results back to default memory. Buffer-assignment dumps confirmed the colored call. The probe proves explicit peer visibility and operation-scoped access, but not end-to-end copy elimination.
- Interpretation: adding existing memory-space attributes to the full MoK FFI would move staging traffic into XLA rather than remove it. Production zero-copy remains blocked on an upstream JAX/OpenXLA contract that either propagates collective memory across surrounding producers/consumers or guarantees peer-visible ordinary device allocations without changing memory space. The supported staged backend remains unchanged.
- Next action: propose the upstream buffer/lifetime contract in `.agents/projects/jax_ffi_peer_visible_buffers/`; do not integrate the colored probe into normal Grug until optimized HLO and Nsight show the identified copies are absent.

### 2026-08-10 22:55 PT - MOK-ZC-004 default-space peer access is copy-free on the pinned runtime

- Hypothesis: the pinned JAX 0.11/OpenXLA CUDA allocator already maps ordinary memory-space-zero buffers for peer access, allowing the same direct FFI ring without collective-memory boundary copies.
- Command: `/dlwh/mok-zc-004-default-space-20260810-2255`, four GB200s, memory space zero, single execution followed by two concurrent invocations of the same compiled executable, no retry.
- Result: the job succeeded in 16.99 seconds. The native typed FFI compiled and loaded. Every single and concurrent remote read/write matched exactly, distinct concurrent inputs remained isolated, and no invocation deadlocked. Optimized HLO made the custom call the root in memory space zero and reported `copy_line_count=0`, `copy_lines=[]`, and `zero_copy=true`.
- Interpretation: symmetric rank-relative addresses are unnecessary because MoK already accepts one pointer per peer. The supported pin can directly dereference each exchanged ordinary XLA pointer under CUDA UVA/VMM. This is proven runtime behavior, not yet a public JAX semantic guarantee, so production use remains an explicit experimental mode with staged fallback.
- Next action: remove only the forward `x` staging copy first, using invocation-local XLA pointer exchange plus the already-proven readiness/completion lifetime protocol; validate numerics/concurrency before touching backward staging.

### 2026-08-10 23:05 PT - MOK-ZC-005/006 forward-activation zero-copy passes native and concurrency gates

- Hypothesis: the normal MoK forward can consume the XLA-owned activation directly, using invocation-local peer pointer exchange and the existing readiness/completion protocol, without retaining symmetric addresses or persistent XLA pointers.
- Commands: `/dlwh/mok-zc-005-forward-x-correctness-20260810-2310` and `/dlwh/mok-zc-006-forward-x-vjp-20260810-2310`, four GB200s, `forward_x_storage=xla_peer_experimental`, production intermediate width 3,072, eight real macrobuffers, offloaded saved context, no retry.
- Result: both attempts succeeded. The first passed forward, `dx`, combine/router, all routed weights, and all shared weights with four forward/four backward handlers and zero drops or protocol anomalies. The second passed the primary tree, two dependent calls in one executable, and two concurrent executions of the same compiled VJP. Dependent and concurrent probes each executed eight forward/eight backward handlers, used both physical slots on every rank, and passed every gradient tree with zero generation mismatch or slot-reuse failure.
- Interpretation: direct ordinary-space XLA pointers remain valid for the full remote-read lifetime, including rematerialized saved context and concurrent RunIds. Equal virtual addresses across ranks are not required; the lease carries one peer-dereferenceable pointer per owner rank.
- Next action: remove the four remaining backward boundary copies as one atomic storage mode and measure forward-only throughput against the staged barrier-free control.

### 2026-08-10 23:10 PT - First forward-only throughput launch catches an inconsistent edit snapshot

- Command: `/dlwh/mok-zc-007-forward-x-100-coord-20260810-2315`, 100-step normal Grug run with direct forward and staged backward.
- Result: deterministic failure before step zero: the Python FFI call supplied the new `backward_peer_storage` attribute while the bundled native binding still expected the prior five-attribute schema. W&B initialized with the intended config but recorded no training metric. No retry was attempted.
- Interpretation: the coordinator bundle was created while the shared worktree's backward implementation was being edited. This is a source-snapshot race, not a numerical or runtime failure. Subsequent bundles must be created only after local schema checks and native staged smoke are complete.
- Next action: finish the backward edit, run one clean staged schema smoke, then launch paired throughput arms from a stable bundle.

### 2026-08-10 23:17 PT - MOK-ZC-008/009 all-direct forward/backward passes native correctness

- Hypothesis: XLA-owned `x`, `d_y`, combine weights, and router-gradient output can replace the runtime staging buffers when all four rank handlers exchange pointers and sizes per invocation; destination readiness prevents remote router writes before the local output clear completes.
- Commands: staged schema smoke `/dlwh/mok-zc-008-staged-schema-smoke-20260810-2330` and all-direct treatment `/dlwh/mok-zc-009-all-direct-correctness-20260810-2330`, four GB200s, production intermediate width 3,072, eight real macrobuffers, offloaded context, no retry.
- Result: both attempts succeeded in 40.26 seconds. The clean worker compiled and loaded the six-attribute backward binding. Staged/staged and direct/direct each passed forward and every input, combine/router, routed-weight, and shared-weight gradient with exact four/four FFI counts, zero drops, and zero generation/reuse anomalies. The direct treatment removes the five identified D2D boundary copies: 2.251953 GiB/layer/GPU, or 108.09375 GiB/GPU per 48-layer step.
- Interpretation: runtime-owned `combine` and `d_x_routed` remain because they are peer-written routed intermediates, not avoidable staging. Direct XLA pointers are held only in the invocation lease and never stored in the persistent runtime.
- Next action: run dependent/concurrent all-direct VJPs, route-edge destination readiness, and paired 100-step throughput arms.

### 2026-08-10 23:22 PT - MOK-ZC-010 all-direct dependent and concurrent VJPs pass

- Command: `/dlwh/mok-zc-010-all-direct-vjp-20260810-2340`, direct forward and backward peer storage, dependent back-to-back VJP plus two concurrent executions, no retry.
- Result: attempt zero succeeded in 40.91 seconds. Primary counts were four/four; dependent and concurrent counts were eight/eight. Every gradient in the primary, dependent, and both concurrent trees passed. Both slots were active on every rank; generation mismatch and slot-reuse failure counts were zero.
- Interpretation: direct backward inputs and router outputs are isolated across both static call sites and concurrent RunIds. The treatment shows no pointer-table cross-talk.
- Next action: complete route-edge gates and score the profile-free throughput windows for forward-only and fully direct treatments.

### 2026-08-10 23:27 PT - Paired zero-copy throughput arms launched

- Commands: forward-only `/dlwh/mok-zc-013-forward-only-100-coord-20260810-2355` and all-direct `/dlwh/mok-zc-014-all-direct-100-coord-20260810-2355`, normal 48-layer Grug, four GB200s, 100 updates, no retry, profiler steps 80-84. Both use the same seed, data, model, capacity factor 1.1, offloaded MoE context, and staged barrier-free baseline.
- Baseline: `mok-bf-010-zero-barrier-100-s80n5-r1-20260810-1900`; steps 60-79 averaged 20,966.88 raw and 20,935.97 drop-adjusted tokens/s, 12.50305 seconds, 24.52169% MFU, and 0.14799% drops.
- Status: both coordinators were accepted. Score steps 60-79 primarily and 40-79 secondarily; exclude profile steps 80-84.
- Next action: validate terminal status, W&B identity, finite loss, profiler upload, runtime counters, and exact paired deltas before promoting the direct mode.

### 2026-08-10 23:48 PT - Forward-only zero-copy wins narrowly; all-direct backward regresses

- Commands: forward-only `/dlwh/mok-zc-013-forward-only-100-coord-20260810-2355` and all-direct `/dlwh/mok-zc-014-all-direct-100-coord-20260810-2355`, each 100 updates with profiler steps 80-84 and no retry.
- Result: both attempts succeeded with 100 finite losses and zero generation/reuse anomalies. Forward-only steps 60-79 averaged 21,074.51 raw and 21,023.83 drop-adjusted tokens/s, 12.43936 seconds, and 24.64757% MFU: +0.513% raw, +0.420% adjusted, and -0.509% step time versus staged. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-zc-013-forward-only-100-s80n5-r1-20260810-2355); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-zc-013-forward-only-100-s80n5-r1-20260810-2355).
- Negative result: all-direct averaged 20,463.28 raw and 20,419.59 adjusted tokens/s, 12.81084 seconds, and 23.93272% MFU. It regressed 2.402% raw versus staged and 2.900% versus forward-only. Backward-pre wait events rose 11.04%, total cycles 26.63%, and mean cycles per event 14.04% versus forward-only; forward/post-completion phases were nearly unchanged. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-zc-014-all-direct-100-s80n5-r1-20260810-2355); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-zc-014-all-direct-100-s80n5-r1-20260810-2355).
- Interpretation: eliminating copies is not itself a performance objective. Direct router-gradient output requires extra per-destination readiness observations that overwhelm the 1 MiB/rank output copy it removes. Keep the large XLA inputs direct but stage the small router-gradient output.
- Next action: benchmark the hybrid mode that removes forward `x` plus backward `x`, `d_y`, and router input copies while retaining one 1 MiB router-gradient output copy per layer/GPU.

### 2026-08-10 23:49 PT - MOK-ZC-015 hybrid backward correctness and copy telemetry pass

- Command: `/dlwh/mok-zc-015-hybrid-vjp-20260810-0010`, direct forward input plus direct backward inputs/staged router-gradient output, dependent and concurrent VJPs, no retry.
- Result: attempt zero succeeded in 45.76 seconds. Every primary, dependent, and concurrent gradient tree passed with exact four/four and eight/eight native counts, both slots active, zero drops, and zero generation/reuse anomalies. Primary copy telemetry was exact: zero forward staging calls/bytes; one 8,192-byte backward router-output copy per rank, or four calls/32,768 bytes across the API invocation.
- Interpretation: the hybrid path removes the 108.09 GiB/GPU/step staging budget except 48 MiB/GPU/step of router-gradient output copies, without the direct-output readiness path implicated in the all-direct regression.
- Next action: run 100 steps with the same scoring window, then promote or reject it based on measured throughput and emitted copy counters.

### 2026-08-11 00:04 PT - Strict factor-four capacity is feasible and dropless in a one-layer screen

- Hypothesis: the current static implementation can provision the four-rank worst-case receiver schedule without OOM, establishing a semantic dropless control before weight-gradient accumulation is optimized.
- Commands: factor 1.1 `/dlwh/mok-dl-001-cap11-1l-2step-coord-20260811-0015` and factor 4 `/dlwh/mok-dl-002-cap4-1l-2step-coord-20260811-0015`, one normal Grug layer, two updates, direct forward plus hybrid backward, no retry.
- Result: both attempts succeeded. Factor 1.1 derived capacity 290,816/9 static macrobuffers; its warm step ran at 34,773.52 tokens/s and dropped 18,826 assignments (1.79539%). Factor 4 derived capacity 1,052,672/33 static macrobuffers; its warm step ran at 32,979.76 tokens/s with zero drops, a provisional 5.158% throughput cost. Both reported about 174.6 GiB peak HBM/GPU and no OOM; the factor-four run retained about 10.43 GiB observed headroom. [factor 1.1 W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-001-cap11-1l-2step-20260811-0015); [factor 4 W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-002-cap4-1l-2step-20260811-0015).
- Interpretation: strict dropless capacity is operationally feasible on GB200, but the one-step performance estimate is too noisy and the current factor-four shape inflates routed FP32 weight-gradient partials from 3.797 to 13.922 GiB/GPU. This is a semantic control, not yet the final capacity-independent implementation.
- Next action: run the factor-four hybrid configuration for 100 steps, require zero drops throughout, and score steps 60-79 against the factor-1.1 hybrid run before deciding whether bounded online weight-gradient accumulation is required.

### 2026-08-11 00:10 PT - Static factor-four capacity cannot train the 48-layer model

- Command: `/dlwh/mok-dl-003-hybrid-cap4-100-coord-20260811-0025`, normal 48-layer Grug, direct forward plus hybrid backward, strict schedule capacity factor four, 100 requested updates, no retry.
- Result: deterministic failure before step zero. XLA reported a 301.67 GiB program after rematerialization against a 185.45 GiB target, then all four ranks failed `ncclCuMemAlloc` with CUDA out-of-memory. W&B initialized but recorded no step, loss, throughput, drop, profile, or runtime counter. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-003-hybrid-cap4-100-s80n5-r1-20260811-0025).
- Interpretation: the one-layer feasibility result does not extend to the full model. The capacity-dependent backward result exposes three FP32 routed-weight partial tensors for every static macrobatch; factor four grows them from 3.797 to 13.922 GiB per active layer. The full compiled program is 116.22 GiB above the rematerialization target. Increasing the allocator fraction cannot repair this shape.
- Next action: make routed FP32 weight-gradient accumulation capacity-independent, returning canonical per-expert gradients rather than one full tensor per static macrobatch. Re-run strict dropless correctness and training only after that change.

### 2026-08-11 00:13 PT - Multi-process local-EP4 scaling contract is locally implemented

- Result: the runtime now permits multiple JAX processes only when each process owns exactly four local CUDA GPUs. Concrete mesh validation requires each four-device expert group to be contained in one process, ordered by local hardware IDs zero through three, with exactly one expert group per process and mesh process IDs matching the JAX world. The launcher supports 1, 2, 16, and 32 nodes with global batch `64 * num_nodes`, one process per four-GPU node, local EP4, and data parallelism across nodes. Profiling is restricted to process zero.
- Runtime seal: successful training resets native call/debug counters immediately before the loop, gathers compact summaries from every process, and rejects any process whose forward/backward handler count differs from `remaining_steps * layers * 4` or reports a generation mismatch or slot-reuse failure. This prevents process-zero-only telemetry from masking a bad node.
- Local evidence: 69 focused tests passed after the distributed runtime audit was added; focused format, lint, and Pyrefly checks passed.
- Next action: do not launch rack-scale work until the single-node hybrid/readiness and dropless candidates are resolved. Then run the ordered two-node, one-rack, and two-rack weak-scaling screens from a clean committed source snapshot.

### 2026-08-11 00:16 PT - Hybrid zero-copy is safe but does not beat forward-only

- Command: `/dlwh/mok-zc-016-hybrid-100-coord-20260811-0010`, normal 48-layer Grug, direct forward `x`, direct backward `x`/`d_y`/router input, staged router-gradient output, 100 updates, no retry, profiler steps 80-84.
- Result: coordinator and child succeeded with 100 finite losses; final loss was 5.31303. Steps 60-79 averaged 20,999.04 raw and 20,966.35 drop-adjusted tokens/s, 12.48395 seconds, 24.55930% MFU, and 0.15647% drops. This is +0.153% raw/+0.145% adjusted versus staged, but -0.358% raw/-0.273% adjusted versus forward-only. Steps 40-79 were effectively tied with staged (+0.050% raw/+0.018% adjusted) and 0.430% below forward-only raw. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-zc-016-hybrid-100-s80n5-r1-20260811-0010); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-zc-016-hybrid-100-s80n5-r1-20260811-0010).
- Copy evidence: forward staging was exactly zero. Backward staged 4,800 one-MiB router-gradient outputs per rank, totaling 18.75 GiB across four ranks over the run. Generation mismatch and slot-reuse failures were zero.
- Wait diagnosis: hybrid backward-ready waits were 27.78M events/2.511T cycles/90,360 mean cycles. Versus forward-only, events fell 7.36% but total cycles rose 2.88% and mean cycles rose 11.05%. The large input-copy savings are offset by exposing producer readiness.
- Interpretation: retain forward zero-copy as the measured best path unless readiness publication recovers the hybrid stalls. Keep the router-gradient output staged; all-direct remains clearly worse.
- Next action: measure the already-correct readiness-order treatment, which publishes backward peer-ready state before clearing local-only routed-weight partials.

### 2026-08-11 00:28 PT - Canonical FP32 routed-weight accumulation passes 8- and 32-macro gates

- Hypothesis: routed weight gradients can be accumulated deterministically inside the native kernel into canonical per-expert FP32 outputs, removing the schedule-capacity dimension from the XLA result without reverting to BF16 accumulation.
- Implementation: backward FFI routed gate/up/down results are now rank-three canonical FP32 arrays. The generated MoK epilogue uses its existing first-contribution predicate and macrobatch serialization: the first contribution TMA-stores FP32, then later macrobatches use FP32 TMA reduction-add into the same expert tile. The Python-side macrobatch `sum(axis=0)` is removed; inactive experts retain exact zero through the native clear and explicit mask. Native build schema is v8.
- Footprint: routed gradient outputs are 432 MiB/GPU/layer invocation at the production shape, independent of capacity. This replaces 3.797 GiB at factor 1.1 and 13.922 GiB at factor four.
- Commands: balanced dependent/concurrent VJP `/dlwh/mok-dl-004-canonical-wgrad-vjp-20260811-0028` and route matrix `/dlwh/mok-dl-005-canonical-wgrad-routes-20260811-0031`, four GB200s, no retry.
- Result: fresh native compilation succeeded. Balanced eight-macro primary, dependent, and concurrent VJPs passed every leaf with exact four/four and eight/eight handler counts, both slots active, and zero protocol anomalies. Zero-token routing passed every leaf with inactive routed gate/up/down gradients exactly zero. All-to-one used exact no-overflow capacity 8,192, exercised 32 real macrobuffers, and passed every leaf. Both route cases had zero drops and exact hybrid copy telemetry.
- Interpretation: canonical FP32 accumulation is numerically correct across ordinary, empty-expert, heavily skewed, dependent, and concurrent execution. The remaining question is full-model memory and throughput at strict dropless capacity.
- Next action: repeat the one-layer factor-four training screen, then retry 48-layer strict-dropless training only if the compiled/HBM footprint is reduced as predicted.

### 2026-08-11 00:42 PT - Two-node local-EP4 weak-scaling screen passes

- First attempt: `/dlwh/mok-scale-001-2node-1l-2step-coord-20260811-0022` failed before runtime initialization because distributed JAX exposed `local_hardware_id=None` on process-one devices. The preflight was corrected to compare each expert group with JAX's canonical process-filtered device ordering rather than optional hardware metadata.
- Corrected command: `/dlwh/mok-scale-002-2node-1l-2step-coord-20260811-0032`, two tasks with four GB200s and one JAX process each, local EP4/DP2, one model layer, two updates, forward-only zero-copy, native v8, no retry.
- Result: parent and both child tasks succeeded. Process zero owned device IDs zero through three; process one owned four through seven despite `local_hardware_id=None`. Both tasks independently built/loaded the process-local runtime. Loss was finite at 11.8080 then 10.5590; the warm second update reached 67,753.6 tokens/s. Process-zero peak observed HBM was 176.02 GiB. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-scale-002-2node-1l-2step-20260811-0032).
- Distributed seal: gathered runtime summaries reported exactly eight forward/eight backward handlers on both processes, matching two steps times one layer times four local ranks. `processes_with_protocol_errors` was zero, and every generation-mismatch and slot-reuse count was zero.
- Interpretation: native pointer exchange remains strictly process-local while JAX data-parallel gradients cross nodes. The concrete mesh and all-process audit work in a real distributed client. This is a correctness/compile screen, not a steady scaling score.
- Next action: after the final single-node candidate is chosen, run 25 updates on two nodes before advancing to one rack.

### 2026-08-11 00:46 PT - Earlier backward readiness removes waits but not step time

- Command: `/dlwh/mok-zc-018-hybrid-readiness-100-coord-20260811-0015`, normal 48-layer Grug, hybrid storage, 100 updates, profiler steps 80-84, no retry. The treatment publishes peer-visible backward inputs after their true prerequisites and before clearing local-only routed-gradient outputs.
- Result: all 100 losses were finite; final loss was 5.31006. Steps 60-79 averaged 20,973.29 raw and 20,944.34 adjusted tokens/s, 12.50031 seconds, 24.52919% MFU, and 0.13885% drops. This is only +0.031% raw versus staged, -0.123% versus the prior hybrid, and -0.480% versus forward-only. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-zc-018-hybrid-readiness-100-s80n5-r1-20260811-0015); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-zc-018-hybrid-readiness-100-s80n5-r1-20260811-0015).
- Counter result: backward-ready polling fell from 27.78M events/2.511T cycles to exactly zero, but step time did not move. Handler counts were exactly 19,200/19,200; distributed protocol errors were zero; hybrid copy telemetry remained exact at 18.75 GiB total staged router-gradient output over the run.
- Interpretation: the observed backward-ready waits were overlapped CTA work, not critical-path latency. Eliminating a counter is not a performance result. Forward-only zero-copy remains the best measured storage mode; the canonical-gradient v8 program needs a fresh apples-to-apples 100-step measurement before promotion.
- Next action: score the v8 factor-1.1 forward-only run, then use it as the control for efficient strict-dropless training.

### 2026-08-11 00:52 PT - Canonical factor-four one-layer training is dropless and lighter

- Command: `/dlwh/mok-dl-006-cap4-canonical-1l-2step-coord-20260811-0039`, one normal Grug layer, two updates, schedule factor four, native v8 canonical FP32 accumulation, direct forward plus hybrid backward, no retry.
- Result: parent and child succeeded. Both losses were finite, 11.80820 to 10.74407, and both steps had exactly zero dropped assignments. Runtime audit reported exactly eight forward/eight backward handlers, zero protocol-error processes, and zero generation/reuse errors. Warm step one reached 33,932.52 tokens/s in 7.72545 seconds: +2.889% throughput versus the old factor-four implementation and -2.419% versus factor 1.1, whose corresponding step dropped 1.795% of assignments. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-006-cap4-canonical-1l-2step-20260811-0039).
- Memory/copy evidence: peak observed HBM was 172.35 GiB, 2.25 GiB below the prior factor-four run. Forward staging was zero; backward copied exactly one one-MiB router-gradient output per rank/step, eight MiB total. A synchronous remote autotune-cache fetch added 8.5 minutes to startup but was outside execution.
- Interpretation: capacity-independent accumulation removes the factor-four result-memory blowup and improves its kernel/program cost, while strict capacity still pays additional schedule work. One warm step is not a throughput conclusion.
- Next action: run the full 48-layer two-update memory/correctness screen. Only a finite, zero-drop result without OOM justifies the 100-step treatment.

### 2026-08-11 01:09 PT - Canonical accumulation makes full-model strict capacity executable

- Command: `/dlwh/mok-dl-008-cap4-canonical-48l-2step-coord-20260811-0054`, normal 48-layer Grug, factor-four capacity, direct forward plus hybrid backward, native v8, two updates, no retry.
- Result: parent and child succeeded. Both losses were finite, 11.80647 to 10.71523, and both steps had zero drops. The warm update reached 11,892.91 tokens/s in 22.0420 seconds at 13.9093% MFU. Runtime audit reported exactly 384 forward/384 backward handlers, zero protocol-error processes, and zero generation/reuse errors. Peak observed HBM was 174.64 GiB/GPU. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-008-cap4-canonical-48l-2step-20260811-0054).
- Memory result: XLA's post-rematerialization estimate fell from 301.67 GiB in the old factor-four program to 288.94 GiB. The old program failed before step zero; the canonical program completed both full-model updates. Hybrid copy telemetry remained exact at zero forward staging and one one-MiB backward router-gradient copy per rank/layer/update.
- Interpretation: capacity-independent routed-weight results remove the deterministic full-model memory blocker. Strict factor four still costs about 3.6% throughput on the single warm full-model update, although that comparison is storage-mode-confounded and not stable enough for promotion.
- Next action: establish a stable 100-step native-v8 factor-1.1 control, then compare factor four using the same allocator, storage mode, seed, and profile window.

### 2026-08-11 01:16 PT - Native v8 factor-1.1 hits a repeated-step allocator cliff

- Command: `/dlwh/mok-dl-007-cap11-canonical-forward-100-coord-20260811-0046`, normal 48-layer Grug, factor 1.1, direct forward plus staged backward, native v8, 100 requested updates, profiler scheduled at steps 80-84, no retry.
- Result: 48 losses were finite through global step 47, ending at 6.02137, then all four GPUs failed within one millisecond on the same 132,687,700,848-byte (123.575 GiB) `jit_train_step` allocation. The cuda-async pool reported only 31.383 GiB live and 156.625 GiB reserved. Request plus live memory was 154.958 GiB, 1.667 GiB below the reservation, but the allocation still failed. No profile step had started. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-007-cap11-canonical-forward-100-s80n5-r1-20260811-0046).
- Negative evidence: step times stayed flat; watch logging was disabled; runtime counters are read outside the loop; and v8's post-rematerialization estimate is 2.60 GiB smaller than the successful pre-v8 forward-only run. Factor-1.1 and factor-four v8 estimates differ by only 24 bytes. This is not a monotonic live-buffer leak or the prior true capacity-wall signature.
- Interpretation: the evidence points to cuda-async pool reuse/fragmentation or an async output-lifetime cliff around one giant fixed-shape temporary. The partial steps 40-47 averaged 21,077.83 raw and 21,021.14 adjusted tokens/s, but do not constitute a stable score.
- Next action: run 55 otherwise-identical updates at fraction 0.85 with profiling disabled and explicitly await the complete `(state, metrics)` result after every update. Crossing step 48 isolates stream-ordered reuse; recurrence justifies a narrowly tagged 0.86 allocator-headroom test.

### 2026-08-11 01:20 PT - Full-step completion allocator discriminator launched

- Command: `/dlwh/mok-dl-009-cap11-v8-fullsync-55-coord-20260811-0119`, factor 1.1, native v8, direct forward plus staged backward, allocator fraction 0.85, 55 updates, `train_step_completion_mode=state_and_metrics`, profiler disabled, no retry.
- Contract: allocator fraction, XLA graph, model, data, seed, storage mode, and capacity match the failed control; only the host completion boundary changes. The launcher records both the completion mode and allocator fraction in the fingerprint and W&B tags.
- Decision rule: if update 49 and terminal update 55 complete, broad asynchronous state/metric reuse caused the prior cliff. If the exact allocation failure recurs, keep normal loss-only completion and test fraction 0.86 as an empirical headroom workaround rather than a semantic fix.

### 2026-08-11 01:37 PT - Full-step completion does not move the allocator cliff

- Result: `/dlwh/mok-dl-009-cap11-v8-fullsync-55-coord-20260811-0119` reproduced the failure at the same boundary. Forty-seven progress updates were durable and finite through loss 5.98. The next update failed on the identical 132,687,700,848-byte allocation with the same 31.38 GiB live use, 156.65 GiB allocator limit, 154.96 GiB high-water, and 156.63 GiB pool reservation. Profiling was disabled and the code awaited the complete `(state, metrics)` result after every update.
- Interpretation: in-flight state or metric outputs do not cause the failure. The repeated fixed-shape executable reaches the same cuda-async pool cliff even when each prior result is fully complete. The exact recurrence also makes the diagnostic completion mode unsuitable as a fix.
- Next command: `/dlwh/mok-dl-010-cap11-v8-mem86-55-coord-20260811-0137`, restoring normal loss-only completion and changing only the XLA allocator fraction from 0.85 to 0.86. The 1.843 GiB larger pool is narrowly above the observed shortfall while retaining about 7.84 GiB for NCCL, native workspaces, and profiling. The run remains a discriminator; a pass is empirical headroom, not proof that allocator fragmentation is eliminated.

### 2026-08-11 02:00 PT - Allocator fraction 0.86 crosses the repeated-step cliff

- Command: `/dlwh/mok-dl-010-cap11-v8-mem86-55-coord-20260811-0137`, otherwise identical native-v8 factor-1.1 forward-zero-copy training, normal loss-only completion, profiler disabled, allocator fraction 0.86, 55 updates, no retry.
- Result: coordinator and child succeeded with all 55 losses finite; final loss was 5.82470. The run crossed the prior update-48 boundary and completed. Steps 50-54 averaged 21,431.53 tokens/s, 12.23176 seconds, 25.06513% MFU, and zero drops. Runtime audit reported exactly 10,560 forward/10,560 backward handlers, zero generation mismatches, zero slot-reuse failures, and zero protocol-error processes. Forward staging remained zero. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-010-cap11-v8-mem86-55-20260811-0137).
- Interpretation: a 1.843 GiB larger cuda-async pool is sufficient to avoid the fixed 123.575 GiB allocation cliff, while broad result synchronization was not. This is a narrow empirical headroom workaround rather than a proof that the allocator's fragmentation/reuse behavior is repaired. Fraction 0.90 remains unsafe because it would leave almost no non-pool headroom.
- Next command: `/dlwh/mok-dl-011-cap11-v8-mem86-100-coord-20260811-0202`, 100 updates with the normal profile at steps 80-84. Require terminal success, stable steps 60-79, exact runtime audit, and a completed XProf before promoting 0.86 or launching strict factor four.

### 2026-08-11 02:29 PT - Allocator fraction 0.86 delays but does not eliminate the cliff

- Command: `/dlwh/mok-dl-011-cap11-v8-mem86-100-coord-20260811-0202`, native-v8 factor-1.1 forward-zero-copy training, normal loss-only completion, cuda-async allocator fraction 0.86, 100 requested updates, no retry.
- Result: the run crossed the prior update-48 boundary but failed after 63 finite updates on the identical 132,687,700,848-byte allocation. The allocator limit rose to 158.50 GiB; live use remained 31.38 GiB. Partial steps 40-62 averaged 21,068.93 raw and 21,023.48 drop-adjusted tokens/s, 12.44243 seconds, 24.64104% MFU, and 0.21570% drops. The required steps 60-79 and profile 80-84 were not reached. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-011-cap11-v8-mem86-100-s80n5-r1-20260811-0202).
- Interpretation: increasing the cuda-async fraction changes when fragmentation/reuse fails but does not make the repeated fixed-shape executable stable. Further fraction tuning is low-information and consumes the HBM needed by the native workspaces and NCCL.
- Next action: test OpenXLA's VMM allocator, which allocates physical pages per logical buffer instead of requiring the giant request to fit a fragmented cuda-async pool.

### 2026-08-11 02:53 PT - VMM avoids pool fragmentation but exposes exact physical-budget pressure

- Command: `/dlwh/mok-dl-012-cap11-v8-vmm85-55-coord-20260811-0253`, same factor-1.1 forward-zero-copy program, VMM allocator at fraction 0.85, 55 requested updates, no retry.
- Result: the run failed before step zero on VMM's explicit physical-allocation budget. Existing physical allocation was 73,431,777,280 bytes, the next request was 106,302,537,728 bytes, and the budget was 168,206,517,665 bytes. The exact 179,734,315,008-byte requirement implies a minimum fraction of 0.908253581. No training metric, profile, or runtime audit was produced. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-012-cap11-v8-vmm85-55-20260811-0253).
- Interpretation: the VMM path is active and converts the opaque repeated-step cuda-async failure into a deterministic up-front capacity check. Two native workspace slots consume about 15.004 GiB/GPU, leaving too little non-XLA headroom even at fractions 0.91-0.92.
- Next action: allocate one production workspace slot. Completed serial training runs never acquired slot one and reported maximum active slots one; retain two slots only for explicit concurrent-call correctness stress.

### 2026-08-11 03:09 PT - One-slot native schema v9 saves 7.502 GiB/GPU and passes correctness

- Implementation: `MokLikeConfig.workspace_slots` now selects one or two active native slots. Runtime initialization compatibility, allocation, peer pointer setup, acquisition, slot telemetry, saved-stamp validation, and teardown honor the active count; fixed host arrays retain capacity two. Production launcher default is one, while the correctness harness defaults to two and rejects concurrent-call stress with one. Native schema is v9.
- Footprint: removing one production slot saves 8,055,161,448 bytes, or 7.50195 GiB/GPU, at 65,536 local tokens, hidden width 6,144, and top-k four. Native workspace storage falls from about 15.0039 to 7.50195 GiB/GPU.
- Local evidence: 78 focused tests passed; targeted Ruff, Black, Pyrefly, license, AST, conflict, and whitespace checks passed; pinned native source generation succeeded.
- Command: `/dlwh/mok-zc-021-vmm91-one-slot-correctness-20260811-0309`, four GB200s, VMM fraction 0.91, one slot, direct forward and staged backward, balanced production intermediate width, offloaded saved context, no retry.
- Result: schema-v9 native compilation/loading and the five-integer runtime initialization succeeded. Forward and all nine gradient/router leaves passed; counts were four forward/four backward, eight real macrobuffers, and zero drops. Every rank acquired only slot zero twice, maximum active slots were one, and generation mismatch/reuse failure counts were zero.
- Interpretation: one-slot mode is an actual HBM reduction with the same serial training semantics, not a launcher-only setting. Two-slot dependent/concurrent VJP coverage remains independently proven.
- Next command: `/dlwh/mok-dl-013-vmm91-one-slot-55-20260811-0312-coord`, factor 1.1, VMM fraction 0.91, one slot, 55 updates. Require step-48 crossing, terminal success, exact handler/audit counters, and comparable throughput before a 100-step seal.

### 2026-08-11 03:15 PT - Full training rules out VMM even with one workspace slot

- Command: `/dlwh/mok-dl-013-vmm91-one-slot-55-20260811-0312-coord`, normal 48-layer factor-1.1 training, one native slot, VMM fraction 0.91, 55 requested updates, no retry.
- Result: native v9 compiled, but the first training update failed before producing a metric. VMM had already allocated 179,736,412,160 bytes (167.393 GiB) when the executable requested another 132,688,904,192 bytes (123.576 GiB); the 0.91 budget was 180,079,919,050 bytes. Even fraction one cannot hold the additive 291 GiB requirement. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-013-vmm91-one-slot-55-20260811-0312).
- Interpretation: VMM does not reuse the full-program physical allocations in the way this graph requires. It is unsuitable for production training regardless of the saved native slot, although it remains numerically compatible with the small correctness graph. Do not test 0.92 or 1.0.
- Next command: `/dlwh/mok-dl-014-cuda85-one-slot-55-20260811-0328-coord`, returning to cuda-async fraction 0.85 with one slot and no other change. All prior cuda-async cliff runs used two slots, so this is the smallest remaining HBM/remapping discriminator. If it fails, test XLA's built-in separate temp-buffer color with default-pool preallocation disabled before adding a custom trim hook.

### 2026-08-11 03:39 PT - One native slot does not repair cuda-async pool reuse

- Command: `/dlwh/mok-dl-014-cuda85-one-slot-55-20260811-0328-coord`, native-v9 factor-1.1 training, direct forward plus staged backward, one workspace slot, cuda-async fraction 0.85, 55 requested updates, profiler steps 5-9, no retry.
- Result: 34 metric rows were finite through global step 33, then the executable again failed on the exact 132,687,700,848-byte allocation. The pool reported 31.383 GiB live and 156.625 GiB reserved, so nominal unused pool storage exceeded the request by 1.667 GiB, while physical device free space was 17.188 GiB. The failed process wedged in downstream rendezvous and was stopped after evidence capture. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-014-cuda85-one-slot-55-20260811-0328); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-dl-014-cuda85-one-slot-55-20260811-0328).
- Partial performance: steps 20-33 averaged 21,022.29 tokens/s, 12.47094 seconds, and 24.5865% MFU. These partial rows are consistent with the prior forward-only result but are not a stable score.
- Interpretation: reducing native persistent HBM does not eliminate the cuda-async cliff. The unchanged request/live/reserved signature continues to implicate default-pool fragmentation or stream-ordered cached backing rather than live-tensor capacity. Failure timing is not monotonic enough to use as a fix criterion.
- Next command: `/dlwh/mok-dl-015-separate-temp-2step-20260811-0345-coord`, one slot and otherwise identical training, but with `--xla_gpu_temp_buffer_use_separate_color=true` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Pinned XLA places the executable temp heap in a dedicated cuda-async pool specifically to isolate allocation interference. Require two finite updates before the 55-step boundary gate.

### 2026-08-11 03:55 PT - A dedicated XLA temp pool fits the full graph

- Command: `/dlwh/mok-dl-015-separate-temp-2step-20260811-0345-coord`, native-v9 factor-1.1 training, one workspace slot, cuda-async fraction 0.85, separate XLA temp-buffer color, default-pool preallocation disabled, two updates, no retry.
- Result: coordinator and child succeeded. Losses 11.8065 and 10.7161 were finite; peak observed allocated HBM was 165.51 GiB/GPU. Runtime audit reported exactly 384 forward/384 backward handlers, zero protocol-error processes, zero generation/reuse errors, and only slot-zero acquisitions with maximum active slots one. Forward staging was zero. The warm update reached 9,194.09 tokens/s in 28.512 seconds, but two updates are compile/warmup dominated and not a throughput score.
- Interpretation: isolating the compiled temp heap in XLA's dedicated cuda-async pool removes the immediate allocation failure without changing model work or native pointer semantics. It must still cross repeated-use boundaries and recover steady throughput before promotion.
- Next command: `/dlwh/mok-dl-016-separate-temp-55-20260811-0359-coord`, identical settings for 55 updates. Require finite updates beyond both prior cuda-async failure boundaries, terminal success, exact runtime audit, and comparable profile-free throughput.

### 2026-08-11 04:24 PT - Temp-pool isolation is stable but too slow

- Command: `/dlwh/mok-dl-016-separate-temp-55-20260811-0359-coord`, native-v9 factor-1.1 training, one slot, cuda-async fraction 0.85, separate XLA temp-buffer pool, 55 updates, profiler steps 5-9, no retry.
- Result: coordinator and child succeeded. All 55 losses were finite through final loss 5.82585, crossing both prior shared-pool failure boundaries. Steps 10-54 averaged 17,298.22 tokens/s; steps 45-54 averaged 17,464.35 tokens/s in 15.0103 seconds. Peak observed allocated HBM was about 165.52 GiB/GPU. Runtime audit reported exactly 10,560 forward/10,560 backward handlers, slot zero only, maximum active slots one, and zero protocol, generation, or reuse errors.
- Interpretation: the fixed temp address/dedicated synchronous allocator eliminates the reuse cliff, confirming default-pool allocation interference as the fault domain. Its roughly 17% throughput loss versus the 21,074.51-token/s forward-only score is unacceptable for promotion.
- Implementation: native schema v10 adds a bounded default-pool trim operation. Python blocks the full update result, native maintenance excludes acquisitions, all devices synchronize, active reservations and workspace slots must be zero, and `cudaMemPoolTrimTo(pool, 0)` records exact used/reserved bytes before and after on every rank. The launcher allows one explicit completed-update boundary only in shared cuda-async mode. Ninety-four focused tests and targeted lint/type checks passed.
- Next command: `/dlwh/mok-dl-017-shared-trim25-55-20260811-0425-coord`, returning to the fast shared pool and trimming once after update 25, before the latest update-33 cliff. Require the trim to release cached backing, cross update 55, retain shared-pool throughput, and finish with exact runtime audit.

### 2026-08-11 04:42 PT - Quiescent pool trim preserves the fast path and crosses the cliff

- Command: `/dlwh/mok-dl-017-shared-trim25-55-20260811-0425-coord`, native-v10 factor-1.1 training, one slot, shared cuda-async pool at fraction 0.85, one trim after completed update 25, 55 updates, profiler steps 5-9, no retry.
- Result: coordinator and child succeeded with all 55 losses finite. Exactly one trim occurred at W&B step 24. Native telemetry reported zero active reservations and zero active workspace slots; in 1.2134 seconds it reduced each rank's pool reservation from 168,174,813,184 to 33,722,204,160 bytes while live use remained about 33,697,105,212 bytes. Training crossed updates 33 and 48 without allocator error. Tail steps 45-54 averaged 21,292.36 tokens/s in 12.3117 seconds, 21.92% faster than the separate-pool tail and consistent with the fast forward-only path. Peak allocated HBM was 167.15 GiB/GPU.
- Runtime audit: exactly 10,560 forward/10,560 backward handlers, only slot-zero acquisitions, maximum active slots one, forward staging zero, and zero protocol, generation, or reuse errors.
- Interpretation: the cuda-async failure is cached default-pool backing that cannot satisfy the next giant temporary despite nominal aggregate space. A quiescent cache reset releases that backing without changing live allocations or steady kernel work. Its measured 1.21-second cost is small enough for a periodic production treatment, unlike the 17.46k-token/s separate-pool fallback.
- Next action: replace the one-shot diagnostic surface with an explicit 25-update trim interval and run the 100-update profile seal. Require trims after updates 25, 50, 75, and 100, stable steps 60-79 including the update-75 cost, profile 80-84, terminal audit, and no OOM.

### 2026-08-11 05:15 PT - A 25-update trim interval is not conservative enough

- Command: `/dlwh/mok-dl-018-cap11-shared-trim25-100-20260811-0450-coord`, native-v10 factor-1.1 training, one slot, shared cuda-async pool at fraction 0.85, trims every 25 completed updates, 100 requested updates, profiler scheduled at 80-84, no retry.
- Result: trims after updates 25 and 50 each succeeded with zero active reservations/slots and released about 133 GiB/GPU in 1.25-1.28 seconds. Sixty-six metric rows were finite through global step 65. The next update failed on the identical 132,687,700,848-byte allocation before the third scheduled trim. Pool live use remained 31.38 GiB and reservation 155.03 GiB; final callbacks and XProf were not reached.
- Partial performance: steps 40-65, including the update-50 trim cost, averaged 20,923.19 raw and 20,883.57 drop-adjusted tokens/s in 12.5386 seconds. Steps 60-65 averaged 21,089.18 raw. These truncated windows show the trim itself retains the fast path, but cannot substitute for the required complete score.
- Interpretation: cached-backing failure can recur only 16 updates after a successful reset, so a 25-update interval lacks adequate margin. The measured 1.2-second trim cost amortizes to about 0.12 seconds/update at interval ten, roughly one percent of step time.
- Next action: rerun the same 100-update seal with interval ten. Require ten exact trim rows, terminal success, steps 60-79 including two trim costs, XProf 80-84, and the expanded all-process zero-copy/slot/trim audit.

### 2026-08-11 05:32 PT - Interval ten exposes an idle allocator/runtime wedge

- Command: `/dlwh/mok-dl-019-cap11-shared-trim10-100-20260811-0516-coord`, native-v10 factor-1.1 training with shared cuda-async, one slot, trims every ten completed updates, attempt-zero child semantics, 100 requested updates.
- Result: trims after updates 10 and 20 each released about 133 GiB/GPU with zero quiescence anomalies. Twenty-eight losses were finite through global step 27. Training then stopped advancing for more than four minutes: W&B system telemetry stayed fresh, all four GPUs were idle at zero SM utilization with about 165.5 GiB allocated, PID 1 slept, and one child process was a zombie. No OOM, traceback, or Iris task event appeared. The run was stopped with authorization and GPUs were released. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-019-cap11-shared-trim10-100-20260811-0516); [Echo incident](https://echo.oa.dev/wiki/103).
- Interpretation: the failure occurred eight updates after the second reset and before update-30 trimming. It is consistent with an allocator operation wedging instead of returning `RESOURCE_EXHAUSTED`, but the captured evidence does not prove the blocked call. A tighter interval is justified as a bounded discriminator; do not report this as an OOM.
- Next action: run 55 updates at interval five. This exercises eleven trims and caps the growth window below the observed eight-update stall. If it is stable, its measured roughly 1.2-second trim cost implies about two percent amortized step overhead and justifies one 100-update seal.

### 2026-08-11 05:57 PT - Interval five is stable across eleven pool resets

- Command: `/dlwh/mok-dl-020-cap11-shared-trim5-55-20260811-0534-coord`, native-v10 factor-1.1 training, one slot, shared cuda-async, trims every five completed updates, 55 updates, profile steps 5-9, attempt-zero semantics.
- Result: coordinator and child succeeded. All 55 losses were finite; exactly eleven trims occurred after updates 5 through 55. Every trim saw zero active reservations and workspace slots. Aggregate trim wall time was 12.8649 seconds; the eleven resets released 5,845,383,380,992 bytes across four GPUs. Training crossed the prior step-27 stall and both allocator failure boundaries. Tail steps 45-54, including two trim costs, averaged 20,322.77 raw and 20,321.18 adjusted tokens/s in 12.9303 seconds. [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-dl-020-cap11-shared-trim5-55-20260811-0534).
- Runtime audit: exactly 10,560 forward/10,560 backward handlers, only slot zero, maximum active slots one, forward staging zero, expected/actual trim count eleven, and zero trim, protocol, generation, or reuse anomalies. Peak allocated HBM was 167.13 GiB/GPU.
- Interpretation: five updates is the first cadence with repeated-run stability evidence. Its roughly 3.6% tail throughput gap versus the old forward-only score is larger than the simple 1.9% trim-time amortization because each reset also perturbs following allocation reuse.
- Next action: parallelize the four independent per-device sync/trim operations under the same native maintenance gate. Validate compilation and exact telemetry on two updates with interval one, then measure a 55/100-step interval-five seal only if wall time falls materially without correctness or allocator regressions.

### 2026-08-11 06:28 PT - Parallel per-device trim is correct but driver-limited

- Implementation: native schema v11 runs the four device synchronizations and four default-pool trim operations in joined rank-local host workers under the same maintenance/quiescence contract. Errors are captured per rank and rethrown in deterministic rank order. The public telemetry ABI and cadence surface are unchanged. One hundred nine focused tests and targeted lint/type checks passed.
- Command: `/dlwh/mok-dl-021-parallel-trim-i1-2step-20260811-0607-coord`, two full-model updates with a trim after every update, direct forward, staged backward, one slot, no retry.
- Result: both updates and both trims succeeded. Losses were finite; runtime audit reported exactly 384 handlers per phase, expected/actual trims two, forward staging zero, only slot zero, and zero trim/protocol/generation/reuse anomalies. Trim wall times were 1.0483 and 1.0004 seconds, only 12.4% below the serial 1.1695-second mean and well above the 0.50-second target.
- Interpretation: independent host dispatch removes little of the pause; CUDA physical-pool release is effectively driver-limited. The optimization is correct and modestly useful, but it does not change the interval-five stability requirement or eliminate maintenance overhead.
- Next action: run one native-v11 100-update factor-1.1 control at interval five. This is the matched control for strict factor four. Keep the earlier v7 forward-only 21,074.51-token/s run as the best non-dropless production score rather than conflating it with the canonical dropless program's allocator treatment.

### 2026-08-11 07:03 PT - Cadence-five trimming does not prevent the update-66 allocation cliff

- Command: `/dlwh/mok-dl-022-cap11-v11-trim5-100-20260811-0631-coord`, native-v11 factor-1.1 training, one slot, shared cuda-async pool at fraction 0.85, forward zero-copy, staged backward, trims every five completed updates, 100 requested updates, profiler scheduled at 80-84, attempt-zero semantics.
- Result: native v11 compiled and loaded, and 66 finite metric rows completed through global step 65. Exactly thirteen trims occurred after updates 5 through 65; every trim observed zero active reservations and workspace slots. The final trim reduced each rank's reported default-pool reservation from 166,463,537,152 to 33,789,313,024 bytes while live pool use stayed about 33,697,105,212 bytes. The immediately following update failed on the same 132,687,700,848-byte allocation seen in earlier runs, with only about 20.1 GB of physical device memory free. Coordinator and child failed without retry and released all four GPUs. The profile window and terminal runtime audit were not reached.
- Partial performance: steps 40-59, including four trim pauses, averaged 20,173.86 raw and 20,131.40 drop-adjusted tokens/s in 13.0265 seconds at 23.5942% MFU. This is about 3.8% below the staged and forward-only references, so the cadence itself is not an acceptable performance treatment even if it were stable.
- Interpretation: trim cadence and the v11 parallel host dispatch are not the root cause. The earlier trim-25 v10 run and this trim-5 v11 run both failed at update 66 on the identical allocation. After trim, only about 33.79 GB is attributed to the default pool while roughly 146 GB remains allocated elsewhere on the device. Native workspace storage accounts for only about 7.50 GiB/rank and pinned-host offload does not consume HBM.
- Next action: extend quiescent trim telemetry with physical free/total memory and CUDA graph-pool reserved/used bytes before and after trim. Use a short native-v12 gate to identify whether the missing allocation is graph-pool storage or another non-default-pool owner before changing allocator policy or launching strict factor four.

### 2026-08-11 07:26 PT - Trim releases physical backing exactly; graph memory is zero

- Implementation: native schema v12 extends the existing quiescent trim telemetry with per-rank physical free/total device bytes before and after trim plus CUDA graph-pool reserved/used bytes after trim. Python reports derived bytes outside the default pool and outside both the default and graph pools. No allocation behavior changed. One hundred eleven focused tests, targeted lint/type checks, and pinned source generation passed.
- Command: `/dlwh/mok-dl-023-cap11-v12-trim5-6-telemetry-20260811-0714-coord`, six full-model updates with one trim after update 5, otherwise matching the v11 control.
- Result: coordinator and child succeeded with all six losses finite, including the update immediately after trim. The v12 library compiled and loaded; exact handler counts were 1,152 forward and 1,152 backward. Trim telemetry was quiescent and clean. Each rank's default-pool reservation fell from 168,174,813,184 to 33,722,204,160 bytes, and physical free memory rose by the exact same 134,452,609,024 bytes. CUDA graph reserved/used bytes were both zero on every rank. Residual memory outside the default and graph pools after trim was only 11,283,005,440 bytes on rank zero and 11,259,936,768 bytes on ranks one through three, about 10.5 GiB. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-023-cap11-v12-trim5-6-telemetry-20260811-0714).
- Interpretation: `cudaMemPoolTrimTo` is effective and returns its released backing to device-global availability exactly. The roughly 125.2 GiB released at update 5 was default-pool cached backing, not graph memory. The update-66 failure therefore requires a separate non-default allocation to grow after this early checkpoint; cadence tuning cannot fix that owner.
- Next command: `/dlwh/mok-dl-024-cap11-v12-trim5-100-telemetry-20260811-0727-coord`, identical v12 telemetry through the known update-66 boundary, with local-only autotune caching to avoid remote stage-down. Measure the non-default residual at every five-update trim and correlate its growth with the terminal allocation failure before modifying allocator policy.

### 2026-08-11 07:50 PT - Update-66 failure is internal default-pool reuse, not an external leak

- Command: `/dlwh/mok-dl-024-cap11-v12-trim5-100-telemetry-20260811-0727-coord`, matching v12 telemetry through the known boundary, using local-only per-fusion autotune caching.
- Result: the local-only mode explicitly skipped remote sync and reduced pre-training startup from about 14.5 minutes in the prior control to under one minute before native build/compile. Training reproduced the identical update-66 failure after the clean update-65 trim: 132,687,700,848 requested bytes, about 20.17 GB physical free, 31.38 GiB pool use, and 154.96 GiB high-water. The failed process wedged in rendezvous and was stopped after evidence capture; all four GPUs were released.
- Telemetry: all thirteen trims were quiescent; graph reserved/used memory was always zero. On every rank, bytes outside the default and graph pools were constant at every checkpoint: 11,283,005,440 bytes on rank zero and 11,259,936,768 bytes on ranks one through three, for an exact regression slope of zero bytes/update. Default-pool live use was also constant. Default reserved bytes after trim increased only twice in 32 MiB steps, at updates 35 and 65.
- Corrected interpretation: the apparent roughly 130 GB external allocation at failure was sampled after the allocator had already regrown the default pool during two failed allocation attempts. Outside-pool memory is flat to the byte. At failure, reported reserved-minus-used slack exceeded the giant request by only about 78.7 MB, yet the pool could not form the allocation after stream synchronization and retry. This is an internal large-allocation reuse/remapping/fragmentation cliff under an extremely tight preallocated pool, not a graph, NCCL, native-workspace, offload, or cumulative memory leak.
- Implementation: a typed `GpuDefaultPoolPreallocation` mode now distinguishes the existing eager slab from shared on-demand allocation. On-demand keeps pinned XLA's fast asynchronous default pool (`create_new_pool=false`, `sync_mode=false`) but skips the initial pre-grow and sets release threshold zero. Separate-temp allocation remains distinct and slow because it uses a new synchronous pool. Focused tests and targeted lint/type checks passed.
- Next command: `/dlwh/mok-dl-025-cap11-v12-shared-ondemand-55-20260811-0752-coord`, shared cuda-async with on-demand preallocation, no manual trims, one slot, local-only autotune cache. Require 55 finite updates, terminal audit, and throughput materially closer to the original forward-zero-copy score than the separate-temp fallback.

### 2026-08-11 08:12 PT - Shared on-demand allocation is stable but loses the fast path

- Command: `/dlwh/mok-dl-025-cap11-v12-shared-ondemand-55-20260811-0752-coord`, native-v12 factor-1.1 training, one slot, shared cuda-async with preallocation disabled, no manual trims, local-only autotune cache, 55 updates, profiler steps 5-9, attempt-zero semantics.
- Result: coordinator and child succeeded. All 55 losses were finite, crossing prior updates 33 and 48 without allocator failure. Runtime audit reported exactly 10,560 handlers per phase, zero trims, forward staging zero, only slot zero, maximum active slots one, and zero protocol, generation, reuse, or trim anomalies. Peak HBM was about 165.5 GiB/GPU. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-025-cap11-v12-shared-ondemand-55-20260811-0752); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-dl-025-cap11-v12-shared-ondemand-55-20260811-0752).
- Performance: steps 45-54 averaged 17,453.04 raw and 17,452.60 drop-adjusted tokens/s in 15.0237 seconds at 20.4121% MFU. This is 16.76% below staged and 17.18% below the original forward-zero-copy score, but within 0.07% of the separate-temp fallback.
- Interpretation: disabling the pre-grown slab avoids the allocator cliff but causes the same roughly 17% penalty as the separate synchronous temp pool. CUDA's threshold-zero backing release dominates even though the shared allocator remains asynchronous. This mode is a stable diagnostic/fallback, not the promoted treatment.
- Next command: `/dlwh/mok-dl-026-cap11-v12-eager80-55-20260811-0815-coord`, restore eager shared cuda-async and lower only the pre-grow/release threshold from fraction 0.85 to 0.80. The compiled temp plus live pool use exceeds that threshold by roughly 8 GB, so ordinary synchronization should release surplus backing while preserving the fast allocator path. Require 55-step stability and near-baseline tail throughput before a 100-step seal.

### 2026-08-11 08:33 PT - A lower eager-pool threshold restores stable baseline throughput

- Command: `/dlwh/mok-dl-026-cap11-v12-eager80-55-20260811-0815-coord`, native-v12 factor-1.1 training, one slot, eager shared cuda-async at fraction 0.80, no manual trims, local-only autotune cache, 55 updates, profiler steps 5-9, attempt-zero semantics.
- Result: coordinator and child succeeded. All 55 losses were finite, crossing updates 33 and 48 without allocation failure. Runtime audit reported exactly 10,560 handlers per phase, zero trims, forward staging zero, only slot zero, maximum active slots one, and zero protocol, generation, reuse, or trim anomalies. Peak HBM was about 165.5 GiB/GPU. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-026-cap11-v12-eager80-55-20260811-0815); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-dl-026-cap11-v12-eager80-55-20260811-0815).
- Performance: steps 45-54 averaged 20,984.08 raw and 20,983.63 drop-adjusted tokens/s in 12.4926 seconds at 24.5418% MFU. This is 0.08% above the staged reference, 0.43% below the best forward-zero-copy score, and 20.23% above the stable on-demand fallback.
- Interpretation: cuda-async's fraction is a pre-grow/release-threshold policy rather than a hard cap. At 0.80, the full temp plus live use exceeds the retained threshold by roughly 8 GB, causing ordinary synchronization to return surplus backing without manual one-second pauses or the threshold-zero performance collapse. This is the first allocator treatment with both repeated-step stability and baseline throughput evidence.
- Next command: `/dlwh/mok-dl-027-cap11-v12-eager80-100-20260811-0834-coord`, identical settings for the decisive 100-update seal with profiler 80-84. Require complete steps 60-79 and 40-79 scores, terminal all-process audit, and no update-66 recurrence before launching the matched factor-four dropless arm.

### 2026-08-11 09:01 PT - Stable 100-update forward-zero-copy control is sealed

- Command: `/dlwh/mok-dl-027-cap11-v12-eager80-100-20260811-0834-coord`, native-v12 factor-1.1 training, one slot, eager shared cuda-async at fraction 0.80, no manual trims, local-only autotune cache, 100 updates, profiler 80-84, attempt-zero semantics.
- Result: coordinator and child succeeded. All 100 losses were finite through final loss 5.3021, including the prior update-66 allocation boundary. The profile uploaded successfully. Runtime audit reported exactly 19,200 handlers per phase, zero trims, forward staging zero, only slot zero with maximum active slots one, and zero protocol, generation, reuse, or trim anomalies. Peak HBM was about 165.5 GiB/GPU. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-027-cap11-v12-eager80-100-20260811-0834); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-dl-027-cap11-v12-eager80-100-20260811-0834).
- Performance: steps 60-79 averaged 20,774.04 raw and 20,730.59 drop-adjusted tokens/s in 12.6191 seconds at 24.2962% MFU, with 0.2102% drops. Steps 40-79 averaged 20,781.30 raw and 20,741.37 adjusted tokens/s in 12.6148 seconds at 24.3047% MFU, with 0.1928% drops. Primary raw throughput is 0.92% below staged and 1.43% below the earlier best forward-zero-copy run.
- Interpretation: the 0.80 eager-pool threshold is the promoted allocator treatment. It eliminates the deterministic update-66 cliff without manual maintenance, preserves the fast shared asynchronous path, and finishes within about one percent of the barrier-free staged reference. Forward peer zero-copy remains exact and saves the named D2D staging traffic; backward remains staged because its direct modes did not improve throughput.
- Next command: `/dlwh/mok-dl-028-cap4-v12-eager80-100-20260811-0902-coord`, identical v12/eager-0.80 settings with strict factor-four schedule capacity. Require zero dropped assignments on every update and candidate raw throughput at least 99% of the factor-1.1 control's drop-adjusted throughput in both scoring windows.

### 2026-08-11 09:29 PT - Strict dropless training passes the matched 100-update gate

- Command: `/dlwh/mok-dl-028-cap4-v12-eager80-100-20260811-0902-coord`, native-v12 strict factor-four training, one slot, forward zero-copy, staged backward, eager shared cuda-async at fraction 0.80, no trims, local-only autotune cache, 100 updates, profiler 80-84, attempt-zero semantics.
- Result: coordinator and child succeeded. All 100 losses were finite through final loss 5.3217. Every individual row reported exactly zero dropped assignments and zero drop fraction. The run crossed the prior allocation boundaries and uploaded the requested profile. Runtime audit reported exactly 19,200 handlers per phase, zero trims, forward staging zero, only slot zero with maximum active slots one, and zero protocol, generation, reuse, or trim anomalies. Peak HBM was about 165.5 GiB/GPU. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-dl-028-cap4-v12-eager80-100-20260811-0902); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-dl-028-cap4-v12-eager80-100-20260811-0902).
- Performance: steps 60-79 averaged 20,664.44 raw and adjusted tokens/s in 12.6863 seconds at 24.1680% MFU. This retains 99.6809% of the factor-1.1 control's 20,730.59 drop-adjusted tokens/s and is 0.5276% below its raw rate. Steps 40-79 averaged 20,678.80 tokens/s in 12.6775 seconds at 24.1848% MFU, retaining 99.6983% of the control's adjusted rate. Both windows pass the 99% gate.
- Interpretation: canonical FP32 routed-weight accumulation removes the capacity-dependent XLA output footprint, and strict factor-four capacity restores dropless semantics at less than one percent matched useful-throughput cost. The factor-four grid still overlaunches capacity-sized no-op clusters, but active routed math is bounded by `schedule.num_tokens`; the measured residual is small enough to promote strict dropless capacity before scheduler-specific optimization.
- Next action: make the sealed dropless/forward-zero-copy/eager-0.80/local-cache configuration the canonical launcher preset, close the peer-failure cancellation gap found in snapshot review, commit and tag the exact source, then rerun a short gate from that snapshot before the 2-node, 1-rack, and 2-rack ladder.

### 2026-08-11 10:01 PT - Peer cancellation is not safe to include in the stable snapshot

- Review finding: the barrier-free protocol's device readiness and completion waits have no peer-visible cancellation path. A synchronous rank-local handler failure can leave peers waiting, and runtime shutdown then refuses the active reservation. A native-v13 candidate added generation-tagged per-slot cancellation polled by device waits and passed 126 local tests plus source generation.
- Command: `/dlwh/mok-bf-013-cancellation-matrix-20260811-0958`, four isolated single-slot forward/backward failures followed by two two-slot concurrent-RunId controls, no retry.
- Result: forward failures before input readiness and before completion both returned in about 0.6 seconds with exact four-forward/zero-backward handlers and zero slot-reuse failures. The backward-before-input-ready case cancelled the internal native waits, but rank zero then aborted before a subsequent XLA AllReduce while ranks one through three entered it. XLA terminated after a 40-second 3-of-4 rendezvous; later matrix cases did not run. The job failed once and released all GPUs.
- Interpretation: device-wait cancellation alone is insufficient. Every peer executable shard must receive the same failure before any can advance to a later XLA collective. Typed FFI handlers return after enqueuing device work, so a peer that observes cancellation asynchronously cannot currently turn that observation into an all-rank FFI error. Shipping the partial patch would replace one hang class with a later collective mismatch.
- Decision: revert native-v13 and keep the measured native-v12 implementation for the stable branch. The stable snapshot documents rank-local handler/CUDA failure closure as an unresolved limitation; sticky device traps remain terminal process-poison tests. Future work must provide invocation-wide host-visible error propagation before handler return, or eliminate/postpone post-FFI collectives so every shard aborts consistently.
- Next action: commit and push the v12 dropless/forward-zero-copy implementation and launcher/scale hardening, then run a short promoted gate from that commit before multi-node scaling.

### 2026-08-11 10:30 PT - Current-source two-node strict-dropless gate prepared

- DRI: dlwh. Source: clean pushed branch `codex/upstream-mok-like` at `80f909f4bb4657166c995293a8ff1931bf601ae8`; native build schema v12 with pinned MoK `6438bf48f88094d305972fbe0fa6deba0f7d4d1a` and ThunderKittens `1c3920d993404dd49a6d4c7267ea11d583bd5c68`.
- Contract: two Iris tasks, one JAX process and four GB200s per task, local EP4 and DP2, global batch 128 at sequence length 4096, 48 layers, 25 optimizer updates. The reviewed `promoted_dropless_v12` preset selects strict factor-four schedule capacity, one workspace slot, forward XLA-peer storage, staged backward storage, shared eager cuda-async at fraction 0.80, local-only autotune cache, no trims, and zero child retries/failure tolerance.
- Output identity: artifact `grug/moe-backend-comparison/mok_like/mok-scale-003-v12-dropless-2node-25-20260811-1030/2026.08.11`; W&B id/name `mok-scale-003-v12-dropless-2node-25-20260811-1030`, project `marin-community/marin_moe`, group `moe-backend-comparison-2node`, resume `allow`. This metrics gate writes no final model checkpoint; `initialize_from` is unset. Default temporary checkpointing is retained only for preemption recovery, with no retry authorized.
- Exact submission, with the secret value scrubbed:

  ```bash
  run_id="mok-scale-003-v12-dropless-2node-25-20260811-1030"
  .venv/bin/python -c 'import iris.cluster.platforms.k8s.service as service; from iris.cluster.platforms.types import find_free_port; service.find_free_port = lambda start=10000: find_free_port(); from iris.cli.main import main; main()' \
    --config lib/iris/config/cw-us-east-08a.yaml job run --no-wait --enable-extra-resources \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 14400 --max-retries 0 \
    --job-name "${run_id}-coord" -e WANDB_API_KEY '<redacted>' -e WANDB_PROJECT marin_moe -- \
    .venv/bin/python -m experiments.grug.moe_hero_ep.launch_mok_like \
      --run-id "$run_id" --backend mok_like --num-steps 25 --num-nodes 2 \
      --mok-like-preset promoted_dropless_v12 --version 2026.08.11 --run
  ```

- Acceptance: both tasks and coordinator terminal-successful; 25 finite losses and zero drops on every row; exact 4,800 forward and 4,800 backward handlers per process; forward staging zero; staged backward copy calls/bytes exact; slot one unused with maximum active slots one; zero generation, reuse, trim, or protocol anomalies across both processes. Score profile-free steps 10-24, report total and per-GPU throughput, mean step time, MFU, peak HBM, and weak-scale efficiency against the sealed four-GPU dropless score. Stop on deterministic topology, allocator, numerical, or protocol failure; do not resubmit.

### 2026-08-11 10:45 PT - Two-node gate passes; one-rack gate prepared

- Result: `/dlwh/mok-scale-003-v12-dropless-2node-25-20260811-1030-coord` and its two-task child succeeded without a retry, failure, or preemption. All 25 losses were finite through final loss 6.30623, and every row reported exactly zero dropped assignments.
- Performance: profile-free steps 10-24 averaged 39,838.317 tokens/s total, 4,979.790 tokens/s/GPU, 13.17164 seconds, and 23.29634% MFU. This is 96.3934% weak-scale efficiency against twice the sealed four-GPU dropless score of 20,664.441 tokens/s. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-scale-003-v12-dropless-2node-25-20260811-1030); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-scale-003-v12-dropless-2node-25-20260811-1030).
- Runtime audit: both processes reported exactly 4,800 forward and 4,800 backward handlers. Forward staging was zero. Backward staged copies were exactly 19,200 calls and 7,741,007,462,400 bytes per process. Slot one was unused, maximum active slots was one, and generation, reuse, trim, and protocol anomalies were all zero. Process-zero peak HBM was 172.96-173.30 GiB/GPU; W&B did not retain process-one system telemetry.
- Decision: promote to one rack. The 16-node request is one hard `nvlink.domain` gang, local EP4/DP16, global batch 1,024, 48 layers, and 25 updates. The same reviewed preset and attempt-zero contract apply. Ideal weak-scale throughput is 330,631.0488 tokens/s; 80% and 70% gates are 264,504.8390 and 231,441.7341 tokens/s.
- Output identity: artifact `grug/moe-backend-comparison/mok_like/mok-scale-004-v12-dropless-1rack-25-20260811-1045/2026.08.11`; W&B id/name `mok-scale-004-v12-dropless-1rack-25-20260811-1045`, project `marin-community/marin_moe`, group `moe-backend-comparison-1rack`, resume `allow`. `initialize_from` is unset and this metrics gate has no final checkpoint.
- Exact submission, with the secret value scrubbed:

  ```bash
  run_id="mok-scale-004-v12-dropless-1rack-25-20260811-1045"
  .venv/bin/python -c 'import iris.cluster.platforms.k8s.service as service; from iris.cluster.platforms.types import find_free_port; service.find_free_port = lambda start=10000: find_free_port(); from iris.cli.main import main; main()' \
    --config lib/iris/config/cw-us-east-08a.yaml job run --no-wait --enable-extra-resources \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 \
    --job-name "${run_id}-coord" -e WANDB_API_KEY '<redacted>' -e WANDB_PROJECT marin_moe -- \
    .venv/bin/python -m experiments.grug.moe_hero_ep.launch_mok_like \
      --run-id "$run_id" --backend mok_like --num-steps 25 --num-nodes 16 \
      --mok-like-preset promoted_dropless_v12 --version 2026.08.11 --run
  ```

- Acceptance: all 16 tasks terminal-successful; 25 finite zero-drop rows; exact 4,800 forward/backward handlers and storage-mode staging telemetry per process; slot one unused; zero protocol anomalies across all processes. Score steps 10-24 and require at least 80% weak-scale efficiency before allocating two racks. Stop on deterministic topology, allocator, numerical, or protocol failure; do not resubmit.

### 2026-08-11 11:20 PT - One-rack request reduced to 96 CPUs per node

- Capacity result: the original 120-CPU request remained `BUILDING` for 30m45s with all 16 tasks scheduling-gated and no GPU allocation or logs. Kueue reported that the hard `multinode-nvlink-ib` topology could fit only 2 of 16 pods; 194 of 201 nodes were excluded solely by the CPU request. There were zero failures and preemptions.
- Evidence for the adjustment: the completed two-node run's 120-CPU process-zero allocation averaged 1.136% host CPU and peaked at 1.698%. Commit `37a0634cba` requests 96 CPUs per four-GB200 task and adds the resource contract to the scale-plan test; 84 focused tests and the required checks passed. GPU count, memory, data, model, mesh, optimizer, XLA graph, and numerical settings are unchanged.
- Decision: stop the unallocated 120-CPU gang and launch a new attempt-zero run from the pushed 96-CPU snapshot. The new identity prevents W&B or artifact lineage from mixing with the pending request.
- Output identity: artifact `grug/moe-backend-comparison/mok_like/mok-scale-005-v12-dropless-1rack-cpu96-25-20260811-1120/2026.08.11`; W&B id/name `mok-scale-005-v12-dropless-1rack-cpu96-25-20260811-1120`, project `marin-community/marin_moe`, group `moe-backend-comparison-1rack`, resume `allow`.
- Exact submission, with the secret value scrubbed:

  ```bash
  run_id="mok-scale-005-v12-dropless-1rack-cpu96-25-20260811-1120"
  .venv/bin/python -c 'import iris.cluster.platforms.k8s.service as service; from iris.cluster.platforms.types import find_free_port; service.find_free_port = lambda start=10000: find_free_port(); from iris.cli.main import main; main()' \
    --config lib/iris/config/cw-us-east-08a.yaml job run --no-wait --enable-extra-resources \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 \
    --job-name "${run_id}-coord" -e WANDB_API_KEY '<redacted>' -e WANDB_PROJECT marin_moe -- \
    .venv/bin/python -m experiments.grug.moe_hero_ep.launch_mok_like \
      --run-id "$run_id" --backend mok_like --num-steps 25 --num-nodes 16 \
      --mok-like-preset promoted_dropless_v12 --version 2026.08.11 --run
  ```

- Acceptance remains the exact 16-process numerical/runtime audit and at least 80% weak-scale efficiency. A second capacity wait is not a training failure; report its exact fit/exclusion state without changing the cluster.

### 2026-08-11 11:25 PT - Admission wait is occupied-node capacity

- Correction: the identical Kueue exclusion at 120 and 96 requested cores does not establish that the request size is the limiting threshold. A read-only node inventory showed that no complete rack slice was currently available even under substantially smaller hypothetical CPU requests.
- Interpretation: other GPU work occupies the nodes needed for the hard 16-node slice. The 96-core setting remains safe relative to the measured 1.698% peak CPU use, but it does not solve the current capacity state.
- Decision: keep `/dlwh/mok-scale-005-v12-dropless-1rack-cpu96-25-20260811-1120-coord` pending at interactive priority. Do not lower the request again, submit competing gangs, or mutate the cluster. Resume the numerical/performance gate when one complete rack becomes available.

### 2026-08-11 11:52 PT - One-rack admission improves while remaining pending

- Status: after 31m58s, all 16 tasks remained scheduling-gated with zero failures, preemptions, logs, or partial GPU allocation. The scheduler's feasible fit improved from 2 of 16 tasks to 10 of 16 tasks as occupied capacity cleared.
- Decision: keep the single interactive request pending. Do not submit a competing gang or change its resources; begin the numerical and throughput gate only after all 16 tasks co-schedule on one complete rack slice.

### 2026-08-11 12:24 PT - One-rack feasible fit returns to two tasks

- Status: after 1h03m38s, the feasible hard-topology fit fell from 10 of 16 tasks back to 2 of 16 as other capacity was occupied. All 16 tasks remained scheduling-gated with zero failures, preemptions, logs, or partial allocation.
- Decision: continue the same pending request. The changing fit confirms an external capacity wait; no resource or launcher change is indicated.

### 2026-08-11 13:00 PT - One-rack compile exceeds the 850 GiB pod limit

- Result: the 16-task gang admitted at about 12:48 PT, then task one was `OOMKilled` with exit 137 during first-step compilation. The other 15 tasks terminated as coscheduled siblings. No optimizer update completed, so W&B has no loss, throughput, drop, runtime-audit, or profile row. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-scale-005-v12-dropless-1rack-cpu96-25-20260811-1120).
- Evidence: process-zero telemetry reached 775,573.875 MiB RSS before the kill, 79.25% process memory, and 93.25% system memory. XLA reported a 288.94 GiB rematerialized device plan against a 176.70 GiB target, but device allocation remained about 156.7 GiB/GPU; the Kubernetes event identifies host/container memory as the failing resource. The successful four-GPU and two-node runs peaked at 827,916.625 and 832,662.0625 MiB RSS under the same graph, showing that the 850 GiB pod limit has little compile headroom.
- Decision: request 900 GiB per four-GB200 task. The cluster advertises about 955.5 GiB allocatable memory per GB200 node, so this adds 50 GiB of cgroup headroom without changing the model, mesh, XLA graph, allocator, or numerical contract. Run one rack before two racks because the same per-process compile peak applies to both topologies.
- Next action: launch a new 16-node, 25-update attempt-zero identity from a clean pushed snapshot. Require all 16 tasks to compile, 25 finite zero-drop rows, exact all-process runtime telemetry, and at least 80% weak-scale efficiency before requesting two racks.

### 2026-08-12 10:18 PT - One-rack 900 GiB retry prepared

- DRI: dlwh. Source: clean pushed branch `codex/upstream-mok-like` at `c8558b7cfa62133d427dba5f509d7c1542d451ed`; native schema v12 with pinned MoK and ThunderKittens revisions unchanged. The launcher requests 16 tasks, each with four GB200s, 96 CPUs, 900 GiB RAM, and 1 TiB disk. Current Iris budget spend is zero under the 128,000 interactive limit.
- Contract: local EP4/DP16, global batch 1,024, 48 layers, 25 optimizer updates, strict factor-four capacity, one workspace slot, forward XLA-peer storage, staged backward storage, shared eager cuda-async at fraction 0.80, local-only autotune cache, profile steps 5-9, and zero retries or tolerated task failures. The run writes metrics and a profile but no final model checkpoint; `initialize_from` is unset.
- Output identity: artifact `grug/moe-backend-comparison/mok_like/mok-scale-006-v12-dropless-1rack-ram900-25-20260812-1018/c8558b7cfa`; W&B id/name `mok-scale-006-v12-dropless-1rack-ram900-25-20260812-1018`, project `marin-community/marin_moe`, group `moe-backend-comparison-1rack`, resume `allow`.
- Exact submission, with the secret value scrubbed:

  ```bash
  .venv/bin/python -c 'import iris.cluster.platforms.k8s.service as service; from iris.cluster.platforms.types import find_free_port; service.find_free_port = lambda start=10000: find_free_port(); from iris.cli.main import main; main()' \
    --config lib/iris/config/cw-us-east-08a.yaml job run --no-wait --enable-extra-resources \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 \
    --job-name mok-scale-006-v12-dropless-1rack-ram900-25-20260812-1018-coord \
    -e WANDB_API_KEY '<redacted>' -e WANDB_PROJECT marin_moe -- \
    .venv/bin/python -m experiments.grug.moe_hero_ep.launch_mok_like \
      --run-id mok-scale-006-v12-dropless-1rack-ram900-25-20260812-1018 \
      --backend mok_like --num-steps 25 --num-nodes 16 \
      --mok-like-preset promoted_dropless_v12 --version c8558b7cfa --run
  ```

- Acceptance: all 16 tasks and coordinator terminal-successful; 25 finite losses with exact zero drops; 4,800 forward and 4,800 backward handlers per process; forward staging zero; exact staged-backward bytes/calls; slot one unused with maximum active slots one; zero generation, reuse, trim, or protocol anomalies. Score steps 10-24 and require at least 80% weak-scale efficiency before launching two racks.

### 2026-08-12 10:20 PT - Scale006 rejected an invalid artifact version

- Result: the coordinator exited 2 before it created a child job, allocated a GPU, or initialized W&B. The launcher rejected `--version c8558b7cfa` because artifact versions must use `YYYY.MM.DD[.N]`, `dev`, or a `-dev` label. This did not exercise the 900 GiB resource contract.
- Decision: retain the clean implementation SHA `c8558b7cfa62133d427dba5f509d7c1542d451ed` as source lineage and use calendar artifact version `2026.08.12`. A new run id prevents the rejected coordinator identity from mixing with the corrected launch.
- Next command: repeat the exact Scale006 contract as `mok-scale-007-v12-dropless-1rack-ram900-25-20260812-1020` with `--version 2026.08.12`. No other argument changes.

### 2026-08-12 10:32 PT - One-rack strict-dropless gate passes

- Result: `/dlwh/mok-scale-007-v12-dropless-1rack-ram900-25-20260812-1020-coord` and all 16 child tasks succeeded with no failure, preemption, or retry. W&B finished with exactly 25 finite rows through final loss 6.09893; every row reported zero dropped assignments. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-scale-007-v12-dropless-1rack-ram900-25-20260812-1020); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-scale-007-v12-dropless-1rack-ram900-25-20260812-1020).
- Performance: profile-free steps 10-24 averaged 310,771.377594 tokens/s total, 4,855.802775 tokens/s/GPU, 13.513913 seconds, and 22.716306% MFU. This is 93.993404% weak-scale efficiency against the 330,631.048752-token/s ideal and passes the 80% promotion gate.
- Runtime audit: every process reported exactly 4,800 forward and 4,800 backward handlers. Forward staging was zero. Backward staged copies were exactly 19,200 calls and 7,741,007,462,400 bytes per process. Slot one was unused, maximum active slots was one, and protocol, generation, reuse, trim, and slot anomalies were zero across all 16 processes.
- Memory: process-zero compile RSS peaked at 890,453.125 MiB (869.583 GiB) and system memory reached 100%. The 900 GiB pod contract passed, but the margin is limited. Peak allocated HBM was 180,552,531,968 bytes on GPU zero and 180,529,463,296 bytes on GPUs one through three.
- Decision: promote the unchanged source and 900 GiB resource contract to the 32-node, two-rack 25-update gate. Require all 32 processes to compile and pass the exact runtime audit, 25 finite zero-drop rows, and at least 80% weak-scale efficiency before the 100-update two-rack seal.

### 2026-08-12 10:40 PT - Two-rack strict-dropless gate prepared

- DRI: dlwh. Source: implementation commit `c8558b7cfa62133d427dba5f509d7c1542d451ed` on clean pushed branch `codex/upstream-mok-like`; later branch commits contain only experiment records. Current Iris spend is zero under the 128,000 interactive budget. The scheduler computes the gang's effective band before admission, so the single 32-task request remains one scheduling unit even though its running resource value exceeds the ordinary interactive budget.
- Contract: 32 tasks across two hard 16-node rack slices, one JAX process and four GB200s per task, 96 CPUs, 900 GiB RAM, and 1 TiB disk per node. Local EP4/DP32 gives global batch 2,048 with the same 16 sequences/GPU. The strict-dropless v12 preset, 48 layers, 25 updates, local-only cache, profile steps 5-9, and zero retry/failure tolerance are unchanged. The run writes metrics/profile artifacts but no final model checkpoint; `initialize_from` is unset.
- Output identity: artifact `grug/moe-backend-comparison/mok_like/mok-scale-008-v12-dropless-2rack-ram900-25-20260812-1040/2026.08.12`; W&B id/name `mok-scale-008-v12-dropless-2rack-ram900-25-20260812-1040`, project `marin-community/marin_moe`, group `moe-backend-comparison-2rack`, resume `allow`.
- Exact submission, with the secret value scrubbed:

  ```bash
  .venv/bin/python -c 'import iris.cluster.platforms.k8s.service as service; from iris.cluster.platforms.types import find_free_port; service.find_free_port = lambda start=10000: find_free_port(); from iris.cli.main import main; main()' \
    --config lib/iris/config/cw-us-east-08a.yaml job run --no-wait --enable-extra-resources \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 \
    --job-name mok-scale-008-v12-dropless-2rack-ram900-25-20260812-1040-coord \
    -e WANDB_API_KEY '<redacted>' -e WANDB_PROJECT marin_moe -- \
    .venv/bin/python -m experiments.grug.moe_hero_ep.launch_mok_like \
      --run-id mok-scale-008-v12-dropless-2rack-ram900-25-20260812-1040 \
      --backend mok_like --num-steps 25 --num-nodes 32 \
      --mok-like-preset promoted_dropless_v12 --version 2026.08.12 --run
  ```

- Acceptance: all 32 tasks and coordinator terminal-successful; 25 finite exact-zero-drop rows; 4,800 forward/backward handlers per process; forward staging zero and staged-backward copies exact; slot one unused, maximum active slots one, and zero protocol/generation/reuse/trim anomalies. Score steps 10-24 against the 661,262.097504-token/s ideal; require at least 80% weak-scale efficiency before the 100-update seal.

### 2026-08-12 10:55 PT - Two-rack 25-update gate passes

- Result: `/dlwh/mok-scale-008-v12-dropless-2rack-ram900-25-20260812-1040-coord` and all 32 training tasks succeeded with no failure, preemption, or retry. Placement was exactly two hard NVLink domains with 16 task nodes each. W&B finished with 25 finite rows through final loss 6.07718 and exact zero drops on every row. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-scale-008-v12-dropless-2rack-ram900-25-20260812-1040); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-scale-008-v12-dropless-2rack-ram900-25-20260812-1040).
- Performance: steps 10-24 averaged 578,736.805369 tokens/s total, 4,521.381292 tokens/s/GPU, 14.502328 seconds, and 21.151823% MFU. This is 87.520033% weak-scale efficiency against the 661,262.097504-token/s ideal and passes the 80% promotion gate.
- Runtime audit: every process reported exactly 4,800 forward and 4,800 backward handlers. Forward staging was zero. Backward staged copies were exactly 19,200 calls and 7,741,007,462,400 bytes per process. Slot one was unused, maximum active slots was one, and protocol, generation, reuse, trim, and slot anomalies were zero across all 32 processes.
- Memory: process-zero compile RSS peaked at 888,605 MiB (867.778 GiB) and system memory reached 100%. Peak allocated HBM was 181,995,372,544 bytes on GPU zero and 181,972,303,872 bytes on GPUs one through three. The 900 GiB contract passed with limited margin.
- Decision: run the final 100-update two-rack seal from the unchanged source and resource contract. Score profile-free steps 60-79 and 40-79; profile process zero at steps 80-84; require 100 finite zero-drop rows and exact 19,200-handler-per-phase audits across every process.

### 2026-08-12 11:00 PT - Two-rack 100-update seal prepared

- Output identity: artifact `grug/moe-backend-comparison/mok_like/mok-scale-009-v12-dropless-2rack-ram900-100-20260812-1100/2026.08.12`; W&B id/name `mok-scale-009-v12-dropless-2rack-ram900-100-20260812-1100`, project `marin-community/marin_moe`, group `moe-backend-comparison-2rack`, resume `allow`. This metrics run has no final model checkpoint and `initialize_from` is unset.
- Exact submission, with the secret value scrubbed:

  ```bash
  .venv/bin/python -c 'import iris.cluster.platforms.k8s.service as service; from iris.cluster.platforms.types import find_free_port; service.find_free_port = lambda start=10000: find_free_port(); from iris.cli.main import main; main()' \
    --config lib/iris/config/cw-us-east-08a.yaml job run --no-wait --enable-extra-resources \
    --priority interactive --cpu 2 --memory 8GB --disk 32GB --timeout 21600 --max-retries 0 \
    --job-name mok-scale-009-v12-dropless-2rack-ram900-100-20260812-1100-coord \
    -e WANDB_API_KEY '<redacted>' -e WANDB_PROJECT marin_moe -- \
    .venv/bin/python -m experiments.grug.moe_hero_ep.launch_mok_like \
      --run-id mok-scale-009-v12-dropless-2rack-ram900-100-20260812-1100 \
      --backend mok_like --num-steps 100 --num-nodes 32 \
      --mok-like-preset promoted_dropless_v12 --version 2026.08.12 --run
  ```

- Acceptance: all 32 tasks and coordinator terminal-successful; 100 finite exact-zero-drop rows; exact 19,200 forward/backward handlers per process; forward staging zero and staged-backward copies exact; slot one unused, maximum active slots one, and zero protocol/generation/reuse/trim anomalies. Report steps 60-79 and 40-79 total/per-GPU throughput, step time, MFU, weak-scale efficiency, and the process-zero XProf upload.

### 2026-08-12 11:37 PT - Two-rack 100-update seal completes below the performance target

- Result: `/dlwh/mok-scale-009-v12-dropless-2rack-ram900-100-20260812-1100-coord` and all 32 training tasks succeeded with no failure, preemption, or retry. Placement was exactly two hard NVLink domains with 16 task nodes each. W&B finished with 100 finite rows through final loss 4.66793 and exact zero drops on every row. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-scale-009-v12-dropless-2rack-ram900-100-20260812-1100); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-scale-009-v12-dropless-2rack-ram900-100-20260812-1100).
- Performance: steps 60-79 averaged 516,415.193428 tokens/s total, 4,034.493699 tokens/s/GPU, 16.404556 seconds, and 18.874077% MFU. Weak-scale efficiency was 78.095387% against the 661,262.097504-token/s ideal. Steps 40-79 averaged 513,978.219612 tokens/s, 4,015.454841 tokens/s/GPU, 16.507216 seconds, and 18.785010% MFU, or 77.726853% efficiency. These windows miss the 80% target by 1.90 and 2.27 percentage points and regress 10.77% and 11.19% from the 25-update two-rack score.
- Runtime audit: every process reported exactly 19,200 forward and 19,200 backward handlers. Forward staging was zero. Backward staged copies were exactly 76,800 calls and 30,964,029,849,600 bytes per process. Slot one was unused, maximum active slots was one, and protocol, generation, reuse, trim, and slot anomalies were zero across all 32 processes.
- Memory: process-zero compile RSS peaked at 888,114.5 MiB (867.299 GiB) and system memory reached 100%, without a host or device OOM under the 900 GiB pod contract. The margin remains limited.
- Diagnosis: input loading and hook time were measured in microseconds and milliseconds, local GPU clocks matched the faster 25-update run, and the training logs contained no NCCL, RAS, watchdog, or collective timeout. The slow window had a much wider step-time distribution, lower mean GPU power, lower sampled NVLink traffic, and higher host-network traffic. This is consistent with intermittent DP32 cross-rack collective or fabric waits, but the available evidence does not isolate a specific collective or link.
- Decision: the v12 implementation is sealed for zero-copy forward storage, strict dropless routing, numerical correctness, and two-rack execution. The 100-update two-rack performance target is not met. Keep the pushed branch as the stable snapshot and treat cross-rack throughput variability as a separate profiling and fabric investigation.

### 2026-08-12 12:05 PT - Confidence campaign starts from the stable v12 snapshot

- Baseline: clean pushed branch `codex/upstream-mok-like` at `728c38a5cee3cf1fbe7f614e20b52994cf364992`; implementation commit `c8558b7cfa62133d427dba5f509d7c1542d451ed`; native schema v12. Normal-path correctness confidence is high, but rank-local failure closure, long stateful parity, repeatability of two-rack performance, and compile-memory margin remain open.
- Failure gate: require a rank-local native failure to become a bounded host-visible error on every rank before any shard can enter a later XLA collective. Inject failures before readiness and completion in forward and backward, then exercise two concurrent RunIds. Reject a design that only cancels device waits or lets one rank return early.
- Numerical gate: run hundreds of fixed-seed sequential comparisons against the independent EP reference. Record per-step output, loss, gradient, optimizer-state, routing, and drop error. Alternate balanced, zero-token, all-to-one, and capacity-boundary routes while reusing workspace slots.
- Reproducibility gate: build from a clean pushed checkout on fresh four-GB200 workers, run the full failure and adversarial matrix, then repeat on two nodes with exact all-process audits.
- Scale gate: run at least two additional two-rack 100-update replicas with steps 80-84 profiles. Require 100 finite zero-drop rows, exact runtime audits, and enough replicas to separate implementation variance from fabric variance. Compare profile-free 60-79 and 40-79 windows with the existing 78.10% and 87.52% efficiency observations.
- Memory gate: identify the source of the 867-870 GiB process RSS compile peak or demonstrate repeatable headroom under the 900 GiB pod contract. Do not promote a lower memory request without a full-rack compile gate.
- Stop conditions: no dirty-tree hardware launch; no retry of a deterministic failure; no promotion after a partial failure-closure gate; no two-rack replica until four-GPU and two-node gates pass from the candidate SHA.

### 2026-08-12 14:35 PT - Stateful parity and cross-process failure closure pass

- Candidate source: clean pushed branch `codex/mok-confidence-v14` at `c88afbb313bb14592f91cbbdb1634a5c812b75fe`; native schema v15. Backward is a destination-passing typed FFI: gradients and failure status are explicit results, while workspace counters and leases remain private protocol state. A full-mesh failure reduction makes the branch uniform before later shared-weight collectives. The negative-gate CLI shuts down distributed JAX explicitly while preserving the injected error.
- Stateful parity: `/dlwh/mok-v14-stateful-parity-maskfix-4gb200-64-20260812-1351` succeeded from `2ae0ac7470f83e0f2bc8500e0850d2cb483b5f5e`. All 64 sequential momentum updates matched the independent reference across one balanced, 16 zero-token, 31 all-to-one, and 16 skewed routes. Output, loss, gradients, parameters, and optimizer state were all close; drops and inactive-expert gradients were zero. The run completed exactly 256 forward and 256 backward handlers, used only slot zero, and reported no generation or reuse anomaly.
- Two-process failure gate: `/dlwh/mok-v15-c88-neg-bwd-inputready-2proc-20260812-1431` succeeded from `c88afbb313bb14592f91cbbdb1634a5c812b75fe`. A synchronous backward failure injected on process zero before input readiness became the same phase-neutral error on both processes in 3.230917 and 3.236696 seconds. Each process reported four forward and four backward handlers, slot acquisitions `[2,0]` on every local rank, maximum active slots one, and zero generation or reuse anomalies. Both runtimes closed, distributed shutdown completed, and both Iris tasks exited zero.
- Prior negative-gate attempts exposed two distinct issues and remain part of the record. Device-only cancellation let healthy ranks enter a later shared-weight collective. The first globally uniform v15 error then persisted in JAX's ordered backward effect token during interpreter cleanup. Returning failure through dataflow and making backward pure removed that retained token; later successful diagnostics now retire the ordinary runtime token before shutdown.
- Remaining gates: complete all forward/backward and concurrent-control injections on four GB200s, measure the healthy-path scalar-agreement overhead, then repeat the two-rack seal. A four-GPU negative matrix is running from `c88afbb313`; no result is claimed until the whole matrix and Iris teardown pass.

### 2026-08-12 14:35 PT - Host-memory attribution finds a pinned-arena boundary

- Control: `/dlwh/mok-memconfidence-001-v12-1node-host192-25-20260812-1324` succeeded from stable v12 SHA `728c38a5cee3cf1fbe7f614e20b52994cf364992`. At the compile checkpoint, cgroup current was 812.83 GiB and summed process-tree PSS was 810.72 GiB. PID one contributed 808.74 GiB anonymous PSS, so the 816.31 GiB cgroup peak is real private anonymous memory rather than descendant or shared-map double counting. Steps 10-24 averaged 20,340.364 tokens/s with 1.9068% CV and exact zero drops.
- Treatment: `/dlwh/mok-memconfidence-002-v12-1node-host128-25-20260812-1412` failed deterministically before step zero. All four host BFC allocators rejected the same 106,300,441,600-byte request. The allocator map implied about 70.4 GiB already occupied, making the immediate compile requirement about 169.4 GiB. No training or profile row exists.
- Decision: skip 160 GiB because it is predicted below the observed compile requirement. Test 176 GiB as the smallest likely feasible arena, with exact cgroup/PSS checkpoints and the 192 GiB throughput control. The 176 GiB run is active under `/dlwh/mok-memconfidence-003-v12-1node-host176-25-20260812-1432-coord`; no result is claimed before terminal evidence.

### 2026-08-12 14:50 PT - The 176 GiB host arena passes the one-node gate

- Result: `/dlwh/mok-memconfidence-003-v12-1node-host176-25-20260812-1432-coord` and its child succeeded from stable v12 SHA `728c38a5cee3cf1fbe7f614e20b52994cf364992`, with no failure, preemption, or retry. W&B finished with 25 finite zero-drop rows. [W&B](https://wandb.ai/marin-community/marin_moe/runs/mok-memconfidence-003-v12-1node-host176-25-20260812-1432); [XProf](https://iris.oa.dev/proxy/xprof/open?uri=s3%3A%2F%2Fmarin-us-east-02a%2Ftmp%2Fttl%3D30d%2Fxprof%2Fmok-memconfidence-003-v12-1node-host176-25-20260812-1432).
- Performance: steps 10-24 averaged 20,853.6004 tokens/s with 2.1528% population CV, 12.576581 seconds, and 24.38921% MFU. Throughput was 2.52324% above the 192 GiB control and passed the three-percent non-regression gate.
- Memory: cgroup peak was 801,148,633,088 bytes (746.1278 GiB), 75,355,389,952 bytes (70.1802 GiB) below the 192 GiB control. The 900 GiB pod retained 153.8722 GiB of measured headroom. All cgroup low, high, max, OOM, OOM-kill, and OOM-group-kill counters stayed zero.
- Runtime audit: 4,800 forward and backward handlers, zero forward staging, exact staged-backward calls and bytes, slot zero only, maximum active slots one, and zero generation, reuse, trim, or protocol anomalies.
- Decision: 176 GiB is the preferred pinned-host arena for the next one-rack compile gate. Do not change the production default until the 16-process run reproduces the memory margin, numerical audit, and throughput result.

### 2026-08-12 14:50 PT - Isolated failure gates pass; concurrent hook self-deadlocks

- Result: `/dlwh/mok-v15-negative-matrix-4gb200-c88afbb313-20260812-143216` ran from candidate SHA `c88afbb313bb14592f91cbbdb1634a5c812b75fe`. Forward and backward injections before input readiness and before completion all passed in 0.66-0.72 seconds with exact handler and slot counters, zero generation or reuse anomalies, clean runtime closure, and clean interpreter exit.
- Concurrent-control failure: the first two-RunId gate waited five minutes inside `ConsumeTestFailure`. The hook blocked the target device's first host callback while waiting for the second invocation to become fully leased; PJRT could not dispatch the target device's second callback on that occupied lane. After the five-minute native timeout, the executions returned exactly one failure and one success, which rules out cross-RunId cancellation. The factor-four ring reference then failed independently because its padded capacity exceeded the assignment population.
- Fix: make the test injection nonblocking. If fewer than two invocations are fully leased, leave the hook armed and let the first matching invocation proceed; a later matching callback injects only after overlap exists. Cap the ring-reference capacity at its available assignment population. Neither change alters the production protocol or native ABI.
- Next action: rerun only the two concurrent-control gates from a clean pushed snapshot. Require one expected failure and one numerically correct success, both slots active, exact per-slot counters, zero anomalies, closure in under 60 seconds, and terminal Iris success.
