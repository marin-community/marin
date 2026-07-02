---
topic: 6597-moe-mgpu-forward
issue: 6597
description: Hopper Pallas Mosaic MGPU expert-parallel MoE forward/permute_up work
---

# 6597 MoE MGPU Forward: Task Logbook

## Scope
- Goal: implement and benchmark a Hopper Pallas Mosaic MGPU `permute_up` path for expert-parallel Grug MoE.
- Primary metric(s): target-shape `permute_up` steady-state time, correctness vs `ragged_all_to_all`, dropped-route count.
- Constraints: use H100 cluster `cw-us-east-02a` for H100 runs; do not restart Iris; babysit launched H100 jobs.
- Coordinating issue/PR: GitHub issue #6597; codex-chat room `6597-forward`.

## Entry Log

### 2026-06-29 00:37 PDT - FWD-SWQ-001 static-workqueue entrypoint scaffold
- Hypothesis: the current branch needs an explicit, triggerable fused `permute_up` workspace before implementing the revised static workqueue kernel body; the older `/59b1` worktree already contains useful Pallas MGPU harness/backend scaffolding.
- Commit Hash: uncommitted worktree.
- Command:
  - `cp /Users/dlwh/.codex/worktrees/59b1/marin/lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py`
  - `cp /Users/dlwh/.codex/worktrees/59b1/marin/lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
  - `cp /Users/dlwh/.codex/worktrees/59b1/marin/lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/src/levanter/grug/_moe/common.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
- Config: added public `implementation="pallas_mgpu"` wiring and `MoeMgpuConfig(dispatch_static_workqueue_permute_up=True)`.
- Result: local compile/import passed; benchmark harness unit tests passed (`37 passed`). No H100 job launched.
- Interpretation: this branch now has a triggerable Pallas MGPU backend/harness and a distinct static-workqueue edit point. The static-workqueue Pallas body is intentionally not complete yet; the transplanted old chunked body is still a balanced-routing diagnostic and diverges from the revised spec for ragged counts/masked stores.
- Next action: implement the Pallas body behind `_permute_up_mgpu_static_workqueue_kernel`, starting with a V0 serial phase loop that accepts aggregate counts and does not assume balanced rows per `(src,dst,expert)`.

### 2026-06-29 00:49 PDT - FWD-SWQ-002 static-workqueue V0 correctness
- Hypothesis: a single-WG serial static phase loop can prove the revised schedule and metadata boundary before adding producer/compute WG overlap.
- Commit Hash: uncommitted worktree.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - H100 smoke: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-v0-smoke-20260629-004226`
  - H100 ragged smoke: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-v0-ragged-smoke-20260629-004326`
  - H100 target balanced: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-v0-target-balanced-20260629-004501`
- Config:
  - New path: `MoeMgpuConfig(dispatch_static_workqueue_permute_up=True)`.
  - V0 schedule: expert-group outer, peer-phase inner, single WG serial `send_phase -> signal -> wait -> compute_phase`.
  - Hot copy path remote-writes only token payloads to destination `recv_x`; compatibility `recv_src_rank`/`recv_src_assignment` are reconstructed outside the Pallas remote-write path.
  - `dispatch_static_workqueue_max_rows_per_src_expert` controls the static loop bound; smoke used `64`/`256`, target balanced used `512`.
- Result:
  - Local compile/import passed; harness tests passed (`37 passed`).
  - Smoke balanced EP=2/T=128/D=128/I=128/E_local=2/K=2: `matches_baseline=true`, `max_abs_diff=0.0`, dropped=0, `steady_state_time=0.0008232628s`.
  - Smoke uniform: `matches_baseline=true`, `max_abs_diff=0.0`, dropped=7, `steady_state_time=0.0008131892s`.
  - Smoke all_to_rank0: `matches_baseline=true`, `max_abs_diff=0.0`, dropped=256, `steady_state_time=0.0007927089s`.
  - Target balanced EP=8/T=32768/D=2560/I=1280/E_local=32/K=4: `matches_baseline=true`, `max_abs_diff=0.0`, dropped=0, compile `148.3841927s`, `steady_state_time=0.0340771002s`, `126.04 TFLOP/s/rank`, `12.74%` roofline.
- Interpretation:
  - Milestone 2 V0 is partially satisfied for EP=2 smoke, ragged/empty-route smoke, and target balanced correctness.
  - This is not a performance win; target V0 is slower than the earlier staged/fused best because it is serial and uses conservative static loop bounds.
  - Remaining production work: remove/tighten the static max-row bound for real random target routing, add EP=8 random/skewed correctness, then implement V1/V2 producer/compute WG overlap.
- Next action: add V1 two-WG lockstep using the same phase helpers and semaphore accounting, then V2 independent producer/compute loops.

### 2026-06-29 01:12 PDT - FWD-SWQ-003 static-workqueue split-WG overlap attempt
- Hypothesis: separating the producer and W13 compute workgroups inside the static phase schedule will overlap token exchange with W13/SwiGLU compute and improve target forward time.
- Commit Hash: uncommitted worktree.
- Command:
  - H100 split smoke failures while iterating scoped SMEM/barrier allocation:
    - `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-split-smoke-20260629-005214`
    - `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-split-smoke2-20260629-005536`
    - `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-split-smoke3-20260629-005814`
  - H100 split smoke success: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-split-smoke4-20260629-010246`
  - H100 target split: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-split-target-20260629-010458`
- Config:
  - V1 lockstep: `dispatch_static_workqueue_permute_up=True`, `dispatch_split_wg_permute_up=True`.
  - V2 overlap: also `dispatch_split_wg_overlap_permute_up=True`.
  - Target shape used EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, balanced routing, `dispatch_static_workqueue_max_rows_per_src_expert=512`, `dispatch_expert_group_size=16`, `dispatch_chunk_copy_tile=512`.
- Result:
  - Smoke lockstep EP=2/T=128/D=128/I=128/E_local=2/K=2: `matches_baseline=true`, `max_abs_diff=0.0`, dropped=0, compile `3.101385873s`, `steady_state_time=0.0008023430s`.
  - Smoke overlap: `matches_baseline=true`, `max_abs_diff=0.0`, dropped=0, compile `3.166313372s`, `steady_state_time=0.0008605780s`.
  - Target lockstep: `matches_baseline=true`, `max_abs_diff=0.0`, dropped=0, compile `146.3438141s`, `steady_state_time=0.0352881522s`, `121.71 TFLOP/s/rank`.
  - Target overlap: `matches_baseline=true`, `max_abs_diff=0.0`, dropped=0, compile `146.9415520s`, `steady_state_time=0.0361040041s`, `118.96 TFLOP/s/rank`.
- Interpretation:
  - Split-WG implementations are correct on the controlled shapes, but slower than serial static V0 target (`0.0340771002s`).
  - Current fused static path is software-structure-bound: more workgroups/barriers/semaphores add overhead before communication overlap pays for itself.
  - V2 required three WGs (producer, loader, compute) rather than two because Mosaic multithreaded WGMMA setup needs thread-collective SMEM/barrier allocation.
- Next action: expose loop-order control on the static workqueue path and test expert-group outer / rotating-peer inner against peer-phase outer / expert-group inner before adding more overlap structure.

### 2026-06-29 01:38 PDT - FWD-SWQ-004 static-workqueue loop-order experiment
- Hypothesis: if the spec order is slower because it repeatedly switches destination peers within an expert group, exchanging the loops to peer-phase outer / expert-group inner may improve locality or reduce handoff overhead.
- Commit Hash: uncommitted worktree.
- Code change:
  - `_permute_up_mgpu_static_workqueue_kernel` now honors `MoeMgpuConfig.dispatch_copy_schedule`.
  - `dispatch_copy_schedule="expert_group_peer"` is the requested expert-group outer / rotating-peer inner schedule.
  - `dispatch_copy_schedule="assignment_major"` runs the exchanged order: peer-phase outer / expert-group inner.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - H100 target attempt: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-looporder-target-20260629-011425`
  - H100 smoke: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-looporder-smoke-20260629-012622`
  - H100 medium group16: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-looporder-medium-20260629-012842`
  - H100 medium group8: `/dlwh/bench-grug-moe-pallas-mgpu-static-swq-group8-medium-20260629-013335`
- Result:
  - Local compile passed; harness tests passed (`37 passed`, 11 warnings).
  - Target attempt EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, group16, `max_rows=512`: first `expert_group_peer` row emitted only the start event and no result after >11 minutes; stopped the job (`JOB_STATE_KILLED`) to avoid burning the reservation.
  - Smoke EP=2/T=128/D=128/I=128/E_local=2/K=2, group1, `max_rows=64`:
    - `expert_group_peer`: compile `2.48375749s`, `steady_state_time=0.00036280796s`, dropped=0.
    - `assignment_major`: compile `2.58799176s`, `steady_state_time=0.00034401395s`, dropped=0.
  - Medium EP=8/T=4096/D=2560/I=1280/E_local=32/K=4, group16, `max_rows=64`:
    - `expert_group_peer`: compile `85.14961777s`, `steady_state_time=0.00720923968s`, `55.85 TFLOP/s/rank`, dropped=0.
    - `assignment_major`: compile `85.46008126s`, `steady_state_time=0.00738581999s`, `54.52 TFLOP/s/rank`, dropped=0.
  - Medium EP=8/T=4096/D=2560/I=1280/E_local=32/K=4, group8, `max_rows=64`:
    - `expert_group_peer`: compile `86.18545449s`, `steady_state_time=0.00726086235s`, `55.46 TFLOP/s/rank`, dropped=0.
    - `assignment_major`: compile `85.92259501s`, `steady_state_time=0.00732990494s`, `54.93 TFLOP/s/rank`, dropped=0.
- Interpretation:
  - On tiny smoke, exchanged loop order is slightly faster, but at realistic EP=8/expert geometry, the requested `expert_group_peer` order is slightly better.
  - Group size 8 versus 16 is nearly flat at medium size; group16 wins slightly in this serial implementation (`0.007209s` vs `0.007261s`).
  - The target compile/no-result behavior needs a separate compile-scaling check; medium compiles reliably in ~85-86s, while target should have been closer to prior ~146s but did not complete before manual stop.
  - Performance remains structure-bound: serial static and split-WG static are correct but far slower than the earlier non-static fused/staged best, so the next useful speed work is reducing phase/barrier count or moving to a true pipeline that does not materialize/zero global `recv_x` for the whole capacity.
- Next action: keep the loop-order switch because it is a useful direct experiment hook; do not pursue `assignment_major` for target unless another change makes the communication/computation overlap real.

### 2026-06-29 01:59 PDT - FWD-SWQ-005 balanced-diagnostic guard and recv_x zeroing negative result
- Hypothesis:
  - Milestone 0 requires the old chunked balanced diagnostic path to fail instead of silently running on real routing.
  - Milestone 7 suggests removing full `recv_x` zeroing because valid rows are uniquely remote-written before compute.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `_require_chunked_balanced_counts(send_counts, rows_per_source_expert)`.
  - `_permute_up_mgpu_fused_chunked_kernel` now calls the guard before using `rows_per_source_expert = assignments // (EP * E_local)`.
  - Added tests proving equal buckets are accepted and unequal buckets raise.
  - Temporarily removed static-workqueue `recv_x` zeroing, then reverted it after H100 smoke failed to make progress.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - H100 capacity-bound no-zero smoke: `/dlwh/bench-grug-moe-pallas-mgpu-static-no-recvx-zero-smoke-20260629-014437`
  - H100 bounded no-zero smoke: `/dlwh/bench-grug-moe-pallas-mgpu-static-no-recvx-zero-bounded-smoke-20260629-015308`
- Config:
  - Smoke shape EP=2/T=128/D=128/I=128/E_local=2/K=2, bf16, `dispatch_static_workqueue_permute_up=True`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=1`, `dispatch_chunk_copy_tile=128`.
  - First no-zero job used `dispatch_static_workqueue_max_rows_per_src_expert=None` (capacity-derived bound).
  - Second no-zero job used `dispatch_static_workqueue_max_rows_per_src_expert=128`.
- Result:
  - Local validation passed: `39 passed`, 11 warnings.
  - Capacity-bound no-zero job: baseline `ragged_all_to_all` uniform row succeeded (`steady_state_time=0.0003463403s`), but the first static row emitted only the start event and no result after several minutes; stopped manually. Final state `JOB_STATE_KILLED`.
  - Bounded no-zero job: baseline uniform row succeeded (`steady_state_time=0.0014115054s`), but the first static row again emitted only the start event and no result after several minutes; stopped manually. Final state `JOB_STATE_KILLED`.
  - Restored `recv_x` zeroing after the failed smokes.
- Interpretation:
  - Milestone 0 guardrail is now implemented for direct library use, not just benchmark CLI use.
  - Removing full `recv_x` zeroing is not a safe standalone change in the current Mosaic kernel structure. It appears to affect compile/progress even at tiny EP=2 shapes, so keep the zeroing until the scratch lifetime/output contract is changed more structurally.
  - The next spec-aligned cleanup should not be another micro-removal inside the current output-scratch structure; it should either introduce a proper internal scratch/lifetime approach or focus on metadata reconstruction/downstream API so `permute_up` does less compatibility work.
- Next action: keep the balanced-count guard, leave `recv_x` zeroing in place, and continue toward variable-count static V0 correctness coverage rather than target performance.

### 2026-06-29 02:16 PDT - FWD-SWQ-006 compact-layout row-block bound and variable-count V0 compare
- Hypothesis:
  - Static V0 compute over compact expert-major rows must account for subranges that do not start on `block_m` boundaries.
  - A source/expert subrange of length `max_rows` can touch `ceil((max_rows + block_m - 1) / block_m)` global blocks, not just `ceil(max_rows / block_m)`.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `_max_blocks_touched_by_unaligned_range(max_rows, block_m)`.
  - Static workqueue compute loop now uses that helper for `max_global_blocks_per_src_expert`.
  - Added tests for the compact-layout tail bound; local suite now has 43 tests.
- Command:
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - Full-forward variable smoke, stopped after no static row: `/dlwh/bench-grug-moe-pallas-mgpu-static-v0-variable-smoke-20260629-020055`
  - Isolated uniform compare: `/dlwh/bench-grug-moe-pallas-mgpu-static-v0-variable-compare-20260629-021006`
  - Isolated all-to-rank0/empty compare: `/dlwh/bench-grug-moe-pallas-mgpu-static-v0-empty-compare-20260629-021259`
- Config:
  - EP=2/T=128/D=128/I=128/E_local=2/K=2, bf16, `dispatch_static_workqueue_permute_up=True`.
  - `dispatch_static_workqueue_max_rows_per_src_expert=128`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=1`, `dispatch_chunk_copy_tile=128`.
- Result:
  - Local tests passed: `43 passed`, 11 warnings.
  - Full-forward variable smoke: `ragged_all_to_all` uniform baseline row succeeded (`steady_state_time=0.0003128470s`), then first full-forward static row emitted start only and no result after several minutes; stopped manually. Final state `JOB_STATE_KILLED`.
  - Isolated `permute_up_compare`, uniform routing:
    - Job succeeded.
    - `matches_baseline=true`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, dropped=0.
    - compile `2.852730321s`, `steady_state_time=0.0004331067s`.
  - Isolated `permute_up_compare`, all_to_rank0 routing:
    - Job succeeded.
    - `matches_baseline=true`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, dropped=192.
    - compile `2.841564004s`, `steady_state_time=0.0004623284s`.
- Interpretation:
  - Static V0 `permute_up` now has direct H100 evidence for variable counts and many empty subranges at EP=2 in the isolated stage harness.
  - Full forward still stalls/compiles too long on the same variable routing smoke; this points outside the isolated `permute_up` stage, likely the downstream full-forward composition or the benchmark candidate structure, not the static workqueue kernel alone.
  - The compact-layout bound fix is required for correctness on non-block-aligned `(src,dst,expert)` row ranges.
- Next action: add a small/medium EP=8 isolated `permute_up_compare` before revisiting full forward, then investigate why full-forward static variable smoke fails to produce a row while isolated `permute_up_compare` succeeds.

### 2026-06-29 02:27 PDT - FWD-SWQ-007 EP=8 isolated compare and user-kernel hook surface
- Hypothesis:
  - Before more full-forward debugging, validate static V0 `permute_up` on EP=8 variable/skewed routing in the isolated stage harness.
  - The branch should expose a clear edit point and explicit config trigger for a hand-written fused `permute_up` kernel, independent of the static-workqueue experiment.
- Commit Hash: uncommitted worktree.
- Code change:
  - `_permute_up_mgpu_user_kernel(...)` is the explicit hand-edit hook selected by `MoeMgpuConfig(dispatch_user_kernel_permute_up=True)`.
  - `moe_mlp(..., pallas_mgpu_config=...)` and `MoEExpertMlp.init(..., pallas_mgpu_config=...)` now thread explicit Pallas MGPU config through the public EP path instead of always constructing defaults.
  - The benchmark already exposes `--dispatch-user-kernel-permute-up` and records the flag in `block_sizes`.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/tests/grug/test_grugformer_moe.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grugformer_moe.py::test_pallas_mgpu_moe_mlp_uses_explicit_config_for_validation -q`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - H100 uniform compare: `/dlwh/bench-grug-moe-pallas-mgpu-static-v0-ep8-uniform-compare-20260629-021744`
  - H100 skew compare: `/dlwh/bench-grug-moe-pallas-mgpu-static-v0-ep8-skew-compare-20260629-022019`
- Config:
  - Uniform: EP=8/T=1024/D=128/I=128/E_local=32/K=4, routing `uniform`, `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=64`, `dispatch_expert_group_size=16`, `dispatch_chunk_copy_tile=128`.
  - Skew: EP=8/T=256/D=128/I=128/E_local=32/K=4, routing `all_to_rank0`, `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=256`, `dispatch_expert_group_size=16`, `dispatch_chunk_copy_tile=128`.
- Result:
  - Local compile passed.
  - Public config validation test passed: `1 passed`, 11 warnings.
  - Benchmark harness tests passed: `43 passed`, 11 warnings.
  - Uniform EP=8 isolated `permute_up_compare`: `matches_baseline=true`, `max_abs_diff=0.0556640625`, `mean_abs_diff=5.142379450262524e-06`, dropped=0, compile `91.31897399004083s`, `steady_state_time=0.0036251016814882555s`, `0.18512270798552888 TFLOP/s/rank`.
  - Skew EP=8 isolated `permute_up_compare`: `matches_baseline=true`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`, dropped=6912, compile `12.898001127992757s`, `steady_state_time=0.003297776992743214s`, `0.0508743193882376 TFLOP/s/rank`.
- Interpretation:
  - Isolated static V0 correctness now covers EP=8 uniform routing and an EP=8 many-empty/overflow skew case. Skew drops are expected from `all_to_rank0` over capacity.
  - Static V0 remains a correctness scaffold, not the right speed path; timings are dominated by software structure and conservative phase/loop overhead rather than WGMMA throughput.
  - The requested hand-written-kernel slot is available without competing with the static path: set `MoeMgpuConfig(dispatch_user_kernel_permute_up=True)` and edit `_permute_up_mgpu_user_kernel`.
- Next action:
  - If continuing with agent implementation, isolate the full-forward static stall separately from isolated `permute_up`.
  - If handing to the user, start from `_permute_up_mgpu_user_kernel` and use `--dispatch-user-kernel-permute-up --include-pallas-stages --pallas-stages permute_up_compare` for direct compare rows.

### 2026-06-29 02:37 PDT - FWD-SWQ-008 static schedule default and swizzle knobs
- Hypothesis:
  - The static-workqueue path should default to the spec schedule, expert-group outer with rotating peer inner.
  - Peer-phase and expert-group swizzles should be explicit config knobs, but must be passed into Mosaic kernels as inputs instead of captured non-scalar constants.
- Commit Hash: uncommitted worktree.
- Code change:
  - Changed `MoeMgpuConfig.dispatch_copy_schedule` default to `expert_group_peer`.
  - Added `dispatch_peer_phase_order: tuple[int, ...] | None` and `dispatch_swizzle_expert_groups: bool`.
  - Added `_static_workqueue_phase_indices(...)` and routed both static-workqueue and synthetic chunked phase mapping through it.
  - Benchmark CLI now exposes `--dispatch-peer-phase-order` and `--dispatch-swizzle-expert-groups`; result rows include both fields.
  - Fixed Mosaic lowering failure by passing the peer-order vector as a kernel input rather than capturing it in the kernel body.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - H100 failed capture smoke: `/dlwh/bench-grug-moe-pallas-mgpu-static-swizzle-smoke-20260629-023256`
  - H100 fixed swizzle smoke: `/dlwh/bench-grug-moe-pallas-mgpu-static-swizzle-smoke2-20260629-023539`
- Config:
  - EP=4/T=256/D=128/I=128/E_local=4/K=2, routing `uniform`.
  - `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=128`.
  - `dispatch_copy_schedule=expert_group_peer`, `dispatch_peer_phase_order=(0, 1, 3, 2)`, `dispatch_swizzle_expert_groups=True`, `dispatch_expert_group_size=2`, `dispatch_chunk_copy_tile=128`.
- Result:
  - Local compile passed.
  - Harness tests passed: `46 passed`, 11 warnings.
  - First H100 smoke reached a structured error row: `ValueError: ... captures non-scalar constants [i32[4]]. You should pass them as inputs.`
  - Fixed H100 smoke succeeded: `matches_baseline=true`, `max_abs_diff=0.0771484375`, `mean_abs_diff=0.0008432127651758492`, dropped=0, compile `4.329810302006081s`, `steady_state_time=0.0011641139863058925s`.
- Interpretation:
  - Milestone 6 swizzle controls are now implemented and H100-validated on a small EP=4 isolated `permute_up_compare`.
  - This is not a speed result; it proves schedule configurability and the Mosaic constant-capture fix.
  - Static V0 remains structurally slower than the tuned balanced synthetic path and still carries full `recv_x` scratch/metadata compatibility costs.
- Next action:
  - Add target/medium swizzle performance only if static V2 becomes competitive enough to make schedule order meaningful.
  - Otherwise continue closing correctness/structure gaps: full-forward static variable stall and production metadata reconstruction/downstream contract.

### 2026-06-29 02:52 PDT - FWD-SWQ-009 full-forward config threading and remaining timeout
- Hypothesis:
  - The earlier full-forward static smoke did not test the same candidate as isolated `permute_up_compare` because benchmark `_benchmark_implementation()` was not passing `MoeMgpuConfig` into public `moe_mlp(... implementation="pallas_mgpu")`.
  - Once the config is actually threaded, full-forward static V0 should either complete or expose a more precise integration bottleneck.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `_config_for_moe_call(implementation, config)` in the benchmark and pass `pallas_mgpu_config=config` only for full `implementation="pallas_mgpu"`.
  - Fixed public dispatcher regression from config threading: `grug_moe.moe_mlp` now passes `pallas_mgpu_config` only to `_moe_mlp_ep_pallas_mgpu_local`, not to `ring`/`ragged_all_to_all`/`deepep` shard functions.
  - Added benchmark unit coverage that only `pallas_mgpu` receives the config.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py lib/levanter/tests/grug/test_grugformer_moe.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py lib/levanter/tests/grug/test_grugformer_moe.py::test_pallas_mgpu_moe_mlp_uses_explicit_config_for_validation -q`
  - Failed baseline smoke before dispatcher fix: `/dlwh/bench-grug-moe-pallas-mgpu-static-full-forward-smoke-20260629-024043`
  - Fixed baseline/full-forward smoke: `/dlwh/bench-grug-moe-pallas-mgpu-static-full-forward-smoke2-20260629-024414`
- Config:
  - EP=2/T=128/D=128/I=128/E_local=2/K=2, routing `uniform`.
  - `--implementations ragged_all_to_all pallas_mgpu --pass-mode forward`.
  - `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=128`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=1`, `dispatch_chunk_copy_tile=128`, `candidate_timeout_seconds=300`.
- Result:
  - Local compile passed.
  - Focused tests passed: `48 passed`, 11 warnings.
  - First H100 smoke before dispatcher fix:
    - `ragged_all_to_all` row errored with `TypeError: _moe_mlp_ep_ragged_a2a_local() got an unexpected keyword argument 'pallas_mgpu_config'`.
    - `pallas_mgpu` static row succeeded without baseline comparison: compile `2.59791513485834s`, `steady_state_time=0.0005899060051888227s`, dropped=0.
  - Fixed H100 smoke:
    - `ragged_all_to_all` baseline succeeded: compile `2.164657220011577s`, `steady_state_time=0.00031093298457562923s`, dropped=0.
    - `pallas_mgpu` static full-forward row emitted a structured timeout: `TimeoutError: benchmark candidate exceeded 300.000s timeout`.
    - The Iris job remained running after the timeout row, so it was stopped manually. Final state: `JOB_STATE_KILLED`, error `Terminated by user`.
- Interpretation:
  - Full-forward benchmark integration is now real: Pallas full-forward rows receive the static-workqueue config, and non-Pallas EP baselines no longer receive Pallas-only kwargs.
  - The full-forward static V0 timeout is now stronger evidence of a full-composition compile/runtime issue, not a harness config omission.
  - Isolated static `permute_up_compare` remains correct on the same small-family shapes, so the remaining issue is likely downstream composition/custom-VJP/full-forward compile structure rather than basic static `permute_up` value correctness.
- Next action:
  - Add a narrower full-forward decomposition that uses static `permute_up` output plus staged `down_unpermute` in separate jitted calls, or add a split full-forward compare stage, to distinguish XLA compile blow-up from a runtime deadlock.
  - Do not claim full-forward static V0 acceptance until a baseline-comparison row completes without timeout.

### 2026-06-29 03:01 PDT - FWD-SWQ-010 direct up/down compare stage still hangs
- Hypothesis:
  - If the full `moe_mlp(... implementation="pallas_mgpu")` timeout is caused mainly by public dispatcher/custom-VJP wrapping, a direct `permute_up_mgpu` + `down_unpermute_mgpu` shard-map benchmark should compile and emit a row on the tiny EP=2 smoke.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added benchmark stage `up_down_compare_split`.
  - The stage compiles/runs a baseline permute-up+down function with static/chunked/user permute-up disabled, then compiles/runs a candidate permute-up+down function with the active config.
  - It host-compares final token outputs and dropped-route counts, avoiding the public `moe_mlp` wrapper and custom-VJP path.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - H100: `/dlwh/bench-grug-moe-pallas-mgpu-up-down-static-smoke-20260629-095737`
- Config:
  - EP=2/T=128/D=128/I=128/E_local=2/K=2, routing `uniform`.
  - `--implementations none --include-pallas-stages --pallas-stages up_down_compare_split`.
  - `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=128`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=1`, `dispatch_chunk_copy_tile=128`.
- Result:
  - Local compile passed.
  - Focused benchmark tests passed: `47 passed`, 11 warnings.
  - H100 job reached the new stage start event at 09:58:13 UTC and emitted no result row by 03:00:54 PDT, after roughly 2 minutes 41 seconds at stage compile/runtime and 2 minutes 58 seconds total job duration.
  - Job was stopped manually to avoid burning the reservation. Final state: `killed`, error `Terminated by user`.
- Interpretation:
  - This narrows the timeout from public `moe_mlp` integration to direct composition of static `permute_up_mgpu` with `down_unpermute_mgpu`.
  - Isolated static `permute_up_compare_split` remains correct, so the failure is likely a software-structure/composition issue around carrying the static-layout outputs into downstream return/combine work, not a pure dispatch value mismatch.
  - Current static path is not yet a usable fused permute-up/down route. The user-editable `_permute_up_mgpu_user_kernel` hook remains the least entangled place to write a new fused permute-up kernel.
- Next action:
  - Do not scale this direct up/down stage to target until the tiny smoke emits a row.
  - Prefer giving the hand-written kernel path a minimal activation/config surface and using `permute_up_compare_split` as the correctness loop, then add downstream composition only after the new kernel has isolated dispatch correctness.

### 2026-06-29 03:04 PDT - FWD-SWQ-011 user-kernel hook cleanup
- Hypothesis:
  - The fastest next iteration path is a clean, exclusive hook for a hand-written fused `permute_up` kernel, validated first through isolated `permute_up_compare_split` rather than direct up/down composition.
- Commit Hash: uncommitted worktree.
- Code change:
  - Kept `_permute_up_mgpu_user_kernel(...)` as the edit point selected by `MoeMgpuConfig(dispatch_user_kernel_permute_up=True)` or benchmark flag `--dispatch-user-kernel-permute-up`.
  - Made user-kernel mode mutually exclusive with `dispatch_static_workqueue_permute_up` and `dispatch_chunked_permute_up` in both `MoeMgpuConfig.__post_init__` and benchmark CLI validation.
  - Improved the hook `NotImplementedError` to name the exact file and a minimal benchmark command shape.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
- Result:
  - Local compile passed.
  - Focused benchmark tests passed: `49 passed`, 11 warnings.
- Interpretation:
  - The hand-written kernel path is now unambiguous: a command using `--dispatch-user-kernel-permute-up` will route through `_permute_up_mgpu_user_kernel` unless the hook itself raises.
  - No H100 run was launched for this cleanup.

### 2026-06-29 03:13 PDT - FWD-SWQ-012 static up/down composition boundary
- Hypothesis:
  - The tiny `up_down_compare_split` hang could be caused by either normal up/down baseline composition, static up/down same-jit composition, or invalid static layout metadata feeding `down_unpermute`.
  - Splitting baseline-only, candidate-only, and separate-jit `down_unpermute` diagnostics should identify the boundary.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added independent benchmark stages `up_down_baseline` and `up_down_candidate`.
  - Threaded `--candidate-timeout-seconds` into Pallas stage diagnostics and recorded it on the new rows.
  - Added tests that these new stages are dependency-isolated.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - Combined H100 baseline+candidate: `/dlwh/bench-grug-moe-pallas-mgpu-up-down-split-diagnose-20260629-100617`
  - Baseline-only H100: `/dlwh/bench-grug-moe-pallas-mgpu-up-down-baseline-only-20260629-100938`
  - Separate-jit static down H100: `/dlwh/bench-grug-moe-pallas-mgpu-static-down-separate-20260629-101132`
- Config:
  - EP=2/T=128/D=128/I=128/E_local=2/K=2, routing `uniform`.
  - Static config for candidate/separate-jit: `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=128`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=1`, `dispatch_chunk_copy_tile=128`.
  - Baseline-only uses the new `up_down_baseline` stage, which explicitly disables static/chunked/user permute-up inside the function.
- Result:
  - Local compile passed.
  - Focused benchmark tests passed: `49 passed`, 11 warnings.
  - Combined H100 job:
    - `up_down_baseline` started at 10:06:52 UTC.
    - `up_down_candidate` started at 10:06:55 UTC, proving baseline-only returned in that process.
    - Candidate emitted no row after roughly 2 minutes 20 seconds at the candidate stage; the job was stopped manually. Final state: `killed`, error `Terminated by user`.
    - Note: `_benchmark_pallas_stages` buffers rows until the whole stage set returns, so the successful baseline row was not emitted from this combined job before the candidate hang.
  - Baseline-only H100 row succeeded:
    - `compile_time=2.8548796160612255s`
    - `steady_state_time=0.0006226190210630497s`
    - `dropped_routes=0`
    - `effective_tflops_per_rank=0.05052412299625917`
  - Separate-jit static down H100 row succeeded:
    - This selected `--pallas-stages down_unpermute`, so the harness first ran static `permute_up_fn` as a separate jitted call, then timed `down_unpermute_fn`.
    - `compile_time=0.9298681269865483s`
    - `steady_state_time=0.0003441936957339446s`
    - `dropped_routes=0`
    - `effective_tflops_per_rank=0.03046470673334267`
- Interpretation:
  - Normal baseline up/down composition is fine.
  - Static `permute_up` output and reconstructed metadata are sufficient for `down_unpermute` when the two pieces are run as separate jitted calls.
  - The bad boundary is same-jit composition of static `permute_up_mgpu` followed by `down_unpermute_mgpu`.
  - This is software-structure-bound, likely XLA/Pallas composition or semaphore/resource interaction across two MGPU kernels in one jitted function, not a pure communication or compute throughput issue.
- Next action:
  - For forward perf iteration, use isolated `permute_up`/`permute_up_compare_split` and separate stage timing rather than same-jit up/down until the composition issue is solved.
  - Consider adding a production/public guard that static workqueue is only enabled for isolated benchmark stages, or investigate whether the static kernel can return a layout that avoids composing two MGPU semaphore-heavy kernels inside the same jitted function.

### 2026-06-29 03:24 PDT - FWD-SWQ-013 barrier fixes same-jit static up/down
- Hypothesis:
  - The same-jit static up/down hang is an XLA/Pallas scheduling/optimization issue across the static `permute_up_mgpu` custom-call boundary, not a layout correctness problem.
  - Applying `lax.optimization_barrier` to the static workqueue layout outputs should force a safe compiler boundary before downstream `down_unpermute_mgpu`.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `lax.optimization_barrier((hidden, recv_src_rank, recv_src_assignment, rows_per_expert))` in the `dispatch_static_workqueue_permute_up` branch before returning `MoeMgpuUpLayout`.
  - Fixed benchmark forward comparison to reshard the baseline output to the candidate output sharding before diffing. This avoids a post-run `ShardingTypeError` when comparing `ragged_all_to_all` vs `pallas_mgpu`.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
  - Candidate-only same-jit H100: `/dlwh/bench-grug-moe-pallas-mgpu-up-down-candidate-barrier-20260629-101553`
  - Same-jit up/down compare H100: `/dlwh/bench-grug-moe-pallas-mgpu-up-down-compare-barrier-20260629-101739`
  - Public full-forward first H100 with comparison sharding failure: `/dlwh/bench-grug-moe-pallas-mgpu-static-full-forward-barrier-20260629-101923`
  - Public full-forward fixed H100: `/dlwh/bench-grug-moe-pallas-mgpu-static-full-forward-barrier2-20260629-102157`
- Config:
  - EP=2/T=128/D=128/I=128/E_local=2/K=2, routing `uniform`.
  - Static config: `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=128`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=1`, `dispatch_chunk_copy_tile=128`.
- Result:
  - Local compile passed.
  - Focused benchmark tests passed: `49 passed`, 11 warnings.
  - Candidate-only same-jit static up/down now succeeds:
    - `compile_time=2.522900876123458s`
    - `steady_state_time=0.0005526226790000995s`
    - `dropped_routes=0`
  - Same-jit up/down compare now succeeds:
    - `matches_baseline=true`
    - `max_abs_diff=0.0`
    - `mean_abs_diff=0.0`
    - `compile_time=3.9794006960000843s`
    - `steady_state_time=0.0011788442886124053s`
    - `dropped_routes=0`
  - Public full-forward first rerun no longer timed out, but comparison failed after candidate execution with `ShardingTypeError: sub got incompatible shardings for broadcasting: ('expert', None), (('data', 'expert'), None).`
  - After the benchmark compare reshard fix, public full-forward succeeds:
    - `ragged_all_to_all`: `compile_time=2.1421090750955045s`, `steady_state_time=0.0003474426921457052s`, dropped=0.
    - `pallas_mgpu` static: `compile_time=1.8389369670767337s`, `steady_state_time=0.0005390216441204151s`, dropped=0.
    - `matches_baseline=true`
    - `max_abs_diff=0.000244140625`
    - `mean_abs_diff=2.350123213545885e-05`
- Interpretation:
  - The barrier fixes the same-jit static up/down hang on the tiny EP=2 smoke.
  - The previous direct-composition failure was not bad static metadata; it was a compiler/scheduling boundary issue.
  - Static V0 full-forward is now usable through the public `implementation="pallas_mgpu"` path on the smoke shape, but still slower than `ragged_all_to_all` on this tiny shape.
- Next action:
  - Re-run at EP=8 and target-like dimensions to see whether the barrier preserves correctness and whether static V0 remains near the earlier isolated `permute_up` result.
  - If target full-forward succeeds, resume performance work on phase granularity/copy tile and V1/V2 overlap.

### 2026-06-29 03:35 PDT - FWD-SWQ-014 target forward static workqueue speedup
- Hypothesis:
  - After the `lax.optimization_barrier` fix, static-workqueue `permute_up` should compose through the public forward path at the exact EP=8 target shape and retain the isolated `permute_up` speed advantage.
- Commit Hash: uncommitted worktree.
- Code change:
  - No additional kernel changes beyond FWD-SWQ-013.
- Command:
  - Target compare H100: `/dlwh/bench-grug-moe-pallas-mgpu-static-target-forward-barrier-20260629-102456`
  - Target repeat H100: `/dlwh/bench-grug-moe-pallas-mgpu-static-target-forward-repeat-barrier-20260629-103155`
- Config:
  - EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, routing `uniform`, capacity factor `1.25`.
  - Static config: `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=1024`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=16`, `dispatch_chunk_copy_tile=256`.
  - Compare run: `--implementations ragged_all_to_all pallas_mgpu --warmup 1 --steps 3`.
  - Repeat run: `--implementations pallas_mgpu --warmup 2 --steps 5`.
- Result:
  - Target compare:
    - `ragged_all_to_all`: `compile_time=49.55544659704901s`, `steady_state_time=0.0829358536672468s`, dropped=0, `38.83996280937942 TFLOP/s/rank`.
    - static `pallas_mgpu`: `compile_time=82.82848281506449s`, `steady_state_time=0.029945751690926652s`, dropped=0, `107.56869639629076 TFLOP/s/rank`.
    - `matches_baseline=true`
    - `max_abs_diff=0.01953125`
    - `mean_abs_diff=0.001220707199536264`
  - Target repeat:
    - static `pallas_mgpu`: `compile_time=83.26943371701054s`, `steady_state_time=0.03021237477660179s`, dropped=0, `106.61940664441589 TFLOP/s/rank`.
- Interpretation:
  - Static-workqueue forward now works at the exact target shape through public `implementation="pallas_mgpu"`.
  - Target steady-state speed is stable around `0.030s`, a nontrivial improvement over:
    - same-run `ragged_all_to_all`: `0.08294s`
    - previous non-static target `pallas_mgpu`: `0.03923s` from `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260628-target-forward-compare-tiled`.
  - This is still static V0, not the overlapped V2 pipeline from the spec, so the next performance work should focus on phase granularity and split-WG overlap without losing this public-forward correctness.
- Next action:
  - Benchmark static target with `expert_group_size` 8/16/32 and `dispatch_chunk_copy_tile` 256/512.
  - Revisit V1/V2 split-WG now that public forward composition is fixed by the barrier.

### 2026-06-29 03:55 PDT - FWD-SWQ-015 static target phase/tile matrix and user kernel hook
- Hypothesis:
  - If the static workqueue is phase-overhead/copy-loop bound, smaller expert groups and wider row copy tiles may improve target forward without changing the protocol.
  - For the user's revised direction, keep a standalone user-kernel edit point available instead of continuing to pile more speculative variants into the static path.
- Commit Hash: uncommitted worktree.
- Code status:
  - `MoeMgpuConfig(dispatch_user_kernel_permute_up=True)` routes `permute_up_mgpu` into `_permute_up_mgpu_user_kernel(...)`.
  - The user hook is mutually exclusive with `dispatch_static_workqueue_permute_up` and `dispatch_chunked_permute_up` in both config validation and benchmark CLI parsing.
  - The benchmark flag is `--dispatch-user-kernel-permute-up`.
  - Recommended local/H100 validation stage for a new user kernel is:
    `--implementations none --include-pallas-stages --pallas-stages permute_up_compare_split --dispatch-user-kernel-permute-up`.
- Command:
  - H100 matrix: `/dlwh/bench-grug-moe-pallas-mgpu-static-target-tune-eg-tile-20260629-103742`
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Local: `uv run --package marin-levanter pytest lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py -q`
- Config:
  - EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, routing `uniform`, capacity factor `1.25`.
  - Static config for matrix: `dispatch_static_workqueue_permute_up=True`, `dispatch_static_workqueue_max_rows_per_src_expert=1024`, `dispatch_copy_schedule=expert_group_peer`, `--implementations pallas_mgpu`, `--pass-mode forward`, `--warmup 1`, `--steps 3`.
- Result:
  - H100 matrix job succeeded, no active H100 jobs remain from this pass.
  - Static target rows:
    - `expert_group_size=8`, `copy_tile=256`: `steady_state_time=0.028846212662756443s`, `111.6689220057976 TFLOP/s/rank`, dropped=0.
    - `expert_group_size=8`, `copy_tile=512`: `steady_state_time=0.02501240566683312s`, `128.78511227216342 TFLOP/s/rank`, dropped=0.
    - `expert_group_size=16`, `copy_tile=256`: `steady_state_time=0.030026244387651484s`, `107.28033217916366 TFLOP/s/rank`, dropped=0.
    - `expert_group_size=16`, `copy_tile=512`: `steady_state_time=0.026737209021424253s`, `120.4772521103031 TFLOP/s/rank`, dropped=0.
    - `expert_group_size=32`, `copy_tile=256`: `steady_state_time=0.031730903002123036s`, `101.51698083677213 TFLOP/s/rank`, dropped=0.
    - `expert_group_size=32`, `copy_tile=512`: `steady_state_time=0.027855980365226667s`, `115.63856054483503 TFLOP/s/rank`, dropped=0.
  - Local compile passed.
  - Focused benchmark tests passed: `49 passed`, 11 warnings.
- Interpretation:
  - Best static target row in this matrix is `expert_group_size=8`, `copy_tile=512` at `0.02501240566683312s`.
  - Wider copy tile helps every expert-group size tested; smaller expert group wins for the static full-forward composition.
  - This confirms the static path is still dominated by software/copy-loop/phase structure. It is faster than the older staged target forward, but it is not the requested overlapped token-exchange/W13 kernel.
  - The worktree now has a clean user hook and benchmark switch for replacing the token exchange/W13 body directly.
- Next action:
  - Stop tuning static V0 unless needed as a reference.
  - Implement the next fused experiment inside `_permute_up_mgpu_user_kernel(...)`, validate with `permute_up_compare_split`, then only promote it into the public `pallas_mgpu` path if it beats the existing best on target.

### 2026-06-29 04:12 PDT - FWD-SWQ-016 compact standalone kernel playground
- Hypothesis:
  - The fastest way for manual iteration is a compact, mostly dependency-free script that exposes the fused balanced-routing kernel and a tuning loop without the full Levanter MoE dispatcher/harness.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`.
  - The script imports only standard library, `jax`, `jax.experimental.pallas`, Mosaic GPU, NumPy, and `jaxtyping`.
  - It contains:
    - `CompactConfig` and `BenchShape`.
    - `compact_permute_up_mgpu(...)`: balanced-routing MGPU token exchange into per-phase scratch, followed by local W13/SwiGLU WGMMA compute.
    - `compact_reference(...)`: slow JAX reference for small correctness checks.
    - `--check` correctness mode comparing hidden values and metadata.
    - A bounded tuning loop over expert-group size, copy tile, copy rows, max concurrent steps, and grid block-N.
    - JSON/JSONL rows with compile time, steady-state time, device info, shape, config, errors, diff metrics, and estimated TFLOP/s/rank.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
  - Local: `uv run --package marin-levanter python lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py --help | head -n 80`
- Result:
  - Local syntax compile passed.
  - CLI import/help passed.
  - No H100 job was launched for this file creation.
- Notes:
  - This compact script intentionally assumes balanced routing: `T*K` must be divisible by `EP*E_local`, and each source/global-expert bucket has `rows_per_source_expert` rows.
  - It omits Marin/Levanter/Haliax routing and dispatcher code by design.
  - It is a playground for the fused balanced kernel, not a production ragged-count implementation.
- Example target command:
  - `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py --ep-size 8 --tokens-per-rank 32768 --hidden-dim 2560 --intermediate-dim 1280 --experts-per-rank 32 --topk 4 --expert-group-sizes 8,16,32 --copy-tiles 256,512 --copy-rows 1 --warmup 1 --steps 3 --jsonl scratch/compact_moe_permute_up_mgpu.jsonl`

### 2026-06-29 09:33 PDT - FWD-SWQ-017 compact static workqueue variant
- Hypothesis:
  - The compact playground should include both edit targets: the scratch/chunked fused kernel and a static workqueue kernel that materializes destination `recv_x` phase-by-phase before W13 compute.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `compact_static_workqueue_permute_up_mgpu(...)` to `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`.
  - Added `--implementations chunked,static` support. Rows now report `pallas_mgpu_compact_chunked` or `pallas_mgpu_compact_static`.
  - The static variant uses the same balanced synthetic routing and output layout as the chunked compact variant.
  - Fixed standalone harness issues found by H100 smoke:
    - create the parent directory for `--jsonl`;
    - use this JAX version's `shard_map(..., check_vma=False)`;
    - strip/restore the local leading EP axis inside the shard body;
    - use a 1D logical device id for compact mesh semaphore signals;
    - suppress noisy git SHA stderr when the Iris job bundle is not a git checkout.
- Command:
  - Local: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
  - Failed H100 smoke, missing jsonl parent: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260629-162709`
  - Failed H100 smoke, `check_rep` shard_map mismatch: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260629-162757`
  - Structured H100 smoke rows, local shape output-spec issue: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260629-162942`
  - Structured H100 smoke rows, 1D semaphore id issue: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260629-163050`
  - Passing H100 smoke: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260629-163145`
- Passing smoke config:
  - EP=8/T=512/D=128/I=128/E_local=4/K=4, bf16.
  - `--implementations chunked,static`, `expert_group_size=4`, `copy_tile=128`, `copy_rows=1`, `warmup=1`, `steps=2`, `--check`.
- Result:
  - Final H100 smoke succeeded, no active H100 jobs remain.
  - `pallas_mgpu_compact_chunked`:
    - `compile_time=2.2498145231511444s`
    - `steady_state_time=0.0016518650809302926s`
    - `effective_tflops_per_rank=0.08125223394419819`
    - `max_abs_diff=9.5367431640625e-07`
    - `mean_abs_diff=2.0388574384355707e-08`
    - `metadata_mismatches=0`
  - `pallas_mgpu_compact_static`:
    - `compile_time=0.5348977209068835s`
    - `steady_state_time=0.0014034620253369212s`
    - `effective_tflops_per_rank=0.09563331645384499`
    - `max_abs_diff=3.790855407714844e-05`
    - `mean_abs_diff=2.2422497636398475e-08`
    - `metadata_mismatches=0`
- Interpretation:
  - The compact static workqueue variant lowers and matches the reference on the small EP=8 smoke.
  - This is a correctness/iteration harness result, not a target-shape performance claim.
  - The file now offers two self-contained edit targets: scratch/chunked and static recv_x workqueue.

### 2026-06-29 09:35 PDT - FWD-SWQ-018 compact target speed sanity
- Hypothesis:
  - The compact chunked and compact static kernels should run at plausible target speeds before treating the script as a useful manual iteration harness.
  - Expected chunked target range is near the prior balanced fused `permute_up` best: roughly `0.0093s` for group32/copytile512/copy_rows1.
  - Static compact has no exact prior apples-to-apples target row; it should be in the same broad isolated-permute-up regime and not near the slower production full-forward static row, because this compact benchmark omits down/unpermute and production metadata/composition costs.
- Commit Hash: uncommitted worktree.
- Code change:
  - No additional kernel changes.
- Command:
  - Target speed H100: `/dlwh/iris-run-job-20260629-163351`
  - Local sanity: direct FLOP-rate recomputation from target shape and row times.
- Config:
  - Shape: EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, bf16.
  - Chunked: `--implementations chunked`, `expert_group_size=32`, `copy_tile=512`, `copy_rows=1`, `max_concurrent_steps=4`, `grid_block_n=2`, `warmup=1`, `steps=3`.
  - Static: `--implementations static`, `expert_group_size=8`, `copy_tile=512`, `copy_rows=1`, `max_concurrent_steps=4`, `grid_block_n=2`, `warmup=1`, `steps=3`.
- Result:
  - H100 job succeeded; no active H100 jobs remain.
  - `pallas_mgpu_compact_chunked`:
    - `compile_time=2.3026913709472865s`
    - `steady_state_time=0.00942928995937109s`
    - `effective_tflops_per_rank=182.19684894646994`
  - `pallas_mgpu_compact_static`:
    - `compile_time=2.222618896048516s`
    - `steady_state_time=0.008486996327216426s`
    - `effective_tflops_per_rank=202.42578789514656`
- Interpretation:
  - Chunked is running at expected target speed. It is within ~1.5% of the previous compact/fused target best `0.0092931247s` from `/dlwh/bench_grug_moe_pallas_mgpu-20260629-052800-copyrows1-copytile512-splitcompare`.
  - Static compact is also healthy. It is faster than chunked in this compact balanced-only harness; this should not be compared directly to production full-forward static `0.0250124057s`, because that row included downstream/full-forward composition and production metadata/scratch structure.
  - Both compact edit targets are now validated for small correctness and target-speed sanity.

### 2026-06-29 11:48 PDT - FWD-SWQ-019 static permute_up tax isolation
- Hypothesis:
  - Compact static is faster than production static because production pays some combination of receiver-capacity padding, ragged routing/block alignment, metadata scaffolding outside the Pallas call, and larger JAX/Mosaic compile graph cost.
  - Compare exact balanced compact static against production `permute_up` only, then toggle capacity factor and routing while keeping static schedule config fixed.
- Commit Hash: uncommitted worktree.
- Command:
  - Production static `permute_up`, random uniform routing, `capacity_factor=1.25`: `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-183547`
  - Production static `permute_up`, random uniform routing, `capacity_factor=1.0`: `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-183929`
  - Production static `permute_up`, balanced routing, `capacity_factor=1.0`: `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-184207`
  - Production static `permute_up`, balanced routing, `capacity_factor=1.25`: `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-184500`
  - Earlier combined decomposition `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-183207` was stopped manually after `permute_up` completed but before rows flushed; the harness buffers stage rows until the whole stage list returns, and the later `up_down_candidate` compile/run blocked useful output.
- Config:
  - Shape: EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, bf16.
  - Static schedule: `dispatch_static_workqueue_permute_up=True`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=8`, `dispatch_chunk_copy_tile=512`, `dispatch_static_workqueue_max_rows_per_src_expert=1024`, `block_m=64`, `block_k=64`, `block_n=128`, `grid_block_n=2`, `max_concurrent_steps=4`.
- Result:
  - Compact static target baseline from FWD-SWQ-018: `compile_time=2.222618896048516s`, `steady_state_time=0.008486996327216426s`.
  - Production static `permute_up`, uniform/cap1.25: `compile_time=87.03831239801366s`, `steady_state_time=0.013293436340366801s`, `effective_tflops_per_rank=161.54465956097138`, dropped routes 0.
  - Production static `permute_up`, uniform/cap1.0: `compile_time=87.75526675907895s`, `steady_state_time=0.013045254706715545s`, `effective_tflops_per_rank=131.69439440040986`, dropped routes 660. This row is not a valid exact-capacity correctness/perf comparison because random routing overflows exact capacity.
  - Production static `permute_up`, balanced/cap1.0: `compile_time=87.69829447683878s`, `steady_state_time=0.011277910321950912s`, `effective_tflops_per_rank=152.33202511427788`, dropped routes 0.
  - Production static `permute_up`, balanced/cap1.25: `compile_time=87.4569153659977s`, `steady_state_time=0.011448079720139503s`, `effective_tflops_per_rank=187.58461685256603`, dropped routes 0.
  - Earlier production static full-forward best, uniform/cap1.25 group8/tile512: `steady_state_time=0.02501240566683312s`.
- Tax split:
  - Exact balanced production static over compact static: `0.011277910321950912 - 0.008486996327216426 = 0.0027909139947344865s` (`2.79ms`, `1.33x` compact).
  - Receiver-capacity padding at balanced routing: `0.011448079720139503 - 0.011277910321950912 = 0.000170169398188591s` (`0.17ms`). Padding is not the main runtime tax.
  - Random/ragged routing shape over balanced at cap1.25: `0.013293436340366801 - 0.011448079720139503 = 0.0018453566202272977s` (`1.85ms`). This is likely from ragged row ranges causing extra touched blocks/alignment waste in the static compute loop, not from receiver padding.
  - Full-forward residual after production static `permute_up` under the earlier uniform/cap1.25 full-forward row: `0.02501240566683312 - 0.013293436340366801 = 0.01171896932646632s` (`11.72ms`) for W2/down-unpermute/composition and remaining full-forward structure.
  - Compile-time tax is structural: production static `permute_up` compiles in ~`87s` versus compact static ~`2.22s` (`~39x` slower compile, +`85s`).
- Interpretation:
  - Runtime tax is not primarily metadata exchange volume or capacity padding.
  - The steady-state production-vs-compact tax that remains under exact balanced capacity is about `2.79ms`; the code path shows production static computes `recv_src_rank` and `recv_src_assignment` outside the Pallas kernel with `all_gather` plus scatter updates, carries general clipped-count metadata, and compiles a much larger wrapper graph.
  - Random ragged routing adds another material steady-state penalty (`~1.85ms`) even with no drops, consistent with the static compute loop touching unaligned per-source/expert ranges.
  - Next best isolation would be a production-style compact benchmark that keeps balanced exact routing but adds one feature at a time: production metadata reconstruction, full `MoeMgpuRoutingMetadata`, then ragged counts. That would directly attribute the remaining `2.79ms`.

### 2026-06-29 12:08 PDT - FWD-SWQ-020 static recv metadata all_gather/scatter split
- Hypothesis:
  - The JAX receive-metadata construction in production static `permute_up` might explain a large part of the `2.79ms` production-over-compact exact-balanced runtime tax.
  - Isolate the static receive metadata block into stages:
    - `static_metadata_bases`: local metadata sort/counts, clipped counts, base offsets.
    - `static_metadata_all_gather`: bases plus materialized large per-source all-gathers.
    - `static_metadata_remote_rows`: all-gather plus remote row/keep construction, no scatter set.
    - `static_metadata_full`: all-gather plus scatter-set construction of `recv_src_rank` and `recv_src_assignment`.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added temporary stage benchmarks to `lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`.
  - Local syntax check: `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`.
- Command:
  - First target run with harness bug: `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-190016`.
    - `static_metadata_bases` succeeded (`0.0006977484095841646s`); other stages failed with `NameError: name 'lax' is not defined`.
  - Fixed harness and relaunched target run: `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-190335`.
  - HLO dump for all-gather stage: `/dlwh/iris-run-job-20260629-190621`.
- Config:
  - Shape: EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, bf16.
  - Routing/capacity: `--routing balanced --capacity-factor 1.0`.
  - Static schedule config unchanged from FWD-SWQ-019.
- Result:
  - Successful target metadata rows from `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-190335`, all with dropped routes 0:
    - `static_metadata_bases`: `compile_time=26.553497771034017s`, `steady_state_time=0.0006119998171925545s`.
    - `static_metadata_all_gather`: `compile_time=3.5518180790822953s`, `steady_state_time=0.0004932146053761244s`.
    - `static_metadata_remote_rows`: `compile_time=3.978591732913628s`, `steady_state_time=0.0004855863749980927s`.
    - `static_metadata_full`: `compile_time=3.76837639301084s`, `steady_state_time=0.0004973131697624921s`.
  - HLO dump for `/dlwh/iris-run-job-20260629-190621`:
    - Before optimization, the all-gather stage has six all-gathers: five large `[1, 131072] -> [8, 131072]` int32 metadata arrays plus the tiny send-counts gather.
    - GPU after-optimization buffer assignment shows a tuple `all-gather-start` carrying the five large `[8, 131072]` outputs plus the small `[8, 8, 32]` counts output, with a single `all-gather-done` tuple. This indicates XLA collective-combines them into one async tuple all-gather, not five independent runtime collectives.
- Interpretation:
  - The receive-metadata JAX work is not the missing `2.79ms`.
  - These stage rows are not additive because each stage returns different materialized outputs and XLA rewrites the graph differently. The safe conclusion is an order-of-magnitude bound: the whole isolated recv-metadata construction is around `0.5-0.6ms` steady-state on the exact-balanced target shape.
  - Scatter-set construction is below measurement noise in this isolation: `static_metadata_full - static_metadata_remote_rows ~= 0.012ms`.
  - The large all-gathers are combined by XLA and the all-gather stage is also around `0.5-0.65ms`; there is no evidence of five separate per-array all-gather costs in steady state.
  - The remaining production-over-compact runtime tax should be sought inside the Pallas static kernel/wrapper differences (`recv_x` materialization, hidden zeroing/output shape, phase loop structure, WGMMA schedule) rather than in JAX receive-metadata construction.

### 2026-06-29 13:23 PDT - FWD-SWQ-021 static kernel tax split: bounds, init, copy loop, dense compute
- Hypothesis:
  - After metadata was bounded at ~`0.5-0.6ms`, test whether the remaining production-over-compact static `permute_up` tax comes from output zeroing, oversized static loop bounds, remote-copy loop lowering, or generic row-range/partial-row compute machinery.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added diagnostic config/CLI flags:
    - `dispatch_static_workqueue_init_recv_x`
    - `dispatch_static_workqueue_init_hidden`
    - `dispatch_static_workqueue_direct_store_hidden`
    - `dispatch_static_workqueue_copy_pl_loop`
    - `dispatch_static_workqueue_dense_balanced_compute`
  - Added stage metadata/tax rows in `lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`.
  - Local syntax check passed:
    - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
- Config:
  - Shape: EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, bf16.
  - Routing/capacity: `--routing balanced --capacity-factor 1.0`.
  - Static schedule unless otherwise noted: `dispatch_static_workqueue_permute_up=True`, `dispatch_copy_schedule=expert_group_peer`, `dispatch_expert_group_size=8`, `dispatch_chunk_copy_tile=512`, `block_m=64`, `block_k=64`, `block_n=128`, `grid_block_n=2`, `max_concurrent_steps=4`.
- Results:
  - Output init tax, `/dlwh/iris-run-job-20260629-194348`:
    - baseline balanced cap1 static: `steady_state_time=0.01115282364965727s`, compile `87.56222728604916s`.
    - no `recv_x` init: `0.01103482135416319s`, compile `87.3863595459843s` (`-0.118ms`).
    - no `hidden` init: `0.010801506345160306s`, compile `86.63012784998864s` (`-0.351ms`).
    - no both: `0.010516481668067476s`, compile `87.12146889592987s` (`-0.636ms`).
  - Direct-store generic diagnostic, `/dlwh/iris-run-job-20260629-195300`:
    - no-init generic store: `0.010641762986779213s`, compile `87.74683582002763s`.
    - no-init direct store: `0.010833886025163034s`, compile `87.70729296107311s`.
    - Direct store is slower in the generic path.
  - Static max-row bound test, `/dlwh/iris-run-job-20260629-195911`:
    - `1024/no-init`: `steady_state_time=0.010565503966063261s`, compile `86.2425032609608s`, dropped 0.
    - `512/no-init`: `0.011493846618880829s`, compile `87.16987772914581s`, dropped 0.
    - `512/init`: `0.012007447037224969s`, compile `86.84557094192132s`, dropped 0.
  - Copy-loop form test, `/dlwh/iris-run-job-20260629-200557`:
    - current `mgpu.nd_loop` copy: `0.010401395304749409s`, compile `85.93726169504225s`, dropped 0.
    - compact-style per-SM `pl.loop` copy: `0.010697968653403223s`, compile `87.25083516503219s`, dropped 0.
  - Dense balanced compute with direct store, `/dlwh/iris-run-job-20260629-201000`:
    - dense + direct-store + current copy: `0.009359903323153654s`, compile `87.3749894129578s`, dropped 0.
    - dense + direct-store + `pl.loop` copy: `0.009551265665019551s`, compile `86.75852058501914s`, dropped 0.
  - Dense direct-store correctness compare, `/dlwh/iris-run-job-20260629-201423`:
    - `matches_baseline=false`, `max_abs_diff_vs_baseline=3.794921875`, `mean_abs_diff_vs_baseline=0.0015488507924601436`.
    - `rank_mismatches=0`, `assignment_mismatches=40959`, base/candidate dropped 0.
  - Generic static compare controls, `/dlwh/iris-run-job-20260629-201740`:
    - `1024/no-init`: `matches_baseline=true`, max/mean diff 0, compile `89.97487393999472s`, compare steady-state `0.03119320096448064s`, dropped 0.
    - `512/no-init`: `matches_baseline=true`, max/mean diff 0, compile `90.09632427291945s`, compare steady-state `0.03142285905778408s`, dropped 0.
  - Dense balanced compute with generic SMEM store, `/dlwh/iris-run-job-20260629-201843`:
    - Timing and compare rows both failed during lowering:
      - `ValueError: Operand 2 of operation "memref.subview" must be a Sequence of Values (is not a Value)`.
- Interpretation:
  - Output zeroing explains only about `0.64ms` of the exact-balanced production static stage.
  - The oversized `max_rows_per_src_expert=1024` hypothesis is falsified for this shape. Tightening to the exact `512` rows/source/expert regressed by about `0.93ms` no-init.
  - Replacing the production remote-copy `mgpu.nd_loop` with a compact-style per-SM `pl.loop` also regresses (`+0.30ms`), so copy-loop lowering is not the missing compact-vs-production tax.
  - Generic static compare controls pass exactly for both `1024` and `512`, so the compare machinery and static metadata row contract are correct.
  - Dense balanced compute identifies the likely profitable area: removing generic row-range/partial-row compute machinery and using dense block assumptions can reach `0.00936s`, close to compact static `0.00849s`. However, the current dense+direct-store diagnostic is not correct, and the dense+generic-store variant does not lower.
  - Current judgment: remaining steady-state tax is software-structure-bound inside the static Pallas compute/store path, not comm-bound and not primarily metadata-bound. A usable patch needs a correctness-preserving dense/block-aligned store path or a lowering-safe fix for the generic SMEM store under dense block assumptions.

### 2026-06-29 14:09 PDT - FWD-SWQ-022 static zero-fill defaults and full-forward validation
- Hypothesis:
  - Static workqueue forward should not need to zero `recv_x` or `hidden` scratch buffers in the production full-forward path.
  - `recv_x` live rows are overwritten by the copy phase before W13 reads them.
  - `hidden` padding is not semantically consumed by W2/down; W2/down use live `rows_per_expert`, so raw scratch-buffer comparisons over padding are the wrong correctness test.
- Commit Hash: uncommitted worktree.
- Code change:
  - Changed `MoeMgpuConfig.dispatch_static_workqueue_init_recv_x` default to `False`.
  - Changed `MoeMgpuConfig.dispatch_static_workqueue_init_hidden` default to `False`.
  - Kept `--dispatch-static-workqueue-init-recv-x` and `--dispatch-static-workqueue-init-hidden` as BooleanOptionalAction flags for diagnostics and scratch-buffer compares.
  - Added CLI-default coverage in `lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py::test_pallas_mgpu_benchmark_cli_defaults_match_kernel_config`
    - Result: `1 passed`; existing JAX shutdown warning emitted after test teardown.
- Correctness result:
  - Corrected full-forward compare job: `/dlwh/iris-run-job-20260629-205717`.
  - Shape: EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, bf16, `capacity_factor=1.25`, random uniform routing.
  - `ragged_all_to_all`: `steady_state_time=0.08192223613150418s`, compile `49.58993618306704s`, dropped routes 0.
  - `pallas_mgpu` static workqueue with both zero fills disabled: `steady_state_time=0.024388630175963044s`, compile `81.23922398500144s`, dropped routes 0.
  - Compare: `matches_baseline=true`, `max_abs_diff_vs_baseline=0.01953125`, `mean_abs_diff_vs_baseline=0.001220707199536264`.
  - Earlier malformed compare job `/dlwh/iris-run-job-20260629-205459` failed because `--implementations ragged_all_to_all,pallas_mgpu` was passed as one comma-separated token; rerun above used separate implementation args.
- Timing matrix:
  - Job: `/dlwh/iris-run-job-20260629-205404`.
  - Shape/config: same target forward shape, static workqueue, `dispatch_copy_schedule=expert_group_peer`, expert group 8, copy tile 512, `capacity_factor=1.25`.
  - Job state: succeeded, exit 0, failures 0, preemptions 0.
  - Balanced routing:
    - both init: `steady_state_time=0.023240805021487176s`, compile `84.27704969001934s`, dropped routes 0.
    - no `recv_x` init: `0.02308348403312266s`, compile `82.5860853549093s`, dropped routes 0.
    - no `hidden` init: `0.02250337301908682s`, compile `83.01493256399408s`, dropped routes 0.
    - no both: `0.02232442202512175s`, compile `83.32194020296447s`, dropped routes 0.
  - Uniform routing:
    - both init: `steady_state_time=0.025791856343857944s`, compile `84.392797999084s`, dropped routes 0.
    - no `recv_x` init: `0.02520231033364932s`, compile `82.90985730395187s`, dropped routes 0.
    - no `hidden` init: `0.02453143533784896s`, compile `84.57700058305636s`, dropped routes 0.
    - no both: `0.02419436036143452s`, compile `83.57761984004173s`, dropped routes 0.
- Tax split:
  - Balanced no-both speedup over initialized baseline: `0.0009163829963654261s` (`0.916ms`, `3.94%`).
  - Uniform no-both speedup over initialized baseline: `0.001597495982423424s` (`1.597ms`, `6.19%`).
  - `hidden` zeroing is the larger single tax in both routes:
    - Balanced no `hidden`: `0.737ms` faster than baseline.
    - Uniform no `hidden`: `1.260ms` faster than baseline.
  - `recv_x` zeroing is smaller but still measurable:
    - Balanced no `recv_x`: `0.157ms` faster than baseline.
    - Uniform no `recv_x`: `0.590ms` faster than baseline.
- Interpretation:
  - Skipping zero fills is a real production full-forward speedup and is safe under the full-forward semantic compare.
  - This tax is software-structure-bound scratch initialization, not communication-bound.
  - The patch should be treated as a forward-path default change; scratch-buffer-only stage compares may still need explicit `--dispatch-static-workqueue-init-hidden` if they compare unused padding bytes.
  - Remaining forward work is still inside the static Pallas compute/store structure: dense/block-aligned compute was faster in FWD-SWQ-021 but not yet correctness-preserving.

### 2026-06-29 16:31 PDT - FWD-SWQ-023 destination-pull permute_up adaptation attempt
- Hypothesis:
  - Adapt the attached source-push fused all2all+GMM destination-pull idea to `permute_up`: destination rank `dst` should invert the existing source-push indexing, remote-read source rank `src` token rows from `x_local`, and feed those rows directly into local W13/SwiGLU without materializing full `recv_x`.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `MoeMgpuConfig.dispatch_static_workqueue_destination_pull` as an explicit experimental flag under `dispatch_static_workqueue_permute_up`.
  - Added benchmark CLI wiring and row metadata for `--dispatch-static-workqueue-destination-pull`.
  - Added CLI default coverage in `lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`.
  - Implemented an edit-point prototype branch in `_permute_up_mgpu_static_workqueue_kernel` that shares the current static schedule and inverts source-push indexing:
    - source-push reads `token_ids_ref[source_expert_offsets[dst * E_local + expert] + local_pos]` on rank `src` and writes to destination row `_remote_row(dst, src, expert, local_pos)`;
    - destination-pull computes `src = (rank + ep_size - peer_phase) % ep_size`, remote-refs source `x_ref`/`token_ids_ref`/`source_expert_offsets_ref`, and attempts to load those same source-local token rows into W13 lhs SMEM.
  - After H100 lowering evidence, left the flag as a clear `NotImplementedError` edit point rather than an opaque compiler failure. The prototype code remains adjacent for continued manual editing.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py::test_pallas_mgpu_benchmark_cli_defaults_match_kernel_config`
    - Result: `1 passed`; existing Haliax/JAX deprecation warnings only.
- H100 commands/config:
  - All jobs used `cw-us-east-02a`, H100x8, balanced exact target shape: EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, bf16, `capacity_factor=1.0`, `--dispatch-static-workqueue-permute-up`, `--dispatch-static-workqueue-dense-balanced-compute`, `--dispatch-static-workqueue-destination-pull`, `--dispatch-static-workqueue-max-rows-per-src-expert 512`, warmup 1, steps 1.
- Results:
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-231515`, vector remote gather into lhs SMEM:
    - `ragged_all_to_all`: `steady_state_time=0.07803318509832025s`, compile `49.63361969101243s`, dropped routes 0.
    - `pallas_mgpu`: `status=error`, `NotImplementedError: int_indexer_shape non-empty`.
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-231850`, scalar remote row loads into lhs SMEM:
    - `ragged_all_to_all`: `steady_state_time=0.07813188014551997s`, compile `49.666998255066574s`, dropped routes 0.
    - `pallas_mgpu`: `status=error`, `ValueError: Failed to infer a possible set of layouts. This should only happen if user-provided layout casts are unsatisfiable.`
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-232147`, `mgpu.copy_gmem_to_smem` from peer `remote_ref`:
    - `ragged_all_to_all`: `steady_state_time=0.07801189017482102s`, compile `49.939604789949954s`, dropped routes 0.
    - `pallas_mgpu`: `status=error`, `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.`
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260629-232600`, hybrid scalar peer loads for lhs plus local RHS `copy_gmem_to_smem`:
    - `ragged_all_to_all`: `steady_state_time=0.07833280018530786s`, compile `49.71072102501057s`, dropped routes 0.
    - `pallas_mgpu`: `status=error`, `ValueError: Failed to infer a possible set of layouts. This should only happen if user-provided layout casts are unsatisfiable.`
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-001731`, explicit `mgpu.load(remote_ref, idx, optimized=False)` for remote token ids/source rows:
    - `ragged_all_to_all`: `steady_state_time=0.07849886396434158s`, compile `49.61705406906549s`, dropped routes 0.
    - `pallas_mgpu`: `status=error`, `ValueError: Failed to infer a possible set of layouts. This should only happen if user-provided layout casts are unsatisfiable.`
- Interpretation:
  - The source/destination indexing adaptation is straightforward and is now represented in code, but the direct destination-pull W13 path is blocked by current Mosaic GPU warpgroup lowering.
  - Supported remote-ref patterns in this codebase are remote GMEM writes and some scalar/register remote reads. Feeding peer GMEM directly into WGMMA lhs SMEM is not currently supported in the forms tested, including the explicit `mgpu.load` route.
  - A viable destination-pull version likely needs a different staging mechanism: e.g. destination-pull into local GMEM scratch with non-warpgroup scalar/vector loads, then local GMEM-to-SMEM WGMMA; a multimem-based source-push/store variant; or an upstream Pallas/Mosaic lowering change for peer GMEM refs as WGMMA copy sources.
  - Until then, source-push static workqueue remains the practical fused `permute_up` path.

### 2026-06-29 21:24 PDT - FWD-SWQ-024 destination-pull layout-unsat source
- Hypothesis:
  - The explicit `mgpu.load(remote_ref, idx, optimized=False)` destination-pull failure might be caused by peer-id GMEM refs in warpgroup lowering, or by a more specific Mosaic layout constraint conflict after the remote row load is lowered to vector load/store.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `--debug-mosaic-layout-inference` to `lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`.
  - The flag monkeypatches Mosaic GPU layout inference to print bounded `MARIN MOSAIC LAYOUT DEBUG` blocks when constraint reduction or top-level assignment search returns `Unsatisfiable`.
  - Fixed the temporary `MARIN_MGPU_DEST_PULL_DEBUG_ALLOW` guard in `lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py`; the previous guard accidentally wrapped the destination-pull prototype body, so a debug run fell through to the source-push static kernel instead of testing the remote-load path.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
- H100 commands/config:
  - Prior miswired debug run:
    - Job: `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-041620`.
    - Shape: target EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, `capacity_factor=1.0`.
    - Result: Iris succeeded, task exit 0, duration `79.897s`.
    - `ragged_all_to_all`: `steady_state_time=0.07855752995237708s`, compile `50.20793724898249s`, dropped routes 0.
    - `pallas_mgpu`: `status=error`, `ValueError: Operand 2 of operation "memref.subview" must be a Sequence of Values (is not a Value)`.
    - No `MARIN MOSAIC LAYOUT DEBUG` block printed because this run did not enter the destination-pull body.
  - Corrected small compile probe:
    - Job: `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-042144`.
    - Shape: EP=8/T=1024/D=256/I=128/E_local=8/K=4, `capacity_factor=1.0`, balanced routing, `--dispatch-static-workqueue-max-rows-per-src-expert 64`, warmup 0, steps 1.
    - Environment: `MARIN_MGPU_DEST_PULL_DEBUG_ALLOW=1 MARIN_MGPU_LAYOUT_DEBUG_MAX_PRINTS=6`.
    - Result: Iris succeeded, task exit 0, duration `26.555s`.
    - `pallas_mgpu`: `status=error`, no timing row, `ValueError: Failed to infer a possible set of layouts. This should only happen if user-provided layout casts are unsatisfiable.`
- Key debug constraints:
  - First unsat block shows the remote row load lowered as a vector load/store into SMEM:
    - `type(V(%331 = "mosaic_gpu.vector_load"(%330):r-0)) != WGSplatFragLayout`
    - `IsTransferableSmemRegisters(V("mosaic_gpu.vector_store"(%331,:o-0) -> V(%238 = "mosaic_gpu.slice_smem"():r-0))`
    - `IsValidMMATiling(V(%237 = "mosaic_gpu.slice_smem"():r-0), 16, allow_unswizzled=False)`
    - `Equals(V(%238 = "mosaic_gpu.slice_smem"():r-0) == V(%237 = "mosaic_gpu.slice_smem"():r-0))`
    - `(1, 64) % V(%238 = "mosaic_gpu.slice_smem"():r-0) == 0`
  - The same conflict repeats for candidate SMEM transforms `(8, 64)`, `(8, 32)`, and `(8, 16)`, then `find_assignments_for returned Unsatisfiable`.
- Interpretation:
  - The current explicit `mgpu.load(remote_ref, idx)` path is not failing at the peer-id GMEM-ref warpgroup check. It gets far enough to become a vector load followed by a vector store into lhs SMEM.
  - The unsat comes from storing one row at a time into the same SMEM allocation that WGMMA uses as lhs:
    - vector store into a `1 x 64` SMEM subview adds a divisibility/transfer constraint for shape `(1, 64)`;
    - WGMMA requires lhs SMEM to use a valid MMA tile transform `(8, swizzle_elems)`;
    - the lhs slice is equated with the WGMMA lhs tile, so Mosaic cannot satisfy both constraints.
  - This explains why scalar/row remote loads into WGMMA lhs SMEM fail even with `optimized=False`. Avoiding remote GMEM -> local GMEM scratch -> local SMEM would still need a way to write the pulled data into WGMMA-compatible tiled SMEM, not row slices.
  - Plausible next experiments:
    - build an 8-row/tile-shaped SMEM store path if Pallas can express a compatible `8 x 64` register value from remote rows;
    - separate the load staging SMEM from the WGMMA lhs SMEM and copy/transform locally, if Mosaic has a supported SMEM-to-SMEM tiled move;
    - otherwise fall back to local GMEM scratch or an upstream Mosaic change that permits non-optimized register-to-tiled-SMEM row stores feeding WGMMA.

### 2026-06-29 21:38 PDT - FWD-SWQ-025 WGMMA-shaped remote-load probes
- Hypothesis:
  - The row-store unsat in FWD-SWQ-024 should be avoidable by loading/storing a WGMMA-compatible tile shape. The critical question is whether Pallas can express the routed-token rows in a register layout that is transferable to WGMMA-tiled SMEM.
- Commit Hash: uncommitted worktree.
- Code change:
  - Modified the destination-pull edit-point body in `lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py` for lowering probes.
  - Current edit-point state is intentionally marked as a lowering probe only: it remote-loads a contiguous full `64 x block_k` tile using `layout=mgpu.Layout.WGMMA`, which is not the correct routed-token gather semantics.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py`
- H100 commands/config:
  - All probes used `cw-us-east-02a`, H100x8, shape EP=8/T=1024/D=256/I=128/E_local=8/K=4, balanced routing, `capacity_factor=1.0`, `--dispatch-static-workqueue-permute-up`, `--dispatch-static-workqueue-dense-balanced-compute`, `--dispatch-static-workqueue-destination-pull`, `--dispatch-static-workqueue-max-rows-per-src-expert 64`, warmup 0, steps 1, and `MARIN_MGPU_DEST_PULL_DEBUG_ALLOW=1`.
- Results:
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-043046`, eight scalar row loads assembled with `jnp.stack` and stored as `8 x 64`:
    - Iris succeeded, task exit 0, duration `26.576s`.
    - `pallas_mgpu`: `status=error`, `NotImplementedError: Unimplemented primitive in Pallas Mosaic GPU lowering: concatenate for lowering semantics LoweringSemantics.Warpgroup and user thread semantics PrimitiveSemantics.Warpgroup.`
    - Interpretation: the `8 x 64` store gets past the original row-slice unsat, but `jnp.stack` lowers to unsupported warpgroup `concatenate`.
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-043249`, no-concat assembly via broadcast/select/add into `8 x 64`:
    - Iris succeeded, task exit 0, duration `63.895s`.
    - `pallas_mgpu`: `status=error`, `ValueError: Layout inference failed to find a solution. Consider adding layout annotations to your program to guide the search.`
    - Debug signal: the final store is `8 x 64`, but the assembly path introduces many `broadcast_in_dim`/`select`/`addf` relayout constraints, including relayouts from splat-shaped masks into non-splat values.
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-043505`, contiguous `8 x 64` remote vector load into `8 x 64` SMEM:
    - Iris succeeded, task exit 0, duration `26.800s`.
    - `pallas_mgpu`: `status=error`, `ValueError: Failed to infer a possible set of layouts. This should only happen if user-provided layout casts are unsatisfiable.`
    - Debug signal: remote vector load chooses `WGStridedFragLayout(shape=(8, 64), vec_size=4)`. `IsTransferableSmemRegisters(WGStridedFragLayout -> TileTransform(8, 64))` is unsatisfiable because current Mosaic constraints only allow non-MMA register layouts to transfer to untiled SMEM.
  - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-043729`, contiguous full `64 x 64` remote vector load with `layout=mgpu.Layout.WGMMA` into full lhs SMEM:
    - Iris succeeded, task exit 0, duration `37.233s`.
    - `pallas_mgpu`: `status=ok`, `compile_time=10.564915537834167s`, `steady_state_time=0.003740105079486966s`, dropped routes 0.
    - This is a lowering capability probe only; it is not semantically correct for MoE because it reads contiguous source rows instead of `token_ids_sorted` rows.
- Interpretation:
  - The user's shape hypothesis is correct in the constrained sense: loading/storing a WGMMA-shaped tile fixes the specific `(1, 64)` divisibility conflict.
  - However, an `8 x 64` value is still not enough unless its register layout is MMA-compatible. Default remote vector loads use `WGStridedFragLayout`, and the current constraint system refuses `WGStridedFragLayout -> WGMMA-tiled SMEM`.
  - A full `64 x 64` remote load can be forced into `Layout.WGMMA` and lowered successfully. That proves direct remote GMEM -> WGMMA-register -> WGMMA-SMEM is possible for contiguous tiles.
  - Remaining hard part: produce a correct routed-token `64 x 64` WGMMA-layout register tile. Direct arbitrary-index gather likely hits `int_indexer_shape non-empty`; assembling rows in warpgroup code hits unsupported concat or layout-inference relayout conflicts.
  - Practical next direction is to make the source side pack per-destination/per-expert rows into contiguous WGMMA-shaped chunks, then destination-pull those chunks with `layout=mgpu.Layout.WGMMA`. That is a structural split but avoids remote GMEM -> local GMEM -> local SMEM for the final W13 load.

### 2026-06-29 22:15 PDT - FWD-SWQ-026 source-packed destination-pull with LHS reuse
- Hypothesis:
  - Arbitrary peer gather is the wrong abstraction. If the source rank packs routed rows in the existing `token_ids_sorted` / `source_expert_offsets` order, destination ranks can pull contiguous `block_m x block_k` WGMMA LHS tiles directly from peer GMEM.
  - A naive destination-pull W13 reloads the same remote LHS tile once per output-N tile, so target performance should improve if each remote LHS tile feeds multiple W13 N tiles before being reloaded.
- Commit Hash: uncommitted worktree.
- Code change:
  - Reworked the experimental `dispatch_static_workqueue_destination_pull` branch in `lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py`:
    - source side now builds `sorted_x = x_local[metadata.token_ids_sorted]`;
    - destination kernel remote-loads contiguous `sorted_x` WGMMA tiles instead of arbitrary token gathers;
    - exact-balanced prototype uses arithmetic source offsets `(dst * E_local + expert) * max_rows_per_src_expert`, avoiding an in-warpgroup remote metadata load;
    - target path groups two output-N tiles per program, reusing one remote LHS SMEM tile for two W13 gate/up tile pairs.
  - Added `static_source_pack` stage timing to `lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
- H100 commands/config:
  - All jobs used `cw-us-east-02a`, H100x8, balanced routing, `capacity_factor=1.0`, `MARIN_MGPU_DEST_PULL_DEBUG_ALLOW=1`, `--dispatch-static-workqueue-permute-up`, `--dispatch-static-workqueue-dense-balanced-compute`, `--dispatch-static-workqueue-destination-pull`.
- Results:
  - Source-packed small probe before N grouping:
    - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-045153`, EP=8/T=1024/D=256/I=128/E_local=8/K=4, max rows 64.
    - `ragged_all_to_all`: `steady_state_time=0.0010968249989673495s`, compile `6.463732519070618s`, dropped 0.
    - `pallas_mgpu`: `steady_state_time=0.0032660829601809382s`, compile `10.123171860002913s`, `matches_baseline=true`, `max_abs_diff=0.00048828125`, mean `3.641754301497713e-05`, dropped 0.
  - Source-packed target before N grouping:
    - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-045326`, EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, max rows 512.
    - `ragged_all_to_all`: `steady_state_time=0.07772745467567195s`, compile `50.13745272997767s`, dropped 0.
    - `pallas_mgpu`: `steady_state_time=0.0450604953415071s`, compile `82.38088049099315s`, `matches_baseline=true`, `max_abs_diff=0.015625`, mean `0.0012174173025414348`, dropped 0.
  - Group-size check:
    - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-045805`, same target, timing-only, `--dispatch-expert-group-size 32`.
    - `pallas_mgpu`: `steady_state_time=0.04486727702897042s`, compile `83.30769508401863s`, dropped 0.
    - Interpretation: phase-count/group-size is not the main tax.
  - Stage tax:
    - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-050121`, same target, stages only.
    - `static_metadata_bases=0.0006425683386623859s`, `static_metadata_all_gather=0.0005286666952694455s`, `static_metadata_full=0.0005025730157891909s`, candidate `permute_up=0.034457909991033375s`.
    - `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-050508`, same target, `static_source_pack=0.0008139143930748105s`.
    - Interpretation: source sort and metadata are not the target tax; the remote-LHS W13 body is.
  - Two-N-tile LHS reuse:
    - Reduced compare `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-050803`, EP=8/T=1024/D=256/I=256/E_local=8/K=4, max rows 64:
      - `pallas_mgpu`: `steady_state_time=0.003334232955239713s`, compile `10.472533004009165s`, `matches_baseline=true`, `max_abs_diff=0.00048828125`, mean `5.1721555792028084e-05`, dropped 0.
    - Target timing `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-050928`:
      - `pallas_mgpu`: `steady_state_time=0.034350360316845276s`, compile `84.010608929093s`, `75.02047587944105 TFLOP/s/rank`, dropped 0.
    - Target compare `/dlwh/iris-run-bench_grug_moe_pallas_mgpu-20260630-051151`:
      - `ragged_all_to_all`: `steady_state_time=0.07784553000237793s`, compile `50.039016231079586s`, dropped 0.
      - `pallas_mgpu`: `steady_state_time=0.03487554902676493s`, compile `82.28238473797683s`, `matches_baseline=true`, `max_abs_diff=0.015625`, mean `0.0012174173025414348`, dropped 0.
- Interpretation:
  - Source-side sorting is the right fix for the arbitrary-gather lowering problem.
  - Direct remote GMEM -> WGMMA SMEM is viable for correct routed rows when the rows are source-packed.
  - The naive destination-pull kernel was slow because it reloaded the same remote LHS tile for every W13 N tile. Reusing LHS for two N tiles improves target forward from `0.0450604953s` to `0.0343503603s`.
  - Compared with the prior public exact target forward row `0.039230s`, the validated timing row is about 12.4% faster; compared with repeated target `0.038529s`, it is about 10.8% faster.
  - Caveats:
    - Current destination-pull branch is exact-balanced only; it assumes `max_rows_per_src_expert` rows per source/destination/expert for the packed source offset.
    - It remains behind the explicit debug environment guard and experimental flag.
    - Remaining bottleneck is remote-LHS W13 efficiency; metadata construction and source pack are sub-millisecond at target.

### 2026-06-29 23:16 PDT - FWD-SWQ-027 destination-pull tile/group tuning
- Hypothesis:
  - The source-packed destination-pull W13 body is now dominated by remote LHS load reuse and WGMMA tile scheduling.
  - Tune `block_k`, output-N grouping, grid-N interleave, and expert-group phase size before changing the structure again.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `MoeMgpuConfig.dispatch_static_workqueue_destination_pull_n_group` and benchmark CLI/result plumbing for `--dispatch-static-workqueue-destination-pull-n-group`.
  - Added config validation for destination-pull `block_m`: it must be a multiple of 64, matching the Mosaic WGMMA accumulator constraint observed during tuning.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Result after the validation edit: `50 passed, 11 warnings`.
- H100 commands/config:
  - Sweeps used `cw-us-east-02a`, H100x8, target shape EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, balanced routing, `capacity_factor=1.0`, `--implementations none --include-pallas-stages --pallas-stages permute_up`, warmup 1, steps 3.
  - All destination-pull runs used `MARIN_MGPU_DEST_PULL_DEBUG_ALLOW=1`, `--dispatch-static-workqueue-permute-up`, `--dispatch-static-workqueue-dense-balanced-compute`, `--dispatch-static-workqueue-destination-pull`, and `--dispatch-static-workqueue-max-rows-per-src-expert 512`.
- Results:
  - Broad sweep `/dlwh/iris-run-job-20260630-053753` succeeded, duration `19m48.73s`.
    - Base `block_m=64, block_n=128, block_k=64, grid_block_n=2, expert_group=8, n_group=2`: `permute_up=0.0237152643s`, `72.44 TFLOP/s/rank`.
    - Same tile with `n_group=1`: `0.0344786133s`, `49.83 TFLOP/s/rank`.
    - `block_n=64,n_group=2`: `0.0346223770s`.
    - `block_n=256,n_group=1`: `0.0330533833s`.
    - `grid_block_n=1`: `0.0232801950s`; `grid_block_n=4`: `0.0237489987s`.
    - `expert_group=16`: `0.0239003387s`; `expert_group=32`: `0.0225034043s`.
    - `block_m=32`: rejected by MLIR WGMMA verification; accumulator first dimension must be a multiple of 64.
    - `block_m=128`: `0.0415832067s`.
    - `block_k=32`: `0.0350873123s`; `block_k=128`: `0.0204824653s`, `83.88 TFLOP/s/rank`.
  - Focused `block_k=128` sweep `/dlwh/iris-run-job-20260630-055914` succeeded, duration `14m16.09s`.
    - `expert_group=8, grid_block_n=2`: `0.0205722700s`.
    - `expert_group=8, grid_block_n=1`: `0.0204390740s`.
    - `expert_group=8, grid_block_n=4`: `0.0206318993s`.
    - `expert_group=16, grid_block_n=2`: `0.0207783097s`.
    - `expert_group=16, grid_block_n=1`: `0.0205459164s`.
    - `expert_group=32, grid_block_n=2`: `0.0205846293s`.
    - `expert_group=32, grid_block_n=1`: `0.0204400020s`.
    - Best: `expert_group=32, grid_block_n=4`: `0.0202995833s`, `84.63 TFLOP/s/rank`.
- Interpretation:
  - Reusing one remote LHS tile for two output-N tiles is essential: `n_group=2` is about 31% faster than `n_group=1` at the baseline tile.
  - `block_k=128` is the largest clean gain, likely because it halves the K-loop and barrier/copy overhead without exceeding SMEM limits for the 2-N-tile path.
  - `block_n=64` and `block_n=256` both lose; the best shape remains `block_n=128` with explicit two-N grouping.
  - Expert-group and grid-N interleave are second-order once `block_k=128` is used. `expert_group=32, grid_block_n=4` is the best stage row, but `expert_group=8, grid_block_n=1` is within noise.
  - Full target forward compare `/dlwh/iris-run-job-20260630-061556` succeeded, duration `2m39.72s`, `capacity_factor=1.25`.
    - `ragged_all_to_all`: `steady_state_time=0.0815642667s`, compile `49.8027186721s`, dropped 0.
    - Tuned `pallas_mgpu`: `steady_state_time=0.0317110477s`, compile `81.6392614100s`, `101.58 TFLOP/s/rank`, dropped 0.
    - Correctness: `matches_baseline=true`, `max_abs_diff=0.015625`, mean `0.0012174173025414348`.
    - Compared with the previous destination-pull full target row `0.0343503603s`, this is about 7.7% faster. Compared with the earlier repeated public target row `0.038529s`, it is about 17.7% faster.

### 2026-06-30 00:27 PDT - FWD-SWQ-028 generalized source-packed destination-pull
- Hypothesis:
  - Keep the dense exact-balanced destination-pull W13 path as the fast path, but remove the exact-balanced source-offset assumption so the same structure can run on realistic uniform/ragged route counts.
  - The general path should use source-local packed offsets from metadata all-gather, and only special-case stores for partial tail blocks.
- Commit Hash: uncommitted worktree.
- Code change:
  - Destination-pull now all-gathers `source_expert_offsets` and computes remote source rows as `source_expert_offsets_by_src[src_rank, dst * local_experts + expert] + local_pos`.
  - Added a ragged partial-tail store path for destination-pull while preserving the dense full-block direct-store path.
  - `permute_up_compare` now masks hidden diffs to valid expert rows because destination-pull intentionally does not zero unused capacity rows.
  - `_baseline_permute_up_config` now clears destination-pull flags when building the staged baseline config.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Result: `51 passed, 11 warnings`.
- H100 results:
  - Small uniform ragged compare `/dlwh/iris-run-job-20260630-071127`, EP=8/T=512/D=256/I=256/E_local=8/K=4:
    - `matches_baseline=true`, `max_abs_diff=0.0`, mean `0.0`, dropped 0.
    - `steady_state_time=0.0030966374s`, compile `10.3639914480s`.
  - Target uniform `block_k=128,n_group=2` attempts failed compilation:
    - `/dlwh/iris-run-job-20260630-071446`
    - `/dlwh/iris-run-job-20260630-072023`
    - Error: `Mosaic GPU kernel exceeds available shared memory: smem_bytes=327712 > max_smem_bytes=232448`.
  - Target uniform ragged compare `/dlwh/iris-run-job-20260630-072411`, EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, `capacity_factor=1.25`, `max_rows_per_src_expert=640`, `block_k=64,n_group=2`:
    - `matches_baseline=true`, `max_abs_diff=0.0`, mean `0.0`, dropped 0.
    - `steady_state_time=0.0523233680s`, `82.09 TFLOP/s/rank`, compile `93.0546630330s`.
- Interpretation:
  - General source-packed destination-pull is correct at target geometry.
  - The dense exact-balanced path remains the only fast path at target: `0.0317110477s`.
  - The ragged partial-tail path is software-structure/SMEM-bound today. It cannot combine the current partial-tail machinery with the tuned `block_k=128,n_group=2` tile under Hopper SMEM limits, and falling back to `block_k=64` costs about 65% versus the dense exact-balanced row.

### 2026-06-30 00:50 PDT - FWD-SWQ-029 dense-profile and follow-up tile/buffering probes
- Hypothesis:
  - The tuned exact-balanced destination-pull path should be exposed as a repeatable profile, then tested against larger K tiles, larger LHS reuse, and a small RHS-buffering experiment.
- Commit Hash: uncommitted worktree.
- Code change:
  - Added `MoeMgpuConfig.destination_pull_dense_hopper(...)`.
  - Added benchmark CLI `--pallas-mgpu-tuning-profile destination_pull_dense_balanced_hopper`; it applies the tuned Hopper dense-balanced destination-pull knobs and infers `max_rows_per_src_expert` from balanced routing, while preserving explicit CLI overrides.
  - Added an isolated `destination_pull_n_group=5` W13 branch for compile/perf probing.
  - Added an experimental destination-pull split-WG RHS-buffering branch for `n_group=1`.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py`
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - Result: `54 passed, 11 warnings`.
- H100 results:
  - Dense profile/tile sweep `/dlwh/iris-run-job-20260630-073159`, target balanced shape, H100x8, stage `permute_up`, `capacity_factor=1.0`:
    - Profile `block_k=128,block_n=128,n_group=2`: `steady_state_time=0.0202888617s`, `84.68 TFLOP/s/rank`, compile `87.4616771320s`.
    - `block_k=256,n_group=1`: `0.0331435887s`, `51.83 TFLOP/s/rank`, compile `87.9439658599s`.
    - `block_k=256,n_group=2`: compile failure, `Mosaic GPU kernel exceeds available shared memory: smem_bytes=294920 > max_smem_bytes=232448`.
    - `block_k=64,block_n=256,n_group=2`: `0.0302525080s`, `56.79 TFLOP/s/rank`, compile `87.4815058070s`.
    - `block_k=128,block_n=256,n_group=1`: `0.0217952987s`, `78.82 TFLOP/s/rank`, compile `87.0830144180s`.
  - N-group 5 probe `/dlwh/iris-run-job-20260630-074028`:
    - `block_k=64,n_group=5`: `0.0479425933s`, `35.83 TFLOP/s/rank`, compile `89.6338702260s`.
    - `block_k=128,n_group=5`: compile failure, `smem_bytes=344072 > max_smem_bytes=232448`.
  - RHS-buffering probe `/dlwh/iris-run-job-20260630-074647`:
    - Synchronous `block_k=128,n_group=1`: `0.0330940127s`, `51.91 TFLOP/s/rank`, compile `86.3973019890s`.
    - Split-WG RHS-buffered `block_k=128,n_group=1`: compile/lowering failure, `NotImplementedError: run_scoped discharge does not support collective_axes yet.`
    - Split-WG RHS-buffered `block_k=64,n_group=1`: same `run_scoped discharge does not support collective_axes yet`.
- Interpretation:
  - The named profile reproduces the best tuned stage row and is the right trigger for dense exact-balanced Hopper experiments.
  - `block_k=256` is not promising: without LHS reuse it is slow, and with two-N reuse it exceeds SMEM.
  - Wider `block_n=256` does not beat `block_n=128,n_group=2`; the best wider-N row is still about 7.4% slower than the profile.
  - Larger `n_group=5` is a dead end in the current unrolled implementation: it either spills/pressures badly (`block_k=64`) or exceeds SMEM (`block_k=128`).
  - A small split-WG RHS-buffering experiment is blocked by Mosaic lowering for nested `run_scoped(..., collective_axes="wg")` in this destination-pull branch. Testing RHS overlap further likely requires restructuring the kernel body around a top-level split-WG scope rather than a nested per-tile scope.

### 2026-06-30 15:51 PDT - FWD-SWQ-030 compact pull script replacement
- Hypothesis:
  - The self-contained compact benchmark should use the new destination-pull/source-packed W13 kernel as its primary edit target, keeping the materialized static path only as a comparison.
- Commit Hash: uncommitted worktree.
- Code change:
  - Replaced `compact_permute_up_mgpu` in `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py` with a dense balanced destination-pull kernel:
    - source-packed local `x` order by destination global expert and local position;
    - destination rank remote-loads source rows directly into W13 SMEM via `mgpu.load(remote_sorted_x_ref, ...)`;
    - supports `n_group=1` and `n_group=2`, defaulting to the tuned `block_k=128, block_n=128, grid_block_n=4, expert_group_size=32, n_group=2`.
  - Renamed the compact implementation selector from `chunked` to `pull`; default CLI is now `--implementations pull,static`.
  - Made config validation implementation-aware so pull does not require static-copy-only knobs such as `copy_tile` to divide `D`.
- Local checks:
  - `uv run --package marin-levanter python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
- H100 results:
  - First smoke `/dlwh/iris-run-job-20260630-170620` failed before compile because validation still required `D=256` divisible by `copy_tile=512`; fixed by implementation-aware validation.
  - Correctness smoke `/dlwh/iris-run-job-20260630-224536`, EP=8/T=1024/D=256/I=256/E_local=8/K=4:
    - `implementation=pallas_mgpu_compact_pull`, `steady_state_time=0.0023575800s`, compile `2.9034495370s`.
    - `max_abs_diff=3.814697265625e-06`, `mean_abs_diff=4.25914237212055e-08`, `metadata_mismatches=0`.
  - Target timing `/dlwh/iris-run-job-20260630-224753`, EP=8/T=32768/D=2560/I=1280/E_local=32/K=4:
    - `implementation=pallas_mgpu_compact_pull`, `block_k=128, block_n=128, expert_group_size=32, grid_block_n=4, n_group=2`.
    - `steady_state_time=0.0188209613s`, `91.28 TFLOP/s/rank`, compile `3.1311350849s`.
- Interpretation:
  - The compact script now exercises the pull-based fused all2all(lhs)+W13 kernel directly.
  - The target compact row is faster than the production-stage `permute_up` row because the compact harness has a synthetic exact-balanced metadata contract and omits the production metadata/source-offset construction around the kernel.

### 2026-06-30 16:05 PDT - FWD-SWQ-031 compact pull roughly-balanced routing
- Hypothesis:
  - Exact-balanced compact routing is not the useful target; the compact pull benchmark should take real per-token `selected_experts` and generate a roughly-balanced route distribution by default.
- Commit Hash: uncommitted worktree.
- Code change:
  - Updated `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`.
  - `compact_permute_up_mgpu` now takes `selected_experts_local: Int[Array, "T K"]`.
  - Added source-side packing by assignment: each source rank sorts assignments by global expert and packs `x` into fixed destination/expert buckets.
  - Added dense destination metadata construction from all-gathered sorted assignment metadata.
  - Added `--routing uniform|balanced`, defaulting to `uniform`; static workqueue remains exact-balanced only.
  - Added `--capacity-factor` and `--max-rows-per-src-expert`; the pull path defaults to block-rounded capacity from `capacity_factor=1.25`.
- Local checks:
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
- H100 results:
  - Uniform correctness smoke `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260630-230301`, EP=8/T=1024/D=256/I=256/E_local=8/K=4, `capacity_factor=2.0`:
    - `steady_state_time=0.0018150274s`, compile `3.5180689860s`.
    - `max_abs_diff=3.814697265625e-06`, `mean_abs_diff=2.1294535912375068e-08`, `metadata_mismatches=0`.
  - Target uniform timing `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260630-230433`, EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, `capacity_factor=1.25`:
    - `steady_state_time=0.0257171070s`, `66.80 TFLOP/s/rank`, compile `3.3428851070s`.
  - `block_m=32,max_rows_per_src_expert=608` probe `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260630-230609`:
    - Compile failure: `mosaic_gpu.wgmma` requires accumulator first dimension multiple of 64, but got 32.
  - `block_m=128,max_rows_per_src_expert=640` probe `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260630-230704`:
    - `steady_state_time=0.0391202977s`, `43.92 TFLOP/s/rank`, compile `5.9350396390s`.
- Interpretation:
  - The compact pull benchmark now exercises roughly-balanced, assignment-driven routing instead of exact-balanced synthetic row arithmetic.
  - The current implementation computes all padded bucket blocks. At target this means capacity `640` rows per source/expert versus mean `512`, so the 0.0257s row includes a deliberate padding-compute tax.
  - Compared with the exact-balanced compact pull row `0.0188209613s`, the rough-routing padded path is about 36.6% slower. The next useful isolation is skipping padded blocks without paying a destination-output zeroing tax.
  - For seed 0 target uniform routing, local count inspection found per-source/global-expert counts from `437` to `584`; `max_rows_per_src_expert=576` would drop routes, so `640` is the smallest no-drop multiple of 64 for `block_m=64`.
  - `block_m=128` is not useful despite fewer CTAs; it is about 52% slower than `block_m=64`.

### 2026-06-30 16:15 PDT - FWD-SWQ-032 skip padded rough-routing blocks and tune expert groups
- Hypothesis:
  - The rough-routing compact pull path should not compute all padded capacity blocks. It should skip blocks whose start row is beyond `ceil(count/block_m)*block_m`.
  - Expert groups, not individual experts, should remain the useful pull/scheduling unit; tune group size directly.
- Commit Hash: uncommitted worktree.
- Code change:
  - Updated `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`.
  - Added `CompactConfig.skip_padded_blocks` and CLI `--skip-padded-blocks`.
  - The pull kernel now gathers per-source send counts and guards W13 compute for padded blocks.
  - The correctness harness now compares hidden values only for real routed rows, while still checking dense metadata exactly.
- Local checks:
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
- H100 results:
  - Skip-block correctness smoke `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260630-231223`, EP=8/T=1024/D=256/I=256/E_local=8/K=4, `capacity_factor=2.0`:
    - `steady_state_time=0.0018600664s`, compile `3.4695892399s`.
    - `max_abs_diff=3.814697265625e-06`, `mean_abs_diff=4.2589071824750135e-08`, `metadata_mismatches=0`.
  - Target skip-block timing `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260630-231318`, EP=8/T=32768/D=2560/I=1280/E_local=32/K=4:
    - `steady_state_time=0.0233611513s`, `73.54 TFLOP/s/rank`, compile `3.3899511560s`.
  - Expert-group sweep `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260630-231356`, same target shape, skip enabled:
    - `expert_group_size=8`: `0.0270230927s`, `63.57 TFLOP/s/rank`.
    - `expert_group_size=16`: `0.0250623534s`, `68.55 TFLOP/s/rank`.
    - `expert_group_size=32`: `0.0234545576s`, `73.25 TFLOP/s/rank`.
  - Grid scheduler sweep `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260630-231454`, same target shape, `expert_group_size=32`, skip enabled:
    - `grid_block_n=1`: `0.0240313623s`, `71.49 TFLOP/s/rank`.
    - `grid_block_n=2`: `0.0233627524s`, `73.54 TFLOP/s/rank`.
    - `grid_block_n=4`: `0.0233073560s`, `73.71 TFLOP/s/rank`.
    - `grid_block_n=8`: `0.0234101349s`, `73.39 TFLOP/s/rank`.
- Interpretation:
  - Skipping padded blocks recovers about 9.4% versus the padded rough-routing row `0.0257171070s`, but still trails exact-balanced compact pull `0.0188209613s` by about 23.8%.
  - Larger expert groups are better in this compact pull schedule. `expert_group_size=32` wins over `16` and `8`, supporting the idea that the pull unit should be the whole local expert group for this target.
  - `grid_block_n=4` is the best measured scheduler value, but the margin over `2` is tiny.

### 2026-07-01 13:05 PDT - FWD-SWQ-033 compact ring expert-group token chunks
- Hypothesis:
  - The ring/pull unit should be an expert group, not one expert or one 64-row WGMMA block.
  - Staging `expert_group_size * ring_block_tokens * D` into a local-GMEM ring should reduce per-entry queue/semaphore overhead enough to approach the direct peer-GMEM pull path.
- Commit Hash: uncommitted worktree.
- Code change:
  - Updated `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`.
  - Added/kept compact `implementation=ring` with a destination-side local-GMEM ring:
    - queue entry decodes `(expert_group, token_chunk, source)`;
    - prefetch copies remote source-packed rows into a local ring slot;
    - W13 reads the ring slot into SMEM and computes all experts/blocks in that expert-group token chunk.
  - Made `ring_block_tokens` real instead of requiring `ring_block_tokens == block_m`; it must now be a multiple of `block_m`.
  - Added/tuned knobs: `ring_block_tokens`, `ring_num_prefetch`, `ring_max_active_srcs`, `ring_queue_order`, `copy_tile`, `expert_group_size`, and `grid_block_n`.
  - Tried moving the ring from an output-backed local GMEM buffer to `scratch_shapes=mgpu.GMEM(...)`; Mosaic returned `NotImplementedError: Unsupported memory space: gmem`, so this was reverted.
- Local checks:
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
- H100 results:
  - Initial blocking-ring smoke `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-193342`, EP=8/T=1024/D=256/I=256/E_local=8/K=4:
    - `steady_state_time=0.0112086915s`, `max_abs_diff=3.814697265625e-06`, `metadata_mismatches=0`.
  - Initial queue smoke `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-193544`, EP=8/T=512/D=256/I=256/E_local=8/K=4:
    - Best row before multi-block chunks: `expert_group_size=2`, `ring_queue_order=src_eg_block`, `steady_state_time=0.0031896159s`.
  - Multi-block ring correctness sweep `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-194233`, EP=8/T=1024/D=256/I=256/E_local=8/K=4:
    - All 12 rows matched reference: `max_abs_diff=3.814697265625e-06`, `metadata_mismatches=0`.
    - Best row: `expert_group_size=8`, `ring_block_tokens=128`, `ring_queue_order=eg_block_src`, `steady_state_time=0.0024283500s`.
  - Target group/chunk sweep `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-194915`, EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, uniform routing:
    - Best row: `expert_group_size=32`, `ring_block_tokens=256`, `ring_queue_order=eg_block_src`, `copy_tile=128`, `ring_num_prefetch=2`, `steady_state_time=0.0249642887s`, `68.82 TFLOP/s/rank`, compile `10.8333007041s`.
    - Other useful rows: `EG=32,block_tokens=128` `0.0276660859s`; `EG=32,block_tokens=512` `0.0274713620s`; `EG=16,block_tokens=512` `0.0281273764s`; `EG=8,block_tokens=512` `0.0333962497s`.
  - Focused tax sweep `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-195358`, same target shape, `EG=32`, `block_tokens=256`, `ring_queue_order=eg_block_src`:
    - Best row: `copy_tile=256`, `grid_block_n=2`, `ring_num_prefetch=2`, `steady_state_time=0.0252261278s`, `68.10 TFLOP/s/rank`, compile `11.0031423101s`.
    - `grid_block_n=1` was consistently worse (`0.0276682594s` to `0.0300691886s`).
    - `ring_num_prefetch=1` did not improve over `2`; `copy_tile=256` did not materially beat the earlier `copy_tile=128` row.
  - GMEM scratch-shape smoke `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-200019`:
    - Failed as an expected negative result: `NotImplementedError: Unsupported memory space: gmem`.
- Interpretation:
  - Target results support the updated unit choice: whole local expert groups are better than smaller groups. `EG=32` beat `EG=16` and `EG=8` at the target shape.
  - The best ring row, `0.0249642887s`, is close but still about 6.9% slower than the direct compact rough-routing skip row `0.0233611513s`.
  - This looks software-structure-bound rather than compute-bound: the extra remote-GMEM -> local-GMEM -> SMEM hop plus per-entry semaphore/queue structure dominates; copy-tile and prefetch-depth sweeps did not recover the gap.
  - Current Mosaic does not provide a usable local GMEM scratch allocation via `scratch_shapes=mgpu.GMEM`, so the prototype still pays for an output-backed ring buffer in this compact harness.

### 2026-07-01 14:55 PDT - FWD-SWQ-034 compact ring tax breakdown
- Hypothesis:
  - The compact ring path is slower because it is a blocking local-GMEM staging implementation, not a true copy/compute pipeline.
  - Need to separate remote-copy cost, compute-from-scratch cost, queue/semaphore cost, queue sparsity, and prefetch-depth behavior before proposing producer/consumer warp specialization.
- Commit Hash: uncommitted worktree.
- Code change:
  - Updated `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`.
  - Added benchmark implementations:
    - `ring_copy`: run the ring remote-source -> local-GMEM scratch refill path and skip WGMMA.
    - `ring_compute`: run WGMMA/SILU from a prefilled local scratch input and skip remote refill.
    - `ring_queue`: run the ring queue/semaphore protocol and skip both refill and WGMMA.
    - `pull`: direct compact baseline, now also honors the `src_eg_block` queue-order analogue where feasible.
  - Added `CompactConfig.ring_stage` and stage dispatch for the variants above.
  - Added target queue-sparsity counters in `queue_stats`: total/live queue entries, no-live-expert entries, no-live-token-block entries, masked copy tiles, and masked compute blocks.
  - Renamed the `n_group == 2` accumulators to make the adjacent-N-tile meaning explicit: `gate_n0_acc`, `up_n0_acc`, `gate_n1_acc`, `up_n1_acc` and matching SMEM names.
- Execution-order finding:
  - Current ring order is:
    1. prologue copies the first `ring_num_prefetch` queue entries into slots;
    2. main loop waits for the current slot's `prefetch_sem`;
    3. computes `_compute_queue_entry(queue_index, slot)`;
    4. signals and waits `compute_sem` for the current entry;
    5. only then calls `_prefetch_next()` into the released slot.
  - `_prefetch_next()` is ordinary blocking copy work in the same worker loop. It is after compute and after the compute semaphore wait, so increasing `ring_num_prefetch` only changes the initial fill and slot count. It does not overlap remote refill for entry `n+1` with WGMMA for entry `n`.
- Async-copy API check:
  - Local Pallas/Mosaic exposes `mgpu.copy_gmem_to_smem` for async GMEM -> SMEM, `mgpu.copy_smem_to_gmem` for async SMEM -> GMEM, and `mgpu.async_prefetch` for L2 prefetch.
  - I did not find a supported async GMEM -> GMEM staging primitive for remote GMEM -> local GMEM scratch.
  - Prior negative test `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-200019` also showed `scratch_shapes=mgpu.GMEM(...)` is rejected with `NotImplementedError: Unsupported memory space: gmem`.
  - Conclusion: this ring version is a blocking staging proof, not a true pipeline. A real overlap design needs a structural producer/consumer split or another supported async refill mechanism.
- Local checks:
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
  - Host-only target queue-sparsity sweep, seed 0 uniform routing:
    - `ring_block_tokens=32` unsupported with current `block_m=64`.
    - `EG=32,block_tokens=64`: `80` entries/rank, live-entry fraction `0.9078`, masked tile fraction `0.1518`.
    - `EG=32,block_tokens=128`: `40` entries/rank, live-entry fraction `1.0`, masked tile fraction `0.1518`.
    - `EG=32,block_tokens=256`: `24` entries/rank, live-entry fraction `1.0`, masked tile fraction `0.2931`.
    - For the best-size configs, a sparse queue would not remove queue entries; the waste is intra-entry masked copy/compute tiles.
- H100 results:
  - Stage-variant smoke `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-211646`, EP=8/T=1024/D=256/I=256/E_local=8/K=4:
    - `pull`: best `0.0024555209s`.
    - `ring`: best `0.0026456320s`, correct vs reference.
    - `ring_copy`: best `0.0023739219s`.
    - `ring_compute`: best `0.0024660882s`.
    - `ring_queue`: best `0.0023732670s`.
  - Target tax sweep `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-212229`, EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, `EG=32`, `copy_tile=512`, `capacity_factor=1.25`, uniform routing:
    - Completed `128` rows.
    - `ring_block_tokens=32`: unsupported because current WGMMA uses `block_m=64`.
    - Full ring best:
      - `block_tokens=64`: `0.034483s`, `prefetch=2`, `eg_block_src`, `80` entries/rank, live `0.9094`, masked `0.1510`.
      - `block_tokens=128`: `0.027634s`, `prefetch=3`, `eg_block_src`, `40` entries/rank, live `1.0`, masked `0.1510`.
      - `block_tokens=256`: `0.025150s`, `prefetch=4`, `eg_block_src`, `24` entries/rank, live `1.0`, masked `0.2925`.
    - Copy-only best:
      - `block_tokens=64`: `0.011551s`.
      - `block_tokens=128`: `0.010945s`.
      - `block_tokens=256`: `0.012126s`.
    - Compute-only best:
      - `block_tokens=64`: `0.025421s`.
      - `block_tokens=128`: `0.019289s`.
      - `block_tokens=256`: `0.016154s`.
    - Queue-only best:
      - `block_tokens=64`: `0.003365s`.
      - `block_tokens=128`: `0.003272s`.
      - `block_tokens=256`: `0.003310s`.
  - Target direct/EG sweep `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-214108`, same target shape, `ring_block_tokens=256`, `ring_num_prefetch=4`:
    - Direct pull:
      - `EG=1`: best `0.053968s`.
      - `EG=2`: best `0.049101s`.
      - `EG=4`: best `0.037149s`.
      - `EG=8`: best `0.040787s`.
      - `EG=16`: best `0.036096s`.
      - `EG=32`: best `0.026536s` with `src_eg_block`.
    - Full ring:
      - `EG=1`: best `0.096610s`, `768` entries/rank, live `0.830`, masked `0.292`.
      - `EG=2`: best `0.059629s`, `384` entries/rank, live `0.921`, masked `0.292`.
      - `EG=4`: best `0.036710s`, `192` entries/rank, live `0.980`, masked `0.292`.
      - `EG=8`: best `0.035853s`, `96` entries/rank, live `1.0`, masked `0.292`.
      - `EG=16`: best `0.028285s`, `48` entries/rank, live `1.0`, masked `0.292`.
      - `EG=32`: best `0.024935s`, `24` entries/rank, live `1.0`, masked `0.292`.
- Interpretation:
  - Increasing `ring_num_prefetch` does not show a monotonic speedup. The best full-ring rows occur at different depths, with noisy regressions at larger depths. Treat this as no effective overlap.
  - The best ring family is `EG=32, block_tokens=256`, because it minimizes queue entries. Smaller expert groups are much slower.
  - Queue sparsity is not the main target-size issue at the winning settings: all queue entries are live for `EG>=16, block_tokens>=128`; a sparse host queue would mainly help small-EG configs that are already slow.
  - The remaining waste is intra-entry masked work and the blocking remote-GMEM -> local-GMEM -> SMEM staging path. The measured target copy-only tax is about `11-12 ms`, compute-only about `16 ms`, queue-only about `3.3 ms`, and full ring about `25 ms`.
  - This supports the expected conclusion: the ring implementation is a correctness/perf-isolation proof. It is not a production performance path unless we get true async refill or split the kernel into a producer/consumer design that can overlap remote staging with WGMMA.

### 2026-07-01 15:15 PDT - FWD-SWQ-035 direct remote-SMEM double-buffer test
- Hypothesis:
  - If Mosaic can issue remote peer GMEM -> local SMEM copies asynchronously, a direct destination-pull K-loop can overlap the LHS fetch for `kk+1` with WGMMA for `kk` without staging through local GMEM.
  - Start with the constrained case: `block_m=64`, `expert_group_size=1`, `n_group in {1,2}`, `block_k in {64,128}`, no ring semaphores, no local GMEM scratch.
- Commit Hash: uncommitted worktree.
- Code change:
  - Updated `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`.
  - Added benchmark implementations:
    - `direct_remote_smem_blocking`: constrained direct destination-pull variant using the existing blocking `mgpu.load(remote_ref, ..., layout=WGMMA)` LHS path.
    - `direct_remote_smem_double_buffer`: attempts explicit async `mgpu.copy_gmem_to_smem(remote_sorted_x_ref.at[...], lhs_smem, barrier)` into two LHS SMEM buffers plus two gate/up SMEM buffer sets.
  - Added `--block-k-values` so the compact harness can sweep `block_k=64,128` without separate commands.
  - Added validation that the direct remote-SMEM prototype runs only with `block_m=64`, `block_k in {64,128}`, and `expert_group_size=1`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
- H100 results:
  - Smoke `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-215740`, EP=8/T=1024/D=256/I=256/E_local=8/K=4, `block_k=64`, `n_group=1`, `EG=1`, `--check`:
    - `direct_remote_smem_blocking`: `0.0035452221s`, compile `5.2569512750s`, `max_abs_diff=3.814697265625e-06`, `metadata_mismatches=0`.
    - `direct_remote_smem_double_buffer`: compile/lowering failure: `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.`
  - Target direct remote-SMEM sweep `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-220012`, EP=8/T=32768/D=2560/I=1280/E_local=32/K=4, `EG=1`, uniform routing:
    - `direct_remote_smem_blocking`, `block_k=64,n_group=1`: `0.068488s`, `25.08 TFLOP/s/rank`.
    - `direct_remote_smem_blocking`, `block_k=64,n_group=2`: `0.071185s`, `24.13 TFLOP/s/rank`.
    - `direct_remote_smem_blocking`, `block_k=128,n_group=1`: `0.059341s`, `28.95 TFLOP/s/rank`.
    - `direct_remote_smem_blocking`, `block_k=128,n_group=2`: `0.056502s`, `30.41 TFLOP/s/rank`.
    - `direct_remote_smem_double_buffer`, `block_k=64,n_group=1`: failed with `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.`
    - `direct_remote_smem_double_buffer`, `block_k=64,n_group=2`: same peer-id GMEM lowering failure.
    - `direct_remote_smem_double_buffer`, `block_k=128,n_group=1`: same peer-id GMEM lowering failure.
    - `direct_remote_smem_double_buffer`, `block_k=128,n_group=2`: failed earlier on SMEM pressure, `smem_bytes=294928 > max_smem_bytes=232448`.
  - Target baseline comparison `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-220407`, same target shape, `EG=32`, `ring_block_tokens=256`, `ring_num_prefetch=4`, `ring_queue_order=eg_block_src`:
    - `pull`, `block_k=64,n_group=1`: `0.041067s`, `41.83 TFLOP/s/rank`.
    - `pull`, `block_k=64,n_group=2`: `0.030538s`, `56.26 TFLOP/s/rank`.
    - `pull`, `block_k=128,n_group=1`: `0.040129s`, `42.81 TFLOP/s/rank`.
    - `pull`, `block_k=128,n_group=2`: `0.034237s`, `50.18 TFLOP/s/rank`.
    - `ring`, `block_k=64,n_group=1`: `0.026284s`, `65.36 TFLOP/s/rank`.
    - `ring`, `block_k=64,n_group=2`: `0.028880s`, `59.49 TFLOP/s/rank`.
    - `ring`, `block_k=128,n_group=1`: `0.022377s`, `76.77 TFLOP/s/rank`.
    - `ring`, `block_k=128,n_group=2`: `0.032636s`, `52.64 TFLOP/s/rank`.
- Interpretation:
  - The intended async remote GMEM -> SMEM K-tile prefetch cannot be tested as a runtime overlap path in current warpgroup lowering: `mgpu.copy_gmem_to_smem` from an `mgpu.remote_ref(..., peer_id)` is rejected.
  - Therefore `double_buffer=true` cannot improve over blocking; it does not produce an executable kernel for the relevant cases.
  - On the executable constrained blocking path, `n_group=2` only helped at `block_k=128` (`0.059341s -> 0.056502s`) and hurt at `block_k=64` (`0.068488s -> 0.071185s`).
  - `block_k=128` beats `block_k=64` for the constrained blocking path, but the path is much slower than the EG=32 baselines because `EG=1` explodes schedule/phase overhead.
  - This reinforces the previous direction: without supported async peer GMEM -> SMEM copy, the next real overlap experiment needs producer/consumer warp specialization or a different supported data movement mechanism, not more direct double-buffer tuning.

### 2026-07-01 15:27 PDT - FWD-SWQ-036 peer GMEM -> SMEM Lane lowering repro
- Hypothesis:
  - The `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.` failure may be specific to warpgroup lowering. Lane lowering may permit a manual `mgpu.copy_gmem_to_smem(remote_ref.at[...], smem, barrier)` path.
- Code change:
  - Added minimal repro `lib/levanter/scripts/bench/repro_remote_gmem_to_smem_peer_lane.py`.
  - The repro runs the same two-rank remote GMEM -> SMEM copy with both `mgpu.LoweringSemantics.Warpgroup` and `mgpu.LoweringSemantics.Lane`.
- Local check:
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_remote_gmem_to_smem_peer_lane.py`
- H100 result:
  - Job `/dlwh/iris-run-repro_remote_gmem_to_smem_peer_lane-20260701-222612`, cluster `cw-us-east-02a`, H100x8, succeeded in `16.27s`.
  - `Warpgroup`: reproduces `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.`
  - `Lane`: succeeds; output `out[:, 0, 0] = [2 1]`, confirming rank 0 pulled rank 1's value and rank 1 pulled rank 0's value.
- Interpretation:
  - The peer-id GMEM restriction is not a blanket prohibition on remote GMEM -> SMEM copies. It is specific to warpgroup lowering for this repro.
  - `compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)` is a concrete next experiment for the direct remote-SMEM/double-buffer path.
  - The main open question is whether Lane lowering still supports the WGMMA-heavy compute path with acceptable performance, or whether this only proves the copy primitive.

### 2026-07-01 15:45 PDT - FWD-SWQ-037 Lane lowering with WGMMA matmul
- Hypothesis:
  - Lane lowering may permit peer GMEM -> SMEM async copies, but WGMMA requires Lane-specific SMEM layout transforms. If operands are allocated as logical SMEM backed by `(TilingTransform((8, swizzle_elems)), SwizzleTransform(128))`, WGMMA should lower under `LoweringSemantics.Lane`.
- Code change:
  - Extended `lib/levanter/scripts/bench/repro_remote_gmem_to_smem_peer_lane.py` with a two-rank WGMMA case:
    - rank 0 pulls rank 1's all-2 LHS tile, rank 1 pulls rank 0's all-1 LHS tile;
    - both multiply by an all-1 RHS tile; expected `out[:, 0, 0] = [128, 64]`.
  - Updated `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`:
    - added `--lowering-semantics {warpgroup,lane}`;
    - added WGMMA SMEM layout helper for Lane lowering;
    - added `--debug-exceptions`;
    - for `direct_remote_smem_double_buffer` under Lane only, statically unrolled the phase loop so async-copy peer ids are derived from `axis_index + static_phase` instead of a `pl.loop` block argument.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_remote_gmem_to_smem_peer_lane.py lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_remote_gmem_to_smem_peer_lane.py lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
- H100 results:
  - Minimal repro, current script: `/dlwh/iris-run-repro_remote_gmem_to_smem_peer_lane-20260701-224410`, `cw-us-east-02a`, H100x8, succeeded in `23.55s`.
    - `copy Warpgroup`: `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.`
    - `copy Lane`: success, `out[:, 0, 0] = [2 1]`.
    - `wgmma Warpgroup`: same peer-id NotImplementedError.
    - `wgmma Lane`: success, `out[:, 0, 0] = [128 64]`.
  - Earlier WGMMA repro without Lane SMEM transforms: `/dlwh/iris-run-repro_remote_gmem_to_smem_peer_lane-20260701-223055`.
    - `wgmma Lane` failed with `ValueError: When WGMMA lhs is passed in as a ref, it must be transformed by swizzling and tiling appropriately.`
  - Compact direct double-buffer Lane smoke before static phase unroll: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-223621`.
    - Failed row: `ValueError: Failed to recompute the async_copy peer id on the host`.
  - Compact debug rerun: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-223808`.
    - Full stack shows `ReplicationError: Can't recompute a value that's a block argument` inside `launch_context.py:init_tma_desc`, confirming the dynamic `pl.loop` phase variable was the peer-id blocker.
  - Compact direct double-buffer Lane smoke after static phase unroll: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-223933`, `cw-us-east-02a`, H100x8, succeeded in `2m10.92s`.
    - Shape: `EP=8,T=1024,D=256,I=256,E_local=8,K=4,routing=uniform`.
    - Config: `direct_remote_smem_double_buffer`, `lowering_semantics=lane`, `block_m=64`, `block_n=128`, `block_k=64`, `expert_group_size=1`, `n_group=1`.
    - `compile_time=82.66401271894574s`.
    - `steady_state_time=0.011414404027163982s`.
    - `effective_tflops_per_rank=0.09406902203958356`.
    - `max_abs_diff=3.814697265625e-06`, `mean_abs_diff=4.2589071824750135e-08`, `metadata_mismatches=0`.
- Interpretation:
  - Lane lowering does work with WGMMA matmul operations, provided WGMMA operands use the expected swizzle/tile SMEM transforms.
  - The compact async-copy path also compiles and is correct on a small shape after removing dynamic `pl.loop` peer ids.
  - Static phase unroll is compile-heavy and not a performance path as-is: even the small smoke spends `82.7s` compiling and runs at only `0.094 TFLOP/s/rank`.
  - For target-size work, Lane lowering looks usable for correctness experiments and for proving remote-GMEM -> WGMMA plumbing. The next performance question is how to express rotating peer phases without dynamic peer ids and without exploding compile size.

### 2026-07-01 16:04 PDT - FWD-SWQ-038 static-peer conditional instead of full phase unroll
- Hypothesis:
  - We do not need to unroll all `expert_groups * ep_size` phases. The peer id only needs to be static/host-recomputable at the `mgpu.remote_ref` site. Keep the dynamic `pl.loop(0, total_phases)`, compute `(expert_group, peer_phase)` dynamically, then use an 8-way `pl.when(peer_phase == static_peer_phase)` branch so each remote-copy branch closes over a Python static peer phase.
- Code change:
  - Updated `lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`.
  - Replaced the Lane-only full phase unroll:
    - old: `for phase in range(total_phases): _compute_phase(phase)`
    - new: `@pl.loop(0, total_phases)` plus static-peer `pl.when` branches inside `_compute_phase`.
  - This leaves Warpgroup/default behavior unchanged.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
- H100 results:
  - Small correctness smoke: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-224927`, `cw-us-east-02a`, H100x8, succeeded.
    - Shape: `EP=8,T=1024,D=256,I=256,E_local=8,K=4,routing=uniform`.
    - Config: `direct_remote_smem_double_buffer`, `lowering_semantics=lane`, `block_k=64`, `n_group=1`, `expert_group_size=1`.
    - `compile_time=10.79410322709009s` vs `82.66401271894574s` for the full phase unroll.
    - `steady_state_time=0.0032046420965343714s` vs `0.011414404027163982s` for the full phase unroll.
    - `effective_tflops_per_rank=0.33505826599519106`.
    - `max_abs_diff=3.814697265625e-06`, `mean_abs_diff=4.2589071824750135e-08`, `metadata_mismatches=0`.
  - Target single row: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-225120`, same target shape as below.
    - `block_k=64,n_group=1`: `compile_time=91.53700038208626s`, `steady_state_time=0.048244417645037174s`, `35.61006645453263 TFLOP/s/rank`.
  - Target sweep: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-225508`, `EP=8,T=32768,D=2560,I=1280,E_local=32,K=4,routing=uniform`, H100x8, succeeded.
    - `block_k=64,n_group=1`: `compile_time=91.53104437398724s`, `steady_state_time=0.045079481011877455s`, `38.1101751803076 TFLOP/s/rank`.
    - `block_k=64,n_group=2`: `compile_time=142.53197380900383s`, `steady_state_time=0.048396044333154954s`, `35.49849873211743 TFLOP/s/rank`.
    - `block_k=128,n_group=1`: `compile_time=107.96291107893921s`, `steady_state_time=0.03846787537137667s`, `44.66030166247046 TFLOP/s/rank`.
    - `block_k=128,n_group=2`: failed at compile with `ValueError: Mosaic GPU kernel exceeds available shared memory: smem_bytes=294928 > max_smem_bytes=232448`.
- Interpretation:
  - Yes, a conditional works, as long as the peer-id expression inside each branch is static with respect to `pl.loop` block arguments.
  - A fully dynamic `remote_ref(..., peer_phase_from_loop)` still fails because async-copy TMA descriptor host init cannot recompute peer ids from block arguments.
  - Static-peer branching avoids the full-unroll compile blowup and is much faster on the small smoke.
  - Target compact best is now `0.038468s` at `block_k=128,n_group=1`. This is a real improvement over the previous constrained direct remote-SMEM blocking rows (`0.056502-0.071185s`) and gets close to the earlier production target forward repeat (`0.038529s`), but it is still a compact harness result, not production integration.

### 2026-07-01 16:10 PDT - FWD-SWQ-039 lax.switch static-peer attempt
- Hypothesis:
  - Since `pl.when` lowers through `jax.lax.cond`, an 8-way `jax.lax.switch` over static-peer branches might express the same static-peer selection more cleanly and perhaps avoid eight separate conditionals.
- Code tried:
  - Replaced the working `pl.when(peer_phase == static_peer_phase)` chain with `lax.switch(peer_phase, branches, expert_group)`, where each branch called `_compute_expert_group_peer(expert_group, static_peer_phase)`.
  - After that failed, tried a zero-operand switch: branches closed over dynamic `expert_group` and called `_compute_expert_group_peer(expert_group, static_peer_phase)`.
  - Reverted back to the working `pl.when` chain after both switch attempts failed.
- H100 results:
  - Explicit-operand switch smoke: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-230700`, same small config as FWD-SWQ-038.
    - Failed row: `ValueError: Failed to recompute the async_copy peer id on the host`.
  - Zero-operand switch smoke: `/dlwh/iris-run-compact_moe_permute_up_mgpu-20260701-230854`, same small config.
    - Failed row: `ValueError: Failed to recompute the async_copy peer id on the host`.
- Local checks after revert:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py`
- Interpretation:
  - `lax.switch` is not equivalent enough for this Mosaic async-copy peer-id use case. Even when each branch closes over a Python static peer phase, the lowered switch/case still leaves the async-copy descriptor host init unable to recompute the peer id.
  - The working form remains the explicit `pl.when` static-peer chain from FWD-SWQ-038.

### 2026-07-01 18:25 PDT - FWD-SWQ-040 Phase 0 source-push primitive repro
- Spec reference:
  - Source-push inbox pipeline spec from `/Users/dlwh/.codex/attachments/47df6d89-bafb-4095-a361-89ad773a8abc/pasted-text-1.txt`.
  - Required Phase 0 control: confirm `copy_smem_to_gmem(local_smem, remote_gmem)` works, and stop optimizing destination-pull async paths.
- Code change:
  - Added `lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py`.
  - Two-rank repro: each rank loads a local `[64, 128]` bf16 tile into SMEM, then calls `mgpu.copy_smem_to_gmem(tile_smem, remote_y_ref.at[...])` into the peer rank's output GMEM and waits with `mgpu.wait_smem_to_gmem(0, wait_read_only=False)`.
  - Expected output after cross-write is `out[:, 0, 0] = [2, 1]`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py`
- H100 result:
  - Job `/dlwh/iris-run-repro_smem_to_remote_gmem_peer-20260702-012447`, cluster `cw-us-east-02a`, H100x8, succeeded in `21.51s`.
  - `Warpgroup`: fails with `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.` from `copy_smem_to_gmem`.
  - `Lane`: succeeds, `out[:, 0, 0] = [2 1]`.
- Interpretation:
  - The source-push primitive works only under Lane lowering for this direct remote-peer GMEM write path.
  - This corrects the spec assumption: `local SMEM -> remote GMEM` is not usable in Warpgroup lowering in the current environment.
  - Phase 1/2 source-push inbox prototypes should use `mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)` unless a different supported remote write mechanism is found.

### 2026-07-01 18:29 PDT - FWD-SWQ-041 Phase 1 minimal source-push inbox semaphore repro
- Hypothesis:
  - A minimal source-push inbox slot can be synchronized with a remote full semaphore under Lane lowering:
    - source writes payload with `copy_smem_to_gmem` into peer inbox GMEM;
    - source writes peer metadata;
    - source signals peer full semaphore;
    - destination waits its local full semaphore before returning/validating the inbox.
- Code change:
  - Added `lib/levanter/scripts/bench/repro_source_push_inbox_sem.py`.
  - Two-rank repro with one inbox slot per rank:
    - payload: bf16 `[64, 128]`;
    - metadata: int32 `[src_rank, valid_rows]`;
    - expected payload after exchange: `inbox[:, 0, 0] = [2, 1]`;
    - expected metadata: `[[1, 64], [0, 64]]`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_sem.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_sem.py`
- H100 result:
  - Job `/dlwh/iris-run-repro_source_push_inbox_sem-20260702-012758`, cluster `cw-us-east-02a`, H100x8, succeeded in `21.61s`.
  - `Warpgroup`: fails with `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.`
  - `Lane`: succeeds:
    - `inbox[:, 0, 0] = [2 1]`
    - `meta = [[1 64], [0 64]]`
- Interpretation:
  - The basic source-push inbox invariant works under Lane for a single slot: remote payload write, remote metadata write, remote full semaphore, local full wait.
  - Warpgroup remains unusable for direct peer GMEM refs in the async copy path.
  - Next Phase 1 step should lift this from a one-slot repro into the compact routing harness: deterministic send entries, `inbox[src, slot, :, :]`, metadata per `(src, slot)`, and a send-only validator/release path before adding W13 compute.

### 2026-07-01 18:49 PDT - FWD-SWQ-042 Queue-shaped source-push inbox send-only repro
- Hypothesis:
  - The one-slot Phase 1 repro should be lifted to a queue-shaped send-only prototype before adding W13 compute:
    - each source rank owns deterministic send entries;
    - source workers load `[BLOCK_M, block_k]` local GMEM tiles into SMEM;
    - source workers push those SMEM tiles into remote destination inbox GMEM;
    - source writes metadata and signals the destination full semaphore;
    - destination waits the full count before host validation.
  - This still does not test compute overlap, but it tests the payload/metadata layout, multi-entry queue shape, and sensitivity to source-send worker count.
- Code change:
  - Added `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - Self-contained JAX/Pallas MGPU script with no Marin/Levanter imports.
  - Output inbox layout: `inbox[dst_rank, src, slot, block_m, hidden_dim]`.
  - Metadata per destination/source/slot: `[src, expert, dst_row_start, valid_rows]`.
  - Current limitation: this send-only repro does not reuse slots yet, so it requires `inbox_slots >= entries_per_rank` and has no destination release path. It is a queue/inbox primitive check, not the full pipeline.
- Local checks:
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 smoke:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-014516`, cluster `cw-us-east-02a`, H100x8, succeeded in `27.04s`.
  - Config: `EP=8`, `entries_per_rank=2`, `inbox_slots=2`, `D=2560`, `block_m=64`, `block_k=128`, `num_send_sms=4`, `num_sms=16`, Lane lowering.
  - Result:
    - `bytes_per_rank=655360`
    - `compile_time=2.100447454955429s`
    - `steady_state_time=0.0014150138013064861s`
    - `send_gbps_per_rank=0.4631474261204409`
    - `max_abs_diff=0.0`, `metadata_mismatches=0`
- H100 send-SM sweep:
  - Job `/dlwh/iris-run-job-20260702-014627`, cluster `cw-us-east-02a`, H100x8, succeeded in `1m19.86s`.
  - Config common fields: `EP=8`, `entries_per_rank=64`, `inbox_slots=64`, `D=2560`, `block_m=64`, `block_k=128`, `num_sms=16`, Lane lowering, validation enabled.
  - Rows:
    - `num_send_sms=1`: `steady_state_time=0.010348283220082521s`, `send_gbps_per_rank=2.026569968562647`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `num_send_sms=2`: `steady_state_time=0.002098269620910287s`, `send_gbps_per_rank=9.994673606770315`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `num_send_sms=4`: `steady_state_time=0.0017919364385306836s`, `send_gbps_per_rank=11.70327225289074`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `num_send_sms=8`: `steady_state_time=0.0015863002277910709s`, `send_gbps_per_rank=13.220397773757444`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
- H100 block-K sweep:
  - Job `/dlwh/iris-run-job-20260702-014826`, cluster `cw-us-east-02a`, H100x8, succeeded in `58.75s`.
  - Config common fields: `EP=8`, `entries_per_rank=64`, `inbox_slots=64`, `D=2560`, `block_m=64`, `num_send_sms=8`, `num_sms=16`, Lane lowering, validation enabled.
  - Rows:
    - `block_k=64`: `steady_state_time=0.0016892635729163885s`, `send_gbps_per_rank=12.414593161323088`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `block_k=128`: `steady_state_time=0.0017605913802981378s`, `send_gbps_per_rank=11.911633917262899`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `block_k=256`: `steady_state_time=0.0015535728074610234s`, `send_gbps_per_rank=13.498897444190842`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
- Interpretation:
  - The queue-shaped source-push inbox primitive is correct under Lane lowering for EP=8 and multi-entry sends.
  - Send-only performance scales strongly from one to two source workers, then tapers. The best row so far is `num_send_sms=8, block_k=256`: `0.0015536s` for `20.97 MB/rank`, or `13.50 GB/s/rank`.
  - This is not yet a real pipeline result. It still blocks on `wait_smem_to_gmem` for every K slice and does not include destination local GMEM -> SMEM or W13 compute.
  - The next meaningful step is to add slot reuse/release and a destination consumer path. Without that, high `inbox_slots` is only a correctness/throughput probe, not the intended bounded-inbox protocol.

### 2026-07-01 19:03 PDT - FWD-SWQ-043 Bounded source-push inbox slot reuse/release
- Hypothesis:
  - Pallas/Mosaic shaped semaphores should be enough for a bounded source-push inbox protocol:
    - source waits local `empty_sem[slot]`;
    - source writes remote destination `inbox[src, slot, :, :]` and metadata;
    - source signals remote destination `full_sem[src, slot]`;
    - destination waits `full_sem[src, slot]`, records the consumed entry, and signals remote source `empty_sem[slot]`.
  - This should allow `inbox_slots < entries_per_rank`, so the repro tests slot reuse instead of relying on one slot per queue entry.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - Replaced the previous scalar full-count wait with shaped semaphores:
    - `empty_sem_ref = pl.get_global(mgpu.SemaphoreType.REGULAR((inbox_slots,)))`
    - `full_sem_ref = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, inbox_slots)))`
  - Added destination receiver program at `sm == num_send_sms`, requiring `num_sms > num_send_sms`.
  - Added `seen_payload[src, entry]` and `seen_meta[src, entry, :]` outputs so host validation checks every consumed generation, not only the final slot contents.
  - Final correct sender schedule is round-major, slot-minor:
    - for each generation/round;
    - for each slot owned by this send worker;
    - send `entry = slot + round * inbox_slots`.
- Bugs found while testing:
  - First bounded version let multiple send workers wait on the same slot. A later generation could wake before an earlier generation, producing incorrect rows:
    - Killed job `/dlwh/iris-run-job-20260702-015335`.
    - Bad first row before kill: `entries_per_rank=64`, `inbox_slots=2`, `num_send_sms=4`, `steady_state_time=0.002219265792518854s`, but `max_abs_diff=22.0`, `metadata_mismatches=918`.
  - FIFO-per-slot sender then sent all generations for one owned slot before moving to the next owned slot. That can deadlock: sender blocks on a later generation of slot 0 while receiver waits for the first generation of slot 4.
    - Killed job `/dlwh/iris-run-job-20260702-015654` after rows through `slots=4`; the next `slots=8` row hung.
  - The round-major slot-owner schedule fixed both issues.
- H100 smoke after slot reuse:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-015242`, cluster `cw-us-east-02a`, H100x8, succeeded in `26.59s`.
  - Config: `entries_per_rank=4`, `inbox_slots=2`, `block_k=128`, `num_send_sms=2`, `num_sms=16`, Lane lowering.
  - Result: `steady_state_time=0.00783512918278575s`, `send_gbps_per_rank=0.1672876055291763`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
- H100 smoke after round-major fix:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-015941`, cluster `cw-us-east-02a`, H100x8, succeeded in `26.27s`.
  - Config: `entries_per_rank=64`, `inbox_slots=8`, `block_k=128`, `num_send_sms=4`, `num_sms=16`, Lane lowering.
  - Result: `steady_state_time=0.00194842298515141s`, `send_gbps_per_rank=10.76333022132272`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
- Final H100 bounded sweep:
  - Job `/dlwh/iris-run-job-20260702-020050`, cluster `cw-us-east-02a`, H100x8, succeeded in `1m51.4s`.
  - Config common fields: `EP=8`, `entries_per_rank=64`, `D=2560`, `block_m=64`, `block_k=128`, `num_sms=16`, Lane lowering, validation enabled.
  - Rows:
    - `inbox_slots=2`, `num_send_sms=4`: `steady_state_time=0.0054614458233118056s`, `send_gbps_per_rank=3.8399209071130045`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `inbox_slots=2`, `num_send_sms=8`: `steady_state_time=0.0021684879902750254s`, `send_gbps_per_rank=9.671033500785136`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `inbox_slots=4`, `num_send_sms=4`: `steady_state_time=0.0018162766005843877s`, `send_gbps_per_rank=11.546435159299199`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `inbox_slots=4`, `num_send_sms=8`: `steady_state_time=0.0018009061925113202s`, `send_gbps_per_rank=11.644981891453059`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `inbox_slots=8`, `num_send_sms=4`: `steady_state_time=0.0018964590039104224s`, `send_gbps_per_rank=11.058251170606676`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
    - `inbox_slots=8`, `num_send_sms=8`: `steady_state_time=0.001735408790409565s`, `send_gbps_per_rank=12.084484137625358`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
- Interpretation:
  - Bounded source-push slot reuse/release is now correct in the compact repro.
  - Correctness requires a real per-slot FIFO discipline. Per-slot semaphores alone prevent overwrites, but they do not impose generation order if multiple workers contend for the same slot.
  - Best bounded row so far is `inbox_slots=8`, `num_send_sms=8`: `0.0017354s`, `12.08 GB/s/rank`.
  - No-reuse send-only best from FWD-SWQ-042 was `0.0015536s`, `13.50 GB/s/rank` with `block_k=256`; bounded reuse at `block_k=128` is close but not faster. This is expected because bounded reuse adds destination receiver semaphore waits and release signals.
  - This still does not prove communication/compute overlap. The next step is to replace the destination receiver's `seen_*` recording with local inbox GMEM -> SMEM -> W13 for one M-owner compute path.

### 2026-07-01 19:25 PDT - FWD-SWQ-044 Source-push inbox W13 M-owner bring-up
- Hypothesis:
  - The source-push inbox protocol can feed W13 directly from the destination's local inbox GMEM:
    - source SMs wait local `empty_sem[slot]`;
    - source SMs push local X tiles through SMEM into destination `inbox[src, slot, :, :]`;
    - destination compute SMs wait `full_sem[src, slot]`;
    - destination loads local inbox GMEM plus local expert weights into WGMMA SMEM and computes `hidden = silu(gate) * up`;
    - destination releases the slot back to the source.
  - A single M-owner receiver should be correct but compute-serialized; slot-owned receivers should expose whether multiple destination SMs can consume independent inbox slots safely.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - Added `implementation=m_owner` and `implementation=m_owner_slots`.
  - Added local expert weights, WGMMA SMEM transforms for Lane lowering, W13 gate/up accumulation, SiLU activation, hidden output, and host reference validation.
  - `m_owner` uses one destination receiver at `sm == num_send_sms`.
  - `m_owner_slots` initializes empty slots with one worker, then lets all `sm >= num_send_sms` consume slots by `slot % (num_sms - num_send_sms)` while preserving per-slot FIFO.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 correctness smokes:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-020906`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: `implementation=m_owner`, `EP=8`, `entries_per_rank=2`, `inbox_slots=2`, `D=256`, `I=256`, `block_k=64`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, Lane lowering.
    - `steady_state_time=0.0018399891s`.
    - `w13_tflops_per_rank=0.0182362`.
    - `hidden_max_abs_diff=0.0069537`, `hidden_mean_abs_diff=0.0011011`, `metadata_mismatches=0`.
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-021001`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: `implementation=m_owner`, `entries_per_rank=8`, `inbox_slots=4`, `D=2560`, `I=1280`, `block_k=128`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, Lane lowering.
    - `steady_state_time=0.0043452730s`.
    - `w13_tflops_per_rank=1.5444`.
    - `hidden_max_abs_diff=0.0143805`, `hidden_mean_abs_diff=0.0019476`, `metadata_mismatches=0`.
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-021700`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: `implementation=m_owner_slots`, `entries_per_rank=4`, `inbox_slots=4`, `D=256`, `I=256`, `block_k=64`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, Lane lowering.
    - `steady_state_time=0.0018883741s`.
    - `w13_tflops_per_rank=0.0355379`.
    - `hidden_max_abs_diff=0.0126772`, `hidden_mean_abs_diff=0.00218024`, `metadata_mismatches=0`.
- H100 timing sweeps:
  - Single M-owner slot-depth sweep: `/dlwh/iris-run-job-20260702-021130`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Common config: `implementation=m_owner`, `entries_per_rank=16`, `D=2560`, `I=1280`, `block_k=128`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation disabled.
    - `inbox_slots=1`: `steady_state_time=0.0075915533s`, `1.76799 TFLOP/s/rank`.
    - `inbox_slots=2`: `steady_state_time=0.0072322971s`, `1.85581 TFLOP/s/rank`.
    - `inbox_slots=4`: `steady_state_time=0.0072396067s`, `1.85394 TFLOP/s/rank`.
    - `inbox_slots=8`: `steady_state_time=0.0072902044s`, `1.84107 TFLOP/s/rank`.
  - Slot-parallel M-owner sweep: `/dlwh/iris-run-job-20260702-021757`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Common config: `implementation=m_owner_slots`, `entries_per_rank=16`, `D=2560`, `I=1280`, `block_k=128`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation disabled.
    - `inbox_slots=1`: `steady_state_time=0.0076613196s`, `1.75189 TFLOP/s/rank`.
    - `inbox_slots=2`: `steady_state_time=0.0053904176s`, `2.48993 TFLOP/s/rank`.
    - `inbox_slots=4`: `steady_state_time=0.0060648710s`, `2.21304 TFLOP/s/rank`.
    - `inbox_slots=8`: `steady_state_time=0.0022848203s`, `5.87432 TFLOP/s/rank`.
- Interpretation:
  - The source-push inbox plus destination W13 path is correct under Lane lowering.
  - The single M-owner receiver is compute-serialized: increasing slots from 1 to 8 is flat around `0.0072s`.
  - Slot-owned destination compute improves substantially when enough slots are available (`0.00228s` at 8 slots), so the semaphore protocol can safely feed several independent consumers.
  - This is still a correctness/protocol primitive, not a production performance path. It only sends to the next rank and does not yet implement all-peer routing or M x N sharded W13. The current bottleneck is software-structure and tensor-core occupancy from the M-owner compute assignment, not raw remote-copy bandwidth.
  - Next source-push experiment should split compute across N tiles, or otherwise give multiple compute SMs work for the same M block, before spending more time on send-SM or slot-depth tuning.

### 2026-07-01 19:33 PDT - FWD-SWQ-045 Source-push inbox M x N slot-sharded W13
- Hypothesis:
  - The M-owner variants underuse tensor cores because one destination worker computes all N tiles for each M block.
  - Sharding each full inbox slot across N-tile workers should improve occupancy while preserving the bounded source-push protocol:
    - source pushes one M block into a destination slot;
    - source signals the full slot once;
    - each N-tile worker observes the full generation, computes one W13 N tile, and signals done;
    - one release worker waits for all N tiles before returning the slot to the source.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - Added `implementation=m_n_slots`.
  - Refactored W13 tile compute into `_compute_hidden_n_tile`.
  - Initial M x N semaphore design used `full_sem` with `inc=n_tiles` and one decrementing wait per N tile. It was correct for a small `n_tiles=2` smoke but hung when reusing multiple target-size slots.
  - Replaced that with monotonic per-slot counters:
    - source signals `full_sem[src, slot]` once per generation;
    - N-tile workers wait with `decrement=False` on `value=round_i + 1`;
    - each N-tile worker signals `done_sem[slot]`;
    - release worker waits with `decrement=False` on `value=(round_i + 1) * n_tiles` before signaling source `empty_sem[slot]`.
- Local checks:
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 correctness/timing:
  - Small checked smoke before monotonic-counter fix: `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-022353`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: `implementation=m_n_slots`, `entries_per_rank=4`, `inbox_slots=4`, `D=256`, `I=256`, `block_k=64`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation enabled.
    - `steady_state_time=0.0015948184s`.
    - `w13_tflops_per_rank=0.0420793`.
    - `hidden_max_abs_diff=0.0126772`, `hidden_mean_abs_diff=0.00218024`, `metadata_mismatches=0`.
  - First target sweep before monotonic-counter fix: `/dlwh/iris-run-job-20260702-022444`, cluster `cw-us-east-02a`, H100x8, killed after slots=2 hung.
    - `inbox_slots=1`: `steady_state_time=0.0025575220s`, `5.24796 TFLOP/s/rank`.
    - `inbox_slots=2`: no row after about 1m24s total job runtime, killed to avoid burning the reservation.
  - Checked target-dim slots=2 smoke after monotonic-counter fix: `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-022744`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: `implementation=m_n_slots`, `entries_per_rank=4`, `inbox_slots=2`, `D=2560`, `I=1280`, `block_k=128`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation enabled.
    - `steady_state_time=0.0016075835s`.
    - `w13_tflops_per_rank=2.08726`.
    - `hidden_max_abs_diff=0.0143805`, `hidden_mean_abs_diff=0.00180969`, `metadata_mismatches=0`.
  - Fixed target-size slot sweep: `/dlwh/iris-run-job-20260702-022847`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Common config: `implementation=m_n_slots`, `entries_per_rank=16`, `D=2560`, `I=1280`, `block_k=128`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation disabled.
    - `inbox_slots=1`: `steady_state_time=0.0025331167s`, `5.29852 TFLOP/s/rank`.
    - `inbox_slots=2`: `steady_state_time=0.0021586060s`, `6.21780 TFLOP/s/rank`.
    - `inbox_slots=4`: `steady_state_time=0.0020214153s`, `6.63979 TFLOP/s/rank`.
    - `inbox_slots=8`: `steady_state_time=0.0020124843s`, `6.66926 TFLOP/s/rank`.
- Interpretation:
  - M x N slot-sharding is correct with monotonic full/done counters and avoids the earlier multi-slot hang.
  - It improves the best compact source-push primitive row from `m_owner_slots` `0.0022848203s` / `5.87432 TFLOP/s/rank` to `m_n_slots` `0.0020124843s` / `6.66926 TFLOP/s/rank`, about `11.9%` faster in this primitive.
  - The improvement is real but still not enough to call this a production direction. Absolute W13 occupancy remains very low because this compact path only models next-rank traffic, has heavy semaphore/control overhead, and still reloads the same M tile from local inbox for every N tile.
  - The next useful experiments are:
    - tune `num_send_sms` vs compute workers for `m_n_slots`;
    - group multiple N tiles per worker (`n_group`) once shared memory permits, to reduce repeated inbox loads and semaphore fanout;
    - extend from next-rank-only to all-peer routing only if the primitive closes more of the occupancy gap.

### 2026-07-01 19:36 PDT - FWD-SWQ-046 Source-push M x N send/compute SM split sweep
- Hypothesis:
  - `m_n_slots` is compute/control-heavy after N-tile sharding. Reducing send workers may help by leaving more compute workers, but too few send workers may underfeed the inbox.
  - Sweep `num_send_sms` at the two best slot depths from FWD-SWQ-045.
- H100 result:
  - Job `/dlwh/iris-run-job-20260702-023131`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `entries_per_rank=16`, `D=2560`, `I=1280`, `block_k=128`, `block_n=128`, `num_sms=16`, validation disabled.
  - `inbox_slots=4`:
    - `num_send_sms=1`: `steady_state_time=0.0022352966s`, `6.00447 TFLOP/s/rank`.
    - `num_send_sms=2`: `steady_state_time=0.0021871936s`, `6.13653 TFLOP/s/rank`.
    - `num_send_sms=4`: `steady_state_time=0.0021312877s`, `6.29749 TFLOP/s/rank`.
    - `num_send_sms=8`: `steady_state_time=0.0022724379s`, `5.90633 TFLOP/s/rank`.
  - `inbox_slots=8`:
    - `num_send_sms=1`: `steady_state_time=0.0019796023s`, `6.78003 TFLOP/s/rank`.
    - `num_send_sms=2`: `steady_state_time=0.0021450970s`, `6.25695 TFLOP/s/rank`.
    - `num_send_sms=4`: `steady_state_time=0.0020981620s`, `6.39692 TFLOP/s/rank`.
    - `num_send_sms=8`: `steady_state_time=0.0024364567s`, `5.50873 TFLOP/s/rank`.
- Interpretation:
  - Best compact source-push primitive row is now `inbox_slots=8,num_send_sms=1`: `0.0019796023s`, `6.78003 TFLOP/s/rank`.
  - That is only `1.6%` faster than the previous best `inbox_slots=8,num_send_sms=4` row from FWD-SWQ-045 (`0.0020124843s`) and about `13.4%` faster than `m_owner_slots` best (`0.0022848203s`).
  - The optimal send/compute split depends on buffering: with `inbox_slots=4`, one send worker underfeeds and four send workers win; with `inbox_slots=8`, one send worker wins because the extra compute workers matter more.
  - This remains software-structure-bound. The compact primitive is still far below a healthy W13 roofline, and this sweep does not justify integrating the source-push path yet.
  - Next highest-signal patch is N grouping (`n_group > 1`) so each worker reuses one inbox M tile for multiple adjacent N tiles, reducing semaphore fanout and repeated local inbox loads. The current script still rejects `n_group != 1`.

### 2026-07-01 19:41 PDT - FWD-SWQ-047 Source-push M x N `n_group=2` negative result
- Hypothesis:
  - In `m_n_slots`, each N tile currently reloads the same inbox M block from local GMEM. Grouping two adjacent N tiles per worker should reuse that LHS SMEM tile inside the K loop and halve the full/done semaphore fanout.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - Added `n_group=2` support for `implementation=m_n_slots`.
  - The grouped path computes two adjacent N tiles with clear accumulator names:
    - `gate_n0_acc`, `up_n0_acc`;
    - `gate_n1_acc`, `up_n1_acc`.
  - The grouped path uses one LHS inbox SMEM load and four RHS SMEM loads per K tile, then issues four WGMMA operations before storing the two hidden N tiles.
  - Validation still rejects `n_group > 1` for the other inbox implementations.
- Local checks:
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 correctness:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-023651`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Config: `implementation=m_n_slots`, `n_group=2`, `entries_per_rank=4`, `inbox_slots=2`, `D=2560`, `I=1280`, `block_k=128`, `block_n=128`, `num_send_sms=1`, `num_sms=16`, validation enabled.
  - `steady_state_time=0.0017425114s`.
  - `w13_tflops_per_rank=1.92564`.
  - `hidden_max_abs_diff=0.0143805`, `hidden_mean_abs_diff=0.00180969`, `metadata_mismatches=0`.
- H100 timing:
  - Job `/dlwh/iris-run-job-20260702-023749`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `n_group=2`, `entries_per_rank=16`, `D=2560`, `I=1280`, `block_k=128`, `block_n=128`, `num_sms=16`, validation disabled.
  - `inbox_slots=4,num_send_sms=1`: `steady_state_time=0.0023088150s`, `5.81327 TFLOP/s/rank`.
  - `inbox_slots=4,num_send_sms=4`: `steady_state_time=0.0072954780s`, `1.83974 TFLOP/s/rank`.
  - `inbox_slots=8,num_send_sms=1`: `steady_state_time=0.0023554047s`, `5.69829 TFLOP/s/rank`.
  - `inbox_slots=8,num_send_sms=4`: `steady_state_time=0.0055771460s`, `2.40657 TFLOP/s/rank`.
- Interpretation:
  - `n_group=2` is correct, but it is slower than `n_group=1` for every tested comparable row.
  - Best `n_group=2` row is `0.0023088150s`, worse than the current `n_group=1` best `0.0019796023s`.
  - The likely cause is extra accumulator/register/SMEM pressure and longer per-worker critical sections. The saved LHS load and reduced semaphore fanout do not compensate.
  - Do not pursue larger `n_group` in this source-push primitive without a different compute layout. At `n_group=4`, RHS SMEM would also exceed the comfortable shared-memory budget for this shape.
  - Current best remains `m_n_slots`, `n_group=1`, `inbox_slots=8`, `num_send_sms=1`: `0.0019796023s`, `6.78003 TFLOP/s/rank`.

### 2026-07-01 20:03 PDT - FWD-SWQ-048 Source-push `lax.switch` peer-loop smoke
- Hypothesis:
  - `pl.when` lowers through `jax.lax.cond`, so the source-push peer loop might avoid a fully duplicated Python peer loop by using `jax.lax.switch` over a dynamic peer ordinal, with each branch closing over a static peer offset.
  - This is different from the earlier destination-pull `lax.switch` failure in FWD-SWQ-039 because the remote operation here is `copy_smem_to_gmem(local_smem, remote_gmem)` under lane lowering.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py`.
    - Added `--selection={direct,pl_when,switch,all}`.
    - The `pl_when` and `switch` modes run a two-phase loop: local write, then remote write.
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `PushInboxConfig.peer_loop`.
    - Added CLI `--peer-loop={static,switch}`.
    - `peer_loop=switch` uses a dynamic `pl.loop` over peer ordinals and `lax.switch` branches that close over static peer offsets for send, initial empty-slot signaling, and receive/release loops.
- Local checks:
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py`
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 primitive result:
  - Job `/dlwh/iris-run-repro_smem_to_remote_gmem_peer-20260702-025104`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - `direct Warpgroup`: expected `NotImplementedError: GMEM refs with peer ids are not supported in warpgroup lowering.`
  - `direct Lane`: success, `out[:, 0, 0] = [2 1]`.
  - `pl_when Warpgroup`: expected same peer-id warpgroup failure.
  - `pl_when Lane`: success, `out[:, 0, 0] = [2 1]`.
  - `switch Warpgroup`: expected same peer-id warpgroup failure.
  - `switch Lane`: success, `out[:, 0, 0] = [2 1]`.
- H100 inbox small all-to-all result:
  - Job `/dlwh/iris-run-job-20260702-025332`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `EP=8`, `entries_per_rank=2`, `inbox_slots=2`, `D=256`, `I=256`, `block_m=64`, `block_k=64`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation enabled.
  - `peer_loop=static`: `compile_time=6.95185s`, `steady_state_time=0.00164852s`, `0.162834 TFLOP/s/rank`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.00695372`.
  - `peer_loop=switch`: `compile_time=7.31261s`, `steady_state_time=0.00163688s`, `0.163992 TFLOP/s/rank`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.00695372`.
- H100 target-dim follow-up:
  - Job `/dlwh/iris-run-job-20260702-025454`, cluster `cw-us-east-02a`, H100x8, failed by command timeout.
  - Config: same source-push `m_n_slots` all-to-all comparison, `entries_per_rank=4`, `D=2560`, `I=1280`, `block_k=128`, validation enabled.
  - Result: no JSON rows emitted before timeout.
  - Iris summary: task exit code `124`, duration `1208443 ms`, state `failed`.
- Interpretation:
  - `lax.switch` is viable for the source-push lane-lowering path.
  - It does not bypass the warpgroup peer-id limitation; all warpgroup variants still fail at the expected remote-GMEM ref lowering point.
  - On the small inbox all-to-all smoke, `switch` is correctness-neutral and runtime-neutral; compile time was slightly worse (`7.31s` vs `6.95s`).
  - Current bottleneck judgment is still software-structure-bound, not compute-bound.
  - The target-dim all-to-all `m_n_slots` graph is now too large/slow to compile with the current structure, even at `entries_per_rank=4`. The switch form is useful as a control-flow option for source-push lane lowering, but it does not solve the large-graph problem by itself.

### 2026-07-01 20:55 PDT - FWD-SWQ-049 Grid peer phase plus `lax.switch`
- Question:
  - Can we avoid fully unrolling the all-peer source-push loop while still giving Mosaic static peer-id branches?
  - `pl.when` lowers through `jax.lax.cond`; test `jax.lax.switch` with the peer ordinal supplied by a grid dimension instead of a `pl.loop` carried value.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `peer_loop=dynamic`, `peer_loop=grid`, and `peer_loop=grid_switch`.
    - `grid` maps `peer_ordinal = pl.program_id(0)` and `sm = pl.program_id(1)`, but still uses dynamic peer arithmetic.
    - `grid_switch` maps peer ordinal the same way, then calls `lax.switch(peer_ordinal, static_peer_branches)` for send, empty-slot init, and recv/release.
    - `send_only` now uses a dummy `(ep_size, 1, 1, 1)` W input and hidden output shape `(1, 1)`, so send-only timing no longer allocates unused multi-GB W13 weights.
  - Added/updated peer-id repros:
    - `lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py` now has `--ep-size` and dynamic nested-loop selections.
    - `lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer_grid.py` tests source-push peer phase as a grid dimension.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer_grid.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer_grid.py`
- Negative H100 results:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-033657`: small all-to-all `send_only`, `peer_loop=dynamic`, failed with `ValueError: Failed to recompute the async_copy peer id on the host`; root was a loop-carried block argument.
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-034619`: first `peer_loop=grid` layout failed with peer-id recompute on `gpu.block_id x`.
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-034954`: after swapping grid order so peer phase is `program_id(0)`, `peer_loop=grid` still failed with peer-id recompute on `gpu.block_id y`.
  - Target-D `send_only` `static`/`switch` attempts before the dummy-W/grid-switch fixes produced no row before being killed or timed out:
    - `/dlwh/iris-run-job-20260702-031957`, killed after `227052 ms`.
    - `/dlwh/iris-run-job-20260702-032452`, killed after `212320 ms`.
    - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-032844`, killed after `339106 ms`.
- Positive H100 repro results:
  - `/dlwh/iris-run-repro_smem_to_remote_gmem_peer-20260702-033437`: minimal dynamic-loop EP=2 source-push lane repro succeeded; warpgroup failed with expected peer-id lowering limitation.
  - `/dlwh/iris-run-job-20260702-034124`: minimal dynamic/nested EP=8 source-push lane repros succeeded, all producing `out[:, 0, 0] = [8 1 2 3 4 5 6 7]`; warpgroup failed as expected.
  - `/dlwh/iris-run-repro_smem_to_remote_gmem_peer_grid-20260702-034405`: minimal grid peer-phase lane repro succeeded for EP=8; warpgroup failed as expected.
- Positive H100 inbox results:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-035138`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: `implementation=send_only`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `EP=8`, `entries_per_rank=2`, `inbox_slots=2`, `D=256`, `I=256`, `block_m=64`, `block_k=64`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation enabled.
    - `compile_time=2.6732348402s`, `steady_state_time=0.0015680999s`, `bytes_per_rank=524288`, `send_gbps_per_rank=0.334346`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-035238`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: target-D `send_only`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `EP=8`, `entries_per_rank=4`, `inbox_slots=2`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation enabled.
    - `compile_time=3.0009198510s`, `steady_state_time=0.0015431187s`, `bytes_per_rank=10485760`, `send_gbps_per_rank=6.79517`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-035336`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: small WGMMA `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `EP=8`, `entries_per_rank=2`, `inbox_slots=2`, `D=256`, `I=256`, `block_m=64`, `block_k=64`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation enabled.
    - `compile_time=7.0569546900s`, `steady_state_time=0.0015161049s`, `w13_tflops_per_rank=0.177056`, `hidden_max_abs_diff=0.00695372`, `hidden_mean_abs_diff=0.00106579`, `metadata_mismatches=0`.
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-035549`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: target-D WGMMA `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `EP=8`, `entries_per_rank=4`, `inbox_slots=2`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation enabled.
    - `compile_time=5.6102150199s`, `steady_state_time=0.0015892386s`, `w13_tflops_per_rank=16.8908`, `send_gbps_per_rank=6.59798`, `hidden_max_abs_diff=0.0143805`, `hidden_mean_abs_diff=0.00192177`, `metadata_mismatches=0`.
- Tuning follow-up:
  - `/dlwh/iris-run-job-20260702-035727`, cluster `cw-us-east-02a`, H100x8, killed after one useful row because the next config produced no row.
    - Sweep intent: `inbox_slots={1,2,4,8}` x `num_send_sms={1,2,4,8}`, target-D all-to-all `m_n_slots grid_switch`, validation disabled.
    - Completed row: `inbox_slots=1,num_send_sms=1`: `compile_time=5.4421144261s`, `steady_state_time=0.0017394261s`, `w13_tflops_per_rank=15.4324`.
    - The next row, `inbox_slots=1,num_send_sms=2`, emitted no row before manual kill at about `92s` job duration.
  - `/dlwh/iris-run-job-20260702-035921`, cluster `cw-us-east-02a`, H100x8, killed after the first focused config emitted no row.
    - Focused intent: `(inbox_slots,num_send_sms)=(2,1),(2,4),(4,1),(4,4),(8,1)`, target-D all-to-all `m_n_slots grid_switch`, validation disabled.
    - The first row, `inbox_slots=2,num_send_sms=1`, emitted no row before manual kill at about `50s` job duration.
  - `/dlwh/iris-run-job-20260702-040031`, cluster `cw-us-east-02a`, was submitted for a narrower `num_send_sms=4` slot sweep but remained pending, so it was killed before starting.
- Interpretation:
  - `lax.switch` is useful here when the peer ordinal comes from a grid dimension and each branch closes over a static peer offset. This avoids both the loop-carried peer-id recompute failure and the target-D no-row behavior seen with the previous static/switch structures.
  - Plain dynamic peer arithmetic is not enough, even with peer phase as `program_id(0)`: the combined send/recv kernel still makes async-copy host peer-id recompute see an unsupported block id.
  - Lane lowering is compatible with the source-push WGMMA path in `grid_switch`; the small `m_n_slots` checked run validates matmul correctness.
  - This is still a source-push protocol/harness result, not yet a production permute_up speedup. The current bottleneck remains software-structure-bound: the protocol can compile and run, but the WGMMA primitive is far below useful tensor-core occupancy, and send-only bandwidth at this small message count is not competitive with the production tiled path.

### 2026-07-01 21:15 PDT - FWD-SWQ-050 Native grid-switch sweep and compile/runtime split
- Question:
  - The first grid-switch tuning attempts suggested low `num_send_sms` might hang. Determine whether that was a real semaphore/progress issue or an artifact of the shell-loop harness.
  - Make sweeps native to the Python repro so row generation and progress events are reliable.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `--separate-compile` and `--progress-events`.
    - Rows now include `lower_compile_time` and `first_run_time` when `--separate-compile` is used.
    - Progress events now cover `validate_start`, `mesh_start`, `make_inputs_start`, `device_put_start`, `jit_start`, and timing phases.
    - Added native sweep flags `--sweep-inbox-slots` and `--sweep-num-send-sms`, avoiding fragile shell loops.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 compile/runtime diagnostics:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-040634`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: target-D all-to-all `m_n_slots grid_switch`, `inbox_slots=2`, `num_send_sms=4`, validation disabled, `--separate-compile`, `warmup=0`, `steps=1`.
    - `lower_compile_time=3.8032942459s`, `first_run_time=1.6628572522s`, `compile_time=5.4661514980s`, `steady_state_time=0.0025271629s`.
    - Setup events showed about `16s` before `lower_start`, dominated by mesh/device discovery, synthetic target-D W construction, and `device_put`.
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-040733`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: same, `inbox_slots=2`, `num_send_sms=1`.
    - `lower_compile_time=3.9671898130s`, `first_run_time=1.5979714841s`, `compile_time=5.5651612971s`, `steady_state_time=0.0023255940s`.
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-040833`, cluster `cw-us-east-02a`, H100x8, succeeded.
    - Config: same, `inbox_slots=1`, `num_send_sms=2`.
    - `lower_compile_time=3.8154163749s`, `first_run_time=1.6651320548s`, `compile_time=5.4805484298s`, `steady_state_time=0.0023940830s`.
- H100 native target-D sweep:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-041149`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `EP=8`, `entries_per_rank=4`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `num_sms=16`, validation disabled, `warmup=1`, `steps=3`.

  | inbox_slots | num_send_sms | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|---:|
  | 1 | 1 | `0.0019036843s` | `14.1008` | `5.5081` |
  | 1 | 2 | `0.0017392787s` | `15.4337` | `6.0288` |
  | 1 | 4 | `0.0017727807s` | `15.1421` | `5.9149` |
  | 1 | 8 | `0.0018025277s` | `14.8922` | `5.8173` |
  | 2 | 1 | `0.0017870436s` | `15.0212` | `5.8677` |
  | 2 | 2 | `0.0016089707s` | `16.6837` | `6.5171` |
  | 2 | 4 | `0.0016030597s` | `16.7452` | `6.5411` |
  | 2 | 8 | `0.0017332930s` | `15.4870` | `6.0496` |
  | 4 | 1 | `0.0016862490s` | `15.9191` | `6.2184` |
  | 4 | 2 | `0.0017943053s` | `14.9604` | `5.8439` |
  | 4 | 4 | `0.0016837710s` | `15.9425` | `6.2275` |
  | 4 | 8 | `0.0018952853s` | `14.1633` | `5.5325` |
- Interpretation:
  - Low `num_send_sms` is not a deadlock in the grid-switch path. The earlier no-row observations came from weak shell-loop orchestration and premature kills, not a proved protocol progress failure.
  - There is a real overlap/buffering signal: `inbox_slots=2` improves over `inbox_slots=1` at comparable send-worker counts; best row is `inbox_slots=2,num_send_sms=4` at `0.0016030597s`, close to the prior checked `0.0015892386s` row.
  - More buffering is not monotonically better: `inbox_slots=4` is comparable but does not beat the best `inbox_slots=2` row.
  - Too many send workers hurts: `num_send_sms=8` regresses at every tested slot count.
  - Current bottleneck remains software-structure-bound / low tensor-core occupancy. The source-push grid-switch prototype answers the spec's overlap question positively in a narrow sense, but at only about `16.7 TFLOP/s/rank` it is still far from a production performance path.

### 2026-07-01 21:18 PDT - FWD-SWQ-051 Grid-switch `block_k=64` targeted sweep
- Question:
  - The source-push spec calls out `block_k=64` and `block_k=128` as initial candidates.
  - Test whether smaller send/compute K tiles improve the best `grid_switch` target-D region.
- H100 result:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-041553`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `EP=8`, `entries_per_rank=4`, `D=2560`, `I=1280`, `block_m=64`, `block_k=64`, `block_n=128`, `n_group=1`, `num_sms=16`, validation disabled, `warmup=1`, `steps=3`.

  | inbox_slots | num_send_sms | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|---:|
  | 2 | 2 | `0.0017569417s` | `15.2786` | `5.9682` |
  | 2 | 4 | `0.0018279493s` | `14.6851` | `5.7364` |
  | 4 | 2 | `0.0017310370s` | `15.5072` | `6.0575` |
  | 4 | 4 | `0.0017507356s` | `15.3327` | `5.9893` |
- Interpretation:
  - `block_k=64` is slower than `block_k=128` in the best target-D grid-switch region.
  - Best tested `block_k=64` row is `inbox_slots=4,num_send_sms=2`: `0.0017310370s`, worse than `block_k=128` best `inbox_slots=2,num_send_sms=4`: `0.0016030597s`.
  - Keep `block_k=128` for this prototype unless a later local K-pipeline rewrite changes the balance.

### 2026-07-01 21:30 PDT - FWD-SWQ-052 `lax.switch` peer dispatch plus local K `emit_pipeline` test
- Question:
  - User suggested replacing a hand-written `pl.when`/`cond` peer selection ladder with `jax.lax.switch`.
  - Test whether the source-push `grid_switch` peer dispatch can remain the static-peer mechanism while swapping the compute-side manual K loop for `mgpu.emit_pipeline`.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `COMPUTE_PIPELINE_MODES = ("manual", "emit")`.
    - Added `PushInboxConfig.compute_pipeline` and `PushInboxConfig.max_concurrent_steps`.
    - Added CLI flags `--compute-pipeline`, `--max-concurrent-steps`, and `--sweep-max-concurrent-steps`.
    - Factored lane WGMMA shared-memory transforms through `_wgmma_transforms(...)`, reused by both manual SMEM refs and `mgpu.BlockSpec(..., transforms=...)`.
    - `compute_pipeline=emit` uses `mgpu.emit_pipeline` across K tiles for the `n_group=1` W13 tile path, while preserving `grid_switch` static peer dispatch.
    - `compute_pipeline=emit` currently rejects `n_group != 1`; the grouped two-N-tile path still uses the manual loop.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 correctness smoke:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-042319`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `compute_pipeline=emit`, `max_concurrent_steps=4`, `EP=8`, `entries_per_rank=2`, `inbox_slots=2`, `D=256`, `I=256`, `block_m=64`, `block_k=64`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, validation enabled.
  - `lower_compile_time=5.8907892320s`, `first_run_time=1.8379488070s`, `compile_time=7.7287380390s`.
  - `steady_state_time=0.0017448770s`, `w13_tflops_per_rank=0.153842`, `send_gbps_per_rank=0.300473`.
  - `hidden_max_abs_diff=0.0069537163`, `hidden_mean_abs_diff=0.0010657860`, `metadata_mismatches=0`.
- H100 target-D `block_k=128` emit sweep:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-042459`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `compute_pipeline=emit`, `EP=8`, `entries_per_rank=4`, `inbox_slots=2`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `num_send_sms=4`, `num_sms=16`, validation disabled, `warmup=1`, `steps=3`.

  | max_concurrent_steps | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | outcome |
  |---:|---:|---:|---:|---|
  | 1 | n/a | n/a | n/a | invalid: `max_concurrent_steps` must exceed `delay_release=1` |
  | 2 | `0.0019022586s` | `14.1114` | `5.5123` | runs, slower than manual |
  | 4 | n/a | n/a | n/a | SMEM overflow: `327712 > 232448` bytes |
  | 6 | n/a | n/a | n/a | SMEM overflow: `491568 > 232448` bytes |

- H100 target-D `block_k=64` emit sweep:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-042714`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: same as above except `block_k=64`.

  | max_concurrent_steps | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|
  | 2 | `0.0017692650s` | `15.1721` | `5.9266` |
  | 3 | `0.0017556247s` | `15.2900` | `5.9727` |
  | 4 | `0.0018021560s` | `14.8952` | `5.8185` |
  | 5 | `0.0020414394s` | `13.1493` | `5.1365` |

- Interpretation:
  - `lax.switch` remains the right way to express static peer branches when the peer ordinal comes from the grid. It traces all branches, but each branch closes over static `src_offset`/`dst_offset`, so it avoids the dynamic peer-id recompute failure.
  - `mgpu.emit_pipeline` is correctness-compatible with lane-lowered source-push WGMMA in this harness.
  - The local K `emit_pipeline` variant is not a speedup at target shape. With `block_k=128`, only `max_concurrent_steps=2` fits and it is slower than the manual best (`0.0019022586s` vs `0.0016030597s`). Deeper staging would need more SMEM than Mosaic allows in this combined send/recv/compute kernel.
  - With `block_k=64`, deeper staging fits but remains slower than manual `block_k=128`; best emit row is `0.0017556247s`.
  - Bottleneck judgment remains software-structure-bound / occupancy-bound. The source-push protocol works and has a narrow buffering signal, but this simple local-K pipeline does not fix the low TFLOP/s path.

### 2026-07-01 21:36 PDT - FWD-SWQ-053 Source-push target-D `num_sms` sweep
- Question:
  - The target-D `grid_switch` source-push path had only been swept at `num_sms=16`.
  - Test whether increasing the total programs per peer phase improves compute occupancy for the manual K-loop path.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added CLI flag `--sweep-num-sms`.
    - Native sweep loop now nests `inbox_slots`, `num_send_sms`, `num_sms`, and `max_concurrent_steps`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 target-D `num_sms` sweep:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-043121`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `compute_pipeline=manual`, `EP=8`, `entries_per_rank=4`, `inbox_slots=2`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `num_send_sms=4`, validation disabled, `warmup=1`, `steps=3`.

  | num_sms | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|
  | 16 | `0.0015406663s` | `17.4233` | `6.8060` |
  | 32 | `0.0016165933s` | `16.6050` | `6.4863` |
  | 64 | `0.0015991673s` | `16.7860` | `6.5570` |
  | 128 | `0.0016346443s` | `16.4216` | `6.4147` |

- H100 target-D best-row repeat:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-043440`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Config: same as above with `num_sms=16`, validation disabled, `warmup=3`, `steps=10`.
  - `lower_compile_time=3.4722816400s`, `first_run_time=1.5514973670s`, `compile_time=5.0237790070s`.
  - `steady_state_time=0.0015381217s`, `w13_tflops_per_rank=17.4522`, `send_gbps_per_rank=6.8172`.
- Interpretation:
  - Increasing `num_sms` does not improve the target-D source-push path. Best remains `num_sms=16`; larger grids add overhead or dilute useful work.
  - The previous best target-D source-push row was `0.0016030597s`; a longer repeat of the same best-shape config now measures `0.0015381217s`, about 4.0% faster. This is a benchmark measurement update, not a production integrated speedup.
  - Bottleneck remains software-structure/occupancy-bound. More independent programs do not unlock higher tensor-core utilization for this combined source-push kernel.

### 2026-07-01 21:52 PDT - FWD-SWQ-054 Full-scale source-push scaling and `lax.switch` peer dispatch
- Question:
  - User noted that `pl.when` lowers to `jax.lax.cond`, so peer selection can use `jax.lax.switch` instead of manually writing an unrolled conditional ladder.
  - Test the `grid_switch` source-push path at a realistic balanced proxy: `entries_per_rank=256`, corresponding to `32` local experts times `512` rows/expert divided by `block_m=64`.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `--sweep-entries-per-rank`.
    - Vectorized `_make_inputs` host construction for target-scale `entries_per_rank=256`, avoiding Python loops over rank/entry/expert.
    - Continued using `peer_loop=grid_switch`: one grid axis carries `peer_phase`, and `jax.lax.switch` selects a branch whose body closes over static peer offsets.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 correctness smoke:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-044100`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Config: small checked all-to-all `m_n_slots grid_switch`, `D=256`, `I=256`, `entries_per_rank=4`, `block_k=64`, `inbox_slots=2`, `num_send_sms=4`, `num_sms=16`.
  - `steady_state_time=0.0014935450s`, `hidden_max_abs_diff=0.0289773941`, `hidden_mean_abs_diff=0.0022519461`, `metadata_mismatches=0`.
- H100 entries scaling:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-043917`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `compute_pipeline=manual`, `EP=8`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `inbox_slots=2`, `num_send_sms=4`, `num_sms=16`, validation disabled, `warmup=1`, `steps=3`.

  | entries_per_rank | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | bytes/rank |
  |---:|---:|---:|---:|---:|
  | 4 | `0.0016284170s` | `16.4844` | `6.4392` | `10485760` |
  | 64 | `0.0038493923s` | `111.5752` | `43.5841` | `167772160` |
  | 256 | `0.0108622830s` | `158.1608` | `61.7815` | `671088640` |

- H100 send-only full-scale tax:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-044220`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Config: `implementation=send_only`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `entries_per_rank=256`, `D=2560`, `block_m=64`, `block_k=128`, `inbox_slots=2`, `num_send_sms=4`, `num_sms=16`, validation enabled.
  - `steady_state_time=0.0048350357s`, `send_gbps_per_rank=138.7970`, `bytes_per_rank=671088640`, `max_abs_diff=0.0`, `metadata_mismatches=0`.
- H100 full-scale overlap sweep:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-044327`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `compute_pipeline=manual`, `EP=8`, `entries_per_rank=256`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `num_sms=16`, validation disabled, `warmup=1`, `steps=3`.

  | inbox_slots | num_send_sms | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|---:|
  | 1 | 1 | `0.0175822653s` | `97.7114` | `38.1685` |
  | 1 | 2 | `0.0174991904s` | `98.1752` | `38.3497` |
  | 1 | 4 | `0.0176462460s` | `97.3571` | `38.0301` |
  | 1 | 8 | `0.0264773947s` | `64.8850` | `25.3457` |
  | 2 | 1 | `0.0109726423s` | `156.5700` | `61.1602` |
  | 2 | 2 | `0.0110186857s` | `155.9158` | `60.9046` |
  | 2 | 4 | `0.0110242110s` | `155.8376` | `60.8741` |
  | 2 | 8 | `0.0152888033s` | `112.3690` | `43.8941` |
  | 4 | 1 | `0.0103051037s` | `166.7122` | `65.1220` |
  | 4 | 2 | `0.0133965090s` | `128.2414` | `50.0943` |
  | 4 | 4 | `0.0118276817s` | `145.2514` | `56.7388` |
  | 4 | 8 | `0.0164114740s` | `104.6821` | `40.8914` |

- Interpretation:
  - `jax.lax.switch` works with lane lowering and the WGMMA matmul path here. It removes source-level hand-unrolling, but all branches are still traced into the compiled switch; runtime selects one branch. This is a good static-peer expression, not a compile-size miracle.
  - Full-scale performance is much better than the tiny `entries_per_rank=4` proxy: best full-scale row is `0.0103051037s` and `166.7 TFLOP/s/rank` for W13/up-half source-push harness work.
  - There is a real slot-depth/overlap signal. `inbox_slots=1` is about `17.5ms`; `inbox_slots=2` drops to about `11.0ms`; best `inbox_slots=4,num_send_sms=1` is `10.3ms`.
  - More send workers are not better. `num_send_sms=8` regresses for every tested slot count.
  - Send-only remote write/semaphore cost at the same full-scale traffic is `4.835ms`; the full W13 path is not just copy-bound, but the exchange is still a large part of the budget.
  - Bottleneck judgment: communication plus software structure, with meaningful but incomplete overlap. This source-push path is more credible at realistic queue depth than the earlier tiny proxy, but it remains a W13-only harness result and not an integrated production forward speedup.

### 2026-07-01 21:59 PDT - FWD-SWQ-055 Full-scale `n_group=2` source-push test
- Question:
  - The source-push spec suggests trying `n_group=2` so each compute job reuses the same inbox LHS tile for two adjacent N tiles.
  - Test whether this improves the full-scale W13/up-half source-push harness.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `--sweep-n-groups` so `n_group=1,2` rows can be produced by one reproducible Python sweep.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 job:
  - Job `/dlwh/iris-run-job-20260702-045326`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Command structure: one checked small `n_group=2` smoke, then a full-scale `n_group=1,2` sweep.
- H100 correctness smoke:
  - Config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `compute_pipeline=manual`, `EP=8`, `entries_per_rank=4`, `inbox_slots=2`, `D=256`, `I=256`, `block_m=64`, `block_k=64`, `block_n=128`, `n_group=2`, `num_send_sms=4`, `num_sms=16`, validation enabled.
  - `steady_state_time=0.0015734613s`, `hidden_max_abs_diff=0.0289773941`, `hidden_mean_abs_diff=0.0022519461`, `metadata_mismatches=0`.
- H100 full-scale sweep:
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `compute_pipeline=manual`, `EP=8`, `entries_per_rank=256`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `block_n=128`, `num_sms=16`, validation disabled, `warmup=1`, `steps=3`.

  | inbox_slots | num_send_sms | n_group | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|---:|---:|
  | 1 | 1 | 1 | `0.0175520747s` | `97.8794` | `38.2341` |
  | 1 | 1 | 2 | `0.0349167467s` | `49.2024` | `19.2197` |
  | 1 | 4 | 1 | `0.0177069133s` | `97.0235` | `37.8998` |
  | 1 | 4 | 2 | `0.0351557560s` | `48.8679` | `19.0890` |
  | 2 | 1 | 1 | `0.0109774067s` | `156.5021` | `61.1336` |
  | 2 | 1 | 2 | `0.0214171403s` | `80.2155` | `31.3342` |
  | 2 | 4 | 1 | `0.0109635550s` | `156.6998` | `61.2109` |
  | 2 | 4 | 2 | `0.0194689583s` | `88.2424` | `34.4697` |
  | 4 | 1 | 1 | `0.0103007983s` | `166.7819` | `65.1492` |
  | 4 | 1 | 2 | `0.0236580810s` | `72.6173` | `28.3661` |
  | 4 | 4 | 1 | `0.0118166576s` | `145.3869` | `56.7917` |
  | 4 | 4 | 2 | `0.0157361350s` | `109.1746` | `42.6463` |

- Interpretation:
  - `n_group=2` is correct in the checked small case but is consistently slower at the full target shape with `block_n=128`.
  - Best row remains `n_group=1,inbox_slots=4,num_send_sms=1`: `0.0103007983s`, matching the prior best within noise.
  - `n_group=2` likely pays too much SMEM/register/scheduling pressure: at `block_n=128`, it needs one LHS tile plus four RHS tiles and four accumulators per compute job. It also halves the number of N work groups per ready slot, which may reduce scheduling flexibility.
  - Next useful test: `block_n=64,n_group=2`, which keeps the four RHS tiles closer to the SMEM footprint of the current `block_n=128,n_group=1` path while still testing LHS reuse.

### 2026-07-01 22:06 PDT - FWD-SWQ-056 `block_n=64` follow-up for grouped N tiles
- Question:
  - FWD-SWQ-055 showed `n_group=2` is much slower at `block_n=128`.
  - Test whether the slowdown is specifically from the large grouped RHS/accumulator footprint by trying `block_n=64,n_group=2`, which keeps the four RHS tiles closer to the SMEM footprint of `block_n=128,n_group=1`.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `--sweep-block-n`.
    - Added `--sweep-block-k`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 job:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-045936`, cluster `cw-us-east-02a`, H100x8, succeeded.
  - Common config: `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `compute_pipeline=manual`, `EP=8`, `entries_per_rank=256`, `D=2560`, `I=1280`, `block_m=64`, `block_k=128`, `num_sms=16`, validation disabled, `warmup=1`, `steps=3`.

  | inbox_slots | num_send_sms | block_n | n_group | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|---:|---:|---:|
  | 2 | 1 | 64 | 1 | `0.0120518717s` | `142.5494` | `55.6834` |
  | 2 | 1 | 64 | 2 | `0.0116488006s` | `147.4819` | `57.6101` |
  | 2 | 4 | 64 | 1 | `0.0154716453s` | `111.0410` | `43.3754` |
  | 2 | 4 | 64 | 2 | `0.0116478840s` | `147.4935` | `57.6146` |
  | 2 | 1 | 128 | 1 | `0.0109705330s` | `156.6001` | `61.1719` |
  | 2 | 1 | 128 | 2 | `0.0207069037s` | `82.9669` | `32.4089` |
  | 2 | 4 | 128 | 1 | `0.0109399877s` | `157.0374` | `61.3427` |
  | 2 | 4 | 128 | 2 | `0.0192205400s` | `89.3829` | `34.9152` |
  | 4 | 1 | 64 | 1 | `0.0163424597s` | `105.1241` | `41.0641` |
  | 4 | 1 | 64 | 2 | `0.0111712187s` | `153.7869` | `60.0730` |
  | 4 | 4 | 64 | 1 | `0.0141456363s` | `121.4500` | `47.4414` |
  | 4 | 4 | 64 | 2 | `0.0124490807s` | `138.0011` | `53.9067` |
  | 4 | 1 | 128 | 1 | `0.0105058403s` | `163.5268` | `63.8777` |
  | 4 | 1 | 128 | 2 | `0.0233965400s` | `73.4291` | `28.6832` |
  | 4 | 4 | 128 | 1 | `0.0117593630s` | `146.0952` | `57.0685` |
  | 4 | 4 | 128 | 2 | `0.0157999107s` | `108.7340` | `42.4742` |

- Interpretation:
  - `block_n=64,n_group=2` partially rescues grouped N tiles: it beats `block_n=64,n_group=1` in 3 of 4 matched rows, and the best grouped-64 row is `0.0111712187s`.
  - It still does not beat the current best `block_n=128,n_group=1` row: `0.0105058403s` in this job and `0.0103007983s` in FWD-SWQ-055.
  - The `block_n=128,n_group=2` regression is therefore at least partly SMEM/register-pressure related, not just a semantic bug in grouped-N scheduling.
  - Decision: keep `block_n=128,n_group=1` as the best source-push W13 harness configuration for now. Stop spending blind tuning on `n_group=2` unless a later structural change reduces grouped-job pressure.

### 2026-07-01 22:32 PDT - FWD-SWQ-057 Routing-derived source-push queues and counters
- Question:
  - Replace the synthetic rectangular source-push queue with routing-derived live send/recv entries before doing more tuning.
  - Validate zeros, tails, ragged source/expert counts, multiple sources to one expert, and many-source/many-expert traffic.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `--queue-mode {rectangular,routing}` and `--routing {balanced,uniform,tail_debug,one_source_one_expert,many_sources_one_expert,many_sources_many_experts}`.
    - Added routing metadata inputs for send and recv queues, with live-entry skipping before semaphore waits.
    - Added queue counters: send/recv/live entries, skipped zero entries, tail entries/fraction, source/destination/expert imbalance, slot wait counts, and max slot reuse depth.
    - Kept the source-push protocol: source local GMEM -> SMEM -> destination inbox GMEM, then signal full; destination waits full, computes, and releases empty.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 validation jobs:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-051328`, cluster `cw-us-east-02a`, succeeded. Checked small `tail_debug` routing smoke: `steady_state_time=0.0018682873s`, `hidden_max_abs_diff=0.0308904648`, `metadata_mismatches=0`, `send_entries_total=511`, `tail_entries_total=511`, `dropped_entries_total=0`.
  - `/dlwh/iris-run-job-20260702-052247`, cluster `cw-us-east-02a`, succeeded. Checked routing matrix:
    - `tail_debug`: `send_entries=511`, `tails=511`, `zero_entries_skipped=1025`, `hidden_max_abs_diff=0.0289773941`, `metadata_mismatches=0`.
    - `one_source_one_expert`: `send_entries=2`, `tails=1`, `zero_entries_skipped=1534`, `hidden_max_abs_diff=0.015625`, `metadata_mismatches=0`.
    - `many_sources_one_expert`: `send_entries=16`, `tails=8`, `zero_entries_skipped=1520`, `hidden_max_abs_diff=0.015625`, `metadata_mismatches=0`.
    - `many_sources_many_experts`: `send_entries=410`, `tails=240`, `zero_entries_skipped=1126`, `hidden_max_abs_diff=0.0289840698`, `metadata_mismatches=0`.
    - `uniform`: `send_entries=751`, `tails=477`, `zero_entries_skipped=785`, `hidden_max_abs_diff=0.0304203033`, `metadata_mismatches=0`.
- H100 benchmark jobs:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-051838`, cluster `cw-us-east-02a`, succeeded. Routing uniform target, `entries_per_rank=288`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `num_sms=16`:

  | inbox_slots | num_send_sms | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|---:|
  | 2 | 1 | `0.0198999327s` | `91.5320` | `35.7547` |
  | 2 | 4 | `0.0130559287s` | `139.5137` | `54.4976` |
  | 4 | 1 | `0.0107049743s` | `170.1528` | `66.4659` |
  | 4 | 4 | `0.0123006187s` | `148.0805` | `57.8439` |

  - `/dlwh/iris-run-job-20260702-052615`, cluster `cw-us-east-02a`, succeeded. A/B/C benchmark with current best synthetic config (`block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `inbox_slots=4`, `num_send_sms=1`, `num_sms=16`):

  | case | queue | entries_per_rank | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | tail_fraction |
  |---|---|---:|---:|---:|---:|---:|
  | A | rectangular balanced | 256 | `0.0112180463s` | `153.1449` | `59.8222` | `0.0` |
  | B | routing balanced | 256 | `0.0097151733s` | `176.8354` | `69.0763` | `0.0` |
  | C | routing uniform target | 288 | `0.0106858497s` | `170.4573` | `66.5849` | `0.1160554948` |

- Interpretation:
  - Routing-derived queues are correct across zero-count, tail, one-source/one-expert, many-source/one-expert, many-source/many-expert, and random ragged cases.
  - The routing queue preserves the overlap signal: `inbox_slots=4` is materially faster than `inbox_slots=2`; `num_send_sms=1` remains best at `inbox_slots=4`.
  - Matched balanced routing is faster than the rectangular baseline in this harness, so the rectangular queue was not an optimistic bound.
  - Full uniform routing target is close to the best synthetic source-push W13 harness row: `0.0106858497s`, `170.46 TFLOP/s/rank`, no dropped entries.
  - Bottleneck judgment remains communication plus software structure, but the routing-derived source-push path is now a credible forward-permute-up experiment path instead of only a synthetic correctness proof.

### 2026-07-02 00:00 PDT - FWD-SWQ-058 Static recv metadata and direct-self compute
- Question:
  - Test whether moving slot metadata out of the source-push kernel, and bypassing inbox traffic for self-rank entries, removes enough software/metadata tax to move the source-push inbox harness toward the 200 TFLOP/s/rank target.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `--metadata-mode {remote_slot,static_recv}`.
    - Added `--direct-self-compute`.
    - For `metadata_mode=static_recv`, `m_n_slots` sources no longer write per-slot metadata; the receiver reads static routing metadata after waiting for the full semaphore.
    - For `direct_self_compute`, self-rank entries skip inbox sends and slot semaphore traffic, and the destination computes directly from the rank-local input for those entries.
    - Added counters for `direct_self_entries_total`, `payload_send_entries_total`, `remote_send_entries_total`, and send-side rounded-row counts.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 jobs:
  - `/dlwh/repro-source-push-self-bypass-20260702-0538`, cluster `cw-us-east-02a`, stopped after the first over-aggressive direct-self attempt wedged. The implementation was then corrected to keep the slot protocol balanced for remote entries.
  - `/dlwh/repro-source-push-self-bypass-smoke2-20260702-0547`, cluster `cw-us-east-02a`, succeeded for an intermediate self-payload-bypass smoke: `steady_state_time=0.0015181369s`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.0289773941`.
  - `/dlwh/repro-source-push-static-self-smoke-20260702-0552`, cluster `cw-us-east-02a`, succeeded for the final true direct-self/static metadata small smoke.
  - `/dlwh/repro-source-push-static-self-target-20260702-0555`, cluster `cw-us-east-02a`, succeeded for the target A/B/C comparison.
- Checked small final direct-self smoke:
  - Config: `routing=tail_debug`, `metadata_mode=static_recv`, `direct_self_compute=True`, `implementation=m_n_slots`, `peer_loop=grid_switch`, `EP=8`, `entries_per_rank=24`, `D=256`, `I=256`, `inbox_slots=2`, `num_send_sms=4`, `num_sms=16`.
  - `steady_state_time=0.0014902499970048666s`, `w13_tflops_per_rank=0.7191039584994546`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.028977394104003906`, `hidden_mean_abs_diff=0.0012497357092797756`.
  - Counters: `direct_self_entries_total=63`, `remote_send_entries_total=448`, `live_entries_total=511`, `send_wait_empty_count=448`, `slot_full_waits=896`.
- Target A/B/C comparison:

  | metadata_mode | direct_self | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | live entries | remote send entries |
  |---|---:|---:|---:|---:|---:|---:|
  | `remote_slot` | false | `0.009905381322217485s` | `183.88806148375832` | `71.8312740170931` | `17371` | `17371` |
  | `static_recv` | false | `0.009813299674230317s` | `185.6135479468952` | `72.50529216675594` | `17371` | `17371` |
  | `static_recv` | true | `0.010159598003762463s` | `179.28675612218515` | `61.277133186711424` | `17371` | `15199` |

- Interpretation:
  - `static_recv` metadata is correct and gives a small target win, about 0.9% over per-slot remote metadata in this run.
  - True direct-self bypass is correct in the checked small smoke but regresses at target shape despite reducing remote send entries from `17371` to `15199`.
  - The direct-self branch likely hurts scheduling/branch uniformity and removes some useful uniformity from the steady source-push loop. It is not a performance path as currently written.

### 2026-07-02 00:03 PDT - FWD-SWQ-059 Perf output mode, grouped compute jobs, and target tuning sweep
- Question:
  - Test whether debug-output stores, done-semaphore fan-in, larger/smaller tiles, extra inbox buffering, or additional send/compute SMs are the remaining tax in the source-push inbox harness.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `--output-mode {debug,perf}`. `perf` keeps the real `inbox` and hidden output shape but shrinks debug/validation side outputs, and is disabled when `--check` is used.
    - Added `--n-groups-per-job` and `--sweep-n-groups-per-job`. Compute jobs can now cover multiple adjacent N workgroups per queue entry, wait full once, and signal done once.
    - Added stats for `n_groups_per_job`, `n_work_groups_per_entry`, `num_compute_jobs_per_entry`, and `done_signals_per_entry`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 jobs:
  - `/dlwh/repro-source-push-ngroups-smoke-20260702-0559`, cluster `cw-us-east-02a`, succeeded.
  - `/dlwh/repro-source-push-full-sweep-20260702-0603`, cluster `cw-us-east-02a`, completed the Cartesian target sweep rows and was stopped when the trailing static peer-loop spot blocked.
  - `/dlwh/repro-source-push-best-check-20260702-0647`, cluster `cw-us-east-02a`, produced repeat/check rows and was stopped when a trailing switch peer-loop spot blocked.
  - `/dlwh/repro-source-push-direct-self-best-20260702-0656`, cluster `cw-us-east-02a`, succeeded.
- Checked small `n_groups_per_job=2` smoke:
  - `steady_state_time=0.0014919049572199583s`, `w13_tflops_per_rank=0.7183062612761347`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.028977394104003906`.
  - Counters: `n_groups_per_job=2`, `n_work_groups_per_entry=2`, `num_compute_jobs_per_entry=1`, `done_signals_per_entry=1`.
- Full target sweep summary:
  - Completed rows: `120` successful Cartesian rows plus `24` expected error rows.
  - All error rows were `block_k=256,block_n=256` shared-memory overflows: `smem_bytes=294920 > max_smem_bytes=232448`.
  - Best single no-check Cartesian row:
    - `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `n_groups_per_job=1`, `output_mode=perf`, `peer_loop=grid_switch`, `direct_self_compute=False`, `metadata_mode=static_recv`.
    - `steady_state_time=0.00912814901676029s`, `w13_tflops_per_rank=199.54553395826022`, `live_entries_total=17371`, `remote_send_entries_total=17371`, `slot_full_waits=173710`.
  - Other top rows:

  | block_k | block_n | inbox_slots | num_send_sms | num_sms | n_groups_per_job | steady_state_time | W13 TFLOP/s/rank |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 256 | 128 | 4 | 1 | 16 | 1 | `0.009502197344166538s` | `191.6905431056133` |
  | 128 | 128 | 4 | 2 | 32 | 2 | `0.009681603677260378s` | `188.13839424951837` |
  | 128 | 128 | 4 | 1 | 16 | 1 | `0.009755055652931333s` | `186.72178144392814` |
  | 64 | 128 | 8 | 2 | 32 | 1 | `0.009787799674086273s` | `186.09712399636356` |

- Repeat/check follow-up for the best-looking row:
  - Repeat no-check row, same best structural config with `warmup=2`, `steps=7`, `output_mode=perf`:
    - `steady_state_time=0.010859257408550807s`, `w13_tflops_per_rank=167.73535252656663`.
  - Checked target row, same structural config with `output_mode=debug`, `warmup=1`, `steps=1`:
    - `steady_state_time=0.009240464074537158s`, `w13_tflops_per_rank=197.12011809225453`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.2509002685546875`, `hidden_mean_abs_diff=0.020084695890545845`.
  - Direct-self at the best SMS shape:
    - `steady_state_time=0.017455345989825826s`, `w13_tflops_per_rank=104.35091751614`, `send_gbps_per_rank=35.665350910996864`, `direct_self_entries_total=2172`, `remote_send_entries_total=15199`, `send_rounded_rows_per_rank_mean=121592.0`, `rounded_rows_per_rank_mean=138968.0`.
- Interpretation:
  - `block_k=128,block_n=128,inbox_slots=4,n_groups_per_job=1,peer_loop=grid_switch` remains the best stable shape.
  - The single `199.55 TFLOP/s/rank` row is not stable enough to claim a 200 TFLOP/s/rank path; the immediate longer repeat fell to `167.74 TFLOP/s/rank`.
  - The checked target row reached `197.12 TFLOP/s/rank` with `metadata_mismatches=0`, but max diff was larger than the small-shape smoke, so this is a useful harness result rather than an integration-quality production proof.
  - `output_mode=perf` removes debug-output tax in principle, but did not produce a stable win.
  - `n_groups_per_job>1` reduces done semaphore fan-in but generally regresses, consistent with reduced scheduling flexibility or higher per-job pressure.
  - `inbox_slots=8` did not beat `4`.
  - `block_n=256` is not competitive, and `block_k=256,block_n=256` is invalid on shared-memory footprint.
  - Static/switch peer loops blocked in this harness; treat `grid_switch` as the viable peer scheduling path for now.
  - Current bottleneck judgment: communication plus software structure. The source-push inbox protocol can approach 200 TFLOP/s/rank, but the current implementation does not have a stable checked target win beyond the existing best row.

### 2026-07-02 01:10 PDT - FWD-SWQ-060 Receiver scheduling and fixed SMS follow-up
- Question:
  - Test whether receiver-side head-of-line blocking in the source-push inbox harness is the reason the best rows are not stable above 200 TFLOP/s/rank.
  - Specifically, try a ready-style receiver schedule, lower-risk ordered slot variants, and a fixed-only SMS/slot repeat around the current best shape.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `receiver_schedule`, `scan_order`, `scan_candidates`, `workers_per_slot`, and `source_push_mode` config/CLI fields and queue stats.
    - Added `ordered_wait`, a diagnostic that keeps the same wait/release protocol but changes receiver slot visit order.
    - Tried multiple `ready_scan` variants during the pass; the current file now rejects `receiver_schedule=ready_scan` at validation because H100 smokes either deadlocked or produced wrong hidden values.
    - `slot_group` is still rejected as not implemented.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- Ready-scan H100 results on `cw-us-east-02a`:
  - `/dlwh/repro-source-push-ready-scan-smoke-20260702-001238`: failed before useful row because the tiny test capacity was invalid for 32 experts.
  - `/dlwh/repro-source-push-ready-scan-smoke-20260702-001506`: fixed tiny row passed (`steady_state_time=0.001476853s`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.0289774`); ready row hung and was stopped.
  - `/dlwh/repro-source-push-ready-scan-min-smoke-20260702-002159`: ready-only, `scan_candidates=1`; reached `first_call_start` and hung, stopped.
  - `/dlwh/repro-source-push-ready-scan-min-smoke-20260702-003002`: SMEM flag attempt failed lowering with `memref<4xi32, #gpu.address_space<workgroup>> must have a number of elements that is a multiple of 128`.
  - `/dlwh/repro-source-push-ready-scan-min-smoke-20260702-003332`: padded SMEM flag attempt compiled, then hung in first run, stopped.
  - `/dlwh/repro-source-push-ready-scan-min-smoke-20260702-003952`: HBM flag attempt compiled, then hung in first run, stopped.
  - `/dlwh/repro-source-push-ready-scan-min-smoke-20260702-004115`: current fallback-style ready scan completed but was wrong: `steady_state_time=0.002412586s`, `w13_tflops_per_rank=0.22166`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.671875`.
  - `/dlwh/repro-source-push-ready-scan-min-smoke-20260702-004828`: compiled then made no progress; stopped.
- Ordered/fixed receiver schedule rows:

  | job | schedule | scan_order | steady_state_time | W13 TFLOP/s/rank | correctness |
  |---|---|---|---:|---:|---|
  | `/dlwh/repro-source-push-ordered-wait-smoke-20260702-004459` | fixed | current | `0.001839216s` | `0.29076` | `metadata_mismatches=0`, `hidden_max_abs_diff=0.0289774` |
  | `/dlwh/repro-source-push-ordered-wait-smoke-20260702-004459` | ordered | current | `0.001978554s` | `0.27029` | `metadata_mismatches=0`, `hidden_max_abs_diff=0.0289774` |
  | `/dlwh/repro-source-push-live-mask-smoke-20260702-010245` | fixed | rotated | `0.002558922s` | `0.20898` | `metadata_mismatches=0`, `hidden_max_abs_diff=0.0289774` |
  | `/dlwh/repro-source-push-live-mask-smoke-20260702-010245` | ordered | rotated | `0.002221976s` | `0.24067` | `metadata_mismatches=0`, `hidden_max_abs_diff=0.0289774` |
  | `/dlwh/repro-source-push-ordered-target-20260702-004805` | fixed | rotated | `0.010260116s` | `177.53` | perf-only |
  | `/dlwh/repro-source-push-ordered-target-20260702-010700` | fixed | current | `0.010889653s` | `167.27` | perf-only |
  | `/dlwh/repro-source-push-ordered-target-20260702-010700` | fixed | rotated | `0.013704646s` | `132.91` | perf-only |
  | `/dlwh/repro-source-push-ordered-target-20260702-010700` | fixed | hash | `0.009793478s` | `185.99` | perf-only |
  | `/dlwh/repro-source-push-ordered-target-20260702-010700` | ordered | current | `0.012363681s` | `147.33` | perf-only |

- Ordered target outcome:
  - `/dlwh/repro-source-push-ordered-target-20260702-004805` hung after the first fixed row and was stopped.
  - `/dlwh/repro-source-push-ordered-target-20260702-010700` hung at the ordered/rotated first call and was stopped.
  - The only target ordered row that completed was slower than fixed (`147.33` vs fixed/hash `185.99` TFLOP/s/rank in the same job).
- Fixed-only SMS/slot sweep:
  - Job: `/dlwh/repro-source-push-fixed-sms-sweep-20260702-010151`, cluster `cw-us-east-02a`, succeeded with 12 rows.

  | inbox_slots | num_send_sms | num_sms | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|---:|---:|
  | 4 | 2 | 32 | `0.009513282s` | `191.47` | `74.79` |
  | 8 | 1 | 32 | `0.011302224s` | `161.16` | `62.95` |
  | 8 | 1 | 16 | `0.012038354s` | `151.31` | `59.10` |
  | 4 | 2 | 16 | `0.012392647s` | `146.98` | `57.41` |
  | 4 | 1 | 32 | `0.013233860s` | `137.64` | `53.76` |
  | 8 | 2 | 32 | `0.013294513s` | `137.01` | `53.52` |
  | 4 | 2 | 64 | `0.014473559s` | `125.85` | `49.16` |
  | 4 | 1 | 16 | `0.014937385s` | `121.94` | `47.63` |
  | 4 | 1 | 64 | `0.018163113s` | `100.28` | `39.17` |
  | 8 | 2 | 64 | `0.020159322s` | `90.35` | `35.29` |
  | 8 | 2 | 16 | `0.021094127s` | `86.35` | `33.73` |
  | 8 | 1 | 64 | `0.032533448s` | `55.99` | `21.87` |

- Interpretation:
  - No stable speedup this pass. The best fixed-only row here (`191.47 TFLOP/s/rank`) is below the prior one-shot `199.55` and checked `197.12` target rows from FWD-SWQ-059.
  - Increasing total SMS to 64 regressed badly in this sweep. `inbox_slots=8` was also mostly worse. The best shape remains around `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`.
  - `ordered_wait` is useful only as a diagnostic. It preserves blocking waits and does not create real overlap; at target it is either slower or hangs for some orders.
  - The ready-scan attempts did not establish a correct nonblocking receiver schedule. Current evidence points to a software/protocol synchronization problem rather than a WGMMA compute limit.
  - No source-push H100 jobs are left running.

### 2026-07-02 01:21 PDT - FWD-SWQ-061 Block-M and tail-padding tax
- Question:
  - Is the target uniform-routing source-push path losing mainly to tail padding/masked row work, or to per-entry/semaphore overhead?
  - Compare `block_m=64` against `block_m=128`, which roughly halves live entries and semaphore waits but increases padded rows per tail entry.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added `--sweep-block-m`.
    - Added queue counters for masked row work: `masked_rows_total`, `masked_rows_per_rank_*`, `masked_row_fraction`, `send_masked_rows_per_rank_*`, and `send_masked_row_fraction`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 jobs on `cw-us-east-02a`:
  - `/dlwh/repro-source-push-blockm-tail-sweep-20260702-011709`, succeeded.
  - `/dlwh/repro-source-push-blockm128-target-20260702-011900`, succeeded; this was an extra `block_m=128` target row found while checking active jobs.

  | job | block_m | entries_per_rank | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | live entries | masked rows/rank | masked row fraction | tail entry fraction | result |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | `/dlwh/repro-source-push-blockm-tail-sweep-20260702-011709` | 32 | 576 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | invalid: WGMMA requires `m` multiple of 64 |
  | `/dlwh/repro-source-push-blockm-tail-sweep-20260702-011709` | 64 | 288 | `0.011840745s` | `153.83` | `60.09` | `17371` | `7896` | `5.6819%` | `11.6055%` | valid, slower/noisy |
  | `/dlwh/repro-source-push-blockm-tail-sweep-20260702-011709` | 128 | 160 | `0.014755931s` | `130.43` | `50.95` | `9177` | `15760` | `10.7334%` | `21.9680%` | valid regression |
  | `/dlwh/repro-source-push-blockm128-target-20260702-011900` | 128 | 192 | `0.012782207s` | `150.57` | `58.81` | `9177` | `15760` | `10.7334%` | `21.9680%` | valid regression |

- Interpretation:
  - `block_m=32` is not available for this WGMMA path.
  - `block_m=128` cuts live entries from `17371` to `9177`, but it doubles masked-row fraction from `5.68%` to `10.73%` and is slower in both 128-row runs.
  - Therefore the target path is not simply dominated by the number of slot semaphore events. Larger M reduces queue pressure but hurts the WGMMA/copy shape and padding enough to lose.
  - The 64-row row in this sweep was slower than the earlier best 64-row rows, so absolute timing is noisy; the relative `block_m=128` regression is still consistent across two runs.
  - No source-push H100 jobs are left running.

### 2026-07-02 01:25 PDT - FWD-SWQ-062 Nearby inbox slot-depth sweep
- Question:
  - Test non-power-of-two inbox depths around the current best source-push pocket.
  - Keep fixed receiver scheduling and the current best target geometry: `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `entries_per_rank=288`, `num_sms=32`, `output_mode=perf`, `metadata_mode=static_recv`.
- H100 job:
  - `/dlwh/repro-source-push-slot-depth-nearby-20260702-012247`, cluster `cw-us-east-02a`, succeeded.

  | inbox_slots | num_send_sms | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | max slot reuse depth | masked row fraction |
  |---:|---:|---:|---:|---:|---:|---:|
  | 3 | 1 | `0.012471006s` | `146.06` | `57.05` | `93` | `5.6819%` |
  | 3 | 2 | `0.017623122s` | `103.36` | `40.37` | `93` | `5.6819%` |
  | 4 | 1 | `0.012033473s` | `151.37` | `59.13` | `70` | `5.6819%` |
  | 4 | 2 | `0.009473210s` | `192.28` | `75.11` | `70` | `5.6819%` |
  | 5 | 1 | `0.011551082s` | `157.69` | `61.60` | `56` | `5.6819%` |
  | 5 | 2 | `0.013462776s` | `135.30` | `52.85` | `56` | `5.6819%` |
  | 6 | 1 | `0.015101117s` | `120.62` | `47.12` | `47` | `5.6819%` |
  | 6 | 2 | `0.012726683s` | `143.12` | `55.91` | `47` | `5.6819%` |

- Interpretation:
  - `inbox_slots=4,num_send_sms=2,num_sms=32` is again the best local pocket (`192.28 TFLOP/s/rank` here, `191.47` in FWD-SWQ-060).
  - Reducing slot reuse depth below 70 with 5 or 6 slots does not help. More buffering is not the missing overlap.
  - `inbox_slots=3` is worse, and `send_sms=2` is especially bad at 3 slots, likely from poorer modulo assignment/slot-owner balance.
  - No source-push H100 jobs are left running.

### 2026-07-02 01:33 PDT - FWD-SWQ-063 Source-aware ready-scan smoke
- Question:
  - The earlier ready-scan attempts only scanned slots within one source loop. The revised objective describes a full `(src_ordinal, slot)` candidate space.
  - Implement a source-aware `ready_scan` for `peer_loop=grid_switch` where each `(peer_phase, compute_sm)` is a global receiver worker and candidates are `(src_ordinal, slot, n_job)`.
- Code change:
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
    - Added a source-aware `ready_scan` branch for `implementation=m_n_slots`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `metadata_mode=static_recv`.
    - Global ownership uses `candidate_id % (num_sources * num_compute_sms)` so each `(src, slot, n_job)` is owned by one receiver worker.
    - Slot release ownership is also global over `(src, slot)`.
    - Re-disabled `receiver_schedule=ready_scan` in validation after the checked smoke deadlocked, to avoid accidental target sweeps.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `uv run --package marin-levanter --group test python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 checked smoke:
  - Job: `/dlwh/repro-source-push-global-ready-smoke-20260702-013121`, cluster `cw-us-east-02a`, stopped after ready-scan first-run hang.
  - Config: `routing=tail_debug`, `D=256`, `I=256`, `experts_per_rank=4`, `entries_per_rank=24`, `inbox_slots=2`, `block_m=64`, `block_k=64`, `block_n=128`, `num_send_sms=4`, `num_sms=16`, fixed vs ready, `--check`, `--separate-compile`.
  - Fixed row passed: `steady_state_time=0.009216695s`, `w13_tflops_per_rank=0.05802`, `metadata_mismatches=0`, `hidden_max_abs_diff=0.0289774`, `hidden_mean_abs_diff=0.00114439`.
  - Ready row: lower/compile completed, then no progress after `first_run_start`; job was stopped.
- Interpretation:
  - Source-aware ready scanning still deadlocks before correctness can be established.
  - The failure is now on the full candidate-space design requested by the revised objective, not the earlier slot-local approximation.
  - Current status satisfies the "ready_scan correctness/deadlock issue" branch of the stop-condition tree for this design. Continue fixed-schedule tuning or move to a different producer/consumer design; do not run target ready-scan sweeps with the current implementation.

### 2026-07-02 01:38 PDT - FWD-SWQ-064 Fixed-wait tile sweep around current pocket
- Question:
  - With ready-scan unsafe, check whether the fixed schedule has an untuned local W13 tile around the current best pocket.
  - Sweep `block_k={64,128,256}` x `block_n={64,128,256}` with `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `n_group=1`, `entries_per_rank=288`, `output_mode=perf`.
- H100 job:
  - `/dlwh/repro-source-push-fixed-tile-sweep-20260702-013448`, cluster `cw-us-east-02a`, succeeded.

  | block_k | block_n | compute jobs/entry | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | result |
  |---:|---:|---:|---:|---:|---:|---|
  | 64 | 64 | 20 | `0.013798211s` | `132.01` | `51.57` | slower |
  | 128 | 64 | 20 | `0.010916507s` | `166.86` | `65.18` | slower |
  | 256 | 64 | 20 | `0.015641414s` | `116.45` | `45.49` | slower |
  | 64 | 128 | 10 | `0.011568177s` | `157.46` | `61.51` | slower |
  | 128 | 128 | 10 | `0.015814460s` | `115.18` | `44.99` | noisy/regressed in this run |
  | 256 | 128 | 10 | `0.021322850s` | `85.42` | `33.37` | slower |
  | 64 | 256 | 5 | `0.029516344s` | `61.71` | `24.11` | slower |
  | 128 | 256 | 5 | `0.030248649s` | `60.22` | `23.52` | slower |
  | 256 | 256 | 5 | n/a | n/a | n/a | invalid: `smem_bytes=294920 > max_smem_bytes=232448` |

- Interpretation:
  - No tile candidate beats the current local pocket. `block_n=64` and `block_n=256` are slower; `block_k=256` is consistently bad; `256x256` is invalid on SMEM.
  - The `128x128` row was much slower than the same structural config in FWD-SWQ-060/062 (`191-192 TFLOP/s/rank`), so this job should not reset the baseline. Treat it as a noisy negative sweep for alternate tiles.
  - Best stable-looking structural config remains `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `fixed_wait`, `static_recv`, `output_mode=perf`.
  - No source-push H100 jobs are left running.

Addendum:
- Live-row validation now separates correctness from unwritten hidden arena contents. In `/dlwh/repro-source-push-live-mask-smoke-20260702-010245`, both fixed and ordered rows had live `hidden_max_abs_diff=0.028977394104003906`, while `hidden_all_max_abs_diff=0.671875` and `hidden_unwritten_max_abs=0.671875`. The former is the useful correctness signal; the latter is the expected no-zeroing diagnostic for rows never written by live routing metadata.
- `/dlwh/repro-source-push-ordered-missing-target-20260702-011330` was launched only for missing `ordered_wait` rotated/hash target rows. It reached `ordered_wait/rotated` `first_call_start`, emitted no completed rows after several minutes, and was stopped. `ordered_wait/rotated` is therefore a runtime stall at target shape in the current implementation.
- `block_m=128` source-push target probe `/dlwh/repro-source-push-blockm128-target-20260702-011900` succeeded but regressed: `steady_state_time=0.0127822074241s`, `w13_tflops_per_rank=150.565260486`, `send_gbps_per_rank=58.8145548775`, `live_entries_total=9177`, `slot_full_waits=91770`, `rounded_rows_per_rank_mean=146832`, dropped entries `0`. Halving queue entries and semaphore waits did not offset larger CTA/accumulator pressure and extra rounded-row work.
- Slot-depth nearby sweep `/dlwh/repro-source-push-slot-depth-nearby-20260702-012247` succeeded with 8 rows around the current best shape (`block_m=64`, `block_k=128`, `block_n=128`, `num_sms=32`, uniform target, perf mode):

  | inbox_slots | num_send_sms | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|---:|
  | 4 | 2 | `0.0094732100144s` | `192.277102147` | `75.1082430262` |
  | 5 | 1 | `0.0115510824136s` | `157.689236764` | `61.5973581108` |
  | 4 | 1 | `0.0120334734209s` | `151.367880735` | `59.1280784121` |
  | 3 | 1 | `0.0124710059725s` | `146.057292701` | `57.0536299614` |
  | 6 | 2 | `0.0127266826108s` | `143.12302941` | `55.9074333633` |
  | 5 | 2 | `0.0134627759922s` | `135.297606575` | `52.8506275682` |
  | 6 | 1 | `0.0151011172216s` | `120.618980892` | `47.1167894108` |
  | 3 | 2 | `0.0176231221762s` | `103.357472722` | `40.3740127821` |

  This reinforces `inbox_slots=4,num_send_sms=2,num_sms=32` as the best current fixed-wait neighborhood, but still below the 210 TFLOP/s/rank goal.

- Uncommitted global ready-scan smoke `/dlwh/repro-source-push-ready-scan-global-smoke-20260702-013200`:
  - A later ready-scan variant that scans across all source/slot/job candidates compiled on the tiny checked target (`compile_done` after about `1.91s`) but stalled after `first_run_start`.
  - It emitted no row and was stopped.
  - Conclusion unchanged: current ready-scan structure still deadlocks/stalls before it can be a performance path.

- Fixed-wait tile sweep `/dlwh/repro-source-push-fixed-tile-sweep-20260702-013448` succeeded with 9 rows at the target uniform shape (`inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, perf mode):

  | block_k | block_n | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | result |
  |---:|---:|---:|---:|---:|---|
  | 128 | 64 | `0.0109165066388s` | `166.855701176` | `65.1780082718` | valid |
  | 64 | 128 | `0.0115681770258s` | `157.456215058` | `61.5063340071` | valid |
  | 64 | 64 | `0.0137982110027s` | `132.008516846` | `51.565826893` | valid |
  | 256 | 64 | `0.0156414137688s` | `116.452476517` | `45.4892486395` | valid |
  | 128 | 128 | `0.0158144599758s` | `115.178221222` | `44.9914926648` | valid but noisy/slow |
  | 256 | 128 | `0.0213228495792s` | `85.4239187327` | `33.368718255` | valid |
  | 64 | 256 | `0.0295163440052s` | `61.7109412086` | `24.1058364096` | valid |
  | 128 | 256 | `0.0302486493718s` | `60.2169487706` | `23.5222456135` | valid |
  | 256 | 256 | n/a | n/a | n/a | invalid shared-memory footprint (`294920 > 232448`) |

  The best tile-sweep row is still below the current `block_k=128,block_n=128,inbox_slots=4,num_send_sms=2,num_sms=32` neighborhood, so tile-size changes did not unlock the 210 target.

### 2026-07-02 01:45 PDT - FWD-SWQ-065 Manual vs emit pipeline comparison
- Question:
  - Check whether replacing the current manual WGMMA staging loop with `mgpu.emit_pipeline` improves the best fixed source-push target pocket.
- H100 job:
  - `/dlwh/repro-source-push-pipeline-compare-20260702-013920`, cluster `cw-us-east-02a`, killed after emitting 5 rows.

  | compute_pipeline | max_concurrent_steps | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | result |
  |---|---:|---:|---:|---:|---|
  | manual | 4 | `0.015715236403s` | `115.905438702` | `45.2755619929` | valid but slow/noisy |
  | emit | 2 | `0.020857731998s` | `87.3288318106` | `34.112824926` | valid regression |
  | emit | 3 | n/a | n/a | n/a | invalid shared-memory footprint (`245784 > 232448`) |
  | emit | 4 | n/a | n/a | n/a | invalid shared-memory footprint (`327712 > 232448`) |
  | emit | 5 | n/a | n/a | n/a | invalid shared-memory footprint (`409640 > 232448`) |

- Interpretation:
  - `mgpu.emit_pipeline` does not improve the source-push target path here.
  - `max_concurrent_steps=2` is a regression versus the already-noisy manual row, and higher in-flight depth exceeds SMEM.
  - No source-push H100 jobs are left running.

### 2026-07-02 01:52 PDT - FWD-SWQ-066 Best fixed pocket process-level repeat
- Question:
  - Re-run the current best fixed source-push pocket enough times to separate a real 190-ish ceiling from single-row noise.
- H100 job:
  - `/dlwh/repro-source-push-fixed-best-repeat-20260702-084602`, cluster `cw-us-east-02a`, succeeded.
  - Config: `fixed_wait`, `static_recv`, `output_mode=perf`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `entries_per_rank=288`, `warmup=2`, `steps=7`, `--separate-compile`, `--no-check`.

  | repeat | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|
  | 0 | `0.009558173262s` | `190.567938` | `74.440601` |
  | 1 | `0.015324508139s` | `118.860674` | `46.429951` |
  | 2 | `0.010587501406s` | `172.040720` | `67.203406` |
  | 3 | `0.009565891432s` | `190.414180` | `74.380539` |
  | 4 | `0.009538531569s` | `190.960354` | `74.593888` |
  | 5 | `0.014714029584s` | `123.792151` | `48.356309` |
  | 6 | `0.009544583437s` | `190.839274` | `74.546591` |
  | 7 | `0.009648417721s` | `188.785501` | `73.744336` |

- Summary:
  - Median: `0.009607154577s`, `189.599840 TFLOP/s/rank`.
  - Range: `118.860674` to `190.960354 TFLOP/s/rank`.
  - The fast mode is stable around `189-191`, but process-level repeats have a large slow tail.
- Follow-up:
  - Added `--repeat-runs` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py` so one setup/compile can emit multiple steady-state rows. This will isolate process/input/setup effects from kernel execution variability.

### 2026-07-02 01:56 PDT - FWD-SWQ-067 Best fixed pocket in-process repeat
- Question:
  - Does the slow tail from FWD-SWQ-066 come from separate Python processes/input rebuilding, or from repeated kernel execution itself?
- Code change:
  - `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py` now supports `--repeat-runs`.
  - `_time_jitted` reuses the same device inputs and compiled/jitted function, then emits one row per repeated steady-state timing block.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 job:
  - `/dlwh/repro-source-push-fixed-inproc-repeat-20260702-085359`, cluster `cw-us-east-02a`, succeeded.
  - Config: same best fixed pocket as FWD-SWQ-066, but one process and `--repeat-runs 8`.

  | repeat_run | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|
  | 0 | `0.011876592113s` | `153.367342` | `59.909118` |
  | 1 | `0.009596478161s` | `189.807275` | `74.143467` |
  | 2 | `0.015052653716s` | `121.007326` | `47.268487` |
  | 3 | `0.009564737856s` | `190.437145` | `74.389510` |
  | 4 | `0.010157193856s` | `179.329192` | `70.050466` |
  | 5 | `0.015546231431s` | `117.165461` | `45.767758` |
  | 6 | `0.010841142419s` | `168.015630` | `65.631105` |
  | 7 | `0.014964028129s` | `121.724001` | `47.548438` |

- Summary:
  - Median: `0.011358867266s`, `160.691486 TFLOP/s/rank`.
  - Range: `117.165461` to `190.437145 TFLOP/s/rank`.
- Interpretation:
  - The slow tail persists when compilation, device inputs, and process setup are reused.
  - Current fixed source-push path is therefore not just process/setup noisy; repeated kernel execution itself is unstable.
  - This points at software-structure/scheduling contention in the semaphore/slot fixed-wait path, not an isolated WGMMA tile-size issue.
  - No source-push H100 jobs are left running.

### 2026-07-02 01:59 PDT - FWD-SWQ-068 Send-only in-process repeat
- Question:
  - Does the in-process slow tail persist when WGMMA/W13 compute is removed?
- H100 job:
  - `/dlwh/repro-source-push-send-only-inproc-repeat-20260702-085749`, cluster `cw-us-east-02a`, succeeded.
  - Config: same source-push routing/slot geometry as FWD-SWQ-067, but `implementation=send_only`, `output_mode=debug`, `--repeat-runs 8`, `warmup=2`, `steps=7`, `--separate-compile`, `--no-check`.

  | repeat_run | steady_state_time | send GB/s/rank |
  |---:|---:|---:|
  | 0 | `0.007546702866s` | `94.281724` |
  | 1 | `0.008228565866s` | `86.469036` |
  | 2 | `0.012115386980s` | `58.728307` |
  | 3 | `0.006489105855s` | `109.647797` |
  | 4 | `0.006054872547s` | `117.511336` |
  | 5 | `0.006713915583s` | `105.976334` |
  | 6 | `0.011479832132s` | `61.979666` |
  | 7 | `0.008532974869s` | `83.384303` |

- Summary:
  - Median: `0.007887634366s`, `90.375380 GB/s/rank`.
  - Range: `58.728307` to `117.511336 GB/s/rank`.
- Interpretation:
  - Removing WGMMA does not remove the slow tail.
  - The unstable part is already present in source-push copy/semaphore/output-scratch behavior; W13 scheduling is not the only limiter.

### 2026-07-02 02:03 PDT - FWD-SWQ-069 Slot-group receiver schedule sweep
- Question:
  - If fixed-wait instability is driven by per-entry semaphore/job fanout, does assigning a fixed group of receiver workers to each slot improve median throughput?
- H100 job:
  - `/dlwh/repro-source-push-slot-group-sweep-20260702-085941`, cluster `cw-us-east-02a`, succeeded.
  - Config: target source-push pocket with `receiver_schedule=slot_group`, `workers_per_slot={1,2,3,5,6,10,15,30}`, `repeat_runs=3`, `warmup=2`, `steps=7`, `--separate-compile`, `--no-check`.

  | workers_per_slot | repeat TFLOP/s/rank rows | median TFLOP/s/rank | min | max |
  |---:|---|---:|---:|---:|
  | 1 | `64.23, 62.68, 62.86` | `62.86` | `62.68` | `64.23` |
  | 2 | `81.58, 89.56, 99.19` | `89.56` | `81.58` | `99.19` |
  | 3 | `98.33, 137.45, 100.79` | `100.79` | `98.33` | `137.45` |
  | 5 | `155.35, 117.52, 145.41` | `145.41` | `117.52` | `155.35` |
  | 6 | `143.41, 118.19, 180.65` | `143.41` | `118.19` | `180.65` |
  | 10 | `140.99, 137.21, 200.70` | `140.99` | `137.21` | `200.70` |
  | 15 | `137.46, 153.77, 148.56` | `148.56` | `137.46` | `153.77` |
  | 30 | `92.11, 110.29, 98.30` | `98.30` | `92.11` | `110.29` |

- Interpretation:
  - `slot_group` is not a speedup over fixed-wait. It reduces semaphore fanout but loses too much parallelism or introduces worse lane imbalance.
  - The `workers_per_slot=10` single `200.70 TFLOP/s/rank` row is not stable enough to treat as progress; its median is `140.99`.
  - Current best remains fixed-wait, but its stable median is below 210 and its in-process distribution has a large slow tail.

### 2026-07-02 02:07 PDT - FWD-SWQ-070 Scratch inbox storage smoke
- Question:
  - Can the full remote-write inbox be moved from a returned output buffer to non-returned GMEM scratch to reduce output/scratch allocation pressure?
- Code change:
  - Added experimental `--inbox-storage {output,scratch}` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - The scratch variant wires the full inbox through `scratch_shapes=(mgpu.GMEM(...),)` and returns a tiny placeholder inbox in `output_mode=perf`.
- H100 smoke:
  - `/dlwh/repro-source-push-scratch-inbox-smoke-20260702-090531`, cluster `cw-us-east-02a`, succeeded as a job but scratch row failed.

  | inbox_storage | repeat_run | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | result |
  |---|---:|---:|---:|---:|---|
  | output | 0 | `0.003919158597s` | `0.136451` | `0.266506` | tiny-shape smoke baseline |
  | output | 1 | `0.010931410672s` | `0.048921` | `0.095549` | tiny-shape smoke baseline |
  | scratch | n/a | n/a | n/a | n/a | `NotImplementedError: Unsupported memory space: gmem` |

- Interpretation:
  - Pallas/Mosaic currently does not support `mgpu.GMEM` in `scratch_shapes` for this kernel path.
  - The scratch-inbox approach cannot be used as a small patch unless we move away from `mgpu.kernel` scratch allocation or find a different aliasing mechanism.
  - `inbox_storage=scratch` is now disabled in validation to avoid accidental target sweeps.

### 2026-07-02 02:28 PDT - FWD-SWQ-071 Re-enabled simple ready-scan branch
- Question:
  - The current code still contained a simpler per-source `ready_scan` branch, while validation rejected all `ready_scan` runs based on the older global candidate-space deadlock. Does the simpler branch work, and can it stabilize the target?
- Code change:
  - Removed the unconditional `receiver_schedule=ready_scan` validation rejection in `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- H100 smokes:
  - `/dlwh/repro-source-push-ready-scan-local-smoke-20260702-091236`, cluster `cw-us-east-02a`, stopped.
    - `fixed_wait` tiny rows completed: `0.001838723974s` / `0.001584498057s`.
    - `ready_scan` with `scan_order=rotated` compiled, then stalled after `first_run_start`.
  - `/dlwh/repro-source-push-ready-scan-current-smoke-20260702-091636`, cluster `cw-us-east-02a`, succeeded.
    - `ready_scan`, `scan_order=current`, tiny rows: `0.001771671697s`, `0.001612790627s`.
- H100 target comparison:
  - `/dlwh/repro-source-push-ready-current-target-20260702-091827`, cluster `cw-us-east-02a`, stopped after target `ready_scan/current` stalled after `first_run_start`.

  | schedule | repeat_run | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | result |
  |---|---:|---:|---:|---:|---|
  | fixed_wait | 0 | `0.009551624002s` | `190.698605` | `74.491642` | completed |
  | fixed_wait | 1 | `0.009404732413s` | `193.677107` | `75.655120` | completed |
  | fixed_wait | 2 | `0.009309633269s` | `195.655545` | `76.427947` | completed |
  | fixed_wait | 3 | `0.009444840552s` | `192.854645` | `75.333846` | completed |
  | fixed_wait | 4 | `0.009636285848s` | `189.023177` | `73.837179` | completed |
  | fixed_wait | 5 | `0.014543765990s` | `125.241383` | `48.922415` | slow tail |
  | fixed_wait | 6 | `0.009429359875s` | `193.171264` | `75.457525` | completed |
  | fixed_wait | 7 | `0.009362845715s` | `194.543563` | `75.993579` | completed |
  | ready_scan/current | n/a | n/a | n/a | n/a | target first-run stall |

- Summary:
  - Fixed target median in this job: `0.009437100213s`, `193.012955 TFLOP/s/rank`, `75.395685 GB/s/rank`.
  - `ready_scan/rotated` is not safe even at the tiny smoke shape.
  - `ready_scan/current` works on the tiny shape but stalls at the target shape before emitting a row.
- Interpretation:
  - Lightweight ready scanning did not produce a target performance path.
  - The target stall is a correctness/liveness issue, not a performance regression.
  - Current best remains fixed-wait; the best repeated median seen in this pass is about `193 TFLOP/s/rank`, still below the 210 Stage 1 target.

### 2026-07-02 02:28 PDT - FWD-SWQ-072 Inbox input/output alias smoke
- Question:
  - Can the full inbox be passed as an input/output aliased buffer so repeated calls avoid allocating a fresh returned inbox output?
- Code change:
  - Added experimental `inbox_storage=alias` path using `pl.pallas_call(..., input_output_aliases={4: 0})`.
  - `_time_jitted` can now thread an aliased output buffer into the next call.
- H100 smoke:
  - `/dlwh/repro-source-push-inbox-alias-smoke-20260702-092534`, cluster `cw-us-east-02a`, succeeded as a job but alias row failed.

  | inbox_storage | repeat_run | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | result |
  |---|---:|---:|---:|---:|---|
  | output | 0 | `0.001737541364s` | `0.307776` | `0.601125` | tiny-shape smoke baseline |
  | output | 1 | `0.001736621683s` | `0.307939` | `0.601444` | tiny-shape smoke baseline |
  | alias | n/a | n/a | n/a | n/a | `ValueError: Mosaic GPU kernel exceeds available shared memory: smem_bytes=14730256 > max_smem_bytes=232448` |

- Interpretation:
  - The quick `pl.pallas_call` alias wrapper maps the full inbox into the kernel memory footprint and blows SMEM, so this is not a viable small patch.
  - `inbox_storage=alias` is now disabled in validation alongside `inbox_storage=scratch`.
  - Avoiding returned-inbox allocation likely requires a deeper wrapper/lowering change or a different kernel boundary.

### 2026-07-02 02:08 PDT - FWD-SWQ-071 Send-only perf-output repeat
- Question:
  - How much of the send-only source-push tax in FWD-SWQ-068 came from debug `seen_payload`/`seen_meta` writes and their output allocation, rather than the inbox copy/semaphore protocol itself?
- Code change:
  - `output_mode=perf` now supports `implementation=send_only` with `metadata_mode=static_recv`.
  - In this mode the send-only path skips slot metadata writes and receiver debug `seen_*` writes; it still remote-writes the full inbox and uses the same empty/full semaphore protocol.
- H100 job:
  - `/dlwh/repro-source-push-send-only-perf-repeat-20260702-020508`, cluster `cw-us-east-02a`, succeeded.
  - Config: same target source-push routing/slot geometry as FWD-SWQ-068, but `implementation=send_only`, `output_mode=perf`, `--repeat-runs 8`, `warmup=2`, `steps=7`, `--separate-compile`, `--no-check`.

  | repeat_run | steady_state_time | send GB/s/rank |
  |---:|---:|---:|
  | 0 | `0.012597401294s` | `56.481186` |
  | 1 | `0.005783228570s` | `123.030960` |
  | 2 | `0.006046383575s` | `117.676319` |
  | 3 | `0.006035404147s` | `117.890392` |
  | 4 | `0.007975454442s` | `89.213244` |
  | 5 | `0.012893703167s` | `55.183228` |
  | 6 | `0.005775786703s` | `123.189480` |
  | 7 | `0.005699594006s` | `124.836288` |

- Summary:
  - Median: `0.006040893861s`, `117.783355 GB/s/rank`.
  - Prior send-only debug-output median from FWD-SWQ-068: `0.007887634366s`, `90.375380 GB/s/rank`.
- Interpretation:
  - Debug side-output writes/allocations are a real tax for send-only: removing them improved median by about `23.4%` in time (`0.0078876s -> 0.0060409s`).
  - The slow tail did not disappear: repeats `0` and `5` are still about `0.0126-0.0129s`.
  - Current bottleneck judgment remains source-push copy/semaphore/output-scratch scheduling instability, not just WGMMA or debug-output overhead.

### 2026-07-02 02:22 PDT - FWD-SWQ-072 Observed concurrent ready-current target job
- Coordination:
  - Observed active H100 job `/dlwh/repro-source-push-ready-current-target-20260702-091827` on cluster `cw-us-east-02a`.
  - I did not launch this job and did not stop it.
  - Posted coordination update in `6597-forward`.
- Fixed-wait half completed before the job entered `ready_scan`:

  | repeat_run | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank |
  |---:|---:|---:|---:|
  | 0 | `0.009551624002s` | `190.698605` | `74.491642` |
  | 1 | `0.009404732413s` | `193.677107` | `75.655120` |
  | 2 | `0.009309633269s` | `195.655545` | `76.427947` |
  | 3 | `0.009444840552s` | `192.854645` | `75.333846` |
  | 4 | `0.009636285848s` | `189.023177` | `73.837179` |
  | 5 | `0.014543765990s` | `125.241383` | `48.922415` |
  | 6 | `0.009429359875s` | `193.171264` | `75.457525` |
  | 7 | `0.009362845715s` | `194.543563` | `75.993579` |

- Summary:
  - Median: `0.009437100213s`, about `193.012955 TFLOP/s/rank`.
  - One slow-tail repeat remains: repeat `5` at `0.014543765990s`, `125.241383 TFLOP/s/rank`.
- Ready-scan status:
  - The job then compiled `receiver_schedule=ready_scan`.
  - Logs reached `first_run_start` at `2026-07-02 09:19:31 UTC` and had not emitted `first_run_done` or result rows by `02:22 PDT`.
- Interpretation:
  - This reproduces the fixed-wait fast pocket and the same slow-tail behavior.
  - The subsequent `ready_scan` start-only behavior matches earlier ready-scan deadlock/stall observations.
  - I am holding off on launching the fixed-wait `num_send_sms=2,3,4` repeat sweep until this H100 reservation clears or is handed off.

### 2026-07-02 02:26 PDT - FWD-SWQ-073 Fixed-wait producer-SM repeat sweep
- Question:
  - Does changing producer ownership stabilize the fixed-wait source-push path?
  - Sweep `num_send_sms=1,2,3,4` at the current fixed pocket while keeping `num_sms=32`.
- H100 job:
  - `/dlwh/repro-source-push-send-sms-repeat-20260702-0924`, cluster `cw-us-east-02a`, succeeded.
  - Config: `implementation=m_n_slots`, `routing=uniform`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_sms=32`, `repeat_runs=8`, `warmup=2`, `steps=7`, `--no-check`.

  | num_send_sms | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|
  | 1 | `0.011332655858s` | `160.730592` | `101.649665` | `163.746946` |
  | 2 | `0.010458210932s` | `174.196073` | `116.080274` | `196.091615` |
  | 3 | `0.011350601496s` | `160.484537` | `106.459776` | `162.648930` |
  | 4 | `0.011415319484s` | `159.564951` | `115.026196` | `160.275030` |

- Interpretation:
  - `num_send_sms=2` remains the least-bad full-path setting in this repeat, but the median was only `174.20 TFLOP/s/rank`.
  - Increasing producers to 3 or 4 does not stabilize the full WGMMA path.
  - The slow tail is still present, so producer count alone is not the missing overlap/scheduling fix.

### 2026-07-02 02:34 PDT - FWD-SWQ-074 Hidden-output sink diagnostic
- Question:
  - Is full hidden output materialization or giant output-buffer initialization a major part of the source-push tax?
- Code change:
  - Added experimental `--hidden-output-mode {full,sink}` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - First sink attempt tried to write one scalar from the tiled accumulator and failed in lowering because Mosaic requires accumulator slices to be multiples of tile shape `(64, 8)`.
  - Corrected sink mode stores each computed `64 x 128` tile into a much smaller cyclic `(src, slot)` hidden sink. This keeps WGMMA/SILU live and reduces output shape, but still performs a tile-shaped store.
- H100 jobs:
  - `/dlwh/repro-source-push-hidden-sink-20260702-0927`, cluster `cw-us-east-02a`, failed during scalar-sink lowering after completing the full-output half.
  - `/dlwh/repro-source-push-hidden-sink2-20260702-0932`, cluster `cw-us-east-02a`, succeeded with corrected tile-sink mode.

  | mode | job | median steady_state_time | median W13 TFLOP/s/rank | result |
  |---|---|---:|---:|---|
  | full | `/dlwh/repro-source-push-hidden-sink-20260702-0927` | `0.009547218564s` | `190.787500` | baseline half completed |
  | scalar sink | `/dlwh/repro-source-push-hidden-sink-20260702-0927` | n/a | n/a | failed: `slice shape [1, 1]` not multiple of accumulator tile `(64, 8)` |
  | tile sink | `/dlwh/repro-source-push-hidden-sink2-20260702-0932` | `0.012997329850s` | `141.267949` | valid but slower |

- Interpretation:
  - Full hidden output shape is not an obvious easy win in this harness.
  - The corrected sink mode is slower, likely because a small cyclic sink creates repeated writes to the same GMEM rows and a less favorable output layout.
  - This does not rule out a production W2-fused path, but it argues against spending more time on hidden-output shrinking inside the current source-push harness.

### 2026-07-02 02:36 PDT - FWD-SWQ-075 Send-only producer-SM sweep
- Question:
  - Without WGMMA/hidden stores, does the producer/copy side prefer more send workers?
- H100 job:
  - `/dlwh/repro-source-push-send-only-sms-sweep-20260702-0935`, cluster `cw-us-east-02a`, succeeded.
  - Config: same routing and slot geometry as FWD-SWQ-073, but `implementation=send_only`, `output_mode=perf`, `repeat_runs=8`.

  | num_send_sms | median steady_state_time | median send GB/s/rank | min GB/s/rank | max GB/s/rank |
  |---:|---:|---:|---:|---:|
  | 1 | `0.012627916361s` | `56.458212` | `49.821655` | `71.999437` |
  | 2 | `0.005767165783s` | `123.374177` | `68.960984` | `125.401970` |
  | 3 | `0.005901318775s` | `121.558835` | `65.988476` | `135.224364` |
  | 4 | `0.004562460784s` | `158.764683` | `57.438985` | `183.208139` |

- Interpretation:
  - Send-only strongly prefers more producer workers; `num_send_sms=4` is the best median and reaches `183.21 GB/s/rank`.
  - This conflicts with the full WGMMA path, where `num_send_sms=4` is slower. The issue is not raw producer copy bandwidth; it is interference with the compute-side schedule/protocol.

### 2026-07-02 02:38 PDT - FWD-SWQ-076 Send4 compute-worker balance sweep
- Question:
  - Can the full WGMMA path use the faster `num_send_sms=4` producer setting if we increase `num_sms` to keep enough compute workers?
- H100 job:
  - `/dlwh/repro-source-push-send4-compute-balance-20260702-0937`, cluster `cw-us-east-02a`, succeeded.
  - Config: same fixed pocket as FWD-SWQ-073, but `num_send_sms=4`, `sweep_num_sms=32,34,36,40,48`, `repeat_runs=6`.

  | num_sms | compute workers | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|---:|
  | 32 | 28 | `0.012146341997s` | `150.503083` | `98.901626` | `161.420185` |
  | 34 | 30 | `0.012657967080s` | `143.900029` | `105.574300` | `145.861262` |
  | 36 | 32 | `0.016199639067s` | `112.449405` | `109.145886` | `140.905683` |
  | 40 | 36 | `0.015663144910s` | `116.462717` | `104.584664` | `135.690994` |
  | 48 | 44 | `0.015521660497s` | `117.461902` | `102.247753` | `149.282296` |

- Interpretation:
  - More total CTAs do not make the fast send-only producer setting viable for the full path.
  - The extra producer/compute programs appear to interfere with WGMMA scheduling more than they help overlap copy.
  - Current bottleneck judgment: software/protocol scheduling interference between source-push copy/semaphore CTAs and WGMMA CTAs, not raw copy bandwidth, hidden-output buffer size, or WGMMA tile math alone.

### 2026-07-02 02:43 PDT - FWD-SWQ-077 Fixed-wait compute-worker count sweeps
- Question:
  - Does increasing receiver/compute worker count reduce fixed-wait slow tails for the current source-push pocket?
  - Does the current best `num_send_sms=2` setting improve with nearby `num_sms` values around 32?
- H100 jobs:
  - `/dlwh/repro-source-push-fixed-sms64-sweep-20260702-093509`, cluster `cw-us-east-02a`, succeeded.
  - `/dlwh/repro-source-push-send2-compute-nearby-20260702-0940`, cluster `cw-us-east-02a`, succeeded.
- Shared config:
  - `implementation=m_n_slots`, `routing=uniform`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `hidden_output_mode=full`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `warmup=2`, `steps=7`, `--no-check`.
- `num_sms=64` sweep:

  | num_send_sms | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|
  | 1 | `0.020542044857s` | `88.672623` | `78.232393` | `102.507006` |
  | 2 | `0.015388849375s` | `118.363878` | `102.048281` | `142.585191` |
  | 4 | `0.014866538928s` | `122.530442` | `93.650632` | `142.729282` |

- `num_send_sms=2` nearby `num_sms` sweep:

  | num_sms | median steady_state_time | median W13 TFLOP/s/rank | median send GB/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|---:|
  | 30 | `0.015023226284s` | `122.480188` | `47.843823` | `102.291225` | `148.943942` |
  | 32 | `0.009580117983s` | `190.132532` | `74.270520` | `133.410604` | `193.270701` |
  | 34 | `0.013888717064s` | `131.158661` | `51.233852` | `91.799253` | `133.612599` |
  | 36 | `0.016027360439s` | `113.729346` | `44.425526` | `107.347938` | `136.950840` |
  | 40 | `0.014844690067s` | `123.061323` | `48.070829` | `100.283028` | `130.602520` |

- Interpretation:
  - `num_sms=64` is strongly negative; adding more receiver workers increases overhead/contention instead of reducing slow tails.
  - The nearby `num_send_sms=2` sweep shows a sharp pocket at `num_sms=32`; even there two of six repeats are slow (`150.25` and `133.41 TFLOP/s/rank`).
  - Current bottleneck judgment remains software/protocol-structure-bound in the fixed-wait source-push semaphore/slot loop. It is not fixed by more receiver workers, more producer workers, hidden-output shrinking, or WGMMA tile retuning.

### 2026-07-02 02:56 PDT - FWD-SWQ-078 Direct-owned receiver job schedule
- Question:
  - Can we keep fixed-wait correctness/protocol but reduce receiver-side software work by having each compute worker iterate only the `(slot, n_job)` work groups it owns?
- Code change:
  - Added experimental `receiver_schedule=owned_jobs` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - The new branch preserves the existing `full_sem` wait and `done_sem` release protocol, but replaces the per-worker scan over all `inbox_slots * n_compute_jobs` candidates with a direct modular owner loop.
  - Added queue stats: `local_work_groups_per_source` and `owned_work_groups_per_worker_round`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `git diff --check -- lib/levanter/scripts/bench/repro_source_push_inbox_queue.py .agents/logbooks/6597-moe-mgpu-forward.md`
- H100 checked smoke attempts:
  - `/dlwh/repro-source-push-owned-jobs-smoke-20260702-0947`, cluster `cw-us-east-02a`, succeeded as a job but both rows failed in host reference before kernel execution: default `entries_per_rank=2` was too small for `T=4096` routing.
  - `/dlwh/repro-source-push-owned-jobs-smoke2-20260702-0948`, cluster `cw-us-east-02a`, succeeded as a job but both rows failed in host reference with `entries_per_rank=36`.
  - `/dlwh/repro-source-push-owned-jobs-smoke3-20260702-0950`, cluster `cw-us-east-02a`, succeeded as a job but failed in host reference with `entries_per_rank=48`; local input stats showed `max_live_end=24896` with `hidden_rows=24576` because dropped earlier buckets can leave later live row bases beyond the compact hidden buffer.
  - Local capacity check showed `entries_per_rank=56` has `dropped_entries_total=0`, `hidden_rows=28672`, `max_live_end=24960`.
- Successful H100 smoke:
  - `/dlwh/repro-source-push-owned-jobs-smoke4-20260702-0953`, cluster `cw-us-east-02a`, succeeded.
  - Config: `tokens_per_rank=4096`, `entries_per_rank=56`, `routing=uniform`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=debug`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `warmup=1`, `steps=3`, checked.

  | receiver_schedule | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | metadata_mismatches | max_abs_diff |
  |---|---:|---:|---:|---:|---:|
  | `fixed_wait` | `0.008288928618s` | `37.875058` | `14.794945` | `0` | `0.061573` |
  | `owned_jobs` | `0.003172392646s` | `98.961159` | `38.656703` | `0` | `0.061573` |

- Interpretation:
  - `owned_jobs` is a real correctness-preserving schedule improvement on the small checked smoke.
  - For this shape, `local_work_groups_per_source=40` and `owned_work_groups_per_worker_round=2`, so the receiver worker scans far fewer dead candidates than fixed wait while keeping the same `10` done signals per entry.
  - Target perf repeat is still required; held off because `/dlwh/repro-source-push-ready-multi-smoke-20260702-0953` is currently running and appears start-only after `first_run_start`.

### 2026-07-02 02:59 PDT - FWD-SWQ-079 Direct-owned target repeat
- Question:
  - Does the small-shape `owned_jobs` win survive at target shape?
- H100 job:
  - `/dlwh/repro-source-push-owned-jobs-target-20260702-0957`, cluster `cw-us-east-02a`, succeeded.
  - Config: target current fixed pocket, `tokens_per_rank=32768`, `entries_per_rank=288`, `routing=uniform`, `traffic_pattern=all_to_all`, `implementation=m_n_slots`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `hidden_output_mode=full`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`, `--separate-compile`.

  | receiver_schedule | median steady_state_time | median W13 TFLOP/s/rank | median send GB/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---|---:|---:|---:|---:|---:|
  | `fixed_wait` | `0.009230798925s` | `197.326526` | `77.080674` | `142.586310` | `198.696572` |
  | `owned_jobs` | `0.010791040441s` | `169.154311` | `66.075903` | `110.940220` | `178.910594` |

- Interpretation:
  - Target result is negative: `owned_jobs` reduces candidate scanning but regresses median target time by about `16.9%` versus the same-job fixed-wait row.
  - Likely cause: the branch computes all directly owned jobs for a round and only then releases all slots, delaying empty-slot signals for earlier slots. This can stall producer reuse even though receiver candidate scanning is cheaper.
  - Next experiment: keep direct job ownership but restore per-slot ordering/release, so each slot is released immediately after its owned N jobs complete.

### 2026-07-02 03:05 PDT - FWD-SWQ-080 Direct-owned per-slot release schedule
- Question:
  - Can direct job ownership help at target if we preserve the fixed-wait per-slot release order?
- Code change:
  - Added experimental `receiver_schedule=owned_slots` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - The branch loops slots in order, computes only directly owned `job_i` values for that slot, then calls the existing `_release_completed_slot` immediately.
  - Added queue stat `owned_slot_jobs_per_worker`; target value is `1`.
- Local checks:
  - `uvx black@25.9.0 --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `git diff --check -- lib/levanter/scripts/bench/repro_source_push_inbox_queue.py .agents/logbooks/6597-moe-mgpu-forward.md`
- H100 checked smoke:
  - `/dlwh/repro-source-push-owned-slots-smoke-20260702-1000`, cluster `cw-us-east-02a`, succeeded.
  - Config: same zero-drop checked smoke as FWD-SWQ-078, sweeping `fixed_wait,owned_slots`.

  | receiver_schedule | steady_state_time | W13 TFLOP/s/rank | send GB/s/rank | metadata_mismatches | max_abs_diff |
  |---|---:|---:|---:|---:|---:|
  | `fixed_wait` | `0.003095567382s` | `101.417161` | `39.616078` | `0` | `0.061573` |
  | `owned_slots` | `0.005162347651s` | `60.814125` | `23.755518` | `0` | `0.061573` |

- H100 target repeat:
  - `/dlwh/repro-source-push-owned-slots-target-20260702-1003`, cluster `cw-us-east-02a`, succeeded.
  - Config: target current fixed pocket, `tokens_per_rank=32768`, `entries_per_rank=288`, `routing=uniform`, `traffic_pattern=all_to_all`, `implementation=m_n_slots`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `hidden_output_mode=full`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`, `--separate-compile`.

  | receiver_schedule | median steady_state_time | median W13 TFLOP/s/rank | median send GB/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---|---:|---:|---:|---:|---:|
  | `fixed_wait` | `0.010753012129s` | `169.393029` | `66.169152` | `124.929544` | `197.632067` |
  | `owned_slots` | `0.009304244570s` | `195.768926` | `76.472237` | `120.401905` | `197.654049` |

- Interpretation:
  - `owned_slots` is a target same-job win over `fixed_wait` in a noisy run and avoids the `owned_jobs` target regression.
  - Absolute median `195.77 TFLOP/s/rank` is still not a confirmed improvement over the best fixed-wait fast pocket (`~197 TFLOP/s/rank` in FWD-SWQ-079, with historical rows around `190-193` and occasional fast rows).
  - Next action: repeat `owned_slots` alone to measure stability without compiling/running a paired fixed-wait row.

### 2026-07-02 03:09 PDT - FWD-SWQ-081 Owned-slots solo stability
- Question:
  - Is the `owned_slots` target same-job win stable when run alone for more repeats?
- H100 job:
  - `/dlwh/repro-source-push-owned-slots-solo-20260702-1006`, cluster `cw-us-east-02a`, succeeded.
  - Config: target current fixed pocket with `receiver_schedule=owned_slots`, `tokens_per_rank=32768`, `entries_per_rank=288`, `routing=uniform`, `traffic_pattern=all_to_all`, `implementation=m_n_slots`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `hidden_output_mode=full`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=16`, `--no-check`, `--separate-compile`.

  | metric | value |
  |---|---:|
  | median steady_state_time | `0.009547218996s` |
  | median W13 TFLOP/s/rank | `190.821228` |
  | median send GB/s/rank | `74.539542` |
  | min W13 TFLOP/s/rank | `125.735202` |
  | max W13 TFLOP/s/rank | `198.085746` |

- Interpretation:
  - `owned_slots` is not a confirmed target speedup when run alone; median falls back near the fixed-wait baseline range because slow repeats remain.
  - The branch still often reaches the fast pocket (`~194-198 TFLOP/s/rank`), but it does not remove the slow-tail mechanism.
  - Next action: test whether reducing done-signal count per entry via `n_group` / `n_groups_per_job` helps now that per-slot ownership is available.

### 2026-07-02 03:12 PDT - FWD-SWQ-082 Owned-slots N-group / jobs-per-entry sweep
- Question:
  - Does reducing done signals per entry help `owned_slots` once direct per-slot ownership is in place?
- H100 job:
  - `/dlwh/repro-source-push-owned-slots-ngroup-20260702-1010`, cluster `cw-us-east-02a`, succeeded.
  - Config: target current fixed pocket with `receiver_schedule=owned_slots`, `tokens_per_rank=32768`, `entries_per_rank=288`, `routing=uniform`, `traffic_pattern=all_to_all`, `implementation=m_n_slots`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `hidden_output_mode=full`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`, `--separate-compile`.

  | n_group | n_groups_per_job | median steady_state_time | median W13 TFLOP/s/rank | median send GB/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | 1 | `0.009227763213s` | `197.391456` | `77.106038` | `136.677539` | `198.431101` |
  | 1 | 2 | `0.009646938432s` | `188.815363` | `73.756001` | `124.465835` | `191.552094` |
  | 2 | 1 | `0.023298971421s` | `78.187533` | `30.542005` | `69.425444` | `84.784374` |
  | 2 | 2 | `0.021237426920s` | `85.770951` | `33.504278` | `82.006268` | `92.736513` |

- Interpretation:
  - Grouping N work is negative. `n_groups_per_job=2` loses even without the two-N-tile fused body, and `n_group=2` is a severe regression.
  - The best row remains plain `owned_slots`, `n_group=1`, `n_groups_per_job=1`.
  - This sweep produced a fast `owned_slots` median (`197.39 TFLOP/s/rank`), but FWD-SWQ-081 showed the same config is not stable over 16 solo repeats (`190.82 TFLOP/s/rank` median). Treat this as noisy evidence, not a new baseline.

### 2026-07-02 03:17 PDT - FWD-SWQ-083 Owned-slots bounded interaction sweep
- Question:
  - Does `owned_slots` prefer a different `block_k`, producer count, or total SM count than the fixed-wait pocket?
- H100 job:
  - `/dlwh/repro-source-push-owned-slots-interact-20260702-1014`, cluster `cw-us-east-02a`, succeeded.
  - Config: target `owned_slots`, `tokens_per_rank=32768`, `entries_per_rank=288`, `routing=uniform`, `traffic_pattern=all_to_all`, `implementation=m_n_slots`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `hidden_output_mode=full`, `block_m=64`, `block_n=128`, `inbox_slots=4`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`, `--separate-compile`.

  | block_k | num_send_sms | num_sms | median steady_state_time | median W13 TFLOP/s/rank | median send GB/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 128 | 1 | 16 | `0.010667869489s` | `171.567473` | `67.018544` | `118.654045` | `188.397041` |
  | 128 | 1 | 32 | `0.011029648561s` | `165.144423` | `64.509540` | `132.104969` | `167.097249` |
  | 128 | 2 | 16 | `0.011941052724s` | `152.539605` | `59.585783` | `121.789318` | `153.166650` |
  | 128 | 2 | 32 | `0.009183862513s` | `198.336084` | `77.475033` | `126.667036` | `198.963395` |
  | 256 | 1 | 16 | `0.009462767447s` | `192.489482` | `75.191204` | `117.747700` | `193.420874` |
  | 256 | 1 | 32 | `0.025886109298s` | `70.384603` | `27.493986` | `65.222722` | `77.928563` |
  | 256 | 2 | 16 | `0.011592252213s` | `157.129233` | `61.378607` | `102.977414` | `158.583550` |
  | 256 | 2 | 32 | `0.021712419810s` | `83.900228` | `32.773526` | `70.720900` | `88.210721` |

- Interpretation:
  - The best interaction remains `block_k=128`, `num_send_sms=2`, `num_sms=32`, i.e. the current fixed-wait pocket plus `owned_slots`.
  - This job produced the best recent repeated median (`198.34 TFLOP/s/rank`) for `owned_slots`, with six fast rows near `198-199` and two slow rows.
  - The result is still not stable enough to call solved because the same config was `190.82 TFLOP/s/rank` median in the 16-repeat solo run. The remaining problem is the slow-tail probability, not peak fast-pocket speed.

### 2026-07-02 02:42 PDT - FWD-SWQ-078 Send2 nearby compute-worker sweep
- Question:
  - Is the current `num_send_sms=2,num_sms=32` full-path pocket slightly under- or over-provisioned on compute workers?
- H100 job:
  - `/dlwh/repro-source-push-send2-compute-nearby-20260702-0940`, cluster `cw-us-east-02a`, succeeded.
  - Config: same fixed pocket as FWD-SWQ-073, but `num_send_sms=2`, `sweep_num_sms=30,32,34,36,40`, `repeat_runs=6`.

  | num_sms | compute workers | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|---:|
  | 30 | 28 | `0.015023226284s` | `122.480188` | `102.291225` | `148.943942` |
  | 32 | 30 | `0.009580117983s` | `190.132532` | `133.410604` | `193.270701` |
  | 34 | 32 | `0.013888717064s` | `131.158661` | `91.799253` | `133.612599` |
  | 36 | 34 | `0.016027360439s` | `113.729346` | `107.347938` | `136.950840` |
  | 40 | 38 | `0.014844690067s` | `123.061323` | `100.283028` | `130.602520` |

- Interpretation:
  - This confirms the current local pocket: `num_send_sms=2,num_sms=32`.
  - Nearby lower/higher total worker counts regress. The fixed-wait path is sensitive to CTA count/scheduling, and simply adding compute workers does not stabilize the slow tail.
  - No speedup over the existing fixed-wait baseline was found in this pass.

### 2026-07-02 02:58 PDT - FWD-SWQ-079 owned-jobs target regression and ready-scan deadlock
- Question:
  - Does direct receiver job ownership reduce fixed-wait software overhead at the target shape?
  - Can a multi-candidate ready scan compute all currently ready local jobs and reduce head-of-line waits?
- Code changes:
  - Kept experimental `receiver_schedule=owned_jobs` in `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - Tried a multi-candidate `ready_scan` receiver pass that marks already-computed candidates in SMEM and falls back to blocking fixed-wait for the rest of the round.
  - Disabled `receiver_schedule=ready_scan` again after the checked smoke deadlocked.
- H100 jobs:
  - `/dlwh/repro-source-push-owned-jobs-target-20260702-0951`, cluster `cw-us-east-02a`, succeeded.
  - `/dlwh/repro-source-push-ready-multi-smoke-20260702-0953`, cluster `cw-us-east-02a`, stopped after compile succeeded but first execution produced no row.
- Target config:
  - `implementation=m_n_slots`, `queue_mode=routing`, `routing=uniform`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `lowering_semantics=lane`, `metadata_mode=static_recv`, `output_mode=perf`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `repeat_runs=6`.

  | receiver_schedule | median steady_state_time | median W13 TFLOP/s/rank | median send GB/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---|---:|---:|---:|---:|---:|
  | `fixed_wait` | `0.009461655647s` | `192.512947` | `75.200370` | `140.384635` | `194.780626` |
  | `owned_jobs` | `0.010478099725s` | `173.837684` | `67.905345` | `134.911777` | `175.936923` |

- Ready-scan smoke:
  - Job `/dlwh/repro-source-push-ready-multi-smoke-20260702-0953` compiled (`compile_done` after about `10.52s`) and then stayed at `first_run_start` with zero result rows.
  - Stopped manually after `4 minutes and 55.83 seconds`; final state `killed`, exit `0`.
- Interpretation:
  - `owned_jobs` is correct and was faster on a small checked shape, but target regresses by about `9.7%` vs the fixed-wait median. Reducing rectangular scan work alone is not enough; the altered receiver order likely hurts the target scheduling/CTA cadence.
  - Multi-candidate ready scan is not a viable near-term path. It can deadlock even on a checked smoke after successful compile, so keep it gated off.
  - Bottleneck judgment remains software-structure/scheduling-bound: fixed-wait has a sharp fast pocket but slow tails, while attempts to reduce work or add speculative readiness checks perturb the schedule more than they remove overhead.
- Next action:
  - Keep `fixed_wait` as the speed baseline.
  - Test narrower fixed-wait variants that reduce release/done semaphore overhead or combine N work without changing producer count or receiver ownership.

### 2026-07-02 03:01 PDT - FWD-SWQ-080 fixed-wait N job granularity sweep
- Question:
  - Can the current fixed-wait target pocket improve by computing multiple N groups per receiver job, reducing full/done semaphore waits, done signals, and receiver candidate work per received block?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-095941`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - `implementation=m_n_slots`, `queue_mode=routing`, `routing=uniform`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `lowering_semantics=lane`, `metadata_mode=static_recv`, `output_mode=perf`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `receiver_schedule=fixed_wait`, `warmup=2`, `steps=7`, `repeat_runs=6`.

  | n_groups_per_job | median steady_state_time | median W13 TFLOP/s/rank | median send GB/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|---:|
  | 1 | `0.010455207799s` | `176.172684` | `68.817455` | `119.486605` | `197.413588` |
  | 2 | `0.011545417847s` | `157.774537` | `61.630679` | `128.538772` | `187.557457` |
  | 5 | `0.017643561570s` | `103.541350` | `40.445840` | `90.216763` | `109.394312` |
  | 10 | `0.028281084428s` | `64.455824` | `25.178056` | `58.577082` | `67.530847` |

- Interpretation:
  - Reducing the number of receiver N jobs is strongly negative. `n_groups_per_job=2` already regresses, and the larger values collapse WGMMA utilization.
  - The full/done semaphore count is not the dominant cost when traded against N-tile CTA parallelism. Keep one N group per job in the current fixed-wait structure.
  - The baseline row itself showed the same slow-tail issue as previous runs: fast repeats reached `194.73`, `195.74`, and `197.41 TFLOP/s/rank`, but slow repeats at `134.88`, `157.61`, and `119.49` pulled the median down.
- Next action:
  - Do not pursue `n_groups_per_job > 1`.
  - Look for a way to preserve `n_groups_per_job=1` while stabilizing the slow-tail scheduling pocket.

### 2026-07-02 03:06 PDT - FWD-SWQ-081 owned-slots receiver schedule
- Question:
  - Can we preserve fixed-wait round/slot order but prune each worker's per-slot job loop to only the N jobs it owns?
  - This is narrower than `owned_jobs`: it keeps slot order and release shape, but avoids scanning all `n_compute_jobs` for every worker.
- Code state:
  - Tested the existing experimental `receiver_schedule=owned_slots` in `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
- Checked smoke:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-100338`, cluster `cw-us-east-02a`, succeeded.
  - Config: `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=1280`, `experts_per_rank=32`, `entries_per_rank=56`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, `receiver_schedule=fixed_wait,owned_slots`, checked.

  | receiver_schedule | steady_state_time | W13 TFLOP/s/rank | metadata_mismatches | hidden_max_abs_diff |
  |---|---:|---:|---:|---:|
  | `fixed_wait` | `0.001785373703s` | `17.584198` | `0` | `0.044435501` |
  | `owned_slots` | `0.001740596335s` | `18.036557` | `0` | `0.044435501` |

- Target repeat:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-100529`, cluster `cw-us-east-02a`, succeeded.
  - Config: target current fixed pocket, `tokens_per_rank=32768`, `entries_per_rank=288`, `routing=uniform`, `metadata_mode=static_recv`, `output_mode=perf`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | receiver_schedule | median steady_state_time | median W13 TFLOP/s/rank | median send GB/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
  |---|---:|---:|---:|---:|---:|
  | `fixed_wait` | `0.009798858913s` | `186.118165` | `72.702408` | `135.195429` | `197.602250` |
  | `owned_slots` | `0.010483440710s` | `176.059348` | `68.773183` | `132.986509` | `198.042040` |

- Interpretation:
  - `owned_slots` is correct on the checked smoke and can hit the same fast mode as fixed-wait (`198.04 TFLOP/s/rank` max), but it worsens the target median.
  - Receiver job-loop pruning does not remove the slow-tail source. Like `owned_jobs`, it perturbs schedule cadence enough to lose median throughput.
  - Current best remains plain `fixed_wait` with the target source-push pocket. The unresolved problem is run-to-run/kernel-call slow-tail instability, not an obvious surplus job-loop scan.
- Next action:
  - Stop spending target runs on receiver-pruning variants unless a new mechanism also changes slot refill/scheduling stability.
  - Shift to diagnosing why the same kernel alternates between fast and slow modes.

### 2026-07-02 03:10 PDT - FWD-SWQ-082 per-launch slow-mode diagnostics
- Question:
  - Is the fixed-wait slow tail an artifact of averaging seven kernel launches per row, or does it show up on individual launches?
  - Does the same per-launch slow mode remain when WGMMA/hidden compute is removed?
- H100 jobs:
  - Full W13 path: `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-100753`, cluster `cw-us-east-02a`, succeeded.
  - Send-only perf path: `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-100934`, cluster `cw-us-east-02a`, succeeded.
- Shared config:
  - Target current fixed pocket, `routing=uniform`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `block_m=64`, `block_k=128`, `block_n=128`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=1`, `repeat_runs=32`, `--no-check`.
- Full W13 path result:
  - Median `steady_state_time=0.010136504541s`, median `179.755242 TFLOP/s/rank`.
  - Range: `67.358396` to `195.556826 TFLOP/s/rank`.
  - `9/32` launches were `>=190 TFLOP/s/rank`; `13/32` launches were `<160 TFLOP/s/rank`.
  - Slow launches appeared in contiguous bands, especially repeats `10-16` and `20-25`, then returned to fast mode.
- Send-only perf result:
  - Median `steady_state_time=0.005503618042s`, median `129.281551 GB/s/rank`.
  - Range: `125.767321` to `132.757171 GB/s/rank`.
  - `32/32` launches were `>=110 GB/s/rank`; `0/32` launches were `<70 GB/s/rank`.
- Interpretation:
  - The slow tail is per-kernel-launch behavior, not just seven-launch averaging.
  - In perf mode, send-only source-push copy/semaphore/inbox output is stable. The full-path slow mode requires WGMMA/hidden compute interacting with the source-push protocol.
  - This supersedes the earlier send-only debug-output result: debug `seen_*` outputs could be noisy, but the current perf send-only path is not.
  - Current bottleneck judgment: WGMMA/producer CTA coexistence or hidden-output compute/store scheduling, not raw remote-copy throughput alone and not receiver scan overhead.
- Next action:
  - Run the same per-launch diagnostic on compute-only or copy/compute-separated variants if available, then test whether changing compute CTA occupancy or producer/compute overlap shape removes the slow bands.

### 2026-07-02 03:16 PDT - FWD-SWQ-083 hidden sink no-result and fine CTA-count step=1 sweep
- Questions:
  - Does replacing the full hidden scatter with the compact `hidden_output_mode=sink` path remove full-path slow bands?
  - Do adjacent total CTA counts around the current `num_sms=32` pocket change the per-launch slow-mode behavior?
- Hidden sink job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-101240`, cluster `cw-us-east-02a`, killed.
  - Config: current target pocket with `hidden_output_mode=sink`, `steps=1`, `repeat_runs=24`.
  - Result: no kernel row. The job stayed before `device_put_start` after `make_inputs_start` and was stopped after `1 minute and 32.72 seconds`.
  - Interpretation: no performance signal; treat as a host-side harness stall/no-result.
- Fine CTA-count H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-101447`, cluster `cw-us-east-02a`, succeeded.
  - Config: current target fixed-wait pocket with `num_send_sms=2`, sweep `num_sms=31,32,33`, `warmup=2`, `steps=1`, `repeat_runs=16`, `output_mode=perf`, `--no-check`.

  | num_sms | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 31 | `0.011112826527s` | `163.909244` | `72.028581` | `166.038985` | `0/16` | `6/16` |
  | 32 | `0.009233551566s` | `197.267713` | `195.978266` | `199.878653` | `16/16` | `0/16` |
  | 33 | `0.009244400077s` | `197.036235` | `193.908557` | `199.106016` | `16/16` | `0/16` |

- Interpretation:
  - `num_sms=31` is not viable; after early slow launches it stabilizes around `164-166 TFLOP/s/rank`.
  - `num_sms=32` and `33` both showed stable fast per-launch behavior in this process, unlike the earlier single-config step=1 run where `num_sms=32` had contiguous slow bands.
  - The slow mode is therefore not a deterministic property of the `num_sms=32` kernel. It is sensitive to run/process/GPU state or burn-in, while the fast-mode ceiling remains about `198-200 TFLOP/s/rank`.
  - This improves the stability diagnosis but does not yet cross the `210 TFLOP/s/rank` target.
- Next action:
  - Re-test `num_sms=32/33` with normal `steps=7` after a longer burn-in to see whether the stable fast per-launch mode carries over to the benchmark metric.

### 2026-07-02 03:18 PDT - FWD-SWQ-084 long-warmup steps=7 no-result
- Question:
  - Does the stable fast per-launch mode from FWD-SWQ-083 carry over to the normal benchmark metric with a longer burn-in?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-101714`, cluster `cw-us-east-02a`, killed.
- Config:
  - Current target fixed-wait pocket, `num_send_sms=2`, sweep `num_sms=32,33`, `warmup=24`, `steps=7`, `repeat_runs=8`, `output_mode=perf`, `--no-check`.
- Result:
  - No kernel row.
  - The job emitted `validate_start`, `mesh_start`, and `make_inputs_start` for `num_sms=32`, then never reached `device_put_start`.
  - Stopped manually after `1 minute and 34.53 seconds`; final state `killed`, exit `0`.
- Interpretation:
  - No performance signal. Treat this as a host setup/input-construction stall rather than a kernel result.
  - The best actionable signal remains FWD-SWQ-083: with `steps=1`, `num_sms=32/33` can run in a stable fast mode around `197 TFLOP/s/rank`, but current source-push W13 still does not demonstrate a stable `>=210 TFLOP/s/rank` path.

### 2026-07-02 03:24 PDT - FWD-SWQ-085 burn-in normal-steps stability result
- Question:
  - Does the stable fast per-launch mode from FWD-SWQ-083 carry over to the normal `steps=7` metric if the job avoids the host-side no-result seen with `warmup=24`?
- H100 job:
  - `/dlwh/repro-source-push-fixed-burnin-normal-20260702-1020`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `num_send_sms=2`, sweep `num_sms=32,33`, `warmup=16`, `steps=7`, `repeat_runs=8`, `output_mode=perf`, `--no-check`.

  | num_sms | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 32 | `0.010824148211s` | `170.192751` | `117.910635` | `197.501167` | `3/8` | `4/8` |
  | 33 | `0.011810014207s` | `154.239423` | `114.277907` | `195.996906` | `2/8` | `5/8` |

- Interpretation:
  - Longer burn-in does not make the stable `steps=1` fast pocket survive the normal seven-launch timing loop.
  - Both `num_sms=32` and `33` can still hit fast individual rows around `195-198 TFLOP/s/rank`, but the median remains dominated by slow rows.
  - Current bottleneck judgment remains software-structure / WGMMA-producer interaction. Send-only is stable, and CTA-count changes can influence fast-mode entry, but the full compute path still has slow-mode launches under the benchmark metric.
- Next action:
  - Stop treating warmup/CTA-count alone as the fix.
  - Add or run a tighter decomposition that separates full hidden stores, WGMMA compute, and producer protocol under the same repeated-call metric.

### 2026-07-02 03:29 PDT - FWD-SWQ-086 hidden-output sink decomposition
- Question:
  - Is the slow normal-metric tail tied to full hidden scatter/output footprint, or does it remain when WGMMA writes only a compact per-slot sink?
- H100 job:
  - `/dlwh/repro-source-push-hidden-sink-paired-20260702-102544`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=8`, `output_mode=perf`, `--no-check`.

  | hidden_output_mode | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---|---:|---:|---:|---:|---:|---:|
  | `full` | `0.011921322139s` | `153.267857` | `117.833982` | `196.789983` | `2/8` | `4/8` |
  | `sink` | `0.009300199504s` | `195.854501` | `123.755533` | `197.208545` | `5/8` | `2/8` |

- Interpretation:
  - Compact sink output materially improves the median and fast-row frequency, so full hidden output footprint/scatter is part of the normal benchmark tax.
  - The fast-mode ceiling remains about `197 TFLOP/s/rank`, and the sink mode still has slow rows. Full hidden stores are not the whole slow-mode mechanism.
  - Current bottleneck judgment: WGMMA/source-push interaction is still the primary instability source; full hidden scatter increases how often the slow mode appears.

### 2026-07-02 03:35 PDT - FWD-SWQ-087 store-zero decomposition
- Question:
  - With the same `m_n_slots` fixed-wait receiver/job/semaphore schedule, how much time is output store/protocol without WGMMA?
- Code change:
  - Added harness-only `hidden_compute_mode=store_zero`.
  - This keeps the source-push receiver schedule and N-tile job structure but replaces WGMMA/SwiGLU with zero hidden stores.
- H100 job:
  - `/dlwh/repro-source-push-store-zero-20260702-102957`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=8`, `output_mode=perf`, `hidden_compute_mode=store_zero`, `--no-check`.

  | hidden_output_mode | median steady_state_time | effective TFLOP/s/rank denominator | min effective TF | max effective TF | fast-equivalent >=300 | slow-equivalent <200 |
  |---|---:|---:|---:|---:|---:|---:|
  | `full` | `0.005726062204s` | `318.273186` | `314.710574` | `322.773802` | `8/8` | `0/8` |
  | `sink` | `0.005612719288s` | `324.528725` | `144.152636` | `326.583586` | `5/8` | `2/8` |

- Interpretation:
  - Store/protocol-only time is around `0.0056-0.0058s`, much faster than WGMMA sink (`~0.0093s`) or WGMMA full (`~0.0119s`) in FWD-SWQ-086.
  - Full-output zero stores are stable in this run. Therefore the full-output store path alone is not enough to create the observed slow WGMMA tail.
  - The slow-mode mechanism requires WGMMA/hidden compute interacting with the source-push receiver/protocol. Compact sink reduces the frequency but does not remove it.
- Next action:
  - Test the existing `compute_pipeline=emit` WGMMA staging under the same fixed-wait target schedule.

### 2026-07-02 03:37 PDT - FWD-SWQ-088 emit pipeline default SMEM failure
- Question:
  - Does Mosaic `compute_pipeline=emit` improve WGMMA staging stability or fast-mode ceiling versus the manual K loop?
- H100 job:
  - `/dlwh/repro-source-push-emit-pipeline-20260702-103420`, cluster `cw-us-east-02a`, succeeded with error rows.
- Config:
  - Current target fixed-wait pocket, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=8`, `compute_pipeline=emit`, swept `hidden_output_mode=sink,full`.
- Result:
  - Both rows failed during lowering:
    - `ValueError: Mosaic GPU kernel exceeds available shared memory: smem_bytes=327712 > max_smem_bytes=232448`
- Interpretation:
  - Default emit-pipeline staging is not directly viable at this tile shape.
- Next action:
  - Retry emit with smaller `max_concurrent_steps` to see whether a lower-buffered emit path fits and times.

### 2026-07-02 03:39 PDT - FWD-SWQ-089 emit reduced-buffer sweep
- Question:
  - Can smaller `max_concurrent_steps` make `compute_pipeline=emit` fit and improve the WGMMA sink path?
- H100 job:
  - `/dlwh/repro-source-push-emit-mcs-sweep-20260702-103652`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `num_send_sms=2`, `num_sms=32`, `hidden_output_mode=sink`, `compute_pipeline=emit`, swept `max_concurrent_steps=1,2,3`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.
- Results:
  - `max_concurrent_steps=1` failed lowering:
    - `ValueError: max_concurrent_steps must be greater than all delay_release values, but max_concurrent_steps=1 and delay_release_levels=[1].`
  - `max_concurrent_steps=2` fit and timed:

    | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank |
    |---:|---:|---:|---:|
    | `0.022424937292s` | `81.225783` | `71.873548` | `89.749197` |

  - `max_concurrent_steps=3` failed lowering:
    - `ValueError: Mosaic GPU kernel exceeds available shared memory: smem_bytes=245784 > max_smem_bytes=232448`
- Interpretation:
  - Emit-pipeline staging is not a performance path at this tile shape. The only fitting setting is much slower than manual WGMMA sink (`~195.85 TFLOP/s/rank` in FWD-SWQ-086).
- Next action:
  - Stop pursuing emit for this repro.
  - Tune manual-loop tile geometry that changes K-loop/barrier count without increasing SMEM beyond the current viable path.

### 2026-07-02 03:43 PDT - FWD-SWQ-090 manual block_k sink sweep
- Question:
  - Does changing manual WGMMA K tile size improve the source-push sink path by reducing K-loop/barrier count or improving WGMMA staging?
- H100 job:
  - `/dlwh/repro-source-push-blockk-sink-20260702-103952`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `hidden_output_mode=sink`, `compute_pipeline=manual`, `num_send_sms=2`, `num_sms=32`, swept `block_k=64,128,256`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | block_k | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 64 | `0.010711036855s` | `170.175136` | `103.360957` | `171.216860` | `0/8` | `2/8` |
  | 128 | `0.009197200283s` | `198.047402` | `131.252848` | `199.844759` | `6/8` | `1/8` |
  | 256 | `0.023309882281s` | `78.150542` | `73.756341` | `87.122337` | `0/8` | `8/8` |

- Interpretation:
  - `block_k=128` remains the only good K tile in this source-push sink pocket.
  - `block_k=64` likely pays too much loop/barrier overhead; `block_k=256` likely hurts occupancy/SMEM or WGMMA scheduling badly.
- Next action:
  - Sweep `block_n` at `block_k=128` to test output-tile width / CTA-count effects.

### 2026-07-02 03:46 PDT - FWD-SWQ-091 manual block_n sink sweep
- Question:
  - Does changing output tile width improve CTA count, accumulator shape, or slow-tail stability for the manual WGMMA sink path?
- H100 job:
  - `/dlwh/repro-source-push-blockn-sink-20260702-104304`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `hidden_output_mode=sink`, `compute_pipeline=manual`, `block_k=128`, `num_send_sms=2`, `num_sms=32`, swept `block_n=64,128,256`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | block_n | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 64 | `0.010833995426s` | `168.309840` | `109.044382` | `176.042405` | `0/8` | `3/8` |
  | 128 | `0.011294078147s` | `181.172568` | `107.388406` | `199.179513` | `4/8` | `3/8` |
  | 256 | `0.028477389432s` | `63.972562` | `57.729078` | `68.906014` | `0/8` | `8/8` |

- Interpretation:
  - `block_n=128` remains the only good output tile, but this run reproduced the launch-to-launch slow-tail instability even in sink mode.
  - `block_n=64` adds too many CTAs/jobs; `block_n=256` is a severe occupancy/scheduling regression.
- Next action:
  - Test `block_m=128` with adjusted `entries_per_rank=160` to trade larger CTAs/padding for fewer live blocks and fewer semaphore/job events.

### 2026-07-02 03:49 PDT - FWD-SWQ-092 block_m=128 adjusted-capacity sink test
- Question:
  - Can `block_m=128` improve the source-push sink path by reducing live block count, slot traffic, and receiver jobs?
- H100 job:
  - `/dlwh/repro-source-push-blockm128-sink-20260702-104627`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - `block_m=128`, `entries_per_rank=160`, `block_k=128`, `block_n=128`, `hidden_output_mode=sink`, `compute_pipeline=manual`, current fixed-wait target schedule, `num_send_sms=2`, `num_sms=32`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|
  | `0.011908851139s` | `161.366736` | `110.851990` | `162.732348` | `0/8` | `3/8` |

- Interpretation:
  - `block_m=128` is slower despite fewer live blocks and semaphore/job events.
  - Larger CTAs/padding/occupancy cost dominates. Keep `block_m=64`.
- Next action:
  - Test the existing `direct_self_compute` path to remove source-push protocol for local self-routes while preserving remote source-push behavior.

### 2026-07-02 03:54 PDT - FWD-SWQ-093 direct-self sink test
- Question:
  - Does skipping source-push for local self-routes reduce protocol pressure enough to improve the sink path?
- H100 job:
  - `/dlwh/repro-source-push-direct-self-sink-20260702-104906`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `hidden_output_mode=sink`, `block_m=64`, `block_k=128`, `block_n=128`, `num_send_sms=2`, `num_sms=32`, `direct_self_compute=True`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|
  | `0.009616135071s` | `189.551055` | `121.539628` | `196.230145` | `4/8` | `1/8` |

- Interpretation:
  - Direct self compute is not a speedup at this target config.
  - It reduces counted send bytes for self routes and changes schedule shape, but the WGMMA sink fast ceiling remains under `200 TFLOP/s/rank`.
- Next action:
  - Sweep send/compute CTA allocation in sink mode to see whether the slow-tail frequency responds to producer pressure.

### 2026-07-02 03:58 PDT - FWD-SWQ-094 send-SM allocation sink sweep
- Question:
  - Does producer CTA allocation change slow-tail frequency or sink-path throughput?
- H100 job:
  - `/dlwh/repro-source-push-send-sms-sink-20260702-105406`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `hidden_output_mode=sink`, `block_m=64`, `block_k=128`, `block_n=128`, `num_sms=32`, swept `num_send_sms=1,2,3,4`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | num_send_sms | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | `0.010948145298s` | `166.265372` | `117.405066` | `168.238500` | `0/8` | `3/8` |
  | 2 | `0.009461618078s` | `192.514468` | `118.640292` | `198.800220` | `5/8` | `2/8` |
  | 3 | `0.010942319929s` | `166.452003` | `108.701753` | `168.215376` | `0/8` | `3/8` |
  | 4 | `0.012279707639s` | `148.893065` | `111.048242` | `164.103263` | `0/8` | `4/8` |

- Interpretation:
  - `num_send_sms=2` remains the best allocation.
  - The slow-tail issue is not fixed by simply adding or removing producer CTAs. Too few or too many send CTAs regress the compute schedule.
- Next action:
  - Sweep total `num_sms` in sink mode around the current pocket while keeping `num_send_sms=2`.

### 2026-07-02 04:02 PDT - FWD-SWQ-095 total num_sms sink sweep
- Question:
  - Does changing total producer+compute CTA count stabilize the normal `steps=7` sink path?
- H100 job:
  - `/dlwh/repro-source-push-num-sms-sink-20260702-105750`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `hidden_output_mode=sink`, `num_send_sms=2`, swept `num_sms=28,30,32,34,36`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | num_sms | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 28 | `0.011688691431s` | `155.971801` | `108.156654` | `164.423576` | `0/8` | `5/8` |
  | 30 | `0.011380845348s` | `160.050163` | `98.885528` | `163.009789` | `0/8` | `4/8` |
  | 32 | `0.009361910933s` | `194.908301` | `115.486052` | `199.042827` | `6/8` | `2/8` |
  | 34 | `0.013562356138s` | `134.304564` | `100.884211` | `136.134647` | `0/8` | `8/8` |
  | 36 | `0.012745696300s` | `142.754081` | `94.883516` | `143.728891` | `0/8` | `8/8` |

- Interpretation:
  - Total `num_sms=32` remains the only good pocket.
  - The slow-tail is sensitive to CTA count, but moving away from 32 quickly collapses WGMMA throughput.
- Next action:
  - Sweep inbox slot count to test producer/receiver pacing and slot reuse depth without changing WGMMA tile shape.

### 2026-07-02 04:06 PDT - FWD-SWQ-096 inbox slot sink sweep
- Question:
  - Does increasing inbox buffering reduce producer/receiver pacing stalls and stabilize the sink path?
- H100 job:
  - `/dlwh/repro-source-push-inbox-slots-sink-20260702-110202`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Current target fixed-wait pocket, `hidden_output_mode=sink`, `num_send_sms=2`, `num_sms=32`, swept `inbox_slots=2,4,8`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | inbox_slots | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | fast >=210 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 2 | `0.014361527651s` | `121.321169` | `105.396758` | `150.449475` | `0/8` | `0/8` | `8/8` |
  | 4 | `0.011568964708s` | `162.863359` | `114.070002` | `197.191185` | `5/8` | `0/8` | `4/8` |
  | 8 | `0.008729882289s` | `208.649496` | `132.120815` | `210.142061` | `6/8` | `2/8` | `1/8` |

- Interpretation:
  - `inbox_slots=8` is a real improvement and the first run in this pass to hit `>=210 TFLOP/s/rank` on some normal `steps=7` rows.
  - Median is still below the stable target (`~208.65 TFLOP/s/rank`) because one slow row and one mid row remain.
  - Bottleneck hypothesis updates: receiver/producer pacing and slot reuse depth are a major part of the slow-tail mechanism. Larger buffering reduces, but does not fully eliminate, WGMMA slow mode.
- Next action:
  - Focus on the `inbox_slots=8` pocket: repeat for stability and sweep nearby total CTA counts before touching more structural code.

### 2026-07-02 04:10 PDT - FWD-SWQ-097 slots=8 total CTA focused sweep
- Question:
  - In the improved `inbox_slots=8` sink pocket, does a nearby total CTA count push median throughput over `210 TFLOP/s/rank`?
- H100 job:
  - `/dlwh/repro-source-push-slots8-num-sms-20260702-110551`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - `inbox_slots=8`, `hidden_output_mode=sink`, `num_send_sms=2`, swept `num_sms=31,32,33`, target shape, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=7`, `repeat_runs=12`, `--no-check`.

  | num_sms | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=210 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 31 | `0.008889216091s` | `204.679486` | `113.011646` | `207.156522` | `0/12` | `3/12` |
  | 32 | `0.008793331013s` | `207.269478` | `108.022979` | `209.591879` | `0/12` | `3/12` |
  | 33 | `0.008644239012s` | `210.718777` | `126.409954` | `213.667406` | `6/12` | `2/12` |

- Interpretation:
  - `inbox_slots=8`, `num_sms=33`, `num_send_sms=2` is the best sink-output pocket so far and crosses the `210 TFLOP/s/rank` median target in this 12-repeat run.
  - This is still a diagnostic sink-output path, not full hidden output. The full-output path must be measured before treating this as production-relevant forward speed.
- Next action:
  - Run the same `inbox_slots=8`, `num_sms=33` pocket with `hidden_output_mode=full`.

### 2026-07-02 04:13 PDT - FWD-SWQ-098 slots=8 full-output measurement
- Question:
  - Does the `inbox_slots=8`, `num_sms=33` sink-output 210+ pocket survive full hidden materialization?
- H100 job:
  - `/dlwh/repro-source-push-slots8-full-20260702-110941`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `hidden_output_mode=full`, target shape, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=7`, `repeat_runs=12`, `--no-check`.

  | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=210 | slow <160 |
  |---:|---:|---:|---:|---:|---:|
  | `0.010121502215s` | `182.902172` | `107.452429` | `215.106489` | `4/12` | `6/12` |

- Interpretation:
  - Full output can hit faster individual rows than the sink run (`215.11 TFLOP/s/rank` max), but slow rows are frequent enough to pull the median down.
  - The sink-output 210+ median is therefore not yet a production-materialization win.
  - Full hidden materialization interacts with WGMMA/source-push pacing; store-only full output was stable in FWD-SWQ-087, so the full-output slow tail still requires WGMMA.
- Next action:
  - Try deeper buffering (`inbox_slots=16`) for the full-output path.

### 2026-07-02 04:16 PDT - FWD-SWQ-099 slots=16 full-output measurement
- Question:
  - Does deeper buffering (`inbox_slots=16`) fix the full-output slow-tail seen at `inbox_slots=8`?
- H100 job:
  - `/dlwh/repro-source-push-slots16-full-20260702-111312`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - `inbox_slots=16`, `num_send_sms=2`, `num_sms=33`, `hidden_output_mode=full`, target shape, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=7`, `repeat_runs=12`, `--no-check`.

  | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=210 | slow <160 |
  |---:|---:|---:|---:|---:|---:|
  | `0.008865553852s` | `205.456016` | `146.911398` | `209.424840` | `0/12` | `2/12` |

- Interpretation:
  - `inbox_slots=16` removes most of the full-output slow-tail but also lowers the fast ceiling compared with the best `slots=8` full rows.
  - This is a better full-output median than `slots=8` (`205.46` vs `182.90 TFLOP/s/rank`) but does not reach `210`.
- Next action:
  - Sweep nearby `num_sms` for `inbox_slots=16` full output.

### 2026-07-02 04:20 PDT - FWD-SWQ-100 slots=16 full-output num_sms sweep
- Question:
  - Can nearby total CTA counts improve the `inbox_slots=16` full-output pocket?
- H100 job:
  - `/dlwh/repro-source-push-slots16-full-num-sms-20260702-111625`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - `inbox_slots=16`, `hidden_output_mode=full`, `num_send_sms=2`, swept `num_sms=32,33,34,35`, target shape, `warmup=2`, `steps=7`, `repeat_runs=8`, `--no-check`.

  | num_sms | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=210 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 32 | `0.010251194342s` | `179.480750` | `129.656141` | `200.481863` | `0/8` | `4/8` |
  | 33 | `0.009717192368s` | `188.640718` | `128.552792` | `207.166075` | `0/8` | `2/8` |
  | 34 | `0.011824262491s` | `154.402426` | `102.283443` | `162.373991` | `0/8` | `4/8` |
  | 35 | `0.011688236014s` | `156.049376` | `108.215546` | `163.079219` | `0/8` | `4/8` |

- Interpretation:
  - This smaller sweep did not reproduce the earlier `205.46 TFLOP/s/rank` median for `num_sms=33`; run-to-run variance remains substantial.
  - No slots=16 full-output configuration crosses `210`.
  - Best confirmed diagnostic path remains `inbox_slots=8`, `num_sms=33`, sink output from FWD-SWQ-097.
- Current bottleneck judgment:
  - Sink output can cross 210 median with enough buffering and the right CTA count.
  - Full hidden materialization still causes WGMMA/source-push slow-tail variance. Store-only full output is stable, so the full-output issue is not pure store bandwidth; it is the combination of WGMMA, output materialization, and source-push pacing.
- Next action:
  - Treat `inbox_slots=8`, `num_sms=33`, `num_send_sms=2` as the best diagnostic schedule.
  - For production relevance, prefer avoiding full hidden materialization by fusing/streaming into the next stage, or add a separate output-consumer schedule instead of further blind tile sweeps.

### 2026-07-02 03:29 PDT - FWD-SWQ-086 producer send double-buffer smoke
- Question:
  - Can the source-push producer legally keep two SMEM-to-GMEM copies in flight before waiting, and does that preserve correctness?
- Code change:
  - Added `send_pipeline_depth` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - `send_pipeline_depth=1` preserves the existing single-SMEM-buffer path.
  - `send_pipeline_depth=2` uses two producer SMEM tiles and waits with `mgpu.wait_smem_to_gmem(1)` before reusing a tile, then `wait_smem_to_gmem(0)` before signaling the inbox slot full.
  - Depth 2 is currently allowed only for `peer_loop in {"static", "switch", "grid_switch"}`; dynamic/grid sender loops remain single-buffered.
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-102549`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Checked small uniform all-to-all smoke, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=debug`, `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=1280`, `entries_per_rank=56`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=1`, `steps=3`, `repeat_runs=1`, `--check`.

  | send_pipeline_depth | steady_state_time | W13 TFLOP/s/rank | max_abs_diff | metadata_mismatches |
  |---:|---:|---:|---:|---:|
  | 1 | `0.001798146327s` | `17.459294` | `0.0444355011` | `0` |
  | 2 | `0.001744095003s` | `18.000376` | `0.0444355011` | `0` |

- Interpretation:
  - Two pending producer copies compile, run, and preserve the checked result at smoke shape.
  - The small-shape signal is only a legality/correctness check; the speedup is too small to claim target-path value.
- Next action:
  - Run target repeated perf with `send_pipeline_depth=1,2` to see whether producer-copy buffering reduces the full W13 slow-mode bands or improves the fast-mode ceiling.

### 2026-07-02 03:34 PDT - FWD-SWQ-087 target producer send double-buffer result
- Question:
  - Does producer-side double buffering reduce the repeated-launch full-W13 slow mode at target shape?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-103001`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=1`, `repeat_runs=16`, `--no-check`.

  | send_pipeline_depth | median steady_state_time | median W13 TFLOP/s/rank | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | `0.009581376449s` | `190.164023` | `77.866541` | `196.413938` | `8/16` | `5/16` |
  | 2 | `0.009104238939s` | `200.069752` | `185.835695` | `204.200094` | `15/16` | `0/16` |

- Depth 1 TFLOP rows:
  - `196.413938, 196.225680, 195.393448, 196.316367, 185.760759, 119.870520, 77.866541, 165.285197, 141.191185, 89.111128, 142.592685, 186.854920, 195.763827, 195.002174, 194.901120, 193.473126`
- Depth 2 TFLOP rows:
  - `185.835695, 197.059849, 198.967216, 200.248455, 203.866550, 199.779717, 199.142934, 202.600809, 199.516249, 202.696672, 197.528409, 202.907930, 201.214476, 201.687592, 204.200094, 199.891048`
- Interpretation:
  - Producer double buffering is useful: it removes the severe `<160 TFLOP/s/rank` slow launches in this target step-1 repeated run.
  - It does not reach the stage-1 target; the observed fast-mode ceiling is still about `204 TFLOP/s/rank`, below `210`.
  - Current bottleneck judgment: still software-structure / CTA coexistence, but less dominated by producer-copy wait serialization. Next useful knob is CTA allocation under the depth-2 path.
- Next action:
  - Sweep nearby `num_send_sms` and `num_sms` with `send_pipeline_depth=2` to see whether the stabilized path has a better compute/producer allocation.

### 2026-07-02 03:39 PDT - FWD-SWQ-088 depth-2 CTA allocation screen
- Question:
  - Once producer double buffering stabilizes the path, is there a better sender/compute CTA split than `num_send_sms=2`, `num_sms=32`?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-103448`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `send_pipeline_depth=2`, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=288`, `inbox_slots=4`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=1`, `repeat_runs=8`, `--no-check`.
  - Swept `num_send_sms in {1,2,3}` and `num_sms in {30,32,34,36}`.

  | num_send_sms | num_sms | median W13 TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 2 | 32 | `202.090737` | `0.009013194474s` | `197.295991` | `206.849809` | `8/8` | `0/8` |
  | 3 | 30 | `199.632144` | `0.009124219068s` | `157.204719` | `204.744471` | `7/8` | `1/8` |
  | 1 | 32 | `188.275815` | `0.009674541419s` | `83.887018` | `189.466785` | `0/8` | `1/8` |
  | 1 | 30 | `174.164749` | `0.010458382661s` | `170.779086` | `175.473157` | `0/8` | `0/8` |
  | 3 | 32 | `165.337279` | `0.011016816483s` | `163.163128` | `167.206972` | `0/8` | `0/8` |
  | 2 | 30 | `160.479901` | `0.011350225075s` | `160.072437` | `162.884679` | `0/8` | `0/8` |
  | 3 | 34 | `148.767223` | `0.012243957957s` | `134.805010` | `150.529815` | `0/8` | `8/8` |
  | 3 | 36 | `147.075783` | `0.012384790112s` | `142.641449` | `149.152805` | `0/8` | `8/8` |
  | 2 | 36 | `143.420241` | `0.012700309046s` | `142.023091` | `146.018434` | `0/8` | `8/8` |
  | 2 | 34 | `133.323603` | `0.013663021964s` | `70.373342` | `136.315786` | `0/8` | `8/8` |
  | 1 | 34 | `104.280438` | `0.017511342070s` | `79.881167` | `116.683913` | `0/8` | `8/8` |
  | 1 | 36 | `102.850096` | `0.017941443482s` | `60.730091` | `115.725349` | `0/8` | `8/8` |

- Interpretation:
  - The original `num_send_sms=2`, `num_sms=32` split remains best under double buffering.
  - Higher total CTA counts (`34`, `36`) are consistently bad for this path; widening total CTAs is not the route to `210`.
  - `num_send_sms=3`, `num_sms=30` is near the best but less stable and still below the current best.
- Next action:
  - Keep `num_send_sms=2`, `num_sms=32` and test deeper producer copy buffering (`send_pipeline_depth=3,4`) before considering a more structural producer/consumer split.

### 2026-07-02 03:51 PDT - FWD-SWQ-089 deeper producer buffering smokes
- Question:
  - Are producer copy depths 3 and 4 legal/correct, and do they look promising enough for target timing?
- Code change:
  - Extended `send_pipeline_depth` validation to allow `1,2,3,4`.
  - Added static producer-copy branches for depths 3 and 4 using three/four SMEM tiles and `mgpu.wait_smem_to_gmem(2/3)` before reusing a tile.
- Multi-depth smoke no-result:
  - Job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-104055`, cluster `cw-us-east-02a`, stopped.
  - Config swept `send_pipeline_depth=2,3,4` at small checked shape.
  - Result: no kernel row; stuck before `device_put_start` after `make_inputs_start` for depth 2.
  - Interpretation: recurring host setup/no-result mode, not a kernel signal.
- Single-depth smoke results:

  | send_pipeline_depth | job | result | steady_state_time | W13 TFLOP/s/rank | max_abs_diff | metadata_mismatches |
  |---:|---|---|---:|---:|---:|---:|
  | 3 | `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-104547` | succeeded | `0.001829319013s` | `17.161777` | `0.0444355011` | `0` |
  | 4 | `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-104830` | succeeded | `0.003804757648s` | `8.251344` | `0.0444355011` | `0` |

- Interpretation:
  - Depth 3 and 4 are legal/correct at smoke shape.
  - Depth 4 is clearly too slow and should not be taken to target.
  - Depth 3 is correct but slower than prior depth-2 small smoke (`18.000376 TFLOP/s/rank`), so it only gets one target screen before declaring deeper buffering negative.
- Next action:
  - Run a single target screen for `send_pipeline_depth=3`, `num_send_sms=2`, `num_sms=32`.

### 2026-07-02 03:55 PDT - FWD-SWQ-090 target depth-3 producer buffering result
- Question:
  - Does producer copy depth 3 help at target shape even though it is slower at smoke shape?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-105141`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `send_pipeline_depth=3`, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=1`, `repeat_runs=8`, `--no-check`.
- Result:
  - Median `steady_state_time=0.009091190528s`, median `200.356748 TFLOP/s/rank`.
  - Range: `173.587496` to `202.563616 TFLOP/s/rank`.
  - `7/8` rows were `>=190 TFLOP/s/rank`; `0/8` rows were `<160`.
  - TFLOP rows: `200.361320, 199.778335, 199.351255, 201.636279, 200.352176, 202.563616, 201.354652, 173.587496`.
- Interpretation:
  - Depth 3 is worse than the depth-2 CTA screen best (`202.090737` median, `206.849809` max).
  - Deeper producer buffering beyond 2 is negative; keep `send_pipeline_depth=2`.
- Next action:
  - Move to the spec's physical tile candidates: test larger `block_k=256` and `block_m=128` under the depth-2 path.

### 2026-07-02 04:01 PDT - FWD-SWQ-091 tile screen A, block_m=64 block_k sweep
- Question:
  - Does the spec's `64 x 256` copy/compute tile improve the stabilized depth-2 path?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-105602`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `send_pipeline_depth=2`, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_n=128`, sweep `block_k in {128,256}`, `warmup=2`, `steps=1`, `repeat_runs=8`, `--no-check`.

  | block_k | median W13 TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank |
  |---:|---:|---:|---:|---:|
  | 128 | `198.060451` | `0.009196653962s` | `148.283591` | `200.364163` |
  | 256 | `87.746351` | `0.020758488565s` | `75.455626` | `87.981646` |

- Interpretation:
  - `block_k=256` is a hard loss in the `block_m=64` path.
  - Keep `block_k=128`.
- Next action:
  - Test `block_m=128` with reduced queue capacity (`entries_per_rank=160`) under the same depth-2 path.

### 2026-07-02 04:05 PDT - FWD-SWQ-092 tile screen B, block_m=128 block_k sweep
- Question:
  - Does the spec's `128 x 128` or `128 x 256` tile improve the stabilized depth-2 path by reducing queue/metadata overhead?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-110204`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `send_pipeline_depth=2`, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=160`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=128`, `block_n=128`, sweep `block_k in {128,256}`, `warmup=2`, `steps=1`, `repeat_runs=8`, `--no-check`.

  | block_k | median W13 TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 128 | `160.152317` | `0.012017086963s` | `157.286402` | `161.125454` | `0/8` | `4/8` |
  | 256 | `67.239333` | `0.028622489539s` | `66.005997` | `67.986773` | `0/8` | `8/8` |

- Interpretation:
  - `128 x 128` and `128 x 256` are both negative.
  - Original `64 x 128` remains the only viable tile among the tested physical copy candidates.
- Next action:
  - Test `n_group=2` under `block_m=64`, `block_k=128`, `send_pipeline_depth=2`; this targets N scheduling/adjacent-tile overhead rather than physical copy tile shape.

### 2026-07-02 04:09 PDT - FWD-SWQ-093 n_group screen
- Question:
  - Does computing two adjacent N tiles per entry reduce scheduling overhead enough to improve target performance?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-110605`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `send_pipeline_depth=2`, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, sweep `n_group in {1,2}`, `n_groups_per_job=1`, `warmup=2`, `steps=1`, `repeat_runs=8`, `--no-check`.

  | n_group | median W13 TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | fast >=190 | slow <160 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 1 | `199.403200` | `0.009134744061s` | `192.082613` | `204.125815` | `8/8` | `0/8` |
  | 2 | `79.965641` | `0.022793063428s` | `49.467314` | `84.392264` | `0/8` | `8/8` |

- Interpretation:
  - The adjacent-N grouped path is a hard loss.
  - Keep `n_group=1`.
- Next action:
  - Check the existing `compute_pipeline` mode as the last bounded non-structural WGMMA scheduling knob before returning to producer/consumer structure.

### 2026-07-02 04:14 PDT - FWD-SWQ-094 emit compute-pipeline screen
- Question:
  - Does `mgpu.emit_pipeline` improve receiver-side WGMMA scheduling versus the manual copy/WGMMA loop?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-111009`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `send_pipeline_depth=2`, `compute_pipeline=emit`, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, sweep `max_concurrent_steps in {1,2,4}`, `warmup=2`, `steps=1`, `repeat_runs=8`, `--no-check`.
- Result:
  - `max_concurrent_steps=1`: invalid, `ValueError: max_concurrent_steps must be greater than all delay_release values`.
  - `max_concurrent_steps=2`: median `steady_state_time=0.020948013524s`, median `86.953368 TFLOP/s/rank`, range `59.129428` to `89.445244`; `0/8 >=190`, `8/8 <160`.
  - `max_concurrent_steps=4`: compile failure, `ValueError: Mosaic GPU kernel exceeds available shared memory: smem_bytes=327712 > max_smem_bytes=232448`.
- Interpretation:
  - `compute_pipeline=emit` is a hard loss.
  - Manual receiver compute remains best.
- Next action:
  - Inspect and test `direct_self_compute` under all-to-all traffic to skip the self-rank inbox path.

### 2026-07-02 04:17 PDT - FWD-SWQ-095 direct-self compute smoke
- Question:
  - Does skipping the self-rank inbox send preserve correctness and look promising enough for target timing?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-111445`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Checked small uniform all-to-all, `direct_self_compute=true`, `send_pipeline_depth=2`, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=debug`, `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=1280`, `entries_per_rank=56`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=1`, `steps=3`, `repeat_runs=1`, `--check`.
- Result:
  - `steady_state_time=0.005313082676s`, `5.908880 TFLOP/s/rank`.
  - `max_abs_diff=0.0444355011`, `metadata_mismatches=0`.
  - `direct_self_entries_total=367`, `payload_send_entries_total=2627`.
- Interpretation:
  - Correctness passes, but smoke performance collapses versus the non-direct-self depth-2 smoke (`18.000376 TFLOP/s/rank`).
  - Do not take `direct_self_compute` to target.
- Next action:
  - Screen remaining receiver schedules (`ordered_wait`, `slot_group`) while keeping `ready_scan` disabled.

### 2026-07-02 04:24 PDT - FWD-SWQ-096 mixed receiver-schedule screen stopped
- Question:
  - Do `ordered_wait` or `slot_group` improve over `fixed_wait`?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-111827`, cluster `cw-us-east-02a`, stopped manually.
- Config:
  - Target uniform all-to-all, `send_pipeline_depth=2`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, sweep `receiver_schedule in {fixed_wait, ordered_wait, slot_group}`, `workers_per_slot=1`, `warmup=2`, `steps=1`, `repeat_runs=8`, `--no-check`.
- Partial result:
  - `fixed_wait`: median `steady_state_time=0.013173047453s`, median `146.759327 TFLOP/s/rank`, range `77.158404` to `199.214376`, `3/8 >=190`, `4/8 <160`.
  - `ordered_wait`: hung in `first_run_start` with no timing row.
  - `slot_group`: did not run because `ordered_wait` blocked the sweep.
- Interpretation:
  - `ordered_wait` is not viable in this harness shape; it appears to deadlock/hang.
  - The fixed-wait rows in this run were unusually noisy and are not a new best-signal baseline.
- Next action:
  - Run `slot_group` alone so it is not blocked by `ordered_wait`.

### 2026-07-02 04:28 PDT - FWD-SWQ-097 slot_group receiver schedule result
- Question:
  - Does `receiver_schedule=slot_group` improve over `fixed_wait` when run alone?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-112417`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `send_pipeline_depth=2`, `receiver_schedule=slot_group`, `workers_per_slot=1`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `entries_per_rank=288`, `inbox_slots=4`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=1`, `repeat_runs=8`, `--no-check`.
- Result:
  - Median `steady_state_time=0.028224333539s`, median `64.630832 TFLOP/s/rank`.
  - Range: `45.152868` to `67.953045 TFLOP/s/rank`.
  - `0/8 >=190`, `8/8 <160`.
- Interpretation:
  - `slot_group` is a hard loss.
  - `ordered_wait` hung, `ready_scan` remains disabled, and `slot_group` is slow; keep `fixed_wait`.
- Current overall bottleneck judgment:
  - The only positive local tuning result in this sequence is `send_pipeline_depth=2`, which stabilized the target step-1 path around `200 TFLOP/s/rank` and removed severe `<160` rows in that run.
  - CTA allocation, deeper buffering, physical tile shape, N grouping, emit pipeline, direct self compute, and receiver schedule variants did not find a stable `>=210 TFLOP/s/rank` candidate.
  - Remaining path to `>=210` likely requires a structural producer/consumer overlap change, not another local tile/schedule knob.

### 2026-07-02 04:34 PDT - FWD-SWQ-098 queue-output correctness smoke
- Question:
  - Is `hidden_output_mode=queue` correct when combined with the best known buffering knobs (`send_pipeline_depth=2`, `inbox_slots=8`, `num_sms=33`)?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-113119`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Checked small uniform all-to-all, `send_pipeline_depth=2`, `hidden_output_mode=queue`, `peer_loop=grid_switch`, `receiver_schedule=fixed_wait`, `metadata_mode=static_recv`, `output_mode=debug`, `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=1280`, `entries_per_rank=56`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=1`, `steps=3`, `repeat_runs=1`, `--check`.
- Result:
  - `steady_state_time=0.003052875710s`, `10.283539 TFLOP/s/rank`.
  - `max_abs_diff=0.0444355011`, `hidden_max_abs_diff=0.0444355011`, `hidden_mean_abs_diff=0.0029928540`, `metadata_mismatches=0`.
- Interpretation:
  - `hidden_output_mode=queue` is correct at smoke shape.
  - It is now safe to test the target schedule across `hidden_output_mode in {sink, queue, full}`.

### 2026-07-02 04:31 PDT - FWD-SWQ-101 queue-output target candidate
- Question:
  - Can a correct receive-queue hidden layout preserve the fast sink-output schedule without overwriting slots?
- Code change under test:
  - Added `hidden_output_mode=queue` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - The queue mode writes W13 hidden output by `(recv_src_ordinal, queue_entry, block_m, n_tile)`, avoiding the expert-row scatter in `hidden_output_mode=full` while preserving one persistent output row per live queue block.
- Correctness smoke:
  - H100 job `/dlwh/repro-source-push-queue-smoke-20260702-112416`, cluster `cw-us-east-02a`, succeeded.
  - Small checked uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `output_mode=debug`, `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=256`, `entries_per_rank=64`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=1`, `steps=3`, `repeat_runs=1`, `--check`, `--separate-compile`.
  - Result: `metadata_mismatches=0`, `max_abs_diff=0.0444355011`, `hidden_max_abs_diff=0.0444355011`, `hidden_unwritten_max_abs=0.015625`, `dropped_entries_total=0`, `dropped_rows_total=0`.
- Target timing:
  - H100 job `/dlwh/repro-source-push-queue-target-20260702-112659`, cluster `cw-us-east-02a`, succeeded.
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `block_m=64`, `block_k=128`, `block_n=128`, `receiver_schedule=fixed_wait`, `warmup=2`, `steps=7`, `repeat_runs=12`, `--no-check`, `--separate-compile`.
- Target result:
  - Median `steady_state_time=0.008591692545s`, median `212.004953 TFLOP/s/rank`.
  - Range: `106.955105` to `214.932681 TFLOP/s/rank`.
  - `8/12 >=210`, `2/12 <160`.
  - Per-row TFLOP/s/rank: `208.799280`, `213.001762`, `211.993486`, `142.836548`, `165.966235`, `214.932681`, `213.326043`, `212.710309`, `213.254830`, `212.016420`, `210.411280`, `106.955105`.
- Interpretation:
  - This is the first correct, non-overwriting hidden-output layout in this harness to clear `>=210 TFLOP/s/rank` median at the target shape.
  - It is not stable enough to call complete: the schedule still has intermittent slow rows, including two severe rows below `160 TFLOP/s/rank`.
  - The result suggests the full expert-row output scatter was a meaningful part of the full-output tax, but instability remains in source-push/inbox scheduling rather than the final W13 store alone.
- Next action:
  - Run a longer target repeat for stability and a bounded `hidden_output_mode=queue` sweep over nearby slot/CTA settings before consolidating the path.

### 2026-07-02 04:39 PDT - FWD-SWQ-102 mixed hidden-output sweep after queue mode
- Question:
  - Does the best depth-2/source-push schedule still clear `>=210 TFLOP/s/rank` when `sink`, `queue`, and `full` hidden-output modes are swept in one process?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-113542`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `queue_mode=routing`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `lowering_semantics=lane`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `n_groups_per_job=1`, `receiver_schedule=fixed_wait`, `send_pipeline_depth=2`, `warmup=2`, `steps=7`, `repeat_runs=12`, `--no-check`, `--separate-compile`; swept `hidden_output_mode in {sink,queue,full}`.
- Result:

  | hidden_output_mode | median W13 TFLOP/s/rank | mean W13 TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | <160 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `full` | `208.508561` | `198.402069` | `0.008735763142s` | `124.560425` | `210.645685` | `4/12` | `1/12` |
  | `queue` | `205.769014` | `183.729782` | `0.008852089922s` | `118.596089` | `209.717749` | `0/12` | `4/12` |
  | `sink` | `200.534513` | `188.505011` | `0.009096722651s` | `130.680923` | `210.873266` | `1/12` | `2/12` |

- Per-row TFLOP/s/rank:
  - `full`: `208.481054`, `209.723717`, `210.364821`, `210.240864`, `210.045948`, `210.645685`, `188.005588`, `124.560425`, `207.184939`, `208.536068`, `207.437484`, `185.598241`
  - `queue`: `122.902087`, `206.184465`, `208.630668`, `209.717749`, `209.328071`, `209.457292`, `149.270741`, `157.460157`, `205.447567`, `206.090461`, `201.672041`, `118.596089`
  - `sink`: `178.059956`, `174.083600`, `145.086935`, `209.950543`, `209.602767`, `208.359783`, `210.873266`, `208.554385`, `185.738943`, `130.680923`, `192.783233`, `208.285793`
- Interpretation:
  - The combined sweep did not reproduce the queue-only median from FWD-SWQ-101 (`212.004953 TFLOP/s/rank`).
  - `queue` being slower than `full` in this mixed run argues against a simple routed-output-scatter-only tax; the remaining issue is still slow-tail instability in the source-push/WGMMA schedule.
  - Treat this as a diagnostic miss, not a regression: it may include per-process/order interference from sweeping three separately compiled output modes.
- Next action:
  - Run an isolated `hidden_output_mode=queue` target repeat with more rows before changing the kernel structure again.

### 2026-07-02 04:42 PDT - FWD-SWQ-103 queue-only stability repeat
- Question:
  - Was the prior queue-only `212.004953 TFLOP/s/rank` median a stable candidate or a fast-mode sample?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-113953`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Isolated target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `queue_mode=routing`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `lowering_semantics=lane`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `n_groups_per_job=1`, `receiver_schedule=fixed_wait`, `send_pipeline_depth=2`, `warmup=2`, `steps=7`, `repeat_runs=24`, `--no-check`, `--separate-compile`.
- Result:
  - Median `steady_state_time=0.009692862846s`, median `189.099314 TFLOP/s/rank`.
  - Mean `181.668463 TFLOP/s/rank`.
  - Range: `130.949784` to `210.625815 TFLOP/s/rank`.
  - `12/24 >=190`, `2/24 >=210`, `6/24 <160`.
- Per-row TFLOP/s/rank:
  - `173.569778`, `136.317238`, `209.624036`, `209.842886`, `149.472691`, `131.260001`, `208.481845`, `209.247967`, `209.066908`, `207.760075`, `134.885700`, `174.164956`, `208.049762`, `206.982275`, `166.087865`, `151.760521`, `204.033672`, `210.493102`, `210.625815`, `171.001947`, `165.324213`, `208.529926`, `172.510145`, `130.949784`
- Interpretation:
  - The earlier queue-only `>=210` median was not stable over a longer repeat.
  - Queue output remains useful as a correctness-preserving diagnostic layout, but it does not solve the source-push/WGMMA slow-tail.
- Next action:
  - Inspect whether slow rows correlate with queue metadata changes across repeats or with nondeterministic runtime scheduling for identical metadata.

### 2026-07-02 04:46 PDT - FWD-SWQ-104 queue-output per-call timing distribution
- Question:
  - Do normal `steps=7` slow rows come from occasional individual slow invocations, whole seven-step batches entering slow mode, or different routing metadata?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-114348`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Same queue-only target config as FWD-SWQ-103, but `warmup=2`, `steps=1`, `repeat_runs=96`.
- Result:
  - Median `steady_state_time=0.008701298968s`, median `209.334425 TFLOP/s/rank`.
  - Mean `192.161062 TFLOP/s/rank`.
  - Percentiles: p10 `126.013778`, p25 `202.976401`, p75 `211.067498`, p90 `212.398281`.
  - Range: `87.433856` to `216.176655 TFLOP/s/rank`.
  - `73/96 >=190`, `72/96 >=200`, `41/96 >=210`, `17/96 <160`, `14/96 <140`.
- Pattern:
  - The first 13 timed invocations were mostly slow, rows 13-35 were fast, rows 36-47 were another slow cluster, and rows 48-95 were mostly clean fast-mode rows around `208-213 TFLOP/s/rank`.
- Interpretation:
  - The same metadata and compiled executable are reused for every repeat, so the slow rows are not caused by routing balance changes.
  - Slow rows are clustered, not independent single-call outliers.
  - Queue-output fast mode is close to the Stage 1 target, but normal `steps=7` timing is polluted by clustered slow bands.
- Next action:
  - Compare with `hidden_output_mode=sink` under the same per-call timing to see whether the cluster is driven mainly by large queue/full hidden outputs.

### 2026-07-02 04:48 PDT - FWD-SWQ-105 sink-output per-call timing distribution
- Question:
  - Does the clustered slow mode disappear when W13 writes only the tiny per-slot sink output?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-114612`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Same target config as FWD-SWQ-104, but `hidden_output_mode=sink`, `warmup=2`, `steps=1`, `repeat_runs=96`.
- Result:
  - Median `steady_state_time=0.008918763953s`, median `204.230250 TFLOP/s/rank`.
  - Mean `194.205325 TFLOP/s/rank`.
  - Percentiles: p10 `147.419997`, p25 `202.105671`, p75 `206.132526`, p90 `208.305902`.
  - Range: `82.352096` to `211.051663 TFLOP/s/rank`.
  - `84/96 >=190`, `84/96 >=200`, `3/96 >=210`, `11/96 <160`, `9/96 <140`.
- Pattern:
  - Slow rows again appeared in clusters, including a severe cluster around rows 26-30 and another around rows 53-58.
  - The fast-mode ceiling is lower than queue output (`~204-208` for sink versus `~208-213` for queue).
- Interpretation:
  - The clustered slow mode is not mainly the large queue/full hidden-output size; it persists with the tiny sink output.
  - Queue output is still the better fast-mode layout, but the schedule needs the clustered slow bands removed or avoided.
- Next action:
  - Run a queue-output normal-metric test with a long warmup (`warmup=64`, `steps=7`) to determine whether the clusters are early transient or recurring.

### 2026-07-02 04:51 PDT - FWD-SWQ-106 queue-output long-warmup no-signal
- Question:
  - Does queue-output normal `steps=7` timing stabilize if the early clustered slow bands from FWD-SWQ-104 are burned off with `warmup=64`?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-114814`, cluster `cw-us-east-02a`, stopped manually.
- Config:
  - Same queue-only target config as FWD-SWQ-103, but `warmup=64`, `steps=7`, `repeat_runs=12`, `--progress-events`.
- Result:
  - No timing rows.
  - The job emitted `validate_start`, `mesh_start`, and `make_inputs_start`, then did not reach `device_put_start`.
  - Stopped after it remained in that state for about 2.5 minutes.
- Interpretation:
  - This reproduces the earlier long-warmup no-signal pattern from FWD-SWQ-084.
  - Because `warmup` is not used until after inputs are created and moved to devices, this is a host setup/resource stall rather than a kernel performance signal.
  - Do not count this for or against the source-push kernel.
- Next action:
  - Stop treating long warmup alone as a path. Focus on avoiding repeated output/inbox lifecycle or changing source-push scheduling.

### 2026-07-02 04:57 PDT - FWD-SWQ-107 GMEM-spec alias experiment
- Question:
  - Can the inbox output be passed as an input/output alias so repeated calls avoid allocating/returning a fresh full inbox buffer?
- Code change under test:
  - Re-enabled the existing `inbox_storage=alias` scaffold and added explicit `pl.BlockSpec(..., memory_space=mgpu.GMEM)` specs for the direct `pl.pallas_call` alias wrapper.
  - This was intended to avoid the earlier alias failure where the wrapper mapped the full inbox into the kernel memory footprint and exceeded SMEM.
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-115259`, cluster `cw-us-east-02a`, succeeded as an Iris job but emitted an error row.
- Config:
  - Small uniform all-to-all smoke, `implementation=m_n_slots`, `hidden_output_mode=queue`, `inbox_storage=alias`, `output_mode=perf`, `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=1280`, `entries_per_rank=56`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `block_m=64`, `block_k=128`, `block_n=128`, `send_pipeline_depth=2`, `warmup=1`, `steps=3`, `repeat_runs=1`, `--no-check`, `--separate-compile`.
- Result:
  - The job reached `lower_start` and produced no timing row.
  - Error:
    - `IndexError: Slice DynamicSlice(base=64, length=64) along axis 0 is out of bounds for shape [64]`
    - The traceback is inside `jax/_src/pallas/mosaic_gpu/lowering.py` while lowering `pl.get_global(mgpu.SemaphoreType.REGULAR(...))` / `reserve_semaphores`.
- Interpretation:
  - Explicit GMEM specs avoid the previous whole-inbox SMEM accounting failure, but raw `pl.pallas_call` is still not equivalent to `mgpu.kernel` for this 2D grid with global semaphores.
  - `mgpu.kernel` passes Mosaic GPU mesh/grid names into lowering; `pl.pallas_call`/`GridSpec` does not expose the same grid-name environment, which likely changes global semaphore scoping.
  - `inbox_storage=alias` is re-disabled in validation with this updated reason. A real fix likely needs an `mgpu.kernel`-level alias mechanism or a different kernel boundary.
- Local checks:
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `git diff --check -- lib/levanter/scripts/bench/repro_source_push_inbox_queue.py .agents/logbooks/6597-moe-mgpu-forward.md`

### 2026-07-02 05:02 PDT - FWD-SWQ-108 slot-order scheduling knob and checked smoke
- Question:
  - Can changing inbox slot traversal order break fixed-wait source/receiver lockstep without changing the default schedule?
- Code change:
  - Added `slot_order in {current,rotated,hash}` and `--sweep-slot-order` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - Default `slot_order=current` preserves existing behavior.
  - Send workers and `receiver_schedule=fixed_wait` now use `_slot_for_position(...)` so rotated/hash variants can traverse slots out of ascending order.
  - `queue_stats` records `slot_order`.
- Local checks:
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `python lib/levanter/scripts/bench/repro_source_push_inbox_queue.py --help`
  - `git diff --check -- lib/levanter/scripts/bench/repro_source_push_inbox_queue.py .agents/logbooks/6597-moe-mgpu-forward.md`
  - `uv run black --config lib/levanter/pyproject.toml lib/levanter/scripts/bench/repro_source_push_inbox_queue.py` failed locally because `black` is not installed in this environment.
- H100 checked smoke:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-115917`, cluster `cw-us-east-02a`, succeeded.
- Smoke config:
  - Small uniform all-to-all, `hidden_output_mode=queue`, `output_mode=debug`, `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=1280`, `entries_per_rank=56`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `send_pipeline_depth=2`, `warmup=1`, `steps=3`, `repeat_runs=1`, `--check`, swept `slot_order in {current,rotated,hash}`.
- Smoke result:

  | slot_order | steady_state_time | W13 TFLOP/s/rank | max_abs_diff | hidden_max_abs_diff | metadata_mismatches |
  |---|---:|---:|---:|---:|---:|
  | `current` | `0.002519001641s` | `12.463019` | `0.0444355011` | `0.0444355011` | `0` |
  | `rotated` | `0.001903161018s` | `16.495906` | `0.0444355011` | `0.0444355011` | `0` |
  | `hash` | `0.001997661001s` | `15.715562` | `0.0444355011` | `0.0444355011` | `0` |

- Interpretation:
  - Rotated/hash slot traversal is correct at the checked small shape and materially faster there.
  - This is a plausible small schedule perturbation for the target slow-band problem.
- Next action:
  - Run target queue-output timing with `slot_order in {current,rotated,hash}`.

### 2026-07-02 05:06 PDT - FWD-SWQ-109 target slot-order timing sweep
- Question:
  - Does rotated/hash inbox slot traversal reduce the target queue-output slow bands?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-120241`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `queue_mode=routing`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `lowering_semantics=lane`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `n_groups_per_job=1`, `receiver_schedule=fixed_wait`, `send_pipeline_depth=2`, `warmup=2`, `steps=7`, `repeat_runs=12`, `--no-check`, swept `slot_order in {current,rotated,hash}`.
- Result:

  | slot_order | median W13 TFLOP/s/rank | mean W13 TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | <160 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | `current` | `210.512307` | `193.890519` | `0.008652632441s` | `142.728924` | `212.140232` | `8/12` | `2/12` |
  | `hash` | `146.825110` | `138.096847` | `0.012405795378s` | `105.322104` | `147.507356` | `0/12` | `12/12` |
  | `rotated` | `156.920350` | `153.038900` | `0.011607894424s` | `110.026310` | `158.545555` | `0/12` | `12/12` |

- Per-row TFLOP/s/rank:
  - `current`: `165.951213`, `159.341014`, `212.140232`, `210.833654`, `210.190959`, `210.157932`, `142.728924`, `169.728124`, `210.933028`, `211.382664`, `211.350334`, `211.948150`
  - `rotated`: `156.156325`, `156.082301`, `155.294404`, `158.309141`, `110.026310`, `154.316531`, `158.545555`, `157.593150`, `157.689663`, `158.317248`, `157.888616`, `156.247550`
  - `hash`: `146.928601`, `142.917045`, `105.322104`, `147.240016`, `147.366417`, `147.294542`, `147.152473`, `123.270633`, `113.204813`, `147.507356`, `146.721619`, `142.236541`
- Interpretation:
  - Rotated/hash slot traversal is a hard target-shape loss despite the checked small-shape speedup.
  - The current/default slot order again shows a 210+ median fast-mode sample, but the result is not stable: two rows are below `160`, and FWD-SWQ-103 current-only 24-repeat fell to `189.099314`.
- Next action:
  - Rerun the unchanged `slot_order=current` path alone after this code shape with 24 repeats. If it does not reproduce, treat the 210+ median as another fast-mode sample.

### 2026-07-02 05:09 PDT - FWD-SWQ-110 current slot-order stability repeat
- Question:
  - Does the `slot_order=current` 210+ median from FWD-SWQ-109 reproduce in isolation after the slot-order code shape change?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-120626`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `slot_order=current`, `queue_mode=routing`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `lowering_semantics=lane`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `n_groups_per_job=1`, `receiver_schedule=fixed_wait`, `send_pipeline_depth=2`, `warmup=2`, `steps=7`, `repeat_runs=24`, `--no-check`.
- Result:
  - Median `steady_state_time=0.008699176517s`, median `209.385969 TFLOP/s/rank`.
  - Mean `192.354136 TFLOP/s/rank`.
  - Range: `119.070001` to `212.513595 TFLOP/s/rank`.
  - `18/24 >=190`, `16/24 >=200`, `10/24 >=210`, `4/24 <160`.
- Per-row TFLOP/s/rank:
  - `210.578172`, `212.039398`, `166.813992`, `138.966451`, `210.605313`, `212.513595`, `211.716096`, `119.070001`, `191.338379`, `209.070655`, `207.933264`, `195.996810`, `206.618671`, `122.643655`, `211.432916`, `208.991296`, `209.817492`, `211.125180`, `212.500330`, `147.309576`, `166.214967`, `209.701282`, `212.034249`, `211.467523`
- Interpretation:
  - Not stable enough to call a 210 candidate.
  - The queue/current fast mode is real and close, but clustered slow bands remain.
- Next action:
  - Sweep nearby `num_sms` for `hidden_output_mode=queue`, `slot_order=current` before abandoning this local scheduling line.

### 2026-07-02 05:12 PDT - FWD-SWQ-111 queue/current total CTA-count screen
- Question:
  - Can a nearby total persistent-CTA count stabilize the near-210 queue/current path?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-120923`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `slot_order=current`, `num_send_sms=2`, swept `num_sms in {31,32,33,34,35}`, `queue_mode=routing`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `lowering_semantics=lane`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=8`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `n_groups_per_job=1`, `receiver_schedule=fixed_wait`, `send_pipeline_depth=2`, `warmup=2`, `steps=7`, `repeat_runs=12`, `--no-check`.
- Result:

  | num_sms | median W13 TFLOP/s/rank | mean W13 TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | <160 |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 31 | `193.880373` | `180.234649` | `0.009394879004s` | `127.680033` | `196.815846` | `0/12` | `3/12` |
  | 32 | `204.837545` | `188.728931` | `0.008892332786s` | `123.477270` | `207.559119` | `0/12` | `2/12` |
  | 33 | `206.953291` | `191.666553` | `0.008801440941s` | `144.695123` | `209.428387` | `0/12` | `2/12` |
  | 34 | `142.823734` | `140.571208` | `0.012781691421s` | `114.336567` | `161.639467` | `0/12` | `8/12` |
  | 35 | `157.678354` | `147.046572` | `0.011601282799s` | `101.923592` | `170.537003` | `0/12` | `6/12` |

- Interpretation:
  - Nearby total CTA count does not stabilize the queue/current fast mode.
  - `num_sms=33` remains the best local value in this sweep, but it did not cross `210 TFLOP/s/rank`.
  - Higher counts (`34`, `35`) are hard losses.
- Next action:
  - Run a final bounded producer/compute split screen for queue/current (`num_send_sms`) around the best `num_sms=33` pocket.

### 2026-07-02 05:16 PDT - FWD-SWQ-112 queue/current producer split screen
- Question:
  - Can changing the source-push producer/compute persistent-CTA split stabilize the near-210 queue/current path?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-121246`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `slot_order=current`, `num_sms=33`, swept `num_send_sms in {1,2,3,4}`, `queue_mode=routing`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `lowering_semantics=lane`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=8`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `n_groups_per_job=1`, `receiver_schedule=fixed_wait`, `send_pipeline_depth=2`, `warmup=2`, `steps=7`, `repeat_runs=12`, `--no-check`.
- Result:

  | num_send_sms | median W13 TFLOP/s/rank | mean W13 TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | <160 |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 1 | `176.750356` | `164.673174` | `0.010305392423s` | `116.022834` | `179.741248` | `0/12` | `3/12` |
  | 2 | `207.352762` | `185.182695` | `0.008784519858s` | `121.674884` | `211.428421` | `2/12` | `4/12` |
  | 3 | `189.818445` | `171.057626` | `0.009603278645s` | `123.021412` | `198.547419` | `0/12` | `5/12` |
  | 4 | `195.044347` | `177.154725` | `0.009339340696s` | `115.352191` | `198.984136` | `0/12` | `2/12` |

- Interpretation:
  - `num_send_sms=2` remains the best split, but still has frequent slow bands.
  - Producer/compute split does not stabilize the queue/current fast mode.
  - The bounded local tuning line (`slot_order`, total CTAs, producer split) is exhausted unless the job granularity or kernel boundary changes.
- Next action:
  - Re-check prior `n_groups_per_job`/job granularity experiments before deciding whether to reduce semaphore/job count or move to a larger kernel-boundary change.

### 2026-07-02 04:53 PDT - FWD-SWQ-107 queue-output post-compaction repeat and nearby sweep
- Correction/note:
  - The queue-output target candidate in FWD-SWQ-101 used `send_pipeline_depth=1`, confirmed from the JSON row config. Later depth-2 runs are separate candidates.
- Question:
  - Does `hidden_output_mode=queue` remain a stable `>=210 TFLOP/s/rank` path when rerun for more rows, and do nearby `inbox_slots`/`num_sms` settings suppress the slow rows?
- Long repeat of initial queue candidate:
  - H100 job `/dlwh/repro-source-push-queue-longrepeat-20260702-113253`, cluster `cw-us-east-02a`, succeeded.
  - Config: target uniform all-to-all, `hidden_output_mode=queue`, `inbox_slots=8`, `num_send_sms=2`, `num_sms=33`, `send_pipeline_depth=1`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=7`, `repeat_runs=24`, `--no-check`, `--separate-compile`.
  - Result: median `steady_state_time=0.009217505360s`, median `197.977160 TFLOP/s/rank`, range `115.114956` to `215.137282`, `8/24 >=210`, `12/24 <190`, `7/24 <160`.
  - Interpretation: the earlier `212.004953 TFLOP/s/rank` median was not stable over 24 rows.
- Nearby depth-1 slot/CTA sweep:
  - H100 job `/dlwh/repro-source-push-queue-slots-sms-20260702-113344`, cluster `cw-us-east-02a`, succeeded.
  - Config: same target config, `hidden_output_mode=queue`, `send_pipeline_depth=1`, swept `inbox_slots in {8,12,16}` and `num_sms in {32,33,34}`, `repeat_runs=6`.
- Result:

  | inbox_slots | num_sms | median TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | <190 | <160 |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 8 | 32 | `189.643527` | `0.009680599207s` | `122.195700` | `209.877307` | `0/6` | `3/6` | `1/6` |
  | 8 | 33 | `148.885965` | `0.012300048439s` | `118.439105` | `210.065057` | `1/6` | `5/6` | `4/6` |
  | 8 | 34 | `161.190875` | `0.011300345916s` | `129.494530` | `162.987340` | `0/6` | `6/6` | `2/6` |
  | 12 | 32 | `188.928575` | `0.009843685282s` | `141.717373` | `216.679483` | `3/6` | `3/6` | `2/6` |
  | 12 | 33 | `209.681560` | `0.008686903432s` | `208.306618` | `211.138944` | `2/6` | `0/6` | `0/6` |
  | 12 | 34 | `136.175747` | `0.013403092361s` | `101.072595` | `151.837312` | `0/6` | `6/6` | `6/6` |
  | 16 | 32 | `186.093886` | `0.009788017288s` | `144.704199` | `187.434557` | `0/6` | `6/6` | `2/6` |
  | 16 | 33 | `173.066149` | `0.010525345486s` | `117.718170` | `206.953252` | `0/6` | `4/6` | `2/6` |
  | 16 | 34 | `161.621979` | `0.011270010295s` | `158.589233` | `162.825440` | `0/6` | `6/6` | `1/6` |

- Interpretation:
  - `inbox_slots=12,num_sms=33,send_pipeline_depth=1` was the only stable-looking neighbor in this small sweep, but its median was just below `210`.
  - The original `slots=8,num_sms=33` candidate was noisy again.
- Next action:
  - Isolate `slots=12` with `send_pipeline_depth` and `num_sms` sweeps.

### 2026-07-02 04:54 PDT - FWD-SWQ-108 slots=12 depth/CTA focused sweep and isolated repeats
- Question:
  - Does source-push depth 2 recover a stable `>=210 TFLOP/s/rank` median at the more stable `inbox_slots=12` setting?
- Focused sweep:
  - H100 job `/dlwh/repro-source-push-queue-slots12-depth-sms-20260702-113826`, cluster `cw-us-east-02a`, succeeded.
  - Config: target uniform all-to-all, `hidden_output_mode=queue`, `inbox_slots=12`, `num_send_sms=2`, swept `num_sms in {31,32,33}` and `send_pipeline_depth in {1,2}`, `warmup=2`, `steps=7`, `repeat_runs=8`.
- Result:

  | send_pipeline_depth | num_sms | median TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | <190 | <160 |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 1 | 31 | `165.793488` | `0.011021286848s` | `116.027655` | `176.890003` | `0/8` | `8/8` | `4/8` |
  | 1 | 32 | `205.445823` | `0.008879936284s` | `124.026270` | `220.599763` | `4/8` | `3/8` | `2/8` |
  | 1 | 33 | `194.027426` | `0.009453364515s` | `125.702837` | `211.091806` | `4/8` | `4/8` | `2/8` |
  | 2 | 31 | `158.643037` | `0.011603117438s` | `131.766857` | `175.308175` | `0/8` | `8/8` | `4/8` |
  | 2 | 32 | `212.120094` | `0.008587092843s` | `138.457557` | `213.720344` | `5/8` | `1/8` | `1/8` |
  | 2 | 33 | `207.951806` | `0.008759172500s` | `128.722923` | `209.793682` | `0/8` | `2/8` | `1/8` |

- Isolated repeats:
  - `/dlwh/repro-source-push-queue-s12-s33-d1-repeat-20260702-114147`, cluster `cw-us-east-02a`, succeeded. `inbox_slots=12`, `num_sms=33`, `send_pipeline_depth=1`, `repeat_runs=24`: median `208.474787 TFLOP/s/rank`, median `0.008737186436s`, range `128.865983` to `211.903813`, `8/24 >=210`, `6/24 <190`, `4/24 <160`.
  - `/dlwh/repro-source-push-queue-s12-s32-d2-repeat-20260702-114147`, cluster `cw-us-east-02a`, succeeded. `inbox_slots=12`, `num_sms=32`, `send_pipeline_depth=2`, `repeat_runs=24`: median `210.142612 TFLOP/s/rank`, median `0.008667834718s`, range `140.022184` to `215.244428`, `13/24 >=210`, `7/24 <190`, `5/24 <160`.
- Interpretation:
  - `inbox_slots=12,num_sms=32,send_pipeline_depth=2` is the best correct queue-output median found in this pass, but it still has severe slow rows.
  - `inbox_slots=12,num_sms=33,send_pipeline_depth=1` is not stable once isolated for 24 rows.
- Next action:
  - Split the slow-tail source between output/store/protocol and WGMMA/receiver scheduling using `store_zero` and `sink`.

### 2026-07-02 04:56 PDT - FWD-SWQ-109 store/protocol versus WGMMA slow-tail diagnostics
- Question:
  - Are the severe slow rows from large queue hidden-output materialization, store/protocol overhead, or WGMMA/receiver scheduling?
- Diagnostics at current best median config:
  - Base config: target uniform all-to-all, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `block_m=64`, `block_k=128`, `block_n=128`, `warmup=2`, `steps=7`, `repeat_runs=24`.
- Queue + store-zero:
  - H100 job `/dlwh/repro-source-push-queue-s12-s32-d2-zero-20260702-114452`, cluster `cw-us-east-02a`, succeeded.
  - Config delta: `hidden_output_mode=queue`, `hidden_compute_mode=store_zero`.
  - Result: median effective `359.021792 TFLOP/s/rank`, median `0.005073456875s`, range `174.561827` to `366.504074`, `20/24 >=300`, `4/24 <250`, `0/24 <160`.
- Sink + WGMMA:
  - H100 job `/dlwh/repro-source-push-sink-s12-s32-d2-repeat-20260702-114452`, cluster `cw-us-east-02a`, succeeded.
  - Config delta: `hidden_output_mode=sink`, WGMMA on.
  - Result: median `209.321992 TFLOP/s/rank`, median `0.008701815636s`, range `124.993411` to `214.212530`, `11/24 >=210`, `9/24 <190`, `7/24 <160`.
- Interpretation:
  - The severe `<160 TFLOP/s/rank` slow rows are not primarily caused by the large persistent queue hidden output; they persist with the tiny sink output.
  - Store/protocol has smaller variability, but the main slow-tail is WGMMA/receiver scheduling or source-push synchronization.
  - Current bottleneck judgment: software-structure/scheduling-bound, not pure compute-bound and not pure output-bandwidth-bound.
- Follow-up screen:
  - H100 job `/dlwh/repro-source-push-queue-s12-s32-d2-ngroups-20260702-114931`, cluster `cw-us-east-02a`, stopped manually.
  - Config: same best median config, swept `n_groups_per_job in {1,2,5,10}`, `repeat_runs=8`, no `--progress-events`.
  - Result: no timing rows; raw logs only showed Iris setup and dependency install. Stopped after remaining quiet far longer than the identical single-config `n_groups_per_job=1` runs.
  - Interpretation: this screen was inconclusive. If revisited, rerun with `--progress-events` and a single `n_groups_per_job` value at a time.
- Next action:
  - Do not claim the queue-output candidate is complete. The best median candidate is `slots=12,num_sms=32,send_pipeline_depth=2` at `210.142612 TFLOP/s/rank`, but it is still too unstable (`5/24 <160`).
  - A real forward performance path likely needs a receiver scheduling/producer-consumer structural change or a way to eliminate the clustered slow mode, not just output layout tuning.

### 2026-07-02 05:16 PDT - FWD-SWQ-110 alias-output dead end and step-1 recurrence check
- Question:
  - Are the severe slow bands a transient/output-buffer lifecycle artifact that can be removed by output aliasing or by burning off early calls?
- Alias-output attempt:
  - Local patch tried extending the existing `inbox_storage=alias` path to alias both inbox and hidden outputs across timed calls.
  - H100 job `/dlwh/repro-source-push-alias-smoke-20260702-115903`, cluster `cw-us-east-02a`, failed immediately at validation.
  - Failure reason: the file already gates `inbox_storage=alias` with `ValueError: inbox_storage=alias is disabled: forcing direct pl.pallas_call specs to GMEM avoids the earlier whole-inbox SMEM blowup, but global semaphore lowering then fails with an out-of-bounds reserve_semaphores slice. This path needs an mgpu.kernel-level alias mechanism.`
  - Action: reverted the local hidden-alias experiment. Do not pursue raw `pl.pallas_call` aliasing for this semaphore-using MGPU kernel unless Mosaic gains an `mgpu.kernel`-level alias mechanism.
- Step-1 recurrence diagnostic:
  - H100 job `/dlwh/repro-source-push-queue-s12-s32-d2-step1-20260702-120125`, cluster `cw-us-east-02a`, succeeded.
  - Config: current best uniform queue config, `hidden_output_mode=queue`, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `slot_order=current`, `warmup=2`, `steps=1`, `repeat_runs=128`, `--progress-events`, `--no-check`.
- Result:
  - All rows: median `209.940126 TFLOP/s/rank`, mean `196.204918`, range `88.135396` to `219.089260`, `63/128 >=210`, `22/128 <190`, `19/128 <160`.
  - Tail after 64 timed calls: median `207.674254`, mean `190.632694`, range `88.135396` to `214.464433`, `20/64 >=210`, `15/64 <190`, `12/64 <160`.
- Interpretation:
  - The slow bands recur late; this is not just an early transient that a larger warmup can honestly remove.
  - Larger warmup is not a valid stabilization fix for the uniform target config.

### 2026-07-02 05:17 PDT - FWD-SWQ-111 current-best slot-order target sweep
- Question:
  - Does the small-shape slot-order win transfer to the current best `inbox_slots=12,num_sms=32,send_pipeline_depth=2` target config?
- H100 job:
  - `/dlwh/repro-source-push-queue-s12-s32-d2-slotorder-20260702-120406`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `queue_mode=routing`, `traffic_pattern=all_to_all`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `receiver_schedule=fixed_wait`, `warmup=2`, `steps=7`, `repeat_runs=12`, swept `slot_order in {current,rotated,hash}`.
- Result:

  | slot_order | median TFLOP/s/rank | mean TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | <190 | <160 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | `current` | `200.855893` | `186.178821` | `0.009079848499s` | `114.696808` | `212.873949` | `2/12` | `4/12` | `2/12` |
  | `hash` | `178.771520` | `159.126419` | `0.010188879844s` | `107.477189` | `181.026590` | `0/12` | `12/12` | `4/12` |
  | `rotated` | `152.238298` | `147.928264` | `0.012077857624s` | `117.697389` | `168.016227` | `0/12` | `12/12` | `6/12` |

- Interpretation:
  - Non-current slot orders are hard target losses at the current best config.
  - Keep `slot_order=current`; the small-shape rotated/hash win does not transfer.

### 2026-07-02 05:18 PDT - FWD-SWQ-112 exact-balanced diagnostic
- Question:
  - Are the uniform target slow bands caused primarily by multinomial tails/masked work, or do they persist with exact balanced routing?
- H100 job:
  - `/dlwh/repro-source-push-queue-s12-s32-d2-balanced-20260702-120833`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Same as FWD-SWQ-111 current-best config, but `routing=balanced`, `slot_order=current`, `repeat_runs=24`.
- Result:
  - Median `steady_state_time=0.007915820211s`, median `217.032155 TFLOP/s/rank`, mean `204.105129`.
  - Range: `132.585203` to `221.655624 TFLOP/s/rank`.
  - `18/24 >=210`, `5/24 <190`, `3/24 <160`.
  - Queue stats: `tail_fraction=0.0`, `masked_row_fraction=0.0`, `entries_by_src_min=max=2048`, `entries_by_dst_min=max=2048`, `max_slot_reuse_depth=22`.
- Interpretation:
  - Exact balance removes tail/masked work and raises the median/ceiling substantially.
  - The severe slow-band problem still persists, so tails are not the root cause.
  - This is diagnostic only: exact balanced is not the final interesting production condition.

### 2026-07-02 05:19 PDT - FWD-SWQ-113 exact-balanced slot-count diagnostic
- Question:
  - Under exact balanced routing, does aligning `inbox_slots` with the 8 blocks/expert structure improve stability?
- H100 job:
  - `/dlwh/repro-source-push-balanced-slots-20260702-121146`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Exact-balanced target, `hidden_output_mode=queue`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `slot_order=current`, swept `inbox_slots in {8,12,16}`, `warmup=2`, `steps=7`, `repeat_runs=12`.
- Result:

  | inbox_slots | median TFLOP/s/rank | mean TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | <190 | <160 | max slot reuse |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 8 | `208.932462` | `200.558971` | `0.008222729284s` | `129.168040` | `211.453188` | `2/12` | `2/12` | `1/12` | `32` |
  | 12 | `217.607463` | `207.058181` | `0.007894891713s` | `111.051930` | `219.912773` | `10/12` | `1/12` | `1/12` | `22` |
  | 16 | `192.464500` | `166.296772` | `0.008928788560s` | `101.918180` | `198.318502` | `0/12` | `6/12` | `5/12` | `16` |

- Interpretation:
  - `inbox_slots=12` remains best even under exact balance.
  - Aligning to 8 slots does not stabilize or speed up the path; reducing reuse depth to 16 slots is worse.
  - Current judgment: exact-balanced routing can produce a repeated `>=210` median, but the roughly balanced/uniform target still lacks a stable `>=210` median and both cases retain recurring slow bands.

### 2026-07-02 05:25 PDT - FWD-SWQ-114 low receiver-CTA screen for current best queue path
- Question:
  - Are the recurring slow bands in the current best queue-output source-push path caused by receiver grid pressure from `grid_switch` launching `8 * num_sms` receiver CTAs per rank?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-122137`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `queue_mode=routing`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=12`, `num_send_sms=2`, `send_pipeline_depth=2`, `slot_order=current`, `receiver_schedule=fixed_wait`, `warmup=2`, `steps=7`, `repeat_runs=12`, swept `num_sms in {16,20,24,28,32}`.
- Result:

  | num_sms | median TFLOP/s/rank | mean TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | >=200 | <160 | median send GB/s/rank |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 16 | `145.139208` | `136.968712` | `0.012549894646s` | `98.188436` | `146.073841` | `0/12` | `0/12` | `12/12` | `56.695003` |
  | 20 | `182.509923` | `164.128926` | `0.009980363572s` | `116.627086` | `186.507895` | `0/12` | `0/12` | `4/12` | `71.292939` |
  | 24 | `167.311028` | `162.978609` | `0.010886810221s` | `114.794813` | `168.857067` | `0/12` | `0/12` | `1/12` | `65.355870` |
  | 28 | `165.509056` | `153.094848` | `0.011005342273s` | `119.487737` | `166.856386` | `0/12` | `0/12` | `4/12` | `64.651975` |
  | 32 | `210.637540` | `202.653704` | `0.008647467924s` | `117.599549` | `213.518905` | `8/12` | `11/12` | `1/12` | `82.280289` |

- Interpretation:
  - Lowering receiver CTA count is not a stabilization fix. It underfills the WGMMA/receiver side and drops median performance sharply.
  - The current best `num_sms=32` remains the only row near the target median, but this screen still has a slow outlier (`117.6 TFLOP/s/rank`), consistent with the earlier repeated slow-band evidence.
  - The bottleneck hypothesis shifts away from simple grid oversubscription; the path still looks WGMMA/receiver scheduling/synchronization-bound rather than communication-bound.

### 2026-07-02 05:30 PDT - FWD-SWQ-115 single `n_groups_per_job=2` check at slots=12
- Question:
  - Does reducing per-entry N-job fanout from 10 to 5 improve the current best queue-output geometry now that the earlier slots=12 grouped-job sweep was inconclusive?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-122643`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `queue_mode=routing`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `slot_order=current`, `receiver_schedule=fixed_wait`, `n_groups_per_job=2`, `warmup=2`, `steps=7`, `repeat_runs=12`.
- Result:
  - Median `steady_state_time=0.008622289003s`, median `211.449565 TFLOP/s/rank`, mean `191.344788`.
  - Range: `123.859057` to `220.837295 TFLOP/s/rank`.
  - `6/12 >=210`, `8/12 >=200`, `8/12 >=190`, `3/12 <160`.
  - Median send bandwidth: `82.597486 GB/s/rank`.
  - Queue stats subset: `n_work_groups_per_entry=10`, `num_compute_jobs_per_entry=5`, `done_signals_per_entry=5`, `compute_wait_full_count=86855`, `max_slot_reuse_depth=24`, `tail_fraction=0.11605549479016752`, `masked_row_fraction=0.056818835991019515`.
- Interpretation:
  - Grouping two adjacent N jobs raises the fast ceiling (`220.84 TFLOP/s/rank`) and keeps a `>210` median in this short run, so semaphore/job overhead is visible in the fast mode.
  - It worsens stability versus the same `num_sms=32` `n_groups_per_job=1` low-CTA row (`1/12 <160` there vs `3/12 <160` here), and the mean falls to `191.34`.
  - Do not treat this as a solved path. If explored further, it needs CTA-count tuning around `n_groups_per_job=2`; otherwise keep `n_groups_per_job=1` as the less unstable baseline.

### 2026-07-02 05:37 PDT - FWD-SWQ-116 `n_groups_per_job=2` producer split with 30 compute workers
- Question:
  - Can the higher fast-mode ceiling from `n_groups_per_job=2` be stabilized by changing producer count while keeping receiver compute workers fixed at 30?
- H100 job:
  - `/dlwh/iris-run-job-20260702-123114`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `queue_mode=routing`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=12`, `send_pipeline_depth=2`, `slot_order=current`, `receiver_schedule=fixed_wait`, `n_groups_per_job=2`, `warmup=2`, `steps=7`, `repeat_runs=12`.
  - Paired sweep keeps `num_sms - num_send_sms = 30` for every row.
- Result:

  | num_sms | num_send_sms | median TFLOP/s/rank | mean TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | >=200 | <160 | median send GB/s/rank |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 31 | 1 | `179.269308` | `170.725624` | `0.010160592202s` | `113.531295` | `183.689814` | `0/12` | `0/12` | `2/12` | `70.027073` |
  | 32 | 2 | `218.227439` | `210.722064` | `0.008346821714s` | `130.472201` | `220.590474` | `11/12` | `11/12` | `1/12` | `85.245094` |
  | 33 | 3 | `210.680226` | `201.233427` | `0.008645720131s` | `134.180181` | `213.049521` | `8/12` | `10/12` | `1/12` | `82.296963` |
  | 34 | 4 | `145.029587` | `140.765402` | `0.012559385787s` | `99.906788` | `145.800899` | `0/12` | `0/12` | `12/12` | `56.652182` |

- Interpretation:
  - One producer is a hard loss even with 30 compute workers; median send bandwidth falls to about `70 GB/s/rank`.
  - Four producers are also a hard loss, likely because producer CTAs interfere with the WGMMA/receiver side.
  - The existing two-producer split is still the best point. In this paired run, `n_groups_per_job=2,num_send_sms=2,num_sms=32` is a strong candidate (`218.23` median, `11/12 >=210`), but it contradicts the prior isolated 12-row sample from FWD-SWQ-115 (`211.45` median, `3/12 <160`).
  - Next action: run a longer isolated repeat of exactly `n_groups_per_job=2,num_send_sms=2,num_sms=32` before treating this as a stable speedup.

### 2026-07-02 05:42 PDT - FWD-SWQ-117 isolated 48-repeat validation of grouped-job queue candidate
- Question:
  - Does the strong `n_groups_per_job=2,num_send_sms=2,num_sms=32` paired-row result hold in a longer isolated repeat?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-123853`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `queue_mode=routing`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `slot_order=current`, `receiver_schedule=fixed_wait`, `n_group=1`, `n_groups_per_job=2`, `warmup=2`, `steps=7`, `repeat_runs=48`, `--no-check`.
- Result:
  - Median `steady_state_time=0.008288984719s`, median `219.747234 TFLOP/s/rank`, mean `208.494879`.
  - Range: `131.283484` to `223.101574 TFLOP/s/rank`.
  - `20/48 >=220`, `38/48 >=210`, `39/48 >=200`, `41/48 >=190`, `5/48 <160`.
  - Median send bandwidth: `85.838763 GB/s/rank`.
  - First 24 repeats: median `219.204851`.
  - Tail 24 repeats: median `220.210963`, `2/24 <160`.
- Interpretation:
  - This is the best repeated median found for the source-push queue harness so far.
  - Compared with the prior best `n_groups_per_job=1` isolated queue candidate (`0.008667834718s`, `210.142612 TFLOP/s/rank`, FWD-SWQ-108), this is about `4.57%` faster by median time and `4.57%` higher by median TFLOP/s/rank.
  - It does not eliminate the severe slow tail: `5/48 <160` still appears. The improvement is a strong median/fast-mode improvement from reduced N-job/semaphore fanout, not a complete scheduling-stability fix.
  - Next action: run a checked smoke for the same grouped-job schedule before treating it as a correctness-preserving candidate.

### 2026-07-02 05:46 PDT - FWD-SWQ-118 checked smoke for grouped-job queue candidate
- Question:
  - Does the `n_groups_per_job=2` queue candidate preserve correctness in a checked smoke?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-124258`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Same schedule knobs as FWD-SWQ-117 but small checked geometry: `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=1280`, `entries_per_rank=56`, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `slot_order=current`, `n_groups_per_job=2`, `output_mode=debug`, `--check`, `warmup=1`, `steps=3`, `repeat_runs=1`.
- Result:
  - `metadata_mismatches=0`.
  - `max_abs_diff=0.04443550109863281`.
  - `hidden_max_abs_diff=0.04443550109863281`.
  - `hidden_mean_abs_diff=0.0029928539879620075`.
  - `hidden_all_max_abs_diff=0.04443550109863281`.
  - `hidden_unwritten_max_abs=0.015625`.
  - Timing row: `steady_state_time=0.001716642718s`, `18.288235 TFLOP/s/rank`.
- Interpretation:
  - The grouped-job queue schedule passes the same style of small checked smoke used for earlier source-push harness variants.
  - Current candidate status: correctness-smoked, target median speedup validated over 48 repeats, but slow-tail outliers remain.

### 2026-07-02 05:50 PDT - FWD-SWQ-119 grouped-job inbox slot sweep
- Question:
  - Does changing inbox buffering / slot reuse depth stabilize the `n_groups_per_job=2` queue candidate?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-124709`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Target uniform all-to-all, `implementation=m_n_slots`, `hidden_output_mode=queue`, `queue_mode=routing`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `slot_order=current`, `receiver_schedule=fixed_wait`, `n_groups_per_job=2`, `warmup=2`, `steps=7`, `repeat_runs=12`, swept `inbox_slots in {6,12,18,24}`.
- Result:

  | inbox_slots | median TFLOP/s/rank | mean TFLOP/s/rank | median steady_state_time | min TFLOP/s/rank | max TFLOP/s/rank | >=210 | >=200 | <160 | median send GB/s/rank | max slot reuse | local work groups/source |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 6 | `181.532853` | `173.822052` | `0.010131654296s` | `127.028813` | `200.740505` | `0/12` | `2/12` | `5/12` | `70.911271` | `47` | `30` |
  | 12 | `217.568665` | `208.711074` | `0.008371987929s` | `125.780223` | `220.670654` | `10/12` | `11/12` | `1/12` | `84.987760` | `24` | `60` |
  | 18 | `168.847808` | `158.322952` | `0.010787724002s` | `116.060427` | `171.346620` | `0/12` | `0/12` | `3/12` | `65.956175` | `16` | `90` |
  | 24 | `174.270285` | `163.780910` | `0.010468218309s` | `129.598134` | `183.850283` | `0/12` | `0/12` | `5/12` | `68.074330` | `12` | `120` |

- Interpretation:
  - `inbox_slots=12` remains the best point for the grouped-job queue candidate.
  - Reducing slot reuse depth with more buffering (`18` or `24` slots) does not stabilize the path and instead collapses median throughput.
  - `inbox_slots=6` under-buffers / under-schedules this path.
  - Remaining slow-tail outliers are not explained by simple slot reuse depth.

### 2026-07-02 05:57 PDT - FWD-SWQ-120 grouped-job sink-output diagnostic
- Question:
  - Does replacing queue hidden output stores with tiny sink output stabilize the `n_groups_per_job=2` candidate?
- H100 job:
  - `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-125338`, cluster `cw-us-east-02a`, succeeded.
- Config:
  - Same as FWD-SWQ-117 except `hidden_output_mode=sink`: target uniform all-to-all, `implementation=m_n_slots`, `queue_mode=routing`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `output_mode=perf`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `slot_order=current`, `receiver_schedule=fixed_wait`, `n_group=1`, `n_groups_per_job=2`, `warmup=2`, `steps=7`, `repeat_runs=48`, `--no-check`.
- Result:
  - Median `steady_state_time=0.008325638566s`, median `218.779985 TFLOP/s/rank`, mean `199.344143`.
  - Range: `108.412940` to `223.383378 TFLOP/s/rank`.
  - `18/48 >=220`, `31/48 >=210`, `34/48 >=200`, `35/48 >=190`, `8/48 <160`.
  - Median send bandwidth: `85.460932 GB/s/rank`.
  - First 24 repeats: median `218.014956`.
  - Tail 24 repeats: median `220.642147`, `3/24 <160`.
- Interpretation:
  - Sink output is not a stabilization fix. It has a comparable median to queue output but a worse slow-tail count (`8/48 <160` vs `5/48 <160` for queue).
  - The remaining slow-tail outliers are not primarily caused by queue hidden-output stores.
  - Current best candidate remains `hidden_output_mode=queue`, `n_groups_per_job=2`, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `send_pipeline_depth=2`, `slot_order=current`.

### 2026-07-02 06:00 PDT - FWD-SWQ-121 named profile for current best queue candidate
- Change:
  - Added `--source-push-profile hopper_queue_ngroups2_210` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - The profile sets the current best repeated target candidate as parser defaults while still allowing explicit CLI overrides for one-axis sweeps.
- Profile defaults:
  - `implementation=m_n_slots`
  - `queue_mode=routing`
  - `routing=uniform`
  - `traffic_pattern=all_to_all`
  - `peer_loop=grid_switch`
  - `lowering_semantics=lane`
  - `metadata_mode=static_recv`
  - `output_mode=perf`
  - `hidden_output_mode=queue`
  - `tokens_per_rank=32768`
  - `hidden_dim=2560`
  - `intermediate_dim=1280`
  - `experts_per_rank=32`
  - `topk=4`
  - `entries_per_rank=288`
  - `inbox_slots=12`
  - `num_send_sms=2`
  - `num_sms=32`
  - `block_m=64`
  - `block_k=128`
  - `block_n=128`
  - `n_group=1`
  - `n_groups_per_job=2`
  - `receiver_schedule=fixed_wait`
  - `send_pipeline_depth=2`
  - `slot_order=current`
  - `warmup=2`
  - `steps=7`
  - `repeat_runs=48`
  - `check=False`
  - `separate_compile=True`
  - `progress_events=True`
- Evidence behind profile:
  - FWD-SWQ-117 target validation job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-123853`: median `0.008288984719s`, `219.747234 TFLOP/s/rank`, `38/48 >=210`, `5/48 <160`.
  - FWD-SWQ-118 checked smoke job `/dlwh/iris-run-repro_source_push_inbox_queue-20260702-124258`: `metadata_mismatches=0`, `max_abs_diff=0.04443550109863281`.
- Local checks:
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `python lib/levanter/scripts/bench/repro_source_push_inbox_queue.py --source-push-profile hopper_queue_ngroups2_210 --help`
  - parser assertion that the profile applies the expected defaults
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
- Interpretation:
  - This does not make the kernel production-ready, but it turns the current best candidate into a reproducible named profile and gives the production branch a stable boundary to extract from.

### 2026-07-02 06:05 PDT - FWD-SWQ-122 source-push profile parser regression test
- Change:
  - Added a benchmark CLI regression test for `--source-push-profile hopper_queue_ngroups2_210` in `lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`.
  - The test imports the self-contained source-push script directly and asserts that the profile applies the exact current-best defaults while preserving explicit override behavior through the normal parser path.
- Local checks:
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py::test_source_push_profile_applies_current_best_candidate_defaults`
    - Result: `1 passed, 11 warnings`.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: `55 passed, 11 warnings`.
- Interpretation:
  - The profile is now covered by a cheap local parser regression test.
  - This is still harness consolidation rather than a production kernel boundary; the current best H100 candidate remains the FWD-SWQ-117 queue profile.

### 2026-07-02 06:06 PDT - FWD-SWQ-123 roughly-balanced routing and target tuning
- Hypothesis:
  - Exact balanced is too artificial and multinomial uniform adds high seed noise; a deterministic low-variance `roughly_balanced` routing mode should exercise tails and masked work while preserving per-source assignment totals and avoiding capacity drops.
- Code change:
  - Added `routing=roughly_balanced` to `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - The new mode jitters each source's global expert counts around the balanced count, clips to a bounded range, then adjusts counts back to the exact per-source assignment total.
  - Target shape host stats for seed 0: per-source sums `131072`, count range `449..575`, no drops with `entries_per_rank=288`, live entries/rank mean `2172.75`, max live entries/pair `276`, tail fraction `0.1169025429`, masked row fraction `0.0574157174`.
- Local checks:
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`
  - `git diff --check -- lib/levanter/scripts/bench/repro_source_push_inbox_queue.py .agents/logbooks/6597-moe-mgpu-forward.md`
- H100 jobs:
  - Checked smoke: `/dlwh/repro-source-push-roughly-balanced-smoke-20260702-052211`, succeeded, metadata mismatches `0`, dropped entries/rows `0`, reduced-shape steady state `0.0017134467s`.
  - Mistake row: `/dlwh/repro-source-push-roughly-balanced-target-20260702-052459`, succeeded with harness error only because `output_mode=perf` was launched without `--no-check`.
  - Target seed 0 depth 2: `/dlwh/repro-source-push-roughly-balanced-target-nocheck-20260702-052728`, succeeded.
  - Slots/depth sweep: `/dlwh/repro-source-push-roughly-balanced-slots-depth-20260702-053037`, succeeded.
  - Depth 1 seed confirmation: `/dlwh/repro-source-push-roughly-balanced-depth1-seeds-20260702-053553`, succeeded.
  - Longer averaging check: `/dlwh/repro-source-push-roughly-balanced-depth1-steps21-20260702-054124`, succeeded.
  - Compute-tax sweep: `/dlwh/repro-source-push-roughly-balanced-compute-tax-20260702-054640`, succeeded.
  - N-group sweep: `/dlwh/repro-source-push-roughly-balanced-ngroup-20260702-054947`, succeeded.
  - `n_groups_per_job=2` seed confirmation: `/dlwh/repro-source-push-roughly-balanced-ngpj2-seeds-20260702-055716`, succeeded.
  - SM-count sweep: `/dlwh/repro-source-push-roughly-balanced-sm-count-20260702-060010`, succeeded.
- Results:

  | Experiment | Best config / note | Median steady_state_time | Median TFLOP/s/rank | Slow bands |
  |---|---|---:|---:|---:|
  | Initial target | slots 12, depth 2, `n_groups_per_job=1`, seed 0 | `0.008540005s` | `213.423170` | `14/24 >=210`, `8/24 <190`, `5/24 <160` |
  | Slots/depth sweep | slots 12, depth 1 | `0.008309496s` | `219.343614` | `11/12 >=210`, `1/12 <190`, `1/12 <160` |
  | Depth 1 seeds | slots 12, depth 1, seeds 0/1/2 | `0.008544623s` overall | `213.418` overall | `49/72 >=210`, `17/72 <190`, `11/72 <160` |
  | Steps 21 | slots 12, depth 1, seed 0 | `0.008507563s` | `214.268338` | `13/24 >=210`, `10/24 <190`, `1/24 <160` |
  | Compute tax | `store_zero` vs WGMMA, slots 12, depth 1 | `0.005645133s` vs `0.008390887s` | equivalent `322.869` vs `217.216` | store-zero still has `1/24 <160` |
  | N grouping | `n_group=1`, `n_groups_per_job=2` | `0.008288255s` | `219.906` | `9/12 >=210`, `3/12 <190`, `2/12 <160` |
  | `n_groups_per_job=2` seeds | seeds 0/1/2 | `0.008391041s` overall | `217.325` overall | `51/72 >=210`, `19/72 <190`, `11/72 <160` |
  | SM-count sweep | `num_send_sms=2`, `num_sms=32` remains best | `0.008406207s` | `216.820` | `10/12 >=210`, `2/12 <190`, `1/12 <160` |

- Interpretation:
  - `roughly_balanced` gives a controlled middle ground between exact-balanced and multinomial uniform: no dropped entries, realistic tails/masked rows, and low source/destination imbalance.
  - For roughly-balanced target, the best config is `inbox_slots=12`, `send_pipeline_depth=1`, `n_group=1`, `n_groups_per_job=2`, `num_send_sms=2`, `num_sms=32`, `slot_order=current`.
  - `n_groups_per_job=2` is the only positive tuning result in this pass; it halves compute semaphore waits/signals per entry and improves the multi-seed median from about `213.4` to `217.3 TFLOP/s/rank`.
  - `n_group=2` is a hard loss (`41.9` to `57.6 TFLOP/s/rank` medians in the sweep), and larger `num_sms` values are also hard losses.
  - Slow-tail outliers remain even in `store_zero`, so the tail is not purely WGMMA compute. The bottleneck judgment remains software-structure/runtime-tail-bound, with meaningful compute cost on top of a ~`5.65ms` source-push/slot/store-zero floor.
- Next action:
  - Keep `roughly_balanced` as the target-like deterministic harness mode.
  - If turning this into a profile, use the rough-balanced best config above; do not switch the existing uniform profile without a separate uniform confirmation.

### 2026-07-02 06:10 PDT - FWD-SWQ-123 source-push profile coverage check
- Observation:
  - The current worktree already includes `--source-push-profile hopper_queue_roughly_balanced_ngroups2_210` in `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py`.
  - The profile keeps the FWD-SWQ-117 uniform profile separate and encodes the FWD-SWQ-122 rough-balanced depth-1, `n_groups_per_job=2` candidate as its own harness entrypoint.
- Local checks:
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: `57 passed, 11 warnings`.
- Interpretation:
  - Both named source-push harness profiles are locally covered:
    - `hopper_queue_ngroups2_210`: uniform, depth 2, 48 repeats.
    - `hopper_queue_roughly_balanced_ngroups2_210`: rough-balanced, depth 1, 48 repeats.
  - This remains harness consolidation. The source-push inbox kernel is still not extracted into the production `pallas_mgpu.py` permute_up hook.

### 2026-07-02 06:24 PDT - FWD-SWQ-124 source-push profile boundary consolidation
- Goal:
  - Make the current source-push queue candidates easier to extract into a package-private production branch without changing kernel behavior.
- Code change:
  - Added `--source-push-profile hopper_queue_roughly_balanced_ngroups2_210`.
  - Documented `--source-push-profile` in the repro docstring as the package-private extraction boundary for current Hopper source-push candidates.
  - Converted the source-push parser regression test to cover both named profiles:
    - `hopper_queue_ngroups2_210`: uniform-routing repeated target candidate from FWD-SWQ-117, `send_pipeline_depth=2`.
    - `hopper_queue_roughly_balanced_ngroups2_210`: target-like deterministic routing diagnostic from FWD-SWQ-123, `send_pipeline_depth=1`.
  - Added an explicit override test so profile defaults do not prevent one-axis sweeps.
  - Renumbered the previous rough-balanced entry from duplicate `FWD-SWQ-122` to `FWD-SWQ-123`.
- Local checks:
  - `python -m py_compile lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - `git diff --check -- lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py .agents/logbooks/6597-moe-mgpu-forward.md`
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py::test_source_push_profile_applies_current_best_candidate_defaults lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py::test_source_push_profile_allows_explicit_overrides`
    - Result: `3 passed, 11 warnings`.
  - `./infra/pre-commit.py --fix --files lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: `57 passed, 11 warnings`.
  - `python lib/levanter/scripts/bench/repro_source_push_inbox_queue.py --source-push-profile hopper_queue_roughly_balanced_ngroups2_210 --help`
    - Result: parser/help succeeds.
- H100:
  - No H100 job launched for this patch; it only changes named defaults, tests, and documentation around already-measured configs.
- Interpretation:
  - Stage 1 evidence remains the uniform profile from FWD-SWQ-117: median `219.747234 TFLOP/s/rank` over 48 target repeats with a checked smoke in FWD-SWQ-118.
  - The rough-balanced profile is now an explicit target-like diagnostic boundary backed by FWD-SWQ-123, not a replacement for the uniform production candidate.
  - Remaining Stage 2 work is to extract the profile-backed source-push kernel/harness into a cleaner package-private module/branch and keep the benchmark script as an experiment driver rather than the only API boundary.

### 2026-07-02 06:32 PDT - FWD-SWQ-125 package-private source-push profile module
- Goal:
  - Move the source-push candidate profile boundary out of the benchmark script and into package-private `_moe` code as the first production-branch extraction step.
- Code change:
  - Added `lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py`.
  - Moved `SOURCE_PUSH_PROFILES` and named profile defaults into that module.
  - Added `source_push_profile_defaults(profile)` which returns a copy of the profile defaults.
  - Updated `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py` to import the profile choices/defaults from `_moe`.
  - Added `test_source_push_profile_defaults_are_copied`.
- Local checks:
  - `python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - `git diff --check -- lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py .agents/logbooks/6597-moe-mgpu-forward.md`
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py::test_source_push_profile_applies_current_best_candidate_defaults lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py::test_source_push_profile_allows_explicit_overrides lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py::test_source_push_profile_defaults_are_copied`
    - Result: `4 passed, 11 warnings`.
  - `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/repro_source_push_inbox_queue.py --source-push-profile hopper_queue_roughly_balanced_ngroups2_210 --help`
    - Result: parser/help succeeds.
  - `./infra/pre-commit.py --fix --files lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed, including Pyrefly.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: `58 passed, 11 warnings`.
- H100:
  - No H100 job launched. This patch moves/covers profile metadata only and does not change kernel behavior or measured config values.
- Interpretation:
  - This is not the full Stage 2 production branch yet, but it creates a concrete `_moe` package-private boundary for the profile-backed source-push candidates.
  - Next extraction step is to move the source-push Pallas kernel/wrapper behind `_moe` code and leave the benchmark script as a thin CLI around the package-private entrypoint.

### 2026-07-02 06:40 PDT - FWD-SWQ-126 package-private source-push harness extraction
- Observation:
  - The source-push inbox prototype has now been moved behind package-private `_moe` code.
  - `lib/levanter/src/levanter/grug/_moe/source_push_inbox.py` contains `PushInboxConfig`, the Pallas kernel factory, the sharded wrapper, input generation, validation, timing, parser, and `main()`.
  - Added `run_source_push_inbox(config, ...)` as the explicit package-private runner used by the CLI wrapper path.
  - `lib/levanter/scripts/bench/repro_source_push_inbox_queue.py` is now only a CLI wrapper that imports `main()` from `_moe.source_push_inbox`.
  - `lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py` now documents the actual boundary: profiles live next to the package-private source-push harness and benchmark scripts should remain thin wrappers.
- Local checks:
  - `python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
  - `git diff --check -- lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py .agents/logbooks/6597-moe-mgpu-forward.md`
  - `uv run --package marin-levanter --group test python lib/levanter/scripts/bench/repro_source_push_inbox_queue.py --source-push-profile hopper_queue_roughly_balanced_ngroups2_210 --help`
    - Result: parser/help succeeds.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: `59 passed, 11 warnings`.
  - `./infra/pre-commit.py --fix --files lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed, including Pyrefly.
- H100:
  - Launched post-extraction wrapper/profile validation job `/dlwh/repro-source-push-extracted-profile-20260702-0640`, cluster `cw-us-east-02a`.
  - Config: `--source-push-profile hopper_queue_ngroups2_210`, JSONL `scratch/repro_source_push_extracted_profile_20260702_0640.jsonl`.
  - Job state: succeeded, exit `0`, duration `1 minute and 45.29 seconds`.
  - Results over 48 repeats:

  | metric | value |
  |---|---:|
  | median steady_state_time | `0.008370955358s` |
  | mean steady_state_time | `0.009399717622s` |
  | min steady_state_time | `0.008172136299s` |
  | max steady_state_time | `0.015787412845s` |
  | median W13 TFLOP/s/rank | `217.595407` |
  | mean W13 TFLOP/s/rank | `200.679220` |
  | min W13 TFLOP/s/rank | `115.375546` |
  | max W13 TFLOP/s/rank | `222.889255` |
  | median send GB/s/rank | `84.998206` |
  | rows >=210 TFLOP/s/rank | `35/48` |
  | rows <190 TFLOP/s/rank | `10/48` |
  | rows <160 TFLOP/s/rank | `8/48` |
  | dropped entries / rows | `0 / 0` |

  - Errors: none.
- Interpretation:
  - Stage 2 extraction is materially further along: the source-push kernel/harness no longer lives only in `scripts/bench`.
  - The post-extraction wrapper/profile run preserves the fast source-push profile behavior, but this repeat is noisier than FWD-SWQ-117 (`217.60` vs `219.75` median TFLOP/s/rank and `35/48` vs `38/48` rows >=210).
  - This is still a package-private prototype boundary, not public `moe_mlp` integration.

### 2026-07-02 06:22 PDT - FWD-SWQ-127 extracted-wrapper H100 checked smoke
- Question:
  - Does the thin `scripts/bench/repro_source_push_inbox_queue.py` wrapper correctly execute the extracted package-private `source_push_inbox.py` implementation on H100?
- H100 job:
  - `/dlwh/repro_source_push_extract_smoke-20260702-0619`, cluster `cw-us-east-02a`, succeeded.
- Command shape:
  - Wrapper CLI with `--source-push-profile hopper_queue_ngroups2_210`.
  - Overrides: `tokens_per_rank=4096`, `hidden_dim=256`, `intermediate_dim=1280`, `entries_per_rank=56`, `output_mode=debug`, `--check`, `warmup=1`, `steps=3`, `repeat_runs=1`, `--no-separate-compile`.
- Result:
  - Terminal state: `JOB_STATE_SUCCEEDED`.
  - `metadata_mismatches=0`.
  - `max_abs_diff=0.04443550109863281`.
  - `hidden_max_abs_diff=0.04443550109863281`.
  - `hidden_mean_abs_diff=0.0029928539879620075`.
  - `hidden_unwritten_max_abs=0.015625`.
  - Debug-shape `steady_state_time=0.006517660338431597s`, `w13_tflops_per_rank=4.816815208193975`, `send_gbps_per_rank=1.8815684407007711`.
- Interpretation:
  - The extraction is not only a local import change; the wrapper drives the extracted package-private implementation through compile/run/check on H100.
  - This is a correctness/import smoke only. It does not replace the target performance evidence from FWD-SWQ-117.

### 2026-07-02 06:29 PDT - FWD-SWQ-128 extracted rough-balanced 96-repeat validation
- Question:
  - Does the extracted package-private source-push path preserve a stable median above `210 TFLOP/s/rank` on the target-like `roughly_balanced` routing profile?
- H100 job:
  - `/dlwh/repro-source-push-rough-extracted-96-20260702-0645`, cluster `cw-us-east-02a`, succeeded.
- Command shape:
  - `uv run --package marin-iris --extra controller iris --cluster=cw-us-east-02a job run --no-wait --job-name repro-source-push-rough-extracted-96-20260702-0645 --cpu 16 --memory 128GB --disk 16GB --gpu H100x8 --reserve H100x8 --enable-extra-resources --extra gpu -- timeout 3600s uv run --package marin-levanter --group test python lib/levanter/scripts/bench/repro_source_push_inbox_queue.py --source-push-profile hopper_queue_roughly_balanced_ngroups2_210 --repeat-runs 96 --jsonl scratch/repro_source_push_rough_extracted_96_20260702_0645.jsonl`
- Config:
  - Profile `hopper_queue_roughly_balanced_ngroups2_210`: `implementation=m_n_slots`, `routing=roughly_balanced`, `hidden_output_mode=queue`, `peer_loop=grid_switch`, `metadata_mode=static_recv`, `tokens_per_rank=32768`, `hidden_dim=2560`, `intermediate_dim=1280`, `experts_per_rank=32`, `topk=4`, `entries_per_rank=288`, `inbox_slots=12`, `num_send_sms=2`, `num_sms=32`, `block_m=64`, `block_k=128`, `block_n=128`, `n_group=1`, `n_groups_per_job=2`, `receiver_schedule=fixed_wait`, `send_pipeline_depth=1`, `slot_order=current`, `warmup=2`, `steps=7`, `repeat_runs=96`, `--no-check`, `--separate-compile`.
- Results:

  | metric | value |
  |---|---:|
  | median steady_state_time | `0.008262450441s` |
  | mean steady_state_time | `0.009176977219s` |
  | min steady_state_time | `0.008118119590s` |
  | max steady_state_time | `0.015110662274s` |
  | median W13 TFLOP/s/rank | `220.592534` |
  | mean W13 TFLOP/s/rank | `204.663667` |
  | min W13 TFLOP/s/rank | `120.619121` |
  | max W13 TFLOP/s/rank | `224.514407` |
  | median send GB/s/rank | `86.168958` |
  | rows >=220 TFLOP/s/rank | `58/96` |
  | rows >=210 TFLOP/s/rank | `69/96` |
  | rows >=200 TFLOP/s/rank | `73/96` |
  | rows <190 TFLOP/s/rank | `21/96` |
  | rows <160 TFLOP/s/rank | `13/96` |
  | first 48 median W13 TFLOP/s/rank | `220.533451` |
  | tail 48 median W13 TFLOP/s/rank | `220.738143` |
  | dropped entries / rows | `0 / 0` |

  - Compile time: `5.523906586925s` total, `3.840899118921s` lower/compile, `1.683007468004s` first run.
  - Queue stats: `live_entries_total=17382`, `rounded_rows_per_rank_mean=139056.0`, `masked_row_fraction=0.05741571740881372`.
  - Errors: none.
- Interpretation:
  - This is the strongest Stage 1 evidence so far for the target-like route distribution: the extracted package-private rough-balanced profile has a replicated median well above `210 TFLOP/s/rank` over 96 repeats, and both halves of the run have medians above `220`.
  - Severe slow-tail rows remain (`13/96 <160`), so the schedule is not outlier-free. The current claim should be `stable median >=210`, not eliminated variance.
  - Stage 2 should now focus on PR consolidation: keep this as the current rough-balanced profile evidence, document the slow-tail limitation, and avoid more blind schedule tuning unless a new structural idea directly attacks the tail.

### 2026-07-02 06:29 PDT - FWD-SWQ-128 post-compaction source-push extraction validation
- Scope:
  - Revalidated the extracted package-private source-push inbox boundary after context compaction.
  - No kernel scheduling/control-flow changes.
- Local checks:
  - `python -m py_compile lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: `60 passed, 11 warnings`.
  - `./infra/pre-commit.py --fix --files lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed, including Pyrefly.
  - `git diff --check -- .agents/logbooks/6597-moe-mgpu-forward.md .agents/projects/20260629-mgpu_permute_up.md lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed.
- H100:
  - No new H100 job launched for this entry.
- Interpretation:
  - The package-private extraction remains locally clean.
  - Continue from the source-push inbox path; the closed `lax.switch` experiment should not be revisited.

### 2026-07-02 06:34 PDT - FWD-SWQ-129 full touched-file cleanup before WIP snapshot
- Scope:
  - Cleaned mechanical lint issues across the current MoE/Pallas forward WIP set before snapshotting the branch.
  - Moved Mosaic debug imports in `bench_grug_moe_pallas_mgpu.py` to module scope, removed one unused metadata gather in `pallas_mgpu.py`, accepted Black formatting, and added Levanter license headers to the small peer-copy repro scripts.
- Local checks:
  - `./infra/pre-commit.py --fix --files .agents/logbooks/6597-moe-mgpu-forward.md .agents/projects/20260629-mgpu_permute_up.md lib/levanter/src/levanter/grug/_moe/pallas_mgpu.py lib/levanter/src/levanter/grug/_moe/source_push_inbox.py lib/levanter/src/levanter/grug/_moe/source_push_inbox_profiles.py lib/levanter/src/levanter/grug/_moe/common.py lib/levanter/src/levanter/grug/grug_moe.py lib/levanter/scripts/bench/bench_grug_moe_pallas_mgpu.py lib/levanter/scripts/bench/compact_moe_permute_up_mgpu.py lib/levanter/scripts/bench/repro_source_push_inbox_queue.py lib/levanter/scripts/bench/repro_source_push_inbox_sem.py lib/levanter/scripts/bench/repro_remote_gmem_to_smem_peer_lane.py lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer.py lib/levanter/scripts/bench/repro_smem_to_remote_gmem_peer_grid.py lib/levanter/tests/grug/test_grugformer_moe.py lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: passed, including Pyrefly.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grug_moe_pallas_mgpu_bench.py`
    - Result: `60 passed, 11 warnings`.
  - `uv run --package marin-levanter --group test pytest -q lib/levanter/tests/grug/test_grugformer_moe.py::test_pallas_mgpu_moe_mlp_uses_explicit_config_for_validation`
    - Result: `1 passed, 11 warnings`.
  - `git diff --check -- <current MoE/Pallas forward WIP files>`
    - Result: passed.
- H100:
  - No new H100 job launched for this entry.
- Interpretation:
  - The branch is locally clean enough for a WIP checkpoint commit/push of the current forward package.
