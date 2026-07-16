---
topic: Source-push EP MoE follow-up (building on PR #6841)
issue: https://github.com/marin-community/marin/issues/7276
pull_request: https://github.com/marin-community/marin/pull/6841
description: Follow-up investigation on the source-push/semantic EP MoE path — review-driven avenues ranked by potential/effort, executed as parallel experiments.
author: Matt Wittmann
---

# 6841 Source-Push Follow-up Logbook

Branch: `research/mcwitt/6841-source-push-followup` (based on PR #6841 head `26711f86e`).

Prior logbooks: `.agents/logbooks/6597-moe-mgpu-forward.md`,
`.agents/logbooks/6597-moe-fused-comm-tensorcore.md`.

Experiment ID prefix: `SPF-###`. Tags: `spf`, `6597`, `6841`.

## Fixed Target (inherited from #6841)

```text
EP ranks:               8
tokens/rank:            32768
top-k:                  4
experts/rank:           32
hidden dimension:       2560
intermediate dimension: 1280
routing:                roughly balanced
target:                 250 useful TFLOP/s/rank aggregate forward + backward
```

Baselines at PR head:

| path | time | useful TFLOP/s/rank |
|---|---|---|
| staged W13 inbox forward | 8.31 ms | 206.8 |
| staged full forward | 13.88 ms | 185.7 |
| staged honest custom-VJP fwd+bwd | 63.04 ms | 122.7 |
| manual integrated graph fwd+bwd | 51.47 ms | 150.3 |
| fused production custom VJP fwd+bwd | 135.49 ms | 57.1 |

## Strategic frame (from SPF-000 review synthesis)

- The **staged semantic path** is the fastest source-push configuration:
  44.152 ms / 175.19 useful fwd+bwd (manual decomposition), 63.038 ms / 122.70
  through the honest custom-VJP API. The fused persistent-kernel path is
  135.49 ms / 57.09 and additionally has **no finite end-to-end numerical
  validation** (FUSED-MOE-094 failed with y max-abs-diff 1024 and was never
  re-cleared; -129 checksum nonfinite). Near-term optimization spend goes to
  the staged path; the fused path gets free measurements + cheap fixes only.
- The **public `ring` backend is the bar**: 38.224 ms / 202.25 useful fwd+bwd
  at the same target (job `/dlwh/ring-mlp-fwd-bwd-3d88a8720-target`). Any
  source-push claim must be made relative to ring, not only to itself.
- External expert guidance (2026-07-15 call): SM-carveout specialization over
  comm-into-GEMM fusion — consistent with the measured fused ceiling and with
  the review findings (unpipelined fused consumers, serialized barriers).

## Ranked avenues (potential / effort)

| # | Avenue | Type | Est. gain | Effort | Disposition |
|---|---|---|---|---|---|
| 1 | bf16-cast dy before the replicated reshard (`semantic_mlp.py:586`); AG payload 2.35→1.17 GB | perf | ~5.8 ms off 63.0 ms API | S | SPF-001 |
| 2 | Switch dx return to existing owner-sharded `psum_scatter` wrapper (`semantic_mlp.py:621`); AR 4.70→2.35 GB | perf | ~3-6 ms (also helps manual graph) | S | SPF-002 |
| 3 | Measure unlogged tip commit `26711f86e` (grouped W13 raw gathers) vs 23.025 ms fused-W13 baseline | measurement | decisive for FUSED-MOE-129 thread | free | SPF-003 |
| 4 | Fused W13-bwd unconditional zero-fill remote-writes 2x NVLink bytes (`fused_w13_backward.py:1040`) | bug(perf) | ~2-4 ms of fused 135.5 | S | SPF-003 |
| 5 | Combine kernel at ~21% HBM roofline (per-row scalar loads, `source_push_combine.py:145`) | perf | 1.34→~0.5 ms fwd | S | SPF-004 |
| 6 | Host-side numpy planner cost at target shape (excluded from all benchmark numbers; O(268M) element ops/plan) | measurement | unknown, possibly dominates | S (CPU) | SPF-005 |
| 7 | Source-push dy route (fix dcombine in_specs `backward_pallas.py:2280`, reuse existing dy-route remote-write kernel, add global phase barrier) | structural | rest of the 11.6 ms API tax | M-L | next |
| 8 | W2-return kernel: double-buffer k-loop + deferred `wait_smem_to_gmem` (`source_push_w2_return.py:680-765`) | perf | 4.05→~2.7 ms fwd | M | next |
| 9 | Correctness cluster: single-jit remote-write completion barrier is scalar-only (`source_push_forward.py:1109`); fused dX-return race (no wait on `ready_sem`); failed integrated gate FUSED-MOE-094 | bug | blocking for fused promotion / single-jit | S-M | next |
| 10 | SM-carveout restructure: persistent SM-capped comm kernel + pipelined near-stock GEMM consumers; de-fuse W2-return/combine (dense fixed-shape permute) | structural | only credible route to 250 | L | design |

Sum of S-effort staged wins (1+2+5): 63.0 → ~50-52 ms API (~150 useful),
converging toward the 44.2 ms manual bound; ring (38.2 ms) still ahead until
avenue 7/10 land.

## Hypothesis Queue

| ID | Hypothesis | Status | Decisive evidence |
| --- | --- | --- | --- |
| SPF-001 | bf16 dy before reshard saves ≥4 ms with grad parity within bf16 tolerance | Running | fwd_bwd median vs 63.038 ms + parity check |
| SPF-002 | Owner-sharded dx return (psum_scatter) saves ≥3 ms with dx parity | Running | fwd_bwd median (alone and stacked on SPF-001) |
| SPF-003a | Tip gather grouping improves fused W13 vs 23.025 ms | CONFIRMED (small): 22.769 ms, -1.11% | see SPF-003 entry |
| SPF-003b | Zero-fill double-write fix saves ≥2 ms on fused W13-bwd | FALSIFIED: +0.32 ms vs control, bit-exact; discarded | see SPF-003 entry |
| SPF-006 | Reduced-shape fused-W13 compare regression (valid_error_count 0 -> 5 -> 3) introduced in d2ce47ca35..088831b4b | Open (needs bisect) | reduced `semantic_permute_w13_compare` valid_error_count per commit |
| SPF-004 | XLA gather-sum reaches ≥1.5 TB/s effective on combine | CONFIRMED: 0.481 ms / 1745 GB/s, bitwise-identical to kernel; ADOPT | see SPF-004 entry |
| SPF-007 | Staged w2_return stage regressed 4.05 -> 7.08 ms between 89f3267fc (2026-07-03) and branch head | Open (needs bisect) | staged_host_sync w2_return stage time per commit |
| SPF-005 | Host-side plan build at target shape costs >2 ms/plan (likely ≫), making planner device-siding the top structural priority | CONFIRMED (~190x over threshold) | see SPF-005 entry: 380 ms plan build, ~1.85 s total public-path host work per plan |

## 2026-07-16 SPF-000 - Kickoff: review fan-out

Four parallel review agents launched over the PR #6841 code at head
`26711f86e`: (a) forward path + plan/metadata, (b) backward path + custom-VJP
boundary (dy all-gather focus), (c) fused semantic stages (135 ms budget,
gather bottleneck, SM topology), (d) logbook/harness mining (falsified ideas,
open hypotheses, best-known numbers, harness bugs). Synthesis will produce a
ranked avenue list and the first experiment matrix.

External context: 2026-07-15 expert call (NVIDIA/Dao-lab) recommends
SM-carveout specialization (persistent SM-capped comm kernels + independently
launched GEMMs) over comm-into-GEMM fusion; consistent with the measured fused
ceiling in FUSED-MOE-129.

## 2026-07-16 SPF-005 - Host planner cost at target shape: 380 ms/plan, 1.85 s public path (CONFIRMED, ~190x threshold)

CPU-only measurement (no GPU jobs). Script
`/home/marin/.claude/jobs/84d3f127/tmp/spf005_plan_cost.py`, run via
`uv run --package marin-levanter --group test python <script>` from the
worktree root (jax 0.10.1 CPU, numpy 2.3.5, 32 vCPU). 2 warmup + 10 reps,
target shape EP8/T32768/topk4/E32/H2560/cf1.25, `roughly_balanced` seed 0,
rough_balanced_216 capacity (entries_per_rank=288, block_m=64), routing built
with the bench harness's own helpers.

| Component | median ms | p90 |
|---|---:|---:|
| `build_source_push_plan` (entries=288) | 379.9 | 409.8 |
| `build_source_push_plan` (entries=auto, public path) | 389.0 | 423.0 |
| `make_source_push_forward_plan_inputs_from_plan` (total) | 90.5 | 92.0 |
| — `_make_route_inverse` | 63.9 | 65.9 |
| — send/recv meta + row bases + h-row weights | ~17.5 | — |
| `source_push_mlp_route_table_from_plan` (public path) | 1372.6 | 1380.4 |
| semantic JAX planner, jitted steady state (CPU) | 156.0 | 159.7 |

Totals: public staged path ≈ **1852 ms host work per plan**; plan build alone
= 27.4x the 13.876 ms device forward and 6.0x the 63.038 ms fwd+bwd. Per
`source_push_public.py:94-110` the plan, inputs, and route table are rebuilt
unconditionally per call, and `_STATIC_PUBLIC_ROUTE_ERROR`
(`source_push_public.py:36-40`) forces concrete routing at Python time — fresh
routing per microbatch x per MoE layer means this host work is fully
serializing in real training. The only cache (`_STAGED_FORWARD_CALL_CACHE`,
`source_push_forward.py:98`) caches compiled callables, not plans. Dominant
costs: the per-(src x dst x expert) mask scan (`source_push_plan.py:636-662`)
and the route-table builder's Python triple-loop scatter.

Interpretation: every published #6841 benchmark number times steady-state
kernels with a prebuilt plan; on the public API the host planner would
dominate device time by 6-30x. The JAX-native semantic planner
(`build_source_push_plan_semantic_jax`) already runs in-jit at ~1.3-1.5 ms on
GPU (era-1 logbook), so **device-siding the planner (or wiring the semantic
planner under the staged forward) is the top structural priority for any
production use of this path** — ahead of all kernel tuning. Caveats: dev-box
CPU, not the pod host (GPU hosts add D2H/H2D syncs, likely worse); the 156 ms
semantic-planner CPU number is a contrast point, not a GPU projection.

## 2026-07-16 SPF-003 - Fused measurements: tip grouping +1.1% real; zero-fill fix falsified; compare regression found

Branch `spf/003-fused-measurements` (final `ff398b51f`; tested fix at
`a4952eccd`). All jobs cw-rno2a H100x8, canonical flags
(`--rows-per-src-dst-capacity auto --plan-builder jax --separate-compile`;
target warmup 1 / steps 3 / repeats 3; reduced compare EP8/T128/E4/topk2/cf4.0).
Zero drops/overflows, no error rows.

**Arm A (tip `26711f86e` grouped W13 raw gathers, no code change).** Job
`/marin/spf003-arma-r3-26711f86`: target `semantic_permute_w13_pallas` median
**22.769 ms / 75.45 useful** vs baseline 23.025 ms / 74.61 (`d2ce47ca35`) =
**-0.257 ms (-1.11%)**, target checksum identical to baseline. Verdict: real
but modest; the gather implementation remains ~2.7x off the 8.4 ms inbox
floor.

**Compare-gate regression (new, SPF-006).** Reduced
`semantic_permute_w13_compare` shows `valid_error_count` = 3 at tip, 5 at
`088831b4b` (pre-grouping), 0 at `d2ce47ca35` — a validity-flag regression vs
the JAX reference introduced somewhere in the 72-commit
`d2ce47ca35..088831b4b` segment (grouping itself moved 5 -> 3, so it is not
the origin). z/h float diffs are bit-identical across all three commits
(commit-independent bf16-boundary artifact per FUSED-MOE-003). Blocks
promotion of anything from this branch segment until bisected.

**Arm B (fused W13-bwd zero-fill double-write fix).** Fix `a4952eccd`
(zero-fill only when `valid_rows == 0`; publication order untouched; local
tests 11 passed). Reduced compare **bit-exact** (dx/dw13 diff 0.0). Target
`semantic_fused_w13_backward_pallas`: fix median **63.249 ms** vs same-day tip
control **62.933 ms** (job `/marin/spf003-armb-ctl-26711f86`) and 62.697 ms
selected baseline — **slower, non-overlapping repeat ranges. FALSIFIED**: the
~671 MB/rank duplicate NVLink zero-writes were fully overlapped; the added
predicate costs ~0.3-0.5 ms. Reverted (`ff398b51f`).

**Infra notes.** (1) Iris client `cuda_toolchain_setup_script` on main runs
`uv pip install --offline --reinstall-package nvidia-cudnn-cu13` — fails
outright on cold-cache nodes before the user command (killed two jobs);
workaround: submit without client `--extra gpu`, sync in-task. (2) This
worktree's iris predates cw-rno2a config and IRIS_USER (jobs landed under
`/marin/`); submitted with the main-checkout client. (3) `/tmp/iris`
port-lock dir owned by another user breaks the tunnel — manual
`kubectl port-forward` + `--controller-url` works.

## 2026-07-16 SPF-004 - Combine: plain XLA beats the Pallas kernel, bitwise-identical; W2-return regressed at head

Branch `spf/004-combine-roofline` (final `8b5e2038a`; combine mode at
`2ece2c284`, W2 hoist at `65629623e`). Jobs on cw-us-east-02a H100x8
(this branch's iris predates cw-rno2a), 216 profile, seed 0.

**Combine (adopt).** New opt-in `combine_mode="xla_gather_sum"`
(`source_push_combine.py`): flat row-granularity `jnp.take` + fixed-slot-order
f32 sum mirroring the kernel's arithmetic exactly. Job
`/marin/spf004-combine-2ece2c284`: baseline `direct_gather_sum` median
**0.5852 ms (1433 GB/s)**; `xla_gather_sum` median **0.4807 ms (1745 GB/s)**,
bitwise-deterministic across 3 runs and **bitwise-identical to the kernel
output**. Decision rule (≥1.5 TB/s) met. Note: the 1.338 ms/0.63 TB/s review
premise was stale — the head kernel already runs 0.585 ms standalone (~0.92 ms
in stage context); the XLA path still wins -18% and deletes a Pallas kernel.

**W2-return regression discovered (SPF-007).** The staged forward's w2_return
stage measures **7.077 ms at head** (`/marin/spf004-w2base-2ece2c284`,
staged_host_sync, roughly_balanced cf1.25) vs the logged 4.050 ms at
`89f3267fc` (2026-07-03) — a ~75% stage regression from branch drift, larger
than every S-tier win combined. Needs a bisect.

**W2 from_h weight-copy hoist (keep).** Route-weight tile staged once before
the k-loop (mirroring the compact variant): w2_return median 7.077 -> **6.803
ms (-3.87%)**, total staged forward 16.724 -> 16.516 ms. Full-forward
`--check` reports max_abs_diff=512.0 **bit-identically on baseline and hoist**
(pre-existing target-shape reference limitation — historical gates all ran
`--no-check`; another correctness-debt item). H100 pytest: same 2 pre-existing
failures before and after (one is the avenue-9 in_specs bug), 102 passed; no
new failures.

**Infra.** Iris auto-tunnel port-lock dir `/tmp/iris` owned by another user →
PermissionError swallowed per-port (`iris/cluster/backends/types.py:70-97`);
manual `kubectl port-forward` workaround. Worker pytest needs explicit
`uv pip install pytest pytest-xdist` after the cudnn pin; plain re-sync skews
jax/pallas.
