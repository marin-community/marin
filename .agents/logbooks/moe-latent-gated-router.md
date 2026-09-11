---
topic: Gated latent router ablation
issue: https://github.com/marin-community/marin/issues/9110
description: Compare full-width routing with a GatedNorm latent shared by router and experts.
author: Kaiyue Wen
---

# Gated latent router: task logbook

## Scope

Idea from Zihan Qiu. Follow Agent MoE Gate 1 at d512/d768 and proceed to
Gate 2 at d1024/d1280 only if both smaller scales improve effective speedup.
Reuse the existing Agent MoE baseline runs; do not launch new baseline jobs.
Experiment prefix: MOE-LGR. W&B group: moe-latent-gated-router-9110 in marin_moe.

## Current baseline

EP hero main `d891fba48a7d3729fdac2595df3c6ca292a53b74`, scaled to small widths.
The control is `RMSNorm(down(h))` for experts and `router(h)` for routing.
The treatment is `z=GatedNorm(down(h))`, used by both router and experts.
The control is retained for numerical checks and analytic budget calculation.
Only the treatment is trained. Historical comparisons have architecture,
optimizer, hardware, and token-count differences.
See the [plan](../projects/moe-latent-gated-router/plan.md) and
[variant README](../../experiments/grug/moe_latent_gated_router/README.md).

## Hypothesis queue

- Active MOE-LGR-001: The shared gated latent improves loss enough to offset
  runtime overhead at d512 and d768 versus the existing Agent MoE reference.
  Evidence is pending; fail if either effective speedup is <=1. This reference
  comparison cannot isolate the requested intervention.
- Conditional MOE-LGR-002: The gain persists at d1024/d1280 and improves
  projected loss at 1e21 and 1e23 FLOPs. Run only after MOE-LGR-001 passes.

## Entry log

### 2026-09-11 — Design and prior work

- Effort: low. The requested intervention is concrete; stop after checking
  current hero code, nearby latent experiments, and norm/optimizer history.
- Source: [hero model](https://github.com/marin-community/marin/blob/d891fba48a7d3729fdac2595df3c6ca292a53b74/experiments/grug/moe_hero_ep/model.py#L914).
  Router reads full width; latent RMSNorm feeds experts.
- [#8105](https://github.com/marin-community/marin/issues/8105): dynamic latent
  retained RMSNorm and found a dropless tie (3.0329 vs 3.0335). It does not
  answer whether the requested shared gated latent helps.
- Echo query: `latent MoE router gated norm ablation`, execution 5444.
  Related norm-placement discussion in #4952 and optimizer-routing work in
  #5750; neither snippet established an existing result for this intervention.
- Scope: combined change, no component-level attribution. GatedNorm has no
  built-in RMS normalization; use it literally as the replacement requested.
- Four initial full cells: d512 control/treatment and d768 control/treatment.
  Nemotron mix and v5p-8 follow Agent MoE; current hero shape and Aug optimizer
  are matched across arms. Harrier data and GB200 transport are not reproduced.
- Cluster read-only preflight: `uv run --no-sync iris --cluster=marin cluster status`.
  Running True, Healthy True, 363/363 workers healthy.
- Issue #9110 created and attached as a sub-issue of #4281. Zihan Qiu credited.
- Next: numerical tests, lint, bounded accelerator smoke, then full Gate 1.

### 2026-09-11 — Local verification

- Numerical suite: `uv run --no-project infra/ci/run_tests.py --base-ref marin/main --workers 2`:
  4 passed. Control output/routing exactly matches the upstream hero; the treatment
  is invariant to down-projection-nullspace perturbations; router gradients reach
  the down projection and latent gate; dense all-expert output matches grouped dispatch.
- Variant contracts: `uv run --no-sync pytest tests/test_grug_variant_contracts.py -q -n 2`:
  19 passed, 1 pre-existing skip.
- Explicit Pyrefly checks on the new model, trainer, launcher and test file: 0 errors.
- Required `infra/pre-commit.py` lint/format checks on all changed files: passed.
- The new gate key is folded with 4, disjoint from the existing four initialization
  splits. The source model's obsolete shard_map compatibility import was replaced
  with the current JAX import. HFCheckpointConverter support is omitted from this
  training-only variant; the research checkpoint format is the native Grug state.
- Dry-run lowering: only the research checkpoint is mutable; the nine training
  caches are pinned and all report SUCCESS in us-central1.
- Computed full schedules: d512 13,642 steps / 1,788,084,224 tokens;
  d768 19,378 / 5,079,826,432; d1024 16,425 / 8,611,430,400;
  d1280 14,473 / 15,176,040,448. Both arms use these same schedules.
- A scratch dense-reference test initially had ambiguous explicit sharding.
  The reference now uses NumPy dense expert evaluation; no tolerance was relaxed.

### 2026-09-11 — Placement preflight

- All 32 pinned dependencies report SUCCESS in us-central1: nine training,
  sixteen Paloma and seven Uncheatable caches. This check read status metadata only.
- The first smoke submission was rejected before creating a job: the CPU parent
  combined a us-central1 region constraint, inferred non-preemptible placement,
  and `--reserve v5p-8` availability. The controller reported no matching groups.
- Pin the TPU child explicitly to us-central1 and retry the CPU parent without
  the accelerator availability constraint. No production job was changed.

### 2026-09-11 — Smoke submission and scope correction

- Source checkpoint: `da78e56d0`. The smoke parent
  `/kaiyuew/moe-lgr-9110-d512-gated-smoke` is running and dispatched only
  `/kaiyuew/moe-lgr-9110-d512-gated-smoke/grug-train-moe-lgr-9110-d512-gated-smoke`.
  The child is capacity-pending, with no training progress verified yet.
- User clarification: “我们不需要跑baseline吧”. Do not submit any new baseline
  training. Reuse existing Agent MoE references; only two treatment cells per gate.
  The fresh-control plan above is superseded. No baseline job was submitted.
- The existing May Recipe references differ in architecture, optimizer, token count
  and v4-32 hardware. Report compute-equivalent gain and measured wall-clock
  comparison separately; this reference cannot isolate the requested intervention.
- Monitoring ownership transfers to the current task's heartbeat
  `follow-gated-latent-moe-ablation-9110` (10-minute cadence). The state file is
  `scratch/20260911-1028_moe-lgr-9110_monitoring_state.json`; the durable
  [handoff](../projects/moe-latent-gated-router/handoff.md) records commands, gates,
  checkpoint locations, and recovery limits. Stop the heartbeat after the gate
  decision or an unrecoverable failure. Normal capacity waits should stay quiet.
- Verified the existing W&B references are both `finished`: d512 Paloma macro
  3.54216671 at 1,439,170,560 tokens; d768 3.22727251 at 4,423,680,000 tokens.
  Full config/summary snapshots are in `scratch/9110-existing-baselines.json`.
  Retrieve their final 100 throughput samples when calculating the final comparison.
- 17:36 UTC update: smoke child allocated, JAX initialized, expected W&B run
  started, and checkpoint loader confirmed a fresh start. Training loss and final
  smoke completion are still pending.

### 2026-09-11 — Automatic preemption recovery

- At 17:48 UTC W&B showed `crashed` while Iris still showed the child running.
  `iris task describe` resolved the apparent conflict: attempt 0 was scheduler-
  preempted, and attempt 1 was already running on another v5p-8 worker.
- Attempt-specific logs show the same run ID reinitialized, read the existing
  caches and entered training at 0/5 at 17:49:13 UTC. A thread profile confirmed
  a live Python process during cache loading. No model exception was observed.
- No job was manually stopped/resubmitted and no code or cluster settings changed.
  Manual recovery count remains zero. Full Gate 1 is still conditional on smoke success.
- CLI correction: this checkout has `task describe` and `attempt logs`, but no
  `job summary`; use the current task/attempt inspection surface on future ticks.
