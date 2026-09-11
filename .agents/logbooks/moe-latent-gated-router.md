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
Use paired fresh controls and leave the production hero unchanged.
Experiment prefix: MOE-LGR. W&B group: moe-latent-gated-router-9110 in marin_moe.

## Current baseline

EP hero main `d891fba48a7d3729fdac2595df3c6ca292a53b74`, scaled to small widths.
The control is `RMSNorm(down(h))` for experts and `router(h)` for routing.
The treatment is `z=GatedNorm(down(h))`, used by both router and experts.
Both arms use the same TPU, token budget, data order and optimizer schedule.
See the [plan](../projects/moe-latent-gated-router/plan.md) and
[variant README](../../experiments/grug/moe_latent_gated_router/README.md).

## Hypothesis queue

- Active MOE-LGR-001: The shared gated latent improves loss enough to offset
  runtime overhead at d512 and d768. Evidence is pending; fail if either
  paired effective speedup is <=1.
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
