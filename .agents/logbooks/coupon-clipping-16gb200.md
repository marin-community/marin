---
topic: coupon-clipping-16gb200
issue: https://github.com/marin-community/marin/issues/7836
description: Test shallow knowledge acquisition and layerwise shared-expert capacity on 16 GB200 GPUs.
author: rjpower
---

# Coupon clipping on 16 GB200 GPUs: Research Logbook

## Scope

- Goal: measure whether shallow-to-deep growth or input-adjacent active capacity reaches an interesting early loss faster than uniform full-depth training.
- Primary metrics: held-out Datamix and Paloma loss against allocation wall time, valid tokens, and estimated training FLOPs; factual retrieval versus manipulation probes.
- Constraints: four 4-GPU GB200 nodes per arm, at most 12 hours, with a comparison checkpoint after four hours.
- Design artifact: https://loom.rjp.io/s/9eoiqdub/artifacts/coupon-clipping-experiment-design
- Coordinating issue: https://github.com/marin-community/marin/issues/7836

## Current TL;DR

The pyramid and L1-to-L48 paths are implemented and passing their focused tests. The frozen design uses a `d3072/L48/E64/top4` Grug MoE with 5.295B active and 46.064B stored parameters. The production-path canaries are ready to submit; no training result has landed.

## Current Baseline

- Arm: `CC16-C0`, uniform full-depth control.
- Model: `d3072/L48/E64/top4`, routed expert width 1536, shared expert width 1280.
- Provisional horizon: 6,400 updates at batch 256 and sequence length 4,096, or 6.711B valid tokens.
- Hardware: 16 GB200 GPUs across four nodes; production mesh and horizon are gated on a 128-update sustained pilot.

## Hypothesis Queue

### Active

- `CC16-H1`: a one-layer source expanded to 48 layers can approach the token-matched C0 loss with at least 30% fewer GB200-hours. Next test: `CC16-D1` after measuring post-expansion mixing tokens.
- `CC16-H2`: a fixed-wall one-layer source plus a calibrated full-depth tail reaches C0's loss earlier. Next test: `CC16-D2` after the mixing interval is known.
- `CC16-H3`: placing four `i4096` shared experts at layers 0-3 beats the uniform `i1280` allocation and the same capacity at layers 22-25. Next test: `CC16-P1` versus `CC16-C0` and `CC16-P2`.

### Blocked

- `CC16-H4`: a 12-layer source is a safer growth platform than a one-layer source. Blocker: it is conditional on `CC16-D1` failing the mixing or throughput gate.

### Falsified / Dead End

- None.

### Promoted

- None.

## Decision Log

- 2026-07-31: use approximately 5B active parameters. The earlier 15.8B-active design would receive too few tokens on 16 GPUs.
- 2026-07-31: vary shared-expert width for the primary pyramid. This applies the treatment capacity to every token and removes early router specialization as a required mechanism.
- 2026-07-31: compare fat-first with fat-middle. The middle placement controls for generic layer heterogeneity.
- 2026-07-31: collect a coordinated comparison after four hours; continue healthy runs toward the 12-hour horizon.
- 2026-07-31: use FSDP-only (`expert=1`, `replica=1`, `data=16`). `sonic_cute` is a local SM100 backend and rejects expert-parallel axes larger than one.
- 2026-07-31: preregister D1 at update 4,480 (70%) so the token-matched arm retains 1,920 full-depth updates before the shared 640-update decay.

## Negative Results Index

- None.

## Entry Log

### 2026-07-31 01:20 - CC16-000 experiment charter

- Hypothesis: shallow data exposure and input-adjacent shared capacity may improve early time-to-loss on an intentionally undertrained 5.2B-active MoE.
- Commit Hash: `2bcabffaa906b6d643988f60aefdf457fda37d03` (pre-implementation base).
- Command: launch commands pending implementation and systems validation.
- Config: five core arms (`CC16-C0`, `CC16-D1`, `CC16-D2`, `CC16-P1`, `CC16-P2`) on 16 GB200 GPUs; provisional batch 256, sequence length 4,096, and 6.711B-token full-depth horizon.
- Result: no run data yet.
- Interpretation: undertraining is intentional; conclusions will be scoped to early training.
- Next action: implement the shared-expert segments and depth-growth checkpoint transform, then run the production-path systems/LR gate.

### 2026-07-31 15:35 - CC16-001 implementation gate

- Hypothesis: matched shared-expert placement and deterministic L1-to-L48 growth can be exercised on the corrected GB200 Grug stack without changing the data, optimizer, or mesh contracts between arms.
- Commit Hash: pending research snapshot.
- Command: `uv run --with pytest --with pytest-timeout --with pytest-asyncio pytest -q experiments/grug/coupon_clipping tests/test_grug_depth_growth.py tests/test_grug_variant_contracts.py`
- Config: P0/P1/P2 use segment lengths `(4, 18, 4, 22)` and exact matched accounting of 46,063,592,448 stored / 5,294,957,568 active parameters. D1 uses L1 through update 4,480 and L48 through update 6,400. All production and pilot stages retain the 6,400-step optimizer and Datamix horizons.
- Result: 25 passed and 1 skipped in the wider Grug contract run. One unrelated base-Grug CPU test failed on pre-existing label-concatenation sharding; the 12 coupon/depth tests pass. Artifact construction passes for all C0/P1/P2/D1 and canary entry points.
- Interpretation: parameter, scan, optimizer-bucket, transition-state, data-offset, and artifact dependency contracts are locally enforced. On-hardware memory, compile time, throughput, router stability, and checkpoint restore remain unmeasured.
- Next action: commit and push the reproducible snapshot, submit the bounded canary wave, then launch C0, D1, P1, and P2 in parallel if the gate passes.
