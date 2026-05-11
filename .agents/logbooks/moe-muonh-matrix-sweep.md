# MoE MuonH Matrix Sweep: Research Logbook

## Scope

- Goal: Test whether replacing AdamH/AdamH-expert with MuonH while preserving the AdamH baseline Adam group improves effective speedup versus the v16 AdamH baseline.
- Primary metrics: `eval/paloma/macro_loss`, `throughput/tokens_per_second`, `throughput/total_tokens`, run state.
- Constraints: Keep model sizing, data, batch size, step count, schedule, z-loss, eval cadence, and lm-head AdamH behavior fixed.
- Issue: https://github.com/marin-community/marin/issues/5596

## Baseline

- Date: 2026-05-09
- Code refs: `experiments/grug/moe/README.md`, `experiments/grug/moe/heuristic.py`, `experiments/grug/moe/launch.py`
- Baseline numbers: use the README compute-optimal table. Gate 1 compares d512 at 2.19e17 FLOPs and d768 at 1.70e18 FLOPs.

## Experiment Log

### 2026-05-09 11:20 - MOE-MH-001 implementation validation

- Hypothesis: MuonH can replace AdamH on Grug MoE matrix-shaped leaves while keeping the lm head on AdamH and vectors/scalars on Adam.
- Command: `uv run pytest -o addopts='' tests/test_grug_moe_optimizer.py tests/test_grug_variant_contracts.py -q`
- Config: `GrugMoeMuonHConfig`, `MUONH_MATRIX_GATE=1` launcher defaults.
- Result: 11 tests passed. The launch-builder smoke check produced gate-1 steps `muonh-matrix-d512-2.19e17` and `muonh-matrix-d768-1.70e18`.
- Interpretation: The optimizer mask, single-device update path, and Grug variant contracts are ready for gate-1 launch.
- Next action: Open PR, launch gate-1 Iris jobs, and monitor W&B runs under group `muonh-matrix-sweep`.

### 2026-05-09 11:23 - MOE-MH-002 first launch killed after sharding failure

- Hypothesis: The local single-device update test was insufficient for stacked expert tensor sharding.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --preemptible --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e MUONH_MATRIX_GATE 1 -- python -m experiments.grug.moe.muonh_matrix_sweep`
- Config: PR #5597 commit `4804d6350`, gate 1, v5p-8.
- Result: Jobs `/kaiyue/iris-run-job-20260509-181301/grug-train-muonh-matrix-d512-2.19e17` and `/kaiyue/iris-run-job-20260509-181301/grug-train-muonh-matrix-d768-1.70e18` reached W&B startup, then d768 failed during lowering with `ShardingTypeError: mul got incompatible shardings for broadcasting`. The parent job and both children were killed.
- Interpretation: `_grug_scale_with_muon` can return a direction update whose leading-axis sharding differs from stacked expert parameter sharding. The direction must be resharded to the parameter layout before the hyperball update multiplies by per-expert norms.
- Next action: Add an abstract-mesh regression covering `P("expert", "data", "model")` expert tensors, fix the shared hyperball helper, then relaunch.

### 2026-05-09 11:37 - MOE-MH-003 gate-1 relaunch after sharding fix

- Hypothesis: Resharding Muon directions to parameter layouts before hyperball norm math removes the stacked-expert lowering failure.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --preemptible --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e MUONH_MATRIX_GATE 1 -- python -m experiments.grug.moe.muonh_matrix_sweep`
- Config: PR #5597 commit `4a7867902`, gate 1, v5p-8 preemptible.
- Result: Parent `/kaiyue/iris-run-job-20260509-183227` is running. Child jobs `/kaiyue/iris-run-job-20260509-183227/grug-train-muonh-matrix-d512-2.19e17` and `/kaiyue/iris-run-job-20260509-183227/grug-train-muonh-matrix-d768-1.70e18` were created and are pending while the Iris autoscaler brings up preemptible v5p-8 workers.
- Interpretation: The fixed code was accepted by Iris and progressed past parent submission; live TPU compile validation is blocked on capacity.
- Next action: Wait for workers, confirm W&B startup, and check that lowering passes for both child jobs.

### 2026-05-09 11:58 - MOE-MH-004 baseline Adam group correction

- Hypothesis: The ablation should preserve the AdamH baseline's Adam group so router, token embedding, attention gate, router bias, and vector/scalar behavior stays fixed while only AdamH/AdamH-expert matrix groups switch to MuonH.
- Command: `uv run pytest -o addopts='' tests/test_grug_moe_optimizer.py::test_grug_moe_muonh_keeps_adamh_baseline_adam_group_on_adam tests/test_grug_moe_optimizer.py::test_muonh_matrix_sweep_suffix_builds_distinct_relaunch_steps -q`
- Config: `GrugMoeMuonHConfig`, gate-1 relaunch suffix `baseline-adam-mask`.
- Result: Added failing tests, then updated the mask and launcher so the baseline Adam group remains Adam and corrected relaunches use distinct run IDs such as `muonh-matrix-baseline-adam-mask-d512-2.19e17`.
- Interpretation: The already-running MOE-MH-003 jobs are useful as a broader matrix-swap reference, but the corrected experimental comparison requires a new suffixed launch.
- Next action: Run full focused validation, push PR update, then launch corrected gate-1 jobs without stopping MOE-MH-003.

### 2026-05-09 12:10 - MOE-MH-005 corrected gate-1 launch

- Hypothesis: Preserving the AdamH baseline Adam group isolates the AdamH-to-MuonH matrix replacement from router/token-embedding/attention-gate effects.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --preemptible --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e MUONH_MATRIX_GATE 1 -e MUONH_MATRIX_RUN_SUFFIX baseline-adam-mask -- python -m experiments.grug.moe.muonh_matrix_sweep`
- Config: PR #5597 commit `64bb37fc3`, gate 1, v5p-8 preemptible, run suffix `baseline-adam-mask`.
- Result: Parent `/kaiyue/iris-run-job-20260509-190202` is running. Child `/kaiyue/iris-run-job-20260509-190202/grug-train-muonh-matrix-baseline-adam-mask-d768-1.70e18` is running and has a W&B run `muonh-matrix-baseline-adam-mask-d768-1.70e18`. Child `/kaiyue/iris-run-job-20260509-190202/grug-train-muonh-matrix-baseline-adam-mask-d512-2.19e17` is pending on v5p-8 preemptible capacity.
- Interpretation: The corrected suffixed launch is accepted by Iris and avoids W&B/output-path collisions with earlier runs.
- Next action: Wait for d512 capacity and for d768 to log training steps, then compare against the v16 AdamH baseline.

### 2026-05-09 12:14 - MOE-MH-006 corrected W&B startup

- Hypothesis: Corrected MuonH gate-1 jobs should get through startup without the previous expert-sharding lowering failure.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job logs /kaiyue/iris-run-job-20260509-190202 --since-seconds 900 --tail --max-lines 800 | rg -i "wandb:|traceback|shardingtypeerror|exception|\\berror\\b|global_step|tokens_per_second|finished|compil|lower"`
- Config: Parent `/kaiyue/iris-run-job-20260509-190202`, run suffix `baseline-adam-mask`.
- Result: Both corrected MuonH children are running with zero failures/preemptions. W&B runs are live at `muonh-matrix-baseline-adam-mask-d512-2.19e17` and `muonh-matrix-baseline-adam-mask-d768-1.70e18`. No `Traceback` or `ShardingTypeError` appeared in the recent log scan.
- Interpretation: Corrected MuonH reached W&B startup for both gate-1 points; wait for step metrics before assessing quality/throughput.
- Next action: Monitor W&B until both runs log `global_step`, `throughput/tokens_per_second`, and eval metrics.

### 2026-05-10 17:00 - MOE-MH-007 gate-1 completion and verdict

- Hypothesis: Corrected MuonH baseline-adam-mask passes gate 1 against the v16 AdamH README baseline.
- Command: wandb pull from `marin-community/marin_moe`.
- Config: PR #5597 commit `c6dd1df58`, suffix `baseline-adam-mask`.
- Result: Both gate-1 runs `state=finished`. d768 was preempted at step 3000 and resumed cleanly from checkpoint.

| Scale | Run | macro_loss | tps | step |
|---|---|---|---|---|
| d512 (2.19e17) | v16 AdamH baseline | 3.8104 | 405,630 | 6386 |
| d512 (2.19e17) | MuonH baseline-adam-mask | 3.7542 | 411,620 | 6386 |
| d768 (1.70e18) | v16 AdamH baseline | 3.4339 | 273,532 | 10342 |
| d768 (1.70e18) | MuonH baseline-adam-mask | 3.3988 | 281,211 | 10342 |

Effective speedup (`L_inf=1.6, alpha=0.0941`):

| Scale | Wall-clock | Step-wise |
|---|---|---|
| d512 | 1.33 | 1.32 |
| d768 | 1.26 | 1.23 |

- Interpretation: Passes gate 1 cleanly. ~+1-2% tps gain plus a ~0.04-0.06 macro_loss reduction at both scales.
- W&B report: https://wandb.ai/marin-community/marin_moe/reports/MuonH-vs-v16-AdamH-—-gate-1-(d512-+-d768)--VmlldzoxNjgyNzg5Ng==
- Next action: Launch gate 2 (d1024 at 9.00e18 and d1280 at 2.83e19) under the same suffix.

### 2026-05-10 17:07 - MOE-MH-008 gate-2 launch and preemption-driven relaunches

- Hypothesis: MuonH should continue to beat the v16 AdamH baseline at d1024 and d1280.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --preemptible --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e MUONH_MATRIX_GATE 2 -e MUONH_MATRIX_RUN_SUFFIX baseline-adam-mask -- python -m experiments.grug.moe.muonh_matrix_sweep`
- Config: PR #5597 commit `054f7c6cc` (entity-pin on top of `c6dd1df58`), suffix `baseline-adam-mask`, preemptible v5p-8.
- Result: Three preemption / capacity events required relaunches:
  1. `/kaiyue/iris-run-job-20260510-000716` — d1024 and d1280 both preempted at 05:30 UTC. Parent exhausted `max_task_failures`, went to `failed`.
  2. `/kaiyue/iris-run-job-20260510-060004` — relaunch stuck pending ~50 min on `tpu_v5e-preemptible_16-europe-west4-b` autoscaler capacity. Terminated.
  3. `/kaiyue/iris-run-job-20260510-065540` — relaunched with `--region us-east5 --region us-central2` (had to drop `--reserve` as the CLI rejects the combination). Landed in us-east5-a where prior gate-1 runs ran. d1024 succeeded; d1280 was preempted again at step 9010.
  4. `/kaiyue/iris-run-job-20260511-033447` — final relaunch with `--region us-east5` only. d1280 resumed from the latest GCS checkpoint and ran to completion.
- Interpretation: The us-east5-a v5p-8 preemptible pool (145 ready slices, `availability_status=requesting`) is the proven landing zone for this workload. `us-central1-a` is `quota_exceeded`; `us-central2` has no v5p-8 pool. Submitting with `--region us-east5` (no `--reserve`) is the right pattern for this workload going forward.
- Next action: Pull final metrics for d1024 and d1280, compute speedups + scaling-law projection, record gate-2 verdict.

### 2026-05-11 03:50 - MOE-MH-009 gate-2 completion and scaling projection

- Hypothesis: MuonH passes gate 2 — wall-clock speedup > 1 at all four scales AND projected MuonH macro_loss < baseline projection at 1e21 and 1e23.
- Command: wandb pull for all four MuonH runs; scaling-law fit via `scipy.optimize.curve_fit` on `loss(C) = 1.6 + A * C^(-alpha)`.
- Config: Same code (PR #5597 commit `054f7c6cc`), suffix `baseline-adam-mask`, all four runs in `marin-community/marin_moe`.
- Result: All four MuonH gate points finished.

| Scale | Budget | Baseline macro_loss | Baseline tps | MuonH macro_loss | MuonH tps | MuonH step |
|---|---|---|---|---|---|---|
| d512 | 2.19e17 | 3.8104 | 405,630 | 3.7542 | 411,620 | 6,386 |
| d768 | 1.70e18 | 3.4339 | 273,532 | 3.3988 | 281,211 | 10,342 |
| d1024 | 9.00e18 | 3.1605 | 175,165 | 3.1357 | 179,731 | 12,648 |
| d1280 | 2.83e19 | 3.0065 | 128,277 | 2.9888 | 133,289 | 11,806 |

Effective speedup (`L_inf=1.6, alpha=0.0941`):

| Scale | Wall-clock | Step-wise |
|---|---|---|
| d512 | 1.33 | 1.32 |
| d768 | 1.26 | 1.23 |
| d1024 | 1.22 | 1.19 |
| d1280 | **1.19** | 1.14 |

MuonH scaling-law fit on the 4 measured optima (L∞=1.6 pinned):

- `A = 80.22`
- `alpha = 0.0906`
- Fit residuals (predicted vs actual): d512 −0.003, d768 +0.007, d1024 −0.005, d1280 +0.0001.

Projections:

| Budget | MuonH projected | Baseline projected | Δ |
|---|---|---|---|
| 1e21 | 2.6055 | 2.606 | **−0.0005** (MuonH lower, essentially tied) |
| 1e23 | 2.2626 | 2.252 | **+0.0106** (MuonH higher — fails) |

- Interpretation: Gate 2 passes the **all-scales speedup** criterion (>1 wall-clock at every measured budget) but **fails the 1e23 projection criterion**: MuonH's fitted alpha (0.0906) is slightly shallower than the baseline's (0.0941), so extrapolating with L∞=1.6 pinned predicts the baseline catches up and overtakes MuonH between 1e21 and 1e23. In the measured range (2.19e17 to 2.83e19) MuonH is uniformly better; outside that range it depends on whether the alpha estimate is real or just a small-sample artifact.
- W&B reports:
  - Gate 1: https://wandb.ai/marin-community/marin_moe/reports/MuonH-vs-v16-AdamH-—-gate-1-(d512-+-d768)--VmlldzoxNjgyNzg5Ng==
  - Gate 2: https://wandb.ai/marin-community/marin_moe/reports/MuonH-vs-v16-AdamH-—-gate-2-(d1024-+-d1280)--VmlldzoxNjgyODM0Ng==
- Next action: Decide on promotion. The speedup is real and consistent in-range, but the projection caveat at 1e23 is a real concern. Consider (a) re-running one of the existing baseline points with the same recipe to tighten the alpha estimate, or (b) collecting a fifth MuonH point at higher compute to anchor the fit. The router-zloss-4e-3 variant (#5600) is a candidate refinement that has shown ~+6% wall-clock over baseline-adam-mask MuonH at d768 and may have a steeper alpha.
