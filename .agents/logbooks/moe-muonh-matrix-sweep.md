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
