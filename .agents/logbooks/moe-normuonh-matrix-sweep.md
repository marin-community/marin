# MoE NorMuonH Matrix Sweep: Research Logbook

## Scope

- Goal: Test whether NorMuon inside the Grug hyperball update improves when preserving the AdamH baseline Adam group and replacing AdamH/AdamH-expert matrix groups.
- Primary metrics: `eval/paloma/macro_loss`, `throughput/tokens_per_second`, `throughput/total_tokens`, run state.
- Constraints: Keep model sizing, data, batch size, step count, schedule, z-loss, eval cadence, and lm-head AdamH behavior fixed.
- Issue: https://github.com/marin-community/marin/issues/5598
- Paired comparison: https://github.com/marin-community/marin/issues/5596
- Paper: https://arxiv.org/abs/2510.05491v1

## Baseline

- Date: 2026-05-09
- Code refs: `experiments/grug/moe/README.md`, `experiments/grug/moe/heuristic.py`, `experiments/grug/moe/muonh_matrix_sweep.py`
- Baseline numbers: use the README compute-optimal table. Gate 1 compares d512 at 2.19e17 FLOPs and d768 at 1.70e18 FLOPs.

## Implementation Notes

- NorMuon tracks `mean_cols(O * O)` for `W in R^{m x n}` in the paper.
- Grug arrays use `(fan_in, fan_out)` layout, unlike the PyTorch-style convention in the paper.
- The Grug implementation therefore tracks the NorMuon statistic on the trailing output axis.
- For stacked expert tensors `(num_experts, fan_in, fan_out)`, the NorMuon statistic shape is `(num_experts, fan_out)`.
- Attempted to add #5598 as a sub-issue of #4281, but GitHub rejected it because #4281 already has 100 sub-issues.

## Experiment Log

### 2026-05-09 11:18 - MOE-NMH-001 implementation validation

- Hypothesis: Output-axis NorMuon normalization inside the hyperball update may improve over MuonH by reducing per-output update imbalance after orthogonalization.
- Command: `uv run pytest -o addopts='' tests/test_grug_moe_optimizer.py -q`
- Config: `GrugMoeNorMuonHConfig`, `NORMUONH_MATRIX_GATE=1` launcher defaults.
- Result: 7 tests passed. The focused test suite includes a state-shape check pinning the Grug `(fan_in, fan_out)` output-axis convention.
- Interpretation: The optimizer mask, output-axis second-moment state, single-device update path, and gate-1 launcher are ready for broader Grug validation.
- Next action: Run Grug variant contracts, update PR #5597, then launch gate-1 Iris jobs.

### 2026-05-09 11:23 - MOE-NMH-002 shared expert-sharding regression

- Hypothesis: NorMuonH has the same stacked-expert sharding risk as MuonH because it also consumes `_grug_scale_with_muon` directions before the hyperball update.
- Command: `uv run pytest -o addopts='' tests/test_grug_moe_optimizer.py::test_grug_hyperball_update_handles_expert_parameter_sharding -q`
- Config: abstract mesh `("data", "expert", "model")`, expert parameter sharding `P("expert", "data", "model")`.
- Result: The regression initially failed for both MuonH and NorMuonH with incompatible direction/norm shardings. After resharding direction updates to parameter layouts inside the shared hyperball helper, both variants passed.
- Interpretation: The sharding fix covers NorMuonH without a separate optimizer-specific path.
- Next action: Run the full focused optimizer and Grug contract tests, then update PR #5597.

### 2026-05-09 11:37 - MOE-NMH-003 gate-1 launch

- Hypothesis: NorMuon output-axis normalization can be tested under the same gate-1 sizing as the MuonH ablation once the shared expert-sharding fix is in place.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --preemptible --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e NORMUONH_MATRIX_GATE 1 -- python -m experiments.grug.moe.normuonh_matrix_sweep`
- Config: PR #5597 commit `4a7867902`, gate 1, v5p-8 preemptible.
- Result: Parent `/kaiyue/iris-run-job-20260509-183259` is running. Child jobs `/kaiyue/iris-run-job-20260509-183259/grug-train-normuonh-matrix-d512-2.19e17` and `/kaiyue/iris-run-job-20260509-183259/grug-train-normuonh-matrix-d768-1.70e18` were created and are pending while the Iris autoscaler brings up preemptible v5p-8 workers.
- Interpretation: The NorMuonH launcher and job tree are accepted by Iris; live TPU compile validation is blocked on capacity.
- Next action: Wait for workers, confirm W&B startup, and check that lowering passes for both child jobs.

### 2026-05-09 11:58 - MOE-NMH-004 baseline Adam group correction

- Hypothesis: NorMuonH should preserve the AdamH baseline's Adam group so router, token embedding, attention gate, router bias, and vector/scalar behavior stays fixed while only AdamH/AdamH-expert matrix groups switch to NorMuonH.
- Command: `uv run pytest -o addopts='' tests/test_grug_moe_optimizer.py::test_grug_moe_normuonh_keeps_adamh_baseline_adam_group_on_adam tests/test_grug_moe_optimizer.py::test_normuonh_matrix_sweep_suffix_builds_distinct_relaunch_steps -q`
- Config: `GrugMoeNorMuonHConfig`, gate-1 relaunch suffix `baseline-adam-mask`.
- Result: Added failing tests, then updated the mask and launcher so the baseline Adam group remains Adam and corrected relaunches use distinct run IDs such as `normuonh-matrix-baseline-adam-mask-d512-2.19e17`.
- Interpretation: The already-running MOE-NMH-003 jobs are useful as a broader matrix-swap reference, but the corrected experimental comparison requires a new suffixed launch.
- Next action: Run full focused validation, push PR update, then launch corrected gate-1 jobs without stopping MOE-NMH-003.

### 2026-05-09 12:10 - MOE-NMH-005 corrected gate-1 launch

- Hypothesis: Preserving the AdamH baseline Adam group isolates NorMuonH's effect on the AdamH/AdamH-expert matrix groups from router/token-embedding/attention-gate effects.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --preemptible --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e NORMUONH_MATRIX_GATE 1 -e NORMUONH_MATRIX_RUN_SUFFIX baseline-adam-mask -- python -m experiments.grug.moe.normuonh_matrix_sweep`
- Config: PR #5597 commit `64bb37fc3`, gate 1, v5p-8 preemptible, run suffix `baseline-adam-mask`.
- Result: Parent `/kaiyue/iris-run-job-20260509-190231` is running. Children `/kaiyue/iris-run-job-20260509-190231/grug-train-normuonh-matrix-baseline-adam-mask-d512-2.19e17` and `/kaiyue/iris-run-job-20260509-190231/grug-train-normuonh-matrix-baseline-adam-mask-d768-1.70e18` are pending on v5p-8 preemptible capacity.
- Interpretation: The corrected suffixed launch is accepted by Iris and avoids W&B/output-path collisions with earlier runs.
- Next action: Wait for capacity and W&B startup, then compare against the v16 AdamH baseline and MuonH corrected launch.

### 2026-05-09 12:14 - MOE-NMH-006 corrected W&B startup

- Hypothesis: Corrected NorMuonH gate-1 jobs should get through startup without the previous expert-sharding lowering failure.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job logs /kaiyue/iris-run-job-20260509-190231 --since-seconds 900 --tail --max-lines 800 | rg -i "wandb:|traceback|shardingtypeerror|exception|\\berror\\b|global_step|tokens_per_second|finished|compil|lower"`
- Config: Parent `/kaiyue/iris-run-job-20260509-190231`, run suffix `baseline-adam-mask`.
- Result: Both corrected NorMuonH children are running with zero failures/preemptions. W&B runs are live at `normuonh-matrix-baseline-adam-mask-d512-2.19e17` and `normuonh-matrix-baseline-adam-mask-d768-1.70e18`. No `Traceback` or `ShardingTypeError` appeared in the recent log scan.
- Interpretation: Corrected NorMuonH reached W&B startup for both gate-1 points; wait for step metrics before assessing quality/throughput.
- Next action: Monitor W&B until both runs log `global_step`, `throughput/tokens_per_second`, and eval metrics.

### 2026-05-11 16:30 - MOE-NMH-007 gate-1 completion and verdict

- Hypothesis: NorMuonH baseline-adam-mask passes gate 1 against the v16 AdamH README baseline.
- Command: wandb pull from `marin-community/marin_moe`. d768 resumed from preemption via parent `/kaiyue/iris-run-job-20260511-162245` (preemptible v5p-8, `--region us-east5`).
- Config: PR #5597 commit on branch `research/moe-muonh-lr-sweep`, suffix `baseline-adam-mask`.
- Result: Both gate-1 runs `state=finished`. d768 absorbed several preemptions before completing (Iris auto-restart from checkpoints; no manual intervention required).

| Scale | Run | macro_loss | tps | step |
|---|---|---|---|---|
| d512 (2.19e17) | v16 AdamH baseline | 3.8104 | 405,630 | 6386 |
| d512 (2.19e17) | MuonH baseline-adam-mask (#5596) | 3.7542 | 411,620 | 6386 |
| d512 (2.19e17) | NorMuonH baseline-adam-mask | 3.7512 | 407,697 | 6386 |
| d768 (1.70e18) | v16 AdamH baseline | 3.4339 | 273,532 | 10342 |
| d768 (1.70e18) | MuonH baseline-adam-mask (#5596) | 3.3988 | 281,211 | 10342 |
| d768 (1.70e18) | NorMuonH baseline-adam-mask | 3.3964 | 280,626 | 10342 |

Effective speedup (`L_inf=1.6, alpha=0.0941`):

| Scale | vs v16 AdamH (wall) | vs v16 AdamH (step) | vs MuonH (wall) | vs MuonH (step) |
|---|---|---|---|---|
| d512 | 1.34 | 1.33 | 1.005 | 1.01 |
| d768 | 1.28 | 1.25 | 1.012 | 1.01 |

- Interpretation: NorMuonH passes gate 1 cleanly against the v16 AdamH baseline and is essentially tied with MuonH at d512 / slightly better (~+1% wall-clock) at d768. The output-axis normalization is at worst neutral and may help marginally at larger scale.
- W&B report: https://wandb.ai/marin-community/marin_moe/reports/NorMuonH-vs-MuonH-vs-v16-AdamH-—-gate-1--VmlldzoxNjg0Nzc3Nw==
- Next action: Auto-launched gate 2 (d1024 at 9.00e18 and d1280 at 2.83e19) under parent `/kaiyue/iris-run-job-20260511-232705` without waiting for confirmation, per user's standing instruction. A gate-2 babysitter is running it to completion with auto-build of the W&B report, scaling-law fit, projection vs baseline at 1e21/1e23, and PushNotification on completion.
