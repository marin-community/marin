# MoE MuonH + Router z-loss 4e-3: Research Logbook

## Scope

- Goal: Test whether raising `GrugModelConfig.router_z_loss_coef` from `1e-3` to `4e-3` on top of the corrected MuonH matrix-swap improves effective speedup over both the v16 AdamH baseline and the MuonH baseline-adam-mask gate-1 results from #5596.
- Primary metrics: `eval/paloma/macro_loss`, `throughput/tokens_per_second`, `throughput/total_tokens`, run state.
- Constraints: Keep model sizing, data, batch size, step count, schedule, output z-loss (`z_loss_weight=1e-4`), eval cadence, and the MuonH optimizer mask fixed; the only knob that changes is `router_z_loss_coef`.
- Issue: https://github.com/marin-community/marin/issues/5600
- PR: https://github.com/marin-community/marin/pull/5601 (stacked on #5597)

## Baseline

- Date: 2026-05-09
- Code refs: `experiments/grug/moe/muonh_router_zloss_sweep.py`, `experiments/grug/moe/muonh_matrix_sweep.py`, `experiments/grug/moe/heuristic.py`, `experiments/grug/moe/launch.py`, `experiments/grug/moe/model.py:72` (default `router_z_loss_coef=0.001`).
- Comparators: README compute-optimal table (`experiments/grug/moe/README.md`) and the corrected MuonH baseline-adam-mask gate-1 results from #5596.

## Experiment Log

### 2026-05-09 14:25 - MOE-RZ-001 launcher and validation

- Hypothesis: Raising the router z-loss coefficient should be a single isolated knob that doesn't interact with the rest of the MuonH stack.
- Command: `uv run pytest -o addopts='' tests/test_grug_moe_optimizer.py tests/test_grug_variant_contracts.py -q`
- Config: `experiments/grug/moe/muonh_router_zloss_sweep.py`, `MUONH_RZLOSS_GATE=1` default, `entity="marin-community"` pinned.
- Result: 19 tests passed. `./infra/pre-commit.py --fix experiments/grug/moe/muonh_router_zloss_sweep.py` clean. Smoke build of `_build_steps('1')` confirmed `router_z_loss_coef=0.004` on both d512 and d768 model configs.
- Interpretation: Launcher is ready; only `GrugModelConfig.router_z_loss_coef` differs from the corrected MuonH baseline-adam-mask launcher.
- Next action: Open PR #5601 and launch gate 1 on Iris preemptible v5p-8.

### 2026-05-09 15:29 - MOE-RZ-002 gate-1 launch

- Hypothesis: Tighter router-logit regularization (4× the README default) under MuonH may stabilize routing and improve effective speedup at gate 1.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --preemptible --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e MUONH_RZLOSS_GATE 1 -- python -m experiments.grug.moe.muonh_router_zloss_sweep`
- Config: PR #5601 commit `2f91bba79`, gate 1, v5p-8 preemptible.
- Result: Parent `/kaiyue/iris-run-job-20260509-222905` running; both children created and progressed past startup with no `Traceback` or `ShardingTypeError`.
- Interpretation: The 4e-3 router z-loss coefficient runs cleanly through MuonH without compile/lower regressions.
- Next action: Wait for both runs to finish, then compare against #5596 MuonH baseline-adam-mask and the README v16 AdamH baseline.

### 2026-05-09 23:00 - MOE-RZ-003 gate-1 completion and verdict

- Hypothesis: Router z-loss bumped from 1e-3 to 4e-3 helps the MuonH variant.
- Command: wandb pull from `marin-community/marin_moe`.
- Config: Same launcher and run IDs as MOE-RZ-002.
- Result: Both runs `state=finished`, parent `/kaiyue/iris-run-job-20260509-222905` succeeded.

| Scale | Run | macro_loss | tps | total_tokens | step |
|---|---|---|---|---|---|
| d512 (2.19e17) | v16 AdamH baseline | 3.8104 | 407,378 | 8.37e8 | 6386 |
| d512 (2.19e17) | MuonH baseline-adam-mask (#5596) | 3.7542 | 411,620 | 8.37e8 | 6386 |
| d512 (2.19e17) | MuonH + router-zloss-4e-3 | **3.7499** | 411,230 | 8.37e8 | 6386 |
| d768 (1.70e18) | v16 AdamH baseline | 3.4339 | 274,431 | 2.71e9 | 10342 |
| d768 (1.70e18) | MuonH baseline-adam-mask (#5596) | 3.3988 | 281,211 | 2.71e9 | 10342 |
| d768 (1.70e18) | MuonH + router-zloss-4e-3 | **3.3881** | 280,188 | 2.71e9 | 10342 |

Effective speedup (`L_inf=1.6, alpha=0.0941`):

| Scale | vs v16 AdamH (wall) | vs v16 AdamH (step-wise) | vs MuonH baseline-adam-mask (wall) | vs MuonH baseline-adam-mask (step-wise) |
|---|---|---|---|---|
| d512 | **1.36** | 1.34 | 1.02 | 1.02 |
| d768 | **1.34** | 1.31 | 1.06 | 1.07 |

- Interpretation: Router z-loss 4e-3 passes gate 1 cleanly against the v16 AdamH baseline at both scales (1.36× / 1.34× wall-clock), and improves over the corrected MuonH baseline-adam-mask by ~+2% at d512 and ~+6% at d768. The wider gain at d768 suggests the regularization compounds with scale.
- W&B report: https://wandb.ai/marin-community/marin_moe/reports/MuonH-router-zloss-4e-3-vs-MuonH-vs-v16-AdamH-—-gate-1--VmlldzoxNjgyOTMwNg==
- Next action: Decide on launching gate 2 (d1024 at 9.00e18 and d1280 at 2.83e19) for the router-zloss-4e-3 variant; the d768 trend is encouraging.
