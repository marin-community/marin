# MoE Per-Expert MuonH LR sqrt(sparsity): Research Logbook

## Scope

- Goal: Test whether routed expert MuonH weights should use a lower LR because each expert sees sparse token traffic.
- Primary metrics: `eval/paloma/macro_loss`, `throughput/tokens_per_second`, effective speedup from `experiments/grug/moe/agent.md`.
- Constraints: Build on #5763 (`may_arch + GN->MuonH + 1% warmup + no grad-clip`) and run gate 1 before spending gate-2 compute.
- Issue: https://github.com/marin-community/marin/issues/5799
- Branch: `moe_muonh_may_arch_per_expert_lr_gate1`
- Launcher: `experiments/grug/moe/muonh_may_arch_per_expert_lr_gate1.py`

## Baseline

- Date: 2026-05-17
- Code refs:
  - `experiments/grug/moe/README.md`
  - `experiments/grug/moe/agent.md`
  - `experiments/grug/moe/muonh_may_arch_gn_muonh_1pct_noclip_sweep.py`
- Gate-1 baseline points:
  - d512, 2.19e17 FLOPs, Paloma macro 3.8104, 405,630 v5p-8 tok/s.
  - d768, 1.70e18 FLOPs, Paloma macro 3.4339, 273,532 v5p-8 tok/s.

## Experiment Log

### 2026-05-17 11:44 - MOE-PELR-001 gate-1 plan

- Hypothesis: Routed expert weights see `sparsity = num_experts_per_token / num_experts = 4 / 256 = 1/64` of token traffic, so their LR should shrink by `sqrt(sparsity) = 0.125` relative to dense MuonH weights.
- Command:

```bash
.venv/bin/iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --reserve v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe.muonh_may_arch_per_expert_lr_gate1
```

- Config:
  - `shrink-expert`: routed expert MuonH LR = 0.125x heuristic; remaining MuonH LR = 1.0x heuristic.
  - `boost-nonexpert`: routed expert MuonH LR = 1.0x heuristic; remaining MuonH LR = 8.0x heuristic.
  - Adam/AdamH groups unchanged.
  - Gate-1 points only: d512 at 2.19e17 FLOPs, d768 at 1.70e18 FLOPs.
- Result: submitted as `/kaiyue/iris-run-job-20260517-184930`.
- Interpretation: pending.
- Next action: validate code, commit, push branch, submit gate-1 job.

### 2026-05-17 12:00 - MOE-PELR-002 mid-ratio add-on

- Hypothesis: The fixed 8:1 non-expert/expert LR ratio may be useful, but the two initial candidates also change the absolute LR scale. A geometric-middle point can separate ratio from absolute scale.
- Command:

```bash
.venv/bin/iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --reserve v5p-8 \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe.muonh_may_arch_per_expert_lr_mid_ratio_gate1
```

- Config:
  - `mid-ratio`: routed expert MuonH LR = `sqrt(0.125 * 1.0) = 0.353553x` heuristic.
  - Remaining MuonH LR keeps the same 8:1 non-expert/expert ratio: `2.828427x` heuristic.
  - Adam/AdamH groups unchanged.
  - Gate-1 points only: d512 at 2.19e17 FLOPs, d768 at 1.70e18 FLOPs.
- Result: submitted as `/kaiyue/iris-run-job-20260517-185409`.
- Interpretation: pending.
- Next action: validate add-on launcher, commit, push branch, submit add-on gate-1 job.
