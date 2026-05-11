# MoE MuonH No-Warmup Sweep: Research Logbook

## Scope

- Goal: Test whether removing the LR-schedule warmup phase (0.1 -> 0.0) on top of the corrected MuonH matrix swap improves effective speedup vs the v16 AdamH baseline.
- Primary metrics: `eval/paloma/macro_loss`, `throughput/tokens_per_second`, `throughput/total_tokens`, run state.
- Constraints: Keep model sizing, data, batch size, step count, schedule shape (cosine to `min_lr_ratio`), z-loss, eval cadence, and lm-head AdamH behavior fixed. Only `warmup` changes (0.1 -> 0.0). The optimizer mask follows the corrected MuonH ablation (`baseline-adam-mask`): MuonH on AdamH/AdamH-expert matrix groups, AdamH on the lm head, Adam preserved on the baseline Adam group.
- Issue: https://github.com/marin-community/marin/issues/5619
- PR: https://github.com/marin-community/marin/pull/5620

## Baseline

- Reference: gate 1 of the corrected MuonH matrix sweep (`muonh-matrix-baseline-adam-mask`, see `moe-muonh-matrix-sweep.md`).
- v16 AdamH README baseline: d512 macro_loss 3.8104 (tps 405,630), d768 macro_loss 3.4339 (tps 273,532).
- Corrected MuonH gate-1: d512 macro_loss 3.7542 (tps 411,620), d768 macro_loss 3.3988 (tps 281,211).

## Experiment Log

### 2026-05-11 09:25 - MOE-MH-NW-001 gate-1 launch

- Hypothesis: Removing warmup on the corrected MuonH matrix swap improves both training loss and step efficiency without harming throughput.
- Command: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --preemptible --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -e MUONH_NOWARMUP_GATE 1 -- python -m experiments.grug.moe.muonh_no_warmup_sweep`
- Config: PR #5620 (commit `3271a0360`), gate 1, v5p-8 preemptible, no run suffix.
- Result: Parent `/kaiyue/iris-run-job-20260511-162524` reached `running` at 16:25 UTC. Both children dispatched at 16:28:22 UTC and entered `running` by 16:58 UTC.
- Interpretation: Launch is healthy on us-east5 preemptible v5p-8; nothing to fix.
- Next action: Babysit until both children reach a terminal state, then pull final metrics.

### 2026-05-11 12:35 - MOE-MH-NW-002 gate-1 completion and verdict

- Hypothesis: MuonH-no-warmup beats both the v16 AdamH baseline and the corrected-mask MuonH variant at gate 1.
- Command: wandb pull from `marin-community/marin_moe`; `iris job list --prefix /kaiyue/iris-run-job-20260511-162524` for terminal state.
- Config: Same as MOE-MH-NW-001 (PR #5620, `3271a0360`).
- Result: Both gate-1 runs `state=finished`, no preemption events.

| Scale | Run | macro_loss | tps (median) | step |
|---|---|---|---|---|
| d512 (2.19e17) | v16 AdamH baseline | 3.8104 | 405,630 | 6,386 |
| d512 (2.19e17) | MuonH baseline-adam-mask | 3.7542 | 411,620 | 6,386 |
| d512 (2.19e17) | **MuonH no-warmup** | **3.7404** | **411,158** | 6,386 |
| d768 (1.70e18) | v16 AdamH baseline | 3.4339 | 273,532 | 10,342 |
| d768 (1.70e18) | MuonH baseline-adam-mask | 3.3988 | 281,211 | 10,342 |
| d768 (1.70e18) | **MuonH no-warmup** | **3.3834** | **280,250** | 10,342 |

Effective speedup vs v16 AdamH baseline (`L_inf=1.6, alpha=0.0941`):

| Scale | Wall-clock | Step-wise |
|---|---|---|
| d512 | **1.43** | 1.41 |
| d768 | **1.38** | 1.35 |

Effective speedup vs corrected-mask MuonH (`L_inf=1.6, alpha=0.0941`):

| Scale | Wall-clock | Step-wise |
|---|---|---|
| d512 | 1.07 | 1.07 |
| d768 | 1.09 | 1.10 |

- Interpretation: Removing warmup is a clean win on top of the corrected MuonH matrix swap. Both scales improve in absolute macro_loss (-0.014 at d512, -0.015 at d768) at essentially identical throughput, and the cumulative speedup over the v16 AdamH baseline jumps from ~1.3 (baseline-adam-mask alone) to ~1.4 wall-clock. Gate 1 passes cleanly.
- W&B runs:
  - `marin-community/marin_moe/muonh-nowarmup-d512-2.19e17` (run id `muonh-nowarmup-d512-2.19e17`, runtime 5,110 s)
  - `marin-community/marin_moe/muonh-nowarmup-d768-1.70e18` (run id `muonh-nowarmup-d768-1.70e18`, runtime 13,536 s)
- W&B report (3-way gate 1): https://wandb.ai/marin-community/marin_moe/reports/MuonH-no-warmup-vs-MuonH-vs-v16-AdamH----gate-1--VmlldzoxNjg0Njk4OA==
- Next action: Launch gate 2 (d1024 at 9.00e18, d1280 at 2.83e19) under the same code to confirm the speedup persists at scale.
