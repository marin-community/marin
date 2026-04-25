# MoE MuonH Heuristic / AdamH 2x / Muon Search: Research Logbook

## Scope
- Goal: Define a provisional MuonH heuristic, run an AdamH 2x-batch control, and launch a Vizier search over Muon LR and beta/momentum terms at nearby small scales.
- Primary metric(s): `eval/paloma/macro_loss`, `train/loss`, `throughput/tokens_per_second`, and effective speedup versus the compute-optimal AdamH baselines.
- Constraints: Keep Grug MoE architecture, Nemotron mix, eval setup, v5p-8 hardware, sequence length, and fixed-token compute accounting unchanged unless the experiment axis says otherwise.
- Issue: https://github.com/marin-community/marin/issues/5167

## Baseline
- Date: 2026-04-24
- Code refs: `experiments/grug/moe/README.md`, `experiments/grug/moe/agent.md`
- Baseline numbers:
  - `d512 @ 2.19e17`: Paloma macro `3.8104`, avg tok/s `405,630`
  - `d768 @ 1.70e18`: Paloma macro `3.4339`, avg tok/s `273,532`
- Prior related issue: #5134

## Experiment Log
### 2026-04-24 17:00 - kickoff and implementation
- Hypothesis: MuonH should use AdamH model and batch sizing, but expose an independent matrix-step multiplier and coefficient choice. AdamH 2x-batch controls test whether the prior MuonH 2x result is a batch-size effect rather than an optimizer-specific gain. Muon may improve if the AOL preset is allowed larger LR multipliers and retuned momentum / Adam beta terms.
- Command:
  - Validation: `./infra/pre-commit.py --fix experiments/grug/moe/optimizer.py experiments/grug/moe/heuristic.py experiments/grug/moe/adamh_batch2x_gate1.py experiments/grug/moe/muon_vizier_search.py`
  - Planned AdamH 2x submission: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.adamh_batch2x_gate1`
  - Planned Muon Vizier submission: `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.muon_vizier_search`
- Config:
  - Branch: `research/moe-muonh-heuristic-adamh2x-muon-search`
  - MuonH heuristic: inherits AdamH model/batch sizing, Adam fallback LR, epsilon, beta2, and schedule; adds `muonh_lr_multiplier`, `adam_lr_multiplier`, `momentum`, `coefficient_type`, and `muon_epsilon`.
  - AdamH 2x control: `d512` batch `64`, steps `3194`; `d768` batch `128`, steps `5172`, preserving baseline token counts.
  - Muon Vizier search scales: `d512-c3e17` with batch `32`, steps `8750`; `d768-c1e18` with batch `32`, steps `12168`.
  - Muon Vizier search space: `muon_lr_multiplier in [0.5, 3.0]`, `adam_lr_multiplier in [0.5, 3.0]`, `momentum in [0.92, 0.99]`, `beta1 in [0.70, 0.95]`, `beta2 in [0.95, 0.999]`.
- Result: code validates with targeted pre-commit. Issue #5167 created and added as a sub-issue of #4281.
- Interpretation: the launch surface is ready for Iris submission. The Vizier design runs two independent studies, one per scale, with three loops of four suggestions each.
- Next action: commit, push, submit the AdamH 2x and Muon Vizier jobs, then record Iris job IDs.

### 2026-04-24 17:10 - Iris submission
- Hypothesis: AdamH 2x-batch control and Vizier Muon search can run through the standard Grug MoE Iris path on v5p-8.
- Command:
  - `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.adamh_batch2x_gate1`
  - `.venv/bin/iris --config lib/iris/examples/marin.yaml job run --no-wait --reserve v5p-8 -e WANDB_API_KEY "$WANDB_API_KEY" -- python -m experiments.grug.moe.muon_vizier_search`
- Config:
  - Commit: `9d815d70c`
  - AdamH 2x parent job: `/pc0618/iris-run-job-20260425-000952`
  - Muon Vizier parent job: `/pc0618/iris-run-job-20260425-001006`
  - Hardware reservation: `v5p-8`
- Result: both parent jobs submitted and running; both v5p-8 reservations were initially pending due to no matching worker with sufficient capacity.
- Interpretation: launch succeeded. The next useful check is whether the reservations bind to workers and child training jobs start emitting W&B runs.
- Next action: monitor Iris status and W&B run creation for `moe-adamh-batch2x-*` and `moe-muon-vizier-lr-beta-*`.
