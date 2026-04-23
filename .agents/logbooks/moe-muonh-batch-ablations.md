# MoE MuonH Batch Ablations: Research Logbook

## Scope
- Goal: Test whether a clean AdamH to MuonH swap can beat the current AdamH-based Grug MoE recipe at gate 1, and whether MuonH benefits from doubling batch size at the same learning-rate schedule.
- Primary metric(s): `eval/paloma/macro_loss`, `throughput/tokens_per_second`, effective speedup versus the compute-optimal AdamH baseline.
- Constraints: Keep the model architecture, data mix, compute-optimal budgets, and AdamH-derived learning-rate schedule fixed at `d512 / 2.19e17` and `d768 / 1.70e18`. Only change the matrix optimizer family and batch size.
- Issue: https://github.com/marin-community/marin/issues/5134

## Baseline
- Date: 2026-04-23
- Code refs: `experiments/grug/moe/README.md`, `experiments/grug/moe/agent.md`
- Baseline numbers:
  - `d512 @ 2.19e17`: macro `3.8104`, avg tok/s `405,630`
  - `d768 @ 1.70e18`: macro `3.4339`, avg tok/s `273,532`

## Experiment Log
### 2026-04-23 00:00 - kickoff
- Hypothesis: MuonH can reuse the AdamH heuristic learning rates on the matrix/expert parameters, and the larger-step geometry may show a clearer gain when batch size is doubled.
- Command: pending issue creation and Iris submission.
- Config:
  - Branch: `research/moe-muonh-batch-ablations`
  - Optimizer variant: `GrugMoeMuonHConfig`
  - Newton-Schulz coefficient type: `quintic`
  - Batch variants: baseline heuristic batch and `2x` batch
  - MoE budgets: `d512 / 2.19e17`, `d768 / 1.70e18`
- Result: pending
- Interpretation: pending
- Next action: validate the MuonH launcher locally, create the experiment issue, then submit the four gate-1 runs on Iris.
