# MoE Muon AOL: Research Logbook

## Scope
- Goal: Test whether a Muon-based optimizer recipe using AOL Newton-Schulz coefficients can beat the current AdamH-based Grug MoE recipe at the gate-1 compute-optimal scales.
- Primary metric(s): `eval/paloma/macro_loss`, `throughput/tokens_per_second`, effective speedup versus the compute-optimal AdamH baseline.
- Constraints: Keep the model architecture, data mix, and compute-optimal budgets fixed at `d512 / 2.19e17` and `d768 / 1.70e18`. Only swap the optimizer path.
- Issue: https://github.com/marin-community/marin/issues/5115

## Baseline
- Date: 2026-04-22
- Code refs: `experiments/grug/moe/README.md`, `experiments/grug/moe/agent.md`
- Baseline numbers:
  - `d512 @ 2.19e17`: macro `3.8104`, avg tok/s `405,630`
  - `d768 @ 1.70e18`: macro `3.4339`, avg tok/s `273,532`

## Experiment Log
### 2026-04-22 00:00 - kickoff
- Hypothesis: Replacing AdamH on the matrix/expert weights with a tuned Muon recipe using AOL Newton-Schulz coefficients can recover enough quality and/or throughput to beat the current AdamH baseline at the two small compute-optimal MoE budgets.
- Command: pending issue creation and Iris submission.
- Config:
  - Branch: `research/moe-muon-coefficients`
  - Optimizer variant: `GrugMoeMuonConfig`
  - Newton-Schulz coefficient type: `aol`
  - MoE budgets: `d512 / 2.19e17`, `d768 / 1.70e18`
- Result: pending
- Interpretation: pending
- Next action: validate the new launcher locally, create the experiment issue, then submit gate-1 on Iris.
