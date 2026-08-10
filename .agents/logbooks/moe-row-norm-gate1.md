---
topic: moe-row-norm-gate1
issue: https://github.com/marin-community/marin/issues/8131
description: Evaluate factorized output scales with row-wise MuonH on the Grug MoE baseline.
author: kaiyuew
---

# MoE Row Norm Gate 1: Task Logbook

## Current TL;DR

The d512 row-norm cell finished at Paloma macro loss 3.5934 and 344,365
tokens/s, for 0.846 effective speedup against the July control; it fails Gate
1. The d768 cell is still running. A matched rerun of the current canonical
d512 baseline is ready at commit `768bdfaa3985d5cf6821572fb287c19f79da45ea`
to distinguish a real regression from baseline drift.

## Scope

- Goal: determine whether writing each linear as `v * (xW)`, preserving
  mathematical output-row norms of `W`, and preserving the L2 norm of `v`
  improves effective training speed.
- Primary metrics: final `eval/paloma/macro_loss`, mean
  `throughput/tokens_per_second` over the final 100 steps, and effective
  speedup from `experiments/grug/moe/agent.md`.
- Constraints: `W` has the baseline initialization, every `v` starts at
  one, the LM-head matrix remains AdamH, and router/attention-gate matrices
  retain baseline Adam.
- Coordinating issue: https://github.com/marin-community/marin/issues/8131

## Current Baseline

The July baseline uses 8192-token sequences on v5p-8. Its batch sizes are half
the older 4096-token May table, preserving the Gate-1 token/FLOP cells.

| ID | Width | Batch | Steps | Budget | Paloma macro loss | Tokens/s | W&B |
|---|---:|---:|---:|---:|---:|---:|---|
| Control 1 | 512 | 16 | 10,980 | 3.82e17 | 3.5667 | 352,609 | [run](https://wandb.ai/marin-community/dial_moe/runs/moe_may_july_baseline_d512) |
| Control 2 | 768 | 32 | 16,875 | 2.81e18 | 3.2272 | 249,954 | [run](https://wandb.ai/marin-community/dial_moe/runs/moe_may_july_baseline_d768) |

## Hypothesis Queue

### Active

- `MOE-ROW-NORM-CTRL-H1`: the current canonical d512 baseline materially
  differs from the historical July control under the current stack. Evidence:
  the row-norm d512 result is 0.75% worse in loss and 2.34% slower than the
  historical control. Next test: `MOE-ROW-NORM-CTRL-001-d512`.

### Blocked

- None.

### Falsified / Dead End

- `MOE-ROW-NORM-H1` at d512: the row-norm variant reached Paloma macro loss
  3.5934 at 344,365 tokens/s versus historical-control loss 3.5667 at 352,609
  tokens/s, yielding 0.846 effective speedup. The d768 replication remains in
  flight, but the proposed two-width Gate-1 promotion condition is already
  unmet.

### Promoted

- None.

## Background Research Brief

- Effort: low.
- Stop rule: stopped after local architecture/optimizer/history searches and
  primary reparameterization references stopped changing the single requested
  Gate-1 hypothesis.
- Date: 2026-08-10.

### Question

Does explicit per-output scale/direction factorization improve this Grug MoE
baseline when the two factors are constrained to their initial norms?

### Current Marin Context

- Grug stores dense linear matrices as `(input, output)` and expert matrices
  as `(expert, input, output)`.
- Therefore a mathematical row of conventional `W @ x` is a stored column,
  and its norm reduces over stored axis `-2`.
- The baseline optimizer sends most matrices through MuonH, the LM head through
  AdamH, and router/attention-gate matrices through Adam.

### Internal Prior Work

- `experiments/grug/moe/model.py`: canonical model initialization, forward
  layout, fused expert boundary, and inference export.
- `experiments/grug/moe/optimizer.py`: canonical MuonH/AdamH/Adam grouping and
  hyperball implementation.
- `experiments/grug/moe/agent.md`: Gate-1 decision rule and effective-speedup
  calculation.
- No reusable factorized-linear or per-output-row hyperball implementation was
  found in the repository.

### External Prior Art

- [Weight Normalization](https://arxiv.org/abs/1602.07868) directly motivates
  separating a weight vector's magnitude from its direction. This experiment
  differs by keeping both factor norms fixed and allowing relative coordinates
  within the output-scale vector to move.
- [The Newton-Muon Optimizer](https://arxiv.org/abs/2604.01472) analyzes Muon's
  matrix-gradient orthogonalization, but does not establish that the proposed
  per-output-row projection improves language-model training.

### Negative / Failed Leads

- Repository search found no existing per-row MuonH projection or complete
  Grug factorized-linear variant to reuse.
- The May README throughput controls are not hardware/sequence matched to this
  trial; the July 8k v5p-8 runs above are the relevant controls.

### Evidence Map

#### Claim: axis `-2` is the correct row-norm axis

- Support:
  - Marin model einsums consume `(..., input, output)` weights.
  - Initialization-equivalence tests fuse `W * v` on stored output columns.
- Contradictions:
  - Conventional mathematical notation commonly stores `(output, input)`,
    which would make the row axis appear to be `-1`.
- Directness to Marin: exact implementation layout.
- Confidence: high.
- Action: test norm preservation for both 2D dense and 3D expert weights.

#### Claim: factorization may improve optimization

- Support:
  - Weight Normalization reports benefits from decoupling vector magnitude and
    direction.
- Contradictions:
  - The external result uses a different parameterization, tasks, and
    optimizers; it does not predict the sign in this MoE regime.
- Directness to Marin: low.
- Confidence: exploratory.
- Action: run the two smallest decision-changing baseline cells.

### Recommended Next Experiments

#### 1. Gate-1 two-width comparison

- Minimum experiment: d512 and d768 at the exact July baseline token budgets.
- Baseline/control: the two W&B runs in Current Baseline.
- Expected signal: effective speedup greater than 1 at both widths.
- Falsifier: effective speedup less than or equal to 1 at either width.
- Cost/risk: two v5p-8 runs; the factorization may add optimizer state and
  elementwise overhead.
- Sources: Marin Gate-1 guide, July baseline W&B runs, Weight Normalization.

### Hypothesis Queue Update

- Add: `MOE-ROW-NORM-H1`.
- Revise: none.
- Falsify / stop: none.
- Promote: none.

### Source Ledger

| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|
| Grug MoE model | Marin code | `experiments/grug/moe/model.py` | Stored matrix convention | High | Exact baseline |
| Grug MoE optimizer | Marin code | `experiments/grug/moe/optimizer.py` | Parameter groups and hyperball update | High | Exact baseline |
| MoE agent guide | Marin code | `experiments/grug/moe/agent.md` | Gate and speedup rule | High | Required workflow |
| July d512/d768 runs | W&B | Links above | Baseline quality and throughput | High | Hardware/sequence matched |
| Weight Normalization | Paper | https://arxiv.org/abs/1602.07868 | Scale/direction precedent | Medium | Different constraint and regime |
| Newton-Muon | Paper | https://arxiv.org/abs/2604.01472 | Muon matrix-update context | Low | No direct row-wise result |

### Handoff

- Suggested issue Prior work block: scale/direction decoupling has precedent,
  but no direct evidence covers fixed-norm `v` plus per-row MuonH in this MoE.
- Suggested logbook entry: launch both Gate-1 cells from the pinned snapshot.
- Open questions: throughput cost and whether any quality delta is consistent
  across widths.
- Stop reason: the next uncertainty is empirical and the minimum discriminating
  run is fully specified.

## Entry Log

### 2026-08-10 14:03 PDT - MOE-ROW-NORM-000 implementation snapshot

- Hypothesis: a unit-initialized output-scale factorization can preserve the
  exact baseline function at step zero while exposing independent constrained
  scale and direction updates.
- Commit Hash: `87426191277502d955b50806873435b43bcc8206`.
- Command:
  - `./infra/pre-commit.py --changed-files --fix`
  - `uv run --package marin-core --group test pytest -q experiments/grug/moe_row_norm/test_optimizer.py tests/test_grug_variant_contracts.py`
  - `uv run python -m experiments.grug.moe_row_norm.launch --version 2026.08.10 --max-concurrent 2`
- Config: all linear output scales initialized to one; baseline MuonH matrices
  use axis-`-2` row-wise MuonH; all `v` use vector AdamH; LM-head `W`
  remains AdamH; router and attention-gate `W` remain Adam.
- Result: required lint/type checks passed; focused suite passed 24 tests with
  1 skipped; dry run resolved only the two intended checkpoint artifacts and
  pinned dataset dependencies.
- Interpretation: initialization, optimizer routing, and mathematical/stored
  axis conventions are locally validated; TPU behavior remains untested.
- Next action: push the snapshot and launch both Gate-1 artifacts on v5p-8.

### 2026-08-10 14:18 PDT - Gate-1 launcher placement recovery

- Hypothesis: constraining the CPU StepRunner parent to a region containing
  v5p-8 groups allows its two Fray children to inherit a schedulable region
  without attaching an accelerator to the parent.
- Commit Hash: `cdf354d93ff82efc8aeff94f9ba19cf0b26b1e3c`.
- Command:
  - Initial: CPU parent without a region or availability constraint.
  - Rejected adjustment: CPU parent with `--reserve v5p-8 --preemptible`.
  - Active: CPU parent with `--region us-central1`.
- Config: parent uses 1 CPU, 2 GiB RAM, and the `cpu` extra; child
  `ResourceConfig` remains v5p-8. No training configuration changed.
- Result: the unconstrained parent landed in `us-central2`, causing both
  child submissions to fail because that region has no v5p groups. The
  availability-constrained CPU parent had no matching CPU scaling group and was
  stopped while unallocated. The region-pinned parent is accepted and waiting
  for an ordinary CPU worker in `us-central1`. No W&B run or TPU allocation
  occurred in either failed attempt.
- Interpretation: parent location is inherited by Fray children, so the
  launcher needs an explicit TPU-capable region; `--reserve` is not a valid
  CPU-parent placement mechanism on the current scaling-group catalog.
- Next action: wait for the CPU parent, verify exactly two v5p-8 children, then
  validate W&B startup and configuration.

### 2026-08-10 14:42 PDT - Both Gate-1 cells healthy

- Hypothesis: both exact cells start from scratch with the intended
  factorized-row-norm configuration and produce finite, advancing training
  metrics.
- Commit Hash: `cde4e2704131ba2bc929cd2af1b35b9271bb7cfa`.
- Command: `uv run python scratch/monitor_moe_row_norm.py`, plus targeted
  Iris log and W&B config queries.
- Config:
  - d512: batch 16, 10,980 steps, LR 0.0098041002, Adam LR 0.0022624847,
    checkpoint root
    `gs://marin-us-central1/grug/moe_row_norm_gate1_d512/2026.08.10/checkpoints`.
  - d768: batch 32, 16,875 steps, LR 0.0083729565, Adam LR 0.0019322207,
    checkpoint root
    `gs://marin-us-central1/grug/moe_row_norm_gate1_d768/2026.08.10/checkpoints`.
  - Both: 8192-token context, scratch initialization, z-loss 0, v5p-8.
- Result: Iris shows exactly the parent plus
  `grug-train-MOE-ROW-NORM-001-d512` and
  `grug-train-MOE-ROW-NORM-002-d768`, all running. At the config check, d512
  was at step 1,941 with finite loss 4.0748 and about 346,885 tokens/s; d768 was
  at step 439 with finite loss 4.2815 and about 254,738 tokens/s. Representative
  parameter norms remained constant from step 0 to 830 within floating-point
  error: `v_q` 22.6274185 to 22.6274338, `w_q` 11.1722631 to 11.1722679,
  and the 3D expert `v_gate` global norm 256.0 to 256.0002.
- Interpretation: the launch, runtime routing, and norm constraints are healthy.
  Early throughput is close to the July controls, but quality and effective
  speedup require terminal metrics.
- Next action: monitor both cells to successful terminal state, verify final
  checkpoint metadata, and compute effective speedup at each width.

### 2026-08-10 16:46 PDT - Matched d512 baseline-control snapshot

- Hypothesis: rerunning the current canonical MoE stack at the exact d512
  Gate-1 cell will determine whether the comparison to the July control is
  confounded by implementation or infrastructure drift.
- Commit Hash: `768bdfaa3985d5cf6821572fb287c19f79da45ea`.
- Command:
  - `uv run --package marin-core --group test pytest -q experiments/grug/moe_row_norm/test_baseline_control.py`
  - `./infra/pre-commit.py --changed-files --fix`
  - `uv run python -m experiments.grug.moe_row_norm.baseline_control --version 2026.08.10`
- Config: canonical `GrugModelConfig` and `GrugMoeMuonHConfig`, hidden width
  512, batch 16, 10,980 steps, 8192-token context, scratch initialization,
  z-loss 0, and v5p-8. The control keeps canonical AdamH/MuonH routing and has
  no scale-vector factorization.
- Result: the focused recipe-equivalence test passed, required lint and type
  checks passed, and the dry run resolved exactly one intended checkpoint
  artifact plus pinned dataset dependencies.
- Interpretation: model architecture, data, batch, schedule, evaluation, and
  scalar optimizer settings match the completed row-norm d512 cell; only the
  parameterization and optimizer implementation differ.
- Next action: launch `MOE-ROW-NORM-CTRL-001-d512`, verify its Iris child and
  W&B identity, and compare its terminal quality and throughput to both runs.
