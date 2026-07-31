# Task and Evidence Policy

## Scientific Boundary

The primary task is the 39-bucket Delphi 3e18 setting. Inputs are per-phase mixture weights, phase fractions, bucket sizes or exposure multipliers, and optionally predeclared bucket families. Targets are Uncheatable BPB and OLMoBaseEval Table-9 macro BPB. Lower is better.

Every candidate must provide a concise equation; a latent state and transition law; a response link; units or dimensionless variables; parameter symmetries; limiting cases; an algebraically tied restriction; and an independently refitted one-phase restriction.

## Data Use

- The 280 one-phase and 280 two-phase fit tables are exact aggregate-matched pairs and are the primary fitting evidence.
- The heldout table is append-only exposed development evidence, not IID and not confirmatory.
- `training_series`, `proposal_target`, `candidate_kind`, `policy_class`, and `group_id` must be retained in evaluation.
- Target-matched proposal evaluation is primary; cross-target evaluation is secondary transfer evidence.
- Exact aliases and fit overlaps must not be counted as coordinate-disjoint heldouts.
- Do not tune equations, output calibration, hyperparameters, or feature definitions directly against heldout outcomes.
- Work in frozen batches: derive candidates from fit, algebraic, and mechanistic evidence; freeze them; then evaluate the batch once on heldouts and log the exposure.

## Prohibited Shortcuts

Do not use unconstrained output calibrators, proposal-series indicators, lookup tables, nearest-neighbor residuals, ensembles, model disagreement as the prediction, or stronger deployment regularization as evidence that a raw model is correct. Do not reopen a rejected route without a materially new state variable, invariant, transition law, response mechanism, or identification strategy.

## Required Diagnostics

Report grouped OOF RMSE and rank metrics, exact-fiber phase-delta prediction, heldout bias and calibration, target-matched Regret@1/@3/@5, lower-tail optimism, errors above 0.05 BPB, proposal-stratified behavior, parameter stability, and raw-optimum geometry before any deployment regularization.
