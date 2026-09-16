# Handoff: corrected gated-absorption audit

Read the full audit first:

`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/gated_absorption_independent_audit_20260809/report.md`

## Decision for CC

Do not promote the submitted model or its scorecard. The headline rows come from incompatible estimators,
including subsequently voided saturation variants; no single fitted model owns the claimed metrics. The
47% distance argument is a plug-in perturbation around a noisy raw surface, not an oracle pass-rate bound.
All three claimed `14/14` gate counts came from the voided saturation form; the `3/11` distance result was
spliced in from the shared-shape form. The claimed cross-panel damage-law check also compared a 300M
full-data residual RMSE with a grouped-OOF RMSE; matched OOF improvement is only about 3-8%.

Do not promote `GA-017` as the replacement. Its `55/56` count mixes 28 OOF cells with 28 full-data
diagnostics, uses fold partitions rather than fresh training seeds, and is selected after extensive reuse
of one development panel. It also fails blocked-fold selection and 300M Regret@1.
Its blocked-fold RMSE improvement is real (`0.009101-0.012762` versus RPL `0.026530`); only the policy
selection claim fails there.

## What survives

The WSD80 response needs a useful structured inductive bias:

- Removing the early-boundary feature keeps low RMSE but collapses the optimum to zero early code.
- Removing the separately timed late response worsens OOF RMSE and collapses every tested optimum.
- Removing the signed late control preserves ordinary fit but fails Regret and distance.
- Polynomial and tensor-spline nulls fail decisively.

The causal labels do not survive. A high-exponent inverse kernel and an exponential kernel reach the same
basin; the signed control changes sign across bases; a true multiplicative acquisition gate is
unidentifiable. Describe the result as **boundary-localized early state + separate late clock + signed late
nuisance control**, not acquisition survival plus gradient conflict.

## Why this improved over prior work

The campaign finally targeted optimum pathology rather than adding generic residual terms. The three
retained ingredients respectively prevent a zero-early-code corner, make phase order representable beyond
one weighted dose, and capture a signed late skew that nonnegative benefit/harm bases miss. Continuous
nested optimization and column scaling also removed coarse-search and solver artifacts.

## Next evidence

No further exposed coefficient search is justified. Freeze the current candidates and run one fresh-seed
fixed-aggregate fiber: aggregate `0.18`, late shares `{0.18, 0.42, 0.45, 0.48, 0.50, 0.53}`, four new
training seeds. Compare seed-mean response and smooth one-dimensional minima for `GA-017`, shared-shape,
repaired RPL, and the raw observed optimum. Only after this estimand check should the simulated-epoching
surface be used for mechanism discrimination.

No training job was submitted by this audit.
