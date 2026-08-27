# Delphi crossed-prefix continuation panel: v2 prelaunch review

## Why this is a second review

Your first review returned `do not launch` for the proposed 10-state by 40-action panel. It identified three blockers: hardware and branch-code nesting, one residual degree of freedom per state, and no prefix-seed null-interaction control. It also found an uncentered design criterion, discarded exposure/TV stratification, inadequate low-only sentinels, and an unaudited support contract.

The implementation was rebuilt rather than relabeled. Please verify that the changes below actually resolve the blockers and give a direct `approve`, `approve with required changes`, or `do not launch` verdict.

## Revised design

The fit panel crosses the complete frozen, label-blind 50-action bank with nine exact phase-boundary states:

- four cap-10 v5p seed-0 states: observed incumbent and KL 0.05, 0.2, and 0.5;
- three cap-4 v6e seed-0 states: KL 0, 0.05, and 0.2;
- the cap-4 KL=0 v6e seed-1 checkpoint as a null-interaction state;
- a new v6e replay of the cap-10 KL=0.05 seed-0 prefix as a phase-0 hardware bridge.

All 50 fit actions are rerun under one new branch code commit. No old branch result is reused. This removes branch-code version nesting and restores the frozen exposure/TV stratification. The centered 50-by-38 tangent design has rank 38, condition number 23.27, minimum singular value 0.02818, and 11 residual degrees of freedom per state.

The bridge prefix is an upstream dependency in the same Iris parent. Its continuation rows cannot start until its permanent step-2399 checkpoint and deterministic provenance exist. Existing states can fan out immediately. Prefix and continuation children use v6e-8 in `us-east5-b`; the Iris parent is pinned to `us-east5-a`; all artifacts use `gs://marin-us-east5`.

Each state has three controls outside the 50-row fit budget:

- its tied continuation at the common fit seed;
- a repeat of low-exposure `fit_maximin_00` at data seed 930001;
- a repeat of high-exposure `fit_maximin_26` at data seed 930002.

The fit cells use data seed 930000 and continuation trainer seed 0. The cap-4 seed-1 prefix therefore isolates variation in the incoming trained state while holding continuation settings fixed.

## Support and inference contracts

The design now audits every prefix-action pair. All 50 actions remain inside the frozen phase-1 support envelope. Cap-4 KL=0 itself exceeds the old cap-10 total-exposure envelope on one bucket, so all 50 cells from its two prefix seeds are explicitly marked controlled extrapolation; state-specific action filtering was rejected because it would destroy the common-action estimand.

The manifest pre-registers:

- the primary comparison as prefix-shared versus prefix-varying parameters in a bounded-shape DSP branch-response model;
- the unrestricted 38-dimensional tangent fit as descriptive and a lack-of-fit diagnostic, not the primary mechanism claim;
- exact idempotent rerun of any missing row, with no action replacement; an irrecoverably missing fit row blocks the confirmatory interaction comparison;
- one exact branch commit for every new cell, with zero reuse.

## Files and verification

- `experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_phase1_crossed_prefix_panel_20260827.py`
- `experiments/domain_phase_mix/launch_delphi_3e18_phase1_crossed_prefix_panel.py`
- `tests/test_delphi_phase1_crossed_prefix_panel.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_crossed_prefix_panel_v2_20260827/`

Current focused result: five tests pass; pyrefly reports zero errors; targeted pre-commit passes; launcher dry-run validates 477 rows.

## Questions

1. Does the nine-state by 50-action design now identify the intended state-conditioned response well enough for launch?
2. Does the v6e cap-10 bridge break the phase-0 hardware alias in the intended way?
3. Is using the cap-4 seed-1 checkpoint with common continuation seed 0 a valid null-interaction reference?
4. Is the controlled cap-4 KL=0 support extrapolation acceptable when explicitly modeled, or should it block those states?
5. Does the upstream bridge dependency and runtime provenance validation safely guarantee exact checkpoint restoration?
6. List any remaining launch blocker. Do not edit files or submit jobs.
