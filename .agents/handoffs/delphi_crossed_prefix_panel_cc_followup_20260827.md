# Delphi crossed-prefix panel: required-change follow-up

Your second prelaunch review approved the 9-state by 50-action panel subject to five required changes. Inspect the current files directly and determine whether those requirements are now satisfied. Do not repeat the full design review unless a new blocker appears.

## Changes made

1. `FIT_DATA_SEED` is now 970000, disjoint from every cap-10 and cap-4 prefix data seed and from the two sentinel seeds. The nine tied controls use this same disjoint seed.
2. `HarshCandidatePrefixTrainingConfig` now takes an explicit `experiment_name` and an optional aliases hash. The bridge writes its own experiment identity and JSON `null` for the absent cap-10 aliases artifact. Existing harsh-cap launches still pass their canonical experiment and aliases hash explicitly.
3. Runtime prefix validation now checks `experiment_name` and `candidate_aliases_sha256` in addition to the previously frozen core fields.
4. The manifest labels the bridge as a composite hardware-by-code detector that cannot support a hardware-only correction.
5. `observed_cap10_best` is descriptive-only and excluded from the confirmatory model comparison.
6. The cap-4 seed pair is labeled a single-draw diagnostic, not a null distribution. The primary comparison must be repeated without cap-4 KL=0 because that state is outside the historical total-exposure envelope.
7. The manifest now specifies the sole confirmatory target, action-blocked CV comparison, common-action bootstrap unit, state set, sensitivity analysis, multiplicity policy, and the intentionally broad exposure range.

## Frozen artifacts

Directory: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_crossed_prefix_panel_v3_20260827/`

- manifest: `afd3726c4924871b4081012c1b1c59b4beb7ea2f87afa704fff13259365c91be`
- prefix registry: `81993db01c30afaaaba9dda373df233e07e1933fce2277a4b25a5e055d83cb5e`
- panel rows: `148d66bceca853119c68dbe52be99862fc3f81d1d58bf9cd1eaf680d7ab20eeb`
- panel weights: `daa000aa487437dafd9daab2989efc9e1fa9a656e8b8910bb2d81cbfca4f3c3e`

## Files to inspect

- `experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_phase1_crossed_prefix_panel_20260827.py`
- `experiments/domain_phase_mix/launch_delphi_3e18_phase1_crossed_prefix_panel.py`
- `experiments/domain_phase_mix/launch_delphi_3e18_phase0_harsh_cap_candidates.py`
- `experiments/domain_phase_mix/launch_delphi_3e18_phase1_harsh_cap_branches.py`
- `tests/test_delphi_phase1_crossed_prefix_panel.py`
- the four v3 frozen artifacts above

End with exactly one of: `APPROVE`, `APPROVE WITH REQUIRED CHANGES`, or `DO NOT LAUNCH`. List only actual remaining blockers before that verdict. Do not edit files or submit jobs.
