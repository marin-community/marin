# Review brief: Delphi proportional-prefix branch search

## Decision to review

We want to determine whether phase-1 optimization alone can recover or exceed the best validated Delphi 3e18 Uncheatable frontier when phase 0 is fixed to the proportional mixture.

Three exact proportional phase-boundary checkpoints exist, including optimizer state. The proposed Wave 1 uses prefix seed 0 for 80 fit branches and 8 sealed geometry referees, plus 14 controls/repeats. It includes five observations each for the seed-0 tied mixture, seed-1 tied mixture, and validated-frontier continuation from seed 0. Final candidate certification will cross prefix seeds 0, 1, and 2.

The fit panel is adapted from `design_delphi_phase1_harsh_cap_branches_20260825.py`: runtime-exact 1/2048 mixtures, full-rank square-root-simplex coverage, local and mid-radius tranches, deployment anchors, and a 10-total-materialized-epoch support cap. Wave 2 is intended to use the already frozen two-part policy: model-guided acquisition plus an outcome-blind coverage tranche.

Primary target: scalar Uncheatable BPB. Primary effect: paired gain against the exact tied continuation from the same prefix and data seed. Frontier target: mean BPB below `0.9798883332146539`, pinned to Fieldbook validation `val_01m0xnbbg6jc6cj6awyfgrq2d1` and its frozen candidate contract. The exact frontier continuation and four fresh repeats are in Wave 1. Repeats do not count against the fit budget.

The proportional prefixes were trained on v5p-8 in east5a and the branches run on v6e-8 in east5b. Successful source executor records and their hashes now attest the prefix configuration. The manifest labels this a cross-hardware discovery panel; any canonical frontier claim requires same-hardware confirmation.

## Files

- `.agents/logbooks/delphi-proportional-prefix-branch-search.md`
- `experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_phase1_proportional_prefix_wave1_20260825.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_phase1_harsh_cap_branches_20260825.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_proportional_prefix_wave1_20260825/validated_frontier_contract.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_proportional_prefix_wave2_contract_20260826/contract.json`
- `experiments/domain_phase_mix/launch_delphi_3e18_phase1_proportional_prefix_wave1.py`
- `experiments/domain_phase_mix/launch_delphi_3e18_phase1_harsh_cap_branches.py`

## Questions

1. Does fixing phase 0 to proportional make the causal question interpretable, or does the design still conflate prefix quality and branch recoverability?
2. Is the 80-row Wave-1 geometry sufficient and well allocated for a 38-dimensional tangent response, or should local/global allocations change?
3. Is the 10-total-materialized-epoch cap an appropriate support guard for this prefix, or could it exclude the phase-1 mechanism being tested?
4. Are tied, fresh-data, and cross-prefix-state controls sufficient for a paired gain claim and a frontier claim?
5. Should the first adaptive model remain the scalar low-complexity response model, given joint atomic and low-rank curvature heads did not promote in the cap-4 panel?
6. Identify any launch-safety, idempotence, leakage, or sequential-design blocker that must be fixed before submission.
