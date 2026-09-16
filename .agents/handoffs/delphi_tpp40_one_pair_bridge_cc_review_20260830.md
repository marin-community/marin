# Review brief: Delphi TPP40 one-pair multiregion bridge

## Decision already made

Before the Europe bridge produced a phase-boundary or endpoint result, the user changed the production gate from four matched coordinates to the single immutable source-panel coordinate `run_order=2`. Do not revisit that scope decision. Review whether the implementation faithfully enforces it without weakening the previously frozen numerical, provenance, completeness, or idempotence checks.

The canonical completed East5 v5p trajectory is now the reference. The already-running Europe v6e bridge trajectory is the candidate. Europe rows 120, 240, and 260 remain supplementary and must not block this gate.

## Review targets

- `experiments/domain_phase_mix/launch_delphi_tpp40_bridge_uncheatable_eval.py`
- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_acceptance.py`
- `experiments/domain_phase_mix/collect_delphi_tpp40_bridge_idempotence.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v3.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_paths_v2.json`
- the four `*run2*command*.txt` files in that reference-output directory
- `tests/test_analyze_delphi_tpp40_bridge_acceptance.py`
- `tests/test_collect_delphi_tpp40_bridge_idempotence.py`
- `tests/test_delphi_tpp40_bridge_uncheatable_eval.py`

## Facts already verified locally

- Contract SHA-256: `be5398dc3ad2883fb5a558bded483f06daa4cfaae2a72175579c2059c0b77f63`.
- Frozen path-manifest SHA-256: `b51faea20a12176c9c091c07c5a04e04bb5d3a55c6b95ca60a7d57f6a479372a`.
- East reference output resolves to `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815/fit_002_run_00002-29ef42`.
- Its exact Orbax checkpoints at steps 21855 and 27335 exist, are permanent, and have metadata SHA-256 values `e2a3134d3f1de4258b4ec814c5c32b155192b4336be7f27708dafaa77c6777cc` and `2b6c2df1bac9e50112ef8749e7264d033748041e96f7f4b9d70468a31eaae524`.
- All four exact launch commands pass region-local launch-safety validation.
- Both one-row training launchers and both Uncheatable launchers pass dry-run construction.
- Focused suite: 68 tests pass.

## Questions

1. Are there any stale four-row assumptions, count checks, paths, or command hashes that could silently block or incorrectly pass a one-pair gate?
2. Does the East side reconstruct the canonical historical trajectory and its Table-9 output exactly, rather than constructing a scientifically different replacement?
3. With one pair, does the unchanged `mean <= 0.002` and `any <= 0.005` logic correctly make the 0.002 threshold binding at phase 0, endpoint, and Table-9?
4. Does the mechanical idempotence collector prove unchanged reruns emitted zero children and left every frozen output byte inventory unchanged?
5. Do the four frozen commands and path manifest support the intended sequence: evaluate East immediately, evaluate Europe as checkpoints appear, then rerun both training and evaluation launchers after completion to prove idempotence?
6. Identify only production blockers or scientifically material omissions. Distinguish them from polish.

Return a clear `GO` or `NO-GO`, with file and line references for every blocker.
