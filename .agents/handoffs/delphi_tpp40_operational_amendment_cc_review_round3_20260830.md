# Delphi TPP40 operational amendment, round 3 review

Review the exact current versions of:

- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_operational_amendment.py`
- `experiments/domain_phase_mix/materialize_delphi_tpp40_multiregion_assignment.py`
- `tests/test_analyze_delphi_tpp40_bridge_operational_amendment.py`
- `tests/test_delphi_tpp40_multiregion_assignment.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_operational_amendment_v2.json`

The previous review correctly found that the assignment authenticated only itself and that stale pre-quiescence assignments could pass. Verify that the current revision closes both issues:

1. amendment v2 has explicit `UNFROZEN` file and semantic assignment digests, so authorization is impossible before post-quiescence rematerialization and a final amendment refreeze;
2. the analyzer compares both observed digests against the amendment, validates the exact roots, materializer observation inventory, completed artifacts, sorted disjoint coverage, resumable-East5 placement, exact legacy-parent identity, terminal state, and UTC observation time;
3. the materializer embeds the legacy-parent quiescence snapshot in the semantic digest;
4. the assignment bytes are read once for both parsing and file hashing;
5. historical amendment v1 remains byte-identical at SHA-256 `32b6fd0b2a27dbacde1168c1bca297ff6528c447b045c754f9f4a67e9f609765`.

The macro-only 0.005 BPB screen is a user-directed operational policy, not an equivalence claim; do not propose a per-component gate. Identify only remaining launch-safety or scientific-integrity blockers. Return `GO` or `NO-GO` with exact evidence.

Local verification: 42 focused tests pass and targeted Marin pre-commit passes.
