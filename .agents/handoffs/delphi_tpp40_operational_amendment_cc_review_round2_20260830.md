# Delphi TPP40 operational amendment, round 2 review

Review the exact current versions of:

- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_operational_amendment.py`
- `tests/test_analyze_delphi_tpp40_bridge_operational_amendment.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_operational_amendment_v1.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_report_v3.json`

The user explicitly chose one matched pair and a macro-only 0.005 BPB large-effect screen to unblock a region-balanced production run. This is not an equivalence claim. Do not propose changing that scientific decision into a per-component gate. Large component deltas must instead remain conspicuous diagnostics and trigger common-region re-evaluation before any frontier claim.

The first review found four implementation gaps. Verify that the revision now:

1. has no `--allow-incomplete` success path;
2. binds the frozen noise-audit digest and records the consumed v4 report and contract digests;
3. requires and validates a concrete 280-row disjoint production assignment before production authorization can become true;
4. propagates the frozen v4 failure, thresholds, component warnings, and production follow-up into the standalone authorization artifact.

Also check whether the assignment validation is truly fail-closed and whether a stale pre-quiescence assignment can be mistaken for a fresh assignment. Identify only launch-safety or scientific-integrity blockers, not stylistic polish. Return `GO` or `NO-GO`, with exact file/line evidence.

Local verification already run: 33 focused tests pass and targeted Marin pre-commit passes.
