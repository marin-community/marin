# Delphi TPP40 operational amendment, terminal-state follow-up

Review the sole post-GO change in:

- `experiments/domain_phase_mix/materialize_delphi_tpp40_multiregion_assignment.py`
- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_operational_amendment.py`
- their two focused test files.

Iris empirically reports a user-cancelled parent as `State: killed`, so the accepted terminal-state set changed from `{cancelled, failed, succeeded}` to `{failed, killed, succeeded}`. Verify that this matches Iris semantics, remains fail-closed against non-terminal states, and does not weaken the assignment freshness gate. Local targeted pre-commit and all 42 focused tests pass. Return `GO` or `NO-GO`, listing only launch-safety blockers.
