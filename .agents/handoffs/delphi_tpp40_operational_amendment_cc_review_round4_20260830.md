# Delphi TPP40 operational amendment, final mechanical review

Review only the two remaining provenance fixes from round 3 in the exact current tree:

- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_operational_amendment.py`
- `tests/test_analyze_delphi_tpp40_bridge_operational_amendment.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_operational_amendment_v1.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_operational_amendment_v2.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_operational_authorization_v2.json`

Verify:

1. historical amendment v1 is pinned in code and tested at SHA-256 `32b6fd0b2a27dbacde1168c1bca297ff6528c447b045c754f9f4a67e9f609765`;
2. the irreproducible intermediate authorization v1 was removed and authorization v2 is regenerated from amendment v2, with the current incomplete/UNFROZEN state failing closed;
3. no previously closed assignment, one-pair macro-screen, or provenance issue regressed.

Local verification: targeted Marin pre-commit passes; 42 focused tests pass; local `shasum` confirms v1 `32b6fd0...` and v2 `937dd76a...`. Return `GO` or `NO-GO`; report only launch-safety or scientific-integrity blockers.
