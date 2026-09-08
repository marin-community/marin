# Delphi TPP40 Europe bridge acceptance-contract review

Return `GO` or `NO-GO` as the first line. This is a read-only preregistration review before any Europe bridge result exists.

The full exact-command review already returned GO, but identified one residual gap: the historical Table-9 single-run standard deviation and its provenance were described rather than numerically frozen. Review whether the new acceptance contract closes that gap without changing the already reviewed launch or weakening its gates.

Read these files:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v2.json`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/europe_bridge_command_v2.txt`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/analyze_prefix_search_evidence_20260819.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/design_prefix_search_20260819.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/runtime_cache_audit.json`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/evaluation_cache_audit.json`
- `/tmp/delphi_tpp40_europe_bridge_v2_dry_run/launch_audit.json`

Check specifically:

1. The frozen `0.0034193565526723828` Table-9 SD is reproducible from the named, hashed inputs using the stated estimator and supports the rounded historical `0.00342` value.
2. The derived Table-9 thresholds are arithmetically correct: `max(0.002, 0.10 * SD) = 0.002` for the four-row mean absolute paired delta, and `max(0.005, 0.20 * SD) = 0.005` for every individual paired row.
3. The contract cannot silently pass missing phase-boundary/end checkpoints, missing Uncheatable or native Table-9 evaluations, failed rows, or a non-idempotent rerun.
4. The four run orders and all source/scientific/runtime/evaluation hashes match the dry-run and audit artifacts. Distinguish the launcher's ordered validation-path hash from the audit's named Europe-path hash.
5. No result-dependent choice remains. State any blocker that must be fixed before submitting the four-row Europe bridge; otherwise say the bridge is GO while the production multiregion launch remains gated on bridge results and the unchanged idempotent rerun.

Do not edit files and do not launch anything.
