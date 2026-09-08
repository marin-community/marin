# Delphi TPP40 bridge acceptance collector review, round 2

Review the exact implementation below as a fail-closed scientific launch gate. Do not edit files.

The purpose is to decide whether four matched East5-v5p and Europe-v6e bridge trajectories are sufficiently equivalent to permit a later 280-row multiregion TPP40 launch. The production launch must remain blocked unless all numerical, identity, completion, and unchanged-rerun checks pass.

## Files

- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_acceptance.py`
- `experiments/domain_phase_mix/collect_delphi_tpp40_bridge_idempotence.py`
- `experiments/domain_phase_mix/launch_delphi_tpp40_bridge_uncheatable_eval.py`
- `experiments/domain_phase_mix/audit_delphi_tpp40_evaluation_caches.py`
- `experiments/domain_phase_mix/delphi_tpp40_evaluation_identity.py`
- `tests/test_analyze_delphi_tpp40_bridge_acceptance.py`
- `tests/test_collect_delphi_tpp40_bridge_idempotence.py`
- `tests/test_delphi_tpp40_bridge_uncheatable_eval.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v2.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_paths_v1.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/evaluation_cache_audit.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/runtime_cache_audit.json`

## Changes since the prior NO-GO review

1. Evaluation inputs are now bound by live object size/CRC identities. The seven Uncheatable validation caches and Table-9 request set have equal cross-region payload digests, and the v2 evaluation audit is source-frozen.
2. The source panel, coordinate, fixed/scientific identity, runtime-cache audit, launcher cache-path digest, checkpoint metadata, and result manifests are checked rather than merely reported.
3. Numerical payloads reject nonpositive or component-degenerate results.
4. The unchanged-rerun collector queries Iris `jobs` plus `job_config`, compares stored `entrypoint_json` and `submit_argv_json` against the frozen command with only the parent name changed, requires a succeeded parent with zero descendants submitted after the before snapshot, and requires byte-identical complete output inventories before/after.
5. Focused tests now cover mean-only threshold failure, exact/above per-row threshold boundaries, phase-0 versus endpoint independence, missing pairs, cross-side model/data/seed mismatches, Table-9 request identity, missing results, one-ULP macro tampering, corrupted rerun counts, frozen-manifest mutation, changed rerun arguments, stale reruns, and any submitted child.

Local validation before review: 68 focused tests pass, Pyrefly reports zero errors, and targeted pre-commit checks pass.

## Questions

1. Can any missing, malformed, mismatched, stale, or partially rerun state still produce `production_launch_authorized=true`?
2. Is the mechanical zero-child rerun plus unchanged complete result inventory sufficient evidence that all 4 training, 8 Uncheatable, and 4 Table-9 units per side were skipped?
3. Are model, data, optimizer/checkpoint, evaluator, and result identities bound strongly enough for the numerical cross-accelerator comparison?
4. Are the preregistered mean and any-row tolerances implemented exactly and independently at phase 0, endpoint, and Table-9?
5. Identify concrete blocker/high/medium findings with file and line references. End with `GO` only if it is safe to proceed to the bridge evaluations and eventual idempotence reruns; otherwise end with `NO-GO`.
