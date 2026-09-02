# Review request: TPP40 bridge acceptance collector

Review the strict acceptance collector that gates an East5-v5p versus Europe-v6e deployment bridge before a 280-row multiregion TPP40 launch.

## Frozen scientific contract

Read:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v2.json`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_paths_v1.json`

The path manifest was independently materialized twice with stable SHA-256 `6445ffe2d0cf360914822fe92513334775f4c24510b72274a23b130f7226083b`. Its 16 Uncheatable output paths exactly match the earlier frozen sidecar manifests. It also freezes eight native endpoint Table-9 paths.

## Code under review

Read only:

- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_acceptance.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/tests/test_analyze_delphi_tpp40_bridge_acceptance.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/launch_delphi_tpp40_bridge_uncheatable_eval.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/lib/marin/src/marin/evaluation/olmo_base_eval/run.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/lib/marin/src/marin/evaluation/olmo_base_eval/aggregate.py`
- `/Users/calvinxu/Projects/Work/Marin/marin/lib/marin/src/marin/evaluation/olmo_base_eval/components.py`

Current local validation: 13 focused tests pass, targeted pyrefly reports zero errors, targeted pre-commit passes, and an incomplete live run fails closed with zero paired results and production authorization false.

## Questions

1. Can any missing, stale, duplicate, wrong-checkpoint, wrong-model, wrong-request-set, malformed, or failed result silently enter a paired numerical comparison or yield `production_launch_authorized=true`?
2. Does the collector apply every preregistered threshold exactly: four matched run orders, separate phase-boundary and endpoint Uncheatable tests, endpoint Table-9, both mean-absolute and any-row limits?
3. Is cross-region scientific identity checked strongly enough, especially model configuration, parameter count, seeds, checkpoint step/path, Table-9 request inventory/version, and evaluator provenance?
4. Is the idempotence evidence schema sufficiently fail-closed to prove an unchanged rerun skipped all 4 training, 8 Uncheatable, and 4 Table-9 units per side without changing result inventories?
5. Are path and command hashes frozen in a way that prevents analysis drift?
6. Do the tests exercise the load-bearing pass and fail cases? Identify any missing regression test that should block use.

Return findings ordered by severity, distinguish blockers from non-blocking polish, and end with a GO or NO-GO verdict for using this collector as the production launch gate. Do not edit files.
