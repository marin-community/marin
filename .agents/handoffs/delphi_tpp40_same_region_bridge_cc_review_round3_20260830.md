# Delphi TPP40 same-region bridge review, round 3

Please review only the two launch blockers from your round-2 response and return `GO` or `NO-GO` for the eventual acceptance gate. The exact East-reference evaluator has already been submitted under your explicit Q4 GO; this review concerns the later unchanged-rerun proof and production authorization.

## Blocker 1 fix: mutable executor bookkeeping

- `tree_payload_identity` now accepts an optional exact `excluded_relative_paths` tuple. Default behavior is byte-for-byte unchanged, so all source-frozen mirror digests retain their original meaning.
- `result_inventory` alone passes `(".executor_info",)` for training, Uncheatable, and Table-9 result roots.
- The filtered digest includes the exclusion policy in its canonical payload, so it cannot be confused with an unfiltered tree that happens not to contain `.executor_info`.
- Mirror trees remain fully unfiltered.
- Regression test mutates `.executor_info` and proves the result inventory identity is stable, then mutates `results.json` and proves the identity changes.

Files:
- `experiments/domain_phase_mix/delphi_tpp40_evaluation_identity.py`
- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_acceptance.py`
- `tests/test_delphi_tpp40_evaluation_identity.py`

## Blocker 2 fix: after-snapshot race

- `after_evidence` now validates all three rerun parents as succeeded before starting the after-inventory audit.
- It records `after_inventory_started_at_ms` immediately after rerun validation and before reading any result tree, then records `after_inventory_captured_at_ms` after the audit.
- It rejects any rerun whose `finished_at_ms` is later than the audit start.
- The analyzer independently validates the interval and each rerun completion timestamp before authorizing production.
- Regression test places a rerun completion one millisecond after the audit start and proves authorization fails.

Files:
- `experiments/domain_phase_mix/collect_delphi_tpp40_bridge_idempotence.py`
- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_acceptance.py`
- `tests/test_analyze_delphi_tpp40_bridge_acceptance.py`

## Additional mirror closure

- The audit now hashes the whole `MIRROR_ROOT` and requires its total object and byte counts to equal the sum of the three frozen trees. This rejects extra objects elsewhere under the mirror root, closing your wording caveat.
- New tests cover destination mutation and an extra object outside the three frozen trees.

Files:
- `experiments/domain_phase_mix/launch_delphi_tpp40_bridge_same_region_east5_eval.py`
- `tests/test_delphi_tpp40_bridge_same_region_east5_eval.py`

## Local evidence

- 112 focused TPP40 tests pass.
- Targeted pre-commit passes.
- Pyrefly reports 0 errors.
- Contract, mirror manifest, path manifest, and exact command SHA-256 values still match their source-frozen constants.

Questions:
1. Are both blockers closed mechanically?
2. Does the result-inventory exclusion remain appropriately narrow without weakening scientific-output or mirror mutation detection?
3. Is the acceptance gate now GO pending results and the post-results idempotence evidence freeze?

Do not edit files.
