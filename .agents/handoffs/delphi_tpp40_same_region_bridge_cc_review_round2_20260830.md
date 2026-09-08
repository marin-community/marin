# Delphi TPP40 same-region bridge review, round 2

Please perform a read-only launch-blocking review of the revised one-pair bridge. The user has authorized `run_order=2` alone to unblock production; East5 capacity must not block the Europe production share. Thresholds are unchanged. Review whether the implementation now closes B1-B4 from your first response and whether the exact East-reference evaluator may be submitted.

## Revised design

- Logical East reference: immutable canonical East5 v5p-8 `run_order=2` trajectory.
- Candidate: Europe v6e-8 `run_order=2` trajectory.
- Both logical sides are evaluated on Europe `v6e-8` in `europe-west4-a` using the same region-local Uncheatable caches and Table-9 request set.
- Only the two East Orbax checkpoints and endpoint HF export were copied, without Storage Transfer Service. The source-frozen mirror manifest records 16 objects and 10,049,866,394 bytes. Every launch and analysis rechecks source and destination tree identities, including object names, sizes, and CRC32C.
- Contract v4 explicitly limits the gate to training-deployment drift. It does not authorize pooling raw East5- and Europe-evaluated numbers. Supplementary cross-region evaluator checks retain the preregistered 0.002-BPB thresholds; failure requires common-region reevaluation before scientific pooling.
- Idempotence now has three unchanged rerun roles: one combined East-reference evaluator parent, Europe training, and Europe Uncheatable. The East mirror trees are included in before/after inventories.

## Frozen artifacts

- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v4.json`
  - SHA-256 `f0441b8927e3e7d32bbdbe781ed3008dbb46a1cd98ff540661423e850ee936df`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/east5_row2_europe_mirror_manifest_v1.json`
  - SHA-256 `08c6160b4bc181a139c432ed642945f8c2fd72b61b280d2590ffd435deb48202`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_paths_v3.json`
  - SHA-256 `aa55556044b37a6990d2032bedff0a5776b51c7db6183f3baca7ffbc0825d1f7`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/east5_same_region_reference_eval_command_v1.txt`
  - SHA-256 `eb587151cfd2d7de7f2469e143e4dbaf88f4f50d1a7bc826008f26d934bb3b8c`

## Code to review

- `experiments/domain_phase_mix/launch_delphi_tpp40_bridge_same_region_east5_eval.py`
- `experiments/domain_phase_mix/launch_delphi_tpp40_bridge_uncheatable_eval.py`
- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_acceptance.py`
- `experiments/domain_phase_mix/collect_delphi_tpp40_bridge_idempotence.py`
- `tests/test_analyze_delphi_tpp40_bridge_acceptance.py`
- `tests/test_collect_delphi_tpp40_bridge_idempotence.py`
- `tests/test_delphi_tpp40_bridge_uncheatable_eval.py`

## Current local evidence

- `uv run python -m py_compile` passes on all four bridge modules.
- Focused bridge suite: 32 passed.
- Exact command has already passed Europe launch-safety validation and dry-run; these will be rerun after this review.

## Questions

1. Are B1-B4 fully closed without weakening the one-pair numerical thresholds?
2. Does the mirror audit fail closed on source mutation, destination mutation, missing objects, and extra destination objects?
3. Do the asymmetric idempotence roles prove every training, mirror, Uncheatable, and Table-9 output is unchanged and skipped?
4. Is there any blocker to submitting the exact East same-region reference command now?

Return `GO` or `NO-GO`, with launch blockers first. Do not edit files.
