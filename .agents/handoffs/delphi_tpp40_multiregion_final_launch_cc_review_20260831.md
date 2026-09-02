# Delphi TPP40 multiregion final launch review

Give a concise `GO` or `NO-GO` verdict on submitting the two exact production commands below. This is a read-only launch-safety review. Identify only concrete blockers or material caveats; do not edit files.

## Decision context

The user explicitly directed that one completed East5/Europe matched pair is sufficient for an operational production screen and that we should not wait for more East5 bridge capacity. The original strict v4 equivalence verdict must remain failed; the dated v2 operational amendment separately permits launch only when all three absolute paired macro deltas are at most 0.005 BPB, idempotence passes, and the post-quiescence assignment is digest-bound.

Observed Europe-minus-East5 deltas for run order 2 are:

- phase-0 Uncheatable: `+0.0036705732345581055`
- endpoint Uncheatable: `+0.002214491367340088`
- endpoint native Table-9 macro: `+0.001886141969370092`

All are within the operational 0.005 BPB screen. The strict v4 report still records `numerical_acceptance_passed=false` and `production_launch_authorized=false` because the phase-0 and endpoint deltas exceed its 0.002 mean-absolute threshold.

The legacy East5 parent `/calvinxu/dm-delphi-augmented-swarm-tpp40-phase0ckpt-interactive-retry8-20260825` was explicitly cancelled and observed `State: killed` at `2026-08-31T05:48:57Z`. Its full Iris root tree then had 20 succeeded jobs, 57 killed jobs, and zero pending/building/running jobs.

## Exact frozen artifacts

- Operational amendment: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_operational_amendment_v2.json`
  - SHA-256: `178ca635b116ee1b6bf1c09a80e1de43d639a99594300f57402ad55fdb9dcd5a`
- Operational authorization: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_operational_authorization_v2.json`
  - SHA-256: `a938a89287267b51a59bc7edeb894eed9176f28cf0f3df1c4ec7002a6662e6a4`
  - Expected: `operational_bridge_screen_passed=true`, `operational_production_launch_authorized=true`, no blockers.
- Strict v4 report: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_report_v3.json`
  - SHA-256: `1a2d9e9bf11f461f92827826ca7302e2031d5407279b20930925e487d494f393`
- Idempotence evidence: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_idempotence_evidence_v3.json`
  - SHA-256: `74d273cdd4fa38796692c171e5356141a1e9ae93c186f099db62610e943aa510`
  - Three unchanged rerun parents each had one successful zero-exit parent task, emitted zero children, and preserved both regional output inventory hashes.
- Assignment: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/assignment_v2.json`
  - File SHA-256: `eeb657d51d9891c25e66f9c559c93337aa1a8b154c6122539ca0e652d70c0473`
  - Semantic SHA-256: `8074b0d3a92e5e002336389849f33bbd630d9be2ea1580ccf436dfb2b40ea836`
  - Partitions: 29 completed, 125 East5, 126 Europe; all 280 rows covered exactly once; row 30 is the sole resumable East5 row.
  - Byte-identical 6979-byte copies with the file SHA above are present in both regional buckets at `gs://marin-{us-east5,eu-west4}/experiments/domain_phase_mix/delphi_tpp40_multiregion_assignment_20260830/assignment_v2.json`.
- Resolved-path audit: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/east5_resolved_path_audit_v3.json`
  - SHA-256: `89b38aeb2fc399c87c5c5eafb9ab79260243238e88c85b11dd7fc41c0f5578e0`
  - `passed=true`; 29 completed, 1 resumable, 124 fresh, 154 total East5 coordinates resolved to the frozen East5 root.

## Exact commands

- East5: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/east5_production_command_v2.txt`
- Europe: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/europe_production_command_v2.txt`

Both exact commands passed `east5_launch_safety`. East5 uses a parent and v5p-8 training in `us-east5-a`, native Table-9 on v6e-8 in `us-east5-b`, and only `gs://marin-us-east5`. Europe uses a parent in `europe-west4-b`, v6e-8 training/Table-9 in `europe-west4-a`, and only `gs://marin-eu-west4`. Both are interactive, non-preemptible, `max-concurrent=56`, and require semantic assignment SHA `8074b0d3...`.

## Relevant implementation

- `experiments/domain_phase_mix/launch_delphi_augmented_swarm_tpp40.py`
- `experiments/domain_phase_mix/materialize_delphi_tpp40_multiregion_assignment.py`
- `experiments/domain_phase_mix/audit_delphi_tpp40_multiregion_resolved_paths.py`
- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_acceptance.py`
- `experiments/domain_phase_mix/analyze_delphi_tpp40_bridge_operational_amendment.py`
- `experiments/domain_phase_mix/collect_delphi_tpp40_bridge_idempotence.py`

Local verification immediately before review: 88 focused tests passed; targeted pre-commit passed; both exact launch-safety checks returned `ok=true` with no warnings.

Please verify:

1. The two commands consume disjoint assignment partitions and preserve existing East5 completed/resumable work without routing any Europe row into East5 state.
2. The assignment freeze, both digests, and authorization form a fail-closed chain with no stale-v1 escape hatch.
3. The command region, zone, bucket, accelerator, and native Table-9 placement are internally consistent.
4. The one-pair operational decision is represented honestly without erasing the strict v4 failure.
5. No remaining issue should block submitting both parents now. State any post-launch scientific obligations separately from launch blockers.
