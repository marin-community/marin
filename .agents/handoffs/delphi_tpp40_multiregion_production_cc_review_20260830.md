# Delphi TPP40 multiregion production launch review

Review the frozen production assignment and exact East5/Europe commands for the 280-row Delphi TPP40 swarm. This is a read-only, fail-closed launch review. Return an explicit **GO** or **NO-GO**, with blockers first. Do not edit files or submit jobs.

## Scientific and operational objective

Run the remaining immutable 280-policy TPP40 panel on as much available regional compute as possible without duplicating a training coordinate or changing the training/evaluation configuration. The legacy East5 parent is already killed. A separate four-row Europe bridge is running under a disjoint experiment name and is not part of this production namespace. Production launch remains gated on a preregistered cross-accelerator acceptance pair; these commands are being frozen and reviewed ahead of that result, not authorized yet.

## Files to inspect

- `experiments/domain_phase_mix/materialize_delphi_tpp40_multiregion_assignment.py`
- `experiments/domain_phase_mix/launch_delphi_augmented_swarm_tpp40.py`
- `experiments/domain_phase_mix/east5_launch_safety.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/assignment_v1.json`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/east5_production_command_v1.txt`
- `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/europe_production_command_v1.txt`
- `tests/test_delphi_tpp40_multiregion_assignment.py`
- `tests/test_delphi_augmented_swarm_tpp40.py`

## Frozen evidence

- Assignment semantic SHA-256: `873959336b2fb3488317d1824c74954767f264d48c308ecd15dc8b1403837a3f`.
- Assignment file byte SHA-256: `72f88de70e32e2f50d7a26018236c4c4228232bc900f294623b8155bb36a0f58`.
- Both region-local GCS copies have that exact byte hash:
  - `gs://marin-us-east5/experiments/domain_phase_mix/delphi_tpp40_multiregion_assignment_20260830/assignment_v1.json`
  - `gs://marin-eu-west4/experiments/domain_phase_mix/delphi_tpp40_multiregion_assignment_20260830/assignment_v1.json`
- East command byte SHA-256: `e1de0384ed36fe2d65a51c743a6f5eca7cfc38f38c85a608dcf37bcc5529f886`.
- Europe command byte SHA-256: `4a8228c6c0631307aab780f30bbc32351e06ec9865ff3c2c30ff5f07504278ef`.
- Live-state assignment counts: completed 27, East-assigned 126, Europe-assigned 127, East resumable 2.
- East graph dry-run: 153 specs = 27 completed/idempotent replay-for-eval + 126 East assigned; 140 runtime caches and 23 validation caches accepted; Table-9 v6e-8 in `us-east5-b`.
- Europe graph dry-run: 127 fresh specs; 140 runtime caches and 23 validation caches accepted; training/Table-9 v6e-8 in `europe-west4-a`.
- Independent cross-check: union covers all 280 run orders and noncompleted overlap is zero.
- Strict launch-safety checks passed with parent, training-child, Table-9-child, and bucket locality specified separately.
- `tests/test_delphi_tpp40_multiregion_assignment.py` plus `tests/test_delphi_augmented_swarm_tpp40.py`: 27 passed.

## Design details to scrutinize

1. The assignment includes all 27 completed rows only in the East graph so training skips idempotently while missing native Table-9 obligations can still run.
2. Two East rows with a permanent phase-0 checkpoint but no final checkpoint remain assigned to East and should resume on v5p-8 rather than restart in Europe.
3. Fresh remaining rows are deterministically balanced by frozen panel stratum: 126 East and 127 Europe.
4. Both parents use the canonical production experiment name, but `MARIN_PREFIX` makes their output roots region-local and disjoint.
5. East training uses v5p-8 in `us-east5-a`; East Table-9 uses v6e-8 in `us-east5-b`. Europe training and Table-9 use v6e-8 in `europe-west4-a`; its CPU parent is in `europe-west4-b`.
6. Both parents use interactive priority, `--max-concurrent 56`, strict executor mode, and a seven-day parent timeout.
7. The separate Europe bridge uses experiment name `delphi_augmented_swarm_tpp40_europe_v6e_bridge_v2_20260830`; it must not collide with production.

## Questions

1. Is the assignment exhaustive, disjoint, deterministic, and safe against both duplicate training and loss of downstream Table-9 obligations?
2. Is it correct to include completed rows in the East graph and rely on executor idempotence while excluding them from Europe?
3. Can the two East phase-0 rows resume safely under this command, including exact accelerator/config identity?
4. Do the commands enforce complete region locality and avoid cross-region training-data access?
5. Are the concurrency, timeout, parent placement, and accelerator choices operationally sound for maximizing throughput?
6. Does any code path ignore the assignment hash, silently fall back to all rows, or produce the same production output path from both parents?
7. Is there any blocker that should prevent these exact commands from being submitted immediately after the one-pair acceptance/idempotence gate passes?
