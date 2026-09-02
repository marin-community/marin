---
topic: delphi-tpp40-multiregion
issue: https://github.com/marin-community/marin/issues/8797
description: Complete the 280-row Delphi TPP40 swarm across disjoint East5 and Europe assignments.
author: Calvin Xu
---

# Delphi TPP40 Multiregion: Task Logbook

## Current TL;DR

- The strict one-pair v4 equivalence gate remains failed. A separate user-authorized operational screen passed at an absolute `0.005` BPB bound for phase-boundary Uncheatable, endpoint Uncheatable, and endpoint Table-9 macro; this permits launch but does not establish cross-accelerator equivalence.
- Frozen assignment semantic SHA-256: `8074b0d3a92e5e002336389849f33bbd630d9be2ea1580ccf436dfb2b40ea836`; byte SHA-256: `eeb657d51d9891c25e66f9c559c93337aa1a8b154c6122539ca0e652d70c0473`.
- The assignment covers 29 completed East5 rows, 125 East5-assigned rows including resumable row 30, and 126 fresh Europe rows. All 280 rows are covered exactly once.
- The exact East lineage audit passed for all 154 East graph rows: 29 completed, 1 resumable, and 124 fresh. Both v2 production commands passed locality checks and received a final subscription-safe CC `GO`; their launch bundle identities are recorded after submission.

## Scope

- Goal: complete all 280 preregistered Delphi two-phase policies at total tokens per parameter 40.
- Primary metrics: endpoint Uncheatable BPB and Table-9 macro BPB; phase-boundary Uncheatable BPB is used for the cross-accelerator bridge.
- Constraints: preserve policy coordinates, seeds, batch size, model, optimizer, tokenizer, datasets, validation payloads, and checkpoint steps; keep training data and mutable artifacts region-local; do not use Storage Transfer Service or copy training corpora across regions.
- Coordinating issue: https://github.com/marin-community/marin/issues/8797
- Fieldbook experiment: `exp_01kz3nq7y7mp3a51kz26cvv4tr`

## Baseline

- Date: 2026-08-30
- Code ref: `88ad3e0038fc9603d713c0586c11b7c7c157a1ec` with an approved dirty-tree launch bundle; tracked working-tree diff SHA-256 at 18:29 PDT was `fff7cad10931e90d951d361c16859eff247df18cd37f0845a5ef28409c913de2`.
- Baseline state: 27 of 280 East5 trajectories complete; run orders 27 and 29 have permanent phase-boundary checkpoints; 251.4009 approximate full-run equivalents remain.
- Checkpoint observations: one optimizer-inclusive checkpoint is about 4.3 GB and the final HF export is about 1.45 GB. The launcher retains one rolling temporary checkpoint per active row in a region-local 14-day TTL prefix and permanent checkpoints at steps 21855 and 27335. At 112 simultaneous rows, rolling resume storage is approximately 482 GB. Two permanent Orbax checkpoints plus the HF export are approximately 10.1 GB per completed row.

## Entry Log

### 2026-08-30 18:30 PDT - Candidate assignment and launch contract

- Hypothesis: a region-local Europe v6e-8 deployment can reproduce the East5 v5p-8 trajectory within the frozen paired threshold, allowing the remaining panel to use both regions without changing the scientific design.
- Commit Hash: `88ad3e0038fc9603d713c0586c11b7c7c157a1ec`; production source bundle identity will be appended after submission.
- Assignment: 27 completed East5 rows, 126 East5 rows including resumable run orders 27 and 29, and 127 fresh Europe rows. The candidate assignment is stored at `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/assignment_v1.json`.
- East command file: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/east5_production_command_v1.txt`, SHA-256 `e1de0384ed36fe2d65a51c743a6f5eca7cfc38f38c85a608dcf37bcc5529f886`.
- Europe command file: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/europe_production_command_v1.txt`, SHA-256 `4a8228c6c0631307aab780f30bbc32351e06ec9865ff3c2c30ff5f07504278ef`.
- East command:

```bash
UV_FROZEN=1 uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml --working-dir-exclude .agents/ --working-dir-exclude .github/ --working-dir-exclude docs/ --working-dir-exclude scripts/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-include experiments/domain_phase_mix/exploratory/general_scaling_models.py --working-dir-include experiments/domain_phase_mix/exploratory/dsre_ceq_tools.py --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-exclude checkpoints/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude .experiments/ --working-dir-exclude .experiments.zip -- --no-wait --no-preemptible --job-name dm-delphi-tpp40-east5-v5p8-multiregion-v1-20260830 --region us-east5 --zone us-east5-a --priority interactive --enable-extra-resources --cpu 1 --memory 8GB --disk 16GB --timeout 604800 --extra cpu -e MARIN_PREFIX gs://marin-us-east5 -e MARIN_EXECUTOR_STRICT 1 -- python -m experiments.domain_phase_mix.launch_delphi_augmented_swarm_tpp40 --tpu-type v5p-8 --tpu-region us-east5 --tpu-zone us-east5-a --table9-tpu-type v6e-8 --table9-tpu-zone us-east5-b --assignment-file gs://marin-us-east5/experiments/domain_phase_mix/delphi_tpp40_multiregion_assignment_20260830/assignment_v1.json --assignment-region east5 --expect-assignment-sha256 873959336b2fb3488317d1824c74954767f264d48c308ecd15dc8b1403837a3f --max-concurrent 56
```

- Europe command:

```bash
UV_FROZEN=1 uv run python -m marin.run.iris_run --config lib/iris/config/marin.yaml --working-dir-exclude .agents/ --working-dir-exclude .github/ --working-dir-exclude docs/ --working-dir-exclude scripts/ --working-dir-exclude experiments/domain_phase_mix/exploratory/ --working-dir-include experiments/domain_phase_mix/exploratory/general_scaling_models.py --working-dir-include experiments/domain_phase_mix/exploratory/dsre_ceq_tools.py --working-dir-exclude experiments/domain_phase_mix/manifests/ --working-dir-exclude checkpoints/ --working-dir-exclude tests/ --working-dir-exclude infra/grafana/ --working-dir-exclude .experiments/ --working-dir-exclude .experiments.zip -- --no-wait --no-preemptible --job-name dm-delphi-tpp40-europe-v6e8-multiregion-v1-20260830 --region europe-west4 --zone europe-west4-b --priority interactive --enable-extra-resources --cpu 1 --memory 8GB --disk 16GB --timeout 604800 --extra cpu -e MARIN_PREFIX gs://marin-eu-west4 -e MARIN_EXECUTOR_STRICT 1 -- python -m experiments.domain_phase_mix.launch_delphi_augmented_swarm_tpp40 --tpu-type v6e-8 --tpu-region europe-west4 --tpu-zone europe-west4-a --table9-tpu-type v6e-8 --table9-tpu-zone europe-west4-a --assignment-file gs://marin-eu-west4/experiments/domain_phase_mix/delphi_tpp40_multiregion_assignment_20260830/assignment_v1.json --assignment-region europe --expect-assignment-sha256 873959336b2fb3488317d1824c74954767f264d48c308ecd15dc8b1403837a3f --max-concurrent 56
```

- Config: East parent `us-east5-a`, training `v5p-8` in `us-east5-a`, Table 9 `v6e-8` in `us-east5-b`; Europe parent `europe-west4-b`, training and Table 9 `v6e-8` in `europe-west4-a`; interactive priority; 56 children per parent; final checkpoint step 27335; phase-boundary checkpoint step 21855; W&B training group `delphi_tpp40_augmented_swarm`; Table-9 group `olmo_base_eval_table9_delphi_tpp40_augmented_swarm`; DRI Calvin Xu; monitoring owner Codex heartbeat `finish-europe-and-launch-tpp40` at a 30-minute cadence before launch and 15-minute cadence after launch.
- Output: content-addressed row roots below `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815/` and `gs://marin-eu-west4/pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815/`. Temporary resume checkpoints use each region's `tmp/ttl=14d/checkpoints-temp/` prefix. Iris executor state and logs remain region-local. No Ray runtime is used.
- Initialization: fresh rows have no explicit `initialize_from`; idempotent content-addressed executors resume existing exact-path checkpoints. East run orders 27 and 29 resume from permanent step-21855 metadata after the assignment is re-frozen.
- Result: the East resolved-path audit passed 153 of 153 rows. All 27 completed rows have successful executor status and final markers; run orders 27 and 29 have phase-boundary metadata; all 124 fresh East rows are noncolliding. Audit artifact SHA-256: `3f9468e84ffc1f447de121f2fa15b0353838cc65556712277f21721203267934`.
- Interpretation: assignment arithmetic, locality, lineage identity, and launch commands are ready. The run-order-2 numerical and idempotence gate remains the only scientific launch blocker.
- Next action: finish the Europe bridge, materialize paired Uncheatable and Table-9 results, prove no-op idempotent reruns, pass the final gate, re-materialize the assignment byte-identically, and submit both parents.

### 2026-08-30 23:05 PDT - Operational bridge acceptance and v2 production freeze

- Code ref: `3a5552e10d73af82d394d60fbdb6e56e7d804e97` with dirty-tree Iris bundle `e5fffed68c0cc120d6e38bd53d9572cc17de38b9612ffa556213f22d49011980`. Both regional parents use this same content-addressed bundle.
- Strict result: `bridge_acceptance_report_v3.json` preserves `numerical_acceptance_passed=false` and `production_launch_authorized=false`; SHA-256 `1a2d9e9bf11f461f92827826ca7302e2031d5407279b20930925e487d494f393`.
- Operational result: Europe-minus-East5 deltas are `+0.0036705732` phase-0 Uncheatable, `+0.0022144914` endpoint Uncheatable, and `+0.0018861420` endpoint Table-9 macro. All pass the separate `0.005` BPB operational screen. `bridge_operational_authorization_v2.json` has no blockers and SHA-256 `a938a89287267b51a59bc7edeb894eed9176f28cf0f3df1c4ec7002a6662e6a4`.
- Idempotence: three unchanged parents succeeded with one zero-exit task each, emitted zero child jobs, and left East5 and Europe result inventories unchanged. Evidence SHA-256: `74d273cdd4fa38796692c171e5356141a1e9ae93c186f099db62610e943aa510`.
- Quiescence: the legacy East5 parent was killed and its complete Iris root tree contained 20 succeeded jobs, 57 killed jobs, and zero active jobs at the `2026-08-31T05:48:57Z` assignment observation.
- Assignment: `assignment_v2.json` contains 29 completed, 125 East5, and 126 Europe rows. Row 30 is the sole resumable East5 row. Byte-identical copies were verified in both regional buckets. File SHA-256 is `eeb657d51d9891c25e66f9c559c93337aa1a8b154c6122539ca0e652d70c0473`; semantic SHA-256 is `8074b0d3a92e5e002336389849f33bbd630d9be2ea1580ccf436dfb2b40ea836`.
- East resolved-path audit: 29 completed, 1 resumable, and 124 fresh rows all resolve under the frozen East5 root. Audit SHA-256: `89b38aeb2fc399c87c5c5eafb9ab79260243238e88c85b11dd7fc41c0f5578e0`.
- East command: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/east5_production_command_v2.txt`, SHA-256 `3b8edb0118c6f8544a160d9e3fa4fe7237095ae32ed6eb3f4c432a848938f3fc`.
- Europe command: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_multiregion_assignment_20260830/europe_production_command_v2.txt`, SHA-256 `2af54cdfb6ba43c83c4693333a261eacbd3c2990f5c862ce7890b9dd587e1bc7`.
- Verification: 88 focused tests and targeted pre-commit passed. East5 launch safety passed with Table-9 explicitly placed in `us-east5-b`; Europe launch safety passed with training and Table-9 in `europe-west4-a`. Subscription-safe Claude Code review used `claude-opus-5`, max effort, read-only tools, and returned `GO` with no launch blockers.
- Required follow-up: retain row-to-region assignment; estimate a region fixed effect and region-by-mixture interaction; pool only if the 95% upper bound on interaction RMS is below `0.002` BPB; re-evaluate selected and near-frontier checkpoints in one common region. The single `ngd3dm2_stratified_300m_6b` row is Europe-only and must be excluded from the region fixed-effect estimate or modeled explicitly.
