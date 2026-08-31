---
topic: delphi-tpp40-multiregion
issue: https://github.com/marin-community/marin/issues/8797
description: Complete the 280-row Delphi TPP40 swarm across disjoint East5 and Europe assignments.
author: Calvin Xu
---

# Delphi TPP40 Multiregion: Task Logbook

## Current TL;DR

- Production is gated on the paired run-order-2 bridge. The frozen threshold is an absolute BPB difference of at most `0.002` for phase-boundary Uncheatable, endpoint Uncheatable, and endpoint Table 9.
- Candidate assignment semantic SHA-256: `873959336b2fb3488317d1824c74954767f264d48c308ecd15dc8b1403837a3f`; byte SHA-256: `72f88de70e32e2f50d7a26018236c4c4228232bc900f294623b8155bb36a0f58`.
- The assignment covers 27 completed East5 rows, 126 East5-assigned rows including two resumable rows, and 127 fresh Europe rows.
- The exact East lineage audit passed for all 153 East rows. Production commands have passed locality checks and dry runs. They remain unsubmitted until the bridge gate passes and the assignment is re-materialized byte-identically.

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
