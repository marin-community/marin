# Delphi TPP40 East5 bridge-reference repair review

Return `GO` or `NO-GO` as the first line. This is a high-stakes, read-only launch review. Do not edit files or launch jobs.

## Why this repair exists

The Europe bridge is already training the preregistered run orders `2,120,240,260`. Before inspecting any Europe result, a live Iris/GCS audit found that the row-set correction inherited an invalid assumption from an older draft: only run order 2 has a historical East5 TPP40 result; 120, 240, and 260 do not. The frozen representative row set and every numerical threshold remain unchanged. The proposed repair trains all four East5 references together in a new isolated namespace, so the comparison uses one current code bundle rather than mixing historical and fresh East5 rows.

Production multiregion launch remains blocked until both bridge halves complete, detached phase-boundary evaluation is complete, every frozen numerical threshold passes, and unchanged idempotent reruns skip all work.

## Exact inputs

- East5 command: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/east5_bridge_command_v1.txt`
- East5 dry-run audit: `/tmp/delphi_tpp40_east5_bridge_v1_dry_run/launch_audit.json`
- East5 dry-run run specs: `/tmp/delphi_tpp40_east5_bridge_v1_dry_run/run_specs.json`
- Running Europe command: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/europe_bridge_command_v3.txt`
- Frozen acceptance contract: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_tpp40_europe_readiness_20260830/bridge_acceptance_contract_v2.json`
- Launcher: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/launch_delphi_augmented_swarm_tpp40.py`
- Launch-safety checker: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/east5_launch_safety.py`

## Verified local evidence

- Command SHA-256: `259f6aa0c318a103db9dccd1ccbc434fa8c876fa866ed8c0c0d0482b203e61d3`.
- Launch safety passes with parent `us-east5-a`, training `v5p-8` in `us-east5-a`, Table-9 `v6e-8` in `us-east5-b`, and only `gs://marin-us-east5` paths.
- Dry run selects exactly `[2,120,240,260]` and reports the frozen source-panel, coordinate, fixed-identity, and scientific-identity hashes.
- Dry run reports phase-0 checkpoint step 21,855 and endpoint step 27,335.
- East5 experiment, training W&B group, and Table-9 W&B group are explicit and isolated from both the production East5 namespace and the Europe bridge namespace.
- The current legacy production parent has not emitted rows 120, 240, or 260. Its currently pending rows are 26-38. The proposed command writes to a different experiment namespace regardless.
- Focused launcher and launch-safety tests: 42 passed.

## Review questions

1. Is launching a fresh isolated East5 half for the already-frozen four rows a scientifically valid repair, given that the gap was found before inspecting Europe results and neither row selection nor thresholds change?
2. Does the exact command preserve data, seeds, model, optimizer, schedule, mixture coordinates, phase transition, batch size, and evaluations while changing only the intended deployment from Europe `v6e-8` to East5 `v5p-8`?
3. Can this command collide with or contaminate the legacy production East5 graph, the Europe bridge, or future production assignment state?
4. Are parent, training, Table-9, runtime-cache, validation-cache, checkpoint, source-panel, and executor paths correctly region-local?
5. Are the explicit East5 Table-9 zone and filtered working-directory bundle correct?
6. Is the command idempotently resubmittable, and will all four rows retain the required phase-0 and endpoint checkpoints plus endpoint Uncheatable and native Table-9 outputs?
7. Identify any blocker that should prevent submission now. Distinguish later detached phase-boundary evaluation and numerical acceptance work from blockers to starting this isolated reference training.
