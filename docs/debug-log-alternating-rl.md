# Debugging log for alternating RL

Bring the alternating single-allocation RL path closer to something that can run correctly end to end on TPU pods.

## Initial status

The alternating controller, phase entrypoints, manifests, and tests already exist. The next pass is focused on runtime-correctness issues that are easy to miss in unit tests: Levanter checkpoint resume semantics across phases, optimizer schedule consistency across phase boundaries, and failure handling when one sampling host dies before the rest of the phase completes.

## Hypothesis 1

Training-phase resume is currently using the wrong checkpoint root and can miss prior Levanter checkpoints entirely.

## Changes to make

- Inspect how Levanter expands `checkpointer.base_path` with `run_id`.
- Update alternating training to resume from the exact checkpoint path recorded in run state, and discover new checkpoints under the resolved trainer checkpoint root instead of the parent directory.
- Add tests that pin the expected checkpoint root behavior.

## Future Work

- [ ] Validate the identical-input `hax.shard()` loading path on real multi-host TPU runtime.
- [ ] Validate alternating phase metrics on a real WandB-backed TPU run.
- [ ] Add an integration test that exercises one full alternating phase with a tiny local trainer stub.

## Results

- Confirmed the original bug: alternating training was discovering checkpoints
  under the parent checkpoint directory even though Levanter appends the trainer
  run id to `checkpointer.base_path` by default.
- Fixed alternating training to resume from the exact checkpoint path recorded
  in run state and to discover new checkpoints under the resolved trainer
  checkpoint root.
- Added `tests/rl/test_alternating_training_phase.py` to lock in the corrected
  checkpoint-root behavior.

## Hypothesis 2

Training currently rebuilds the optimizer against the current phase boundary instead of the total run horizon, which can distort LR schedules across phases.

## Changes to make

- Rebuild the optimizer against the full configured training horizon.
- Add a targeted unit test that the alternating training phase uses the full run schedule even when the current phase stops early.

## Results

- Fixed alternating training to build the optimizer against the full configured
  run horizon while still stopping each phase at the local boundary.
- Added a regression test that captures the optimizer build argument and phase
  target step.

## Hypothesis 3

Sampling wait logic should fail fast when a host writes a failure status, instead of waiting until timeout behind a dead host.

## Changes to make

- Add a helper that scans any written host status manifests during wait.
- Raise immediately when a failure status appears.
- Add a unit test covering early failure detection.

## Results

- Added fail-fast sampling wait logic in `ExistingPodPhaseHooks` so the
  controller raises immediately once any host reports `FAILED`.
- Added `tests/rl/test_alternating_runtime.py` to cover this path.

## Hypothesis 4

The spill-to-disk materializer still has an avoidable pass-2 memory spike
because it reloads every spilled rollout object before batch packing.

## Changes to make

- Plan deterministic batch assignments from lightweight per-env index metadata.
- Stream spill shards once, routing selected records into local per-batch append
  files.
- Finalize each `TrainingBatch` from its local append file instead of from a
  global in-memory replay map.
- Add a regression that forces two materialized batches and checks that
  `max_samples=1` prevents rollout reuse.

## Results

- Replaced the in-memory pass 2 with a planned streaming pass:
  - batch planning now keeps only per-env index/usage metadata in memory
  - spill shards are streamed through local per-batch append files
  - final `TrainingBatch` packing happens one batch at a time
- Added a multi-batch materialization regression in
  `tests/rl/test_alternating_materialization.py`.

## Hypothesis 5

Operational debugging on real TPU will be painful unless the alternating path
records explicit per-phase timings and survives silent sampler death.

## Changes to make

- Persist per-phase timing metrics durably so controller/runtime/training stages
  can contribute to one shared metrics record.
- Add container-liveness checks during sampling wait so a dead host without a
  status file fails quickly.
- Log phase timing metrics to the active tracker when training/export has a
  tracker context.

## Results

- Added durable `PhaseMetricsManifest` support and per-phase metrics paths.
- Runtime now records prepare-sampling, sampling, curriculum-update, and
  materialization timings.
- Training/export now record training/export timings and log the aggregated
  phase metrics to the active tracker when present.
- The controller now logs a per-phase timing summary from the durable metrics
  manifest after each completed phase, so TPU smoke tests have readable
  boundary-cost breadcrumbs even before we trust dashboard wiring.
- Added a runtime regression covering the "container exited without status"
  path.

## Hypothesis 6

Recovery from "training finished, export did not" should not require rerunning
the whole training phase.

## Changes to make

- Split policy export into a reusable helper.
- Add an export-only recovery entrypoint that reads run state plus the
  materialized manifest, verifies the latest checkpoint step, and writes the
  next policy manifest.
- Avoid reloading the original HF checkpoint on resumed zero-KL phases where
  trainer checkpoint restore immediately overwrites the model weights.

## Results

- Added `export_policy_only(...)` plus the `export-policy` experiment subcommand.
- Added a regression test for export-only recovery.
- Resumed zero-KL training phases now skip the unnecessary initial checkpoint
  load while preserving the original reference-model path for nonzero-KL runs.
