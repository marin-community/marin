# Debugging log for alternating W&B lifecycle

Alternating RL trainer subprocesses were showing up in W&B as crashed with no
runtime even when the durable controller state had already advanced to the next
phase. The goal is to make the alternating training W&B run accurately reflect a
successful phase boundary.

## Initial status

The active soak `alt-tpu-002e-v5p-soak5` advanced durably to:

- `status=sampling`
- `phase_id=1`
- `policy_version=1`
- `source_global_step=1`

But the linked W&B trainer run
`alt-tpu-002e-v5p-soak5-alternating-train` was marked `crashed` and the W&B API
reported:

- `state=crashed`
- `summary_runtime=None`
- `summary_step=None`

## Hypothesis 1

Alternating training starts a trainer tracker but never explicitly finishes it,
so the subprocess exits without flushing the run. Normal Levanter subprocess
entrypoints explicitly call `trainer.tracker.finish()` after training.

## Changes to make

- Update `lib/marin/src/marin/rl/alternating/training_phase.py` to finish the
  trainer tracker on the successful phase path after logging alternating phase
  metrics.
- Add a regression test in `tests/rl/test_alternating_training_phase.py` that
  verifies tracker `finish()` is called after phase metrics are logged.

## Future Work

- [ ] Verify the next TPU run reports non-empty runtime and step in W&B
- [ ] Decide whether alternating should also have a controller-level W&B run in
      addition to the trainer subprocess run

## Results

Confirmed the lifecycle gap in code:

- alternating training exits `Trainer(...)` but never calls `tracker.finish()`
- `Trainer.__exit__` only unwinds context managers; it does not finish trackers
- standard Levanter training entrypoints explicitly call `trainer.tracker.finish()`

Patched alternating training to finish the tracker after phase metrics are
logged, matching the standard subprocess pattern.

## Follow-up issue

Even after the trainer lifecycle fix, the active soak still had a visibility
gap: W&B only showed the trainer subprocess run. Sampling, materialization, and
controller state transitions were only visible in durable manifests and TPU
logs, which made the soak look idle or missing in W&B between training phases.

## Hypothesis 2

Alternating RL never created a controller-level tracker. The controller already
had the durable data needed for W&B:

- run state transitions in `state/run_state.json`
- per-phase timing manifests in `state/phase_metrics/`
- sampling host completion manifests under `sampling/phase_*/host_*/status.json`
- materialization manifests under `materialized/phase_*/manifest.json`

So the visibility gap was not missing data. It was missing W&B wiring in the
controller loop.

## Additional changes

- Added `lib/marin/src/marin/rl/alternating/wandb.py` to derive explicit
  alternating W&B identities from the base trainer `WandbConfig`
- Added a resumed controller run:
  - run id: `<run_id>-alternating-controller`
  - display name: `<run_id>-alternating-controller`
- Updated the trainer subprocess run metadata so future alternating trainer runs
  are explicit and group with the controller run:
  - display name: `<run_id>-alternating-train`
  - group: `<run_id>`
  - extra tags: `alternating`, role-specific tag
- Updated `lib/marin/src/marin/rl/alternating/controller.py` to log:
  - controller start/resume
  - sampling manifest creation
  - sampling completion and host counts
  - materialization start/completion
  - training completion
  - phase advance
  - terminal completed / failed state
  - per-phase timing metrics whenever the phase metrics manifest exists
- Added controller regression tests in
  `tests/rl/test_alternating_controller.py`
- Added trainer metadata regression coverage in
  `tests/rl/test_alternating_training_phase.py`

## Validation

- `uv run pytest tests/rl/test_alternating_controller.py -o addopts=`
- `uv run pytest tests/rl/test_alternating_training_phase.py -o addopts=`
- `./infra/pre-commit.py --all-files --fix`

## Remaining caveat

The currently running soak on `alt-v5p-probe-001` is still using the older TPU
image, so it will not retroactively gain the new controller W&B run. The next
image build and TPU launch should show both alternating runs continuously.

## Follow-up issue 2

The first relaunch attempt with the new controller tracker still spent minutes
before phase 0 doing local W&B startup work. Process sampling showed the
controller inside `wandb.run.log_code(...)`, hashing and copying files from the
entire repo before entering the controller loop.

## Hypothesis 3

Alternating W&B runs do not need repo code snapshots on every controller or
trainer startup. The durable experiment record already contains:

- exact image digest
- exact run id
- exact controller command
- repo revision in the local workspace / logbook

So `save_code=True` is pure startup tax here, especially for the controller.

## Additional changes 2

- Updated `lib/marin/src/marin/rl/alternating/wandb.py` so alternating
  role-specific `WandbConfig` objects set:
  - `save_code=False`
- Extended regression coverage so both controller and trainer alternating W&B
  configs assert `save_code` is disabled

## Result

After rebuilding the TPU image with `save_code=False`, the next relaunch moved
through controller W&B initialization immediately and reached live phase-0
sampling on the active probe. That removed the controller startup stall without
reducing metric visibility.
