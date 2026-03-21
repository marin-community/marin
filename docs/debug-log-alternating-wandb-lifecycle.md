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

## Follow-up issue 3

Even after controller + trainer runs were visible, the phase-0 trainer run
still looked empty in W&B during `ALT-TPU-002E retry3`. The TPU logs showed the
trainer really did:

- load the base model
- run one train step
- save checkpoint `step-1`
- export `policy_0001`

But the W&B API still reported zero history rows for both the trainer and
controller runs while the job was live.

## Hypothesis 4

Alternating training was not actually using the async RL training hook stack.
It only relied on bare `Trainer` defaults plus the final alternating
phase-metrics log, which meant:

- no async-style step timing metrics
- no async-style `train/samples` table
- no MFU / throughput performance hook from the RL train worker

There was also an extra one-step-run hazard: alternating phases use
`steps_per_phase=1`, so if every metric lands on a single explicit W&B step,
that row needs an explicit commit to avoid looking empty until a later step
arrives.

## Additional changes 3

- Added shared RL training observability helpers in:
  - `lib/marin/src/marin/rl/training_observability.py`
- Moved the common async RL metric hooks there so alternating and async now use
  the same code path for:
  - step timing metrics
  - `train/samples` W&B table
  - throughput / MFU logging
  - explicit per-step W&B commit
- Updated `lib/marin/src/marin/rl/train_worker.py` to use the shared helper
  instead of maintaining a parallel local hook implementation
- Updated `lib/marin/src/marin/rl/alternating/training_phase.py` to install the
  same hook stack during alternating training
- Updated `lib/marin/src/marin/rl/alternating/materialization.py` to persist a
  per-batch sample-preview sidecar so alternating can log the same async-style
  `train/samples` table even though it trains from materialized batches rather
  than a live replay loader

## Validation 2

- `uv run pytest tests/rl/test_alternating_training_phase.py -o addopts=`
- `uv run pytest tests/rl/test_training_observability.py -o addopts=`
- `uv run python -m py_compile lib/marin/src/marin/rl/training_observability.py lib/marin/src/marin/rl/alternating/training_phase.py lib/marin/src/marin/rl/alternating/materialization.py lib/marin/src/marin/rl/train_worker.py`

## Remaining caveat 2

This fix is in repo only. The currently running `ALT-TPU-002E retry3` TPU image
does not include it, so that live run will not retroactively gain the new
async-style trainer metrics. The next image build / relaunch is required to
verify W&B history on real hardware.
