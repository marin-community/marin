# Debugging log for iris-rl-e4ms2-500-ckptdbg retro

Retro goal: determine what is destabilizing the long-running trainer in
`/ahmed/iris-rl-e4ms2-500-0328-ckptdbg`, and specifically whether the observed
retries are pre-emptions or some other failure class.

## Initial status

User suspicion:
- the run is not obviously being pre-empted, so something else must be causing
  the trainer to restart.

Current Iris state at retro time:
- trainer child `failure_count=3`
- trainer child `preemption_count=0`
- rollout-0 `failure_count=0`, `preemption_count=0`
- rollout-1 `failure_count=1`, `preemption_count=0`

## Hypothesis 1

The trainer failures are not pre-emptions at all; they are repeated checkpoint
save failures in the same serializer subphase.

## Changes to make

No code changes for this retro. Pull the full trainer log history and extract:
- checkpoint start/phase lines
- coordination fatal lines
- wandb resume lines
- checkpoint load lines

## Results

Confirmed. The active run has had three trainer failures, all with
`preemption_count=0`, and each failure matches the same signature:

1. The trainer starts a checkpoint and enters
   `PHASE: CHECKPOINT ... phase=tensorstore_serialize`.
2. JAX logs:
   - `Waiting for previous serialization to finish`
   - `Thread joined successfully`
   - `Error check finished successfully`
3. The run remains stuck in `tensorstore_serialize`.
4. No `Starting commit to storage layer by process: 0` ever appears.
5. JAX distributed then kills the process with:
   - `UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats)`
   - `RPC: /tensorflow.CoordinationService/PollForError`
6. The trainer later resumes from the last completed checkpoint.

Observed failure windows:

- Failure 1:
  - checkpoint start: `step=98` at `2026-03-29 08:37:21 UTC`
  - fatal coordination error: `2026-03-29 08:39:11 UTC`
  - resumed wandb: `2026-03-29 08:40:22 UTC`
  - resumed from checkpoint: `step-94`

- Failure 2:
  - checkpoint start: `step=153` at `2026-03-29 12:33:42 UTC`
  - fatal coordination error: `2026-03-29 12:35:46 UTC`
  - rollout side immediately saw `FlightUnavailableError` / `Connection refused`
    because the trainer Arrow Flight endpoint vanished
  - trainer then restarted

- Failure 3:
  - checkpoint start: `step=248` at `2026-03-29 18:52:21 UTC`
  - fatal coordination error: `2026-03-29 18:54:12 UTC`
  - resumed wandb: `2026-03-29 18:55:28 UTC`
  - resumed from checkpoint: `step-244`

What this rules out:
- TPU pre-emption as the primary driver for these trainer retries
- rollout workers failing first
- commit/metadata-write as the first failing checkpoint subphase

What this supports:
- repeated trainer death in checkpoint pre-commit serialization
- the surfaced coordination fatal is secondary, not the underlying root cause
- rollback to the last durable completed checkpoint is working, but only after
  losing several steps of progress each time

Secondary fallout after trainer restart:
- rollout workers keep generating briefly against the old weight lineage
- replay buffer then logs a burst of `Skipping stale rollout batch (...)`
- this is expected recovery fallout, not the initial cause

## Hypothesis 2

The remaining unknown is the lower-level reason inside
`tensorstore_serialize`: TPU runtime wedge, TensorStore/JAX serialization bug,
or some other native crash before Python can log a traceback.

## Results

Still unresolved from trainer logs alone. The trainer logs are good enough to
pin the failure boundary, but not the deeper TPU-side cause.

## Future Work

- [ ] Compare the three failure windows against worker-attempt logs if Iris can
      surface lower-level task-attempt diagnostics.
- [ ] Correlate checkpoint RSS and serializer duration with failure frequency.
- [ ] Investigate why some checkpoints succeed after 180s+ while others die
      earlier in the same serializer phase.
