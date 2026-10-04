# Rollout engine

`marin-rolloutengine` executes TaskCompendium tasks through Shellbox and returns
the original prompt and generated token IDs, loss masks, log probabilities, and grades.
Callers can add a failure record when they convert an interrupted rollout.

The `rolloutengine.contracts` module defines `TaskSession`, `SessionStart`, and
the model and result data classes. The `rolloutengine.engine` module implements
`ShellboxRolloutEngine`.
Callers supply inference and `TaskSession` creation as callables.
`run()` asynchronously executes one task. Callers control task concurrency and
cancel the coroutine when they no longer need its result. Cancellation waits for
active operations and machine cleanup.

`machines.py` owns machine setup and cleanup. `grading.py` executes private
graders and collects rewards. `task_session.py` prepares model requests and
implements Shellbox task operations. `engine.py` controls model calls, token
accumulation, and stage progression.

TaskCompendium owns task definitions, serialization, importers, and grading
contracts. It does not depend on this package or Shellbox.
SkyRL supplies inference, schedules rollout workers, and converts rollout
records to training batches.

See the [task rollout reference](../../docs/references/task-rollouts.md) for
interfaces, task formats, and backend configuration.

Run the CPU tests from the Marin repository root:

```bash
uv run --project lib/rolloutengine --frozen --group test pytest lib/rolloutengine/tests -q
```
