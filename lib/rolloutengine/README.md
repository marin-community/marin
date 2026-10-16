# Rollout engine

`marin-rolloutengine` runs one attempt of a TaskCompendium task and returns `RolloutData`: the
conversation, exact tokens, loss mask, optional log probabilities, grade, per-turn steps, and run outputs.

```python
from rolloutengine.engine import RolloutEngine
from rolloutengine.shell import shell_harness

engine = RolloutEngine(machine_factories, {"shellbox": shell_harness(command_timeout=30)})
record = await engine.run(spec, model)
```

- `RolloutSpec` holds an unchanged `TaskSpec` and the choices that run it: the harness name, the task and
  verifier machines, the limits, and the trace mode.
- The model is the policy under rollout, an async function from `ModelRequest` to `ModelTurn`. Pass it to each
  `engine.run` call.
- A harness runs the model side of a task. Register each `Harness(open, check, grade)` in the `harnesses`
  mapping; `RolloutSpec.harness` selects one by its key. `check` rejects a spec before any machine starts,
  `open` starts an episode for one attempt, and `grade` grades a task whose grader is `SessionGrader`, which
  leaves grading to the harness. The engine grades every other task with the task's grader.
- An episode returns the first prompt (`prepare`), handles each model turn (`advance`), and releases its
  resources (`close`). Each `Transition` that `advance` returns becomes one of `RolloutData.steps`, and it
  can carry the harness's own step data, such as rewards for training.
- Built-in harnesses: `shell_harness` serves a shell tool on the task machine and host tools from
  registered providers; `agent_harness` runs an agent that owns its prompts and tool loop, for example an
  installed agent behind an HTTP proxy, and sends each of its model calls through the engine.
- The engine owns the machines, the turn loop and its limits, token accumulation, grading, cleanup, and
  progress events for an optional observer, `RolloutEngine(..., observer=...)`.

`TraceMode.EXACT_TOKENS`, the default, enforces the exact-token contract. `TraceMode.MESSAGES` accepts chat
endpoints that re-tokenize each request and records an all-zero loss mask. Never train on those rollouts:
their token IDs need not be the tokens that the model sampled.
See the [task rollout reference](../../docs/references/task-rollouts.md) for the harness lifecycle, the
exact-token contract, trace modes, failure handling, and backend configuration.

From the Marin repository root:

```bash
uv run --package marin-rolloutengine --frozen --group test pytest lib/rolloutengine/tests -q
```
