# Rollout engine

`marin-rolloutengine` runs and grades one attempt of a TaskCompendium task. It returns a `GradedRollout`: the
`Rollout`, with its conversations, exact tokens, loss masks, optional log probabilities, per-turn steps, and run
outputs, and its grade.

```python
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness

engine = RolloutEngine(machine_factories, {"shellbox": ShellHarness(command_timeout=30)})
result = await engine.run_task(spec, model)
```

- `TaskRolloutSpec` holds an unchanged `TaskSpec` and the choices that run it: the harness name, the task and
  verifier machines, the limits, the trace mode, and an optional judge connection that is never serialized.
- The model is the policy under rollout, an async function from `ModelRequest` to `ModelTurn`. Pass it to each
  `engine.run_task` call.
- A harness is the agent loop of a task. It implements `Harness`: `check(task, limits)` rejects a task before
  any machine starts, and `run(context, session)` runs one attempt. The harness calls the model through
  its `ModelSession` (`start`, `reply`, and `record`), which records the exact tokens and applies the turn
  and token limits. Register each harness in the `harnesses` mapping; `TaskRolloutSpec.harness` selects one by
  its key.
- A harness does not grade. The engine grades the last conversation that has a model turn with the task's
  grader. A harness can attach its own step data, such as rewards for training, with `record(data=...)`.
- Built-in harnesses: `ShellHarness` serves a shell tool on the task machine and host tools from registered
  providers; `AgentHarness` runs an agent that owns its prompts and tool loop, for example an installed agent
  behind an HTTP proxy; `StepHarness` runs an `Episode` that returns the first prompt and handles each turn.
  Without interactive tools, `ShellHarness` answers an ordinary conversation and finishes after one response.
  `AssistantPrefixHarness` continues the final assistant text once. Select it explicitly for unfinished assistant
  messages; it can advertise final tool schemas but cannot execute tools.
- `harness_rollout` runs one harness on one task with one model, without grading. `task_machine` starts the
  task machine for it.
- The engine owns the machines, the attempt deadline, grading, cleanup, and progress events for an optional
  observer, `RolloutEngine(..., observer=...)`.

`TraceMode.EXACT_TOKENS`, the default, enforces the exact-token contract. `TraceMode.MESSAGES` accepts chat
endpoints that re-tokenize each request and records an all-zero loss mask. Never train on those rollouts:
their token IDs need not be the tokens that the model sampled.
See the [task rollout reference](../../docs/references/task-rollouts.md) for the harness contract, the
exact-token contract, trace modes, failure handling, and backend configuration.

From the Marin repository root:

```bash
uv run --package marin-rolloutengine --frozen --group test pytest lib/rolloutengine/tests -q
```
