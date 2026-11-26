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

`rolloutengine.completion` and `rolloutengine.likelihood` provide raw-text
generation and continuation-scoring transports. Corpus scheduling, FineStore
results, and resume belong to `marin.rollouts.evaluation`; dataset protocols and
corpus readers live in TaskCompendium. See
[evaluation tasks](../../docs/references/evaluation-tasks.md).

From the Marin repository root:

```bash
uv run --package marin-rolloutengine --frozen --group test pytest -c lib/rolloutengine/pyproject.toml lib/rolloutengine/tests -q
```

## Inference admission and execution clocks

SkyRL task workers bind a scoped `InferenceClock` to each trial. The model transport
returns a request-correlated engine `SCHEDULED` or first-token notice over the same
cancellable Ray stream as the result. Submission alone does not start execution.
The recorder's total-turn and model-turn budgets begin at that notice; the attempt
clock still bounds setup before the first request and starts its execution budget
afresh at the first admission. Later queue waits pause remaining execution budgets.
Harbor's attempt and agent-phase deadlines use the same clock. Installed-agent HTTP
callbacks restore the registering trial's clock; proxy and client transport
watchdogs and bearer lifetimes include the bounded queue allowance. Detached agent
commands retain a hard setup + queue + execution safety watchdog, while their
local active agent budget stays unchanged.

`trajectory_runner.inference_queue_timeout` bounds cumulative admission wait
separately (2280 seconds by default). A never-scheduled request becomes an
`InferenceAdmissionTimeout`, not a total-turn timeout or a scored zero. The rollout
reports `timing/inference_queue_wait` separately from `timing/inference_generation`,
plus scheduled-request and queue-timeout counts. Engine `request_timing` logs retain
`QUEUED`/`SCHEDULED`/first-token evidence. Missing traces are not evidence that no
engine work occurred.

Verifier-unavailable rows have zero token-loss masks and are excluded from RLOO-N
baselines, with zero advantage. Any stored reward zero is an archive placeholder.
The 1770-second active partial-output cutoff also starts after scheduling; it does
not consume queue wait. Adopt these controls only with a fresh run version.

Micro210 infrastructure recoveries select original row indices and carry a
`_source_uid` column through filtering, so native trial identities and their frozen
per-row settings survive subsetting. Merge only originally unavailable identities;
keep every original graded outcome unchanged and the assigned cohort fixed.
