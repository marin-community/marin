# Rollout engine

`marin-rolloutengine` executes one TaskCompendium task and returns
`RolloutData`: the conversation, exact tokens, loss mask, optional log probabilities, grade, and per-turn records.

Callers supply a model callable, machine factories, and optional `TaskSession`
factories. Each factory receives a `TaskSessionContext` for one attempt;
a session class with that constructor signature can be registered directly.
Each session prepares model requests, handles responses and tools, and grades the attempt.
The built-in `shellbox` session uses the same execution
loop as registered sessions and handles shell tools. Optional `ToolProvider`s, keyed by
`ProviderRequirement.action_interface`, serve host tools that the session advertises beside the shell tool. `AgentTaskSession` adapts
an agent that controls its own prompt and tool loop. Its factory can use the
default grader supplied by the engine.
For text, number, JSON, and native-action answers, it adds the task's answer-format instruction and tools to the model request.
For workspace-state tasks, this session adds shell and final-response instructions to the model request.
Importers retain the task problem without these interface instructions.
The engine owns model calls, conversation and token accumulation, deadlines, the optional
`max_total_tokens` budget, and resource cleanup.
An optional `RolloutObserver` passed as `RolloutEngine(..., observer=...)` receives typed progress events
from `rolloutengine.contracts` while the attempt runs, so callers can log or trace a rollout before `run()` returns.
Each event carries the `run_key` passed to `run()`, which defaults to the task id.
The engine delivers events from a queue and never awaits the observer, so a slow or failing observer does not change the rollout.

`RolloutEngine.run(lowered)` accepts a `LoweredTaskSpec` and asynchronously returns one rollout.
The lowered record preserves its `TaskSpec` and adds machine selections and session limits.
The supported scope is single-stage tasks with prebuilt, digest-pinned images, local package-lock environments, or the built-in ShellSim filesystem.
After the turn loop, the engine captures the files under the task's `output_paths` from the task machine once
and returns them in `RolloutData.outputs`; final grading uses that capture.
The default session then grades an in-process `VerifyitGrader` on the host.
A `ScriptGrader`, or a `VerifyitGrader` with an environment, grades in a separate verifier machine using its digest-pinned image or a local environment built from its package lock.
A `SessionGrader` requires a registered task session. A `NoGrader` task receives an `unavailable` grade.
`TaskSessionSpec.trace` selects `TraceMode.EXACT_TOKENS` (the default) or `TraceMode.MESSAGES`.
`MESSAGES` accepts chat-endpoint turns without checking or requiring token IDs, records an all-zero loss mask, and marks each record with `metrics["trace"] = "messages"`; never train on those rollouts.
See the [task rollout reference](../../docs/references/task-rollouts.md)
for the session lifecycle, exact-token contract, trace modes, failure handling, and backend configuration.

From the Marin repository root:

```bash
uv run --package marin-rolloutengine --frozen --group test pytest lib/rolloutengine/tests -q
```
