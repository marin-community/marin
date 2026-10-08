# Rollout engine

`marin-rolloutengine` executes one TaskCompendium task through Shellbox and returns
`RolloutData`: the conversation, exact tokens, loss mask, optional log probabilities, grade, and per-turn records.

Callers supply a model callable, machine factories, and optional `TaskSession`
factories for task actions and grading.
The default session handles shell tools.
For text, number, JSON, and native-action answers, it adds the task's answer-format instruction and tools to the model request.
For workspace-state tasks, this session adds shell and final-response instructions to the model request.
Importers retain the task problem without these interface instructions.
The engine owns model calls, conversation and token accumulation, deadlines, and resource cleanup.

`ShellboxRolloutEngine.run(lowered)` accepts a `LoweredTaskSpec` and asynchronously returns one rollout.
The lowered record preserves its `TaskSpec` and adds machine selections and session limits.
The supported scope is single-stage tasks with prebuilt, digest-pinned images or the built-in ShellSim filesystem.
After the turn loop, the default session grades an in-process `VerifyitGrader` on the host.
A `ScriptGrader`, or a `VerifyitGrader` with an environment, grades in a separate verifier machine built from its digest-pinned image.
A `SessionGrader` requires a registered task session. A `NoGrader` task receives an `unavailable` grade.
The Harbor importer accepts only separate verifier environments.
An unset verifier mode without a separate environment selects shared mode and causes rejection.
See the [task rollout reference](../../docs/references/task-rollouts.md)
for the session lifecycle, exact-token contract, failure handling, and backend configuration.

From the Marin repository root:

```bash
uv run --project lib/rolloutengine --frozen --group test pytest lib/rolloutengine/tests -q
```
