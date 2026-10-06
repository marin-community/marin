# Rollout engine

`marin-rolloutengine` executes one TaskCompendium task through Shellbox and returns
`RolloutData`: the conversation, exact tokens, loss mask, optional log probabilities, grade, and per-turn records.

Callers supply a model callable, machine factories, and optional `TaskSession`
factories for task actions and grading.
The default session handles shell tools.
The engine owns model calls, conversation and token accumulation, stage progression, and resource cleanup.

`ShellboxRolloutEngine.run(task, execution=...)` accepts a `TaskSpec` and separate
`TaskExecution` settings, then asynchronously returns one rollout.
See the [task rollout reference](../../docs/references/task-rollouts.md)
for the session lifecycle, exact-token contract, failure handling, and backend configuration.

From the Marin repository root:

```bash
uv run --project lib/rolloutengine --frozen --group test pytest lib/rolloutengine/tests -q
```
