# Taskforge

Taskforge is an effect-driven task pipeline. Keep stages and models pure; put I/O,
clock, id, randomness, and resource mutation behind executors. A launch resumes at
its last stage boundary; inside a stage, cached effects hit by their keys and
sandbox effects run live, never replayed.

- Use frozen dataclasses with `slots=True`. Model fields use `Mapping`,
  `Sequence`, `tuple`, or `frozenset`; no mutable collections or module state.
- Annotate every function. Use PEP 695 generics, `TypeIs` from
  `typing_extensions`, and `@override` on protocol implementations. Keep `Any`
  inside JSON boundary adapters only.
- The effect decisions are `Transient`, `Restart`, `Hold`, and `LogFailure`.
  A new exception needs a written decision rationale in `effects/issues.py`.
  No bare catches or catch-all outside the process-boundary driver. Executors
  own retries; observers report issues at scope checks.
- Follow root `TESTING.md`: behavioral tests, fake executors, no tautologies.
  Network and sandbox tests must be marked `cluster` or `live_glm`.
- Run from the workspace root:
  `uv run --frozen --package marin-taskforge --group test pytest lib/taskforge/tests -q`;
  `uv run --frozen pyrefly check`;
  `uv run --frozen --package marin-taskforge --group test pyrefly check --config lib/taskforge/pyproject.toml`;
  `./infra/pre-commit.py --all-files --fix`.
- Keep dependency direction: model <- effects <- resources <- lowering <- harness
  <- config <- stages <- coordinator. Copied code carries no source-citation header; its provenance belongs in
  the commit message and the PR description.
- `taskforge.llm.client.ModelClient` is the one owner of model retries; a new model host is a `Provider`
  and a new model is a `ModelProfile`, never a second client or a retry loop around one.
- `Event` alone uses an explicit covariant `TypeVar`: PEP 695 infers invariance
  for its phantom result type. The event-class registry, keyed by each class's stable `kind`, is the module-state
  exception needed for typed replay; logs never dynamically import classes.
