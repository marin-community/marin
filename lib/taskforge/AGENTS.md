# Taskforge

Taskforge is an effect-driven task pipeline. Keep stages and models pure; put I/O,
clock, id, randomness, and resource mutation behind executors. Replay resolved
effects, never whole stages.

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
  <- config <- stages <- coordinator. Copy modules with a source citation.
- `taskforge.llm` is the GLM transport kept from the previous package; extend it
  rather than adding a second client.
- `Event` alone uses an explicit covariant `TypeVar`: PEP 695 infers invariance
  for its phantom result type. The event-class registry is the module-state
  exception needed for typed replay; logs never dynamically import classes.
