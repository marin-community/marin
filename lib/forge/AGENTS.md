# Forge

Forge is an effect-driven task pipeline. Keep stages and models pure; put I/O,
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
  `uv run --frozen --package marin-forge --group test pytest lib/forge/tests -q`;
  `uv run --frozen pyrefly check`;
  `uv run --frozen --package marin-forge --group test pyrefly check --config lib/forge/pyproject.toml`;
  `./infra/pre-commit.py --all-files --fix`.
- Use SSH, branches `mark/forge-<wp>`, and squash merges. Never force-push.
  Do not add attribution trailers. Label PRs `agent-generated`, report net
  library lines against the work-package budget, and address automated review.
  Do not request human review or merge; report readiness to the coordinator.
- Do not edit `lib/taskforge`; it remains frozen during the rewrite.
- Report verbatim copies from `origin/mark/taskforge-integration` separately as
  copied lines; they are exempt from the work-package library line budgets.
