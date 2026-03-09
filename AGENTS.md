# Agent Guidelines for Marin

Start with the shared practices below. Consult subproject manuals for directory-specific guidance:

- `lib/levanter/AGENTS.md` — Levanter (JAX training library)
- `lib/marin/AGENTS.md` — Marin (pipeline framework)
- `lib/iris/AGENTS.md` — Iris (job orchestration)

## Workflow Playbooks

Agent-friendly skills live in `.agents/skills/` (also accessible as `.claude/skills/`). Load a skill by reading its `SKILL.md`. Key ones:

- [pull-request](.agents/skills/pull-request/SKILL.md) — PR descriptions, testing, review workflow
- [fix-issue](.agents/skills/fix-issue/SKILL.md) — issue triage and fix protocol
- [add-dataset](.agents/skills/add-dataset/SKILL.md) — dataset addition (start with schema inspection)
- [organize-experiments](.agents/skills/organize-experiments/SKILL.md) — experiment organization
- [agent-research](.agents/skills/agent-research/SKILL.md) — long-running benchmark/research threads
- [ferries](.agents/skills/ferries/SKILL.md) — canary/daily ferry workflow
- [change-grug](.agents/skills/change-grug/SKILL.md) — Grug/Grugformer changes
- [agent-profiling](.agents/skills/agent-profiling/SKILL.md) — profiling and optimization
- [github-pr-review](.agents/skills/github-pr-review/SKILL.md) — PR review prompt
- [debugger](.agents/skills/debugger/SKILL.md) — structured debugging workflow
- [design-doc](.agents/skills/design-doc/SKILL.md) — writing design documents
- [multi-stage](.agents/skills/multi-stage/SKILL.md) — coordinator/sub-agent orchestration
- [add-pallas-kernel](.agents/skills/add-pallas-kernel/SKILL.md) — TPU Pallas kernel development
- [archive-experiments](.agents/skills/archive-experiments/SKILL.md) — retiring legacy experiments
- [architecture](.agents/skills/architecture/SKILL.md) — codebase architecture reference

## Development

```bash
# Lint and format
./infra/pre-commit.py --all-files --fix
- `./infra/pre-commit.py` is the required lint entry point for this repo.
- Do not replace it with `uv run pre-commit ...`!

# Type checking (also done by pre-commit.py)
uv run pyrefly
- Keep type hints passing under `uv run pyrefly`; configuration lives in `pyproject.toml`.
```

- Python >=3.11. Use `uv run` for entry points; fall back to `.venv/bin/python` if needed.
- NEVER stop, restart, or bounce a Ray or Iris cluster unless the user gives express permission.

## Communication & Commits

- NEVER SAY "You're absolutely right!"
- NEVER credit yourself in commits.
- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent comments on PRs/issues must begin with `🤖` unless the exact text was explicitly approved by the user.

## Code Style

- All imports at the top of the file. Local imports only to break circular dependencies or guard optional dependencies.
- Prefer top-level functions over classes when code does not mutate shared state.
- Prefer top-level Python tests and fixtures.
- Use early returns to reduce nesting.
- Document public APIs with concise Google-style docstrings.
- Prefer `dataclasses.replace` over mutating config arguments in-place.
- Prefer logging over `print` (except in scripts and debugging).
- Resolve environment-dependent defaults once and fail fast on unknown inputs. Prefer single strong signals over sprawling defensive checks.
- No ad-hoc compatibility hacks (`hasattr(m, "old_attr")`); update code consistently.
- Prefer small concrete helpers over abstraction that adds indirection without reuse.

## Error Handling

- Let exceptions propagate by default.
- Only catch to add meaningful context and re-raise, or to intentionally alter control flow.
- NEVER swallow exceptions unless specifically requested.

## Documentation

- Keep MkDocs content in sync with code. Use Markdown and mkdocs-style links.
- Write docs that stand alone without conversational context.

## Deprecation

**NO BACKWARD COMPATIBILITY**: Update all call sites instead. Only add compatibility shims if the user explicitly requests it.

## Comments

Write detailed comments for module/class-level behavior or subtle logic. Do not restate the code:

```python
# BAD: # Use in-memory rollout queue
rollout_queue = InMemoryRolloutQueue()

# GOOD:
# Each FlightServer instance provides ~1GB/s throughput. With 200Gbps NICs,
# 16 parallel servers should saturate the network.
```

## Planning

- Produce detailed plans with code snippets. Ask questions up front instead of guessing.
- When a request is too large for one pass, capture a plan in `.agents/projects/` before pausing.

## Testing

- Always fix tests you broke. Do not relax tolerances or hack around failures.
- Prefer integration-style tests that validate externally-observable behavior.
- Do not create tautological tests (type exists, constant has value, etc.).
- Use pytest fixtures and parameterization to avoid duplication.
