# Agent Guidelines for Marin

Start with the shared practices below. Consult subproject manuals for directory-specific guidance:

- `lib/levanter/AGENTS.md` — Levanter (JAX training library)
- `lib/marin/AGENTS.md` — Marin (pipeline framework)
- `lib/iris/AGENTS.md` — Iris (job orchestration)

## Workflow Playbooks

Agent-friendly recipes live in `docs/recipes/`. Key ones:

- [pull-request.md](docs/recipes/pull-request.md) — PR descriptions, testing, review workflow
- [fix_issue.md](docs/recipes/fix_issue.md) — issue triage and fix protocol
- [add_dataset.md](docs/recipes/add_dataset.md) — dataset addition (start with schema inspection)
- [organize_experiments.md](docs/recipes/organize_experiments.md) — experiment organization
- [agent_research.md](docs/recipes/agent_research.md) — long-running benchmark/research threads
- [ferries.md](docs/recipes/ferries.md) — canary/daily ferry workflow
- [change_grug.md](docs/recipes/change_grug.md) — Grug/Grugformer changes
- [agent_profiling.md](docs/recipes/agent_profiling.md) — profiling and optimization

## Development

```bash
# Lint and format
./infra/pre-commit.py --all-files --fix

# Type checking
uv run pyrefly
```

- Python >=3.11. Use `uv run` for entry points; fall back to `.venv/bin/python` if needed.
- NEVER stop, restart, or bounce a Ray cluster unless the user gives express permission.

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
