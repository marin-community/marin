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

- All imports at the top of the file. No local imports except to break circular dependencies or guard optional deps. No `TYPE_CHECKING` guards — fix cycles structurally via protocols.
- Prefer top-level functions over classes when code does not mutate shared state. Reduce deep inheritance hierarchies.
- Use early returns to reduce nesting.
- Document public APIs with concise Google-style docstrings. Skip docstrings on trivial functions with clear names.
- Prefer `dataclasses.replace` over mutating config arguments in-place.
- Prefer logging over `print` (except in scripts and debugging).
- Resolve environment-dependent defaults once and fail fast on unknown inputs.
- No ad-hoc compatibility hacks (`hasattr(m, "old_attr")`); update code consistently.
- Prefer small concrete helpers over abstraction that adds indirection without reuse. Start simple; abstract only under real pressure.
- Delete dead code: unused parameters, stale options, old experiments.
- Top-level constants for magic strings/numbers.
- Separate computation from I/O (split compute from upload/write).
- Use context managers for resource lifecycle.

## Naming

- No `*_utils.py` — use descriptive names like `text_cleaning.py`.
- Function names should reflect return types (`probe_task` → `task_status`).
- No `_s` suffix for seconds (assumed in this codebase). No abbreviations like `exe` — use `exec` or full words.

## Types & Data Structures

- Dataclass/namedtuple over raw dicts. `StrEnum` over string keys.
- Use `Protocol` for decoupling; avoid hard-coupling to concrete types.
- Avoid `X | str` unions that require `isinstance` checks — pick one input type.
- Replace compound booleans encoding state with an enum.

## Configuration

- No `default_*` wrappers that obscure underlying mechanisms.
- Force explicit specification of critical parameters (no silent defaults).
- Centralize defaults in one canonical location.
- Prefer explicit constructor/config parameters over env vars.
- Composition over inheritance: embed sub-configs, don't subclass.

## API Design

- Accept only what's necessary. Replace boolean flags with meaningful parameters (e.g., `num_workers: int` instead of `parallel: bool`).
- Use separate classes over boolean flags for variant behavior (`NativeVllm` / `DockerVllm`, not `Vllm(docker=True)`).
- Normalize inputs to a standard format once at the boundary, not throughout.

## Error Handling

- Let exceptions propagate by default.
- Only catch to add meaningful context and re-raise, or to intentionally alter control flow.
- NEVER swallow exceptions unless specifically requested.
- Assert liberally; prefer `raise ValueError` over silent fallbacks.

## Documentation

- Keep MkDocs content in sync with code. Use Markdown and mkdocs-style links.
- Write docs that stand alone without conversational context.

## Deprecation

**NO BACKWARD COMPATIBILITY**: Update all call sites instead. Only add compatibility shims if the user explicitly requests it.

## Comments

- Write comments for module/class-level behavior or subtle logic. Do not restate the code.
- Delete stale comments immediately on discovery.
- Inline comments to clarify non-obvious boolean arguments.

## File I/O

- Assert absolute paths at boundaries.
- Atomic writes: write to a temp file, then rename.

## Performance / Distributed

- Consider coordinator bottlenecks: N workers × M operations can overwhelm head nodes.
- Overlap I/O with computation using thread pools and queues.

## LLM-Generated Code Pitfalls

Watch for and eliminate these patterns in generated code:
- Over-protective try/except and defensive None checks
- Tautological tests (type exists, constant has value)
- Verbose/redundant docstrings and `__all__` in `__init__.py`
- Boolean dispatch instead of separate classes
- Environment variables instead of explicit parameters

## Planning

- Produce detailed plans with code snippets. Ask questions up front instead of guessing.
- When a request is too large for one pass, capture a plan in `.agents/projects/` before pausing.

## Code Reuse

Before writing any utility function, helper, or data structure:
1. Search the codebase for existing implementations
2. Check subproject utils: `lib/marin/src/marin/`, `lib/iris/src/iris/`, `lib/levanter/`
3. Check `pyproject.toml` for available third-party packages before adding new ones

If a suitable implementation exists, use it. Do not create parallel implementations.

## Testing

- Always fix tests you broke. Do not relax tolerances or hack around failures.
- Prefer integration-style tests that validate externally-observable behavior.
- Do not write tautological tests: tests must fail if behavior is wrong, not just if implementation changes.
- Use pytest fixtures and parameterization to avoid duplication.
- Prefer top-level `def test_*` with fixtures over test classes.
- Search for existing test files before creating new ones. Extend existing files first.
- No mocks unless testing I/O boundaries (network, filesystem). Test against real behavior.
- No `time.sleep()` in tests — inject `now=time.time()` or mock time instead.
- Mock at boundaries (e.g., wandb), not internal logger output.
