# Agent Guidelines for Marin

Start with the shared practices below. Consult subproject manuals for directory-specific guidance:

- `lib/levanter/AGENTS.md` — Levanter (JAX training library)
- `lib/marin/AGENTS.md` — Marin (pipeline framework)
- `lib/iris/AGENTS.md` — Iris (job orchestration)
- `lib/zephyr/AGENTS.md` — Zephyr (dataset processing)
- `lib/fray/AGENTS.md` — Fray (distributed execution)

## Operational Guides

For debugging and operating live infrastructure, read the relevant OPS.md:

- `lib/iris/OPS.md` — cluster lifecycle, job/task management, profiling, SQL queries, GCP/CoreWeave operations
- `lib/zephyr/OPS.md` — pipeline debugging, straggler diagnosis, coordinator queries, diagnostic patterns

Zephyr OPS.md references Iris OPS.md for shared infrastructure commands — read Iris first when debugging zephyr jobs on Iris.

## Workflow Playbooks

Skills are task-focused playbooks in `.agents/skills/` (also accessible as
`.claude/skills/`). **Before starting any non-trivial task, check whether a
matching skill exists** by scanning the skill descriptions in your system
prompt. If a skill matches, invoke it via the Skill tool — do not skip it in
favor of ad-hoc commands.

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
- NEVER stop, restart, or bounce an Iris cluster unless the user gives express permission.
- In general, never read or write large amounts of data across GCS regions or to the open internet; storage and bandwidth are major cost drivers for this project.
- do not use storage transfer service to move files from one region to another unless the user says "I personally will write grants for Percy to pay for this"

## Communication & Commits

- NEVER SAY "You're absolutely right!"
- NEVER credit yourself in commits.
- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent comments on PRs/issues must begin with `🤖` unless the exact text was explicitly approved by the user.
- When using `gh` to inspect issues or PRs, prefer `--json <fields>` or explicit narrow flags such as `--comments`; avoid plain `gh issue view` / `gh pr view`, which can fail on this repo because GitHub classic project fields are deprecated.

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

Dependency direction: {`iris`, `haliax`} → {`levanter`, `zephyr`} → `marin`. Each layer may only import from layers to its left. Never introduce reverse dependencies (e.g., levanter importing from marin).

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
