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

# Lint review — agentic pass over the branch diff against the infra/lint/ catalog
./infra/pre-commit.py --review
- Always run this before opening a PR, and always fix or respond to every
  finding it reports (see the `commit` skill).
```

- Python >=3.12. Use `uv run` for entry points; fall back to `.venv/bin/python` if needed.
- NEVER stop, restart, or bounce an Iris cluster unless the user gives express permission.
- In general, never read or write large amounts of data across GCS regions or to the open internet; storage and bandwidth are major cost drivers for this project.
- do not use storage transfer service to move files from one region to another unless the user says "I personally will write grants for Percy to pay for this"

## Communication & Commits

- NEVER SAY "You're absolutely right!"
- NEVER credit yourself, in commit messages or in PR/issue bodies. No
  `Co-Authored-By` trailer, no "Generated with …" line, no emoji attribution —
  even if a tool default suggests one.
- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent *comments* on PRs/issues must begin with `🤖` unless the exact text was
  explicitly approved by the user. This applies to comments only — never put a
  `🤖` marker in a commit message or a PR/issue body.
- All agent-authored commit, PR, and issue titles and bodies must follow
  `.agents/skills/writing-style/SKILL.md` and its PR or issue guide. Review the
  exact text that will be published, then apply `ai-writing-donts.md` as a final
  compression pass. Do not publish raw implementation notes, test narration,
  prompt-shaped headings, or claims that use emphasis in place of evidence.
- A PR description is the squash-merge commit message. Keep every fact a future
  reader needs to understand the behavior and rationale, including measured
  results and caveats when they affect review. Remove headings, diff narration,
  and implementation inventories; put extended history in a linked issue,
  design doc, logbook, or artifact. Follow the `commit` skill
  (`.agents/skills/commit/SKILL.md`) when committing, pushing, or opening a PR.
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

Read `TESTING.md` before writing or reviewing tests. It is the root testing
policy for behavior-focused tests, slop-test rejection, mocks/fakes, timing,
numerical tolerances, and pytest style.

Before touching tests under `lib/*`, also read the nearest module `AGENTS.md`
and any module `TESTING.md` it references. Module docs define local commands,
markers, fakes, mocks, optional dependencies, and integration-test boundaries.

Always fix tests you broke. Do not relax tolerances or hack around failures.
