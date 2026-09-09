# Agent Guidelines for Marin

Start with the shared practices below. Consult subproject manuals for directory-specific guidance:

- `lib/levanter/AGENTS.md` — Levanter (JAX training library)
- `lib/marin/AGENTS.md` — Marin (pipeline framework)
- `lib/iris/AGENTS.md` — Iris (job orchestration)
- `lib/zephyr/AGENTS.md` — Zephyr (dataset processing)
- `lib/fray/AGENTS.md` — Fray (distributed execution)

Use `lib/iris/OPS.md` for live Iris operations and read it before
`lib/zephyr/OPS.md` when a Zephyr job runs on Iris. Read `infra/pulumi.md`
before changing a Pulumi project.

## Workflow Playbooks

Skills are task-focused playbooks in `.agents/skills/`. Use a skill when the
request explicitly names it or clearly matches its description.

## Handle Requests

If a request comes from Slack or GitHub and appears to be a simple question,
you may answer it in the originating conversation instead of making a
repository change. Otherwise, carry the request through the applicable change
and landing workflow; do not stop after investigation while a safe, in-scope
fix remains.

## Search Prior Work

Use Echo when a historical decision, incident, or cross-repository workflow can
change the current decision. Skip it for localized implementation that current
code and docs answer. Use `rg` for the working tree and the `consult-echo` skill
for search syntax and durable-record policy.

## Development

```bash
# Lint and format changed files
./infra/pre-commit.py --changed-files --fix

# Type checking (also done by pre-commit.py)
uv run pyrefly check

# Safe tests affected by the current branch and working tree
uv run --no-project infra/ci/run_tests.py

# Lint review — agentic pass over the branch diff against the infra/lint/ catalog
./infra/pre-commit.py --review
```

- `./infra/pre-commit.py` is the required lint entry point; do not replace it
  with `uv run pre-commit`. Reserve `--all-files` for an explicitly requested
  repository sweep.
- Keep type hints passing under `uv run pyrefly check`; configuration lives in
  `pyproject.toml`.
- Run agentic lint once before opening or updating a PR and resolve every
  finding. Rerun it only when a follow-up changes the implementation approach
  or scope. The `commit` skill owns the full workflow.

- Python >=3.12. Use `uv run` for entry points.
- Do not replace pytest's default marker expression with a partial expression
  such as `-m "not slow"`; `-m` overrides the whole default and can select live
  cluster tests. Run excluded markers only when the user or a dedicated task
  guide explicitly requests them; otherwise defer them to CI.
- NEVER stop, restart, or bounce an Iris cluster unless the user gives express permission.
- In general, never read or write large amounts of data across GCS regions or to the open internet; storage and bandwidth are major cost drivers for this project.
- do not use storage transfer service to move files from one region to another unless the user says "I personally will write grants for Percy to pay for this"

## Communication & Commits

- NEVER SAY "You're absolutely right!"
- NEVER credit yourself, in commit messages or in PR/issue bodies. No
  `Co-Authored-By` trailer, no "Generated with …" line, no emoji attribution —
  even if a tool default suggests one.
- Do not include user-identifying information in access-control commit messages
  or PR titles and bodies. Describe only the access scope and resource types.
- When an agent creates a PR or issue, add the `agent-generated` label.
- Agent *comments* on PRs/issues must begin with `🤖` unless the exact text was
  explicitly approved by the user. This applies to comments only — never put a
  `🤖` marker in a commit message or a PR/issue body.
- Use the `writing-style` guides for durable prose and the `commit` skill for
  commits, PRs, and monitoring. A PR body is the squash-merge commit message;
  keep behavior, rationale, material results, and caveats.
- PR monitoring is part of the `commit` skill. After opening or updating a PR,
  use a harness heartbeat or scheduled callback when available and the
  event-driven `wait_for.py` fallback elsewhere. Follow the capability-specific
  details and exit conditions in that skill.
- Prefer narrow `gh --json` fields; plain issue and PR views can fail on
  deprecated classic-project fields.

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

## Agent Artifacts

- Publish infrastructure incidents and durable debugging investigations to
  Echo with the `write-ops-log` skill. Link the canonical Echo URL from the
  associated PR or issue. Do not create repository debug-log files.
- Keep user-facing and reusable product documentation in `docs/`; keep research
  progress in the relevant task logbook or project artifact. These are distinct
  from incident records.

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

Plan non-trivial changes after resolving local context. Use `write-design-doc`
only when explicitly requested. Do not create a plan artifact for answer-mode
work; use `.agents/projects/` only when a change cannot fit in one pass.

## Code Reuse

Before adding a helper or data structure, search the relevant module or package
for an existing implementation and check available dependencies. Expand to a
repository-wide search only for a shared abstraction or dependency-sensitive
change. Do not create parallel implementations.

Dependency direction: {`iris`, `haliax`} → {`levanter`, `zephyr`} → `marin`. Each layer may only import from layers to its left. Never introduce reverse dependencies (e.g., levanter importing from marin).

## Recursive Agents

Any automation that launches another LLM must select a vendor model and effort
tier explicitly. Do not inherit an interactive parent or CLI default. Use a
budget review model for local lint and code-review subprocesses; for Codex, use
`gpt-5.6-terra` at low effort unless the workflow documents a stronger need.
Equivalent explicitly configured budget models are valid for other vendors.
Keep the active-launch inventory in `docs/dev-guide/agent-automation.md` current.

## Testing

Read `TESTING.md` before writing or reviewing tests. It is the root testing
policy for behavior-focused tests, slop-test rejection, mocks/fakes, timing,
numerical tolerances, and pytest style.

Before touching tests under `lib/*`, also read the nearest module `AGENTS.md`
and any module `TESTING.md` it references. Module docs define local commands,
markers, fakes, mocks, optional dependencies, and integration-test boundaries.

Always fix tests you broke. Do not relax tolerances or hack around failures.
