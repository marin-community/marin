# Agent Guidelines for Marin

## How to Use This Guide

- Start with the shared practices below; if you discover missing guidance, expand this document so the next agent benefits.
- When you uncover directory-specific guidance, add it to the relevant subproject manual so the next agent stays aligned.
- Consult the subproject manuals when working in submodule trees:
  * `lib/levanter/AGENTS.md` for Levanter-specific conventions.
  * `lib/marin/AGENTS.md` for Marin-specific conventions
- When a recipe exists, follow it—the agent-friendly playbooks live in `docs/recipes/`. Some live in the individual `lib/*/docs` directories.

## Shared Workflow Playbooks

- Begin with the agent-friendly recipes in `docs/recipes/`.
- The first step for dataset addition is schema inspection. See the [add_dataset.md](docs/recipes/add_dataset.md) recipe for details.
- You can help organize experiments using the [organize_experiments.md](docs/recipes/organize_experiments.md) recipe.
- When making significant changes to Grug/Grugformer, follow [change_grug.md](docs/recipes/change_grug.md).
- Follow the rules and examples in each recipe to ensure compatibility and automation-friendliness.

## Shared Coding Practices

### Tooling

- Assume Python >=3.11.
- Always use `uv run` for Python entry points. If that fails, try `.venv/bin/python` directly.
- Run `uv run python infra/pre-commit.py --all-files` before sending changes; formatting and linting are enforced with `ruff`.
- Keep type hints passing under `uv run pyrefly`; configuration lives in `pyproject.toml`.
- If `uv.lock` gets merge conflicts, the quickest fix is usually to delete `uv.lock` and regenerate it with `uv sync --all-packages --group test`.

### Communication & Commits

- NEVER SAY "You're absolutely right!"
- You never credit yourself in commits.
- NEVER EVER EVER credit yourself in commit messages.

### Code Style

- Put all imports at the top of the file. Avoid local imports unless technically necessary (for example, to break circular dependencies or guard optional dependencies).
- Prefer top-level functions when code does not mutate shared state; use classes to encapsulate data when that improves clarity.
- Prefer top-level Python tests and fixtures.
- Separation of responsibilities: when a change introduces a new subsystem (e.g., serving/inference, data access, evaluation), encapsulate lifecycle/configuration in a dedicated module and have callers depend on the interface rather than re-implementing setup/teardown details.
- Disprefer internal mutation of function arguments, especially config dataclasses; prefer returning a modified copy (e.g., via `dataclasses.replace`) so call sites remain predictable and side effects are explicit.
- Use early returns (`if not x: return None`) when they reduce nesting.
- Do not introduce ad-hoc compatibility hacks like `hasattr(m, "old_attr")`; update the code consistently instead.
- Do not use `from future import ...` statements.
- Document public APIs with concise Google-style docstrings.

### Error Handling

- Let exceptions propagate by default.
- Only catch exceptions when you can add meaningful context and re-raise, or when you are intentionally altering control flow.
- NEVER EVER SWALLOW EXCEPTIONS unless specifically requested by the user.

### Documentation

- Keep MkDocs content in sync with code. Docs live in `docs/` or in the subproject's `docs/` directory; use Markdown and mkdocs-style links when referencing symbols.
- Public-facing modules and APIs need concise Google-style docstrings; align terminology across code and docs.

### Deprecation

- Unless specifically requested, do not introduce deprecation or fallback paths—update all call sites instead.

## Comments

You write detailed comments when appropriate to describe code behavior as a
whole, e.g. at the module or class level, or when describing some subtle
behavior.

You don't generate comments that merely restate the code, e.g.

<bad>
     # Use in-memory rollout queue
    rollout_queue = InMemoryRolloutQueue()
</bad>

<good>
# We have found that each instance of a FlightServer can provide approximately 1GB/s
# of throughput. As our typical VMs run with 200Gbps NICs, running 16 parallel servers
# should be sufficient to saturate the network.
</good>

## Planning

- When planning, you produce detailed plans including code snippets.
- You ask questions up front when building a plan instead of guessing.
- When a request feels too large for one pass, capture a plan (for example in `.agents/projects/` when the subproject provides one) before pausing.

## Testing

- Always fix tests if you broke them.
- Do not fix tests by relaxing tolerances or hacking around them.
- Avoid “tautological” tests that merely restate implementation logic as asserts; prefer tests that validate externally-observable behavior, integration points, or realistic failure modes.
- Run the appropriate tests for your changes (for example, `uv run pytest` under the relevant directory); consult subproject guides for preferred markers.
- Use pytest features like fixtures and parameterization to avoid duplication and write clean code.

PREFER:

- Integration style tests which exercise behavior and test the output

DO NOT:

- Create tests which validate obvious features: if a type exists, a constant has a value, etc.


## Environment

- Prefer to use `uv` when possible. If you can't (for instance, due to sandbox restrictions) you can use `.venv/bin/python`

> This file will be expanded as agent workflows and best practices evolve.
