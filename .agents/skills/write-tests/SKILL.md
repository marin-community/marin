---
name: write-tests
description: Write or revise Marin tests with an emphasis on behavior, regression coverage, pytest style, and avoiding "slop tests." Use when adding tests, fixing failing tests, reviewing test quality, or deciding what test would catch a bug.
---

# Write Tests

Use this skill when a change needs tests or when existing tests look too coupled
to implementation details.

Read root `TESTING.md` before writing or reviewing non-trivial tests. It is the
shared project-wide testing policy used by `write-tests`, `review-pr`, and
`commit`.

Also read the relevant module-specific docs before choosing test style,
fixtures, mocks, markers, or commands:

- root `AGENTS.md`
- root `TESTING.md`
- the nearest subproject `AGENTS.md` for files under `lib/*`
- package testing docs referenced from that `AGENTS.md`, such as
  `lib/iris/TESTING.md`

## Workflow

1. Find existing tests for the touched behavior before creating a new file.
2. Check module-specific testing rules for commands, markers, fakes, mocks, and
   numerical tolerances.
3. Name the behavior that should fail if the code is wrong.
4. Write the smallest test that observes that behavior through a public API,
   structured output, persisted state, or real side effect.
5. Prefer a regression test before the fix when fixing a bug. Ensure the test fails before implementing the bug fix.
6. Keep test setup realistic but small. Use fixtures and parameterization to
   remove duplication.
7. Run the narrow test first, then the relevant package test command. Before a
   PR, run the repo lint entry point required by `AGENTS.md`.

## Default Commands

- Root or mixed changes: `uv run pytest` over touched test directories. Do not
  override the repository's default marker expression; it excludes tests that
  require slow, integration, live-cluster, Docker, or manual execution.
- For package-specific commands, use the relevant `lib/*/AGENTS.md` or testing
  doc.

For PR preparation, run `./infra/pre-commit.py --changed-files --fix` or
`./infra/pre-commit.py --all-files --fix` as appropriate. Do not replace it
with `uv run pre-commit`.
