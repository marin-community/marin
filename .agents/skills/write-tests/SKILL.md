---
name: write-tests
description: Add, revise, or review Marin tests for an explicit behavior change, regression, or test-quality request.
---

# Write Tests

Read these before choosing test style, fixtures, mocks, markers, or commands:

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
5. For a bug, write the regression test before the fix and confirm it fails.
6. Keep test setup realistic but small. Use fixtures and parameterization to
   remove duplication.
7. Run the narrow test first, then the relevant package test command. Before a
   PR, run the repo lint entry point required by `AGENTS.md`.

## Default Commands

- Run a narrow test path while editing, then run
  `uv run --no-project infra/ci/run_tests.py`. Do not override the repository's
  default marker expression.
- For package-specific commands, use the relevant `lib/*/AGENTS.md` or testing
  doc.

Before a PR, run `./infra/pre-commit.py --changed-files --fix` or `--all-files`
as appropriate. Do not substitute `uv run pre-commit`.
