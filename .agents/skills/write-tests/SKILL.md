---
name: write-tests
description: Write, fix, or review Marin tests for behavioral regression coverage and pytest quality.
---

# Write Tests

Read root `AGENTS.md` and `TESTING.md`. For files under `lib/*`, also read the
nearest module `AGENTS.md` and its referenced testing guide; those documents
own local commands, markers, fakes, mocks, optional dependencies, and numerical
tolerances.

1. Find existing coverage for the behavior before adding a file.
2. State the observable behavior that should fail when the code is wrong.
3. Exercise it through a public API, structured output, persisted state, or real
   side effect. Avoid implementation-detail and tautological assertions.
4. For a bug, prefer a regression test that fails before the fix.
5. Keep setup realistic and small; use fixtures or parameterization only when
   they remove meaningful duplication.
6. Run the narrow test, then the relevant package command. The safe branch-wide
   default is `uv run --no-project infra/ci/run_tests.py`; do not override its
   marker expression.
7. Before a PR, run `./infra/pre-commit.py --changed-files --fix`.

Do not substitute `uv run pre-commit` for the repository lint entry point.
