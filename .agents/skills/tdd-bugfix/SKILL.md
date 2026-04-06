---
name: tdd-bugfix
description: Fix a bug using test-driven development. Analyze the problem, write a failing test first, fix the code, validate, then commit. Use for in-branch bug fixes where the issue is already understood. For bugs originating from a GitHub issue, prefer fix-issue instead.
---

# Skill: TDD Bug Fix

Fix a bug by writing the test first, then the fix. Every phase must complete
before advancing to the next.

## Prerequisites

Read first:

@AGENTS.md

## Phase 1: Analyze

Understand the bug before touching any code.

1. **Reproduce mentally** — read the reported symptoms, stack traces, and any
   linked code. Identify the module and call chain involved.
2. **Locate root cause** — trace the failure to the specific function/line. Read
   surrounding code to understand invariants.
3. **State the bug in one sentence** — e.g. "X returns Y when it should return
   Z because condition C is not checked." If you cannot write this sentence, you
   do not understand the bug yet; keep reading.
4. **Find the right test file** — search `tests/` for existing tests covering
   the affected module. You will add your test there. Only create a new test
   file if no relevant file exists.

Write a brief analysis to the user before proceeding:
- Root cause (1-2 sentences)
- Affected code (file:line links)
- Which test file you will use

## Phase 2: Write a Failing Test

Write a minimal test that fails because of the bug and will pass once fixed.

Rules:
- **Minimal infrastructure** — use the simplest fixture setup that triggers the
  bug. No unnecessary mocks; no elaborate scenarios. The test should be
  understandable in isolation.
- **Test the observable behavior**, not the implementation detail. Assert on
  return values, raised exceptions, or externally visible side effects.
- **Name it clearly** — `test_<function>_<scenario>` or
  `test_<module>_<bug_description>`.
- **One test function** unless the bug has genuinely distinct failure modes.
  Prefer `pytest.mark.parametrize` over copy-pasted variants.
- Follow existing file conventions (fixtures, imports, style).

Run the test and confirm it **fails** for the expected reason:

```bash
uv run pytest <test_file>::<test_name> -xvs --timeout=60
```

If the test passes, your test does not reproduce the bug — revise it. If it
fails for an unrelated reason, fix the test setup.

## Phase 3: Fix the Bug

Now — and only now — fix the production code.

- Make the **smallest change** that fixes the bug. Do not refactor, do not clean
  up adjacent code, do not add features.
- If the fix requires changing a public API, flag it to the user before
  proceeding.

Run the failing test again and confirm it **passes**:

```bash
uv run pytest <test_file>::<test_name> -xvs
```

Then run the broader test suite for the affected module to check for regressions:

```bash
uv run pytest <test_file> -x --timeout=120
```

If other tests break, fix them before proceeding. PR CI will catch broader
regressions beyond this file.

## Phase 4: Commit and Push

Delegate to @.agents/skills/commit/SKILL.md for lint, staging, and push.

Use this commit message format:

```
[<scope>] Fix <concise description>

<Root cause in one sentence. Reference the test that now guards against
regression.>
```
