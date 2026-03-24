# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightshift cleanup: single agent picks a subproject, finds something meaty, opens a PR."""

import datetime
import secrets
import subprocess

SUBPROJECTS = ["lib/marin/src/marin", "lib/iris/src/iris", "lib/zephyr/src/zephyr", "lib/levanter/src/levanter"]

CLEANUP_PROMPT = """\
You are the Nightshift Cleanup Agent.

Your random seed is: {haiku_seed}
Use this seed to compose a haiku about code maintenance. Include it
as the epigraph of your PR description.

## Your Mission

Pick one of these subprojects to focus on: {subprojects}

Browse the subproject and find something **meaty** to improve — not cosmetic
lint, not renaming, but a genuine code quality win. Look for things like:

- Dead code: unused functions, stale TODO/FIXME (>90 days via `git blame`),
  commented-out blocks
- Duplicated logic that should use an existing helper from elsewhere in the repo
- Copy-paste with slight variation that should be unified
- Overly defensive error handling (try/except that swallows exceptions,
  redundant None checks)
- Parameter sprawl, stringly-typed code, leaky abstractions
- Unnecessary work: redundant computations, repeated file reads, N+1 patterns
- Test issues: tests with no assertions, `time.sleep()` in tests, mocks of
  internal functions

Read `AGENTS.md` for project conventions.

## Rules of Engagement

- Only make changes you are confident are correct improvements.
- Do NOT refactor working code for style alone.
- Run `./infra/pre-commit.py --all-files --fix` before committing.
- Run relevant tests: `uv run pytest -x` on any test files you modified or
  that test modules you changed.
- If you find issues but the fix is non-trivial, file a GitHub issue instead
  of making a risky change.
- Keep the PR focused: one coherent improvement.

## Output

Create a branch named `nightshift/cleanup-{date}` and open a PR with:
- Title: `[nightshift] <concise description>`
- Body: your haiku, then a summary of what was cleaned and why
- Labels: `agent-generated`, `nightshift`

If you find nothing worth changing, exit cleanly — no branch, no PR.
"""


def main() -> None:
    date = datetime.date.today().strftime("%Y%m%d")
    prompt = CLEANUP_PROMPT.format(
        haiku_seed=secrets.token_hex(4),
        subprojects=", ".join(SUBPROJECTS),
        date=date,
    )

    subprocess.run(
        [
            "claude",
            "--model=opus",
            "--print",
            "--dangerously-skip-permissions",
            "--tools=Read,Write,Edit,Glob,Grep,Bash",
            "--max-turns",
            "800",
            "--",
            prompt,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
