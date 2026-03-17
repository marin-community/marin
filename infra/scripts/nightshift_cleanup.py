# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nightshift cleanup: single agent picks a subproject, finds something meaty, opens a PR."""

import datetime
import subprocess

SUBPROJECTS = ["lib/marin/src/marin", "lib/iris/src/iris", "lib/zephyr/src/zephyr", "lib/levanter/src/levanter"]

CLEANUP_PROMPT = """\
You are the Nightshift Cleanup Agent.

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
Read `.agents/skills/pull-request/SKILL.md` before opening any PR.

## Rules of Engagement

- Only make changes you are confident are correct improvements.
- Do NOT refactor working code for style alone.
- Run `./infra/pre-commit.py --all-files --fix` before committing.
- Run relevant tests: `uv run pytest -x` on any test files you modified or
  that test modules you changed.
- If you find issues but the fix is non-trivial, file a GitHub issue instead
  of making a risky change.
- If you open a PR, follow `.agents/skills/pull-request/SKILL.md`: plain-text
  commit-style title/body, reference an issue with `Fixes #NNNN` or
  `Part of #NNNN`, and add the `agent-generated` and `nightshift` labels.
- Keep the PR focused: one coherent improvement.

## Output

Create a branch named `nightshift/cleanup-{date}` and open a PR with:
- A concise `[nightshift] ...` title that matches the pull-request skill
- A plain-text body that follows the pull-request skill and links the issue
- Labels: `agent-generated`, `nightshift`

If you find nothing worth changing, exit cleanly — no branch, no PR.
"""


def main() -> None:
    date = datetime.date.today().strftime("%Y%m%d")
    prompt = CLEANUP_PROMPT.format(
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
            "80",
            "--",
            prompt,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
