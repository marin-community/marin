# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Confirm intent before `pulumi up` touches a guarded stack from a non-`main` branch.

Some stacks (production clusters and shared GCP grants) should normally be updated only after
the change has merged to `main`. This is a soft guard, not a gate: on a `pulumi up` against a
guarded stack from another branch it asks for confirmation on the controlling terminal, and
proceeds unprompted when there is no terminal to ask on (e.g. `--yes` in CI). `pulumi preview`
is never affected.
"""

import subprocess
from collections.abc import Callable

MAIN_BRANCH = "main"


def current_git_branch() -> str | None:
    """Return the current git branch, or None when it can't be named (detached HEAD, no git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    branch = result.stdout.strip()
    if not branch or branch == "HEAD":  # empty output or detached HEAD
        return None
    return branch


def confirm_on_tty(prompt: str) -> bool | None:
    """Ask a yes/no question on the controlling terminal.

    Returns True/False for the answer, or None when there is no terminal to ask on. Under
    `pulumi up` the program's stdin is not the terminal, so this reads `/dev/tty` directly
    rather than `input()`.
    """
    try:
        with open("/dev/tty", "r+") as tty:
            tty.write(prompt)
            tty.flush()
            answer = tty.readline()
    except OSError:
        return None
    return answer.strip().lower() in ("y", "yes")


def guard_pulumi_up(
    *,
    stack: str,
    branch: str | None,
    is_preview: bool,
    main_only_stacks: frozenset[str],
    confirm: Callable[[str], bool | None],
    warn: Callable[[str], None],
) -> None:
    """Abort a non-`main` `pulumi up` on a guarded stack unless the operator confirms.

    No-op on preview, on unguarded stacks, and on the `main` branch. Raises SystemExit when the
    operator declines. Proceeds with a warning when no terminal is available to ask on, so this
    never blocks non-interactive automation.
    """
    if is_preview or stack not in main_only_stacks or branch == MAIN_BRANCH:
        return
    where = f"branch {branch!r}" if branch else "a detached HEAD"
    decision = confirm(
        f"You are about to run `pulumi up` on stack {stack!r} from {where}, not {MAIN_BRANCH!r}.\n"
        f"Production stacks should be deployed from {MAIN_BRANCH!r} after merging. Continue anyway? [y/N] "
    )
    if decision is None:
        warn(f"`pulumi up` on {stack!r} from {where} (no terminal to confirm on; proceeding).")
        return
    if not decision:
        raise SystemExit(f"Aborted `pulumi up` on {stack!r}: not on the {MAIN_BRANCH!r} branch.")
