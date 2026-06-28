# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run provenance: who/when/which-commit/which-argv produced an artifact.

A single :class:`Provenance` value captured once in the launching process and recorded on every
artifact built by that invocation, so an artifact answers "what produced me" without the executor.
"""

import getpass
import os
import subprocess
import sys
from datetime import datetime

from pydantic import BaseModel


def _git(*args: str) -> str | None:
    """Run a read-only ``git`` command at the cwd, or ``None`` outside a checkout/on failure."""
    if not os.path.exists(".git"):
        return None
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


class Provenance(BaseModel):
    """Who/when/where an artifact was built, captured in the launching process.

    Nests inside :class:`~marin.execution.artifact.ArtifactRecord`. Every field is best-effort:
    outside a git checkout the git fields are ``None``.
    """

    git_remote: str | None = None
    git_commit: str | None = None
    git_branch: str | None = None
    user: str | None = None
    created_at: str = ""
    """ISO-8601 timestamp of the launching invocation (shared by every step it builds)."""
    command_line: list[str] = []

    @classmethod
    def capture(cls) -> "Provenance":
        """Snapshot the current launch: git remote/commit/branch, OS user, time, and argv."""
        return cls(
            git_remote=_git("remote", "get-url", "origin"),
            git_commit=_git("rev-parse", "HEAD"),
            git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
            user=_current_user(),
            created_at=datetime.now().isoformat(),
            command_line=list(sys.argv),
        )


def _current_user() -> str | None:
    # getpass.getuser() raises when no username can be resolved from the environment or the
    # password database; provenance is best-effort metadata, so a missing user is not an error.
    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return None
