# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run provenance: who/when/which-commit/which-argv produced an artifact.

Small, dependency-free helpers so an artifact record can capture provenance without
pulling in the executor.
"""

import os
import subprocess
import sys
from datetime import datetime


def get_command_line() -> list[str]:
    """The launching process's command line (``sys.argv``), recorded as provenance."""
    return list(sys.argv)


def get_git_commit() -> str | None:
    """The current ``HEAD`` commit, or ``None`` outside a git checkout."""
    if os.path.exists(".git"):
        return os.popen("git rev-parse HEAD").read().strip()
    return None


def get_user() -> str | None:
    """The OS user running the build."""
    return subprocess.check_output("whoami", shell=True).strip().decode("utf-8")


def created_now() -> str:
    """An ISO-8601 timestamp for the current moment."""
    return datetime.now().isoformat()
