# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What a converted task looks like on disk.

The verifier spec (``tasktrove_verify.spec``) lives at ``tests/verifier.toml``: Harbor copies only
``tests/`` into the container, and only at verify time, so the expected answer never reaches the
agent. ``tests/test.sh`` is the same three lines for every task. ``task.toml`` keeps Harbor's own
tables plus a ``[metadata]`` block with the selection tags. The task's own Dockerfile is kept and
gets the tool install appended.
"""

import hashlib
import re

import tomlkit
from tasktrove_verify.spec import Mode

VERIFIER_TOML = "tests/verifier.toml"
TESTS_MOUNT = "/tests"
"""Where Harbor mounts the task's ``tests/`` directory at grading time."""

VERIFY_TEST_SH = f"""#!/bin/bash
set -euo pipefail
exec tasktrove-verify {TESTS_MOUNT}/verifier.toml
"""

VERIFY_TOOL_URL = "git+https://github.com/marin-community/marin@{ref}#subdirectory=lib/tasktrove-verify"
TOOL_PYTHON = ">=3.11"
UV_IMAGE = "ghcr.io/astral-sh/uv:0.8"
INSTALL_MARKER = "# --- tasktrove-verify ---"
OLD_GRADER_LINE = re.compile(r"rewardkit|litellm", re.IGNORECASE)
"""A Dockerfile line installing the old judge graders; converters strip it and the graded step
rejects a task that still carries one."""


def drop_dockerfile_lines(dockerfile: str, pattern: re.Pattern[str]) -> str:
    """The Dockerfile without the lines ``pattern`` matches, ending in a newline."""
    kept = [line for line in dockerfile.splitlines() if not pattern.search(line)]
    return "\n".join(kept) + "\n"


# Which tool extras a mode needs installed in the task image. Modes absent here run on the core.
MODE_EXTRAS: dict[Mode, tuple[str, ...]] = {
    Mode.MATH: ("answer",),
    Mode.JSON_SCHEMA: ("schema",),
    Mode.REASONING_GYM: ("reasoning-gym",),
    Mode.JUDGE: ("judge",),
}

_BLANK_RUN = re.compile(r"\n{3,}")


def tool_install_block(tool_ref: str, extras: tuple[str, ...]) -> str:
    """The Dockerfile lines that install ``tasktrove-verify`` at ``tool_ref``.

    Every task with the same extras gets the same bytes, so images built from otherwise equal
    Dockerfiles still share layers. Git is installed only when the base image lacks it; every base
    image in the corpus is Debian or Ubuntu derived. The binary lands in ``/usr/local/bin`` rather
    than uv's default ``~/.local/bin``, which a login shell's ``/etc/profile`` drops from PATH. The
    tool needs Python 3.11; ``--python`` lets uv fetch a managed interpreter when the image's own is
    older, leaving the task's Python untouched.
    """
    package = "tasktrove-verify" + (f"[{','.join(extras)}]" if extras else "")
    url = VERIFY_TOOL_URL.format(ref=tool_ref)
    return (
        f"{INSTALL_MARKER}\n"
        "RUN command -v git >/dev/null || (apt-get update && apt-get install -y --no-install-recommends git"
        " && rm -rf /var/lib/apt/lists/*)\n"
        f"COPY --from={UV_IMAGE} /uv /usr/local/bin/uv\n"
        f'RUN UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --python "{TOOL_PYTHON}" "{package} @ {url}"\n'
    )


def edit_dockerfile(original: str, tool_ref: str, extras: tuple[str, ...]) -> str:
    """Normalize the task's Dockerfile and append the tool install.

    Normalization is limited to what never changes the image: trailing whitespace, runs of blank
    lines, and a single trailing newline. Two tasks whose Dockerfiles differ only in that get one
    image.
    """
    lines = [line.rstrip() for line in original.splitlines()]
    body = _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip("\n")
    return f"{body}\n\n{tool_install_block(tool_ref, extras)}"


def dockerfile_id(dockerfile: str) -> str:
    return hashlib.sha256(dockerfile.encode()).hexdigest()[:12]


def render_task_toml(agent_timeout: float, verifier_timeout: float, metadata: dict) -> str:
    """Harbor's task.toml: timeouts plus the selection metadata; no grading information."""
    doc = {
        "version": "1.0",
        "agent": {"timeout_sec": agent_timeout},
        "verifier": {"timeout_sec": verifier_timeout, "restart_environment": False},
        "metadata": metadata,
    }
    return tomlkit.dumps(doc)
