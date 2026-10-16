# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute each SWE-smith instance's fail-to-pass restore payload from a bare clone of its swesmith/<repo> mirror.

The payload is exactly what the upstream verifier's 'git checkout HEAD~1' changes: files that differ between the task
branch and its parent, at their parent contents, plus files the parent does not have (deleted).
"""

import subprocess
from pathlib import Path

MIRROR = "https://github.com/swesmith/{name}"


def mirror(name: str, root: Path) -> Path:
    path = root / f"{name}.git"
    if not path.exists():
        subprocess.run(["git", "clone", "-q", "--bare", MIRROR.format(name=name), str(path)], check=True)
    return path


def git(repository: Path, *args: str) -> bytes:
    return subprocess.run(["git", "--git-dir", str(repository), *args], check=True, capture_output=True).stdout


def restore_payload(repository: Path, instance: str) -> tuple[dict[str, bytes], list[str]]:
    """Return ({path: contents at <instance>~1}, [paths absent at <instance>~1]) for paths the last commit changed."""
    branch = f"refs/heads/{instance}"
    subjects = git(repository, "log", "--format=%s", "-3", branch).decode().splitlines()
    if subjects[:2] != ["Remove F2P Tests", "Bug Patch"]:
        raise ValueError(f"{instance}: unexpected history {subjects}")
    files, deleted = {}, []
    for line in git(repository, "diff", "--no-renames", "--name-status", "-z", branch, f"{branch}~1").split(b"\0"):
        if not line:
            continue
        files_or_status = line.decode()
        if files_or_status in ("A", "M", "D", "T"):
            status = files_or_status
            continue
        path = files_or_status
        if status == "D":
            deleted.append(path)
        else:
            files[path] = git(repository, "show", f"{branch}~1:{path}")
    if not files and not deleted:
        raise ValueError(f"{instance}: the last commit changed nothing")
    return files, deleted


def clean_only_lines(repository: Path, instance: str, minimum: int = 25) -> list[str]:
    """Lines the bug patch removed from source files (present only in the clean code), for leak probes."""
    branch = f"refs/heads/{instance}"
    diff = git(repository, "diff", "--no-renames", "-U0", f"{branch}~2", f"{branch}~1").decode(errors="replace")
    removed = {line[1:].strip() for line in diff.splitlines() if line.startswith("-") and not line.startswith("---")}
    added = {line[1:].strip() for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")}
    candidates = sorted(line for line in removed - added if len(line) >= minimum)
    # Drop lines that also occur somewhere in the agent's (bugged) tree; finding those would not be a leak.
    present = set()
    for line in candidates:
        found = subprocess.run(
            ["git", "--git-dir", str(repository), "grep", "-qF", "-e", line, branch], capture_output=True
        )
        if found.returncode == 0:
            present.add(line)
    return [line for line in candidates if line not in present]
