# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild SWE-smith Harbor packages so the agent cannot recover the fix from git (issue #10212).

Upstream packages (DCAgent/swesmith-sandboxes-with_tests-oracle_verified_120s) check out the task branch in the
agent's image. That branch's history is 'Initial commit' (clean code) -> 'Bug Patch' -> 'Remove F2P Tests', and the
clone keeps a local main branch, every other instance branch and the GitHub remote, so 'git diff HEAD main' prints
the fix. The upstream verifier relies on that history: test.sh runs 'git checkout <branch>; git checkout HEAD~1' to
bring back the fail-to-pass tests the last commit removed.

The fixed package:
- environment/Dockerfile: after the checkout, replace .git with a fresh repository holding one commit of the task
  tree, one branch (main), no remote and no reflog.
- tests/f2p_restore/: the files that 'git checkout HEAD~1' would change (their HEAD~1 contents) and the paths it would
  delete. The verifier copies them in; the agent never sees them because Harbor uploads tests/ only after the agent
  phase.
- tests/test.sh: the two history checkouts replaced by that restore step. Everything else is unchanged.
"""

import io
import json
import re
import tarfile

STRIP_HISTORY = (
    " && rm -rf .git"
    " && git init -q"
    " && git symbolic-ref HEAD refs/heads/main"
    " && git add -A"
    ' && git -c user.name=swesmith -c user.email=swesmith@localhost commit -q -m "Initial commit"'
    " && git reflog expire --expire=now --all"
    " && git gc -q --prune=now"
)
RESTORE = (
    "cd /testbed && cp -a /tests/f2p_restore/files/. /testbed/"
    " && (cd /testbed && xargs -r -d '\\n' rm -f -- < /tests/f2p_restore/deleted.txt); "
)


def read_package(task_binary: bytes) -> dict[str, bytes]:
    """Return {relative path: bytes} for every regular file in one gzipped package tarball."""
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as archive:
        return {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}


def instance_id(package: dict[str, bytes]) -> str:
    return json.loads(package["tests/config.json"])["instance_id"]


def base_image(package: dict[str, bytes]) -> str:
    return re.search(r"^FROM\s+(\S+)", package["environment/Dockerfile"].decode(), re.MULTILINE)[1]


def fixed_dockerfile(dockerfile: str, instance: str) -> str:
    checkout = f"RUN git fetch && git checkout {instance}"
    lines = dockerfile.splitlines()
    if lines.count(checkout) != 1:
        raise ValueError(f"Unexpected Dockerfile checkout for {instance}")
    return "\n".join(line + STRIP_HISTORY if line == checkout else line for line in lines) + "\n"


def fixed_test_sh(test_sh: str, instance: str) -> str:
    checkouts = f"git checkout {instance}; git checkout HEAD~1; "
    if test_sh.count(checkouts) != 1:
        raise ValueError(f"Unexpected test.sh history checkout for {instance}")
    return test_sh.replace(checkouts, RESTORE)


def fixed_package(package: dict[str, bytes], restore_files: dict[str, bytes], deleted: list[str]) -> dict[str, bytes]:
    """Return the fixed package's files; `restore_files` are /testbed-relative paths at the parent commit."""
    instance = instance_id(package)
    fixed = dict(package)
    fixed["environment/Dockerfile"] = fixed_dockerfile(package["environment/Dockerfile"].decode(), instance).encode()
    fixed["tests/test.sh"] = fixed_test_sh(package["tests/test.sh"].decode(), instance).encode()
    for path, data in restore_files.items():
        fixed[f"tests/f2p_restore/files/{path}"] = data
    fixed["tests/f2p_restore/deleted.txt"] = "".join(f"{path}\n" for path in deleted).encode()
    return fixed


def write_package(files: dict[str, bytes]) -> bytes:
    """Gzipped tarball in the upstream layout (executable bit kept for shell scripts)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(files):
            info = tarfile.TarInfo(path)
            info.size = len(files[path])
            info.mode = 0o755 if path.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(files[path]))
    return buffer.getvalue()
