# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop privileges before a source grader executes candidate code (container only)."""

import os
import sys
from pathlib import Path

CANDIDATE_UID = 65534


def main() -> None:
    if os.getuid() != 0 or not Path("/input/specification.json").is_file():
        raise RuntimeError("Candidate launcher requires the TaskCompendium container supervisor")
    argv = sys.argv[1:]
    if not argv:
        raise ValueError("Candidate command is required")
    tests = os.environ.get("TASKTROVE_TESTS_DIR")
    if tests:
        directory = Path(tests)
        directory.parent.chmod(0o711)
        directory.chmod(0o555)
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("Verifier resources must not be symlinks")
            path.chmod(0o555 if path.is_dir() else 0o444)
    # Source pytest restore creates root-owned tests after the snapshot copy.
    # Make those readable to the test process without giving it write access.
    for path in Path(os.environ["TASKCOMPENDIUM_WORKSPACE"]).rglob("*"):
        if path.is_symlink():
            continue
        metadata = path.stat()
        if metadata.st_uid == 0:
            path.chmod(0o555 if path.is_dir() or metadata.st_mode & 0o111 else 0o444)
    for argument in argv:
        if argument.startswith("--json-report-file="):
            Path(argument.split("=", 1)[1]).parent.chmod(0o777)
    logs = os.environ.get("TASKTROVE_LOGS_DIR")
    if logs:
        Path(logs).chmod(0o777)
    home = Path("/tmp/taskcompendium-home")
    home.mkdir(exist_ok=True)
    os.chown(home, CANDIDATE_UID, CANDIDATE_UID)
    os.setgroups([])
    os.setgid(CANDIDATE_UID)
    os.setuid(CANDIDATE_UID)
    os.environ["HOME"] = str(home)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
