# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted initialization of an immutable C83 reference without source setup.

This standalone guest program uses only the standard library. It never derives
grading references from the final actor's workspace or dependency installation.
"""

import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath


def initialize(reference_path: Path, bundle_path: Path) -> None:
    reference = json.loads(reference_path.read_text())
    contract = reference["contract"]
    root, workspace = Path(contract["protected_root"]), Path(contract["workspace"])
    if root.exists():
        raise FileExistsError(root)
    if not workspace.is_dir() or any(workspace.iterdir()):
        raise ValueError("Reference-only initialization requires the original empty workspace")
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != reference["bundle_sha256"]:
        raise ValueError("Frozen reference changed before trusted initialization")
    verifyit = shutil.which("verifyit")
    if verifyit is None:
        raise FileNotFoundError("The original source Verifyit CLI is missing")
    root.mkdir(mode=0o700, parents=True)
    with tarfile.open(bundle_path) as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.isdev() or member.isfifo():
                raise ValueError(f"Unsafe frozen reference member: {member.name}")
        archive.extractall(root, filter="data")
    if json.loads((root / "contract.json").read_text()) != contract:
        raise ValueError("Installed frozen reference differs from its declared contract")
    if hashlib.sha256(Path(contract["harness"]).read_bytes()).hexdigest() != contract["harness_sha256"]:
        raise ValueError("Installed grading controller differs from its pinned source")
    for uid, name in ((contract["agent_uid"], "taskcompendium-agent"), (contract["grader_uid"], "taskcompendium-grade")):
        try:
            pwd.getpwuid(uid)
        except KeyError:
            subprocess.run(["/usr/sbin/useradd", "--create-home", "--uid", str(uid), "--user-group", name], check=True)
    bootstrap = Path(contract["protected_root"] + "-bootstrap")
    bootstrap.mkdir(mode=0o750)
    shutil.copy2(Path(contract["harness"]), bootstrap / "control.py")
    (bootstrap / "python").symlink_to(sys.executable)
    (bootstrap / "verifyit").symlink_to(verifyit)
    runtime = json.loads((root / "runtime.json").read_text())
    for site in runtime["sites"]:
        subprocess.run(["/usr/bin/chgrp", "-R", str(contract["agent_uid"]), site], check=True)
        subprocess.run(["/usr/bin/chmod", "-R", "g+rwX", site], check=True)
    os.chown(workspace, contract["agent_uid"], contract["agent_uid"])
    for directory in (root, bootstrap):
        for path in (directory, *directory.rglob("*")):
            if path.is_symlink():
                continue
            os.chown(path, 0, contract["grader_uid"])
            path.chmod(0o750 if path.is_dir() or os.access(path, os.X_OK) else 0o640)
    reference_path.unlink()
    bundle_path.unlink()


if __name__ == "__main__":
    initialize(Path(sys.argv[1]), Path(sys.argv[2]))
