# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NixOS host-program dependencies for the local backend."""

import subprocess
from pathlib import Path, PurePosixPath

from shellbox.machine import Mount

SYSTEM_PROFILE = Path("/run/current-system/sw")
QUERY_TIMEOUT = 30


def system_programs() -> tuple[tuple[Mount, ...], tuple[Path, ...]]:
    """Return read-only program mounts and their command directories on NixOS."""
    if not SYSTEM_PROFILE.is_dir():
        return (), ()
    profile = SYSTEM_PROFILE.resolve()
    roots = [profile]
    # Managed executables use FHS loaders that may sit outside the program profile's closure.
    for directory in ("/lib", "/lib32", "/lib64", "/libx32"):
        for loader in Path(directory).glob("ld*.so*"):
            target = loader.resolve()
            if target.is_relative_to("/nix/store"):
                roots.append(target)
    closure = subprocess.run(
        [str(profile / "bin/nix-store"), "--query", "--requisites", *map(str, roots)],
        check=True,
        capture_output=True,
        text=True,
        timeout=QUERY_TIMEOUT,
    ).stdout.splitlines()
    mounts = (
        *(Mount(Path(path), PurePosixPath(path)) for path in closure),
        Mount(profile, PurePosixPath(SYSTEM_PROFILE)),
    )
    return mounts, (SYSTEM_PROFILE / "bin",)
