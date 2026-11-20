# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NixOS host-program dependencies for the local backend."""

from pathlib import Path, PurePosixPath

from shellbox.machine import Mount

SYSTEM_PROFILE = Path("/run/current-system/sw")
STORE = Path("/nix/store")


def system_programs() -> tuple[tuple[Mount, ...], tuple[Path, ...]]:
    """Return the read-only Nix store/profile mounts and program directory."""
    if not SYSTEM_PROFILE.is_dir():
        return (), ()
    mounts = (
        Mount(STORE, PurePosixPath(STORE)),
        Mount(SYSTEM_PROFILE.resolve(), PurePosixPath(SYSTEM_PROFILE)),
    )
    return mounts, (SYSTEM_PROFILE / "bin",)
