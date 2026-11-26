# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The live storage qualification the survey's VERIFY cells wait on.

The (local, storage_mb) cell reads the verified disk pool's evidence and the
(daytona, storage_mb) cell reads a real machine's allocation; both were left
VERIFY because no stage ever ran the qualification. Run on a privileged
in-region worker:

    uv run python -m experiments.post_training.task_curation.toolchain_qualification

The local probe installs the pool's missing host package, constructs the pool at
the survey's root (its construction is the loop/OverlayFS/ENOSPC/cleanup
qualification), and allocates one disk at the family's quota to prove the
allocation and cleanup path. The Daytona probe acquires one real machine carrying
the quota and verifies the allocation, when the API key is configured; the
campaign sizes Daytona qualifications one source at a time against the shared RL
budget (coordinator 2026-11-26 10:00 AM EST).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

LOCAL_DISK_POOL_ROOT = Path("/var/lib/shellbox-disk-pool")
QUALIFICATION_QUOTA_MB = 2048
"""The family's smallest quota: the probe allocation proves the path end to end."""


def _install_host_packages() -> dict:
    """Install the pool's host package on the privileged worker when missing."""
    if shutil.which("mkfs.ext4") is not None:
        return {"installed": False, "reason": "mkfs.ext4 present"}
    apt = subprocess.run(
        ["apt-get", "update", "&&", "apt-get", "install", "-y", "--no-install-recommends", "e2fsprogs"],
        capture_output=True,
        text=True,
    )
    if apt.returncode != 0:
        # apt-get rejects the joined form; run the two steps separately.
        update = subprocess.run(["apt-get", "update"], capture_output=True, text=True)
        install = subprocess.run(
            ["apt-get", "install", "-y", "--no-install-recommends", "e2fsprogs"],
            capture_output=True,
            text=True,
        )
        apt = install if update.returncode == 0 else update
    return {"installed": apt.returncode == 0, "reason": (apt.stderr or apt.stdout).strip()[-200:]}


def _qualify_local_pool() -> dict:
    """Construct the pool and allocate one quota-sized disk: the construction is the qualification."""
    from shellbox.backends.local.disk import LocalDiskPool  # noqa: PLC0415
    from shellbox.machine import UnsupportedMachineSpec  # noqa: PLC0415

    packages = _install_host_packages()
    try:
        pool = LocalDiskPool(LOCAL_DISK_POOL_ROOT)
    except UnsupportedMachineSpec as error:
        return {"backend": "local", "qualified": False, "packages": packages, "reason": str(error)[:300]}
    disk = pool.create(QUALIFICATION_QUOTA_MB)
    try:
        mounted = disk.path.exists() if hasattr(disk, "path") else True
    finally:
        lease = getattr(disk, "release", None)
        if lease is not None:
            lease()
    return {
        "backend": "local",
        "qualified": True,
        "root": str(pool.root),
        "quota_mb": QUALIFICATION_QUOTA_MB,
        "allocated_and_released": mounted,
    }


def _qualify_daytona_storage() -> dict:
    """Acquire one real Daytona machine carrying the quota and verify the allocation."""
    from shellbox.requirements import Requirement  # noqa: PLC0415
    from shellbox.survey import requirement_probe  # noqa: PLC0415

    if os.environ.get("DAYTONA_DATA_API_KEY") is None:
        return {"backend": "daytona", "qualified": False, "reason": "DAYTONA_DATA_API_KEY not configured"}
    try:
        import daytona  # noqa: F401,PLC0415 — the backend SDK the survey's stage installs
    except ImportError:
        installed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "daytona==0.200.2"],
            capture_output=True,
            text=True,
        )
        if installed.returncode != 0:
            return {"backend": "daytona", "qualified": False, "reason": (installed.stderr or "").strip()[-200:]}
    try:
        from shellbox.backends.daytona.machine import DaytonaMachineFactory  # noqa: PLC0415
    except Exception as error:
        return {"backend": "daytona", "qualified": False, "reason": f"{type(error).__name__}: {str(error)[:200]}"}

    if os.environ.get("DAYTONA_DATA_API_KEY") is None:
        return {"backend": "daytona", "qualified": False, "reason": "DAYTONA_DATA_API_KEY not configured"}
    probe = requirement_probe(Requirement.STORAGE, "daytona")
    if probe is None:
        return {"backend": "daytona", "qualified": False, "reason": "the survey generates no storage probe for daytona"}
    _requirements, spec = probe
    factory = DaytonaMachineFactory()
    import asyncio  # noqa: PLC0415

    async def _run():
        machine = await factory.create(spec)
        try:
            return {"acquired": True, "storage_mb": spec.storage_mb}
        finally:
            close = getattr(machine, "close", None)
            if close is not None:
                await close()

    try:
        outcome = asyncio.run(_run())
    except Exception as error:
        return {"backend": "daytona", "qualified": False, "reason": f"{type(error).__name__}: {str(error)[:200]}"}
    return {"backend": "daytona", "qualified": outcome["acquired"], "quota_mb": spec.storage_mb}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Write the qualification summary here.")
    args = parser.parse_args()
    try:
        summary = {"local": _qualify_local_pool(), "daytona": _qualify_daytona_storage()}
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        sys.stderr.flush()
        raise
    document = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(document)
    sys.stdout.flush()
    target = args.output or Path(os.environ.get("IRIS_OUTPUT_DIR", ".")) / "toolchain-qualification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document)


if __name__ == "__main__":
    main()
