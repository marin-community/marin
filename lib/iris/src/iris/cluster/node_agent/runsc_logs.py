# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Maintain CoreWeave runsc logs on a host (Python 3.12+, no Iris dependencies).

Run this file with host Python, including through a node-debug pod's chroot.
The default is a dry run; neither command restarts containerd or sandboxes.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

LOG_ROOT = Path("/mnt/local/iris-runtime/runsc")
LEGACY_LOG_ROOT = Path("/var/log/runsc")
CONFIG_PATH = Path("/etc/containerd/runsc.toml")
RUNTIME_ID = re.compile(r"[a-f0-9]{64}")
MIN_AGE = 24 * 60 * 60


def redirected_config(source: str, log_root: Path) -> str:
    """Redirect only the shim and runsc log destinations, preserving other options."""
    before = tomllib.loads(source)
    expected = {**before, "log_path": f"{log_root}/%ID%/shim.log"}
    expected["runsc_config"] = {**before["runsc_config"], "debug-log": f"{log_root}/%ID%/gvisor.%COMMAND%.log"}
    updated = source
    for key, value in [("log_path", expected["log_path"]), ("debug-log", expected["runsc_config"]["debug-log"])]:
        updated, count = re.subn(
            rf"(?m)^([ \t]*{key}[ \t]*=[ \t]*)[^\n]*$",
            lambda match, value=value: match[1] + json.dumps(value),
            updated,
        )
        if count != 1:
            raise ValueError(f"Expected one {key} setting in runsc config; found {count}")
    if tomllib.loads(updated) != expected:
        raise ValueError("Refusing to change unrelated runtime configuration")
    return updated


def configure(config_path: Path, log_root: Path, *, apply: bool) -> None:
    """Move future runtime logs to NVMe without moving files open by existing shims."""
    if config_path.is_symlink() or not stat.S_ISREG(config_path.stat().st_mode):
        raise ValueError("runsc config must be a regular file")
    # DirectoryOrCreate on a missing NVMe mount would recreate the original failure on root.
    nvme = Path("/mnt/local")
    if nvme.stat().st_dev == Path("/").stat().st_dev:
        raise ValueError("/mnt/local must be backed by a different filesystem from root")
    if not log_root.is_relative_to(nvme) or ".." in log_root.parts:
        raise ValueError("runsc log directory must be under /mnt/local")
    source = config_path.read_text()
    updated = redirected_config(source, log_root)
    print(f"runsc log destination: {log_root}; config_changed={source != updated}; apply={apply}", flush=True)
    if not apply:
        return
    log_root.mkdir(parents=True, exist_ok=True)
    if log_root.resolve().stat().st_dev != nvme.stat().st_dev:
        raise ValueError("runsc log directory must reside on the NVMe filesystem")
    if source == updated:
        return
    backup = log_root.parent / f"runsc.toml.{hashlib.sha256(source.encode()).hexdigest()[:12]}.bak"
    backup.write_text(source)
    metadata = config_path.stat()
    with tempfile.NamedTemporaryFile(mode="w", dir=config_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            os.fchown(temporary.fileno(), metadata.st_uid, metadata.st_gid)
            os.fchmod(temporary.fileno(), stat.S_IMODE(metadata.st_mode))
            temporary.write(updated)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path.replace(config_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    print(f"config backup: {backup}", flush=True)


def protected_runtime_ids(log_roots: tuple[Path, ...], proc: Path = Path("/proc")) -> set[str]:
    """Protect CRI sandboxes/containers, running process IDs, and open log files."""
    known: set[str] = set()
    for args, key in [("pods", "items"), ("ps", "containers")]:
        command = ["crictl", args, *(["-a"] if args == "ps" else []), "-o", "json"]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        items = json.loads(result.stdout)[key]
        if not items:
            raise ValueError("Refusing cleanup without populated CRI sandbox and container inventories")
        known.update(item["id"] for item in items)
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            command = (process / "cmdline").read_bytes().decode(errors="replace")
            known.update(re.findall(r"(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])", command))
            for descriptor in (process / "fd").iterdir():
                try:
                    target = Path(os.readlink(descriptor))
                except FileNotFoundError:
                    continue
                for root in log_roots:
                    if target.is_relative_to(root):
                        known.add(target.relative_to(root).parts[0])
        except (FileNotFoundError, ProcessLookupError):
            # Processes and descriptors may disappear while enumerating /proc.
            continue
    return known


def prune(log_roots: tuple[Path, ...], known: set[str], *, cutoff: float, apply: bool) -> tuple[int, int]:
    """Remove old orphan log directories; leave active, recent, and unknown entries."""
    directories = allocated_bytes = 0
    for root in log_roots:
        if root.is_symlink():
            raise ValueError(f"Refusing a symlink log root: {root}")
        if not root.exists():
            continue
        for directory in root.iterdir():
            if directory.is_symlink() or not directory.is_dir() or not RUNTIME_ID.fullmatch(directory.name):
                continue
            if directory.name in known:
                continue
            metadata = [directory.stat(), *(path.lstat() for path in directory.rglob("*"))]
            if max(entry.st_mtime for entry in metadata) >= cutoff:
                continue
            directories += 1
            allocated_bytes += sum(entry.st_blocks * 512 for entry in metadata)
            if apply:
                shutil.rmtree(directory)
    return directories, allocated_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("configure", "prune"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--min-age", type=int, default=MIN_AGE, help="Minimum orphan log age in seconds (at least 3600)")
    args = parser.parse_args()
    if args.command == "configure":
        configure(CONFIG_PATH, LOG_ROOT, apply=args.apply)
        return
    if args.min_age < 3600:
        parser.error("--min-age must be at least 3600 seconds")
    roots = (LEGACY_LOG_ROOT, LOG_ROOT)
    known = protected_runtime_ids(roots)
    directories, allocated = prune(roots, known, cutoff=time.time() - args.min_age, apply=args.apply)
    print(json.dumps({"directories": directories, "allocated_bytes": allocated, "apply": args.apply}))
    print(f"root_filesystem={shutil.disk_usage('/')}", flush=True)


if __name__ == "__main__":
    main()
