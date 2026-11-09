# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move new CoreWeave shim and gVisor logs onto the host's NVMe disk.

CoreWeave provides /etc/containerd/runsc.toml with log paths under /var/log/runsc.
Abandoned runtime logs can fill the small RAM-backed root filesystem and prevent
container startup and exec. The node-agent DaemonSet runs this module in an init
container with the host config directory mounted at /host-containerd and the NVMe
disk at /mnt/local, so replacement nodes receive the same log-path correction.

Only the shim log_path and runsc debug-log destinations change. Setup preserves
the provider's runtime options, backs up the original file on NVMe, and replaces
it atomically. New shims read the updated config without a runtime restart;
existing shims keep their previous log paths. Historical logs require separate
cleanup.
"""

import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import tomllib
from pathlib import Path

CONFIG_PATH = Path("/host-containerd/runsc.toml")
logger = logging.getLogger(__name__)


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


def configure(config_path: Path, nvme: Path) -> None:
    """Redirect new shims' logs without restarting runtimes or moving open files."""
    if not config_path.exists():
        logger.info("No runsc configuration at %s; skipping runtime log setup", config_path)
        return
    if config_path.is_symlink() or not stat.S_ISREG(config_path.stat().st_mode):
        raise ValueError("runsc config must be a regular file")
    # Both paths are host mounts; the init container's root is an overlay filesystem.
    if nvme.stat().st_dev == config_path.parent.stat().st_dev:
        raise ValueError("NVMe must be backed by a different filesystem from the host config directory")
    log_root = nvme / "iris-runtime/runsc"
    source = config_path.read_text()
    updated = redirected_config(source, log_root)
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
    logger.info("Configured runsc logs at %s; previous configuration saved at %s", log_root, backup)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    configure(CONFIG_PATH, Path("/mnt/local"))
