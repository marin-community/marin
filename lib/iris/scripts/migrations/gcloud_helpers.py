# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared gcloud helpers for Iris migration scripts."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_cmd(
    *args: str,
    timeout: float = 120,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with logging."""
    cmd = list(args)
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout, check=check)


def discover_controller_vm(project: str, label_prefix: str) -> tuple[str, str] | None:
    """Find controller VM by its Iris label. Returns (vm_name, zone) or None."""
    r = run_cmd(
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter=labels.iris-{label_prefix}-controller=true",
        "--format=json(name,zone)",
        timeout=30,
    )
    instances = json.loads(r.stdout)
    if not instances:
        return None
    vm = instances[0]
    zone = vm["zone"].rsplit("/", 1)[-1]
    return vm["name"], zone


def scp_from_controller(vm: str, zone: str, project: str, remote_path: str, local_path: Path) -> None:
    """SCP a file from the controller VM to a local path."""
    run_cmd(
        "gcloud",
        "compute",
        "scp",
        f"{vm}:{remote_path}",
        str(local_path),
        f"--zone={zone}",
        f"--project={project}",
        timeout=600,
    )


def scp_to_controller(local_path: Path, vm: str, zone: str, project: str, remote_path: str) -> None:
    """SCP a local file to the controller VM."""
    run_cmd(
        "gcloud",
        "compute",
        "scp",
        str(local_path),
        f"{vm}:{remote_path}",
        f"--zone={zone}",
        f"--project={project}",
        timeout=600,
    )


def gcloud_ssh(vm: str, zone: str, project: str, command: str, timeout: float = 30) -> str:
    """Run a command on a VM via gcloud SSH and return stdout."""
    r = run_cmd(
        "gcloud",
        "compute",
        "ssh",
        vm,
        f"--zone={zone}",
        f"--project={project}",
        f"--command={command}",
        timeout=timeout,
    )
    return r.stdout.strip()
