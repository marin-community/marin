# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Activate services on existing GCE VMs without replacing the VM."""

import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rigging.timing import ExponentialBackoff, retry_with_backoff

DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_RETRY_INTERVAL = 5


@dataclass(frozen=True)
class GceVmTarget:
    """Connection coordinates for one existing GCE VM."""

    project: str
    zone: str
    instance: str
    user: str | None = None
    ssh_key_file: Path | None = None
    impersonate_service_account: str | None = None
    tunnel_through_iap: bool = False


class StartupScriptPersistence(StrEnum):
    """When an activated script becomes the VM's reboot-time startup script."""

    BEFORE_ACTIVATION = "before_activation"
    AFTER_SUCCESS = "after_success"


def _identity_args(target: GceVmTarget) -> list[str]:
    arguments: list[str] = []
    if target.impersonate_service_account:
        arguments.append(f"--impersonate-service-account={target.impersonate_service_account}")
    return arguments


def ssh_arguments(
    target: GceVmTarget,
    command: str,
    *,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
) -> list[str]:
    """Return the noninteractive ``gcloud compute ssh`` arguments for ``target``."""
    destination = f"{target.user}@{target.instance}" if target.user else target.instance
    arguments = [
        "gcloud",
        "compute",
        "ssh",
        destination,
        f"--project={target.project}",
        f"--zone={target.zone}",
        *_identity_args(target),
        "--quiet",
        "--ssh-flag=-oBatchMode=yes",
        f"--ssh-flag=-oConnectTimeout={connect_timeout}",
    ]
    if target.ssh_key_file is not None:
        arguments.append(f"--ssh-key-file={target.ssh_key_file}")
    if target.tunnel_through_iap:
        arguments.append("--tunnel-through-iap")
    arguments.extend(("--command", command))
    return arguments


def run_remote(
    target: GceVmTarget,
    command: str,
    *,
    stdin: str | None = None,
    timeout: int,
    connect_timeout: int = DEFAULT_CONNECT_TIMEOUT,
    attempts: int = 1,
    retry_interval: float = DEFAULT_RETRY_INTERVAL,
) -> subprocess.CompletedProcess[str]:
    """Run one command on ``target`` and stream its output to the caller."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    arguments = ssh_arguments(target, command, connect_timeout=connect_timeout)

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            arguments,
            input=stdin,
            text=True,
            check=True,
            timeout=timeout,
        )

    if attempts == 1:
        return run()
    if retry_interval <= 0:
        raise ValueError("retry_interval must be positive")
    return retry_with_backoff(
        run,
        retryable=lambda error: isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)),
        max_attempts=attempts,
        backoff=ExponentialBackoff(
            initial=retry_interval,
            maximum=retry_interval,
            factor=1.0,
            jitter=0,
        ),
        operation=f"SSH command on {target.instance}",
    )


def set_startup_script(target: GceVmTarget, script: str, *, timeout: int) -> None:
    """Persist ``script`` as the VM's GCE startup-script metadata."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh") as script_file:
        script_file.write(script)
        script_file.flush()
        subprocess.run(
            [
                "gcloud",
                "compute",
                "instances",
                "add-metadata",
                target.instance,
                f"--project={target.project}",
                f"--zone={target.zone}",
                *_identity_args(target),
                f"--metadata-from-file=startup-script={script_file.name}",
                "--quiet",
            ],
            text=True,
            check=True,
            timeout=timeout,
        )


def activate_startup_script(
    target: GceVmTarget,
    script: str,
    *,
    persistence: StartupScriptPersistence,
    timeout: int,
    attempts: int = 1,
    retry_interval: float = DEFAULT_RETRY_INTERVAL,
) -> None:
    """Run ``script`` over SSH and persist it according to ``persistence``.

    ``BEFORE_ACTIVATION`` makes an interrupted rollout reboot into the intended
    script. ``AFTER_SUCCESS`` leaves the last successful script installed when
    the candidate fails. The caller owns service health checks and rollback.
    """
    if persistence is StartupScriptPersistence.BEFORE_ACTIVATION:
        set_startup_script(target, script, timeout=timeout)

    run_remote(
        target,
        "bash -s",
        stdin=script,
        timeout=timeout,
        attempts=attempts,
        retry_interval=retry_interval,
    )

    if persistence is StartupScriptPersistence.AFTER_SUCCESS:
        set_startup_script(target, script, timeout=timeout)
