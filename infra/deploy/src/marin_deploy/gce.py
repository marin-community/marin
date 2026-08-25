# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Activate services on existing GCE VMs without replacing the VM."""

import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from rigging.gce import gce_ssh_arguments
from rigging.timing import ExponentialBackoff, retry_with_backoff

DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_RETRY_INTERVAL = 5


@dataclass(frozen=True)
class GceVmTarget:
    """Connection coordinates for one existing GCE VM."""

    project: str
    zone: str
    instance: str
    ssh_key_file: Path | None = None
    impersonate_service_account: str | None = None


class StartupScriptPersistence(StrEnum):
    """When an activated script becomes the VM's reboot-time startup script."""

    BEFORE_ACTIVATION = "before_activation"
    AFTER_SUCCESS = "after_success"


def _identity_args(target: GceVmTarget) -> list[str]:
    arguments: list[str] = []
    if target.impersonate_service_account:
        arguments.append(f"--impersonate-service-account={target.impersonate_service_account}")
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
    """Run one command on ``target`` over noninteractive GCE SSH."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    arguments = gce_ssh_arguments(
        project=target.project,
        zone=target.zone,
        instance=target.instance,
        command=command,
        ssh_key_file=str(target.ssh_key_file) if target.ssh_key_file is not None else None,
        impersonate_service_account=target.impersonate_service_account,
        connect_timeout=connect_timeout,
    )

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
