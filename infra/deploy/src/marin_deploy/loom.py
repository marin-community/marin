# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Activate the Pulumi-managed Loom host over the shared GCE SSH path."""

import json
import os
from pathlib import Path
from urllib.request import urlopen

from rigging.timing import ExponentialBackoff, retry_with_backoff

from marin_deploy.gce import GceVmTarget, run_remote

SSH_CONNECT_TIMEOUT = 15
RESTART_TIMEOUT = 600
READY_ATTEMPTS = 90
READY_INTERVAL = 10
READY_REQUEST_TIMEOUT = 10
STARTUP_RESTART_COMMAND = """
set -euo pipefail
sudo rm -f /run/loom-startup-succeeded
sudo systemctl restart google-startup-scripts.service
sudo test -f /run/loom-startup-succeeded
""".strip()


class LoomNotReadyError(RuntimeError):
    """The Loom readiness endpoint has not reported ready yet."""


def _require_ready(url: str) -> None:
    with urlopen(url, timeout=READY_REQUEST_TIMEOUT) as response:  # noqa: S310
        status = json.load(response).get("status")
    if status != "ready":
        raise LoomNotReadyError(f"Loom readiness status is {status!r}")


def activate_loom(target: GceVmTarget, readiness_url: str) -> None:
    """Restart Loom through its startup unit and wait for public readiness."""
    if target.ssh_key_file is None:
        raise ValueError("Loom activation requires an explicit Compute Engine SSH key")
    public_key = Path(f"{target.ssh_key_file}.pub")
    if not target.ssh_key_file.is_file() or not public_key.is_file():
        raise FileNotFoundError(
            f"no Compute Engine SSH key at {target.ssh_key_file}; run "
            f"'gcloud compute ssh {target.instance} --zone={target.zone} --project={target.project}' once"
        )

    run_remote(
        target,
        STARTUP_RESTART_COMMAND,
        timeout=RESTART_TIMEOUT,
        connect_timeout=SSH_CONNECT_TIMEOUT,
    )
    retry_with_backoff(
        lambda: _require_ready(readiness_url),
        retryable=lambda error: isinstance(error, (OSError, json.JSONDecodeError, LoomNotReadyError)),
        max_attempts=READY_ATTEMPTS,
        backoff=ExponentialBackoff(
            initial=READY_INTERVAL,
            maximum=READY_INTERVAL,
            factor=1.0,
            jitter=0,
        ),
        operation="Loom readiness probe",
    )


def main() -> None:
    project = os.environ["LOOM_PROJECT"]
    zone = os.environ["LOOM_ZONE"]
    instance = os.environ["LOOM_INSTANCE"]
    domain = os.environ["LOOM_DOMAIN"]
    activate_loom(
        GceVmTarget(
            project=project,
            zone=zone,
            instance=instance,
            ssh_key_file=Path.home() / ".ssh" / "google_compute_engine",
        ),
        f"https://{domain}/api/ready",
    )


if __name__ == "__main__":
    main()
