# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Command construction for noninteractive GCE SSH."""


def gce_ssh_arguments(
    *,
    project: str,
    zone: str,
    instance: str,
    command: str,
    ssh_user: str | None = None,
    ssh_key_file: str | None = None,
    impersonate_service_account: str | None = None,
    connect_timeout: int | None = None,
) -> list[str]:
    """Return a noninteractive ``gcloud compute ssh`` command."""
    destination = f"{ssh_user}@{instance}" if ssh_user else instance
    arguments = [
        "gcloud",
        "compute",
        "ssh",
        destination,
        f"--project={project}",
        f"--zone={zone}",
    ]
    if ssh_key_file:
        arguments.append(f"--ssh-key-file={ssh_key_file}")
    if impersonate_service_account:
        arguments.append(f"--impersonate-service-account={impersonate_service_account}")
    arguments.extend(("--quiet", "--ssh-flag=-oBatchMode=yes"))
    if connect_timeout is not None:
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        arguments.append(f"--ssh-flag=-oConnectTimeout={connect_timeout}")
    arguments.extend(("--command", command))
    return arguments
