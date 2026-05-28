# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared SSH helpers for GCP providers."""

from __future__ import annotations

from iris.rpc import config_pb2

# Instance metadata that opts a VM into OS Login. Applied at VM creation by
# the controller and worker-VM creation paths so `gcloud compute ssh` picks
# up the OS Login transport automatically — no per-call client dispatch.
OS_LOGIN_METADATA = {
    "enable-oslogin": "TRUE",
    "block-project-ssh-keys": "TRUE",
}


def ssh_impersonate_service_account(
    ssh_config: config_pb2.SshConfig | None,
) -> str | None:
    """Return the explicit impersonation SA from ssh_config, or None."""
    if ssh_config and ssh_config.impersonate_service_account:
        return ssh_config.impersonate_service_account
    return None
