# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared SSH and OS Login helpers for GCP providers."""

from __future__ import annotations

import os

from iris.rpc import config_pb2

OS_LOGIN_METADATA = {
    "enable-oslogin": "TRUE",
    "block-project-ssh-keys": "TRUE",
}


def uses_os_login(ssh_config: config_pb2.SshConfig | None) -> bool:
    if ssh_config is None:
        return False
    return ssh_config.auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN


def ssh_key_file(ssh_config: config_pb2.SshConfig | None) -> str | None:
    if ssh_config and ssh_config.key_file:
        return ssh_config.key_file
    if uses_os_login(ssh_config):
        return os.path.expanduser("~/.ssh/google_compute_engine")
    return None


def ssh_impersonate_service_account(
    ssh_config: config_pb2.SshConfig | None,
    service_account: str | None = None,
) -> str | None:
    if ssh_config and ssh_config.impersonate_service_account:
        return ssh_config.impersonate_service_account
    if service_account:
        return service_account
    return None
