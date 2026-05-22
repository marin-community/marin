# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared SSH and OS Login helpers for GCP providers."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

from iris.rpc import config_pb2

logger = logging.getLogger(__name__)

OS_LOGIN_METADATA = {
    "enable-oslogin": "TRUE",
    "block-project-ssh-keys": "TRUE",
}


def uses_os_login(ssh_config: config_pb2.SshConfig | None) -> bool:
    if ssh_config is None:
        return False
    return ssh_config.auth_mode == config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN


class OsLoginKeyProvisioner:
    """Lazily provisions an SSH keypair and registers it with OS Login.

    Thread-safe. The keypair is generated once on disk and persists across
    restarts. The OS Login registration has a TTL and is re-registered
    when expired.
    """

    _TTL = 3600  # 1 hour
    _REFRESH_MARGIN = 300  # re-register 5 min before expiry

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._registration_expiry: float | None = None

    def ensure_key(self, key_path: str, impersonate_sa: str | None) -> None:
        if os.path.exists(key_path) and not self._needs_registration():
            return
        with self._lock:
            self._generate_keypair(key_path)
            if self._needs_registration():
                self._register_os_login(key_path + ".pub", impersonate_sa)

    def _needs_registration(self) -> bool:
        return self._registration_expiry is None or time.monotonic() >= self._registration_expiry

    def _generate_keypair(self, key_path: str) -> None:
        if os.path.exists(key_path):
            return
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        logger.info("Generating SSH keypair at %s", key_path)
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", "", "-q"],
            check=True,
        )

    def _register_os_login(self, pub_key_path: str, impersonate_sa: str | None) -> None:
        self._purge_stale_keys(impersonate_sa)
        cmd = [
            "gcloud",
            "compute",
            "os-login",
            "ssh-keys",
            "add",
            f"--key-file={pub_key_path}",
            f"--ttl={self._TTL}s",
        ]
        if impersonate_sa:
            cmd.append(f"--impersonate-service-account={impersonate_sa}")
        logger.info("Registering SSH key with OS Login (ttl=%ds, sa=%s)", self._TTL, impersonate_sa)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to register SSH key with OS Login: {result.stderr.strip()}")
        self._registration_expiry = time.monotonic() + self._TTL - self._REFRESH_MARGIN

    def _purge_stale_keys(self, impersonate_sa: str | None) -> None:
        """Remove expired and no-expiration SSH keys from the OS Login profile.

        The profile has a 32 KiB limit. Old keys registered without --ttl have
        expirationTimeUsec=0 and accumulate forever, so we remove those too.
        Keys with a future expiration are left alone.
        """
        list_cmd = [
            "gcloud",
            "compute",
            "os-login",
            "ssh-keys",
            "list",
            "--format=value(fingerprint,expirationTimeUsec)",
        ]
        if impersonate_sa:
            list_cmd.append(f"--impersonate-service-account={impersonate_sa}")
        result = subprocess.run(list_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning("Failed to list OS Login keys for purge: %s", result.stderr.strip())
            return

        now_us = int(time.time() * 1e6)
        removed = 0
        for line in (result.stdout or "").splitlines():
            parts = line.split()
            if not parts:
                continue
            fingerprint = parts[0]
            expiry_us = 0
            if len(parts) >= 2:
                try:
                    expiry_us = int(parts[1])
                except ValueError:
                    pass

            # Remove keys that are expired OR have no expiration (legacy keys
            # registered without --ttl). Keys with a future expiry are kept.
            if expiry_us > now_us:
                continue

            rm_cmd = ["gcloud", "compute", "os-login", "ssh-keys", "remove", f"--key={fingerprint}"]
            if impersonate_sa:
                rm_cmd.append(f"--impersonate-service-account={impersonate_sa}")
            subprocess.run(rm_cmd, capture_output=True, text=True)
            removed += 1
        if removed:
            logger.info("Purged %d stale OS Login SSH keys", removed)


_os_login_key_provisioner = OsLoginKeyProvisioner()


def ssh_key_file(
    ssh_config: config_pb2.SshConfig | None,
    impersonate_service_account: str | None = None,
) -> str | None:
    if ssh_config and ssh_config.key_file:
        return ssh_config.key_file
    if uses_os_login(ssh_config):
        path = os.path.expanduser("~/.ssh/google_compute_engine")
        effective_sa = impersonate_service_account or ssh_impersonate_service_account(ssh_config)
        _os_login_key_provisioner.ensure_key(path, effective_sa)
        return path
    return None


def ssh_impersonate_service_account(
    ssh_config: config_pb2.SshConfig | None,
) -> str | None:
    """Return the explicit impersonation SA from ssh_config, or None."""
    if ssh_config and ssh_config.impersonate_service_account:
        return ssh_config.impersonate_service_account
    return None
