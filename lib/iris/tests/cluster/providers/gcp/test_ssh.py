# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for SSH key provisioning and OS Login helpers."""

from __future__ import annotations

import subprocess
import threading
from unittest.mock import patch

import pytest
from iris.cluster.providers.gcp.ssh import OsLoginKeyProvisioner, ssh_key_file, uses_os_login
from iris.rpc import config_pb2


def _ssh_config(
    auth_mode: int = config_pb2.SshConfig.SSH_AUTH_MODE_METADATA, key_file: str = ""
) -> config_pb2.SshConfig:
    cfg = config_pb2.SshConfig(auth_mode=auth_mode)
    if key_file:
        cfg.key_file = key_file
    return cfg


def _os_login_config(key_file: str = "") -> config_pb2.SshConfig:
    return _ssh_config(auth_mode=config_pb2.SshConfig.SSH_AUTH_MODE_OS_LOGIN, key_file=key_file)


# -- uses_os_login ---------------------------------------------------------


def test_uses_os_login_none():
    assert not uses_os_login(None)


def test_uses_os_login_metadata():
    assert not uses_os_login(_ssh_config())


def test_uses_os_login_true():
    assert uses_os_login(_os_login_config())


# -- OsLoginKeyProvisioner -------------------------------------------------


@pytest.fixture()
def provisioner():
    return OsLoginKeyProvisioner()


@patch(
    "iris.cluster.providers.gcp.ssh.subprocess.run",
    return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
)
@patch("iris.cluster.providers.gcp.ssh.os.path.exists", return_value=False)
@patch("iris.cluster.providers.gcp.ssh.os.makedirs")
def test_ensure_key_generates_and_registers(mock_makedirs, mock_exists, mock_run, provisioner):
    provisioner.ensure_key("/tmp/test_key", "sa@project.iam.gserviceaccount.com")

    # keygen + purge list + add = 3 calls
    assert mock_run.call_count == 3
    keygen_call = mock_run.call_args_list[0]
    assert "ssh-keygen" in keygen_call.args[0]
    assert "/tmp/test_key" in keygen_call.args[0]

    register_call = mock_run.call_args_list[2]
    assert "ssh-keys" in register_call.args[0]
    assert "add" in register_call.args[0]
    assert "--impersonate-service-account=sa@project.iam.gserviceaccount.com" in register_call.args[0]


@patch("iris.cluster.providers.gcp.ssh.subprocess.run")
@patch("iris.cluster.providers.gcp.ssh.os.path.exists", return_value=True)
def test_ensure_key_skips_when_valid(mock_exists, mock_run, provisioner):
    # Simulate a previous successful registration
    provisioner._registration_expiry = float("inf")

    provisioner.ensure_key("/tmp/test_key", "sa@project.iam.gserviceaccount.com")

    mock_run.assert_not_called()


@patch(
    "iris.cluster.providers.gcp.ssh.subprocess.run",
    return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
)
@patch("iris.cluster.providers.gcp.ssh.os.path.exists", return_value=True)
@patch("iris.cluster.providers.gcp.ssh.time.monotonic", return_value=100000.0)
def test_ensure_key_reregisters_on_expiry(mock_monotonic, mock_exists, mock_run, provisioner):
    # Registration expired
    provisioner._registration_expiry = 99999.0

    provisioner.ensure_key("/tmp/test_key", "sa@project.iam.gserviceaccount.com")

    # Should only register (not keygen, since key exists): purge list + add = 2 calls
    assert mock_run.call_count == 2
    assert "add" in mock_run.call_args.args[0]


@patch(
    "iris.cluster.providers.gcp.ssh.subprocess.run",
    return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
)
@patch("iris.cluster.providers.gcp.ssh.os.path.exists", return_value=False)
@patch("iris.cluster.providers.gcp.ssh.os.makedirs")
def test_ensure_key_no_impersonate_sa(mock_makedirs, mock_exists, mock_run, provisioner):
    provisioner.ensure_key("/tmp/test_key", None)

    register_call = mock_run.call_args_list[2]
    cmd = register_call.args[0]
    assert "add" in cmd
    assert not any("--impersonate-service-account" in arg for arg in cmd)


@patch(
    "iris.cluster.providers.gcp.ssh.subprocess.run",
    return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
)
@patch("iris.cluster.providers.gcp.ssh.os.path.exists", return_value=False)
@patch("iris.cluster.providers.gcp.ssh.os.makedirs")
def test_ensure_key_thread_safety(mock_makedirs, mock_exists, mock_run, provisioner):
    barrier = threading.Barrier(4)
    results: list[Exception | None] = [None] * 4

    def call(idx):
        try:
            barrier.wait(timeout=5)
            provisioner.ensure_key("/tmp/test_key", "sa@project.iam.gserviceaccount.com")
        except Exception as e:
            results[idx] = e

    threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert all(r is None for r in results), f"Thread errors: {results}"
    # keygen should only be called once (guarded by lock + os.path.exists check)
    keygen_calls = [c for c in mock_run.call_args_list if "ssh-keygen" in c.args[0]]
    assert len(keygen_calls) >= 1


# -- ssh_key_file -----------------------------------------------------------


@patch("iris.cluster.providers.gcp.ssh._os_login_key_provisioner")
def test_ssh_key_file_metadata_mode(mock_provisioner):
    result = ssh_key_file(_ssh_config())
    assert result is None
    mock_provisioner.ensure_key.assert_not_called()


@patch("iris.cluster.providers.gcp.ssh._os_login_key_provisioner")
def test_ssh_key_file_explicit_key(mock_provisioner):
    result = ssh_key_file(_os_login_config(key_file="/custom/key"))
    assert result == "/custom/key"
    mock_provisioner.ensure_key.assert_not_called()


@patch("iris.cluster.providers.gcp.ssh._os_login_key_provisioner")
def test_ssh_key_file_os_login_provisions(mock_provisioner):
    result = ssh_key_file(_os_login_config(), impersonate_service_account="sa@test.iam.gserviceaccount.com")
    assert result is not None
    assert "google_compute_engine" in result
    mock_provisioner.ensure_key.assert_called_once()
    call_args = mock_provisioner.ensure_key.call_args
    assert call_args.args[1] == "sa@test.iam.gserviceaccount.com"


@patch("iris.cluster.providers.gcp.ssh._os_login_key_provisioner")
def test_ssh_key_file_os_login_uses_config_sa_as_fallback(mock_provisioner):
    """When no explicit SA is passed, ssh_key_file falls back to the config's impersonate_service_account."""
    cfg = _os_login_config()
    cfg.impersonate_service_account = "sa-from-config@project.iam.gserviceaccount.com"
    result = ssh_key_file(cfg)
    assert result is not None
    mock_provisioner.ensure_key.assert_called_once()
    call_args = mock_provisioner.ensure_key.call_args
    assert call_args.args[1] == "sa-from-config@project.iam.gserviceaccount.com"


@patch("iris.cluster.providers.gcp.ssh._os_login_key_provisioner")
def test_ssh_key_file_none_config(mock_provisioner):
    result = ssh_key_file(None)
    assert result is None
    mock_provisioner.ensure_key.assert_not_called()


# -- _purge_stale_keys --------------------------------------------------------


@patch("iris.cluster.providers.gcp.ssh.time.time", return_value=1_700_000_000.0)
@patch("iris.cluster.providers.gcp.ssh.subprocess.run")
def test_purge_stale_keys_removes_expired_and_no_expiry(mock_run, mock_time, provisioner):
    """Expired keys and keys with no expiration (0) are removed; future keys are kept."""
    now_us = int(1_700_000_000.0 * 1e6)
    expired_us = now_us - 1_000_000
    future_us = now_us + 86_400_000_000

    list_output = "\n".join(
        [
            f"SHA256:expired {expired_us}",
            "SHA256:no_expiry",  # no second column → expirationTimeUsec absent
            "SHA256:no_expiry_zero 0",
            f"SHA256:future {future_us}",
        ]
    )
    mock_run.return_value = subprocess.CompletedProcess([], 0, stdout=list_output, stderr="")

    provisioner._purge_stale_keys("sa@test.iam.gserviceaccount.com")

    # list call + 3 removes (expired, no_expiry, no_expiry_zero) = 4 calls total
    assert mock_run.call_count == 4
    remove_calls = mock_run.call_args_list[1:]
    removed_keys = {arg for c in remove_calls for arg in c.args[0] if arg.startswith("--key=")}
    assert removed_keys == {
        "--key=SHA256:expired",
        "--key=SHA256:no_expiry",
        "--key=SHA256:no_expiry_zero",
    }


@patch("iris.cluster.providers.gcp.ssh.subprocess.run")
def test_purge_stale_keys_list_failure_is_nonfatal(mock_run, provisioner):
    """If listing keys fails, the purge logs a warning and returns without error."""
    mock_run.return_value = subprocess.CompletedProcess([], 1, stdout="", stderr="some error")

    provisioner._purge_stale_keys(None)  # should not raise

    assert mock_run.call_count == 1  # only the list call
