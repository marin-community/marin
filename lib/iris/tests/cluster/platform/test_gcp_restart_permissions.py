# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import subprocess

import pytest

from iris.cluster.platform import gcp
from iris.rpc import config_pb2


def _completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["gcloud"], returncode=returncode, stdout=stdout, stderr=stderr)


def _gcp_config(project_id: str = "test-project") -> config_pb2.IrisClusterConfig:
    config = config_pb2.IrisClusterConfig()
    config.platform.gcp.project_id = project_id
    return config


def test_active_account_has_project_role_checks_user_or_service_account_binding(monkeypatch):
    seen_command: list[str] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        nonlocal seen_command
        seen_command = cmd
        return _completed_process("roles/resourcemanager.projectIamAdmin\n")

    monkeypatch.setattr(gcp, "_run_gcloud_restart_permission_command", fake_run)

    has_role = gcp._active_account_has_project_role_for_restart_permissions(
        project_id="test-project",
        account="test-user@example.com",
        role="roles/resourcemanager.projectIamAdmin",
    )

    assert has_role
    filter_arg = next(arg for arg in seen_command if arg.startswith("--filter="))
    assert "bindings.members=user:test-user@example.com" in filter_arg
    assert "bindings.members=serviceAccount:test-user@example.com" in filter_arg


def test_missing_project_permissions_returns_ungranted_subset(monkeypatch):
    monkeypatch.setattr(
        gcp,
        "_run_gcloud_restart_permission_command",
        lambda cmd: _completed_process('{"permissions":["compute.instances.list"]}'),
    )
    required = ("compute.instances.list", "compute.instances.delete")

    missing = gcp._missing_project_permissions_for_restart("test-project", required)

    assert missing == ["compute.instances.delete"]


def test_ensure_gcp_restart_permissions_errors_without_active_account(monkeypatch):
    monkeypatch.setattr(gcp, "_active_gcloud_account_for_restart_permissions", lambda: None)

    with pytest.raises(RuntimeError, match="No active gcloud account found"):
        gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="cluster")


def test_ensure_gcp_restart_permissions_errors_when_required_role_missing(monkeypatch):
    monkeypatch.setattr(gcp, "_active_gcloud_account_for_restart_permissions", lambda: "test-user@example.com")
    monkeypatch.setattr(
        gcp,
        "_active_account_has_project_role_for_restart_permissions",
        lambda project_id, account, role: False,
    )

    with pytest.raises(RuntimeError, match="missing required role"):
        gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="cluster")


def test_ensure_gcp_restart_permissions_errors_when_required_permissions_missing(monkeypatch):
    monkeypatch.setattr(gcp, "_active_gcloud_account_for_restart_permissions", lambda: "test-user@example.com")
    monkeypatch.setattr(
        gcp,
        "_active_account_has_project_role_for_restart_permissions",
        lambda project_id, account, role: True,
    )
    monkeypatch.setattr(
        gcp,
        "_missing_project_permissions_for_restart",
        lambda project_id, permissions: ["tpu.nodes.list"],
    )

    with pytest.raises(RuntimeError, match="missing required IAM permissions"):
        gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="cluster")


def test_ensure_gcp_restart_permissions_uses_scope_specific_permissions(monkeypatch):
    seen_permissions: list[tuple[str, ...]] = []

    monkeypatch.setattr(gcp, "_active_gcloud_account_for_restart_permissions", lambda: "test-user@example.com")
    monkeypatch.setattr(
        gcp,
        "_active_account_has_project_role_for_restart_permissions",
        lambda project_id, account, role: True,
    )

    def fake_missing(project_id: str, permissions: tuple[str, ...]) -> list[str]:
        seen_permissions.append(permissions)
        return []

    monkeypatch.setattr(gcp, "_missing_project_permissions_for_restart", fake_missing)

    gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="controller")

    assert seen_permissions == [gcp._GCP_REQUIRED_PERMISSIONS_BY_SCOPE["controller"]]


def test_ensure_gcp_restart_permissions_errors_for_unknown_scope(monkeypatch):
    monkeypatch.setattr(gcp, "_active_gcloud_account_for_restart_permissions", lambda: "test-user@example.com")
    monkeypatch.setattr(
        gcp,
        "_active_account_has_project_role_for_restart_permissions",
        lambda project_id, account, role: True,
    )

    with pytest.raises(RuntimeError, match="Unknown restart permission scope"):
        gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="unknown")
