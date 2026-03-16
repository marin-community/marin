# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
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
    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("\n")
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "_run_gcloud_restart_permission_command", fake_run)

    with pytest.raises(RuntimeError, match="No active gcloud account found"):
        gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="cluster")


def test_ensure_gcp_restart_permissions_errors_when_required_role_missing(monkeypatch):
    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("test-user@example.com\n")
        if cmd[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _completed_process("")
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "_run_gcloud_restart_permission_command", fake_run)

    with pytest.raises(RuntimeError, match="missing required role"):
        gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="cluster")


def test_ensure_gcp_restart_permissions_errors_when_required_permissions_missing(monkeypatch):
    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("test-user@example.com\n")
        if cmd[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _completed_process("roles/resourcemanager.projectIamAdmin\n")
        if cmd[:3] == ["gcloud", "projects", "test-iam-permissions"]:
            return _completed_process(json.dumps({"permissions": ["compute.instances.list"]}))
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "_run_gcloud_restart_permission_command", fake_run)

    with pytest.raises(RuntimeError, match="missing required IAM permissions"):
        gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="cluster")


def test_ensure_gcp_restart_permissions_uses_scope_specific_permissions(monkeypatch):
    seen_permissions: list[tuple[str, ...]] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("test-user@example.com\n")
        if cmd[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _completed_process("roles/resourcemanager.projectIamAdmin\n")
        if cmd[:3] == ["gcloud", "projects", "test-iam-permissions"]:
            permissions_arg = next(arg for arg in cmd if arg.startswith("--permissions="))
            permissions = tuple(permissions_arg.split("=", 1)[1].split(","))
            seen_permissions.append(permissions)
            return _completed_process(json.dumps({"permissions": list(permissions)}))
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "_run_gcloud_restart_permission_command", fake_run)

    gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="controller")

    assert seen_permissions == [gcp._GCP_REQUIRED_PERMISSIONS_BY_SCOPE["controller"]]


def test_ensure_gcp_restart_permissions_errors_for_unknown_scope(monkeypatch):
    seen_commands: list[list[str]] = []

    def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen_commands.append(cmd)
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("test-user@example.com\n")
        if cmd[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _completed_process("roles/resourcemanager.projectIamAdmin\n")
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "_run_gcloud_restart_permission_command", fake_run)

    with pytest.raises(RuntimeError, match="Unknown restart permission scope"):
        gcp.ensure_gcp_restart_permissions(_gcp_config(), scope="unknown")

    assert all(cmd[:3] != ["gcloud", "projects", "test-iam-permissions"] for cmd in seen_commands)
