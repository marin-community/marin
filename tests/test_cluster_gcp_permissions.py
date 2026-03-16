# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

import pytest

from marin.cluster import gcp


def _completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["gcloud"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_get_active_account_returns_email(monkeypatch):
    monkeypatch.setattr(
        gcp,
        "run_gcloud_command",
        lambda cmd, **kwargs: _completed_process("test-user@example.com\n"),
    )
    assert gcp.get_active_account() == "test-user@example.com"


def test_account_has_project_role_checks_user_or_service_account_binding(monkeypatch):
    seen_command: list[str] = []

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        nonlocal seen_command
        seen_command = cmd
        return _completed_process(f"{gcp.RESTART_REQUIRED_PROJECT_ROLE}\n")

    monkeypatch.setattr(gcp, "run_gcloud_command", fake_run)

    has_role = gcp.account_has_project_role(
        project="test-project",
        account="test-user@example.com",
        role=gcp.RESTART_REQUIRED_PROJECT_ROLE,
    )

    assert has_role
    filter_arg = next(arg for arg in seen_command if arg.startswith("--filter="))
    assert "bindings.members=user:test-user@example.com" in filter_arg
    assert "bindings.members=serviceAccount:test-user@example.com" in filter_arg


def test_ensure_active_account_has_project_role_errors_without_active_account(monkeypatch):
    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("\n")
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "run_gcloud_command", fake_run)

    with pytest.raises(RuntimeError, match="No active gcloud account found"):
        gcp.ensure_active_account_has_project_role("test-project")


def test_ensure_active_account_has_project_role_errors_for_missing_role(monkeypatch):
    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("test-user@example.com\n")
        if cmd[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _completed_process("")
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "run_gcloud_command", fake_run)

    with pytest.raises(RuntimeError, match="missing required role"):
        gcp.ensure_active_account_has_project_role("test-project")


def test_ensure_active_account_has_project_role_returns_active_account(monkeypatch):
    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("test-user@example.com\n")
        if cmd[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _completed_process(f"{gcp.RESTART_REQUIRED_PROJECT_ROLE}\n")
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "run_gcloud_command", fake_run)

    assert gcp.ensure_active_account_has_project_role("test-project") == "test-user@example.com"


def test_missing_project_permissions_returns_ungranted_subset(monkeypatch):
    monkeypatch.setattr(
        gcp,
        "run_gcloud_command",
        lambda cmd, **kwargs: _completed_process('{"permissions":["compute.instances.list"]}'),
    )
    required = ("compute.instances.list", "compute.instances.delete")

    missing = gcp.missing_project_permissions("test-project", required)

    assert missing == ["compute.instances.delete"]


def test_ensure_active_account_can_restart_cluster_errors_for_missing_permissions(monkeypatch):
    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("test-user@example.com\n")
        if cmd[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _completed_process(f"{gcp.RESTART_REQUIRED_PROJECT_ROLE}\n")
        if cmd[:3] == ["gcloud", "projects", "test-iam-permissions"]:
            return _completed_process(json.dumps({"permissions": ["compute.instances.list"]}))
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "run_gcloud_command", fake_run)

    with pytest.raises(RuntimeError, match="missing required IAM permissions"):
        gcp.ensure_active_account_can_restart_cluster("test-project")


def test_ensure_active_account_can_restart_cluster_returns_active_account(monkeypatch):
    seen_permissions: list[tuple[str, ...]] = []

    def fake_run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["gcloud", "auth", "list"]:
            return _completed_process("test-user@example.com\n")
        if cmd[:3] == ["gcloud", "projects", "get-iam-policy"]:
            return _completed_process(f"{gcp.RESTART_REQUIRED_PROJECT_ROLE}\n")
        if cmd[:3] == ["gcloud", "projects", "test-iam-permissions"]:
            permissions_arg = next(arg for arg in cmd if arg.startswith("--permissions="))
            permissions = tuple(permissions_arg.split("=", 1)[1].split(","))
            seen_permissions.append(permissions)
            return _completed_process(json.dumps({"permissions": list(permissions)}))
        pytest.fail(f"unexpected gcloud command: {cmd}")

    monkeypatch.setattr(gcp, "run_gcloud_command", fake_run)

    account = gcp.ensure_active_account_can_restart_cluster("test-project")
    assert account == "test-user@example.com"
    assert seen_permissions == [gcp.RESTART_REQUIRED_PERMISSIONS]
