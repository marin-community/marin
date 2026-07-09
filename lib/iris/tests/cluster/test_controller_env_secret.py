# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The controller's own credentials reach it through a Secret task pods never mount."""

import pytest
from iris.cluster.config import AuthConfig, IrisClusterConfig
from iris.cluster.inject_env import TASK_ENV_SECRET_NAME
from iris.cluster.platforms.k8s.controller import (
    CONTROLLER_ENV_SECRET_NAME,
    SIGNING_KEY_ENV_VAR,
    _build_controller_deployment,
    _controller_env,
)


def _env_from_names(**kwargs) -> list[str]:
    manifest = _build_controller_deployment(namespace="iris", image="img", port=10000, node_selector={}, **kwargs)
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    return [ref["secretRef"]["name"] for ref in container.get("envFrom", [])]


def test_controller_mounts_its_own_secret_alongside_the_task_env_secret():
    assert _env_from_names(task_env_secret=True, controller_env_secret=True) == [
        TASK_ENV_SECRET_NAME,
        CONTROLLER_ENV_SECRET_NAME,
    ]


def test_controller_mounts_only_its_own_secret_when_no_task_env_is_projected():
    assert _env_from_names(task_env_secret=False, controller_env_secret=True) == [CONTROLLER_ENV_SECRET_NAME]


def test_controller_declares_no_env_sources_when_neither_secret_is_projected():
    manifest = _build_controller_deployment(namespace="iris", image="img", port=10000, node_selector={})
    assert "envFrom" not in manifest["spec"]["template"]["spec"]["containers"][0]


def test_a_signing_key_is_resolved_in_the_operator_shell(monkeypatch):
    monkeypatch.setenv(SIGNING_KEY_ENV_VAR, "resolved-pem")
    config = IrisClusterConfig(name="c", auth=AuthConfig(signing_key=f"env:{SIGNING_KEY_ENV_VAR}"))

    assert _controller_env(config) == {SIGNING_KEY_ENV_VAR: "resolved-pem"}


def test_a_cluster_without_a_signing_key_projects_no_controller_secret():
    assert _controller_env(IrisClusterConfig(name="c", auth=AuthConfig())) == {}


def test_a_cluster_with_no_auth_block_projects_no_controller_secret():
    assert _controller_env(IrisClusterConfig(name="c")) == {}


def test_a_signing_key_the_pod_cannot_reach_is_refused_at_deploy_time():
    """A pod holds no cloud credentials, so `gcp-secret://` alone would resolve nowhere."""
    config = IrisClusterConfig(name="c", auth=AuthConfig(signing_key="gcp-secret://projects/p/secrets/s/versions/1"))

    with pytest.raises(ValueError, match=f"env:{SIGNING_KEY_ENV_VAR}"):
        _controller_env(config)
