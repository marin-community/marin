# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker bootstrap script generation."""

from __future__ import annotations

import pytest

from iris.cluster.platform.bootstrap import build_worker_bootstrap_script, render_template
from iris.rpc import config_pb2


def _bootstrap_config(**overrides: object) -> config_pb2.BootstrapConfig:
    cfg = config_pb2.BootstrapConfig(
        docker_image="gcr.io/test/iris-worker:latest",
        worker_port=10001,
        cache_dir="/var/cache/iris",
        controller_address="10.0.0.10:10000",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_build_worker_bootstrap_script_includes_controller_address() -> None:
    script = build_worker_bootstrap_script(_bootstrap_config(), vm_address="10.0.0.2")

    assert "--controller-address 10.0.0.10:10000" in script
    assert "--config /etc/iris/config.yaml" not in script
    assert "gcr.io/test/iris-worker:latest" in script
    assert "IRIS_VM_ADDRESS=10.0.0.2" in script


def test_build_worker_bootstrap_script_requires_controller_address() -> None:
    cfg = _bootstrap_config()
    cfg.controller_address = ""

    with pytest.raises(ValueError, match="controller_address"):
        build_worker_bootstrap_script(cfg, vm_address="10.0.0.2")


def test_build_worker_bootstrap_script_includes_env_vars() -> None:
    """Env vars in BootstrapConfig appear in the generated script."""
    cfg = _bootstrap_config()
    cfg.env_vars["IRIS_WORKER_ATTRIBUTES"] = '{"region": "us-west4"}'
    cfg.env_vars["IRIS_SCALE_GROUP"] = "west-group"

    script = build_worker_bootstrap_script(cfg, vm_address="10.0.0.2")

    assert "IRIS_WORKER_ATTRIBUTES=" in script
    assert "us-west4" in script
    assert "IRIS_SCALE_GROUP=" in script
    assert "west-group" in script


def test_render_template_preserves_docker_templates() -> None:
    template = 'docker ps --format "{{.Names}} {{.Status}}" and {{ value }}'
    rendered = render_template(template, value="x")
    assert rendered == 'docker ps --format "{{.Names}} {{.Status}}" and x'


def test_render_template_preserves_shell_variables() -> None:
    template = "echo ${PATH} and {{ value }}"
    rendered = render_template(template, value="x")
    assert rendered == "echo ${PATH} and x"
