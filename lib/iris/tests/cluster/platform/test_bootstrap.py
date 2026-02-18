# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkerBootstrap.bootstrap_vm(), focusing on per-VM bootstrap behavior."""

from __future__ import annotations

import pytest

from iris.cluster.platform.base import PlatformError
from iris.cluster.platform.bootstrap import WorkerBootstrap, render_template
from iris.rpc import config_pb2
from iris.time_utils import Timestamp
from tests.cluster.platform.fakes import FakeVmHandle


def _make_cluster_config() -> config_pb2.IrisClusterConfig:
    return config_pb2.IrisClusterConfig(
        defaults=config_pb2.DefaultsConfig(
            bootstrap=config_pb2.BootstrapConfig(
                docker_image="gcr.io/test/iris-worker:latest",
                worker_port=10001,
                cache_dir="/var/cache/iris",
            ),
        ),
    )


def _make_vm(address: str = "10.0.0.1", vm_id: str = "vm-0") -> FakeVmHandle:
    return FakeVmHandle(
        vm_id=vm_id,
        address=address,
        created_at_ms=Timestamp.now().epoch_ms(),
    )


def test_bootstrap_vm_succeeds():
    """bootstrap_vm() should call bootstrap on the VM and return its log."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)
    vm = _make_vm("10.0.0.1")

    log = bootstrap.bootstrap_vm(vm)

    assert vm._bootstrap_count == 1
    assert log == vm.bootstrap_log


def test_bootstrap_vm_raises_on_empty_address():
    """bootstrap_vm() should raise PlatformError when a VM has no internal address."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)
    vm = _make_vm(address="")

    with pytest.raises(PlatformError, match="has no internal address"):
        bootstrap.bootstrap_vm(vm)


def test_bootstrap_vm_raises_on_connection_timeout():
    """bootstrap_vm() should raise PlatformError when wait_for_connection times out."""
    config = _make_cluster_config()
    bootstrap = WorkerBootstrap(config)
    vm = _make_vm("10.0.0.1")
    vm._wait_for_connection_succeeds = False

    with pytest.raises(PlatformError, match="failed to become reachable"):
        bootstrap.bootstrap_vm(vm)


def test_render_template_basic_substitution():
    """render_template() should substitute {{ variable }} placeholders."""
    template = "Hello {{ name }}, you are {{ age }} years old"
    result = render_template(template, name="Alice", age=30)
    assert result == "Hello Alice, you are 30 years old"


def test_render_template_preserves_docker_templates():
    """render_template() should not touch Docker {{.Field}} syntax (no spaces)."""
    template = 'docker ps --format "{{.Names}} {{.Status}}" and {{ my_var }}'
    result = render_template(template, my_var="test")
    assert result == 'docker ps --format "{{.Names}} {{.Status}}" and test'


def test_render_template_preserves_shell_variables():
    """render_template() should not touch shell ${VAR} syntax."""
    template = "echo ${PATH} and {{ iris_var }}"
    result = render_template(template, iris_var="value")
    assert result == "echo ${PATH} and value"


def test_render_template_raises_on_missing_variable():
    """render_template() should raise ValueError when a variable is missing."""
    template = "Hello {{ name }} and {{ missing }}"
    with pytest.raises(ValueError, match="Template variable 'missing' not provided"):
        render_template(template, name="Alice")


def test_render_template_raises_on_unused_variable():
    """render_template() should raise ValueError when extra variables are passed."""
    template = "Hello {{ name }}"
    with pytest.raises(ValueError, match="Unused template variables: extra"):
        render_template(template, name="Alice", extra="unused")


def test_render_template_rejects_sloppy_whitespace():
    """render_template() requires exactly one space inside braces."""
    # Extra spaces should NOT match, leaving the placeholder unsubstituted.
    # Since the variable is passed but never used, we expect an "unused" error.
    template = "Hello {{  name  }}"
    with pytest.raises(ValueError, match="Unused template variables: name"):
        render_template(template, name="Alice")


def test_render_template_mixed_syntaxes():
    """render_template() handles Iris, Docker, and shell syntaxes together."""
    template = 'echo ${PATH}; docker inspect --format "{{.State.Status}}" {{ container }}; ' "echo {{ greeting }}"
    result = render_template(template, container="myapp", greeting="hi")
    assert result == 'echo ${PATH}; docker inspect --format "{{.State.Status}}" myapp; echo hi'


def test_worker_bootstrap_script_generation():
    """build_worker_bootstrap_script() should generate valid script without escaping issues."""
    from iris.cluster.platform.bootstrap import build_worker_bootstrap_script

    config = _make_cluster_config()
    script = build_worker_bootstrap_script(config, vm_address="10.0.0.1")

    # Verify the script contains Docker template syntax (not escaped)
    assert "{{.Status}}" in script or "{{.State}}" in script or "{{.Names}}" in script
    # Verify our variables were substituted
    assert "gcr.io/test/iris-worker:latest" in script
    assert "10001" in script
    assert "/var/cache/iris" in script
    # Verify no {{ iris_var }} patterns remain (all substituted)
    assert "{{ docker_image }}" not in script
    assert "{{ cache_dir }}" not in script
    assert "{{ worker_port }}" not in script
