# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker bootstrap script generation."""

from __future__ import annotations

import pytest

from iris.cluster.platform.bootstrap import (
    build_worker_bootstrap_script,
    parse_artifact_registry_tag,
    render_template,
    rewrite_artifact_registry_region,
)
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


class TestParseArtifactRegistryTag:
    def test_standard_ar_tag(self) -> None:
        result = parse_artifact_registry_tag("us-west4-docker.pkg.dev/my-project/marin/iris-worker:v1.0")
        assert result == ("us-west4", "my-project", "iris-worker", "v1.0")

    def test_no_version_defaults_to_latest(self) -> None:
        result = parse_artifact_registry_tag("europe-west4-docker.pkg.dev/proj/repo/image")
        assert result == ("europe-west4", "proj", "image", "latest")

    def test_non_ar_image_returns_none(self) -> None:
        assert parse_artifact_registry_tag("gcr.io/project/image:tag") is None
        assert parse_artifact_registry_tag("ghcr.io/org/image:latest") is None
        assert parse_artifact_registry_tag("ubuntu:22.04") is None

    def test_malformed_ar_tag_returns_none(self) -> None:
        assert parse_artifact_registry_tag("us-west4-docker.pkg.dev/project") is None


class TestRewriteArtifactRegistryRegion:
    def test_rewrites_region(self) -> None:
        original = "us-west4-docker.pkg.dev/my-project/marin/iris-worker:latest"
        result = rewrite_artifact_registry_region(original, "europe-west4")
        assert result == "europe-west4-docker.pkg.dev/my-project/marin/iris-worker:latest"

    def test_same_region_noop(self) -> None:
        original = "us-west4-docker.pkg.dev/my-project/marin/iris-worker:latest"
        result = rewrite_artifact_registry_region(original, "us-west4")
        assert result == original

    def test_non_ar_image_passthrough(self) -> None:
        original = "ghcr.io/org/iris-worker:latest"
        result = rewrite_artifact_registry_region(original, "europe-west4")
        assert result == original

    def test_preserves_full_tag(self) -> None:
        original = "us-central1-docker.pkg.dev/hai-gcp-models/marin/iris-worker:abc123"
        result = rewrite_artifact_registry_region(original, "europe-west4")
        assert result == "europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-worker:abc123"
