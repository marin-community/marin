# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker bootstrap script generation."""

from __future__ import annotations

import pytest

from iris.cluster.platform.bootstrap import (
    build_controller_bootstrap_script_from_config,
    build_worker_bootstrap_script,
    render_template,
    rewrite_ghcr_to_ar_remote,
    zone_to_multi_region,
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


def test_build_worker_bootstrap_script_configures_ar_auth() -> None:
    ar_image = "us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-worker:latest"
    cfg = _bootstrap_config(docker_image=ar_image)

    script = build_worker_bootstrap_script(cfg, vm_address="10.0.0.2")

    assert f'if echo "{ar_image}" | grep -q -- "-docker.pkg.dev/"' in script
    assert 'sudo gcloud auth configure-docker "$AR_HOST" -q || true' in script


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


@pytest.mark.parametrize(
    "zone, expected",
    [
        ("us-central1-a", "us"),
        ("us-west4-b", "us"),
        ("europe-west4-b", "europe"),
    ],
)
def test_zone_to_multi_region(zone: str, expected: str) -> None:
    assert zone_to_multi_region(zone) == expected


def test_zone_to_multi_region_unknown_prefix() -> None:
    assert zone_to_multi_region("southamerica-east1-a") is None


@pytest.mark.parametrize("zone", ["asia-east1-a", "me-west1-a"])
def test_zone_to_multi_region_unsupported_raises(zone: str) -> None:
    with pytest.raises(ValueError, match="no AR remote repo provisioned"):
        zone_to_multi_region(zone)


@pytest.mark.parametrize(
    "image_tag, multi_region, project, expected",
    [
        (
            "ghcr.io/marin-community/iris-worker:v1",
            "us",
            "hai-gcp-models",
            "us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-worker:v1",
        ),
        (
            "ghcr.io/marin-community/iris-controller:latest",
            "europe",
            "hai-gcp-models",
            "europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-controller:latest",
        ),
        (
            "ghcr.io/myorg/myimage:abc123",
            "us",
            "my-project",
            "us-docker.pkg.dev/my-project/ghcr-mirror/myorg/myimage:abc123",
        ),
    ],
)
def test_rewrite_ghcr_to_ar_remote(image_tag: str, multi_region: str, project: str, expected: str) -> None:
    assert rewrite_ghcr_to_ar_remote(image_tag, multi_region, project) == expected


def test_rewrite_ghcr_to_ar_remote_non_ghcr_passthrough() -> None:
    assert rewrite_ghcr_to_ar_remote("ubuntu:22.04", "us", "proj") == "ubuntu:22.04"
    assert rewrite_ghcr_to_ar_remote("gcr.io/proj/img:v1", "us", "proj") == "gcr.io/proj/img:v1"


def test_rewrite_ghcr_to_ar_remote_custom_mirror_repo() -> None:
    result = rewrite_ghcr_to_ar_remote("ghcr.io/org/image:v1", "us", "proj", mirror_repo="custom-mirror")
    assert result == "us-docker.pkg.dev/proj/custom-mirror/org/image:v1"


def test_build_controller_bootstrap_script_from_config_rewrites_ghcr_to_ar() -> None:
    from unittest.mock import MagicMock

    config = config_pb2.IrisClusterConfig()
    config.controller.image = "ghcr.io/marin-community/iris-controller:latest"
    config.controller.gcp.zone = "europe-west4-b"
    config.controller.gcp.port = 10000
    config.platform.gcp.project_id = "hai-gcp-models"

    platform = MagicMock()
    platform.resolve_image.return_value = (
        "europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-controller:latest"
    )
    script = build_controller_bootstrap_script_from_config(config, platform=platform)

    platform.resolve_image.assert_called_once_with(
        "ghcr.io/marin-community/iris-controller:latest", zone="europe-west4-b"
    )
    assert (
        "Pulling image: europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-controller:latest"
        in script
    )
    assert 'sudo gcloud auth configure-docker "$AR_HOST" -q || true' in script


def test_build_controller_bootstrap_script_from_config_non_ghcr_passthrough() -> None:
    """Non-GHCR images are not rewritten."""
    from unittest.mock import MagicMock

    config = config_pb2.IrisClusterConfig()
    config.controller.image = "us-docker.pkg.dev/proj/repo/iris-controller:latest"
    config.controller.gcp.zone = "europe-west4-b"
    config.controller.gcp.port = 10000

    platform = MagicMock()
    platform.resolve_image.side_effect = lambda image, zone=None: image
    script = build_controller_bootstrap_script_from_config(config, platform=platform)

    assert "Pulling image: us-docker.pkg.dev/proj/repo/iris-controller:latest" in script


# --- Platform.resolve_image() tests ---


def test_gcp_platform_resolve_image_rewrites_ghcr() -> None:
    """GcpPlatform.resolve_image() rewrites GHCR images for the correct continent."""
    from iris.cluster.platform.gcp import GcpPlatform
    from unittest.mock import patch

    # GcpPlatform.__init__ tries to detect project/zones — mock that
    with patch.object(GcpPlatform, "__init__", lambda self, *a, **kw: None):
        platform = GcpPlatform.__new__(GcpPlatform)
        platform._project_id = "my-proj"

    assert platform.resolve_image("ghcr.io/org/img:v1", zone="us-central1-a") == (
        "us-docker.pkg.dev/my-proj/ghcr-mirror/org/img:v1"
    )
    assert platform.resolve_image("ghcr.io/org/img:v1", zone="europe-west4-b") == (
        "europe-docker.pkg.dev/my-proj/ghcr-mirror/org/img:v1"
    )


def test_gcp_platform_resolve_image_passthrough_non_ghcr() -> None:
    """GcpPlatform.resolve_image() returns non-GHCR images unchanged."""
    from iris.cluster.platform.gcp import GcpPlatform
    from unittest.mock import patch

    with patch.object(GcpPlatform, "__init__", lambda self, *a, **kw: None):
        platform = GcpPlatform.__new__(GcpPlatform)
        platform._project_id = "my-proj"

    assert platform.resolve_image("docker.io/library/ubuntu:latest", zone="us-central1-a") == (
        "docker.io/library/ubuntu:latest"
    )


def test_gcp_platform_resolve_image_requires_zone_for_ghcr() -> None:
    """GcpPlatform.resolve_image() raises when zone is missing for GHCR images."""
    from iris.cluster.platform.gcp import GcpPlatform
    from unittest.mock import patch

    with patch.object(GcpPlatform, "__init__", lambda self, *a, **kw: None):
        platform = GcpPlatform.__new__(GcpPlatform)
        platform._project_id = "my-proj"

    with pytest.raises(ValueError, match="zone is required"):
        platform.resolve_image("ghcr.io/org/img:v1")


def test_local_platform_resolve_image_passthrough() -> None:
    """LocalPlatform.resolve_image() always returns the image unchanged."""
    from iris.cluster.platform.local import LocalPlatform

    platform = LocalPlatform(label_prefix="test")
    assert platform.resolve_image("ghcr.io/org/img:v1", zone="us-central1-a") == "ghcr.io/org/img:v1"
    assert platform.resolve_image("docker.io/library/ubuntu:latest") == "docker.io/library/ubuntu:latest"
