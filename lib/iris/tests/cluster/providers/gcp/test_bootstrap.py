# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker bootstrap script generation."""

from __future__ import annotations

import pytest

from iris.cluster.providers.gcp.bootstrap import (
    build_controller_bootstrap_script_from_config,
    build_worker_bootstrap_script,
    render_template,
    rewrite_ghcr_to_ar_remote,
    zone_to_multi_region,
)
from iris.cluster.dashboard_common import DASHBOARD_TITLE_ENV_VAR
from iris.rpc import config_pb2


def _worker_config(**overrides: object) -> config_pb2.WorkerConfig:
    cfg = config_pb2.WorkerConfig(
        docker_image="gcr.io/test/iris-worker:latest",
        port=10001,
        cache_dir="/var/cache/iris",
        controller_address="10.0.0.10:10000",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_build_worker_bootstrap_script_includes_controller_address() -> None:
    script = build_worker_bootstrap_script(_worker_config())

    assert "controller_address" in script
    assert "10.0.0.10:10000" in script
    assert "gcr.io/test/iris-worker:latest" in script


def test_build_worker_bootstrap_script_configures_ar_auth() -> None:
    ar_image = "us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-worker:latest"
    cfg = _worker_config(docker_image=ar_image)

    script = build_worker_bootstrap_script(cfg)

    assert f'if echo "{ar_image}" | grep -q -- "-docker.pkg.dev/"' in script
    assert 'sudo gcloud auth configure-docker "$AR_HOST" -q || true' in script


def test_build_worker_bootstrap_script_requires_controller_address() -> None:
    cfg = _worker_config()
    cfg.controller_address = ""

    with pytest.raises(ValueError, match="controller_address"):
        build_worker_bootstrap_script(cfg)


def test_build_worker_bootstrap_script_embeds_worker_config_json() -> None:
    """WorkerConfig fields appear in the embedded JSON in the generated script."""
    cfg = _worker_config()
    cfg.task_env["IRIS_SCALE_GROUP"] = "west-group"

    script = build_worker_bootstrap_script(cfg)

    assert "IRIS_SCALE_GROUP" in script
    assert "west-group" in script
    assert "worker_config.json" in script


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
    config = config_pb2.IrisClusterConfig()
    config.controller.image = "ghcr.io/marin-community/iris-controller:latest"
    config.controller.gcp.zone = "europe-west4-b"
    config.controller.gcp.port = 10000
    config.platform.gcp.project_id = "hai-gcp-models"

    def resolve_image(image: str, zone: str | None = None) -> str:
        return "europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-controller:latest"

    script = build_controller_bootstrap_script_from_config(config, resolve_image=resolve_image)

    assert (
        "Pulling image: europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-controller:latest"
        in script
    )
    assert 'sudo gcloud auth configure-docker "$AR_HOST" -q || true' in script


def test_build_controller_bootstrap_script_sets_dashboard_title() -> None:
    config = config_pb2.IrisClusterConfig()
    config.platform.label_prefix = "marin-dev"
    config.controller.image = "ghcr.io/marin-community/iris-controller:latest"
    config.controller.gcp.zone = "us-central1-a"
    config.controller.gcp.port = 10000
    config.platform.gcp.project_id = "hai-gcp-models"

    script = build_controller_bootstrap_script_from_config(config, resolve_image=lambda image, zone=None: image)

    assert f"-e {DASHBOARD_TITLE_ENV_VAR}=marin-dev" in script


# --- GcpWorkerProvider.resolve_image() tests ---


def _make_gcp_worker_provider(project_id: str = "my-proj"):
    """Build a GcpWorkerProvider backed by InMemoryGcpService for testing."""
    from iris.cluster.providers.gcp.fake import InMemoryGcpService
    from iris.cluster.providers.gcp.workers import GcpWorkerProvider
    from iris.cluster.service_mode import ServiceMode

    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id=project_id)
    gcp_config = config_pb2.GcpPlatformConfig(project_id=project_id)
    return GcpWorkerProvider(gcp_config=gcp_config, label_prefix="iris", gcp_service=gcp_service)


def test_gcp_provider_resolve_image_rewrites_ghcr() -> None:
    """GcpWorkerProvider.resolve_image() rewrites GHCR images for the correct continent."""
    provider = _make_gcp_worker_provider()

    assert provider.resolve_image("ghcr.io/org/img:v1", zone="us-central1-a") == (
        "us-docker.pkg.dev/my-proj/ghcr-mirror/org/img:v1"
    )
    assert provider.resolve_image("ghcr.io/org/img:v1", zone="europe-west4-b") == (
        "europe-docker.pkg.dev/my-proj/ghcr-mirror/org/img:v1"
    )


def test_gcp_provider_resolve_image_passthrough_non_ghcr() -> None:
    """GcpWorkerProvider.resolve_image() returns non-GHCR images unchanged."""
    provider = _make_gcp_worker_provider()

    assert provider.resolve_image("docker.io/library/ubuntu:latest", zone="us-central1-a") == (
        "docker.io/library/ubuntu:latest"
    )


def test_worker_bootstrap_tunes_network_sysctls() -> None:
    """Worker bootstrap configures sysctl for expanded port range and TIME_WAIT reuse."""
    script = build_worker_bootstrap_script(_worker_config())
    assert 'sysctl -w net.ipv4.ip_local_port_range="1024 65535"' in script
    assert "sysctl -w net.ipv4.tcp_tw_reuse=1" in script


def test_gcp_provider_resolve_image_requires_zone_for_ghcr() -> None:
    """GcpWorkerProvider.resolve_image() raises when zone is missing for GHCR images."""
    provider = _make_gcp_worker_provider()

    with pytest.raises(ValueError, match="zone is required"):
        provider.resolve_image("ghcr.io/org/img:v1")
