# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker bootstrap script generation."""

import re

import pytest
from iris.cluster.config import GcpPlatformConfig, WorkerConfig
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.worker_bootstrap import (
    build_worker_bootstrap_script,
    render_template,
    rewrite_ghcr_to_ar_remote,
    zone_to_multi_region,
)
from iris.cluster.platforms.gcp.workers import GcpWorkerProvider
from iris.cluster.service_mode import ServiceMode


def _worker_config(**overrides: object) -> WorkerConfig:
    cfg = WorkerConfig(
        docker_image="gcr.io/test/iris-worker:latest",
        port=10001,
        cache_dir="/var/cache/iris",
        controller_address="10.0.0.10:10000",
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_build_worker_bootstrap_script_requires_controller_address() -> None:
    cfg = _worker_config()
    cfg.controller_address = ""

    with pytest.raises(ValueError, match="controller_address"):
        build_worker_bootstrap_script(cfg)


def test_bootstrap_renders_versioned_runsc_url() -> None:
    """Every gVisor URL in the bootstrap must use the numeric release path:
    the GCS layout is releases/release/<YYYYMMDD.P>/, and a URL built from the
    release-<version> tag name 404s, leaving workers without the runtime."""
    script = build_worker_bootstrap_script(_worker_config())
    urls = re.findall(r"https://storage\.googleapis\.com/gvisor/\S+", script)
    assert urls, "bootstrap no longer downloads runsc"
    for url in urls:
        assert re.match(r"https://storage\.googleapis\.com/gvisor/releases/release/\d{8}\.\d+/", url), url


def test_bootstrap_keeps_task_port_range_below_ephemeral_floor() -> None:
    """Task named ports must be excluded from kernel ephemeral assignment: a
    task binds its allocated port only after container setup, and an outbound
    socket handed that port in the window kills the task with EADDRINUSE
    (#7392). The default range sits below the ephemeral floor, and the
    reservation list pins it as defense-in-depth."""
    script = build_worker_bootstrap_script(_worker_config())
    assert 'sysctl -w net.ipv4.ip_local_port_range="14000 65535"' in script
    assert 'sysctl -w net.ipv4.ip_local_reserved_ports="8081,8431,8470-8482,12000-13999"' in script


def test_bootstrap_reserves_configured_task_port_range() -> None:
    """A cluster overriding port_range into the ephemeral span is still
    protected by the reservation (end bound exclusive, like PortAllocator)."""
    script = build_worker_bootstrap_script(_worker_config(port_range="20000-25000"))
    assert 'sysctl -w net.ipv4.ip_local_reserved_ports="8081,8431,8470-8482,20000-24999"' in script


def test_bootstrap_rejects_malformed_task_port_range() -> None:
    with pytest.raises(ValueError):
        build_worker_bootstrap_script(_worker_config(port_range="all-of-them"))


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


# --- GcpWorkerProvider.resolve_image() tests ---


def _make_gcp_worker_provider(project_id: str = "my-proj"):
    """Build a GcpWorkerProvider backed by InMemoryGcpService for testing."""

    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id=project_id)
    gcp_config = GcpPlatformConfig(project_id=project_id)
    return GcpWorkerProvider(gcp_config=gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)


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


def test_gcp_provider_resolve_image_requires_zone_for_ghcr() -> None:
    """GcpWorkerProvider.resolve_image() raises when zone is missing for GHCR images."""
    provider = _make_gcp_worker_provider()

    with pytest.raises(ValueError, match="zone is required"):
        provider.resolve_image("ghcr.io/org/img:v1")
