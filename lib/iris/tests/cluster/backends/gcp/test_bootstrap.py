# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker bootstrap script generation."""

import re

import pytest
from iris.cluster.config import GcpPlatformConfig, WorkerConfig
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.worker_bootstrap import (
    build_worker_bootstrap_script,
    docker_hub_repo_path,
    render_template,
    rewrite_image_to_mirror,
    upstream_registry,
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


# registry_mirrors map mirroring the shape committed in config/marin.yaml.
_MIRRORS = {
    "ghcr.io": {
        "us": "us-docker.pkg.dev/hai-gcp-models/ghcr-mirror",
        "europe": "europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror",
    },
    "docker.io": {
        "us": "us-docker.pkg.dev/hai-gcp-models/docker-mirror",
        "europe": "europe-docker.pkg.dev/hai-gcp-models/docker-mirror",
    },
}


@pytest.mark.parametrize(
    "image_tag, expected",
    [
        # Docker Hub aliases all collapse to the canonical docker.io key.
        ("ubuntu:24.04", "docker.io"),
        ("bitnami/redis:latest", "docker.io"),
        ("docker.io/library/python:3.12", "docker.io"),
        ("index.docker.io/tensorflow/tensorflow:latest", "docker.io"),
        ("registry-1.docker.io/library/nginx:stable", "docker.io"),
        # Everything else keys on its literal host.
        ("ghcr.io/marin-community/iris-worker:v1", "ghcr.io"),
        ("gcr.io/proj/img:v1", "gcr.io"),
        ("localhost:5000/img:dev", "localhost:5000"),
    ],
)
def test_upstream_registry(image_tag: str, expected: str) -> None:
    assert upstream_registry(image_tag) == expected


@pytest.mark.parametrize(
    "image_tag, expected",
    [
        # Bare official image: gains the implicit library/ namespace.
        ("ubuntu:24.04", "library/ubuntu:24.04"),
        ("python", "library/python"),
        # Namespaced Docker Hub image: kept as-is.
        ("bitnami/redis:latest", "bitnami/redis:latest"),
        # Explicit docker.io / index.docker.io prefixes.
        ("docker.io/library/python:3.12", "library/python:3.12"),
        ("index.docker.io/tensorflow/tensorflow:latest", "tensorflow/tensorflow:latest"),
        ("docker.io/nginx:stable", "library/nginx:stable"),
        # Other registries are not Docker Hub.
        ("gcr.io/proj/img:v1", None),
        ("ghcr.io/marin-community/iris-worker:v1", None),
        ("us-docker.pkg.dev/hai-gcp-models/docker-mirror/library/ubuntu:24.04", None),
        ("localhost:5000/img:dev", None),
    ],
)
def test_docker_hub_repo_path(image_tag: str, expected: str | None) -> None:
    assert docker_hub_repo_path(image_tag) == expected


@pytest.mark.parametrize(
    "image_tag, zone, expected",
    [
        (
            "ghcr.io/marin-community/iris-worker:v1",
            "us-central1-a",
            "us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-worker:v1",
        ),
        (
            "ghcr.io/marin-community/iris-controller:latest",
            "europe-west4-b",
            "europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror/marin-community/iris-controller:latest",
        ),
        (
            "ubuntu:24.04",
            "us-central1-a",
            "us-docker.pkg.dev/hai-gcp-models/docker-mirror/library/ubuntu:24.04",
        ),
        (
            "bitnami/redis:latest",
            "europe-west4-b",
            "europe-docker.pkg.dev/hai-gcp-models/docker-mirror/bitnami/redis:latest",
        ),
        (
            "docker.io/library/python:3.12",
            "us-west4-b",
            "us-docker.pkg.dev/hai-gcp-models/docker-mirror/library/python:3.12",
        ),
    ],
)
def test_rewrite_image_to_mirror(image_tag: str, zone: str, expected: str) -> None:
    assert rewrite_image_to_mirror(image_tag, zone, _MIRRORS) == expected


def test_rewrite_image_to_mirror_unlisted_registry_passthrough() -> None:
    assert rewrite_image_to_mirror("gcr.io/proj/img:v1", "us-central1-a", _MIRRORS) == "gcr.io/proj/img:v1"
    # An already-rewritten reference keys on the AR host, which is unlisted, so
    # a second pass is a no-op instead of stacking mirror prefixes.
    mirrored = rewrite_image_to_mirror("ubuntu:24.04", "us-central1-a", _MIRRORS)
    assert rewrite_image_to_mirror(mirrored, "us-central1-a", _MIRRORS) == mirrored


def test_rewrite_image_to_mirror_unlisted_zone_prefix_passthrough() -> None:
    assert rewrite_image_to_mirror("ubuntu:24.04", "asia-east1-a", _MIRRORS) == "ubuntu:24.04"
    assert rewrite_image_to_mirror("ghcr.io/org/img:v1", "me-west1-a", _MIRRORS) == "ghcr.io/org/img:v1"


# --- GcpWorkerProvider.resolve_image() tests ---


def _make_gcp_worker_provider(registry_mirrors: dict[str, dict[str, str]] | None = None):
    """Build a GcpWorkerProvider backed by InMemoryGcpService for testing."""

    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="my-proj")
    gcp_config = GcpPlatformConfig(project_id="my-proj", registry_mirrors=registry_mirrors or {})
    return GcpWorkerProvider(gcp_config=gcp_config, label_prefix="iris", worker_port=10001, gcp_service=gcp_service)


def test_gcp_provider_resolve_image_routes_by_registry_mirrors() -> None:
    """resolve_image() rewrites each mirrored upstream for the zone's continent."""
    provider = _make_gcp_worker_provider(_MIRRORS)

    assert provider.resolve_image("ghcr.io/org/img:v1", zone="us-central1-a") == (
        "us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/org/img:v1"
    )
    assert provider.resolve_image("ghcr.io/org/img:v1", zone="europe-west4-b") == (
        "europe-docker.pkg.dev/hai-gcp-models/ghcr-mirror/org/img:v1"
    )
    assert provider.resolve_image("ubuntu:24.04", zone="us-central1-a") == (
        "us-docker.pkg.dev/hai-gcp-models/docker-mirror/library/ubuntu:24.04"
    )
    assert provider.resolve_image("bitnami/redis:latest", zone="europe-west4-b") == (
        "europe-docker.pkg.dev/hai-gcp-models/docker-mirror/bitnami/redis:latest"
    )
    # Other registries (incl. Artifact Registry) pass through untouched.
    assert provider.resolve_image("gcr.io/proj/img:v1", zone="us-central1-a") == "gcr.io/proj/img:v1"


def test_gcp_provider_resolve_image_passthrough_without_mirrors() -> None:
    """With no registry_mirrors configured, every image pulls straight from its upstream."""
    provider = _make_gcp_worker_provider()

    assert provider.resolve_image("ghcr.io/org/img:v1", zone="us-central1-a") == "ghcr.io/org/img:v1"
    assert provider.resolve_image("ubuntu:24.04", zone="us-central1-a") == "ubuntu:24.04"
    # Unmirrored images need no zone at all.
    assert provider.resolve_image("ghcr.io/org/img:v1") == "ghcr.io/org/img:v1"


def test_gcp_provider_resolve_image_requires_zone_for_mirrored_upstream() -> None:
    """A mirrored upstream needs a zone to pick the continent."""
    provider = _make_gcp_worker_provider(_MIRRORS)

    with pytest.raises(ValueError, match="zone is required"):
        provider.resolve_image("ghcr.io/org/img:v1")
    with pytest.raises(ValueError, match="zone is required"):
        provider.resolve_image("ubuntu:24.04")
