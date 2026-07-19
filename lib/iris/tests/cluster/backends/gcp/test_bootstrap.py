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
    rewrite_docker_hub_to_ar_remote,
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
    "image_tag, multi_region, project, expected",
    [
        (
            "ubuntu:24.04",
            "us",
            "hai-gcp-models",
            "us-docker.pkg.dev/hai-gcp-models/docker-mirror/library/ubuntu:24.04",
        ),
        (
            "bitnami/redis:latest",
            "europe",
            "hai-gcp-models",
            "europe-docker.pkg.dev/hai-gcp-models/docker-mirror/bitnami/redis:latest",
        ),
        (
            "docker.io/library/python:3.12",
            "us",
            "my-project",
            "us-docker.pkg.dev/my-project/docker-mirror/library/python:3.12",
        ),
    ],
)
def test_rewrite_docker_hub_to_ar_remote(image_tag: str, multi_region: str, project: str, expected: str) -> None:
    assert rewrite_docker_hub_to_ar_remote(image_tag, multi_region, project, "docker-mirror") == expected


def test_rewrite_docker_hub_to_ar_remote_other_registry_passthrough() -> None:
    assert rewrite_docker_hub_to_ar_remote("gcr.io/proj/img:v1", "us", "proj", "docker-mirror") == "gcr.io/proj/img:v1"
    assert rewrite_docker_hub_to_ar_remote("ghcr.io/org/img:v1", "us", "proj", "docker-mirror") == "ghcr.io/org/img:v1"


def test_rewrite_ghcr_to_ar_remote_custom_mirror_repo() -> None:
    result = rewrite_ghcr_to_ar_remote("ghcr.io/org/image:v1", "us", "proj", mirror_repo="custom-mirror")
    assert result == "us-docker.pkg.dev/proj/custom-mirror/org/image:v1"


# --- GcpWorkerProvider.resolve_image() tests ---


def _make_gcp_worker_provider(project_id: str = "my-proj", docker_hub_mirror_repo: str = ""):
    """Build a GcpWorkerProvider backed by InMemoryGcpService for testing."""

    gcp_service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id=project_id)
    gcp_config = GcpPlatformConfig(project_id=project_id, docker_hub_mirror_repo=docker_hub_mirror_repo)
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
    """Without a docker_hub_mirror_repo, Docker Hub images pull straight from Docker Hub."""
    provider = _make_gcp_worker_provider()

    assert provider.resolve_image("docker.io/library/ubuntu:latest", zone="us-central1-a") == (
        "docker.io/library/ubuntu:latest"
    )
    assert provider.resolve_image("ubuntu:24.04", zone="us-central1-a") == "ubuntu:24.04"


def test_gcp_provider_resolve_image_requires_zone_for_ghcr() -> None:
    """GcpWorkerProvider.resolve_image() raises when zone is missing for GHCR images."""
    provider = _make_gcp_worker_provider()

    with pytest.raises(ValueError, match="zone is required"):
        provider.resolve_image("ghcr.io/org/img:v1")


def test_gcp_provider_resolve_image_rewrites_docker_hub_when_configured() -> None:
    """With docker_hub_mirror_repo set, Docker Hub images route through the regional AR cache."""
    provider = _make_gcp_worker_provider(docker_hub_mirror_repo="docker-mirror")

    assert provider.resolve_image("ubuntu:24.04", zone="us-central1-a") == (
        "us-docker.pkg.dev/my-proj/docker-mirror/library/ubuntu:24.04"
    )
    assert provider.resolve_image("bitnami/redis:latest", zone="europe-west4-b") == (
        "europe-docker.pkg.dev/my-proj/docker-mirror/bitnami/redis:latest"
    )
    # ghcr still routes through ghcr-mirror even when the Docker Hub mirror is on.
    assert provider.resolve_image("ghcr.io/org/img:v1", zone="us-central1-a") == (
        "us-docker.pkg.dev/my-proj/ghcr-mirror/org/img:v1"
    )
    # Other registries (incl. Artifact Registry) still pass through untouched.
    assert provider.resolve_image("gcr.io/proj/img:v1", zone="us-central1-a") == "gcr.io/proj/img:v1"


def test_gcp_provider_resolve_image_requires_zone_for_docker_hub_when_configured() -> None:
    """A configured Docker Hub mirror needs a zone to pick the continent, like GHCR."""
    provider = _make_gcp_worker_provider(docker_hub_mirror_repo="docker-mirror")

    with pytest.raises(ValueError, match="zone is required"):
        provider.resolve_image("ubuntu:24.04")
