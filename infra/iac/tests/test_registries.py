# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact Registry mirror provisioning — upstream dispatch, cleanup policies, and component wiring.

Guards the GCP-arm pull-through caches: ghcr.io routes through a custom-URI remote repo and Docker
Hub through GCP's predefined public repo, each created per multi-region with age/keep cleanup.
"""

import pulumi
import pulumi.runtime
from iac.config import (
    DEFAULT_MIRROR_CLEANUP_POLICIES,
    DOCKER_HUB_UPSTREAM,
    GcpRemoteRepositorySpec,
    load_iris_config,
    load_provisioning,
)
from iac.gcp.registries import (
    GcpArtifactRegistries,
    GcpArtifactRegistriesArgs,
    _cleanup_policy,
    _import_id,
    _remote_config,
)


def test_marin_provisioning_declares_both_mirrors() -> None:
    """The committed marin stack encodes the ghcr + docker mirrors the workers rewrite to."""
    gcp = load_provisioning("marin").gcp
    assert gcp is not None
    by_name = {r.name: r for r in gcp.registries}
    assert by_name["ghcr-mirror"].docker_upstream == "https://ghcr.io"
    assert by_name["docker-mirror"].docker_upstream == DOCKER_HUB_UPSTREAM
    assert by_name["ghcr-mirror"].locations == ["us", "europe"]
    assert by_name["docker-mirror"].locations == ["us", "europe"]


def test_registry_mirrors_route_to_provisioned_repos() -> None:
    """Every runtime mirror target in the GCP cluster configs is a repo this IaC provisions.

    platform.gcp.registry_mirrors routes worker image pulls; provisioning.gcp.registries
    (marin.yaml) declares the repos. A mirror entry naming an undeclared repo or location,
    or routing a registry to a repo that proxies a different upstream, would 404 at pull.
    """
    provisioning = load_provisioning("marin").gcp
    assert provisioning is not None
    provisioned = {(r.name, loc): r.docker_upstream for r in provisioning.registries for loc in r.locations}

    for cluster in ("marin", "marin-dev", "ci-gcp-smoke"):
        gcp = load_iris_config(cluster).platform.gcp
        assert gcp is not None and gcp.registry_mirrors, f"{cluster} routes no registries through the mirrors"
        for upstream_host, zone_map in gcp.registry_mirrors.items():
            for zone_prefix, target in zone_map.items():
                location, _, remainder = target.partition("-docker.pkg.dev/")
                project, _, repo = remainder.partition("/")
                assert project == "hai-gcp-models", target
                # AR multi-region caches serve their own continent's zones.
                assert zone_prefix == location, target
                expected = DOCKER_HUB_UPSTREAM if upstream_host == "docker.io" else f"https://{upstream_host}"
                assert provisioned.get((repo, location)) == expected, (cluster, upstream_host, target)


def test_default_cleanup_policies_applied_when_unspecified() -> None:
    spec = GcpRemoteRepositorySpec(name="docker-mirror", docker_upstream=DOCKER_HUB_UPSTREAM, locations=["us"])
    delete, keep = spec.cleanup_policies
    assert delete.action == "DELETE" and delete.older_than == "2592000s"
    assert keep.action == "KEEP" and keep.keep_count == 16


def test_remote_config_docker_hub_uses_predefined_public_repo() -> None:
    spec = GcpRemoteRepositorySpec(name="docker-mirror", docker_upstream=DOCKER_HUB_UPSTREAM, locations=["us"])
    docker = _remote_config(spec).docker_repository
    assert docker.public_repository == "DOCKER_HUB"
    assert docker.custom_repository is None


def test_remote_config_custom_url_uses_custom_repo() -> None:
    spec = GcpRemoteRepositorySpec(name="ghcr-mirror", docker_upstream="https://ghcr.io", locations=["us"])
    docker = _remote_config(spec).docker_repository
    assert docker.public_repository is None
    assert docker.custom_repository.uri == "https://ghcr.io"


def test_cleanup_policy_delete_carries_age_condition() -> None:
    delete = _cleanup_policy(DEFAULT_MIRROR_CLEANUP_POLICIES[0])
    assert delete.action == "DELETE"
    assert delete.condition.older_than == "2592000s"
    assert delete.condition.tag_state == "ANY"
    assert delete.most_recent_versions is None


def test_cleanup_policy_keep_carries_version_count() -> None:
    keep = _cleanup_policy(DEFAULT_MIRROR_CLEANUP_POLICIES[1])
    assert keep.action == "KEEP"
    assert keep.most_recent_versions.keep_count == 16
    assert keep.condition is None


def test_import_id_is_regional_repository_path() -> None:
    assert _import_id("hai-gcp-models", "europe", "docker-mirror") == (
        "projects/hai-gcp-models/locations/europe/repositories/docker-mirror"
    )


class _RecordingMocks(pulumi.runtime.Mocks):
    """Capture every resource the program registers so a test can assert on the graph it built."""

    def __init__(self) -> None:
        self.resources: list[tuple[str, str, dict]] = []

    def new_resource(self, args: pulumi.runtime.MockResourceArgs):
        self.resources.append((args.typ, args.name, args.inputs))
        return f"{args.name}-id", args.inputs

    def call(self, args: pulumi.runtime.MockCallArgs):
        return {}


def test_component_creates_one_remote_repo_per_registry_and_location() -> None:
    """Two mirrors x two continents = four REMOTE_REPOSITORY Docker repos with valid kwargs."""
    mocks = _RecordingMocks()
    pulumi.runtime.set_mocks(mocks, preview=False)

    # @pulumi.runtime.test drives the resource-registration event loop to completion, so the
    # mock's `resources` list is fully populated once the decorated program returns. Asserting
    # inside a resource `.apply()` would instead fire before the child repos finish registering.
    @pulumi.runtime.test
    def _program():
        GcpArtifactRegistries(
            "registries",
            GcpArtifactRegistriesArgs(
                project="hai-gcp-models",
                registries=[
                    GcpRemoteRepositorySpec(
                        name="ghcr-mirror", docker_upstream="https://ghcr.io", locations=["us", "europe"]
                    ),
                    GcpRemoteRepositorySpec(
                        name="docker-mirror", docker_upstream=DOCKER_HUB_UPSTREAM, locations=["us", "europe"]
                    ),
                ],
            ),
            gcp_provider=pulumi.ProviderResource("pulumi:providers:gcp", "gcp-test", {}),
        )

    _program()

    repos = [
        (name, inputs) for typ, name, inputs in mocks.resources if typ == "gcp:artifactregistry/repository:Repository"
    ]
    assert {name for name, _ in repos} == {
        "repo-ghcr-mirror-us",
        "repo-ghcr-mirror-europe",
        "repo-docker-mirror-us",
        "repo-docker-mirror-europe",
    }
    for _, inputs in repos:
        assert inputs["mode"] == "REMOTE_REPOSITORY"
        assert inputs["format"] == "DOCKER"
