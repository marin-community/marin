# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact Registry remote repositories (pull-through caches) for the GCP arm.

Declares the Docker pull-through caches Iris workers pull public base images through: the
`ghcr-mirror` (proxies ghcr.io, for iris worker/controller/task images) and the `docker-mirror`
(proxies Docker Hub, for harbor sandbox base images). Each is created once per multi-region so a
worker pulls from its own continent, and each carries cleanup policies that bound cache growth.
Workers rewrite image tags to these repos in `GcpWorkerProvider.resolve_image` — see
lib/iris/docs/image-push.md.
"""

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp

from iac.config import (
    DOCKER_HUB_UPSTREAM,
    GcpArtifactRegistryCleanupPolicy,
    GcpDeleteCleanupPolicy,
    GcpRemoteRepositorySpec,
)

DOCKER_HUB_URI = "https://registry-1.docker.io"


@dataclass(frozen=True)
class GcpArtifactRegistriesArgs:
    project: str
    registries: list[GcpRemoteRepositorySpec]
    # Adoption mode: stamp import_=<repo id> on each resource so `pulumi preview` shows the real
    # adoption diff against the live repos instead of planning creates. Set via `marin-iac:import`.
    adopt: bool = False


def _import_id(project: str, location: str, repo_name: str) -> str:
    # google_artifact_registry_repository import id.
    return f"projects/{project}/locations/{location}/repositories/{repo_name}"


def remote_repository_config(
    spec: GcpRemoteRepositorySpec,
) -> gcp.artifactregistry.RepositoryRemoteRepositoryConfigArgs:
    upstream_uri = DOCKER_HUB_URI if spec.docker_upstream == DOCKER_HUB_UPSTREAM else spec.docker_upstream
    return gcp.artifactregistry.RepositoryRemoteRepositoryConfigArgs(
        common_repository=gcp.artifactregistry.RepositoryRemoteRepositoryConfigCommonRepositoryArgs(uri=upstream_uri)
    )


def _cleanup_policy(policy: GcpArtifactRegistryCleanupPolicy) -> gcp.artifactregistry.RepositoryCleanupPolicyArgs:
    """Translate a cleanup-policy spec into its Artifact Registry arg (DELETE by age, KEEP N newest)."""
    if isinstance(policy, GcpDeleteCleanupPolicy):
        condition = gcp.artifactregistry.RepositoryCleanupPolicyConditionArgs(
            tag_state=policy.tag_state, older_than=policy.older_than
        )
        most_recent = None
    else:
        condition = None
        most_recent = gcp.artifactregistry.RepositoryCleanupPolicyMostRecentVersionsArgs(keep_count=policy.keep_count)
    return gcp.artifactregistry.RepositoryCleanupPolicyArgs(
        id=policy.id, action=policy.action, condition=condition, most_recent_versions=most_recent
    )


class GcpArtifactRegistries(pulumi.ComponentResource):
    """Create one Docker remote-repository (pull-through cache) per (registry, location)."""

    def __init__(
        self,
        name: str,
        args: GcpArtifactRegistriesArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:gcp:GcpArtifactRegistries", name, None, opts)
        for spec in args.registries:
            for location in spec.locations:
                gcp.artifactregistry.Repository(
                    f"repo-{spec.name}-{location}",
                    project=args.project,
                    location=location,
                    repository_id=spec.name,
                    format="DOCKER",
                    mode="REMOTE_REPOSITORY",
                    description=spec.description or None,
                    remote_repository_config=remote_repository_config(spec),
                    cleanup_policies=[_cleanup_policy(p) for p in spec.cleanup_policies],
                    # Enforce the policies (actually delete), not just report a dry-run plan.
                    cleanup_policy_dry_run=False,
                    opts=pulumi.ResourceOptions(
                        parent=self,
                        provider=gcp_provider,
                        import_=_import_id(args.project, location, spec.name) if args.adopt else None,
                    ),
                )
        self.register_outputs({})
