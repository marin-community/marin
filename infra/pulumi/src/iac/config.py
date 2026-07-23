# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed `provisioning:` schema for Marin IaC.

The `provisioning:` section lives in the per-cluster Iris config
(`lib/iris/config/<cluster>.yaml`). Iris carries it as an opaque dict
(`IrisClusterConfig.provisioning`); this module owns the typed validation, so there
is no reverse dependency from Iris onto the IaC schema.

Single-source rules: values already present in the Iris config are NOT repeated here.
Namespace derives from `kubernetes_provider.namespace`, the Kueue ClusterQueue name from
`kubernetes_provider.kueue.cluster_queue`, and NodePools from `scale_groups` (see
`iac.nodepools`). `provisioning.coreweave` carries only the residual cluster facts.
"""

import enum
from enum import StrEnum
from typing import Annotated, Literal

from iris.cluster.config import IrisClusterConfig, load_config
from iris.cluster.platforms.k8s.kueue_manifests import (
    DEFAULT_CLIENT_CONNECTION_BURST,
    DEFAULT_CLIENT_CONNECTION_QPS,
)
from iris.cluster.platforms.k8s.network_manifests import DEFAULT_CLUSTER_ISSUER
from pydantic import BaseModel, Field
from rigging.config_discovery import resolve_cluster_config

# IaC reads only the reviewed, in-tree cluster config — deliberately NOT Iris's runtime
# search path (IRIS_CLUSTER_CONFIG_DIRS), which checks the operator override
# ~/.config/marin/clusters *first*. A `pulumi preview/up` plan must derive its NodePools,
# Kueue/RBAC, and GCP addresses from the committed stack inputs so the production plan is
# reproducible and review-gated, never from a private local override. Relative to the marin
# project root (resolved by rigging.config_discovery).
IAC_CLUSTER_CONFIG_DIR = "lib/iris/config"


class Provider(StrEnum):
    COREWEAVE = enum.auto()
    GCP = enum.auto()


class CksClusterSpec(BaseModel):
    """The CoreWeave CKS cluster object (coreweave_cks_cluster)."""

    name: str
    zone: str


class KueueClientConnectionSpec(BaseModel):
    """Kubernetes API client rate limit for the Kueue controller-manager."""

    qps: float = Field(default=DEFAULT_CLIENT_CONNECTION_QPS, gt=0)
    burst: int = Field(default=DEFAULT_CLIENT_CONNECTION_BURST, gt=0)


class KueueProvisioningSpec(BaseModel):
    """Cluster-scoped Kueue objects owned by IaC (KueueAddon).

    The ResourceFlavor name and the Topology set are canonical constants in
    iris.cluster.platforms.k8s.kueue_manifests (shared with install_kueue.py so IaC and the
    script render identically), not per-cluster knobs. cluster_queue derives from the Iris
    config. The flavor→topology binding and controller-manager resource overrides are the
    cluster-specific values that live here.
    """

    # Which topology the ResourceFlavor binds (spec.topologyName). NVL72 clusters bind
    # `multinode-nvlink-ib` to expose the nvlink.domain level; IB clusters bind `infiniband`.
    flavor_topology: str = "infiniband"
    # Override controllerManager.manager.resources' memory (requests == limits)
    manager_memory_limit: str | None = None
    # Override Iris's shared Kueue client-side API rate limit.
    client_connection: KueueClientConnectionSpec = Field(default_factory=KueueClientConnectionSpec)


# Egress addresses of the marin-side controllers that federate into every CoreWeave cluster
# (reserved as iris-marin-fed-egress / iris-marin-dev-fed-egress in project hai-gcp-models).
# The federation ingress admits only these; the set is the same for every CW cluster. Modeled
# as a plain input with these constants for now rather than as a GCP address reservation that
# the GCP arm looks up (that consolidation is deferred). Mirrors FEDERATION_ALLOW_SOURCES in
# lib/iris/scripts/install_cw_network.py — keep the two in sync until that script's own copy is
# deleted (see README.md's "Future work").
MARIN_FEDERATION_EGRESS_SOURCES = ["34.27.183.11", "35.254.13.19"]
CLOUDFLARE_OA_DEV_ZONE_ID = "169959d6aafcbfd77764b8efafa3a509"


class IngressSpec(BaseModel):
    """Traefik + cert-manager + ACME issuers, and the IP-locked federation route (TraefikAddon)."""

    ingress_class: str = "traefik"
    acme_email: str
    cluster_issuers: list[str]
    # Which of cluster_issuers annotates the federation Ingress right now
    # (cert-manager.io/cluster-issuer). install_cw_network.py defaults a fresh install to
    # staging to avoid Let's Encrypt rate limits while DNS/allowlist are shaken out; flip to
    # prod here once that cert validates (matches the script's --cluster-issuer flag).
    active_cluster_issuer: str = DEFAULT_CLUSTER_ISSUER
    # Sources allowed through the federation ingress ipAllowList. A Pulumi input to the
    # (deferred) federation-ingress component; constant default covers every CW cluster.
    federation_allow_sources: list[str] = Field(default_factory=lambda: list(MARIN_FEDERATION_EGRESS_SOURCES))


class FederationDnsSpec(BaseModel):
    """Cloudflare CNAME for the CoreWeave federation ingress."""

    zone_id: str = CLOUDFLARE_OA_DEV_ZONE_ID
    hostname: str
    target: str


class RbacSpec(BaseModel):
    """Controller RBAC ceded from ensure_rbac(). namespace derives from the Iris config."""

    service_account: str = "iris-controller"


class CoreweaveProvisioning(BaseModel):
    cluster: CksClusterSpec
    kueue: KueueProvisioningSpec
    ingress: IngressSpec
    federation_dns: FederationDnsSpec | None = None
    rbac: RbacSpec = RbacSpec()


class GcpAddressSpec(BaseModel):
    """A reserved external static IP (google_compute_address)."""

    name: str  # e.g. "iris-marin-fed-egress"
    region: str  # e.g. "us-central1"
    address: str  # the pinned IP, e.g. "34.27.183.11"
    # description is immutable on a compute Address (any change forces replacement, which
    # releases the pinned IP), so it must match the live reservation exactly for adoption to
    # be a no-op. None => the live reservation has no description.
    description: str | None = None


class GcpDeleteCleanupPolicy(BaseModel):
    """A DELETE cleanup policy: prune versions whose age exceeds a threshold."""

    id: str
    action: Literal["DELETE"] = "DELETE"
    older_than: str  # Go duration string, e.g. "2592000s" = 30d
    tag_state: str = "ANY"  # which tag states the condition matches ("ANY", "TAGGED", "UNTAGGED")


class GcpKeepCleanupPolicy(BaseModel):
    """A KEEP cleanup policy: retain the most-recent N versions regardless of age."""

    id: str
    action: Literal["KEEP"] = "KEEP"
    keep_count: int


# `action` discriminates two disjoint field sets (older_than/tag_state vs. keep_count); the
# discriminated union rejects a policy that carries fields from the wrong arm instead of
# silently ignoring them.
GcpArtifactRegistryCleanupPolicy = Annotated[
    GcpDeleteCleanupPolicy | GcpKeepCleanupPolicy, Field(discriminator="action")
]


# Default for mirrors of versioned image streams (the iris images on ghcr): drop versions cached
# more than 30 days ago (older_than counts from caching time; an evicted version is re-fetched on
# the next pull), keeping the 16 newest versions of each package so the current release train
# stays warm. Repos caching a few mutable base tags override cleanup_policies with a plain short
# TTL instead — with so few versions per package, a keep floor would protect them all
# indefinitely (see docker-mirror in marin.yaml).
DEFAULT_MIRROR_CLEANUP_POLICIES = [
    GcpDeleteCleanupPolicy(id="delete-older-than-30d", older_than="2592000s"),
    GcpKeepCleanupPolicy(id="keep-latest", keep_count=16),
]

# Sentinel `docker_upstream` value selecting GCP's predefined Docker Hub upstream rather than a
# custom registry URL. Matches the `DOCKER-HUB` keyword of `gcloud ... --remote-docker-repo`.
DOCKER_HUB_UPSTREAM = "DOCKER_HUB"


class GcpRemoteRepositorySpec(BaseModel):
    """A Docker Artifact Registry remote repository (pull-through cache) across multi-regions.

    ``docker_upstream`` selects the upstream: the sentinel ``"DOCKER_HUB"`` uses GCP's predefined
    Docker Hub public repository; any other value is a custom registry URL (e.g. ``"https://ghcr.io"``).
    The repo is created once per entry in ``locations`` (``us``, ``europe``, …) so a worker pulls
    from its own continent. See lib/iris/docs/image-push.md for how workers rewrite image tags to it.
    """

    name: str  # repository_id, e.g. "docker-mirror"
    docker_upstream: str  # "DOCKER_HUB" or a URL like "https://ghcr.io"
    locations: list[str] = Field(default_factory=list)  # multi-regions, e.g. ["us", "europe"]
    description: str = ""
    cleanup_policies: list[GcpArtifactRegistryCleanupPolicy] = Field(
        default_factory=lambda: list(DEFAULT_MIRROR_CLEANUP_POLICIES)
    )


class GcpProvisioning(BaseModel):
    """GCP-arm provisioning: the project, reserved static IPs, and Artifact Registry mirrors."""

    project: str
    addresses: list[GcpAddressSpec] = Field(default_factory=list)
    registries: list[GcpRemoteRepositorySpec] = Field(default_factory=list)


class ProvisioningConfig(BaseModel):
    """Top-level `provisioning:` section. Exactly one provider block is populated."""

    provider: Provider
    coreweave: CoreweaveProvisioning | None = None
    gcp: GcpProvisioning | None = None


def _validate_provider_block(provisioning: ProvisioningConfig) -> ProvisioningConfig:
    """Reject a provisioning section whose selected provider has no matching block."""
    if provisioning.provider is Provider.COREWEAVE and provisioning.coreweave is None:
        raise ValueError("provisioning.provider is 'coreweave' but no 'coreweave:' block is present")
    if provisioning.provider is Provider.GCP and provisioning.gcp is None:
        raise ValueError("provisioning.provider is 'gcp' but no 'gcp:' block is present")
    return provisioning


def load_iris_config(cluster: str) -> IrisClusterConfig:
    """Load the per-cluster Iris config from the reviewed in-tree config dir.

    Deliberately does NOT use Iris's runtime search path (IRIS_CLUSTER_CONFIG_DIRS): that
    checks ~/.config/marin/clusters first, so an operator override could feed `pulumi
    preview/up` inputs that never went through review. IaC plans read only the committed
    config (see IAC_CLUSTER_CONFIG_DIR).
    """
    return load_config(resolve_cluster_config(cluster, dirs=(IAC_CLUSTER_CONFIG_DIR,)))


def load_provisioning(cluster: str) -> ProvisioningConfig:
    """Load and validate the `provisioning:` section of lib/iris/config/<cluster>.yaml.

    Reads the same file the Iris config loader reads (Iris carries `provisioning:` as an
    opaque dict). Raises pydantic.ValidationError on a malformed section, ValueError if the
    section is absent or the selected provider block is missing.
    """
    iris_config = load_iris_config(cluster)
    if iris_config.provisioning is None:
        raise ValueError(f"cluster {cluster!r} has no `provisioning:` section in its Iris config")
    return _validate_provider_block(ProvisioningConfig.model_validate(iris_config.provisioning))
