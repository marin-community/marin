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

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.cluster.config import IrisClusterConfig, load_config
from pydantic import BaseModel, Field
from rigging.config_discovery import resolve_cluster_config


class Provider(StrEnum):
    COREWEAVE = enum.auto()
    GCP = enum.auto()


class CksClusterSpec(BaseModel):
    """The CoreWeave CKS cluster object (coreweave_cks_cluster)."""

    name: str
    zone: str
    vpc: str | None = None
    import_existing: bool = False


class KueueProvisioningSpec(BaseModel):
    """Cluster-scoped Kueue objects owned by IaC (KueueAddon).

    cluster_queue and pod_namespace are NOT here — they derive from the Iris config.
    """

    resource_flavor: str
    topologies: list[str]
    # Which topology the ResourceFlavor binds (spec.topologyName). NVL72 clusters bind
    # `multinode-nvlink-ib` to expose the nvlink.domain level; IB clusters bind `infiniband`.
    flavor_topology: str = "infiniband"


# Egress addresses of the marin-side controllers that federate into every CoreWeave cluster
# (reserved as iris-marin-fed-egress / iris-marin-dev-fed-egress in project hai-gcp-models).
# The federation ingress admits only these; the set is the same for every CW cluster. Modeled
# as a plain input with these constants for now rather than as a GCP address reservation that
# the GCP arm looks up (that consolidation is deferred). Mirrors FEDERATION_ALLOW_SOURCES in
# lib/iris/scripts/install_cw_network.py — keep the two in sync until the federation-ingress
# component reads this field and that constant is deleted (see .agents/projects/iac/gaps.md).
MARIN_FEDERATION_EGRESS_SOURCES = ["34.27.183.11", "35.254.13.19"]


class IngressSpec(BaseModel):
    """Traefik + cert-manager + ACME issuers, and the IP-locked federation route (TraefikAddon)."""

    ingress_class: str = "traefik"
    acme_email: str
    cluster_issuers: list[str]
    # Sources allowed through the federation ingress ipAllowList. A Pulumi input to the
    # (deferred) federation-ingress component; constant default covers every CW cluster.
    federation_allow_sources: list[str] = Field(default_factory=lambda: list(MARIN_FEDERATION_EGRESS_SOURCES))


class RbacSpec(BaseModel):
    """Controller RBAC ceded from ensure_rbac(). namespace derives from the Iris config."""

    service_account: str = "iris-controller"


class BucketSpec(BaseModel):
    name: str
    region: str


class ObjectStorageSpec(BaseModel):
    """Buckets + access key(s). Bucket lifecycle rules are out of scope (configure_buckets.py)."""

    buckets: list[BucketSpec] = Field(default_factory=list)
    access_key_secret_ref: str | None = None


class CoreweaveProvisioning(BaseModel):
    region: str
    cluster: CksClusterSpec
    kueue: KueueProvisioningSpec
    ingress: IngressSpec
    rbac: RbacSpec = RbacSpec()
    object_storage: ObjectStorageSpec = ObjectStorageSpec()


class GcpAddressSpec(BaseModel):
    """A reserved external static IP (google_compute_address)."""

    name: str  # e.g. "iris-marin-fed-egress"
    region: str  # e.g. "us-central1"
    address: str  # the pinned IP, e.g. "34.27.183.11"
    # description is immutable on a compute Address (any change forces replacement, which
    # releases the pinned IP), so it must match the live reservation exactly for adoption to
    # be a no-op. None => the live reservation has no description.
    description: str | None = None


class GcpProvisioning(BaseModel):
    """GCP-arm provisioning: the project and its reserved static IP addresses (GcpStaticAddresses)."""

    project: str
    addresses: list[GcpAddressSpec] = Field(default_factory=list)


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
    """Resolve and load the per-cluster Iris config via the standard iris search path."""
    return load_config(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS))


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
