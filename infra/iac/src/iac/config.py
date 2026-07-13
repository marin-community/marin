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


class IngressSpec(BaseModel):
    """Traefik + cert-manager + ACME issuers (TraefikAddon)."""

    ingress_class: str = "traefik"
    acme_email: str
    cluster_issuers: list[str]


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


class ProvisioningConfig(BaseModel):
    """Top-level `provisioning:` section. Exactly one provider block is populated."""

    provider: Provider
    coreweave: CoreweaveProvisioning | None = None
    # gcp: GcpProvisioning | None = None  # ported after CoreWeave proves the pattern


def _validate_provider_block(provisioning: ProvisioningConfig) -> ProvisioningConfig:
    """Reject a provisioning section whose selected provider has no matching block."""
    if provisioning.provider is Provider.COREWEAVE and provisioning.coreweave is None:
        raise ValueError("provisioning.provider is 'coreweave' but no 'coreweave:' block is present")
    if provisioning.provider is Provider.GCP:
        raise ValueError("provisioning.provider 'gcp' is not yet supported by iac")
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
