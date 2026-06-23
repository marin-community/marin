# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The full cluster config: the shared manifest plus the admin-only sections.

``marin_cluster`` sits above iris/finelog, so it is the layer that owns the
sections of ``config/<cluster>.yaml`` no client library reads:

- ``provisioning:`` — the one-time GCP rollout spec (project, zone, service
  accounts, IAP/GCLB resources) that drives ``marin-cluster admin``. Today these
  values are baked into ``setup_iam.py`` / ``iap_gclb.py`` as constants; modeling
  them here is what makes the rollouts config-driven instead of hard-coded.

It composes (does not subclass) rigging's :class:`~rigging.cluster_manifest.ClusterManifest`,
which carries the identity/auth slice every client shares. Secrets in
``provisioning`` are *references* (e.g. ``gcp-secret://…``), never inline values.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rigging.cluster_manifest import ClusterAuth, ClusterManifest, load_manifest

_CURRENT_CLUSTER_POINTER = Path.home() / ".config" / "marin" / "cluster"


@dataclass(frozen=True)
class GcpProvisioning:
    """GCP placement for a cluster's resources."""

    project: str
    default_zone: str
    network: str = "default"


@dataclass(frozen=True)
class IamProvisioning:
    """Service accounts and principals for ``marin-cluster admin iam``.

    Role-sets and required APIs are deliberately *not* here: they are platform
    mechanics (the same for every GCP cluster), so they stay as code constants.
    Only cluster-specific identity lives in config.
    """

    controller_service_account: str
    worker_service_account: str
    ci_principal: str | None = None
    operators: tuple[str, ...] = ()


@dataclass(frozen=True)
class IapGclbProvisioning:
    """The IAP/GCLB front-door rollout for ``marin-cluster admin iap``.

    Resource names (NEG, backend service, URL map, cert, forwarding rule) all
    derive from ``resource_prefix``, matching the ``iris-{cluster}`` convention the
    current script bakes in.
    """

    domain: str
    resource_prefix: str
    controller_port: int = 10000
    discovery_label: str | None = None
    enabled: bool = True
    apply_allow_rule: bool = True
    deny_public_default: bool = False


@dataclass(frozen=True)
class Provisioning:
    """The admin-only rollout spec. ``gcp`` absent ⇒ not a GCP cluster."""

    gcp: GcpProvisioning | None = None
    iam: IamProvisioning | None = None
    iap_gclb: IapGclbProvisioning | None = None


@dataclass(frozen=True)
class ClusterConfig:
    """A cluster's complete config: shared manifest + admin-only provisioning."""

    manifest: ClusterManifest
    provisioning: Provisioning | None = field(default=None)

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def dashboard_url(self) -> str | None:
        return self.manifest.dashboard_url

    @property
    def auth(self) -> ClusterAuth:
        return self.manifest.auth

    @property
    def gcp(self) -> GcpProvisioning:
        """The GCP placement, or raise if this is not a GCP cluster."""
        if self.provisioning is None or self.provisioning.gcp is None:
            raise ValueError(f"cluster {self.name!r} has no provisioning.gcp — not a GCP cluster")
        return self.provisioning.gcp

    @classmethod
    def load(cls, cluster: str | None = None) -> "ClusterConfig":
        """Load and parse the full config for ``cluster`` (see :func:`load_manifest`)."""
        manifest = load_manifest(cluster)
        return cls(manifest=manifest, provisioning=_parse_provisioning(manifest.document.get("provisioning")))


def _parse_provisioning(raw: Mapping[str, Any] | None) -> Provisioning | None:
    if not raw:
        return None
    return Provisioning(
        gcp=_parse_gcp(raw.get("gcp")),
        iam=_parse_iam(raw.get("iam")),
        iap_gclb=_parse_iap_gclb(raw.get("iap_gclb")),
    )


def _parse_gcp(raw: Mapping[str, Any] | None) -> GcpProvisioning | None:
    if not raw:
        return None
    project = raw.get("project")
    default_zone = raw.get("default_zone")
    if not project or not default_zone:
        raise ValueError("provisioning.gcp requires 'project' and 'default_zone'")
    return GcpProvisioning(
        project=str(project),
        default_zone=str(default_zone),
        network=str(raw.get("network") or "default"),
    )


def _parse_iam(raw: Mapping[str, Any] | None) -> IamProvisioning | None:
    if not raw:
        return None
    accounts = raw.get("service_accounts") or {}
    controller = accounts.get("controller")
    worker = accounts.get("worker")
    if not controller or not worker:
        raise ValueError("provisioning.iam.service_accounts requires 'controller' and 'worker'")
    principals = raw.get("principals") or {}
    return IamProvisioning(
        controller_service_account=str(controller),
        worker_service_account=str(worker),
        ci_principal=_opt_str(principals.get("ci")),
        operators=tuple(str(o) for o in principals.get("operators") or ()),
    )


def _parse_iap_gclb(raw: Mapping[str, Any] | None) -> IapGclbProvisioning | None:
    if not raw:
        return None
    domain = raw.get("domain")
    if not domain:
        raise ValueError("provisioning.iap_gclb requires a 'domain'")
    resources = raw.get("resources") or {}
    controller = raw.get("controller") or {}
    firewall = raw.get("firewall") or {}
    return IapGclbProvisioning(
        domain=str(domain),
        resource_prefix=str(resources.get("prefix") or domain.split(".")[0]),
        controller_port=int(controller.get("port") or 10000),
        discovery_label=_opt_str(controller.get("discovery_label")),
        enabled=bool(raw.get("enabled", True)),
        apply_allow_rule=bool(firewall.get("apply_allow_rule", True)),
        deny_public_default=bool(firewall.get("deny_public_default", False)),
    )


def current_cluster() -> str | None:
    """The cluster pinned by ``marin-cluster config use``, or None."""
    if _CURRENT_CLUSTER_POINTER.is_file():
        name = _CURRENT_CLUSTER_POINTER.read_text().strip()
        return name or None
    return None


def set_current_cluster(name: str) -> None:
    """Pin ``name`` as the current cluster (a secret-free pointer)."""
    _CURRENT_CLUSTER_POINTER.parent.mkdir(parents=True, exist_ok=True)
    _CURRENT_CLUSTER_POINTER.write_text(name + "\n")


def _opt_str(value: Any) -> str | None:
    return str(value) if value is not None else None
