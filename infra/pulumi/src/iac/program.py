# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi program for Marin IaC.

Reads the target cluster from stack config (`marin-iac:cluster`), loads its Iris config +
typed `provisioning:` section, and declares that cluster's resources. One stack per cluster;
`pulumi up` provisions all of a stack's declared resources together. The provider decides
which resources: CoreWeave declares the controller RBAC, reserved NodePools, Kueue objects,
the Traefik/cert-manager/federation-ingress stack, and configured Cloudflare CNAMEs; GCP declares
the reserved federation-egress static IPs, Artifact Registry pull-through mirrors, the shared
GCLB/IAP ingress, and every IAM grant on the project (`iac.gcp.iam.GcpIam`, replacing
`infra/permissions`). Components not yet implemented are tracked in README.md's "Future work".
"""

import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import pulumi
import pulumi_gcp as gcp
import pulumi_kubernetes as k8s
from finelog.deploy.config import GcpDeployment
from iris.cluster.endpoints import gcp_instance_internal_ip
from iris.cluster.platforms.types import Labels
from iris.cluster.platforms.vm_lifecycle import DEFAULT_CONTROLLER_PORT, controller_vm_name
from rigging.auth import MARIN_DESKTOP_OAUTH_CLIENT
from rigging.secrets import resolve_secret_spec

from iac.config import (
    CLOUDFLARE_TOKEN_SECRET,
    GcpGclbSpec,
    Provider,
    load_iac_finelog_config,
    load_iris_config,
    load_provisioning,
)
from iac.coreweave.cluster import CoreweaveCluster, CoreweaveClusterArgs
from iac.coreweave.dns import FederationDns, FederationDnsArgs
from iac.coreweave.kueue import KueueAddon, KueueAddonArgs
from iac.coreweave.rbac import GrafanaObserverRbac, GrafanaObserverRbacArgs, IrisRbac, IrisRbacArgs
from iac.coreweave.traefik import TraefikAddon, TraefikAddonArgs
from iac.gcp.addresses import GcpStaticAddresses, GcpStaticAddressesArgs
from iac.gcp.gclb import ControllerIngress, FinelogIngress, GcpGclbIap, GcpGclbIapArgs
from iac.gcp.iam import GcpIam, GcpIamArgs
from iac.gcp.iam_config import load_iam_config
from iac.gcp.registries import GcpArtifactRegistries, GcpArtifactRegistriesArgs
from iac.imports import NO_IMPORTS, ImportRegistrar
from iac.nodepools import derive_nodepools

DEFAULT_NAMESPACE = "iris"
IAP_AUDIENCE_PATTERN = re.compile(
    r"^/projects/(?P<project_number>[0-9]+)/global/backendServices/(?P<backend_id>[0-9]+)$"
)


@dataclass(frozen=True)
class _IapBackendIdentity:
    domain: str
    project_number: str
    backend_service_id: str
    programmatic_clients: tuple[str, ...]


def _warn_if_no_persistent_signing_key(cluster: str, iris_config) -> None:
    """Warn (non-fatal) if this cluster has no persistent `auth.signing_key`.

    Without one, the controller falls back to an ephemeral in-process keypair that does not
    survive a restart. That's fine for a cluster with no federation peers, but TraefikAddon
    (declared below) exposes the controller's RPC surface behind only an IP allowlist — a
    stable identity plus `auth.trusted_cidrs` is what actually makes that enforcing rather
    than permissive (see install_cw_network.py's own NULL-AUTH warning, docs/coreweave.md).
    Pulumi does not provision the signing-key secret itself — the key material must never pass
    through Pulumi state (see README.md's "Unsupported"); this is purely a read-only reminder
    pointing at the one command that does.
    """
    auth = iris_config.auth
    if auth is not None and auth.signing_key:
        return
    secret = f"iris-{cluster}-signing-key"
    pulumi.log.warn(
        f"cluster {cluster!r} has no persistent auth.signing_key (ephemeral in-process keypair; "
        "tokens do not survive a restart). To mint one:\n"
        f"    uv run iris cluster init-keys --gcp-secret projects/hai-gcp-models/secrets/{secret}\n"
        f"then add the printed `auth.signing_key: gcp-secret://…/versions/N` (and `auth.trusted_cidrs`, "
        f"to make the federation ingress actually enforcing) to lib/iris/config/{cluster}.yaml. "
        "See docs/coreweave.md."
    )


def _build_coreweave(cluster: str, *, imports: ImportRegistrar) -> None:
    provisioning = load_provisioning(cluster)
    assert provisioning.coreweave is not None  # guaranteed by load_provisioning
    coreweave_provisioning = provisioning.coreweave
    iris_config = load_iris_config(cluster)
    _warn_if_no_persistent_signing_key(cluster, iris_config)

    # Single-source: namespace + queue name come from the Iris config, not provisioning.
    kubernetes_provider = iris_config.kubernetes_provider
    if kubernetes_provider and kubernetes_provider.namespace:
        namespace = kubernetes_provider.namespace
    else:
        namespace = DEFAULT_NAMESPACE

    platform_coreweave = iris_config.platform.coreweave
    if platform_coreweave is None:
        raise ValueError(
            f"cluster {cluster!r} has no platform.coreweave config; "
            "the minimal IaC cut needs an out-of-cluster Kubernetes target"
        )
    # Require an explicit context: omitting it makes the provider use the kubeconfig's
    # current-context (whatever kubectl last pointed at), which can silently retarget
    # another CoreWeave cluster sharing ~/.kube/coreweave-iris.
    if not platform_coreweave.kube_context:
        raise ValueError(f"cluster {cluster!r} missing required platform.coreweave.kube_context")

    if not os.environ.get("KUBECONFIG"):
        raise ValueError("pulumi up requires KUBECONFIG (e.g. export KUBECONFIG=~/.kube/coreweave-iris).")

    # kubeconfig="": bypasses the Python SDK's default of copying $KUBECONFIG into
    # the provider input (which persists a machine-local path in state, creating spurious diffs)
    # The provider process still loads credentials from the ambient KUBECONFIG env.
    # enable_patch_force=True: allows pulumi to take control of resources it didn't create.
    # This should not happen in practice, but could during manual operations.
    k8s_provider = k8s.Provider(
        "cw-k8s",
        kubeconfig="",
        context=platform_coreweave.kube_context,
        enable_patch_force=True,
    )

    coreweave_cluster = CoreweaveCluster(
        "cluster",
        CoreweaveClusterArgs(
            cluster=coreweave_provisioning.cluster,
            nodepools=derive_nodepools(iris_config),
        ),
        k8s_provider=k8s_provider,
        imports=imports,
    )
    rbac = IrisRbac(
        "rbac",
        IrisRbacArgs(namespace=namespace, spec=coreweave_provisioning.rbac),
        k8s_provider=k8s_provider,
        imports=imports,
    )
    if grafana_observer := coreweave_provisioning.grafana_observer_rbac:
        GrafanaObserverRbac(
            "grafana-observer-rbac",
            GrafanaObserverRbacArgs(usernames=grafana_observer.usernames),
            k8s_provider=k8s_provider,
            imports=imports,
        )

    kueue_config = kubernetes_provider.kueue if kubernetes_provider else None
    if kueue_config is None or not kueue_config.cluster_queue:
        raise ValueError(
            f"cluster {cluster!r} has no kubernetes_provider.kueue.cluster_queue; "
            "KueueAddon needs the ClusterQueue name Iris binds its LocalQueue to"
        )
    KueueAddon(
        "kueue",
        KueueAddonArgs(
            namespace=namespace,
            cluster_queue=kueue_config.cluster_queue,
            spec=coreweave_provisioning.kueue,
        ),
        k8s_provider=k8s_provider,
        imports=imports,
        opts=pulumi.ResourceOptions(depends_on=[coreweave_cluster]),
    )

    controller_coreweave = iris_config.controller.coreweave
    if controller_coreweave is None or not controller_coreweave.service_name:
        raise ValueError(
            f"cluster {cluster!r} has no controller.coreweave.service_name; "
            "TraefikAddon needs the controller Service the federation route backends onto"
        )
    TraefikAddon(
        "traefik",
        TraefikAddonArgs(
            cluster=cluster,
            namespace=namespace,
            service_name=controller_coreweave.service_name,
            port=controller_coreweave.port or DEFAULT_CONTROLLER_PORT,
            spec=coreweave_provisioning.ingress,
            namespace_dependency=rbac.namespace,
        ),
        k8s_provider=k8s_provider,
        imports=imports,
    )

    federation_dns = coreweave_provisioning.federation_dns
    if federation_dns is not None:
        cloudflare_api_token = resolve_secret_spec(CLOUDFLARE_TOKEN_SECRET).value
        FederationDns(
            "dns",
            FederationDnsArgs(
                spec=federation_dns,
                api_token=pulumi.Output.secret(cloudflare_api_token),
            ),
        )


def _iap_backend_identity(cluster: str, iris_config) -> _IapBackendIdentity:
    auth = iris_config.auth
    iap = auth.iap if auth is not None else None
    if iap is None or not iap.url or not iap.signed_header_audience:
        raise ValueError(f"cluster {cluster!r} needs auth.iap.url and auth.iap.signed_header_audience for GCLB")
    parsed_url = urlparse(iap.url)
    if parsed_url.scheme != "https" or not parsed_url.hostname or parsed_url.path not in ("", "/"):
        raise ValueError(f"cluster {cluster!r} auth.iap.url must be an HTTPS origin")
    audience = IAP_AUDIENCE_PATTERN.fullmatch(iap.signed_header_audience)
    if audience is None:
        raise ValueError(f"cluster {cluster!r} has an invalid auth.iap.signed_header_audience")
    desktop_client = iap.oauth_client_id or MARIN_DESKTOP_OAUTH_CLIENT.client_id
    programmatic_clients = tuple(dict.fromkeys((desktop_client, *iap.programmatic_audiences)))
    return _IapBackendIdentity(
        domain=parsed_url.hostname,
        project_number=audience["project_number"],
        backend_service_id=audience["backend_id"],
        programmatic_clients=programmatic_clients,
    )


def _build_gclb(
    project: str,
    spec: GcpGclbSpec,
    *,
    gcp_provider: pulumi.ProviderResource,
    imports: ImportRegistrar,
) -> None:
    controllers: list[ControllerIngress] = []
    project_numbers: set[str] = set()
    for controller_spec in spec.controllers:
        iris_config = load_iris_config(controller_spec.cluster)
        controller_config = iris_config.controller.gcp
        platform_config = iris_config.platform.gcp
        if controller_config is None or platform_config is None:
            raise ValueError(f"cluster {controller_spec.cluster!r} is not configured for GCP")
        if platform_config.project_id != project:
            raise ValueError(
                f"cluster {controller_spec.cluster!r} belongs to {platform_config.project_id!r}, not {project!r}"
            )
        iap_identity = _iap_backend_identity(controller_spec.cluster, iris_config)
        project_numbers.add(iap_identity.project_number)
        label_prefix = iris_config.platform.label_prefix or "iris"
        instance = controller_vm_name(label_prefix)
        controllers.append(
            ControllerIngress(
                cluster=controller_spec.cluster,
                domain=iap_identity.domain,
                zone=controller_config.zone,
                instance=instance,
                ip_address=gcp_instance_internal_ip(
                    instance,
                    project=project,
                    zone=controller_config.zone,
                ),
                backend_service_id=iap_identity.backend_service_id,
                network_tag=Labels(label_prefix).iris_controller,
                programmatic_clients=iap_identity.programmatic_clients,
                port=controller_config.port or DEFAULT_CONTROLLER_PORT,
                token_proxy=controller_spec.token_proxy,
                deny_public=controller_spec.deny_public,
            )
        )

    if len(project_numbers) != 1:
        raise ValueError("all GCLB controller audiences must use one GCP project number")
    (project_number,) = project_numbers

    finelogs: list[FinelogIngress] = []
    for finelog_spec in spec.finelogs:
        finelog_config = load_iac_finelog_config(finelog_spec.cluster)
        deployment = finelog_config.deployment.gcp
        if not isinstance(deployment, GcpDeployment):
            raise ValueError(f"finelog {finelog_spec.cluster!r} is not configured for GCP")
        if deployment.project != project:
            raise ValueError(f"finelog {finelog_spec.cluster!r} belongs to {deployment.project!r}, not {project!r}")
        network_tag = f"{finelog_config.name}-lb"
        if network_tag not in deployment.network_tags:
            raise ValueError(f"finelog {finelog_spec.cluster!r} must declare network tag {network_tag!r}")
        finelogs.append(
            FinelogIngress(
                cluster=finelog_spec.cluster,
                domain=finelog_spec.domain,
                zone=deployment.zone,
                instance=finelog_config.name,
                ip_address=gcp_instance_internal_ip(
                    finelog_config.name,
                    project=project,
                    zone=deployment.zone,
                ),
                port=finelog_config.port,
                network_tag=network_tag,
                sender_source_ranges=tuple(finelog_spec.sender_source_ranges),
            )
        )

    gclb = GcpGclbIap(
        "gclb",
        GcpGclbIapArgs(
            project=project,
            project_number=project_number,
            frontend_name=spec.frontend_name,
            controllers=tuple(controllers),
            finelogs=tuple(finelogs),
            network=spec.network,
            subnetwork=spec.subnetwork,
        ),
        gcp_provider=gcp_provider,
        imports=imports,
    )
    pulumi.export("gclb_ip_address", gclb.ip_address)


def _build_gcp(cluster: str, *, imports: ImportRegistrar) -> None:
    provisioning = load_provisioning(cluster)
    assert provisioning.gcp is not None  # guaranteed by load_provisioning
    gcp_provisioning = provisioning.gcp
    iam_config = load_iam_config()

    gcp_provider = gcp.Provider("gcp", project=gcp_provisioning.project)
    GcpStaticAddresses(
        "addresses",
        GcpStaticAddressesArgs(
            project=gcp_provisioning.project,
            addresses=gcp_provisioning.addresses,
        ),
        gcp_provider=gcp_provider,
        imports=imports,
    )
    GcpArtifactRegistries(
        "registries",
        GcpArtifactRegistriesArgs(
            project=gcp_provisioning.project,
            registries=gcp_provisioning.registries,
        ),
        gcp_provider=gcp_provider,
        imports=imports,
    )
    if gcp_provisioning.gclb is not None:
        _build_gclb(
            gcp_provisioning.project,
            gcp_provisioning.gclb,
            gcp_provider=gcp_provider,
            imports=imports,
        )
    GcpIam(
        "iam",
        GcpIamArgs(
            project=gcp_provisioning.project,
            kms_location=iam_config.kms_location,
            kms_key_ring=iam_config.kms_key_ring,
            kms_key=iam_config.kms_key,
            custom_roles=iam_config.custom_roles,
            owned_service_accounts=iam_config.owned_service_accounts,
            project_grants=iam_config.project_grants,
            kms_grants=iam_config.kms_grants,
            secrets=iam_config.secrets,
            buckets=iam_config.buckets,
            artifact_repositories=iam_config.artifact_repositories,
            service_accounts=iam_config.service_accounts,
        ),
        gcp_provider=gcp_provider,
        imports=imports,
    )


def build_stack(imports: ImportRegistrar = NO_IMPORTS) -> None:
    """Declare one configured Marin IaC stack and optionally collect import IDs."""

    config = pulumi.Config("marin-iac")
    cluster = config.require("cluster")
    provider = load_provisioning(cluster).provider
    if provider is Provider.COREWEAVE:
        _build_coreweave(cluster, imports=imports)
    elif provider is Provider.GCP:
        _build_gcp(cluster, imports=imports)
    else:
        raise NotImplementedError(f"provider {provider!r} not yet implemented in iac")
