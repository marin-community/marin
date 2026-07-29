# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for Marin IaC.

Reads the target cluster from stack config (`marin-iac:cluster`), loads its Iris config +
typed `provisioning:` section, and declares that cluster's resources. One stack per cluster;
`pulumi up` provisions all of a stack's declared resources together. The provider decides
which resources: CoreWeave declares the controller RBAC, reserved NodePools, Kueue objects,
the Traefik/cert-manager/federation-ingress stack, and configured Cloudflare CNAMEs; GCP declares
the reserved federation-egress static IPs and the Artifact Registry pull-through mirrors. Components not yet implemented
(object storage, the CKS cluster object itself; GCP IAM/GCLB+IAP/buckets) are tracked in
README.md's "Future work".
"""

import os
import sys

# Make the `iac` package importable without a separate install step: Pulumi runs this file
# from infra/pulumi/, so add its src/ to the path. Deps (pulumi, pulumi-kubernetes) and
# marin-iris/marin-rigging come from the shared repo virtualenv (marin-iac is a workspace
# member; run `uv sync --all-packages`).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pulumi
import pulumi_gcp as gcp
import pulumi_kubernetes as k8s
from iac.config import CLOUDFLARE_TOKEN_SECRET, Provider, load_iris_config, load_provisioning
from iac.coreweave.cluster import CoreweaveCluster, CoreweaveClusterArgs
from iac.coreweave.dns import FederationDns, FederationDnsArgs
from iac.coreweave.kueue import KueueAddon, KueueAddonArgs
from iac.coreweave.rbac import GrafanaObserverRbac, GrafanaObserverRbacArgs, IrisRbac, IrisRbacArgs
from iac.coreweave.traefik import TraefikAddon, TraefikAddonArgs
from iac.gcp.addresses import GcpStaticAddresses, GcpStaticAddressesArgs
from iac.gcp.registries import GcpArtifactRegistries, GcpArtifactRegistriesArgs
from iac.nodepools import derive_nodepools
from rigging.secrets import resolve_secret_spec

DEFAULT_NAMESPACE = "iris"
DEFAULT_COREWEAVE_KUBECONFIG = os.path.expanduser("~/.kube/coreweave-iris")


def _require_kubeconfig(cluster: str) -> None:
    """Require KUBECONFIG in the execution environment, offering the default CoreWeave kubeconfig path."""
    if os.environ.get("KUBECONFIG"):
        return
    candidate = DEFAULT_COREWEAVE_KUBECONFIG
    if os.path.isfile(candidate) and sys.stdin.isatty():
        answer = input(f"KUBECONFIG is unset; use {candidate}? [Y/n] ").strip().lower()
        if answer in ("", "y", "yes"):
            os.environ["KUBECONFIG"] = candidate
            return
    hint = f" (e.g. export KUBECONFIG={candidate})" if os.path.isfile(candidate) else ""
    raise ValueError(
        f"cluster {cluster!r} requires KUBECONFIG; "
        f"Pulumi reads Kubernetes credentials from the execution environment{hint}"
    )


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


def _build_coreweave(cluster: str, *, adopt: bool) -> None:
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
    _require_kubeconfig(cluster)
    # kubeconfig="": bypasses the Python SDK's default of copying $KUBECONFIG into
    # the provider input (which persists a machine-local path in state, creating spurious diffs).
    # The provider process still loads credentials from the ambient KUBECONFIG env.
    # enable_patch_force=True: Iris's controller re-applies RBAC/NodePools under its own field
    # manager on every restart, so a plain SSA dry-run reports a field conflict without forced
    # ownership (README §Adoption check). Declared here until the "cede" ships (spec.md §4).
    k8s_provider = k8s.Provider(
        "cw-k8s",
        kubeconfig="",
        context=platform_coreweave.kube_context,
        enable_patch_force=True,
    )

    CoreweaveCluster(
        "cluster",
        CoreweaveClusterArgs(
            cluster=coreweave_provisioning.cluster,
            nodepools=derive_nodepools(iris_config),
            adopt=adopt,
        ),
        k8s_provider=k8s_provider,
    )
    rbac = IrisRbac(
        "rbac",
        IrisRbacArgs(namespace=namespace, spec=coreweave_provisioning.rbac, adopt=adopt),
        k8s_provider=k8s_provider,
    )
    if grafana_observer := coreweave_provisioning.grafana_observer_rbac:
        GrafanaObserverRbac(
            "grafana-observer-rbac",
            GrafanaObserverRbacArgs(usernames=grafana_observer.usernames, adopt=adopt),
            k8s_provider=k8s_provider,
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
            adopt=adopt,
        ),
        k8s_provider=k8s_provider,
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
            port=controller_coreweave.port or 10000,
            spec=coreweave_provisioning.ingress,
            namespace_dependency=rbac.namespace,
            adopt=adopt,
        ),
        k8s_provider=k8s_provider,
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


def _build_gcp(cluster: str, *, adopt: bool) -> None:
    provisioning = load_provisioning(cluster)
    assert provisioning.gcp is not None  # guaranteed by load_provisioning
    gcp_provisioning = provisioning.gcp

    gcp_provider = gcp.Provider("gcp", project=gcp_provisioning.project)
    GcpStaticAddresses(
        "addresses",
        GcpStaticAddressesArgs(
            project=gcp_provisioning.project,
            addresses=gcp_provisioning.addresses,
            adopt=adopt,
        ),
        gcp_provider=gcp_provider,
    )
    GcpArtifactRegistries(
        "registries",
        GcpArtifactRegistriesArgs(
            project=gcp_provisioning.project,
            registries=gcp_provisioning.registries,
            adopt=adopt,
        ),
        gcp_provider=gcp_provider,
    )


def main() -> None:
    config = pulumi.Config("marin-iac")
    cluster = config.require("cluster")
    # Adoption recon: `pulumi config set marin-iac:import true` stamps import_ on every
    # resource so `pulumi preview` shows the real adoption diff (provider- and parent-correct)
    # instead of planning creates. Never run `pulumi up` through a destructive NodePool diff.
    adopt = config.get_bool("import") or False
    provider = load_provisioning(cluster).provider
    if provider is Provider.COREWEAVE:
        _build_coreweave(cluster, adopt=adopt)
    elif provider is Provider.GCP:
        _build_gcp(cluster, adopt=adopt)
    else:
        raise NotImplementedError(f"provider {provider!r} not yet implemented in iac")


main()
