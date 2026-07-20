# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TraefikAddon — the ingress stack a CoreWeave controller needs for Iris federation.

Parent Marin controllers federate jobs by dialing a CoreWeave controller's RPC surface
inbound, so each cluster runs its own ingress. This component declares that stack: the
Traefik ingress controller and cert-manager, a Let's Encrypt HTTP-01 ClusterIssuer per
configured environment, and one IP-locked Ingress (a Traefik ipAllowList Middleware)
admitting only the parent controllers' egress IPs over the whole controller host.

Manifest shapes come from iris.cluster.platforms.k8s.network_manifests, shared with
install_cw_network.py so the two render identical objects.
"""

from dataclasses import dataclass

import pulumi
import pulumi_kubernetes as k8s
from iris.cluster.platforms.k8s.network_manifests import (
    CERT_MANAGER_CHART,
    DEFAULT_CERT_MANAGER_NAMESPACE,
    DEFAULT_CERT_MANAGER_RELEASE,
    DEFAULT_TLS_SECRET,
    DEFAULT_TRAEFIK_NAMESPACE,
    DEFAULT_TRAEFIK_RELEASE,
    ISSUER_ENVS,
    MIDDLEWARE_NAME,
    TRAEFIK_CHART,
    default_federation_host,
    federation_ingress,
    http01_issuer,
    ipallowlist_middleware,
    normalize_source,
)

from iac.config import IngressSpec

# Chart versions, pinned for reproducibility (same discipline as KueueAddon.CKS_KUEUE_VERSION) —
# latest published at https://charts.core-services.ingress.coreweave.com/index.yaml as of this
# writing. Bump these in lockstep with a chart upgrade.
TRAEFIK_VERSION = "1.35.0"
CERT_MANAGER_VERSION = "1.20.0"


@dataclass(frozen=True)
class TraefikAddonArgs:
    cluster: str  # Iris cluster name; derives the federation host (iris-cw-<cluster>.oa.dev)
    namespace: str  # controller namespace; the federation Ingress + Middleware live here
    service_name: str  # controller Service the federation route backends onto (controller.coreweave)
    port: int  # controller port (controller.coreweave)
    spec: IngressSpec
    # IrisRbac's Namespace resource (args.namespace must already exist before the
    # namespace-scoped Middleware/Ingress below apply). Pulumi does not order sibling
    # ComponentResources on its own, so a fresh cluster with no namespace yet needs this wired
    # explicitly — see IrisRbac.namespace.
    namespace_dependency: pulumi.Resource | None = None
    # Adoption mode: stamp import_ on each object so `pulumi preview` shows the real adoption
    # diff instead of planning creates. Set via the `marin-iac:import` stack flag.
    adopt: bool = False


class TraefikAddon(pulumi.ComponentResource):
    """Traefik, cert-manager, HTTP-01 ClusterIssuers, and the IP-locked federation Ingress.

    The federation Ingress and its ipAllowList Middleware admit only
    `spec.federation_allow_sources` — the parent controllers' egress IPs — over the whole
    controller host.
    """

    def __init__(
        self,
        name: str,
        args: TraefikAddonArgs,
        *,
        k8s_provider: pulumi.ProviderResource,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:TraefikAddon", name, None, opts)

        unknown_issuers = set(args.spec.cluster_issuers) - set(ISSUER_ENVS)
        if unknown_issuers:
            raise ValueError(
                f"spec.cluster_issuers has unknown issuer name(s) {sorted(unknown_issuers)!r}; "
                f"valid names are {sorted(ISSUER_ENVS)!r}"
            )
        if args.spec.active_cluster_issuer not in args.spec.cluster_issuers:
            raise ValueError(
                f"spec.active_cluster_issuer {args.spec.active_cluster_issuer!r} is not in "
                f"spec.cluster_issuers {args.spec.cluster_issuers!r}; the federation Ingress would "
                "reference a ClusterIssuer that TraefikAddon never creates"
            )

        def child_opts(import_id: str | None = None, depends_on: list | None = None) -> pulumi.ResourceOptions:
            return pulumi.ResourceOptions(
                parent=self,
                provider=k8s_provider,
                depends_on=depends_on,
                import_=import_id if (args.adopt and import_id) else None,
            )

        # These two Releases omit `repository_opts`: with it set, Pulumi intermittently fails to
        # resolve the chart at preview time (upstream pulumi-kubernetes#935). Without it,
        # `chart="coreweave/traefik"` resolves through the local `helm` CLI repo config, which
        # must have the `coreweave` alias registered first (`helm repo add coreweave <url>`).
        # That prerequisite, and the options for folding it into `pulumi up`, are documented in
        # one place: infra/pulumi/README.md Prerequisites and gaps.md's "Pulumi Helm chart
        # resolution". `KueueAddon`'s `cks-kueue` Release keeps `repository_opts`, never having failed.
        cert_manager_release = k8s.helm.v3.Release(
            "cert-manager",
            name=DEFAULT_CERT_MANAGER_RELEASE,
            chart=CERT_MANAGER_CHART,
            version=CERT_MANAGER_VERSION,
            namespace=DEFAULT_CERT_MANAGER_NAMESPACE,
            create_namespace=True,
            opts=child_opts(f"{DEFAULT_CERT_MANAGER_NAMESPACE}/{DEFAULT_CERT_MANAGER_RELEASE}"),
        )
        traefik_release = k8s.helm.v3.Release(
            "traefik",
            name=DEFAULT_TRAEFIK_RELEASE,
            chart=TRAEFIK_CHART,
            version=TRAEFIK_VERSION,
            namespace=DEFAULT_TRAEFIK_NAMESPACE,
            create_namespace=True,
            opts=child_opts(f"{DEFAULT_TRAEFIK_NAMESPACE}/{DEFAULT_TRAEFIK_RELEASE}"),
        )

        # HTTP-01 ClusterIssuers named in spec.cluster_issuers (normally both staging + prod).
        #
        # No explicit CRD-readiness wait: `depends_on=[cert_manager_release]` orders these after
        # the Release's readiness check (healthy cert-manager pods), which in practice clears
        # after the CRDs it ships are registered, and the k8s provider retries a CustomResource
        # create up to 5 times if the CRD is not yet found. See gaps.md's "Traefik/cert-manager
        # CRD-registration race" for the investigation and the accepted-risk rationale.
        issuers = []
        for issuer_name in args.spec.cluster_issuers:
            env = ISSUER_ENVS[issuer_name]
            manifest = http01_issuer(env, args.spec.acme_email, args.spec.ingress_class)
            issuers.append(
                k8s.apiextensions.CustomResource(
                    f"cluster-issuer-{env}",
                    api_version=manifest["apiVersion"],
                    kind=manifest["kind"],
                    metadata=manifest["metadata"],
                    spec=manifest["spec"],
                    opts=child_opts(issuer_name, depends_on=[cert_manager_release]),
                )
            )

        # Both objects below are namespace-scoped to args.namespace, which only IrisRbac's
        # Namespace resource creates — depend on it explicitly (see TraefikAddonArgs.namespace_dependency).
        namespace_deps = [args.namespace_dependency] if args.namespace_dependency is not None else []

        source_ranges = [normalize_source(source) for source in args.spec.federation_allow_sources]
        middleware_manifest = ipallowlist_middleware(namespace=args.namespace, source_ranges=source_ranges)
        k8s.apiextensions.CustomResource(
            "federation-ipallowlist",
            api_version=middleware_manifest["apiVersion"],
            kind=middleware_manifest["kind"],
            metadata=middleware_manifest["metadata"],
            spec=middleware_manifest["spec"],
            opts=child_opts(f"{args.namespace}/{MIDDLEWARE_NAME}", depends_on=[traefik_release, *namespace_deps]),
        )

        ingress_manifest = federation_ingress(
            namespace=args.namespace,
            service_name=args.service_name,
            port=args.port,
            host=default_federation_host(args.cluster),
            ingress_class=args.spec.ingress_class,
            tls_secret=DEFAULT_TLS_SECRET,
            cluster_issuer=args.spec.active_cluster_issuer,
        )
        k8s.networking.v1.Ingress(
            "federation-ingress",
            metadata=ingress_manifest["metadata"],
            spec=ingress_manifest["spec"],
            opts=child_opts(
                f"{args.namespace}/{ingress_manifest['metadata']['name']}",
                # Depends on the ClusterIssuer it references via cert-manager.io/cluster-issuer —
                # applying the Ingress before that issuer exists means cert-manager's
                # ingress-shim can't find it on first reconcile.
                depends_on=[traefik_release, *issuers, *namespace_deps],
            ),
        )
        self.register_outputs({})
