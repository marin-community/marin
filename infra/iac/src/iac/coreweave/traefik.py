# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TraefikAddon — Traefik, cert-manager, HTTP-01 ClusterIssuers, and the IP-locked federation ingress.

Reproduces what `lib/iris/scripts/install_cw_network.py install` installs, so an already-
installed cluster adopts with no change. Manifest shapes come from the shared
`iris.cluster.platforms.k8s.network_manifests` builders, so IaC and the script render
identically.
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
    build_federation_ingress,
    build_http01_issuer,
    build_ipallowlist_middleware,
    default_federation_host,
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
    """Traefik + cert-manager + HTTP-01 ClusterIssuers + the IP-locked federation ingress.

    The federation Ingress + Middleware admit only `spec.federation_allow_sources` (the
    marin-side controllers' egress IPs) over the whole controller host — there is no
    world-open surface. Does not remove the controller's legacy `iris-controller-proxy`
    Ingress: that is a one-time migration cleanup the imperative script performs, not a
    stable piece of declared state; run the script by hand once if that cleanup is still
    needed on a given cluster.
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

        # WORKAROUND, deliberate and load-bearing — see gaps.md's "Pulumi Helm chart resolution"
        # section for the full investigation before touching this.
        #
        # Neither Release below sets `repository_opts`. That's the fix, not an oversight: Pulumi's
        # embedded Helm client intermittently fails to resolve a `repository_opts`-based chart
        # with "chart ... version ... not found in ... repository", even though the chart/version
        # genuinely exist (confirmed by direct download and by the real `helm` CLI resolving them
        # instantly). Extensive testing (documented in gaps.md — ~20 preview runs, multiple
        # dependency-graph shapes) found no way to make `repository_opts` resolve reliably;
        # forcing resolution order via `depends_on` made failures *more* deterministic, not less.
        #
        # Without `repository_opts`, `chart="coreweave/traefik"` (repo-alias/chart-name syntax)
        # is resolved by Pulumi falling back to the LOCAL `helm` CLI's own repo config —
        # `~/Library/Preferences/helm/repositories.yaml` on macOS — which must already have the
        # `coreweave` alias registered. This is the one non-obvious operational cost of this
        # workaround, and it is REQUIRED, not optional:
        #
        #     helm repo add coreweave https://charts.core-services.ingress.coreweave.com
        #     helm repo update coreweave
        #
        # Run this once per machine, and again any time TRAEFIK_VERSION/CERT_MANAGER_VERSION
        # below is bumped to a version published after your last `helm repo update` (a stale
        # local index just won't list the new version — a real, non-flaky error in that case).
        # This is now documented in infra/iac/README.md's Prerequisites, and MUST be added as an
        # explicit step in any CI workflow that runs `pulumi preview`/`up` against this stack —
        # CI runners are ephemeral, so this can never be a "ran it once" step there.
        #
        # Residual risk, accepted: resolution now trusts whatever `coreweave` happens to be
        # aliased to locally, with no URL pinned in code to verify against. Low-likelihood
        # (nobody has a reason to alias `coreweave` to something else), but real.
        #
        # Root cause is upstream and unfixed: https://github.com/pulumi/pulumi-kubernetes/issues/935
        # (Pulumi caches nothing for Helm chart resolution, by design, open since 2020).
        # `KueueAddon`'s `cks-kueue` Release has never failed in this investigation and still
        # uses `repository_opts` (untouched) — this workaround is scoped to the two Releases that
        # actually failed, not applied everywhere by default.
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
        # No explicit CRD-readiness wait here (unlike install_cw_network.py's wait_for_crd), by
        # design: `depends_on=[cert_manager_release]` already orders this after the Release's
        # readiness check, which waits for cert-manager's Deployments to have healthy pods — a
        # bar that in practice clears well after the CRDs it ships are registered, since pod
        # readiness (image pull, container start, probe passes) takes far longer than API-server
        # CRD registration. Pulumi's k8s provider also retries a CustomResource create up to 5
        # times with backoff if the CRD genuinely isn't found yet (verified against
        # https://github.com/pulumi/pulumi-kubernetes/issues/1446 — hardcoded, not configurable
        # via custom_timeouts, and still an open enhancement request upstream as of this writing).
        # Between the two, this has never failed in real testing on this repo. If it ever does,
        # the fix is upstream (a higher retry budget), not a bespoke polling loop here — see
        # gaps.md's "Traefik/cert-manager CRD-registration race" entry.
        issuers = []
        for issuer_name in args.spec.cluster_issuers:
            env = ISSUER_ENVS[issuer_name]
            manifest = build_http01_issuer(env, args.spec.acme_email, args.spec.ingress_class)
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
        middleware_manifest = build_ipallowlist_middleware(namespace=args.namespace, source_ranges=source_ranges)
        k8s.apiextensions.CustomResource(
            "federation-ipallowlist",
            api_version=middleware_manifest["apiVersion"],
            kind=middleware_manifest["kind"],
            metadata=middleware_manifest["metadata"],
            spec=middleware_manifest["spec"],
            opts=child_opts(f"{args.namespace}/{MIDDLEWARE_NAME}", depends_on=[traefik_release, *namespace_deps]),
        )

        ingress_manifest = build_federation_ingress(
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
