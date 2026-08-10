# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure Traefik/cert-manager/federation-ingress manifest builders and their constants.

One source of truth for the CoreWeave network stack Iris installs: the install script
(``scripts/install_cw_network.py``) and the IaC component (``iac.coreweave.traefik``)
both import these builders so they render byte-identical manifests. Everything here is
pure — the functions return plain dicts and do no I/O.
"""

import ipaddress

# CoreWeave-hosted charts (same repo Kueue's cks-kueue chart comes from).
CW_REPO_NAME = "coreweave"
CW_REPO_URL = "https://charts.core-services.ingress.coreweave.com"
TRAEFIK_CHART = f"{CW_REPO_NAME}/traefik"
CERT_MANAGER_CHART = f"{CW_REPO_NAME}/cert-manager"

DEFAULT_TRAEFIK_NAMESPACE = "traefik"
DEFAULT_TRAEFIK_RELEASE = "traefik"
DEFAULT_CERT_MANAGER_NAMESPACE = "cert-manager"
DEFAULT_CERT_MANAGER_RELEASE = "cert-manager"

LE_ACME = {
    "prod": "https://acme-v02.api.letsencrypt.org/directory",
    "staging": "https://acme-staging-v02.api.letsencrypt.org/directory",
}
# ClusterIssuer names this module creates; use as controller.coreweave.cluster_issuer.
ISSUER_NAMES = {"prod": "letsencrypt-http01-prod", "staging": "letsencrypt-http01-staging"}
# Reverse lookup: issuer name (as it appears in IngressSpec.cluster_issuers) -> LE env key.
ISSUER_ENVS = {name: env for env, name in ISSUER_NAMES.items()}

CLUSTERISSUER_CRD = "clusterissuers.cert-manager.io"
MIDDLEWARE_CRD = "middlewares.traefik.io"

# Federation route: an IP-locked Ingress + ipAllowList Middleware covering the WHOLE
# controller host. It is the only off-cluster ingress for a CoreWeave controller — there
# is no world-open surface (users reach Iris via iris.oa.dev, marin federates outward, so
# marin's egress is the only external caller).
INGRESS_NAME = "iris-federation"
MIDDLEWARE_NAME = "iris-federation-ipallowlist"

# The controller's legacy world-open /proxy Ingress. The IP-locked route here supersedes
# it, so install removes it (leaving it would keep /proxy world-open, since Traefik
# prefers its longer path prefix).
CONTROLLER_PROXY_INGRESS_NAME = "iris-controller-proxy"

# TLS secret cert-manager issues for the federation route.
DEFAULT_TLS_SECRET = "iris-controller-fed-tls"
# Staging first to avoid Let's Encrypt rate limits while DNS/allowlist are shaken out;
# flip to the prod issuer once the staging cert validates.
DEFAULT_CLUSTER_ISSUER = ISSUER_NAMES["staging"]


def default_federation_host(cluster: str) -> str:
    """The controller's public host, ``iris-cw-<cluster>.oa.dev`` (cluster minus a leading ``cw-``)."""
    short = cluster[len("cw-") :] if cluster.startswith("cw-") else cluster
    return f"iris-cw-{short}.oa.dev"


def normalize_source(value: str) -> str:
    """Validate one allowlist entry and return it in CIDR form (bare IP -> /32,/128)."""
    if "/" in value:
        ipaddress.ip_network(value, strict=False)
        return value
    ip = ipaddress.ip_address(value)
    return f"{value}/{32 if ip.version == 4 else 128}"


def http01_issuer(env: str, email: str, ingress_class: str) -> dict:
    """A Let's Encrypt HTTP-01 ClusterIssuer validated through ``ingress_class``.

    HTTP-01 (not CoreWeave's bundled DNS-01) so it can issue for a custom host, which
    the coreweave.app DNS-01 webhook cannot. Requires the host to already resolve to
    the Traefik LoadBalancer before issuance.
    """
    return {
        "apiVersion": "cert-manager.io/v1",
        "kind": "ClusterIssuer",
        "metadata": {"name": ISSUER_NAMES[env]},
        "spec": {
            "acme": {
                "server": LE_ACME[env],
                "email": email,
                "privateKeySecretRef": {"name": f"{ISSUER_NAMES[env]}-account-key"},
                "solvers": [{"http01": {"ingress": {"ingressClassName": ingress_class}}}],
            }
        },
    }


def ipallowlist_middleware(*, namespace: str, source_ranges: list[str], xff_depth: int = 0) -> dict:
    """A Traefik ``ipAllowList`` Middleware admitting only ``source_ranges``.

    By default Traefik matches the client's direct transport peer (``RemoteAddr``). If
    the CoreWeave LoadBalancer SNATs (so Traefik sees the LB, not the real client), set
    ``xff_depth`` to the number of trusted proxy hops and Traefik reads the client IP
    from ``X-Forwarded-For`` instead. Verify which applies by testing a refused request
    from a non-allowlisted host.
    """
    ip_allow_list: dict = {"sourceRange": source_ranges}
    if xff_depth > 0:
        ip_allow_list["ipStrategy"] = {"depth": xff_depth}
    return {
        "apiVersion": "traefik.io/v1alpha1",
        "kind": "Middleware",
        "metadata": {"name": MIDDLEWARE_NAME, "namespace": namespace},
        "spec": {"ipAllowList": ip_allow_list},
    }


def federation_ingress(
    *,
    namespace: str,
    service_name: str,
    port: int,
    host: str,
    ingress_class: str,
    tls_secret: str,
    cluster_issuer: str,
) -> dict:
    """Single catch-all Ingress routing the whole controller host, IP-locked by the Middleware.

    One ``/`` path attaches the ``ipAllowList`` Middleware via the router-middlewares
    annotation (``<ns>-<name>@kubernetescrd``), so the entire controller surface is
    reachable only from the allowlisted source. cert-manager issues the cert via
    ``cluster_issuer``; its HTTP-01 solver runs on its own unrestricted Ingress, so the
    allowlist does not block ACME validation.
    """
    annotations = {
        "traefik.ingress.kubernetes.io/router.middlewares": f"{namespace}-{MIDDLEWARE_NAME}@kubernetescrd",
    }
    if cluster_issuer:
        annotations["cert-manager.io/cluster-issuer"] = cluster_issuer
    spec: dict = {
        "ingressClassName": ingress_class,
        "rules": [
            {
                "host": host,
                "http": {
                    "paths": [
                        {
                            "path": "/",
                            "pathType": "Prefix",
                            "backend": {"service": {"name": service_name, "port": {"number": port}}},
                        }
                    ]
                },
            }
        ],
    }
    if tls_secret:
        spec["tls"] = [{"hosts": [host], "secretName": tls_secret}]
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {"name": INGRESS_NAME, "namespace": namespace, "annotations": annotations},
        "spec": spec,
    }
