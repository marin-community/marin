# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FederationDns — the oa.dev CNAME pointing a cluster's federation host at CoreWeave's LB.

CoreWeave's External Hostname Controller allocates a wildcard record under
`<tenant>-<cks-cluster-name>.coreweave.app` once TraefikAddon's LoadBalancer comes up, resolving
any label under it to that LoadBalancer. Rather than polling the live Service for that
asynchronously-allocated value (the approach install_cw_network.py's read_traefik_fqdn takes),
the CNAME target is computed directly: it always follows
`<federation-host-label>.<tenant>-<cks-cluster-name>.coreweave.app`, confirmed byte-for-byte
against every live CoreWeave record in the oa.dev zone (2026-07-22) — see
`federation_dns_target`'s docstring for the four checked values.
"""

from dataclasses import dataclass

import pulumi
import pulumi_cloudflare as cloudflare
from iris.cluster.platforms.k8s.network_manifests import default_federation_host
from rigging.secrets import resolve_secret_spec

from iac.config import CksClusterSpec

# CoreWeave account/tenant ID for this org. Stable across every cluster on the account, not
# derived from anything cluster-specific; also hardcoded in infra/grafana/src/config.py and
# lib/iris/docs/coreweave.md's Grafana dashboard link.
COREWEAVE_TENANT_ID = "208261"

# oa.dev's Cloudflare zone ID. Same value as infra/grafana/Pulumi.marin-grafana.yaml's
# dns_zone_id; not secret (zone IDs aren't credentials, only the API token is), so it's a plain
# constant here rather than duplicated stack config across every CoreWeave cluster.
OA_DEV_ZONE_ID = "169959d6aafcbfd77764b8efafa3a509"

# Same token infra/grafana's DnsRecord uses (see its README's "one-time" setup step) — a plain
# constant, not per-cluster config, since every CoreWeave cluster's federation CNAME lives in
# the same oa.dev zone. "latest" (not a pinned version) matches how it's rotated today; a
# malformed/absent secret raises SecretResolutionError rather than falling back silently.
CLOUDFLARE_TOKEN_SECRET = "gcp-secret://projects/hai-gcp-models/secrets/cloudflare-oa-dns-token/versions/latest"


def federation_dns_target(cluster: str, cks_cluster_name: str) -> str:
    """The CNAME target CoreWeave allocates for `cluster`'s federation host.

    Checked against every live oa.dev record as of 2026-07-22:
        cw-rno2a        (marin-rn02a)      -> iris-cw-rno2a.208261-marin-rn02a.coreweave.app
        cw-us-east-02a  (marin-gpu)        -> iris-cw-us-east-02a.208261-marin-gpu.coreweave.app
        cw-us-east-08a  (marin-us-east-08a)-> iris-cw-us-east-08a.208261-marin-us-east-08a.coreweave.app
        cw-us-west-04a  (marin)            -> iris-cw-us-west-04a.208261-marin.coreweave.app
    """
    host_label = default_federation_host(cluster).split(".", 1)[0]
    return f"{host_label}.{COREWEAVE_TENANT_ID}-{cks_cluster_name}.coreweave.app"


@dataclass(frozen=True)
class FederationDnsArgs:
    cluster: str  # Iris cluster name; derives the record name (iris-cw-<cluster>.oa.dev)
    cks_cluster: CksClusterSpec  # .name feeds the CNAME target
    # One-shot adoption: the live record's Cloudflare-generated `<zone_id>/<dns_record_id>`.
    # Unlike the K8s resources' `marin-iac:import` flow, Cloudflare's ID isn't derivable from our
    # own config — look it up via the API or dashboard and set it via
    # `marin-iac:dns_record_import_id` for exactly one `up`, same one-shot discipline as
    # `marin-iac:import` elsewhere (see README.md's "Adopting a new cluster").
    import_id: str | None = None


class FederationDns(pulumi.ComponentResource):
    """The oa.dev CNAME pointing this cluster's federation host at its Traefik LoadBalancer."""

    def __init__(
        self,
        name: str,
        args: FederationDnsArgs,
        *,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:FederationDns", name, None, opts)

        # Fetched directly from Secret Manager rather than the CLOUDFLARE_API_TOKEN env var, so
        # `pulumi preview`/`up` need no manual export step. pulumi.Output.secret marks it so the
        # CLI and state file never render it in plaintext (the provider's own schema already
        # treats api_token as sensitive; this is belt-and-suspenders for the raw string in between).
        token = resolve_secret_spec(CLOUDFLARE_TOKEN_SECRET).value
        cloudflare_provider = cloudflare.Provider(
            "cloudflare",
            api_token=pulumi.Output.secret(token),
            opts=pulumi.ResourceOptions(parent=self),
        )

        host = default_federation_host(args.cluster)
        target = federation_dns_target(args.cluster, args.cks_cluster.name)
        self.record = cloudflare.DnsRecord(
            "federation-cname",
            zone_id=OA_DEV_ZONE_ID,
            name=host,
            type="CNAME",
            content=target,
            ttl=300,
            proxied=False,
            opts=pulumi.ResourceOptions(parent=self, provider=cloudflare_provider, import_=args.import_id),
        )
        self.register_outputs({})
