# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cloudflare DNS for a CoreWeave federation ingress."""

from dataclasses import dataclass

import pulumi
import pulumi_cloudflare as cloudflare

from iac.config import FederationDnsSpec


@dataclass(frozen=True)
class FederationDnsArgs:
    spec: FederationDnsSpec
    api_token: pulumi.Input[str]


class FederationDns(pulumi.ComponentResource):
    """A protected, DNS-only CNAME for the CoreWeave controller ingress."""

    def __init__(
        self,
        name: str,
        args: FederationDnsArgs,
        *,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:coreweave:FederationDns", name, None, opts)

        provider = cloudflare.Provider(
            "cloudflare",
            api_token=args.api_token,
            opts=pulumi.ResourceOptions(parent=self),
        )
        record = cloudflare.DnsRecord(
            "federation-cname",
            zone_id=args.spec.zone_id,
            name=args.spec.hostname,
            type="CNAME",
            content=args.spec.target,
            ttl=300,
            proxied=False,
            opts=pulumi.ResourceOptions(parent=self, provider=provider, protect=True),
        )
        self.register_outputs({"hostname": record.name})
