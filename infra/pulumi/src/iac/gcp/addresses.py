# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reserved external static IPs (google_compute_address) for the GCP arm.

Currently holds the federation-egress IPs the CoreWeave controllers' ipAllowList admits — the
marin / marin-dev controller egress, mirrored as `IngressSpec.federation_allow_sources`.
"""

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp

from iac.config import GcpAddressSpec
from iac.imports import NO_IMPORTS, ImportRegistrar


@dataclass(frozen=True)
class GcpStaticAddressesArgs:
    project: str
    addresses: list[GcpAddressSpec]


def _import_id(project: str, address: GcpAddressSpec) -> str:
    # Regional google_compute_address import id.
    return f"projects/{project}/regions/{address.region}/addresses/{address.name}"


class GcpStaticAddresses(pulumi.ComponentResource):
    """Create one EXTERNAL google_compute_address per `args.addresses`.

    Each pins `address` to a fixed IP, so Pulumi owns the reservation and keeps that exact IP;
    Program-first adoption uses each resource's registered provider ID.
    """

    def __init__(
        self,
        name: str,
        args: GcpStaticAddressesArgs,
        *,
        gcp_provider: pulumi.ProviderResource,
        imports: ImportRegistrar = NO_IMPORTS,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:gcp:GcpStaticAddresses", name, None, opts)
        for address in args.addresses:
            resource = gcp.compute.Address(
                f"address-{address.name}",
                name=address.name,
                project=args.project,
                region=address.region,
                address=address.address,
                address_type="EXTERNAL",
                description=address.description,
                opts=pulumi.ResourceOptions(
                    parent=self,
                    provider=gcp_provider,
                    # These IPs are baked into every CoreWeave federation allowlist; a stray
                    # `pulumi destroy`/rename must never release the reservation.
                    retain_on_delete=True,
                ),
            )
            imports.register(resource, parent=self, provider_id=_import_id(args.project, address))
        self.register_outputs({})
