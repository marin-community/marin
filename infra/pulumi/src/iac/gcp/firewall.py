# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared GCE firewall rule declarations."""

from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp

from iac.imports import NO_IMPORTS, ImportRegistrar


@dataclass(frozen=True)
class FirewallPort:
    protocol: str
    ports: tuple[str, ...]


@dataclass(frozen=True)
class GcpFirewallRuleArgs:
    project: str
    name: str
    source_ranges: tuple[str, ...]
    target_tags: tuple[str, ...]
    priority: int
    allows: tuple[FirewallPort, ...] = ()
    denies: tuple[FirewallPort, ...] = ()
    network: str = "default"
    direction: str = "INGRESS"


def create_firewall_rule(
    logical_name: str,
    args: GcpFirewallRuleArgs,
    *,
    gcp_provider: pulumi.ProviderResource | None = None,
    parent: pulumi.Resource | None = None,
    imports: ImportRegistrar = NO_IMPORTS,
    opts: pulumi.ResourceOptions | None = None,
) -> gcp.compute.Firewall:
    """Declare one non-authoritative GCE firewall rule and catalog its import ID."""
    if bool(args.allows) == bool(args.denies):
        raise ValueError("a firewall rule must define exactly one of allows or denies")

    resource_options = pulumi.ResourceOptions.merge(
        opts,
        pulumi.ResourceOptions(parent=parent, provider=gcp_provider),
    )
    resource = gcp.compute.Firewall(
        logical_name,
        project=args.project,
        name=args.name,
        network=args.network,
        direction=args.direction,
        priority=args.priority,
        source_ranges=list(args.source_ranges),
        target_tags=list(args.target_tags),
        allows=(
            [gcp.compute.FirewallAllowArgs(protocol=rule.protocol, ports=list(rule.ports)) for rule in args.allows]
            if args.allows
            else None
        ),
        denies=(
            [gcp.compute.FirewallDenyArgs(protocol=rule.protocol, ports=list(rule.ports)) for rule in args.denies]
            if args.denies
            else None
        ),
        opts=resource_options,
    )
    imports.register(resource, parent=parent, provider_id=f"projects/{args.project}/global/firewalls/{args.name}")
    return resource
