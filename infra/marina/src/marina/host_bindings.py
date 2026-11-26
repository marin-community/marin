# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hostname validation, applet host access, DNS zone resolution and applet-host import."""

import re
from collections.abc import Iterable
from enum import StrEnum

from sqlalchemy import text
from sqlalchemy.engine import Connection

HOSTNAME_PATTERN = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


def normalize_hostname(hostname: str) -> str:
    hostname = hostname.lower().rstrip(".")
    if HOSTNAME_PATTERN.fullmatch(hostname) is None:
        raise ValueError("hostname must be a DNS name without a scheme, port, or path")
    return hostname


def dns_zone_name(hostname: str, zone_names: Iterable[str]) -> str:
    """Return the most specific configured DNS zone that contains ``hostname``."""
    matches = [zone for zone in zone_names if hostname == zone or hostname.endswith(f".{zone}")]
    if not matches:
        raise ValueError(f"no DNS zone is configured for {hostname}")
    return max(matches, key=len)


def host_resource_slug(hostname: str, zone_name: str, primary_zone_name: str) -> str:
    """Name a host's Pulumi resources by its first label, qualified outside the primary zone.

    Hosts in the primary zone keep their first label so existing resource names do not change.
    """
    label = hostname.split(".")[0]
    if zone_name == primary_zone_name:
        return label
    return f"{label}-{zone_name.replace('.', '-')}"


class AppletHostAccess(StrEnum):
    """The Cloud Run surface that serves a named applet host."""

    AUTHENTICATED = "authenticated"
    PUBLIC = "public"


def applet_host_access(applet_domains: list[str], public_applet_domains: list[str]) -> dict[str, AppletHostAccess]:
    """Map each configured applet host to the surface that serves it; a host may be in only one list."""
    if overlap := set(applet_domains) & set(public_applet_domains):
        raise ValueError(f"applet hosts {sorted(overlap)} are in both applet_domains and public_applet_domains")
    return {
        **{host: AppletHostAccess.AUTHENTICATED for host in applet_domains},
        **{host: AppletHostAccess.PUBLIC for host in public_applet_domains},
    }


def applet_domains_env(hosts: dict[str, AppletHostAccess]) -> dict[AppletHostAccess, str]:
    """``MARINA_APPLET_DOMAINS`` per surface.

    The authenticated service lists every host so that operators can bind public hosts through it;
    the public reader lists only the hosts it serves.
    """
    return {
        AppletHostAccess.AUTHENTICATED: ",".join(hosts),
        AppletHostAccess.PUBLIC: ",".join(host for host, access in hosts.items() if access is AppletHostAccess.PUBLIC),
    }


def domain_mapping_name(slug: str, access: AppletHostAccess) -> str:
    """Pulumi logical name of a host's Cloud Run domain mapping, from the host's resource slug.

    The name carries the surface. Moving a host between surfaces in one deploy therefore creates a
    second mapping for the same hostname, which Cloud Run rejects, instead of silently keeping the
    old route (the route itself is ignored on update).
    """
    return f"{slug}-domain" if access is AppletHostAccess.AUTHENTICATED else f"{slug}-public-domain"


def migrate_host_bindings(connection: Connection) -> None:
    """Import existing bindings once, without resetting later rebindings or deletions."""
    connection.execute(
        text(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
    )
    pending = connection.execute(
        text("INSERT INTO schema_migrations (name) VALUES ('m0001_applet_hosts') ON CONFLICT DO NOTHING RETURNING name")
    ).scalar()
    if pending is None:
        return
    connection.execute(
        text(
            """CREATE TABLE applet_hosts (
            hostname TEXT PRIMARY KEY,
            applet_id UUID REFERENCES applets(id),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )"""
        )
    )
    # This is the complete host-to-applet inventory from Pulumi.marin-marina.yaml.
    # Fresh development databases do not contain the production applet.
    connection.execute(
        text(
            """INSERT INTO applet_hosts (hostname, applet_id)
            SELECT 'filer.marina.oa.dev', id FROM applets
            WHERE id = '7f536ce3-0562-40f1-b209-e6110d8ab4e8'"""
        )
    )
