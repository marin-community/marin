# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hostname validation and the one-time import of revision-configured applet hosts."""

import re
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


def domain_mapping_name(host: str, access: AppletHostAccess) -> str:
    """Pulumi logical name of a host's Cloud Run domain mapping.

    The name carries the surface. Moving a host between surfaces in one deploy therefore creates a
    second mapping for the same hostname, which Cloud Run rejects, instead of silently keeping the
    old route (the route itself is ignored on update).
    """
    slug = host.split(".")[0]
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
