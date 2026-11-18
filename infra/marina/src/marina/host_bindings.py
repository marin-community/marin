# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hostname validation, DNS zone resolution and the one-time import of revision-configured applet hosts."""

import re
from collections.abc import Iterable

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
