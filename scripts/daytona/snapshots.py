#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit and reclaim stale Daytona snapshots."""

import argparse

from marin.daytona.client import create_daytona_client
from marin.daytona.config import DaytonaConfig, resolve_daytona_credentials
from marin.daytona.snapshots import audit_snapshots, delete_audited_snapshots, list_snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Daytona snapshots; deletion is opt-in.")
    parser.add_argument("--api-key-env", default="DAYTONA_API_KEY")
    parser.add_argument("--endpoint")
    parser.add_argument("--target")
    parser.add_argument("--stale-after-days", type=float, default=14.0)
    parser.add_argument("--name-prefix", required=True, help="Only this explicit snapshot namespace is deletable.")
    parser.add_argument("--delete", action="store_true", help="Delete audit-selected snapshots.")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion without an interactive prompt.")
    args = parser.parse_args()
    if args.yes and not args.delete:
        parser.error("--yes requires --delete")
    credentials = resolve_daytona_credentials(DaytonaConfig(args.endpoint, args.target, args.api_key_env))
    client = create_daytona_client(credentials)
    rows = audit_snapshots(
        list_snapshots(client.snapshot), stale_after_days=args.stale_after_days, name_prefix=args.name_prefix
    )
    for row in rows:
        idle = "n/a" if row.idle_days is None else f"{row.idle_days:.1f}d"
        print(f"{row.snapshot_id}\t{row.name}\t{row.state}\t{idle}\teligible={row.delete_eligible}")
    if not args.delete:
        return 0
    deleted = delete_audited_snapshots(client.snapshot, rows, confirm=lambda count: args.yes and count > 0)
    print(f"deleted {len(deleted)} snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
