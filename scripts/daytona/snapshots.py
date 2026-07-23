#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit and reclaim stale Daytona snapshots."""

import argparse
import json

from marin.daytona.client import create_daytona_client
from marin.daytona.config import DaytonaConfig, resolve_daytona_credentials
from marin.daytona.snapshots import SnapshotAuditRow, audit_snapshots, delete_audited_snapshots, list_snapshots


def _audit_json(rows: list[SnapshotAuditRow]) -> str:
    """Render a snapshot audit without SDK objects or credentials."""

    return json.dumps(
        [
            {
                "snapshot_id": row.snapshot_id,
                "name": row.name,
                "state": row.state,
                "idle_days": row.idle_days,
                "protected": row.protected,
                "delete_eligible": row.delete_eligible,
            }
            for row in rows
        ],
        indent=2,
        sort_keys=True,
    )


def _confirm(count: int, assume_yes: bool) -> bool:
    if not count:
        return False
    if assume_yes:
        return True
    try:
        response = input(f"Delete these {count} snapshots? [y/N] ").strip().lower()
    except EOFError:
        return False
    return response in {"y", "yes"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Daytona snapshots; deletion is opt-in.")
    parser.add_argument("--api-key-env", default="DAYTONA_API_KEY")
    parser.add_argument("--endpoint")
    parser.add_argument("--target")
    parser.add_argument("--stale-after-days", type=float, default=14.0)
    parser.add_argument("--name-prefix", help="Only this explicit snapshot namespace is deletable.")
    parser.add_argument("--delete", action="store_true", help="Delete audit-selected snapshots.")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion without an interactive prompt.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable audit.")
    args = parser.parse_args()
    if args.yes and not args.delete:
        parser.error("--yes requires --delete")
    if args.delete and not args.name_prefix:
        parser.error("--delete requires a non-empty --name-prefix")
    if args.json and args.delete and not args.yes:
        parser.error("--json --delete requires --yes")
    credentials = resolve_daytona_credentials(DaytonaConfig(args.endpoint, args.target, args.api_key_env))
    client = create_daytona_client(credentials)
    rows = audit_snapshots(
        list_snapshots(client.snapshot), stale_after_days=args.stale_after_days, name_prefix=args.name_prefix
    )
    if args.json:
        print(_audit_json(rows))
    else:
        for row in rows:
            idle = "n/a" if row.idle_days is None else f"{row.idle_days:.1f}d"
            print(f"{row.snapshot_id}\t{row.name}\t{row.state}\t{idle}\teligible={row.delete_eligible}")
    if not args.delete:
        return 0
    eligible_count = sum(row.delete_eligible for row in rows)
    confirmed = _confirm(eligible_count, args.yes)
    if eligible_count and not confirmed:
        return 3
    deleted = delete_audited_snapshots(client.snapshot, rows, confirm=lambda _: confirmed)
    if not args.json:
        print(f"deleted {len(deleted)} snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
