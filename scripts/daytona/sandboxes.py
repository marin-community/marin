#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit and reclaim stale Daytona sandboxes."""

import argparse

from marin.daytona.client import create_daytona_client
from marin.daytona.config import DaytonaConfig, resolve_daytona_credentials
from marin.daytona.sandboxes import audit_sandboxes, delete_audited_sandboxes


def _confirm(count: int, assume_yes: bool) -> bool:
    if not assume_yes:
        return False
    return count > 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Daytona sandboxes; deletion is opt-in.")
    parser.add_argument("--api-key-env", default="DAYTONA_API_KEY")
    parser.add_argument("--endpoint")
    parser.add_argument("--target")
    parser.add_argument("--stale-after-minutes", type=float, default=60.0)
    parser.add_argument("--id-prefix", help="Restrict deletion to sandbox ids with this prefix.")
    parser.add_argument("--delete", action="store_true", help="Delete audit-selected sandboxes.")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion without an interactive prompt.")
    args = parser.parse_args()
    if args.yes and not args.delete:
        parser.error("--yes requires --delete")
    if args.delete and not args.id_prefix:
        parser.error("--delete requires a non-empty --id-prefix")
    credentials = resolve_daytona_credentials(DaytonaConfig(args.endpoint, args.target, args.api_key_env))
    client = create_daytona_client(credentials)
    rows = audit_sandboxes(client.list(), stale_after_minutes=args.stale_after_minutes, id_prefix=args.id_prefix)
    for row in rows:
        age = "n/a" if row.age_minutes is None else f"{row.age_minutes:.1f}m"
        print(f"{row.sandbox_id}\t{row.state}\t{age}\t{row.reason}")
    if not args.delete:
        return 0
    results = delete_audited_sandboxes(rows, confirm=lambda count: _confirm(count, args.yes))
    for result in results:
        if result.error is None:
            print(f"deleted\t{result.sandbox_id}")
        else:
            print(f"failed\t{result.sandbox_id}\t{result.error}")
    return int(any(result.error for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
