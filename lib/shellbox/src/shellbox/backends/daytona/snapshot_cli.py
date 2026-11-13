# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operate Daytona snapshots without ad-hoc scripts.

``shellbox-daytona-snapshots`` compares an account inventory with supplied
snapshot specs and reclaims stale storage before submitting a campaign.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from daytona import AsyncDaytona
from daytona.common.snapshot import Snapshot

from shellbox.backends.daytona.connection import (
    DEFAULT_DAYTONA_API_URL,
    DaytonaConnection,
)
from shellbox.backends.daytona.snapshot import (
    SNAPSHOT_STALE_AFTER,
    SnapshotClassification,
    SnapshotRequest,
    classify_inventory,
    classify_snapshot,
    ensure_snapshot,
    list_snapshots,
    purge_stale_snapshots,
    snapshot_last_used,
)


def _load_specs(source: str) -> list[SnapshotRequest]:
    """Read the snapshot-spec interchange emitted by ``trial_driver snapshot-specs``."""
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(entry, dict) for entry in payload):
        raise SystemExit("Snapshot specs must be one JSON object or a list of objects")
    return [SnapshotRequest.from_dict(entry) for entry in payload]


def _connection(arguments: argparse.Namespace) -> DaytonaConnection:
    api_key = os.environ.get(arguments.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing Daytona credential environment variable: {arguments.api_key_env}")
    if not arguments.target:
        raise SystemExit("Set --target or DAYTONA_TARGET")
    return DaytonaConnection(api_key=api_key, api_url=arguments.api_url, target=arguments.target)


def _row(name: str, state: str, cpus: int | None, memory: int | None, disk: int | None) -> tuple[str, ...]:
    return (
        name,
        state,
        "-" if cpus is None else str(cpus),
        "-" if memory is None else str(memory),
        "-" if disk is None else str(disk),
    )


def _print_rows(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(column) for column in header]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    print("  ".join(column.ljust(width) for column, width in zip(header, widths, strict=True)))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))


def _inventory_entries(
    snapshots: list[Snapshot],
    *,
    now: datetime,
    minimum_idle: timedelta,
) -> list[dict[str, Any]]:
    entries = []
    for snapshot in snapshots:
        last_used = snapshot_last_used(snapshot)
        entries.append(
            {
                "name": snapshot.name,
                "state": snapshot.state.value,
                "cpu": snapshot.cpu,
                "memory_gib": snapshot.mem,
                "disk_gib": snapshot.disk,
                "gpu": snapshot.gpu,
                "general": snapshot.general,
                "organization_id": snapshot.organization_id,
                "last_used_at": None if last_used is None else last_used.isoformat(),
                "classification": classify_snapshot(snapshot, now=now, minimum_idle=minimum_idle).value,
            }
        )
    return entries


async def _inventory(
    client: AsyncDaytona,
    *,
    now: datetime,
    minimum_idle: timedelta,
    as_json: bool,
) -> None:
    entries = _inventory_entries(await list_snapshots(client), now=now, minimum_idle=minimum_idle)
    if as_json:
        print(json.dumps(entries, indent=2, sort_keys=True))
        return
    rows = [
        (
            entry["name"],
            entry["state"],
            entry["classification"],
            entry["last_used_at"] or "-",
            f"{entry['cpu']}cpu/{entry['memory_gib']}g/{entry['disk_gib']}g",
        )
        for entry in entries
    ]
    _print_rows(("NAME", "STATE", "CLASS", "LAST USED", "RESOURCES"), rows)


async def _plan(client: AsyncDaytona, requests: list[SnapshotRequest]) -> None:
    present = {snapshot.name: snapshot for snapshot in await list_snapshots(client)}
    rows = []
    for request in requests:
        snapshot = present.get(request.name)
        rows.append(
            _row(
                request.name,
                "missing" if snapshot is None else snapshot.state.value,
                request.resources.cpu,
                request.resources.memory,
                request.resources.disk,
            )
        )
    _print_rows(("NAME", "STATE", "CPU", "MEMORY_GIB", "DISK_GIB"), rows)


async def _ensure(client: AsyncDaytona, requests: list[SnapshotRequest]) -> None:
    rows = []
    for request in requests:
        name = await ensure_snapshot(client, request)
        rows.append(
            _row(
                name,
                "active",
                request.resources.cpu,
                request.resources.memory,
                request.resources.disk,
            )
        )
    _print_rows(("NAME", "STATE", "CPU", "MEMORY_GIB", "DISK_GIB"), rows)


async def _purge_dry_run(client: AsyncDaytona, *, now: datetime, minimum_idle: timedelta) -> None:
    classes = classify_inventory(await list_snapshots(client), now=now, minimum_idle=minimum_idle)
    stale = [name for name, classification in classes.items() if classification is SnapshotClassification.STALE]
    unclassified = [
        name for name, classification in classes.items() if classification is SnapshotClassification.UNCLASSIFIED
    ]
    for name in stale:
        print(f"would delete {name}")
    for name in unclassified:
        print(f"unclassified {name} (missing last-use evidence)")
    if not stale and not unclassified:
        print("no stale snapshots")


async def _purge_apply(client: AsyncDaytona, *, now: datetime, minimum_idle: timedelta) -> None:
    report = await purge_stale_snapshots(client, now=now, minimum_idle=minimum_idle)
    for name in report.deleted:
        print(f"deleted {name}")
    for name in report.skipped:
        print(f"kept {name} (used since inventory)")
    for name in report.unclassified:
        print(f"unclassified {name} (missing last-use evidence)")
    for name in report.remaining:
        print(f"removal pending {name}")
    if not report.deleted:
        print("no stale snapshots deleted")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shellbox-daytona-snapshots",
        description=(
            "Inventory, plan, require and reclaim Daytona snapshots identified "
            "by Shellbox image sources and resources."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        required=True,
        help="Environment variable holding the selected account's API key.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("DAYTONA_API_URL", DEFAULT_DAYTONA_API_URL),
        help="Daytona API URL.",
    )
    parser.add_argument(
        "--target",
        default=os.environ.get("DAYTONA_TARGET"),
        help="Daytona target region (defaults to DAYTONA_TARGET).",
    )
    parser.add_argument(
        "--min-idle-seconds",
        type=int,
        default=int(SNAPSHOT_STALE_AFTER.total_seconds()),
        help="Idle threshold before a snapshot is stale.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="List every snapshot and its stale classification.")
    inventory.add_argument("--json", action="store_true", help="Emit JSON.")

    for name, help_text in (
        ("plan", "Compare required snapshots with the account inventory."),
        ("ensure", "Create required snapshots that are missing."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--specs",
            required=True,
            help="Snapshot-spec JSON file; use - to read stdin.",
        )

    purge = commands.add_parser("purge", help="Delete every stale account snapshot.")
    purge.add_argument(
        "--apply",
        action="store_true",
        help="Delete targets; without this flag the command only reports them.",
    )
    return parser


async def _run(arguments: argparse.Namespace) -> None:
    connection = _connection(arguments)
    minimum_idle = timedelta(seconds=arguments.min_idle_seconds)
    now = datetime.now(UTC)
    async with connection.client_factory()() as client:
        if arguments.command == "inventory":
            await _inventory(
                client,
                now=now,
                minimum_idle=minimum_idle,
                as_json=arguments.json,
            )
        elif arguments.command in {"plan", "ensure"}:
            requests = _load_specs(arguments.specs)
            if arguments.command == "plan":
                await _plan(client, requests)
            else:
                await _ensure(client, requests)
        else:
            purge = _purge_apply if arguments.apply else _purge_dry_run
            await purge(
                client,
                now=now,
                minimum_idle=minimum_idle,
            )


def main() -> None:
    arguments = _parser().parse_args()
    asyncio.run(_run(arguments))


if __name__ == "__main__":
    main()
