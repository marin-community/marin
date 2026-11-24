# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory or delete idle sandboxes in explicitly selected Daytona accounts."""

import argparse
import asyncio
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TextIO

from daytona import AsyncDaytona, AsyncSandbox, DaytonaError

from shellbox.backends.daytona.connection import ACCOUNT_KEY_ENVS, DEFAULT_DAYTONA_API_URL, DaytonaConnection
from shellbox.backends.daytona.inventory import inventory_sandboxes
from shellbox.backends.daytona.stale import DeletionState, SandboxActivity, SandboxDecision, select_idle_sandboxes

REQUEST_TIMEOUT = 60
DELETE_INTERVAL = 1.0
DEFAULT_THRESHOLD_MINUTES = 60


@dataclass(frozen=True)
class PurgeCounts:
    deleted: int
    skipped: int
    errors: int


def _activity(sandbox: AsyncSandbox) -> SandboxActivity:
    state = sandbox.state.value if sandbox.state else ""
    deletion_state = DeletionState.NOT_DELETING
    if state in (DeletionState.DESTROYING, DeletionState.DESTROYED):
        deletion_state = DeletionState(state)
    return SandboxActivity(sandbox.id, sandbox.last_activity_at, sandbox.updated_at, deletion_state)


def _record(ledger: TextIO, account: str, decision: SandboxDecision, outcome: str) -> None:
    ledger.write(
        json.dumps(
            {
                "account": account,
                **asdict(decision),
                "last_used_at": decision.last_used_at.isoformat() if decision.last_used_at else None,
                "recorded_at": datetime.now(UTC).isoformat(),
                "outcome": outcome,
            }
        )
        + "\n"
    )
    ledger.flush()
    os.fsync(ledger.fileno())


async def _delete_candidates(
    client: AsyncDaytona,
    account: str,
    candidates: tuple[SandboxDecision, ...],
    minimum_idle: timedelta,
    ledger: TextIO,
) -> PurgeCounts:
    """Recheck candidate IDs and return confirmed deletions, skips and errors."""
    deleted = skipped = errors = 0
    for decision in candidates:
        # Throttle rechecks and deletions together; no bursts within an org.
        await asyncio.sleep(DELETE_INTERVAL)
        try:
            sandbox = await client.get(decision.id, request_timeout=REQUEST_TIMEOUT)
            if sandbox.id != decision.id:
                raise DaytonaError("Daytona returned a different sandbox ID")
            rechecked = select_idle_sandboxes([_activity(sandbox)], now=datetime.now(UTC), minimum_idle=minimum_idle)
            if not rechecked.candidates:
                skipped += 1
                _record(ledger, account, rechecked.skipped[0], "skipped on recheck")
                continue
            decision = rechecked.candidates[0]
            _record(ledger, account, decision, "delete requested")
            await sandbox.delete(timeout=REQUEST_TIMEOUT, wait=True)
        except DaytonaError as error:
            errors += 1
            # Provider exception messages can contain credential-bearing URLs.
            outcome = f"error: {type(error).__name__}"
            _record(ledger, account, decision, outcome)
            print(f"  {decision.id}: {outcome}", flush=True)
            continue
        _record(ledger, account, decision, "deleted")
        deleted += 1
        if deleted % 50 == 0:
            print(f"{account}: deleted={deleted} errors={errors}", flush=True)
    return PurgeCounts(deleted, skipped, errors)


async def _account_error_count(
    account: str, connection: DaytonaConnection, minimum_idle: timedelta, ledger: TextIO | None
) -> int:
    """Report one account, optionally purge idle IDs, and return its error count."""
    async with connection.client_factory()() as client:
        sandboxes = await inventory_sandboxes(client)
        selection = select_idle_sandboxes(map(_activity, sandboxes), now=datetime.now(UTC), minimum_idle=minimum_idle)
        print(
            f"{account}: total={len(sandboxes)} idle_candidates={len(selection.candidates)} "
            f"skipped={len(selection.skipped)}",
            flush=True,
        )
        for reason, count in Counter(row.reason for row in selection.skipped).items():
            print(f"  skipped={count}: {reason}", flush=True)
        if ledger is None:
            for row in selection.candidates:
                print(f"  candidate {row.id}: {row.reason}, last_used_at={row.last_used_at}", flush=True)
            return 0
        counts = await _delete_candidates(client, account, selection.candidates, minimum_idle, ledger)
        print(
            f"{account}: deleted={counts.deleted} skipped={len(selection.skipped) + counts.skipped} "
            f"errors={counts.errors}",
            flush=True,
        )
        return counts.errors


async def _run(arguments: argparse.Namespace, ledger: TextIO | None) -> int:
    accounts = ACCOUNT_KEY_ENVS.items() if arguments.all else ((arguments.api_key_env, arguments.api_key_env),)
    errors = 0
    for account, name in accounts:
        api_key = os.environ.get(name)
        if not api_key:
            print(f"{account}: error: missing {name}", flush=True)
            errors += 1
            continue
        connection = DaytonaConnection(api_key=api_key, api_url=arguments.api_url, target=arguments.target)
        try:
            errors += await _account_error_count(account, connection, timedelta(minutes=arguments.threshold), ledger)
        except DaytonaError as error:
            print(f"{account}: inventory error: {type(error).__name__}", flush=True)
            errors += 1
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge idle sandboxes account-wide, including those created outside Shellbox; dry-run by default."
    )
    account = parser.add_mutually_exclusive_group(required=True)
    account.add_argument(
        "--all", action="store_true", help="Select eval, rl and data via their dedicated API key variables."
    )
    account.add_argument("--api-key-env", help="Environment variable holding the selected account's API key.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Report candidates without deletion (default).")
    mode.add_argument("--delete", action="store_true", help="Delete idle sandbox IDs after an activity recheck.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_MINUTES,
        help=f"Minimum idle time in minutes (default: {DEFAULT_THRESHOLD_MINUTES}).",
    )
    parser.add_argument("--ledger", type=Path, help="Append-only JSONL ledger; required with --delete.")
    parser.add_argument("--target", default=os.environ.get("DAYTONA_TARGET"), help="Daytona target region.")
    parser.add_argument("--api-url", default=os.environ.get("DAYTONA_API_URL", DEFAULT_DAYTONA_API_URL))
    arguments = parser.parse_args()
    if not arguments.target:
        parser.error("Set --target or DAYTONA_TARGET")
    if not math.isfinite(arguments.threshold) or arguments.threshold <= 0:
        parser.error("--threshold must be finite and positive")
    if arguments.delete and arguments.ledger is None:
        parser.error("--delete requires --ledger")
    if arguments.ledger is not None and not arguments.delete:
        parser.error("--ledger requires --delete")
    if arguments.delete:
        with arguments.ledger.open("a", encoding="utf-8") as ledger:
            errors = asyncio.run(_run(arguments, ledger))
        print(f"ledger: {arguments.ledger}")
    else:
        errors = asyncio.run(_run(arguments, None))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
