# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report a Daytona project's limits and live usage to size concurrency.

``shellbox-daytona-usage`` reads the project's limits and sandbox inventory
through one explicitly selected API key and prints the remaining capacity and
a suggested maximum concurrency.
"""

import argparse
import asyncio
import json
import os
from typing import Any

from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.usage import (
    DEFAULT_HEADROOM,
    Capacity,
    ProjectUsage,
    SandboxProfile,
    max_concurrency,
    profile_violations,
    project_usage,
    usage_report,
)

# The accounts use dedicated keys, not the SDK's ambient DAYTONA_API_KEY.
ALL_ACCOUNTS: tuple[tuple[str, str], ...] = (
    ("eval", "DAYTONA_EVAL_API_KEY"),
    ("rl", "DAYTONA_RL_API_KEY"),
    ("data", "DAYTONA_DATA_API_KEY"),
)


def _profile(arguments: argparse.Namespace) -> SandboxProfile | None:
    if (arguments.cpu is None) != (arguments.memory_gib is None):
        raise SystemExit("Provide both --cpu and --memory-gib to check a sandbox profile")
    if arguments.cpu is None:
        return None
    return SandboxProfile(cpus=arguments.cpu, memory_gib=arguments.memory_gib)


def _cell(value: float | None) -> str:
    if value is None:
        return "-"
    return str(int(value)) if value == int(value) else f"{value:.1f}"


def _print_project(usage: ProjectUsage, profile: SandboxProfile | None, headroom: float) -> None:
    print(f"organization {usage.organization}  project {usage.project}")
    print("  " + "  ".join(column.ljust(20) for column in ("limit", "max", "used", "remaining")))
    rows: tuple[tuple[str, Capacity], ...] = (
        ("concurrent sandboxes", usage.sandboxes),
        ("images", usage.images),
        ("volumes", usage.volumes),
        ("snapshots", usage.snapshots),
    )
    for name, capacity in rows:
        cells = (name, _cell(capacity.limit), _cell(capacity.used), _cell(capacity.remaining))
        print("  " + "  ".join(cell.ljust(20) for cell in cells))
    print(
        f"  per sandbox: {_cell(usage.max_cores_per_sandbox)} cores, "
        f"{_cell(usage.max_memory_gib_per_sandbox)} GiB memory"
    )
    states = "  ".join(f"{state} {count}" for state, count in usage.sandbox_states) or "none"
    print(f"  sandboxes: {states}")
    limit = max_concurrency(usage, headroom=headroom)
    print(f"  max concurrency {'unlimited' if limit is None else limit} more sandboxes, {headroom:.0%} headroom")
    if profile is not None:
        for violation in profile_violations(usage, profile):
            print(f"  warning: {violation}; such creates fail")


async def _run(arguments: argparse.Namespace) -> None:
    profile = _profile(arguments)
    headroom = arguments.headroom
    accounts = ALL_ACCOUNTS if arguments.all else (("selected", arguments.api_key_env),)
    reports: dict[str, Any] = {}
    for account, name in accounts:
        api_key = os.environ.get(name)
        if not api_key:
            if not arguments.all:
                raise SystemExit(f"Missing Daytona credential environment variable: {name}")
            reports[account] = {"skipped": f"missing {name}"}
            if not arguments.json:
                print(f"{account}: skipped (missing {name})")
            continue
        usage = await project_usage(DaytonaConnection(api_key=api_key))
        reports[account] = usage_report(usage, profile, headroom=headroom)
        if not arguments.json:
            if arguments.all:
                print(f"account {account}")
            _print_project(usage, profile, headroom)
    if arguments.json:
        print(json.dumps(reports if arguments.all else reports["selected"], indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shellbox-daytona-usage",
        description=(
            "Report a Daytona project's limits, live usage and sandbox inventory, and size "
            "the maximum additional concurrency from the concurrent-sandbox limit."
        ),
    )
    account = parser.add_mutually_exclusive_group(required=True)
    account.add_argument("--api-key-env", help="Environment variable holding the selected account's API key.")
    account.add_argument(
        "--all",
        action="store_true",
        help="Read the three known accounts (eval, rl, data) from their standard key variables.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--cpu", type=int, help="Per-sandbox vCPU request to check against the maxima.")
    parser.add_argument("--memory-gib", type=float, help="Per-sandbox memory request in GiB.")
    parser.add_argument(
        "--headroom",
        type=float,
        default=DEFAULT_HEADROOM,
        help=f"Fraction of the remaining sandbox limit kept in reserve (default {DEFAULT_HEADROOM * 100:.0f}%%).",
    )
    return parser


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()
