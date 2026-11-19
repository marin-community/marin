# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report a Daytona account's quota and live usage to size concurrency.

``shellbox-daytona-usage`` reads the account's sandbox inventory and usage
overview through one explicitly selected API key and prints the remaining
capacity and a suggested maximum concurrency.
"""

import argparse
import asyncio
import json
import os
from typing import Any

from shellbox.backends.daytona.connection import DEFAULT_DAYTONA_API_URL, DaytonaConnection
from shellbox.backends.daytona.usage import (
    DEFAULT_HEADROOM,
    AccountUsage,
    RegionCapacity,
    SandboxProfile,
    account_usage,
    max_concurrency,
    usage_report,
)

# The three known accounts in report order. The eval account's key is
# DAYTONA_API_KEY; some launchers load the same key as DAYTONA_EVAL_API_KEY.
ALL_ACCOUNTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("eval", ("DAYTONA_API_KEY", "DAYTONA_EVAL_API_KEY")),
    ("rl", ("DAYTONA_RL_API_KEY",)),
    ("data", ("DAYTONA_DATA_API_KEY",)),
)


def _api_key(names: tuple[str, ...]) -> str | None:
    for name in names:
        if value := os.environ.get(name):
            return value
    return None


def _connection(api_key: str, arguments: argparse.Namespace) -> DaytonaConnection:
    if not arguments.target:
        raise SystemExit("Set --target or DAYTONA_TARGET")
    return DaytonaConnection(api_key=api_key, api_url=arguments.api_url, target=arguments.target)


def _profile(arguments: argparse.Namespace) -> SandboxProfile | None:
    given = [arguments.cpu is not None, arguments.memory_gib is not None, arguments.disk_gib is not None]
    if any(given) and not all(given):
        raise SystemExit("Provide all of --cpu, --memory-gib and --disk-gib to size concurrency")
    if arguments.headroom is not None and not all(given):
        raise SystemExit("--headroom sizes concurrency; provide --cpu, --memory-gib and --disk-gib")
    if arguments.cpu is None:
        return None
    return SandboxProfile(cpu=arguments.cpu, memory_gib=arguments.memory_gib, disk_gib=arguments.disk_gib)


def _number(value: float | None) -> str:
    if value is None:
        return "-"
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def _over_maximum_line(profile: SandboxProfile, usage: AccountUsage) -> str | None:
    for region in usage.regions:
        maxima = (
            ("cpu", profile.cpu, region.max_cpu_per_sandbox),
            ("memory-gib", profile.memory_gib, region.max_memory_gib_per_sandbox),
            ("disk-gib", profile.disk_gib, region.max_disk_gib_per_sandbox),
        )
        exceeded = [
            f"--{name} {value} exceeds the per-sandbox maximum {_number(maximum)} "
            f"in {region.region_id} ({region.sandbox_class}); requests above the maximum fail at creation"
            for name, value, maximum in maxima
            if maximum is not None and value > maximum
        ]
        if exceeded:
            return "; ".join(exceeded)
    return None


def _capacity_rows(usage: AccountUsage) -> list[tuple[str, str, str, str]]:
    return [
        (name, _number(capacity.quota), _number(capacity.used), _number(capacity.remaining))
        for name, capacity in (
            ("cpu", usage.cpu),
            ("memory_gib", usage.memory_gib),
            ("disk_gib", usage.disk_gib),
            ("snapshots", usage.snapshots),
            ("volumes", usage.volumes),
        )
    ]


def _region_rows(region: RegionCapacity) -> list[tuple[str, str, str, str]]:
    return [
        (name, _number(capacity.quota), _number(capacity.used), _number(capacity.remaining))
        for name, capacity in (
            ("cpu", region.cpu),
            ("memory_gib", region.memory_gib),
            ("disk_gib", region.disk_gib),
            ("gpu", region.gpu),
        )
    ]


def _print_account(usage: AccountUsage, profile: SandboxProfile | None, headroom: float) -> None:
    print(f"organization  {usage.organization_id}")
    print("  " + "  ".join(column.ljust(12) for column in ("resource", "quota", "used", "remaining")))
    for row in _capacity_rows(usage):
        print("  " + "  ".join(column.ljust(12) for column in row))
    if len(usage.regions) > 1:
        for region in usage.regions:
            print(f"  region {region.region_id} ({region.sandbox_class})")
            for row in _region_rows(region):
                print("  " + "  ".join(column.ljust(12) for column in row))
    elif usage.regions:
        print(f"  region {usage.regions[0].region_id} ({usage.regions[0].sandbox_class})")
    states = "  ".join(f"{state} {count}" for state, count in usage.sandbox_states.items()) or "none"
    print(f"  sandboxes {usage.sandboxes}  states: {states}")
    if profile is not None:
        print(
            f"  max concurrency {max_concurrency(usage, profile, headroom=headroom)} sandboxes "
            f"at {profile.cpu}cpu/{_number(profile.memory_gib)}g/{_number(profile.disk_gib)}g, "
            f"{headroom:.0%} headroom"
        )
        exceeded = _over_maximum_line(profile, usage)
        if exceeded:
            print(f"  warning: {exceeded}")


async def _run(arguments: argparse.Namespace) -> None:
    profile = _profile(arguments)
    headroom = DEFAULT_HEADROOM if arguments.headroom is None else arguments.headroom
    if arguments.all:
        reports: dict[str, Any] = {}
        for account, names in ALL_ACCOUNTS:
            api_key = _api_key(names)
            if api_key is None:
                if arguments.json:
                    reports[account] = {"skipped": f"missing {' or '.join(names)}"}
                else:
                    print(f"{account}: skipped (missing {' or '.join(names)})")
                continue
            usage = await account_usage(_connection(api_key, arguments))
            if arguments.json:
                reports[account] = usage_report(usage, profile, headroom=headroom)
            else:
                print(f"account {account}")
                _print_account(usage, profile, headroom)
        if arguments.json:
            print(json.dumps(reports, indent=2, sort_keys=True))
        return

    api_key = _api_key((arguments.api_key_env,))
    if api_key is None:
        raise SystemExit(f"Missing Daytona credential environment variable: {arguments.api_key_env}")
    usage = await account_usage(_connection(api_key, arguments))
    if arguments.json:
        print(json.dumps(usage_report(usage, profile, headroom=headroom), indent=2, sort_keys=True))
    else:
        _print_account(usage, profile, headroom)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shellbox-daytona-usage",
        description=(
            "Report a Daytona account's quota, live usage and sandbox inventory, and size "
            "the maximum additional concurrency from the remaining capacity."
        ),
    )
    account = parser.add_mutually_exclusive_group(required=True)
    account.add_argument(
        "--api-key-env",
        help="Environment variable holding the selected account's API key.",
    )
    account.add_argument(
        "--all",
        action="store_true",
        help="Read the three known accounts (eval, rl, data) from their standard key variables.",
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
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--cpu", type=int, help="Per-sandbox CPU request for concurrency sizing.")
    parser.add_argument("--memory-gib", type=float, help="Per-sandbox memory request in GiB.")
    parser.add_argument("--disk-gib", type=float, help="Per-sandbox disk request in GiB.")
    parser.add_argument(
        "--headroom",
        type=float,
        default=None,
        help="Fraction of remaining capacity kept in reserve (default 20%%).",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    asyncio.run(_run(arguments))


if __name__ == "__main__":
    main()
