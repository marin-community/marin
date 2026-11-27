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
from collections.abc import Mapping
from typing import Any

from shellbox.backends.daytona.connection import DAYTONA_TARGET, DaytonaConnection
from shellbox.backends.daytona.usage import (
    DEFAULT_HEADROOM,
    DEFAULT_SANDBOX_CLASS,
    AccountUsage,
    RegionCapacity,
    ResourceCapacity,
    SandboxProfile,
    account_resource_capacities,
    account_usage,
    max_concurrency,
    region_for_launch,
    region_resource_capacities,
    usage_report,
)

# The accounts use dedicated keys, not the SDK's ambient DAYTONA_API_KEY.
ALL_ACCOUNTS: tuple[tuple[str, str], ...] = (
    ("eval", "DAYTONA_EVAL_API_KEY"),
    ("rl", "DAYTONA_RL_API_KEY"),
    ("data", "DAYTONA_DATA_API_KEY"),
)


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


def _over_maximum_line(profile: SandboxProfile, region: RegionCapacity) -> str | None:
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
    return "; ".join(exceeded) if exceeded else None


def _capacity_rows(capacities: Mapping[str, ResourceCapacity]) -> list[tuple[str, str, str, str]]:
    return [
        (name, _number(capacity.quota), _number(capacity.used), _number(capacity.remaining))
        for name, capacity in capacities.items()
    ]


def _print_account(
    usage: AccountUsage, profile: SandboxProfile | None, headroom: float, region: RegionCapacity | None
) -> None:
    print(f"organization  {usage.organization_id}")
    print("  " + "  ".join(column.ljust(12) for column in ("resource", "quota", "used", "remaining")))
    for row in _capacity_rows(account_resource_capacities(usage)):
        print("  " + "  ".join(column.ljust(12) for column in row))
    if len(usage.regions) > 1:
        for row_region in usage.regions:
            print(f"  region {row_region.region_id} ({row_region.sandbox_class})")
            for row in _capacity_rows(region_resource_capacities(row_region)):
                print("  " + "  ".join(column.ljust(12) for column in row))
    elif usage.regions:
        print(f"  region {usage.regions[0].region_id} ({usage.regions[0].sandbox_class})")
    states = "  ".join(f"{state} {count}" for state, count in usage.sandbox_states) or "none"
    print(f"  sandboxes {usage.sandboxes}  states: {states}")
    if profile is not None and region is not None:
        print(
            f"  max concurrency {max_concurrency(region, profile, headroom=headroom)} sandboxes "
            f"at {profile.cpu}cpu/{_number(profile.memory_gib)}g/{_number(profile.disk_gib)}g, "
            f"{headroom:.0%} headroom, in {region.region_id} ({region.sandbox_class})"
        )
        exceeded = _over_maximum_line(profile, region)
        if exceeded:
            print(f"  warning: {exceeded}")


def _launch_region(
    usage: AccountUsage, arguments: argparse.Namespace, profile: SandboxProfile | None
) -> RegionCapacity | None:
    """The quota row the launch is charged against, or None when nothing sizes."""
    if profile is None:
        return None
    return region_for_launch(usage, target=DAYTONA_TARGET, sandbox_class=arguments.sandbox_class)


async def _run(arguments: argparse.Namespace) -> None:
    profile = _profile(arguments)
    headroom = DEFAULT_HEADROOM if arguments.headroom is None else arguments.headroom
    if arguments.all:
        reports: dict[str, Any] = {}
        for account, name in ALL_ACCOUNTS:
            api_key = os.environ.get(name)
            if not api_key:
                if arguments.json:
                    reports[account] = {"skipped": f"missing {name}"}
                else:
                    print(f"{account}: skipped (missing {name})")
                continue
            usage = await account_usage(DaytonaConnection(api_key=api_key))
            try:
                region = _launch_region(usage, arguments, profile)
            except RuntimeError as error:
                # One account's missing quota row must not lose the reports
                # already read for the rest of the sweep.
                if arguments.json:
                    reports[account] = {"skipped": str(error)}
                else:
                    print(f"{account}: skipped ({error})")
                continue
            if arguments.json:
                reports[account] = usage_report(usage, profile, region=region, headroom=headroom)
            else:
                print(f"account {account}")
                _print_account(usage, profile, headroom, region)
        if arguments.json:
            print(json.dumps(reports, indent=2, sort_keys=True))
        return

    api_key = os.environ.get(arguments.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing Daytona credential environment variable: {arguments.api_key_env}")
    usage = await account_usage(DaytonaConnection(api_key=api_key))
    region = _launch_region(usage, arguments, profile)
    if arguments.json:
        print(json.dumps(usage_report(usage, profile, region=region, headroom=headroom), indent=2, sort_keys=True))
    else:
        _print_account(usage, profile, headroom, region)


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
        "--sandbox-class",
        default=DEFAULT_SANDBOX_CLASS,
        help=f"Sandbox class of the launch (default {DEFAULT_SANDBOX_CLASS}).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--cpu", type=int, help="Per-sandbox CPU request for concurrency sizing.")
    parser.add_argument("--memory-gib", type=float, help="Per-sandbox memory request in GiB.")
    parser.add_argument("--disk-gib", type=float, help="Per-sandbox disk request in GiB.")
    parser.add_argument(
        "--headroom",
        type=float,
        default=None,
        help=f"Fraction of remaining capacity kept in reserve (default {DEFAULT_HEADROOM * 100:.0f}%%).",
    )
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    asyncio.run(_run(arguments))


if __name__ == "__main__":
    main()
