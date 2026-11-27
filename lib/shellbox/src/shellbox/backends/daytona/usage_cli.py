# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report per-region headroom from paginated Daytona inventory, without usage API access."""

import argparse
import asyncio
import json
import os
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import TypedDict

from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.usage import (
    DEFAULT_HEADROOM,
    AccountLimits,
    RegionLimits,
    SandboxProfile,
    UsageReport,
    account_usage,
    usage_report,
    validate_headroom,
)


class Account(StrEnum):
    EVAL = "eval"
    RL = "rl"
    DATA = "data"


# Single quota source of record. Owner-supplied organization limits (2026-11-19)
# recorded for the accounts' us region, the one Daytona region Marin launches in
# (DAYTONA_TARGET). Update these presets when the account owner changes the
# limits, and add regions when the owner supplies their rows.
QUOTA_SOURCE = (
    "account-owner limits (2026-11-19), ACCOUNT_LIMITS in lib/shellbox/src/shellbox/backends/daytona/usage_cli.py"
)
PER_SANDBOX_MAXIMUM = SandboxProfile(cpu=4, memory_gib=8, disk_gib=10)
ACCOUNT_LIMITS = MappingProxyType(
    {
        Account.EVAL: AccountLimits(
            regions=MappingProxyType({"us": RegionLimits(2500, 7500, 25000)}),
            per_sandbox=PER_SANDBOX_MAXIMUM,
        ),
        Account.RL: AccountLimits(
            regions=MappingProxyType({"us": RegionLimits(6000, 12000, 20000)}),
            per_sandbox=PER_SANDBOX_MAXIMUM,
        ),
        Account.DATA: AccountLimits(
            regions=MappingProxyType({"us": RegionLimits(2000, 4000, 10000)}),
            per_sandbox=PER_SANDBOX_MAXIMUM,
        ),
    }
)
ACCOUNT_KEY_ENVS = MappingProxyType(
    {
        Account.EVAL: "DAYTONA_EVAL_API_KEY",
        Account.RL: "DAYTONA_RL_API_KEY",
        Account.DATA: "DAYTONA_DATA_API_KEY",
    }
)


class SkippedAccount(TypedDict):
    skipped: str


class AccountReport(UsageReport):
    account: str
    quota_source: str


def _profile(arguments: argparse.Namespace) -> SandboxProfile | None:
    given = [arguments.cpu is not None, arguments.memory_gib is not None, arguments.disk_gib is not None]
    if any(given) and not all(given):
        raise SystemExit("Provide all of --cpu, --memory-gib and --disk-gib to size concurrency")
    if not any(given):
        return None
    return SandboxProfile(arguments.cpu, arguments.memory_gib, arguments.disk_gib)


def _selected_accounts(arguments: argparse.Namespace) -> Mapping[Account, str]:
    if arguments.all:
        if arguments.limits is not None:
            raise SystemExit("--limits applies to a single --api-key-env selection")
        return ACCOUNT_KEY_ENVS
    account = arguments.limits
    if account is None:
        account = next((name for name, key in ACCOUNT_KEY_ENVS.items() if key == arguments.api_key_env), None)
    if account is None:
        raise SystemExit("A custom --api-key-env requires --limits eval, rl or data")
    return {Account(account): arguments.api_key_env}


def _print_report(report: AccountReport) -> None:
    print(f"account {report['account']}  organization {report['organization_id'] or 'empty inventory'}")
    print(f"  sandboxes {report['sandboxes']['total']}  started {report['sandboxes']['started']}")
    maximum = report["per_sandbox"]
    print(
        f"  per-sandbox maximum {maximum['cpu']} vCPU / {maximum['memory_gib']} GiB RAM / {maximum['disk_gib']} GiB disk"
    )
    for name, region in report["regions"].items():
        counts = region["sandboxes"]
        print(
            f"  region {name}  sandboxes {counts['total']}"
            f"  compute charged {counts['compute_charged']}  disk charged {counts['disk_charged']}"
        )
        print("    resource       quota         used          remaining     available")
        for resource, capacity in region["resources"].items():
            print(
                f"    {resource:12} "
                + " ".join(f"{capacity[field]:13.2f}" for field in ("quota", "used", "remaining", "available"))
            )
        if "max_concurrency" in region:
            sizing = region["max_concurrency"]
            print(f"    additional sandboxes {sizing['limit']}")
            if sizing["exceeds_per_sandbox_maximum"]:
                print("    profile exceeds per-sandbox maximum: " + ", ".join(sizing["exceeds_per_sandbox_maximum"]))
    print(f"  reserve {report['headroom_fraction']:.0%} of each region's quota; memory and disk in GiB")


async def _run(arguments: argparse.Namespace) -> None:
    validate_headroom(arguments.headroom)
    profile = _profile(arguments)
    reports: dict[str, AccountReport | SkippedAccount] = {}
    for account, key_env in _selected_accounts(arguments).items():
        api_key = os.environ.get(key_env)
        if not api_key:
            if not arguments.all:
                raise SystemExit(f"Missing Daytona credential environment variable: {key_env}")
            reports[account] = {"skipped": f"missing {key_env}"}
            if not arguments.json:
                print(f"{account}: skipped (missing {key_env})")
            continue
        connection = DaytonaConnection(api_key=api_key)
        usage = await account_usage(connection, ACCOUNT_LIMITS[account])
        report: AccountReport = {
            **usage_report(usage, profile, headroom=arguments.headroom),
            "account": account,
            "quota_source": QUOTA_SOURCE,
        }
        reports[account] = report
        if not arguments.json:
            _print_report(report)
    if arguments.json:
        print(json.dumps(reports if arguments.all else next(iter(reports.values())), indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shellbox-daytona-usage",
        description="Read sandbox inventory and size per-region headroom from owner-provided quota limits.",
    )
    account = parser.add_mutually_exclusive_group(required=True)
    account.add_argument("--api-key-env", help="Environment variable holding the selected account's API key.")
    account.add_argument(
        "--all", action="store_true", help="Read eval, rl and data using their dedicated key variables."
    )
    parser.add_argument(
        "--limits", type=Account, choices=list(Account), help="Quota preset; inferred for standard key variables."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument("--cpu", type=int, help="Per-sandbox vCPU allocation.")
    parser.add_argument("--memory-gib", type=int, help="Per-sandbox RAM allocation in whole GiB.")
    parser.add_argument("--disk-gib", type=int, help="Per-sandbox disk allocation in whole GiB.")
    parser.add_argument(
        "--headroom",
        type=float,
        default=DEFAULT_HEADROOM,
        help=f"Fraction of total quota to keep free (default {DEFAULT_HEADROOM}).",
    )
    return parser


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()
