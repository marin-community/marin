# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report organization headroom from paginated Daytona inventory, without usage API access."""

import argparse
import asyncio
import json
import os
from enum import StrEnum
from typing import Any

from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.usage import (
    DEFAULT_HEADROOM,
    OrganizationLimits,
    SandboxProfile,
    account_usage,
    usage_report,
)


class Account(StrEnum):
    EVAL = "eval"
    RL = "rl"
    DATA = "data"


# Owner-provided limits from experiments/active/DAYTONA-QUOTAS.md (2026-11-19).
PER_SANDBOX_MAXIMUM = SandboxProfile(cpu=4, memory_gib=8, disk_gib=10)
ACCOUNT_LIMITS = {
    Account.EVAL: OrganizationLimits(2500, 7500, 25000, PER_SANDBOX_MAXIMUM),
    Account.RL: OrganizationLimits(6000, 12000, 20000, PER_SANDBOX_MAXIMUM),
    Account.DATA: OrganizationLimits(2000, 4000, 10000, PER_SANDBOX_MAXIMUM),
}
ACCOUNT_KEY_ENVS = {
    Account.EVAL: "DAYTONA_EVAL_API_KEY",
    Account.RL: "DAYTONA_RL_API_KEY",
    Account.DATA: "DAYTONA_DATA_API_KEY",
}


def _profile(arguments: argparse.Namespace) -> SandboxProfile | None:
    given = [arguments.cpu is not None, arguments.memory_gib is not None, arguments.disk_gib is not None]
    if any(given) and not all(given):
        raise SystemExit("Provide all of --cpu, --memory-gib and --disk-gib to size concurrency")
    if not any(given):
        return None
    return SandboxProfile(arguments.cpu, arguments.memory_gib, arguments.disk_gib)


def _selected_accounts(arguments: argparse.Namespace) -> dict[Account, str]:
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


def _print_report(report: dict[str, Any]) -> None:
    print(f"account {report['account']}  organization {report['organization_id'] or 'empty inventory'}")
    print(
        f"  sandboxes {report['sandboxes']['total']}  started {report['sandboxes']['started']}"
        f"  charged {report['sandboxes']['charged']}"
    )
    print("  resource       quota         used          remaining     available")
    for name, resource in report["resources"].items():
        print(
            f"  {name:14} "
            + " ".join(f"{resource[field]:13.2f}" for field in ("quota", "used", "remaining", "available"))
        )
    print(f"  reserve {report['headroom_fraction']:.0%} of total quota; memory and disk in GiB")
    maximum = report["org_limits"]["per_sandbox"]
    print(
        f"  per-sandbox maximum {maximum['cpu']} vCPU / {maximum['memory_gib']} GiB RAM / {maximum['disk_gib']} GiB disk"
    )
    if "max_concurrency" in report:
        sizing = report["max_concurrency"]
        print(f"  additional sandboxes {sizing['limit']}")
        if sizing["exceeds_per_sandbox_maximum"]:
            print("  profile exceeds per-sandbox maximum: " + ", ".join(sizing["exceeds_per_sandbox_maximum"]))


async def _run(arguments: argparse.Namespace) -> None:
    if not 0 <= arguments.headroom < 1:
        raise SystemExit("Headroom must be in [0, 1)")
    profile = _profile(arguments)
    reports: dict[str, Any] = {}
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
        report = {"account": account, **usage_report(usage, profile, headroom=arguments.headroom)}
        reports[account] = report
        if not arguments.json:
            _print_report(report)
    if arguments.json:
        print(json.dumps(reports if arguments.all else next(iter(reports.values())), indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shellbox-daytona-usage",
        description="Read sandbox inventory and size organization headroom from owner-provided quota limits.",
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
        "--headroom", type=float, default=DEFAULT_HEADROOM, help="Fraction of total quota to keep free (default 0.2)."
    )
    return parser


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()
