#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit GitHub Actions credentials against stack config and workflow references."""

import argparse
import json
import sys
from pathlib import Path

from iac.github.audit import audit_credentials, discover_secret_references
from iac.github.credentials import load_stack_manifest
from iac.github.inventory import github_secret_inventory

STACK_DIR = Path(__file__).resolve().parent
REPO_ROOT = STACK_DIR.parents[2]
DEFAULT_STACK_CONFIG = STACK_DIR / "Pulumi.marin-community.yaml"


def _print_section(title: str, values: tuple[str, ...]) -> None:
    print(f"{title} ({len(values)}):")
    for value in values:
        print(f"  {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-config", type=Path, default=DEFAULT_STACK_CONFIG)
    parser.add_argument("--live", action="store_true", help="include live GitHub metadata")
    parser.add_argument("--json", action="store_true", help="print structured output")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = load_stack_manifest(args.stack_config)
    references = discover_secret_references(REPO_ROOT)
    live_secrets = github_secret_inventory(manifest) if args.live else None
    report = audit_credentials(manifest, references, live_secrets)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        _print_section("Removal candidates", report.removal_candidates)
        _print_section("Referenced but missing", report.referenced_missing)
        _print_section("Shadowed organization secrets", report.shadowed)
        _print_section("Credentials recoverable without owner recovery", report.recoverable_credentials)
        _print_section("Credentials requiring owner recovery", report.manual_recovery_credentials)
        if report.errors:
            print(f"Errors ({len(report.errors)}):")
            for finding in report.errors:
                label = f" [{finding.credential}]" if finding.credential else ""
                print(f"  {finding.code}{label}: {finding.detail}")
        else:
            mode = "live and offline" if args.live else "offline"
            print(f"No {mode} catalog drift detected.")
    if report.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
