#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sync the canonical Finelog telemetry catalog into Python package data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CATALOG = REPOSITORY_ROOT / "lib/finelog/rust/telemetry_catalog.v1.json"
PACKAGED_MIRRORS = (REPOSITORY_ROOT / "lib/rigging/src/rigging/telemetry_catalog.v1.json",)


def sync_catalog(*, check: bool) -> bool:
    canonical = CANONICAL_CATALOG.read_bytes()
    stale = [mirror for mirror in PACKAGED_MIRRORS if not mirror.exists() or mirror.read_bytes() != canonical]
    if not stale:
        return True
    if check:
        for mirror in stale:
            print(f"stale telemetry catalog mirror: {mirror.relative_to(REPOSITORY_ROOT)}", file=sys.stderr)
        print("run: uv run python scripts/sync_telemetry_catalog.py", file=sys.stderr)
        return False
    for mirror in stale:
        mirror.write_bytes(canonical)
        print(f"updated {mirror.relative_to(REPOSITORY_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of updating stale package mirrors")
    args = parser.parse_args()
    return 0 if sync_catalog(check=args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
