#!/usr/bin/env python3
"""Bounded, read-only preflight for the rebuilt c32 candidate snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

PROBES = Path("/Users/k3sc0re/openathena/build_envs/envgen/probes")
BOUND_PATHS = (
    "/etc/hostname",
    "/etc/hosts",
    "/etc/resolv.conf",
    "/etc/daytona/netleash/ca.crt",
    "/usr/local/bin/daytona",
    "/usr/local/lib/daytona-computer-use",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("DAYTONA_API_KEY"):
        raise SystemExit("DAYTONA_API_KEY is required")

    sys.path.insert(0, str(PROBES))
    import dtx  # type: ignore[import-not-found]

    expected_hashes = json.loads(args.expected.read_text())
    client = dtx.client()
    snapshot = client.snapshot.get(args.snapshot)
    build_info = getattr(snapshot, "build_info", None)
    recipe = args.recipe.read_text()
    provider_recipe = getattr(build_info, "dockerfile_content", None)
    receipt = {
        "schema_version": "c32-candidate-snapshot-preflight-v1",
        "snapshot": {
            "name": snapshot.name,
            "id": snapshot.id,
            "ref": snapshot.ref,
            "state": str(snapshot.state),
            "cpu": snapshot.cpu,
            "memory_gb": snapshot.mem,
            "disk_gb": snapshot.disk,
            "recipe_sha256": digest(args.recipe),
            "provider_recipe_sha256": hashlib.sha256(
                (provider_recipe or "").encode()
            ).hexdigest(),
            "provider_recipe_exact_match": provider_recipe == recipe,
        },
        "network_block_all": True,
        "create_failures": [],
        "sandbox_id": None,
        "cleanup": None,
    }
    if "ACTIVE" not in receipt["snapshot"]["state"] or provider_recipe != recipe:
        raise RuntimeError("snapshot identity/build_info mismatch")

    sandbox = None
    try:
        for attempt in range(1, 5):
            try:
                sandbox, seconds = dtx.create(
                    client,
                    args.snapshot,
                    purpose="c32-publication-preflight",
                    ttl_min=60,
                    block_all=True,
                )
                receipt["sandbox_id"] = sandbox.id
                receipt["create_seconds"] = seconds
                break
            except Exception as error:  # noqa: BLE001
                receipt["create_failures"].append(
                    {"attempt": attempt, "error_type": type(error).__name__}
                )
                if attempt < 4:
                    time.sleep(5 * attempt)
        if sandbox is None:
            receipt["state"] = "infrastructure_error"
            return 1

        sql_command = (
            "test -f /tmp/task-ready && "
            "PGPASSWORD=storm psql -h 127.0.0.1 -U storm -d storm -X -tAc 'SELECT 1'"
        )
        for attempt in range(1, 61):
            sql = dtx.sh(sandbox, sql_command, timeout=30)
            if sql["exit"] == 0 and sql["stdout"].strip() == "1":
                receipt["ready"] = {"attempts": attempt, "sql_result": "1"}
                break
            time.sleep(2)
        else:
            raise RuntimeError("candidate did not become SQL-ready")

        measured = {}
        for path, expected in expected_hashes.items():
            result = dtx.sh(sandbox, "sha256sum -- " + shlex.quote(path), timeout=60)
            words = (result.get("stdout") or "").split(maxsplit=1)
            actual = words[0] if words else None
            measured[path] = {"expected": expected, "actual": actual, "match": actual == expected}
            if result["exit"] != 0 or actual != expected:
                raise RuntimeError(f"ready hash mismatch: {path}")
        receipt["ready_hashes"] = measured

        root_dev = dtx.sh(sandbox, "stat -c %d /", timeout=30)["stdout"].strip()
        pg_dev = dtx.sh(
            sandbox, "stat -c %d /var/lib/postgresql/data", timeout=30
        )["stdout"].strip()
        receipt["mounts"] = {
            "root_device": root_dev,
            "pgdata_device": pg_dev,
            "pgdata_distinct": bool(root_dev and pg_dev and root_dev != pg_dev),
            "provider_paths": {},
        }
        if not receipt["mounts"]["pgdata_distinct"]:
            raise RuntimeError("PGDATA is not a distinct mount")
        for path in BOUND_PATHS:
            query = (
                "if test -e " + shlex.quote(path) + "; then "
                "printf 'present\\n'; findmnt -T " + shlex.quote(path)
                + " -n -o TARGET,FSTYPE,OPTIONS; stat -c %d " + shlex.quote(path)
                + "; else printf 'absent\\n'; fi"
            )
            result = dtx.sh(sandbox, query, timeout=30)
            lines = [line for line in result["stdout"].splitlines() if line]
            if result["exit"] != 0 or not lines:
                raise RuntimeError(f"mount query failed: {path}")
            receipt["mounts"]["provider_paths"][path] = {
                "present": lines[0] == "present",
                "findmnt": lines[1] if len(lines) > 1 else None,
                "device": lines[2] if len(lines) > 2 else None,
            }
        receipt["state"] = "passed"
        return 0
    finally:
        if sandbox is not None:
            try:
                sandbox.delete()
                observations = []
                absent = False
                for attempt in range(5):
                    if attempt:
                        time.sleep(3 * attempt)
                    try:
                        client.get(sandbox.id)
                        observations.append("present")
                    except Exception as error:  # noqa: BLE001
                        if getattr(error, "status_code", None) == 404 or "not found" in str(error).lower():
                            observations.append("not_found")
                            absent = True
                            break
                        observations.append(type(error).__name__)
                receipt["cleanup"] = {
                    "delete_requested": True,
                    "verified_absent": absent,
                    "observations": observations,
                }
                if not absent:
                    receipt["state"] = "cleanup_unverified"
            except Exception as error:  # noqa: BLE001
                receipt["cleanup"] = {
                    "delete_requested": False,
                    "verified_absent": False,
                    "error_type": type(error).__name__,
                }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
