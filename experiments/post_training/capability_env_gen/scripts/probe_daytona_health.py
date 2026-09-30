#!/usr/bin/env python3
"""One bounded, network-blocked Daytona create/delete health probe."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from daytona import CreateSandboxFromSnapshotParams

from capability_pipeline.daytona_snapshot import wait_for_sandbox_deletion
from capability_pipeline.provider_retry import provision_with_rate_limit_retry


def _dt():
    """Load the maintained SDK helper without importing the Harbor runtime."""
    path = Path(os.environ["CAPABILITY_DAYTONA_TOOLS"]) / "dt.py"
    spec = importlib.util.spec_from_file_location("health_daytona_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("maintained Daytona helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def complete(receipt: dict, state: str) -> None:
    receipt["state"] = state
    receipt["completed_at"] = datetime.now(UTC).isoformat()


def run(snapshot_name: str, output: Path) -> int:
    receipt = {
        "schema_version": "capability-daytona-health-v1",
        "state": "pending",
        "daytona_sdk_version": version("daytona"),
        "snapshot": snapshot_name,
        "snapshot_id": None,
        "snapshot_state": "not_checked",
        "network_block_all": None,
        "network_block_all_requested": True,
        "network_block_all_observed": None,
        "sandbox_id": None,
        "provisioning_attempts": [],
        "deleted": False,
        "lookup_after_delete": "not_checked",
        "deletion_lookups": [],
    }
    if not os.environ.get("DAYTONA_API_KEY"):
        receipt["error_type"] = "MissingDaytonaApiKey"
        complete(receipt, "failed")
        atomic_json(output, receipt)
        return 1
    try:
        client = _dt().client()
        snapshot = client.snapshot.get(snapshot_name)
    except Exception as error:  # noqa: BLE001 - retain only provider error class
        receipt["error_type"] = type(error).__name__
        complete(receipt, "failed")
        atomic_json(output, receipt)
        return 1
    state = getattr(snapshot.state, "value", snapshot.state)
    receipt["snapshot"] = snapshot.name
    receipt["snapshot_id"] = snapshot.id
    receipt["snapshot_state"] = str(state)
    if str(state).lower() not in {"active", "snapshotstate.active"}:
        receipt["error_type"] = "SnapshotNotActive"
        complete(receipt, "failed")
        atomic_json(output, receipt)
        return 1
    parameters = CreateSandboxFromSnapshotParams(
        snapshot=snapshot.name,
        labels={"envgen": "1", "envgen_purpose": "provider-health"},
        ephemeral=True,
        auto_stop_interval=0,
        ttl_minutes=15,
        network_block_all=True,
    )
    sandbox = None
    try:
        sandbox, attempts = provision_with_rate_limit_retry(
            lambda: client.create(parameters, timeout=600)
        )
        receipt["sandbox_id"] = sandbox.id
        receipt["provisioning_attempts"] = attempts
        observed = client.get(sandbox.id)
        receipt["network_block_all_observed"] = getattr(
            observed, "network_block_all", None
        )
        receipt["network_block_all"] = receipt["network_block_all_observed"]
    except Exception as error:  # noqa: BLE001 - retain only provider error class
        if isinstance(getattr(error, "attempts", None), list):
            receipt["provisioning_attempts"] = error.attempts
        receipt["error_type"] = type(error).__name__
    finally:
        if sandbox is not None:
            try:
                sandbox.delete()
                receipt["deleted"] = True
                lookup_state, observations = wait_for_sandbox_deletion(
                    client, sandbox.id
                )
                receipt["lookup_after_delete"] = lookup_state
                receipt["deletion_lookups"] = observations
            except Exception as delete_error:  # noqa: BLE001
                receipt["delete_error_type"] = type(delete_error).__name__
    passed = (
        "error_type" not in receipt
        and receipt["network_block_all_requested"] is True
        and receipt["network_block_all_observed"] is True
        and receipt["deleted"]
        and receipt["lookup_after_delete"] == "not_found"
    )
    complete(receipt, "passed" if passed else "failed")
    atomic_json(output, receipt)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return run(args.snapshot, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
