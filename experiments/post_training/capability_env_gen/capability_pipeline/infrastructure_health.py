"""In-job sandbox provider health probe for automatic infrastructure retries.

``synthesize --retry-infrastructure`` used to require an operator-supplied
``capability-daytona-health-v1`` receipt.  The conveyor cannot wait for an
operator, so the job produces the same receipt itself: one fresh,
network-blocked sandbox created from a snapshot the item's grader already uses,
observed, deleted, and confirmed gone.  The receipt passes the unchanged
``synthesis._validate_infrastructure_health_receipt`` check, so the automatic
path makes exactly the claim the operator path made.

Credential-free: the client comes from the staged ``dt.py`` (Daytona or the
silo drop-in) and the receipt records only ids, states and error class names.
Everything here is bounded; nothing retries a grade.
"""

from __future__ import annotations

import json
import os
import re
import time
import types
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import sandbox_provider
from .daytona_snapshot import resource_not_found, wait_for_sandbox_deletion
from .provider_retry import provision_with_rate_limit_retry

SCHEMA = "capability-daytona-health-v1"
_SNAPSHOT_NAME = re.compile(r"cap-verifier-[0-9a-f]{20}")
_DIGEST_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


def load_dt(tools: Path) -> types.ModuleType:
    """Load the staged provider helper exactly as the runtime does (no pycache)."""
    path = Path(tools) / "dt.py"
    module = types.ModuleType("capability_health_dt")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def _grading_records(value: Any):
    if isinstance(value, dict):
        yield value
        for machine in value.get("machine_results") or []:
            yield from _grading_records(machine)
        detail = value.get("detail")
        if isinstance(detail, dict):
            yield detail
            for machine in detail.get("machine_results") or []:
                yield from _grading_records(machine)


def recorded_verifier_snapshots(item_root: Path) -> list[str]:
    """Verifier snapshots this item's own graded trials were created from."""
    found: list[str] = []
    for parent in (item_root / "runtime-trials", item_root / "judge-calibration"):
        for path in sorted(parent.rglob("verifier/taskcompendium-result.json")):
            try:
                value = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for record in _grading_records(value):
                name = record.get("verifier_snapshot")
                if isinstance(name, str) and _SNAPSHOT_NAME.fullmatch(name) and name not in found:
                    found.append(name)
    return found


def specification_verifier_snapshots(specification: Path) -> list[str]:
    """Default-profile verifier snapshot names for the container runtimes in a spec."""
    from .daytona_policy import verifier_snapshot_recipe
    from .daytona_resources import VERIFIER_DEFAULT, snapshot_name

    try:
        spec = json.loads(Path(specification).read_text())
    except (OSError, ValueError):
        return []
    names: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            image = value.get("image")
            if value.get("kind") == "container" and isinstance(image, str) and _DIGEST_IMAGE.fullmatch(image):
                python = value.get("supervisor_python") or "python3"
                try:
                    name = snapshot_name(
                        "cap-verifier", verifier_snapshot_recipe(image, python), VERIFIER_DEFAULT
                    )
                except (TypeError, ValueError):
                    name = None
                if name and name not in names:
                    names.append(name)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(spec)
    return names


def probe_snapshots(item_root: Path) -> list[str]:
    names = recorded_verifier_snapshots(item_root)
    for spec in (item_root / "harbor" / "specification.json", item_root / "workspace" / "task" / "specification.json"):
        for name in specification_verifier_snapshots(spec):
            if name not in names:
                names.append(name)
    return names


def _complete(receipt: dict[str, Any], state: str) -> dict[str, Any]:
    receipt["state"] = state
    receipt["completed_at"] = datetime.now(UTC).isoformat()
    return receipt


def _create_parameters(snapshot: str) -> Any:
    fields = {
        "snapshot": snapshot,
        "labels": {"envgen": "1", "envgen_purpose": "provider-health"},
        "ephemeral": True,
        "auto_stop_interval": 0,
        "ttl_minutes": 15,
        "network_block_all": True,
    }
    try:
        from daytona import CreateSandboxFromSnapshotParams
    except ImportError:
        # The silo client reads parameters by attribute; the Daytona SDK is
        # only needed for Daytona itself.
        return SimpleNamespace(**fields)
    return CreateSandboxFromSnapshotParams(**fields)


def run_probe(
    snapshots: list[str],
    *,
    client_factory: Callable[[], Any],
    sleeper: Callable[[float], None] = time.sleep,
    create_timeout: float = 600,
) -> dict[str, Any]:
    """Create/observe/delete one sandbox from the first active snapshot.

    Returns a receipt.  ``state`` is ``passed``, ``failed`` (the provider could
    not do it: an infrastructure signal), or ``unavailable`` (no active snapshot
    to probe with, which says nothing about health).
    """
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "state": "pending",
        "provider": sandbox_provider.provider(),
        "probe": "in_job_automatic",
        "started_at": datetime.now(UTC).isoformat(),
        "snapshot": None,
        "snapshot_id": None,
        "snapshot_state": "not_checked",
        "snapshots_considered": list(snapshots),
        "network_block_all": None,
        "network_block_all_requested": True,
        "network_block_all_observed": None,
        "sandbox_id": None,
        "provisioning_attempts": [],
        "deleted": False,
        "lookup_after_delete": "not_checked",
        "deletion_lookups": [],
    }
    if not sandbox_provider.credentials_present():
        receipt["error_type"] = "MissingProviderCredentials"
        return _complete(receipt, "unavailable")
    if not snapshots:
        receipt["error_type"] = "NoProbeSnapshot"
        return _complete(receipt, "unavailable")
    started = time.monotonic()
    try:
        client = client_factory()
    except Exception as error:  # noqa: BLE001 - provider error class only
        receipt["error_type"] = type(error).__name__
        return _complete(receipt, "failed")
    snapshot = None
    lookups: list[dict[str, str]] = []
    for name in snapshots:
        try:
            record = client.snapshot.get(name)
        except Exception as error:  # noqa: BLE001
            if resource_not_found(error, "snapshot"):
                lookups.append({"snapshot": name, "state": "not_found"})
                continue
            # The control plane itself did not answer: that is the signal.
            receipt["snapshot_lookups"] = lookups + [{"snapshot": name, "state": "lookup_error", "error_type": type(error).__name__}]
            receipt["error_type"] = type(error).__name__
            return _complete(receipt, "failed")
        state = str(getattr(record.state, "value", record.state)).lower().removeprefix("snapshotstate.")
        lookups.append({"snapshot": name, "state": state})
        if state == "active":
            snapshot = record
            break
    receipt["snapshot_lookups"] = lookups
    if snapshot is None:
        receipt["error_type"] = "NoActiveProbeSnapshot"
        return _complete(receipt, "unavailable")
    receipt["snapshot"] = snapshot.name
    receipt["snapshot_id"] = getattr(snapshot, "id", None)
    receipt["snapshot_state"] = "active"
    sandbox = None
    try:
        sandbox, attempts = provision_with_rate_limit_retry(
            lambda: client.create(_create_parameters(snapshot.name), timeout=create_timeout),
            sleep=sleeper,
        )
        receipt["sandbox_id"] = sandbox.id
        receipt["provisioning_attempts"] = attempts
        receipt["create_seconds"] = round(time.monotonic() - started, 1)
        observed = client.get(sandbox.id)
        receipt["network_block_all_observed"] = getattr(observed, "network_block_all", None)
        receipt["network_block_all"] = receipt["network_block_all_observed"]
    except Exception as error:  # noqa: BLE001 - provider error class only
        if isinstance(getattr(error, "attempts", None), list):
            receipt["provisioning_attempts"] = error.attempts
        receipt["error_type"] = type(error).__name__
    finally:
        if sandbox is not None:
            try:
                sandbox.delete()
                receipt["deleted"] = True
                state, observations = wait_for_sandbox_deletion(client, sandbox.id, sleeper=sleeper)
                receipt["lookup_after_delete"] = state
                receipt["deletion_lookups"] = observations
            except Exception as error:  # noqa: BLE001
                receipt["delete_error_type"] = type(error).__name__
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 1)
    passed = (
        "error_type" not in receipt
        and receipt["network_block_all_observed"] is True
        and receipt["deleted"] is True
        and receipt["lookup_after_delete"] == "not_found"
    )
    return _complete(receipt, "passed" if passed else "failed")


def probe_item_provider(
    item_root: Path,
    output: Path,
    *,
    daytona_tools: Path | None,
    client_factory: Callable[[], Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Probe the provider with this item's verifier snapshot; write the receipt."""
    if client_factory is None:
        tools = daytona_tools or (Path(os.environ["CAPABILITY_DAYTONA_TOOLS"]) if os.environ.get("CAPABILITY_DAYTONA_TOOLS") else None)
        if tools is None or not (Path(tools) / "dt.py").is_file():
            receipt = _complete(
                {"schema_version": SCHEMA, "probe": "in_job_automatic", "error_type": "NoProviderHelper"},
                "unavailable",
            )
            _write(output, receipt)
            return receipt

        def client_factory() -> Any:
            return load_dt(Path(tools)).client()

    receipt = run_probe(probe_snapshots(item_root), client_factory=client_factory, sleeper=sleeper)
    _write(output, receipt)
    return receipt


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
