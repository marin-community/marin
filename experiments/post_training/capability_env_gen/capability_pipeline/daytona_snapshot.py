"""Bounded lifecycle policy for Daytona task-environment snapshots."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from typing import Any


class SnapshotRecipeEvidenceError(RuntimeError):
    """A cached snapshot cannot be bound to the requested build recipe."""


def _provider_field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def validate_snapshot_recipe(
    snapshot: Any, *, expected_name: str, expected_dockerfile: str
) -> dict[str, str]:
    """Require the provider's exact Dockerfile before reusing a named snapshot."""
    actual_name = str(_provider_field(snapshot, "name") or "")
    if actual_name != expected_name:
        raise SnapshotRecipeEvidenceError(
            "cached snapshot has the wrong provider name: "
            f"expected {expected_name!r}, got {actual_name!r}"
        )
    build_info = _provider_field(snapshot, "build_info")
    actual_dockerfile = _provider_field(build_info, "dockerfile_content")
    if not isinstance(actual_dockerfile, str) or not actual_dockerfile:
        raise SnapshotRecipeEvidenceError(
            "insufficient snapshot recipe evidence: provider "
            "build_info.dockerfile_content is absent"
        )
    expected_sha256 = hashlib.sha256(expected_dockerfile.encode()).hexdigest()
    actual_sha256 = hashlib.sha256(actual_dockerfile.encode()).hexdigest()
    if actual_dockerfile != expected_dockerfile:
        raise SnapshotRecipeEvidenceError(
            "cached snapshot Dockerfile does not match the requested recipe: "
            f"expected sha256:{expected_sha256}, got sha256:{actual_sha256}"
        )
    return {
        "snapshot_name": actual_name,
        "dockerfile_sha256": actual_sha256,
        "evidence": "provider-build-info-exact-match",
    }


def resource_not_found(error: Exception, resource: str) -> bool:
    """Recognize a typed or HTTP not-found only when its resource is explicit."""
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    message = str(error).lower()
    return resource.lower() in message and (
        "not found" in message or name == "DaytonaNotFoundError" or status == 404
    )


def snapshot_not_found(error: Exception) -> bool:
    """Recognize the SDK's snapshot-specific 404 without retrying other failures."""
    return resource_not_found(error, "snapshot")


def snapshot_conflict(error: Exception) -> bool:
    """Recognize only a concurrent same-name snapshot creator."""
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    message = str(error).lower()
    return (
        name == "DaytonaConflictError"
        or status == 409
        or ("snapshot" in message and "already exists" in message)
    )


def wait_for_sandbox_deletion(
    client: Any,
    sandbox_id: str,
    *,
    delays: tuple[float, ...] = (0, 5, 10, 15, 30),
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[str, list[dict[str, Any]]]:
    """Observe eventual deletion for at most 60 seconds without creating work."""
    if not delays or delays[0] != 0 or sum(delays) > 60:
        raise ValueError("deletion lookup delays must start at zero and total <= 60s")
    observations: list[dict[str, Any]] = []
    elapsed = 0.0
    for delay in delays:
        if delay:
            sleeper(delay)
            elapsed += delay
        observation: dict[str, Any] = {"elapsed_seconds": elapsed}
        try:
            client.get(sandbox_id)
            observation["state"] = "present"
        except Exception as error:  # noqa: BLE001 - record provider class, not text
            if resource_not_found(error, "sandbox"):
                observation["state"] = "not_found"
                observations.append(observation)
                return "not_found", observations
            observation.update(
                {"state": "lookup_error", "error_type": type(error).__name__}
            )
        observations.append(observation)
    return str(observations[-1]["state"]), observations


def snapshot_identity(
    snapshot: Any, requested_image: str, source: str
) -> dict[str, Any]:
    state = getattr(snapshot, "state", None)
    if hasattr(state, "value"):
        state = state.value
    return {
        "source": source,
        "id": str(getattr(snapshot, "id", "")),
        "name": str(getattr(snapshot, "name", "")),
        "state": str(state),
        "image_name": str(getattr(snapshot, "image_name", "")),
        "requested_image": requested_image,
        "requested_image_sha256": hashlib.sha256(requested_image.encode()).hexdigest(),
    }


def wait_for_snapshot_active(
    get_snapshot: Callable[[str], Any],
    name: str,
    expected_dockerfile: str,
    *,
    initial_snapshot: Any | None = None,
    delays: tuple[float, ...] = (0, 2, 5, 10, 15, 30, 60, 60, 60, 58),
    sleeper: Callable[[float], None] = time.sleep,
) -> Any:
    """Wait for a concurrently built named snapshot, never rebuilding or deleting it."""
    if (
        not delays
        or delays[0] != 0
        or any(delay < 0 for delay in delays)
        or sum(delays) > 300
    ):
        raise ValueError(
            "snapshot readiness waits must start at zero and total <= 300s"
        )
    for index, delay in enumerate(delays):
        if delay:
            sleeper(delay)
        try:
            snapshot = (
                initial_snapshot
                if index == 0 and initial_snapshot is not None
                else get_snapshot(name)
            )
        except Exception as error:
            if snapshot_not_found(error):
                continue
            raise
        if str(_provider_field(snapshot, "name") or "") != name:
            raise SnapshotRecipeEvidenceError(
                "resolved snapshot has the wrong provider name"
            )
        state = _provider_field(snapshot, "state")
        if hasattr(state, "value"):
            state = state.value
        state = str(state or "").lower().removeprefix("snapshotstate.")
        if state == "active":
            if not str(_provider_field(snapshot, "id") or ""):
                raise SnapshotRecipeEvidenceError("active snapshot lacks provider ID")
            validate_snapshot_recipe(
                snapshot, expected_name=name, expected_dockerfile=expected_dockerfile
            )
            return snapshot
        if state not in {"pending", "building", "creating", "queued"}:
            raise RuntimeError(
                f"resolved Daytona snapshot has terminal or unknown state: {state}"
            )
    raise RuntimeError(
        "resolved Daytona snapshot did not become active within 300 seconds"
    )


def provision_with_snapshot_recovery(
    *,
    base_name: str,
    requested_image: str,
    initial_snapshot: Any,
    create_sandbox: Callable[[str], tuple[Any, list[dict[str, Any]]]],
    reconstruct: Callable[[str], Any],
    recovery_suffix: str,
    record: Callable[[list[dict[str, Any]]], None],
) -> tuple[Any, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Create once, then rebuild once only for a snapshot-specific not-found."""
    active = snapshot_identity(initial_snapshot, requested_image, "resolved")
    if not active["id"] or active["name"] != base_name:
        raise ValueError("resolved Daytona snapshot has the wrong provider identity")
    if active["state"].lower() not in {"active", "snapshotstate.active"}:
        raise RuntimeError("resolved Daytona snapshot is not active")
    events = [{"event": "snapshot_resolved", **active}]
    record(events)
    try:
        sandbox, attempts = create_sandbox(active["name"])
    except Exception as error:
        if not snapshot_not_found(error):
            raise
        events.append(
            {
                "event": "sandbox_create_snapshot_not_found",
                "snapshot_id": active["id"],
                "snapshot_name": active["name"],
                "error_type": type(error).__name__,
            }
        )
        record(events)
        replacement_name = f"{base_name}-recovery-{recovery_suffix}"
        replacement = reconstruct(replacement_name)
        active = snapshot_identity(replacement, requested_image, "reconstructed")
        if not active["id"] or active["name"] != replacement_name:
            raise ValueError("reconstructed Daytona snapshot has the wrong identity")
        if active["state"].lower() not in {"active", "snapshotstate.active"}:
            raise RuntimeError("reconstructed Daytona snapshot is not active")
        events.append({"event": "snapshot_reconstructed", **active})
        record(events)
        try:
            sandbox, attempts = create_sandbox(active["name"])
        except Exception as retry_error:
            if snapshot_not_found(retry_error):
                events.append(
                    {
                        "event": "snapshot_reconstruction_not_found",
                        "snapshot_id": active["id"],
                        "snapshot_name": active["name"],
                        "error_type": type(retry_error).__name__,
                    }
                )
                record(events)
            raise
    events.append(
        {
            "event": "sandbox_created",
            "snapshot_id": active["id"],
            "snapshot_name": active["name"],
            "sandbox_id": str(getattr(sandbox, "id", "")),
            "provisioning_attempts": attempts,
        }
    )
    record(events)
    return sandbox, attempts, active, events
