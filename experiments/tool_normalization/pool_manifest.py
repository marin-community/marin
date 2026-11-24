# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pool manifest client for the GLM synthesis serving fleet (2026-11-24 handoff contract).

The serving side publishes manifest.json (a list of [{base_url, api_key_ref}]) plus
status.json (healthy_entry_count, per_engine_concurrency, total_concurrency, model,
cluster, validated_at, proof_ref, sustained_ref) under one S3 prefix, only after both
serving gates pass. Consumers treat a per-entry registry 404/connect/5xx as
unavailable and rotate entries; model-name/schema errors are fatal; the fleet-wide
request budget (status.total_concurrency) is split across pods deterministically.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import fsspec

logger = logging.getLogger(__name__)


class PoolManifestError(RuntimeError):
    """The published manifest/status fails the handoff contract."""


@dataclass(frozen=True)
class PoolEntry:
    base_url: str
    api_key_ref: str
    api_key_file: Path


@dataclass(frozen=True)
class PoolPlan:
    entries: tuple[PoolEntry, ...]
    request_slots: int
    rr_offset: int
    model: str
    total_concurrency: int
    per_engine_concurrency: int
    validated_at: int


def _fetch_json(ref: str):
    with fsspec.open(ref, "rb") as stream:
        return json.loads(stream.read().decode())


def fetch_credential(ref: str, dest: Path) -> None:
    """Materialize an endpoint-scoped JWT; fail fast on an empty credential."""
    with fsspec.open(ref, "rb") as stream:
        material = stream.read().strip()
    if not material:
        # Empty Bearer headers caused the silent pool-era HTTP 401 storm
        # (coordinator directive 2026-11-24). Never boot with one.
        raise PoolManifestError(f"Credential ref {ref!r} is EMPTY")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(material.decode())
    dest.chmod(0o600)


def check_status(status: dict, entry_count: int, expected_model: str | None = None) -> None:
    """Enforce the published-only-after-both-gates and consistency contract."""
    for field in (
        "validated_at",
        "total_concurrency",
        "per_engine_concurrency",
        "healthy_entry_count",
        "proof_ref",
        "sustained_ref",
        "model",
    ):
        if field not in status:
            raise PoolManifestError(f"status.json missing {field!r}")
    if not status["validated_at"]:
        raise PoolManifestError("status.json has no validated_at (serving gates not passed)")
    if status["total_concurrency"] < 1:
        raise PoolManifestError("status.json total_concurrency < 1")
    if status["healthy_entry_count"] != entry_count:
        raise PoolManifestError(
            f"manifest/status mismatch: {entry_count} entries vs "
            f"{status['healthy_entry_count']} healthy; pausing rather than mixing generations"
        )
    if status["total_concurrency"] != status["healthy_entry_count"] * status["per_engine_concurrency"]:
        raise PoolManifestError(
            f"status.json inconsistent: total {status['total_concurrency']} != "
            f"{status['healthy_entry_count']} * {status['per_engine_concurrency']}"
        )
    if expected_model and status["model"] != expected_model:
        raise PoolManifestError(f"status model {status['model']!r} != expected {expected_model!r}")


def split_slots(total_concurrency: int, expected_pods: int, pod_index: int) -> int:
    """Deterministic fleet-wide budget split.

    Per-pod share = total // pods, with the first (total % pods) pods taking one extra.
    A zero-share pod pauses requests entirely: a floor of one per pod would violate the
    aggregate cap (e.g. 24 slots across 32 pods).
    """
    if expected_pods < 1:
        raise ValueError(f"expected_pods must be >= 1, got {expected_pods}")
    if not 0 <= pod_index < expected_pods:
        raise ValueError(f"pod_index {pod_index} outside [0, {expected_pods})")
    base, remainder = divmod(total_concurrency, expected_pods)
    return base + (1 if pod_index < remainder else 0)


def refresh_pool(
    manifest_ref: str,
    workdir: Path,
    previous: PoolPlan | None = None,
    *,
    expected_model: str | None = None,
    expected_pods: int = 1,
    pod_index: int = 0,
    pause_seconds: int = 60,
    max_attempts: int = 1,
) -> PoolPlan | None:
    """Manifest/status refresh for one cycle (or boot with retries); None means pause.

    Per-cycle (max_attempts=1): an inconsistent publication (e.g. mid-roll
    healthy_entry_count mismatch), a zero share, or a fetch error pauses the pod without
    claiming work — generations are never mixed and the caller's next cycle re-refreshes.
    Boot (max_attempts > 1): waits pause_seconds between attempts so a mid-roll
    publication does not churn the whole fleet. A rolled generation (new validated_at,
    entry set, or share) is adopted by the caller for subsequent requests.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            plan = load_pool(manifest_ref, workdir, expected_model=expected_model, expected_pods=expected_pods, pod_index=pod_index)
        except Exception as error:  # any refresh failure (including transient S3 errors) pauses
            if attempt >= max_attempts:
                logger.warning("pool manifest refresh failed after %d attempts; pausing (%s)", attempt, error)
                return None
            logger.warning("pool manifest refresh failed (%d/%d): %s; retrying in %ds", attempt, max_attempts, error, pause_seconds)
            time.sleep(pause_seconds)
            continue
        if plan.request_slots < 1:
            logger.warning(
                "pool budget %d across %d pods leaves pod %d zero slots; pausing",
                plan.total_concurrency,
                expected_pods,
                pod_index,
            )
            return None
        if previous is not None and (
            plan.validated_at != previous.validated_at
            or [entry.base_url for entry in plan.entries] != [entry.base_url for entry in previous.entries]
            or [entry.api_key_ref for entry in plan.entries] != [entry.api_key_ref for entry in previous.entries]
            or plan.request_slots != previous.request_slots
        ):
            logger.info(
                "pool generation rolled: %d entries, %d slots/pod (previous: %d entries, %d slots)",
                len(plan.entries),
                plan.request_slots,
                len(previous.entries),
                previous.request_slots,
            )
        return plan
    return None


def load_pool(
    manifest_ref: str,
    workdir: Path,
    *,
    expected_model: str | None = None,
    expected_pods: int = 1,
    pod_index: int = 0,
) -> PoolPlan:
    """Read the manifest + status, gate them, and materialize per-entry credentials."""
    manifest = _fetch_json(manifest_ref)
    if not isinstance(manifest, list) or not manifest:
        raise PoolManifestError(f"{manifest_ref} is not a non-empty entry list")
    for entry in manifest:
        if not entry.get("base_url") or not entry.get("api_key_ref"):
            raise PoolManifestError(f"manifest entry missing base_url/api_key_ref: {entry!r}")
    status_ref = manifest_ref.replace("manifest.json", "status.json")
    status = _fetch_json(status_ref)
    check_status(status, len(manifest), expected_model or os.environ.get("GLM_POOL_MODEL"))
    entries = []
    for i, entry in enumerate(manifest):
        key_file = workdir / f"pool-api-key-{i}.txt"
        fetch_credential(entry["api_key_ref"], key_file)
        entries.append(PoolEntry(entry["base_url"].rstrip("/"), entry["api_key_ref"], key_file))
    slots = split_slots(status["total_concurrency"], expected_pods, pod_index)
    logger.info(
        "pool manifest: %d entries, %d request slots for pod %d, model %s, validated %s",
        len(entries), slots, pod_index, status["model"],
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(status["validated_at"])),
    )
    return PoolPlan(
        entries=tuple(entries),
        request_slots=slots,
        rr_offset=pod_index,
        model=status["model"],
        total_concurrency=status["total_concurrency"],
        per_engine_concurrency=status["per_engine_concurrency"],
        validated_at=int(status["validated_at"]),
    )
