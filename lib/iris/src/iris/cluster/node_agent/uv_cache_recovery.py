# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a Kubernetes node after repeated uv cache failures."""

import logging
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from rigging import telemetry
from rigging.telemetry.serialization import EventBody

from iris.cluster.platforms.k8s.service import K8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.runtime.env import UV_CACHE_PATH, UV_CACHE_RECOVERY_SIGNAL_PREFIX, cache_host_dirname

logger = logging.getLogger(__name__)

UV_CACHE_RECOVERY_THRESHOLD = 3
UV_CACHE_RECOVERY_WINDOW = 30 * 60.0
UV_CACHE_RECOVERY_INTERVAL = 30.0
UV_CACHE_RESET_MARKER = ".iris-uv-cache-reset-boot-id"
COREWEAVE_PENDING_STATE_CONDITION = "PendingPhaseState"
COREWEAVE_PENDING_STATE_LABEL = "node.coreweave.cloud/pending-state"
COREWEAVE_PENDING_STATE_LABEL_PATH = "/metadata/labels/" + COREWEAVE_PENDING_STATE_LABEL.replace("~", "~0").replace(
    "/", "~1"
)
COREWEAVE_POWER_RESET_STATE = "production-powerreset"
UV_CACHE_RECOVERY_OBSERVED_EVENT = "uv_cache_recovery_observed"
UV_CACHE_REBOOT_REQUESTED_EVENT = "uv_cache_reboot_requested"
UV_CACHE_RESET_COMPLETED_EVENT = "uv_cache_reset_completed"
_OBSERVED_RECOVERY_SIGNAL_PREFIX = ".iris-recovery-observed-"
_RECOVERY_FAILURES = telemetry.counter("iris_uv_cache_recovery_failures", unit="{failure}")


def _uv_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / cache_host_dirname(UV_CACHE_PATH)


def _emit_event(name: str, **fields: str | int | float | bool) -> None:
    telemetry.event(name, EventBody(fields))


def complete_uv_cache_reset(cache_dir: Path, boot_id: str) -> None:
    """Clear a marked uv cache only when the machine boot ID has changed."""
    marker = cache_dir / UV_CACHE_RESET_MARKER
    if not marker.exists():
        return
    requested_boot_id = marker.read_text().strip()
    if requested_boot_id == boot_id:
        return

    uv_cache_dir = _uv_cache_dir(cache_dir)
    if uv_cache_dir.exists():
        shutil.rmtree(uv_cache_dir)
    uv_cache_dir.mkdir(parents=True)
    marker.unlink()
    logger.warning("cleared uv cache %s after node reboot", uv_cache_dir)
    _emit_event(
        UV_CACHE_RESET_COMPLETED_EVENT,
        requested_boot_id=requested_boot_id,
        current_boot_id=boot_id,
        cache_path=str(uv_cache_dir),
    )


def _prune_and_count_recovery_signals(cache_dir: Path, now: float) -> int:
    cutoff = now - UV_CACHE_RECOVERY_WINDOW
    attempt_uids: set[str] = set()
    for signal in _uv_cache_dir(cache_dir).glob(f"{UV_CACHE_RECOVERY_SIGNAL_PREFIX}*"):
        if signal.stat().st_mtime < cutoff:
            signal.unlink()
            continue
        if signal.name.startswith(_OBSERVED_RECOVERY_SIGNAL_PREFIX):
            attempt_uid = signal.name.removeprefix(_OBSERVED_RECOVERY_SIGNAL_PREFIX)
        else:
            attempt_uid = signal.name.removeprefix(UV_CACHE_RECOVERY_SIGNAL_PREFIX)
            _emit_event(UV_CACHE_RECOVERY_OBSERVED_EVENT, attempt_uid=attempt_uid)
            signal.replace(signal.with_name(f"{_OBSERVED_RECOVERY_SIGNAL_PREFIX}{attempt_uid}"))
        attempt_uids.add(attempt_uid)
    return len(attempt_uids)


def _request_coreweave_safe_reboot(
    k8s: K8sService,
    node_name: str,
    now: datetime,
    recovery_count: int,
) -> None:
    node = k8s.get_json(K8sResource.NODES, node_name)
    if node is None:
        raise ConnectionError(f"Kubernetes node {node_name!r} is not visible")

    conditions = node.get("status", {}).get("conditions", [])
    active_pending_state = next(
        (
            condition
            for condition in conditions
            if condition.get("type") == COREWEAVE_PENDING_STATE_CONDITION and condition.get("status") == "True"
        ),
        None,
    )
    labels = node.get("metadata", {}).get("labels", {})
    has_power_reset_label = labels.get(COREWEAVE_PENDING_STATE_LABEL) == COREWEAVE_POWER_RESET_STATE
    has_power_reset_condition = (
        active_pending_state is not None and active_pending_state.get("reason") == COREWEAVE_POWER_RESET_STATE
    )
    if active_pending_state is not None and not has_power_reset_condition:
        logger.warning(
            "node %s already has active CoreWeave lifecycle operation %s",
            node_name,
            active_pending_state.get("reason", "unknown"),
        )
        return

    if not has_power_reset_condition:
        timestamp = now.astimezone(UTC).isoformat().replace("+00:00", "Z")
        k8s.patch_json(
            K8sResource.NODES,
            node_name,
            [
                {
                    "op": "add",
                    "path": "/status/conditions/-",
                    "value": {
                        "type": COREWEAVE_PENDING_STATE_CONDITION,
                        "status": "True",
                        "lastHeartbeatTime": timestamp,
                        "lastTransitionTime": timestamp,
                        "reason": COREWEAVE_POWER_RESET_STATE,
                        "message": "Iris uv cache recovery threshold exceeded",
                    },
                }
            ],
            subresource="status",
        )
    if not has_power_reset_label:
        k8s.patch_json(
            K8sResource.NODES,
            node_name,
            [{"op": "add", "path": COREWEAVE_PENDING_STATE_LABEL_PATH, "value": COREWEAVE_POWER_RESET_STATE}],
        )
    if not has_power_reset_condition or not has_power_reset_label:
        logger.error("requested CoreWeave safe reboot for node %s", node_name)
        _emit_event(
            UV_CACHE_REBOOT_REQUESTED_EVENT,
            recovery_count=recovery_count,
            recovery_window_seconds=UV_CACHE_RECOVERY_WINDOW,
            lifecycle_state=COREWEAVE_POWER_RESET_STATE,
        )


def reconcile_uv_cache_recovery(
    k8s: K8sService,
    node_name: str,
    cache_dir: Path,
    boot_id: str,
    *,
    now: float,
) -> None:
    """Request node repair when recent task-local uv cache recoveries cross the threshold."""
    recovery_count = _prune_and_count_recovery_signals(cache_dir, now)
    if recovery_count < UV_CACHE_RECOVERY_THRESHOLD:
        return

    marker = cache_dir / UV_CACHE_RESET_MARKER
    marker.write_text(f"{boot_id}\n")
    _request_coreweave_safe_reboot(k8s, node_name, datetime.fromtimestamp(now, UTC), recovery_count)


def run_uv_cache_recovery(
    k8s: K8sService,
    node_name: str,
    cache_dir: Path,
    boot_id: str,
    stop: threading.Event,
) -> None:
    """Watch task recovery signals and request safe node repair when they cluster."""
    while not stop.is_set():
        try:
            reconcile_uv_cache_recovery(k8s, node_name, cache_dir, boot_id, now=time.time())
        except Exception as error:
            logger.exception("uv cache recovery check failed for node %s", node_name)
            _RECOVERY_FAILURES.add(1, attributes={"failure_kind": type(error).__name__})
        stop.wait(UV_CACHE_RECOVERY_INTERVAL)
