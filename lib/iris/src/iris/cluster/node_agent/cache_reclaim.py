# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reclaim stale task-cache entries from an idle Kubernetes node."""

import logging
import os
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rigging import telemetry
from rigging.timing import Duration, Timestamp

from iris.cluster.platforms.k8s.service import K8sService
from iris.cluster.platforms.k8s.types import (
    IRIS_KUBERNETES_RUNTIME,
    IRIS_MANAGED_LABEL,
    IRIS_RUNTIME_LABEL,
    K8sResource,
    KubectlError,
)

logger = logging.getLogger(__name__)

CACHE_RECLAIM_INTERVAL = Duration.from_minutes(5)
_CACHE_RECLAIM_PREFIX = ".iris-reclaim-"
_TERMINAL_POD_PHASES = frozenset({"Failed", "Succeeded"})
_RECLAIM_FAILURES = telemetry.counter("iris_cache_reclaim_failures", unit="{failure}")
_RECLAIM_TAINT_EFFECT = "NoSchedule"
_RECLAIM_TAINT_NAME = "cache-reclaim"


def _record_failure(failure_kind: str, node_name: str) -> None:
    _RECLAIM_FAILURES.add(1, attributes={"failure_kind": failure_kind, "node_name": node_name})


def _entry_last_modified(entry: Path) -> Timestamp:
    latest = Timestamp.from_seconds(entry.lstat().st_mtime)
    if entry.is_symlink() or not entry.is_dir():
        return latest

    errors: list[OSError] = []
    for root, directories, files in os.walk(entry, followlinks=False, onerror=errors.append):
        for name in directories + files:
            try:
                latest = max(latest, Timestamp.from_seconds((Path(root) / name).lstat().st_mtime))
            except FileNotFoundError:
                continue
    if errors:
        raise errors[0]
    return latest


def _remove_cache_entry(entry: Path) -> None:
    if entry.name.startswith(_CACHE_RECLAIM_PREFIX):
        tombstone = entry
    else:
        # A task scheduled after the idle-node check can refill the original path
        # without racing the slower recursive deletion.
        tombstone = entry.with_name(f"{_CACHE_RECLAIM_PREFIX}{uuid.uuid4().hex}")
        entry.rename(tombstone)
    if tombstone.is_symlink() or not tombstone.is_dir():
        tombstone.unlink()
    else:
        shutil.rmtree(tombstone)


@contextmanager
def _block_task_admission(kubectl: K8sService, node_name: str) -> Iterator[bool]:
    taint_key = f"{kubectl.namespace}.iris.marin.community/{_RECLAIM_TAINT_NAME}"
    try:
        kubectl.add_node_taint(
            node_name,
            key=taint_key,
            value="true",
            effect=_RECLAIM_TAINT_EFFECT,
        )
    except KubectlError as error:
        logger.warning("cache reclamation could not block scheduling on node %s: %s", node_name, error)
        _record_failure("admission_block", node_name)
        yield False
        return

    try:
        yield True
    finally:
        try:
            kubectl.remove_node_taint(
                node_name,
                key=taint_key,
                value="true",
                effect=_RECLAIM_TAINT_EFFECT,
            )
        except KubectlError as error:
            logger.warning("cache reclamation could not unblock scheduling on node %s: %s", node_name, error)
            _record_failure("admission_unblock", node_name)


def _node_is_idle(kubectl: K8sService, node_name: str) -> bool:
    labels = {IRIS_MANAGED_LABEL: "true", IRIS_RUNTIME_LABEL: IRIS_KUBERNETES_RUNTIME}
    try:
        pods = kubectl.list_json(
            K8sResource.PODS,
            labels=labels,
            field_selector=f"spec.nodeName={node_name}",
        )
    except KubectlError as error:
        logger.warning("cache reclamation could not inspect tasks on node %s: %s", node_name, error)
        _record_failure("task_inspection", node_name)
        return False
    return not any(pod.get("status", {}).get("phase", "") not in _TERMINAL_POD_PHASES for pod in pods)


def reclaim_cache(
    cache_dir: Path,
    *,
    max_age: Duration,
    kubectl: K8sService,
    node_name: str,
    now: Timestamp | None = None,
) -> int:
    """Remove stale top-level cache entries while this node has no Iris tasks."""
    if not _node_is_idle(kubectl, node_name):
        return 0
    with _block_task_admission(kubectl, node_name) as admission_blocked:
        if not admission_blocked:
            return 0
        if not _node_is_idle(kubectl, node_name):
            return 0
        if not cache_dir.exists():
            return 0

        cutoff = (now or Timestamp.now()).add_ms(-max_age.to_ms())
        reclaimed = 0
        for namespace in cache_dir.iterdir():
            if namespace.is_symlink() or not namespace.is_dir():
                continue
            for entry in namespace.iterdir():
                try:
                    if not entry.name.startswith(_CACHE_RECLAIM_PREFIX) and _entry_last_modified(entry) > cutoff:
                        continue
                    _remove_cache_entry(entry)
                    reclaimed += 1
                except FileNotFoundError:
                    continue
                except OSError as error:
                    logger.warning("could not reclaim cache entry %s: %s", entry, error)
                    _record_failure("entry_removal", node_name)
        if reclaimed:
            logger.info("reclaimed %d stale cache entries from %s", reclaimed, cache_dir)
        return reclaimed


def run_cache_reclaimer(
    cache_dir: Path,
    max_age: Duration,
    kubectl: K8sService,
    node_name: str,
    stop: threading.Event,
) -> None:
    """Sweep the task cache periodically until the node agent stops."""
    while not stop.is_set():
        try:
            reclaim_cache(cache_dir, max_age=max_age, kubectl=kubectl, node_name=node_name)
        except OSError:
            logger.exception("cache reclamation failed for %s", cache_dir)
            _record_failure("cache_scan", node_name)
        stop.wait(CACHE_RECLAIM_INTERVAL.to_seconds())
