# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe task S3 storage from nodes; quarantine isolated failures."""

import hashlib
import json
import logging
import subprocess
import sys
import threading
from enum import IntEnum

import click
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    CredentialRetrievalError,
    NoCredentialsError,
    PartialCredentialsError,
)
from pydantic import BaseModel, ValidationError
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Timestamp

from iris.cluster.config import NodeStorageHealthConfig
from iris.cluster.platforms.k8s.service import K8sService
from iris.cluster.platforms.k8s.types import K8sResource, KubectlError

logger = logging.getLogger(__name__)
HEALTH_ANNOTATION = "iris.marin.community/storage-health"
CORDON_ANNOTATION = "iris.marin.community/storage-health-cordon"
PROBE_PAYLOAD = b"iris-node-storage-health\n"


class ProbeResult(IntEnum):
    HEALTHY = 0
    FAILED = 10
    CONFIGURATION_ERROR = 11


class StorageHealthReport(BaseModel):
    node_uid: str
    boot_id: str
    target: str
    started_at: float
    checked_at: float
    failure_since: float
    failures: int
    result: ProbeResult


def target_id(config: NodeStorageHealthConfig) -> str:
    target = [config.scratch, config.environment_revision]
    return hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()[:16]


def probe_storage(path: str) -> ProbeResult:
    """Round-trip one tiny object and delete it, including after a failed read.

    A fixed node-incarnation key bounds abandoned objects after process timeouts.
    The next successful probe deletes that key; bucket lifecycle expiry covers a
    node that never recovers. Caller must enforce a process-level deadline.
    """
    object_path = StoragePath(path)
    fs, key = url_to_fs(path)
    result = ProbeResult.HEALTHY
    try:
        object_path.write_bytes(PROBE_PAYLOAD)
        if fs.cat_file(key, start=0, end=len(PROBE_PAYLOAD) + 1) != PROBE_PAYLOAD:
            result = ProbeResult.FAILED
    except (OSError, ValueError, BotoCoreError, ClientError) as error:
        result = storage_error_result(error)
    finally:
        try:
            object_path.rm()
        except (OSError, ValueError, BotoCoreError, ClientError) as error:
            cleanup_result = storage_error_result(error)
            if result == ProbeResult.HEALTHY or cleanup_result == ProbeResult.CONFIGURATION_ERROR:
                result = cleanup_result
    return result


def storage_error_result(error: Exception) -> ProbeResult:
    if isinstance(
        error,
        (
            PermissionError,
            FileNotFoundError,
            ValueError,
            NoCredentialsError,
            PartialCredentialsError,
            CredentialRetrievalError,
        ),
    ):
        return ProbeResult.CONFIGURATION_ERROR
    if isinstance(error, ClientError):
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if 400 <= status < 500:
            return ProbeResult.CONFIGURATION_ERROR
    return ProbeResult.FAILED


def bounded_probe(path: str, timeout: float) -> ProbeResult:
    """Isolate DNS, credential discovery, and SDK retries behind a hard deadline."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", __name__, path],
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult.FAILED
    if completed.returncode in (ProbeResult.HEALTHY, ProbeResult.FAILED, ProbeResult.CONFIGURATION_ERROR):
        return ProbeResult(completed.returncode)
    # A broken installation is not evidence that its host should be cordoned.
    return ProbeResult.CONFIGURATION_ERROR


def run_storage_health(k8s: K8sService, node_name: str, config: NodeStorageHealthConfig, stop: threading.Event) -> None:
    """Publish health independently of the telemetry service's availability."""
    failures = 0
    failure_since = 0.0
    incarnation = ""
    while not stop.is_set():
        try:
            node = k8s.get_json(K8sResource.NODES, node_name)
            if node is None:
                raise RuntimeError(f"storage health node {node_name} disappeared")
            uid = node["metadata"]["uid"]
            boot_id = node["status"]["nodeInfo"]["bootID"]
            if incarnation != f"{uid}/{boot_id}":
                failures = 0
                failure_since = 0.0
                incarnation = f"{uid}/{boot_id}"
            started_at = Timestamp.now().epoch_seconds()
            result = bounded_probe(str(StoragePath.parse(config.scratch) / uid / boot_id), config.timeout)
            if result == ProbeResult.FAILED:
                failure_since = failure_since if failures else started_at
                failures += 1
            else:
                failures = 0
                failure_since = 0.0
            report = StorageHealthReport(
                node_uid=uid,
                boot_id=boot_id,
                target=target_id(config),
                started_at=started_at,
                checked_at=Timestamp.now().epoch_seconds(),
                failure_since=failure_since,
                failures=failures,
                result=result,
            )
            k8s.patch_node(node_name, {"metadata": {"annotations": {HEALTH_ANNOTATION: report.model_dump_json()}}})
            if result != ProbeResult.HEALTHY:
                logger.warning(
                    "node storage probe node=%s result=%s consecutive_failures=%d", node_name, result.name, failures
                )
        except KubectlError:
            logger.exception("could not publish node storage health for %s", node_name)
            # Lost publication breaks the consecutive evidence visible to controller.
            failures = 0
            failure_since = 0.0
        stop.wait(config.interval)


def current_report(node: dict, config: NodeStorageHealthConfig, now: float) -> StorageHealthReport | None:
    """Return a fresh report for this node and target, or None if absent, invalid, or stale."""
    metadata = node.get("metadata", {})
    raw = metadata.get("annotations", {}).get(HEALTH_ANNOTATION)
    if raw is None:
        return None
    try:
        report = StorageHealthReport.model_validate_json(raw)
    except ValidationError:
        return None
    if (
        report.node_uid != metadata.get("uid")
        or report.boot_id != node.get("status", {}).get("nodeInfo", {}).get("bootID")
        or report.target != target_id(config)
        or not 0 <= now - report.checked_at <= 2 * (config.interval + config.timeout)
    ):
        return None
    return report


def reconcile_storage_health(k8s: K8sService, config: NodeStorageHealthConfig, max_cordoned_nodes: int) -> None:
    """Cordon persistent outliers and release Iris cordons after recovery."""
    try:
        nodes = k8s.list_json(K8sResource.NODES)
        now = Timestamp.now().epoch_seconds()
        reports = [(node, current_report(node, config, now)) for node in nodes]
        cordoned_count = sum(CORDON_ANNOTATION in n.get("metadata", {}).get("annotations", {}) for n in nodes)
        released = 0
        for node, report in reports:
            metadata = node["metadata"]
            raw_cordon = metadata.get("annotations", {}).get(CORDON_ANNOTATION)
            if raw_cordon is None or report is None or report.result != ProbeResult.HEALTHY:
                continue
            try:
                cordon = StorageHealthReport.model_validate_json(raw_cordon)
            except ValidationError:
                continue
            if (
                cordon.result != ProbeResult.FAILED
                or report.node_uid != cordon.node_uid
                or report.boot_id != cordon.boot_id
                or report.target != cordon.target
                or report.started_at <= cordon.checked_at
            ):
                continue
            k8s.patch_node(
                metadata["name"],
                {
                    "metadata": {
                        "resourceVersion": metadata["resourceVersion"],
                        "annotations": {CORDON_ANNOTATION: None},
                    },
                    "spec": {"unschedulable": False},
                },
            )
            logger.info("uncordoned node=%s after successful storage probe", metadata["name"])
            released += 1
        remaining = max_cordoned_nodes - cordoned_count + released
        if remaining <= 0:
            return
        for node, report in reports:
            if (
                report is None
                or report.result != ProbeResult.FAILED
                or report.failures < config.failure_threshold
                or node.get("spec", {}).get("unschedulable", False)
                or CORDON_ANNOTATION in node.get("metadata", {}).get("annotations", {})
            ):
                continue
            # A majority must have completed a successful probe after this failure
            # began. Missing/stale/rebooted agents count against the majority.
            healthy_peers = sum(
                peer is not None and peer.result == ProbeResult.HEALTHY and peer.started_at > report.failure_since
                for _, peer in reports
            )
            if healthy_peers < max(config.minimum_healthy_nodes, len(nodes) // 2 + 1):
                continue
            metadata = node["metadata"]
            k8s.patch_node(
                metadata["name"],
                {
                    "metadata": {
                        "resourceVersion": metadata["resourceVersion"],
                        "annotations": {CORDON_ANNOTATION: report.model_dump_json()},
                    },
                    "spec": {"unschedulable": True},
                },
            )
            logger.warning("cordoned node=%s after %d storage probe failures", metadata["name"], report.failures)
            remaining -= 1
            if remaining <= 0:
                return
    except KubectlError:
        # Health maintenance must not prevent task termination/reconciliation.
        logger.exception("node storage health reconciliation failed")


@click.command()
@click.argument("path")
def main(path: str) -> None:
    raise SystemExit(probe_storage(path))


if __name__ == "__main__":
    main()
