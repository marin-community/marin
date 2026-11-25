# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture allocation timestamps before Iris garbage-collects completed GPU pods."""

import contextlib
import logging
import subprocess
import threading
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import yaml
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS
from iris.client.client import IrisClient
from iris.cluster.types import JobName
from rigging.config_discovery import resolve_cluster_config
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.knotmeter.artifacts import read_record, write_record
from experiments.evaluation.knotmeter.recipe import TrialPlan

logger = logging.getLogger(__name__)
# Project only timing and identity fields. Task environment values never enter this output.
POD_TIMINGS = (
    r'{range .items[*]}{.metadata.uid}{"\t"}{.metadata.labels.iris\.attempt_uid}{"\t"}'
    r'{.metadata.name}{"\t"}{.spec.nodeName}{"\t"}'
    r'{.status.conditions[?(@.type=="PodScheduled")].status}{"\t"}'
    r'{.status.conditions[?(@.type=="PodScheduled")].lastTransitionTime}{"\t"}'
    r'{.status.phase}{"\t"}{.status.initContainerStatuses[*].state.terminated.finishedAt}{" "}'
    r'{.status.containerStatuses[*].state.terminated.finishedAt}{"\n"}{end}'
)


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def capture_allocations(client: IrisClient, plan: TrialPlan) -> None:
    """Read only this trial's native pods and retain compact timing snapshots."""
    try:
        tasks = client.list_tasks(JobName.from_string(plan["gpu_job_id"]))
    except ConnectError as error:
        if error.code == Code.NOT_FOUND:
            return  # CPU preparation has not submitted the serving child yet.
        raise
    uids = [attempt.attempt_uid for task in tasks for attempt in client.task_status(task.task_id).attempts]
    if not uids:
        return
    path = resolve_cluster_config(plan["cluster"], dirs=IRIS_CLUSTER_CONFIG_DIRS)
    cluster = yaml.safe_load(Path(path).read_text())
    platform = cluster["platform"]["coreweave"]
    command = [
        "kubectl",
        "--kubeconfig",
        str(Path(platform["kubeconfig_path"]).expanduser()),
        "--context",
        platform["kube_context"],
        "--namespace",
        platform["namespace"],
        "get",
        "pods",
        "--selector",
        f"iris.attempt_uid in ({','.join(uids)})",
        "-o",
        f"jsonpath={POD_TIMINGS}",
    ]
    output = subprocess.check_output(command, text=True, timeout=30)
    for line in output.splitlines():
        uid, attempt, name, node, scheduled, scheduled_at, phase, finishes = line.split("\t")
        record = {
            "pod_uid": uid,
            "attempt_uid": attempt,
            "pod_name": name,
            "node_name": node,
            "observed_unix": time.time(),
            "phase": phase,
            "scheduled_unix": timestamp(scheduled_at) if scheduled == "True" and scheduled_at else None,
            "finished_unix": (
                max((timestamp(value) for value in finishes.split()), default=None)
                if phase in {"Succeeded", "Failed"}
                else None
            ),
        }
        write_record(prefix_join(plan["trial_uri"], f"attempts/{attempt}/allocation.json"), record)


@contextlib.contextmanager
def track_allocations(client: IrisClient, plan: TrialPlan) -> Iterator[None]:
    """Keep timing evidence while the local terminal waits; Iris owns job lifetime."""
    stopped = threading.Event()

    def capture() -> None:
        try:
            capture_allocations(client, plan)
        except Exception as error:
            # A timing read cannot cancel a running GPU trial. Preserve the
            # failure and leave cost incomplete until recovery can resolve it.
            logger.warning("Allocation timing capture failed: %s", error)
            write_record(prefix_join(plan["trial_uri"], "allocation-error.json"), {"error": str(error)})

    def watch() -> None:
        while not stopped.is_set():
            capture()
            stopped.wait(10)

    thread = threading.Thread(target=watch, name="knotmeter-allocation-capture")
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()
        capture()


def allocation_cost(
    plan: dict, attempt_uid: str, pod_uid: str, iris_finished: float | None, gpus: int
) -> tuple[float | None, dict]:
    """Return allocated GPU-seconds and measured, estimated or unknown timing evidence."""
    uri = prefix_join(plan["trial_uri"], f"attempts/{attempt_uid}/allocation.json")
    try:
        record = read_record(uri)
    except FileNotFoundError:
        return None, {"status": "unknown", "reason": "No retained pod allocation timestamp"}
    if record["pod_uid"] != pod_uid:
        raise ValueError("Allocation timestamps belong to a different native pod UID")
    start = record["scheduled_unix"]
    finish = record["finished_unix"] or iris_finished
    if start is None or finish is None:
        return None, record | {"status": "unknown"}
    if finish < start:
        raise ValueError("Allocation finish precedes pod scheduling")
    return (finish - start) * gpus, record | {
        "status": "measured" if record["finished_unix"] else "estimated",
        "end_basis": "pod-container-finish" if record["finished_unix"] else "iris-observed-terminal",
        "end_unix": finish,
    }
