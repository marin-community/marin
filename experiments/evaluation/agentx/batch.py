# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Advance one saved batch through native Iris jobs, then exit.

There is no resident coordinator. Actions serializes callers; Iris owns queued
jobs and replacement attempts. A collector can stop at any point and repeat the
same operation: stable job names and upstream ingest conflict keys make this safe.
"""

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from enum import Enum
from pathlib import Path

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fsspec.implementations.local import LocalFileSystem
from iris.client.client import IrisClient
from iris.client.workload import JobStatus
from iris.cluster.types import JobName
from iris.resources.state import is_job_finished
from rigging.filesystem import atomic
from rigging.filesystem.buckets import filesystem_for
from rigging.timing import Timestamp

from experiments.evaluation.agentx.submit import SERIES_PATH, point_manifest, submit_point


def json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (JobName, Path)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Timestamp):
        return {"epoch_ms": value.epoch_ms()}
    raise TypeError(f"Cannot serialize {type(value)}")


def read_record(uri: str) -> dict:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rt") as stream:
        return json.load(stream)


def write_record(uri: str, record: dict) -> None:
    filesystem, path = filesystem_for(uri)
    if isinstance(filesystem, LocalFileSystem):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    with atomic.atomic_rename(path, filesystem=filesystem) as temporary:
        # fsspec.open's automatic mkdir issues CreateBucket on S3. The existing
        # bucket is infrastructure; writing an object must not try to create it.
        with filesystem.open(temporary, "wt") as stream:
            json.dump(record, stream, indent=2, default=json_value)


def create_batch(
    source: dict,
    profiles: list[str],
    cluster: str,
    priority: str,
    iris_user: str,
    retention_uri: str,
    now: float,
    selected_concurrency: int | None,
) -> dict:
    """Freeze all comparison inputs before the first submission."""
    if source["kind"] != "github-actions" or source["run_id"] <= 0 or source["run_attempt"] <= 0:
        raise ValueError("A batch requires its real GitHub run ID and attempt")
    series = json.loads(SERIES_PATH.read_text())
    prefix = f"agentx-{source['run_id']}-{source['run_attempt']}"
    points = []
    for profile in profiles:
        target = cluster if profile == "h100" else series["profiles"][profile]["cluster"]
        for concurrency in series["profiles"][profile]["concurrencies"]:
            if selected_concurrency is not None and concurrency != selected_concurrency:
                continue
            name = f"{prefix}-{profile}-c{concurrency}"
            manifest = point_manifest(profile, concurrency, source)
            points.append(
                {
                    "name": name,
                    "job_id": f"/{iris_user}/{name}",
                    "cluster": target,
                    "manifest": manifest,
                    "state": "unsubmitted",
                }
            )
    immutable = {
        "source": source,
        "priority": priority,
        "points": points,
        "retention_uri": retention_uri,
        "created_unix": now,
        "deadline_unix": now + 7 * 86400,
    }
    digest = hashlib.sha256(json.dumps(immutable, sort_keys=True, default=json_value).encode()).hexdigest()
    # Normalize tuple-valued release metadata once, exactly as a saved JSON plan does.
    return json.loads(json.dumps(immutable | {"plan_sha256": digest, "status": "active"}, default=json_value))


def status_or_missing(client: IrisClient, job_id: str) -> JobStatus | None:
    try:
        return client.job_status(JobName.from_string(job_id))
    except ConnectError as error:
        if error.code == Code.NOT_FOUND:
            return None
        raise


def cancel_batch(client: IrisClient, batch: dict, checkpoint: str) -> None:
    """Persist cancellation before touching jobs, so collection cannot resurrect them."""
    batch["status"] = "cancelled"
    write_record(checkpoint, batch)
    for point in batch["points"]:
        status = status_or_missing(client, point["job_id"])
        if status is None:
            point["state"] = "unsubmitted"
            continue
        status = dataclasses.replace(status, tasks=tuple(client.list_tasks(status.job_id)))
        point["iris_status"] = json.loads(json.dumps(status, default=json_value))
        if not is_job_finished(status.state):
            point["state"] = "cancelling"
            write_record(checkpoint, batch)
            client.cancel_job(status.job_id)
        elif point.get("receipt", {}).get("status") != "valid":
            point["state"] = "cancelled"
    write_record(checkpoint, batch)


def advance_batch(
    client: IrisClient, batch: dict, checkpoint: str, collect: Callable[[JobStatus, dict, str], dict | None], now: float
) -> dict:
    """Make bounded progress, using one allocation at a time across all points."""
    if batch["status"] == "cancelled":
        cancel_batch(client, batch, checkpoint)
        return batch
    if batch["status"] == "complete":
        return batch
    if now >= batch["deadline_unix"]:
        cancel_batch(client, batch, checkpoint)
        return batch
    for point in batch["points"]:
        if point["state"] in {"collected", "failed"}:
            continue
        status = status_or_missing(client, point["job_id"])
        if status is None:
            # The intent is durable before the RPC. KEEP recovers an ambiguous submit.
            point["state"] = "submitting"
            write_record(checkpoint, batch)
            job = submit_point(client, point["name"], point["manifest"], point["cluster"], batch["priority"])
            point.update({"job_id": str(job.job_id), "state": "submitted"})
            write_record(checkpoint, batch)
            return batch
        point["iris_status"] = json.loads(json.dumps(status, default=json_value))
        if not is_job_finished(status.state):
            point["state"] = status.state.value
            write_record(checkpoint, batch)
            return batch
        # Collection retains terminal failure diagnostics too, but never processes
        # them as benchmark points. A successful command may still be finalizing its archive.
        # Federated GetJobStatus can omit tasks. ListTasks is the explicit
        # task/attempt read and forwards to the execution peer.
        status = dataclasses.replace(status, tasks=tuple(client.list_tasks(status.job_id)))
        receipt = collect(status, point["manifest"], batch["retention_uri"])
        if receipt is None:
            point["state"] = "awaiting-output"
            write_record(checkpoint, batch)
            return batch
        point.update({"receipt": receipt, "state": "collected" if receipt["status"] == "valid" else "failed"})
        write_record(checkpoint, batch)
        # Start the next point on the next collector invocation. The current
        # invocation can finish artifact upload/ingestion before another allocation.
        return batch
    batch["status"] = "complete"
    batch["finished_unix"] = time.time()
    write_record(checkpoint, batch)
    return batch
