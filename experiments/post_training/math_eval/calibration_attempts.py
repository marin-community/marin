# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Permit calibration startup retries only before an immutable measurement marker."""

import time

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.pool import canonical_json


def claim_measurement(output_uri, *, task_id, attempt_id, attempt_uid, binding_sha256, source_commit):
    """Persist the measurement boundary before any completion request can be sent."""
    output = StoragePath(output_uri)
    if output.exists():
        raise ValueError("Calibration measurement or failure evidence already exists")
    marker = {
        "schema": "calibration_measurement_start_v1",
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_uid": attempt_uid,
        "binding_sha256": binding_sha256,
        "source_commit": source_commit,
        "started_at_ms": time.time_ns() // 1_000_000,
    }
    raw = canonical_json(marker).encode()
    target = output / "measurement-start.json"
    target.write_bytes(raw)
    if target.read_bytes() != raw:
        raise ValueError("Calibration measurement marker failed immutable readback")
    return marker


def validate_calibration_task(generation, native_tasks, native_job):
    """Bind the measured attempt and retain every earlier startup allocation."""
    tasks = native_tasks.get("tasks", [])
    if len(tasks) != 1:
        raise ValueError("Calibration requires one native task")
    task, controller = tasks[0], native_job["job"]
    allocation = generation.get("controller_allocation", {})
    gpu = controller.get("resources", {}).get("device", {}).get("gpu", {})
    allocated = allocation.get("resources", {}).get("device", {})
    if (
        controller.get("job_id", "") + "/0" != task["task_id"]
        or task["task_id"] != generation.get("task_id")
        or controller.get("state") != "JOB_STATE_SUCCEEDED"
        or controller.get("exit_code") != 0
        or controller.get("cluster") != "cw-us-east-02a"
        or task.get("cluster") != "cw-us-east-02a"
        or task.get("state") != "TASK_STATE_SUCCEEDED"
        or task.get("exit_code") != 0
        or controller.get("task_count") != 1
        or controller.get("completed_count") != 1
        or gpu != {"variant": "H100", "count": 1}
        or any(allocated.get(k) != v for k, v in {"kind": "gpu", "variant": "H100", "count": 1}.items())
        or allocation.get("controller_scope") not in {"local", "cw-us-east-02a"}
        or generation.get("worker_region_hint") not in {None, "cw-us-east-02a"}
    ):
        raise ValueError("Calibration native resources, region or task identity changed")
    attempts = task.get("attempts", [])
    current_id = task.get("current_attempt_id")
    if (
        type(current_id) is not int
        or current_id not in (0, 1, 2)
        or generation.get("attempt_id") != current_id
        or [a.get("attempt_id") for a in attempts] != list(range(current_id + 1))
        or len({a.get("attempt_uid") for a in attempts}) != len(attempts)
    ):
        raise ValueError("Calibration attempt chain is incomplete or mismatched")
    current = attempts[-1]
    start = int(current["started_at"]["epoch_ms"])
    finish = int(current["finished_at"]["epoch_ms"])
    if (
        current.get("state") != "TASK_STATE_SUCCEEDED"
        or current.get("exit_code") != 0
        or not current.get("attempt_uid")
        or generation.get("attempt_uid") != current["attempt_uid"]
        or allocation.get("attempt_uid") != current["attempt_uid"]
        or allocation.get("started_at_ms") != start
        or not 0 < start < finish
        or int(task["finished_at"]["epoch_ms"]) != finish
    ):
        raise ValueError("Calibration current attempt lacks its native identity and interval")
    marker = generation.get("measurement_start", {})
    expected = {key: generation[key] for key in ("task_id", "attempt_id", "attempt_uid")}
    expected.update(
        schema="calibration_measurement_start_v1",
        binding_sha256=generation["binding"]["binding_sha256"],
        source_commit=generation["specification"]["source_commit"],
    )
    if (
        any(marker.get(key) != value for key, value in expected.items())
        or type(marker.get("started_at_ms")) is not int
        or not start <= marker["started_at_ms"] <= generation["panels"][0]["timing"]["started_at_ms"]
    ):
        raise ValueError("Calibration measurement marker belongs to another attempt or starts too late")
    intervals, previous_end = [], 0
    for attempt in attempts:
        prior = attempt is not current
        if prior and (
            attempt.get("state") not in {"TASK_STATE_FAILED", "TASK_STATE_PREEMPTED"} or not attempt.get("attempt_uid")
        ):
            raise ValueError("Calibration retried a successful or unidentified attempt")
        began = int(attempt.get("started_at", {}).get("epoch_ms", 0))
        ended = int(attempt.get("finished_at", {}).get("epoch_ms", 0))
        known = began > 0 and ended > began
        if known and (began < previous_end or (prior and ended > start)):
            raise ValueError("Calibration attempts have overlapping native intervals")
        if known:
            previous_end = ended
        intervals.append(
            {
                "attempt_id": attempt["attempt_id"],
                "attempt_uid": attempt["attempt_uid"],
                "state": attempt["state"],
                "started_at_ms": began or None,
                "finished_at_ms": ended or None,
                "task_h100_hours": (ended - began) / 3_600_000 if known else None,
                "scope": "startup before measurement" if prior else "measurement attempt",
            }
        )
    known_hours = sum(item["task_h100_hours"] or 0 for item in intervals)
    return {
        "start": start,
        "finish": finish,
        "attempts": intervals,
        "known_task_gpu_hours": known_hours,
        "task_gpu_hours": known_hours if all(item["task_h100_hours"] is not None for item in intervals) else None,
        "measurement_start_sha256": audit.canonical_sha(marker),
    }
