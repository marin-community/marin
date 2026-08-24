# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes task-output uploader entry point."""

import logging
import os
import signal
import sys
import threading
from pathlib import Path

from google.protobuf import json_format
from rigging.timing import Deadline, Duration

from iris.cluster.config import TaskOutputDestination, TaskOutputPolicy
from iris.cluster.runtime.env import OUTPUT_PATH
from iris.cluster.runtime.output_capture import (
    TaskOutputLimits,
    capture_task_outputs,
    resolve_task_output_destination,
)
from iris.cluster.types import AttemptUid
from iris.cluster.types import TaskAttempt as TaskAttemptIdentity
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

_RELEASE_PATH = Path("/iris/output-control/release")
_TERMINATION_LOG = Path("/dev/termination-log")


def _policy_from_environment() -> TaskOutputPolicy:
    return TaskOutputPolicy(
        destination=TaskOutputDestination(os.environ["IRIS_TASK_OUTPUT_DESTINATION"]),
        ttl_days=int(os.environ["IRIS_TASK_OUTPUT_TTL_DAYS"]),
        max_bytes=int(os.environ["IRIS_TASK_OUTPUT_MAX_BYTES"]),
        max_entries=int(os.environ["IRIS_TASK_OUTPUT_MAX_ENTRIES"]),
        finalization_timeout=Duration.from_seconds(float(os.environ["IRIS_TASK_OUTPUT_TIMEOUT_SECONDS"])),
    )


def _wait_for_release(stop: threading.Event) -> bool:
    while not stop.wait(0.2):
        if _RELEASE_PATH.exists():
            return True
    return False


def _write_result(result: job_pb2.TaskOutputArchive) -> None:
    _TERMINATION_LOG.write_text(
        json_format.MessageToJson(result, preserving_proto_field_name=True, indent=None),
        encoding="utf-8",
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s outputship %(message)s")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    if not _wait_for_release(stop):
        _write_result(
            job_pb2.TaskOutputArchive(
                state=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_UNAVAILABLE,
                error="pod_terminated_before_capture",
            )
        )
        return 0

    try:
        policy = _policy_from_environment()
        identity = TaskAttemptIdentity.from_wire(os.environ["IRIS_TASK_ID"])
        destination = resolve_task_output_destination(
            policy,
            identity.task_id,
            AttemptUid(os.environ["IRIS_ATTEMPT_UID"]),
            local_root=Path("/tmp"),
            source_prefix=os.environ.get("IRIS_TASK_OUTPUT_SOURCE_PREFIX"),
        )
        result = capture_task_outputs(
            Path(OUTPUT_PATH),
            destination,
            TaskOutputLimits(max_bytes=policy.max_bytes, max_entries=policy.max_entries),
            Deadline.from_now(policy.finalization_timeout),
            stop,
        )
    except Exception as exc:
        logger.exception("Failed to finalize task outputs")
        result = job_pb2.TaskOutputArchive(
            state=job_pb2.TaskOutputArchive.TASK_OUTPUT_ARCHIVE_STATE_FAILED,
            error=f"storage_error: {str(exc)[:900]}",
        )
    _write_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
