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

from iris.cluster.backends.k8s.output_contract import (
    ATTEMPT_UID_ENV,
    OUTPUT_RELEASE_PATH,
    SOURCE_PREFIX_ENV,
    TASK_ID_ENV,
    output_policy_from_environment,
)
from iris.cluster.runtime.env import OUTPUT_PATH
from iris.cluster.runtime.output_capture import capture_task_outputs_for_attempt, task_output_storage_failure
from iris.cluster.types import AttemptUid
from iris.cluster.types import TaskAttempt as TaskAttemptIdentity
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

_TERMINATION_LOG = Path("/dev/termination-log")


def _wait_for_release(stop: threading.Event) -> bool:
    while not stop.wait(0.2):
        if Path(OUTPUT_RELEASE_PATH).exists():
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
        policy = output_policy_from_environment(os.environ)
        identity = TaskAttemptIdentity.from_wire(os.environ[TASK_ID_ENV])
        result = capture_task_outputs_for_attempt(
            Path(OUTPUT_PATH),
            policy,
            identity.task_id,
            AttemptUid(os.environ[ATTEMPT_UID_ENV]),
            local_root=Path("/tmp"),
            source_prefix=os.environ.get(SOURCE_PREFIX_ENV),
            stop=stop,
        )
    except Exception as exc:
        logger.exception("Failed to finalize task outputs")
        result = task_output_storage_failure(exc)
    _write_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
