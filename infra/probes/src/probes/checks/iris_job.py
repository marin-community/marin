# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submits a tiny CPU job to a specific zone, polls until terminal, asserts SUCCEEDED."""

from __future__ import annotations

import logging
import time

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.constraints import zone_constraint
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    JobName,
    ResourceSpec,
    is_job_finished,
)
from iris.rpc import job_pb2
from rigging.timing import Duration

from probes.probe import ErrorClass, ProbeOutcome, ProbeResult

logger = logging.getLogger(__name__)


_TERMINAL_OK = job_pb2.JOB_STATE_SUCCEEDED
_USER = "probes"


class IrisJobSubmit:
    """Submits a one-task CPU job constrained to ``zone`` and polls to terminal."""

    def __init__(self, client: RemoteClusterClient, zone: str, image: str):
        if not zone:
            raise ValueError("zone must be non-empty")
        self._client = client
        self._zone = zone
        self._image = image

    def run(self, deadline_seconds: float) -> ProbeResult:
        job_id = JobName.root(_USER, f"canary-{self._zone}-{int(time.time())}")
        entrypoint = Entrypoint.from_command("python", "-c", "import time; time.sleep(1)")
        resources = ResourceSpec(cpu=1.0, memory="256m").to_proto()
        environment = EnvironmentSpec().to_proto()
        constraints = [zone_constraint(self._zone).to_proto()]

        submit_start = time.monotonic()
        try:
            submitted_id = self._client.submit_job(
                job_id=job_id,
                entrypoint=entrypoint,
                resources=resources,
                environment=environment,
                constraints=constraints,
                max_retries_failure=0,
                max_retries_preemption=0,
                timeout=Duration.from_seconds(60),
                task_image=self._image,
            )
        except Exception as exc:
            return ProbeResult.remote_error(ErrorClass.SUBMIT_REJECTED, f"{type(exc).__name__}: {exc}")
        submit_latency_ms = int((time.monotonic() - submit_start) * 1000)

        poll_deadline = max(0.5, deadline_seconds - (time.monotonic() - submit_start))
        try:
            status = self._client.wait_for_job(submitted_id, timeout=poll_deadline)
        except TimeoutError as exc:
            return ProbeResult(
                outcome=ProbeOutcome.TIMEOUT,
                error_class=ErrorClass.TIMEOUT,
                error_detail=str(exc),
                target_id=submitted_id.to_wire(),
                extras={"submit_latency_ms": submit_latency_ms},
            )
        except Exception as exc:
            return ProbeResult.remote_error(
                ErrorClass.RPC_ERROR,
                f"poll failed: {type(exc).__name__}: {exc}",
                target_id=submitted_id.to_wire(),
                extras={"submit_latency_ms": submit_latency_ms},
            )

        state = int(status.state)
        if not is_job_finished(state):
            # wait_for_job is supposed to either return terminal or raise; defensive.
            return ProbeResult(
                outcome=ProbeOutcome.TIMEOUT,
                error_class=ErrorClass.TIMEOUT,
                error_detail=f"non-terminal state {state}",
                target_id=submitted_id.to_wire(),
                extras={"submit_latency_ms": submit_latency_ms},
            )

        queue_latency_ms = _queue_latency_ms(status)
        extras: dict[str, str | int | float] = {
            "submit_latency_ms": submit_latency_ms,
            "queue_latency_ms": queue_latency_ms,
        }

        if state == _TERMINAL_OK:
            return ProbeResult.success(target_id=submitted_id.to_wire(), extras=extras)

        error_class = (
            ErrorClass.JOB_CANCELLED
            if state in (job_pb2.JOB_STATE_KILLED, job_pb2.JOB_STATE_WORKER_FAILED)
            else ErrorClass.JOB_FAILED
        )
        return ProbeResult.remote_error(
            error_class,
            f"terminal state {state}: {status.error}".strip(),
            target_id=submitted_id.to_wire(),
            extras=extras,
        )


def _queue_latency_ms(status: job_pb2.JobStatus) -> int:
    """Best-effort queue latency: submitted → first task start. Returns 0 if
    the relevant timestamps aren't populated on this status proto."""
    submitted_ms = status.submitted_at.epoch_ms if status.HasField("submitted_at") else 0
    started_ms = status.started_at.epoch_ms if status.HasField("started_at") else 0
    if submitted_ms and started_ms and started_ms >= submitted_ms:
        return started_ms - submitted_ms
    return 0
