import asyncio
from uuid import uuid4

import pytest
from rigging import telemetry
from rigging.telemetry.serialization import EventBody

from harbor.models.job.config import TelemetryConfig
from harbor.telemetry.observer import (
    FAILURE_KIND_LLM_REQUEST,
    FAILURE_KIND_QUEUE_WAIT,
    bind_telemetry,
    record_failure,
    record_job_terminal,
    shutdown_telemetry,
    start_telemetry,
)


@pytest.mark.parametrize("ownership", ["foreign", "different_run", "stale_release"])
def test_telemetry_ownership_preserves_exports_and_job_errors(
    telemetry_capture, ownership
):
    config = TelemetryConfig(
        endpoint=telemetry_capture.endpoint,
        root_run_uid="training-run",
        execution_uid="native-attempt",
    )
    identity = uuid4()
    acquired = []
    try:
        if ownership == "foreign":
            telemetry.configure(
                endpoint=telemetry_capture.endpoint,
                service="other-application",
                attributes={"run_id": "other-run"},
            )
            refused = start_telemetry(config, job_uid=identity, job_name="first")
            acquired.append(refused)
            assert refused is None
            shutdown_telemetry(refused)
            telemetry.event("still_active", EventBody({"value": 1}))
            telemetry.shutdown(1)
            records = telemetry_capture.records
            assert any(record["name"] == "still_active" for _, record in records)
            assert all(
                resource["service"] == "other-application"
                and resource["attributes"]["run_id"] == "other-run"
                for resource, _ in records
            )
            return

        first = start_telemetry(config, job_uid=identity, job_name="first")
        acquired.append(first)
        if ownership == "different_run":
            refused = start_telemetry(
                config.model_copy(update={"root_run_uid": "different-run"}),
                job_uid=uuid4(),
                job_name="refused",
            )
            acquired.append(refused)
            assert refused is None
            shutdown_telemetry(refused)
            with bind_telemetry(first):
                record_failure(FAILURE_KIND_QUEUE_WAIT, asyncio.CancelledError())
            record_job_terminal(first, asyncio.CancelledError())
            shutdown_telemetry(first)
            events = {
                record["name"]: record["body"]
                for _, record in telemetry_capture.records
                if record["kind"] == "event"
            }
            assert events["job_terminal"]["outcome"] == "cancelled"
            assert "message" not in events["job_terminal"]
            assert "message" not in events["failure"]
            assert "status_code" not in events["failure"]
        else:
            second = start_telemetry(config, job_uid=uuid4(), job_name="second")
            acquired.append(second)
            shutdown_telemetry(first)
            restarted = start_telemetry(config, job_uid=identity, job_name="restarted")
            acquired.append(restarted)
            shutdown_telemetry(first)
            shutdown_telemetry(second)
            with bind_telemetry(restarted):
                record_failure(
                    FAILURE_KIND_LLM_REQUEST, RuntimeError("agent output \ud842 tail")
                )
            endpoint = "https://iris.example/proxy/t/scoped-token/eval/v1"
            record_job_terminal(
                restarted, RuntimeError(f"request to {endpoint} failed")
            )
            shutdown_telemetry(restarted)
            events = {
                record["name"]: record["body"]
                for _, record in telemetry_capture.records
                if record["kind"] == "event"
                and record["attributes"]["harbor_job_name"] == "restarted"
            }
            assert events["job_terminal"]["message"] == f"request to {endpoint} failed"
            assert events["job_terminal"]["outcome"] == "failed"
            assert "\\ud842" in events["failure"]["message"]
            assert "status_code" not in events["failure"]

        assert all(
            resource["service"] == "harbor"
            and resource["attributes"]["run_id"] == "training-run"
            and resource["attributes"]["execution_uid"] == "native-attempt"
            for resource, _ in telemetry_capture.records
        )
    finally:
        for runtime in acquired:
            shutdown_telemetry(runtime)
