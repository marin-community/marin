# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for controller RPC service implementation.

These tests verify the RPC contract (input -> output) of the ControllerServiceImpl.
State changes are verified via RPC calls rather than internal state inspection.
"""

from datetime import date, timedelta

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.constraints import ConstraintOp, WellKnownAttribute, device_variant_constraint
from iris.cluster.controller.service import (
    FEATURE_INTRODUCTION_DATE,
    FRESHNESS_WINDOW,
    ControllerServiceImpl,
    _check_client_freshness,
)
from iris.log_server.server import LogServiceImpl
from iris.cluster.controller.codec import constraints_from_json
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.types import JobName, WorkerId, tpu_device
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from rigging.timing import Timestamp

from .conftest import (
    make_job_request,
    make_test_entrypoint,
    make_worker_metadata,
    query_job as _query_job,
    query_tasks_with_attempts as _query_tasks_with_attempts,
)

# =============================================================================
# Test Helpers
# =============================================================================


def _register_worker(state: ControllerTransitions, worker_id: WorkerId) -> None:
    metadata = job_pb2.WorkerMetadata(
        hostname=str(worker_id),
        ip_address="127.0.0.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
    )
    state.register_or_refresh_worker(
        worker_id=worker_id,
        address=f"{worker_id}:8080",
        metadata=metadata,
        ts=Timestamp.now(),
    )


def _set_job_state(state: ControllerTransitions, job_id: JobName, state_value: int) -> None:
    state._db.execute(
        "UPDATE jobs SET state = ? WHERE job_id = ?",
        (state_value, job_id.to_wire()),
    )


def _assign_and_transition(
    state: ControllerTransitions,
    task_id: JobName,
    worker_id: WorkerId,
    target_state: int,
    *,
    error: str | None = None,
) -> None:
    state.queue_assignments([Assignment(task_id=task_id, worker_id=worker_id)])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)],
        )
    )
    if target_state != job_pb2.TASK_STATE_RUNNING:
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[TaskUpdate(task_id=task_id, attempt_id=0, new_state=target_state, error=error)],
            )
        )


@pytest.fixture
def service(controller_service):
    return controller_service


# =============================================================================
# Job Launch Tests
# =============================================================================


def test_launch_job_returns_job_id(service):
    """Verify launch_job returns a job_id and job can be queried via RPC."""
    request = make_job_request("test-job")

    response = service.launch_job(request, None)

    assert response.job_id == JobName.root("test-user", "test-job").to_wire()

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "test-job").to_wire()), None
    )
    assert status_response.job.job_id == JobName.root("test-user", "test-job").to_wire()
    assert status_response.job.state == job_pb2.JOB_STATE_PENDING


def test_launch_job_bundle_blob_rewrites_to_controller_bundle_id(service, state):
    request = make_job_request("bundle-job")
    request.bundle_blob = b"bundle-bytes"
    service.launch_job(request, None)

    job = _query_job(state, JobName.root("test-user", "bundle-job"))
    assert job is not None
    assert len(job.bundle_id) == 64


def test_launch_job_rejects_tpu_chip_count_mismatch(service):
    """A job requesting fewer chips than the variant's chips_per_vm is rejected."""
    request = make_job_request("bad-tpu-chip-count")
    request.resources.device.CopyFrom(tpu_device("v6e-8", count=4))

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.INVALID_ARGUMENT
    assert "chip count mismatch" in exc_info.value.message


def test_launch_job_rejects_mixed_vm_shape_alternatives(service):
    """device-variant IN constraint with mismatched chips_per_vm is rejected."""
    request = make_job_request("mixed-tpu-variants")
    request.resources.device.CopyFrom(tpu_device("v6e-4"))
    # User-provided IN constraint that mixes a 4-chip/VM and an 8-chip/VM variant.
    request.constraints.append(device_variant_constraint(["v6e-4", "v6e-8"]).to_proto())

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.INVALID_ARGUMENT
    # Mismatched shapes necessarily imply a chip-count mismatch for at least one
    # candidate, so the per-candidate count check fires first.
    assert "chip count mismatch" in exc_info.value.message
    assert "v6e-8" in exc_info.value.message


def test_launch_job_rejects_variant_override_with_smaller_primary(service):
    """Explicit device-variant constraint overrides the primary; chip count must match it.

    Regression for Codex review: primary v6e-4 (chips_per_vm=4) with an explicit
    `device-variant EQ v6e-8` constraint would schedule onto a single v6e-8 VM
    while reserving only 4 of its 8 chips — the exact partial-VM collision we
    want to block. The validator must check chip count against every effective
    candidate, not just the primary.
    """
    request = make_job_request("variant-override-mismatch")
    request.resources.device.CopyFrom(tpu_device("v6e-4"))
    request.constraints.append(device_variant_constraint(["v6e-8"]).to_proto())

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.INVALID_ARGUMENT
    assert "chip count mismatch" in exc_info.value.message


def test_launch_job_accepts_same_shape_alternatives(service):
    """Alternatives sharing vm_count/chips_per_vm (e.g. v4-8 + v5p-8) are accepted."""
    request = make_job_request("matched-tpu-variants")
    request.resources.device.CopyFrom(tpu_device("v4-8"))
    request.constraints.append(device_variant_constraint(["v4-8", "v5p-8"]).to_proto())

    response = service.launch_job(request, None)
    assert response.job_id == JobName.root("test-user", "matched-tpu-variants").to_wire()


def test_launch_job_rejects_duplicate_name(service):
    """Verify launch_job rejects duplicate job names for running jobs."""
    request = make_job_request("duplicate-job")

    response = service.launch_job(request, None)
    assert response.job_id == JobName.root("test-user", "duplicate-job").to_wire()

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "still running" in exc_info.value.message


def test_launch_job_replaces_finished_job_by_default(service, state):
    """Verify launch_job replaces finished jobs by default."""
    request = make_job_request("replaceable-job")
    job_id = JobName.root("test-user", "replaceable-job")

    # Submit initial job
    response = service.launch_job(request, None)
    assert response.job_id == job_id.to_wire()

    # Mark the job as failed
    job = _query_job(state, job_id)
    assert job is not None
    tasks = _query_tasks_with_attempts(state, job.job_id)
    assert len(tasks) == 1
    _set_job_state(state, job.job_id, job_pb2.JOB_STATE_FAILED)

    # Verify job is now failed
    job = _query_job(state, job_id)
    assert job.state == job_pb2.JOB_STATE_FAILED

    # Submit again - should succeed (replaces the finished job)
    response = service.launch_job(request, None)
    assert response.job_id == job_id.to_wire()

    # Verify the new job is pending
    job = _query_job(state, job_id)
    assert job.state == job_pb2.JOB_STATE_PENDING


def test_launch_job_error_policy_prevents_replacement(service, state):
    """Verify EXISTING_JOB_POLICY_ERROR prevents replacing finished jobs."""
    request = make_job_request("no-replace-job")
    job_id = JobName.root("test-user", "no-replace-job")

    # Submit initial job
    response = service.launch_job(request, None)
    assert response.job_id == job_id.to_wire()

    # Mark the job as succeeded
    job = _query_job(state, job_id)
    _set_job_state(state, job.job_id, job_pb2.JOB_STATE_SUCCEEDED)

    # Verify job is now succeeded
    job = _query_job(state, job_id)
    assert job.state == job_pb2.JOB_STATE_SUCCEEDED

    # Submit again with ERROR policy - should fail
    request_no_replace = make_job_request("no-replace-job")
    request_no_replace.existing_job_policy = job_pb2.EXISTING_JOB_POLICY_ERROR

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request_no_replace, None)

    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "SUCCEEDED" in exc_info.value.message


def test_existing_job_policy_keep_running(service, state):
    """KEEP policy on a running job returns the existing handle without re-creating."""
    request = make_job_request("keep-job")
    job_id = JobName.root("test-user", "keep-job")

    service.launch_job(request, None)

    # Job is still running (PENDING). Submit again with KEEP policy.
    request_keep = make_job_request("keep-job")
    request_keep.existing_job_policy = job_pb2.EXISTING_JOB_POLICY_KEEP
    response = service.launch_job(request_keep, None)

    assert response.job_id == job_id.to_wire()
    # Job should still be in its original PENDING state (not replaced).
    job = _query_job(state, job_id)
    assert job.state == job_pb2.JOB_STATE_PENDING


def test_existing_job_policy_recreate_running(service, state):
    """RECREATE policy cancels a running job and replaces it."""
    request = make_job_request("recreate-job")
    job_id = JobName.root("test-user", "recreate-job")

    service.launch_job(request, None)
    # Confirm job exists and is pending
    job = _query_job(state, job_id)
    assert job.state == job_pb2.JOB_STATE_PENDING

    request_recreate = make_job_request("recreate-job")
    request_recreate.existing_job_policy = job_pb2.EXISTING_JOB_POLICY_RECREATE
    response = service.launch_job(request_recreate, None)

    assert response.job_id == job_id.to_wire()
    # New job should be pending (the old one was cancelled and removed).
    job = _query_job(state, job_id)
    assert job.state == job_pb2.JOB_STATE_PENDING


def test_existing_job_policy_error_any_state(service, state):
    """ERROR policy rejects submission regardless of job state."""
    request = make_job_request("error-policy-job")
    job_id = JobName.root("test-user", "error-policy-job")

    service.launch_job(request, None)

    # Running job with ERROR policy -> error
    request_err = make_job_request("error-policy-job")
    request_err.existing_job_policy = job_pb2.EXISTING_JOB_POLICY_ERROR
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request_err, None)
    assert exc_info.value.code == Code.ALREADY_EXISTS

    # Mark job as finished, ERROR policy should still reject
    _set_job_state(state, job_id, job_pb2.JOB_STATE_SUCCEEDED)
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request_err, None)
    assert exc_info.value.code == Code.ALREADY_EXISTS


def test_existing_job_policy_unspecified_preserves_current_behavior(service, state):
    """Default (UNSPECIFIED) policy replaces finished jobs and errors on running ones."""
    request = make_job_request("default-policy-job")
    job_id = JobName.root("test-user", "default-policy-job")

    service.launch_job(request, None)

    # Running job -> error (same as before)
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)
    assert exc_info.value.code == Code.ALREADY_EXISTS
    assert "still running" in exc_info.value.message

    # Finished job -> replaced
    _set_job_state(state, job_id, job_pb2.JOB_STATE_FAILED)
    response = service.launch_job(request, None)
    assert response.job_id == job_id.to_wire()
    job = _query_job(state, job_id)
    assert job.state == job_pb2.JOB_STATE_PENDING


def test_launch_job_rejects_empty_name(service, state):
    """Verify launch_job rejects empty job names."""
    request = controller_pb2.Controller.LaunchJobRequest(
        name="",
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
    )

    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)

    assert exc_info.value.code == Code.INVALID_ARGUMENT


# =============================================================================
# Job Status Tests
# =============================================================================


def test_get_job_status_returns_status(service):
    """Verify get_job_status returns correct status for launched job."""
    service.launch_job(make_job_request("test-job"), None)

    request = controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "test-job").to_wire())
    response = service.get_job_status(request, None)

    assert response.job.job_id == JobName.root("test-user", "test-job").to_wire()
    assert response.job.state == job_pb2.JOB_STATE_PENDING


def test_get_job_status_reports_has_children(service, state):
    """GetJobStatus sets has_children so the dashboard can render the expand toggle."""
    service.launch_job(make_job_request("parent-job"), None)
    parent_id = JobName.root("test-user", "parent-job")

    child_id = JobName.from_wire(parent_id.to_wire() + "/child")
    child_req = controller_pb2.Controller.LaunchJobRequest(
        name=child_id.to_wire(),
        entrypoint=job_pb2.RuntimeEntrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
    )
    child_req.entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    state.submit_job(child_id, child_req, Timestamp.now())

    parent = service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=parent_id.to_wire()), None)
    assert parent.job.has_children is True

    child = service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=child_id.to_wire()), None)
    assert child.job.has_children is False


def test_get_job_status_not_found(service):
    """Verify get_job_status raises ConnectError for unknown job."""
    request = controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "nonexistent").to_wire())

    with pytest.raises(ConnectError) as exc_info:
        service.get_job_status(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


def test_redact_request_env_vars_does_not_mutate_original():
    """Verify redact_request_env_vars returns a copy and does not mutate the input."""
    from iris.cluster.redaction import REDACTED_VALUE, redact_request_env_vars

    original = controller_pb2.Controller.LaunchJobRequest(
        name="/test-user/job",
        entrypoint=make_test_entrypoint(),
        environment=job_pb2.EnvironmentConfig(env_vars={"WANDB_API_KEY": "secret", "SAFE": "ok"}),
    )
    redacted = redact_request_env_vars(original)

    assert original.environment.env_vars["WANDB_API_KEY"] == "secret"
    assert redacted.environment.env_vars["WANDB_API_KEY"] == REDACTED_VALUE
    assert redacted.environment.env_vars["SAFE"] == "ok"


def test_submit_argv_roundtrips_through_get_job_status(service):
    """submit_argv set on LaunchJob must survive storage and reconstruction."""
    job_name = JobName.root("test-user", "submit-argv-test")
    launch_req = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        submit_argv=["iris", "job", "run", "-e", "LOG_LEVEL", "info", "--", "python", "t.py"],
    )
    service.launch_job(launch_req, None)

    response = service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=job_name.to_wire()), None)
    assert list(response.request.submit_argv) == [
        "iris",
        "job",
        "run",
        "-e",
        "LOG_LEVEL",
        "info",
        "--",
        "python",
        "t.py",
    ]


def test_submit_argv_empty_when_omitted(service):
    """Programmatic submissions without submit_argv should reconstruct as empty."""
    job_name = JobName.root("test-user", "submit-argv-empty")
    launch_req = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
    )
    service.launch_job(launch_req, None)

    response = service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=job_name.to_wire()), None)
    assert list(response.request.submit_argv) == []


def test_get_job_status_redacts_sensitive_env_vars(service):
    """Verify get_job_status redacts env var values whose keys match sensitive patterns."""
    from iris.cluster.redaction import REDACTED_VALUE

    job_name = JobName.root("test-user", "redact-test")
    launch_req = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(
            env_vars={
                "WANDB_API_KEY": "wk-secret123",
                "HF_TOKEN": "hf_tok456",
                "MY_SECRET": "s3cret",
                "DB_PASSWORD": "hunter2",
                "SAFE_VAR": "visible",
                "NUM_WORKERS": "4",
            }
        ),
    )
    service.launch_job(launch_req, None)

    status_req = controller_pb2.Controller.GetJobStatusRequest(job_id=job_name.to_wire())
    response = service.get_job_status(status_req, None)

    env = dict(response.request.environment.env_vars)
    assert env["WANDB_API_KEY"] == REDACTED_VALUE
    assert env["HF_TOKEN"] == REDACTED_VALUE
    assert env["MY_SECRET"] == REDACTED_VALUE
    assert env["DB_PASSWORD"] == REDACTED_VALUE
    assert env["SAFE_VAR"] == "visible"
    assert env["NUM_WORKERS"] == "4"


def test_get_job_status_omits_per_task_detail(service):
    """GetJobStatus never populates per-task detail (callers use ListTasks)."""
    service.launch_job(make_job_request("task-test"), None)
    job_id = JobName.root("test-user", "task-test")

    request = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire())
    response = service.get_job_status(request, None)

    assert response.job.state == job_pb2.JOB_STATE_PENDING
    assert len(response.job.tasks) == 0
    assert response.job.task_count == 1


# =============================================================================
# Job Termination Tests
# =============================================================================


def test_terminate_job_marks_as_killed(service):
    """Verify terminate_job sets job state to KILLED via get_job_status."""
    service.launch_job(make_job_request("test-job"), None)

    request = controller_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "test-job").to_wire())
    response = service.terminate_job(request, None)

    assert isinstance(response, job_pb2.Empty)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "test-job").to_wire()), None
    )
    assert status_response.job.state == job_pb2.JOB_STATE_KILLED
    assert status_response.job.finished_at.epoch_ms > 0


def test_terminate_job_not_found(service):
    """Verify terminate_job raises ConnectError for unknown job."""
    request = controller_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "nonexistent").to_wire())

    with pytest.raises(ConnectError) as exc_info:
        service.terminate_job(request, None)

    assert exc_info.value.code == Code.NOT_FOUND
    assert "nonexistent" in exc_info.value.message


def test_terminate_pending_job(service):
    """Verify terminate_job works on pending jobs (not just running)."""
    service.launch_job(make_job_request("test-job"), None)

    request = controller_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "test-job").to_wire())
    service.terminate_job(request, None)

    # Verify via get_job_status RPC
    status_response = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "test-job").to_wire()), None
    )
    assert status_response.job.state == job_pb2.JOB_STATE_KILLED
    assert status_response.job.finished_at.epoch_ms > 0


def test_terminate_job_cascades_to_children(service):
    """Verify terminate_job terminates all children when parent is terminated."""
    service.launch_job(make_job_request("parent"), None)
    service.launch_job(make_job_request("/test-user/parent/child1"), None)
    service.launch_job(make_job_request("/test-user/parent/child2"), None)

    request = controller_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "parent").to_wire())
    service.terminate_job(request, None)

    # Verify all jobs are killed via get_job_status RPC
    for job_name in [
        JobName.root("test-user", "parent"),
        JobName.from_string("/test-user/parent/child1"),
        JobName.from_string("/test-user/parent/child2"),
    ]:
        status = service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=job_name.to_wire()), None)
        assert status.job.state == job_pb2.JOB_STATE_KILLED, f"Job {job_name} should be KILLED"


def test_terminate_job_only_affects_descendants(service):
    """Verify terminate_job does not affect sibling jobs."""
    service.launch_job(make_job_request("parent"), None)
    service.launch_job(make_job_request("/test-user/parent/child1"), None)
    service.launch_job(make_job_request("/test-user/parent/child2"), None)

    # Terminate only child1
    request = controller_pb2.Controller.TerminateJobRequest(
        job_id=JobName.from_string("/test-user/parent/child1").to_wire()
    )
    service.terminate_job(request, None)

    # Verify states via get_job_status RPC
    child1_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/test-user/parent/child1").to_wire()),
        None,
    )
    assert child1_status.job.state == job_pb2.JOB_STATE_KILLED

    child2_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/test-user/parent/child2").to_wire()),
        None,
    )
    assert child2_status.job.state == job_pb2.JOB_STATE_PENDING

    parent_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "parent").to_wire()), None
    )
    assert parent_status.job.state == job_pb2.JOB_STATE_PENDING


def test_terminate_job_skips_already_finished_children(service, state):
    """Verify terminate_job skips children already in terminal state."""
    # Launch parent via RPC
    service.launch_job(make_job_request("parent"), None)

    # Create child and transition it to SUCCEEDED.
    service.launch_job(make_job_request("/test-user/parent/child-succeeded"), None)
    child_succeeded_job = JobName.from_string("/test-user/parent/child-succeeded")
    child_task = _query_tasks_with_attempts(state, child_succeeded_job)[0]
    done_worker = WorkerId("w-child-succeeded")
    _register_worker(state, done_worker)
    _assign_and_transition(state, child_task.task_id, done_worker, job_pb2.TASK_STATE_SUCCEEDED)

    # Launch running child via RPC
    service.launch_job(make_job_request("/test-user/parent/child-running"), None)

    # Terminate parent
    request = controller_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "parent").to_wire())
    service.terminate_job(request, None)

    # Verify states via get_job_status RPC
    succeeded_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(
            job_id=JobName.from_string("/test-user/parent/child-succeeded").to_wire()
        ),
        None,
    )
    assert succeeded_status.job.state == job_pb2.JOB_STATE_SUCCEEDED

    running_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(
            job_id=JobName.from_string("/test-user/parent/child-running").to_wire()
        ),
        None,
    )
    assert running_status.job.state == job_pb2.JOB_STATE_KILLED

    parent_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "parent").to_wire()),
        None,
    )
    assert parent_status.job.state == job_pb2.JOB_STATE_KILLED


# =============================================================================
# Authorization Tests
# =============================================================================


def test_terminate_job_allowed_by_owner(service):
    """Job owner can terminate their own job."""
    from iris.rpc.auth import _verified_identity, VerifiedIdentity

    service.launch_job(make_job_request("/alice/my-job"), None)

    token = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        request = controller_pb2.Controller.TerminateJobRequest(job_id="/alice/my-job")
        service.terminate_job(request, None)
    finally:
        _verified_identity.reset(token)

    status = service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id="/alice/my-job"), None)
    assert status.job.state == job_pb2.JOB_STATE_KILLED


def test_terminate_job_rejected_for_non_owner(state, mock_controller, tmp_path):
    """Non-owner gets PERMISSION_DENIED when trying to terminate another user's job."""
    from iris.cluster.bundle import BundleStore
    from iris.cluster.controller.auth import ControllerAuth
    from iris.rpc.auth import _verified_identity, VerifiedIdentity

    auth_service = ControllerServiceImpl(
        state,
        state._store,
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles_owner")),
        log_service=LogServiceImpl(),
        auth=ControllerAuth(provider="static"),
    )

    auth_service.launch_job(make_job_request("/alice/my-job"), None)

    token = _verified_identity.set(VerifiedIdentity(user_id="bob", role="user"))
    try:
        request = controller_pb2.Controller.TerminateJobRequest(job_id="/alice/my-job")
        with pytest.raises(ConnectError) as exc_info:
            auth_service.terminate_job(request, None)
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(token)

    # Job should still be running
    status = auth_service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id="/alice/my-job"), None)
    assert status.job.state == job_pb2.JOB_STATE_PENDING


def test_launch_child_job_rejected_for_non_owner(state, mock_controller, tmp_path):
    """Cannot submit a child job under another user's hierarchy."""
    from iris.cluster.bundle import BundleStore
    from iris.cluster.controller.auth import ControllerAuth
    from iris.rpc.auth import _verified_identity, VerifiedIdentity

    auth_service = ControllerServiceImpl(
        state,
        state._store,
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles_child")),
        log_service=LogServiceImpl(),
        auth=ControllerAuth(provider="static"),
    )

    auth_service.launch_job(make_job_request("/alice/parent-job"), None)

    token = _verified_identity.set(VerifiedIdentity(user_id="bob", role="user"))
    try:
        with pytest.raises(ConnectError) as exc_info:
            auth_service.launch_job(make_job_request("/alice/parent-job/sneaky-child"), None)
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(token)


def test_terminate_job_allowed_when_auth_disabled(service):
    """When auth is disabled (no verified user), anyone can terminate."""
    service.launch_job(make_job_request("/alice/my-job"), None)

    # No _verified_identity set => auth disabled
    request = controller_pb2.Controller.TerminateJobRequest(job_id="/alice/my-job")
    service.terminate_job(request, None)

    status = service.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id="/alice/my-job"), None)
    assert status.job.state == job_pb2.JOB_STATE_KILLED


def test_parent_job_failure_cascades_to_children(service, state):
    """Verify when a parent job fails, all children are automatically cancelled."""
    # Launch parent and children via RPC
    service.launch_job(make_job_request("parent"), None)
    service.launch_job(make_job_request("/test-user/parent/child1"), None)
    service.launch_job(make_job_request("/test-user/parent/child2"), None)

    # Get parent task and mark it as failed
    parent_job = _query_job(state, JobName.root("test-user", "parent"))
    parent_task = _query_tasks_with_attempts(state, parent_job.job_id)[0]
    worker_id = WorkerId("w-parent")
    _register_worker(state, worker_id)
    _assign_and_transition(
        state,
        parent_task.task_id,
        worker_id,
        job_pb2.TASK_STATE_FAILED,
        error="Parent task failed",
    )

    # Verify all jobs are now in terminal states via get_job_status RPC
    parent_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.root("test-user", "parent").to_wire()), None
    )
    assert parent_status.job.state == job_pb2.JOB_STATE_FAILED

    child1_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/test-user/parent/child1").to_wire()),
        None,
    )
    assert child1_status.job.state == job_pb2.JOB_STATE_KILLED, "Child 1 should be killed when parent fails"

    child2_status = service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=JobName.from_string("/test-user/parent/child2").to_wire()),
        None,
    )
    assert child2_status.job.state == job_pb2.JOB_STATE_KILLED, "Child 2 should be killed when parent fails"


def test_launch_job_rejects_child_of_failed_parent(service, state):
    """Verify launch_job rejects submissions to a failed parent's namespace."""
    # Launch and fail parent
    service.launch_job(make_job_request("failed-parent"), None)
    parent_job = _query_job(state, JobName.root("test-user", "failed-parent"))
    parent_task = _query_tasks_with_attempts(state, parent_job.job_id)[0]
    worker_id = WorkerId("w-failed-parent")
    _register_worker(state, worker_id)
    _assign_and_transition(
        state,
        parent_task.task_id,
        worker_id,
        job_pb2.TASK_STATE_FAILED,
        error="Parent task failed",
    )

    # Try to submit a child job - should fail
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(make_job_request("/test-user/failed-parent/new-child"), None)

    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert "terminated" in exc_info.value.message.lower() or "failed" in exc_info.value.message.lower()


def test_launch_job_rejects_child_of_absent_parent(service):
    """Reject child submissions when the parent row is missing from the DB.

    Simulates a controller restart where the checkpoint did not capture the
    parent row but running processes keep submitting descendants. Previously
    the guard only rejected terminated parents, leaving absent-parent children
    inserted with `parent_job_id = NULL` and an orphaned `depth`.
    """
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(make_job_request("/test-user/absent-parent/new-child"), None)

    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert "absent" in exc_info.value.message.lower() or "not found" in exc_info.value.message.lower()


# =============================================================================
# Job List Tests
# =============================================================================


def test_list_jobs_returns_all_jobs(service):
    """Verify list_jobs returns all jobs launched via RPC."""
    service.launch_job(make_job_request("job-1"), None)
    service.launch_job(make_job_request("job-2"), None)
    service.launch_job(make_job_request("job-3"), None)

    # Terminate one to get different state
    service.terminate_job(
        controller_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "job-3").to_wire()), None
    )

    request = controller_pb2.Controller.ListJobsRequest()
    response = service.list_jobs(request, None)

    assert len(response.jobs) == 3
    job_ids = {j.job_id for j in response.jobs}
    assert job_ids == {
        JobName.root("test-user", "job-1").to_wire(),
        JobName.root("test-user", "job-2").to_wire(),
        JobName.root("test-user", "job-3").to_wire(),
    }

    states_by_id = {j.job_id: j.state for j in response.jobs}
    assert states_by_id[JobName.root("test-user", "job-1").to_wire()] == job_pb2.JOB_STATE_PENDING
    assert states_by_id[JobName.root("test-user", "job-2").to_wire()] == job_pb2.JOB_STATE_PENDING
    assert states_by_id[JobName.root("test-user", "job-3").to_wire()] == job_pb2.JOB_STATE_KILLED


def test_list_jobs_sql_pagination(service):
    """SQL-level pagination returns correct page when sorting by date."""
    for i in range(5):
        service.launch_job(make_job_request(f"job-{i}"), None)

    # Request page of 2
    request = controller_pb2.Controller.ListJobsRequest(query=controller_pb2.Controller.JobQuery(offset=0, limit=2))
    response = service.list_jobs(request, None)

    assert len(response.jobs) == 2
    assert response.total_count == 5
    assert response.has_more is True

    # Second page
    request2 = controller_pb2.Controller.ListJobsRequest(query=controller_pb2.Controller.JobQuery(offset=2, limit=2))
    response2 = service.list_jobs(request2, None)

    assert len(response2.jobs) == 2
    assert response2.total_count == 5
    assert response2.has_more is True

    # No overlap between pages
    page1_ids = {j.job_id for j in response.jobs}
    page2_ids = {j.job_id for j in response2.jobs}
    assert page1_ids.isdisjoint(page2_ids)

    # Last page
    request3 = controller_pb2.Controller.ListJobsRequest(query=controller_pb2.Controller.JobQuery(offset=4, limit=2))
    response3 = service.list_jobs(request3, None)

    assert len(response3.jobs) == 1
    assert response3.has_more is False


def test_list_jobs_state_filter(service):
    """SQL pagination respects state_filter."""
    service.launch_job(make_job_request("job-a"), None)
    service.launch_job(make_job_request("job-b"), None)
    service.terminate_job(
        controller_pb2.Controller.TerminateJobRequest(job_id=JobName.root("test-user", "job-b").to_wire()), None
    )

    # Filter to killed only
    request = controller_pb2.Controller.ListJobsRequest(
        query=controller_pb2.Controller.JobQuery(state_filter="killed", limit=10)
    )
    response = service.list_jobs(request, None)

    assert len(response.jobs) == 1
    assert response.jobs[0].state == job_pb2.JOB_STATE_KILLED


def test_list_jobs_name_filter(service):
    """Name filter returns only matching jobs."""
    service.launch_job(make_job_request("alpha-job"), None)
    service.launch_job(make_job_request("beta-job"), None)

    request = controller_pb2.Controller.ListJobsRequest(query=controller_pb2.Controller.JobQuery(name_filter="alpha"))
    response = service.list_jobs(request, None)

    assert len(response.jobs) == 1
    assert "alpha" in response.jobs[0].name.lower()


def test_list_jobs_all_scope_includes_descendants(service, state):
    """Legacy ListJobs behavior returns all jobs, including descendants."""
    service.launch_job(make_job_request("parent-job"), None)
    parent_id = JobName.root("test-user", "parent-job")
    child_id = JobName.from_wire(parent_id.to_wire() + "/child")
    child_req = controller_pb2.Controller.LaunchJobRequest(
        name=child_id.to_wire(),
        entrypoint=job_pb2.RuntimeEntrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
    )
    child_req.entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    state.submit_job(child_id, child_req, Timestamp.now())

    request = controller_pb2.Controller.ListJobsRequest()
    response = service.list_jobs(request, None)

    # Both parent and child should appear; total_count counts all matching jobs.
    job_ids = [j.job_id for j in response.jobs]
    assert parent_id.to_wire() in job_ids
    assert child_id.to_wire() in job_ids
    assert response.total_count == 2


def test_list_jobs_job_query_roots_and_children(service, state):
    """JobQuery supports roots-only and direct-children scopes."""
    service.launch_job(make_job_request("parent-job"), None)
    parent_id = JobName.root("test-user", "parent-job")
    child_id = JobName.from_wire(parent_id.to_wire() + "/child")
    child_req = controller_pb2.Controller.LaunchJobRequest(
        name=child_id.to_wire(),
        entrypoint=job_pb2.RuntimeEntrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
    )
    child_req.entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    state.submit_job(child_id, child_req, Timestamp.now())

    roots_response = service.list_jobs(
        controller_pb2.Controller.ListJobsRequest(
            query=controller_pb2.Controller.JobQuery(scope=controller_pb2.Controller.JOB_QUERY_SCOPE_ROOTS)
        ),
        None,
    )
    assert [job.job_id for job in roots_response.jobs] == [parent_id.to_wire()]
    assert roots_response.jobs[0].has_children is True

    children_response = service.list_jobs(
        controller_pb2.Controller.ListJobsRequest(
            query=controller_pb2.Controller.JobQuery(
                scope=controller_pb2.Controller.JOB_QUERY_SCOPE_CHILDREN,
                parent_job_id=parent_id.to_wire(),
            )
        ),
        None,
    )
    assert [job.job_id for job in children_response.jobs] == [child_id.to_wire()]


# =============================================================================
# SQL Aggregation Tests
# =============================================================================


def test_task_summaries_sql_group_by(state, service):
    """_task_summaries_for_jobs SQL GROUP BY produces correct aggregates."""
    from iris.cluster.controller.service import _task_summaries_for_jobs

    # Launch a job with 3 replicas
    service.launch_job(make_job_request("multi-task", replicas=3), None)

    job_id = JobName.root("test-user", "multi-task")
    summaries = _task_summaries_for_jobs(state._db, {job_id})

    assert job_id in summaries
    s = summaries[job_id]
    assert s.task_count == 3
    # All tasks should be pending
    assert s.task_state_counts.get(job_pb2.TASK_STATE_PENDING, 0) == 3
    assert s.completed_count == 0
    assert s.failure_count == 0
    assert s.preemption_count == 0


def test_live_user_stats_sql_aggregation(state, service):
    """_live_user_stats SQL GROUP BY produces correct per-user counts."""
    from iris.cluster.controller.service import _live_user_stats

    service.launch_job(make_job_request("job-x", replicas=2), None)
    service.launch_job(make_job_request("job-y"), None)

    stats_list = _live_user_stats(state._db)
    assert len(stats_list) >= 1

    user_stats = {s.user: s for s in stats_list}
    assert "test-user" in user_stats
    s = user_stats["test-user"]

    # 2 jobs
    total_jobs = sum(s.job_state_counts.values())
    assert total_jobs == 2

    # 3 tasks total (2 + 1)
    total_tasks = sum(s.task_state_counts.values())
    assert total_tasks == 3


def test_worker_addresses_for_tasks(state, service):
    """_worker_addresses_for_tasks fetches only referenced workers."""
    from iris.cluster.controller.service import _worker_addresses_for_tasks

    # Register workers
    _register_worker(state, WorkerId("w-1"))
    _register_worker(state, WorkerId("w-2"))
    _register_worker(state, WorkerId("w-3"))

    # Launch job and assign one task to w-1
    service.launch_job(make_job_request("assigned-job"), None)
    job_id = JobName.root("test-user", "assigned-job")
    task_id = JobName.from_wire(job_id.to_wire() + "/0")
    state.queue_assignments([Assignment(task_id=task_id, worker_id=WorkerId("w-1"))])

    # Get tasks with attempts
    tasks = _query_tasks_with_attempts(state, job_id)
    addresses = _worker_addresses_for_tasks(state._db, tasks)

    # Should only have w-1
    assert WorkerId("w-1") in addresses
    assert len(addresses) == 1


# =============================================================================
# Worker Tests
# =============================================================================


def test_list_workers_returns_all(service, state):
    """Verify list_workers returns all registered workers."""
    from iris.rpc.auth import _verified_identity, VerifiedIdentity

    db = state._db
    db.ensure_user("system:worker", Timestamp.now(), role="worker")
    token = _verified_identity.set(VerifiedIdentity(user_id="system:worker", role="worker"))
    try:
        for i in range(3):
            request = controller_pb2.Controller.RegisterRequest(
                address=f"host{i}:8080",
                metadata=make_worker_metadata(),
                worker_id=f"worker-{i}",
            )
            service.register(request, None)
    finally:
        _verified_identity.reset(token)

    request = controller_pb2.Controller.ListWorkersRequest()
    response = service.list_workers(request, None)

    assert len(response.workers) == 3

    # All workers should be healthy after registration
    for w in response.workers:
        assert w.healthy is True


# =============================================================================
# Constraint Injection Tests
# =============================================================================


def test_launch_job_injects_device_constraints_from_tpu_resource(service, state):
    """Job with TPU resource spec gets auto-injected device-type and device-variant constraints."""
    request = controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "tpu-job").to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
    )
    request.resources.device.CopyFrom(tpu_device("v5litepod-16"))

    service.launch_job(request, None)

    job = _query_job(state, JobName.root("test-user", "tpu-job"))
    stored_constraints = constraints_from_json(job.constraints_json)
    keys = {c.key for c in stored_constraints}
    assert WellKnownAttribute.DEVICE_TYPE in keys
    assert WellKnownAttribute.DEVICE_VARIANT in keys

    dt = next(c for c in stored_constraints if c.key == WellKnownAttribute.DEVICE_TYPE)
    assert dt.values[0].value == "tpu"
    dv = next(c for c in stored_constraints if c.key == WellKnownAttribute.DEVICE_VARIANT)
    assert dv.values[0].value == "v5litepod-16"


def test_launch_job_user_constraints_override_auto(service, state):
    """Explicit user constraints for canonical keys replace auto-generated ones."""
    user_variant = device_variant_constraint(["v5litepod-16", "v6e-16"])

    request = controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "multi-variant-job").to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
    )
    request.resources.device.CopyFrom(tpu_device("v5litepod-16"))
    request.constraints.append(user_variant.to_proto())

    service.launch_job(request, None)

    job = _query_job(state, JobName.root("test-user", "multi-variant-job"))
    stored_constraints = constraints_from_json(job.constraints_json)

    # device-variant should be the user's IN constraint, not the auto EQ
    dv_constraints = [c for c in stored_constraints if c.key == WellKnownAttribute.DEVICE_VARIANT]
    assert len(dv_constraints) == 1
    assert dv_constraints[0].op == ConstraintOp.IN

    # device-type should still be auto-injected
    dt_constraints = [c for c in stored_constraints if c.key == WellKnownAttribute.DEVICE_TYPE]
    assert len(dt_constraints) == 1
    assert dt_constraints[0].values[0].value == "tpu"


def test_launch_job_cpu_resource_no_constraints_injected(service, state):
    """CPU-only jobs get no auto-injected device constraints."""
    request = controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", "cpu-job").to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
    )

    service.launch_job(request, None)

    job = _query_job(state, JobName.root("test-user", "cpu-job"))
    assert len(constraints_from_json(job.constraints_json)) == 0


# =============================================================================
# Register Role-Gating Tests
# =============================================================================


def test_register_requires_worker_role(state, mock_controller, tmp_path):
    """Non-worker user gets PERMISSION_DENIED on register()."""
    from iris.cluster.bundle import BundleStore
    from iris.cluster.controller.auth import ControllerAuth
    from iris.rpc.auth import _verified_identity, VerifiedIdentity

    db = state._db
    now = Timestamp.now()
    db.ensure_user("alice", now, role="user")

    auth = ControllerAuth(provider="static")
    service = ControllerServiceImpl(
        state,
        state._store,
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=LogServiceImpl(),
        auth=auth,
    )

    token = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        with pytest.raises(ConnectError) as exc_info:
            service.register(
                controller_pb2.Controller.RegisterRequest(
                    worker_id="w-1",
                    address="localhost:8080",
                    metadata=make_worker_metadata(),
                ),
                None,
            )
        assert exc_info.value.code == Code.PERMISSION_DENIED
    finally:
        _verified_identity.reset(token)


def test_register_allows_worker_role(state, mock_controller, tmp_path):
    """Worker-role user can call register()."""
    from iris.cluster.bundle import BundleStore
    from iris.cluster.controller.auth import ControllerAuth
    from iris.rpc.auth import _verified_identity, VerifiedIdentity

    db = state._db
    now = Timestamp.now()
    db.ensure_user("system:worker", now, role="worker")

    auth = ControllerAuth(provider="static")
    service = ControllerServiceImpl(
        state,
        state._store,
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=LogServiceImpl(),
        auth=auth,
    )

    token = _verified_identity.set(VerifiedIdentity(user_id="system:worker", role="worker"))
    try:
        resp = service.register(
            controller_pb2.Controller.RegisterRequest(
                worker_id="w-1",
                address="localhost:8080",
                metadata=make_worker_metadata(),
            ),
            None,
        )
        assert resp.accepted
    finally:
        _verified_identity.reset(token)


def test_get_scheduler_state_with_running_task(controller_service, state):
    """get_scheduler_state must not crash when there are running tasks.

    Regression: JobName has no `.job` attribute — the code must use `.parent`.
    """
    from iris.rpc.auth import VerifiedIdentity, _verified_identity

    # Submit a job and move a task to RUNNING
    job_id = JobName.root("alice", "sched-test")
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_id.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    state.submit_job(job_id, request, Timestamp.now())

    w1 = WorkerId("w1")
    state.register_or_refresh_worker(
        worker_id=w1,
        address="w1:8080",
        metadata=job_pb2.WorkerMetadata(
            hostname="w1",
            ip_address="127.0.0.1",
            cpu_count=8,
            memory_bytes=16 * 1024**3,
            disk_bytes=100 * 1024**3,
        ),
        ts=Timestamp.now(),
    )
    task_id = job_id.task(0)
    _assign_and_transition(state, task_id, w1, job_pb2.TASK_STATE_RUNNING)

    token = _verified_identity.set(VerifiedIdentity(user_id="alice", role="user"))
    try:
        resp = controller_service.get_scheduler_state(
            controller_pb2.Controller.GetSchedulerStateRequest(),
            None,
        )
        assert resp.total_running == 1
        assert resp.running_tasks[0].job_id == job_id.to_wire()
        assert resp.running_tasks[0].user_id == "alice"
        # alice has no explicit user_budgets row but has an active task — the
        # scheduler state must report her spend using UserBudgetDefaults so the
        # dashboard renders Spent/Limit/Utilization instead of '-'.
        alice_budget = next((b for b in resp.user_budgets if b.user_id == "alice"), None)
        assert alice_budget is not None
        assert alice_budget.budget_spent > 0
        assert alice_budget.budget_limit == controller_service._user_budget_defaults.budget_limit
    finally:
        _verified_identity.reset(token)


# =============================================================================
# Client freshness tests
# =============================================================================


# Fixed reference point for helper unit tests so freshness behavior is
# reproducible regardless of wall-clock date. Pick something far enough in the
# future that FEATURE_INTRODUCTION_DATE has aged out for the past-grace test.
_REF_NOW = date(2026, 6, 1)


def test_check_client_freshness_accepts_today():
    """A client built today is inside the window (upper edge)."""
    _check_client_freshness(_REF_NOW.isoformat(), _REF_NOW)


def test_check_client_freshness_accepts_at_window_edge():
    """A client exactly FRESHNESS_WINDOW old is still accepted (lower edge)."""
    edge = _REF_NOW - FRESHNESS_WINDOW
    _check_client_freshness(edge.isoformat(), _REF_NOW)


def test_check_client_freshness_rejects_over_window():
    """A client one day past the window is rejected."""
    stale = _REF_NOW - FRESHNESS_WINDOW - timedelta(days=1)
    with pytest.raises(ConnectError) as exc_info:
        _check_client_freshness(stale.isoformat(), _REF_NOW)
    assert exc_info.value.code == Code.FAILED_PRECONDITION
    assert stale.isoformat() in exc_info.value.message


def test_check_client_freshness_empty_is_introduction_date():
    """Empty string is substituted with FEATURE_INTRODUCTION_DATE."""
    # Right at ship time: empty clients still inside window, succeed.
    _check_client_freshness("", FEATURE_INTRODUCTION_DATE)
    # Well past the grace period: empty clients fail.
    well_past = FEATURE_INTRODUCTION_DATE + FRESHNESS_WINDOW + timedelta(days=1)
    with pytest.raises(ConnectError) as exc_info:
        _check_client_freshness("", well_past)
    assert exc_info.value.code == Code.FAILED_PRECONDITION


def test_check_client_freshness_rejects_malformed():
    """Non-ISO strings are rejected as INVALID_ARGUMENT."""
    with pytest.raises(ConnectError) as exc_info:
        _check_client_freshness("not-a-date", _REF_NOW)
    assert exc_info.value.code == Code.INVALID_ARGUMENT


def test_launch_job_root_with_fresh_client_date(service):
    """Root submission with today's date succeeds end-to-end through launch_job."""
    request = make_job_request("fresh-client")
    request.client_revision_date = date.today().isoformat()
    response = service.launch_job(request, None)
    assert response.job_id == JobName.root("test-user", "fresh-client").to_wire()


def test_launch_job_root_with_stale_client_date(service):
    """Root submission with an ancient date is rejected end-to-end."""
    request = make_job_request("stale-client")
    request.client_revision_date = "2000-01-01"
    with pytest.raises(ConnectError) as exc_info:
        service.launch_job(request, None)
    assert exc_info.value.code == Code.FAILED_PRECONDITION


def test_launch_job_nested_with_stale_client_date_is_exempt(service):
    """Nested submissions bypass the freshness check (parent already running)."""
    service.launch_job(make_job_request("parent-job"), None)
    parent_id = JobName.root("test-user", "parent-job")

    child_id = JobName.from_wire(parent_id.to_wire() + "/child")
    assert not child_id.is_root
    child_req = controller_pb2.Controller.LaunchJobRequest(
        name=child_id.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
    )
    child_req.client_revision_date = "2000-01-01"

    response = service.launch_job(child_req, None)
    assert response.job_id == child_id.to_wire()
