# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fidelity of the persisted Job resource specification.

A queued federated handoff is rebuilt from the parent's stored job state and delivered
to the peer, so a request field this round trip drops is a field the peer never runs
with. Two federation outages came from exactly that: a dropped ``client_revision_date``
(the peer's freshness gate rejected every handoff) and dropped inline ``workdir_files``
(the peer ran a ``from_callable`` task with no ``_callable_runner.py``).
"""

from dataclasses import replace

from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.controller import ops, reads
from iris.cluster.controller.codec import reconstruct_job_spec
from iris.cluster.controller.resources.legacy_rpc import job_spec_from_legacy_request, job_spec_to_legacy_request
from iris.cluster.types import JobName, tpu_device
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp


def _fully_populated_request(job_id: JobName) -> controller_pb2.Controller.LaunchJobRequest:
    """A LaunchJobRequest with every field set to a non-default value."""
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_id.to_wire(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=2000, memory_bytes=8 * 1024**3, disk_bytes=64 * 1024**3, device=tpu_device("v6e-8")
        ),
        environment=job_pb2.EnvironmentConfig(env_vars={"LOG_LEVEL": "info"}),
        bundle_id="a" * 64,
        bundle_blob=b"PK\x03\x04",
        ports=["http"],
        max_task_failures=3,
        max_retries_failure=2,
        max_retries_preemption=7,
        replicas=2,
        fail_if_exists=True,
        preemption_policy=job_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN,
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_RECREATE,
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        task_image="custom/image:dev",
        submit_argv=["iris", "job", "run", "--", "python", "train.py"],
        client_revision_date="2026-07-12",
        container_profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED,
    )
    request.entrypoint.setup_commands.append("uv sync")
    request.entrypoint.run_command.argv[:] = ["python", "train.py"]
    request.entrypoint.workdir_files["_callable_runner.py"] = b"import pickle"
    request.entrypoint.workdir_file_refs["_callable.pkl"] = "b" * 64
    request.constraints.append(Constraint.create(key="device-variant", op=ConstraintOp.EQ, value="v6e-8").to_proto())
    request.coscheduling.group_by = "tpu-name"
    request.scheduling_timeout.milliseconds = 60_000
    request.timeout.milliseconds = 3_600_000
    request.federation.requester_id = "parent"
    return request


def test_every_launch_request_field_survives_storage(state):
    """A typed specification survives persistence and reconstruction."""
    job_id = JobName.root("test-user", "codec-fidelity")
    request = _fully_populated_request(job_id)

    spec = job_spec_from_legacy_request(request)
    assert job_spec_to_legacy_request(spec).resources == request.resources

    with state._db.transaction() as cur:
        ops.job.insert_job_and_config(
            cur,
            job_id=job_id,
            spec=spec,
            ts=Timestamp.now(),
            priority_band=int(request.priority_band),
        )

    with state._db.read_snapshot() as tx:
        job = reads.get_job_detail(tx, job_id)
        reconstructed = reconstruct_job_spec(job, workdir_files=reads.get_workdir_files(tx, job_id))

    assert reconstructed == replace(spec, client_revision_date="")
