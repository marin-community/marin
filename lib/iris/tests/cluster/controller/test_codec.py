# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fidelity of the stored-job -> LaunchJobRequest reconstruction.

A queued federated handoff is rebuilt from the parent's stored job state and delivered
to the peer, so a request field this round trip drops is a field the peer never runs
with. Two federation outages came from exactly that: a dropped ``client_revision_date``
(the peer's freshness gate rejected every handoff) and dropped inline ``workdir_files``
(the peer ran a ``from_callable`` task with no ``_callable_runner.py``).
"""

from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.controller import ops, reads
from iris.cluster.controller.codec import reconstruct_launch_job_request
from iris.cluster.types import JobName, tpu_device
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp

# The only LaunchJobRequest fields the job store deliberately does not keep.
NOT_PERSISTED = {
    # Uploaded to the bundle store at submit; survives as ``bundle_id``.
    "bundle_blob",
    # Describes the client of the hop that submitted the request: the parent gates the
    # user's CLI build at submit, and a peer exempts a received handoff (whose wire
    # client is the parent controller, not a CLI).
    "client_revision_date",
    # Stamped by the federation manager onto the request it delivers to a peer.
    "federation",
}


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
    """Every field of a submitted request comes back out of the job store.

    The reconstruction rebuilds the request field by field, so it silently drifts from
    the proto whenever a field is added. Adding one fails this test until it is either
    persisted or named (with its reason) in ``NOT_PERSISTED``.
    """
    job_id = JobName.root("test-user", "codec-fidelity")
    request = _fully_populated_request(job_id)

    unset = {f.name for f in request.DESCRIPTOR.fields} - {f.name for f, _ in request.ListFields()}
    assert not unset, f"_fully_populated_request must set every field; missing {sorted(unset)}"

    with state._db.transaction() as cur:
        ops.job.insert_job_and_config(cur, job_id=job_id, request=request, ts=Timestamp.now())

    with state._db.read_snapshot() as tx:
        job = reads.get_job_detail(tx, job_id)
        reconstructed = reconstruct_launch_job_request(job, workdir_files=reads.get_workdir_files(tx, job_id))

    expected = controller_pb2.Controller.LaunchJobRequest()
    expected.CopyFrom(request)
    for field in NOT_PERSISTED:
        expected.ClearField(field)
    assert reconstructed == expected
