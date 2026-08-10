# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fidelity of the persisted Job resource specification.

A queued federated handoff is rebuilt from the parent's stored job state and delivered
to the peer, so a request field this round trip drops is a field the peer never runs
with. Two federation outages came from exactly that: a dropped ``client_revision_date``
(the peer's freshness gate rejected every handoff) and dropped inline ``workdir_files``
(the peer ran a ``from_callable`` task with no ``_callable_runner.py``).
"""

import json
from dataclasses import replace
from datetime import date

from iris.cluster.bundle import BundleStore
from iris.cluster.config import BackendConfig
from iris.cluster.constraints import Constraint, ConstraintMode, ConstraintOp
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.controller import CapabilityUrlConfig, Controller
from iris.cluster.controller.endpoint_registry import EndpointRegistry
from iris.cluster.controller.operations import OperationalServices
from iris.cluster.controller.persistence import operations as ops
from iris.cluster.controller.persistence import reads
from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.persistence.json_codec import reconstruct_job_spec
from iris.cluster.controller.persistence.schema import job_config_table
from iris.cluster.types import UserBudgetDefaults
from iris.resources.execution import (
    CommandEntrypoint,
    Environment,
    GpuDevice,
    ResourceSpec,
    RuntimeEntrypoint,
    tpu_device,
)
from iris.resources.identity import ResourceKey, ResourceKind
from iris.resources.job import (
    ContainerProfile,
    CoschedulingConfig,
    ExistingJobPolicy,
    JobPreemptionPolicy,
    JobSpec,
    PriorityBand,
)
from iris.resources.names import JobName
from iris.rpc import controller_pb2, job_pb2, resource_pb2
from iris.rpc.endpoint_service import EndpointServiceImpl
from iris.rpc.legacy.controller_service import LegacyControllerService
from iris.rpc.legacy.job_codec import (
    constraint_to_proto,
    device_to_proto,
)
from iris.rpc.legacy.job_codec import (
    device_from_proto as legacy_device_from_proto,
)
from iris.rpc.legacy.job_codec import (
    resource_spec_from_proto as legacy_resource_spec_from_proto,
)
from iris.rpc.legacy.job_service_codec import job_spec_from_legacy_request, job_spec_to_legacy_request
from iris.rpc.resource_codec import (
    device_from_proto,
    job_spec_from_proto,
    job_spec_to_proto,
    resource_spec_from_proto,
)
from rigging.timing import Duration, Timestamp
from sqlalchemy import select, update
from tests.cluster.controller._test_support import ControllerTestState


def _fully_populated_request(job_id: JobName) -> controller_pb2.Controller.LaunchJobRequest:
    """A LaunchJobRequest with every field set to a non-default value."""
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_id.to_wire(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=2000,
            memory_bytes=8 * 1024**3,
            disk_bytes=64 * 1024**3,
            device=device_to_proto(tpu_device("v6e-8")),
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
    request.constraints.append(
        constraint_to_proto(Constraint.create(key="device-variant", op=ConstraintOp.EQ, value="v6e-8"))
    )
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
    assert job.res_device_json == '{"tpu": {"variant": "v6e-8", "topology": "", "count": 8}}'


def _controller_boundaries(
    db: ControllerDB,
    mock_controller,
    tmp_path,
    log_client,
    *,
    initialize_projections: bool = True,
) -> tuple[Controller, LegacyControllerService]:
    if initialize_projections:
        ControllerTestState(db)
    bundle_store = BundleStore(storage_dir=str(tmp_path / "bundles"))
    endpoint_service = EndpointServiceImpl(EndpointRegistry(db=db))
    resources = Controller(
        cluster_id="test",
        db=db,
        runtime=mock_controller,
        bundle_store=bundle_store,
        endpoint_registry=endpoint_service.registry,
        auth=ControllerAuth(),
        user_budget_defaults=UserBudgetDefaults(),
        capability_url_config=CapabilityUrlConfig(cluster_name="test"),
        backends=mock_controller.backends,
        backend_configs={backend_id: BackendConfig(kind="worker_daemon") for backend_id in mock_controller.backends},
    )
    legacy = LegacyControllerService(
        runtime=mock_controller,
        bundle_store=bundle_store,
        log_client=log_client,
        operations=OperationalServices.from_database(db),
        endpoint_service=endpoint_service,
        controller=resources,
    )
    return resources, legacy


def test_native_job_spec_survives_public_and_legacy_reopen(tmp_path, mock_controller, log_client) -> None:
    setup_commands = ["prepare"]
    argv = ["python", "train.py"]
    workdir_files = {"config.json": b'{"batch": 8}'}
    workdir_file_refs = {"z-last.bin": "z" * 64, "a-first.bin": "a" * 64}
    env_vars = {"ZED": "last", "EMPTY": "", "MODE": "train"}
    setup_scripts = ["install"]
    ports = ["http"]
    submit_argv = ["iris", "job", "run"]
    constraints = [
        Constraint.create(
            key="ordinal",
            op=ConstraintOp.EQ,
            value=2**63 - 1,
            mode=ConstraintMode.PREFERRED,
        )
    ]
    spec = JobSpec(
        version=1,
        name="/alice/native-fidelity",
        entrypoint=RuntimeEntrypoint(
            setup_commands,
            CommandEntrypoint(argv),
            workdir_files,
            workdir_file_refs,
        ),
        resources=ResourceSpec(cpu=0.125, memory=1024, disk=2048),
        environment=Environment(env_vars, setup_scripts),
        bundle_id="bundle-1",
        scheduling_timeout=Duration.from_seconds(3),
        ports=ports,
        max_task_failures=4,
        max_retries_failure=2,
        max_retries_preemption=7,
        constraints=constraints,
        coscheduling=CoschedulingConfig(group_by="rack"),
        replicas=2,
        timeout=Duration.from_seconds(30),
        fail_if_exists=True,
        preemption_policy=JobPreemptionPolicy.PRESERVE_CHILDREN,
        existing_job_policy=ExistingJobPolicy.ERROR,
        priority_band=PriorityBand.INTERACTIVE,
        task_image="task@sha256:" + "b" * 64,
        submit_argv=submit_argv,
        client_revision_date=date.today().isoformat(),
        container_profile=ContainerProfile.DEFAULT,
    )

    setup_commands.append("changed")
    argv.append("changed")
    workdir_files["config.json"] = b"changed"
    workdir_file_refs["z-last.bin"] = "changed"
    env_vars["MODE"] = "changed"
    setup_scripts.append("changed")
    ports.append("changed")
    submit_argv.append("changed")
    constraints.clear()

    assert job_spec_from_proto(job_spec_to_proto(spec)) == spec
    assert job_spec_from_legacy_request(job_spec_to_legacy_request(spec)) == spec

    db_dir = tmp_path / "db"
    db = ControllerDB(db_dir)
    try:
        resources, legacy = _controller_boundaries(db, mock_controller, tmp_path, log_client)
        response = legacy.launch_job(job_spec_to_legacy_request(spec), None)
        identity = resources.list_jobs().items[0].identity
        assert response.job_id == identity.key.resource_id
    finally:
        db.close()

    reopened = ControllerDB(db_dir)
    try:
        resources, legacy = _controller_boundaries(reopened, mock_controller, tmp_path, log_client)
        detail = resources.describe_job(identity.key)
        legacy_detail = legacy.get_job_status(
            controller_pb2.Controller.GetJobStatusRequest(job_id=identity.key.resource_id),
            None,
        )
        with reopened.read_snapshot() as tx:
            stored = tx.execute(
                select(
                    job_config_table.c.constraints_json,
                    job_config_table.c.entrypoint_json,
                    job_config_table.c.environment_json,
                ).where(job_config_table.c.job_id == JobName.from_wire(spec.name))
            ).one()
    finally:
        reopened.close()

    assert detail.spec == replace(spec, client_revision_date="")
    assert job_spec_from_legacy_request(legacy_detail.request) == detail.spec
    assert stored.constraints_json == ('[{"key": "ordinal", "value": {"int_value": "9223372036854775807"}, "mode": 1}]')
    assert stored.entrypoint_json == (
        '{"setup_commands": ["prepare"], "run_command": {"argv": ["python", "train.py"]}, '
        '"workdir_file_refs": {"a-first.bin": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", '
        '"z-last.bin": "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"}}'
    )
    assert stored.environment_json == (
        '{"env_vars": {"EMPTY": "", "MODE": "train", "ZED": "last"}, "setup_scripts": ["install"]}'
    )


def test_omitted_gpu_count_defaults_once_across_wires_and_storage(state, mock_controller, tmp_path, log_client) -> None:
    resource_device = device_from_proto(resource_pb2.DeviceConfig(gpu=resource_pb2.GpuDevice(variant="H100")))
    legacy_device = legacy_device_from_proto(job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant="H100")))
    assert resource_device == legacy_device == GpuDevice(variant="H100", count=1)

    job_id = JobName.root("test-user", "gpu-default")
    spec = replace(
        job_spec_from_legacy_request(_fully_populated_request(job_id)),
        resources=ResourceSpec(device=resource_device),
    )
    with state._db.transaction() as tx:
        ops.job.insert_job_and_config(
            tx,
            job_id=job_id,
            spec=spec,
            ts=Timestamp.now(),
            priority_band=int(spec.priority_band),
        )
        tx.execute(
            update(job_config_table)
            .where(job_config_table.c.job_id == job_id)
            .values(res_device_json='{"gpu":{"variant":"H100"}}')
        )
    resources, _ = _controller_boundaries(
        state._db,
        mock_controller,
        tmp_path,
        log_client,
        initialize_projections=False,
    )
    detail = resources.describe_job(ResourceKey("test", ResourceKind.JOB, job_id.to_wire()))
    with state._db.read_snapshot() as tx:
        row = reads.get_job_detail(tx, job_id)
        assert row is not None
        assert json.loads(row.res_device_json) == {"gpu": {"variant": "H100"}}
        reconstructed = reconstruct_job_spec(row, workdir_files=reads.get_workdir_files(tx, job_id))

    assert detail.spec.resources.device == GpuDevice(variant="H100", count=1)
    assert reconstructed.resources.device == GpuDevice(variant="H100", count=1)


def test_resource_spec_with_present_empty_device_decodes_device_less_across_wires() -> None:
    legacy = job_pb2.ResourceSpecProto(cpu_millicores=2_000)
    legacy.device.SetInParent()
    resource = resource_pb2.ResourceSpecProto(cpu_millicores=2_000)
    resource.device.SetInParent()

    assert legacy_resource_spec_from_proto(legacy) == ResourceSpec(cpu=2.0)
    assert resource_spec_from_proto(resource) == ResourceSpec(cpu=2.0)
