# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-daemon actions for Iris product journeys."""

from dataclasses import dataclass, field
from pathlib import Path

from iris.backends.rpc.backend import RpcTaskBackend
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.composition import compose_controller_process
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.runtime import ControllerConfig
from iris.cluster.types import DEFAULT_BACKEND_ID
from iris.managed_thread import ThreadContainer
from iris.resources.execution import CommandEntrypoint, Environment, ResourceSpec, RuntimeEntrypoint
from iris.resources.identity import NodeIdentity, NodeLocator, ResourceKey, ResourceKind
from iris.resources.job import (
    ContainerProfile,
    ExistingJobPolicy,
    JobPreemptionPolicy,
    JobSpec,
    PriorityBand,
)
from iris.resources.names import JobName
from iris.resources.node import NodeDetail, NodeQuery
from iris.resources.state import TaskState
from iris.resources.task import TaskDetail
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.rpc.worker_client import RpcWorkerClient
from rigging.timing import Duration, Timestamp


@dataclass(slots=True)
class WorkerJourneyClock:
    epoch_ms: int = 1_704_067_200_000

    def now(self) -> Timestamp:
        current = Timestamp.from_ms(self.epoch_ms)
        self.epoch_ms += 1
        return current

    def advance(self, seconds: float) -> None:
        self.epoch_ms += int(seconds * 1000)


@dataclass(frozen=True, slots=True)
class WorkerJob:
    job_id: str

    @property
    def task_id(self) -> str:
        return JobName.from_wire(self.job_id).task(0).to_wire()


@dataclass(slots=True)
class WorkerDaemon:
    """An execution process observed only through the worker RPC boundary."""

    worker_id: str
    reachable: bool = True
    failures_remaining: int = 0
    acknowledge_stops: bool = True
    queued_observations: list[worker_pb2.Worker.AttemptObservation] = field(default_factory=list)
    delivered_observation_uids: set[str] = field(default_factory=set)

    def fail_next_reconciles(self, count: int) -> None:
        self.failures_remaining += count

    def queue_observation(self, attempt_uid: str, state: int) -> None:
        self.queued_observations.append(worker_pb2.Worker.AttemptObservation(attempt_uid=attempt_uid, state=state))

    def reconcile(self, request: worker_pb2.Worker.ReconcileRequest) -> worker_pb2.Worker.ReconcileResponse:
        if not self.reachable or self.failures_remaining:
            self.failures_remaining = max(0, self.failures_remaining - 1)
            raise ConnectionError(f"worker {self.worker_id} is unreachable")

        observations = self.queued_observations
        self.queued_observations = []
        self.delivered_observation_uids.update(observation.attempt_uid for observation in observations)
        for desired in request.desired:
            if desired.HasField("run"):
                observations.append(
                    worker_pb2.Worker.AttemptObservation(
                        attempt_uid=desired.attempt_uid,
                        state=job_pb2.TASK_STATE_RUNNING,
                    )
                )
                continue

            if self.acknowledge_stops:
                observations.append(
                    worker_pb2.Worker.AttemptObservation(
                        attempt_uid=desired.attempt_uid,
                        state=job_pb2.TASK_STATE_KILLED,
                    )
                )

        return worker_pb2.Worker.ReconcileResponse(
            worker_id=self.worker_id,
            observed=observations,
            health=worker_pb2.Worker.WorkerHealth(healthy=True),
        )


@dataclass(slots=True)
class _AddressWorkerStub:
    address: str
    fleet: "WorkerFleet"

    async def reconcile(
        self,
        request: worker_pb2.Worker.ReconcileRequest,
        *,
        timeout_ms: int | None = None,
    ) -> worker_pb2.Worker.ReconcileResponse:
        del timeout_ms
        return self.fleet.daemons[self.address].reconcile(request)


@dataclass(slots=True)
class WorkerFleet:
    """Address resolver standing in for the external worker transport."""

    daemons: dict[str, WorkerDaemon] = field(default_factory=dict)

    def attach(self, address: str, worker_id: str) -> WorkerDaemon:
        daemon = WorkerDaemon(worker_id)
        self.daemons[address] = daemon
        return daemon

    def get_stub(self, address: str) -> _AddressWorkerStub:
        return _AddressWorkerStub(address, self)

    def evict(self, address: str) -> None:
        return None

    def close(self) -> None:
        return None


def _worker_metadata(*, cpu_millicores: int) -> job_pb2.WorkerMetadata:
    metadata = job_pb2.WorkerMetadata(
        hostname="journey-worker",
        ip_address="127.0.0.1",
        cpu_count=max(1, cpu_millicores // 1000),
        memory_bytes=8 * 1024**3,
        disk_bytes=8 * 1024**3,
        device=job_pb2.DeviceConfig(cpu=job_pb2.CpuDevice(variant="cpu")),
    )
    metadata.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "cpu"
    return metadata


class WorkerJourney:
    """Drive worker lifecycle stories through ControllerRuntime and worker RPC APIs."""

    def __init__(self, root: Path, monkeypatch) -> None:
        self.clock = WorkerJourneyClock()
        monkeypatch.setattr(Timestamp, "now", classmethod(lambda cls: self.clock.now()))
        self.fleet = WorkerFleet()
        self.backend = RpcTaskBackend(
            worker_client=RpcWorkerClient(self.fleet),
            unreachable_grace=Duration.from_ms(100),
        )
        state_dir = root / "controller"
        config = ControllerConfig(
            cluster_id="worker-journey",
            remote_state_dir=f"file://{root / 'remote'}",
            local_state_dir=state_dir,
        )
        self.controller = compose_controller_process(
            config=config,
            backends={DEFAULT_BACKEND_ID: self.backend},
            log_stack=build_log_stack(
                log_service_address="",
                local_log_dir=state_dir / "log-server",
                host=config.host,
                worker_token=None,
            ),
            threads=ThreadContainer(name="worker-journey"),
            db=ControllerDB(db_dir=root / "db"),
        )

    def close(self) -> None:
        self.controller.stop()

    def add_worker(
        self,
        worker_id: str,
        address: str,
        *,
        cpu_millicores: int = 1000,
    ) -> WorkerDaemon:
        existing = self.fleet.daemons.get(address)
        daemon = (
            existing
            if existing is not None and existing.worker_id == worker_id
            else self.fleet.attach(address, worker_id)
        )
        response = self.controller.controller_service.register(
            controller_pb2.Controller.RegisterRequest(
                worker_id=worker_id,
                address=address,
                metadata=_worker_metadata(cpu_millicores=cpu_millicores),
            ),
            None,
        )
        if not response.accepted:
            raise AssertionError(f"worker registration rejected: {worker_id}")
        return daemon

    def replace_daemon(self, address: str, worker_id: str) -> WorkerDaemon:
        return self.fleet.attach(address, worker_id)

    def submit(
        self,
        name: str,
        *,
        cpu_millicores: int = 1000,
        priority_band: int = PriorityBand.BATCH,
        preemption_retries: int = 1,
    ) -> WorkerJob:
        entrypoint = RuntimeEntrypoint((), CommandEntrypoint(("python", "-c", "pass")), {}, {})
        identity = self.controller.controller.submit_job(
            JobSpec(
                version=1,
                name=JobName.root("journey", name).to_wire(),
                entrypoint=entrypoint,
                resources=ResourceSpec(cpu=cpu_millicores / 1_000, memory=1024**3),
                environment=Environment({}, ()),
                bundle_id="",
                scheduling_timeout=None,
                ports=(),
                max_task_failures=0,
                max_retries_failure=0,
                max_retries_preemption=preemption_retries,
                constraints=(),
                coscheduling=None,
                replicas=1,
                timeout=None,
                fail_if_exists=False,
                preemption_policy=JobPreemptionPolicy.UNSPECIFIED,
                existing_job_policy=ExistingJobPolicy.UNSPECIFIED,
                priority_band=PriorityBand(priority_band),
                task_image="",
                submit_argv=(),
                client_revision_date="",
                container_profile=ContainerProfile.UNSPECIFIED,
            ),
            enforce_client_freshness=False,
        )
        return WorkerJob(identity.key.resource_id)

    def preempt(self, job: WorkerJob) -> None:
        task = self.task(job)
        current = task.summary.current_attempt
        if current is None:
            raise AssertionError(f"{job.task_id} has no current Attempt")
        self.controller.controller.retry_task(
            task.summary.identity,
            expected_attempt_uid=current.attempt_uid,
            idempotency_key=f"worker-journey-preempt:{current.attempt_uid}",
        )

    def step(self) -> None:
        self.controller.runtime.run_control_tick()

    def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)

    def run_until_task_state(self, job: WorkerJob, state: int, *, max_ticks: int = 12) -> None:
        for _ in range(max_ticks):
            if self.task(job).summary.state == state:
                return
            self.step()
        raise AssertionError(
            f"{job.task_id} did not reach {TaskState(state).name}; " f"last={self.task(job).summary.state.name}"
        )

    def run_until_worker_releases_task(self, worker_id: str, job: WorkerJob, *, max_ticks: int = 6) -> None:
        for _ in range(max_ticks):
            if self.task(job).summary.state is TaskState.PENDING and worker_id not in self.worker_ids():
                return
            self.step()
        raise AssertionError(f"{worker_id} still owns {job.task_id} after {max_ticks} control ticks")

    def task(self, job: WorkerJob) -> TaskDetail:
        return self.controller.controller.describe_task(
            ResourceKey(self.controller.controller.cluster_id, ResourceKind.TASK, job.task_id)
        )

    def worker(self, worker_id: str) -> NodeDetail:
        page = self.controller.controller.list_nodes(NodeQuery(contains=worker_id, page_size=100))
        summary = next(node for node in page.items if node.identity.key.resource_id == worker_id)
        return self.controller.controller.describe_node(
            NodeLocator(summary.identity.key, summary.identity.backend_id, summary.identity.node_uid)
        )

    def node(self, identity: NodeIdentity) -> NodeDetail:
        return self.controller.controller.describe_node(
            NodeLocator(identity.key, identity.backend_id, identity.node_uid)
        )

    def worker_ids(self) -> set[str]:
        return {
            node.identity.key.resource_id
            for node in self.controller.controller.list_nodes(NodeQuery(page_size=100)).items
        }
