# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-daemon actions for Iris product journeys."""

from dataclasses import dataclass, field
from pathlib import Path

from rigging.timing import Duration, Timestamp

from iris.cluster.backends.rpc.backend import RpcTaskBackend
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.types import DEFAULT_BACKEND_ID, JobName
from iris.managed_thread import ThreadContainer
from iris.rpc import controller_pb2, job_pb2, worker_pb2


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
    """Drive worker lifecycle stories through Controller and worker RPC APIs."""

    def __init__(self, root: Path, monkeypatch) -> None:
        self.clock = WorkerJourneyClock()
        monkeypatch.setattr(Timestamp, "now", classmethod(lambda cls: self.clock.now()))
        self.fleet = WorkerFleet()
        self.backend = RpcTaskBackend(
            stub_factory=self.fleet,
            unreachable_grace=Duration.from_ms(100),
        )
        state_dir = root / "controller"
        config = ControllerConfig(
            cluster_id="worker-journey",
            remote_state_dir=f"file://{root / 'remote'}",
            local_state_dir=state_dir,
        )
        self.controller = Controller(
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
        response = self.controller.register_worker(
            controller_pb2.Controller.RegisterRequest(
                worker_id=worker_id,
                address=address,
                metadata=_worker_metadata(cpu_millicores=cpu_millicores),
            )
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
        priority_band: int = job_pb2.PRIORITY_BAND_BATCH,
        preemption_retries: int = 1,
    ) -> WorkerJob:
        entrypoint = job_pb2.RuntimeEntrypoint()
        entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
        response = self.controller.launch_job(
            controller_pb2.Controller.LaunchJobRequest(
                name=JobName.root("journey", name).to_wire(),
                entrypoint=entrypoint,
                environment=job_pb2.EnvironmentConfig(),
                resources=job_pb2.ResourceSpecProto(
                    cpu_millicores=cpu_millicores,
                    memory_bytes=1024**3,
                ),
                replicas=1,
                max_retries_preemption=preemption_retries,
                priority_band=priority_band,
            )
        )
        return WorkerJob(response.job_id)

    def preempt(self, job: WorkerJob) -> None:
        response = self.controller.kick_tasks(
            controller_pb2.Controller.KickTasksRequest(
                targets=[job.task_id],
                desired_state=job_pb2.TASK_STATE_PREEMPTED,
                reason="journey preemption",
            )
        )
        if len(response.results) != 1 or not response.results[0].queued:
            raise AssertionError(f"preemption was not queued: {response}")

    def step(self) -> None:
        self.controller.run_control_tick()

    def advance(self, seconds: float) -> None:
        self.clock.advance(seconds)

    def run_until_task_state(self, job: WorkerJob, state: int, *, max_ticks: int = 12) -> None:
        for _ in range(max_ticks):
            if self.task(job).state == state:
                return
            self.step()
        raise AssertionError(
            f"{job.task_id} did not reach {job_pb2.TaskState.Name(state)}; "
            f"last={job_pb2.TaskState.Name(self.task(job).state)}"
        )

    def run_until_worker_releases_task(self, worker_id: str, job: WorkerJob, *, max_ticks: int = 6) -> None:
        for _ in range(max_ticks):
            if self.task(job).state == job_pb2.TASK_STATE_PENDING and worker_id not in self.worker_ids():
                return
            self.step()
        raise AssertionError(f"{worker_id} still owns {job.task_id} after {max_ticks} control ticks")

    def task(self, job: WorkerJob) -> job_pb2.TaskStatus:
        return self.controller.get_task_status(job.task_id).task

    def worker(self, worker_id: str) -> controller_pb2.Controller.WorkerHealthStatus:
        return self.controller.get_worker_status(worker_id).worker

    def worker_ids(self) -> set[str]:
        return {worker.worker_id for worker in self.controller.list_workers().workers}
