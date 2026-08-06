# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed domain actions and public observations for Iris journeys."""

import contextlib
from dataclasses import dataclass
from pathlib import Path

from iris.cluster.config import PeerConfig
from iris.cluster.controller.checkpoint import CheckpointResult, download_checkpoint_to_local
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.controller.transition_reader import DbTransitionReader
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.types import DEFAULT_BACKEND_ID, JobName
from iris.managed_thread import ThreadContainer
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp
from tests.journeys.backend import BackendEvent, ScriptedObservation, ScriptedTaskBackend, UnavailableTaskBackend


@dataclass(frozen=True, slots=True)
class TaskRef:
    wire_id: str


@dataclass(frozen=True, slots=True)
class JobRef:
    wire_id: str
    tasks: int
    coscheduled: bool = False

    def __getitem__(self, task_index: int) -> TaskRef:
        if task_index < 0 or task_index >= self.tasks:
            raise IndexError(task_index)
        return TaskRef(JobName.from_wire(self.wire_id).task(task_index).to_wire())


@dataclass(slots=True)
class ManualClock:
    epoch_ms: int = 1_704_067_200_000

    def now(self) -> Timestamp:
        current = Timestamp.from_ms(self.epoch_ms)
        self.epoch_ms += 1
        return current

    def advance(self, seconds: float) -> None:
        self.epoch_ms += int(seconds * 1000)


class JourneyWorld:
    """Own real controller persistence and expose concise domain operations."""

    def __init__(
        self,
        root: Path,
        monkeypatch,
        *,
        capacity_available: bool = True,
        cluster_id: str = "journey",
        peer_configs: dict[str, PeerConfig] | None = None,
        federation_peers: list[FederationPeer] | None = None,
    ) -> None:
        self.root = root
        self.clock = ManualClock()
        self._db_dir = root / "db"
        self._incarnation = 0
        self._capacity_available = capacity_available
        self._cluster_id = cluster_id
        self._peer_configs = peer_configs or {}
        self._federation_peers = federation_peers
        self._jobs: dict[str, JobRef] = {}
        self._checkpoint_jobs: dict[str, frozenset[str]] = {}
        self._task_history: dict[str, tuple[tuple[int, int], ...]] = {}
        self._terminal_tasks: set[str] = set()
        self._prior_backend_events: list[BackendEvent] = []
        self.trace: list[str] = []

        db = ControllerDB(db_dir=self._db_dir)
        monkeypatch.setattr(Timestamp, "now", classmethod(lambda cls: self.clock.now()))
        self.controller, self.backend = self._build_controller(db)

    def _build_controller(self, db: ControllerDB) -> tuple[Controller, ScriptedTaskBackend]:
        self._incarnation += 1
        state_dir = self.root / f"controller-{self._incarnation}"
        config = ControllerConfig(
            cluster_id=self._cluster_id,
            remote_state_dir=f"file://{self.root / 'remote'}",
            local_state_dir=state_dir,
            peers=self._peer_configs,
        )
        backend_type = ScriptedTaskBackend if self._capacity_available else UnavailableTaskBackend
        backend = backend_type(DbTransitionReader(db))
        log_stack = build_log_stack(
            log_service_address="",
            local_log_dir=state_dir / "log-server",
            host=config.host,
            worker_token=None,
        )
        controller = Controller(
            config=config,
            backends={DEFAULT_BACKEND_ID: backend},
            log_stack=log_stack,
            threads=ThreadContainer(name=f"journey-{self._incarnation}"),
            db=db,
            federation_peers=self._federation_peers,
        )
        return controller, backend

    def close(self) -> None:
        self.controller.stop()

    def restart(self) -> None:
        """Close and reconstruct the controller over the same SQLite directory."""
        self.trace.append("restart")
        self._prior_backend_events.extend(self.backend.events)
        self.controller.stop()
        db = ControllerDB(db_dir=self._db_dir)
        self.controller, self.backend = self._build_controller(db)
        self._check_invariants()

    def checkpoint(self) -> tuple[str, CheckpointResult]:
        path, result = self.controller.begin_checkpoint()
        self._checkpoint_jobs[path] = frozenset(self._jobs)
        self.trace.append(f"checkpoint {path}")
        return path, result

    def restore(self, checkpoint: str) -> None:
        """Replace the live DB with an exact published checkpoint and reopen."""
        self.trace.append(f"restore {checkpoint}")
        self._prior_backend_events.extend(self.backend.events)
        self.controller.stop()
        restored = download_checkpoint_to_local(
            f"file://{self.root / 'remote'}",
            self._db_dir,
            checkpoint_dir=checkpoint,
        )
        if not restored:
            raise AssertionError(f"checkpoint did not restore: {checkpoint}")
        retained = self._checkpoint_jobs[checkpoint]
        self._jobs = {job_id: job for job_id, job in self._jobs.items() if job_id in retained}
        self._task_history = {
            task_id: history
            for task_id, history in self._task_history.items()
            if any(task_id.startswith(f"{job_id}/") for job_id in retained)
        }
        self._terminal_tasks = {
            task_id for task_id in self._terminal_tasks if any(task_id.startswith(f"{job_id}/") for job_id in retained)
        }
        db = ControllerDB(db_dir=self._db_dir)
        self.controller, self.backend = self._build_controller(db)
        self._check_invariants()

    def submit(
        self,
        name: str,
        *,
        tasks: int = 1,
        failure_retries: int = 0,
        preemption_retries: int = 0,
        max_task_failures: int | None = None,
        coscheduled: bool = False,
        scheduling_timeout: float | None = None,
        execution_timeout: float | None = None,
        required_attributes: dict[str, str] | None = None,
    ) -> JobRef:
        return self._launch(
            JobName.root("journey", name),
            tasks=tasks,
            failure_retries=failure_retries,
            preemption_retries=preemption_retries,
            max_task_failures=max_task_failures,
            coscheduled=coscheduled,
            scheduling_timeout=scheduling_timeout,
            execution_timeout=execution_timeout,
            required_attributes=required_attributes,
        )

    def submit_child(
        self,
        parent: JobRef,
        name: str,
        *,
        tasks: int = 1,
        failure_retries: int = 0,
        preemption_retries: int = 0,
    ) -> JobRef:
        """Submit a child Job under a live parent."""
        child = JobName.from_wire(parent.wire_id).child(name)
        return self._launch(
            child,
            tasks=tasks,
            failure_retries=failure_retries,
            preemption_retries=preemption_retries,
        )

    def _launch(
        self,
        job_name: JobName,
        *,
        tasks: int,
        failure_retries: int,
        preemption_retries: int,
        max_task_failures: int | None = None,
        coscheduled: bool = False,
        scheduling_timeout: float | None = None,
        execution_timeout: float | None = None,
        required_attributes: dict[str, str] | None = None,
    ) -> JobRef:
        entrypoint = job_pb2.RuntimeEntrypoint()
        entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
        request = controller_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=entrypoint,
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
            replicas=tasks,
            max_retries_failure=failure_retries,
            max_retries_preemption=preemption_retries,
            # The public journey vocabulary counts retries. Unless a journey is
            # specifically about the job-wide budget, allow those retries at
            # the aggregate level too.
            max_task_failures=failure_retries if max_task_failures is None else max_task_failures,
        )
        if coscheduled:
            request.coscheduling.group_by = "journey.group"
        if scheduling_timeout is not None:
            request.scheduling_timeout.milliseconds = int(scheduling_timeout * 1000)
        if execution_timeout is not None:
            request.timeout.milliseconds = int(execution_timeout * 1000)
        for key, value in (required_attributes or {}).items():
            constraint = request.constraints.add(key=key, op=job_pb2.CONSTRAINT_OP_EQ)
            constraint.value.string_value = value
        response = self.controller.launch_job(request)
        ref = JobRef(response.job_id, tasks, coscheduled)
        self._jobs[ref.wire_id] = ref
        self.trace.append(f"submit {ref.wire_id} tasks={tasks}")
        self._check_invariants()
        return ref

    def succeed(self, task: TaskRef, *, attempt_id: int | None = None) -> None:
        self._observe(task, job_pb2.TASK_STATE_SUCCEEDED, attempt_id=attempt_id)

    def succeed_all(self, job: JobRef) -> None:
        for task_index in range(job.tasks):
            self.succeed(job[task_index])

    def fail(
        self,
        task: TaskRef,
        *,
        error: str = "application failure",
        exit_code: int = 1,
        attempt_id: int | None = None,
    ) -> None:
        self._observe(task, job_pb2.TASK_STATE_FAILED, error=error, exit_code=exit_code, attempt_id=attempt_id)

    def lose_runtime(self, task: TaskRef, *, error: str = "runtime disappeared") -> None:
        self._observe(task, job_pb2.TASK_STATE_WORKER_FAILED, error=error)

    def preempt(self, task: TaskRef, *, error: str = "preempted") -> None:
        self._observe(task, job_pb2.TASK_STATE_PREEMPTED, error=error)

    def _observe(
        self,
        task: TaskRef,
        state: int,
        *,
        error: str = "",
        exit_code: int | None = None,
        attempt_id: int | None = None,
    ) -> None:
        self.backend.observe(
            task.wire_id,
            ScriptedObservation(state, error=error, exit_code=exit_code, attempt_id=attempt_id),
        )
        self.trace.append(f"observe {task.wire_id} {job_pb2.TaskState.Name(state)}")

    def cancel(self, job: JobRef) -> None:
        self.controller.terminate_job(job.wire_id)
        self.trace.append(f"cancel {job.wire_id}")

    def backend_outage(self, *, ticks: int) -> None:
        self.backend.fail_reconcile(times=ticks)
        self.trace.append(f"backend unavailable ticks={ticks}")

    def wait_through_outage(self, *, ticks: int) -> None:
        """Run ticks that must fail at the scripted backend boundary."""
        for _ in range(ticks):
            try:
                self.controller.run_control_tick()
            except ConnectionError:
                self.trace.append("tick backend-unavailable")
                self._check_invariants()
            else:
                raise AssertionError(f"backend unexpectedly reconciled: {self.timeline}")

    def step(self) -> None:
        self.controller.run_control_tick()
        self.trace.append("tick")
        self._check_invariants()

    def settle(self, *, max_ticks: int = 20) -> None:
        previous = self._fingerprint()
        for _ in range(max_ticks):
            self.step()
            current = self._fingerprint()
            if current == previous and not self.backend.has_pending_observations:
                return
            previous = current
        raise AssertionError(
            f"journey did not quiesce after {max_ticks} ticks; pending={self.backend.pending_task_ids}; "
            f"previous={previous}; current={self._fingerprint()}: {self.timeline}"
        )

    @property
    def timeline(self) -> str:
        return " -> ".join(self.trace)

    def job(self, job: JobRef) -> job_pb2.JobStatus:
        return self.controller.get_job_status(job.wire_id).job

    def tasks(self, job: JobRef) -> list[job_pb2.TaskStatus]:
        return list(self.controller.list_tasks(job.wire_id).tasks)

    def task(self, task: TaskRef) -> job_pb2.TaskStatus:
        return self.controller.get_task_status(task.wire_id).task

    def backend_events(self, *, kind: str | None = None) -> list[BackendEvent]:
        events = [*self._prior_backend_events, *self.backend.events]
        if kind is None:
            return events
        return [event for event in events if event.kind == kind]

    def _fingerprint(self) -> tuple:
        return tuple(
            (
                job.wire_id,
                self.job(job).state,
                tuple(
                    (
                        task.task_id,
                        task.state,
                        task.current_attempt_id,
                        tuple(
                            (attempt.attempt_id, attempt.state) for attempt in self.task(TaskRef(task.task_id)).attempts
                        ),
                    )
                    for task in self.tasks(job)
                ),
            )
            for job in self._jobs.values()
        )

    def _check_invariants(self) -> None:
        active_states = {
            job_pb2.TASK_STATE_ASSIGNED,
            job_pb2.TASK_STATE_BUILDING,
            job_pb2.TASK_STATE_RUNNING,
        }
        terminal_states = {
            job_pb2.TASK_STATE_SUCCEEDED,
            job_pb2.TASK_STATE_FAILED,
            job_pb2.TASK_STATE_KILLED,
            job_pb2.TASK_STATE_UNSCHEDULABLE,
            job_pb2.TASK_STATE_COSCHED_FAILED,
        }
        for job in self._jobs.values():
            listed = self.tasks(job)
            status = self.job(job)
            counts: dict[str, int] = {}
            listed_states = {task.state for task in listed}
            if job.coscheduled and listed_states & active_states and job_pb2.TASK_STATE_PENDING in listed_states:
                raise AssertionError(f"coscheduled Job split between active and pending Tasks: {self.timeline}")
            for task in listed:
                detail = self.task(TaskRef(task.task_id))
                attempts = tuple((attempt.attempt_id, attempt.state) for attempt in detail.attempts)
                previous = self._task_history.get(task.task_id, ())
                if tuple(attempt_id for attempt_id, _ in attempts[: len(previous)]) != tuple(
                    attempt_id for attempt_id, _ in previous
                ):
                    raise AssertionError(f"Attempt history was rewritten for {task.task_id}: {self.timeline}")
                terminal_attempt_states = {
                    job_pb2.TASK_STATE_SUCCEEDED,
                    job_pb2.TASK_STATE_FAILED,
                    job_pb2.TASK_STATE_KILLED,
                    job_pb2.TASK_STATE_PREEMPTED,
                    job_pb2.TASK_STATE_WORKER_FAILED,
                    job_pb2.TASK_STATE_UNSCHEDULABLE,
                    job_pb2.TASK_STATE_COSCHED_FAILED,
                }
                for (_, previous_state), (_, current_state) in zip(previous, attempts, strict=False):
                    if previous_state in terminal_attempt_states and current_state != previous_state:
                        raise AssertionError(f"terminal Attempt changed for {task.task_id}: {self.timeline}")
                self._task_history[task.task_id] = attempts
                if sum(attempt.state in active_states for attempt in detail.attempts) > 1:
                    raise AssertionError(f"multiple live Attempts for {task.task_id}: {self.timeline}")
                if task.state in terminal_states and any(attempt.state in active_states for attempt in detail.attempts):
                    raise AssertionError(f"terminal Task retained a live Attempt: {task.task_id}: {self.timeline}")
                if task.task_id in self._terminal_tasks and task.state not in terminal_states:
                    raise AssertionError(f"terminal Task revived: {task.task_id}: {self.timeline}")
                if task.state in terminal_states:
                    self._terminal_tasks.add(task.task_id)
                state_name = job_pb2.TaskState.Name(task.state).removeprefix("TASK_STATE_").lower()
                counts[state_name] = counts.get(state_name, 0) + 1
            if dict(status.task_state_counts) != counts:
                raise AssertionError(
                    f"Job fold disagrees for {job.wire_id}: public={dict(status.task_state_counts)} tasks={counts}: "
                    f"{self.timeline}"
                )

        launches = [(event.task_id, event.attempt_id) for event in self.backend_events(kind="launched")]
        if len(launches) != len(set(launches)):
            raise AssertionError(f"duplicate backend launch: {self.timeline}")


@contextlib.contextmanager
def journey_world(root: Path, monkeypatch):
    world = JourneyWorld(root, monkeypatch)
    try:
        yield world
    finally:
        world.close()
