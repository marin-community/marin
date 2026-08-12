# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Domain actions and public observations for Iris journeys."""

import contextlib
from dataclasses import dataclass
from pathlib import Path

from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from rigging.server_auth import VerifiedIdentity, identity_scope
from rigging.timing import Timestamp

from iris.cluster.config import PeerConfig
from iris.cluster.controller.checkpoint import CheckpointResult, download_checkpoint_to_local
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.controller.transition_reader import DbTransitionReader
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.log_keys import task_log_key
from iris.cluster.types import DEFAULT_BACKEND_ID, JobName, TaskAttempt
from iris.managed_thread import ThreadContainer
from iris.rpc import controller_pb2, job_pb2
from iris.testing.journeys.backend import (
    BackendEvent,
    ScriptedObservation,
    ScriptedTaskBackend,
    UnavailableTaskBackend,
)


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
        dry_run: bool = False,
        backend_advertisements: dict[str, dict[str, set[str]]] | None = None,
        unavailable_backend_ids: set[str] | None = None,
    ) -> None:
        self.root = root
        self.clock = ManualClock()
        self._db_dir = root / "db"
        self._incarnation = 0
        self._capacity_available = capacity_available
        self._cluster_id = cluster_id
        self._peer_configs = peer_configs or {}
        self._federation_peers = federation_peers
        self._dry_run = dry_run
        self._backend_advertisements = backend_advertisements
        self._unavailable_backend_ids = unavailable_backend_ids or set()
        self._jobs: dict[str, JobRef] = {}
        self._checkpoint_jobs: dict[str, frozenset[str]] = {}
        self._task_history: dict[str, tuple[tuple[int, int], ...]] = {}
        self._terminal_tasks: set[str] = set()
        self._prior_backend_events: list[BackendEvent] = []
        self._prior_backend_calls: list[tuple[str, str]] = []
        self.trace: list[str] = []

        db = ControllerDB(db_dir=self._db_dir)
        monkeypatch.setattr(Timestamp, "now", classmethod(lambda cls: self.clock.now()))
        self.controller, self.backends = self._build_controller(db)
        self.backend = next(iter(self.backends.values()))

    def _build_controller(self, db: ControllerDB) -> tuple[Controller, dict[str, ScriptedTaskBackend]]:
        self._incarnation += 1
        state_dir = self.root / f"controller-{self._incarnation}"
        config = ControllerConfig(
            cluster_id=self._cluster_id,
            remote_state_dir=f"file://{self.root / 'remote'}",
            local_state_dir=state_dir,
            peers=self._peer_configs,
            dry_run=self._dry_run,
        )
        advertisements = self._backend_advertisements or {
            DEFAULT_BACKEND_ID: {"region": {"us-central1"}},
        }
        backends: dict[str, ScriptedTaskBackend] = {}
        for backend_id, advertised in advertisements.items():
            backend_type = (
                UnavailableTaskBackend
                if not self._capacity_available or backend_id in self._unavailable_backend_ids
                else ScriptedTaskBackend
            )
            backend = backend_type(DbTransitionReader(db), backend_id=backend_id)
            backend.configure_routing(advertised)
            backends[backend_id] = backend
        log_stack = build_log_stack(
            log_service_address="",
            local_log_dir=state_dir / "log-server",
            host=config.host,
            worker_token=None,
        )
        self.log_stack = log_stack
        controller = Controller(
            config=config,
            backends=backends,
            log_stack=log_stack,
            threads=ThreadContainer(name=f"journey-{self._incarnation}"),
            db=db,
            federation_peers=self._federation_peers,
        )
        return controller, backends

    def close(self) -> None:
        self.controller.stop()

    def restart(self) -> None:
        """Close and reconstruct the controller over the same SQLite directory."""
        self.trace.append("restart")
        self._remember_backend_activity()
        self.controller.stop()
        db = ControllerDB(db_dir=self._db_dir)
        self.controller, self.backends = self._build_controller(db)
        self.backend = next(iter(self.backends.values()))
        self._check_invariants()

    def checkpoint(self) -> tuple[str, CheckpointResult]:
        path, result = self.controller.begin_checkpoint()
        self._checkpoint_jobs[path] = frozenset(self._jobs)
        self.trace.append(f"checkpoint {path}")
        return path, result

    def restore(self, checkpoint: str) -> None:
        """Replace the live DB with an exact published checkpoint and reopen."""
        self.trace.append(f"restore {checkpoint}")
        self._remember_backend_activity()
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
        self.controller, self.backends = self._build_controller(db)
        self.backend = next(iter(self.backends.values()))
        self._check_invariants()

    def submit(
        self,
        name: str,
        *,
        user: str = "journey",
        tasks: int = 1,
        failure_retries: int = 0,
        preemption_retries: int = 0,
        max_task_failures: int | None = None,
        coscheduled: bool = False,
        scheduling_timeout: float | None = None,
        execution_timeout: float | None = None,
        required_attributes: dict[str, str] | None = None,
        include_resources: bool = True,
    ) -> JobRef:
        return self._launch(
            JobName.root(user, name),
            tasks=tasks,
            failure_retries=failure_retries,
            preemption_retries=preemption_retries,
            max_task_failures=max_task_failures,
            coscheduled=coscheduled,
            scheduling_timeout=scheduling_timeout,
            execution_timeout=execution_timeout,
            required_attributes=required_attributes,
            include_resources=include_resources,
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
        include_resources: bool = True,
    ) -> JobRef:
        entrypoint = job_pb2.RuntimeEntrypoint()
        entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
        request = controller_pb2.Controller.LaunchJobRequest(
            name=job_name.to_wire(),
            entrypoint=entrypoint,
            environment=job_pb2.EnvironmentConfig(),
            replicas=tasks,
            max_retries_failure=failure_retries,
            max_retries_preemption=preemption_retries,
            # The public journey vocabulary counts retries. Unless a journey is
            # specifically about the job-wide budget, allow those retries at
            # the aggregate level too.
            max_task_failures=failure_retries if max_task_failures is None else max_task_failures,
        )
        if include_resources:
            request.resources.CopyFrom(job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3))
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
        backend = self._backend_for_task(task)
        backend.observe(
            task.wire_id,
            ScriptedObservation(state, error=error, exit_code=exit_code, attempt_id=attempt_id),
        )
        self.trace.append(f"observe {task.wire_id} {job_pb2.TaskState.Name(state)}")

    def cancel(self, job: JobRef) -> None:
        self.controller.terminate_job(job.wire_id)
        self.trace.append(f"cancel {job.wire_id}")

    def kick(
        self,
        task: TaskRef,
        *,
        desired_state: int = job_pb2.TASK_STATE_PREEMPTED,
        reason: str = "journey operator override",
        attempt_id: int | None = None,
    ) -> controller_pb2.Controller.KickResult:
        """Queue one public administrative override for a Task or exact Attempt."""
        target = task.wire_id if attempt_id is None else f"{task.wire_id}:{attempt_id}"
        response = self.controller.kick_tasks(
            controller_pb2.Controller.KickTasksRequest(
                targets=[target],
                desired_state=desired_state,
                reason=reason,
            )
        )
        (result,) = response.results
        self.trace.append(f"kick {target} {job_pb2.TaskState.Name(desired_state)} queued={result.queued}")
        return result

    def jobs(self, *, state: str = "") -> list[job_pb2.JobStatus]:
        """List public Jobs, optionally filtered by friendly state name."""
        request = controller_pb2.Controller.ListJobsRequest(query=controller_pb2.Controller.JobQuery(state_filter=state))
        return list(self.controller.list_jobs(request).jobs)

    def set_budget(self, user: str, *, limit: int = 10_000) -> None:
        """Create or replace one budget row through the public service."""
        with identity_scope(VerifiedIdentity(user_id="journey-admin", role="admin")):
            self.controller.set_user_budget(
                controller_pb2.Controller.SetUserBudgetRequest(
                    user_id=user,
                    budget_limit=limit,
                    max_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
                )
            )

    def budget_spent(self, user: str) -> int:
        """Return public current spend for a configured user budget."""
        with identity_scope(VerifiedIdentity(user_id=user, role="user")):
            return self.controller.get_user_budget(user).budget_spent

    def register_endpoint(
        self,
        task: TaskRef,
        name: str,
        address: str,
        *,
        endpoint_id: str,
        attempt_id: int | None = None,
    ) -> str:
        """Register one endpoint against the Task's current public Attempt."""
        current_attempt_id = self.task(task).current_attempt_id if attempt_id is None else attempt_id
        response = self.controller.register_endpoint(
            controller_pb2.Controller.RegisterEndpointRequest(
                name=name,
                address=address,
                task_id=task.wire_id,
                attempt_id=current_attempt_id,
                endpoint_id=endpoint_id,
            )
        )
        self.trace.append(f"endpoint {name} {endpoint_id} attempt={current_attempt_id}")
        return response.endpoint_id

    def endpoints(self, *, name: str = "") -> list[controller_pb2.Controller.Endpoint]:
        request = controller_pb2.Controller.ListEndpointsRequest(prefix=name, exact=bool(name))
        return list(self.controller.list_endpoints(request).endpoints)

    def backend_outage(self, *, ticks: int) -> None:
        if len(self.backends) != 1:
            raise ValueError("backend_outage requires one backend; address a backend in a multi-backend journey")
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
            if current == previous and not any(backend.has_pending_observations for backend in self.backends.values()):
                return
            previous = current
        raise AssertionError(
            f"journey did not quiesce after {max_ticks} ticks; pending={self.pending_task_ids}; "
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

    def task_detail(self, task: TaskRef) -> controller_pb2.Controller.GetTaskStatusResponse:
        """Read one Task plus public Attempt-derived diagnostics."""
        return self.controller.get_task_status(task.wire_id)

    def push_task_logs(self, task: TaskRef, lines: list[str], *, attempt_id: int = 0) -> None:
        """Publish Task logs through the real finelog RPC boundary."""
        entries = []
        for index, data in enumerate(lines):
            entry = logging_pb2.LogEntry(source="stdout", data=data)
            entry.timestamp.epoch_ms = 1_000 + index
            entries.append(entry)
        client = LogServiceClientSync(address=self.log_stack.address)
        client.push_logs(
            logging_pb2.PushLogsRequest(
                key=task_log_key(TaskAttempt(JobName.from_wire(task.wire_id), attempt_id)),
                entries=entries,
            )
        )

    def backend_events(self, *, kind: str | None = None, backend_id: str | None = None) -> list[BackendEvent]:
        events = [*self._prior_backend_events, *self._current_backend_events()]
        if kind is None:
            selected = events
        else:
            selected = [event for event in events if event.kind == kind]
        if backend_id is not None:
            selected = [event for event in selected if event.backend_id == backend_id]
        return selected

    def backend_calls(self, *, kind: str | None = None, backend_id: str | None = None) -> list[tuple[str, str]]:
        """Return externally visible backend calls as ``(backend_id, kind)`` rows."""
        calls = [*self._prior_backend_calls]
        calls.extend((candidate_id, call) for candidate_id, backend in self.backends.items() for call in backend.calls)
        if kind is not None:
            calls = [row for row in calls if row[1] == kind]
        if backend_id is not None:
            calls = [row for row in calls if row[0] == backend_id]
        return calls

    @property
    def pending_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(task_id for backend in self.backends.values() for task_id in backend.pending_task_ids))

    def _backend_for_task(self, task: TaskRef) -> ScriptedTaskBackend:
        backend_id = self.task(task).backend_id
        if backend_id in self.backends:
            return self.backends[backend_id]
        if len(self.backends) == 1:
            return next(iter(self.backends.values()))
        owners = [backend for backend in self.backends.values() if backend.owns_task(task.wire_id)]
        if len(owners) != 1:
            raise AssertionError(f"expected one backend for {task.wire_id}, found {len(owners)}: {self.timeline}")
        return owners[0]

    def _current_backend_events(self) -> list[BackendEvent]:
        return [event for backend in self.backends.values() for event in backend.events]

    def _remember_backend_activity(self) -> None:
        self._prior_backend_events.extend(self._current_backend_events())
        self._prior_backend_calls.extend(
            (backend_id, call) for backend_id, backend in self.backends.items() for call in backend.calls
        )

    def _fingerprint(self) -> tuple:
        return tuple(
            (
                job.wire_id,
                self.job(job).state,
                tuple(
                    (
                        task.task_id,
                        task.state,
                        task.backend_id,
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
