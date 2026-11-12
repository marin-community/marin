import contextlib
import math
import os
import socket
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import ray
from loguru import logger

from marinskyrl.environment_contract import TRAINING_TYPE_ENV, TrainingType
from skyrl_train.metric_names import ENVIRONMENT_METRIC_PREFIX

try:
    from rigging import telemetry
    from rigging.telemetry.serialization import EventBody
except ImportError as error:
    # An installed rigging without the telemetry submodule raises ImportError, not
    # ModuleNotFoundError; the name check still keeps a failure inside rigging visible.
    if error.name != "rigging":
        raise
    from skyrl_train import inert_telemetry as telemetry
    from skyrl_train.inert_telemetry import EventBody


# A process that forwards a foreign system's metrics publishes under that system's name; this one
# covers everything MarinSkyRL measures about itself.
SERVICE = "marinskyrl"
DRIVER_ROLE = "driver"
TRAINER_ROLE = "trainer"
CONTROLLER_ROLE = "controller"
WORKER_ROLE = "worker"
SHUTDOWN_TIMEOUT_SECONDS = 2.0


work_completed = telemetry.counter("work_completed", unit="{item}")
phase_duration = telemetry.histogram("phase_duration_seconds", unit="s")
# rigging publishes `queue_depth` and `progress_time_seconds` for its own exporter from the process
# that configures the service, so a plain name here lands its rows in our namespace -- on a real run
# it outnumbered ours under `progress_time_seconds` by more than two orders of magnitude. Hence the
# `rollout_` prefix below. `progress_time_seconds` keeps the plain name because levanter writes it
# and finelog's TRAINING_STATUS_NAMES reads it; ours carry `work_kind`, rigging's `progress_kind`.
progress_timestamp = telemetry.gauge("progress_time_seconds", unit="s")
policy_step = telemetry.gauge("policy_step", unit="{step}")
rollout_queue_depth = telemetry.gauge("rollout_queue_depth", unit="{item}")
rollout_capacity = telemetry.gauge("rollout_capacity", unit="{item}")
rollout_staleness = telemetry.histogram("rollout_staleness_steps", unit="{step}")
training_metric = telemetry.histogram("training_metric_value")
training_nonfinite_values = telemetry.counter("training_nonfinite_values", unit="{value}")


def record_event(
    name: str,
    fields: dict[str, str | int | float | bool | None],
    *,
    attributes: dict[str, str] | None = None,
) -> None:
    """Enqueue a flat event, omitting unavailable values."""
    telemetry.event(
        name, EventBody({key: value for key, value in fields.items() if value is not None}), attributes=attributes
    )


@dataclass(frozen=True)
class ConsumedWork:
    """Rows and tokens one optimizer step consumed, excluding data-parallel padding."""

    sequences: int
    response_tokens: int
    loss_tokens: int


def record_consumed_work(work: ConsumedWork, *, step: int) -> None:
    """Record useful work after an optimizer step completes successfully."""
    attributes = {"role": TRAINER_ROLE, "step": str(step)}
    for kind, count in (
        ("consumed_sample", work.sequences),
        ("consumed_response_token", work.response_tokens),
        ("consumed_loss_token", work.loss_tokens),
    ):
        work_completed.add(count, attributes={**attributes, "work_kind": kind})


def record_training_metrics(metrics: Mapping[str, object], *, step: int, kind: str) -> None:
    """Mirror the selected trainer scalar families without changing their values."""
    for name, value in metrics.items():
        if not name.startswith(
            (
                "policy/",
                "reward/",
                "loss/",
                "async/",
                "consumed/",
                "generate/",
                "generator/",
                "val/",
                "eval/",
                "env/",
                ENVIRONMENT_METRIC_PREFIX,
            )
        ):
            continue
        if not isinstance(value, (int, float)):
            continue
        attributes = {"metric": name, "step": str(step), "role": TRAINER_ROLE, "payload_kind": kind}
        if math.isfinite(value):
            training_metric.record(float(value), attributes=attributes)
        else:
            training_nonfinite_values.add(1, attributes=attributes)


class _BackgroundCollector(Protocol):
    def start(self) -> None: ...

    def stop(self, *, timeout: float) -> None: ...


class _InertCollector:
    def start(self) -> None:
        pass

    def stop(self, *, timeout: float) -> None:
        pass


_inert_collector = _InertCollector()


@dataclass
class _ProcessState:
    owner: "ProcessTelemetry | None" = None
    policy_step: int | None = None
    last_progress_timestamp: float | None = None
    queue_depth: int | None = None
    queue_capacity: int | None = None

    def claim(self, owner: "ProcessTelemetry") -> bool:
        if self.owner is not None:
            return False
        self.owner = owner
        self.policy_step = None
        self.last_progress_timestamp = None
        self.queue_depth = None
        self.queue_capacity = None
        return True

    def release(self, owner: "ProcessTelemetry") -> None:
        if self.owner is owner:
            self.owner = None


# Rigging has one process-wide runtime, so application progress follows its single lifecycle owner.
_process_state = _ProcessState()


@dataclass(frozen=True)
class TelemetryConfig:
    endpoint: str | None = None
    run_id: str | None = None
    execution_uid: str | None = None
    serving_job_id: str | None = None
    training_type: TrainingType | None = None

    @classmethod
    def from_environment(cls) -> "TelemetryConfig":
        def text(environment_name: str) -> str | None:
            value = os.environ.get(environment_name, "").strip()
            return value or None

        return cls(
            endpoint=text("SKYRL_TELEMETRY_ENDPOINT"),
            run_id=text("SKYRL_RUN_ID"),
            execution_uid=text("SKYRL_EXECUTION_UID") or _iris_execution_uid(),
            serving_job_id=text("SKYRL_SERVING_JOB_ID"),
            training_type=TrainingType(value) if (value := text(TRAINING_TYPE_ENV)) else None,
        )


def _iris_execution_uid() -> str | None:
    if attempt_uid := os.environ.get("IRIS_ATTEMPT_UID"):
        return f"iris:{attempt_uid}"
    return None


def _iris_resources() -> dict[str, str]:
    task_with_attempt = os.environ.get("IRIS_TASK_ID")
    resources: dict[str, str] = {}
    if task_with_attempt:
        task_id, separator, attempt = task_with_attempt.rpartition(":")
        if not separator or "/" in attempt:
            task_id, attempt = task_with_attempt, ""
        job_id, separator, _ = task_id.rpartition("/")
        if separator:
            resources.update(job_id=job_id, task_id=task_id)
        if attempt:
            resources["attempt"] = attempt
    for environment_name, resource_name in (
        ("IRIS_WORKER_ID", "worker"),
        ("IRIS_MULTIGPU_PROCESS_INDEX", "process_index"),
        ("IRIS_NODE_NAME", "node_name"),
    ):
        if value := os.environ.get(environment_name):
            resources[resource_name] = value
    return resources


def _ray_resources() -> dict[str, str]:
    if not ray.is_initialized():
        return {}
    try:
        context = ray.get_runtime_context()
    except Exception:
        logger.warning("Could not read Ray identity for telemetry; continuing without it", exc_info=True)
        return {}

    resources: dict[str, str] = {}
    for name, getter_name, keep_zero in (
        ("ray_job_id", "get_job_id", False),
        ("ray_task_id", "get_task_id", False),
        ("ray_task_attempt", "get_task_attempt_number", True),
        ("actor_uid", "get_actor_id", False),
        ("node_uid", "get_node_id", False),
    ):
        try:
            raw_value = getattr(context, getter_name)()
        except Exception:
            logger.warning(f"Could not read {name} for telemetry; continuing without it", exc_info=True)
            continue
        if raw_value is None:
            continue
        value = str(raw_value)
        if value and (keep_zero or set(value) != {"0"}):
            resources[name] = value
    return resources


def _resources(config: TelemetryConfig, role: str) -> dict[str, str]:
    resources = {
        **_iris_resources(),
        **_ray_resources(),
        "host": socket.gethostname(),
        "run_id": config.run_id or "",
        "execution_uid": config.execution_uid or "",
        "role": role,
    }
    if config.serving_job_id:
        resources["serving_job_id"] = config.serving_job_id
    if config.training_type is not None:
        resources["training_type"] = config.training_type.value
    return resources


def phase_attributes(*, phase: str, root: str, parent: str | None, clock_domain: str) -> dict[str, str]:
    """Attributes that place one duration in a phase tree; a root carries no parent."""
    attributes = {"phase": phase, "root": root, "clock_domain": clock_domain}
    if parent is not None:
        attributes["parent"] = parent
    return attributes


@contextlib.contextmanager
def critical_phase(phase: Literal["rollout_or_inference_wait", "train_step"], step: int) -> Iterator[None]:
    started = time.perf_counter()
    outcome = "success"
    try:
        yield
    except BaseException:
        outcome = "failure"
        raise
    finally:
        phase_duration.record(
            time.perf_counter() - started,
            attributes={
                "phase": phase,
                "clock_domain": "critical_path",
                "role": TRAINER_ROLE,
                "outcome": outcome,
                "step": str(step),
            },
        )


def record_policy_step(step: int) -> None:
    progress_time = time.time()
    _process_state.policy_step = step
    _process_state.last_progress_timestamp = progress_time
    attributes = {"work_kind": "policy_step", "role": TRAINER_ROLE}
    work_completed.add(1, attributes={**attributes, "step": str(step)})
    progress_timestamp.set(progress_time, attributes=attributes)
    policy_step.set(step, attributes={"role": TRAINER_ROLE})


@dataclass(frozen=True)
class GeneratedWork:
    """Rollouts, samples, and generated tokens in one rollout result."""

    rollout_count: int
    sample_count: int
    generated_token_count: int

    @classmethod
    def from_batch(cls, response_ids: Sequence[Sequence[int]], is_last_step: Sequence[bool] | None) -> "GeneratedWork":
        sample_count = len(response_ids)
        return cls(
            rollout_count=sample_count if is_last_step is None else sum(is_last_step),
            sample_count=sample_count,
            generated_token_count=sum(len(response) for response in response_ids),
        )


def record_generated_work(work: GeneratedWork, weights_step: int) -> None:
    """Count generated work against the policy version that produced it.

    Not against the step that recorded it: a group can wait in the buffer across step boundaries, so
    the producer cannot know which step will consume it. Staleness is recorded by
    `record_rollout_staleness` where the trainer measures it.
    """
    rollout_count = work.rollout_count
    progress_time = time.time()
    if work.sample_count:
        _process_state.last_progress_timestamp = progress_time
    for work_kind, count in (
        ("rollout", rollout_count),
        ("sample", work.sample_count),
        ("generated_token", work.generated_token_count),
    ):
        if count:
            work_completed.add(
                count,
                attributes={"work_kind": work_kind, "role": TRAINER_ROLE, "weights_step": str(weights_step)},
            )
    if rollout_count:
        progress_timestamp.set(
            progress_time,
            attributes={"work_kind": "rollout", "role": TRAINER_ROLE},
        )


def record_rollout_staleness(stalenesses: Sequence[int], step: int) -> None:
    """How far behind the consuming step each admitted group's policy was.

    Measured where the trainer measures it, which is the only place it is known: the same values it
    asserts against `max_staleness_steps` and reports as `async/staleness_*`.
    """
    for staleness in stalenesses:
        rollout_staleness.record(staleness, attributes={"role": TRAINER_ROLE, "step": str(step)})


def record_rollout_buffer(depth: int, queue_capacity: int) -> None:
    _process_state.queue_depth = depth
    _process_state.queue_capacity = queue_capacity
    attributes = {"queue": "rollout_buffer", "role": TRAINER_ROLE}
    rollout_queue_depth.set(depth, attributes=attributes)
    rollout_capacity.set(queue_capacity, attributes=attributes)


class ProcessTelemetry:
    def __init__(self, config: TelemetryConfig, role: str) -> None:
        self._config = config
        self._role = role
        self._configured = False

    def __enter__(self) -> "ProcessTelemetry":
        if not _process_state.claim(self):
            logger.warning("Telemetry already has a process owner; leaving the nested lifecycle inert")
            return self
        if self._config.endpoint is None:
            return self
        if self._config.run_id is None or self._config.execution_uid is None:
            logger.warning("Telemetry requires run_id and execution_uid; export remains inert")
            return self
        telemetry.configure(
            endpoint=self._config.endpoint,
            service=SERVICE,
            attributes=_resources(self._config, self._role),
        )
        self._configured = telemetry.runtime_status().configured
        if self._configured:
            record_event("lifecycle", {"state": "started"}, attributes={"role": self._role})
        return self

    def collector_or_inert(self, collector: _BackgroundCollector) -> _BackgroundCollector:
        return collector if self._configured else _inert_collector

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc, traceback
        if self._configured:
            export = telemetry.runtime_status()
            record_event(
                "terminal",
                {
                    "status": "completed" if exc_type is None else "failed",
                    "reason": "normal_exit" if exc_type is None else getattr(exc_type, "__name__", "exception"),
                    "export_queued_records": export.queued_records,
                    "export_lost_records": export.lost_records,
                    "policy_step": _process_state.policy_step,
                    "last_progress_time_seconds": _process_state.last_progress_timestamp,
                    "queue_depth": _process_state.queue_depth,
                    "queue_capacity": _process_state.queue_capacity,
                },
                attributes={"role": self._role},
            )
            telemetry.shutdown(SHUTDOWN_TIMEOUT_SECONDS)
        self._configured = False
        _process_state.release(self)
        return False


def process_telemetry(role: str) -> ProcessTelemetry:
    return ProcessTelemetry(TelemetryConfig.from_environment(), role)
