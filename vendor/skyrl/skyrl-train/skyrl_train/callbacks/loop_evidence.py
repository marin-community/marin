"""Persist bounded numeric observations for a separately verified loop result."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from numbers import Real

from marinskyrl.resource_locator import join_resource_path
from skyrl_train.io import io

from .base import TrainerCallback, TrainerControl, TrainerState
from .types import CallbackErrorBehavior


type MetricValue = float | int | bool | None


@dataclass
class StepObservation:
    step: int
    metrics: dict[str, MetricValue]


@dataclass
class LoopEvidence:
    schema_version: int = 1
    observations_only: bool = True
    expected_steps: int | None = None
    started_at: str | None = None
    recorded_at: str | None = None
    wandb_url: str | None = None
    steps: list[StepObservation] = field(default_factory=list)
    evaluations: dict[str, dict[str, MetricValue]] = field(default_factory=dict)


def numeric_metrics(metrics: Mapping[str, object]) -> dict[str, MetricValue]:
    """Retain finite numeric observations without text or structured payloads."""
    result: dict[str, MetricValue] = {}
    for name, value in metrics.items():
        if value is None or isinstance(value, bool):
            result[name] = value
        elif isinstance(value, Real):
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite loop observation: {name}")
            result[name] = int(value) if isinstance(value, int) else float(value)
    return result


class LoopEvidenceCallback(TrainerCallback):
    """Record observations without claiming terminal or checkpoint success."""

    error_behavior = CallbackErrorBehavior.RAISE

    def __init__(self, export_path: str):
        self.path = join_resource_path(export_path, "loop-evidence.json")
        self.record = LoopEvidence()

    def _persist(self, trainer) -> None:
        run = trainer.tracker.logger.get("wandb")
        self.record.wandb_url = None if run is None else run.url
        self.record.recorded_at = datetime.now(timezone.utc).isoformat()
        io.write_bytes_atomic(self.path, json.dumps(asdict(self.record), allow_nan=False).encode() + b"\n")

    def on_train_begin(self, state: TrainerState, control: TrainerControl, **kwargs):
        self.record.expected_steps = state.total_steps
        self.record.started_at = datetime.now(timezone.utc).isoformat()
        self._persist(kwargs["trainer"])
        return control

    def on_step_end(self, state: TrainerState, control: TrainerControl, **kwargs):
        self.record.steps.append(StepObservation(step=state.global_step, metrics=numeric_metrics(state.metrics)))
        self._persist(kwargs["trainer"])
        return control

    def on_evaluate(self, state: TrainerState, control: TrainerControl, metrics: Mapping[str, object], **kwargs):
        self.record.evaluations[str(state.global_step)] = numeric_metrics(metrics)
        self._persist(kwargs["trainer"])
        return control
