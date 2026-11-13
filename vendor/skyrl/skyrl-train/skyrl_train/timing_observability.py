"""Sink-neutral timing observations and their publishing adapters."""

from __future__ import annotations

import time
import math
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from skyrl_train.telemetry import TRAINER_ROLE, phase_attributes, phase_duration


TIMING_PARENTS: dict[str, str | None] = {
    "step": None,
    "generate": "step",
    "wait_for_generation_buffer": "step",
    "assemble_generation_group_mini_batch": "step",
    "postprocess_trajectory_batch": "step",
    "convert_to_training_input": "step",
    "run_training": "step",
    "fwd_logprobs_values_reward": "run_training",
    "apply_reward_kl_penalty": "run_training",
    "compute_advantages_and_returns": "run_training",
    "train_critic_and_policy": "run_training",
    "critic_train": "train_critic_and_policy",
    "policy_train": "train_critic_and_policy",
    "policy_critic_overlap_train": "train_critic_and_policy",
    "backload_policy_optimizer_to_gpu": "train_critic_and_policy",
    "sync_weights": "step",
    "offload_policy_model_to_cpu": "step",
    "dump_data_batch": "run_training",
    "init_weight_sync_state": None,
    "save_checkpoints": "step",
    # Background elapsed time may span several steps; only the await is step-blocking.
    "checkpoint_upload": None,
    "checkpoint_upload_blocking": "step",
    "cleanup_old_checkpoints": "save_checkpoints",
    "save_hf_model": "step",
    "queue_hf_export": "step",
    "eval": "step",
    "update_ref_with_policy": "step",
}


@dataclass(frozen=True)
class PhaseTiming:
    name: str
    duration_seconds: float
    root: str
    parent: str | None


class PhaseBreakdown:
    """Split one root phase's wall time into child phases and publish the unattributed residual."""

    def __init__(
        self,
        root: str,
        parents: Mapping[str, str] | None = None,
        *,
        enabled: bool,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.root = root
        self.enabled = enabled
        self._parents = parents or {}
        self.clock = clock
        self._started = clock() if enabled else 0.0
        self._durations: dict[str, float] = {}

    @contextmanager
    def span(self, phase: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = self.clock()
        try:
            yield
        finally:
            self._durations[phase] = self._durations.get(phase, 0.0) + self.clock() - started

    def durations(self) -> dict[str, float]:
        """Each entered phase's accumulated seconds."""
        return dict(self._durations)

    def add(self, phase: str, seconds: float) -> None:
        """Count time spent in a phase measured elsewhere, such as in another process."""
        self._durations[phase] = self._durations.get(phase, 0.0) + seconds

    def publish(self, *, clock_domain: str, attributes: Mapping[str, str]) -> float:
        """Record the root, each entered phase under its parent and the residual; return the root's duration."""
        if not self.enabled:
            return 0.0
        total = self.clock() - self._started
        parents = {phase: self._parents.get(phase, self.root) for phase in self._durations}
        children = sum(duration for phase, duration in self._durations.items() if parents[phase] == self.root)
        rows = [
            (self.root, total, None),
            *((phase, duration, parents[phase]) for phase, duration in self._durations.items()),
        ]
        rows.append((f"{self.root}_residual", total - children, self.root))
        for phase, duration, parent in rows:
            phase_duration.record(
                duration,
                attributes={
                    **attributes,
                    **phase_attributes(phase=phase, root=self.root, parent=parent, clock_domain=clock_domain),
                },
            )
        return total


STEP_WALL_PHASES = (
    "group_admission",
    "batch_assembly",
    "training_preparation",
    "advantages",
    "policy_training",
    "group_bookkeeping",
    "weight_sync",
    "step_end_bookkeeping",
    "checkpoint_work",
    "evaluation",
    "unaccounted",
)


class StepWallTime:
    """Exclusive optimizer-step wall clock; legacy inclusive timers are not additive."""

    def __init__(self, budgets: Mapping[str, float], *, clock: Callable[[], float] | None = None) -> None:
        if set(budgets) != set(STEP_WALL_PHASES):
            raise ValueError(f"Step phase budgets must name exactly {STEP_WALL_PHASES}")
        self.budgets = {name: float(budgets[name]) for name in STEP_WALL_PHASES}
        if any(not math.isfinite(value) or value < 0 for value in self.budgets.values()):
            raise ValueError("Step phase budgets must be finite nonnegative seconds")
        self.clock = clock or time.monotonic
        self.durations = {name: 0.0 for name in STEP_WALL_PHASES}
        self._active: str | None = None
        self._started: float | None = None

    def start(self, phase: str) -> None:
        if phase not in self.durations or phase == "unaccounted":
            raise ValueError(f"Invalid step wall phase: {phase}")
        now = self.clock()
        if self._active is not None:
            assert self._started is not None
            self.durations[self._active] += now - self._started
        self._active, self._started = phase, now

    def finish(
        self, step_seconds: float, *, ended_at: float | None = None, tolerance: float = 0.01
    ) -> dict[str, float]:
        if self._active is None:
            raise ValueError("Step wall clock was not started")
        now = self.clock() if ended_at is None else ended_at
        assert self._started is not None
        self.durations[self._active] += now - self._started
        residual = step_seconds - sum(self.durations.values())
        if residual < -tolerance:
            raise ValueError(f"Step phases exceed timing/step by {-residual:.3f}s")
        if residual > 0:
            self.durations["unaccounted"] = residual
        self._active = None
        return {
            key: value
            for phase in STEP_WALL_PHASES
            for key, value in (
                (f"timing/step_wall/{phase}", self.durations[phase]),
                (f"timing/step_wall_budget/{phase}", self.budgets[phase]),
                (f"timing/step_wall_overrun/{phase}", max(0.0, self.durations[phase] - self.budgets[phase])),
            )
        }


class TimingSink(Protocol):
    def publish(self, observations: Sequence[PhaseTiming], step: int) -> None: ...


class Tracker(Protocol):
    def log(self, metrics: Mapping[str, float], *, step: int, commit: bool) -> None: ...


def nearest_recorded_parent(name: str, recorded: Mapping[str, object]) -> str | None:
    parent = TIMING_PARENTS.get(name)
    while parent is not None and parent not in recorded:
        parent = TIMING_PARENTS.get(parent)
    return parent


def declared_root(name: str) -> str:
    root = name
    while TIMING_PARENTS.get(root) is not None:
        root = TIMING_PARENTS[root]
    return root


def phase_timing_observations(timings: Mapping[str, float]) -> tuple[PhaseTiming, ...]:
    """Preserve measured wall durations; async spans may overlap and are not additive."""
    known = {name: float(duration) for name, duration in timings.items() if name in TIMING_PARENTS}
    return tuple(
        PhaseTiming(name, duration, declared_root(name), nearest_recorded_parent(name, known))
        for name, duration in known.items()
    )


class FinelogTimingSink:
    def publish(self, observations: Sequence[PhaseTiming], step: int) -> None:
        for observation in observations:
            phase_duration.record(
                observation.duration_seconds,
                attributes={
                    **phase_attributes(
                        phase=observation.name,
                        root=observation.root,
                        parent=observation.parent,
                        clock_domain="inclusive_wall",
                    ),
                    "role": TRAINER_ROLE,
                    "step": str(step),
                },
            )


def publish_step_timings(timings: Mapping[str, float], step: int, sinks: Sequence[TimingSink] | None = None) -> None:
    observations = phase_timing_observations(timings)
    for sink in (FinelogTimingSink(),) if sinks is None else sinks:
        sink.publish(observations, step)


def publish_startup_timings(
    startup_timings: MutableMapping[str, float],
    step_timings: MutableMapping[str, float],
    *,
    step: int,
    tracker: Tracker,
    console: Callable[..., None],
) -> None:
    """Move step timings into startup timings, clear them, then publish them."""
    startup_timings.update(step_timings)
    step_timings.clear()
    if not startup_timings:
        return
    payload = {f"startup/{name}": duration for name, duration in startup_timings.items()}
    console(payload, step=step, kind="startup")
    tracker.log(payload, step=step, commit=False)
