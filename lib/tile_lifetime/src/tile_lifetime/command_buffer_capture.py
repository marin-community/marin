# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for host callbacks while benchmarking command-buffer replay."""

from __future__ import annotations

import copy
import itertools
import json
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CounterbalancedVariant:
    """One measured implementation and its logical handler calls per execution."""

    name: str
    function: Callable[[], object]
    handler_calls_per_execution: Mapping[str, int]


@dataclass(frozen=True)
class CallbackCheckpoint:
    """Handler counts surrounding one variant in a counterbalanced sample."""

    sample_index: int
    order: tuple[str, ...]
    variant: str
    before: Mapping[str, int]
    after: Mapping[str, int]
    delta: Mapping[str, int]
    logical_handler_calls: Mapping[str, int]


@dataclass(frozen=True)
class CounterbalancedMeasurement:
    """Raw timings and callback evidence from a counterbalanced measurement."""

    measurements: Mapping[str, Mapping[str, Any]]
    execution_order: tuple[tuple[str, ...], ...]
    callback_checkpoints: tuple[CallbackCheckpoint, ...]


class CaptureBehavior(StrEnum):
    """Observed host-callback behavior during command-buffer measurement."""

    CAPTURED_REPLAY = "captured_replay"
    BOUNDED_ORDER_SPECIFIC_RECAPTURE = "bounded_order_specific_recapture"
    PER_LOGICAL_CALL_FALLBACK = "per_logical_call_fallback"
    UNBOUNDED_RECAPTURE = "unbounded_recapture"
    UNATTRIBUTED_CALLBACK = "unattributed_callback"
    MISSING_INITIAL_CAPTURE = "missing_initial_capture"
    INVALID_CHECKPOINT_SEQUENCE = "invalid_checkpoint_sequence"


@dataclass(frozen=True)
class CaptureAcceptancePolicy:
    """Bounded policy for command-buffer capture and replay evidence."""

    require_initial_capture: bool = True
    maximum_recaptures_per_handler_per_order: int = 1
    recapture_only_on_first_sample_of_order: bool = True
    reject_callbacks_from_uninstrumented_variants: bool = True


@dataclass(frozen=True)
class OrderCaptureSummary:
    """Callback deltas attributed to one complete counterbalanced order."""

    order: tuple[str, ...]
    first_sample_index: int
    sample_indices: tuple[int, ...]
    handler_deltas: Mapping[str, int]
    recapture_sample_indices: tuple[int, ...]


@dataclass(frozen=True)
class CaptureAssessment:
    """Capture classification and the evidence used to accept or reject it."""

    accepted: bool
    behavior: CaptureBehavior
    reasons: tuple[str, ...]
    policy: CaptureAcceptancePolicy
    initial_counts: Mapping[str, int]
    final_counts: Mapping[str, int]
    total_timed_deltas: Mapping[str, int]
    order_summaries: tuple[OrderCaptureSummary, ...]

    def to_json(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "status": "accepted" if self.accepted else "rejected",
            "accepted": self.accepted,
            "behavior": self.behavior.value,
            "reasons": list(self.reasons),
            "policy": asdict(self.policy),
            "initial_counts": dict(self.initial_counts),
            "final_counts": dict(self.final_counts),
            "total_timed_deltas": dict(self.total_timed_deltas),
            "order_summaries": [
                {
                    "order": list(summary.order),
                    "first_sample_index": summary.first_sample_index,
                    "sample_indices": list(summary.sample_indices),
                    "handler_deltas": dict(summary.handler_deltas),
                    "recapture_sample_indices": list(summary.recapture_sample_indices),
                }
                for summary in self.order_summaries
            ],
        }


class CaptureAcceptanceError(RuntimeError):
    """Raised after a rejected capture result has been serialized."""

    def __init__(self, result: Mapping[str, Any]):
        assessment = result["capture_acceptance"]
        super().__init__(
            f"command-buffer capture acceptance failed: {assessment['behavior']}: " + "; ".join(assessment["reasons"])
        )
        self.result = result


def _normalized_counts(counts: Mapping[str, int]) -> dict[str, int]:
    normalized = {name: int(value) for name, value in counts.items()}
    if any(value < 0 for value in normalized.values()):
        raise ValueError(f"handler counts must be nonnegative: {normalized}")
    return dict(sorted(normalized.items()))


def _normalized_deltas(deltas: Mapping[str, int]) -> dict[str, int]:
    return dict(sorted((name, int(value)) for name, value in deltas.items()))


def measure_counterbalanced_variants(
    variants: Sequence[CounterbalancedVariant],
    *,
    repeats: int,
    iterations: int,
    synchronize: Callable[[object], None],
    read_handler_counts: Callable[[], Mapping[str, int]],
    clock: Callable[[], float] = time.perf_counter,
) -> CounterbalancedMeasurement:
    """Measure variants while attributing callback deltas to every sample phase."""
    if not variants:
        raise ValueError("counterbalanced measurement requires at least one variant")
    if repeats <= 0 or iterations <= 0:
        raise ValueError("repeats and iterations must be positive")
    names = tuple(variant.name for variant in variants)
    if len(set(names)) != len(names):
        raise ValueError(f"variant names must be unique: {names}")
    if any(calls < 0 for variant in variants for calls in variant.handler_calls_per_execution.values()):
        raise ValueError("logical handler calls per execution must be nonnegative")
    orderings = tuple(itertools.permutations(variants))
    if repeats % len(orderings):
        raise ValueError(f"repeats must be divisible by {len(orderings)} counterbalanced orders")

    samples: dict[str, list[float]] = {variant.name: [] for variant in variants}
    execution_order: list[tuple[str, ...]] = []
    checkpoints: list[CallbackCheckpoint] = []
    for sample_index in range(repeats):
        ordering = orderings[sample_index % len(orderings)]
        order = tuple(variant.name for variant in ordering)
        execution_order.append(order)
        for variant in ordering:
            before = _normalized_counts(read_handler_counts())
            started = clock()
            result = None
            for _ in range(iterations):
                result = variant.function()
            synchronize(result)
            elapsed = clock() - started
            after = _normalized_counts(read_handler_counts())
            handlers = tuple(sorted(set(before) | set(after) | set(variant.handler_calls_per_execution)))
            logical_calls = {
                handler: int(variant.handler_calls_per_execution.get(handler, 0)) * iterations for handler in handlers
            }
            checkpoints.append(
                CallbackCheckpoint(
                    sample_index=sample_index,
                    order=order,
                    variant=variant.name,
                    before={handler: before.get(handler, 0) for handler in handlers},
                    after={handler: after.get(handler, 0) for handler in handlers},
                    delta={handler: after.get(handler, 0) - before.get(handler, 0) for handler in handlers},
                    logical_handler_calls=logical_calls,
                )
            )
            samples[variant.name].append(elapsed * 1e3 / iterations)

    return CounterbalancedMeasurement(
        measurements={
            name: {
                "samples_ms": values,
                "median_ms": statistics.median(values),
                "minimum_ms": min(values),
            }
            for name, values in samples.items()
        },
        execution_order=tuple(execution_order),
        callback_checkpoints=tuple(checkpoints),
    )


def assess_command_buffer_capture(
    initial_counts: Mapping[str, int],
    checkpoints: Sequence[CallbackCheckpoint],
    *,
    policy: CaptureAcceptancePolicy = CaptureAcceptancePolicy(),
) -> CaptureAssessment:
    """Classify bounded recapture separately from per-logical-call fallback."""
    if policy.maximum_recaptures_per_handler_per_order < 0:
        raise ValueError("maximum recaptures per handler per order must be nonnegative")
    initial = _normalized_counts(initial_counts)
    handlers = tuple(
        sorted(
            set(initial)
            | {handler for checkpoint in checkpoints for handler in checkpoint.delta}
            | {handler for checkpoint in checkpoints for handler in checkpoint.logical_handler_calls}
        )
    )
    reasons: list[str] = []
    invalid_sequence = not checkpoints
    if not checkpoints:
        reasons.append("timed measurement contains no callback checkpoints")
    missing_initial_capture = policy.require_initial_capture and any(
        initial.get(handler, 0) == 0 for handler in handlers
    )
    if missing_initial_capture:
        reasons.append("at least one handler had no capture callback before timed measurement")

    expected_before = initial
    total_deltas = {handler: 0 for handler in handlers}
    first_sample_by_order: dict[tuple[str, ...], int] = {}
    samples_by_order: dict[tuple[str, ...], set[int]] = {}
    order_deltas: dict[tuple[str, ...], dict[str, int]] = {}
    recapture_samples: dict[tuple[str, ...], set[int]] = {}
    per_logical_call_fallback = False
    unattributed_callback = False
    unbounded_recapture = False

    for checkpoint in checkpoints:
        before = _normalized_counts(checkpoint.before)
        after = _normalized_counts(checkpoint.after)
        delta = _normalized_deltas(checkpoint.delta)
        if any(before.get(handler, 0) != expected_before.get(handler, 0) for handler in handlers):
            invalid_sequence = True
        computed_delta = {handler: after.get(handler, 0) - before.get(handler, 0) for handler in handlers}
        if any(computed_delta[handler] != delta.get(handler, 0) for handler in handlers):
            invalid_sequence = True
        if any(value < 0 for value in computed_delta.values()):
            invalid_sequence = True
        expected_before = after

        order = checkpoint.order
        first_sample_by_order.setdefault(order, checkpoint.sample_index)
        samples_by_order.setdefault(order, set()).add(checkpoint.sample_index)
        order_deltas.setdefault(order, {handler: 0 for handler in handlers})
        recapture_samples.setdefault(order, set())
        for handler in handlers:
            observed = computed_delta[handler]
            logical = int(checkpoint.logical_handler_calls.get(handler, 0))
            total_deltas[handler] += observed
            order_deltas[order][handler] += observed
            if observed == 0:
                continue
            recapture_samples[order].add(checkpoint.sample_index)
            if logical == 0:
                unattributed_callback = True
                continue
            if observed >= logical:
                per_logical_call_fallback = True
                continue
            if (
                policy.recapture_only_on_first_sample_of_order
                and checkpoint.sample_index != first_sample_by_order[order]
            ):
                unbounded_recapture = True

    for deltas in order_deltas.values():
        if any(value > policy.maximum_recaptures_per_handler_per_order for value in deltas.values()):
            unbounded_recapture = True

    if invalid_sequence:
        reasons.append("callback checkpoints are discontinuous or contain inconsistent deltas")
    if unattributed_callback and policy.reject_callbacks_from_uninstrumented_variants:
        reasons.append("a callback occurred in a variant that declared no logical handler calls")
    if per_logical_call_fallback:
        reasons.append("a timed phase invoked a host callback at least once per logical handler call")
    if unbounded_recapture:
        reasons.append("timed recapture exceeded the per-order bound or occurred after an order's first sample")

    rejected = (
        missing_initial_capture
        or invalid_sequence
        or per_logical_call_fallback
        or unbounded_recapture
        or (unattributed_callback and policy.reject_callbacks_from_uninstrumented_variants)
    )
    if invalid_sequence:
        behavior = CaptureBehavior.INVALID_CHECKPOINT_SEQUENCE
    elif per_logical_call_fallback:
        behavior = CaptureBehavior.PER_LOGICAL_CALL_FALLBACK
    elif unattributed_callback and policy.reject_callbacks_from_uninstrumented_variants:
        behavior = CaptureBehavior.UNATTRIBUTED_CALLBACK
    elif missing_initial_capture:
        behavior = CaptureBehavior.MISSING_INITIAL_CAPTURE
    elif unbounded_recapture:
        behavior = CaptureBehavior.UNBOUNDED_RECAPTURE
    elif any(total_deltas.values()):
        behavior = CaptureBehavior.BOUNDED_ORDER_SPECIFIC_RECAPTURE
    else:
        behavior = CaptureBehavior.CAPTURED_REPLAY

    summaries = tuple(
        OrderCaptureSummary(
            order=order,
            first_sample_index=first_sample_by_order[order],
            sample_indices=tuple(sorted(samples_by_order[order])),
            handler_deltas=order_deltas[order],
            recapture_sample_indices=tuple(sorted(recapture_samples[order])),
        )
        for order in first_sample_by_order
    )
    return CaptureAssessment(
        accepted=not rejected,
        behavior=behavior,
        reasons=tuple(reasons),
        policy=policy,
        initial_counts=initial,
        final_counts=expected_before,
        total_timed_deltas=total_deltas,
        order_summaries=summaries,
    )


def serialize_then_assess_capture(
    path: Path,
    raw_result: Mapping[str, Any],
    assessment_factory: Callable[[], CaptureAssessment],
) -> dict[str, Any]:
    """Persist raw timings before capture assessment and reject only afterward."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = copy.deepcopy(dict(raw_result))
    pending["capture_acceptance"] = {"status": "pending"}
    path.write_text(json.dumps(pending, indent=2, sort_keys=True) + "\n")

    assessment = assessment_factory()
    result = copy.deepcopy(dict(raw_result))
    result["capture_acceptance"] = assessment.to_json()
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not assessment.accepted:
        raise CaptureAcceptanceError(result)
    return result
