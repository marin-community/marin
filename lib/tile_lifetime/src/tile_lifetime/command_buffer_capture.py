# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate steady-state command-buffer replay without workload-specific bounds."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from tile_lifetime.xla_hlo_recovery import parse_hlo_module_text

_CUSTOM_CALL_TARGET = re.compile(r'(?:^|,\s*)custom_call_target="(?P<target>[^"]+)"')


@dataclass(frozen=True)
class CaptureSite:
    """One instrumented handler target and its final-HLO occurrence count."""

    handler: str
    target: str
    occurrences: int


@dataclass(frozen=True)
class CaptureSiteManifest:
    """Static callback topology derived from one final optimized HLO module."""

    executable: str
    hlo_sha256: str
    sites: tuple[CaptureSite, ...]

    @classmethod
    def uninstrumented(cls, executable: str) -> CaptureSiteManifest:
        """Describe an executable that must not invoke instrumented handlers."""
        return cls(executable=executable, hlo_sha256="", sites=())

    @property
    def handler_calls_per_execution(self) -> dict[str, int]:
        """Return statically derived logical calls for one executable execution."""
        calls: Counter[str] = Counter()
        for site in self.sites:
            calls[site.handler] += site.occurrences
        return dict(sorted(calls.items()))

    @property
    def target_handlers(self) -> dict[str, str]:
        """Return the instrumented counter corresponding to each registered target."""
        return {site.target: site.handler for site in self.sites}

    def to_json(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "executable": self.executable,
            "hlo_sha256": self.hlo_sha256,
            "sites": [asdict(site) for site in self.sites],
        }


def derive_capture_site_manifest(
    executable: str,
    final_hlo: str,
    target_handlers: Mapping[str, str],
    *,
    expected_target_occurrences: Mapping[str, int] | None = None,
) -> CaptureSiteManifest:
    """Derive an instrumented callback-site manifest from final optimized HLO."""
    if not executable:
        raise ValueError("capture-site manifest requires an executable name")
    normalized_targets = dict(sorted(target_handlers.items()))
    if not normalized_targets:
        raise ValueError("instrumented capture-site manifest requires handler targets")
    if any(not target or not handler for target, handler in normalized_targets.items()):
        raise ValueError("capture-site handler names and targets must be nonempty")

    module = parse_hlo_module_text(final_hlo)
    occurrences: Counter[str] = Counter()
    for computation in module.computations:
        for instruction in computation.instructions:
            if instruction.opcode != "custom-call":
                continue
            match = _CUSTOM_CALL_TARGET.search(instruction.attributes)
            if match is not None:
                occurrences[match.group("target")] += 1

    missing = tuple(target for target in normalized_targets if occurrences[target] == 0)
    if missing:
        raise ValueError(f"final HLO is missing registered capture targets: {missing}")
    if expected_target_occurrences is not None:
        expected = dict(sorted(expected_target_occurrences.items()))
        if expected.keys() != normalized_targets.keys():
            raise ValueError("expected capture-site targets must exactly match registered targets")
        mismatches = {
            target: {"expected": count, "actual": occurrences[target]}
            for target, count in expected.items()
            if count <= 0 or occurrences[target] != count
        }
        if mismatches:
            raise ValueError(f"final-HLO capture-site multiplicities changed: {mismatches}")
    sites = tuple(
        CaptureSite(handler=handler, target=target, occurrences=occurrences[target])
        for target, handler in normalized_targets.items()
    )
    return CaptureSiteManifest(
        executable=executable,
        hlo_sha256=hashlib.sha256(final_hlo.encode()).hexdigest(),
        sites=sites,
    )


@dataclass(frozen=True)
class CounterbalancedVariant:
    """One executable variant and its final-HLO-derived capture sites."""

    name: str
    function: Callable[[], object]
    capture_sites: CaptureSiteManifest


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


@dataclass(frozen=True)
class CaptureAcceptancePolicy:
    """Fixed bounded stabilization policy applied before timed measurement."""

    maximum_stabilization_rounds: int = 8
    required_consecutive_quiescent_rounds: int = 2
    reject_callbacks_from_uninstrumented_variants: bool = True
    require_zero_timed_callbacks: bool = True

    def __post_init__(self) -> None:
        if (
            self.maximum_stabilization_rounds != 8
            or self.required_consecutive_quiescent_rounds != 2
            or not self.reject_callbacks_from_uninstrumented_variants
            or not self.require_zero_timed_callbacks
        ):
            raise ValueError(
                "capture acceptance is fixed at eight stabilization rounds, two consecutive quiescent rounds, "
                "no uninstrumented callbacks, and zero timed callbacks"
            )


@dataclass(frozen=True)
class CaptureStabilizationRound:
    """One untimed traversal of every measurement order."""

    round_index: int
    quiescent: bool
    callback_checkpoints: tuple[CallbackCheckpoint, ...]


@dataclass(frozen=True)
class CaptureStabilization:
    """Bounded topology-matched command-buffer stabilization evidence."""

    stabilized: bool
    initial_counts: Mapping[str, int]
    final_counts: Mapping[str, int]
    rounds: tuple[CaptureStabilizationRound, ...]
    capture_sites: tuple[CaptureSiteManifest, ...]
    policy: CaptureAcceptancePolicy

    def to_json(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        return {
            "stabilized": self.stabilized,
            "initial_counts": dict(self.initial_counts),
            "final_counts": dict(self.final_counts),
            "policy": asdict(self.policy),
            "capture_sites": [manifest.to_json() for manifest in self.capture_sites],
            "rounds": [
                {
                    "round_index": round_result.round_index,
                    "quiescent": round_result.quiescent,
                    "callback_checkpoints": [asdict(checkpoint) for checkpoint in round_result.callback_checkpoints],
                }
                for round_result in self.rounds
            ],
        }


class CaptureBehavior(StrEnum):
    """Observed host-callback behavior around steady-state measurement."""

    CAPTURED_REPLAY = "captured_replay"
    FAILED_TO_STABILIZE = "failed_to_stabilize"
    PER_LOGICAL_CALL_FALLBACK = "per_logical_call_fallback"
    UNATTRIBUTED_CALLBACK = "unattributed_callback"
    MISSING_INITIAL_CAPTURE = "missing_initial_capture"
    INVALID_CHECKPOINT_SEQUENCE = "invalid_checkpoint_sequence"
    STEADY_STATE_RECAPTURE = "steady_state_recapture"


@dataclass(frozen=True)
class OrderCaptureSummary:
    """Timed callback deltas attributed to one complete counterbalanced order."""

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
    stabilization: CaptureStabilization
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
            "policy": asdict(self.stabilization.policy),
            "stabilization": self.stabilization.to_json(),
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


def _variant_orderings(variants: Sequence[CounterbalancedVariant]) -> tuple[tuple[CounterbalancedVariant, ...], ...]:
    if not variants:
        raise ValueError("counterbalanced execution requires at least one variant")
    names = tuple(variant.name for variant in variants)
    if len(set(names)) != len(names):
        raise ValueError(f"variant names must be unique: {names}")
    target_handlers: dict[str, str] = {}
    for variant in variants:
        if variant.capture_sites.executable != variant.name:
            raise ValueError(
                f"variant {variant.name!r} uses capture sites for executable {variant.capture_sites.executable!r}"
            )
        for target, handler in variant.capture_sites.target_handlers.items():
            previous = target_handlers.setdefault(target, handler)
            if previous != handler:
                raise ValueError(f"capture target {target!r} maps to both {previous!r} and {handler!r}")
    return tuple(itertools.permutations(variants))


def _manifest_handlers(variants: Sequence[CounterbalancedVariant]) -> tuple[str, ...]:
    return tuple(
        sorted({handler for variant in variants for handler in variant.capture_sites.handler_calls_per_execution})
    )


def _execute_variant(
    variant: CounterbalancedVariant,
    *,
    iterations: int,
    synchronize: Callable[[object], None],
) -> None:
    result = None
    for _ in range(iterations):
        result = variant.function()
    synchronize(result)


def _callback_checkpoint(
    variant: CounterbalancedVariant,
    *,
    sample_index: int,
    order: tuple[str, ...],
    iterations: int,
    before: Mapping[str, int],
    after: Mapping[str, int],
    manifest_handlers: tuple[str, ...],
) -> CallbackCheckpoint:
    before = _normalized_counts(before)
    after = _normalized_counts(after)
    handlers = tuple(sorted(set(manifest_handlers) | set(before) | set(after)))
    calls_per_execution = variant.capture_sites.handler_calls_per_execution
    return CallbackCheckpoint(
        sample_index=sample_index,
        order=order,
        variant=variant.name,
        before={handler: before.get(handler, 0) for handler in handlers},
        after={handler: after.get(handler, 0) for handler in handlers},
        delta={handler: after.get(handler, 0) - before.get(handler, 0) for handler in handlers},
        logical_handler_calls={handler: calls_per_execution.get(handler, 0) * iterations for handler in handlers},
    )


def stabilize_counterbalanced_variants(
    variants: Sequence[CounterbalancedVariant],
    *,
    iterations: int,
    synchronize: Callable[[object], None],
    read_handler_counts: Callable[[], Mapping[str, int]],
    policy: CaptureAcceptancePolicy = CaptureAcceptancePolicy(),
) -> CaptureStabilization:
    """Run the timed launch topology until two complete rounds are quiescent."""
    if iterations <= 0:
        raise ValueError("stabilization iterations must be positive")
    orderings = _variant_orderings(variants)
    manifest_handlers = _manifest_handlers(variants)
    initial_counts = _normalized_counts(read_handler_counts())
    rounds: list[CaptureStabilizationRound] = []
    consecutive_quiescent = 0
    for round_index in range(policy.maximum_stabilization_rounds):
        checkpoints: list[CallbackCheckpoint] = []
        for order_index, ordering in enumerate(orderings):
            order = tuple(variant.name for variant in ordering)
            sample_index = round_index * len(orderings) + order_index
            for variant in ordering:
                before = _normalized_counts(read_handler_counts())
                _execute_variant(variant, iterations=iterations, synchronize=synchronize)
                after = _normalized_counts(read_handler_counts())
                checkpoints.append(
                    _callback_checkpoint(
                        variant,
                        sample_index=sample_index,
                        order=order,
                        iterations=iterations,
                        before=before,
                        after=after,
                        manifest_handlers=manifest_handlers,
                    )
                )
        quiescent = all(not any(checkpoint.delta.values()) for checkpoint in checkpoints)
        rounds.append(
            CaptureStabilizationRound(
                round_index=round_index,
                quiescent=quiescent,
                callback_checkpoints=tuple(checkpoints),
            )
        )
        consecutive_quiescent = consecutive_quiescent + 1 if quiescent else 0
        if consecutive_quiescent == policy.required_consecutive_quiescent_rounds:
            break

    final_counts = _normalized_counts(read_handler_counts())
    return CaptureStabilization(
        stabilized=consecutive_quiescent == policy.required_consecutive_quiescent_rounds,
        initial_counts=initial_counts,
        final_counts=final_counts,
        rounds=tuple(rounds),
        capture_sites=tuple(variant.capture_sites for variant in variants),
        policy=policy,
    )


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
    if repeats <= 0 or iterations <= 0:
        raise ValueError("repeats and iterations must be positive")
    orderings = _variant_orderings(variants)
    if repeats % len(orderings):
        raise ValueError(f"repeats must be divisible by {len(orderings)} counterbalanced orders")
    manifest_handlers = _manifest_handlers(variants)
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
            _execute_variant(variant, iterations=iterations, synchronize=synchronize)
            elapsed = clock() - started
            after = _normalized_counts(read_handler_counts())
            checkpoint = _callback_checkpoint(
                variant,
                sample_index=sample_index,
                order=order,
                iterations=iterations,
                before=before,
                after=after,
                manifest_handlers=manifest_handlers,
            )
            checkpoints.append(checkpoint)
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


def _flatten_stabilization_checkpoints(stabilization: CaptureStabilization) -> tuple[CallbackCheckpoint, ...]:
    return tuple(checkpoint for round_result in stabilization.rounds for checkpoint in round_result.callback_checkpoints)


def _validate_checkpoint_sequence(
    initial_counts: Mapping[str, int],
    checkpoints: Sequence[CallbackCheckpoint],
) -> tuple[bool, dict[str, int]]:
    handlers = tuple(
        sorted(
            set(initial_counts)
            | {handler for checkpoint in checkpoints for handler in checkpoint.before}
            | {handler for checkpoint in checkpoints for handler in checkpoint.after}
            | {handler for checkpoint in checkpoints for handler in checkpoint.delta}
        )
    )
    expected = {handler: int(initial_counts.get(handler, 0)) for handler in handlers}
    invalid = False
    for checkpoint in checkpoints:
        before = _normalized_counts(checkpoint.before)
        after = _normalized_counts(checkpoint.after)
        delta = _normalized_deltas(checkpoint.delta)
        if any(before.get(handler, 0) != expected[handler] for handler in handlers):
            invalid = True
        computed = {handler: after.get(handler, 0) - before.get(handler, 0) for handler in handlers}
        if any(computed[handler] != delta.get(handler, 0) or computed[handler] < 0 for handler in handlers):
            invalid = True
        expected = {handler: after.get(handler, 0) for handler in handlers}
    return invalid, expected


def _has_unattributed_callback(checkpoints: Sequence[CallbackCheckpoint]) -> bool:
    return any(
        checkpoint.delta.get(handler, 0) > 0 and checkpoint.logical_handler_calls.get(handler, 0) == 0
        for checkpoint in checkpoints
        for handler in checkpoint.delta
    )


def _has_repeated_per_logical_call_fallback(stabilization: CaptureStabilization) -> bool:
    handlers = tuple(sorted({site.handler for manifest in stabilization.capture_sites for site in manifest.sites}))
    streak = {handler: 0 for handler in handlers}
    for round_result in stabilization.rounds:
        for handler in handlers:
            observed = sum(checkpoint.delta.get(handler, 0) for checkpoint in round_result.callback_checkpoints)
            logical = sum(
                checkpoint.logical_handler_calls.get(handler, 0) for checkpoint in round_result.callback_checkpoints
            )
            streak[handler] = streak[handler] + 1 if logical > 0 and observed >= logical else 0
            if streak[handler] >= 2:
                return True
    return False


def _order_summaries(checkpoints: Sequence[CallbackCheckpoint]) -> tuple[OrderCaptureSummary, ...]:
    handlers = tuple(sorted({handler for checkpoint in checkpoints for handler in checkpoint.delta}))
    first_sample_by_order: dict[tuple[str, ...], int] = {}
    samples_by_order: dict[tuple[str, ...], set[int]] = {}
    order_deltas: dict[tuple[str, ...], dict[str, int]] = {}
    recapture_samples: dict[tuple[str, ...], set[int]] = {}
    for checkpoint in checkpoints:
        order = checkpoint.order
        first_sample_by_order.setdefault(order, checkpoint.sample_index)
        samples_by_order.setdefault(order, set()).add(checkpoint.sample_index)
        order_deltas.setdefault(order, {handler: 0 for handler in handlers})
        recapture_samples.setdefault(order, set())
        for handler in handlers:
            observed = checkpoint.delta.get(handler, 0)
            order_deltas[order][handler] += observed
            if observed:
                recapture_samples[order].add(checkpoint.sample_index)
    return tuple(
        OrderCaptureSummary(
            order=order,
            first_sample_index=first_sample_by_order[order],
            sample_indices=tuple(sorted(samples_by_order[order])),
            handler_deltas=order_deltas[order],
            recapture_sample_indices=tuple(sorted(recapture_samples[order])),
        )
        for order in first_sample_by_order
    )


def assess_command_buffer_capture(
    stabilization: CaptureStabilization,
    timed_checkpoints: Sequence[CallbackCheckpoint],
) -> CaptureAssessment:
    """Accept only a stabilized manifest whose timed callback counts stay fixed."""
    calibration_checkpoints = _flatten_stabilization_checkpoints(stabilization)
    invalid_calibration, calibration_final = _validate_checkpoint_sequence(
        stabilization.initial_counts,
        calibration_checkpoints,
    )
    invalid_timed, timed_final = _validate_checkpoint_sequence(calibration_final, timed_checkpoints)
    invalid_sequence = (
        invalid_calibration
        or invalid_timed
        or calibration_final != _normalized_counts(stabilization.final_counts)
        or not timed_checkpoints
    )
    all_checkpoints = calibration_checkpoints + tuple(timed_checkpoints)
    unattributed_callback = _has_unattributed_callback(all_checkpoints)
    per_logical_call_fallback = _has_repeated_per_logical_call_fallback(stabilization)
    manifest_handlers = tuple(
        sorted({site.handler for manifest in stabilization.capture_sites for site in manifest.sites})
    )
    missing_capture = not manifest_handlers or any(
        calibration_final.get(handler, 0) == 0 for handler in manifest_handlers
    )
    total_timed_deltas = {
        handler: sum(checkpoint.delta.get(handler, 0) for checkpoint in timed_checkpoints)
        for handler in sorted(
            set(manifest_handlers) | {name for checkpoint in timed_checkpoints for name in checkpoint.delta}
        )
    }
    steady_state_recapture = any(total_timed_deltas.values())

    reasons: list[str] = []
    if invalid_sequence:
        reasons.append("callback checkpoints are empty, discontinuous, or internally inconsistent")
    if unattributed_callback and stabilization.policy.reject_callbacks_from_uninstrumented_variants:
        reasons.append("a callback occurred in a variant with no matching final-HLO capture site")
    if per_logical_call_fallback:
        reasons.append("callbacks matched logical handler calls in two consecutive stabilization rounds")
    if not stabilization.stabilized:
        reasons.append("capture counts did not reach two consecutive quiescent rounds within eight rounds")
    if missing_capture:
        reasons.append("at least one final-HLO capture site was never observed before timed measurement")
    if steady_state_recapture and stabilization.policy.require_zero_timed_callbacks:
        reasons.append("a host callback occurred after the capture counts reached a steady-state plateau")

    rejected = bool(reasons)
    if invalid_sequence:
        behavior = CaptureBehavior.INVALID_CHECKPOINT_SEQUENCE
    elif unattributed_callback and stabilization.policy.reject_callbacks_from_uninstrumented_variants:
        behavior = CaptureBehavior.UNATTRIBUTED_CALLBACK
    elif per_logical_call_fallback:
        behavior = CaptureBehavior.PER_LOGICAL_CALL_FALLBACK
    elif not stabilization.stabilized:
        behavior = CaptureBehavior.FAILED_TO_STABILIZE
    elif missing_capture:
        behavior = CaptureBehavior.MISSING_INITIAL_CAPTURE
    elif steady_state_recapture:
        behavior = CaptureBehavior.STEADY_STATE_RECAPTURE
    else:
        behavior = CaptureBehavior.CAPTURED_REPLAY

    return CaptureAssessment(
        accepted=not rejected,
        behavior=behavior,
        reasons=tuple(reasons),
        stabilization=stabilization,
        final_counts=timed_final,
        total_timed_deltas=total_timed_deltas,
        order_summaries=_order_summaries(timed_checkpoints),
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
