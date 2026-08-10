# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from tile_lifetime.command_buffer_capture import (
    CallbackCheckpoint,
    CaptureAcceptanceError,
    CaptureBehavior,
    CounterbalancedVariant,
    assess_command_buffer_capture,
    measure_counterbalanced_variants,
    serialize_then_assess_capture,
)


def _checkpoint(
    *,
    sample_index: int,
    order: tuple[str, ...],
    variant: str,
    before: int,
    after: int,
    logical_calls: int,
) -> CallbackCheckpoint:
    return CallbackCheckpoint(
        sample_index=sample_index,
        order=order,
        variant=variant,
        before={"handler": before},
        after={"handler": after},
        delta={"handler": after - before},
        logical_handler_calls={"handler": logical_calls},
    )


def test_counterbalanced_measurement_attributes_counts_to_sample_order() -> None:
    handler_counts = {"forward": 0, "reverse": 0}
    clock_value = 0.0

    def generated() -> None:
        handler_counts["forward"] += 1
        handler_counts["reverse"] += 1

    def clock() -> float:
        nonlocal clock_value
        clock_value += 0.001
        return clock_value

    measurement = measure_counterbalanced_variants(
        (
            CounterbalancedVariant(
                name="generated",
                function=generated,
                handler_calls_per_execution={"forward": 1, "reverse": 1},
            ),
            CounterbalancedVariant(name="reference", function=lambda: None, handler_calls_per_execution={}),
        ),
        repeats=2,
        iterations=3,
        synchronize=lambda _: None,
        read_handler_counts=lambda: handler_counts,
        clock=clock,
    )

    assert measurement.execution_order == (("generated", "reference"), ("reference", "generated"))
    assert [checkpoint.variant for checkpoint in measurement.callback_checkpoints] == [
        "generated",
        "reference",
        "reference",
        "generated",
    ]
    assert [checkpoint.delta for checkpoint in measurement.callback_checkpoints] == [
        {"forward": 3, "reverse": 3},
        {"forward": 0, "reverse": 0},
        {"forward": 0, "reverse": 0},
        {"forward": 3, "reverse": 3},
    ]
    assert all(len(summary["samples_ms"]) == 2 for summary in measurement.measurements.values())


def test_capture_assessment_accepts_only_first_sample_recapture_for_each_order() -> None:
    first_order = ("generated", "reference")
    second_order = ("reference", "generated")
    checkpoints = (
        _checkpoint(
            sample_index=0,
            order=first_order,
            variant="generated",
            before=1,
            after=2,
            logical_calls=1000,
        ),
        _checkpoint(
            sample_index=0,
            order=first_order,
            variant="reference",
            before=2,
            after=2,
            logical_calls=0,
        ),
        _checkpoint(
            sample_index=1,
            order=second_order,
            variant="reference",
            before=2,
            after=2,
            logical_calls=0,
        ),
        _checkpoint(
            sample_index=1,
            order=second_order,
            variant="generated",
            before=2,
            after=3,
            logical_calls=1000,
        ),
        _checkpoint(
            sample_index=2,
            order=first_order,
            variant="generated",
            before=3,
            after=3,
            logical_calls=1000,
        ),
        _checkpoint(
            sample_index=2,
            order=first_order,
            variant="reference",
            before=3,
            after=3,
            logical_calls=0,
        ),
    )

    assessment = assess_command_buffer_capture({"handler": 1}, checkpoints)

    assert assessment.accepted
    assert assessment.behavior is CaptureBehavior.BOUNDED_ORDER_SPECIFIC_RECAPTURE
    assert [(summary.order, summary.handler_deltas) for summary in assessment.order_summaries] == [
        (first_order, {"handler": 1}),
        (second_order, {"handler": 1}),
    ]


def test_capture_assessment_rejects_recapture_after_order_is_established() -> None:
    order = ("generated", "reference")
    checkpoints = (
        _checkpoint(
            sample_index=0,
            order=order,
            variant="generated",
            before=1,
            after=2,
            logical_calls=1000,
        ),
        _checkpoint(
            sample_index=1,
            order=order,
            variant="generated",
            before=2,
            after=3,
            logical_calls=1000,
        ),
    )

    assessment = assess_command_buffer_capture({"handler": 1}, checkpoints)

    assert not assessment.accepted
    assert assessment.behavior is CaptureBehavior.UNBOUNDED_RECAPTURE


def test_rejected_capture_serializes_raw_timings_before_assessment(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    raw_samples = [0.031, 0.029, 0.030]
    raw_result: Mapping[str, object] = {
        "measurements": {"generated": {"samples_ms": raw_samples, "median_ms": 0.030}},
        "execution_order": [["generated"]] * len(raw_samples),
    }
    checkpoints = (
        _checkpoint(
            sample_index=0,
            order=("generated",),
            variant="generated",
            before=1,
            after=1001,
            logical_calls=1000,
        ),
    )

    def assess_after_raw_write():
        pending = json.loads(output.read_text())
        assert pending["measurements"]["generated"]["samples_ms"] == raw_samples
        assert pending["capture_acceptance"] == {"status": "pending"}
        return assess_command_buffer_capture({"handler": 1}, checkpoints)

    with pytest.raises(CaptureAcceptanceError) as error:
        serialize_then_assess_capture(output, raw_result, assess_after_raw_write)

    serialized = json.loads(output.read_text())
    assert serialized["measurements"]["generated"]["samples_ms"] == raw_samples
    assert serialized["execution_order"] == raw_result["execution_order"]
    assert serialized["capture_acceptance"]["status"] == "rejected"
    assert serialized["capture_acceptance"]["behavior"] == "per_logical_call_fallback"
    assert error.value.result["measurements"] == serialized["measurements"]
