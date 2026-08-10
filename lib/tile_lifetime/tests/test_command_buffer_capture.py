# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from tile_lifetime.command_buffer_capture import (
    CaptureAcceptanceError,
    CaptureAcceptancePolicy,
    CaptureBehavior,
    CaptureSite,
    CaptureSiteManifest,
    CounterbalancedVariant,
    assess_command_buffer_capture,
    derive_capture_site_manifest,
    measure_counterbalanced_variants,
    serialize_then_assess_capture,
    stabilize_counterbalanced_variants,
)


def _manifest(executable: str, *, occurrences: int = 1) -> CaptureSiteManifest:
    return CaptureSiteManifest(
        executable=executable,
        hlo_sha256=f"{executable}-hlo",
        sites=(CaptureSite(handler="handler", target="shuttle.test.handler", occurrences=occurrences),),
    )


def _variants(generated, reference=lambda: None, *, occurrences: int = 1):
    return (
        CounterbalancedVariant(
            name="generated",
            function=generated,
            capture_sites=_manifest("generated", occurrences=occurrences),
        ),
        CounterbalancedVariant(
            name="reference",
            function=reference,
            capture_sites=CaptureSiteManifest.uninstrumented("reference"),
        ),
    )


def _stabilize_and_measure(variants, handler_counts, *, iterations: int = 2):
    stabilization = stabilize_counterbalanced_variants(
        variants,
        iterations=iterations,
        synchronize=lambda _: None,
        read_handler_counts=lambda: handler_counts,
    )
    measurement = measure_counterbalanced_variants(
        variants,
        repeats=2,
        iterations=iterations,
        synchronize=lambda _: None,
        read_handler_counts=lambda: handler_counts,
    )
    return stabilization, measurement


def test_capture_acceptance_policy_cannot_be_weakened() -> None:
    with pytest.raises(ValueError, match="zero timed callbacks"):
        CaptureAcceptancePolicy(require_zero_timed_callbacks=False)


def test_capture_site_manifest_counts_final_hlo_targets() -> None:
    hlo = """HloModule capture_sites

ENTRY main {
  parameter = f32[] parameter(0)
  first = f32[] custom-call(parameter), custom_call_target="shuttle.forward.0"
  second = f32[] custom-call(first), custom_call_target="shuttle.forward.1"
  ROOT result = f32[] copy(second)
}
"""

    manifest = derive_capture_site_manifest(
        "generated",
        hlo,
        {"shuttle.forward.0": "handler", "shuttle.forward.1": "handler"},
    )

    assert manifest.handler_calls_per_execution == {"handler": 2}
    assert manifest.sites == (
        CaptureSite(handler="handler", target="shuttle.forward.0", occurrences=1),
        CaptureSite(handler="handler", target="shuttle.forward.1", occurrences=1),
    )


def test_capture_site_manifest_rejects_registered_target_absent_from_final_hlo() -> None:
    hlo = """HloModule capture_sites

ENTRY main {
  parameter = f32[] parameter(0)
  ROOT result = f32[] custom-call(parameter), custom_call_target="shuttle.actual"
}
"""

    with pytest.raises(ValueError, match="missing registered capture targets"):
        derive_capture_site_manifest("generated", hlo, {"shuttle.expected": "handler"})


def test_capture_site_manifest_validates_repeated_sites_in_full_composition() -> None:
    selected = {
        "shuttle.source_fold": 1,
        "shuttle.weighted_fold": 1,
        "shuttle.normalized_forward": 1,
        "shuttle.normalized_reverse": 1,
        "shuttle.low_rank_forward": 6,
        "shuttle.low_rank_reverse": 4,
    }
    unselected = {"shuttle.remaining": 9}
    targets = [target for target, count in {**selected, **unselected}.items() for _ in range(count)]
    instructions = []
    operand = "parameter"
    for index, target in enumerate(targets):
        name = f"call_{index}"
        instructions.append(f'  {name} = f32[] custom-call({operand}), custom_call_target="{target}"')
        operand = name
    hlo = "\n".join(
        (
            "HloModule full_composition",
            "",
            "ENTRY main {",
            "  parameter = f32[] parameter(0)",
            *instructions,
            f"  ROOT result = f32[] copy({operand})",
            "}",
        )
    )

    manifest = derive_capture_site_manifest(
        "generated",
        hlo,
        {target: target for target in selected},
        expected_target_occurrences=selected,
    )

    assert sum(site.occurrences for site in manifest.sites) == 14
    assert manifest.handler_calls_per_execution["shuttle.low_rank_forward"] == 6
    with pytest.raises(ValueError, match="multiplicities changed"):
        derive_capture_site_manifest(
            "generated",
            hlo,
            {target: target for target in selected},
            expected_target_occurrences={**selected, "shuttle.low_rank_forward": 5},
        )


def test_counterbalanced_measurement_uses_manifest_occurrences_for_logical_calls() -> None:
    handler_counts = {"handler": 0}
    clock_value = 0.0

    def generated() -> None:
        handler_counts["handler"] += 2

    def clock() -> float:
        nonlocal clock_value
        clock_value += 0.001
        return clock_value

    measurement = measure_counterbalanced_variants(
        _variants(generated, occurrences=2),
        repeats=2,
        iterations=3,
        synchronize=lambda _: None,
        read_handler_counts=lambda: handler_counts,
        clock=clock,
    )

    generated_checkpoints = tuple(
        checkpoint for checkpoint in measurement.callback_checkpoints if checkpoint.variant == "generated"
    )
    assert [checkpoint.delta for checkpoint in generated_checkpoints] == [{"handler": 6}, {"handler": 6}]
    assert [checkpoint.logical_handler_calls for checkpoint in generated_checkpoints] == [
        {"handler": 6},
        {"handler": 6},
    ]
    assert measurement.execution_order == (("generated", "reference"), ("reference", "generated"))


def test_finite_multi_stream_capture_stabilizes_before_timing() -> None:
    handler_counts = {"handler": 0}
    remaining_recordings = 3

    def generated() -> None:
        nonlocal remaining_recordings
        if remaining_recordings:
            handler_counts["handler"] += 1
            remaining_recordings -= 1

    stabilization, measurement = _stabilize_and_measure(_variants(generated), handler_counts)
    assessment = assess_command_buffer_capture(stabilization, measurement.callback_checkpoints)

    assert stabilization.stabilized
    assert [round_result.quiescent for round_result in stabilization.rounds] == [False, True, True]
    assert assessment.accepted
    assert assessment.behavior is CaptureBehavior.CAPTURED_REPLAY
    assert assessment.total_timed_deltas == {"handler": 0}


def test_nonquiescent_sublogical_callbacks_fail_stabilization() -> None:
    handler_counts = {"handler": 0}
    invocation = 0

    def generated() -> None:
        nonlocal invocation
        invocation += 1
        if invocation % 2 == 0:
            handler_counts["handler"] += 1

    stabilization, measurement = _stabilize_and_measure(_variants(generated), handler_counts, iterations=4)
    assessment = assess_command_buffer_capture(stabilization, measurement.callback_checkpoints)

    assert len(stabilization.rounds) == 8
    assert not stabilization.stabilized
    assert not assessment.accepted
    assert assessment.behavior is CaptureBehavior.FAILED_TO_STABILIZE


def test_per_logical_call_fallback_requires_two_nonquiescent_rounds() -> None:
    handler_counts = {"handler": 0}

    def generated() -> None:
        handler_counts["handler"] += 1

    stabilization, measurement = _stabilize_and_measure(_variants(generated), handler_counts, iterations=3)
    assessment = assess_command_buffer_capture(stabilization, measurement.callback_checkpoints)

    assert not assessment.accepted
    assert assessment.behavior is CaptureBehavior.PER_LOGICAL_CALL_FALLBACK


def test_callback_from_uninstrumented_variant_is_rejected() -> None:
    handler_counts = {"handler": 0}
    generated_recorded = False
    reference_recorded = False

    def generated() -> None:
        nonlocal generated_recorded
        if not generated_recorded:
            handler_counts["handler"] += 1
            generated_recorded = True

    def reference() -> None:
        nonlocal reference_recorded
        if not reference_recorded:
            handler_counts["handler"] += 1
            reference_recorded = True

    stabilization, measurement = _stabilize_and_measure(_variants(generated, reference), handler_counts)
    assessment = assess_command_buffer_capture(stabilization, measurement.callback_checkpoints)

    assert stabilization.stabilized
    assert not assessment.accepted
    assert assessment.behavior is CaptureBehavior.UNATTRIBUTED_CALLBACK


def test_callback_after_quiescent_plateau_rejects_timed_measurement() -> None:
    handler_counts = {"handler": 0}
    remaining_recordings = 1

    def generated() -> None:
        nonlocal remaining_recordings
        if remaining_recordings:
            handler_counts["handler"] += 1
            remaining_recordings -= 1

    variants = _variants(generated)
    stabilization = stabilize_counterbalanced_variants(
        variants,
        iterations=2,
        synchronize=lambda _: None,
        read_handler_counts=lambda: handler_counts,
    )
    remaining_recordings = 1
    measurement = measure_counterbalanced_variants(
        variants,
        repeats=2,
        iterations=2,
        synchronize=lambda _: None,
        read_handler_counts=lambda: handler_counts,
    )
    assessment = assess_command_buffer_capture(stabilization, measurement.callback_checkpoints)

    assert stabilization.stabilized
    assert not assessment.accepted
    assert assessment.behavior is CaptureBehavior.STEADY_STATE_RECAPTURE
    assert assessment.total_timed_deltas == {"handler": 1}


def test_rejected_steady_state_recapture_preserves_raw_timings(tmp_path: Path) -> None:
    handler_counts = {"handler": 0}
    remaining_recordings = 1

    def generated() -> None:
        nonlocal remaining_recordings
        if remaining_recordings:
            handler_counts["handler"] += 1
            remaining_recordings -= 1

    variants = _variants(generated)
    stabilization = stabilize_counterbalanced_variants(
        variants,
        iterations=2,
        synchronize=lambda _: None,
        read_handler_counts=lambda: handler_counts,
    )
    remaining_recordings = 1
    measurement = measure_counterbalanced_variants(
        variants,
        repeats=2,
        iterations=2,
        synchronize=lambda _: None,
        read_handler_counts=lambda: handler_counts,
    )
    output = tmp_path / "result.json"
    raw_samples = [0.031, 0.029]
    raw_result: Mapping[str, object] = {
        "measurements": {"generated": {"samples_ms": raw_samples, "median_ms": 0.030}},
        "execution_order": [list(order) for order in measurement.execution_order],
        "capture_stabilization": stabilization.to_json(),
    }

    def assess_after_raw_write():
        pending = json.loads(output.read_text())
        assert pending["measurements"]["generated"]["samples_ms"] == raw_samples
        assert pending["capture_acceptance"] == {"status": "pending"}
        return assess_command_buffer_capture(stabilization, measurement.callback_checkpoints)

    with pytest.raises(CaptureAcceptanceError) as error:
        serialize_then_assess_capture(output, raw_result, assess_after_raw_write)

    serialized = json.loads(output.read_text())
    assert serialized["measurements"]["generated"]["samples_ms"] == raw_samples
    assert serialized["execution_order"] == raw_result["execution_order"]
    assert serialized["capture_acceptance"]["status"] == "rejected"
    assert serialized["capture_acceptance"]["behavior"] == "steady_state_recapture"
    assert error.value.result["measurements"] == serialized["measurements"]
