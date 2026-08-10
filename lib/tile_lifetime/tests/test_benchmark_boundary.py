# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from tile_lifetime.benchmark_boundary import (
    BenchmarkRepeatabilityMode,
    BenchmarkRepeatabilityPolicy,
    DenseBufferContract,
    DTypeRepeatabilityTolerance,
    NumericalAcceptanceContract,
    benchmark_repeatability_report,
    derive_dense_minor_to_major_layout,
    numerical_error,
    verify_benchmark_repeatability,
    verify_dense_buffer_boundary,
    verify_numerical_acceptance,
)
from tile_lifetime.plan import NumericalPolicy


def test_dense_layout_derivation_matches_shape_mutations() -> None:
    cases = (
        ((1, 2048, 32, 128), (8388608, 4096, 128, 1)),
        ((2, 320, 7, 64), (143360, 448, 64, 1)),
        ((1, 2048, 8, 128), (2097152, 1024, 128, 1)),
    )

    for shape, strides in cases:
        assert derive_dense_minor_to_major_layout(shape, strides) == (3, 2, 1, 0)


def test_dense_layout_derivation_rejects_non_dense_or_mismatched_strides() -> None:
    with pytest.raises(ValueError, match="rank mismatch"):
        derive_dense_minor_to_major_layout((2, 3), (3,))
    with pytest.raises(ValueError, match="unpadded dense"):
        derive_dense_minor_to_major_layout((2, 3, 5), (30, 10, 1))


def _bshd_contract(name: str, shape: tuple[int, ...], *, derive_layout: bool = False) -> DenseBufferContract:
    _batch, sequence, heads, dimension = shape
    strides = (sequence * heads * dimension, heads * dimension, dimension, 1)
    if derive_layout:
        return DenseBufferContract.from_strides(name=name, dtype="bf16", shape=shape, strides=strides)
    return DenseBufferContract(
        name=name,
        dtype="bf16",
        shape=shape,
        strides=strides,
        minor_to_major=(3, 2, 1, 0),
    )


def test_dense_buffer_boundary_accepts_exact_training_inputs_and_results() -> None:
    inputs = (
        _bshd_contract("query", (1, 2048, 32, 128)),
        _bshd_contract("key", (1, 2048, 8, 128)),
        _bshd_contract("value", (1, 2048, 8, 128)),
        _bshd_contract("output_cotangent", (1, 2048, 32, 128)),
    )
    results = (
        _bshd_contract("forward_output", (1, 2048, 32, 128)),
        _bshd_contract("query_cotangent", (1, 2048, 32, 128)),
        _bshd_contract("key_cotangent", (1, 2048, 8, 128)),
        _bshd_contract("value_cotangent", (1, 2048, 8, 128)),
    )
    observed_inputs = tuple(_bshd_contract(contract.name, contract.shape, derive_layout=True) for contract in inputs)
    observed_results = tuple(_bshd_contract(contract.name, contract.shape, derive_layout=True) for contract in results)

    assert verify_dense_buffer_boundary(inputs, observed_inputs, boundary_name="inputs") == observed_inputs
    assert verify_dense_buffer_boundary(results, observed_results, boundary_name="results") == observed_results


@pytest.mark.parametrize(
    "observed",
    (
        (
            DenseBufferContract.from_strides(
                name="query",
                dtype="bf16",
                shape=(1, 2048, 32, 128),
                strides=(8388608, 128, 262144, 1),
            ),
        ),
        (_bshd_contract("query", (1, 1024, 32, 128), derive_layout=True),),
        (
            DenseBufferContract(
                name="query",
                dtype="fp32",
                shape=(1, 2048, 32, 128),
                strides=(8388608, 4096, 128, 1),
                minor_to_major=(3, 2, 1, 0),
            ),
        ),
    ),
)
def test_dense_buffer_boundary_rejects_physical_or_logical_mismatch(
    observed: tuple[DenseBufferContract, ...],
) -> None:
    expected = (_bshd_contract("query", (1, 2048, 32, 128)),)

    with pytest.raises(ValueError, match="dense buffer boundary mismatch"):
        verify_dense_buffer_boundary(expected, observed, boundary_name="inputs")


def test_numerical_acceptance_checks_declared_rounding_thresholds() -> None:
    error = numerical_error(
        np.array([0.0, 1.03125], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    )
    contract = NumericalAcceptanceContract(
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        maximum_absolute_error=0.125,
        mean_absolute_error=0.01,
    )

    assert error.maximum_absolute_error == 0.03125
    assert error.mean_absolute_error == 0.015625
    with pytest.raises(ValueError, match="mean absolute error"):
        verify_numerical_acceptance(
            {"output": error},
            contract=contract,
            boundary_name="candidate",
        )

    accepted = numerical_error(
        np.array([0.0, 1.015625], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    )
    comparisons = {"output": accepted}
    assert (
        verify_numerical_acceptance(
            comparisons,
            contract=contract,
            boundary_name="candidate",
        )
        == comparisons
    )


@pytest.mark.parametrize(
    ("actual", "message"),
    (
        (np.array([np.nan], dtype=np.float32), "nonfinite"),
        (np.array([0.25], dtype=np.float32), "maximum absolute error"),
    ),
)
def test_numerical_acceptance_fails_closed_before_timing(actual, message: str) -> None:
    error = numerical_error(actual, np.array([0.0], dtype=np.float32))
    contract = NumericalAcceptanceContract(
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        maximum_absolute_error=0.125,
        mean_absolute_error=0.01,
    )

    with pytest.raises(ValueError, match=message):
        verify_numerical_acceptance(
            {"output": error},
            contract=contract,
            boundary_name="candidate",
        )


def test_bitwise_acceptance_rejects_nonzero_thresholds() -> None:
    with pytest.raises(ValueError, match="bitwise-exact"):
        NumericalAcceptanceContract(
            numerical_policy=NumericalPolicy.BITWISE_EXACT,
            maximum_absolute_error=0.125,
            mean_absolute_error=0.01,
        )


def _rounding_acceptance() -> NumericalAcceptanceContract:
    return NumericalAcceptanceContract(
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        maximum_absolute_error=0.125,
        mean_absolute_error=0.05,
    )


def _bounded_bf16_policy(*, maximum: float = 0.125, mean: float = 0.05) -> BenchmarkRepeatabilityPolicy:
    return BenchmarkRepeatabilityPolicy(
        mode=BenchmarkRepeatabilityMode.BOUNDED_DRIFT,
        minimum_repeats=3,
        dtype_tolerances=(
            DTypeRepeatabilityTolerance(
                dtype="bf16",
                maximum_absolute_error=maximum,
                mean_absolute_error=mean,
            ),
        ),
    )


def test_bounded_repeatability_accepts_three_semantically_valid_repeats_and_serializes_all_pairs() -> None:
    reference = (
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([2.0, 3.0], dtype=np.float32),
    )
    repeats = (
        (reference[0].copy(), reference[1].copy()),
        (
            np.array([0.03125, 0.984375], dtype=np.float32),
            np.array([2.015625, 2.984375], dtype=np.float32),
        ),
        (
            np.array([-0.015625, 1.015625], dtype=np.float32),
            np.array([1.984375, 3.015625], dtype=np.float32),
        ),
    )
    report = benchmark_repeatability_report(
        ("output", "cotangent"),
        repeats,
        reference,
        output_dtypes={"output": "bf16", "cotangent": "bf16"},
        policy=_bounded_bf16_policy(),
    )

    assert not report.bitwise_repeat
    assert len(report.repeats) == 3
    assert len(report.pairwise_drift) == 3
    assert all(len(record.outputs) == 2 and all(output.sha256 for output in record.outputs) for record in report.repeats)
    assert all(len(pair.outputs) == 2 for pair in report.pairwise_drift)
    serialized = json.loads(json.dumps(asdict(report)))
    assert len(serialized["repeats"]) == 3
    assert len(serialized["pairwise_drift"]) == 3
    assert all(repeat["outputs"][0]["semantic_error"] for repeat in serialized["repeats"])
    assert (
        verify_benchmark_repeatability(
            report,
            numerical_acceptance=_rounding_acceptance(),
            boundary_name="expert oracle",
        )
        == report
    )


def test_bounded_repeatability_rejects_excess_pairwise_drift_before_timing() -> None:
    reference = (np.zeros(8, dtype=np.float32),)
    negative = reference[0].copy()
    positive = reference[0].copy()
    negative[0] = -0.1
    positive[0] = 0.1
    report = benchmark_repeatability_report(
        ("output",),
        (
            (negative,),
            (reference[0].copy(),),
            (positive,),
        ),
        reference,
        output_dtypes={"output": "bf16"},
        policy=_bounded_bf16_policy(),
    )

    with pytest.raises(ValueError, match="maximum repeat drift"):
        verify_benchmark_repeatability(
            report,
            numerical_acceptance=_rounding_acceptance(),
            boundary_name="expert oracle",
        )


def test_bounded_repeatability_rejects_a_semantically_invalid_repeat() -> None:
    reference = (np.array([0.0], dtype=np.float32),)
    report = benchmark_repeatability_report(
        ("output",),
        (
            (np.array([0.0], dtype=np.float32),),
            (np.array([0.0], dtype=np.float32),),
            (np.array([0.25], dtype=np.float32),),
        ),
        reference,
        output_dtypes={"output": "bf16"},
        policy=_bounded_bf16_policy(),
    )

    with pytest.raises(ValueError, match=r"repeat 2.*maximum absolute error"):
        verify_benchmark_repeatability(
            report,
            numerical_acceptance=_rounding_acceptance(),
            boundary_name="expert oracle",
        )


def test_bounded_repeatability_rejects_tolerance_weaker_than_semantic_contract() -> None:
    output = np.array([0.0], dtype=np.float32)
    report = benchmark_repeatability_report(
        ("output",),
        ((output,), (output.copy(),), (output.copy(),)),
        (output,),
        output_dtypes={"output": "bf16"},
        policy=_bounded_bf16_policy(maximum=0.25),
    )

    with pytest.raises(ValueError, match="weaker than the semantic error contract"):
        verify_benchmark_repeatability(
            report,
            numerical_acceptance=_rounding_acceptance(),
            boundary_name="expert oracle",
        )


def test_source_ordered_policy_rejects_bounded_oracle_drift() -> None:
    output = np.array([0.0], dtype=np.float32)
    report = benchmark_repeatability_report(
        ("output",),
        ((output,), (output.copy(),), (output.copy(),)),
        (output,),
        output_dtypes={"output": "bf16"},
        policy=_bounded_bf16_policy(maximum=0.0, mean=0.0),
    )
    source_ordered = NumericalAcceptanceContract(
        numerical_policy=NumericalPolicy.BITWISE_EXACT,
        maximum_absolute_error=0.0,
        mean_absolute_error=0.0,
    )

    with pytest.raises(ValueError, match="source-ordered"):
        verify_benchmark_repeatability(
            report,
            numerical_acceptance=source_ordered,
            boundary_name="expert oracle",
        )


def test_generated_bitwise_repeatability_rejects_mutated_output() -> None:
    reference = (np.array([0.0], dtype=np.float32),)
    report = benchmark_repeatability_report(
        ("output",),
        ((reference[0],), (np.array([0.03125], dtype=np.float32),)),
        reference,
        output_dtypes={"output": "bf16"},
        policy=BenchmarkRepeatabilityPolicy(
            mode=BenchmarkRepeatabilityMode.BITWISE,
            minimum_repeats=2,
        ),
    )

    with pytest.raises(ValueError, match="bitwise repeatability"):
        verify_benchmark_repeatability(
            report,
            numerical_acceptance=_rounding_acceptance(),
            boundary_name="generated Shuttle",
        )


def test_repeatability_verifier_rejects_incomplete_or_duplicate_evidence() -> None:
    output = np.array([0.0], dtype=np.float32)
    report = benchmark_repeatability_report(
        ("output",),
        ((output,), (output.copy(),), (output.copy(),)),
        (output,),
        output_dtypes={"output": "bf16"},
        policy=_bounded_bf16_policy(),
    )
    duplicate_repeats = replace(report, repeats=(report.repeats[0],) * 3)
    missing_output = replace(
        report,
        repeats=(replace(report.repeats[0], outputs=()), *report.repeats[1:]),
    )
    missing_pairs = replace(report, pairwise_drift=())

    for incomplete_report, message in (
        (duplicate_repeats, "repeat indices"),
        (missing_output, "do not match declared outputs"),
        (missing_pairs, "does not cover each repeat pair exactly once"),
    ):
        with pytest.raises(ValueError, match=message):
            verify_benchmark_repeatability(
                incomplete_report,
                numerical_acceptance=_rounding_acceptance(),
                boundary_name="expert oracle",
            )


def test_repeatability_verifier_recomputes_bitwise_summary_from_hashes() -> None:
    reference = (np.array([0.0], dtype=np.float32),)
    report = benchmark_repeatability_report(
        ("output",),
        ((reference[0],), (np.array([0.03125], dtype=np.float32),)),
        reference,
        output_dtypes={"output": "bf16"},
        policy=BenchmarkRepeatabilityPolicy(
            mode=BenchmarkRepeatabilityMode.BITWISE,
            minimum_repeats=2,
        ),
    )

    with pytest.raises(ValueError, match="contradicts recorded output hashes"):
        verify_benchmark_repeatability(
            replace(report, bitwise_repeat=True),
            numerical_acceptance=_rounding_acceptance(),
            boundary_name="generated Shuttle",
        )


def test_repeatability_verifier_rejects_nonfinite_bitwise_pair_evidence() -> None:
    output = np.array([0.0], dtype=np.float32)
    report = benchmark_repeatability_report(
        ("output",),
        ((output,), (output.copy(),)),
        (output,),
        output_dtypes={"output": "bf16"},
        policy=BenchmarkRepeatabilityPolicy(
            mode=BenchmarkRepeatabilityMode.BITWISE,
            minimum_repeats=2,
        ),
    )
    pair = report.pairwise_drift[0]
    error = replace(pair.outputs[0].error, finite=False, nonfinite_values=1)
    pair = replace(pair, outputs=(replace(pair.outputs[0], error=error),))

    with pytest.raises(ValueError, match="invalid drift"):
        verify_benchmark_repeatability(
            replace(report, pairwise_drift=(pair,)),
            numerical_acceptance=_rounding_acceptance(),
            boundary_name="generated Shuttle",
        )


def test_repeatability_verifier_rejects_impossible_error_summary() -> None:
    output = np.array([0.0], dtype=np.float32)
    report = benchmark_repeatability_report(
        ("output",),
        ((output,), (output.copy(),), (output.copy(),)),
        (output,),
        output_dtypes={"output": "bf16"},
        policy=_bounded_bf16_policy(),
    )
    pair = report.pairwise_drift[0]
    error = replace(pair.outputs[0].error, maximum_absolute_error=0.0, mean_absolute_error=0.01)
    pair = replace(pair, outputs=(replace(pair.outputs[0], error=error),))

    with pytest.raises(ValueError, match="mean absolute error greater than maximum"):
        verify_benchmark_repeatability(
            replace(report, pairwise_drift=(pair, *report.pairwise_drift[1:])),
            numerical_acceptance=_rounding_acceptance(),
            boundary_name="expert oracle",
        )


def test_bounded_repeatability_requires_three_repeats_and_explicit_dtype_tolerance() -> None:
    with pytest.raises(ValueError, match="at least three"):
        BenchmarkRepeatabilityPolicy(
            mode=BenchmarkRepeatabilityMode.BOUNDED_DRIFT,
            minimum_repeats=2,
            dtype_tolerances=(
                DTypeRepeatabilityTolerance(
                    dtype="bf16",
                    maximum_absolute_error=0.125,
                    mean_absolute_error=0.05,
                ),
            ),
        )
    with pytest.raises(ValueError, match="explicit dtype tolerance"):
        BenchmarkRepeatabilityPolicy(
            mode=BenchmarkRepeatabilityMode.BOUNDED_DRIFT,
            minimum_repeats=3,
        )
