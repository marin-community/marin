# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from tile_lifetime.benchmark_boundary import (
    DenseBufferContract,
    NumericalAcceptanceContract,
    derive_dense_minor_to_major_layout,
    numerical_error,
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
            deterministic=True,
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
            deterministic=True,
            contract=contract,
            boundary_name="candidate",
        )
        == comparisons
    )


@pytest.mark.parametrize(
    ("actual", "deterministic", "message"),
    (
        (np.array([np.nan], dtype=np.float32), True, "nonfinite"),
        (np.array([0.0], dtype=np.float32), False, "deterministic"),
        (np.array([0.25], dtype=np.float32), True, "maximum absolute error"),
    ),
)
def test_numerical_acceptance_fails_closed_before_timing(actual, deterministic: bool, message: str) -> None:
    error = numerical_error(actual, np.array([0.0], dtype=np.float32))
    contract = NumericalAcceptanceContract(
        numerical_policy=NumericalPolicy.ALLOW_ROUNDING_REORDER,
        maximum_absolute_error=0.125,
        mean_absolute_error=0.01,
    )

    with pytest.raises(ValueError, match=message):
        verify_numerical_acceptance(
            {"output": error},
            deterministic=deterministic,
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
