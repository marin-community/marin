# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed physical and numerical contracts for accelerator benchmarks."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from tile_lifetime.plan import NumericalPolicy


@dataclass(frozen=True)
class DenseBufferContract:
    """One named dense buffer's logical shape and physical representation."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    minor_to_major: tuple[int, ...]

    def __post_init__(self) -> None:
        derived = derive_dense_minor_to_major_layout(self.shape, self.strides)
        if self.minor_to_major != derived:
            raise ValueError(
                f"buffer {self.name!r} declares minor-to-major {self.minor_to_major}, "
                f"but shape/strides require {derived}"
            )

    @classmethod
    def from_strides(
        cls,
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
    ) -> "DenseBufferContract":
        """Construct a contract only when the strides describe dense storage."""
        return cls(
            name=name,
            dtype=dtype,
            shape=shape,
            strides=strides,
            minor_to_major=derive_dense_minor_to_major_layout(shape, strides),
        )


@dataclass(frozen=True)
class NumericalError:
    """Pointwise error summary against an independent semantic reference."""

    maximum_absolute_error: float
    mean_absolute_error: float
    finite: bool


@dataclass(frozen=True)
class NumericalAcceptanceContract:
    """Declared numerical and repeatability gate applied before timing."""

    numerical_policy: NumericalPolicy
    maximum_absolute_error: float
    mean_absolute_error: float
    require_determinism: bool = True

    def __post_init__(self) -> None:
        thresholds = (self.maximum_absolute_error, self.mean_absolute_error)
        if not all(np.isfinite(threshold) and threshold >= 0.0 for threshold in thresholds):
            raise ValueError(f"numerical acceptance thresholds must be finite and nonnegative, found {thresholds}")
        if self.numerical_policy is NumericalPolicy.BITWISE_EXACT and thresholds != (0.0, 0.0):
            raise ValueError("bitwise-exact acceptance requires zero error thresholds")


def derive_dense_minor_to_major_layout(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
) -> tuple[int, ...]:
    """Recover an unpadded dense layout from logical-axis element strides."""
    if len(shape) != len(strides):
        raise ValueError(f"dense layout rank mismatch: shape {shape}, strides {strides}")
    if any(extent <= 0 for extent in shape):
        raise ValueError(f"dense layout requires positive extents, found {shape}")
    if any(stride <= 0 for stride in strides):
        raise ValueError(f"dense layout requires positive strides, found {strides}")
    minor_to_major = tuple(sorted(range(len(shape)), key=strides.__getitem__))
    expected = _layout_strides(shape, minor_to_major)
    if expected != strides:
        raise ValueError(f"strides {strides} do not describe an unpadded dense layout for shape {shape}")
    return minor_to_major


def verify_dense_buffer_boundary(
    expected: tuple[DenseBufferContract, ...],
    observed: tuple[DenseBufferContract, ...],
    *,
    boundary_name: str,
) -> tuple[DenseBufferContract, ...]:
    """Reject a benchmark boundary with any logical or physical mismatch."""
    if expected != observed:
        raise ValueError(f"{boundary_name} dense buffer boundary mismatch: observed {observed}, expected {expected}")
    return observed


def numerical_error(actual: object, expected: object) -> NumericalError:
    """Compare two arrays in FP32 and retain nonfinite status explicitly."""
    actual_array = np.asarray(actual, dtype=np.float32)
    expected_array = np.asarray(expected, dtype=np.float32)
    if actual_array.shape != expected_array.shape:
        raise ValueError(f"numerical comparison shape mismatch: {actual_array.shape} != {expected_array.shape}")
    difference = np.abs(actual_array - expected_array)
    finite = bool(
        np.isfinite(actual_array).all() and np.isfinite(expected_array).all() and np.isfinite(difference).all()
    )
    return NumericalError(
        maximum_absolute_error=float(difference.max(initial=0.0)),
        mean_absolute_error=float(difference.mean()) if difference.size else 0.0,
        finite=finite,
    )


def verify_numerical_acceptance(
    comparisons: Mapping[str, NumericalError],
    *,
    deterministic: bool,
    contract: NumericalAcceptanceContract,
    boundary_name: str,
) -> Mapping[str, NumericalError]:
    """Reject invalid output values, excessive error, or nondeterminism."""
    if not comparisons:
        raise ValueError(f"{boundary_name} numerical acceptance requires at least one output")
    if contract.require_determinism and not deterministic:
        raise ValueError(f"{boundary_name} violates the declared deterministic execution contract")
    for name, error in comparisons.items():
        if not error.finite:
            raise ValueError(f"{boundary_name} output {name!r} contains nonfinite values or error")
        if error.maximum_absolute_error > contract.maximum_absolute_error:
            raise ValueError(
                f"{boundary_name} output {name!r} maximum absolute error {error.maximum_absolute_error} exceeds "
                f"{contract.maximum_absolute_error}"
            )
        if error.mean_absolute_error > contract.mean_absolute_error:
            raise ValueError(
                f"{boundary_name} output {name!r} mean absolute error {error.mean_absolute_error} exceeds "
                f"{contract.mean_absolute_error}"
            )
    return comparisons


def _layout_strides(shape: tuple[int, ...], minor_to_major: tuple[int, ...]) -> tuple[int, ...]:
    if tuple(sorted(minor_to_major)) != tuple(range(len(shape))):
        raise ValueError(f"minor-to-major layout must permute rank {len(shape)}, found {minor_to_major}")
    strides = [0] * len(shape)
    stride = 1
    for axis in minor_to_major:
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)
