# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed physical and numerical contracts for accelerator benchmarks."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

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
    nonfinite_values: int


@dataclass(frozen=True)
class NumericalAcceptanceContract:
    """Declared semantic-reference error gate applied before timing."""

    numerical_policy: NumericalPolicy
    maximum_absolute_error: float
    mean_absolute_error: float

    def __post_init__(self) -> None:
        thresholds = (self.maximum_absolute_error, self.mean_absolute_error)
        if not all(np.isfinite(threshold) and threshold >= 0.0 for threshold in thresholds):
            raise ValueError(f"numerical acceptance thresholds must be finite and nonnegative, found {thresholds}")
        if self.numerical_policy is NumericalPolicy.BITWISE_EXACT and thresholds != (0.0, 0.0):
            raise ValueError("bitwise-exact acceptance requires zero error thresholds")


class BenchmarkRepeatabilityMode(StrEnum):
    """How repeated executions establish benchmark admissibility."""

    BITWISE = "bitwise"
    BOUNDED_DRIFT = "bounded_drift"


@dataclass(frozen=True)
class DTypeRepeatabilityTolerance:
    """Precommitted repeat-drift bounds for one physical output dtype."""

    dtype: str
    maximum_absolute_error: float
    mean_absolute_error: float

    def __post_init__(self) -> None:
        if not self.dtype:
            raise ValueError("repeatability tolerance dtype must not be empty")
        thresholds = (self.maximum_absolute_error, self.mean_absolute_error)
        if not all(np.isfinite(threshold) and threshold >= 0.0 for threshold in thresholds):
            raise ValueError(f"repeatability thresholds must be finite and nonnegative, found {thresholds}")


@dataclass(frozen=True)
class BenchmarkRepeatabilityPolicy:
    """Benchmark repeat policy independent of implementation determinism."""

    mode: BenchmarkRepeatabilityMode
    minimum_repeats: int
    dtype_tolerances: tuple[DTypeRepeatabilityTolerance, ...] = ()

    def __post_init__(self) -> None:
        if self.minimum_repeats < 2:
            raise ValueError("repeatability policy requires at least two repeats")
        tolerance_dtypes = tuple(tolerance.dtype for tolerance in self.dtype_tolerances)
        if len(tolerance_dtypes) != len(set(tolerance_dtypes)):
            raise ValueError(f"repeatability policy repeats dtype tolerances: {tolerance_dtypes}")
        if self.mode is BenchmarkRepeatabilityMode.BITWISE:
            if self.dtype_tolerances:
                raise ValueError("bitwise repeatability does not accept drift tolerances")
            return
        if self.minimum_repeats < 3:
            raise ValueError("bounded-drift repeatability requires at least three repeats")
        if not self.dtype_tolerances:
            raise ValueError("bounded-drift repeatability requires an explicit dtype tolerance")


@dataclass(frozen=True)
class BenchmarkOutputComparison:
    """One output's hash and semantic-reference error for one execution."""

    output_name: str
    sha256: str
    semantic_error: NumericalError


@dataclass(frozen=True)
class BenchmarkRepeatRecord:
    """Serializable evidence from one untimed benchmark execution."""

    repeat_index: int
    combined_sha256: str
    outputs: tuple[BenchmarkOutputComparison, ...]


@dataclass(frozen=True)
class BenchmarkOutputDrift:
    """One output's numerical drift between two untimed executions."""

    output_name: str
    error: NumericalError


@dataclass(frozen=True)
class BenchmarkPairwiseDrift:
    """Serializable pairwise drift evidence between two repeats."""

    left_repeat_index: int
    right_repeat_index: int
    outputs: tuple[BenchmarkOutputDrift, ...]


@dataclass(frozen=True)
class BenchmarkRepeatabilityReport:
    """Hashes and errors needed to decide benchmark-oracle admissibility."""

    policy: BenchmarkRepeatabilityPolicy
    output_dtypes: tuple[tuple[str, str], ...]
    bitwise_repeat: bool
    repeats: tuple[BenchmarkRepeatRecord, ...]
    pairwise_drift: tuple[BenchmarkPairwiseDrift, ...]


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
    finite_difference = difference[np.isfinite(difference)]
    nonfinite_values = int(
        np.count_nonzero(~np.isfinite(actual_array))
        + np.count_nonzero(~np.isfinite(expected_array))
        + np.count_nonzero(~np.isfinite(difference))
    )
    return NumericalError(
        maximum_absolute_error=float(finite_difference.max(initial=0.0)),
        mean_absolute_error=float(finite_difference.mean()) if finite_difference.size else 0.0,
        finite=nonfinite_values == 0,
        nonfinite_values=nonfinite_values,
    )


def verify_numerical_acceptance(
    comparisons: Mapping[str, NumericalError],
    *,
    contract: NumericalAcceptanceContract,
    boundary_name: str,
) -> Mapping[str, NumericalError]:
    """Reject invalid output values or excessive semantic-reference error."""
    if not comparisons:
        raise ValueError(f"{boundary_name} numerical acceptance requires at least one output")
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


def benchmark_repeatability_report(
    output_names: tuple[str, ...],
    repeats: Sequence[Sequence[object]],
    semantic_reference: Sequence[object],
    *,
    output_dtypes: Mapping[str, str],
    policy: BenchmarkRepeatabilityPolicy,
) -> BenchmarkRepeatabilityReport:
    """Build complete hash, semantic-error, and pairwise-drift evidence."""
    if not output_names:
        raise ValueError("benchmark repeatability requires at least one output")
    if len(set(output_names)) != len(output_names):
        raise ValueError(f"benchmark repeatability repeats output names: {output_names}")
    if set(output_dtypes) != set(output_names):
        raise ValueError(f"output dtype names {tuple(output_dtypes)} do not match outputs {output_names}")
    reference = tuple(semantic_reference)
    if len(reference) != len(output_names):
        raise ValueError(f"semantic reference has {len(reference)} outputs, expected {len(output_names)}")

    materialized_repeats = tuple(tuple(outputs) for outputs in repeats)
    records = tuple(
        _repeat_record(output_names, outputs, reference, repeat_index=repeat_index)
        for repeat_index, outputs in enumerate(materialized_repeats)
    )
    pairwise_drift = tuple(
        _pairwise_drift(output_names, materialized_repeats[left], materialized_repeats[right], left, right)
        for left in range(len(materialized_repeats))
        for right in range(left + 1, len(materialized_repeats))
    )
    first_hashes = tuple(output.sha256 for output in records[0].outputs) if records else ()
    bitwise_repeat = bool(records) and all(
        tuple(output.sha256 for output in record.outputs) == first_hashes for record in records[1:]
    )
    return BenchmarkRepeatabilityReport(
        policy=policy,
        output_dtypes=tuple((name, output_dtypes[name]) for name in output_names),
        bitwise_repeat=bitwise_repeat,
        repeats=records,
        pairwise_drift=pairwise_drift,
    )


def verify_benchmark_repeatability(
    report: BenchmarkRepeatabilityReport,
    *,
    numerical_acceptance: NumericalAcceptanceContract,
    boundary_name: str,
) -> BenchmarkRepeatabilityReport:
    """Reject a candidate whose pre-timing repeat evidence is inadmissible."""
    policy = report.policy
    output_names, output_dtypes = _verify_repeatability_report_structure(report, boundary_name=boundary_name)
    if len(report.repeats) < policy.minimum_repeats:
        raise ValueError(
            f"{boundary_name} repeatability requires at least {policy.minimum_repeats} repeats, "
            f"found {len(report.repeats)}"
        )
    for repeat in report.repeats:
        verify_numerical_acceptance(
            {output.output_name: output.semantic_error for output in repeat.outputs},
            contract=numerical_acceptance,
            boundary_name=f"{boundary_name} repeat {repeat.repeat_index}",
        )
    if policy.mode is BenchmarkRepeatabilityMode.BITWISE:
        if not report.bitwise_repeat:
            raise ValueError(f"{boundary_name} violates the declared bitwise repeatability policy")
        for pair in report.pairwise_drift:
            for output in pair.outputs:
                if (
                    not output.error.finite
                    or output.error.maximum_absolute_error != 0.0
                    or output.error.mean_absolute_error != 0.0
                ):
                    raise ValueError(
                        f"{boundary_name} bitwise report has invalid drift for output {output.output_name!r}"
                    )
        return report
    if numerical_acceptance.numerical_policy is not NumericalPolicy.ALLOW_ROUNDING_REORDER:
        raise ValueError(f"{boundary_name} cannot use bounded drift for a source-ordered numerical contract")

    tolerance_by_dtype = {tolerance.dtype: tolerance for tolerance in policy.dtype_tolerances}
    for name in output_names:
        dtype = output_dtypes[name]
        if dtype not in tolerance_by_dtype:
            raise ValueError(f"{boundary_name} has no repeatability tolerance for output {name!r} dtype {dtype!r}")
        tolerance = tolerance_by_dtype[dtype]
        if tolerance.maximum_absolute_error > numerical_acceptance.maximum_absolute_error:
            raise ValueError(
                f"{boundary_name} maximum repeatability tolerance for dtype {dtype!r} is weaker than "
                "the semantic error contract"
            )
        if tolerance.mean_absolute_error > numerical_acceptance.mean_absolute_error:
            raise ValueError(
                f"{boundary_name} mean repeatability tolerance for dtype {dtype!r} is weaker than "
                "the semantic error contract"
            )
    for pair in report.pairwise_drift:
        for output in pair.outputs:
            error = output.error
            tolerance = tolerance_by_dtype[output_dtypes[output.output_name]]
            pair_name = f"repeats {pair.left_repeat_index} and {pair.right_repeat_index} output {output.output_name!r}"
            if not error.finite:
                raise ValueError(f"{boundary_name} {pair_name} has nonfinite repeat drift")
            if error.maximum_absolute_error > tolerance.maximum_absolute_error:
                raise ValueError(
                    f"{boundary_name} {pair_name} maximum repeat drift {error.maximum_absolute_error} exceeds "
                    f"{tolerance.maximum_absolute_error}"
                )
            if error.mean_absolute_error > tolerance.mean_absolute_error:
                raise ValueError(
                    f"{boundary_name} {pair_name} mean repeat drift {error.mean_absolute_error} exceeds "
                    f"{tolerance.mean_absolute_error}"
                )
    return report


def _verify_repeatability_report_structure(
    report: BenchmarkRepeatabilityReport,
    *,
    boundary_name: str,
) -> tuple[tuple[str, ...], dict[str, str]]:
    output_names = tuple(name for name, _dtype in report.output_dtypes)
    if not output_names:
        raise ValueError(f"{boundary_name} repeatability report has no outputs")
    if len(set(output_names)) != len(output_names):
        raise ValueError(f"{boundary_name} repeatability report repeats output names: {output_names}")
    output_dtypes = dict(report.output_dtypes)
    if any(not dtype for dtype in output_dtypes.values()):
        raise ValueError(f"{boundary_name} repeatability report has an empty output dtype")

    repeat_indices = tuple(repeat.repeat_index for repeat in report.repeats)
    expected_repeat_indices = tuple(range(len(report.repeats)))
    if repeat_indices != expected_repeat_indices:
        raise ValueError(
            f"{boundary_name} repeat indices {repeat_indices} do not match the complete sequence "
            f"{expected_repeat_indices}"
        )
    for repeat in report.repeats:
        _verify_sha256(repeat.combined_sha256, boundary_name=f"{boundary_name} repeat {repeat.repeat_index} combined")
        repeat_output_names = tuple(output.output_name for output in repeat.outputs)
        if repeat_output_names != output_names:
            raise ValueError(
                f"{boundary_name} repeat {repeat.repeat_index} outputs {repeat_output_names} "
                f"do not match declared outputs {output_names}"
            )
        for output in repeat.outputs:
            _verify_sha256(
                output.sha256,
                boundary_name=f"{boundary_name} repeat {repeat.repeat_index} output {output.output_name!r}",
            )
            _verify_numerical_error_structure(
                output.semantic_error,
                boundary_name=f"{boundary_name} repeat {repeat.repeat_index} output {output.output_name!r}",
            )

    expected_pairs = {
        (left, right) for left in expected_repeat_indices for right in expected_repeat_indices if left < right
    }
    actual_pairs = tuple((pair.left_repeat_index, pair.right_repeat_index) for pair in report.pairwise_drift)
    if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
        raise ValueError(
            f"{boundary_name} pairwise evidence {actual_pairs} does not cover each repeat pair exactly once: "
            f"{tuple(sorted(expected_pairs))}"
        )
    for pair in report.pairwise_drift:
        pair_output_names = tuple(output.output_name for output in pair.outputs)
        if pair_output_names != output_names:
            raise ValueError(
                f"{boundary_name} repeats {pair.left_repeat_index} and {pair.right_repeat_index} outputs "
                f"{pair_output_names} do not match declared outputs {output_names}"
            )
        for output in pair.outputs:
            _verify_numerical_error_structure(
                output.error,
                boundary_name=(
                    f"{boundary_name} repeats {pair.left_repeat_index} and {pair.right_repeat_index} "
                    f"output {output.output_name!r}"
                ),
            )

    first_hashes = tuple(output.sha256 for output in report.repeats[0].outputs) if report.repeats else ()
    bitwise_repeat = bool(report.repeats) and all(
        tuple(output.sha256 for output in repeat.outputs) == first_hashes for repeat in report.repeats[1:]
    )
    if report.bitwise_repeat != bitwise_repeat:
        raise ValueError(f"{boundary_name} bitwise summary {report.bitwise_repeat} contradicts recorded output hashes")
    if bitwise_repeat:
        combined_hashes = {repeat.combined_sha256 for repeat in report.repeats}
        if len(combined_hashes) != 1:
            raise ValueError(f"{boundary_name} bitwise output hashes contradict recorded combined hashes")
    return output_names, output_dtypes


def _verify_sha256(value: str, *, boundary_name: str) -> None:
    if len(value) != 64 or value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{boundary_name} has an invalid SHA-256 digest: {value!r}")


def _verify_numerical_error_structure(error: NumericalError, *, boundary_name: str) -> None:
    metrics = (error.maximum_absolute_error, error.mean_absolute_error)
    if not all(np.isfinite(metric) and metric >= 0.0 for metric in metrics):
        raise ValueError(f"{boundary_name} has invalid numerical error metrics: {metrics}")
    if error.mean_absolute_error > error.maximum_absolute_error:
        raise ValueError(f"{boundary_name} has mean absolute error greater than maximum absolute error: {metrics}")
    if error.nonfinite_values < 0 or error.finite != (error.nonfinite_values == 0):
        raise ValueError(
            f"{boundary_name} has inconsistent finite evidence: finite={error.finite}, "
            f"nonfinite_values={error.nonfinite_values}"
        )


def _repeat_record(
    output_names: tuple[str, ...],
    outputs: tuple[object, ...],
    semantic_reference: tuple[object, ...],
    *,
    repeat_index: int,
) -> BenchmarkRepeatRecord:
    if len(outputs) != len(output_names):
        raise ValueError(f"repeat {repeat_index} has {len(outputs)} outputs, expected {len(output_names)}")
    digest = hashlib.sha256()
    comparisons = []
    for name, output, reference in zip(output_names, outputs, semantic_reference, strict=True):
        array = np.asarray(output)
        payload = array.tobytes()
        output_hash = hashlib.sha256(payload).hexdigest()
        digest.update(payload)
        comparisons.append(
            BenchmarkOutputComparison(
                output_name=name,
                sha256=output_hash,
                semantic_error=numerical_error(array, reference),
            )
        )
    return BenchmarkRepeatRecord(
        repeat_index=repeat_index,
        combined_sha256=digest.hexdigest(),
        outputs=tuple(comparisons),
    )


def _pairwise_drift(
    output_names: tuple[str, ...],
    left_outputs: tuple[object, ...],
    right_outputs: tuple[object, ...],
    left_repeat_index: int,
    right_repeat_index: int,
) -> BenchmarkPairwiseDrift:
    if len(left_outputs) != len(output_names) or len(right_outputs) != len(output_names):
        raise ValueError("pairwise repeat drift output count does not match output names")
    return BenchmarkPairwiseDrift(
        left_repeat_index=left_repeat_index,
        right_repeat_index=right_repeat_index,
        outputs=tuple(
            BenchmarkOutputDrift(output_name=name, error=numerical_error(left, right))
            for name, left, right in zip(output_names, left_outputs, right_outputs, strict=True)
        ),
    )


def _layout_strides(shape: tuple[int, ...], minor_to_major: tuple[int, ...]) -> tuple[int, ...]:
    if tuple(sorted(minor_to_major)) != tuple(range(len(shape))):
        raise ValueError(f"minor-to-major layout must permute rank {len(shape)}, found {minor_to_major}")
    strides = [0] * len(shape)
    stride = 1
    for axis in minor_to_major:
        strides[axis] = stride
        stride *= shape[axis]
    return tuple(strides)
