"""Array comparison utilities for testing model implementations."""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass
class ComparisonResult:
    """Result of comparing two arrays."""

    name: str
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    diff_locations: tuple[tuple[int, ...], ...]  # Indices where diff is largest
    expected_shape: tuple[int, ...]
    actual_shape: tuple[int, ...]
    atol: float
    rtol: float
    failure_summary: str | None = None

    def __str__(self) -> str:
        if self.passed:
            return f"✓ {self.name}: PASS (max diff: {self.max_abs_diff:.2e})"
        return f"✗ {self.name}: FAIL\n{self.failure_summary}"

    def __repr__(self) -> str:
        return self.__str__()


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert array-like to numpy."""
    if hasattr(arr, "__array__"):
        return np.asarray(arr)
    return np.array(arr)


def compare(
    expected: Any,
    actual: Any,
    name: str = "comparison",
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> ComparisonResult:
    """Compare two arrays, return detailed result.

    Args:
        expected: The expected array (e.g., from HuggingFace)
        actual: The actual array (e.g., from Grug implementation)
        name: Name for this comparison (for reporting)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        ComparisonResult with detailed comparison information
    """
    expected_np = _to_numpy(expected)
    actual_np = _to_numpy(actual)

    expected_shape = expected_np.shape
    actual_shape = actual_np.shape

    # Check shapes first
    if expected_shape != actual_shape:
        return ComparisonResult(
            name=name,
            passed=False,
            max_abs_diff=float("inf"),
            max_rel_diff=float("inf"),
            diff_locations=(),
            expected_shape=expected_shape,
            actual_shape=actual_shape,
            atol=atol,
            rtol=rtol,
            failure_summary=f"Shape mismatch: expected {expected_shape}, got {actual_shape}",
        )

    # Compute differences
    abs_diff = np.abs(expected_np - actual_np)
    max_abs_diff = float(np.max(abs_diff))

    # Relative diff (avoid division by zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = abs_diff / (np.abs(expected_np) + 1e-12)
        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0.0)
    max_rel_diff = float(np.max(rel_diff))

    # Check if within tolerance
    passed = np.allclose(expected_np, actual_np, atol=atol, rtol=rtol)

    # Find locations of largest differences
    flat_idx = np.argsort(abs_diff.ravel())[-5:][::-1]  # Top 5
    diff_locations = tuple(np.unravel_index(idx, abs_diff.shape) for idx in flat_idx)
    diff_locations = tuple(tuple(int(i) for i in loc) for loc in diff_locations)

    failure_summary = None
    if not passed:
        lines = [
            f"Arrays differ beyond tolerance (atol={atol}, rtol={rtol})",
            f"  Max absolute diff: {max_abs_diff:.6e}",
            f"  Max relative diff: {max_rel_diff:.6e}",
            f"  Shape: {expected_shape}",
            "  Largest differences at:",
        ]
        for loc in diff_locations[:3]:
            exp_val = expected_np[loc]
            act_val = actual_np[loc]
            diff = abs(exp_val - act_val)
            lines.append(f"    {loc}: expected={exp_val:.6e}, actual={act_val:.6e}, diff={diff:.6e}")
        failure_summary = "\n".join(lines)

    return ComparisonResult(
        name=name,
        passed=passed,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        diff_locations=diff_locations,
        expected_shape=expected_shape,
        actual_shape=actual_shape,
        atol=atol,
        rtol=rtol,
        failure_summary=failure_summary,
    )
