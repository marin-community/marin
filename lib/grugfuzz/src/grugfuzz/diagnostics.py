"""Diagnostic utilities for debugging array differences."""

from typing import Any

import numpy as np


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert array-like to numpy."""
    if hasattr(arr, "__array__"):
        return np.asarray(arr)
    return np.array(arr)


def diagnose_diff(
    expected: Any,
    actual: Any,
    name: str = "comparison",
    num_worst: int = 5,
) -> str:
    """Generate detailed diagnostic for a failed comparison.

    Returns human-readable string with:
    - Shape comparison
    - Statistics (mean, std, min, max for both)
    - Largest differences and their locations
    - Suggestions based on diff patterns

    Args:
        expected: The expected array (e.g., from HuggingFace)
        actual: The actual array (e.g., from Grug implementation)
        name: Name for this comparison
        num_worst: Number of worst differences to show

    Returns:
        Detailed diagnostic string
    """
    expected_np = _to_numpy(expected)
    actual_np = _to_numpy(actual)

    lines = [f"=== Diagnostic for '{name}' ===", ""]

    # Shape comparison
    lines.append("SHAPES:")
    lines.append(f"  Expected: {expected_np.shape}")
    lines.append(f"  Actual:   {actual_np.shape}")

    if expected_np.shape != actual_np.shape:
        lines.append("")
        lines.append("⚠️  SHAPE MISMATCH - Cannot compute detailed diff")
        lines.append("")
        lines.append(_suggest_shape_fix(expected_np.shape, actual_np.shape))
        return "\n".join(lines)

    lines.append("")

    # Statistics
    lines.append("STATISTICS:")
    lines.append("  Expected:")
    lines.append(f"    mean={expected_np.mean():.6e}, std={expected_np.std():.6e}")
    lines.append(f"    min={expected_np.min():.6e}, max={expected_np.max():.6e}")
    lines.append("  Actual:")
    lines.append(f"    mean={actual_np.mean():.6e}, std={actual_np.std():.6e}")
    lines.append(f"    min={actual_np.min():.6e}, max={actual_np.max():.6e}")
    lines.append("")

    # Difference analysis
    abs_diff = np.abs(expected_np - actual_np)
    rel_diff = abs_diff / (np.abs(expected_np) + 1e-12)

    lines.append("DIFFERENCE SUMMARY:")
    lines.append(f"  Max absolute diff: {abs_diff.max():.6e}")
    lines.append(f"  Mean absolute diff: {abs_diff.mean():.6e}")
    lines.append(f"  Max relative diff: {rel_diff.max():.6e}")
    lines.append(f"  Mean relative diff: {rel_diff.mean():.6e}")

    # Percentage of values that differ
    num_diff = np.sum(abs_diff > 1e-7)
    pct_diff = 100 * num_diff / abs_diff.size
    lines.append(f"  Values differing (>1e-7): {num_diff}/{abs_diff.size} ({pct_diff:.1f}%)")
    lines.append("")

    # Worst differences
    flat_idx = np.argsort(abs_diff.ravel())[-num_worst:][::-1]
    lines.append(f"TOP {num_worst} LARGEST DIFFERENCES:")
    for i, idx in enumerate(flat_idx):
        loc = np.unravel_index(idx, abs_diff.shape)
        loc_str = str(tuple(int(x) for x in loc))
        exp_val = expected_np[loc]
        act_val = actual_np[loc]
        diff = abs_diff[loc]
        rel = rel_diff[loc]
        lines.append(f"  {i+1}. {loc_str}")
        lines.append(f"     expected={exp_val:.6e}, actual={act_val:.6e}")
        lines.append(f"     abs_diff={diff:.6e}, rel_diff={rel:.6e}")
    lines.append("")

    # Pattern analysis and suggestions
    suggestions = _analyze_diff_pattern(expected_np, actual_np, abs_diff)
    if suggestions:
        lines.append("SUGGESTIONS:")
        for s in suggestions:
            lines.append(f"  • {s}")
        lines.append("")

    return "\n".join(lines)


def _suggest_shape_fix(expected_shape: tuple, actual_shape: tuple) -> str:
    """Suggest fixes for shape mismatches."""
    suggestions = ["POSSIBLE FIXES:"]

    exp = list(expected_shape)
    act = list(actual_shape)

    # Check for transpose
    if sorted(exp) == sorted(act):
        suggestions.append("  • Shapes have same dimensions in different order - try transpose/permute")
        # Find the permutation
        for i, e in enumerate(exp):
            if e in act:
                j = act.index(e)
                if i != j:
                    suggestions.append(f"    Expected dim {i} (size {e}) might correspond to actual dim {j}")

    # Check for missing/extra dimension
    if len(exp) == len(act) + 1:
        suggestions.append("  • Expected has one more dimension - actual might need unsqueeze/expand_dims")
    elif len(act) == len(exp) + 1:
        suggestions.append("  • Actual has one more dimension - actual might need squeeze")

    # Check for flattened dimension
    if len(exp) > len(act):
        prod_act = np.prod(act)
        prod_exp = np.prod(exp)
        if prod_act == prod_exp:
            suggestions.append("  • Same total elements - check if dimensions were flattened/merged")

    # Check for specific common issues
    if len(exp) == 4 and len(act) == 3:
        if exp[0] * exp[1] == act[0] and exp[2:] == act[1:]:
            suggestions.append("  • Looks like batch and heads might be merged in actual")
    elif len(exp) == 3 and len(act) == 4:
        if act[0] * act[1] == exp[0] and act[2:] == exp[1:]:
            suggestions.append("  • Looks like batch and heads might be split in actual")

    return "\n".join(suggestions)


def _analyze_diff_pattern(
    expected: np.ndarray,
    actual: np.ndarray,
    abs_diff: np.ndarray,
) -> list[str]:
    """Analyze difference patterns and return suggestions."""
    suggestions = []

    # Check for constant offset
    diff = actual - expected
    mean_diff = diff.mean()
    if np.allclose(diff, mean_diff, rtol=0.01):
        suggestions.append(f"Constant offset detected (mean={mean_diff:.6e}) - check for bias term")

    # Check for scale factor
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = actual / (expected + 1e-12)
        ratio = np.where(np.isfinite(ratio) & (np.abs(expected) > 1e-8), ratio, np.nan)
    if np.nanstd(ratio) < 0.01 * np.abs(np.nanmean(ratio)):
        scale = np.nanmean(ratio)
        suggestions.append(f"Constant scale factor detected (~{scale:.4f}) - check normalization")

    # Check for sign flip
    if np.allclose(actual, -expected, rtol=1e-4):
        suggestions.append("Sign flip detected - check for negation in implementation")

    # Check if diff is localized
    nonzero_mask = abs_diff > 1e-6
    if nonzero_mask.any():
        # Check if errors are localized to certain dimensions
        for axis in range(len(abs_diff.shape)):
            axis_max = abs_diff.max(axis=tuple(i for i in range(len(abs_diff.shape)) if i != axis))
            if axis_max.max() > 10 * axis_max.min() and axis_max.min() < 1e-5:
                bad_indices = np.where(axis_max > 1e-5)[0]
                if len(bad_indices) < len(axis_max) // 2:
                    suggestions.append(
                        f"Errors concentrated on axis {axis}, indices {bad_indices[:5].tolist()}{'...' if len(bad_indices) > 5 else ''}"
                    )

    # Check for epsilon difference (numerical precision)
    max_val = max(np.abs(expected).max(), np.abs(actual).max())
    if abs_diff.max() < 1e-5 * max_val:
        suggestions.append("Differences are very small relative to values - likely just numerical precision")

    # Check for inf/nan issues
    if np.any(~np.isfinite(actual)) and np.all(np.isfinite(expected)):
        suggestions.append("Actual contains inf/nan where expected doesn't - check for division by zero or overflow")

    return suggestions


def compare_structures(
    expected_dict: dict[str, Any],
    actual_dict: dict[str, Any],
    name: str = "state_dict",
) -> str:
    """Compare two dictionaries of arrays (e.g., state dicts) structurally.

    Useful for debugging weight loading issues.

    Args:
        expected_dict: Expected state dict
        actual_dict: Actual state dict
        name: Name for this comparison

    Returns:
        Diagnostic string showing structural differences
    """
    lines = [f"=== Structure comparison for '{name}' ===", ""]

    exp_keys = set(expected_dict.keys())
    act_keys = set(actual_dict.keys())

    missing = exp_keys - act_keys
    extra = act_keys - exp_keys
    common = exp_keys & act_keys

    lines.append(f"Keys in expected: {len(exp_keys)}")
    lines.append(f"Keys in actual: {len(act_keys)}")
    lines.append(f"Common keys: {len(common)}")
    lines.append("")

    if missing:
        lines.append(f"MISSING in actual ({len(missing)}):")
        for k in sorted(missing)[:20]:
            exp_shape = _get_shape(expected_dict[k])
            lines.append(f"  - {k}: {exp_shape}")
        if len(missing) > 20:
            lines.append(f"  ... and {len(missing) - 20} more")
        lines.append("")

    if extra:
        lines.append(f"EXTRA in actual ({len(extra)}):")
        for k in sorted(extra)[:20]:
            act_shape = _get_shape(actual_dict[k])
            lines.append(f"  + {k}: {act_shape}")
        if len(extra) > 20:
            lines.append(f"  ... and {len(extra) - 20} more")
        lines.append("")

    # Check shape mismatches in common keys
    shape_mismatches = []
    for k in common:
        exp_shape = _get_shape(expected_dict[k])
        act_shape = _get_shape(actual_dict[k])
        if exp_shape != act_shape:
            shape_mismatches.append((k, exp_shape, act_shape))

    if shape_mismatches:
        lines.append(f"SHAPE MISMATCHES ({len(shape_mismatches)}):")
        for k, exp_shape, act_shape in shape_mismatches[:20]:
            lines.append(f"  {k}: expected {exp_shape}, got {act_shape}")
        if len(shape_mismatches) > 20:
            lines.append(f"  ... and {len(shape_mismatches) - 20} more")
        lines.append("")

    if not missing and not extra and not shape_mismatches:
        lines.append("✓ Structures match perfectly")

    return "\n".join(lines)


def _get_shape(arr: Any) -> tuple:
    """Get shape of array-like."""
    if hasattr(arr, "shape"):
        return tuple(arr.shape)
    return ()
