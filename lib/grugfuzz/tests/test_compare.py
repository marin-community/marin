"""Tests for compare module."""

import jax.numpy as jnp
import numpy as np
import pytest

from grugfuzz import ComparisonResult, compare


class TestCompare:
    def test_identical_arrays_pass(self):
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = compare(arr, arr, name="identical")
        assert result.passed
        assert result.max_abs_diff == 0.0
        assert "PASS" in str(result)

    def test_small_diff_passes(self):
        expected = jnp.array([1.0, 2.0, 3.0])
        actual = jnp.array([1.0 + 1e-6, 2.0 - 1e-6, 3.0 + 1e-6])
        result = compare(expected, actual, name="small_diff", atol=1e-4)
        assert result.passed

    def test_large_diff_fails(self):
        expected = jnp.array([1.0, 2.0, 3.0])
        actual = jnp.array([1.1, 2.0, 3.0])
        result = compare(expected, actual, name="large_diff", atol=1e-4)
        assert not result.passed
        assert result.max_abs_diff == pytest.approx(0.1, rel=1e-6)
        assert "FAIL" in str(result)

    def test_shape_mismatch_fails(self):
        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        actual = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = compare(expected, actual, name="shape_mismatch")
        assert not result.passed
        assert "Shape mismatch" in result.failure_summary
        assert result.expected_shape == (2, 2)
        assert result.actual_shape == (4,)

    def test_diff_locations_identified(self):
        expected = jnp.zeros((3, 3))
        actual = jnp.zeros((3, 3)).at[1, 2].set(1.0)
        result = compare(expected, actual, name="single_diff", atol=1e-4)
        assert not result.passed
        # The largest diff should be at (1, 2)
        assert (1, 2) in result.diff_locations

    def test_numpy_input(self):
        expected = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])
        result = compare(expected, actual, name="numpy")
        assert result.passed

    def test_mixed_input(self):
        expected = jnp.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])
        result = compare(expected, actual, name="mixed")
        assert result.passed

    def test_tolerances(self):
        expected = jnp.array([1.0, 2.0])
        actual = jnp.array([1.001, 2.002])

        # Should fail with tight tolerance
        result = compare(expected, actual, atol=1e-4, rtol=1e-4)
        assert not result.passed

        # Should pass with looser tolerance
        result = compare(expected, actual, atol=1e-2, rtol=1e-2)
        assert result.passed

    def test_3d_array(self):
        expected = jnp.ones((2, 4, 8))
        actual = jnp.ones((2, 4, 8)) + 1e-6
        result = compare(expected, actual, name="3d")
        assert result.passed

    def test_result_str_repr(self):
        arr = jnp.array([1.0])
        result = compare(arr, arr, name="test")
        assert "test" in str(result)
        assert "test" in repr(result)
