"""Tests for diagnostics module."""

import jax.numpy as jnp
import numpy as np
import pytest

from grugfuzz import compare_structures, diagnose_diff


class TestDiagnoseDiff:
    def test_basic_output(self):
        expected = jnp.array([1.0, 2.0, 3.0])
        actual = jnp.array([1.1, 2.0, 3.0])

        output = diagnose_diff(expected, actual, name="test")

        assert "test" in output
        assert "SHAPES:" in output
        assert "STATISTICS:" in output
        assert "DIFFERENCE SUMMARY:" in output
        assert "LARGEST DIFFERENCES:" in output

    def test_shape_mismatch(self):
        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        actual = jnp.array([1.0, 2.0, 3.0, 4.0])

        output = diagnose_diff(expected, actual, name="shape_test")

        assert "SHAPE MISMATCH" in output
        assert "Expected: (2, 2)" in output
        assert "Actual:   (4,)" in output
        assert "POSSIBLE FIXES:" in output

    def test_constant_offset_detection(self):
        expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = expected + 0.5  # Constant offset

        output = diagnose_diff(expected, actual)

        assert "SUGGESTIONS:" in output
        assert "offset" in output.lower()

    def test_scale_factor_detection(self):
        expected = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = expected * 2.0  # Constant scale

        output = diagnose_diff(expected, actual)

        assert "SUGGESTIONS:" in output
        assert "scale" in output.lower()

    def test_sign_flip_detection(self):
        expected = jnp.array([1.0, -2.0, 3.0, -4.0])
        actual = -expected  # Sign flip

        output = diagnose_diff(expected, actual)

        assert "SUGGESTIONS:" in output
        assert "sign" in output.lower() or "Sign" in output

    def test_shows_worst_differences(self):
        expected = jnp.zeros((3, 3))
        actual = jnp.zeros((3, 3)).at[1, 2].set(100.0)

        output = diagnose_diff(expected, actual, num_worst=3)

        # Should show the location of the difference
        assert "(1, 2)" in output

    def test_numerical_precision_suggestion(self):
        expected = jnp.array([1e6, 2e6, 3e6])
        actual = expected + 1e-2  # Very small relative diff

        output = diagnose_diff(expected, actual)

        # May suggest numerical precision is the issue
        if "SUGGESTIONS:" in output:
            assert "precision" in output.lower() or "small" in output.lower()

    def test_numpy_input(self):
        expected = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.1, 2.0, 3.0])

        output = diagnose_diff(expected, actual)
        assert "STATISTICS:" in output


class TestCompareStructures:
    def test_identical_structures(self):
        dict1 = {"a": jnp.ones((2, 3)), "b": jnp.zeros((4,))}
        dict2 = {"a": jnp.ones((2, 3)), "b": jnp.zeros((4,))}

        output = compare_structures(dict1, dict2)

        assert "match perfectly" in output.lower()

    def test_missing_keys(self):
        expected = {"a": jnp.ones((2,)), "b": jnp.ones((3,)), "c": jnp.ones((4,))}
        actual = {"a": jnp.ones((2,))}

        output = compare_structures(expected, actual)

        assert "MISSING in actual" in output
        assert "b" in output
        assert "c" in output

    def test_extra_keys(self):
        expected = {"a": jnp.ones((2,))}
        actual = {"a": jnp.ones((2,)), "x": jnp.ones((3,)), "y": jnp.ones((4,))}

        output = compare_structures(expected, actual)

        assert "EXTRA in actual" in output
        assert "x" in output
        assert "y" in output

    def test_shape_mismatches(self):
        expected = {"a": jnp.ones((2, 3)), "b": jnp.ones((4, 5))}
        actual = {"a": jnp.ones((3, 2)), "b": jnp.ones((4, 5))}  # a has different shape

        output = compare_structures(expected, actual)

        assert "SHAPE MISMATCHES" in output
        assert "a" in output
        assert "(2, 3)" in output
        assert "(3, 2)" in output

    def test_shows_counts(self):
        expected = {"a": jnp.ones((2,)), "b": jnp.ones((3,))}
        actual = {"a": jnp.ones((2,)), "c": jnp.ones((4,))}

        output = compare_structures(expected, actual)

        assert "Keys in expected: 2" in output
        assert "Keys in actual: 2" in output
        assert "Common keys: 1" in output

    def test_named_comparison(self):
        dict1 = {"x": jnp.ones((1,))}
        dict2 = {"x": jnp.ones((1,))}

        output = compare_structures(dict1, dict2, name="my_weights")

        assert "my_weights" in output


class TestTransposeSuggestion:
    def test_suggests_transpose_for_swapped_dims(self):
        expected = jnp.ones((2, 3, 4))
        actual = jnp.ones((2, 4, 3))  # Swapped last two dims

        output = diagnose_diff(expected, actual)

        assert "transpose" in output.lower() or "permute" in output.lower()

    def test_suggests_squeeze_for_extra_dim(self):
        expected = jnp.ones((2, 3))
        actual = jnp.ones((2, 3, 1))  # Extra dim

        output = diagnose_diff(expected, actual)

        assert "squeeze" in output.lower() or "dimension" in output.lower()
