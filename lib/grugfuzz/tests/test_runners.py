"""Tests for runners module."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn as nn

from grugfuzz import SuiteResult, compare, run_comparison_suite, run_hf


class TestRunHf:
    def test_simple_linear(self):
        # Simple PyTorch linear layer
        layer = nn.Linear(4, 8, bias=False)
        nn.init.ones_(layer.weight)

        x = jnp.ones((2, 4))
        out = run_hf(layer, x)

        assert isinstance(out, jnp.ndarray)
        assert out.shape == (2, 8)
        # With all-ones weights and input, output should be 4.0
        np.testing.assert_allclose(np.array(out), 4.0)

    def test_with_bias(self):
        layer = nn.Linear(3, 2, bias=True)
        nn.init.zeros_(layer.weight)
        nn.init.ones_(layer.bias)

        x = jnp.zeros((1, 3))
        out = run_hf(layer, x)

        # Output should just be bias (1.0)
        np.testing.assert_allclose(np.array(out), 1.0)

    def test_tuple_output(self):
        # Module that returns a tuple
        class TupleModule(nn.Module):
            def forward(self, x):
                return x * 2, x * 3

        module = TupleModule()
        x = jnp.array([1.0, 2.0])

        out0 = run_hf(module, x, output_idx=0)
        out1 = run_hf(module, x, output_idx=1)

        np.testing.assert_allclose(np.array(out0), [2.0, 4.0])
        np.testing.assert_allclose(np.array(out1), [3.0, 6.0])

    def test_invalid_output_idx(self):
        layer = nn.Linear(2, 2)
        x = jnp.ones((1, 2))

        with pytest.raises(ValueError, match="output is not a tuple"):
            run_hf(layer, x, output_idx=0)

    def test_multiple_inputs(self):
        class AddModule(nn.Module):
            def forward(self, a, b):
                return a + b

        module = AddModule()
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])

        out = run_hf(module, a, b)
        np.testing.assert_allclose(np.array(out), [4.0, 6.0])

    def test_numpy_input(self):
        layer = nn.Linear(2, 2, bias=False)
        nn.init.eye_(layer.weight)

        x = np.array([[1.0, 2.0]])
        out = run_hf(layer, x)

        assert isinstance(out, jnp.ndarray)
        np.testing.assert_allclose(np.array(out), [[1.0, 2.0]])

    def test_torch_input(self):
        layer = nn.Linear(2, 2, bias=False)
        nn.init.eye_(layer.weight)

        x = torch.tensor([[1.0, 2.0]])
        out = run_hf(layer, x)

        assert isinstance(out, jnp.ndarray)

    def test_integer_tokens(self):
        layer = nn.Embedding(10, 4)
        nn.init.ones_(layer.weight)

        tokens = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        out = run_hf(layer, tokens)

        assert out.dtype == jnp.float32
        np.testing.assert_allclose(np.array(out), 1.0)


class TestRunComparisonSuite:
    def test_all_pass(self):
        arr1 = jnp.array([1.0, 2.0, 3.0])
        arr2 = jnp.array([4.0, 5.0, 6.0])

        tests = [
            ("test1", lambda x: x, lambda x: x, {"x": arr1}),
            ("test2", lambda x: x * 2, lambda x: x * 2, {"x": arr2}),
        ]

        result = run_comparison_suite(tests)

        assert result.all_passed
        assert result.num_passed == 2
        assert result.num_failed == 0
        assert result.first_failure is None

    def test_one_failure(self):
        arr = jnp.array([1.0, 2.0, 3.0])

        tests = [
            ("pass_test", lambda x: x, lambda x: x, {"x": arr}),
            ("fail_test", lambda x: x, lambda x: x + 1, {"x": arr}),  # Different!
        ]

        result = run_comparison_suite(tests, stop_on_failure=True)

        assert not result.all_passed
        assert result.num_passed == 1
        assert result.num_failed == 1
        assert result.first_failure.name == "fail_test"
        # Should stop at failure
        assert len(result.results) == 2

    def test_continue_on_failure(self):
        arr = jnp.array([1.0, 2.0, 3.0])

        tests = [
            ("pass1", lambda x: x, lambda x: x, {"x": arr}),
            ("fail1", lambda x: x, lambda x: x + 1, {"x": arr}),
            ("pass2", lambda x: x, lambda x: x, {"x": arr}),
            ("fail2", lambda x: x, lambda x: x + 2, {"x": arr}),
        ]

        result = run_comparison_suite(tests, stop_on_failure=False)

        assert not result.all_passed
        assert result.num_passed == 2
        assert result.num_failed == 2
        assert len(result.results) == 4

    def test_hf_function_error(self):
        def bad_hf(x):
            raise RuntimeError("HF error")

        tests = [("error_test", bad_hf, lambda x: x, {"x": jnp.array([1.0])})]

        result = run_comparison_suite(tests)

        assert not result.all_passed
        assert result.num_failed == 1
        assert "HF function raised" in result.first_failure.failure_summary

    def test_grug_function_error(self):
        def bad_grug(x):
            raise RuntimeError("Grug error")

        tests = [("error_test", lambda x: x, bad_grug, {"x": jnp.array([1.0])})]

        result = run_comparison_suite(tests)

        assert not result.all_passed
        assert result.num_failed == 1
        assert "Grug function raised" in result.first_failure.failure_summary

    def test_suite_str(self):
        arr = jnp.array([1.0])
        tests = [("test", lambda x: x, lambda x: x, {"x": arr})]
        result = run_comparison_suite(tests)

        s = str(result)
        assert "1/1" in s
        assert "passed" in s

    def test_custom_tolerance(self):
        arr = jnp.array([1.0, 2.0])

        tests = [
            ("tight", lambda x: x, lambda x: x + 0.001, {"x": arr}),
        ]

        # Should fail with tight tolerance
        result = run_comparison_suite(tests, atol=1e-5, rtol=1e-5)
        assert not result.all_passed

        # Should pass with loose tolerance
        result = run_comparison_suite(tests, atol=1e-2, rtol=1e-2)
        assert result.all_passed
