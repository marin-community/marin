# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Kelp evaluation metrics."""


from experiments.kelp.eval.metrics import (
    ValidityResult,
    check_python_syntax,
    compute_pass_at_k,
    compute_validity_rate,
)


class TestCheckPythonSyntax:
    def test_valid_code(self):
        is_valid, error = check_python_syntax("x = 1")
        assert is_valid
        assert error is None

    def test_invalid_code(self):
        is_valid, error = check_python_syntax("def foo(")
        assert not is_valid
        assert error is not None
        assert "SyntaxError" in error

    def test_empty_code(self):
        is_valid, error = check_python_syntax("")
        assert is_valid
        assert error is None

    def test_function_definition(self):
        code = """
def add(a, b):
    return a + b
"""
        is_valid, error = check_python_syntax(code)
        assert is_valid


class TestComputeValidityRate:
    def test_all_valid(self):
        codes = ["x = 1", "y = 2", "z = x + y"]
        rate = compute_validity_rate(codes)
        assert rate == 1.0

    def test_all_invalid(self):
        codes = ["def (", "if:", "class"]
        rate = compute_validity_rate(codes)
        assert rate == 0.0

    def test_mixed(self):
        codes = ["x = 1", "def (", "y = 2"]
        rate = compute_validity_rate(codes)
        assert abs(rate - 2.0 / 3.0) < 0.01

    def test_empty_list(self):
        rate = compute_validity_rate([])
        assert rate == 0.0

    def test_return_details(self):
        codes = ["x = 1", "def ("]
        result = compute_validity_rate(codes, return_details=True)
        assert isinstance(result, ValidityResult)
        assert result.total == 2
        assert result.valid == 1
        assert len(result.invalid_samples) == 1


class TestComputePassAtK:
    def test_all_pass(self):
        results = [[True, True, True], [True, True, True]]
        result = compute_pass_at_k(results, k=1)
        assert result.pass_rate == 1.0

    def test_none_pass(self):
        results = [[False, False, False], [False, False, False]]
        result = compute_pass_at_k(results, k=1)
        assert result.pass_rate == 0.0

    def test_partial_pass(self):
        results = [[True, False, False], [False, False, True]]
        result = compute_pass_at_k(results, k=1)
        assert result.pass_rate > 0.0
        assert result.pass_rate < 1.0

    def test_pass_at_10(self):
        results = [[False] * 9 + [True]]
        result = compute_pass_at_k(results, k=10)
        assert result.pass_rate == 1.0

    def test_empty_results(self):
        result = compute_pass_at_k([], k=1)
        assert result.pass_rate == 0.0
        assert result.total_problems == 0

    def test_k_larger_than_samples(self):
        results = [[True, False]]
        result = compute_pass_at_k(results, k=10)
        assert result.pass_rate > 0.0
