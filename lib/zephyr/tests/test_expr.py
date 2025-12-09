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

"""Tests for expression evaluation and conversion."""

from zephyr.expr import (
    col,
    lit,
    to_pyarrow_expr,
)


class TestColumnExpr:
    def test_evaluate_existing_column(self):
        expr = col("name")
        assert expr.evaluate({"name": "alice"}) == "alice"

    def test_evaluate_missing_column(self):
        expr = col("name")
        assert expr.evaluate({"other": "bob"}) is None

    def test_repr(self):
        expr = col("score")
        assert repr(expr) == "col('score')"


class TestLiteralExpr:
    def test_evaluate_integer(self):
        expr = lit(42)
        assert expr.evaluate({}) == 42

    def test_evaluate_string(self):
        expr = lit("hello")
        assert expr.evaluate({"anything": "ignored"}) == "hello"

    def test_evaluate_float(self):
        expr = lit(3.14)
        assert expr.evaluate({}) == 3.14

    def test_repr(self):
        assert repr(lit(42)) == "lit(42)"
        assert repr(lit("hello")) == "lit('hello')"


class TestCompareExpr:
    def test_equal(self):
        expr = col("score") == 100
        assert expr.evaluate({"score": 100}) is True
        assert expr.evaluate({"score": 50}) is False

    def test_not_equal(self):
        expr = col("score") != 100
        assert expr.evaluate({"score": 100}) is False
        assert expr.evaluate({"score": 50}) is True

    def test_less_than(self):
        expr = col("score") < 100
        assert expr.evaluate({"score": 50}) is True
        assert expr.evaluate({"score": 100}) is False
        assert expr.evaluate({"score": 150}) is False

    def test_less_equal(self):
        expr = col("score") <= 100
        assert expr.evaluate({"score": 50}) is True
        assert expr.evaluate({"score": 100}) is True
        assert expr.evaluate({"score": 150}) is False

    def test_greater_than(self):
        expr = col("score") > 100
        assert expr.evaluate({"score": 50}) is False
        assert expr.evaluate({"score": 100}) is False
        assert expr.evaluate({"score": 150}) is True

    def test_greater_equal(self):
        expr = col("score") >= 100
        assert expr.evaluate({"score": 50}) is False
        assert expr.evaluate({"score": 100}) is True
        assert expr.evaluate({"score": 150}) is True

    def test_compare_columns(self):
        expr = col("a") > col("b")
        assert expr.evaluate({"a": 10, "b": 5}) is True
        assert expr.evaluate({"a": 5, "b": 10}) is False

    def test_repr(self):
        expr = col("score") > 50
        assert repr(expr) == "(col('score') > lit(50))"


class TestLogicalExpr:
    def test_and_both_true(self):
        expr = (col("a") > 0) & (col("b") > 0)
        assert expr.evaluate({"a": 1, "b": 1}) is True

    def test_and_one_false(self):
        expr = (col("a") > 0) & (col("b") > 0)
        assert expr.evaluate({"a": 1, "b": -1}) is False
        assert expr.evaluate({"a": -1, "b": 1}) is False

    def test_and_short_circuit(self):
        expr = (col("a") > 0) & (col("b") > 0)
        # If a <= 0, b is not evaluated (short-circuit)
        assert expr.evaluate({"a": -1, "b": 1}) is False

    def test_or_both_false(self):
        expr = (col("a") > 0) | (col("b") > 0)
        assert expr.evaluate({"a": -1, "b": -1}) is False

    def test_or_one_true(self):
        expr = (col("a") > 0) | (col("b") > 0)
        assert expr.evaluate({"a": 1, "b": -1}) is True
        assert expr.evaluate({"a": -1, "b": 1}) is True

    def test_complex_logical(self):
        # (a > 0 AND b > 0) OR c > 0
        expr = ((col("a") > 0) & (col("b") > 0)) | (col("c") > 0)
        assert expr.evaluate({"a": 1, "b": 1, "c": -1}) is True
        assert expr.evaluate({"a": -1, "b": -1, "c": 1}) is True
        assert expr.evaluate({"a": -1, "b": -1, "c": -1}) is False

    def test_repr(self):
        expr = (col("a") > 0) & (col("b") > 0)
        assert repr(expr) == "((col('a') > lit(0)) & (col('b') > lit(0)))"


class TestNotExpr:
    def test_not_true(self):
        expr = ~(col("flag") == True)  # noqa: E712
        assert expr.evaluate({"flag": True}) is False
        assert expr.evaluate({"flag": False}) is True

    def test_not_comparison(self):
        expr = ~(col("score") > 50)
        assert expr.evaluate({"score": 60}) is False
        assert expr.evaluate({"score": 40}) is True

    def test_repr(self):
        expr = ~(col("flag") == True)  # noqa: E712
        assert repr(expr) == "~(col('flag') == lit(True))"


class TestArithmeticExpr:
    def test_add(self):
        expr = col("a") + col("b")
        assert expr.evaluate({"a": 10, "b": 5}) == 15

    def test_subtract(self):
        expr = col("a") - col("b")
        assert expr.evaluate({"a": 10, "b": 3}) == 7

    def test_multiply(self):
        expr = col("a") * col("b")
        assert expr.evaluate({"a": 4, "b": 5}) == 20

    def test_divide(self):
        expr = col("a") / col("b")
        assert expr.evaluate({"a": 10, "b": 4}) == 2.5

    def test_add_literal(self):
        expr = col("a") + 10
        assert expr.evaluate({"a": 5}) == 15

    def test_complex_arithmetic(self):
        # (a + b) * c
        expr = (col("a") + col("b")) * col("c")
        assert expr.evaluate({"a": 2, "b": 3, "c": 4}) == 20

    def test_repr(self):
        expr = col("a") + col("b")
        assert repr(expr) == "(col('a') + col('b'))"


class TestFieldAccessExpr:
    def test_nested_field(self):
        expr = col("meta")["score"]
        assert expr.evaluate({"meta": {"score": 0.8}}) == 0.8

    def test_nested_field_missing(self):
        expr = col("meta")["score"]
        assert expr.evaluate({"meta": {"other": 1}}) is None

    def test_nested_field_with_comparison(self):
        expr = col("meta")["score"] > 0.5
        assert expr.evaluate({"meta": {"score": 0.8}}) is True
        assert expr.evaluate({"meta": {"score": 0.2}}) is False

    def test_deeply_nested(self):
        expr = col("data")["level1"]["level2"]
        assert expr.evaluate({"data": {"level1": {"level2": "value"}}}) == "value"

    def test_non_dict_parent(self):
        expr = col("meta")["score"]
        assert expr.evaluate({"meta": "not a dict"}) is None

    def test_repr(self):
        expr = col("meta")["score"]
        assert repr(expr) == "col('meta')['score']"


class TestPyArrowConversion:
    def test_column(self):
        expr = col("score")
        pa_expr = to_pyarrow_expr(expr)
        assert pa_expr is not None

    def test_literal(self):
        expr = lit(42)
        pa_expr = to_pyarrow_expr(expr)
        assert pa_expr is not None

    def test_comparison(self):
        expr = col("score") > 0.5
        pa_expr = to_pyarrow_expr(expr)
        assert pa_expr is not None

    def test_logical_and(self):
        expr = (col("score") > 0.5) & (col("category") == "A")
        pa_expr = to_pyarrow_expr(expr)
        assert pa_expr is not None

    def test_logical_or(self):
        expr = (col("a") > 0) | (col("b") > 0)
        pa_expr = to_pyarrow_expr(expr)
        assert pa_expr is not None

    def test_not(self):
        expr = ~(col("flag") == True)  # noqa: E712
        pa_expr = to_pyarrow_expr(expr)
        assert pa_expr is not None

    def test_arithmetic(self):
        expr = col("a") + col("b")
        pa_expr = to_pyarrow_expr(expr)
        assert pa_expr is not None


class TestExprHash:
    def test_same_expr_same_hash(self):
        expr1 = col("score") > 0.5
        expr2 = col("score") > 0.5
        assert hash(expr1) == hash(expr2)

    def test_different_expr_different_hash(self):
        expr1 = col("score") > 0.5
        expr2 = col("score") < 0.5
        assert hash(expr1) != hash(expr2)

    def test_expr_in_set(self):
        expr1 = col("score") > 0.5
        expr2 = col("score") > 0.5
        s = {expr1}
        assert expr2 in s

    def test_expr_as_dict_key(self):
        expr = col("score") > 0.5
        d = {expr: "value"}
        assert d[expr] == "value"
