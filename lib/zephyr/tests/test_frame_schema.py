# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephyr.frame_schema."""

import polars as pl
import pytest
from zephyr.frame_schema import (
    assert_compatible_sort_key_dtypes,
    is_safe_sort_key_dtype_pair,
    unified_schema,
)


def test_is_safe_sort_key_dtype_pair_allows_widenings():
    assert is_safe_sort_key_dtype_pair(pl.Null, pl.Utf8)
    assert is_safe_sort_key_dtype_pair(pl.Int32, pl.Int64)
    assert is_safe_sort_key_dtype_pair(pl.Float32, pl.Float64)
    assert is_safe_sort_key_dtype_pair(pl.Utf8, pl.String)
    assert is_safe_sort_key_dtype_pair(pl.Binary, pl.Binary)


def test_is_safe_sort_key_dtype_pair_rejects_cross_family():
    assert not is_safe_sort_key_dtype_pair(pl.Int64, pl.Utf8)
    assert not is_safe_sort_key_dtype_pair(pl.Int64, pl.Float64)
    assert not is_safe_sort_key_dtype_pair(pl.Boolean, pl.Int64)


def test_assert_compatible_sort_key_dtypes_rejects_int_vs_utf8():
    schemas = [
        pl.Schema({"__zephyr_sort_key__": pl.Struct({"key": pl.Int64, "sort_value": pl.Null})}),
        pl.Schema({"__zephyr_sort_key__": pl.Struct({"key": pl.Utf8, "sort_value": pl.Null})}),
    ]
    with pytest.raises(ValueError, match="incompatible scatter sort-key"):
        assert_compatible_sort_key_dtypes(schemas, sort_key_col="__zephyr_sort_key__")


def test_assert_compatible_sort_key_dtypes_allows_int_widening():
    schemas = [
        pl.Schema({"__zephyr_sort_key__": pl.Struct({"key": pl.Int32, "sort_value": pl.Null})}),
        pl.Schema({"__zephyr_sort_key__": pl.Struct({"key": pl.Int64, "sort_value": pl.Null})}),
    ]
    assert_compatible_sort_key_dtypes(schemas, sort_key_col="__zephyr_sort_key__")


def test_unified_schema_diagonal_relaxed_fills_missing_columns():
    a = pl.DataFrame({"a": [1], "b": ["x"]})
    b = pl.DataFrame({"a": [2]})
    schema = unified_schema([a, b], how="diagonal_relaxed")
    assert schema.names() == ["a", "b"]
