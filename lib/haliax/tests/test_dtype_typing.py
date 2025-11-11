# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import typing

import jax.numpy as jnp

from haliax import Axis, NamedArray
from haliax.haxtyping import Float, Int, f32, i32


def test_dtype_and_axes_annotation():
    def foo(x: f32["batch embed"]):  # type: ignore  # noqa: F722
        pass

    ann = typing.get_args(typing.get_type_hints(foo, include_extras=True)["x"])
    assert ann[0] is NamedArray
    spec = ann[1]
    assert spec.dtype == jnp.float32
    assert spec.before == ("batch", "embed")


def test_other_dtype_annotation():
    def bar(x: i32["batch"]):  # type: ignore  # noqa: F722
        pass

    spec = typing.get_args(typing.get_type_hints(bar, include_extras=True)["x"])[1]
    assert spec.dtype == jnp.int32
    assert spec.before == ("batch",)


def test_dtype_category_annotation_and_check():
    def baz(x: Float["b"]):  # type: ignore  # noqa: F722
        pass

    spec = typing.get_args(typing.get_type_hints(baz, include_extras=True)["x"])[1]
    assert str(spec.dtype) == "float"

    B = Axis("b", 1)
    arr = NamedArray(jnp.ones((B.size,), dtype=jnp.float32), (B,))
    assert arr.matches_axes(Float["b"])  # type: ignore
    assert not arr.matches_axes(Int["b"])  # type: ignore
