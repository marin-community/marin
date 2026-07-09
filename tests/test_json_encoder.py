# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import jax.numpy as jnp
import numpy as np
import pytest
from marin.utilities.json_encoder import CustomJsonEncoder


def _dumps(o) -> str:
    return json.dumps(o, cls=CustomJsonEncoder)


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (jnp.float32, "float32"),
        (jnp.bfloat16, "bfloat16"),
        (np.float32, "float32"),
        (np.dtype("float32"), "float32"),
        (np.dtype(jnp.bfloat16), "bfloat16"),
    ],
)
def test_dtype_serializes_to_canonical_name(dtype, expected):
    # Config dtype fields (numpy scalar types, jax scalar-type wrappers, and
    # np.dtype objects) serialize to their name — not the "<class ...>" repr the
    # generic str() fallback would produce, and without a jax import.
    assert _dumps(dtype) == json.dumps(expected)


def test_dtype_field_on_dataclass_round_trips():
    @dataclasses.dataclass
    class ModelConfig:
        compute_dtype: object
        name: str

    result = json.loads(_dumps(ModelConfig(compute_dtype=jnp.bfloat16, name="tiny")))
    assert result == {"compute_dtype": "bfloat16", "name": "tiny"}


def test_array_like_input_does_not_crash():
    # Regression: the old ``o in (float32, bfloat16)`` membership test evaluated
    # ``o == dtype`` element-wise on arrays and raised "truth value ambiguous".
    # An array is not JSON data, so the encoder just coerces it to str.
    assert _dumps(np.array([1, 2, 3])) == json.dumps("[1 2 3]")


def test_unserializable_object_falls_back_to_str():
    # Contract: unknown leaf objects are coerced to str rather than raising, so
    # a config with an exotic value still produces valid JSON.
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    assert _dumps(Opaque()) == json.dumps("<opaque>")


def test_arbitrary_class_is_not_mislabeled_as_dtype():
    # np.dtype() coerces any class to the object dtype; such classes must fall
    # through to the str fallback, not be reported as a dtype name.
    class Widget:
        pass

    assert "Widget" in json.loads(_dumps(Widget))
