# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import jax.numpy as jnp
import equinox as eqx
import pytest

import haliax as hax


class M(eqx.Module):
    a: jnp.ndarray = hax.field(axis_names=("batch",))


def test_axis_names_metadata():
    field = M.__dataclass_fields__["a"]
    assert field.metadata["axis_names"] == ("batch",)


def test_axis_names_static_exclusive():
    with pytest.raises(ValueError):
        hax.field(static=True, axis_names=("x",))
