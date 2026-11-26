# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_dataclass

from levanter.pipeline import (
    evenly_partition_layers,
    reshape_batch_into_microbatches,
)


def test_evenly_partition_layers_returns_contiguous_balanced_ranges():
    assert evenly_partition_layers(10, 4) == ((0, 3), (3, 6), (6, 8), (8, 10))


@register_dataclass
@dataclass(frozen=True)
class _Batch:
    tokens: jnp.ndarray
    weights: jnp.ndarray
    label: str


def test_reshape_batch_into_microbatches_adds_leading_axis():
    batch = _Batch(
        tokens=jnp.arange(24).reshape(6, 4),
        weights=jnp.arange(6),
        label="train",
    )

    reshaped = reshape_batch_into_microbatches(batch, 3)

    assert reshaped.tokens.shape == (3, 2, 4)
    assert reshaped.weights.shape == (3, 2)
    assert reshaped.label == "train"
    np.testing.assert_array_equal(reshaped.tokens[1], np.arange(8, 16).reshape(2, 4))


def test_reshape_batch_into_microbatches_preserves_explicit_batch_sharding():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("batch",), axis_types=(AxisType.Explicit,))
    batch = jax.device_put(jnp.arange(24).reshape(6, 4), NamedSharding(mesh, P("batch", None)))

    reshaped = reshape_batch_into_microbatches(batch, 3)

    assert reshaped.sharding.spec == P(None, "batch", None)


def test_pipeline_helpers_reject_impossible_partitions():
    with pytest.raises(ValueError):
        evenly_partition_layers(2, 3)
    with pytest.raises(ValueError):
        reshape_batch_into_microbatches(jnp.zeros((5, 2)), 2)
