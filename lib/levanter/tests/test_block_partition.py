# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from levanter.optim._block_partition import (
    BlockPartitioner,
    merge_small_dimensions,
    pad_and_stack_matrices,
    unstack_and_unpad_matrices,
)


@pytest.mark.parametrize(
    "shape,block_size,dim_diag",
    [
        ((24, 17), 8, [False, False]),
        ((24, 17), 8, [True, False]),
        ((32, 24), 16, [False, False]),
        ((6, 4), 8, [False, False]),
        ((13,), 8, [False]),
        ((5, 7, 9), 4, [False, True, False]),
    ],
)
def test_partition_pad_stack_round_trip(shape, block_size, dim_diag):
    """The full block pipeline used by SOAP and Kron must be lossless."""
    x = jax.random.normal(jax.random.PRNGKey(0), shape)

    partitioner = BlockPartitioner(shape, block_size, dim_diag)
    blocks = partitioner.partition(x)
    block_shapes = [b.shape for b in blocks]

    stacked = pad_and_stack_matrices(blocks, block_size)
    assert stacked.shape[0] == len(blocks)
    assert all(dim % block_size == 0 for dim in stacked.shape[1:])

    recovered = partitioner.merge_partitions(unstack_and_unpad_matrices(stacked, block_shapes))

    assert recovered.shape == shape
    np.testing.assert_array_equal(np.asarray(recovered), np.asarray(x))


def test_partition_splits_only_non_diagonal_dims():
    """Dimensions flagged diagonal are preconditioned whole, so they must not be split."""
    x = jax.random.normal(jax.random.PRNGKey(0), (24, 24))

    blocks = BlockPartitioner((24, 24), 8, [False, True]).partition(x)

    assert [b.shape for b in blocks] == [(8, 24)] * 3


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (1024, 4, 4), (7,), (1,), ()])
def test_merge_small_dimensions_preserves_size(shape):
    merged_shape, merged_diag, sharding = merge_small_dimensions(shape, 16, [False] * len(shape))

    assert np.prod(merged_shape, dtype=np.int64) == np.prod(shape, dtype=np.int64)
    assert len(merged_diag) == max(len(merged_shape), 1)
    assert sharding is None
    if shape:
        x = jax.random.normal(jax.random.PRNGKey(0), shape)
        np.testing.assert_array_equal(np.asarray(x.reshape(merged_shape).reshape(shape)), np.asarray(x))


def test_merge_small_dimensions_merges_sharding_and_diag_flags():
    shape = (4, 4, 512)
    merged_shape, merged_diag, sharding = merge_small_dimensions(
        shape, 16, [True, False, False], sharding_to_merge=PartitionSpec(None, "data", "model")
    )

    assert merged_shape == [16, 512]
    # a merged dimension is diagonal only if every dimension folded into it is
    assert merged_diag == [False, False]
    assert sharding == PartitionSpec("data", "model")
