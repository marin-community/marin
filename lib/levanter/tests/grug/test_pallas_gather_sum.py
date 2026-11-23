# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare ring_filtered's row gathers (Pallas-Triton kernels, clamped token sum) with the XLA reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug._moe.ep_ring import _gather_sum_slots, _take_valid_rows
from levanter.grug._moe.ep_ring_filtered import _gather_sum_slots_clamped
from levanter.grug._moe.pallas_gather_sum import gather_sum_rows, gather_valid_rows, pallas_gather_supported

pytestmark = pytest.mark.skipif(jax.default_backend() == "tpu", reason="Pallas-Triton kernels need a GPU or CPU")


def _bits(x: jax.Array) -> np.ndarray:
    x = np.asarray(x)
    return x.view(np.uint16 if x.dtype.itemsize == 2 else np.uint32)


def _rows(key, shape, dtype):
    rows = jax.random.normal(key, shape, jnp.float32)
    # Include signed zeros to check the gather-sum's accumulation order.
    rows = rows.at[::7].set(-0.0).at[1::11].set(0.0)
    return rows.astype(dtype)


def _slots(key, num_out, topk, num_src):
    # Each source row read at most once, as in the dispatch backward, with sentinels at and past num_src.
    perm = jax.random.permutation(key, num_out * topk)
    slots = jnp.where(perm < num_src, perm, num_src + perm % 3).reshape(num_out, topk)
    return slots.astype(jnp.int32)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize(
    ("num_src", "num_out", "topk", "hidden"),
    [(96, 70, 4, 64), (40, 33, 1, 48), (300, 129, 3, 1024)],
    ids=["k4_tail_rows", "k1_narrow", "k3_two_col_blocks"],
)
def test_gather_sum_rows_matches_xla_bitwise(num_src, num_out, topk, hidden, dtype):
    k_rows, k_slots = jax.random.split(jax.random.key(0))
    rows = _rows(k_rows, (num_src, hidden), dtype)
    slots = _slots(k_slots, num_out, topk, num_src)
    expected = jax.jit(_gather_sum_slots)(rows, slots)
    actual = jax.jit(gather_sum_rows)(rows, slots)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_gather_valid_rows_matches_xla_bitwise(dtype):
    k_x, k_idx, k_valid = jax.random.split(jax.random.key(1), 3)
    x = _rows(k_x, (50, 80), dtype)
    index = jax.random.randint(k_idx, (77,), 0, 50)
    valid = jax.random.bernoulli(k_valid, 0.7, (77,))
    expected = jax.jit(_take_valid_rows)(x, index, valid)
    actual = jax.jit(gather_valid_rows)(x, index, valid)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def test_pallas_gather_supported_needs_a_power_of_two_divisor_of_at_least_16():
    widths = [8, 15, 16, 24, 32, 40, 48, 2560]
    assert [pallas_gather_supported(h) for h in widths] == [False, False, True, False, True, False, True, True]


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_gather_sum_slots_clamped_matches_take_bitwise(dtype):
    k_rows, k_slots = jax.random.split(jax.random.key(3))
    num_rows, tokens, slots_per_token, hidden = 90, 40, 8, 48
    rows = _rows(k_rows, (num_rows, hidden), dtype)
    # The sender's slots: in [0, num_rows), or the sentinel num_rows for a destination the token does not reach.
    slots = jax.random.randint(k_slots, (tokens, slots_per_token), 0, 2 * num_rows)
    slots = jnp.where(slots < num_rows, slots, num_rows).astype(jnp.int32)
    expected = jax.jit(_gather_sum_slots)(rows, slots)
    actual = jax.jit(_gather_sum_slots_clamped)(rows, slots)
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


@pytest.mark.parametrize("topk", [1, 4])
def test_pallas_gathers_match_xla_bitwise_at_june_shapes_on_gpu(topk):
    if jax.default_backend() != "gpu":
        pytest.skip("compiles the Triton kernel; needs a GPU")
    # The dispatch backward's shape at 1/8 scale: 14988 received rows from 16384 expert rows, hidden 2560.
    k_rows, k_slots, k_idx, k_valid = jax.random.split(jax.random.key(2), 4)
    rows = _rows(k_rows, (16384, 2560), jnp.bfloat16)
    slots = _slots(k_slots, 14988, topk, 16384)
    np.testing.assert_array_equal(
        _bits(jax.jit(gather_sum_rows)(rows, slots)), _bits(jax.jit(_gather_sum_slots)(rows, slots))
    )
    index = jax.random.randint(k_idx, (14988,), 0, 16384)
    valid = jax.random.bernoulli(k_valid, 0.8, (14988,))
    np.testing.assert_array_equal(
        _bits(jax.jit(gather_valid_rows)(rows, index, valid)), _bits(jax.jit(_take_valid_rows)(rows, index, valid))
    )
