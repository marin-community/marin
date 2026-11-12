# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug.grug_moe import qb_beta_topk_shard
from levanter.kernels.pallas.kth_largest import kth_largest, kth_largest_reference


def _rows(kind: str, rows: int, columns: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((rows, columns)) * 10.0 ** rng.uniform(-20, 20)).astype(np.float32)
    if kind == "ties":
        x = np.round(x / np.abs(x).max() * 4) / 2
    elif kind == "masked":
        x[rng.random(x.shape) < 0.6] = -np.inf
    elif kind == "zeros_and_subnormals":
        x[rng.random(x.shape) < 0.3] = 0.0
        x[rng.random(x.shape) < 0.1] = np.float32(1e-42)
        x[rng.random(x.shape) < 0.1] = np.float32(-1e-42)
    elif kind == "infinities":
        x[rng.random(x.shape) < 0.05] = np.inf
        x[rng.random(x.shape) < 0.05] = -np.inf
    return x


def _top_k_kth(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    values = np.asarray(jax.lax.top_k(jnp.asarray(x), int(k.max()))[0])
    return values[np.arange(x.shape[0]), k - 1]


def _bits(x) -> np.ndarray:
    return np.asarray(x).view(np.uint32)


@pytest.mark.parametrize(
    ("kind", "rows", "columns"),
    [
        ("normal", 5, 37),
        ("ties", 16, 512),
        ("masked", 3, 1000),
        ("zeros_and_subnormals", 5, 37),
        ("infinities", 16, 512),
    ],
)
def test_kth_largest_matches_top_k_bit_for_bit(kind, rows, columns):
    x = _rows(kind, rows, columns, seed=rows * columns)
    rng = np.random.default_rng(columns)
    per_row_k = rng.integers(1, columns + 1, rows).astype(np.int32)
    per_row_k[0] = 1
    per_row_k[1] = columns
    want = _bits(_top_k_kth(x, per_row_k))
    for radix_bits in (1, 2, 4):
        np.testing.assert_array_equal(
            _bits(kth_largest_reference(jnp.asarray(x), jnp.asarray(per_row_k), radix_bits=radix_bits)), want
        )
    interpret = jax.default_backend() == "cpu"
    kernel = jax.jit(kth_largest, static_argnames=("block", "interpret"))
    np.testing.assert_array_equal(_bits(kernel(jnp.asarray(x), jnp.asarray(per_row_k), interpret=interpret)), want)
    if kind == "normal":
        # The short block forces several tiles and a padded final tile.
        np.testing.assert_array_equal(
            _bits(kernel(jnp.asarray(x), jnp.asarray(per_row_k), block=16, interpret=interpret)), want
        )


def test_qb_thresholds_from_radix_select_equal_top_k():
    tokens, experts, topk = 256, 8, 2
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.standard_normal((tokens, experts)).astype(np.float32))
    for valid in (np.ones(tokens, bool), rng.random(tokens) < 0.7, np.zeros(tokens, bool)):
        kwargs = dict(physical_count=tokens * topk // experts, num_experts_per_token=topk, num_experts=experts)
        want = qb_beta_topk_shard(scores, jnp.asarray(valid), threshold_impl="top_k", **kwargs)
        got = qb_beta_topk_shard(scores, jnp.asarray(valid), threshold_impl="radix_select", **kwargs)
        np.testing.assert_array_equal(_bits(got[0]), _bits(want[0]))
        assert int(got[1]) == int(want[1])
