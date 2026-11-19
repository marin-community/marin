# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug import grug_moe
from levanter.grug.grug_moe import qb_beta_topk_shard
from levanter.kernels.pallas.radix_select import radix_select, radix_select_reference


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
def test_radix_select_matches_top_k_bit_for_bit(kind, rows, columns):
    x = _rows(kind, rows, columns, seed=rows * columns)
    rng = np.random.default_rng(columns)
    per_row_k = rng.integers(1, columns + 1, rows).astype(np.int32)
    per_row_k[0] = 1
    per_row_k[1] = columns
    want = _bits(_top_k_kth(x, per_row_k))
    for radix_bits in (1, 2, 4):
        np.testing.assert_array_equal(
            _bits(radix_select_reference(jnp.asarray(x), jnp.asarray(per_row_k), radix_bits=radix_bits)), want
        )
    interpret = jax.default_backend() == "cpu"
    kernel = jax.jit(radix_select, static_argnames=("block", "interpret"))
    np.testing.assert_array_equal(_bits(kernel(jnp.asarray(x), jnp.asarray(per_row_k), interpret=interpret)), want)
    if kind == "normal":
        # The short block forces several tiles and a padded final tile.
        np.testing.assert_array_equal(
            _bits(kernel(jnp.asarray(x), jnp.asarray(per_row_k), block=16, interpret=interpret)), want
        )


@pytest.mark.parametrize(
    ("radix_bits", "block", "num_warps"), [(1, 64, 1), (2, 128, 2), (4, 32, 4), (2, 512, 8), (4, 1024, 2)]
)
def test_radix_select_launch_shapes_match_top_k(radix_bits, block, num_warps):
    # Rows with ties, -0.0 and +0.0, -inf and repeated maxima; k at 1, at n and in between. A block shorter than the
    # row makes each lane count several tiles, the last one padded; a longer one keeps the padded row in registers.
    rows, columns = 8, 300
    rng = np.random.default_rng(radix_bits)
    x = np.round(rng.standard_normal((rows, columns)) * 2).astype(np.float32) / 2
    x[1] = 1.5
    x[2, ::3] = -0.0
    x[2, 1::3] = 0.0
    x[3, rng.random(columns) < 0.5] = -np.inf
    x[4, :7] = x[4].max() + 1
    k = rng.integers(1, columns + 1, rows).astype(np.int32)
    k[0], k[1], k[4], k[5] = 1, columns, 7, columns - 1
    want = _top_k_kth(x, k)
    kernel = jax.jit(radix_select, static_argnames=("radix_bits", "block", "num_warps", "interpret"))
    got = kernel(
        jnp.asarray(x), jnp.asarray(k), radix_bits=radix_bits, block=block, num_warps=num_warps, interpret=True
    )
    # lax.top_k may return either zero where -0.0 and +0.0 tie; the reference orders them as totalOrder does.
    np.testing.assert_array_equal(np.asarray(got), want)
    np.testing.assert_array_equal(_bits(got), _bits(radix_select_reference(jnp.asarray(x), jnp.asarray(k))))


@pytest.mark.parametrize("valid_fraction", [1.0, 0.7, 0.0])
def test_qb_thresholds_on_amd_instinct_equal_top_k(monkeypatch, valid_fraction):
    # AMD Instinct takes the radix-select kernel (run here in the interpreter); every other device takes lax.top_k.
    tokens, experts, topk = 256, 8, 2
    rng = np.random.default_rng(0)
    scores = jnp.asarray(rng.standard_normal((tokens, experts)).astype(np.float32))
    valid = jnp.asarray(rng.random(tokens) < valid_fraction)
    kwargs = dict(physical_count=tokens * topk // experts, num_experts_per_token=topk, num_experts=experts)
    want = qb_beta_topk_shard(scores, valid, **kwargs)
    monkeypatch.setattr(grug_moe, "is_amd_instinct", lambda: True)
    monkeypatch.setattr(grug_moe, "radix_select", functools.partial(radix_select, interpret=True))
    got = qb_beta_topk_shard(scores, valid, **kwargs)
    np.testing.assert_array_equal(_bits(got[0]), _bits(want[0]))
    assert int(got[1]) == int(want[1])
