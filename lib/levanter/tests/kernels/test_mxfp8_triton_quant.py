# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bitwise gate for the Triton MXFP8 dual-quantize producer.

The inputs are the aligned receiver buffer the MXFP8 expert MLP quantizes in place: each expert
starts on a GROUP_PAD-row boundary, and the rows in the gaps between experts and past the last
were not received and hold NaN and +inf. `dual_quantize_padded` must return the bytes of the
pure-JAX composition ``dual_quantize_activation`` of that buffer with every unreceived row zeroed:
both FP8 copies and both swizzled scale layouts. `quantize_rows_padded` must return its row copy
and row scales. The received rows hold experts without rows and 32x32 blocks built to hit each
corner of the scale and rounding rules. The kernel runs only on a GPU, so these tests skip
elsewhere.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.grug._moe.mxfp8.layout import GROUP_PAD, aligned_dispatch_layout, wgrad_block_perm
from levanter.grug._moe.mxfp8.quantize import dual_quantize_activation
from levanter.grug._moe.mxfp8.triton_quant import dual_quantize_padded, quantize_rows_padded

_OUTPUTS = ("q_rows", "sfa", "q_tokens", "sf_wgrad")
_BLOCK = 32
_BF16_MAX = 3.3895313892515355e38
_HERO_CAPACITY = 302_336
# Hero H; the down projection's cotangent has I = H = 3072 columns too, so one width covers x and dY.
_HERO_HIDDEN = 3072


def _requires_gpu():
    if jax.default_backend() != "gpu":
        pytest.skip("requires the JAX GPU backend")


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def _pow2(exponent):
    return jax.lax.bitcast_convert_type((exponent + 127) << 23, jnp.float32)


@functools.partial(jax.jit, static_argnames=("hidden",))
def _adversarial_buffer(key, received, *, hidden: int):
    """A bf16 ``[rows, hidden]`` buffer whose received rows' 32x32 blocks each take one of eight kinds.

    The other rows alternate NaN and +inf, which the quantizer must read as zero.
    """
    rows = received.shape[0]
    keys = jax.random.split(key, 10)
    block_shape = (-(-rows // _BLOCK), hidden // _BLOCK)

    def per_element(block_values):
        return jnp.repeat(jnp.repeat(block_values, _BLOCK, 0), _BLOCK, 1)[:rows]

    kind = per_element(jax.random.randint(keys[0], block_shape, 0, 8))
    block_exponent = per_element(jax.random.randint(keys[1], block_shape, -20, 21))
    shape = (rows, hidden)
    normal = jax.random.normal(keys[2], shape, jnp.float32)
    negative = jax.random.bernoulli(keys[3], 0.5, shape)
    sign = jnp.where(negative, -1.0, 1.0)
    row = jnp.arange(rows)[:, None]
    col = jnp.arange(hidden)[None, :]
    # One element of every block row and every block column, so each row block and token block
    # inside a 32x32 block holds one.
    diagonal = row % _BLOCK == col % _BLOCK
    block_scale = _pow2(block_exponent)
    # Values with one more mantissa bit than E4M3 at scale 1, halfway between two E4M3 numbers when
    # the bit is set, down into E4M3's subnormals.
    tie = (16 + jax.random.randint(keys[4], shape, 0, 16)) / 16 * _pow2(jax.random.randint(keys[5], shape, -12, 8))
    kinds = [
        # One magnitude per block.
        normal * block_scale,
        # 2^48 of range inside every row and token block, so most elements land in E4M3's subnormals
        # or flush to zero.
        normal * _pow2(block_exponent + jax.random.randint(keys[6], shape, -24, 25)),
        # Signed zeros: the scale takes the amax floor.
        sign * 0.0,
        # Amax exactly 448 * 2^k, where the scale must not round up.
        jnp.where(jax.random.bernoulli(keys[7], 0.25, shape) | diagonal, sign * 448.0, normal.clip(-3.9, 3.9) * 112.0)
        * block_scale,
        # Round-to-nearest-even ties at scale 2^k.
        jnp.where(diagonal, sign * 448.0, sign * tie) * block_scale,
        # Kind 5 is replaced by bf16 subnormals below.
        normal,
        # Amaxes around the 1e-4 floor.
        jax.random.uniform(keys[8], shape, minval=-2e-4, maxval=2e-4) * _pow2(-(block_exponent % 3)),
        # Up to the largest finite bf16.
        sign * jnp.where(diagonal, _BF16_MAX, jax.random.uniform(keys[9], shape, minval=0.5, maxval=1.0) * 3.38e38),
    ]
    x = kinds[0]
    for index, values in enumerate(kinds[1:], start=1):
        x = jnp.where(kind == index, values, x)
    x = x.astype(jnp.bfloat16)
    subnormal_bits = (negative.astype(jnp.uint16) << 15) | (jnp.abs(normal * 40).astype(jnp.uint16) % 127 + 1)
    x = jnp.where(kind == 5, jax.lax.bitcast_convert_type(subnormal_bits, jnp.bfloat16), x)
    unreceived = jnp.where(row % 2 == 0, jnp.nan, jnp.inf).astype(jnp.bfloat16)
    return jnp.where(received[:, None], x, unreceived)


@jax.jit
def _reference(x, received, block_perm):
    return dual_quantize_activation(jnp.where(received[:, None], x, 0), block_perm)


_kernel = jax.jit(dual_quantize_padded)
_rows_kernel = jax.jit(quantize_rows_padded)


def _assert_bitwise_equal(got, want):
    assert len(got) == len(want), f"{len(got)} outputs, expected {len(want)}"
    for name, g, w in zip(_OUTPUTS, got, want, strict=False):
        assert (g.shape, g.dtype) == (w.shape, w.dtype), f"{name}: {g.dtype}{g.shape} vs {w.dtype}{w.shape}"
        differ = jax.lax.bitcast_convert_type(g, jnp.uint8) != jax.lax.bitcast_convert_type(w, jnp.uint8)
        mismatches = int(jnp.count_nonzero(differ))
        if mismatches:
            first = np.argwhere(np.asarray(differ))[:8].tolist()
            pytest.fail(f"{name}: {mismatches} of {w.size} bytes differ, first at {first}")


def _case(capacity: int, hidden: int, active_sizes: tuple[int, ...], seed: int):
    """The adversarial aligned receiver buffer, its layout, its received rows, and the token scales' tile order.

    The ragged backend gives ``capacity`` received rows room for every expert to start on a
    GROUP_PAD-row boundary, and the last expert's span takes the buffer's trailing rows.
    """
    received = np.zeros(_round_up(capacity + (GROUP_PAD - 1) * len(active_sizes), GROUP_PAD), bool)
    physical, start = [], 0
    for size in active_sizes:
        received[start : start + size] = True
        physical.append(_round_up(size, GROUP_PAD))
        start += physical[-1]
    physical[-1] += received.size - start
    layout = aligned_dispatch_layout(
        jnp.asarray(physical, jnp.int32), jnp.asarray(active_sizes, jnp.int32), rows=received.size
    )
    block_perm = wgrad_block_perm(layout.padded_group_sizes, rows=hidden, total_tokens=layout.padded_rows)
    received = jnp.asarray(received)
    x = _adversarial_buffer(jax.random.key(seed), received, hidden=hidden)
    return x, layout, received, block_perm


def _check(capacity: int, hidden: int, active_sizes: tuple[int, ...], seed: int):
    x, layout, received, block_perm = _case(capacity, hidden, active_sizes, seed)
    _assert_bitwise_equal(_kernel(x, layout, block_perm), _reference(x, received, block_perm))


def _check_rows(capacity: int, hidden: int, active_sizes: tuple[int, ...], seed: int):
    x, layout, received, block_perm = _case(capacity, hidden, active_sizes, seed)
    q_rows, sfa, _q_tokens, _sf_wgrad = _reference(x, received, block_perm)
    _assert_bitwise_equal(_rows_kernel(x, layout), (q_rows, sfa))


_SMALL_CASES = [
    pytest.param(1000, 256, (300, 0, 513), id="zero-token-expert-gaps-and-tail"),
    pytest.param(768, 128, (256, 256, 256), id="full-capacity-no-gaps"),
    pytest.param(1500, 384, (0, 1, 1200, 0), id="odd-width-empty-ends"),
    pytest.param(512, 512, (0, 0), id="no-received-rows"),
    pytest.param(4096, 3072, (1, 2500, 37, 1000), id="hero-width-ragged"),
]


@pytest.mark.parametrize(("capacity", "hidden", "active_sizes"), _SMALL_CASES)
def test_dual_quantize_padded_matches_reference_bitwise(capacity, hidden, active_sizes):
    _requires_gpu()
    _check(capacity, hidden, active_sizes, seed=capacity + hidden)


@pytest.mark.parametrize(("capacity", "hidden", "active_sizes"), _SMALL_CASES)
def test_quantize_rows_padded_matches_reference_row_copy_bitwise(capacity, hidden, active_sizes):
    _requires_gpu()
    _check_rows(capacity, hidden, active_sizes, seed=capacity + hidden)


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "active_sizes",
    [
        pytest.param((182_000, 0, 117_500), id="skewed-zero-expert-tail"),
        pytest.param((100_779, 100_779, 100_778), id="balanced-full"),
    ],
)
def test_dual_quantize_padded_matches_reference_bitwise_at_hero_chunk(active_sizes):
    _requires_gpu()
    _check(_HERO_CAPACITY, _HERO_HIDDEN, active_sizes, seed=sum(active_sizes))


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_quantize_rows_padded_matches_reference_row_copy_bitwise_at_hero_chunk():
    _requires_gpu()
    active_sizes = (182_000, 0, 117_500)  # skewed, an expert without rows, and a NaN/inf tail
    _check_rows(_HERO_CAPACITY, _HERO_HIDDEN, active_sizes, seed=sum(active_sizes))
