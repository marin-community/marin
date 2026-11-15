# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the MXFP8 dispatch layout, scale swizzles and reference quantizer."""

import numpy as np
import pytest

import jax.numpy as jnp

from levanter.grug._moe.ep_common import _aligned_receiver_rows
from levanter.grug._moe.mxfp8.layout import GROUP_PAD, aligned_dispatch_layout, wgrad_block_perm
from levanter.grug._moe.mxfp8.quantize import (
    WeightMajor,
    build_sfb,
    dequantize_mxfp8,
    dequantize_mxfp8_tokens,
    dual_quantize_activation,
    e8m0_to_f32,
    quantize_mxfp8,
    quantize_mxfp8_tokens,
    quantize_weights,
    saturating_cast_e4m3,
)

# (received rows per expert, logical receiver capacity).
LAYOUT_CASES = [
    pytest.param([3, 0, 511, 256, 30], 900, id="ragged-with-empty-expert-and-tail"),
    pytest.param([257, 1, 1, 1], 260, id="worst-case-rounding-fills-buffer"),
    pytest.param([0, 0, 0, 7], 64, id="mostly-empty"),
    pytest.param([0, 0, 0, 0], 32, id="nothing-received"),
    pytest.param([300, 0], 400, id="empty-last-expert"),
    pytest.param([1024], 1024, id="single-expert-no-tail"),
    pytest.param([1] * 8, 8, id="all-tiny"),
]


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def _aligned_spans(active: list[int], capacity: int) -> list[int]:
    """Each expert's span in the aligned receiver buffer: its rows rounded up, the last taking the rest."""
    rows = _aligned_receiver_rows(capacity, len(active), GROUP_PAD)
    spans = [_round_up(n, GROUP_PAD) for n in active]
    spans[-1] += rows - sum(spans)
    return spans


def _swizzle_reference(scales: np.ndarray) -> np.ndarray:
    """SM100 block-scaled layout from its index formula: 128x4 tiles, 512 bytes each, row-block-major."""
    rows, cols = scales.shape
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    tile = (r // 128) * (cols // 4) + c // 4
    offset = tile * 512 + (r % 32) * 16 + (r % 128) // 32 * 4 + c % 4
    out = np.empty(rows * cols, np.uint8)
    out[offset.ravel()] = scales.ravel()
    return out


def _bytes(a) -> np.ndarray:
    return np.asarray(a).view(np.uint8)


@pytest.mark.parametrize(("active", "capacity"), LAYOUT_CASES)
def test_aligned_dispatch_layout_keeps_each_experts_received_rows_and_zeroes_the_rest(active, capacity):
    # The ragged backend lays rows out for the MXFP8 expert MLP from GROUP_PAD-aligned starts and
    # leaves every other row unwritten (NaN here). Read in place through the layout, the buffer must
    # give the kernels each expert's received rows at its aligned start, zeros everywhere else, and
    # groups that end at the span ends, except the last, which ends at its received rows rounded up:
    # the kernels skip the buffer's unused rows.
    rng = np.random.default_rng(len(active) * 1000 + capacity)
    packed = rng.standard_normal((sum(active), 8)).astype(np.float32)
    spans = _aligned_spans(active, capacity)
    rows = sum(spans)
    assert rows % GROUP_PAD == 0 and all(span >= n for span, n in zip(spans, active, strict=True))
    aligned = np.full((rows, packed.shape[1]), np.nan, np.float32)
    expected = np.zeros_like(aligned)
    aligned_start = packed_start = 0
    for span, n in zip(spans, active, strict=True):
        aligned[aligned_start : aligned_start + n] = packed[packed_start : packed_start + n]
        expected[aligned_start : aligned_start + n] = packed[packed_start : packed_start + n]
        aligned_start += span
        packed_start += n

    layout = aligned_dispatch_layout(jnp.asarray(spans, jnp.int32), jnp.asarray(active, jnp.int32), rows=rows)

    assert layout.padded_rows == rows
    read = np.where(np.asarray(layout.source_valid)[:, None], aligned, 0)
    np.testing.assert_array_equal(read, expected)
    groups = [*spans[:-1], _round_up(active[-1], GROUP_PAD)]
    np.testing.assert_array_equal(np.asarray(layout.group_ends), np.cumsum(groups))
    np.testing.assert_array_equal(np.asarray(layout.padded_group_sizes), groups)


@pytest.mark.parametrize(("active", "capacity"), LAYOUT_CASES)
@pytest.mark.parametrize("rows", [128, 384])
def test_wgrad_block_perm_matches_host_expert_chunk_order(active, capacity, rows):
    # The experts' groups as the layout holds them, the last stopping short of the buffer's unused
    # rows, whose tiles come after every expert's chunk.
    spans = _aligned_spans(active, capacity)
    total_tokens = sum(spans)
    groups = [*spans[:-1], _round_up(active[-1], GROUP_PAD)]
    token_blocks = total_tokens // 128

    expected = []
    token_block_start = 0
    for group in [*groups, total_tokens - sum(groups)]:
        for row_block in range(rows // 128):
            first = row_block * token_blocks + token_block_start
            expected.extend(range(first, first + group // 128))
        token_block_start += group // 128

    perm = wgrad_block_perm(jnp.asarray(groups, jnp.int32), rows=rows, total_tokens=total_tokens)
    np.testing.assert_array_equal(np.asarray(perm), expected)


@pytest.mark.parametrize(("active", "capacity"), LAYOUT_CASES[:3])
def test_dual_quantize_activation_scales_match_per_expert_swizzles(active, capacity):
    hidden = 256
    spans = _aligned_spans(active, capacity)
    padded_rows = sum(spans)
    rng = np.random.default_rng(3)
    t = rng.standard_normal((padded_rows, hidden)) * 2.0 ** rng.integers(-8, 8, (padded_rows, 1))
    t = jnp.asarray(t, jnp.float32)
    perm = wgrad_block_perm(jnp.asarray(spans, jnp.int32), rows=hidden, total_tokens=padded_rows)

    _, sfa, _, sf_wgrad = dual_quantize_activation(t, perm)

    row_scales = np.asarray(quantize_mxfp8(t)[1])
    token_scales = np.asarray(quantize_mxfp8_tokens(t)[1]).T  # [hidden, tokens / 32]
    expected_sfa, expected_wgrad = [], []
    start = 0
    for group in spans:
        if group:
            expected_sfa.append(_swizzle_reference(row_scales[start : start + group]))
            expected_wgrad.append(_swizzle_reference(token_scales[:, start // 32 : (start + group) // 32]))
        start += group
    np.testing.assert_array_equal(_bytes(sfa).ravel(), np.concatenate(expected_sfa))
    np.testing.assert_array_equal(_bytes(sf_wgrad).ravel(), np.concatenate(expected_wgrad))


def test_build_sfb_swizzles_each_expert_independently():
    rng = np.random.default_rng(4)
    weight_scales = rng.integers(100, 140, (3, 256, 8), dtype=np.uint8)
    expected = np.stack([_swizzle_reference(s) for s in weight_scales])
    np.testing.assert_array_equal(_bytes(build_sfb(jnp.asarray(weight_scales))), expected)


def test_quantize_weights_in_their_stored_layout_matches_quantizing_the_contraction_last_transpose():
    # A [E, K, N] weight quantized in place for an N-major GEMM must give the kernels the operand of
    # the [E, N, K] transpose quantized along K: the same values, transposed, and the same scale bytes.
    rng = np.random.default_rng(5)
    magnitudes = 2.0 ** rng.integers(-12, 12, (3, 1, 256))
    weights = jnp.asarray(rng.standard_normal((3, 384, 256)) * magnitudes, jnp.bfloat16)  # [E, K, N]

    n_major, n_major_scales = quantize_weights(weights, WeightMajor.N)
    k_major, k_major_scales = quantize_weights(jnp.swapaxes(weights, 1, 2), WeightMajor.K)

    np.testing.assert_array_equal(_bytes(n_major), np.swapaxes(_bytes(k_major), 1, 2))
    np.testing.assert_array_equal(_bytes(n_major_scales), _bytes(k_major_scales))


def test_quantize_mxfp8_rounds_scaled_values_half_to_even():
    cases = [
        (1.0625, 1.0),  # tie between 1 and 1.125: even mantissa
        (1.1875, 1.25),  # tie between 1.125 and 1.25
        (1.0625 + 2**-20, 1.125),  # just above a tie; rounding through bf16 or f16 first gives 1.0
        (-1.0625 - 2**-20, -1.125),
        (3 * 2**-10, 2**-8),  # tie between two subnormals
        (2**-10, 0.0),  # half the smallest subnormal
        (250.0, 256.0),
        (447.0, 448.0),
    ]
    block_scale = 2.0**-5
    block = np.zeros(32, np.float32)
    block[0] = 448.0 * block_scale
    block[1 : 1 + len(cases)] = [value * block_scale for value, _ in cases]

    q, scales = quantize_mxfp8(jnp.asarray(block))

    assert float(e8m0_to_f32(scales)[0]) == block_scale
    expected = np.array([448.0] + [rounded for _, rounded in cases], np.float32)
    np.testing.assert_array_equal(np.asarray(q.astype(jnp.float32))[: len(expected)], expected)


def test_saturating_cast_e4m3_clamps_out_of_range_values_to_448():
    values = np.array([448.0, 449.0, 464.0, 465.0, 1e6, -1e6], np.float32)
    cast = np.asarray(saturating_cast_e4m3(jnp.asarray(values)).astype(jnp.float32))
    np.testing.assert_array_equal(cast, [448.0, 448.0, 448.0, 448.0, 448.0, -448.0])


def test_quantize_mxfp8_scale_is_smallest_power_of_two_covering_the_block():
    rng = np.random.default_rng(6)
    # Down to blocks near 2^-110, whose scales stay above the smallest E8M0 scale.
    magnitudes = 2.0 ** rng.integers(-110, 40, (64, 8))
    x = (rng.standard_normal((64, 8, 32)) * magnitudes[..., None]).astype(np.float32)

    _, scales = quantize_mxfp8(jnp.asarray(x.reshape(64, 256)))

    scale = np.asarray(e8m0_to_f32(scales), np.float64)
    mantissa, _ = np.frexp(scale)
    np.testing.assert_array_equal(mantissa, 0.5)
    amax = np.abs(x).max(axis=-1).astype(np.float64)
    assert (amax <= 448.0 * scale).all()
    assert (amax > 224.0 * scale).all()


def test_quantize_mxfp8_zero_blocks_quantize_to_zero_with_the_smallest_scale():
    x = np.zeros((2, 128), np.float32)
    x[0, 32:64] = 3.0

    q, scales = quantize_mxfp8(jnp.asarray(x))

    zero_blocks = np.array([[True, False, True, True], [True, True, True, True]])
    np.testing.assert_array_equal(np.asarray(e8m0_to_f32(scales))[zero_blocks], 2.0**-126)
    np.testing.assert_array_equal(np.asarray(dequantize_mxfp8(q, scales)), x)


def test_quantize_mxfp8_quantizes_tiny_blocks_as_finely_as_unit_blocks():
    # The expert-output cotangent of a mean-reduced loss without loss scaling is about 1e-9. Scaled
    # by a power of two, a block must keep its E4M3 values and shift its scale; a floor on the block
    # amax would flush such blocks into E4M3's subnormals instead.
    rng = np.random.default_rng(9)
    x = rng.standard_normal((64, 256)).astype(np.float32)

    unit_values, unit_scales = quantize_mxfp8(jnp.asarray(x))
    tiny_values, tiny_scales = quantize_mxfp8(jnp.asarray(x * 2.0**-30))

    np.testing.assert_array_equal(_bytes(tiny_values), _bytes(unit_values))
    np.testing.assert_array_equal(np.asarray(tiny_scales, np.int32), np.asarray(unit_scales, np.int32) - 30)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_dequantize_mxfp8_error_is_within_half_an_e4m3_step(dtype):
    rng = np.random.default_rng(7)
    magnitudes = 2.0 ** rng.integers(-20, 20, (128, 8))
    x = jnp.asarray(rng.standard_normal((128, 8, 32)) * magnitudes[..., None], dtype).reshape(128, 256)

    q, scales = quantize_mxfp8(x)
    error = np.abs(np.asarray(dequantize_mxfp8(q, scales)) - np.asarray(x, np.float32))

    scale = np.repeat(np.asarray(e8m0_to_f32(scales)), 32, axis=1)
    # E4M3 keeps 3 mantissa bits; its subnormals are spaced 2^-9 apart in units of the scale.
    bound = np.maximum(np.abs(np.asarray(x, np.float32)) * 2.0**-4, scale * 2.0**-10)
    assert (error <= bound).all()
    assert error.max() > 0


def test_quantize_mxfp8_tokens_matches_row_quantization_of_the_transpose():
    rng = np.random.default_rng(8)
    x = jnp.asarray(rng.standard_normal((128, 64)) * 2.0 ** rng.integers(-10, 10, (1, 64)), jnp.float32)

    q_tokens, token_scales = quantize_mxfp8_tokens(x)
    q_rows, row_scales = quantize_mxfp8(x.T)

    np.testing.assert_array_equal(_bytes(q_tokens), _bytes(q_rows).T)
    np.testing.assert_array_equal(np.asarray(token_scales), np.asarray(row_scales).T)
    np.testing.assert_array_equal(
        np.asarray(dequantize_mxfp8_tokens(q_tokens, token_scales)), np.asarray(dequantize_mxfp8(q_rows, row_scales)).T
    )
