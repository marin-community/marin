# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import math

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PRNGKeyArray

import haliax
import haliax.random as hrandom
from haliax.axis import Axis, AxisSelection, AxisSelector, AxisSpec, axis_name, axis_spec_to_shape_dict
from haliax.core import NamedArray
from haliax.types import PrecisionLike

# With attention, we usually distinguish between the mask and the bias, though the former is just a special case of the
# latter. In practice, the mask is a boolean array that is applied using `where` to the logits, while the bias is a
# float array that is added to the logits. The mask is usually used to prevent attention to certain positions, while
# the bias is usually used to encourage or discourage attention to certain positions.
# The mask usually is head-independent, while the bias is frequently head-dependent

# because we use named axis we can be fairly loose about the shape of masks and biases: want to have a different
# mask for each head? fine. want to broadcast across the key sequence length? fine. etc etc


def dot_product_attention_weights(
    Key: AxisSelector,
    KPos: AxisSelection,
    query: NamedArray,
    key: NamedArray,
    mask: NamedArray | None = None,
    bias: NamedArray | None = None,
    attention_dtype: jnp.dtype | None = None,
    precision: PrecisionLike = None,
    scaling_factor: float | None = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention. Computes the logits for the attention weights. Note that the
    "Pos" axis in query must be distinct from the "Pos" axis in key.

    :param Key: Axis of head dimension
    :param KPos: Axis or axes that are attended to
    :param query: NamedArray of shape (QPos, KeySize)
    :param key: NamedArray of shape (KPos, KeySize)
    :param mask: NamedArray | None broadcast compatible with (KeySize, QPos, KPos). Should be boolean
    :param bias: NamedArray | None broadcast compatible with (KeySize, QPos, KPos). Should be float
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :param scaling_factor: Optional float as scaling factor for attention score. Default to 1/sqrt(D)
    :return: NamedArray of shape (QPos, KPos)
    """
    # cf https://github.com/google/flax/blob/509bf97ea272e130d932920f45307ac98947d994/flax/linen/attention.py#L40

    orig_dtype = query.dtype
    if scaling_factor is None:
        scaling_factor = 1.0 / jnp.sqrt(query.axis_size(Key))

    query = query * scaling_factor

    if attention_dtype is not None:
        query = query.astype(attention_dtype)
        key = key.astype(attention_dtype)

    weights = haliax.dot(query, key, precision=precision, axis=Key)

    if bias is not None:
        weights = weights + bias
    if mask is not None:
        weights = haliax.where(mask, weights, -1e9)

    weights = haliax.nn.softmax(weights, axis=KPos)

    return weights.astype(orig_dtype)


def dot_product_attention(
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: NamedArray | None = None,
    bias: NamedArray | None = None,
    attention_dtype: jnp.dtype | None = None,
    precision: PrecisionLike = None,
) -> NamedArray:
    """
    NamedArray version of dot product attention. This can be multi-headed or not.

    :param KPos: Axis of key sequence length
    :param Key: Axis of head dimension
    :param query: NamedArray of shape {..., QPos, KeySize}
    :param key: NamedArray of shape {..., KPos, KeySize}
    :param value: NamedArray of shape {..., KPos, KeySize}
    :param mask: NamedArray | None broadcast compatible with (KeySize, QPos, KPos). Should be boolean
    :param bias: NamedArray | None broadcast compatible with (KeySize, QPos, KPos). Should be float
    :param attention_dtype: Optional dtype to use for attention
    :param precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
    :return: NamedArray of shape (QPos, KeySize)

    Mask and bias are given as separate arguments because they are often computed separately and have different shapes.
    For example, mask is frequently just a boolean array of shape (QPos, KPos), while bias is frequently a float
    array of shape (KeySize, QPos, KPos) or (KeySize, KPos)
    """
    if not isinstance(query, NamedArray):
        raise TypeError(
            f"query must be a NamedArray, got {type(query)}. Probably you are still using the old signature"
            "of dot_product_attention. It no longer takes a QPos argument."
        )
    KPos = axis_spec_to_shape_dict(KPos)
    KPos = key.resolve_axis(KPos)
    # any axis in KPos that's in query is a problem
    for axis in KPos:
        if query.has_axis(axis):
            raise ValueError(
                f"Axis {axis} in KPos is also in query. Attended-to axes must be distinct from query axis"
            )

    weights = dot_product_attention_weights(
        Key, KPos, query, key, mask=mask, bias=bias, attention_dtype=attention_dtype, precision=precision
    )

    return haliax.dot(weights, value, axis=KPos)


def _get_query_pos_renames(Pos):
    new_Pos: list[Axis] = []
    renames: dict[str, str] = {}
    for i, axis in enumerate(Pos):
        ax_name = axis_name(axis)
        axis = axis.alias(f"q_{ax_name}")
        renames[ax_name] = axis.name
        new_Pos.append(axis)

    return tuple(new_Pos), renames


def mask_to_bias(mask: NamedArray, mask_value: float = -1e9) -> NamedArray:
    return mask * mask_value


def combine_masks_and(mask1: NamedArray | None, mask2: NamedArray | None) -> NamedArray | None:
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
    return mask1 & mask2.broadcast_axis(mask1.axes)


def combine_masks_or(mask1: NamedArray | None, mask2: NamedArray | None) -> NamedArray | None:
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1
    return mask1 | mask2.broadcast_axis(mask1.axes)


def causal_mask(QPos: Axis, KPos: Axis, q_start: int | NamedArray = 0, k_start: int | NamedArray = 0) -> NamedArray:
    """
    Creates a materialized causal mask for attention.

    :param QPos: Axis of query sequence length
    :param KPos: Axis of key sequence length
    :return: NamedArray of shape (QPos, KPos)
    """
    # if q_start is a named array, we vmap the arange
    if isinstance(q_start, NamedArray):
        q_range = haliax.vmap(haliax.arange, q_start.axes)(QPos, start=q_start)
    else:
        q_range = haliax.arange(QPos, start=q_start)

    if isinstance(k_start, NamedArray):
        k_range = haliax.vmap(haliax.arange, k_start.axes)(KPos, start=k_start)
    else:
        k_range = haliax.arange(KPos, start=k_start)

    return q_range >= k_range.broadcast_axis(QPos)


def prefix_lm_mask(QSeqLen: Axis, KSeqLen: Axis, prefix_len: int, q_start: int = 0, k_start: int = 0) -> NamedArray:
    """Mask for the PrefixLM objective: fully connected before prefix_len, then causal after."""
    # sometimes prefix_len is a tracer so we can't assert
    if isinstance(prefix_len, int):
        assert prefix_len >= 0
        # assert prefix_len <= KSeqLen.size

    causal = causal_mask(QSeqLen, KSeqLen, q_start=q_start, k_start=k_start)
    prefix = haliax.arange(KSeqLen, start=k_start) < (prefix_len + k_start)

    return prefix | causal


def dropout_mask(axes: AxisSpec, dropout_rate: float, *, key: PRNGKeyArray) -> NamedArray:
    """
    Really just an alias for haliax.random.bernoulli. You can pass in e.g. Head, QPos and KPos
    """
    return hrandom.bernoulli(key, shape=axes, p=1 - dropout_rate)


def forgetful_causal_mask(KPos: Axis, mask_prob: float, sample_prob: bool = True, *, key: PRNGKeyArray) -> NamedArray:
    """
    Forgetful Context Masking a la https://arxiv.org/abs/2210.13432. Randomly drops out positions from the key sequence.
    Reportedly better than normal attention dropout. Almost certainly faster.

    You're always allowed to attend to the 0th position. (They say BOS token, but we don't always start with bos)

    :param KPos: Axis of key sequence length
    :param mask_prob: Probability a position to mask
    :param sample_prob: If True, sample the prob between 0 and the provided prob (this is what the paper does)
    """
    zeroth_on = haliax.nn.one_hot(0, KPos, dtype=jnp.bool_)  # always allow 0th position
    if mask_prob == 0:
        return haliax.ones((KPos,), dtype=jnp.bool_)
    elif mask_prob == 1:
        return zeroth_on
    else:
        if sample_prob:
            key, subkey = jax.random.split(key)
            mask_prob = jax.random.uniform(subkey, shape=(), minval=0, maxval=mask_prob)
        base: NamedArray = hrandom.bernoulli(key, shape=(KPos,), p=1 - mask_prob)
        return base | zeroth_on


def _get_alibi_slopes(heads: int, bias_max: float) -> list[float]:
    # Mosaic supports "bias_max"
    log_bias_max = math.log2(bias_max)
    # from https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742

    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - log_bias_max)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(heads).is_integer():
        return get_slopes_power_of_2(heads)
    closest_power_of_2 = 2 ** math.floor(math.log2(heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]
    )


def alibi_attention_bias(Heads: Axis, KPos: Axis, bias_max: float = 8, dtype=jnp.float32) -> NamedArray:
    """
    Creates an attention bias for alibi attention.

    :param KPos: Axis of (key) sequence length
    :param Heads: Axis of heads
    :return: NamedArray of shape (Heads, KPos)
    """
    slopes = haliax.named(np.array(_get_alibi_slopes(Heads.size, bias_max)), Heads)
    positions = haliax.arange(KPos).broadcast_axis(Heads)

    biases = slopes * positions
    return biases.astype(dtype)
