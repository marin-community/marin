# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
import logging
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Optional, Union, overload

import equinox as eqx
import jax
import jax.random as jrandom
from jax import numpy as jnp

from ..inference.utils import is_valid

try:
    from jax.experimental.pallas.ops.tpu.ragged_paged_attention import (
        ragged_paged_attention as tpu_ragged_paged_attention,
    )

    raise ImportError("Disabling TPU ragged paged attention until bugs are fixed.")
except Exception:  # pragma: no cover - optional dep
    tpu_ragged_paged_attention = None

import haliax
import haliax as hax
import haliax.haxtyping as ht
import haliax.nn as hnn
from haliax import Axis, AxisSelection, AxisSelector, NamedArray, axis_name
from haliax.jax_utils import maybe_rng_split, named_call
from haliax.nn.attention import causal_mask, combine_masks_and, combine_masks_or
from haliax.nn.normalization import LayerNormBase
from haliax.partitioning import pspec_for_axis
from haliax.types import PrecisionLike
from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from jaxtyping import PRNGKeyArray

from ..inference.page_table import PageBatchInfo, PageTable
from .normalization import LayerNormConfigBase
from .rotary import RotaryEmbeddings, RotaryEmbeddingsConfig

logger = logging.getLogger(__name__)


class AttentionBackend(Enum):
    DEFAULT = "default"  # use the default attention type for the accelerator
    NVTE = "nvte"  # with Transformer Engine on NVIDIA GPUs
    SPLASH = "splash"  # on TPU.
    JAX_FLASH = "jax_flash"  # Use the JAX reference implementation
    VANILLA = "vanilla"  # regular dot product attention


def default_attention_type() -> AttentionBackend:
    accelerator_type = jax.local_devices()[0].platform
    if accelerator_type == "gpu":
        return AttentionBackend.NVTE
    elif accelerator_type == "tpu":
        return AttentionBackend.SPLASH
    else:
        return AttentionBackend.JAX_FLASH


@named_call
def dot_product_attention(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union["AttentionMask", NamedArray]] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    use_flash: Optional[bool] = None,
    attn_backend: Optional[AttentionBackend] = None,
    flash_block_size: Optional[int] = None,
    dropout: float = 0.0,
    *,
    logits_soft_cap: float | None = None,
    scaling_factor: float | None = None,
    inference: bool = True,
    prng: PRNGKeyArray | None = None,
):
    """
    This method is similar to [haliax.nn.attention.dot_product_attention][] but it can use different backends for
    attention. In particular, it can use the Transformer Engine for NVIDIA GPUs, the Splash Attention kernel for TPUs,
    or a pure JAX reference flash attention 2 implementation for other platforms, or it can fall back to regular dot
    product attention.

    It also uses the [AttentionMask][] class, which we might move to haliax.nn.attention in the future.
    Unlike the Haliax version, it requires that the QPos and KPos already be different.

    Args:
        Key: Size of key dimension
        QPos: Axis of query sequence length. Can be an AxisSpec to attend along more than one axis.
        KPos: Axis of key sequence length. Can be an AxisSpec to attend along more than one axis.
        query: shape at least {QPos, KeySize}
        key: shape at least {KPos, KeySize}
        value: shape at least {KPos, ValueSize}
        mask: attention mask
        bias: Optional[NamedArray] broadcast compatible with (KeySize, QPos, KPos). Should be float
        attention_dtype: Optional dtype to use for attention
        precision: PrecisionLike for dot product. See precision argument to jax.lax.dot_general
        use_flash: whether to use flash attention
        attn_backend: AttentionBackend to use. If None, will use the default for the accelerator.
        flash_block_size: block size for flash attention. If None, will use an appropriate default
        dropout: dropout rate
        inference: whether to use inference mode
        prng: PRNGKeyArray for dropout
        scaling_factor: If not None, query will be multiplied by this value before attention.
             default is 1/sqrt(HeadSize.size)
        logits_soft_cap: If not None, the attention logits will be soft_capped with tanh(logits / logits_soft_cap) * logits_soft_cap.
    Returns:
        NamedArray of shape (value.axes - KPos + QPos)
    """
    if axis_name(QPos) == axis_name(KPos):
        raise ValueError("QPos and KPos must have different names")

    if use_flash is not None:
        if attn_backend is None:
            if not use_flash:
                attn_backend = AttentionBackend.VANILLA
            else:
                attn_backend = AttentionBackend.DEFAULT
        else:
            if attn_backend != AttentionBackend.VANILLA and not use_flash:
                raise ValueError("use_flash is False, but flash_backend is not VANILLA")
            elif attn_backend == AttentionBackend.VANILLA and use_flash:
                raise ValueError("use_flash is True, but flash_backend is VANILLA")
    elif use_flash is None and attn_backend is None:
        # if the block_size doesn't divide the seq lens, we can't use flash. Previously default was use_flash=False
        if flash_block_size is not None:
            qlen = query.axis_size(QPos)
            klen = key.axis_size(KPos)
            if qlen % flash_block_size != 0 or klen % flash_block_size != 0:
                use_flash = False
                attn_backend = AttentionBackend.VANILLA

    if attn_backend is None or attn_backend == AttentionBackend.DEFAULT:
        was_default = True
        attn_backend = default_attention_type()
    else:
        was_default = False

    if scaling_factor is None:
        scaling_factor = 1 / math.sqrt(query.resolve_axis(Key).size)

    match attn_backend:
        case AttentionBackend.NVTE:
            attention_out = _try_te_attention(
                QPos,
                KPos,
                Key,
                query,
                key,
                value,
                mask,
                bias,
                dropout,
                inference,
                prng=prng,
                attention_dtype=attention_dtype,
                precision=precision,
                flash_block_size=flash_block_size,
                force_te=not was_default,
                scaling_factor=scaling_factor,
                logits_soft_cap=logits_soft_cap,
            )
        case AttentionBackend.SPLASH:
            attention_out = _try_tpu_splash_attention(
                QPos,
                KPos,
                Key,
                query,
                key,
                value,
                mask,
                bias,
                dropout,
                inference,
                force_flash=not was_default,
                prng=prng,
                attention_dtype=attention_dtype,
                precision=precision,
                block_size=flash_block_size,
                scaling_factor=scaling_factor,
                logits_soft_cap=logits_soft_cap,
            )
        case AttentionBackend.VANILLA:
            attention_out = simple_attention_with_dropout(
                QPos,
                KPos,
                Key,
                query,
                key,
                value,
                mask,
                bias,
                inference,
                dropout,
                attention_dtype,
                precision,
                prng=prng,
                scaling_factor=scaling_factor,
                logits_soft_cap=logits_soft_cap,
            )
        case _:
            attention_out = None

    if attention_out is not None:
        return attention_out
    else:
        # local import to avoid circular imports
        from levanter.models.flash_attention import flash_attention

        return flash_attention(
            QPos,
            KPos,
            Key,
            query,
            key,
            value,
            block_size=flash_block_size,
            mask=mask,
            bias=bias,
            dropout=dropout,
            inference=inference,
            key=prng,
            dtype=attention_dtype,
            precision=precision,
            scaling_factor=scaling_factor,
            logits_soft_cap=logits_soft_cap,
        )


def dot_product_attention_with_sink(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    attn_sink: NamedArray,
    mask: Optional[Union["AttentionMask", NamedArray]] = None,
    bias: Optional[NamedArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    use_flash: Optional[bool] = None,
    attn_backend: Optional[AttentionBackend] = None,
    flash_block_size: Optional[int] = None,
    dropout: float = 0.0,
    *,
    logits_soft_cap: float | None = None,
    scaling_factor: float | None = None,
    inference: bool = True,
    prng: PRNGKeyArray | None = None,
):
    """Dot-product attention variant with a learned sink term per head.

    The sink is implemented by appending a dummy key/value of zeros and
    inserting the sink logit via the bias term at the final key position.
    """

    QPos = query.resolve_axis(QPos)
    KPos = key.resolve_axis(KPos)
    Key = query.resolve_axis(Key)

    KPos1 = KPos.resize(1)
    KPosPlus = KPos.resize(KPos.size + 1)

    zero_key_axes = tuple(KPos1 if ax == KPos else ax for ax in key.axes)
    zero_key = hax.zeros(zero_key_axes, dtype=key.dtype)
    key = hax.concatenate(KPosPlus, [key, zero_key])

    zero_val_axes = tuple(KPos1 if ax == KPos else ax for ax in value.axes)
    zero_val = hax.zeros(zero_val_axes, dtype=value.dtype)
    value = hax.concatenate(KPosPlus, [value, zero_val])

    m = materialize_mask(mask, QPos, KPos)
    if m is not None:
        sink_mask_axes = tuple(KPos1 if ax == KPos else ax for ax in m.axes)
        sink_mask = hax.ones(sink_mask_axes, dtype=m.dtype)
        m = hax.concatenate(KPosPlus, [m, sink_mask])

    bias_axes_prefix = tuple(ax for ax in query.axes if ax != Key)
    sink_bias = attn_sink
    for ax in bias_axes_prefix:
        if ax not in sink_bias.axes:
            sink_bias = sink_bias.broadcast_axis(ax)
    sink_bias = sink_bias.broadcast_axis(KPos1)

    if bias is not None:
        bias = hax.concatenate(KPosPlus, [bias, sink_bias])
    else:
        zero_bias_axes = bias_axes_prefix + (KPos,)
        zero_bias = hax.zeros(zero_bias_axes, dtype=sink_bias.dtype)
        bias = hax.concatenate(KPosPlus, [zero_bias, sink_bias])

    return dot_product_attention(
        QPos,
        KPosPlus,
        Key,
        query,
        key,
        value,
        m,
        bias,
        attention_dtype,
        precision,
        use_flash,
        attn_backend,
        flash_block_size,
        dropout,
        logits_soft_cap=logits_soft_cap,
        scaling_factor=scaling_factor,
        inference=inference,
        prng=prng,
    )


def simple_attention_with_dropout(
    QPos: AxisSelector,
    KPos: AxisSelector,
    Key: Axis,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    inference: bool = False,
    dropout: float = 0.0,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    *,
    prng: Optional[PRNGKeyArray] = None,
    scaling_factor: float | None = None,
    logits_soft_cap: Optional[float] = None,
):
    QPos = query.resolve_axis(QPos)
    KPos = key.resolve_axis(KPos)
    m = materialize_mask(mask, QPos, KPos)
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

    if logits_soft_cap is not None:
        weights = hax.tanh(weights / logits_soft_cap) * logits_soft_cap

    if m is not None:
        weights = haliax.where(m, weights, -1e9)

    weights = haliax.nn.softmax(weights, axis=KPos)

    weights = weights.astype(orig_dtype)

    out = haliax.nn.dropout(weights, dropout, key=prng, inference=inference)

    return haliax.dot(out, value, axis=KPos)


def _try_te_attention(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    dropout: float = 0.0,
    inference: bool = False,
    *,
    prng: Optional[PRNGKeyArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    flash_block_size: Optional[int] = None,
    force_te: bool,
    scaling_factor: float,
    logits_soft_cap: Optional[float] = None,
):
    try:
        return _te_flash_attention(
            QPos,
            KPos,
            Key,
            query,
            key,
            value,
            mask=mask,
            bias=bias,
            dropout=dropout,
            inference=inference,
            prng=prng,
            attention_dtype=attention_dtype,
            precision=precision,
            block_size=flash_block_size,
            scaling_factor=scaling_factor,
            logits_soft_cap=logits_soft_cap,
        )
    except ImportError as e:
        if "transformer_engine" not in str(e):
            raise

        msg = "transformer_engine is not installed. Please install it to use NVIDIA's optimized fused attention."
        if force_te:
            raise ImportError(msg)

        warnings.warn(f"{msg}. Falling back to the reference implementation.")

        return None
    except NotImplementedError as e:
        message = f"Could not use transformer_engine for flash attention: {str(e)}."
        if force_te:
            raise NotImplementedError(message)

        warnings.warn(f"{message}. Falling back to the reference implementation.")

        return None
    except ValueError as e:
        message = str(e)
        if message.startswith("Unsupported backend="):
            _dtype = attention_dtype or query.dtype
            msg = "NVTE doesn't work with these arguments. Falling back to the reference implementation.\n"
            "Check nvte_get_fused_attn_backend for supported configurations:\n"
            "https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/fused_attn/fused_attn.cpp#L71"
            if _dtype not in (jnp.float16, jnp.bfloat16, jnp.float8_e5m2, jnp.float8_e4m3fn):
                msg += f"In particular, NVTE doesn't support {_dtype} yet."

            if force_te:
                raise NotImplementedError(msg)

            warnings.warn(msg)
        else:
            raise
        return None


def _te_flash_attention(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    dropout: float = 0.0,
    inference: bool = False,
    *,
    prng: Optional[PRNGKeyArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    block_size: Optional[int] = None,
    scaling_factor: float,
    logits_soft_cap: Optional[float] = None,
):
    from transformer_engine.jax.attention import (  # noqa: F401
        AttnBiasType,
        AttnMaskType,
        QKVLayout,
        fused_attn,  # noqa: F401
    )

    if logits_soft_cap is not None:
        raise NotImplementedError(
            "logits_soft_cap is not supported for NVTE fused attention. "
            "Please use the JAX reference implementation or ask NVIDIA..."
        )

    attention_dtype = attention_dtype or query.dtype
    query = query.astype(attention_dtype)
    key = key.astype(attention_dtype)
    value = value.astype(attention_dtype)

    if precision is not None:
        warnings.warn("precision is not supported for NVTE fused attention. Ignoring.")

    # references: https://github.com/NVIDIA/TransformerEngine/blob/8255f87f3ee8076db21777795ce15b6ddf8754c0/transformer_engine/jax/fused_attn.py#L31
    # https://github.com/NVIDIA/TransformerEngine/blob/8255f87f3ee8076db21777795ce15b6ddf8754c0/transformer_engine/jax/flax/transformer.py#L269

    q_class, k_class, v_class = _bin_and_group_axes_by_function(query, key, value, QPos, KPos, Key)
    q_: jax.Array = _reshape_axes_for_bshd_bins(query, q_class).array
    k_ = _reshape_axes_for_bshd_bins(key, k_class).array
    v_ = _reshape_axes_for_bshd_bins(value, v_class).array

    B, Sq, Hq, D = q_.shape
    Bk, Sk, Hk, Dk = k_.shape

    QPos = query.resolve_axis(QPos)
    KPos = key.resolve_axis(KPos)

    # TODO: must Dk == Dv?
    if k_.shape != v_.shape:
        raise ValueError("k and v must have the same axes")

    if B != Bk:
        raise ValueError(f"Batch axes must be the same for q, k, and v: {q_class['B']} != {k_class['B']}")

    if D != Dk:
        raise ValueError(f"Embedding axes must be the same for q, k, and v: {q_class['D']} != {k_class['D']}")

    # Mask is generated by transformer engine based on AttnMaskType
    attn_mask_type, fused_attn_mask = _te_materialize_mask(KPos, QPos, B, mask)

    is_training = not inference

    # TODO: bias type is probably also configurable
    attn_bias_type = AttnBiasType.NO_BIAS
    fused_attn_bias = None
    if bias:
        raise NotImplementedError("Using bias with flash attention on GPU is not currently implemented.")

    attn_output = fused_attn(
        qkv=(q_, k_, v_),
        bias=fused_attn_bias,
        mask=fused_attn_mask,
        seed=prng,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=QKVLayout.BSHD_BSHD_BSHD,
        scaling_factor=scaling_factor,
        dropout_probability=dropout,
        is_training=is_training,
    )

    # per the NVTE code, the output is BSHD. we can reshape it to match our axes
    # we have to ungroup the axes, then reshape them to match our expected output
    attn_output = haliax.named(attn_output, ("B", "S", "H", "D"))
    # the output shape is B, S_q, H_q, D_v. Right now we're requiring D_k == D_v
    # we can reshape it to match our expected output
    # the output shape is B, S_q, H_q, D_v. Right now we're requiring D_k == D_v
    # we can reshape it to match our expected output
    attn_output = _unflatten_bshd(attn_output, q_class, v_class)

    reference_out_shape = eqx.filter_eval_shape(
        simple_attention_with_dropout,
        QPos,
        KPos,
        Key,
        query,
        key,
        value,
        mask,
        bias,
        inference,
        dropout,
        attention_dtype,
        precision,
        prng=prng,
    )
    attn_output = attn_output.rearrange(reference_out_shape.axes).astype(reference_out_shape.dtype)

    return attn_output


def _te_materialize_mask(KPos, QPos, batch_size, mask):
    from transformer_engine.jax.attention import AttnMaskType

    if isinstance(mask, NamedArray):
        raise NotImplementedError(
            "Custom NamedArray masks are not implemented for flash attention. Please pass an AttentionMask object"
        )
    elif isinstance(mask, AttentionMask):
        if mask.is_causal:
            # NVTE fused attention does not support non-zero causal offsets.
            if mask.causal_offset is not None:
                raise NotImplementedError(
                    "Causal offset is not supported for NVTE fused attention. Please use the JAX reference"
                    " implementation."
                )
            attn_mask_type = AttnMaskType.CAUSAL_MASK

            fused_attn_mask = mask.materialize(QPos, KPos)

            assert (
                fused_attn_mask is not None
            ), "If AttentionMask is causal, the materialized array should never be None. Something is wrong."

            fused_attn_mask = fused_attn_mask.array
            fused_attn_mask = jnp.dstack([fused_attn_mask] * batch_size)

        else:
            raise NotImplementedError(
                "Non-causal AttentionMask is not supported for NVTE fused attention."
                " Please use the JAX reference implementation."
            )
    else:
        attn_mask_type = AttnMaskType.NO_MASK
        fused_attn_mask = jnp.ones((batch_size, QPos.size, KPos.size))
    return attn_mask_type, fused_attn_mask


_DUMMY_HEAD = "__head__"
_DUMMY_BATCH = "__batch__"


def _bin_and_group_axes_by_function(q, k, v, QPos, KPos, Key):
    """
    NVTE and the Splash Attention kernel require the Q, K, and V to be in a specific format. This function groups the axes
    of Q, K, and V into the right bins to match that format.

    NVTE requires Q, K, and V to have shape BSHD (Batch, Sequence, Head, Embed), while Splash Attention requires BHSD.

    The size of the axes is a bit flexible, with the following conditions:
    - B must be the same for all (TODO: is this true?)
    - S must be the same for K and V. Q's S can be different
    - H: Q's H must be a multiple of K's H (for GQA or MQA)
    - D must be the same for all (TODO: is this true? possibly V can be different)

    We can thus classify the axes in q, k, v by their function and populate the NVTE axes in the right order
    - Key is D. ATM we're assuming this is a single axis.
    - QPos and KPos are always S
    - the latest other axis that is present in all three is H. If there are no other axes, we'll add a dummy axis
    - Any other axis that is present in all three is B. If there are no other axes, we'll add a dummy axis
    - If there's an axis present in Q and not in K or V, it's an extra H for Q (as part of GQA).
      These go *after* the primary H because GQA wants these to be minor axes
    - If there are any other axes present in one but not all three, it's an error
     (TODO: we could vmap over these?)
    """
    QPos = q.resolve_axis(QPos)
    KPos = k.resolve_axis(KPos)
    Key = q.resolve_axis(Key)

    q_class = {"B": [], "S": [QPos], "H": [], "D": [Key]}
    k_class = {"B": [], "S": [KPos], "H": [], "D": [Key]}
    v_class = {"B": [], "S": [KPos], "H": [], "D": [Key]}

    present_in_all: set[str] = q.shape.keys() & k.shape.keys() & v.shape.keys()
    spoken_for: set[str] = {QPos.name, KPos.name, Key.name}

    # find the primary H axes: which are axes that are:
    # - present in all three
    # - not spoken for already
    # - come after QPos in Q (if there's already a primary H)
    # - not the 0th axis in Q (even if there's no primary H)
    primary_H: list[Axis] = []
    for a in reversed(q.axes[1:]):
        if a.name in present_in_all and a.name not in spoken_for:
            primary_H.append(a)
        elif a == QPos and primary_H:  # better to always have at least one H?
            break  # anything before QPos we'll say is Batch

    # since we added them in reverse order, we need to reverse them
    primary_H.reverse()

    spoken_for.update([ax.name for ax in primary_H])

    # remaining shared axes are batch axes
    batch_axes = [ax for ax in q.axes if ax.name not in spoken_for and ax.name in present_in_all]

    spoken_for.update([ax.name for ax in batch_axes])

    q_class["B"] = batch_axes
    k_class["B"] = batch_axes
    v_class["B"] = batch_axes

    # if there's an axis in q that's not in k or v, it's an extra H for q
    extra_q_H = [ax for ax in q.axes if ax.name not in spoken_for]

    # we want primary_h to be *before* extra_q_H b/c GQA wants these to be minor axes
    q_class["H"] = primary_H + extra_q_H
    k_class["H"] = primary_H
    v_class["H"] = primary_H

    # now we want to detect any non-spoken-for axes. These are errors
    # eventually we can vmapp over these, but for now we'll just raise an error
    for a in k.axes:
        if a.name not in spoken_for:
            raise ValueError(f"Axis {a.name} is present in k but not in q and/or v")

    for a in v.axes:
        if a.name not in spoken_for:
            raise ValueError(f"Axis {a.name} is present in v but not in q and/or k")

    return q_class, k_class, v_class


def _reshape_axes_for_bshd_bins(q, q_class, output_order=("B", "S", "H", "D")):
    """
    Reshape the axes of a qkv as BSHD to match the bins in q_class
    """

    def _maybe_flatten(q, axes, name):
        if axes:
            q = q.flatten_axes(axes, name)
        else:
            q = q.broadcast_axis(Axis(name, 1))
        return q

    q = _maybe_flatten(q, q_class["B"], "B")
    q = _maybe_flatten(q, q_class["S"], "S")
    q = _maybe_flatten(q, q_class["H"], "H")
    q = _maybe_flatten(q, q_class["D"], "D")
    q = q.rearrange(output_order)
    return q


def _unflatten_bshd(attn_output, q_class, v_class):
    attn_output = attn_output.unflatten_axis("B", q_class["B"])
    attn_output = attn_output.unflatten_axis("S", q_class["S"])
    attn_output = attn_output.unflatten_axis("H", q_class["H"])
    attn_output = attn_output.unflatten_axis("D", v_class["D"])
    return attn_output


def _materialize_segment_mask(
    segment_ids: NamedArray | tuple[NamedArray, NamedArray], QPos, KPos, q_slice, k_slice
) -> NamedArray:
    """
    Make a segment mask for attention. This is a mask that prevents attention between different segments.
    """
    if isinstance(segment_ids, tuple):
        if len(segment_ids) != 2:
            raise ValueError("segment_ids must be a tuple of two NamedArrays")
        q_segment_ids, kv_segment_ids = segment_ids
        kv_segment_ids = kv_segment_ids.rename({QPos.name: KPos.name})[KPos.name, k_slice]
        q_segment_ids = q_segment_ids.rename({QPos.name: QPos})[QPos.name, q_slice]
    else:
        kv_segment_ids = segment_ids.rename({QPos.name: KPos.name})[KPos.name, k_slice]
        q_segment_ids = segment_ids[QPos.name, q_slice]

    return q_segment_ids.broadcast_axis(kv_segment_ids.axes) == kv_segment_ids


def _materialize_sliding_window_mask(
    window: int, QPos: Axis, KPos: Axis, q_slice: haliax.dslice, k_slice: haliax.dslice
) -> NamedArray:
    """Materialize a causal sliding window mask."""
    sub_q = QPos.resize(q_slice.size)
    sub_k = KPos.resize(k_slice.size)
    q_pos = hax.arange(sub_q) + q_slice.start
    k_pos = hax.arange(sub_k) + k_slice.start
    diff = q_pos.broadcast_axis(sub_k) - k_pos.broadcast_axis(sub_q)
    return (diff >= 0) & (diff < window)


class AttentionMask(eqx.Module):
    """

    !!! warning
        This class is still experimental. I'm not super happy with it yet.

    Represents an attention mask in a structured way to make it easier to optimize attention for particular use cases
    (causal, prefix, etc.). It is anticipated that this will be extended with new types of masks as needed.

    The abstraction is based on two concepts:

    1) Materialization: An AttentionMask can be materialized for a particular slice of the query and key position axes.
       Most naively, you can just get the whole mask as a NamedArray. However, in some cases, you might want to
       only get a particular chunk (e.g. for flash attention).
    2) Combination: AttentionMasks are represented as an implicit conjunction of multiple masks, each with different
        kinds of structure. You can combine masks with `&` and `|`. Due to the way jit works, we don't use inheritance
        or similar to represent different kinds of masks. Instead, we use a single class with different fields.

    In general, it should be safe to batch Attention Masks, but it is important that *all members of a batch have the
    same set of combined masks*. Otherwise, the batching will not work and you'll get weird errors

    (Perhaps it's ok to use inheritance here? I'm not sure. Splash attention landed on inheritance, so maybe
    that's a good sign.)

    """

    # If ``is_causal`` is True we apply a lower-triangular causal mask. If ``causal_offset`` is not ``None``
    # we apply a shifted causal mask such that a query at position *i* can attend to key *j* whenever
    # ``j <= i + causal_offset``. A ``None`` offset means a static offset of 0 (i.e., standard causal masking).
    is_causal: bool = eqx.field(default=False, static=True)
    causal_offset: None | NamedArray = None
    explicit_mask: Optional[NamedArray] = None
    segment_ids: tuple[NamedArray, NamedArray] | None = None
    sliding_window: Optional[int] = eqx.field(default=None, static=True)
    # CF https://github.com/jax-ml/jax/blob/47858c4ac2fd4757a3b6fc5bb2981b71a71f00c2/jax/experimental/pallas/ops/tpu/flash_attention.py#L34
    # TODO: add prefixlm
    # cf https://github.com/google-research/t5x/blob/51a99bff8696c373cc03918707ada1e98cbca407/t5x/examples/decoder_only/layers.py#L978

    def materialize(
        self, QPos: Axis, KPos: Axis, q_slice: Optional[haliax.dslice] = None, k_slice: Optional[haliax.dslice] = None
    ) -> Optional[NamedArray]:
        """
        Materialize the mask as a NamedArray. This is useful for attention functions that don't support masks,
        or for the inner loop
        """
        if q_slice is None:
            q_slice = haliax.dslice(0, QPos.size)
        if k_slice is None:
            k_slice = haliax.dslice(0, KPos.size)

        if self.is_causal:
            # None means static 0 offset
            offset = 0 if self.causal_offset is None else self.causal_offset
            shifted_k_start = k_slice.start - offset
            if isinstance(shifted_k_start, NamedArray):
                # need to vmap
                causal = hax.vmap(causal_mask, shifted_k_start.axes)(
                    QPos.resize(q_slice.size),
                    KPos.resize(k_slice.size),
                    q_slice.start,
                    shifted_k_start,  # type: ignore
                )
            else:
                causal = causal_mask(
                    QPos.resize(q_slice.size),
                    KPos.resize(k_slice.size),
                    q_slice.start,
                    shifted_k_start,
                )
        else:
            causal = None

        if self.explicit_mask is not None:
            explicit = self.explicit_mask[QPos, q_slice, KPos, k_slice]
        else:
            explicit = None

        mask = combine_masks_and(causal, explicit)

        if self.sliding_window is not None:
            sw_mask = _materialize_sliding_window_mask(
                self.sliding_window, QPos, KPos, q_slice=q_slice, k_slice=k_slice
            )
            mask = combine_masks_and(mask, sw_mask)

        if self.segment_ids is not None:
            segment_mask = _materialize_segment_mask(self.segment_ids, QPos, KPos, q_slice, k_slice)
            mask = combine_masks_and(mask, segment_mask)

        return mask

    # Static constructors --------------------------------------------------

    @staticmethod
    def causal(*, sliding_window: Optional[int] = None, offset: int | NamedArray | None = None) -> "AttentionMask":
        """Create a causal AttentionMask.

        Args:
            sliding_window: If provided, restrict each query position to attend only to keys within
                ``sliding_window`` previous positions.
            For ``offset == 0`` this is identical to the old ``AttentionMask.causal()``
            behaviour; larger offsets loosen the restriction so that each query can
            see ``offset`` additional future tokens.
        """
        if isinstance(offset, int | Integral):
            causal_offset = hax.named(offset, ())
        else:
            causal_offset = offset

        return AttentionMask(is_causal=True, causal_offset=causal_offset, sliding_window=sliding_window)

    @staticmethod
    def explicit(mask: NamedArray) -> "AttentionMask":
        return AttentionMask(is_causal=False, causal_offset=None, explicit_mask=mask)

    def __post_init__(self):
        # Normalize legacy single-array segment_ids to a tuple for consistency
        if self.segment_ids is not None and not isinstance(self.segment_ids, tuple):
            warnings.warn("Storing segment_ids as a single NamedArray is deprecated. Use a tuple instead.")
            object.__setattr__(self, "segment_ids", (self.segment_ids, self.segment_ids))

    def with_segment_ids(self, segment_ids: NamedArray, kv_segment_ids: NamedArray | None = None) -> "AttentionMask":
        """Attach segment ids to the mask.

        Always stores segment ids internally as a tuple ``(q_segment_ids, kv_segment_ids)``.
        If only a single array is provided, it is used for both queries and keys/values.
        """
        # Always store as a tuple; duplicate if only one provided.
        seg_field: tuple[NamedArray, NamedArray]
        if kv_segment_ids is None:
            seg_field = (segment_ids, segment_ids)
        else:
            seg_field = (segment_ids, kv_segment_ids)

        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            explicit_mask=self.explicit_mask,
            segment_ids=seg_field,
            sliding_window=self.sliding_window,
        )

    def with_sliding_window(self, sliding_window: int | None) -> "AttentionMask":
        """Return a copy of this mask with ``sliding_window`` applied."""
        return AttentionMask(
            is_causal=self.is_causal,
            causal_offset=self.causal_offset,
            explicit_mask=self.explicit_mask,
            segment_ids=self.segment_ids,
            sliding_window=sliding_window,
        )

    def __and__(self, other) -> "AttentionMask":
        # Conjunction: causal if either component is causal.
        if self.is_causal and other.is_causal:
            # If both are causal, offsets must agree if both specified; otherwise take the specified one.
            if self.causal_offset is not None and other.causal_offset is not None:
                causal_offset = eqx.error_if(
                    self.causal_offset,
                    self.causal_offset != other.causal_offset,
                    "Mismatched causal offsets cannot be combined with &",
                )
            else:
                causal_offset = self.causal_offset if self.causal_offset is not None else other.causal_offset
            is_causal = True
        elif self.is_causal:
            causal_offset = self.causal_offset
            is_causal = True
        elif other.is_causal:
            causal_offset = other.causal_offset
            is_causal = True
        else:
            causal_offset = None
            is_causal = False
        explicit_mask = combine_masks_and(self.explicit_mask, other.explicit_mask)
        segment_ids = self._check_for_same_segment_ids(other)
        if self.sliding_window is None:
            sliding_window = other.sliding_window
        elif other.sliding_window is None:
            sliding_window = self.sliding_window
        else:
            sliding_window = min(self.sliding_window, other.sliding_window)

        return AttentionMask(
            is_causal=is_causal,
            causal_offset=causal_offset,
            explicit_mask=explicit_mask,
            segment_ids=segment_ids,
            sliding_window=sliding_window,
        )

    def __or__(self, other) -> "AttentionMask":
        # Union: causal only if both are causal with the same offset; otherwise non-causal
        if (
            self.is_causal
            and other.is_causal
            and (
                (self.causal_offset is None and other.causal_offset is None)
                or (self.causal_offset is not None and self.causal_offset == other.causal_offset)
            )
        ):
            is_causal = True
            causal_offset = self.causal_offset
        else:
            is_causal = False
            causal_offset = None
        explicit_mask = combine_masks_or(self.explicit_mask, other.explicit_mask)
        segment_ids = self._check_for_same_segment_ids(other)
        if self.sliding_window is None or other.sliding_window is None:
            sliding_window = None
        else:
            sliding_window = max(self.sliding_window, other.sliding_window)
        return AttentionMask(
            is_causal=is_causal,
            causal_offset=causal_offset,
            explicit_mask=explicit_mask,
            segment_ids=segment_ids,
            sliding_window=sliding_window,
        )

    def _check_for_same_segment_ids(self, other):
        # Normalize possibly non-tuple representations to tuples for comparison.
        def _as_tuple(si):
            if si is None:
                return None
            if isinstance(si, tuple):
                return si
            else:
                return (si, si)

        self_si = _as_tuple(self.segment_ids)
        other_si = _as_tuple(other.segment_ids)

        if self_si is not None and other_si is not None:
            # only one segment mask is allowed
            # b/c we might do this in jit, we use eqx.error_if
            # in theory we can do this one by just assigning unique ids to each unique pair...
            # (but i don't really anticipate needing this)
            segment_ids = eqx.error_if(
                hax.logical_or(self_si[0] != other_si[0], self_si[1] != other_si[1]),
                "Only one segment mask is allowed",
            )
        elif self_si is not None:
            segment_ids = self_si
        else:
            segment_ids = other_si
        return segment_ids


@overload
def materialize_mask(
    mask: NamedArray | AttentionMask,
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> NamedArray: ...


@overload
def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]: ...


def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]:
    """
    Materialize an attention mask if it is an AttentionMask. Otherwise, just return it.
    """
    if isinstance(mask, AttentionMask):
        mask = mask.materialize(QPos, KPos, q_slice=q_slice, k_slice=k_slice)
        return mask
    elif isinstance(mask, NamedArray):
        if q_slice is not None or k_slice is not None:
            if q_slice is None:
                q_slice = haliax.dslice(0, QPos.size)
            if k_slice is None:
                k_slice = haliax.dslice(0, KPos.size)
            mask = mask[QPos, q_slice, KPos, k_slice]

        return mask
    else:
        assert mask is None
        return None


# TODO: padding mask
# TODO: FCM mask?


def _try_tpu_splash_attention(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    dropout: float = 0.0,
    inference: bool = False,
    *,
    force_flash: bool,
    prng: Optional[PRNGKeyArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    block_size: Optional[int] = None,
    scaling_factor: float,
    logits_soft_cap: float | None,
) -> Optional[NamedArray]:
    if dropout != 0.0:
        if force_flash:
            raise NotImplementedError("Splash attention does not support dropout.")
        warnings.warn("Splash attention does not support. Falling back to the reference implementation.")
        return None

    if bias is not None:
        if force_flash:
            raise NotImplementedError("Splash attention does not support bias.")
        warnings.warn("Splash attention does not support bias. Falling back to the reference implementation.")
        return None

    try:
        return _tpu_splash_attention(
            QPos,
            KPos,
            Key,
            query,
            key,
            value,
            mask,
            bias,
            dropout,
            inference,
            prng=prng,
            attention_dtype=attention_dtype,
            precision=precision,
            block_size=block_size,
            scaling_factor=scaling_factor,
            logits_soft_cap=logits_soft_cap,
        )
    except ImportError as e:
        if "pallas" not in str(e):
            raise
        if force_flash:
            raise ImportError("Could not import splash attention. You need to update your JAX to at least 0.4.26.")
        warnings.warn(
            "Could not import splash attention. You need to update your JAX to at least 0.4.26. "
            "Falling back to the reference implementation."
        )
        return None
    except NotImplementedError as e:
        message = str(e)
        if force_flash:
            raise NotImplementedError(f"Could not use splash attention: {message}")
        warnings.warn(f"Could not use splash attention: {message}. Falling back to the reference")
        return None


# CF https://github.com/google/maxtext/blob/db31dd4b0b686bca4cd7cf940917ec372faa183a/MaxText/layers/attentions.py#L179
def _tpu_splash_attention(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    dropout: float = 0.0,
    inference: bool = False,
    *,
    prng: Optional[PRNGKeyArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    block_size: Optional[int] = None,
    scaling_factor: float,
    logits_soft_cap: float | None = None,
) -> Optional[NamedArray]:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )

    # Splash attention requires BHSD format
    # We need to reshape the input to match this format
    if dropout != 0.0:
        raise NotImplementedError("Splash attention does not support dropout")

    if bias is not None:
        raise NotImplementedError("Splash attention does not support bias")

    # if attention_dtype is not None and attention_dtype != jnp.float32:
    #     warnings.warn("Splash attention only supports float32. Switching to float32.")

    # attention_dtype = jnp.float32

    q_class, k_class, v_class = _bin_and_group_axes_by_function(query, key, value, QPos, KPos, Key)

    query = query * scaling_factor

    q_: jax.Array = _reshape_axes_for_bshd_bins(query, q_class, output_order=list("BHSD")).array
    k_ = _reshape_axes_for_bshd_bins(key, k_class, output_order=list("BHSD")).array
    v_ = _reshape_axes_for_bshd_bins(value, v_class, output_order=list("BHSD")).array

    B, Hq, Sq, D = q_.shape
    Bk, Hk, Sk, Dk = k_.shape

    # number
    if Sk % 128 != 0:
        raise NotImplementedError("Splash attention requires KPos to be a multiple of 128")

    if block_size is not None and block_size % 128 != 0:
        raise NotImplementedError(f"Splash attention requires block_size to be a multiple of 128, got {block_size}")

    QPos = query.resolve_axis(QPos)
    KPos = key.resolve_axis(KPos)

    # TODO: must Dk == Dv?
    if k_.shape != v_.shape:
        raise ValueError("k and v must have the same axes")

    # TODO: this isn't really necessary on TPU?
    if B != Bk:
        raise ValueError(f"Batch axes must be the same for q, k, and v: {q_class['B']} != {k_class['B']}")

    if D != Dk:
        raise ValueError(f"Embedding axes must be the same for q, k, and v: {q_class['D']} != {k_class['D']}")

    def _physical_axis_for_binning(d):
        def flatten(axes):
            if axes is None:
                return axes
            result = []
            for ax in axes:
                if isinstance(ax, tuple):
                    result += list(ax)
                else:
                    result.append(ax)
            return tuple(result)

        b_out = flatten(tuple(ax for ax in pspec_for_axis(d["B"]) if ax is not None) or None)
        h_out = flatten(tuple(ax for ax in pspec_for_axis(d["H"]) if ax is not None) or None)
        s_out = flatten(tuple(ax for ax in pspec_for_axis(d["S"]) if ax is not None) or None)
        d_out = flatten(tuple(ax for ax in pspec_for_axis(d["D"]) if ax is not None) or None)

        return PartitionSpec(b_out, h_out, s_out, d_out)

    # BHSD
    physical_axes_q = _physical_axis_for_binning(q_class)
    physical_axes_k = _physical_axis_for_binning(k_class)
    physical_axes_v = _physical_axis_for_binning(v_class)

    # segment_ids: handle both the new tuple form and legacy single-array form for robustness
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    if segment_ids is not None:
        if isinstance(segment_ids, tuple):
            _seg_axes = segment_ids[0].axes
        else:
            _seg_axes = segment_ids.axes
        physical_axes_segments = pspec_for_axis(_seg_axes)
    else:
        physical_axes_segments = None
    # do we have a batch axis in segment_ids? (needed for vmap below)
    if segment_ids is not None:
        if isinstance(segment_ids, tuple):
            q_segment_ids, kv_segment_ids = segment_ids
            kv_segment_ids = kv_segment_ids
        else:
            assert segment_ids is not None
            q_segment_ids, kv_segment_ids = segment_ids, segment_ids

        segment_ids = SegmentIds(q_segment_ids.array, kv_segment_ids.array)

        q_segment_batch_axis = _find_batch_axis_for_segment_ids(QPos, q_segment_ids)
        kv_segment_batch_axis = _find_batch_axis_for_segment_ids(QPos, kv_segment_ids)

        if q_segment_batch_axis is not None or kv_segment_batch_axis is not None:
            segment_batch_axis = SegmentIds(q_segment_batch_axis, kv_segment_batch_axis)  # type: ignore
        else:
            segment_batch_axis = None
    else:
        segment_batch_axis = None

    # MaxText uses a block size of 512
    block_size = block_size or 512

    # copied from MaxText
    @functools.partial(
        shard_map,
        mesh=haliax.partitioning._get_mesh(),
        in_specs=(
            physical_axes_q,
            physical_axes_k,
            physical_axes_v,
            physical_axes_segments,
        ),
        out_specs=physical_axes_q,
        check_rep=False,
    )
    def wrap_flash_attention(q, k, v, segment_ids):
        # NB: inside the function, q, k, and v are partitioned, so in general the lengths of dims are not the same
        Sq = q.shape[2]
        Sk = k.shape[2]
        Hq = q.shape[1]
        block_sizes = splash_attention_kernel.BlockSizes(
            block_q=min(block_size, Sq),
            block_kv_compute=min(block_size, Sk),
            block_kv=min(block_size, Sk),
            block_q_dkv=min(block_size, Sq),
            block_kv_dkv=min(block_size, Sk),
            block_kv_dkv_compute=min(block_size, Sq),
            block_q_dq=min(block_size, Sq),
            block_kv_dq=min(block_size, Sq),
        )

        if mask is None:
            base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))
        elif isinstance(mask, AttentionMask):
            if mask.is_causal:
                if mask.causal_offset is not None:
                    raise NotImplementedError(
                        "Causal offsets are not supported for splash attention. Please use a standard causal mask."
                    )
                base_mask = splash_attention_mask.CausalMask((Sq, Sk), 0)
            else:
                base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))
            if mask.sliding_window is not None:
                local_mask = splash_attention_mask.LocalMask(
                    shape=(Sq, Sk),
                    window_size=(mask.sliding_window - 1, None),
                    offset=0,
                )
                base_mask = splash_attention_mask.LogicalAnd(base_mask, local_mask)
            # This is going to be a pain to support
            if mask.explicit_mask is not None:
                raise NotImplementedError("Explicit masks are not yet supported for splash attention")

        elif isinstance(mask, NamedArray):
            raise NotImplementedError("NamedArray masks are not yet supported for splash attention")
        else:
            raise ValueError(f"Unknown mask type: {mask}")

        kernel_mask = splash_attention_mask.MultiHeadMask(masks=[base_mask for _ in range(Hq)])

        # copied from MaxText
        splash_kernel = splash_attention_kernel.make_splash_mha(
            mask=kernel_mask,
            head_shards=1,
            q_seq_shards=1,
            block_sizes=block_sizes,
            attn_logits_soft_cap=logits_soft_cap,
        )

        q = q.astype(attention_dtype)
        k = k.astype(attention_dtype)
        v = v.astype(attention_dtype)
        return jax.vmap(
            lambda q, k, v, si: splash_kernel(q, k, v, segment_ids=si), in_axes=(0, 0, 0, segment_batch_axis)
        )(q, k, v, segment_ids)

    attn_output = wrap_flash_attention(q_, k_, v_, segment_ids)

    attn_output = haliax.named(attn_output, ("B", "H", "S", "D"))
    # the output shape is B, S_q, H_q, D_v. Right now we're requiring D_k == D_v
    # we can reshape it to match our expected output
    attn_output = _unflatten_bshd(attn_output, q_class, v_class)
    with haliax.axis_mapping({}):
        reference_out_shape = eqx.filter_eval_shape(
            simple_attention_with_dropout,
            QPos,
            KPos,
            Key,
            query,
            key,
            value,
            mask,
            bias,
            inference,
            dropout,
            attention_dtype,
            precision,
            prng=prng,
        )
    attn_output = attn_output.rearrange(reference_out_shape.axes).astype(reference_out_shape.dtype)

    attn_output = haliax.shard(attn_output)

    return attn_output


def _find_batch_axis_for_segment_ids(Pos, segment_ids) -> Optional[int]:
    index_of_seq_dim = segment_ids.axes.index(Pos)
    other_indices = [i for i in range(len(segment_ids.axes)) if i != index_of_seq_dim]
    if len(other_indices) > 1:
        raise NotImplementedError(
            f"Only one batch axis is supported in segment_ids right now (got {segment_ids.axes})"
        )
    elif len(other_indices) == 1:
        segment_batch_axis = other_indices[0]
    else:
        segment_batch_axis = None

    return segment_batch_axis


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for the Attention module.

    Args:
        Embed: The embedding dimension axis
        num_heads: Number of attention heads
        num_kv_heads: Number of key/value heads (for grouped-query attention)
        use_bias: Whether to use bias in the attention projections
        upcast_attn: Whether to upcast attention to float32 for better numerical stability
        attn_backend: Which attention backend to use
        flash_attention_block_size: Block size for flash attention
        rope: Configuration for rotary position embeddings
        scaling_factor: Optional scaling factor for attention scores. If None, defaults to 1/sqrt(head_size)
        qk_norm: Optional configuration for QK normalization. If None, no normalization is applied.
    """

    Embed: Axis

    num_heads: int
    num_kv_heads: int
    head_dim: int | None = None
    use_bias: bool = False
    use_output_bias: Optional[bool] = None  # If None, uses use_bias
    upcast_attn: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    rope: Optional[RotaryEmbeddingsConfig] = None
    scaling_factor: Optional[float] = None
    logits_soft_cap: Optional[float] = None
    qk_norm: Optional[LayerNormConfigBase] = None
    """Configuration for QK normalization. If None, no normalization is applied."""

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    @property
    def head_size(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.Embed.size // self.num_heads

    @property
    def q_heads_per_group(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def KVHeads(self) -> Axis:
        return Axis("kv_head", self.num_kv_heads)

    @property
    def Heads(self) -> Axis:
        return Axis("heads", self.num_heads)

    @property
    def HeadSize(self) -> Axis:
        return Axis("head_size", self.head_size)

    @property
    def QHeadsPerGroup(self) -> Axis:
        """Axis for query heads per group."""
        return Axis("q_heads_per_group", self.q_heads_per_group)

    @property
    def use_flash_attention(self) -> bool:
        """Whether to use flash attention based on the backend."""
        if self.attn_backend is None:
            return default_attention_type() != AttentionBackend.VANILLA
        return self.attn_backend != AttentionBackend.VANILLA

    # ---------------------------------------------------------------------------------
    # KV-cache helper (paged only)
    # ---------------------------------------------------------------------------------


class Attention(eqx.Module):
    """A multi-head attention layer that uses dot product attention.

    This is a general-purpose attention layer that can be used in various transformer architectures.
    It supports multi-head attention (MHA), multi-query attention (MQA), and grouped-query attention (GQA).

    Supports ROPE and QK normalization. We should probably not add much more stuff.
    """

    config: AttentionConfig = eqx.field(static=True)
    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    o_proj: hnn.Linear
    q_norm: Optional[LayerNormBase] = None
    k_norm: Optional[LayerNormBase] = None
    rot_embs: Optional[RotaryEmbeddings] = None

    @staticmethod
    def init(config: AttentionConfig, *, key) -> "Attention":
        use_bias = config.use_bias
        use_output_bias = config.use_output_bias if config.use_output_bias is not None else use_bias
        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        q_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.QHeadsPerGroup, config.HeadSize),
            key=k_q,
            use_bias=use_bias,
            out_first=True,
        )
        k_proj = hnn.Linear.init(
            In=config.Embed, Out=(config.KVHeads, config.HeadSize), key=k_k, use_bias=use_bias, out_first=True
        )
        v_proj = hnn.Linear.init(
            In=(config.Embed), Out=(config.KVHeads, config.HeadSize), key=k_v, use_bias=use_bias, out_first=True
        )
        o_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize), Out=config.Embed, key=k_o, use_bias=use_output_bias, out_first=True
        )

        q_norm = None
        k_norm = None
        if config.qk_norm is not None:
            q_norm = config.qk_norm.build(config.HeadSize)
            k_norm = config.qk_norm.build(config.HeadSize)

        # Build rotary embeddings once during initialization if configured
        rot_embs = config.rope.build(config.HeadSize) if config.rope is not None else None

        return Attention(config, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rot_embs)

    def empty_page_cache(self, page_table: PageTable, *, dtype) -> "KvPageCache":
        return KvPageCache.init(
            page_table,
            self.config.KVHeads,
            self.config.HeadSize,
            dtype=dtype,
        )

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        key_proj, key_o = maybe_rng_split(key, 2)

        # Shared computation of q, k, v
        q, k, v = self._compute_qkv(x, key=key_proj, pos_ids=pos_ids)

        # Reshape for attention kernels (convert embed → heads/head_size)
        q = q.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_head", "position", "head_size"))
        v = v.rearrange((..., "kv_head", "position", "head_size"))

        # Distinguish key sequence axis for attention
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # Apply attention
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            scaling_factor=self.config.scaling_factor,
            logits_soft_cap=self.config.logits_soft_cap,
            inference=True,
            prng=key,
        )

        # Flatten heads and apply output projection
        attn_output = attn_output.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        attn_output = self.o_proj(attn_output, key=key_o)

        return attn_output

    # Note: the non-paged decode path has been removed. Use paged_decode.

    @named_call
    @jax.profiler.annotate_function
    def paged_decode(
        self,
        x: NamedArray,
        kv_cache: "KvPageCache",
        batch_info: PageBatchInfo,
        *,
        pos_ids: NamedArray,
        key=None,
    ) -> tuple[NamedArray, "KvPageCache"]:
        """Decode-time forward pass using a paged KV cache.

        This method is intended for autoregressive decoding and prefill.  ``batch_info``
        describes where the new keys and values should be written in ``kv_cache``.
        Currently only causal masks are supported.
        """

        key_proj, key_o = maybe_rng_split(key, 2)

        q, k, v = self._compute_qkv(x, key=key_proj, pos_ids=pos_ids)

        kv_cache = kv_cache.update(batch_info, k, v)

        sm_scale = (
            self.config.scaling_factor
            if self.config.scaling_factor is not None
            else 1.0 / math.sqrt(self.config.HeadSize.size)
        )

        attn_tokens = ragged_paged_attention(
            q,
            kv_cache.kv_pages,
            batch_info.seq_lens,
            batch_info.page_indices,
            batch_info.cu_q_lens,
            batch_info.num_seqs,
            sm_scale=sm_scale,
            soft_cap=self.config.logits_soft_cap,
        )

        attn_output = attn_tokens.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        attn_output = self.o_proj(attn_output, key=key_o)

        return attn_output, kv_cache

    def _compute_qkv(
        self,
        x: NamedArray,
        *,
        key,
        pos_ids: NamedArray | None = None,
    ) -> tuple[NamedArray, NamedArray, NamedArray]:
        """Project *x* to Q, K and V and apply all per-head processing."""

        # Split the projection key into three – one for each of Q, K, V
        key_q, key_k, key_v = maybe_rng_split(key, 3)

        # Linear projections
        q = self.q_proj(x, key=key_q)
        k = self.k_proj(x, key=key_k)
        v = self.v_proj(x, key=key_v)

        # Optional QK layer-norm
        if self.config.qk_norm is not None:
            q = self.q_norm(q)  # type: ignore[misc]
            k = self.k_norm(k)  # type: ignore[misc]

        # Apply rotary embeddings if configured
        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"))
            q = self.rot_embs(q, pos_ids)
            k = self.rot_embs(k, pos_ids)

        return q, k, v


class KvPageCache(eqx.Module):
    """
    KvPageCache for paged attention. It contains keys and values for all pages, including
    potentially sequences that are not currently active.

    Contains a global view of all pages and their sequences. This can't be usefully used
    with an accompanying PageTable.
    """

    kv_pages: NamedArray  # [Page, Slot, 2 * KVHeads, Embed]

    @staticmethod
    def init(page_table: PageTable, kv_heads: Axis, head_size: Axis, dtype=jnp.float32) -> "KvPageCache":
        """
        Initialize a KvPageCache with the given page table and dimensions.

        Args:
            page_table: The PageTable instance that defines the pages.
            kv_heads: Axis for key/value heads.
            head_size: Axis for head size.
            dtype: Data type for the cache.
        """
        kv_pages = hax.zeros(
            {
                "page": page_table.num_pages,
                "slot": page_table.page_size,
                "kv_head": 2 * kv_heads.size,
                head_size.name: head_size.size,
            },
            dtype=dtype,
        )
        return KvPageCache(kv_pages)

    def update(
        self,
        batch_info: PageBatchInfo,
        new_k: NamedArray,  # [Tok, KvHeads, HeadDim]
        new_v: NamedArray,  # [Tok, KvHeads, HeadDim]
    ) -> "KvPageCache":
        """Append keys and values to the paged cache using *batch_info* to locate pages."""

        page_size = self.kv_pages.array.shape[1]

        assert page_size == batch_info.page_size, (
            f"Page size mismatch: {page_size} != {batch_info.page_size}. "
            "Ensure that the page size in batch_info matches the kv_pages."
        )

        t_pages, t_slots = batch_info.pages_and_slots()

        # jax.debug.print("Updating kv_pages at pages {t_pages} and slots {t_slots}",
        #                 t_pages=t_pages, t_slots=t_slots)

        new_k = new_k.astype(self.kv_pages.dtype)
        new_v = new_v.astype(self.kv_pages.dtype)
        kv_pages = eqx.error_if(self.kv_pages, hax.any(hax.isnan(self.kv_pages)).scalar(), "NaN in kv_pages pre")
        kv_pages = kv_pages.at["page", t_pages, "slot", t_slots, "kv_head", 0::2].set(new_k, mode="drop")
        kv_pages = kv_pages.at["page", t_pages, "slot", t_slots, "kv_head", 1::2].set(new_v, mode="drop")

        kv_pages = eqx.error_if(kv_pages, hax.any(hax.isnan(kv_pages)).scalar(), "NaN in kv_pages")

        return dataclasses.replace(self, kv_pages=kv_pages)

    def copy_page(self, src_page: int, dst_page: int) -> "KvPageCache":
        """Copy the entire contents of page ``src_page`` into ``dst_page``.

        This is used when creating clones that should have an identical last partial page, but mapped to a fresh page.
        """
        new_k = self.kv_pages.at["page", dst_page].set(self.kv_pages["page", src_page])
        return dataclasses.replace(self, kv_pages=new_k)


def ragged_paged_attention(
    q: NamedArray,  # [Tok, KVHeads, QHeadsPerGroup, HeadSize]
    kv_pages: NamedArray,  # [Page, PageSize, 2 * KVHeads, HeadDim]
    kv_lens: NamedArray,  # i32[Seq]
    page_indices: NamedArray,  # i32[Seq, PagePerSeq]
    cu_q_lens: NamedArray,  # i32[Seq + 1] <-- cumulative lengths for the sequences, including new tokens
    num_seqs: jnp.ndarray,
    sm_scale: float = 1.0,
    soft_cap: float | None = None,
) -> NamedArray:
    """Ragged attention for paged KV caches.

    This function dispatches to the TPU implementation when available and
    supported, otherwise it falls back to :func:`default_ragged_paged_attention`.
    """

    def _tpu_rpa_available() -> bool:
        if tpu_ragged_paged_attention is None:
            return False
        if jax.default_backend() != "tpu":
            return False
        kind = str(getattr(jax.devices()[0], "device_kind", "")).lower()
        if "tpu v2" in kind or "tpu v3" in kind:
            return False
        return True

    if _tpu_rpa_available():
        try:
            out = _do_tpu_ragged_paged_attention(
                q,
                kv_pages,
                kv_lens,
                page_indices,
                cu_q_lens,
                num_seqs,
                sm_scale=sm_scale,
                soft_cap=soft_cap,
            )
            return out
        except Exception:  # pragma: no cover - fall back if kernel fails
            warnings.warn("TPU ragged paged attention failed. Falling back to reference implementation.")
            logger.warning("Failed to use TPU ragged paged attention. Falling back to reference", exc_info=True)

    return default_ragged_paged_attention(
        q,
        kv_pages,
        kv_lens,
        page_indices,
        cu_q_lens.array,
        num_seqs,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
    )


def _do_tpu_ragged_paged_attention(
    q: ht.Float[NamedArray, "position kv_head q_heads_per_group head_size"],
    kv_pages: ht.Float[NamedArray, "page page_size kv_head head_size"],
    kv_lens: ht.i32[NamedArray, " seq"],  # type: ignore[name-defined]
    page_indices: ht.i32[NamedArray, "seq page"],
    cu_q_lens: ht.i32[NamedArray, " seq"],  # type: ignore[name-defined]
    num_seqs: jnp.ndarray,  # scalar int32
    sm_scale: float = 1.0,
    soft_cap: float | None = None,
) -> NamedArray:
    # Usual shardmap dance
    # The TPU kernel expects the second dimension of the query tensor to be the total number of query heads.
    q_flat = q.flatten_axes(("kv_head", "q_heads_per_group"), "kv_head")
    if num_seqs.ndim == 0:
        this_num_seqs = num_seqs.reshape((1,))
    else:
        this_num_seqs = num_seqs

    # the INVALIDs make the TPU sad. mask them with 0:
    this_num_seqs = jnp.where(this_num_seqs < 0, 0, this_num_seqs)
    page_indices = hax.where(~is_valid(page_indices), 0, page_indices)
    kv_lens = hax.where(~is_valid(kv_lens), 0, kv_lens)

    o = shard_map(
        functools.partial(tpu_ragged_paged_attention, sm_scale=sm_scale, soft_cap=soft_cap),
        haliax.partitioning._get_mesh(),
        in_specs=(
            haliax.partitioning.pspec_for_axis(q_flat.axes),
            haliax.partitioning.pspec_for_axis(kv_pages.axes),
            haliax.partitioning.pspec_for_axis(kv_lens.axes),
            haliax.partitioning.pspec_for_axis(page_indices.axes),
            haliax.partitioning.pspec_for_axis(cu_q_lens.axes),
            # haliax.partitioning.pspec_for_axis(num_seqs)
            PartitionSpec(),  # num_seqs
        ),
        out_specs=pspec_for_axis(
            (
                "position",
                "kv_head",
                "head_size",
            )
        ),
        check_rep=False,
    )(
        q_flat.array,
        kv_pages.array,
        kv_lens.array,
        page_indices.array,
        cu_q_lens.array,
        this_num_seqs,
    )

    out = hax.named(
        o,
        ("position", "kv_head", "head_size"),
    )
    out = out.unflatten_axis(
        "kv_head",
        (
            q.resolve_axis("kv_head"),
            q.resolve_axis("q_heads_per_group"),
        ),
    )

    return out


def default_ragged_paged_attention(
    q: NamedArray,  # [tok, KVHeads, QHeadsPerGroup, HeadSize]
    kv_pages: NamedArray,  # [Page, PageSize, 2 * KVHeads, HeadDim]
    kv_lens: NamedArray,  # i32[Seq]
    page_indices: NamedArray,  # i32[Seq, PagePerSeq]
    cu_q_lens: jnp.ndarray,  # i32[Seq + 1] <-- cumulative lengths for the sequences, including new tokens
    num_seqs: jnp.ndarray,  # scalar int32
    sm_scale: float,
    soft_cap: float | None = None,
) -> NamedArray:
    """Default implementation of ragged paged attention.
    This implementation is not optimized for performance and is intended for testing purposes.

    It does each sequence independently
    """

    Q_BS = min(1, q.axis_size("position"))  # block size for query
    KV_BS = min(2, page_indices.axis_size("page"))  # block size for key-value
    Q_B = hax.Axis("position", Q_BS)

    H = q.resolve_axis("kv_head")
    Q_H = q.resolve_axis("q_heads_per_group")

    D = q.resolve_axis("head_size")

    page_size = kv_pages.array.shape[1]

    q = q * sm_scale

    # pad by at least ``Q_BS`` positions so that any block starting within the
    # original array has enough headroom for a full block slice. This avoids the
    # clamping behavior of ``jax.lax.dynamic_slice`` when ``start + size``
    # exceeds the array length.
    padding_amount = (Q_BS - q.axis_size("position") % Q_BS) % Q_BS
    if padding_amount != 0:
        padded_q = hax.concatenate(
            "position",
            [q, hax.zeros_like(q["position", hax.ds(0, padding_amount)])],
        )
    else:
        padded_q = q

    q_orig = q
    q = padded_q

    output = hax.zeros_like(q)

    def _compute_attention_for_seq(seq_id, carry):
        o = carry
        # have to be careful since we're in jit
        q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
        num_q_blocks = (q_len + Q_BS - 1) // Q_BS

        def _compute_attention_for_q_block(q_block_id, carry):
            o = carry
            q_start = cu_q_lens[seq_id] + q_block_id * Q_BS
            q_block = q.at["position", hax.ds(q_start, Q_B)].get(mode="fill", fill_value=float("nan"))
            kv_len = kv_lens["seq", seq_id].scalar()

            # q_start indexes into the global query tensor, so we need to
            # convert it to the token position within this sequence.
            # kv_len is the total length of the sequence in the KV cache,
            # including any prefix tokens. q_len is just the number of query
            # tokens for this sequence. The position of the first query token
            # within the sequence is therefore ``kv_len - q_len``. Adding the
            # block offset ``q_start - cu_q_lens[seq_id]`` yields the absolute
            # position of the current block within the sequence.
            q_pos_id_start = kv_len - q_len + q_start - cu_q_lens[seq_id]
            q_pos_id_end = q_pos_id_start + q_len
            q_tok = hax.arange(q_block.resolve_axis("position"), start=q_pos_id_start)

            kv_pos_per_block = page_size * KV_BS  # how many tokens per kv block

            num_kv_blocks = (kv_len + kv_pos_per_block - 1) // kv_pos_per_block

            def _compute_attention_for_kv_block(kv_block_id, carry):
                o_b, sum_exp_b, max_b = carry

                kv_page_start = kv_block_id * KV_BS
                block_page_idx = page_indices["seq", seq_id, "page", hax.ds(kv_page_start, KV_BS)]

                kv_pos_start = kv_page_start * page_size

                slots = kv_pages["page", block_page_idx, "slot", :]
                kv_block = slots.flatten_axes(("page", "slot"), "kv_position")

                kv_tok = hax.arange(kv_block.resolve_axis("kv_position"), start=kv_pos_start)
                k_block = kv_block["kv_head", 0::2]
                v_block = kv_block["kv_head", 1::2]

                attn_b = hax.dot(q_block, k_block, axis=(D,))

                if soft_cap is not None:
                    attn_b = hax.tanh(attn_b / soft_cap) * soft_cap

                attn_mask = kv_tok.broadcast_axis(q_tok.axes) <= q_tok  # causal
                attn_mask = attn_mask & (kv_tok < kv_len) & (q_tok < q_pos_id_end)  # stay within bounds

                attn_b = hax.where(attn_mask, attn_b, -1e10)

                new_max_b = hax.maximum(max_b, hax.max(attn_b, "kv_position"))
                P_ij = hax.exp(attn_b - new_max_b)
                P_ij = hax.where(attn_mask, P_ij, 0.0)

                exp_diff = hax.exp(max_b - new_max_b)
                sum_exp_b = exp_diff * sum_exp_b + hax.sum(P_ij, axis="kv_position")

                o_b = exp_diff * o_b + hax.dot(P_ij, v_block, axis="kv_position")

                return o_b, sum_exp_b, new_max_b

            # standard flashattention loop with fancy paging
            o_b = o.at["position", hax.ds(q_start, Q_BS)].get(mode="fill", fill_value=float("nan"))
            sum_exp_b = hax.zeros((Q_B, H, Q_H))
            max_b = hax.full((Q_B, H, Q_H), -jnp.inf)

            o_b, sum_exp_b, max_b = jax.lax.fori_loop(
                0, num_kv_blocks, _compute_attention_for_kv_block, (o_b, sum_exp_b, max_b)
            )

            # Normalize
            sum_exp_b = hax.maximum(sum_exp_b, 1e-10)
            o_b = o_b / sum_exp_b
            # mask out anything not in the original query range
            o_b = hax.where(q_tok < q_pos_id_end, o_b, 0.0)
            o = o.at["position", hax.ds(q_start, Q_BS)].set(o_b, mode="drop")
            return o

        o = jax.lax.fori_loop(0, num_q_blocks, _compute_attention_for_q_block, o)

        return o

    output = jax.lax.fori_loop(0, num_seqs, _compute_attention_for_seq, output)
    output = output["position", 0 : q_orig.axis_size("position")]

    return output


@dataclass(frozen=True)
class MultiHeadLatentAttentionConfig:
    """Configuration for MultiHeadLatentAttention adapted from DeepSeek-V3."""

    Embed: Axis
    num_heads: int
    kv_lora_rank: int
    q_lora_rank: int | None = None
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    use_bias: bool = False
    upcast_attn: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    rope: Optional[RotaryEmbeddingsConfig] = None
    scaling_factor: Optional[float] = None
    logits_soft_cap: Optional[float] = None

    @property
    def Heads(self) -> Axis:
        return Axis("heads", self.num_heads)

    @property
    def QHeadSize(self) -> Axis:
        return Axis("q_head_dim", self.qk_rope_head_dim + self.qk_nope_head_dim)

    @property
    def VHeadSize(self) -> Axis:
        return Axis("v_head_dim", self.v_head_dim)

    @property
    def LatentSize(self) -> Axis:
        return Axis("latent", self.kv_lora_rank)

    @property
    def QLoraSize(self) -> Axis:
        return Axis("q_lora_rank", self.q_lora_rank)

    @property
    def KVCombinedSize(self) -> Axis:
        return Axis("kv_combined", self.kv_lora_rank + self.qk_rope_head_dim)


class MultiHeadLatentAttention(eqx.Module):
    """Multi-head attention layer with latent projections inspired by DeepSeek-V3.
    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
    """

    config: MultiHeadLatentAttentionConfig = eqx.field(static=True)
    kv_a_proj: hnn.Linear
    kv_a_norm: LayerNormBase
    kv_b_proj: hnn.Linear
    o_proj: hnn.Linear

    q_proj: hnn.Linear = None
    q_a_proj: Optional[hnn.Linear] = None
    q_a_norm: Optional[LayerNormBase] = None
    q_b_proj: Optional[hnn.Linear] = None

    rot_embs: Optional[RotaryEmbeddings] = eqx.field(default=None)

    @staticmethod
    def init(config: MultiHeadLatentAttentionConfig, *, key) -> "MultiHeadLatentAttention":
        use_bias = config.use_bias
        keys = jrandom.split(key, 5)
        if config.q_lora_rank is None:
            q_proj = hnn.Linear.init(
                In=config.Embed,
                Out=(config.Heads, config.QHeadSize),
                key=keys[0],
                use_bias=False,
                out_first=True,
            )
            q_a_proj = None
            q_a_norm = None
            q_b_proj = None
        else:
            q_a_proj = hnn.Linear.init(
                In=config.Embed,
                Out=config.QLoraSize,
                key=keys[0],
                use_bias=use_bias,
                out_first=True,
            )
            q_a_norm = hnn.RmsNorm.init(Axis("q_lora_rank", config.q_lora_rank), use_bias=False)
            q_b_proj = hnn.Linear.init(
                In=config.QLoraSize,
                Out=(config.Heads, config.QHeadSize),
                key=keys[1],
                use_bias=False,
                out_first=True,
            )
            q_proj = None

        kv_a_proj = hnn.Linear.init(
            In=config.Embed,
            Out=config.KVCombinedSize,
            key=keys[2],
            use_bias=use_bias,
            out_first=True,
        )
        kv_a_norm = hnn.RmsNorm.init(Axis("latent", config.kv_lora_rank), use_bias=False)
        kv_b_proj = hnn.Linear.init(
            In=config.LatentSize,
            Out=(
                config.Heads,
                Axis("kv_out", config.qk_nope_head_dim + config.v_head_dim),
            ),
            key=keys[3],
            use_bias=False,
            out_first=True,
        )
        o_proj = hnn.Linear.init(
            In=(config.Heads, config.VHeadSize),
            Out=config.Embed,
            key=keys[4],
            use_bias=use_bias,
            out_first=True,
        )
        rot_embs = config.rope.build(Axis("q_head_dim", config.qk_rope_head_dim)) if config.rope is not None else None

        return MultiHeadLatentAttention(
            config,
            kv_a_proj,
            kv_a_norm,
            kv_b_proj,
            o_proj,
            q_proj,
            q_a_proj,
            q_a_norm,
            q_b_proj,
            rot_embs,
        )

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        k_q_a, k_q_b, k_kv_a, k_kv_b, k_o = maybe_rng_split(key, 5)

        # Project to a shared latent space for K and V.
        # For inference, this means you just need to cache the reduced size latent.
        kv = self.kv_a_proj(x, key=k_kv_a)
        compressed_kv = kv["kv_combined", : self.config.kv_lora_rank].rename({"kv_combined": "latent"})

        # We can't do RoPE on K without materializing K, so we shave off a
        # qk_rope_head_dim-sized chunk to materialize for RoPE.
        k_pe = (
            kv["kv_combined", self.config.kv_lora_rank :]
            .rename({"kv_combined": "q_head_dim"})
            .broadcast_axis(self.config.Heads)
            .rearrange(("batch", "heads", "position", "q_head_dim"))
        )
        compressed_kv_norm = self.kv_a_norm(compressed_kv)
        kv_out = self.kv_b_proj(compressed_kv_norm, key=k_kv_b)

        # Split the matrix into K_nope and the full V.
        k_nope = kv_out["kv_out", : self.config.qk_nope_head_dim].rename({"kv_out": "q_head_dim"})
        v = kv_out["kv_out", self.config.qk_nope_head_dim :].rename({"kv_out": "v_head_dim"})

        # Optional step of doing LoRA on Q (as done in DeepSeek).
        if self.config.q_lora_rank is None:
            q = self.q_proj(x, key=k_q_a)
        else:
            assert (
                self.q_a_proj is not None and self.q_a_norm is not None and self.q_b_proj is not None
            ), "q_lora_rank defined, but LoRA matrices are not."
            q = self.q_a_proj(x, key=k_q_a)
            q = self.q_a_norm(q)
            q = self.q_b_proj(q, key=k_q_b)
        q = q.rearrange((..., "heads", "position", "q_head_dim"))

        # Prep for partial RoPE.
        q_nope = q["q_head_dim", : self.config.qk_nope_head_dim]
        q_pe = q["q_head_dim", self.config.qk_nope_head_dim :]

        # Apply RoPE to the split-off portion and then merge back together.
        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"), dtype=jnp.int32)
            q_pe = self.rot_embs(q_pe, pos_ids)
            k_pe = self.rot_embs(k_pe, pos_ids)

        query_states = hax.concatenate("q_head_dim", (q_nope, q_pe))
        key_states = hax.concatenate("q_head_dim", (k_nope, k_pe))

        # Rename axes for attention inputs.
        key_states = key_states.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})
        # Build the value tensor AFTER renaming position → key_position.
        v_attn = v.rename({"v_head_dim": "q_head_dim"})

        attn_output = dot_product_attention(
            "position",
            "key_position",
            "q_head_dim",
            query_states,
            key_states,
            v_attn,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            scaling_factor=self.config.scaling_factor,
            logits_soft_cap=self.config.logits_soft_cap,
            dropout=0.0,
            inference=True,
            prng=key,
        )

        attn_output = attn_output.rename({"q_head_dim": "v_head_dim"}).astype(x.dtype)
        assert self.o_proj is not None
        attn_output = self.o_proj(attn_output, key=k_o)
        return attn_output


class AttentionWithSink(Attention):
    """Attention module that includes a learned sink term per head.

    The sink is added to the softmax denominator, reducing the attention mass
    assigned to tokens and allowing some probability to fall into a separate
    bucket. This can improve stability during generation.
    """

    sinks: NamedArray | None = None

    @staticmethod
    def init(config: AttentionConfig, *, key) -> "AttentionWithSink":
        base = Attention.init(config, key=key)
        sinks = hax.zeros((config.KVHeads, config.QHeadsPerGroup), dtype=jnp.float32)
        return AttentionWithSink(
            base.config,
            base.q_proj,
            base.k_proj,
            base.v_proj,
            base.o_proj,
            base.q_norm,
            base.k_norm,
            base.rot_embs,
            sinks,
        )

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        q_proj = self.q_proj(x, key=key_q)
        k_proj = self.k_proj(x, key=key_k)
        v = self.v_proj(x, key=key_v)

        if self.config.qk_norm is not None:
            q = self.q_norm(q_proj)  # type: ignore[misc]
            k = self.k_norm(k_proj)  # type: ignore[misc]
        else:
            q = q_proj
            k = k_proj

        q = q.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_head", "position", "head_size"))
        v = v.rearrange((..., "kv_head", "position", "head_size"))

        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"), dtype=jnp.int32)
            q = self.rot_embs(q, pos_ids)
            k = self.rot_embs(k, pos_ids)

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        attn_output = dot_product_attention_with_sink(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            self.sinks,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            scaling_factor=self.config.scaling_factor,
            logits_soft_cap=self.config.logits_soft_cap,
            dropout=0.0,
            inference=True,
            prng=key,
        )

        attn_output = attn_output.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        attn_output = self.o_proj(attn_output, key=key_o)

        return attn_output
