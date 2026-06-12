# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import logging
import math
import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Optional, TypedDict, Union, cast

import equinox as eqx
import jax
import jax.random as jrandom
from jax import numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel

from levanter.kernels.pallas.splash_attention import (
    DEFAULT_SPLASH_BLOCK_SIZE,
    SPLASH_BLOCK_GRANULARITY,
    SplashSegmentIdsLowering,
    lower_splash_attention_mask,
    lower_splash_segment_ids,
    packed_causal_segment_mask_infos,
    packed_causal_segment_run_mask_infos,
    packed_prefix_lm_mask_infos,
    prefix_lm_mask_infos,
    splash_attention_block_sizes,
    splash_attention_mask_spec_from_fields,
    splash_partition_spec_shard_factor,
)
from levanter.inference.utils import is_valid

try:
    from jax.experimental.pallas.ops.tpu.ragged_paged_attention import (
        ragged_paged_attention as tpu_ragged_paged_attention,
    )
except Exception:  # pragma: no cover - optional dep
    tpu_ragged_paged_attention = None

import haliax
import haliax as hax
import haliax.haxtyping as ht
import haliax.nn as hnn
from haliax import Axis, AxisSelection, AxisSelector, NamedArray, axis_name
from haliax.jax_utils import maybe_rng_split, named_call
from haliax.nn.normalization import LayerNormBase
from haliax.partitioning import pspec_for_axis, shard_map
from haliax.types import PrecisionLike
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import PRNGKeyArray

try:
    from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import _splash_attention
except Exception:
    _SPLASH_KERNEL_SUPPORTS_SINKS = False
else:
    _SPLASH_KERNEL_SUPPORTS_SINKS = "sinks" in inspect.signature(_splash_attention).parameters

from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from .attention_mask import AttentionMask, materialize_mask
from .kv_cache import KvPageCache
from .normalization import LayerNormConfigBase
from .rotary import RotaryEmbeddings, RotaryEmbeddingsConfig

logger = logging.getLogger(__name__)


class AttentionBackend(StrEnum):
    DEFAULT = "default"  # use the default attention type for the accelerator
    NVTE = "nvte"  # with Transformer Engine on NVIDIA GPUs
    SPLASH = "splash"  # on TPU.
    JAX_FLASH = "jax_flash"  # Use the JAX reference implementation
    VANILLA = "vanilla"  # regular dot product attention


_SPLASH_FALLBACK_WARNINGS_EMITTED: set[str] = set()
SPLASH_BATCH_AXIS_NAME = "splash_batch"
SPLASH_HEAD_AXIS_NAME = "splash_head"


def _warn_splash_fallback_once(message: str) -> None:
    if message in _SPLASH_FALLBACK_WARNINGS_EMITTED:
        return
    _SPLASH_FALLBACK_WARNINGS_EMITTED.add(message)
    warnings.warn(message, stacklevel=3)


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
    attn_sink: Optional[NamedArray] = None,
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

    attention_out = None

    match attn_backend:
        case AttentionBackend.NVTE:
            attention_out = _try_te_attention(
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
                flash_block_size=flash_block_size,
                force_te=not was_default,
                scaling_factor=scaling_factor,
                logits_soft_cap=logits_soft_cap,
                attn_sink=attn_sink,
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
                attn_sink=attn_sink,
            )

        case AttentionBackend.VANILLA:
            if attn_sink is None:
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
            else:
                key2, value2, mask2, bias2, KPosPlus = _materialize_sink_as_dummy_kv(
                    QPos=QPos,
                    KPos=KPos,
                    Key=Key,
                    query=query,
                    key=key,
                    value=value,
                    attn_sink=attn_sink,
                    mask=mask,
                    bias=bias,
                )
                attention_out = simple_attention_with_dropout(
                    QPos,
                    KPosPlus,
                    Key,
                    query,
                    key2,
                    value2,
                    mask2,
                    bias2,
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
        from levanter.models.flash_attention import (  # noqa: PLC0415  # circular import: attention -> flash_attention -> attention
            flash_attention,
        )

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
            attn_sink=attn_sink,
        )


def _materialize_sink_as_dummy_kv(
    *,
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    attn_sink: NamedArray,
    mask: Optional[Union["AttentionMask", NamedArray]],
    bias: Optional[NamedArray],
):
    """
    Preprocess for dot-product attention variant with a learned sink term per head.

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

    return key, value, m, bias, KPosPlus


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
    scaling_factor: float | jax.Array | None = None,
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
    attn_sink: Optional[NamedArray] = None,  # NEW
):
    """
    Try NVTE fused attention. If unsupported, either raise (when forced) or warn and return None.
    Also rejects `attn_sink` since NVTE doesn't support it yet. (Centralizing this logic
    matches the review suggestion to keep the 'forced backend must raise' contract here.)
    """
    if attn_sink is not None:
        msg = "NVTE fused attention does not support attention sinks; falling back to reference."
        if force_te:
            raise NotImplementedError("NVTE fused attention does not support attention sinks.")
        warnings.warn(msg)
        return None

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
            if _dtype not in (
                jnp.float16,
                jnp.bfloat16,
                jnp.float8_e5m2,
                jnp.float8_e4m3fn,
            ):
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
    from transformer_engine.jax.attention import (  # type: ignore[import]  # noqa: PLC0415  # optional dep: transformer_engine
        AttnBiasType,
        QKVLayout,
        SequenceDescriptor,
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

    axis_bins = _bin_and_group_axes_by_function(query, key, value, QPos, KPos, Key)
    q_class = axis_bins.q
    k_class = axis_bins.k
    v_class = axis_bins.v
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

    # Determine attention mask type from the mask object
    attn_mask_type = _te_get_mask_type(mask)

    is_training = not inference

    # TODO: bias type is probably also configurable
    attn_bias_type = AttnBiasType.NO_BIAS
    fused_attn_bias = None
    if bias:
        raise NotImplementedError("Using bias with flash attention on GPU is not currently implemented.")

    # TE 2.x uses SequenceDescriptor instead of explicit masks.
    # For non-packed sequences, all sequences have full length.
    q_seqlens = jnp.full((B,), Sq, dtype=jnp.int32)
    kv_seqlens = jnp.full((B,), Sk, dtype=jnp.int32)

    # Extract segment_ids from mask if present
    segment_ids_for_te = None
    if isinstance(mask, AttentionMask) and mask.segment_ids is not None:
        q_segment_ids, kv_segment_ids = map(lambda x: x.astype(jnp.int32), mask.segment_ids)

        batch_axes = tuple(q_class["B"])
        for ax in batch_axes:
            if ax.name not in q_segment_ids.axes:
                q_segment_ids = q_segment_ids.broadcast_axis(ax)
            if ax.name not in kv_segment_ids.axes:
                kv_segment_ids = kv_segment_ids.broadcast_axis(ax)

        q_seg_reshaped = _maybe_flatten(q_segment_ids, batch_axes, "B")
        q_seg_reshaped = q_seg_reshaped.rearrange(("B", QPos)).array
        kv_seg_reshaped = _maybe_flatten(kv_segment_ids, batch_axes, "B")
        kv_seg_reshaped = kv_seg_reshaped.rearrange(("B", KPos)).array
        segment_ids_for_te = (q_seg_reshaped, kv_seg_reshaped)

    sequence_descriptor = SequenceDescriptor.from_seqlens((q_seqlens, kv_seqlens), segment_ids=segment_ids_for_te)

    attn_output = fused_attn(
        (q_, k_, v_),
        fused_attn_bias,
        sequence_descriptor,
        prng,
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


def _te_get_mask_type(mask):
    """Get the TE AttnMaskType from a mask object without materializing it."""
    from transformer_engine.jax.attention import (  # type: ignore[import]  # noqa: PLC0415  # optional dep: transformer_engine
        AttnMaskType,
    )

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
            return AttnMaskType.CAUSAL_MASK
        else:
            raise NotImplementedError(
                "Non-causal AttentionMask is not supported for NVTE fused attention."
                " Please use the JAX reference implementation."
            )
    else:
        return AttnMaskType.NO_MASK


_DUMMY_HEAD = "__head__"
_DUMMY_BATCH = "__batch__"


class _AxisBins(TypedDict):
    B: list[Axis]
    S: list[Axis]
    H: list[Axis]
    D: list[Axis]


@dataclass(frozen=True)
class _QkvAxisBins:
    q: _AxisBins
    k: _AxisBins
    v: _AxisBins


def _bin_and_group_axes_by_function(q, k, v, QPos, KPos, Key) -> _QkvAxisBins:
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

    q_class: _AxisBins = {"B": [], "S": [QPos], "H": [], "D": [Key]}
    k_class: _AxisBins = {"B": [], "S": [KPos], "H": [], "D": [Key]}
    v_class: _AxisBins = {"B": [], "S": [KPos], "H": [], "D": [Key]}

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

    return _QkvAxisBins(q=q_class, k=k_class, v=v_class)


def _maybe_flatten(q, axes, name):
    if axes:
        q = q.flatten_axes(axes, name)
    else:
        q = q.broadcast_axis(Axis(name, 1))
    return q


def _reshape_axes_for_bshd_bins(q, q_class, output_order=("B", "S", "H", "D")):
    """
    Reshape the axes of a qkv as BSHD to match the bins in q_class
    """

    q = _maybe_flatten(q, q_class["B"], "B")
    q = _maybe_flatten(q, q_class["S"], "S")
    q = _maybe_flatten(q, q_class["H"], "H")
    q = _maybe_flatten(q, q_class["D"], "D")
    q = q.rearrange(output_order)
    return q


def _prepare_sinks_for_splash(attn_sink: NamedArray, q_class, physical_axes_q: PartitionSpec):
    """Reshape and broadcast attention sinks to (B, H) for the splash kernel."""

    batch_axes = tuple(q_class["B"])
    head_axes = tuple(q_class["H"])
    allowed_axes = {ax.name for ax in batch_axes + head_axes}

    sink = _prepare_splash_batch_value(
        attn_sink,
        batch_axes=batch_axes,
        allowed_axis_names=allowed_axes,
        value_name="Attention sinks",
    )
    sink_axis_names = {ax.name for ax in sink.axes}
    for ax in head_axes:
        if ax.name not in sink_axis_names:
            sink = sink.broadcast_axis(ax)
            sink_axis_names.add(ax.name)

    if head_axes:
        sink = _maybe_flatten(sink, head_axes, SPLASH_HEAD_AXIS_NAME)
    else:
        sink = _maybe_flatten(sink, (), SPLASH_HEAD_AXIS_NAME)

    sink = sink.rearrange((SPLASH_BATCH_AXIS_NAME, SPLASH_HEAD_AXIS_NAME))

    sinks_array = sink.astype(jnp.float32).array  # also cast to fp32 inside splash
    physical_axes_sink = PartitionSpec(physical_axes_q[0], physical_axes_q[1])

    return sinks_array, physical_axes_sink


def _prepare_prefix_lengths_for_splash(prefix_lengths: NamedArray, q_class, physical_axes_q: PartitionSpec):
    batch_axes = tuple(q_class["B"])
    lengths = _prepare_splash_batch_value(
        prefix_lengths,
        batch_axes=batch_axes,
        allowed_axis_names={ax.name for ax in batch_axes},
        value_name="Prefix lengths",
    )
    lengths = lengths.rearrange((SPLASH_BATCH_AXIS_NAME,))
    return lengths.astype(jnp.int32).array, PartitionSpec(physical_axes_q[0])


def _prepare_splash_batch_value(
    value: NamedArray,
    *,
    batch_axes: tuple[Axis, ...],
    allowed_axis_names: set[str],
    value_name: str,
) -> NamedArray:
    extra_axes = tuple(ax for ax in value.axes if ax.name not in allowed_axis_names)
    if extra_axes:
        raise NotImplementedError(
            f"{value_name} contain axes unsupported by splash attention: {', '.join(ax.name for ax in extra_axes)}"
        )

    axis_names = {ax.name for ax in value.axes}
    for ax in batch_axes:
        if ax.name not in axis_names:
            value = value.broadcast_axis(ax)
            axis_names.add(ax.name)

    if batch_axes:
        return _maybe_flatten(value, batch_axes, SPLASH_BATCH_AXIS_NAME)
    return _maybe_flatten(value, (), SPLASH_BATCH_AXIS_NAME)


def _prepare_prefix_mask_for_splash(
    prefix_mask: NamedArray,
    QPos: Axis,
    KPos: Axis,
    q_class,
    physical_axes_q: PartitionSpec,
    physical_axes_k: PartitionSpec,
):
    batch_axes = tuple(q_class["B"])
    if QPos.name in {ax.name for ax in prefix_mask.axes} and QPos.name != KPos.name:
        prefix_mask = prefix_mask.rename({QPos.name: KPos.name})
    if KPos.name not in {ax.name for ax in prefix_mask.axes}:
        raise ValueError(f"prefix_mask must contain key position axis {KPos.name}.")

    mask = _prepare_splash_batch_value(
        prefix_mask,
        batch_axes=batch_axes,
        allowed_axis_names={ax.name for ax in batch_axes + (KPos,)},
        value_name="Prefix mask",
    )
    mask = mask.rearrange((SPLASH_BATCH_AXIS_NAME, KPos.name))
    return mask.astype(jnp.bool_).array, PartitionSpec(physical_axes_q[0], physical_axes_k[2])


def _batched_splash_kernel_specs(
    kernel,
    *,
    batch_spec,
    head_spec,
    q_seq_spec,
    num_heads: int,
):
    def spec_for_leaf(leaf):
        if leaf is None:
            return None
        if not isinstance(leaf, jax.Array):
            return PartitionSpec()
        if leaf.ndim == 0:
            return PartitionSpec()
        if leaf.ndim >= 4 and leaf.shape[1] == num_heads:
            return PartitionSpec(batch_spec, head_spec, q_seq_spec, *([None] * (leaf.ndim - 3)))
        return PartitionSpec(batch_spec, *([None] * (leaf.ndim - 1)))

    return jax.tree_util.tree_map(spec_for_leaf, kernel)


def _splash_kernel_from_dynamic_metadata(metadata, block_sizes, logits_soft_cap):
    return splash_attention_kernel.SplashAttentionKernel(
        metadata.fwd_mask_info,
        None if block_sizes.use_fused_bwd_kernel else metadata.dq_mask_info,
        metadata.dkv_mask_info,
        block_sizes=block_sizes,
        is_mqa=False,
        save_residuals=False,
        mask_value=splash_attention_kernel.DEFAULT_MASK_VALUE,
        attn_logits_soft_cap=logits_soft_cap,
        residual_checkpoint_name=None,
        mask_function=None,
        interpret=False,
    )


@dataclass(frozen=True)
class _SplashPrefixControls:
    prefix_lengths: jax.Array | None
    physical_axes_prefix_lengths: PartitionSpec | None
    prefix_masks: jax.Array | None
    physical_axes_prefix_mask: PartitionSpec | None


@dataclass(frozen=True)
class _SplashSegmentRunControls:
    segment_lengths: jax.Array | None
    physical_axes_segment_lengths: PartitionSpec | None
    num_segments: jax.Array | None


@dataclass(frozen=True)
class _SplashKernelPlan:
    kernel: object
    kernel_specs: object
    kernel_vmap_axis: int | None
    segment_id_lowering: SplashSegmentIdsLowering


@dataclass(frozen=True)
class _SplashKernelContext:
    q_seq_len: int
    kv_seq_len: int
    num_heads: int
    block_sizes: splash_attention_kernel.BlockSizes
    head_shards: int
    q_seq_shards: int
    physical_axes_q: PartitionSpec
    logits_soft_cap: float | None


@dataclass(frozen=True)
class _SplashPreparedLayout:
    q: jax.Array
    k: jax.Array
    v: jax.Array
    q_class: _AxisBins
    k_class: _AxisBins
    v_class: _AxisBins
    QPos: Axis
    KPos: Axis
    batch: int
    heads: int
    q_seq_len: int
    kv_seq_len: int
    physical_axes_q: PartitionSpec
    physical_axes_k: PartitionSpec
    physical_axes_v: PartitionSpec


@dataclass(frozen=True)
class _SplashInvocationPlan:
    mesh: Mesh
    kernel_plan: _SplashKernelPlan
    sinks: jax.Array | None
    physical_axes_sink: PartitionSpec | None


@dataclass(frozen=True)
class _PackedSegmentIds:
    q: jax.Array
    kv: jax.Array
    q_vmap_axis: int | None
    kv_vmap_axis: int | None


def _flatten_partition_spec_entries(entries):
    if entries is None:
        return entries
    result = []
    for entry in entries:
        if isinstance(entry, tuple):
            result += list(entry)
        else:
            result.append(entry)
    return tuple(result)


def _physical_axis_for_binning(axis_groups):
    b_out = _flatten_partition_spec_entries(
        tuple(ax for ax in pspec_for_axis(axis_groups["B"]) if ax is not None) or None
    )
    h_out = _flatten_partition_spec_entries(
        tuple(ax for ax in pspec_for_axis(axis_groups["H"]) if ax is not None) or None
    )
    s_out = _flatten_partition_spec_entries(
        tuple(ax for ax in pspec_for_axis(axis_groups["S"]) if ax is not None) or None
    )
    d_out = _flatten_partition_spec_entries(
        tuple(ax for ax in pspec_for_axis(axis_groups["D"]) if ax is not None) or None
    )
    return PartitionSpec(b_out, h_out, s_out, d_out)


def _prepare_splash_prefix_controls(
    *,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    QPos: Axis,
    KPos: Axis,
    q_class,
    physical_axes_q: PartitionSpec,
    physical_axes_k: PartitionSpec,
    kv_seq_len: int,
) -> _SplashPrefixControls:
    prefix_lengths = None
    physical_axes_prefix_lengths = None
    prefix_masks = None
    physical_axes_prefix_mask = None

    prefix_lm = mask.prefix_lm_spec if isinstance(mask, AttentionMask) else None
    if prefix_lm is not None and prefix_lm.prefix_lengths is not None:
        if prefix_lm.prefix_mask is not None:
            raise NotImplementedError("Splash attention does not support prefix_lengths combined with prefix_mask.")
        if mask.sliding_window is not None:
            raise NotImplementedError("Splash attention does not support dynamic prefix lengths with sliding windows.")
        prefix_lengths, physical_axes_prefix_lengths = _prepare_prefix_lengths_for_splash(
            prefix_lm.prefix_lengths, q_class, physical_axes_q
        )
        if prefix_lm.prefix_length is not None:
            prefix_lengths = jnp.maximum(prefix_lengths, prefix_lm.prefix_length)

    if prefix_lm is not None and prefix_lm.prefix_mask is not None:
        if not mask.is_causal:
            raise NotImplementedError("Splash attention requires prefix_mask to be part of a causal prefix-LM mask.")
        if mask.sliding_window is not None:
            raise NotImplementedError("Splash attention does not support prefix_mask with sliding windows.")
        prefix_masks, physical_axes_prefix_mask = _prepare_prefix_mask_for_splash(
            prefix_lm.prefix_mask,
            QPos,
            KPos,
            q_class,
            physical_axes_q,
            physical_axes_k,
        )
        if prefix_lm.prefix_length is not None:
            prefix_positions = jnp.arange(kv_seq_len, dtype=jnp.int32)[None, :] < prefix_lm.prefix_length
            prefix_masks = prefix_masks | prefix_positions

    return _SplashPrefixControls(
        prefix_lengths=prefix_lengths,
        physical_axes_prefix_lengths=physical_axes_prefix_lengths,
        prefix_masks=prefix_masks,
        physical_axes_prefix_mask=physical_axes_prefix_mask,
    )


def _prepare_splash_segment_run_controls(
    *,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    q_class,
    physical_axes_q: PartitionSpec,
) -> _SplashSegmentRunControls:
    if not isinstance(mask, AttentionMask) or mask.segment_run_metadata is None:
        return _SplashSegmentRunControls(
            segment_lengths=None,
            physical_axes_segment_lengths=None,
            num_segments=None,
        )

    batch_axes = tuple(q_class["B"])
    segment_lengths = mask.segment_run_metadata.segment_lengths
    if not segment_lengths.axes:
        raise ValueError("segment_run_metadata.segment_lengths must include a segment-run axis.")

    segment_run_axis = segment_lengths.axes[-1]
    segment_lengths = _prepare_splash_batch_value(
        segment_lengths,
        batch_axes=batch_axes,
        allowed_axis_names={ax.name for ax in batch_axes + (segment_run_axis,)},
        value_name="Segment-run lengths",
    )
    segment_lengths = segment_lengths.rearrange((SPLASH_BATCH_AXIS_NAME, segment_run_axis.name))

    num_segments = _prepare_splash_batch_value(
        mask.segment_run_metadata.num_segments,
        batch_axes=batch_axes,
        allowed_axis_names={ax.name for ax in batch_axes},
        value_name="Segment-run counts",
    )
    num_segments = num_segments.rearrange((SPLASH_BATCH_AXIS_NAME,))

    return _SplashSegmentRunControls(
        segment_lengths=segment_lengths.astype(jnp.int32).array,
        physical_axes_segment_lengths=PartitionSpec(physical_axes_q[0], None),
        num_segments=num_segments.astype(jnp.int32).array,
    )


def _lower_splash_attention_segment_ids(
    *,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    QPos: Axis,
    KPos: Axis,
) -> SplashSegmentIdsLowering:
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    if segment_ids is None:
        return lower_splash_segment_ids()

    q_segment_ids, kv_segment_ids = segment_ids
    kv_segment_ids = kv_segment_ids.rename({QPos.name: KPos.name})

    q_segment_batch_axis = _find_batch_axis_for_segment_ids(QPos, q_segment_ids)
    kv_segment_batch_axis = _find_batch_axis_for_segment_ids(KPos, kv_segment_ids)

    return lower_splash_segment_ids(
        q_segment_ids=q_segment_ids.array,
        kv_segment_ids=kv_segment_ids.array,
        q_segment_ids_axes=pspec_for_axis(q_segment_ids.axes),
        kv_segment_ids_axes=pspec_for_axis(kv_segment_ids.axes),
        q_segment_batch_axis=q_segment_batch_axis,
        kv_segment_batch_axis=kv_segment_batch_axis,
    )


def _segment_ids_for_packed_mask(
    segment_id_lowering: SplashSegmentIdsLowering,
    *,
    q_seq_len: int,
    kv_seq_len: int,
) -> _PackedSegmentIds:
    if segment_id_lowering.segment_ids is None:
        q_segment_ids = jnp.zeros((q_seq_len,), dtype=jnp.int32)
        kv_segment_ids = jnp.zeros((kv_seq_len,), dtype=jnp.int32)
        return _PackedSegmentIds(q=q_segment_ids, kv=kv_segment_ids, q_vmap_axis=None, kv_vmap_axis=None)

    assert segment_id_lowering.segment_batch_axis is not None
    return _PackedSegmentIds(
        q=segment_id_lowering.segment_ids.q,
        kv=segment_id_lowering.segment_ids.kv,
        q_vmap_axis=cast(int | None, segment_id_lowering.segment_batch_axis.q),
        kv_vmap_axis=cast(int | None, segment_id_lowering.segment_batch_axis.kv),
    )


def _packed_dynamic_mask_kernel_plan(
    *,
    segment_id_lowering: SplashSegmentIdsLowering,
    context: _SplashKernelContext,
    metadata_values: tuple[jax.Array, ...],
    metadata_in_axes: tuple[int | None, ...],
    metadata_builder,
    batch_spec,
) -> _SplashKernelPlan:
    segment_ids = _segment_ids_for_packed_mask(
        segment_id_lowering,
        q_seq_len=context.q_seq_len,
        kv_seq_len=context.kv_seq_len,
    )
    if batch_spec is None:
        assert segment_id_lowering.segment_ids_axes is not None
        assert segment_ids.q_vmap_axis is not None
        batch_spec = segment_id_lowering.segment_ids_axes.q[segment_ids.q_vmap_axis]

    def make_kernel(*args):
        *metadata_args, q_segment_ids, kv_segment_ids = args
        return metadata_builder(*metadata_args, q_segment_ids, kv_segment_ids)

    return _dynamic_metadata_kernel_plan(
        context=context,
        metadata_values=metadata_values + (segment_ids.q, segment_ids.kv),
        metadata_in_axes=metadata_in_axes + (segment_ids.q_vmap_axis, segment_ids.kv_vmap_axis),
        metadata_builder=make_kernel,
        batch_spec=batch_spec,
        segment_id_lowering=lower_splash_segment_ids(),
    )


def _dynamic_metadata_kernel_plan(
    *,
    context: _SplashKernelContext,
    metadata_values: tuple[jax.Array, ...],
    metadata_in_axes: tuple[int | None, ...],
    metadata_builder,
    batch_spec,
    segment_id_lowering: SplashSegmentIdsLowering,
) -> _SplashKernelPlan:
    def make_kernel(*metadata_args):
        metadata = metadata_builder(*metadata_args)
        return _splash_kernel_from_dynamic_metadata(metadata, context.block_sizes, context.logits_soft_cap)

    splash_kernel = jax.vmap(make_kernel, in_axes=metadata_in_axes)(*metadata_values)
    kernel_specs = _batched_splash_kernel_specs(
        splash_kernel,
        batch_spec=batch_spec,
        head_spec=context.physical_axes_q[1],
        q_seq_spec=context.physical_axes_q[2],
        num_heads=context.num_heads,
    )
    return _SplashKernelPlan(
        kernel=splash_kernel,
        kernel_specs=kernel_specs,
        kernel_vmap_axis=0,
        segment_id_lowering=segment_id_lowering,
    )


def _packed_prefix_kernel_plan(
    *,
    prefix_masks: jax.Array,
    physical_axes_prefix_mask: PartitionSpec,
    segment_id_lowering: SplashSegmentIdsLowering,
    context: _SplashKernelContext,
) -> _SplashKernelPlan:
    def prefix_metadata(prefix_mask, q_segment_ids, kv_segment_ids):
        return packed_prefix_lm_mask_infos(
            prefix_mask=prefix_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_seq_len=context.q_seq_len,
            kv_seq_len=context.kv_seq_len,
            block_sizes=context.block_sizes,
            head_shards=context.head_shards,
            q_seq_shards=context.q_seq_shards,
        )

    return _packed_dynamic_mask_kernel_plan(
        segment_id_lowering=segment_id_lowering,
        context=context,
        metadata_values=(prefix_masks,),
        metadata_in_axes=(0,),
        metadata_builder=prefix_metadata,
        batch_spec=physical_axes_prefix_mask[0],
    )


def _can_use_packed_causal_segment_kernel(
    *,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    prefix_controls: _SplashPrefixControls,
    segment_id_lowering: SplashSegmentIdsLowering,
    context: _SplashKernelContext,
) -> bool:
    if not isinstance(mask, AttentionMask):
        return False
    if context.q_seq_shards != 1:
        return False
    if segment_id_lowering.segment_ids is None or segment_id_lowering.segment_batch_axis is None:
        return False
    if segment_id_lowering.segment_batch_axis.q is None or segment_id_lowering.segment_batch_axis.kv is None:
        return False
    return _is_plain_causal_splash_mask(mask=mask, prefix_controls=prefix_controls)


def _is_plain_causal_splash_mask(
    *,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    prefix_controls: _SplashPrefixControls,
) -> bool:
    return (
        isinstance(mask, AttentionMask)
        and mask.is_causal
        and mask.causal_offset is None
        and mask.sliding_window is None
        and mask.prefix_lm_spec is None
        and mask.explicit_mask is None
        and prefix_controls.prefix_lengths is None
        and prefix_controls.prefix_masks is None
    )


def _can_use_packed_causal_segment_run_kernel(
    *,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    prefix_controls: _SplashPrefixControls,
    segment_run_controls: _SplashSegmentRunControls,
    context: _SplashKernelContext,
) -> bool:
    if not isinstance(mask, AttentionMask):
        return False
    if context.q_seq_shards != 1:
        return False
    if context.q_seq_len != context.kv_seq_len:
        return False
    if segment_run_controls.segment_lengths is None or segment_run_controls.num_segments is None:
        return False
    return _is_plain_causal_splash_mask(mask=mask, prefix_controls=prefix_controls)


def _packed_causal_segment_run_kernel_plan(
    *,
    segment_run_controls: _SplashSegmentRunControls,
    context: _SplashKernelContext,
) -> _SplashKernelPlan:
    assert segment_run_controls.segment_lengths is not None
    assert segment_run_controls.num_segments is not None
    assert segment_run_controls.physical_axes_segment_lengths is not None

    def make_kernel(segment_lengths, num_segments):
        metadata = packed_causal_segment_run_mask_infos(
            segment_lengths=segment_lengths,
            num_segments=num_segments,
            q_seq_len=context.q_seq_len,
            kv_seq_len=context.kv_seq_len,
            block_sizes=context.block_sizes,
            head_shards=context.head_shards,
            q_seq_shards=context.q_seq_shards,
        )
        return _splash_kernel_from_dynamic_metadata(metadata, context.block_sizes, context.logits_soft_cap)

    return _dynamic_metadata_kernel_plan(
        context=context,
        metadata_values=(segment_run_controls.segment_lengths, segment_run_controls.num_segments),
        metadata_in_axes=(0, 0),
        metadata_builder=make_kernel,
        batch_spec=segment_run_controls.physical_axes_segment_lengths[0],
        segment_id_lowering=lower_splash_segment_ids(),
    )


def _packed_causal_segment_kernel_plan(
    *,
    segment_id_lowering: SplashSegmentIdsLowering,
    context: _SplashKernelContext,
) -> _SplashKernelPlan:
    def causal_metadata(q_segment_ids, kv_segment_ids):
        return packed_causal_segment_mask_infos(
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_seq_len=context.q_seq_len,
            kv_seq_len=context.kv_seq_len,
            block_sizes=context.block_sizes,
            head_shards=context.head_shards,
            q_seq_shards=context.q_seq_shards,
        )

    return _packed_dynamic_mask_kernel_plan(
        segment_id_lowering=segment_id_lowering,
        context=context,
        metadata_values=(),
        metadata_in_axes=(),
        metadata_builder=causal_metadata,
        batch_spec=None,
    )


def _prefix_length_kernel_plan(
    *,
    prefix_lengths: jax.Array,
    physical_axes_prefix_lengths: PartitionSpec,
    segment_id_lowering: SplashSegmentIdsLowering,
    context: _SplashKernelContext,
) -> _SplashKernelPlan:
    def make_prefix_lm_kernel(prefix_length):
        prefix_metadata = prefix_lm_mask_infos(
            prefix_length=prefix_length,
            q_seq_len=context.q_seq_len,
            kv_seq_len=context.kv_seq_len,
            num_heads=context.num_heads,
            block_sizes=context.block_sizes,
        )
        return _splash_kernel_from_dynamic_metadata(prefix_metadata, context.block_sizes, context.logits_soft_cap)

    return _dynamic_metadata_kernel_plan(
        context=context,
        metadata_values=(prefix_lengths,),
        metadata_in_axes=(0,),
        metadata_builder=make_prefix_lm_kernel,
        batch_spec=physical_axes_prefix_lengths[0],
        segment_id_lowering=segment_id_lowering,
    )


def _static_splash_kernel_plan(
    *,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    segment_id_lowering: SplashSegmentIdsLowering,
    context: _SplashKernelContext,
    mesh,
) -> _SplashKernelPlan:
    if mask is None:
        mask_spec = None
    elif isinstance(mask, AttentionMask):
        mask_spec = splash_attention_mask_spec_from_fields(
            is_causal=mask.is_causal,
            causal_offset=mask.causal_offset,
            sliding_window=mask.sliding_window,
            prefix_length=None if mask.prefix_lm_spec is None else mask.prefix_lm_spec.prefix_length,
            prefix_lengths=None if mask.prefix_lm_spec is None else mask.prefix_lm_spec.prefix_lengths,
            prefix_mask=None if mask.prefix_lm_spec is None else mask.prefix_lm_spec.prefix_mask,
            explicit_mask=mask.explicit_mask,
        )
    elif isinstance(mask, NamedArray):
        raise NotImplementedError("NamedArray masks are not yet supported for splash attention")
    else:
        raise ValueError(f"Unknown mask type: {mask}")

    mask_lowering = lower_splash_attention_mask(
        mask=mask_spec,
        q_seq_len=context.q_seq_len,
        kv_seq_len=context.kv_seq_len,
        num_heads=context.num_heads,
        q_seq_shards=context.q_seq_shards,
    )

    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=mask_lowering.kernel_mask,
        head_shards=context.head_shards,
        q_seq_shards=context.q_seq_shards,
        block_sizes=context.block_sizes,
        attn_logits_soft_cap=context.logits_soft_cap,
    )

    kernel_sharding = jax.sharding.NamedSharding(
        mesh, PartitionSpec(context.physical_axes_q[1], context.physical_axes_q[2])
    )
    kernel_specs = splash_kernel.manual_sharding_spec(kernel_sharding)
    return _SplashKernelPlan(
        kernel=splash_kernel,
        kernel_specs=kernel_specs,
        kernel_vmap_axis=None,
        segment_id_lowering=segment_id_lowering,
    )


def _splash_kernel_plan(
    *,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    prefix_controls: _SplashPrefixControls,
    segment_run_controls: _SplashSegmentRunControls,
    segment_id_lowering: SplashSegmentIdsLowering,
    context: _SplashKernelContext,
    mesh,
) -> _SplashKernelPlan:
    if prefix_controls.prefix_masks is not None:
        assert prefix_controls.physical_axes_prefix_mask is not None
        return _packed_prefix_kernel_plan(
            prefix_masks=prefix_controls.prefix_masks,
            physical_axes_prefix_mask=prefix_controls.physical_axes_prefix_mask,
            segment_id_lowering=segment_id_lowering,
            context=context,
        )

    if prefix_controls.prefix_lengths is not None:
        assert prefix_controls.physical_axes_prefix_lengths is not None
        return _prefix_length_kernel_plan(
            prefix_lengths=prefix_controls.prefix_lengths,
            physical_axes_prefix_lengths=prefix_controls.physical_axes_prefix_lengths,
            segment_id_lowering=segment_id_lowering,
            context=context,
        )

    if _can_use_packed_causal_segment_run_kernel(
        mask=mask,
        prefix_controls=prefix_controls,
        segment_run_controls=segment_run_controls,
        context=context,
    ):
        return _packed_causal_segment_run_kernel_plan(
            segment_run_controls=segment_run_controls,
            context=context,
        )

    if _can_use_packed_causal_segment_kernel(
        mask=mask,
        prefix_controls=prefix_controls,
        segment_id_lowering=segment_id_lowering,
        context=context,
    ):
        return _packed_causal_segment_kernel_plan(
            segment_id_lowering=segment_id_lowering,
            context=context,
        )

    return _static_splash_kernel_plan(
        mask=mask,
        segment_id_lowering=segment_id_lowering,
        context=context,
        mesh=mesh,
    )


def _unflatten_bshd(attn_output, q_class, v_class):
    attn_output = attn_output.unflatten_axis("B", q_class["B"])
    attn_output = attn_output.unflatten_axis("S", q_class["S"])
    attn_output = attn_output.unflatten_axis("H", q_class["H"])
    attn_output = attn_output.unflatten_axis("D", v_class["D"])
    return attn_output


def _prepare_splash_layout(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    *,
    scaling_factor: float,
    block_size: int | None,
) -> _SplashPreparedLayout:
    axis_bins = _bin_and_group_axes_by_function(query, key, value, QPos, KPos, Key)
    q_class = axis_bins.q
    k_class = axis_bins.k
    v_class = axis_bins.v
    query = query * scaling_factor

    q: jax.Array = _reshape_axes_for_bshd_bins(query, q_class, output_order=list("BHSD")).array
    k = _reshape_axes_for_bshd_bins(key, k_class, output_order=list("BHSD")).array
    v = _reshape_axes_for_bshd_bins(value, v_class, output_order=list("BHSD")).array

    batch, heads, q_seq_len, dim = q.shape
    key_batch, _, kv_seq_len, key_dim = k.shape

    if kv_seq_len % SPLASH_BLOCK_GRANULARITY != 0:
        raise NotImplementedError(f"Splash attention requires KPos to be a multiple of {SPLASH_BLOCK_GRANULARITY}")
    if block_size is not None and block_size % SPLASH_BLOCK_GRANULARITY != 0:
        raise NotImplementedError(
            f"Splash attention requires block_size to be a multiple of {SPLASH_BLOCK_GRANULARITY}, got {block_size}"
        )

    QPos = query.resolve_axis(QPos)
    KPos = key.resolve_axis(KPos)

    if k.shape != v.shape:
        raise ValueError("k and v must have the same axes")
    if batch != key_batch:
        raise ValueError(f"Batch axes must be the same for q, k, and v: {q_class['B']} != {k_class['B']}")
    if dim != key_dim:
        raise ValueError(f"Embedding axes must be the same for q, k, and v: {q_class['D']} != {k_class['D']}")

    return _SplashPreparedLayout(
        q=q,
        k=k,
        v=v,
        q_class=q_class,
        k_class=k_class,
        v_class=v_class,
        QPos=QPos,
        KPos=KPos,
        batch=batch,
        heads=heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        physical_axes_q=_physical_axis_for_binning(q_class),
        physical_axes_k=_physical_axis_for_binning(k_class),
        physical_axes_v=_physical_axis_for_binning(v_class),
    )


def _prepare_splash_invocation_plan(
    *,
    layout: _SplashPreparedLayout,
    mask: Optional[Union[NamedArray, "AttentionMask"]],
    block_size: int,
    logits_soft_cap: float | None,
    attn_sink: Optional[NamedArray],
) -> _SplashInvocationPlan:
    if attn_sink is not None and not _SPLASH_KERNEL_SUPPORTS_SINKS:
        raise NotImplementedError(
            "Attention sinks are not supported by the installed Splash kernel. Update JAX to >= 0.7.2."
        )

    if attn_sink is not None:
        sinks, physical_axes_sink = _prepare_sinks_for_splash(attn_sink, layout.q_class, layout.physical_axes_q)
    else:
        sinks = None
        physical_axes_sink = None

    prefix_controls = _prepare_splash_prefix_controls(
        mask=mask,
        QPos=layout.QPos,
        KPos=layout.KPos,
        q_class=layout.q_class,
        physical_axes_q=layout.physical_axes_q,
        physical_axes_k=layout.physical_axes_k,
        kv_seq_len=layout.kv_seq_len,
    )
    segment_id_lowering = _lower_splash_attention_segment_ids(mask=mask, QPos=layout.QPos, KPos=layout.KPos)
    segment_run_controls = _prepare_splash_segment_run_controls(
        mask=mask,
        q_class=layout.q_class,
        physical_axes_q=layout.physical_axes_q,
    )

    mesh = hax.partitioning._get_mesh()
    if mesh is None or mesh.empty:
        raise NotImplementedError("Splash attention requires a non-empty mesh")
    head_shards = splash_partition_spec_shard_factor(layout.physical_axes_q[1], mesh)
    q_seq_shards = splash_partition_spec_shard_factor(layout.physical_axes_q[2], mesh)
    kv_seq_shards = splash_partition_spec_shard_factor(layout.physical_axes_k[2], mesh)

    if layout.physical_axes_k[2] is not None:
        raise NotImplementedError(
            "Splash attention does not support sharding the KV sequence dimension. "
            f"Got KV sequence spec: {layout.physical_axes_k[2]}"
        )

    block_sizes = splash_attention_block_sizes(
        q_seq_len=layout.q_seq_len,
        kv_seq_len=layout.kv_seq_len,
        q_seq_shards=q_seq_shards,
        kv_seq_shards=kv_seq_shards,
        max_block_size=block_size,
    )
    kernel_context = _SplashKernelContext(
        q_seq_len=layout.q_seq_len,
        kv_seq_len=layout.kv_seq_len,
        num_heads=layout.heads,
        block_sizes=block_sizes,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        physical_axes_q=layout.physical_axes_q,
        logits_soft_cap=logits_soft_cap,
    )

    return _SplashInvocationPlan(
        mesh=mesh,
        kernel_plan=_splash_kernel_plan(
            mask=mask,
            prefix_controls=prefix_controls,
            segment_run_controls=segment_run_controls,
            segment_id_lowering=segment_id_lowering,
            context=kernel_context,
            mesh=mesh,
        ),
        sinks=sinks,
        physical_axes_sink=physical_axes_sink,
    )


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
    attn_sink: Optional[NamedArray] = None,
) -> Optional[NamedArray]:
    if dropout != 0.0:
        if force_flash:
            raise NotImplementedError("Splash attention does not support dropout.")
        _warn_splash_fallback_once("Splash attention does not support dropout. Falling back to the reference.")
        return None

    if bias is not None:
        if force_flash:
            raise NotImplementedError("Splash attention does not support bias.")
        _warn_splash_fallback_once("Splash attention does not support bias. Falling back to the reference.")
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
            attn_sink=attn_sink,
        )
    except ImportError as e:
        if "pallas" not in str(e):
            raise
        if force_flash:
            raise ImportError("Could not import splash attention. You need to update your JAX to at least 0.7.2.")
        _warn_splash_fallback_once(
            "Could not import splash attention. You need to update your JAX to at least 0.7.2. "
            "Falling back to the reference implementation.",
        )
        return None
    except NotImplementedError as e:
        message = str(e)
        if force_flash:
            raise NotImplementedError(f"Could not use splash attention: {message}")
        logger.info("Could not use splash attention. Falling back to the reference implementation: %s", message)
        _warn_splash_fallback_once("Could not use splash attention. Falling back to the reference implementation.")
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
    attn_sink: Optional[NamedArray] = None,
) -> Optional[NamedArray]:
    # Splash attention requires BHSD format
    # We need to reshape the input to match this format
    if dropout != 0.0:
        raise NotImplementedError("Splash attention does not support dropout")

    if bias is not None:
        raise NotImplementedError("Splash attention does not support bias")

    # if attention_dtype is not None and attention_dtype != jnp.float32:
    #     warnings.warn("Splash attention only supports float32. Switching to float32.")

    # attention_dtype = jnp.float32

    layout = _prepare_splash_layout(
        QPos,
        KPos,
        Key,
        query,
        key,
        value,
        scaling_factor=scaling_factor,
        block_size=block_size,
    )
    invocation = _prepare_splash_invocation_plan(
        layout=layout,
        mask=mask,
        block_size=block_size or DEFAULT_SPLASH_BLOCK_SIZE,
        logits_soft_cap=logits_soft_cap,
        attn_sink=attn_sink,
    )
    return _run_prepared_tpu_splash_attention(
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
        scaling_factor=scaling_factor,
        layout=layout,
        invocation=invocation,
    )


def _run_prepared_tpu_splash_attention(
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
    scaling_factor: float,
    layout: _SplashPreparedLayout,
    invocation: _SplashInvocationPlan,
) -> NamedArray:
    """Run Splash Attention with a precomputed mask/kernel invocation plan."""
    if dropout != 0.0:
        raise NotImplementedError("Splash attention does not support dropout")

    if bias is not None:
        raise NotImplementedError("Splash attention does not support bias")

    scaled_query = query * scaling_factor
    q: jax.Array = _reshape_axes_for_bshd_bins(scaled_query, layout.q_class, output_order=list("BHSD")).array
    k = _reshape_axes_for_bshd_bins(key, layout.k_class, output_order=list("BHSD")).array
    v = _reshape_axes_for_bshd_bins(value, layout.v_class, output_order=list("BHSD")).array

    return _run_prepared_tpu_splash_attention_arrays(
        QPos,
        KPos,
        Key,
        query,
        key,
        value,
        q,
        k,
        v,
        mask=mask,
        bias=bias,
        dropout=dropout,
        inference=inference,
        prng=prng,
        attention_dtype=attention_dtype,
        precision=precision,
        layout=layout,
        invocation=invocation,
    )


def _run_prepared_tpu_splash_attention_arrays(
    QPos: AxisSelector,
    KPos: AxisSelection,
    Key: AxisSelector,
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: Optional[Union[NamedArray, "AttentionMask"]] = None,
    bias: Optional[NamedArray] = None,
    dropout: float = 0.0,
    inference: bool = False,
    *,
    prng: Optional[PRNGKeyArray] = None,
    attention_dtype: Optional[jnp.dtype] = None,
    precision: PrecisionLike = None,
    layout: _SplashPreparedLayout,
    invocation: _SplashInvocationPlan,
) -> NamedArray:
    kernel_plan = invocation.kernel_plan

    @functools.partial(
        shard_map,
        mesh=invocation.mesh,
        in_specs=(
            layout.physical_axes_q,
            layout.physical_axes_k,
            layout.physical_axes_v,
            kernel_plan.segment_id_lowering.segment_ids_axes,
            invocation.physical_axes_sink,
            kernel_plan.kernel_specs,
        ),
        out_specs=layout.physical_axes_q,
        check_rep=False,
    )
    def wrap_flash_attention(q, k, v, segment_ids, sinks, kernel):
        q = q.astype(attention_dtype)
        k = k.astype(attention_dtype)
        v = v.astype(attention_dtype)
        sink_in_axes = 0 if sinks is not None else None

        def call_kernel(q_b, k_b, v_b, segment_ids_for_batch, sink, kernel_b):
            if sink is None:
                return kernel_b(q_b, k_b, v_b, segment_ids=segment_ids_for_batch)
            return kernel_b(q_b, k_b, v_b, segment_ids=segment_ids_for_batch, sinks=sink)

        return jax.vmap(
            call_kernel,
            in_axes=(
                0,
                0,
                0,
                kernel_plan.segment_id_lowering.segment_batch_axis,
                sink_in_axes,
                kernel_plan.kernel_vmap_axis,
            ),
        )(q, k, v, segment_ids, sinks, kernel)

    attn_output = wrap_flash_attention(
        q,
        k,
        v,
        kernel_plan.segment_id_lowering.segment_ids,
        invocation.sinks,
        kernel_plan.kernel,
    )

    attn_output = haliax.named(attn_output, ("B", "H", "S", "D"))
    # the output shape is B, S_q, H_q, D_v. Right now we're requiring D_k == D_v
    # we can reshape it to match our expected output
    attn_output = _unflatten_bshd(attn_output, layout.q_class, layout.v_class)
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
        sliding_window: Optional sliding window size for attention masks.
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
    sliding_window: Optional[int] = None
    scaling_factor: Optional[float] = None
    logits_soft_cap: Optional[float] = None
    qk_norm: Optional[LayerNormConfigBase] = None
    gated: Literal["none", "headwise", "elementwise"] = "none"

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

    @property
    def GateSize(self) -> Axis:
        """Axis for the gate output size based on gating mode.

        For headwise gating, returns an axis of size 1 (one scalar per head).
        For elementwise gating, returns an axis of size head_size (one value per element).

        The axis is always named "gate_size" for consistency.
        """
        if self.gated == "none":
            raise ValueError("GateSize is only defined when gating is enabled")
        if self.gated == "headwise":
            return Axis("gate_size", 1)
        else:  # elementwise
            return Axis("gate_size", self.head_size)


class Attention(eqx.Module):
    """A multi-head attention layer that uses dot product attention.

    This is a general-purpose attention layer that can be used in various transformer architectures.
    It supports multi-head attention (MHA), multi-query attention (MQA), and grouped-query attention (GQA).

    Supports ROPE and QK normalization.
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
        if config.gated != "none":
            return GatedAttention.init(config, key=key)

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
            In=config.Embed,
            Out=(config.KVHeads, config.HeadSize),
            key=k_k,
            use_bias=use_bias,
            out_first=True,
        )
        v_proj = hnn.Linear.init(
            In=(config.Embed),
            Out=(config.KVHeads, config.HeadSize),
            key=k_v,
            use_bias=use_bias,
            out_first=True,
        )
        o_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize),
            Out=config.Embed,
            key=k_o,
            use_bias=use_output_bias,
            out_first=True,
        )

        q_norm = None
        k_norm = None
        if config.qk_norm is not None:
            q_norm = config.qk_norm.build(config.HeadSize)
            k_norm = config.qk_norm.build(config.HeadSize)

        # Build rotary embeddings once during initialization if configured
        rot_embs = config.rope.build(config.HeadSize) if config.rope is not None else None

        return Attention(config, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rot_embs)

    def empty_page_cache(self, spec: PageTableSpec, *, dtype) -> "KvPageCache":
        return KvPageCache.init(spec, self.config.KVHeads, self.config.HeadSize, dtype=dtype)

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

        q, k, v = self._compute_qkv(x, key=key_proj, pos_ids=pos_ids)

        # Reshape for attention kernels (convert embed → heads/head_size)
        q = q.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_head", "position", "head_size"))
        v = v.rearrange((..., "kv_head", "position", "head_size"))

        # Distinguish key sequence axis for attention
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        if self.config.sliding_window is not None and isinstance(mask, AttentionMask):
            mask = mask.with_sliding_window(self.config.sliding_window)

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

    @named_call
    def _compute_qkv(
        self,
        x: NamedArray,
        *,
        key,
        pos_ids: NamedArray | None = None,
    ) -> tuple[NamedArray, NamedArray, NamedArray]:
        """Project *x* to Q, K and V and apply all per-head processing."""
        key_q, key_k, key_v = maybe_rng_split(key, 3)

        q = self.q_proj(x, key=key_q)
        k = self.k_proj(x, key=key_k)
        v = self.v_proj(x, key=key_v)

        if self.config.qk_norm is not None:
            q = self.q_norm(q)  # type: ignore[misc]
            k = self.k_norm(k)  # type: ignore[misc]

        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"))
            q = self.rot_embs(q, pos_ids).astype(q.dtype)
            k = self.rot_embs(k, pos_ids).astype(k.dtype)

        return q, k, v


class GatedAttention(Attention):
    """Attention with learnable per-head gating (headwise or elementwise).

    Implements gated attention per https://github.com/qiuzh20/gated_attention.
    A separate linear projection produces gate values that are applied (after sigmoid)
    to the attention output before the output projection.
    """

    gate_proj: Optional[hnn.Linear] = None  # always set by init(); default for dataclass ordering

    @staticmethod
    def init(config: AttentionConfig, *, key) -> "GatedAttention":
        k_q, k_k, k_v, k_o, k_g = jrandom.split(key, 5)
        use_bias = config.use_bias
        use_output_bias = config.use_output_bias if config.use_output_bias is not None else use_bias

        q_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.QHeadsPerGroup, config.HeadSize),
            key=k_q,
            use_bias=use_bias,
            out_first=True,
        )
        k_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.HeadSize),
            key=k_k,
            use_bias=use_bias,
            out_first=True,
        )
        v_proj = hnn.Linear.init(
            In=(config.Embed),
            Out=(config.KVHeads, config.HeadSize),
            key=k_v,
            use_bias=use_bias,
            out_first=True,
        )
        o_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize),
            Out=config.Embed,
            key=k_o,
            use_bias=use_output_bias,
            out_first=True,
        )

        gate_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.QHeadsPerGroup, config.GateSize),
            key=k_g,
            use_bias=use_bias,
            out_first=True,
        )

        q_norm = None
        k_norm = None
        if config.qk_norm is not None:
            q_norm = config.qk_norm.build(config.HeadSize)
            k_norm = config.qk_norm.build(config.HeadSize)

        rot_embs = config.rope.build(config.HeadSize) if config.rope is not None else None

        return GatedAttention(config, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rot_embs, gate_proj)

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
        q, k, v = self._compute_qkv(x, key=key_proj, pos_ids=pos_ids)

        q = q.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_head", "position", "head_size"))
        v = v.rearrange((..., "kv_head", "position", "head_size"))
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        if self.config.sliding_window is not None and isinstance(mask, AttentionMask):
            mask = mask.with_sliding_window(self.config.sliding_window)

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

        assert self.gate_proj is not None
        gate = hax.nn.sigmoid(self.gate_proj(x))
        gate = gate.rename({"gate_size": "head_size"})
        attn_output = attn_output * gate

        attn_output = attn_output.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        return self.o_proj(attn_output, key=key_o)

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

        assert self.gate_proj is not None
        gate = hax.nn.sigmoid(self.gate_proj(x))
        gate = gate.rename({"gate_size": "head_size"})
        attn_tokens = attn_tokens * gate

        attn_output = attn_tokens.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        return self.o_proj(attn_output, key=key_o), kv_cache


@named_call
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
            logger.warning(
                "Failed to use TPU ragged paged attention. Falling back to reference",
                exc_info=True,
            )

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
    if tpu_ragged_paged_attention is None:
        msg = "TPU ragged paged attention kernel is unavailable."
        raise RuntimeError(msg)
    kernel = tpu_ragged_paged_attention

    # Usual shardmap dance
    # Ensure last dimension (head_size) is a multiple of 128 for Pallas kernels
    orig_head_size = q.axis_size("head_size")
    padded_head_size = ((orig_head_size + 127) // 128) * 128

    if padded_head_size != orig_head_size:
        pad_amount = padded_head_size - orig_head_size
        # Pad query on the head_size axis with zeros
        q_padded = hax.concatenate(
            "head_size",
            [q, hax.zeros_like(q["head_size", hax.ds(0, pad_amount)])],
        )
        # Pad kv_pages on the head_size axis with zeros to match
        kv_pages_padded = hax.concatenate(
            "head_size",
            [kv_pages, hax.zeros_like(kv_pages["head_size", hax.ds(0, pad_amount)])],
        )
    else:
        q_padded = q
        kv_pages_padded = kv_pages

    # The TPU kernel expects the second dimension of the query tensor to be the total number of query heads.
    q_flat = q_padded.flatten_axes(("kv_head", "q_heads_per_group"), "kv_head")
    if num_seqs.ndim == 0:
        this_num_seqs = num_seqs.reshape((1,))
    else:
        this_num_seqs = num_seqs

    # the INVALIDs make the TPU sad. mask them with 0:
    this_num_seqs = jnp.where(this_num_seqs < 0, 0, this_num_seqs)
    page_indices = hax.where(~is_valid(page_indices), 0, page_indices)
    kv_lens = hax.where(~is_valid(kv_lens), 0, kv_lens)

    sm_scale_array = jnp.asarray(sm_scale, dtype=q_flat.array.dtype)
    q_scaled = q_flat.array * sm_scale_array

    def _rpa_with_runtime_scale(
        q_arg: jax.Array,
        kv_pages_arg: jax.Array,
        kv_lens_arg: jax.Array,
        page_indices_arg: jax.Array,
        cu_q_lens_arg: jax.Array,
        num_seqs_arg: jax.Array,
    ) -> jax.Array:
        return kernel(
            q_arg,
            kv_pages_arg,
            kv_lens_arg,
            page_indices_arg,
            cu_q_lens_arg,
            num_seqs_arg,
            sm_scale=1.0,
            soft_cap=soft_cap,
        )

    o = shard_map(
        _rpa_with_runtime_scale,
        mesh=hax.partitioning._get_mesh(),
        in_specs=(
            haliax.partitioning.pspec_for_axis(q_flat.axes),
            haliax.partitioning.pspec_for_axis(kv_pages_padded.axes),
            haliax.partitioning.pspec_for_axis(kv_lens.axes),
            haliax.partitioning.pspec_for_axis(page_indices.axes),
            haliax.partitioning.pspec_for_axis(cu_q_lens.axes),
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
        q_scaled,
        kv_pages_padded.array,
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

    # If we padded head_size for the kernel, slice back to the original size
    if padded_head_size != orig_head_size:
        out = out["head_size", hax.ds(0, orig_head_size)]

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
                0,
                num_kv_blocks,
                _compute_attention_for_kv_block,
                (o_b, sum_exp_b, max_b),
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

    q_proj: Optional[hnn.Linear] = None
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
            assert self.q_proj is not None, "q_lora_rank not defined, but q_proj is missing."
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
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        key_proj, key_o = maybe_rng_split(key, 2)

        q, k, v = self._compute_qkv(x, key=key_proj, pos_ids=pos_ids)

        q = q.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_head", "position", "head_size"))
        v = v.rearrange((..., "kv_head", "position", "head_size"))

        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        if self.config.sliding_window is not None and isinstance(mask, AttentionMask):
            mask = mask.with_sliding_window(self.config.sliding_window)

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
            dropout=0.0,
            inference=True,
            prng=key,
            attn_sink=self.sinks,
        )

        attn_output = attn_output.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        attn_output = self.o_proj(attn_output, key=key_o)

        return attn_output
