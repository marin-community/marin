import functools
import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.experimental.pallas.ops.tpu.splash_attention import SegmentIds
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec
from jaxtyping import PRNGKeyArray

import haliax
import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSelection, AxisSelector, NamedArray, axis_name
from haliax.jax_utils import maybe_rng_split, named_call
from haliax.nn.attention import causal_mask, combine_masks_and, combine_masks_or
from haliax.nn.normalization import LayerNormBase
from haliax.partitioning import pspec_for_axis
from haliax.types import PrecisionLike

from .normalization import LayerNormConfigBase
from .rotary import RotaryEmbeddings, RotaryEmbeddingsConfig


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


def simple_attention_with_dropout(
    QPos: Axis,
    KPos: Axis,
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
    from transformer_engine.jax.attention import fused_attn  # noqa: F401
    from transformer_engine.jax.attention import AttnBiasType, AttnMaskType, QKVLayout  # noqa: F401

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
        if mask.causal():
            attn_mask_type = AttnMaskType.CAUSAL_MASK

            fused_attn_mask = mask.materialize(QPos, KPos)

            assert (
                fused_attn_mask is not None
            ), "If AttentionMask is causal, the materialized array should never be None. Something is wrong."

            fused_attn_mask = fused_attn_mask.array
            fused_attn_mask = jnp.dstack([fused_attn_mask] * batch_size)

        else:
            raise NotImplementedError(
                "Non-Causal masks are not implemented for flash attention. Please pass an AttentionMask object"
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


def _materialize_segment_mask(segment_ids, QPos, KPos, q_slice, k_slice) -> NamedArray:
    """
    Make a segment mask for attention. This is a mask that prevents attention between different segments.
    """
    kv_segment_ids = segment_ids.rename({QPos: KPos})[KPos, k_slice]
    q_segment_ids = segment_ids[QPos, q_slice]
    sub_KPos = kv_segment_ids.resolve_axis(KPos.name)

    return q_segment_ids.broadcast_axis(sub_KPos) == kv_segment_ids


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

    is_causal: bool = eqx.field(static=True)
    explicit_mask: Optional[NamedArray] = None
    segment_ids: Optional[NamedArray] = None
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
            causal = causal_mask(QPos.resize(q_slice.size), KPos.resize(k_slice.size), q_slice.start, k_slice.start)
        else:
            causal = None

        if self.explicit_mask is not None:
            explicit = self.explicit_mask[QPos, q_slice, KPos, k_slice]
        else:
            explicit = None

        mask = combine_masks_and(causal, explicit)

        if self.segment_ids is not None:
            segment_mask = _materialize_segment_mask(self.segment_ids, QPos, KPos, q_slice, k_slice)
            mask = combine_masks_and(mask, segment_mask)

        return mask

    @staticmethod
    def causal() -> "AttentionMask":
        return AttentionMask(is_causal=True)

    @staticmethod
    def explicit(mask: NamedArray) -> "AttentionMask":
        return AttentionMask(is_causal=False, explicit_mask=mask)

    def with_segment_ids(self, segment_ids: NamedArray) -> "AttentionMask":
        return AttentionMask(is_causal=self.is_causal, explicit_mask=self.explicit_mask, segment_ids=segment_ids)

    def __and__(self, other) -> "AttentionMask":
        is_causal = self.is_causal or other.is_causal
        explicit_mask = combine_masks_and(self.explicit_mask, other.explicit_mask)
        segment_ids = self._check_for_same_segment_ids(other)

        return AttentionMask(is_causal=is_causal, explicit_mask=explicit_mask, segment_ids=segment_ids)

    def __or__(self, other) -> "AttentionMask":
        is_causal = self.is_causal and other.is_causal
        explicit_mask = combine_masks_or(self.explicit_mask, other.explicit_mask)
        segment_ids = self._check_for_same_segment_ids(other)
        return AttentionMask(is_causal=is_causal, explicit_mask=explicit_mask, segment_ids=segment_ids)

    def _check_for_same_segment_ids(self, other):
        if self.segment_ids is not None and other.segment_ids is not None:
            # only one segment mask is allowed
            # b/c we might do this in jit, we use eqx.error_if
            # in theory we can do this one by just assigning unique ids to each unique pair...
            # (but i don't really anticipate needing this)
            segment_ids = eqx.error_if(
                self.segment_ids,
                not haliax.all(self.segment_ids == other.segment_ids),
                "Only one segment mask is allowed",
            )
        elif self.segment_ids is not None:
            segment_ids = self.segment_ids
        else:
            segment_ids = other.segment_ids
        return segment_ids


@overload
def materialize_mask(
    mask: NamedArray | AttentionMask,
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> NamedArray:
    ...


@overload
def materialize_mask(
    mask: Optional[NamedArray | AttentionMask],
    QPos: Axis,
    KPos: Axis,
    q_slice: Optional[haliax.dslice] = None,
    k_slice: Optional[haliax.dslice] = None,
) -> Optional[NamedArray]:
    ...


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
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel, splash_attention_mask

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

    # segment_ids
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    physical_axes_segments = pspec_for_axis(segment_ids.axes) if segment_ids is not None else None
    # do we have a batch axis in segment_ids? (needed for vmap below)
    if segment_ids is not None:
        index_of_seq_dim = segment_ids.axes.index(QPos)
        other_indices = [i for i in range(len(segment_ids.axes)) if i != index_of_seq_dim]
        if len(other_indices) > 1:
            raise NotImplementedError(
                f"Only one batch axis is supported in segment_ids right now (got {segment_ids.axes})"
            )
        elif len(other_indices) == 1:
            segment_batch_axis = other_indices[0]
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

        if mask.segment_ids is not None:
            # for now only support self attention
            segment_ids = segment_ids.array
            segment_ids = SegmentIds(segment_ids, segment_ids)

        if mask is None:
            base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))
        elif isinstance(mask, AttentionMask):
            if mask.is_causal:
                base_mask = splash_attention_mask.CausalMask(shape=(Sq, Sk))
            else:
                base_mask = splash_attention_mask.FullMask(_shape=(Sq, Sk))
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
        return Axis("kv_heads", self.num_kv_heads)

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


class Attention(eqx.Module):
    """A multi-head attention layer that uses dot product attention.

    This is a general-purpose attention layer that can be used in various transformer architectures.
    It supports multi-head attention (MHA), multi-query attention (MQA), and grouped-query attention (GQA).
    """

    config: AttentionConfig = eqx.field(static=True)
    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    o_proj: hnn.Linear
    q_norm: Optional[LayerNormBase] = None
    k_norm: Optional[LayerNormBase] = None
    rot_embs: Optional[RotaryEmbeddings] = eqx.field(default=None)

    @staticmethod
    def init(config: AttentionConfig, *, key) -> "Attention":
        use_bias = config.use_bias
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
            In=(config.Heads, config.HeadSize), Out=config.Embed, key=k_o, use_bias=use_bias, out_first=True
        )

        q_norm = None
        k_norm = None
        if config.qk_norm is not None:
            q_norm = config.qk_norm.build(config.HeadSize)
            k_norm = config.qk_norm.build(config.HeadSize)

        # Build rotary embeddings once during initialization if configured
        rot_embs = config.rope.build(config.HeadSize) if config.rope is not None else None

        return Attention(config, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rot_embs)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        # Project to query, key, value
        q_proj = self.q_proj(x, key=key_q)
        k_proj = self.k_proj(x, key=key_k)
        v = self.v_proj(x, key=key_v)

        # Apply QK normalization if enabled
        if self.config.qk_norm is not None:
            q = self.q_norm(q_proj)  # type: ignore[misc]
            k = self.k_norm(k_proj)  # type: ignore[misc]
        else:
            q = q_proj
            k = k_proj

        # Reshape for attention
        q = q.rearrange((..., "kv_heads", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_heads", "position", "head_size"))
        v = v.rearrange((..., "kv_heads", "position", "head_size"))

        # Apply rotary position embeddings if configured
        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"), dtype=jnp.int32)
            q = self.rot_embs(q, pos_ids)
            k = self.rot_embs(k, pos_ids)

        # Rename position axis for attention
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
            dropout=0.0,  # TODO: support dropout
            inference=True,  # TODO: support training
            prng=key,
        )

        # Flatten heads and apply output projection
        attn_output = attn_output.flatten_axes(("kv_heads", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        attn_output = self.o_proj(attn_output, key=key_o)

        return attn_output
