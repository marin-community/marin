# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

import haliax as hax
from haliax import Axis, AxisSelector, NamedArray, NamedOrNumeric

from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample


@register_dataclass
@dataclass(frozen=True)
class GrugLmExample:
    """A grug-conformant LM example that stores raw JAX arrays."""

    tokens: jax.Array
    loss_weight: jax.Array
    attn_mask: GrugAttentionMask = GrugAttentionMask.causal()

    @staticmethod
    def causal(
        tokens: jax.Array,
        *,
        loss_weight: jax.Array | None = None,
        ignore_id: int | None = None,
        eos_id: int | None = None,
        segment_ids: jax.Array | None = None,
        sliding_window: int | None = None,
        block_cross_document_attention: bool = True,
    ) -> "GrugLmExample":
        if tokens.ndim != 1:
            raise ValueError("tokens must be a 1D array")

        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("tokens must be an integer array")

        seq_len = tokens.shape[0]
        causal_loss_mask = GrugLmExample.causal_loss_mask(seq_len)

        if loss_weight is not None:
            dtype = jnp.result_type(loss_weight.dtype, jnp.float32)
            loss_weight = loss_weight.astype(dtype) * causal_loss_mask.astype(dtype)
        else:
            dtype = jnp.float32
            loss_weight = causal_loss_mask.astype(dtype)

        if ignore_id is not None:
            ignore_mask = jnp.roll(tokens, -1) != ignore_id
            ignore_mask = ignore_mask.astype(loss_weight.dtype)
            loss_weight = loss_weight * ignore_mask

        loss_weight = loss_weight.astype(dtype)

        attn_mask = GrugAttentionMask.causal(sliding_window=sliding_window)
        if block_cross_document_attention:
            if eos_id is not None and segment_ids is None:
                eos_mask = jnp.roll(tokens, 1) == eos_id
                eos_mask = eos_mask.at[0].set(False).astype(jnp.int32)
                segment_ids = jnp.cumsum(eos_mask, axis=0)
                attn_mask = attn_mask.with_segment_ids(segment_ids)
            elif segment_ids is not None:
                attn_mask = attn_mask.with_segment_ids(segment_ids)

        return GrugLmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=attn_mask)

    @staticmethod
    def from_prompt_and_completion(
        tokens: jax.Array,
        prompt_length: NamedOrNumeric,
        *,
        ignore_id: int | None = None,
        all_causal: bool = True,
        sliding_window: int | None = None,
    ) -> "GrugLmExample":
        if all_causal:
            attn_mask = GrugAttentionMask.causal(sliding_window=sliding_window)
        else:
            raise NotImplementedError("Not implemented yet")

        loss_weight = GrugLmExample.causal_loss_mask(tokens.shape[0], prompt_length=prompt_length).astype(jnp.float32)

        if ignore_id is not None:
            ignore_mask = jnp.roll(tokens, -1) != ignore_id
            loss_weight = loss_weight * ignore_mask.astype(loss_weight.dtype)

        return GrugLmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=attn_mask)

    @staticmethod
    def causal_loss_mask(seq_len: int, prompt_length: NamedOrNumeric | None = None) -> jax.Array:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        loss_weight = jnp.arange(seq_len) < (seq_len - 1)
        if prompt_length is not None:
            prompt_mask = jnp.arange(seq_len) >= (prompt_length - 1)
            loss_weight = jnp.logical_and(loss_weight, prompt_mask)

        return loss_weight


def grug_attention_mask_from_named(mask: AttentionMask) -> GrugAttentionMask:
    if mask.explicit_mask is not None:
        raise NotImplementedError("Explicit attention masks are not supported by GrugAttentionMask.")

    if mask.causal_offset is not None:
        offset = jnp.asarray(mask.causal_offset.array)
        if offset.ndim != 0 or int(offset) != 0:
            raise NotImplementedError("Non-zero causal offsets are not supported by GrugAttentionMask.")

    segment_ids: tuple[jax.Array, jax.Array] | None = None
    if mask.segment_ids is not None:
        q_seg, kv_seg = mask.segment_ids
        segment_ids = (q_seg.array, kv_seg.array)

    return GrugAttentionMask(
        is_causal=mask.is_causal,
        segment_ids=segment_ids,
        sliding_window=mask.sliding_window,
    )


def _resolve_batch_axis(batch_axis: AxisSelector | None, batch_size: int) -> Axis:
    if batch_axis is None:
        return Axis("batch", batch_size)
    if isinstance(batch_axis, Axis):
        if batch_axis.size != batch_size:
            raise ValueError(f"Batch axis size ({batch_axis.size}) must match batched array size ({batch_size}).")
        return batch_axis
    if isinstance(batch_axis, str):
        return Axis(batch_axis, batch_size)
    raise TypeError(f"Unsupported batch axis selector: {batch_axis!r}")


def named_attention_mask_from_grug(
    mask: GrugAttentionMask,
    Pos: Axis,
    batch_axis: AxisSelector | None = None,
) -> AttentionMask:
    KeyPos = Pos.alias("key_position")

    segment_ids: tuple[NamedArray, NamedArray] | None = None
    if mask.segment_ids is not None:
        q_seg, kv_seg = mask.segment_ids

        if q_seg.ndim == 1 and kv_seg.ndim == 1:
            segment_ids = (
                hax.named(q_seg, Pos),
                hax.named(kv_seg, KeyPos),
            )
        elif q_seg.ndim == 2 and kv_seg.ndim == 2:
            Batch = _resolve_batch_axis(batch_axis, q_seg.shape[0])
            if q_seg.shape != (Batch.size, Pos.size):
                raise ValueError(
                    f"Query segment_ids shape {q_seg.shape} must match {(Batch.size, Pos.size)} for axes (Batch, Pos)."
                )
            if kv_seg.shape != (Batch.size, KeyPos.size):
                raise ValueError(
                    f"KV segment_ids shape {kv_seg.shape} must match {(Batch.size, KeyPos.size)} for axes (Batch, KeyPos)."
                )
            segment_ids = (
                hax.named(q_seg, (Batch, Pos)),
                hax.named(kv_seg, (Batch, KeyPos)),
            )
        else:
            raise ValueError(
                f"segment_ids must be both rank-1 or both rank-2, got ranks {q_seg.ndim} and {kv_seg.ndim}"
            )

    return AttentionMask(
        is_causal=mask.is_causal,
        segment_ids=segment_ids,
        sliding_window=mask.sliding_window,
    )


def grug_lm_example_from_named(example: LmExample) -> GrugLmExample:
    if isinstance(example.attn_mask, NamedArray):
        raise NotImplementedError("NamedArray attention masks are not supported for Grug conversion.")

    return GrugLmExample(
        tokens=example.tokens.array,
        loss_weight=example.loss_weight.array,
        attn_mask=grug_attention_mask_from_named(example.attn_mask),
    )


def named_lm_example_from_grug(
    example: GrugLmExample,
    Pos: Axis,
    batch_axis: AxisSelector | None = None,
) -> LmExample:
    if example.tokens.shape != example.loss_weight.shape:
        raise ValueError(
            f"GrugLmExample token shape {example.tokens.shape} must match loss_weight shape {example.loss_weight.shape}."
        )

    if example.tokens.ndim == 1:
        if example.tokens.shape[0] != Pos.size:
            raise ValueError(
                f"GrugLmExample token length ({example.tokens.shape[0]}) must match Pos axis size ({Pos.size})"
            )
        token_axes: tuple[Axis, ...] = (Pos,)
        resolved_batch_axis: AxisSelector | None = None
    elif example.tokens.ndim == 2:
        Batch = _resolve_batch_axis(batch_axis, example.tokens.shape[0])
        if example.tokens.shape[1] != Pos.size:
            raise ValueError(
                f"GrugLmExample position length ({example.tokens.shape[1]}) must match Pos axis size ({Pos.size})"
            )
        token_axes = (Batch, Pos)
        resolved_batch_axis = Batch
    else:
        raise ValueError(f"GrugLmExample tokens must be rank-1 or rank-2, got rank={example.tokens.ndim}")

    return LmExample(
        tokens=hax.named(example.tokens, token_axes),
        loss_weight=hax.named(example.loss_weight, token_axes),
        attn_mask=named_attention_mask_from_grug(example.attn_mask, Pos, batch_axis=resolved_batch_axis),
    )
