# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass

import haliax as hax
from haliax import Axis, AxisSelector, NamedArray, NamedOrNumeric

from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmExample


LOSS_IGNORE_LABEL = 0


@dataclass(frozen=True)
class LossLabelSpan:
    """An exclusive typed span used to evaluate per-span-type LM loss.

    The span is over next-token loss positions, not raw token positions:
    `[start, end)` labels losses for predicting `tokens[start + 1]` through
    `tokens[end]`. Spans are intentionally exclusive so each loss position has
    one semantic type before aggregate metrics roll those types up.
    """

    start: int
    end: int
    label: int


@dataclass(frozen=True)
class LossLabelSpec:
    """Names exclusive loss labels and defines metric rollups.

    `id_to_name` names the leaf span types stored in `LabeledLmExample.loss_labels`.
    `aggregates` maps metric names to one or more leaf label ids, so callers can
    report both specific span types and rollups such as assistant = assistant
    text plus assistant tool calls. If aggregates is omitted, each non-ignored
    label id gets its own metric.
    """

    id_to_name: Mapping[int, str]
    aggregates: Mapping[str, Sequence[int]] | None = None
    dont_score_label: int = LOSS_IGNORE_LABEL

    def __post_init__(self):
        for label_id, name in self.id_to_name.items():
            if not isinstance(label_id, int):
                raise TypeError(f"label id must be an int, got {label_id!r}")
            if not isinstance(name, str):
                raise TypeError(f"label name for id {label_id} must be a str, got {name!r}")
        if len(set(self.id_to_name.values())) != len(self.id_to_name):
            raise ValueError("label names must be unique")

        for name, label_ids in self._aggregate_mapping().items():
            if not isinstance(name, str):
                raise TypeError(f"aggregate name must be a str, got {name!r}")
            if not label_ids:
                raise ValueError(f"aggregate {name!r} must include at least one label id")
            if self.dont_score_label in label_ids:
                raise ValueError(f"aggregate {name!r} includes dont_score_label={self.dont_score_label}")
            for label_id in label_ids:
                if not isinstance(label_id, int):
                    raise TypeError(f"aggregate {name!r} label id must be an int, got {label_id!r}")
                if label_id not in self.id_to_name:
                    raise ValueError(f"aggregate {name!r} references unknown label id {label_id}")

    def _aggregate_mapping(self) -> Mapping[str, Sequence[int]]:
        if self.aggregates is not None:
            return self.aggregates
        return {
            label_name: (label_id,)
            for label_id, label_name in self.id_to_name.items()
            if label_id != self.dont_score_label
        }

    @property
    def aggregate_names(self) -> tuple[str, ...]:
        return tuple(self._aggregate_mapping().keys())

    @property
    def aggregate_label_ids(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(label_ids) for label_ids in self._aggregate_mapping().values())


def loss_labels_from_spans(
    seq_len: int,
    spans: Sequence[LossLabelSpan],
    *,
    default_label: int = LOSS_IGNORE_LABEL,
) -> jax.Array:
    labels = np.full(seq_len, default_label, dtype=np.int32)
    claimed = np.zeros(seq_len, dtype=bool)

    for span in spans:
        if span.start < 0 or span.end > seq_len or span.start >= span.end:
            raise ValueError(f"Invalid label span [{span.start}, {span.end}) for sequence length {seq_len}")
        if np.any(claimed[span.start : span.end]):
            raise ValueError(f"Label span [{span.start}, {span.end}) overlaps a previous span")

        labels[span.start : span.end] = span.label
        claimed[span.start : span.end] = True

    return jnp.asarray(labels)


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
        max_segments: int | None = None,
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

        # Prepacked datasets mark padding positions with segment id -1. A position whose
        # successor is padding predicts a pad token, so it must never contribute loss --
        # otherwise the (arbitrary) padding value would leak into the objective.
        if segment_ids is not None:
            predicts_real_token = (jnp.roll(segment_ids, -1) >= 0).astype(dtype)
            loss_weight = loss_weight * predicts_real_token

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
                attn_mask = attn_mask.with_segment_ids(segment_ids, max_segments=max_segments)
            elif segment_ids is not None:
                attn_mask = attn_mask.with_segment_ids(segment_ids, max_segments=max_segments)

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


@register_dataclass
@dataclass(frozen=True)
class LabeledLmExample:
    """A grug-conformant LM example with exclusive labels for loss evaluation.

    Use this when an eval needs to report loss by token or span type, such as
    assistant text, tool calls, observations, or derived answer spans.
    `loss_labels[i]` labels the loss for predicting `tokens[i + 1]`; the final
    position should normally use `LOSS_IGNORE_LABEL` because it has no next
    token to predict.
    """

    tokens: jax.Array
    loss_labels: jax.Array
    attn_mask: GrugAttentionMask = GrugAttentionMask.causal()


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


def _resolve_token_axes(
    tokens: jax.Array,
    Pos: Axis,
    batch_axis: AxisSelector | None,
    *,
    kind: str,
) -> tuple[tuple[Axis, ...], AxisSelector | None]:
    """Resolve axes for a rank-1 ``(Pos,)`` or rank-2 ``(Batch, Pos)`` example array.

    ``kind`` names the example type (e.g. ``"GrugLmExample"``) so validation errors point at
    the caller's data. Returns the token axes and the resolved batch axis (``None`` when rank-1).
    """
    if tokens.ndim == 1:
        if tokens.shape[0] != Pos.size:
            raise ValueError(f"{kind} token length ({tokens.shape[0]}) must match Pos axis size ({Pos.size})")
        return (Pos,), None
    if tokens.ndim == 2:
        Batch = _resolve_batch_axis(batch_axis, tokens.shape[0])
        if tokens.shape[1] != Pos.size:
            raise ValueError(f"{kind} position length ({tokens.shape[1]}) must match Pos axis size ({Pos.size})")
        return (Batch, Pos), Batch
    raise ValueError(f"{kind} tokens must be rank-1 or rank-2, got rank={tokens.ndim}")


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


def named_lm_example_from_labeled(
    example: LabeledLmExample,
    Pos: Axis,
    batch_axis: AxisSelector | None = None,
    *,
    scored_labels: Sequence[int] | None = None,
) -> tuple[LmExample, NamedArray]:
    if example.tokens.shape != example.loss_labels.shape:
        raise ValueError(
            f"LabeledLmExample token shape {example.tokens.shape} must match loss_labels shape "
            f"{example.loss_labels.shape}."
        )

    token_axes, resolved_batch_axis = _resolve_token_axes(example.tokens, Pos, batch_axis, kind="LabeledLmExample")

    labels = hax.named(example.loss_labels.astype(jnp.int32), token_axes)
    if scored_labels is None:
        loss_weight_array = example.loss_labels != LOSS_IGNORE_LABEL
    else:
        label_ids = jnp.asarray(tuple(scored_labels), dtype=example.loss_labels.dtype)
        if label_ids.size == 0:
            raise ValueError("scored_labels must contain at least one label id")
        loss_weight_array = jnp.isin(example.loss_labels, label_ids)

    lm_example = LmExample(
        tokens=hax.named(example.tokens, token_axes),
        loss_weight=hax.named(loss_weight_array.astype(jnp.float32), token_axes),
        attn_mask=named_attention_mask_from_grug(example.attn_mask, Pos, batch_axis=resolved_batch_axis),
    )
    return lm_example, labels


def named_lm_example_from_grug(
    example: GrugLmExample,
    Pos: Axis,
    batch_axis: AxisSelector | None = None,
) -> LmExample:
    if example.tokens.shape != example.loss_weight.shape:
        raise ValueError(
            f"GrugLmExample token shape {example.tokens.shape} must match loss_weight shape {example.loss_weight.shape}."
        )

    token_axes, resolved_batch_axis = _resolve_token_axes(example.tokens, Pos, batch_axis, kind="GrugLmExample")

    return LmExample(
        tokens=hax.named(example.tokens, token_axes),
        loss_weight=hax.named(example.loss_weight, token_axes),
        attn_mask=named_attention_mask_from_grug(example.attn_mask, Pos, batch_axis=resolved_batch_axis),
    )
