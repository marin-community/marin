# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Adapter to wire the grug model into the LmHeadModel API.

from typing import Any, Protocol, cast

import equinox as eqx
import jax
import haliax as hax
import jax.numpy as jnp
from haliax import Axis, NamedArray
from jaxtyping import PRNGKeyArray, PyTree

from levanter.grug.attention import AttentionMask
from levanter.grug.model import activations as grug_activations
from levanter.grug.model import init_parameters
from levanter.grug.model import loss_fn as grug_loss_fn
from levanter.layers.attention import AttentionMask as LevanterAttentionMask
from levanter.models.lm_model import LmExample, LmHeadModel


class GrugConfigLike(Protocol):
    vocab_size: int
    max_seq_len: int
    hidden_dim: int
    cross_entropy_block_size: int | None = None


class GrugForwardFn(Protocol):
    def __call__(
        self,
        params: PyTree,
        tokens: jax.Array,
        cfg: GrugConfigLike,
        *,
        mask: AttentionMask | jax.Array | None = None,
    ) -> jax.Array: ...


class GrugInitFn(Protocol):
    def __call__(self, cfg: GrugConfigLike, *, key: PRNGKeyArray) -> PyTree: ...


class GrugLmHeadFn(Protocol):
    def __call__(self, params: PyTree) -> jax.Array: ...


def _default_lm_head_fn(params: PyTree) -> jax.Array:
    return params.output_proj  # type: ignore[attr-defined]


def _mask_from_levanter(attn_mask: LevanterAttentionMask | NamedArray | None) -> AttentionMask | jax.Array | None:
    mask: AttentionMask | jax.Array | None = None
    if isinstance(attn_mask, LevanterAttentionMask):
        if attn_mask.explicit_mask is not None:
            raise NotImplementedError("Grug does not support explicit masks yet.")
        if attn_mask.causal_offset is not None:
            raise NotImplementedError("Grug does not support causal offsets yet.")
        segment_ids = None
        if attn_mask.segment_ids is not None:
            q_seg, kv_seg = attn_mask.segment_ids
            segment_ids = (q_seg.array, kv_seg.array)
        mask = AttentionMask(
            is_causal=attn_mask.is_causal,
            segment_ids=segment_ids,
            sliding_window=attn_mask.sliding_window,
        )
    elif isinstance(attn_mask, NamedArray):
        raise NotImplementedError(
            "NamedArray attention masks are not supported by Grug (pass a Levanter AttentionMask)."
        )
    return mask


class GrugWrapper(LmHeadModel[Any]):
    """Minimal LmHeadModel wrapper around the standalone Grug transformer."""

    params: PyTree
    grug_config: GrugConfigLike
    init_fn: GrugInitFn = eqx.field(static=True)
    forward_fn: GrugForwardFn = eqx.field(static=True)
    lm_head_fn: GrugLmHeadFn = eqx.field(static=True, default=_default_lm_head_fn)

    @property
    def config(self) -> GrugConfigLike:
        return self.grug_config

    @property
    def Pos(self) -> Axis:
        return Axis("position", self.grug_config.max_seq_len)

    @property
    def KeyPos(self) -> Axis:
        return self.Pos.alias("key_position")

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.grug_config.vocab_size)

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.grug_config.hidden_dim)

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: GrugConfigLike,
        *,
        key: PRNGKeyArray,
        init_fn: GrugInitFn | None = None,
        forward_fn: GrugForwardFn | None = None,
        lm_head_fn: GrugLmHeadFn | None = None,
    ) -> "GrugWrapper":
        cfg = config
        chosen_init = init_fn or init_parameters
        params = chosen_init(cfg, key=key)
        return cls(
            params=params,
            grug_config=cfg,
            init_fn=chosen_init,
            forward_fn=forward_fn or grug_activations,
            lm_head_fn=lm_head_fn or _default_lm_head_fn,
        )

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: LevanterAttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        del key, pos_ids  # unused in this lightweight wrapper
        mask = _mask_from_levanter(attn_mask)

        hidden = self.forward_fn(
            self.params,
            input_ids.array,
            self.grug_config,
            mask=mask,
        )

        # Map raw hidden states to a NamedArray with the existing axes plus Embed.
        axes = (*input_ids.axes, self.Embed)
        return hax.named(hidden, axes)

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: hax.ReductionFunction | None = cast(hax.ReductionFunction | None, hax.mean),
        reduction_axis: hax.AxisSelection | None = None,
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype | None = jnp.float32,
        logit_soft_cap: float | None = None,
    ) -> jnp.ndarray | NamedArray:
        """Override to use grug's blockwise loss (avoids materializing full logits)."""
        # NOTE: this wrapper is intentionally minimal; grug core currently doesn't use PRNGs.
        assert logit_soft_cap is None, "logit_soft_cap is not supported by GrugWrapper.compute_next_token_loss"
        del key

        # LmExample-ish protocol: expects `.tokens`, `.loss_weight`, `.attn_mask`.
        tokens = example.tokens
        loss_weight = example.loss_weight
        attn_mask = example.attn_mask

        mask = _mask_from_levanter(attn_mask)
        dtype = jnp.float32 if loss_dtype is None else loss_dtype

        if reduction is None:
            per_pos, _ = grug_loss_fn(
                self.params,
                tokens.array,
                loss_weight.array,
                self.grug_config,
                mask=mask,
                reduction="none",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )
            return hax.named(per_pos, tokens.axes)

        # Fast path: scalar mean/sum reduction over all axes.
        if reduction_axis is None and reduction is hax.mean:
            loss, _ = grug_loss_fn(
                self.params,
                tokens.array,
                loss_weight.array,
                self.grug_config,
                mask=mask,
                reduction="mean",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )
            return loss
        if reduction_axis is None and reduction is hax.sum:
            loss, _ = grug_loss_fn(
                self.params,
                tokens.array,
                loss_weight.array,
                self.grug_config,
                mask=mask,
                reduction="sum",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )
            return loss

        per_pos, _ = grug_loss_fn(
            self.params,
            tokens.array,
            loss_weight.array,
            self.grug_config,
            mask=mask,
            reduction="none",
            logsumexp_weight=logsumexp_weight,
            loss_dtype=dtype,
        )
        loss = hax.named(per_pos, tokens.axes)

        return reduction(loss, axis=reduction_axis)

    def get_lm_head(self) -> NamedArray:
        return hax.named(self.lm_head_fn(self.params), (self.Embed, self.Vocab))

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "GrugWrapper":
        raise NotImplementedError("GrugWrapper does not yet support resizing the vocabulary.")
