# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Adapter to wire the grug model into the LmHeadModel API.

from typing import Any, Protocol

import equinox as eqx
import jax
import haliax as hax
from haliax import Axis, NamedArray
from jaxtyping import PRNGKeyArray, PyTree

from levanter.grug.attention import AttentionMask
from levanter.grug.model import activations as grug_activations
from levanter.grug.model import init_parameters
from levanter.layers.attention import AttentionMask as LevanterAttentionMask
from levanter.models.lm_model import LmHeadModel


class GrugConfigLike(Protocol):
    vocab_size: int
    max_seq_len: int
    hidden_dim: int


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


class GrugWrapper(LmHeadModel[Any]):
    """Minimal LmHeadModel wrapper around the standalone Grug transformer."""

    params: PyTree
    grug_config: GrugConfigLike
    init_fn: GrugInitFn = eqx.field(static=True)
    forward_fn: GrugForwardFn = eqx.field(static=True)

    @property
    def config(self) -> GrugConfigLike:
        return self.grug_config

    @property
    def Pos(self) -> Axis:
        max_seq_len = getattr(self.grug_config, "max_seq_len", None)
        if max_seq_len is None:
            raise AttributeError("grug_config must define max_seq_len.")
        return Axis("position", max_seq_len)

    @property
    def KeyPos(self) -> Axis:
        return self.Pos.alias("key_position")

    @property
    def Vocab(self) -> Axis:
        vocab_size = getattr(self.grug_config, "vocab_size", None)
        if vocab_size is None:
            raise AttributeError("grug_config must define vocab_size.")
        return Axis("vocab", vocab_size)

    @property
    def Embed(self) -> Axis:
        hidden_dim = getattr(self.grug_config, "hidden_dim", None)
        if hidden_dim is None:
            raise AttributeError("grug_config must define hidden_dim.")
        return Axis("embed", hidden_dim)

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: GrugConfigLike,
        *,
        key: PRNGKeyArray,
        init_fn: GrugInitFn | None = None,
        forward_fn: GrugForwardFn | None = None,
    ) -> "GrugWrapper":
        cfg = config
        chosen_init = init_fn or init_parameters
        params = chosen_init(cfg, key=key)
        return cls(
            params=params,
            grug_config=cfg,
            init_fn=chosen_init,
            forward_fn=forward_fn or grug_activations,
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
                causal_offset=0,
                segment_ids=segment_ids,
                sliding_window=attn_mask.sliding_window,
            )
        elif isinstance(attn_mask, NamedArray):
            raise NotImplementedError(
                "NamedArray attention masks are not supported by Grug (pass a Levanter AttentionMask)."
            )

        hidden = self.forward_fn(
            self.params,
            input_ids.array,
            self.grug_config,
            mask=mask,
        )

        # Map raw hidden states to a NamedArray with the existing axes plus Embed.
        axes = (*input_ids.axes, self.Embed)
        return hax.named(hidden, axes)

    def get_lm_head(self) -> NamedArray:
        output_proj = getattr(self.params, "output_proj", None)
        if output_proj is None:
            raise AttributeError("GrugWrapper params must have an output_proj field.")
        return hax.named(output_proj, (self.Embed, self.Vocab))

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "GrugWrapper":
        raise NotImplementedError("GrugWrapper does not yet support resizing the vocabulary.")
