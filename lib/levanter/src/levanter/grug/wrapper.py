# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Wrappers to wire up a grug model to the LmHeadModel API.
# Protocols are not grug-brained, but Grug must accept them

from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import haliax as hax
from haliax import Axis, NamedArray
from jaxtyping import PRNGKeyArray, PyTree

from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmHeadModel

from .config import AttentionRuntimeConfig
from .model import forward, init_parameters


class GrugConfigLike(Protocol):
    vocab_size: int
    max_seq_len: int


class GrugForwardFn(Protocol):
    def __call__(
        self,
        params: PyTree,
        tokens: jax.Array,
        cfg: GrugConfigLike,
        runtime_cfg: AttentionRuntimeConfig,
        *,
        mask: jax.Array | None = None,
        causal: bool = True,
    ) -> jax.Array: ...


class GrugInitFn(Protocol):
    def __call__(self, cfg: GrugConfigLike, *, key: PRNGKeyArray) -> PyTree: ...


class GrugWrapper(LmHeadModel[Any]):
    """Minimal LmHeadModel wrapper around the standalone Grug transformer."""

    params: PyTree
    grug_config: GrugConfigLike
    runtime_cfg: AttentionRuntimeConfig
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
        # We return an Embed axis sized to the vocab so we can use an identity LM head.
        return Axis("embed", self.Vocab.size)

    @classmethod
    def init(
        cls,
        Vocab: Axis,
        config: GrugConfigLike,
        *,
        key: PRNGKeyArray,
        init_fn: GrugInitFn | None = None,
        forward_fn: GrugForwardFn | None = None,
        runtime_cfg: AttentionRuntimeConfig | None = None,
    ) -> "GrugWrapper":
        cfg = config
        chosen_init = init_fn or init_parameters
        params = chosen_init(cfg, key=key)
        runtime = runtime_cfg or AttentionRuntimeConfig()
        return cls(
            params=params,
            grug_config=cfg,
            runtime_cfg=runtime,
            init_fn=chosen_init,
            forward_fn=forward_fn or forward,
        )

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        del key, pos_ids  # unused in this lightweight wrapper
        mask_array: jax.Array | None = None
        if isinstance(attn_mask, AttentionMask):
            materialized = attn_mask.materialize(self.Pos, self.KeyPos)
            mask_array = materialized.array if materialized is not None else None
        elif isinstance(attn_mask, NamedArray):
            mask_array = attn_mask.array

        logits = self.forward_fn(
            self.params,
            input_ids.array,
            self.grug_config,
            self.runtime_cfg,
            mask=mask_array,
            causal=True,
        )

        # Map raw logits to a NamedArray with the existing axes plus Vocab.
        axes = (*input_ids.axes, self.Vocab)
        return hax.named(logits, axes)

    def get_lm_head(self) -> NamedArray:
        eye = jnp.eye(self.Vocab.size, dtype=jnp.float32)
        return hax.named(eye, (self.Embed, self.Vocab))

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "GrugWrapper":
        raise NotImplementedError("GrugWrapper does not yet support resizing the vocabulary.")
