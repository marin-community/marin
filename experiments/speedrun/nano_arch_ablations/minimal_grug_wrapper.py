# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Protocol, cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray
from haliax.state_dict import ModuleWithStateDictSerialization
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiments.llama import llama3_tokenizer_vocab_size
from levanter.grug.attention import AttentionMask
from levanter.layers.attention import AttentionMask as LevanterAttentionMask
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from marin.speedrun.speedrun import SpeedrunConfig, default_speedrun


class GrugConfigLike(Protocol):
    vocab_size: int
    max_seq_len: int
    hidden_dim: int


class GrugLossFn(Protocol):
    def __call__(
        self,
        transformer: eqx.Module,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        cfg: GrugConfigLike,
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array: ...


@LmConfig.register_subclass("grug_transformer")
@dataclass(frozen=True)
class WrapperConfig(LmConfig["WrapperLMHeadModel"]):
    # Core dims
    max_seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    grug_config: GrugConfigLike = None
    initializer_std: float = 0.01
    vocab_size: int = llama3_tokenizer_vocab_size
    tokenizer: str | None = None
    layer_norm_eps: float = 0.01
    model_cls_fn: Any = None
    loss_fn: Any = None
    _total_trainable_params: int = None
    _flops_per_token: float = None

    def __post_init__(self):
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim % num_heads must be 0"

    # ---- LmConfig API ----
    @property
    def model_type(self) -> type["WrapperLMHeadModel"]:
        return WrapperLMHeadModel

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> "WrapperLMHeadModel":
        return WrapperLMHeadModel.init(Vocab, self, self.model_cls_fn(), self.loss_fn, key=key)

    Embed = property(lambda self: Axis("embed", self.hidden_dim))

    @property
    def actual_head_size(self) -> int:
        return self.hidden_dim // self.num_heads

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        return self._flops_per_token

    def total_trainable_params(self, vocab_size: int) -> int:
        return self._total_trainable_params


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


class WrapperLMHeadModel(
    ModuleWithStateDictSerialization,
    LmHeadModel[GrugConfigLike],
):
    """Minimal Llama-like implementation of LmHeadModel"""

    transformer: eqx.Module
    loss_fn: GrugLossFn
    _wrapper_config: WrapperConfig = eqx.field(static=True)

    @property
    def config(self) -> WrapperConfig:
        return self._wrapper_config

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self._wrapper_config.vocab_size)

    @classmethod
    def init(
        cls, Vocab: Axis, wrapper_config: WrapperConfig, model_cls: eqx.Module, loss_fn: GrugLossFn, *, key
    ) -> "WrapperLMHeadModel":
        transformer = model_cls.init(wrapper_config.grug_config, key=key)
        return WrapperLMHeadModel(transformer, loss_fn, wrapper_config)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        del key, pos_ids  # unused in this lightweight wrapper
        mask = _mask_from_levanter(attn_mask)
        hidden = self.transformer(input_ids.array, mask)
        axes = (*input_ids.axes, Axis("embed", self._wrapper_config.hidden_dim))
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
            per_pos = self.loss_fn(
                self.transformer,
                tokens.array,
                loss_weight.array,
                self.config,
                mask=mask,
                reduction="none",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )
            return hax.named(per_pos, tokens.axes)

        # Fast path: scalar mean/sum reduction over all axes.
        if reduction_axis is None and reduction is hax.mean:
            return self.loss_fn(
                self.transformer,
                tokens.array,
                loss_weight.array,
                self.config,
                mask=mask,
                reduction="mean",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )
        if reduction_axis is None and reduction is hax.sum:
            return self.loss_fn(
                self.transformer,
                tokens.array,
                loss_weight.array,
                self.config,
                mask=mask,
                reduction="sum",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )

        per_pos = self.loss_fn(
            self.transformer,
            tokens.array,
            loss_weight.array,
            self.config,
            mask=mask,
            reduction="none",
            logsumexp_weight=logsumexp_weight,
            loss_dtype=dtype,
        )
        loss = hax.named(per_pos, tokens.axes)

        return reduction(loss, axis=reduction_axis)

    def get_lm_head(self) -> hax.NamedArray:
        return hax.named(self.transformer.output_proj, (Axis("embed", self._wrapper_config.hidden_dim), self.Vocab))

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "WrapperLMHeadModel":
        pass


def build_speedrun(model_cls, model_cfg, train_cfg, loss_fn, speedrun_name, speedrun_desc, author):
    model_cfg2 = WrapperConfig(
        model_cls_fn=lambda: model_cls,
        _total_trainable_params=model_cfg.total_trainable_params,
        _flops_per_token=model_cfg.flops_per_token,
        grug_config=model_cfg,
        loss_fn=loss_fn,
        max_seq_len=model_cfg.max_seq_len,
        hidden_dim=model_cfg.hidden_dim,
        num_heads=model_cfg.num_heads,
        num_kv_heads=model_cfg.num_kv_heads,
        intermediate_dim=model_cfg.intermediate_dim,
        num_layers=model_cfg.num_layers,
    )
    speedrun = SpeedrunConfig(
        author=author,
        description=speedrun_desc,
        model_config=model_cfg2,
        train_config=train_cfg,
    )
    return default_speedrun(speedrun_name, speedrun)
