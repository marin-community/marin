# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Type, cast

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers.attention import Attention, AttentionMask
from levanter.layers.kv_cache import KvPageCache, ListCache
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig, LlamaEmbedding
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable

silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers.models.apertus.configuration_apertus import ApertusConfig as HfApertusConfig  # noqa: E402


class XIELUActivation(eqx.Module):
    alpha_p: NamedArray
    alpha_n: NamedArray
    beta: NamedArray
    eps: NamedArray
    Param: Axis = eqx.field(static=True)

    @staticmethod
    def init(
        *,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
        dtype=jnp.float32,
    ) -> "XIELUActivation":
        Param = Axis("xielu_param", 1)
        alpha_p = hax.log(hax.expm1(hax.full(Param, alpha_p_init, dtype=dtype)))
        alpha_n = hax.log(hax.expm1(hax.full(Param, alpha_n_init - beta, dtype=dtype)))
        beta_arr = hax.full((), beta, dtype=dtype)
        eps_arr = hax.full((), eps, dtype=dtype)
        return XIELUActivation(alpha_p, alpha_n, beta_arr, eps_arr, Param)

    def _as_scalar(self, value: NamedArray) -> NamedArray:
        return hax.mean(value, axis=self.Param)

    def __call__(self, x: NamedArray) -> NamedArray:
        alpha_p = self._as_scalar(hnn.softplus(self.alpha_p))
        alpha_n = self._as_scalar(hnn.softplus(self.alpha_n))
        beta = self.beta
        eps = self.eps
        alpha_n = beta + alpha_n

        positive = alpha_p * x * x + beta * x
        negative = (hax.expm1(hax.minimum(x, eps)) - x) * alpha_n + beta * x
        return hax.where(x > 0, positive, negative)


@LmConfig.register_subclass("apertus")
@dataclass(frozen=True)
class ApertusConfig(LlamaConfig):
    """Config for ApertusModel (Llama variant with xIELU and QK-norm)."""

    max_seq_len: int = 65536
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.xielu
    use_qk_norm: bool = True
    rope: RotaryEmbeddingsConfig = dataclasses.field(
        default_factory=lambda: Llama3RotaryEmbeddingsConfig(
            theta=12000000.0,
            factor=8.0,
            original_max_position_embeddings=8192,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
        )
    )
    reference_checkpoint: str = "swiss-ai/Apertus-8B-2509"

    def hf_checkpoint_converter(self, ref_checkpoint: Optional[str] = None) -> HFCheckpointConverter["ApertusConfig"]:
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=False,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfApertusConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_params = getattr(hf_config, "rope_parameters", None)
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if rope_params is None:
            rope_params = rope_scaling
        rope_theta = None
        if rope_params is not None:
            rope_theta = rope_params.get("rope_theta")
        if rope_theta is None:
            rope_theta = getattr(hf_config, "rope_theta", Llama3RotaryEmbeddingsConfig.theta)
        rope = RotaryEmbeddingsConfig.from_hf_config(rope_theta, rope_params)
        use_bias = bool(getattr(hf_config, "attention_bias", False))

        return ApertusConfig(
            max_seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(getattr(hf_config, "hidden_act", "xielu")),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope,
            use_bias=use_bias,
            use_qk_norm=getattr(hf_config, "qk_norm", True),
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfApertusConfig:
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_config = self.rope.to_hf_config()
        rope_scaling = rope_config
        rope_parameters: Optional[Dict[str, float | int | str]] = None
        if rope_config is not None:
            rope_parameters = dict(rope_config)
            rope_parameters["rope_theta"] = rope_theta
        else:
            rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}

        hf_config = HfApertusConfig(
            vocab_size=vocab_size,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function.value,
            max_position_embeddings=self.max_seq_len,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=self.use_bias,
            attention_dropout=0.0,
            _attn_implementation="eager",
            **config_overrides,
        )
        if rope_parameters is not None:
            hf_config.rope_parameters = rope_parameters
        return hf_config

    @property
    def model_type(self) -> Type["ApertusLMHeadModel"]:
        return ApertusLMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=False,
        )

    def total_trainable_params(self, vocab_size):
        token_embedding = vocab_size * self.hidden_dim

        head_size = self.actual_head_size
        q_proj = self.hidden_dim * head_size * self.num_heads
        kv_proj = 2 * self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        attn = q_proj + kv_proj + o_proj

        mlp = 2 * self.hidden_dim * self.intermediate_dim

        transformer_layer = attn + mlp + 2 * self.hidden_dim
        if self.hybrid_norm:
            transformer_layer += 2 * self.hidden_dim

        transformer = self.num_layers * transformer_layer + self.hidden_dim
        if self.input_embedding_norm:
            transformer += self.hidden_dim

        lm_head = 0 if self.tie_word_embeddings else token_embedding
        return transformer + token_embedding + lm_head


class ApertusMlp(ModuleWithStateDictSerialization):
    up_proj: hnn.Linear
    down_proj: hnn.Linear
    act_fn: XIELUActivation

    @staticmethod
    def init(
        Embed: AxisSpec,
        Mlp: AxisSpec,
        activation_function: ActivationFunctionEnum | str,
        *,
        key,
    ) -> "ApertusMlp":
        if isinstance(activation_function, ActivationFunctionEnum):
            activation_function = activation_function.value
        if activation_function != "xielu":
            raise ValueError(f"Unsupported activation for Apertus MLP: {activation_function}")
        k_up, k_down = jrandom.split(key, 2)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up, use_bias=False, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down, use_bias=False, out_first=True)
        act_fn = XIELUActivation.init()
        return ApertusMlp(up_proj, down_proj, act_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_up, k_down = maybe_rng_split(key, 2)
        x = self.up_proj(x, key=k_up)
        x = self.act_fn(x)
        x = self.down_proj(x, key=k_down)
        return x


class ApertusDecoderLayer(ModuleWithStateDictSerialization):
    config: ApertusConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: ApertusMlp
    attention_layernorm: hnn.RmsNorm
    feedforward_layernorm: hnn.RmsNorm

    @staticmethod
    def init(config: ApertusConfig, *, key) -> "ApertusDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn_config = config.attention_config()
        attn = Attention.init(attn_config, key=k_attn)
        mlp = ApertusMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
        )
        attention_layernorm = config.mk_LayerNorm(config.Embed)
        feedforward_layernorm = config.mk_LayerNorm(config.Embed)
        return ApertusDecoderLayer(config, attn, mlp, attention_layernorm, feedforward_layernorm)

    @named_call
    def __call__(
        self, x: NamedArray, mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        residual = x
        x = self.attention_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        x = residual + attn_output

        residual = x
        x = self.feedforward_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        output = residual + mlp_output
        return output

    @named_call
    def decode(
        self,
        x: NamedArray,
        kv_cache: KvPageCache,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, KvPageCache]:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        residual = x
        x = self.attention_layernorm(x)
        attn_output, kv_cache = self.self_attn.paged_decode(x, kv_cache, batch_info, pos_ids=pos_ids, key=k_attn)
        x = residual + attn_output

        residual = x
        x = self.feedforward_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        output = residual + mlp_output
        return output, kv_cache

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> KvPageCache:
        return self.self_attn.empty_page_cache(spec, dtype=dtype)


class ApertusTransformer(eqx.Module):
    config: ApertusConfig = eqx.field(static=True)
    layers: BlockFoldable[ApertusDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: ApertusConfig, *, key) -> "ApertusTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(config.Layers, ApertusDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return ApertusTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: NamedArray | AttentionMask | None, *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = cast(NamedArray, self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids))
        x = self.norm(x)
        return x

    @named_call
    def decode(
        self,
        kv_cache: ListCache[KvPageCache],
        x: NamedArray,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, ListCache[KvPageCache]]:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None

        caches = list(kv_cache)
        updated_caches: list[KvPageCache] = []

        for i in range(self.config.num_layers):
            with jax.named_scope("slice layer"):
                layer = hax.tree_util.tree_map(lambda l: l["layer", i], self.layers.stacked)  # type: ignore
            with jax.named_scope("slice cache"):
                this_cache = caches[i]
            x, this_cache = layer.decode(
                x,
                this_cache,
                batch_info,
                pos_ids=pos_ids,
                key=keys[i] if keys is not None else None,
            )
            with jax.named_scope("update cache"):
                updated_caches.append(this_cache)

        x = self.norm(x)
        return x, ListCache(updated_caches)

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> ListCache[KvPageCache]:
        caches = [layer.initial_cache(spec, dtype=dtype) for layer in self.layers.unstacked()]
        return ListCache(caches)


class ApertusLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[ApertusConfig]):
    transformer: ApertusTransformer
    embeddings: LlamaEmbedding
    lm_head: hnn.Linear | None

    @property
    def config(self):
        return self.transformer.config

    @property
    def vocab_size(self) -> int:
        return self.Vocab.size

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: ApertusConfig, *, key) -> "ApertusLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = ApertusTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        return ApertusLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: NamedArray | AttentionMask | None = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head:
            lm_logits = self.lm_head(x, key=k_head)
        else:
            lm_logits = self.embeddings.unembed(x)
        return lm_logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[ApertusConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}

    def initial_cache(self, spec: PageTableSpec, *, dtype) -> ListCache[KvPageCache]:
        return hax.auto_sharded(self.transformer.initial_cache(spec, dtype=dtype))

    @named_call
    def decode(
        self,
        input_ids: NamedArray,
        kv_cache: ListCache[KvPageCache],
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, ListCache[KvPageCache]]:
        x = self.embeddings.embed(input_ids)
        k_t = maybe_rng_split(key, 1)[0] if key is not None else None
        x, kv_cache = self.transformer.decode(kv_cache, x, batch_info, pos_ids=pos_ids, key=k_t)
        if self.lm_head:
            lm_logits = self.lm_head(x, key=None)
        else:
            lm_logits = self.embeddings.unembed(x)
        return lm_logits, kv_cache
