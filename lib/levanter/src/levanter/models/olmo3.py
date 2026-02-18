# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""
OLMo 3 model implementation.

OLMo 3 key differences from other models:
- Mixed attention: 3 out of 4 layers use sliding window attention, every 4th layer uses full attention
- Post-norm architecture: layer normalization applied after attention/MLP output, before residual
- QK normalization: layer norm applied to Q and K after projection
"""

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Type, Union, cast

import equinox as eqx
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockSeq, ScanCheckpointPolicy, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers import RmsNormConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.llama import LlamaMlp
from levanter.models.olmo import Olmo2Embedding
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable

silence_transformer_nag()
from transformers import Olmo3Config as HfOlmo3Config  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("olmo3")
@dataclass(frozen=True)
class Olmo3Config(HFCompatConfig):
    """Config for OLMo3."""

    max_seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int | None = None
    num_kv_heads: int = 32
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 4096
    layer_types: Optional[Sequence[str]] = None

    upcast_attn: bool = False
    use_flash_attention: Optional[bool] = True
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool = True
    scan_layers: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    reference_checkpoint: str = "allenai/Olmo-3-1025-7B"
    tokenizer: Optional[str] = None

    @property
    def Embed(self) -> Axis:
        return Axis(name="embed", size=self.hidden_dim)

    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    KVHeads = property(lambda self: Axis(name="kv_head", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layer", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.actual_head_size))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(self, ref_checkpoint: Optional[str] = None) -> HFCheckpointConverter["Olmo3Config"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfOlmo3Config,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "Olmo3Config":  # type: ignore[override]
        rope_theta = getattr(hf_config, "rope_theta", 10000.0)
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, hf_config.rope_scaling)
        return Olmo3Config(
            max_seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            attention_bias=hf_config.attention_bias,
            attention_dropout=hf_config.attention_dropout,
            sliding_window=getattr(hf_config, "sliding_window", 4096),
            layer_types=getattr(hf_config, "layer_types", None),
            rope=rope_config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfOlmo3Config:
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_scaling = self.rope.to_hf_config()

        return HfOlmo3Config(
            max_position_embeddings=self.max_seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function.name,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            vocab_size=vocab_size,
            sliding_window=self.sliding_window,
            layer_types=list(self.get_layer_types()),
            pad_token_id=None,
            _attn_implementation="eager",
            **config_overrides,
        )

    @property
    def model_type(self) -> Type["Olmo3LMHeadModel"]:
        return Olmo3LMHeadModel

    def mk_LayerNorm(self, axis: AxisSpec) -> hnn.RmsNorm:
        return self.norm_config.build(axis)

    @property
    def norm_config(self) -> RmsNormConfig:
        return RmsNormConfig(
            eps=self.layer_norm_epsilon,
            use_weight=self.use_layer_norm_weight,
            use_bias=self.use_bias,
        )

    @property
    def actual_head_size(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_dim // self.num_heads

    def get_layer_types(self) -> Sequence[str]:
        if self.layer_types is not None:
            if len(self.layer_types) != self.num_layers:
                raise ValueError("layer_types must match num_layers")
            return list(self.layer_types)
        return ["sliding_attention" if (i + 1) % 4 != 0 else "full_attention" for i in range(self.num_layers)]

    def flops_per_token(self, vocab_size: int, context_length: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        """Calculate total trainable parameters for OLMo3 model.

        OLMo3 has:
        - Post-norm architecture (norm after attention/MLP, before residual)
        - QK normalization (per-head normalization of Q and K)
        - SwiGLU MLP (gate, up, down projections)

        Args:
            vocab_size: Size of the vocabulary

        Returns:
            int: Total number of trainable parameters
        """
        head_size = self.actual_head_size

        # Attention parameters
        q_proj = self.hidden_dim * head_size * self.num_heads
        k_proj = self.hidden_dim * head_size * self.num_kv_heads
        v_proj = self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        attn_params = q_proj + k_proj + v_proj + o_proj

        # MLP parameters (SwiGLU: gate, up, down)
        mlp_params = 3 * self.hidden_dim * self.intermediate_dim

        # Layer norm parameters (RMSNorm has only weight, no bias)
        # Post-attention norm + Post-feedforward norm
        layer_norm_params = 2 * self.hidden_dim

        # QK norm parameters (applied per-head)
        # Q norm shape: (num_kv_heads, num_q_per_kv, head_size)
        # K norm shape: (num_kv_heads, head_size)
        q_norm_params = self.num_heads * head_size
        k_norm_params = self.num_kv_heads * head_size
        qk_norm_params = q_norm_params + k_norm_params

        # Per layer total
        per_layer = attn_params + mlp_params + layer_norm_params + qk_norm_params

        # Transformer total (includes final layer norm)
        transformer_params = self.num_layers * per_layer + self.hidden_dim

        # Token embeddings
        embedding_params = vocab_size * self.hidden_dim

        # LM head (0 if tied to embeddings)
        lm_head_params = 0 if self.tie_word_embeddings else embedding_params

        return transformer_params + embedding_params + lm_head_params

    def attention_config(self) -> AttentionConfig:
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=self.attention_bias,
            use_output_bias=self.attention_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.norm_config,
        )

    def attention_config_for_layer(self, layer_idx: int) -> AttentionConfig:
        """Build attention config for a specific layer.

        OLMo3 uses different RoPE configurations per layer type:
        - sliding_attention: uses "default" (vanilla) RoPE + sliding window
        - full_attention: uses the model's rope config (e.g., YARN)

        This matches HuggingFace's implementation which creates separate rotary
        embedding modules for each attention type.
        """
        attn_config = self.attention_config()
        layer_types = self.get_layer_types()
        attention_type = layer_types[layer_idx]
        if attention_type == "sliding_attention":
            # Sliding attention uses vanilla RoPE (not YARN) + sliding window
            vanilla_rope = DefaultRotaryEmbeddingsConfig(theta=self.rope.theta)
            return dataclasses.replace(attn_config, sliding_window=self.sliding_window, rope=vanilla_rope)
        return attn_config

    def init_attention(self, layer_idx: int, *, key) -> Attention:
        """Build an attention module with OLMo3's per-head QK norm axes."""
        attn_config = self.attention_config_for_layer(layer_idx)
        attn = Attention.init(attn_config, key=key)
        if attn_config.qk_norm is not None:
            q_norm = self.mk_LayerNorm((attn_config.KVHeads, attn_config.QHeadsPerGroup, attn_config.HeadSize))
            k_norm = self.mk_LayerNorm((attn_config.KVHeads, attn_config.HeadSize))
            attn = dataclasses.replace(attn, q_norm=q_norm, k_norm=k_norm)
        return attn


class Olmo3DecoderLayer(ModuleWithStateDictSerialization, eqx.Module):
    """OLMo3 decoder layer with post-norm architecture."""

    config: Olmo3Config = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp
    post_attention_layernorm: hnn.RmsNorm
    post_feedforward_layernorm: hnn.RmsNorm

    @staticmethod
    def init(config: Olmo3Config, layer_idx: int, *, key) -> "Olmo3DecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = config.init_attention(layer_idx, key=k_attn)
        mlp = LlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )

        post_attention_ln = config.mk_LayerNorm(config.Embed)
        post_feedforward_ln = config.mk_LayerNorm(config.Embed)

        return Olmo3DecoderLayer(config, attn, mlp, post_attention_ln, post_feedforward_ln)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)

        # Self-attention with post-norm
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        attn_output = self.post_attention_layernorm(attn_output)
        h = x + attn_output

        # MLP with post-norm
        mlp_output = self.mlp(h, key=k_mlp)
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        x = h + mlp_output

        return x


class Olmo3Transformer(ModuleWithStateDictSerialization, eqx.Module):
    config: Olmo3Config = eqx.field(static=True)
    _layers: BlockFoldable[Olmo3DecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def _can_scan(layer_types: Sequence[str]) -> bool:
        """Check if all layers have the same type (can use Stacked scan)."""
        return len(set(layer_types)) <= 1

    @staticmethod
    def init(config: Olmo3Config, *, key) -> "Olmo3Transformer":
        layer_types = config.get_layer_types()
        use_stacked = config.scan_layers and Olmo3Transformer._can_scan(layer_types)

        if use_stacked:
            # All layers same type
            layers = Stacked.init(
                config.Layers, Olmo3DecoderLayer, gradient_checkpointing=config.gradient_checkpointing
            )(
                config,
                layer_idx=0,
                key=shaped_rng_split(key, config.num_layers),
            )
        else:
            # Mixed layer types - create each layer explicitly with concrete layer_idx
            # to avoid JAX tracing issues during eval_shape
            keys = shaped_rng_split(key, config.num_layers)
            blocks = [Olmo3DecoderLayer.init(config, layer_idx=i, key=keys[i]) for i in range(config.num_layers)]
            layers = BlockSeq(blocks, config.Layers, ScanCheckpointPolicy._mk(config.gradient_checkpointing))

        ln_f = config.mk_LayerNorm(config.Embed)
        return Olmo3Transformer(config, layers, ln_f)

    @property
    def layers(self) -> Sequence[Olmo3DecoderLayer]:
        return cast(Sequence[Olmo3DecoderLayer], self._layers.unstacked())

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = cast(NamedArray, self._layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids))
        x = self.norm(x)
        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"_layers": "layers"}


class Olmo3LMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[Olmo3Config]):
    transformer: Olmo3Transformer
    embeddings: Olmo2Embedding
    lm_head: Optional[hnn.Linear]

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
    def init(cls, Vocab: Axis, config: Olmo3Config, *, key) -> "Olmo3LMHeadModel":
        k_t, k_emb, k_head = jrandom.split(key, 3)
        transformer = Olmo3Transformer.init(config, key=k_t)
        embeddings = Olmo2Embedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_head, use_bias=False, out_first=True)

        return Olmo3LMHeadModel(transformer, embeddings, lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {
            "transformer": "model",
            "embeddings": "model",
            "lm_head": "lm_head",
        }

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        k_t, k_head = maybe_rng_split(key, 2)

        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)

        if self.lm_head is not None:
            lm_logits = self.lm_head(x, key=k_head)
        else:
            lm_logits = self.embeddings.unembed(x)

        return lm_logits

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
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

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[Olmo3Config]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)
