# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Union

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.normalization import LayerNormBase
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.normalization import LayerNormConfigBase
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.llama import (  # Gemma attention and MLP is identical to LLama
    LlamaEmbedding,
    LlamaMlp,
)
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable

silence_transformer_nag()
from transformers import Gemma2Config as HfGemma2Config  # noqa: E402
from transformers import Gemma3Config as HfGemma3Config  # noqa: E402
from transformers import GemmaConfig as HfGemmaConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402

# Gemma is... very similar to Llama, so we use much of the same modeling code.
#
# The key differences are:
# * Activation is changed to approximate gelu
# * Embedding weights are tied to the LM head
# * Gemma allows specifying a head dimension independently of the hidden and intermediate dims.

# Gemma 2 adds logit capping and alternating sliding window attention
# Gemma 3 removes logit capping and replaces with qk-norm, uses 5-1 global/local attention,


@LayerNormConfigBase.register_subclass("gemma")
@dataclass(frozen=True)
class GemmaNormConfig(LayerNormConfigBase):
    """Configuration for Gemma's custom RMS normalization."""

    def build(self, axis: AxisSpec) -> "GemmaRMSNorm":
        return GemmaRMSNorm.init(axis, eps=self.eps, use_weight=self.use_weight, use_bias=self.use_bias)


@LmConfig.register_subclass("gemma")
@dataclass(frozen=True)
class GemmaConfig(HFCompatConfig):
    """Config for GemmaModel.

    Defaults are set for Gemma-2B.

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 8192.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 2048.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 16384.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 18.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 8.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 1.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "gelu".
    """

    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.gelu_new
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = True

    seq_len: int = 8192
    hidden_dim: int = 2048
    intermediate_dim: int = 16384
    vocab_size: int = 256_000
    num_layers: int = 18
    num_heads: int = 8
    head_dim: int = 256
    num_kv_heads: int = 1
    attn_dropout = 0.0

    use_bias: bool = False
    input_embedding_norm: bool = False

    # Attention-related config
    upcast_attn: bool = False
    use_flash_attention: bool | None = None
    attn_backend: AttentionBackend | None = None
    flash_attention_block_size: int | None = None

    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)
    query_pre_attn_scalar: float | None = None

    gradient_checkpointing: bool = True
    scan_layers: bool = True

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    KVHeads = property(lambda self: Axis(name="kv_heads", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.head_dim))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(
        self, ref_checkpoint: str = "google/gemma-2b"
    ) -> HFCheckpointConverter["GemmaConfig"]:  # type: ignore
        return HFCheckpointConverter(
            self,
            reference_checkpoint=ref_checkpoint,
            trust_remote_code=True,
            HfConfigClass=HfGemmaConfig,
        )

    # The activation function specified in the Gemma HF repo is "gelu", but this is incorrect, it should
    # be "gelu_pytorch_tanh". For backwards compatibility, HF did not change the value in the repo, but
    # instead patches around it. We mimic this behavior and use the approximate gelu internally, and
    # specify the approximate gelu for HF when appropriate.
    # See https://github.com/huggingface/transformers/pull/29402 for more detail.
    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        # extract the text backbone for gemma3
        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config

        if hf_config.hidden_activation is None:
            activation_function = "gelu_pytorch_tanh"
        else:
            activation_function = "gelu_pytorch_tanh"

        if activation_function == "gelu_pytorch_tanh":
            activation_function = "gelu_new"

        assert activation_function is not None, "No activation function found in HF configuration."
        activation_function_enum = getattr(ActivationFunctionEnum, activation_function)

        rope_theta = hf_config.rope_theta
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, getattr(hf_config, "rope_scaling", None))

        return GemmaConfig(
            seq_len=hf_config.max_position_embeddings,
            activation_function=activation_function_enum,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope_config,
            head_dim=hf_config.head_dim,
            query_pre_attn_scalar=getattr(hf_config, "query_pre_attn_scalar", hf_config.head_dim),
        )

    def to_hf_config(self, vocab_size: int, config_overrides: dict | None = None) -> HfGemmaConfig:
        """Convert to HuggingFace's GemmaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfGemmaConfig: HuggingFace's GemmaConfig
        """
        if config_overrides is None:
            config_overrides = {}

        rope = self.rope
        assert isinstance(
            rope, DefaultRotaryEmbeddingsConfig
        ), "Only DefaultRotaryEmbeddingsConfig is supported for Gemma."
        rope_theta, rope_scaling = rope.to_hf_config()

        config = HfGemmaConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            hidden_activation=(
                "gelu_pytorch_tanh" if self.activation_function == "gelu_new" else self.activation_function
            ),
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            _attn_implementation="eager",
            **config_overrides,
        )
        return config

    @property
    def model_type(self) -> type["GemmaLMHeadModel"]:
        return GemmaLMHeadModel

    def flops_per_token(self, vocab_size: int) -> float | None:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=False,
        )

    def attention_config(self) -> AttentionConfig:
        """Convert this GemmaConfig to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.norm_config if getattr(self, "use_qk_norm", False) else None,
        )

    @property
    def norm_config(self) -> LayerNormConfigBase:
        """Get the normalization configuration for Gemma."""
        return GemmaNormConfig(
            eps=self.layer_norm_epsilon,
            use_weight=True,  # GemmaRMSNorm requires use_weight=True
            use_bias=False,  # GemmaRMSNorm doesn't support bias
        )

    def mk_LayerNorm(self, axis: AxisSpec):
        """Create a layer normalization module using the config."""
        return self.norm_config.build(axis)


class GemmaRMSNorm(LayerNormBase):
    """
    Like Llama, Gemma uses an RMSNorm instead of a layer norm.

    The canonical Gemma model computes the variances calculation in fp32 explicitly, so
    we do the same for compatibility.

    Also note that Gemma's RMSNorm adds 1 to the output, which is different from most RMSNorm implementations.
    """

    @classmethod
    def init(cls, axis: AxisSpec, eps: float = 1e-6, use_weight: bool = True, use_bias: bool = False, dtype=None):
        assert use_weight, "GemmaRMSNorm does not support use_weight=False"
        assert not use_bias, "GemmaRMSNorm does not support use_bias=True"

        weight = hax.zeros(axis)
        bias = None

        return GemmaRMSNorm(axis, weight, bias, eps, dtype=jnp.float32)

    def __call__(self, x: NamedArray) -> NamedArray:
        # Gemma's norm is calculated in fp32 explicitly
        # See https://github.com/google/gemma_pytorch/blob/main/gemma/model.py#L173
        dtype = x.dtype
        x = x.astype(jnp.float32)

        var = hax.mean(hax.square(x), axis=self.axis)
        inv = hax.rsqrt(var + self.eps)
        out = x * inv
        # NB: this differs from most RMS Norms by adding 1.
        # This is probably so we can weight decay to 1?
        out = out * (1.0 + self.weight)
        return out.astype(dtype)


class GemmaDecoderLayer(ModuleWithStateDictSerialization):
    config: GemmaConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp
    input_layernorm: LayerNormBase
    post_attention_layernorm: LayerNormBase

    @staticmethod
    def init(config: GemmaConfig, *, key) -> "GemmaDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn_config = config.attention_config()
        attn = Attention.init(attn_config, key=k_attn)
        mlp = LlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        ln_1 = config.mk_LayerNorm(config.Embed)
        ln_2 = config.mk_LayerNorm(config.Embed)

        return GemmaDecoderLayer(config, attn, mlp, ln_1, ln_2)

    @named_call
    def __call__(
        self, x: NamedArray, mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        output = residual + mlp_output
        return output


class GemmaTransformer(ModuleWithStateDictSerialization):
    config: GemmaConfig = eqx.field(static=True)
    layers: BlockFoldable[GemmaDecoderLayer]
    norm: LayerNormBase

    @staticmethod
    def init(config: GemmaConfig, *, key) -> "GemmaTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(config.Layers, GemmaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return GemmaTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: NamedArray | AttentionMask | None, *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        x = self.norm(x)

        return x


class GemmaLMHeadModel(LmHeadModel[GemmaConfig], ModuleWithStateDictSerialization):
    transformer: GemmaTransformer

    # Gemma ties the weights of the embedding matrix and LM head.  Rather than
    # use eqx.Shared which is a bit cumbersome, we juse re-use the embedding matrix
    # as we do in GPT-2.
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

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    @classmethod
    def init(cls, Vocab: Axis, config: GemmaConfig, *, key) -> "GemmaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = GemmaTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return GemmaLMHeadModel(transformer, embeddings, lm_head)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: NamedArray | AttentionMask | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        """
        Args:
            input_ids (NamedArray): [batch, position]
                Indices of input sequence tokens in the vocabulary.
            attn_mask (Union[NamedArray, AttentionMask], optional): [batch, position]
                Mask to avoid performing attention on the padding token indices of the encoder input.
                The attn_mask from training pipeline may be an AttentionMask object instead of NamedArray
        """
        x = self.embeddings.embed(input_ids)
        normalizer = jnp.sqrt(self.config.hidden_dim).astype(x.dtype)
        x = x * normalizer
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[GemmaConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> dict[str, str | None]:
        """Map from Levanter model names to HF."""
        return {"transformer": "model", "embeddings": None}


# =====================
# Gemma 2 Configuration
# =====================


@LmConfig.register_subclass("gemma2")
@dataclass(frozen=True)
class Gemma2Config(GemmaConfig):
    """Configuration class for Gemma-2 family.

    Gemma-2 mostly reuses the Gemma-1 hyperparameters but introduces logit
    soft-capping and uses a slightly different normalization order inside each
    decoder block (implemented in :class:`Gemma2DecoderLayer`).

    Apart from the few new attributes this subclass relies on all helper
    methods from :class:`GemmaConfig` (HF conversion, flop estimation, etc.).
    """

    # Gemma-2 specific tweaks / defaults
    query_pre_attn_scalar: float | None = 256.0
    final_logit_softcapping: float | None = 30.0
    attn_logit_softcapping: float | None = 50.0

    use_qk_norm: bool = False  # remains False for Gemma-2, True would be Gemma-3

    sliding_window: int | None = None

    # ---------- Convenience ----------
    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401 – property returns type, not a str
        """Return the Levanter model class for Gemma-2."""
        return Gemma2LMHeadModel

    # ---------- HF checkpoint helpers ----------
    def hf_checkpoint_converter(
        self, ref_checkpoint: str = "google/gemma-2-2b"
    ) -> HFCheckpointConverter["Gemma2Config"]:  # type: ignore
        return HFCheckpointConverter(
            self,
            reference_checkpoint=ref_checkpoint,
            trust_remote_code=True,
            HfConfigClass=HfGemma2Config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: dict | None = None):  # type: ignore[override]
        """Create a ``transformers.Gemma2Config`` from this Levanter config.

        We bypass the complex branching logic in :pyfunc:`GemmaConfig.to_hf_config` and
        construct the Hugging-Face config explicitly so that the intent is clear.
        """

        from transformers import (
            Gemma2Config as _HFGemma2Config,  # local import (optional dependency)
        )

        if config_overrides is None:
            config_overrides = {}

        # ───── RoPE settings ────────────────────────────────────────────────
        rope = self.rope
        assert isinstance(
            rope, DefaultRotaryEmbeddingsConfig
        ), "Only DefaultRotaryEmbeddingsConfig is supported for Gemma2."
        rope_theta, rope_scaling = rope.to_hf_config()

        # ───── Common hyper-parameters (mostly shared with Gemma1) ──────────
        common_args = dict(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            hidden_activation=(
                "gelu_pytorch_tanh" if self.activation_function == "gelu_new" else self.activation_function
            ),
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            # Gemma-2 additions
            query_pre_attn_scalar=self.query_pre_attn_scalar or self.head_dim,
            final_logit_softcapping=self.final_logit_softcapping,
            attn_logit_softcapping=self.attn_logit_softcapping,
            sliding_window=self.sliding_window if self.sliding_window is not None else self.seq_len,
        )

        # Merge user-overrides last so callers can tweak anything.
        cfg = _HFGemma2Config(
            **common_args,
            _attn_implementation="eager",
            **config_overrides,
        )

        return cfg

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "Gemma2Config":
        """Create a :class:`Gemma2Config` from a HuggingFace configuration."""
        if hf_config.hidden_activation is None:
            activation_function = "gelu_pytorch_tanh"
        else:
            activation_function = hf_config.hidden_activation

        if activation_function == "gelu_pytorch_tanh":
            activation_function = "gelu_new"

        assert activation_function is not None, "No activation function found in HF configuration."
        activation_function_enum = getattr(ActivationFunctionEnum, activation_function)

        rope_theta, rope_scaling = hf_config.rope_theta, getattr(hf_config, "rope_scaling", None)
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, rope_scaling)

        return Gemma2Config(
            seq_len=hf_config.max_position_embeddings,
            activation_function=activation_function_enum,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope_config,
            head_dim=hf_config.head_dim,
            query_pre_attn_scalar=hf_config.query_pre_attn_scalar,
        )

    def attention_config(self) -> AttentionConfig:
        """Convert this GemmaConfig to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.norm_config if getattr(self, "use_qk_norm", False) else None,
            logits_soft_cap=self.attn_logit_softcapping,
        )


# -----------------------
# Gemma 2 Implementation
# -----------------------


class Gemma2DecoderLayer(ModuleWithStateDictSerialization):
    """Gemma 2 decoder layer.

    Compared with Gemma 1 / Llama style layers, Gemma 2 uses an *additional* pair of
    RMSNorms around both the attention and MLP blocks.  The execution order is:

        x = x + post_attn_norm( self_attn( pre_attn_norm(x) ) )
        x = x + post_ff_norm( mlp( pre_ff_norm(x) ) )
    """

    # Apart from the altered normalization scheme the layer is identical to the
    # original Gemma layer so we reuse [Attention][] and [LlamaMlp][] without modification.

    config: GemmaConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp

    # the four norms required by Gemma-2
    input_layernorm: LayerNormBase  # pre-attention
    post_attention_layernorm: LayerNormBase  # post-attention
    pre_feedforward_layernorm: LayerNormBase  # pre-MLP
    post_feedforward_layernorm: LayerNormBase  # post-MLP

    @staticmethod
    def init(config: GemmaConfig, *, key):
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = Attention.init(config.attention_config(), key=k_attn)
        mlp = LlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )

        ln_input = config.mk_LayerNorm(config.Embed)
        ln_post_attn = config.mk_LayerNorm(config.Embed)
        ln_pre_ff = config.mk_LayerNorm(config.Embed)
        ln_post_ff = config.mk_LayerNorm(config.Embed)

        return Gemma2DecoderLayer(
            config,
            attn,
            mlp,
            ln_input,
            ln_post_attn,
            ln_pre_ff,
            ln_post_ff,
        )

    @named_call
    def __call__(
        self, x: NamedArray, mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)

        # Attention block
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # Feed-forward block
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x, key=k_mlp)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x


class Gemma2Transformer(ModuleWithStateDictSerialization):
    """Transformer built from :class:`Gemma2DecoderLayer`."""

    config: Union[Gemma2Config, "Gemma3Config"] = eqx.field(static=True)
    layers: BlockFoldable[Gemma2DecoderLayer]
    norm: LayerNormBase

    @staticmethod
    def init(config: Gemma2Config, *, key):
        # Choose "Stacked" vs "BlockSeq" depending on scan_layers like the original implementation
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(config.Layers, Gemma2DecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)
        return Gemma2Transformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: NamedArray | AttentionMask | None, *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        x = self.norm(x)
        return x


class Gemma2LMHeadModel(LmHeadModel[Gemma2Config], ModuleWithStateDictSerialization):
    """Gemma 2 language-model head.

    Mostly identical to :class:`GemmaLMHeadModel` but uses
    :class:`Gemma2Transformer` under the hood so that the altered layernorm
    ordering is respected.
    """

    transformer: Gemma2Transformer
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

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    @classmethod
    def init(cls, Vocab: Axis, config: GemmaConfig, *, key):
        k_t, k_emb = jrandom.split(key, 2)
        transformer = Gemma2Transformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return Gemma2LMHeadModel(transformer, embeddings, lm_head)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: NamedArray | AttentionMask | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        normalizer = jnp.sqrt(self.config.hidden_dim).astype(x.dtype)
        x = x * normalizer
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x

    def resize_vocab(self, new_size: int, key=None):  # type: ignore[override]
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> dict[str, str | None]:
        return {"transformer": "model", "embeddings": None}


# =====================
# Gemma 3 Configuration
# =====================


@LmConfig.register_subclass("gemma3")
@dataclass(frozen=True)
class Gemma3Config(Gemma2Config):
    """Configuration for Gemma-3 models.

    Differences w.r.t Gemma-2:
    • Enables QK-norm by default (``use_qk_norm=True``).
    • Logit soft-capping is **disabled** (defaults to ``None``).
    Anything related to alternating local/global attention is ignored for now.
    """

    # Gemma-3 specifics
    use_qk_norm: bool = True  # activate qk-norm
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None
    sliding_window_pattern: int | None = None
    """
    Sliding window pattern for Gemma-3, if None, no sliding window layers
    """

    # Gemma3 specific rope frequency for local attention
    rope_local_base_freq: float = 10_000.0

    @property
    def local_rope(self) -> RotaryEmbeddingsConfig:
        """Local RoPE config used for Gemma-3's alternating local attention."""
        return dataclasses.replace(self.rope, theta=self.rope_local_base_freq)

    # ---------- Convenience ----------
    @property  # type: ignore[override]
    def model_type(self):  # noqa: D401
        return Gemma3LMHeadModel

    # ---------- HF helpers ----------
    def hf_checkpoint_converter(
        self, ref_checkpoint: str = "google/gemma-3-1b-pt"
    ) -> HFCheckpointConverter["Gemma3Config"]:  # type: ignore
        return HFCheckpointConverter(
            self,
            reference_checkpoint=ref_checkpoint,
            trust_remote_code=True,
            HfConfigClass=HfGemma3Config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: dict | None = None):  # type: ignore[override]
        """Convert to ``transformers.Gemma3Config``."""
        from transformers import Gemma3TextConfig as _HFGemma3Config

        if config_overrides is None:
            config_overrides = {}

        rope = self.rope
        assert isinstance(rope, DefaultRotaryEmbeddingsConfig)
        rope_theta, rope_scaling = rope.to_hf_config()

        common_args = dict(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            hidden_activation=(
                "gelu_pytorch_tanh" if self.activation_function == "gelu_new" else self.activation_function
            ),
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_qk_norm=self.use_qk_norm,
            rope_local_base_freq=self.rope_local_base_freq,
            query_pre_attn_scalar=self.query_pre_attn_scalar or self.head_dim,
            final_logit_softcapping=self.final_logit_softcapping,
            attn_logit_softcapping=self.attn_logit_softcapping,
            sliding_window=self.sliding_window if self.sliding_window is not None else self.seq_len,
            # we don't currently suport alternating local/global attention, so every layer is global
            sliding_window_pattern=self.sliding_window_pattern if self.sliding_window_pattern is not None else 1,
            attention_dropout=0.0,
            attention_bias=self.use_bias,  # Gemma3 uses bias in attention
            # we have to set this for some reason even though it's default
            use_cache=True,
        )

        cfg = _HFGemma3Config(
            **common_args,
            _attn_implementation="eager",
            **config_overrides,
        )
        return cfg

    def from_hf_config(cls, hf_config: HfConfig) -> "Gemma3Config":  # type: ignore[override]
        """Create a :class:`Gemma3Config` from a HuggingFace configuration."""
        if hf_config.hidden_activation is None:
            activation_function = "gelu_pytorch_tanh"
        else:
            activation_function = hf_config.hidden_activation

        if activation_function == "gelu_pytorch_tanh":
            activation_function = "gelu_new"

        assert activation_function is not None, "No activation function found in HF configuration."
        activation_function_enum = getattr(ActivationFunctionEnum, activation_function)

        rope_theta, rope_scaling = hf_config.rope_theta, getattr(hf_config, "rope_scaling", None)
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, rope_scaling)

        return Gemma3Config(
            seq_len=hf_config.max_position_embeddings,
            activation_function=activation_function_enum,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope=rope_config,
            head_dim=hf_config.head_dim,
            query_pre_attn_scalar=getattr(hf_config, "query_pre_attn_scalar", hf_config.head_dim),
            use_qk_norm=getattr(hf_config, "use_qk_norm", True),
            rope_local_base_freq=getattr(hf_config, "rope_local_base_freq", 10_000.0),
        )

    def attention_config(self) -> AttentionConfig:  # type: ignore[override]
        """Gemma-3 uses QK-norm by default."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.norm_config,  # always on for Gemma-3
        )


# -----------------------
# Gemma 3 Implementation
# -----------------------


class Gemma3LMHeadModel(LmHeadModel[Gemma3Config], ModuleWithStateDictSerialization):
    """Gemma-3 language-model head.

    Reuses the Gemma-2 transformer/decoder implementation (normalisation order is
    unchanged) but relies on a different configuration that enables QK-norm and
    disables logit soft-capping by default.
    """

    transformer: Gemma2Transformer
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

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    @classmethod
    def init(cls, Vocab: Axis, config: Gemma3Config, *, key):
        k_t, k_emb = jrandom.split(key, 2)
        transformer = Gemma2Transformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return Gemma3LMHeadModel(transformer, embeddings, lm_head)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: NamedArray | AttentionMask | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        normalizer = jnp.sqrt(self.config.hidden_dim).astype(x.dtype)
        x = x * normalizer
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x

    def resize_vocab(self, new_size: int, key=None):  # type: ignore[override]
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> dict[str, str | None]:
        return {"transformer": "model", "embeddings": None}
