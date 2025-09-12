# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type, Union

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers import RmsNormConfig
from levanter.layers.attention import AttentionBackend, AttentionConfig, AttentionMask, dot_product_attention
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddings, RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import Olmo2Config as HfOlmo2Config  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("olmo2")
@dataclass(frozen=True)
class Olmo2Config(HFCompatConfig):
    """Config for Olmo2Model

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 4096.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 32.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        rope_scaling (Dict, optional): dict containing the scaling configuration for the Rotary Positional Embedding.
    """

    seq_len: int = 4096
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-6
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0

    # Attention-related config
    upcast_attn: bool = False
    use_flash_attention: Optional[bool] = True
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool = True
    scan_layers: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    reference_checkpoint: str = "allenai/OLMo-2-1124-7B"
    tokenizer: Optional[str] = None

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    KVHeads = property(lambda self: Axis(name="kv_heads", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["Olmo2Config"]:  # type: ignore
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfOlmo2Config,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta = getattr(hf_config, "rope_theta", 500000)
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, hf_config.rope_scaling)
        return Olmo2Config(
            seq_len=hf_config.max_position_embeddings,
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
            rope=rope_config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfOlmo2Config:
        """Convert to HuggingFace's Olmo2Config

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfOlmo2Config: HuggingFace's Olmo2Config
        """
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_scaling = self.rope.to_hf_config()

        return HfOlmo2Config(
            max_position_embeddings=self.seq_len,
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
            pad_token_id=None,
            _attn_implementation="eager",
            **config_overrides,
        )

    @property
    def model_type(self) -> Type["Olmo2LMHeadModel"]:
        return Olmo2LMHeadModel

    def mk_LayerNorm(self, axis: AxisSpec) -> hnn.RmsNorm:
        return self.norm_config.build(axis)

    def total_trainable_params(self, vocab_size):
        """Calculate total trainable parameters for OLMo 2 model.

        Args:
            vocab_size: Size of the vocabulary

        Returns:
            int: Total number of trainable parameters
        """
        # Token embedding parameters (input embeddings)
        token_embedding = vocab_size * self.hidden_dim

        # Head dimensions
        head_size = self.hidden_dim // self.num_heads

        # Attention module parameters
        q_proj = self.hidden_dim * head_size * self.num_heads
        kv_proj = 2 * self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        attn = q_proj + kv_proj + o_proj

        # MLP parameters (with SiLU activation using gate and up projections)
        mlp = 3 * self.hidden_dim * self.intermediate_dim

        # Layer norm parameters for standard attention and MLP norms
        # RMSNorm only has a single weight vector per dimension (no bias)
        layer_norm = 2 * self.hidden_dim  # post-attention and post-feedforward norms

        # Additional norms for QK normalization in each attention layer
        qk_norm = 2 * self.hidden_dim  # q_norm and k_norm (OLMo 2 applies norm to Q and K)

        # Total parameters per transformer layer
        transformer_layer = attn + mlp + layer_norm + qk_norm

        # All transformer layers plus final layer norm
        transformer = self.num_layers * transformer_layer + self.hidden_dim  # plus final rmsnorm

        # Input embedding norm if used
        if hasattr(self, "input_embedding_norm") and self.input_embedding_norm:
            transformer += self.hidden_dim

        # Total parameters (transformer + embeddings + LM head)
        # LM head shares weights with token embeddings if tie_word_embeddings is True
        lm_head = 0 if (hasattr(self, "tie_word_embeddings") and self.tie_word_embeddings) else token_embedding

        return transformer + token_embedding + lm_head

    def flops_per_token(self, vocab_size: int):
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=True,
        )

    def attention_config(self) -> AttentionConfig:
        """Convert this Olmo2Config to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_bias=self.attention_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
            qk_norm=self.norm_config,  # OLMo2 always uses QK normalization
        )

    @property
    def norm_config(self) -> RmsNormConfig:
        """Return the normalization configuration for OLMo2."""
        return RmsNormConfig(
            eps=self.layer_norm_epsilon,
            use_weight=self.use_layer_norm_weight,
            use_bias=self.use_bias,
        )


class Olmo2MLP(eqx.Module):
    """Multi-layer Perceptron for Olmo2
    Similar to LlamaMlp, adds an up-proj that multiplies with activated gate_proj before down-proj.
    """

    gate_proj: hnn.Linear  # projection from Embed to Mlp
    up_proj: hnn.Linear  # projection from Embed to Mlp
    down_proj: hnn.Linear  # projection from Mlp to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: Axis, Mlp: Axis, activation_fn: Union[ActivationFunctionEnum, Callable], *, key, use_bias: bool = False
    ) -> "Olmo2MLP":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias, out_first=True)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()
        return Olmo2MLP(gate_proj, up_proj, down_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(x, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x, key=k_up)
        outputs = self.down_proj(hidden_states, key=k_down)
        return outputs


class Olmo2Attention(ModuleWithStateDictSerialization, eqx.Module):
    config: Olmo2Config = eqx.field(static=True)
    q_proj: hnn.Linear  # projection from Embed to query
    k_proj: hnn.Linear  # projection from Embed to key
    v_proj: hnn.Linear  # projection from Embed to value
    o_proj: hnn.Linear  # projection from Heads to output
    q_norm: hnn.RmsNorm  # normalization for query
    k_norm: hnn.RmsNorm  # normalization for key
    rot_embs: Optional[RotaryEmbeddings] = eqx.field(default=None)

    @staticmethod
    def init(config: Olmo2Config, *, key) -> "Olmo2Attention":
        use_bias = config.attention_bias
        Embed = config.Embed
        QHeadsPerGroup = hax.Axis("q_heads_per_group", config.num_heads // config.num_kv_heads)
        HeadSize = config.HeadSize

        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        # should this be the same os o_proj
        q_proj = hnn.Linear.init(
            In=Embed, Out=(config.KVHeads, QHeadsPerGroup, HeadSize), key=k_q, use_bias=use_bias, out_first=True
        )
        k_proj = hnn.Linear.init(In=Embed, Out=(config.KVHeads, HeadSize), key=k_k, use_bias=use_bias, out_first=True)
        v_proj = hnn.Linear.init(In=Embed, Out=(config.KVHeads, HeadSize), key=k_v, use_bias=use_bias, out_first=True)
        o_proj = hnn.Linear.init(In=(config.Heads, HeadSize), Out=Embed, key=k_o, use_bias=use_bias, out_first=True)

        # For q_norm, normalization is over the entire hidden dimension
        q_norm = config.mk_LayerNorm((config.KVHeads, QHeadsPerGroup, HeadSize))
        k_norm = config.mk_LayerNorm((config.KVHeads, HeadSize))

        # Build rotary embeddings once during initialization if configured
        rot_embs = config.rope.build(config.HeadSize) if config.rope is not None else None

        return Olmo2Attention(config, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rot_embs)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        # OLMo2 project for q and k and then normalizes
        q_proj = self.q_proj(x, key=key_q)
        q = self.q_norm(q_proj)

        # Project to key
        k_proj = self.k_proj(x, key=key_k)
        k = self.k_norm(k_proj)

        # Regular projection for value
        v = self.v_proj(x, key=key_v)

        # Reshape for attention
        q = q.rearrange((..., "kv_heads", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_heads", "position", "head_size"))
        v = v.rearrange((..., "kv_heads", "position", "head_size"))

        # Apply rotary position embeddings if configured
        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"))
            q = self.rot_embs(q, pos_ids)
            k = self.rot_embs(k, pos_ids)

        # Rename position axis for attention
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        # Apply attention
        c = self.config
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            use_flash=c.use_flash_attention,
            attn_backend=self.config.attn_backend,
            flash_block_size=c.flash_attention_block_size,
            dropout=self.config.attention_dropout,
            inference=not self.config.attention_dropout > 0,
            prng=key,
        )

        # Flatten heads and apply output projection
        attn_output = attn_output.flatten_axes(("kv_heads", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        attn_output = self.o_proj(attn_output, key=key_o)

        return attn_output


class Olmo2DecoderLayer(ModuleWithStateDictSerialization, eqx.Module):
    config: Olmo2Config = eqx.field(static=True)
    self_attn: Olmo2Attention
    mlp: Olmo2MLP
    post_attention_layernorm: hnn.RmsNorm
    post_feedforward_layernorm: hnn.RmsNorm

    @staticmethod
    def init(config: Olmo2Config, *, key) -> "Olmo2DecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn = Olmo2Attention.init(config, key=k_attn)
        mlp = Olmo2MLP.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )

        post_attention_ln = config.mk_LayerNorm(config.Embed)
        post_feedforward_ln = config.mk_LayerNorm(config.Embed)

        return Olmo2DecoderLayer(config, attn, mlp, post_attention_ln, post_feedforward_ln)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)

        # Self attention with norm before residual
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        attn_output = self.post_attention_layernorm(attn_output)
        h = x + attn_output

        # MLP with norm before residual
        mlp_output = self.mlp(h, key=k_mlp)
        mlp_output = self.post_feedforward_layernorm(mlp_output)
        x = h + mlp_output

        return x


class Olmo2Transformer(ModuleWithStateDictSerialization, eqx.Module):
    config: Olmo2Config = eqx.field(static=True)
    layers: Stacked[Olmo2DecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: Olmo2Config, *, key) -> "Olmo2Transformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(config.Layers, Olmo2DecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return Olmo2Transformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        x = self.norm(x)
        return x


class Olmo2Embedding(ModuleWithStateDictSerialization, eqx.Module):
    """Token embedding for Olmo2"""

    Vocab: Axis = eqx.field(static=True)
    token_embeddings: hnn.Embedding

    @staticmethod
    def init(Vocab: Axis, config: Olmo2Config, *, key) -> "Olmo2Embedding":
        return Olmo2Embedding(Vocab, hnn.Embedding.init(Vocab, config.Embed, key=key))

    @named_call
    def embed(self, input_ids, *args):
        input_embeds = self.token_embeddings(input_ids)
        return input_embeds

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "embed_tokens"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_weights)


class Olmo2LMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[Olmo2Config]):
    transformer: Olmo2Transformer
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
    def init(cls, Vocab: Axis, config: Olmo2Config, *, key) -> "Olmo2LMHeadModel":
        k_t, k_emb, k_head = jrandom.split(key, 3)
        transformer = Olmo2Transformer.init(config, key=k_t)
        embeddings = Olmo2Embedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_head, use_bias=False, out_first=True)

        return Olmo2LMHeadModel(transformer, embeddings, lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        """Map model parameter names to HF parameter names"""
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
        """
        Args:
            input_ids (NamedArray): [batch, position]
                Indices of input sequence tokens in the vocabulary.
            attn_mask (Union[NamedArray, AttentionMask], optional): [batch, position]
                Mask to avoid performing attention on the padding token indices of the encoder input.
        """
        k_t, k_head = maybe_rng_split(key, 2)

        # Get token embeddings
        x = self.embeddings.embed(input_ids)

        # Pass through transformer
        x = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)

        # Apply language modeling head
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
        """
        Compute the activations for the next token in a sequence.
        Args:
            input_ids: token IDs with shape {Pos}
            attn_mask: attention mask with shape {Pos, KeyPos}
            key: PRNGKeyArray for random number generation

        Returns:
            NamedArray: activations with shape {Pos, Embed}
        """
        # Get token embeddings
        x = self.embeddings.embed(input_ids)

        # Pass through transformer
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[Olmo2Config]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        if self.lm_head is not None:
            new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
            new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        else:
            return dataclasses.replace(self, embeddings=new_embeddings)
