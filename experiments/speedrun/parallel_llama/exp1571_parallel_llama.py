# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallel Llama implementation where attention and MLP are computed in parallel.
Based on Levanter's Llama implementation but modified for parallel computation.
"""

import dataclasses
from dataclasses import dataclass
from collections.abc import Callable

import equinox as eqx
import jax.random as jrandom
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, AxisSpec, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import ScanCheckpointPolicy, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers import LayerNormConfigBase, RmsNormConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable

silence_transformer_nag()
from transformers import LlamaConfig as HfLlamaConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("parallel_llama")
@dataclass(frozen=True)
class ParallelLlamaConfig(HFCompatConfig):
    """Config for ParallelLlamaModel with parallel attention and MLP computation.

    This config extends the standard LlamaConfig to support parallel computation
    where attention and MLP are computed simultaneously instead of sequentially.

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 2048.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        use_parallel_blocks (bool, optional): whether to use parallel attention and MLP computation. Defaults to True.
    """

    seq_len: int = 2048
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

    # Parallel computation configuration
    use_parallel_blocks: bool = True

    # Attention-related config
    upcast_attn: bool = False
    attn_backend: AttentionBackend | None = None
    flash_attention_block_size: int | None = None

    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    scan_layers: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    reference_checkpoint: str = "NousResearch/Llama-2-7b-hf"
    tokenizer: str | None = None

    # Axis properties
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(self, ref_checkpoint: str | None = None) -> HFCheckpointConverter["ParallelLlamaConfig"]:
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfLlamaConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta = hf_config.rope_theta
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, getattr(hf_config, "rope_scaling", None))
        return ParallelLlamaConfig(
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
            rope=rope_config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: dict | None = None) -> HfLlamaConfig:
        """Convert to HuggingFace's LlamaConfig"""
        if config_overrides is None:
            config_overrides = {}

        if self.rope:
            rope_theta, rope_scaling = self.rope.to_hf_config()
        else:
            rope_theta = None
            rope_scaling = None

        return HfLlamaConfig(
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
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=self.use_bias,
            mlp_bias=self.use_bias,
            _attn_implementation="eager",
            **config_overrides,
        )

    @property
    def model_type(self) -> type["ParallelLlamaLMHeadModel"]:
        return ParallelLlamaLMHeadModel

    @property
    def norm_config(self) -> LayerNormConfigBase:
        return RmsNormConfig(
            use_weight=self.use_layer_norm_weight,
            use_bias=self.use_bias,
            eps=self.layer_norm_epsilon,
        )

    def mk_LayerNorm(self, axis: AxisSpec):
        return self.norm_config.build(axis)

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

    def total_trainable_params(self, vocab_size):
        token_embedding = vocab_size * self.hidden_dim

        head_size = self.actual_head_size
        q_proj = self.hidden_dim * head_size * self.num_heads
        kv_proj = 2 * self.hidden_dim * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * self.hidden_dim
        attn = q_proj + kv_proj + o_proj

        mlp = 3 * self.hidden_dim * self.intermediate_dim

        transformer_layer = attn + mlp + 2 * self.hidden_dim  # plus 2 rmsnorm
        # Note: ParallelLlamaConfig doesn't support hybrid_norm or input_embedding_norm
        # so we don't need those conditional additions

        transformer = self.num_layers * transformer_layer + self.hidden_dim  # plus final rmsnorm

        lm_head = 0 if self.tie_word_embeddings else token_embedding
        return transformer + token_embedding + lm_head

    def attention_config(self) -> AttentionConfig:
        """Convert this ParallelLlamaConfig to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
        )

    @property
    def actual_head_size(self):
        """Returns the actual head size based on the head_dim or calculated from hidden_dim and num_heads."""
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_dim // self.num_heads


class ParallelLlamaMlp(eqx.Module):
    """Multi-layer Perceptron identical to LlamaMlp for compatibility"""

    gate_proj: hnn.Linear
    up_proj: hnn.Linear
    down_proj: hnn.Linear
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: AxisSpec,
        Mlp: AxisSpec,
        activation_fn: ActivationFunctionEnum | Callable,
        *,
        key,
        use_bias: bool = False,
    ) -> "ParallelLlamaMlp":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias, out_first=True)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()
        elif isinstance(activation_fn, str):
            activation_fn = ActivationFunctionEnum(activation_fn).to_fn()
        return ParallelLlamaMlp(gate_proj, up_proj, down_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(x, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x, key=k_up)
        outputs = self.down_proj(hidden_states, key=k_down)
        return outputs


class ParallelLlamaDecoderLayer(eqx.Module):
    """
    Parallel Llama decoder layer where attention and MLP are computed in parallel.

    The key difference from standard LlamaDecoderLayer:
    - Single shared layer normalization for both attention and MLP
    - Attention and MLP computed in parallel on the normalized input
    - Outputs are combined (sum) and added to residual
    """

    config: ParallelLlamaConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: ParallelLlamaMlp
    input_layernorm: hnn.RmsNorm  # Shared normalization for both attention and MLP

    @staticmethod
    def init(config: ParallelLlamaConfig, *, key) -> "ParallelLlamaDecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)

        attn_config = config.attention_config()
        attn = Attention.init(attn_config, key=k_attn)
        mlp = ParallelLlamaMlp.init(
            config.Embed,
            config.Mlp,
            config.activation_function,
            key=k_mlp,
            use_bias=config.use_bias,
        )
        # Single shared layer norm for both attention and MLP
        ln = config.mk_LayerNorm(config.Embed)

        return ParallelLlamaDecoderLayer(config, attn, mlp, ln)

    @named_call
    def __call__(
        self, x: NamedArray, mask: NamedArray | AttentionMask | None, *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)

        # Store residual connection
        residual = x

        # Single shared normalization
        normalized_x = self.input_layernorm(x)

        if self.config.use_parallel_blocks:
            # Parallel computation: attention and MLP computed simultaneously
            attn_output = self.self_attn(x=normalized_x, mask=mask, key=k_attn, pos_ids=pos_ids)
            mlp_output = self.mlp(normalized_x, key=k_mlp)

            # Sum the parallel outputs
            combined_output = attn_output + mlp_output

            # Add residual connection
            output = residual + combined_output
        else:
            # Fallback to sequential computation for compatibility
            # Attention first
            attn_output = self.self_attn(x=normalized_x, mask=mask, key=k_attn, pos_ids=pos_ids)
            x = residual + attn_output

            # MLP second
            residual = x
            normalized_x = self.input_layernorm(x)  # Re-normalize for MLP
            mlp_output = self.mlp(normalized_x, key=k_mlp)
            output = residual + mlp_output

        return output


class ParallelLlamaTransformer(eqx.Module):
    """Parallel Llama transformer using ParallelLlamaDecoderLayer"""

    config: ParallelLlamaConfig = eqx.field(static=True)
    layers: BlockFoldable[ParallelLlamaDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: ParallelLlamaConfig, *, key) -> "ParallelLlamaTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(config.Layers, ParallelLlamaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return ParallelLlamaTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: NamedArray | AttentionMask | None, *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        x = self.norm(x)
        return x


class ParallelLlamaEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    """Embedding layer identical to LlamaEmbedding for compatibility"""

    token_embeddings: hnn.Embedding

    @staticmethod
    def init(Vocab: Axis, config: ParallelLlamaConfig, *, key) -> "ParallelLlamaEmbedding":
        token_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key)
        return ParallelLlamaEmbedding(token_embeddings)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @property
    def Embed(self) -> Axis:
        return self.token_embeddings.Embed

    @named_call
    def embed(self, input_ids, *args):
        return self.token_embeddings(input_ids)

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def _state_dict_key_map(self) -> dict[str, str | None]:
        return {"token_embeddings": "model.embed_tokens"}

    def resize_embeddings(self, new_size: int, key: PRNGKeyArray | None = None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, token_embeddings=new_weights)


class ParallelLlamaLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[ParallelLlamaConfig]):
    """Parallel Llama model with language modeling head"""

    transformer: ParallelLlamaTransformer
    embeddings: ParallelLlamaEmbedding
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
    def init(cls, Vocab: Axis, config: ParallelLlamaConfig, *, key) -> "ParallelLlamaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = ParallelLlamaTransformer.init(config, key=k_t)
        embeddings = ParallelLlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        return ParallelLlamaLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: NamedArray | AttentionMask | None = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        """
        Forward pass through the parallel Llama model.

        Args:
            input_ids (NamedArray): [batch, position] token indices
            attn_mask (Union[NamedArray, AttentionMask], optional): attention mask
            pos_ids: position IDs

        Returns:
            NamedArray: logits with shape {Batch, Pos, Vocab}
        """
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
        """Compute the activations for the next token in a sequence."""
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[ParallelLlamaConfig]":
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
