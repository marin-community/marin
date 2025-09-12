# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Type, Union

import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.layers.attention import AttentionBackend, AttentionConfig, AttentionMask
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig, LlamaEmbedding, LlamaTransformer
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import MistralConfig as HfMistralConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("mistral")
@dataclass(frozen=True)
class MistralConfig(LlamaConfig):
    """Config for MistralModel

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 8192.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 14336.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 8.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        sliding_window (int, optional): window size of sliding window attention. Defaults to 4096.
    """

    seq_len: int = 8192
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-6
    sliding_window: int = 4096

    # Attention-related config
    upcast_attn: bool = False
    use_flash_attention: Optional[bool] = True
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool = True

    use_bias: bool = False
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Heads = property(lambda self: Axis(name="heads", size=self.num_heads))
    KVHeads = property(lambda self: Axis(name="kv_heads", size=self.num_kv_heads))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_dim // self.num_heads))

    def hf_checkpoint_converter(
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["MistralConfig"]:  # type: ignore
        hf_model_path = "mistralai/Mistral-7B-v0.1" if ref_checkpoint is None else ref_checkpoint

        return HFCheckpointConverter(
            self,
            reference_checkpoint=hf_model_path,
            trust_remote_code=True,
            tokenizer=hf_model_path,
            HfConfigClass=HfMistralConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        rope_theta = hf_config.rope_theta
        rope_config = RotaryEmbeddingsConfig.from_hf_config(rope_theta, None)
        return MistralConfig(
            seq_len=hf_config.max_position_embeddings,  # this might be too big...
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            initializer_range=hf_config.initializer_range,
            layer_norm_epsilon=hf_config.rms_norm_eps,
            sliding_window=hf_config.sliding_window,
            rope=rope_config,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfMistralConfig:
        """Convert to HuggingFace's MistralConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfMistralConfig: HuggingFace's MistralConfig
        """
        if config_overrides is None:
            config_overrides = {}

        rope_theta, rope_scaling = self.rope.to_hf_config()

        return HfMistralConfig(
            max_position_embeddings=self.seq_len,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function.name,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            sliding_window=self.sliding_window,
            vocab_size=vocab_size,
            rope_theta=rope_theta,
            # Mistral 1 doesn't use rope_scaling
            # rope_scaling=rope_scaling,
            **config_overrides,
        )

    @property
    def model_type(cls) -> Type["MistralLMHeadModel"]:
        return MistralLMHeadModel

    def flops_per_token(self, vocab_size: int) -> Optional[float]:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            vocab_size=vocab_size,
            glu=False,
        )

    def attention_config(self) -> AttentionConfig:
        """Convert this MistralConfig to an AttentionConfig for use with Attention."""
        return AttentionConfig(
            Embed=self.Embed,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            use_bias=self.use_bias,
            upcast_attn=self.upcast_attn,
            attn_backend=self.attn_backend,
            flash_attention_block_size=self.flash_attention_block_size,
            rope=self.rope,
        )


class MistralLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[MistralConfig]):
    transformer: LlamaTransformer
    embeddings: LlamaEmbedding
    lm_head: hnn.Linear

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
    def init(cls, Vocab: Axis, config: MistralConfig, *, key) -> "MistralLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = LlamaTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return MistralLMHeadModel(transformer, embeddings, lm_head)

    def get_lm_head(self) -> hax.NamedArray:
        assert self.lm_head.bias is None
        return self.lm_head.weight

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[Union[NamedArray, AttentionMask]] = None,
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
        k_t, k_head = maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        return x

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[MistralConfig]":
        new_Vocab = self.Vocab.resize(new_size)
        k1, k2 = maybe_rng_split(key, 2)
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=k1)
        new_lm_matrix = hax.tree_util.resize_axis(self.lm_head.weight, self.Vocab, new_size, key=k2)
        new_lm_head = dataclasses.replace(self.lm_head, Out=new_Vocab, weight=new_lm_matrix)

        return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}
