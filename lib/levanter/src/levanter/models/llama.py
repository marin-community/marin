# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type, Union

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
from levanter.inference.page_table import PageBatchInfo, PageTable
from levanter.layers import LayerNormConfigBase, RmsNormConfig
from levanter.layers.attention import Attention, AttentionBackend, AttentionConfig, AttentionMask, KvPageCache
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable

silence_transformer_nag()
from transformers import LlamaConfig as HfLlamaConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("llama")
@dataclass(frozen=True)
class LlamaConfig(HFCompatConfig):
    """Config for LlamaModel

    Args:
        seq_len (int, optional): maximum length of the input sequence. Defaults to 2048.
        hidden_dim (int, optional): dimension of the hidden state. Defaults to 4096.
        intermediate_dim (int, optional): dimension of the intermediate state. Defaults to 11008.
        num_layers (int, optional): number of hidden layers in the Transformer encoder. Defaults to 32.
        num_heads (int, optional): number of attention heads for each attention layer. Defaults to 32.
        num_kv_heads (int, optional): number of attention heads for keys and values in each attention layer.
            Setting to 1 means MQA. Setting to num_heads means MHA. Otherwise GQA.
            Note that num_heads must be divisible by this number. Defaults to 32.
        activation_function (str, optional): activation function for the hidden layer. Defaults to "silu".
        hybrid_norm (bool, optional): whether to use hybrid normalization with additional layer norms after attention and MLP. Defaults to False.
        input_embedding_norm (bool, optional): whether to use layer normalization after input embeddings. Defaults to False.
    """

    seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int | None = None  # if set, will use this as the head dimension instead of hidden_dim // num_heads
    num_kv_heads: int = 32
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    initializer_range: float = 0.02
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    hybrid_norm: bool = False
    input_embedding_norm: bool = False

    # Attention-related config
    upcast_attn: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    gradient_checkpointing: bool | ScanCheckpointPolicy | str = True
    scan_layers: bool = True

    use_bias: bool = False
    use_layer_norm_weight: bool = True
    rope: RotaryEmbeddingsConfig = dataclasses.field(default_factory=DefaultRotaryEmbeddingsConfig)

    reference_checkpoint: str = "NousResearch/Llama-2-7b-hf"
    tokenizer: Optional[str] = None

    # Axis
    Pos = property(lambda self: Axis(name="position", size=self.seq_len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_dim))
    Layers = property(lambda self: Axis(name="layers", size=self.num_layers))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_dim))

    def __post_init__(self):
        assert (
            self.num_heads % self.num_kv_heads == 0
        ), f"num_heads={self.num_heads} not divisible by num_kv_heads={self.num_kv_heads}."

    def hf_checkpoint_converter(self, ref_checkpoint: Optional[str] = None) -> HFCheckpointConverter["LlamaConfig"]:  # type: ignore
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
        return LlamaConfig(
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

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfLlamaConfig:
        """Convert to HuggingFace's LlamaConfig

        Args:
            vocab_size (int, optional): Vocabulary size of the tokenizer. Defaults to 32000.
            config_overrides (dict, optional): Overrides for the config. Defaults to None.

        Returns:
            HfLlamaConfig: HuggingFace's LlamaConfig

        Raises:
            ValueError: If hybrid_norm or input_embedding_norm are enabled, as these features
                are not supported in the HuggingFace config format.
        """
        if self.hybrid_norm or self.input_embedding_norm:
            raise ValueError(
                "Cannot export to HuggingFace format with hybrid_norm or input_embedding_norm enabled. "
                "These features are not supported in the HuggingFace config format. "
                "Please disable these features before exporting."
            )

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
    def model_type(self) -> Type["LlamaLMHeadModel"]:
        return LlamaLMHeadModel

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
        if self.hybrid_norm:
            transformer_layer += 2 * self.hidden_dim

        transformer = self.num_layers * transformer_layer + self.hidden_dim  # plus final rmsnorm
        if self.input_embedding_norm:
            transformer += self.hidden_dim

        lm_head = 0 if self.tie_word_embeddings else token_embedding
        return transformer + token_embedding + lm_head

    def attention_config(self) -> AttentionConfig:
        """Convert this LlamaConfig to an AttentionConfig for use with Attention."""
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


class LlamaMlp(eqx.Module):
    """Multi-layer Perceptron
    In comparison with GPT2, LlamaMlp adds an up-proj that multiplies with activated gate_proj,
    before down-proj.
    """

    gate_proj: hnn.Linear  # projection from Embed to Mlp
    up_proj: hnn.Linear  # projection from Embed to Mlp
    down_proj: hnn.Linear  # projection from Mlp to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(
        Embed: AxisSpec,
        Mlp: AxisSpec,
        activation_fn: Union[ActivationFunctionEnum, Callable],
        *,
        key,
        use_bias: bool = False,
    ) -> "LlamaMlp":
        k_fc, k_up_proj, k_down_proj = jrandom.split(key, 3)
        gate_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=True)
        up_proj = hnn.Linear.init(Out=Mlp, In=Embed, key=k_up_proj, use_bias=use_bias, out_first=True)
        down_proj = hnn.Linear.init(Out=Embed, In=Mlp, key=k_down_proj, use_bias=use_bias, out_first=True)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()
        elif isinstance(activation_fn, str):
            activation_fn = ActivationFunctionEnum(activation_fn).to_fn()
        return LlamaMlp(gate_proj, up_proj, down_proj, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_gate, k_up, k_down = maybe_rng_split(key, 3)
        hidden_states = self.gate_proj(x, key=k_gate)
        hidden_states = self.act(hidden_states)
        hidden_states = hidden_states * self.up_proj(x, key=k_up)
        outputs = self.down_proj(hidden_states, key=k_down)
        return outputs


class LlamaDecoderLayer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    self_attn: Attention
    mlp: LlamaMlp
    input_layernorm: hnn.RmsNorm
    post_attention_layernorm: hnn.RmsNorm
    post_attn_layernorm: Optional[hnn.RmsNorm] = None
    post_mlp_layernorm: Optional[hnn.RmsNorm] = None

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaDecoderLayer":
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
        post_attn_ln = None
        post_mlp_ln = None
        if config.hybrid_norm:
            post_attn_ln = config.mk_LayerNorm(config.Embed)
            post_mlp_ln = config.mk_LayerNorm(config.Embed)
        return LlamaDecoderLayer(config, attn, mlp, ln_1, ln_2, post_attn_ln, post_mlp_ln)

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output = self.self_attn(x=x, mask=mask, key=k_attn, pos_ids=pos_ids)
        if self.post_attn_layernorm is not None:
            attn_output = self.post_attn_layernorm(attn_output)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        if self.post_mlp_layernorm is not None:
            mlp_output = self.post_mlp_layernorm(mlp_output)
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
        # self attention and skip connection
        residual = x
        x = self.input_layernorm(x)
        attn_output, kv_cache = self.self_attn.paged_decode(x, kv_cache, batch_info, pos_ids=pos_ids, key=k_attn)

        if self.post_attn_layernorm is not None:
            attn_output = self.post_attn_layernorm(attn_output)
        x = residual + attn_output

        # MLP and skip connection
        residual = x
        x = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x, key=k_mlp)
        if self.post_mlp_layernorm is not None:
            mlp_output = self.post_mlp_layernorm(mlp_output)
        output = residual + mlp_output
        return output, kv_cache

    def initial_cache(self, page_table: PageTable, *, dtype) -> KvPageCache:
        """
        Creates an empty page cache for this layer. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        return self.self_attn.empty_page_cache(page_table, dtype=dtype)


class LlamaTransformer(eqx.Module):
    config: LlamaConfig = eqx.field(static=True)
    layers: BlockFoldable[LlamaDecoderLayer]
    norm: hnn.RmsNorm

    @staticmethod
    def init(config: LlamaConfig, *, key) -> "LlamaTransformer":
        S = Stacked
        if not config.scan_layers:
            from haliax.nn.scan import BlockSeq

            S = BlockSeq

        layers = S.init(config.Layers, LlamaDecoderLayer, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return LlamaTransformer(config, layers, ln_f)

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: Optional[NamedArray | AttentionMask], *, key, pos_ids: NamedArray | None = None
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)
        x = self.norm(x)

        return x

    @named_call
    def decode(
        self,
        kv_cache: KvPageCache,
        x: NamedArray,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, KvPageCache]:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None

        x, kv_cache = self.layers.scan_via(LlamaDecoderLayer.decode)(
            x,
            kv_cache,
            batch_info,
            pos_ids=pos_ids,
            key=keys,
        )
        x = self.norm(x)

        return x, kv_cache

    def initial_cache(self, page_table: PageTable, *, dtype) -> KvPageCache:
        """
        Creates an empty page cache for this transformer. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        return self.layers.vmap_via(LlamaDecoderLayer.initial_cache)(page_table, dtype=dtype)


class LlamaEmbedding(ModuleWithStateDictSerialization, eqx.Module):
    """Similar to GPT2 Embedding, except that:
    - Llama doesn't have position embedding in the Embedding layer.
    - Llama doesn't use dropout.
    """

    token_embeddings: hnn.Embedding
    norm: Optional[hnn.RmsNorm] = None

    @staticmethod
    def init(Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaEmbedding":
        token_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key)
        norm = None
        if config.input_embedding_norm:
            norm = config.mk_LayerNorm(config.Embed)
        return LlamaEmbedding(token_embeddings, norm)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    @property
    def Embed(self) -> Axis:
        return self.token_embeddings.Embed

    @named_call
    def embed(self, input_ids, *args):
        input_embeds = self.token_embeddings(input_ids)
        if self.norm is not None:
            input_embeds = self.norm(input_embeds)
        return input_embeds

    def unembed(self, x: NamedArray):
        return self.token_embeddings.unembed(x)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "model.embed_tokens"}

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_weights = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, token_embeddings=new_weights)


class LlamaLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[LlamaConfig]):
    transformer: LlamaTransformer
    embeddings: LlamaEmbedding
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
    def init(cls, Vocab: Axis, config: LlamaConfig, *, key) -> "LlamaLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = LlamaTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        return LlamaLMHeadModel(transformer, embeddings, lm_head)

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
                The attn_mask from training pipeline may be an AttentionMask object instead of NamedArray
            pos_ids: NamedArray | None = None,

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
            pos_ids: position IDs with shape {Pos}

        Returns:
            NamedArray: activations with shape {Pos, Embed}

        """
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)

        return x

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        else:
            return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[LlamaConfig]":
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

    def initial_cache(self, page_table: PageTable, *, dtype) -> KvPageCache:
        """
        Creates an initial cache for this model. Note that in order to create a decoder state, you
        need to couple the KvPageCache to the PageTable's state with a BatchInfo object.
        """
        return hax.auto_sharded(self.transformer.initial_cache(page_table, dtype=dtype))

    @named_call
    def decode(
        self,
        input_ids: NamedArray,  # token IDs for *this* step (shape {Pos} or {Batch, Pos})
        kv_cache: KvPageCache,
        batch_info: PageBatchInfo,
        pos_ids: NamedArray,
        *,
        key=None,
    ) -> tuple[NamedArray, KvPageCache]:
        """Run one decode / pre-fill step with an existing paged-KV *state*.

        Parameters
        ----------
        input_ids : NamedArray
            Token IDs for the positions being decoded **this call**.
        kv_cache : KvPageCache
            Current paged-KV cache (one per layer). Obtain the initial value via
            ``self.initial_cache`` and update with the returned *new_state* each step.
        pos_ids : NamedArray
            Absolute position IDs matching *input_ids* (negative IDs can mark padding as
            in the lower-level API).
        key : jax.random.PRNGKey | None
            RNG key for dropout etc.  Can be omitted during inference.

        Returns
        -------
        logits : NamedArray
            Logits for the provided tokens (axes match *input_ids* + ``Vocab``).
        new_state : KvPageCache
            Updated cache to pass into the next decode call.
        """

        # Embed the incoming token IDs
        x = self.embeddings.embed(input_ids)

        # Propagate through the transformer with paged-KV caching
        k_t = maybe_rng_split(key, 1)[0] if key is not None else None
        x, new_state = self.transformer.decode(kv_cache, x, batch_info, pos_ids, key=k_t)

        # Project to logits
        if self.lm_head is not None:
            logits = self.lm_head(x, key=None)
        else:
            logits = self.embeddings.unembed(x)

        return logits, new_state
