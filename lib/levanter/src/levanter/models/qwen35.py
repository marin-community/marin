# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 model implementation.

Qwen3.5 is a hybrid Gated DeltaNet + Transformer architecture:
- Every `full_attention_interval` layers use full attention with packed output gate
- All other layers use GatedDeltaNet (linear attention)
- Partial rotary embeddings (partial_rotary_factor=0.25)
- (1+w) RMSNorm (Gemma-style)
- QK-normalization on attention layers
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Type, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.scan import BlockSeq, ScanCheckpointPolicy
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers.attention import AttentionBackend, AttentionConfig, AttentionMask, dot_product_attention
from levanter.layers.gated_deltanet import GatedDeltaNet, GatedDeltaNetConfig
from levanter.layers.rotary import PartialRotaryEmbeddings, PartialRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.gemma import GemmaNormConfig, GemmaRMSNorm
from levanter.models.llama import LlamaMlp
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag
from levanter.utils.types import BlockFoldable

silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402


@LmConfig.register_subclass("qwen35")
@dataclass(frozen=True)
class Qwen35Config(HFCompatConfig):
    """Config for Qwen3.5 hybrid GDN + Transformer model."""

    max_seq_len: int = 262144
    hidden_dim: int = 1024
    intermediate_dim: int = 3584
    num_layers: int = 24
    num_heads: int = 8
    num_kv_heads: int = 2
    head_dim: int = 256

    # GDN-specific params
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4

    # Architecture
    full_attention_interval: int = 4
    layer_types: Optional[Sequence[str]] = None

    # Standard
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.silu
    layer_norm_epsilon: float = 1e-6
    tie_word_embeddings: bool = True
    use_bias: bool = False
    vocab_size: int = 248320

    # RoPE
    rope: RotaryEmbeddingsConfig = field(
        default_factory=lambda: PartialRotaryEmbeddingsConfig(theta=10_000_000.0, partial_rotary_factor=0.25)
    )

    # Training
    gradient_checkpointing: Union[bool, str] = True
    scan_layers: bool = True

    # Tokenizer/reference
    reference_checkpoint: str = ""
    tokenizer: Optional[str] = None

    # HF compat
    hf_max_position_embeddings: Optional[int] = None

    # Attention backend
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None
    upcast_attn: bool = False

    @property
    def model_type(self) -> Type["Qwen35LMHeadModel"]:
        return Qwen35LMHeadModel

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def Layers(self) -> Axis:
        return Axis("layer", self.num_layers)

    @property
    def Mlp(self) -> Axis:
        return Axis("mlp", self.intermediate_dim)

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.vocab_size)

    @property
    def actual_head_size(self) -> int:
        return self.head_dim

    @property
    def norm_config(self) -> GemmaNormConfig:
        return GemmaNormConfig(eps=self.layer_norm_epsilon)

    def mk_LayerNorm(self, axis) -> GemmaRMSNorm:
        return self.norm_config.build(axis)

    def get_layer_types(self) -> Sequence[str]:
        if self.layer_types is not None:
            if len(self.layer_types) != self.num_layers:
                raise ValueError("layer_types must match num_layers")
            return list(self.layer_types)
        return [
            "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
            for i in range(self.num_layers)
        ]

    def attention_config(self) -> AttentionConfig:
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

    def gdn_config(self) -> GatedDeltaNetConfig:
        return GatedDeltaNetConfig(
            Embed=self.Embed,
            num_k_heads=self.linear_num_key_heads,
            num_v_heads=self.linear_num_value_heads,
            head_k_dim=self.linear_key_head_dim,
            head_v_dim=self.linear_value_head_dim,
            conv_kernel_size=self.linear_conv_kernel_dim,
            rms_norm_eps=self.layer_norm_epsilon,
        )

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

    def hf_checkpoint_converter(self) -> HFCheckpointConverter:
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint,
            trust_remote_code=True,
            tokenizer=self.tokenizer or self.reference_checkpoint,
            HfConfigClass=HfConfig,  # bypass AutoConfig (qwen3_5 not in transformers <5.0)
            ignore_prefix="model",
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "Qwen35Config":
        # Handle multimodal wrapper: text params are under text_config
        tc_raw = getattr(hf_config, "text_config", None)
        if tc_raw is None:
            tc_raw = hf_config

        # Unified accessor: text_config may be a dict or an object with attributes
        def _g(key, default=None):
            if isinstance(tc_raw, dict):
                return tc_raw.get(key, default)
            return getattr(tc_raw, key, default)

        # Parse rope from rope_parameters (not rope_scaling)
        rope_params = _g("rope_parameters")
        if rope_params is not None:
            if not isinstance(rope_params, dict):
                rope_params = rope_params.to_dict() if hasattr(rope_params, "to_dict") else vars(rope_params)
            rope_theta = rope_params.get("rope_theta", 10_000_000.0)
            partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25)
            rope = PartialRotaryEmbeddingsConfig(theta=rope_theta, partial_rotary_factor=partial_rotary_factor)
        else:
            rope = PartialRotaryEmbeddingsConfig(
                theta=_g("rope_theta", 10_000_000.0), partial_rotary_factor=_g("partial_rotary_factor", 0.25)
            )

        layer_types = _g("layer_types")
        if layer_types is not None:
            layer_types = tuple(layer_types)

        tie = _g("tie_word_embeddings", None)
        if tie is None:
            tie = getattr(hf_config, "tie_word_embeddings", True)

        return Qwen35Config(
            max_seq_len=_g("max_position_embeddings", 262144),
            hidden_dim=_g("hidden_size"),
            intermediate_dim=_g("intermediate_size"),
            num_layers=_g("num_hidden_layers"),
            num_heads=_g("num_attention_heads"),
            num_kv_heads=_g("num_key_value_heads", _g("num_attention_heads")),
            head_dim=_g("head_dim", 256),
            linear_num_key_heads=_g("linear_num_key_heads", 16),
            linear_num_value_heads=_g("linear_num_value_heads", 16),
            linear_key_head_dim=_g("linear_key_head_dim", 128),
            linear_value_head_dim=_g("linear_value_head_dim", 128),
            linear_conv_kernel_dim=_g("linear_conv_kernel_dim", 4),
            full_attention_interval=_g("full_attention_interval", 4),
            layer_types=layer_types,
            activation_function=ActivationFunctionEnum.silu,
            layer_norm_epsilon=_g("rms_norm_eps", 1e-6),
            tie_word_embeddings=tie,
            vocab_size=_g("vocab_size", 248320),
            rope=rope,
            hf_max_position_embeddings=_g("max_position_embeddings", 262144),
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfConfig:
        rope_theta, rope_dict = self.rope.to_hf_config()

        text_config: dict = {
            "hidden_size": self.hidden_dim,
            "intermediate_size": self.intermediate_dim,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "max_position_embeddings": self.hf_max_position_embeddings or self.max_seq_len,
            "rms_norm_eps": self.layer_norm_epsilon,
            "tie_word_embeddings": self.tie_word_embeddings,
            "vocab_size": vocab_size,
            "hidden_act": "silu",
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
            "full_attention_interval": self.full_attention_interval,
            "attn_output_gate": True,
            "layer_types": list(self.get_layer_types()),
            "model_type": "qwen3_5_text",
        }
        if rope_dict is not None:
            text_config["rope_parameters"] = {**rope_dict, "rope_theta": rope_theta}
        else:
            text_config["rope_theta"] = rope_theta

        hf_dict = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": text_config,
            "tie_word_embeddings": self.tie_word_embeddings,
        }
        if config_overrides:
            hf_dict.update(config_overrides)

        return HfConfig.from_dict(hf_dict)


class Qwen35Attention(ModuleWithStateDictSerialization, eqx.Module):
    """Qwen3.5 attention with packed query+gate in q_proj.

    q_proj outputs (num_heads * head_dim * 2): first half is query, second half is gate.
    Gate is applied as sigmoid(gate) * attn_output before o_proj.
    """

    config: AttentionConfig = eqx.field(static=True)
    q_proj: hnn.Linear  # Out: (KVHeads, QHeadsPerGroup, PackedDim) where PackedDim = head_dim * 2
    k_proj: hnn.Linear  # Out: (KVHeads, HeadSize)
    v_proj: hnn.Linear  # Out: (KVHeads, HeadSize)
    o_proj: hnn.Linear  # In: (Heads, HeadSize), Out: Embed
    q_norm: GemmaRMSNorm
    k_norm: GemmaRMSNorm
    rot_embs: Optional[PartialRotaryEmbeddings] = None

    @staticmethod
    def init(config: AttentionConfig, norm_config: GemmaNormConfig, *, key) -> "Qwen35Attention":
        k_q, k_k, k_v, k_o = jrandom.split(key, 4)
        PackedDim = Axis("packed", config.head_size * 2)

        q_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.QHeadsPerGroup, PackedDim),
            key=k_q,
            use_bias=False,
            out_first=True,
        )
        k_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.HeadSize),
            key=k_k,
            use_bias=False,
            out_first=True,
        )
        v_proj = hnn.Linear.init(
            In=config.Embed,
            Out=(config.KVHeads, config.HeadSize),
            key=k_v,
            use_bias=False,
            out_first=True,
        )
        o_proj = hnn.Linear.init(
            In=(config.Heads, config.HeadSize),
            Out=config.Embed,
            key=k_o,
            use_bias=False,
            out_first=True,
        )
        q_norm = norm_config.build(config.HeadSize)
        k_norm = norm_config.build(config.HeadSize)
        rot_embs = config.rope.build(config.HeadSize) if config.rope is not None else None

        return Qwen35Attention(config, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, rot_embs)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask],
        *,
        key=None,
        pos_ids=None,
    ) -> NamedArray:
        PackedDim = Axis("packed", self.config.head_size * 2)
        HeadSize = self.config.HeadSize

        # Project Q+gate packed, K, V
        qg = self.q_proj(x)  # (..., KVHeads, QHeadsPerGroup, PackedDim)
        k = self.k_proj(x)  # (..., KVHeads, HeadSize)
        v = self.v_proj(x)

        # Split packed dim into query and gate, rename to head_size
        q = qg[PackedDim, : HeadSize.size].rename({"packed": HeadSize.name})
        gate = qg[PackedDim, HeadSize.size :].rename({"packed": HeadSize.name})

        # QK-norm (applied to q, not gate)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Partial RoPE
        if self.rot_embs is not None:
            if pos_ids is None:
                pos_ids = hax.arange(x.resolve_axis("position"))
            q = self.rot_embs(q, pos_ids).astype(q.dtype)
            k = self.rot_embs(k, pos_ids).astype(k.dtype)

        # Reshape for attention
        q = q.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        k = k.rearrange((..., "kv_head", "position", "head_size"))
        v = v.rearrange((..., "kv_head", "position", "head_size"))
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            attn_backend=self.config.attn_backend,
            flash_block_size=self.config.flash_attention_block_size,
            inference=True,
            prng=key,
        )

        # Apply output gate: sigmoid(gate) * attn_output
        gate = hax.nn.sigmoid(gate)
        gate = gate.rearrange((..., "kv_head", "q_heads_per_group", "position", "head_size"))
        attn_output = attn_output * gate

        # Flatten heads and project output
        attn_output = attn_output.flatten_axes(("kv_head", "q_heads_per_group"), "heads")
        attn_output = attn_output.astype(x.dtype)
        return self.o_proj(attn_output)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {}


class Qwen35DecoderLayer(ModuleWithStateDictSerialization, eqx.Module):
    """Qwen3.5 decoder layer — either full attention or GatedDeltaNet.

    Uses Optional fields: one is set, the other is None. Haliax state dict
    skips None fields, so keys only match the active module.
    """

    config: Qwen35Config = eqx.field(static=True)
    layer_type: str = eqx.field(static=True)
    mlp: LlamaMlp
    input_layernorm: GemmaRMSNorm
    post_attention_layernorm: GemmaRMSNorm
    self_attn: Optional[Qwen35Attention] = None
    linear_attn: Optional[GatedDeltaNet] = None

    @staticmethod
    def init(config: Qwen35Config, layer_idx: int, *, key) -> "Qwen35DecoderLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        layer_types = config.get_layer_types()
        layer_type = layer_types[layer_idx]

        self_attn = None
        linear_attn = None

        if layer_type == "full_attention":
            self_attn = Qwen35Attention.init(config.attention_config(), config.norm_config, key=k_attn)
        else:
            linear_attn = GatedDeltaNet.init(config.gdn_config(), key=k_attn)

        mlp = LlamaMlp.init(config.Embed, config.Mlp, config.activation_function, key=k_mlp, use_bias=False)
        input_ln = config.mk_LayerNorm(config.Embed)
        post_attn_ln = config.mk_LayerNorm(config.Embed)

        return Qwen35DecoderLayer(config, layer_type, mlp, input_ln, post_attn_ln, self_attn, linear_attn)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        mask: Optional[NamedArray | AttentionMask] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)

        residual = x
        x = self.input_layernorm(x)

        if self.layer_type == "full_attention":
            assert self.self_attn is not None
            x = self.self_attn(x, mask, key=k_attn, pos_ids=pos_ids)
        else:
            assert self.linear_attn is not None
            x, _ = self.linear_attn(x, inference=False, chunk_size=64)

        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x, key=k_mlp)
        x = residual + x

        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {}


class Qwen35Transformer(ModuleWithStateDictSerialization, eqx.Module):
    config: Qwen35Config = eqx.field(static=True)
    _layers: BlockFoldable[Qwen35DecoderLayer]
    norm: GemmaRMSNorm

    @staticmethod
    def init(config: Qwen35Config, *, key) -> "Qwen35Transformer":
        # Always use BlockSeq — layers are heterogeneous (GDN + attention)
        keys = shaped_rng_split(key, config.num_layers)
        blocks = [Qwen35DecoderLayer.init(config, layer_idx=i, key=keys[i]) for i in range(config.num_layers)]
        layers = BlockSeq(blocks, config.Layers, ScanCheckpointPolicy._mk(config.gradient_checkpointing))
        norm = config.mk_LayerNorm(config.Embed)
        return Qwen35Transformer(config, layers, norm)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        attn_mask: Optional[NamedArray | AttentionMask],
        *,
        key,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        x = self._layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids)  # type: ignore[assignment]
        x = self.norm(x)
        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"_layers": "layers"}


class Qwen35Embedding(ModuleWithStateDictSerialization, eqx.Module):
    """Embedding for Qwen3.5 — maps to model.language_model.embed_tokens."""

    token_embeddings: hnn.Embedding

    @staticmethod
    def init(Vocab: Axis, config: Qwen35Config, *, key) -> "Qwen35Embedding":
        token_embeddings = hnn.Embedding.init(Vocab, config.Embed, key=key)
        return Qwen35Embedding(token_embeddings)

    def embed(self, input_ids) -> NamedArray:
        return self.token_embeddings(input_ids)

    def unembed(self, x: NamedArray) -> NamedArray:
        return self.token_embeddings.unembed(x)

    @property
    def Vocab(self) -> Axis:
        return self.token_embeddings.Vocab

    def resize_embeddings(self, new_size: int, key=None):
        new_token_embeddings = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, token_embeddings=new_token_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "language_model.embed_tokens"}


class Qwen35LMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[Qwen35Config]):
    transformer: Qwen35Transformer
    embeddings: Qwen35Embedding
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
    def init(cls, Vocab: Axis, config: Qwen35Config, *, key) -> "Qwen35LMHeadModel":
        k_t, k_emb, k_head = jrandom.split(key, 3)
        transformer = Qwen35Transformer.init(config, key=k_t)
        embeddings = Qwen35Embedding.init(Vocab, config, key=k_emb)
        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_head, use_bias=False, out_first=True)
        return Qwen35LMHeadModel(transformer, embeddings, lm_head)

    @named_call
    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[NamedArray | AttentionMask] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask, key=key, pos_ids=pos_ids)
        if self.lm_head is not None:
            return self.lm_head(x)
        return self.embeddings.unembed(x)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: Optional[NamedArray | AttentionMask] = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        x = self.embeddings.embed(input_ids)
        x = self.transformer(x, attn_mask, key=key, pos_ids=pos_ids)
        return x

    def get_lm_head(self) -> NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "Qwen35LMHeadModel":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        if self.lm_head is not None:
            new_lm_head = self.lm_head.resize_axis("vocab", new_size, key=key)
            return dataclasses.replace(self, embeddings=new_embeddings, lm_head=new_lm_head)
        return dataclasses.replace(self, embeddings=new_embeddings)

    @staticmethod
    def load_from_hf_checkpoint(ref: str, *, config: Optional[Qwen35Config] = None) -> "Qwen35LMHeadModel":
        """Load a Qwen3.5 model from a HuggingFace checkpoint.

        Handles the weight format conversion between HF's flat tensors and
        Levanter's articulated named arrays, including the GDN's split projections.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import json

        # Load config if not provided
        if config is None:
            cfg_path = hf_hub_download(ref, "config.json")
            with open(cfg_path) as f:
                raw = json.load(f)
            from transformers import PretrainedConfig as _HfConfig

            config = Qwen35Config.from_hf_config(_HfConfig.from_dict(raw))

        # Create model template
        Vocab = Axis("vocab", config.vocab_size)
        model = Qwen35LMHeadModel.init(Vocab, config, key=jax.random.PRNGKey(0))

        # Load safetensors
        try:
            idx_path = hf_hub_download(ref, "model.safetensors.index.json")
            with open(idx_path) as f:
                idx = json.load(f)
            shard_files = set(idx["weight_map"].values())
        except Exception:
            shard_files = {"model.safetensors"}

        hf_state = {}
        for shard in shard_files:
            shard_path = hf_hub_download(ref, shard)
            hf_state.update(load_file(shard_path))

        # Strip 'model.' prefix and convert to numpy.
        # Also keep top-level keys (e.g. lm_head.weight for non-tied models).
        state_dict = {}
        for k, v in hf_state.items():
            if k.startswith("model."):
                state_dict[k[len("model.") :]] = v.float().numpy()
            elif not k.startswith("mtp."):
                state_dict[k] = v.float().numpy()

        # Build Levanter-compatible state dict with proper shapes
        lev_sd: dict = {}
        layer_types = config.get_layer_types()

        for i, lt in enumerate(layer_types):
            lp = f"language_model.layers.{i}"

            # Norms + MLP — direct copy (shapes already match)
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ]:
                key = f"{lp}.{suffix}"
                lev_sd[key] = state_dict[key]

            if lt == "full_attention":
                # Attention: copy flat weights as-is. The flatten/unflatten pipeline
                # (from_torch_compatible_state_dict) handles 2D↔articulated conversion.
                ap = f"{lp}.self_attn"
                for suffix in [
                    "q_proj.weight",
                    "k_proj.weight",
                    "v_proj.weight",
                    "o_proj.weight",
                    "q_norm.weight",
                    "k_norm.weight",
                ]:
                    lev_sd[f"{ap}.{suffix}"] = state_dict[f"{ap}.{suffix}"]

            else:
                # GDN: use the GDN's repacking to convert split → packed format
                gdn_prefix = f"{lp}.linear_attn"
                gdn = model.transformer._layers.blocks[i].linear_attn
                packed = gdn._repack_hf_split_to_packed(state_dict, gdn_prefix)

                # packed dict has keys without prefix — add prefix back
                for pk, pv in packed.items():
                    lev_sd[f"{gdn_prefix}.{pk}"] = pv

        # Embedding
        lev_sd["language_model.embed_tokens.weight"] = state_dict["language_model.embed_tokens.weight"]
        # Final norm
        lev_sd["language_model.norm.weight"] = state_dict["language_model.norm.weight"]
        # LM head (non-tied models like 9B)
        if not config.tie_word_embeddings:
            lm_head_key = "language_model.lm_head.weight"
            if lm_head_key not in state_dict:
                lm_head_key = "lm_head.weight"
            lev_sd["lm_head.weight"] = state_dict[lm_head_key]

        # Use from_torch_compatible_state_dict which handles the
        # flatten/unflatten pipeline for Linear 2D↔articulated conversion.
        from haliax.state_dict import from_torch_compatible_state_dict

        return from_torch_compatible_state_dict(model, lev_sd, unflatten=True)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "language_model", "embeddings": None}
