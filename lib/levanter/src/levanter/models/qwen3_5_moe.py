# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Text decoder of the Qwen3.5 and Qwen3.6 mixture-of-experts models, such as Qwen/Qwen3.6-35B-A3B.

The decoder repeats a group of `full_attention_interval - 1` Gated DeltaNet (linear attention) layers
followed by one gated softmax-attention layer. Every layer ends in a sparse mixture of experts with an
always-on shared expert. Parameters keep Hugging Face's `Qwen3_5MoeForCausalLM` names and shapes.

The released checkpoints are vision-language models. This module loads their text decoder and drops
the vision tower and the multi-token-prediction head, so exports are text-only.
"""

import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, Optional, Type, cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split, named_call, shaped_rng_split
from haliax.nn.normalization import LayerNormBase
from haliax.nn.scan import BlockFoldable, BlockSeq, Stacked
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict, with_prefix

from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.compat.hf_config import hf_config_from_kwargs, hf_rope_parameter_mapping
from levanter.layers.attention import AttentionConfig, AttentionMask, GatedAttention
from levanter.layers.gated_deltanet import (
    GatedDeltaNetConfig,
    GatedRmsNorm,
    _causal_depthwise_conv1d_full,
    chunk_gated_delta_rule,
)
from levanter.layers.normalization import LayerNormConfigBase
from levanter.layers.rotary import PartialRotaryEmbeddingsConfig, RotaryEmbeddingsConfig
from levanter.models.gemma import GemmaNormConfig
from levanter.models.llama import LlamaEmbedding, LlamaMlp
from levanter.models.lm_model import LmConfig, LmHeadModel, resize_embeddings_and_lm_head
from levanter.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeExperts, Qwen3MoeSparseMoeBlock
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_attention_flops_per_token
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import Qwen3_5MoeConfig as HfQwen3_5MoeConfig  # noqa: E402
from transformers import Qwen3_5MoeTextConfig as HfQwen3_5MoeTextConfig  # noqa: E402


LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"

# Key layout of the released vision-language checkpoints.
_LANGUAGE_MODEL_PREFIX = "model.language_model."
_TEXT_MODEL_PREFIX = "model."
_NON_TEXT_PREFIXES = ("model.visual.", "mtp.")
_PER_EXPERT_KEY = re.compile(r"(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight")


def _text_config(hf_config: HfConfig) -> HfConfig:
    """Returns the text decoder config of a text-only or vision-language Qwen3.5 MoE config."""
    if isinstance(hf_config, HfQwen3_5MoeConfig):
        return cast(HfConfig, hf_config.text_config)
    return hf_config


def _text_state_dict(state_dict: StateDict) -> StateDict:
    """Keeps the text decoder's tensors and renames them to the text-only `Qwen3_5MoeForCausalLM` layout."""
    text_state_dict: StateDict = {}
    for key, value in state_dict.items():
        if key.startswith(_NON_TEXT_PREFIXES):
            continue
        if key.startswith(_LANGUAGE_MODEL_PREFIX):
            key = _TEXT_MODEL_PREFIX + key.removeprefix(_LANGUAGE_MODEL_PREFIX)
        text_state_dict[key] = value
    return text_state_dict


def _packed_expert_state_dict(state_dict: StateDict) -> StateDict:
    """Packs per-expert weights into the released checkpoints' `gate_up_proj` and `down_proj` tensors.

    Transformers loads either layout, and its `save_pretrained` writes `experts.N.gate_proj.weight`,
    `experts.N.up_proj.weight` and `experts.N.down_proj.weight` for each expert.
    """
    packed: StateDict = {}
    per_expert: dict[str, dict[str, dict[int, jnp.ndarray]]] = {}
    for key, value in state_dict.items():
        match = _PER_EXPERT_KEY.fullmatch(key)
        if match is None:
            packed[key] = value
            continue
        per_expert.setdefault(match[1], {}).setdefault(match[3], {})[int(match[2])] = value

    for experts_prefix, projections in per_expert.items():

        def stacked(projection: str) -> jnp.ndarray:
            weights = projections[projection]
            return jnp.stack([weights[expert] for expert in range(len(weights))])

        packed[f"{experts_prefix}.gate_up_proj"] = jnp.concatenate([stacked("gate_proj"), stacked("up_proj")], axis=1)
        packed[f"{experts_prefix}.down_proj"] = stacked("down_proj")
    return packed


def _layer_types(num_layers: int, full_attention_interval: int) -> list[str]:
    return [
        FULL_ATTENTION if (layer + 1) % full_attention_interval == 0 else LINEAR_ATTENTION
        for layer in range(num_layers)
    ]


def _full_attention_interval(layer_types: list[str]) -> int:
    """Returns the period of a layer pattern made of linear-attention runs that each end in full attention."""
    if FULL_ATTENTION not in layer_types:
        raise ValueError("Qwen3.5 MoE needs at least one full_attention layer.")
    interval = layer_types.index(FULL_ATTENTION) + 1
    if len(layer_types) % interval != 0 or layer_types != _layer_types(len(layer_types), interval):
        raise ValueError(
            f"Unsupported layer_types {layer_types}: expected {interval - 1} linear_attention layers "
            "before each full_attention layer."
        )
    return interval


class Qwen35MoeCheckpointConverter(HFCheckpointConverter["Qwen35MoeConfig"]):
    """Loads the text decoder from text-only and vision-language Qwen3.5 MoE checkpoints."""

    def hf_config_from_hf_checkpoint(self, ref: Optional[str | RepoRef] = None) -> HfConfig:
        return _text_config(super().hf_config_from_hf_checkpoint(ref))

    def load_state_dict(self, ref: Optional[str | RepoRef] = None, dtype: Optional[jnp.dtype] = None) -> dict:
        return _packed_expert_state_dict(_text_state_dict(super().load_state_dict(ref, dtype)))


@LmConfig.register_subclass("qwen3_5_moe")
@dataclass(frozen=True)
class Qwen35MoeConfig(Qwen3MoeConfig):
    """Qwen3.5 and Qwen3.6 MoE text decoder. Defaults describe Qwen/Qwen3.6-35B-A3B.

    Args:
        shared_expert_intermediate_dim: Hidden size of the shared expert that every token uses.
        full_attention_interval: Layers per group. Each group has `full_attention_interval - 1`
            linear-attention layers followed by one full-attention layer.
        linear_num_key_heads: Query and key heads of a linear-attention layer.
        linear_num_value_heads: Value heads of a linear-attention layer. Must be a multiple of the key heads.
        linear_key_head_dim: Size of each linear-attention query and key head.
        linear_value_head_dim: Size of each linear-attention value head.
        linear_conv_kernel_dim: Kernel size of the causal depthwise convolution over queries, keys and values.
        gdn_chunk_size: Chunk length of the chunkwise-parallel delta rule. It changes cost, not the result.
        mrope_section: Multimodal RoPE sections, kept for export. Text positions use plain RoPE.
    """

    max_seq_len: int = 262144
    hidden_dim: int = 2048
    moe_intermediate_dim: int = 512
    shared_expert_intermediate_dim: int = 512
    num_layers: int = 40
    num_heads: int = 16
    head_dim: int | None = 256
    num_kv_heads: int = 2
    num_experts: int = 256
    num_experts_per_tok: int = 8

    full_attention_interval: int = 4
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    gdn_chunk_size: int = 64

    rope: RotaryEmbeddingsConfig = dataclasses.field(
        default_factory=lambda: PartialRotaryEmbeddingsConfig(theta=10_000_000.0, partial_rotary_factor=0.25)
    )
    mrope_section: tuple[int, ...] = (11, 11, 10)

    reference_checkpoint: str = "Qwen/Qwen3.6-35B-A3B"

    def __post_init__(self):
        super().__post_init__()
        if self.num_layers % self.full_attention_interval != 0:
            raise ValueError(
                f"num_layers={self.num_layers} is not a multiple of "
                f"full_attention_interval={self.full_attention_interval}."
            )
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError(
                f"linear_num_value_heads={self.linear_num_value_heads} is not a multiple of "
                f"linear_num_key_heads={self.linear_num_key_heads}."
            )
        if not isinstance(self.rope, PartialRotaryEmbeddingsConfig):
            raise ValueError(f"Qwen3.5 MoE uses partial rotary embeddings, got {type(self.rope).__name__}.")
        if self.use_bias:
            raise NotImplementedError("Qwen3.5 MoE attention biases are not supported.")
        if not self.norm_topk_prob:
            raise ValueError("Qwen3.5 MoE always renormalizes the selected experts' routing weights.")

    @property
    def LayerGroups(self) -> Axis:
        return Axis(name="layer_group", size=self.num_layers // self.full_attention_interval)

    @property
    def SharedMlp(self) -> Axis:
        return Axis(name="mlp", size=self.shared_expert_intermediate_dim)

    @property
    def layer_types(self) -> list[str]:
        return _layer_types(self.num_layers, self.full_attention_interval)

    @property
    def norm_config(self) -> LayerNormConfigBase:
        # Qwen3.5 stores RMSNorm scales as offsets from one, like Gemma.
        return GemmaNormConfig(eps=self.layer_norm_epsilon, use_weight=True, use_bias=False)

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
            qk_norm=self.norm_config,
            gated="elementwise",
        )

    def gated_deltanet_config(self) -> GatedDeltaNetConfig:
        return GatedDeltaNetConfig(
            Embed=self.Embed,
            num_k_heads=self.linear_num_key_heads,
            num_v_heads=self.linear_num_value_heads,
            head_k_dim=self.linear_key_head_dim,
            head_v_dim=self.linear_value_head_dim,
            conv_kernel_size=self.linear_conv_kernel_dim,
            rms_norm_eps=self.layer_norm_epsilon,
        )

    def hf_checkpoint_converter(  # pyrefly: ignore[bad-override]
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["Qwen35MoeConfig"]:  # type: ignore
        return Qwen35MoeCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=True,
            tokenizer=ref_checkpoint if self.tokenizer is None else self.tokenizer,
            HfConfigClass=HfQwen3_5MoeTextConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        text_config = _text_config(hf_config)
        rope_parameters = hf_rope_parameter_mapping(text_config)
        if rope_parameters is None:
            raise ValueError("Qwen3.5 MoE configs must define rope_parameters.")
        return Qwen35MoeConfig(
            max_seq_len=text_config.max_position_embeddings,
            hidden_dim=text_config.hidden_size,
            moe_intermediate_dim=text_config.moe_intermediate_size,
            shared_expert_intermediate_dim=text_config.shared_expert_intermediate_size,
            num_layers=text_config.num_hidden_layers,
            num_heads=text_config.num_attention_heads,
            head_dim=text_config.head_dim,
            num_kv_heads=text_config.num_key_value_heads,
            activation_function=ActivationFunctionEnum(text_config.hidden_act),
            initializer_range=text_config.initializer_range,
            layer_norm_epsilon=text_config.rms_norm_eps,
            tie_word_embeddings=text_config.tie_word_embeddings,
            num_experts=text_config.num_experts,
            num_experts_per_tok=text_config.num_experts_per_tok,
            router_aux_loss_coef=text_config.router_aux_loss_coef,
            use_bias=text_config.attention_bias,
            full_attention_interval=_full_attention_interval(list(text_config.layer_types)),
            linear_num_key_heads=text_config.linear_num_key_heads,
            linear_num_value_heads=text_config.linear_num_value_heads,
            linear_key_head_dim=text_config.linear_key_head_dim,
            linear_value_head_dim=text_config.linear_value_head_dim,
            linear_conv_kernel_dim=text_config.linear_conv_kernel_dim,
            rope=PartialRotaryEmbeddingsConfig(
                theta=float(rope_parameters["rope_theta"]),
                partial_rotary_factor=rope_parameters["partial_rotary_factor"],
            ),
            mrope_section=tuple(rope_parameters.get("mrope_section", (11, 11, 10))),
        )

    def to_hf_config(  # pyrefly: ignore[bad-override]
        self, vocab_size: int, config_overrides: Optional[Dict] = None
    ) -> HfQwen3_5MoeTextConfig:
        if config_overrides is None:
            config_overrides = {}

        rope = cast(PartialRotaryEmbeddingsConfig, self.rope)
        return hf_config_from_kwargs(
            HfQwen3_5MoeTextConfig,
            vocab_size=vocab_size,
            max_position_embeddings=self.max_seq_len,
            hidden_size=self.hidden_dim,
            moe_intermediate_size=self.moe_intermediate_dim,
            shared_expert_intermediate_size=self.shared_expert_intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            head_dim=self.head_dim,
            num_key_value_heads=self.num_kv_heads,
            hidden_act=self.activation_function.name,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.layer_norm_epsilon,
            tie_word_embeddings=self.tie_word_embeddings,
            attention_bias=self.use_bias,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            # HF requires a float; Levanter's None means no load-balancing loss.
            router_aux_loss_coef=0.0 if self.router_aux_loss_coef is None else self.router_aux_loss_coef,
            layer_types=self.layer_types,
            linear_num_key_heads=self.linear_num_key_heads,
            linear_num_value_heads=self.linear_num_value_heads,
            linear_key_head_dim=self.linear_key_head_dim,
            linear_value_head_dim=self.linear_value_head_dim,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": rope.theta,
                "partial_rotary_factor": rope.partial_rotary_factor,
                "mrope_interleaved": True,
                "mrope_section": list(self.mrope_section),
            },
            **config_overrides,
        )

    @property
    def model_type(self) -> Type["Qwen35MoeLMHeadModel"]:  # pyrefly: ignore[bad-override]
        return Qwen35MoeLMHeadModel

    def flops_per_token(self, vocab_size: int, context_length: int) -> float:
        """Forward FLOPs per token, counting the two token mixers separately.

        Full-attention layers use the megatron-lm attention estimate. Gated DeltaNet layers count their
        projections, the depthwise convolution, and per value head the chunked delta rule: three
        `d_k x d_v` state products per token plus the intra-chunk attention over `gdn_chunk_size` tokens.
        """
        embed = self.hidden_dim
        routed = 2 * 3 * embed * self.moe_intermediate_dim * self.num_experts_per_tok
        shared = 2 * 3 * embed * self.shared_expert_intermediate_dim
        router_and_shared_gate = 2 * embed * (self.num_experts + 1)
        mlp = routed + shared + router_and_shared_gate

        head_size = self.actual_head_size
        # q_proj also produces the output gate.
        full_projections = 2 * embed * head_size * (2 * self.num_heads + 2 * self.num_kv_heads + self.num_heads)

        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        qkv_dim = 2 * key_dim + value_dim
        linear_projections = 2 * embed * (qkv_dim + 2 * value_dim + 2 * self.linear_num_value_heads)
        conv = 2 * qkv_dim * self.linear_conv_kernel_dim
        key_head, value_head = self.linear_key_head_dim, self.linear_value_head_dim
        delta_rule_per_head = 6 * key_head * value_head + 2 * self.gdn_chunk_size * (key_head + value_head)
        linear_mixer = linear_projections + conv + self.linear_num_value_heads * delta_rule_per_head

        num_full = self.num_layers // self.full_attention_interval
        num_linear = self.num_layers - num_full
        attention_map = lm_attention_flops_per_token(
            hidden_dim=embed,
            num_heads=self.num_heads,
            num_layers=num_full,
            seq_len=context_length,
            head_dim=head_size,
        ).noncausal
        lm_head = 2 * embed * vocab_size
        mixers = num_full * full_projections + attention_map + num_linear * linear_mixer
        return self.num_layers * mlp + mixers + lm_head

    def total_trainable_params(self, vocab_size):
        embed = self.hidden_dim
        token_embedding = vocab_size * embed
        lm_head = 0 if self.tie_word_embeddings else token_embedding

        router = embed * self.num_experts
        experts = 3 * self.num_experts * embed * self.moe_intermediate_dim
        shared_expert = 3 * embed * self.shared_expert_intermediate_dim + embed
        layer_norms = 2 * embed
        mlp_and_norms = router + experts + shared_expert + layer_norms

        head_size = self.actual_head_size
        # q_proj also produces the output gate, so it is twice the size of a plain query projection.
        q_proj = 2 * embed * head_size * self.num_heads
        kv_proj = 2 * embed * head_size * self.num_kv_heads
        o_proj = head_size * self.num_heads * embed
        full_attention = q_proj + kv_proj + o_proj + 2 * head_size

        key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        qkv_dim = 2 * key_dim + value_dim
        in_proj = embed * (qkv_dim + value_dim + 2 * self.linear_num_value_heads)
        conv = qkv_dim * self.linear_conv_kernel_dim
        decay = 2 * self.linear_num_value_heads
        linear_attention = in_proj + conv + decay + self.linear_value_head_dim + value_dim * embed

        num_full = self.num_layers // self.full_attention_interval
        num_linear = self.num_layers - num_full
        layers = self.num_layers * mlp_and_norms + num_full * full_attention + num_linear * linear_attention
        return layers + embed + token_embedding + lm_head


class Qwen35MoeAttention(GatedAttention, ModuleWithStateDictSerialization):
    """Gated softmax attention whose query projection also produces the output gate, as in the HF checkpoint.

    HF's `q_proj` emits `[query | gate]` for each head. Levanter keeps separate `q_proj` and `gate_proj`
    parameters and packs them only when reading or writing a torch-compatible state dict.
    """

    @staticmethod
    def init(config: AttentionConfig, *, key) -> "Qwen35MoeAttention":
        attention = GatedAttention.init(config, key=key)
        return Qwen35MoeAttention(**{f.name: getattr(attention, f.name) for f in dataclasses.fields(attention)})

    def _per_head_rows(self, weight: jnp.ndarray) -> jnp.ndarray:
        """Splits a flattened projection weight `(..., heads * rows, embed)` into `(..., heads, rows, embed)`."""
        num_heads = self.config.num_heads
        if weight.shape[-2] % (num_heads * self.config.head_size) != 0:
            raise ValueError(
                f"Expected a torch-layout projection with a multiple of {num_heads * self.config.head_size} "
                f"rows, got shape {weight.shape}."
            )
        return weight.reshape(*weight.shape[:-2], num_heads, -1, weight.shape[-1])

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        state_dict = super().to_state_dict(prefix)
        q_key = cast(str, with_prefix(prefix, "q_proj.weight"))
        query = self._per_head_rows(state_dict[q_key])
        gate = self._per_head_rows(state_dict.pop(cast(str, with_prefix(prefix, "gate_proj.weight"))))
        packed = jnp.concatenate([query, gate], axis=-2)
        state_dict[q_key] = packed.reshape(*packed.shape[:-3], -1, packed.shape[-1])
        return state_dict

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "Qwen35MoeAttention":
        q_key = cast(str, with_prefix(prefix, "q_proj.weight"))
        query, gate = jnp.split(self._per_head_rows(state_dict[q_key]), 2, axis=-2)
        unpacked = dict(state_dict)
        unpacked[q_key] = query.reshape(*query.shape[:-3], -1, query.shape[-1])
        unpacked[cast(str, with_prefix(prefix, "gate_proj.weight"))] = gate.reshape(
            *gate.shape[:-3], -1, gate.shape[-1]
        )
        return super().from_state_dict(unpacked, prefix)


class Qwen35MoeGatedDeltaNet(ModuleWithStateDictSerialization):
    """Gated DeltaNet token mixer with Qwen3.5's separate `qkv`, `z`, `b` and `a` input projections.

    Queries, keys and values pass through a causal depthwise convolution and the chunkwise delta rule.
    The result is RMS-normalized, gated by `z` and projected back to the model width. The layer is
    causal, so it needs no attention mask.
    """

    config: GatedDeltaNetConfig = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    in_proj_qkv: hnn.Linear
    in_proj_z: hnn.Linear
    in_proj_b: hnn.Linear
    in_proj_a: hnn.Linear
    conv1d_weight: NamedArray  # [qkv, conv_in, conv_kernel], torch's depthwise Conv1d layout
    A_log: NamedArray  # [v_heads]
    dt_bias: NamedArray  # [v_heads]
    norm: GatedRmsNorm
    out_proj: hnn.Linear

    @staticmethod
    def init(config: GatedDeltaNetConfig, chunk_size: int, *, key) -> "Qwen35MoeGatedDeltaNet":
        k_qkv, k_z, k_b, k_a, k_conv, k_decay, k_out = jrandom.split(key, 7)
        Qkv = Axis("qkv", 2 * config.key_dim + config.value_dim)
        ValueHeads = (config.VHeads, config.VHeadDim)

        def in_proj(Out, key):
            return hnn.Linear.init(In=config.Embed, Out=Out, key=key, use_bias=False, out_first=True)

        conv_shape = (Qkv, Axis("conv_in", 1), Axis("conv_kernel", config.conv_kernel_size))
        conv1d_weight = hax.random.normal(k_conv, conv_shape) * (Qkv.size * config.conv_kernel_size) ** -0.5
        A_log = hax.log(hax.random.uniform(k_decay, config.VHeads, minval=1e-6, maxval=16.0))
        return Qwen35MoeGatedDeltaNet(
            config=config,
            chunk_size=chunk_size,
            in_proj_qkv=in_proj(Qkv, k_qkv),
            in_proj_z=in_proj(ValueHeads, k_z),
            in_proj_b=in_proj(config.VHeads, k_b),
            in_proj_a=in_proj(config.VHeads, k_a),
            conv1d_weight=conv1d_weight,
            A_log=A_log,
            dt_bias=hax.ones(config.VHeads),
            norm=GatedRmsNorm.init(config.VHeadDim, eps=config.rms_norm_eps),
            out_proj=hnn.Linear.init(In=ValueHeads, Out=config.Embed, key=k_out, use_bias=False, out_first=True),
        )

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"conv1d_weight": "conv1d.weight"}

    @named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        cfg = self.config
        batched = x.has_axis("batch")
        if not batched:
            x = x.broadcast_axis(Axis("batch", 1))

        z = self.in_proj_z(x)
        beta = hnn.sigmoid(self.in_proj_b(x))
        # Log-decay per value head: g = -exp(A_log) * softplus(a + dt_bias), computed in fp32 like HF.
        decay_rate = hax.exp(self.A_log.astype(jnp.float32))
        g = -decay_rate * hnn.softplus(self.in_proj_a(x).astype(jnp.float32) + self.dt_bias.astype(jnp.float32))

        Qkv = self.in_proj_qkv.Out
        conv_order = ("batch", Qkv.name, "position")
        qkv = hax.rearrange(self.in_proj_qkv(x), conv_order)
        conv_weight = self.conv1d_weight["conv_in", 0].rearrange((Qkv.name, "conv_kernel"))
        mixed = hax.named(_causal_depthwise_conv1d_full(qkv.array, conv_weight.array), conv_order)

        KeyHeads = (cfg.KHeads, cfg.KHeadDim)
        q = mixed[Qkv.name, hax.ds(0, cfg.key_dim)].unflatten_axis(Qkv.name, KeyHeads)
        k = mixed[Qkv.name, hax.ds(cfg.key_dim, cfg.key_dim)].unflatten_axis(Qkv.name, KeyHeads)
        v = mixed[Qkv.name, hax.ds(2 * cfg.key_dim, cfg.value_dim)].unflatten_axis(
            Qkv.name, (cfg.VHeads, cfg.VHeadDim)
        )

        # Each key head serves `num_v_heads // num_k_heads` consecutive value heads.
        ValueGroup = Axis("v_group", cfg.num_v_heads // cfg.num_k_heads)
        q = q.broadcast_axis(ValueGroup).flatten_axes((cfg.KHeads, ValueGroup), "heads")
        k = k.broadcast_axis(ValueGroup).flatten_axes((cfg.KHeads, ValueGroup), "heads")
        to_heads = {cfg.VHeads.name: "heads"}

        key_order = ("batch", "position", "heads", cfg.KHeadDim.name)
        out, _ = chunk_gated_delta_rule(
            hax.rearrange(q, key_order),
            hax.rearrange(k, key_order),
            hax.rearrange(v.rename(to_heads), ("batch", "position", "heads", cfg.VHeadDim.name)),
            g.rename(to_heads),
            beta.rename(to_heads),
            chunk_size=self.chunk_size,
        )
        out = out.rename({"heads": cfg.VHeads.name})

        y = self.out_proj(self.norm(out, gate=z).astype(x.dtype))
        return y if batched else y["batch", 0]


class Qwen35MoeExperts(Qwen3MoeExperts):
    """Routed experts serialized as HF's packed `gate_up_proj` and `down_proj` tensors."""

    @staticmethod
    def init(  # pyrefly: ignore[bad-override]
        Experts: Axis, Embed: Axis, Mlp: Axis, activation_fn, *, key
    ) -> "Qwen35MoeExperts":
        experts = Qwen3MoeExperts.init(Experts, Embed, Mlp, activation_fn, key=key, use_bias=False)
        return Qwen35MoeExperts(experts.gate_proj, experts.up_proj, experts.down_proj, experts.act)

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        # MoELinear stores (..., experts, in, out); torch stores (experts, out, in).
        gate = jnp.swapaxes(self.gate_proj.weight.array, -1, -2)
        up = jnp.swapaxes(self.up_proj.weight.array, -1, -2)
        return {
            cast(str, with_prefix(prefix, "gate_up_proj")): jnp.concatenate([gate, up], axis=-2),
            cast(str, with_prefix(prefix, "down_proj")): jnp.swapaxes(self.down_proj.weight.array, -1, -2),
        }

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "Qwen35MoeExperts":
        gate_up = jnp.swapaxes(state_dict[cast(str, with_prefix(prefix, "gate_up_proj"))], -1, -2)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        down = jnp.swapaxes(state_dict[cast(str, with_prefix(prefix, "down_proj"))], -1, -2)
        return eqx.tree_at(
            lambda m: [m.gate_proj.weight.array, m.up_proj.weight.array, m.down_proj.weight.array],
            self,
            [gate, up, down],
        )


class Qwen35MoeSparseMoeBlock(ModuleWithStateDictSerialization):
    """Top-k routed experts plus a shared expert scaled by a per-token sigmoid gate."""

    routed_experts: Qwen3MoeSparseMoeBlock
    shared_expert: LlamaMlp
    shared_expert_gate: hnn.Linear

    @staticmethod
    def init(config: Qwen35MoeConfig, *, key) -> "Qwen35MoeSparseMoeBlock":
        k_router, k_experts, k_shared, k_shared_gate = jrandom.split(key, 4)
        activation = config.activation_function.to_fn()
        router = hnn.Linear.init(config.Embed, config.Experts, key=k_router, use_bias=False)
        experts = Qwen35MoeExperts.init(config.Experts, config.Embed, config.MoeMlp, activation, key=k_experts)
        shared_expert = LlamaMlp.init(config.Embed, config.SharedMlp, activation, key=k_shared, use_bias=False)
        shared_expert_gate = hnn.Linear.init(
            In=config.Embed, Out=Axis("shared_expert_gate", 1), key=k_shared_gate, use_bias=False, out_first=True
        )
        return Qwen35MoeSparseMoeBlock(
            Qwen3MoeSparseMoeBlock(config, router, experts), shared_expert, shared_expert_gate
        )

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        # HF keeps the router (`gate`) and `experts` beside the shared expert.
        return {"routed_experts": None}

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> tuple[NamedArray, Dict[str, NamedArray]]:
        k_routed, k_shared, k_shared_gate = maybe_rng_split(key, 3)
        routed, extras = self.routed_experts(x, key=k_routed)
        shared_gate = hnn.sigmoid(self.shared_expert_gate(x, key=k_shared_gate))["shared_expert_gate", 0]
        return routed + shared_gate * self.shared_expert(x, key=k_shared), extras


class Qwen35MoeLinearAttentionLayer(eqx.Module):
    linear_attn: Qwen35MoeGatedDeltaNet
    mlp: Qwen35MoeSparseMoeBlock
    input_layernorm: LayerNormBase
    post_attention_layernorm: LayerNormBase

    @staticmethod
    def init(config: Qwen35MoeConfig, *, key) -> "Qwen35MoeLinearAttentionLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        return Qwen35MoeLinearAttentionLayer(
            Qwen35MoeGatedDeltaNet.init(config.gated_deltanet_config(), config.gdn_chunk_size, key=k_attn),
            Qwen35MoeSparseMoeBlock.init(config, key=k_mlp),
            config.mk_LayerNorm(config.Embed),
            config.mk_LayerNorm(config.Embed),
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> tuple[NamedArray, Dict[str, NamedArray]]:
        x = x + self.linear_attn(self.input_layernorm(x))
        mlp_out, extras = self.mlp(self.post_attention_layernorm(x), key=key)
        return x + mlp_out, extras


class Qwen35MoeFullAttentionLayer(eqx.Module):
    self_attn: Qwen35MoeAttention
    mlp: Qwen35MoeSparseMoeBlock
    input_layernorm: LayerNormBase
    post_attention_layernorm: LayerNormBase

    @staticmethod
    def init(config: Qwen35MoeConfig, *, key) -> "Qwen35MoeFullAttentionLayer":
        k_attn, k_mlp = jrandom.split(key, 2)
        return Qwen35MoeFullAttentionLayer(
            Qwen35MoeAttention.init(config.attention_config(), key=k_attn),
            Qwen35MoeSparseMoeBlock.init(config, key=k_mlp),
            config.mk_LayerNorm(config.Embed),
            config.mk_LayerNorm(config.Embed),
        )

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> tuple[NamedArray, Dict[str, NamedArray]]:
        k_attn, k_mlp = maybe_rng_split(key, 2)
        x = x + self.self_attn(x=self.input_layernorm(x), mask=mask, key=k_attn, pos_ids=pos_ids)
        mlp_out, extras = self.mlp(self.post_attention_layernorm(x), key=k_mlp)
        return x + mlp_out, extras


class Qwen35MoeLayerGroup(eqx.Module):
    """One period of the layer pattern: the linear-attention layers, then the full-attention layer.

    Grouping makes the heterogeneous stack a sequence of identical blocks that `Stacked` can scan.
    """

    linear_layers: list[Qwen35MoeLinearAttentionLayer]
    full_layer: Qwen35MoeFullAttentionLayer

    @staticmethod
    def init(config: Qwen35MoeConfig, *, key) -> "Qwen35MoeLayerGroup":
        *k_linear, k_full = jrandom.split(key, config.full_attention_interval)
        return Qwen35MoeLayerGroup(
            [Qwen35MoeLinearAttentionLayer.init(config, key=k) for k in k_linear],
            Qwen35MoeFullAttentionLayer.init(config, key=k_full),
        )

    @named_call
    def __call__(
        self, x: NamedArray, mask: Optional[NamedArray | AttentionMask], *, key=None, pos_ids: NamedArray | None = None
    ) -> tuple[NamedArray, Dict[str, NamedArray]]:
        *k_linear, k_full = maybe_rng_split(key, len(self.linear_layers) + 1)
        linear_extras = []
        for layer, k in zip(self.linear_layers, k_linear, strict=True):
            x, extras = layer(x, key=k)
            linear_extras.append(extras)
        x, group_extras = self.full_layer(x, mask, key=k_full, pos_ids=pos_ids)
        for extras in linear_extras:
            group_extras = {name: value + extras[name] for name, value in group_extras.items()}
        return x, group_extras


def _hf_layer_keys(state_dict: StateDict, layers_prefix: str, interval: int) -> StateDict:
    """Renames grouped keys (`layers.G.linear_layers.J.*`, `layers.G.full_layer.*`) to HF's flat `layers.N.*`."""
    escaped = re.escape(layers_prefix)
    linear_key = re.compile(rf"{escaped}\.(\d+)\.linear_layers\.(\d+)\.(.*)")
    full_key = re.compile(rf"{escaped}\.(\d+)\.full_layer\.(.*)")
    renamed: StateDict = {}
    for key, value in state_dict.items():
        if match := linear_key.fullmatch(key):
            layer = int(match[1]) * interval + int(match[2])
            key = f"{layers_prefix}.{layer}.{match[3]}"
        elif match := full_key.fullmatch(key):
            layer = int(match[1]) * interval + interval - 1
            key = f"{layers_prefix}.{layer}.{match[2]}"
        renamed[key] = value
    return renamed


def _grouped_layer_keys(state_dict: StateDict, layers_prefix: str, interval: int) -> StateDict:
    """Inverse of `_hf_layer_keys`."""
    layer_key = re.compile(rf"{re.escape(layers_prefix)}\.(\d+)\.(.*)")
    renamed: StateDict = {}
    for key, value in state_dict.items():
        if match := layer_key.fullmatch(key):
            group, offset = divmod(int(match[1]), interval)
            member = "full_layer" if offset == interval - 1 else f"linear_layers.{offset}"
            key = f"{layers_prefix}.{group}.{member}.{match[2]}"
        renamed[key] = value
    return renamed


class Qwen35MoeTransformer(ModuleWithStateDictSerialization):
    config: Qwen35MoeConfig = eqx.field(static=True)
    layers: BlockFoldable[Qwen35MoeLayerGroup]
    norm: LayerNormBase

    @staticmethod
    def init(config: Qwen35MoeConfig, *, key) -> "Qwen35MoeTransformer":
        S = Stacked if config.scan_layers else BlockSeq
        layers = S.init(config.LayerGroups, Qwen35MoeLayerGroup, gradient_checkpointing=config.gradient_checkpointing)(
            config, key=shaped_rng_split(key, config.LayerGroups.size)
        )
        return Qwen35MoeTransformer(config, layers, config.mk_LayerNorm(config.Embed))

    @named_call
    def __call__(
        self, x: NamedArray, attn_mask: AttentionMask | None, *, key, pos_ids: NamedArray | None = None
    ) -> tuple[NamedArray, Dict[str, NamedArray]]:
        keys = maybe_rng_split(key, self.config.LayerGroups.size) if key is not None else None
        x, extras = cast(
            tuple[NamedArray, Dict[str, NamedArray]],
            self.layers.scan(x, mask=attn_mask, key=keys, pos_ids=pos_ids),
        )
        extras = {name: hax.sum(value, axis=self.config.LayerGroups) for name, value in extras.items()}
        return self.norm(x), extras

    def to_state_dict(self, prefix: Optional[str] = None) -> StateDict:
        state_dict = super().to_state_dict(prefix)
        return _hf_layer_keys(
            state_dict, cast(str, with_prefix(prefix, "layers")), self.config.full_attention_interval
        )

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> "Qwen35MoeTransformer":
        grouped = _grouped_layer_keys(
            state_dict, cast(str, with_prefix(prefix, "layers")), self.config.full_attention_interval
        )
        return super().from_state_dict(grouped, prefix)


class Qwen35MoeLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[Qwen35MoeConfig]):
    transformer: Qwen35MoeTransformer
    embeddings: LlamaEmbedding
    lm_head: Optional[hnn.Linear]

    @property
    def config(self):
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: Qwen35MoeConfig, *, key) -> "Qwen35MoeLMHeadModel":
        k_t, k_emb = jrandom.split(key, 2)
        transformer = Qwen35MoeTransformer.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)
        lm_head = None
        if not config.tie_word_embeddings:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)
        return Qwen35MoeLMHeadModel(transformer, embeddings, lm_head)

    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | None = None,
        pos_ids: NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        k_t, k_head = maybe_rng_split(key, 2)
        x, _ = self.activations(input_ids, attn_mask=attn_mask, key=k_t, pos_ids=pos_ids)
        if self.lm_head is not None:
            return self.lm_head(x, key=k_head)
        return self.embeddings.unembed(x)

    def activations(  # pyrefly: ignore[bad-override]  # narrows activations return type for this MoE head
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> tuple[NamedArray, NamedArray | float]:
        x = self.embeddings.embed(input_ids)
        x, extras = self.transformer(x, attn_mask=attn_mask, key=key, pos_ids=pos_ids)
        return x, extras.get("load_balancing_loss", 0)

    def get_lm_head(self) -> hax.NamedArray:
        if self.lm_head is None:
            return self.embeddings.token_embeddings.weight
        return self.lm_head.weight

    def resize_vocab(self, new_size: int, key=None) -> "LmHeadModel[Qwen35MoeConfig]":
        new_embeddings, new_lm_head = resize_embeddings_and_lm_head(
            self.Vocab, self.embeddings, self.lm_head, new_size, key
        )
        if new_lm_head is None:
            return eqx.tree_at(lambda m: m.embeddings, self, new_embeddings)
        return eqx.tree_at(lambda m: (m.embeddings, m.lm_head), self, (new_embeddings, new_lm_head))

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": "model", "embeddings": None}
