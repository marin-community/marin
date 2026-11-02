"""Megatron-Bridge provider and weight mappings for Grug MoE checkpoints.

Importing this module registers ``GrugMoeBridge`` with Megatron-Bridge so
``AutoBridge.from_hf_pretrained`` resolves Grug checkpoints. The provider
turns the HF config into Megatron-Core settings (sliding window with per-layer
skips, per-layer RoPE skips, half-RoPE, grouped experts with a shared expert,
sigmoid routing with a persistent fp32 expert bias) and builds the pipeline
stage from ``grug_megatron.grug_block_spec``.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import extract_expert_number_from_param
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import get_pg_rank

from skyrl_train.models.grug_megatron import GrugGPTModel, grug_block_spec
from skyrl_train.models.grug_moe import (
    GRUG_DEFAULT_GLOBAL_EVERY,
    GRUG_MOE_MODEL_TYPE,
    GRUG_SPLIT_EXPERT_SCHEMA_VERSION,
    grug_long_layer_flags,
)

GRUG_ROTARY_PERCENT = 0.5


@dataclass
class GrugModelProvider(GPTModelProvider):
    """Megatron-Core model provider for Grug MoE."""

    grug_qk_mult: float = 1.0
    grug_qk_mult_long_scale: float = 1.0
    grug_global_every: int = GRUG_DEFAULT_GLOBAL_EVERY
    grug_local_kv_heads: int = 1
    grug_global_kv_heads: int = 1
    grug_num_shared_experts: int = 1
    grug_sconv_kernel: int = 4
    grug_sconv_sites: tuple[str, ...] = ()
    grug_hero: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GrugGPTModel:
        if self.virtual_pipeline_model_parallel_size:
            raise NotImplementedError("Grug on Megatron does not support virtual pipeline parallelism")
        assert self.vocab_size is not None, "vocab_size must be configured before calling provide()"
        assert not self.should_pad_vocab, "Grug keeps the checkpoint vocab size; vLLM pads it on the serving side"

        for count in (self.num_query_groups, self.grug_local_kv_heads, self.grug_global_kv_heads):
            if self.grug_hero and count % self.tensor_model_parallel_size:
                raise ValueError("Grug stored, local, and global KV head counts must divide evenly over TP")
        if self.grug_hero and self.moe_shared_expert_overlap:
            raise ValueError("Hero shared experts require moe_shared_expert_overlap=False")
        if self.grug_sconv_sites and self.mlp_chunks_for_training != 1:
            raise ValueError("ShortConv requires an unchunked MLP sequence")
        pp_group = self._pg_collection.pp
        if pre_process is None:
            pre_process = is_pp_first_stage(pp_group)
        if post_process is None:
            post_process = is_pp_last_stage(pp_group)
        self._vp_stage = vp_stage
        block_spec = grug_block_spec(self, vp_stage=vp_stage, pp_rank=get_pg_rank(pp_group))
        return GrugGPTModel(
            self,
            transformer_layer_spec=block_spec,
            vocab_size=self.vocab_size,
            max_sequence_length=self.seq_length,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=False,
            position_embedding_type="rope",
            rotary_percent=GRUG_ROTARY_PERCENT,
            rotary_base=self.rotary_base,
            pre_process=pre_process,
            post_process=post_process,
            scatter_embedding_sequence_parallel=self.scatter_embedding_sequence_parallel,
            pg_collection=self._pg_collection,
            vp_stage=vp_stage,
        )


class _StackedExpertExport:
    """Mixin for mappings whose HF side is one stacked ``[E, ...]`` tensor per layer.

    Grouped export re-stacks the per-expert Megatron weights; on import each
    mapping slices its own expert out of the stacked HF tensor.
    """

    is_grouped_export = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The base constructor resets this instance attribute, so set it afterwards.
        self.allow_hf_name_mismatch = True

    @property
    def group_key(self) -> str:
        return str(self.hf_param)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        if megatron_weights is not None:
            megatron_weights = megatron_weights.contiguous()
        return super().megatron_to_hf(megatron_weights, megatron_module)


class GrugStackedExpertMapping(_StackedExpertExport, AutoMapping):
    """Map one Megatron per-expert weight to a slice of Grug's stacked HF tensor."""

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module):
        expert = extract_expert_number_from_param(self.megatron_param)
        return super().hf_to_megatron(hf_weights[expert].contiguous(), megatron_module)


class GrugStackedGatedExpertMapping(_StackedExpertExport, GatedMLPMapping):
    """Map one Megatron fused ``[gate; up]`` expert weight to slices of Grug's stacked gate/up tensors."""

    def hf_to_megatron(self, hf_weights: dict[str, torch.Tensor], megatron_module):
        expert = extract_expert_number_from_param(self.megatron_param)
        sliced = {name: weight[expert].contiguous() for name, weight in hf_weights.items()}
        return super().hf_to_megatron(sliced, megatron_module)


def _routed_expert_mappings(schema_version: int) -> list[AutoMapping | GatedMLPMapping]:
    """Map individually saved or stacked HF experts to Megatron's per-expert weights."""
    megatron = "decoder.layers.*.mlp.experts"
    hf = "model.layers.*.mlp.experts"
    if schema_version == GRUG_SPLIT_EXPERT_SCHEMA_VERSION:
        hf += ".*"
        gate_up_mapping = GatedMLPMapping
        down_mapping = AutoMapping
    else:
        gate_up_mapping = GrugStackedGatedExpertMapping
        down_mapping = GrugStackedExpertMapping
    # Megatron's fc1 fuses HF gate/up weights; fc2 holds the down weight.
    return [
        gate_up_mapping(f"{megatron}.linear_fc1.weight*", gate=f"{hf}.gate_proj.weight", up=f"{hf}.up_proj.weight"),
        down_mapping(f"{megatron}.linear_fc2.weight*", f"{hf}.down_proj.weight"),
    ]


def _gated_norm_mappings(megatron_prefix: str, hf_norm: str, hf_gate_prefix: str) -> list[ReplicatedMapping]:
    return [
        ReplicatedMapping(f"{megatron_prefix}.norm.weight", hf_norm),
        ReplicatedMapping(f"{megatron_prefix}.down_proj.weight", f"{hf_gate_prefix}.down_proj.weight"),
        ReplicatedMapping(f"{megatron_prefix}.up_proj.weight", f"{hf_gate_prefix}.up_proj.weight"),
    ]


@MegatronModelBridge.register_bridge(
    source="GrugMoeForCausalLM",
    target=GrugGPTModel,
    provider=GrugModelProvider,
    model_type=GRUG_MOE_MODEL_TYPE,
)
class GrugMoeBridge(MegatronModelBridge):
    """Convert Grug MoE HF checkpoints to and from the Megatron-Core model."""

    REMOTE_FIRST_DIM_SLICE_PATTERNS = (
        "model.layers.*.mlp.experts.gate_proj.weight",
        "model.layers.*.mlp.experts.up_proj.weight",
        "model.layers.*.mlp.experts.down_proj.weight",
    )

    def provider_bridge(self, hf_pretrained) -> GrugModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        config = hf_pretrained.config
        long_flags = grug_long_layer_flags(config.num_hidden_layers, config.global_every)

        provider.params_dtype = torch.bfloat16
        provider.bf16 = True
        provider.fp16 = False
        provider.normalization = "RMSNorm"
        provider.layernorm_epsilon = config.rms_norm_eps
        provider.activation_func = F.silu
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.share_embeddings_and_output_weights = False
        provider.init_method_std = config.initializer_range

        provider.num_query_groups = config.num_key_value_heads
        provider.kv_channels = config.head_dim
        provider.qk_l2_norm = True  # lets the spec install GrugQKNorm in the q/k norm slots
        provider.position_embedding_type = "rope"
        provider.rotary_percent = GRUG_ROTARY_PERCENT
        provider.rotary_base = config.rope_theta
        provider.window_size = (config.sliding_window - 1, 0)
        provider.window_attn_skip_freq = [0 if is_long else 1 for is_long in long_flags]
        provider.no_rope_freq = [1 if is_long else 0 for is_long in long_flags]
        provider.rotary_interleaved = config.rope_fused
        provider.grug_global_every = config.global_every
        provider.grug_local_kv_heads = config.local_kv_heads or config.num_key_value_heads
        provider.grug_global_kv_heads = config.global_kv_heads or config.num_key_value_heads
        provider.grug_num_shared_experts = config.num_shared_experts
        provider.grug_sconv_sites = config.sconv_sites if config.sconv else ()
        provider.grug_sconv_kernel = config.sconv_kernel
        provider.grug_hero = config.uses_hero_architecture
        provider.moe_latent_size = config.latent_dim
        if provider.grug_hero:
            # TE's ring CP cannot apply sliding windows. A2A preserves packed
            # document boundaries; global layers can retain the ring implementation.
            provider.cp_comm_type = ["p2p" if is_long else "a2a" for is_long in long_flags]
        provider.grug_qk_mult = config.qk_mult
        provider.grug_qk_mult_long_scale = config.qk_mult_long_scale

        provider.num_moe_experts = config.num_local_experts
        provider.ffn_hidden_size = config.intermediate_size
        provider.moe_ffn_hidden_size = config.intermediate_size
        provider.moe_shared_expert_intermediate_size = config.shared_expert_intermediate_size
        provider.moe_shared_expert_gate = False
        provider.moe_shared_expert_overlap = False
        provider.moe_router_topk = config.num_experts_per_tok
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_grouped_gemm = True
        # Megatron's unfused unpermute combines the top-k expert outputs with an atomic
        # scatter-add. For k > 2 the summation order varies between runs, so the eval-mode
        # old log-probs never match the training forward. TE's fused kernels reduce in a
        # fixed order and are bit-reproducible.
        provider.moe_permute_fusion = True
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            *_gated_norm_mappings("embed_norm", "model.embed_norm.weight", "model.embed_gated_norm"),
            *_gated_norm_mappings("decoder.final_layernorm", "model.norm.weight", "model.final_gated_norm"),
            *_gated_norm_mappings(
                "decoder.layers.*.input_layernorm",
                "model.layers.*.input_layernorm.weight",
                "model.layers.*.attn_gated_norm",
            ),
            *_gated_norm_mappings(
                "decoder.layers.*.pre_mlp_layernorm",
                "model.layers.*.post_attention_layernorm.weight",
                "model.layers.*.mlp_gated_norm",
            ),
            QKVMapping(
                "decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            AutoMapping("decoder.layers.*.self_attention.linear_proj.weight", "model.layers.*.self_attn.o_proj.weight"),
            AutoMapping(
                "decoder.layers.*.self_attention.attn_gate.weight", "model.layers.*.self_attn.attn_gate.weight"
            ),
            ReplicatedMapping("decoder.layers.*.mlp.router.weight", "model.layers.*.mlp.router.weight"),
            ReplicatedMapping("decoder.layers.*.mlp.router.expert_bias", "model.layers.*.mlp.router.bias"),
        ]
        config = self.hf_config
        if config.uses_hero_architecture:
            shared_expert_prefixes = [
                (f"decoder.layers.*.mlp.shared_experts.experts.{index}", f"model.layers.*.shared_experts.{index}")
                for index in range(config.num_shared_experts)
            ]
        else:
            shared_expert_prefixes = [("decoder.layers.*.mlp.shared_experts", "model.layers.*.shared_expert")]
        for megatron, hf in shared_expert_prefixes:
            mappings.extend(
                [
                    GatedMLPMapping(
                        f"{megatron}.linear_fc1.weight", gate=f"{hf}.gate_proj.weight", up=f"{hf}.up_proj.weight"
                    ),
                    AutoMapping(f"{megatron}.linear_fc2.weight", f"{hf}.down_proj.weight"),
                ]
            )
        mappings.extend(_routed_expert_mappings(config.grugmoe_artifact_schema_version))
        if config.latent_dim is not None:
            mappings.extend(
                [
                    ReplicatedMapping(
                        "decoder.layers.*.mlp.fc1_latent_proj.weight", "model.layers.*.mlp.latent_down_proj.weight"
                    ),
                    ReplicatedMapping(
                        "decoder.layers.*.mlp.fc2_latent_proj.weight", "model.layers.*.mlp.latent_up_proj.weight"
                    ),
                    ReplicatedMapping(
                        "decoder.layers.*.mlp.latent_norm.weight", "model.layers.*.mlp.latent_norm.weight"
                    ),
                ]
            )
        if config.sconv:
            for site, mg, hf, mapping in (
                ("k", "self_attention.sconv_k", "self_attn.sconv_k", RowParallelMapping),
                ("attn", "self_attention.sconv_attn", "sconv_attn", ReplicatedMapping),
                ("mlp", "sconv_mlp", "sconv_mlp", ReplicatedMapping),
            ):
                if site in config.sconv_sites:
                    mappings.append(mapping(f"decoder.layers.*.{mg}.weight", f"model.layers.*.{hf}.weight"))
        return MegatronMappingRegistry(*mappings)
