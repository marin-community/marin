"""Megatron-Core modules for training the Grug MoE policy with ``trainer.strategy=megatron``.

Grug differs from a stock Megatron GPT model in a handful of places that the
layer spec cannot express through configuration alone:

* every norm (embedding, per-layer, final) is followed by a low-rank sigmoid
  gate (``GrugGatedRMSNorm``);
* queries and keys use a weightless RMS norm and a per-layer query scale;
* attention output is projected away from the value direction (XSA) and
  scaled by a per-head sigmoid gate computed from the attention input;
* the router selects the top-(k+1) experts on biased logits, drops the last
  one, and renormalizes sigmoid weights of the survivors;
* Hero adds latent expert projections, separate shared experts, causal ShortConv,
  and different key/value head counts on local and long-attention layers.

Everything else (sliding window on local layers, RoPE skipped on long layers,
half-RoPE, grouped-GEMM experts, GQA) maps onto stock
Megatron-Core settings chosen by ``GrugModelProvider`` in
``grug_megatron_bridge``.
"""

from dataclasses import replace
from functools import partial

import torch
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.mappings import all_gather_last_dim_from_tensor_parallel_region
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlock,
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint, sharded_state_dict_default
from megatron.core.typed_torch import apply_module
from torch import nn

from skyrl_train.models.megatron_router_replay import RouterScoreType
from skyrl_train.models.grug_shortconv import causal_short_conv
from skyrl_train.models.router_instrumentation import NativeRouterObserverEmitter, emit_router_forward
from skyrl_train.models.grug_moe import (
    GRUG_ATTN_GATE_SCALE,
    GRUG_GATED_NORM_RANK,
    GRUG_ROUTER_RENORM_EPS,
    GRUG_ROUTING_RENORM_SUM,
    GRUG_XSA_EPS,
    grug_long_layer_flags,
    grug_rms_norm_no_weight,
    jax_top_k,
)


def _first_present(preferred: torch.Tensor | None, fallback: torch.Tensor | None) -> torch.Tensor | None:
    return fallback if preferred is None else preferred


class GrugGatedRMSNorm(nn.Module):
    """RMSNorm followed by Grug's low-rank sigmoid gate: ``norm(x) * sigmoid(up(silu(down(norm(x)))))``."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()
        self.norm = TENorm(config=config, hidden_size=hidden_size, eps=eps)
        device = torch.cuda.current_device()
        self.down_proj = nn.Linear(
            hidden_size, GRUG_GATED_NORM_RANK, bias=False, device=device, dtype=config.params_dtype
        )
        self.up_proj = nn.Linear(
            GRUG_GATED_NORM_RANK, hidden_size, bias=False, device=device, dtype=config.params_dtype
        )
        # Replicated across TP ranks; the attribute makes Megatron all-reduce their grads under SP.
        for param in (self.down_proj.weight, self.up_proj.weight):
            param.sequence_parallel = config.sequence_parallel

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        gate = torch.sigmoid(self.up_proj(F.silu(self.down_proj(normalized))))
        return normalized * gate


class GrugQKNorm(nn.Module):
    """Weightless RMS norm applied to each query and key head."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return grug_rms_norm_no_weight(hidden_states)


class GrugSelfAttention(SelfAttention):
    """Grug attention on top of Megatron's fused-QKV self attention.

    Compared with ``SelfAttention.forward`` this drops inference support and
    adds the per-layer query scale, XSA, and the per-head output gate. RoPE
    is skipped entirely on long layers via ``config.no_rope_freq``.
    """

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, layer_number: int, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)
        self.attn_gate = TEColumnParallelLinear(
            config.hidden_size,
            config.num_attention_heads,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="attn_gate",
            tp_group=self.pg_collection.tp,
        )
        is_long = grug_long_layer_flags(config.num_layers, config.grug_global_every)[layer_number - 1]
        self.query_scale = config.grug_qk_mult * (config.grug_qk_mult_long_scale if is_long else 1.0)
        self.skip_rope = bool(config.no_rope_freq[layer_number - 1])
        self.logical_kv_heads = config.grug_global_kv_heads if is_long else config.grug_local_kv_heads
        self.sconv_k = None
        self.sconv_attn = None
        if "k" in config.grug_sconv_sites:
            self.sconv_k = GrugShortConv(
                config,
                self.num_query_groups_per_partition * config.kv_channels,
                self.pg_collection,
                channel_parallel=True,
            )
            # K convolution precedes the weightless head norm.
            self.k_layernorm = nn.Identity()
        if "attn" in config.grug_sconv_sites:
            self.sconv_attn = GrugShortConv(config, config.hidden_size, self.pg_collection)
        if self.logical_kv_heads != config.num_query_groups:
            attention_config = replace(config, num_query_groups=self.logical_kv_heads)
            self.core_attention = submodules.core_attention(
                config=attention_config,
                layer_number=layer_number,
                attn_mask_type=self.attn_mask_type,
                attention_type="self",
                cp_comm_type=kwargs.get("cp_comm_type"),
                softmax_scale=config.softmax_scale,
                pg_collection=self.pg_collection,
            )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        if inference_context is not None or inference_params is not None:
            raise NotImplementedError("GrugSelfAttention only supports training forwards")
        if rotary_pos_cos is not None or rotary_pos_sin is not None or rotary_pos_cos_sin is not None:
            raise NotImplementedError("GrugSelfAttention applies RoPE from rotary_pos_emb only")

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        if self.sconv_k is not None:
            key = self.sconv_k(key.flatten(-2), packed_seq_params).reshape(key.shape)
            key = grug_rms_norm_no_weight(key)
        if self.logical_kv_heads != self.config.num_query_groups:
            key, value = (self._logical_kv(tensor) for tensor in (key, value))

        is_thd = packed_seq_params is not None and packed_seq_params.qkv_format == "thd"
        if is_thd:
            query, key, value = query.squeeze(1), key.squeeze(1), value.squeeze(1)

        if rotary_pos_emb is not None and not self.skip_rope:
            if not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2
            q_pos_emb, k_pos_emb = rotary_pos_emb
            cu_seqlens_q = cu_seqlens_kv = None
            if is_thd:
                cu_seqlens_q = _first_present(packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q)
                cu_seqlens_kv = _first_present(packed_seq_params.cu_seqlens_kv_padded, packed_seq_params.cu_seqlens_kv)
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q, cp_group=self.pg_collection.cp
            )
            key = apply_rotary_pos_emb(
                key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv, cp_group=self.pg_collection.cp
            )

        query = query * self.query_scale

        # Grug is causal-only and the trainer removes left padding, so right-padded keys sit after
        # every valid query and the causal (or causal sliding-window) mask alone is exact.
        attention_mask = None
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self._run_core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=self.attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

        core_attn_out = self._apply_xsa(core_attn_out, value)
        if is_thd:
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        gate, _ = self.attn_gate(hidden_states)
        core_attn_out = self._apply_head_gate(core_attn_out, gate)
        output, bias = apply_module(self.linear_proj)(core_attn_out)
        if self.sconv_attn is not None:
            output = self.sconv_attn(output, packed_seq_params)
        return output, bias

    def _logical_kv(self, tensor):
        # The checkpoint stores max(local, global) heads. Retain unused rows for
        # lossless export, but attend only to the leading logical KV heads.
        # A differentiable gather is needed: TP's stored and logical head owners differ.
        gathered = all_gather_last_dim_from_tensor_parallel_region(
            tensor.flatten(-2), group=self.pg_collection.tp
        ).reshape(*tensor.shape[:2], self.config.num_query_groups, self.config.kv_channels)
        count = self.logical_kv_heads // self.pg_collection.tp.size()
        start = self.pg_collection.tp.rank() * count
        return gathered[..., start : start + count, :].contiguous()

    def _apply_xsa(self, core_attn_out: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Remove each head's component along its (GQA-expanded) value vector."""

        # CoreAttention returns TP-local heads. Derive that count from its
        # output: the inherited head-count field can still be global here.
        local_heads = core_attn_out.shape[-1] // self.hidden_size_per_attention_head
        local_query_groups = value.shape[-2]
        if local_heads * self.hidden_size_per_attention_head != core_attn_out.shape[-1]:
            raise ValueError("Grug XSA received a partial attention head")
        if local_heads % local_query_groups:
            raise ValueError("Grug XSA local heads are not divisible by local query groups")
        heads = core_attn_out.view(*value.shape[:-2], local_heads, self.hidden_size_per_attention_head)
        expanded_value = value.repeat_interleave(local_heads // local_query_groups, dim=-2)
        out = heads.float()
        v = expanded_value.float()
        dot = (out * v).sum(dim=-1, keepdim=True)
        v_norm = v.square().sum(dim=-1, keepdim=True)
        out = out - (dot / (v_norm + GRUG_XSA_EPS)) * v
        return out.to(core_attn_out.dtype).reshape(core_attn_out.shape)

    def _apply_head_gate(self, core_attn_out: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Scale every head by ``2 * sigmoid(attn_gate(x))``."""

        heads = core_attn_out.view(*gate.shape, self.hidden_size_per_attention_head)
        gated = heads * (GRUG_ATTN_GATE_SCALE * torch.sigmoid(gate.float())).unsqueeze(-1).to(heads.dtype)
        return gated.reshape(core_attn_out.shape)


def _grug_topk_indices(biased_logits: torch.Tensor, topk: int) -> torch.Tensor:
    """Return the first K indices from Grug's biased top-(K+1) selection."""
    _, indices = jax_top_k(biased_logits, topk + 1)
    return indices[:, :topk]


class GrugTopKRouter(NativeRouterObserverEmitter, TopKRouter):
    """Grug routing: biased top-(k+1) selection with sigmoid weights renormalized to a fixed sum.

    The persistent fp32 ``expert_bias`` buffer is Grug's frozen query bias. It
    only steers expert selection; the combine weights come from the unbiased
    logits of the first ``k`` selected experts.
    """

    replay_score_type = RouterScoreType.LOGITS

    def __init__(self, config: TransformerConfig, pg_collection=None, is_mtp_layer: bool = False):
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        assert self.enable_expert_bias, "GrugTopKRouter requires moe_router_enable_expert_bias=True"

    def routing(self, logits: torch.Tensor, padding_mask: torch.Tensor | None = None):
        logits = logits.view(-1, self.config.num_moe_experts).float()
        biased_logits = logits + self.expert_bias
        natural_selected: torch.Tensor | None = None
        if self.router_replay is None:
            selected = _grug_topk_indices(biased_logits, self.topk)
            natural_selected = selected
        else:
            # Grug overrides MCore's routing method, so its replay hook must be
            # called here. Selection uses biased logits; combine weights below
            # still use the original, unbiased logits.
            def native_topk(scores, topk, num_groups=None, group_topk=None):
                if num_groups is not None or group_topk is not None:
                    raise ValueError("Grug router replay does not support grouped top-k")
                indices = _grug_topk_indices(scores, topk)
                return scores.gather(1, indices), indices

            _, selected = self.router_replay.get_replay_topk(biased_logits, self.topk, None, None, native_topk)
        combine = torch.sigmoid(torch.gather(logits, dim=-1, index=selected))
        combine = combine * (GRUG_ROUTING_RENORM_SUM / (combine.sum(dim=-1, keepdim=True) + GRUG_ROUTER_RENORM_EPS))
        probs = torch.zeros_like(logits).scatter(1, selected, combine)
        routing_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, selected, True)
        emit_router_forward(
            router=self,
            router_inputs=logits,
            selection_logits=biased_logits,
            natural_selected_experts=natural_selected if natural_selected is not None else selected,
            selected_experts=selected,
            combine_weights=combine,
        )
        return probs, routing_map

    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor | None = None):
        self._maintain_float32_expert_bias()
        logits = self.gating(input)
        return self.routing(logits, padding_mask)


class GrugGPTModel(GPTModel):
    """GPTModel with Grug's gated embedding norm on the first pipeline stage."""

    def __init__(self, config: TransformerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        if self.pre_process:
            self.embed_norm = GrugGatedRMSNorm(
                config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
            )

    def _preprocess(
        self,
        input_ids,
        position_ids,
        decoder_input=None,
        inference_context=None,
        packed_seq_params=None,
        padding_mask=None,
    ):
        apply_embed_norm = self.pre_process and decoder_input is None
        outputs = super()._preprocess(
            input_ids,
            position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        if not apply_embed_norm:
            return outputs
        decoder_input, *rest = outputs
        # The decoder's checkpoint does not cover this normalization. Checkpoint it
        # separately to save memory by recomputing it during the backward pass.
        # With frozen embeddings, run it normally so its weights still get gradients:
        # this checkpoint needs an input with requires_grad=True.
        if (
            self.config.recompute_granularity == "full"
            and self.training
            and torch.is_grad_enabled()
            and decoder_input.requires_grad
        ):
            normalized = tensor_parallel.checkpoint(self.embed_norm, False, decoder_input)
        else:
            normalized = self.embed_norm(decoder_input)
        return (normalized, *rest)


class GrugShortConv(nn.Module):
    """ShortConv on CP-local tokens with optional TP sequence or channel sharding."""

    def __init__(self, config, channels, pg_collection, channel_parallel=False):
        super().__init__()
        self.pg_collection = pg_collection
        self.channel_parallel = channel_parallel
        self.sequence_parallel = config.sequence_parallel and not channel_parallel
        self.weight = nn.Parameter(
            torch.zeros(
                config.grug_sconv_kernel, channels, device=torch.cuda.current_device(), dtype=config.params_dtype
            )
        )
        with torch.no_grad():
            self.weight[0].fill_(1)
        # Output convolutions see the full TP sequence on every rank, so their
        # weight gradients are already complete replicas, not SP partials.
        self.weight.sequence_parallel = False
        if channel_parallel:
            tensor_parallel.set_tensor_model_parallel_attributes(self.weight, True, 1, 1)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return make_sharded_tensors_for_checkpoint(
            self.state_dict(keep_vars=True),
            prefix,
            {"weight": 1} if self.channel_parallel else {},
            sharded_offsets=sharded_offsets,
            tp_group=self.pg_collection.tp,
            dp_cp_group=metadata["dp_cp_group"],
        )

    def forward(self, hidden_states, packed_seq_params=None):
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.pg_collection.tp
            )
        lengths = None
        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            cumulative = _first_present(packed_seq_params.cu_seqlens_q_padded, packed_seq_params.cu_seqlens_q)
            lengths = tuple((cumulative[1:] - cumulative[:-1]).tolist())
        result = causal_short_conv(hidden_states, self.weight, lengths, self.pg_collection.cp)
        if self.sequence_parallel:
            result = tensor_parallel.scatter_to_sequence_parallel_region(result, group=self.pg_collection.tp)
        return result


class GrugSharedExperts(nn.Module):
    """Keep Hero's shared experts separate, including their ordered BF16 additions."""

    def __init__(self, config, submodules, pg_collection, gate=False, name=None):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                SharedExpertMLP(config=config, submodules=submodules, pg_collection=pg_collection, gate=gate)
                for _ in range(config.grug_num_shared_experts)
            ]
        )

    def forward(self, hidden_states):
        return tuple(expert(hidden_states) for expert in self.experts)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state = {}
        for index, expert in enumerate(self.experts):
            state.update(sharded_state_dict_default(expert, f"{prefix}experts.{index}.", sharded_offsets, metadata))
        return state

    def backward_dw(self):
        for expert in self.experts:
            expert.backward_dw()


class GrugMoELayer(MoELayer):
    """Native Megatron latent dispatch, with Hero's pre-dispatch latent RMSNorm."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # MCore 0.18 marks expert parameters as dense when EP=1. With a
        # smaller expert TP group, those replicas instead belong to expert DP
        # (which includes the extra attention-TP ranks).
        if (
            config.expert_model_parallel_size == 1
            and config.tensor_model_parallel_size != config.expert_tensor_parallel_size
        ):
            for param in self.experts.parameters():
                param.allreduce = False
        if config.moe_latent_size:
            self.latent_norm = TENorm(config=config, hidden_size=config.moe_latent_size, eps=config.layernorm_epsilon)

    def preprocess(self, hidden_states, probs, routing_map):
        if not self.config.moe_latent_size:
            return super().preprocess(hidden_states, probs, routing_map)
        hidden_states, _ = self.fc1_latent_proj(hidden_states)
        hidden_states = self.latent_norm(hidden_states)
        return self.token_dispatcher.dispatch_preprocess(hidden_states, routing_map, probs)

    def postprocess(self, output, shared_expert_output):
        output = super().postprocess(output, None)
        if shared_expert_output is not None:
            for shared in shared_expert_output:
                output = output + shared
        return output


class GrugTransformerLayer(TransformerLayer):
    """Pass document metadata explicitly to the MLP branch's ShortConv."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.sconv_mlp = GrugShortConv(config, config.hidden_size, self.pg_collection)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        hidden_states, context = self._forward_attention(hidden_states, attention_mask, **kwargs)
        normalized = self._forward_pre_mlp_layernorm(hidden_states)
        padding_mask = kwargs.get("padding_mask")
        packed_seq_params = kwargs.get("packed_seq_params")

        def mlp_forward(x):
            return self.mlp(x, padding_mask=padding_mask)

        if self.recompute_mlp:
            output, bias = tensor_parallel.checkpoint(mlp_forward, False, normalized)
        else:
            output, bias = mlp_forward(normalized)
        output = self.sconv_mlp(output, packed_seq_params)
        return self._forward_post_mlp((output, bias), hidden_states), context


def grug_layer_spec(config: TransformerConfig) -> ModuleSpec:
    """Build the Transformer Engine layer spec for one Grug decoder layer."""

    backend = TESpecProvider()
    moe = get_moe_module_spec_for_backend(backend, num_experts=config.num_moe_experts, moe_grouped_gemm=True)
    moe.keywords["submodules"].router = GrugTopKRouter
    if config.grug_hero:
        shared = moe.keywords["submodules"].shared_experts
        moe.keywords["submodules"].shared_experts = partial(GrugSharedExperts, **shared.keywords)
        moe = partial(GrugMoELayer, **moe.keywords)
    return ModuleSpec(
        module=GrugTransformerLayer if "mlp" in config.grug_sconv_sites else TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=GrugGatedRMSNorm,
            self_attention=ModuleSpec(
                module=GrugSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=backend.column_parallel_linear(),
                    core_attention=backend.core_attention(),
                    linear_proj=backend.row_parallel_linear(),
                    q_layernorm=GrugQKNorm,
                    k_layernorm=GrugQKNorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=GrugGatedRMSNorm,
            mlp=moe,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def grug_block_spec(config: TransformerConfig, vp_stage: int | None, pp_rank: int) -> ModuleSpec:
    """Build this pipeline stage's decoder block spec with Grug's gated final norm."""

    layer_spec = grug_layer_spec(config)
    num_layers = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    return ModuleSpec(
        module=TransformerBlock,
        submodules=TransformerBlockSubmodules(layer_specs=[layer_spec] * num_layers, layer_norm=GrugGatedRMSNorm),
    )
