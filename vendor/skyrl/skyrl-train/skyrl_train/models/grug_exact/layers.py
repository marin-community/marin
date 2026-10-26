"""The Megatron layer-spec modules of the exact Grug trainer: each computes the decode-invariant engine's values and
differentiates them as the trainer's own computation."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer

from skyrl_train.models.grug_exact import engine_regions as regions
from skyrl_train.models.grug_exact.context import ExactForward, ResidualSum
from skyrl_train.models.grug_exact.ep_combine import ep_combine
from skyrl_train.models.grug_exact.gradient_kernels import (
    gated_product_value,
    hybrid_input_norm_value,
    kernel_value,
    logits_value,
    query_key_values,
    swiglu_value,
    xsa_head_gate_value,
)
from skyrl_train.models.grug_exact.vllm_kernels import (
    expert_outputs,
    fa3_attention_sbhd,
    row_invariant_gemm,
    topk_experts,
)
from skyrl_train.models.grug_megatron import GrugGatedRMSNorm, GrugRoute, GrugSelfAttention, GrugTopKRouter


def _rms_norm_single_rounding(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """vLLM's RMSNorm of an fp32 input: ``x * rsqrt(mean(x^2) + eps) * weight`` in fp32, rounded once."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps) * weight.float()).to(weight.dtype)


def _qkv_projection(
    x: torch.Tensor, fused_weight: torch.Tensor, groups: int, query_width: int, head_dim: int
) -> torch.Tensor:
    """Megatron's fused QKV projection computed as three GEMMs, as compiled vLLM computes q, k and v, in the fused
    projection's layout: per KV group, its ``query_width`` query rows, then its key and value rows."""
    grouped = fused_weight.view(groups, query_width + 2 * head_dim, -1)
    weights = (
        grouped[:, :query_width],
        grouped[:, query_width : query_width + head_dim],
        grouped[:, query_width + head_dim :],
    )
    parts = [F.linear(x, weight.reshape(-1, weight.shape[-1])) for weight in weights]
    return torch.cat([part.view(*x.shape[:-1], groups, -1) for part in parts], dim=-1).view(*x.shape[:-1], -1)


def _split_qkv(mixed_qkv: torch.Tensor, groups: int, query_width: int, head_dim: int) -> tuple[torch.Tensor, ...]:
    """Megatron's split of the fused projection into query ``[S, B, heads, head_dim]``, key and value
    ``[S, B, groups, head_dim]``."""
    grouped = mixed_qkv.view(*mixed_qkv.shape[:-1], groups, query_width + 2 * head_dim)
    query, key, value = torch.split(grouped, [query_width, head_dim, head_dim], dim=-1)
    return query.reshape(*mixed_qkv.shape[:-1], -1, head_dim), key, value


class _ExactGatedNorm(GrugGatedRMSNorm):
    """Grug's gated RMSNorm with the gate's product from the engine's kernel; subclasses compute the norm."""

    def __init__(self, config: TransformerConfig, hidden_size: int, eps: float):
        super().__init__(config, hidden_size, eps)
        self.eps = eps

    def gate_logits(self, normalized: torch.Tensor) -> torch.Tensor:
        """The gate's logits with the engine's SiLU kernel, differentiated as ``F.silu``."""
        hidden = self.down_proj(normalized)
        return self.up_proj(kernel_value(regions.gate_silu(hidden), lambda: F.silu(hidden)))

    def gated(self, normalized: torch.Tensor) -> torch.Tensor:
        """``normalized * sigmoid(gate)`` from the engine's kernel, differentiated as the fp32 product rounded once."""
        gate = self.gate_logits(normalized)
        product = regions.gated_product(normalized, gate)
        return gated_product_value(product, normalized, gate)


class ExactPreMlpGatedNorm(_ExactGatedNorm):
    """The gated norm after the attention: the engine's RMSNorm of the stored bf16 hidden states."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        value = regions.rms_norm(hidden_states, self.norm.weight, self.eps)
        return self.gated(kernel_value(value, lambda: self.norm(hidden_states)))


class ExactInputGatedNorm(_ExactGatedNorm):
    """A layer's input gated norm: the engine normalizes the stored bf16 sum by the statistic of the unrounded sum
    that formed it, which the previous layer or stage hands on."""

    def forward(self, hidden_states: torch.Tensor, statistic: torch.Tensor) -> torch.Tensor:
        weight = self.norm.weight
        value = regions.rms_norm_from_square_sum(hidden_states, statistic, weight, self.eps)
        variance = (statistic / hidden_states.shape[-1]).view(*hidden_states.shape[:-1], 1)
        return self.gated(hybrid_input_norm_value(value, hidden_states, variance, weight, self.eps))


class ExactFinalGatedNorm(_ExactGatedNorm):
    """The final gated norm: the engine normalizes the last layer's unrounded sum."""

    def forward(self, hidden_states: torch.Tensor, parts: ResidualSum) -> torch.Tensor:
        weight = self.norm.weight
        # The gradient flows through the bf16 sum's exactly-zero ``residual - residual.detach()``.
        residual = hidden_states.float()

        def reference() -> torch.Tensor:
            mlp = parts.routed.float() + parts.shared.float()
            unrounded = (parts.residual.float() + mlp) + (residual - residual.detach())
            return _rms_norm_single_rounding(unrounded, weight, self.eps)

        value = regions.final_norm(parts.residual, parts.routed, parts.shared, weight, self.eps)
        return self.gated(kernel_value(value, reference))


class ExactEmbeddingGatedNorm(_ExactGatedNorm):
    """The embedding gated norm; returns its output and layer 0's input-norm statistic of the unrounded product."""

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = regions.rms_norm(embeddings, self.norm.weight, self.eps)
        normalized = kernel_value(value, lambda: self.norm(embeddings))
        gate = self.gate_logits(normalized)
        product, statistic = regions.gated_product_square_sum(normalized, gate)
        return gated_product_value(product, normalized, gate), statistic


class ExactSelfAttention(GrugSelfAttention):
    """Grug attention with the engine's q/k chain, FA3 attention, XSA with the head gate, and GEMM shapes."""

    def __init__(self, config: TransformerConfig, submodules, layer_number: int, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)
        is_long = self.skip_rope
        # The engine applies the two query factors one after the other.
        self.qk_mult = config.grug_qk_mult
        self.qk_mult_scale = config.grug_qk_mult_long_scale if is_long else 1.0
        # The engine's FA3 call for this layer: a sliding window of window_size[0] + 1 keys on local layers.
        self.fa3_window = None if is_long else config.window_size[0] + 1
        self.fa3_scale = (
            config.softmax_scale if config.softmax_scale is not None else 1.0 / math.sqrt(config.kv_channels)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        rotary_pos_emb,
        kept_attention: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``residual + attention(hidden_states)`` ``[S, B, H]``, the residual added in the output projection's
        epilogue as the engine adds it, and the FA3 output, which a recompute passes back as ``kept_attention``."""
        groups = self.num_query_groups_per_partition
        head_dim = self.hidden_size_per_attention_head
        query_width = self.num_attention_heads_per_partition // groups * head_dim
        mixed = _qkv_projection(hidden_states, self.linear_qkv.weight, groups, query_width, head_dim)
        query, key, value = _split_qkv(mixed, groups, query_width, head_dim)
        query, key = self._query_key(query, key, rotary_pos_emb)
        if kept_attention is None:
            with torch.no_grad():
                kept_attention = fa3_attention_sbhd(query, key, value, window=self.fa3_window, scale=self.fa3_scale)
        attention = kernel_value(
            kept_attention, lambda: self.core_attention(query, key, value, None, attn_mask_type=self.attn_mask_type)
        )
        rows = hidden_states.reshape(-1, hidden_states.shape[-1])
        # The engine reads the fp32 gate logits as they are: Inductor fuses the sigmoid into the XSA kernel.
        gate = row_invariant_gemm(rows, self.attn_gate.weight).view(*hidden_states.shape[:-1], -1)
        gate = logits_value(gate, hidden_states, self.attn_gate.weight)
        xsa = regions.xsa_head_gate(attention, value.flatten(-2), gate, head_dim)
        gated = xsa_head_gate_value(xsa, attention, value, gate)
        width = hidden_states.shape[-1]
        output = torch.addmm(residual.reshape(-1, width), gated.reshape(-1, width), self.linear_proj.weight.t())
        return output.view_as(hidden_states), kept_attention

    def _query_key(self, query: torch.Tensor, key: torch.Tensor, rotary_pos_emb) -> tuple[torch.Tensor, torch.Tensor]:
        """The engine's q/k norm, RoPE and query scale on the raw ``[S, B, heads, dim]`` projections, differentiated
        as ``query_key_values``."""
        projections = (query.flatten(-2), key.flatten(-2), query.shape[-1], self.qk_mult, self.qk_mult_scale)
        freqs = None
        if rotary_pos_emb is not None and not self.skip_rope:
            engine_query, engine_key = regions.query_key_rope(
                *projections, self.config.rotary_base, self.config.seq_length
            )
            freqs = rotary_pos_emb
        else:
            engine_query, engine_key = regions.query_key(*projections)
        return query_key_values(
            engine_query.view_as(query), engine_key.view_as(key), query, key, freqs, self.qk_mult, self.qk_mult_scale
        )


class ExactTopKRouter(GrugTopKRouter):
    """Grug's router with the engine's invariant GEMM for the logits and its tie-breaking for the selection; the
    forward also returns the selected experts in the engine's slot order."""

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        value = row_invariant_gemm(input.reshape(-1, input.shape[-1]), self.weight)
        return logits_value(value.view(*input.shape[:-1], value.shape[-1]), input, self.weight)

    def select_topk(self, biased_logits: torch.Tensor) -> torch.Tensor:
        return topk_experts(biased_logits, self.topk)

    def forward(self, input: torch.Tensor, padding_mask=None) -> GrugRoute:
        self._maintain_float32_expert_bias()
        return self.route_with_selection(self.gating(input))


class ExactSharedExpertMLP(SharedExpertMLP):
    """Grug's shared expert with the engine's GEMM shapes (gate and up as two GEMMs) and activation kernel."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``shared_expert(hidden_states)``, the down projection's bf16 GEMM output."""
        gate_weight, up_weight = self.linear_fc1.weight.chunk(2, dim=0)
        gate, up = F.linear(hidden_states, gate_weight), F.linear(hidden_states, up_weight)
        value = regions.shared_activation(gate, up)
        activation = swiglu_value(value, torch.cat((gate, up), dim=-1))
        output = torch.mm(activation.reshape(-1, activation.shape[-1]), self.linear_fc2.weight.t())
        return output.view_as(hidden_states)


class ExactGroupedMLP(TEGroupedMLP):
    """Grug's routed experts with each dispatched slot's value from the engine's fused-MoE kernels, differentiated as
    the grouped GEMMs."""

    def engine_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        """The experts' output with the fused-MoE kernels' values; ``forward`` gives the grouped GEMMs' values."""
        with torch.no_grad():
            value = expert_outputs(
                permuted_local_hidden_states,
                tokens_per_expert.tolist(),
                permuted_probs,
                [getattr(self.linear_fc1, f"weight{index}") for index in range(self.linear_fc1.num_gemms)],
                [getattr(self.linear_fc2, f"weight{index}") for index in range(self.linear_fc2.num_gemms)],
            )
        if not torch.is_grad_enabled():
            return value, None
        reference, bias = super().forward(permuted_local_hidden_states, tokens_per_expert, permuted_probs)
        return kernel_value(value, lambda: reference), bias


class ExactMoELayer(MoELayer):
    """Grug's MoE layer: the routed output added in the engine's expert-parallel order, and the shared expert's
    output, which the decoder layer adds to the routed output and the residual as the engine adds them."""

    def forward(
        self, hidden_states: torch.Tensor, exact: ExactForward, recompute: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The routed and the shared experts' outputs."""
        route = self.router(hidden_states)
        dispatched, probs = self.preprocess(hidden_states, route.probs, route.routing_map)
        dispatched, probs = self.dispatch(dispatched, probs)
        dispatched, tokens_per_expert, permuted_probs = self.token_dispatcher.dispatch_postprocess(dispatched, probs)
        if recompute:
            # A recompute only rebuilds the gradient, which the grouped GEMMs' forward carries.
            output, _ = self.experts(dispatched, tokens_per_expert, permuted_probs)
        else:
            output, _ = self.experts.engine_forward(dispatched, tokens_per_expert, permuted_probs)
        output = self.combine(self.token_dispatcher.combine_preprocess(output))
        return self._combine_postprocess(output, route.selected, exact), self.shared_experts(hidden_states)

    def _combine_postprocess(self, output: torch.Tensor, selected: torch.Tensor, exact: ExactForward) -> torch.Tensor:
        """Each token's routed output added as the engine adds it, differentiated as the trainer's unpermute."""
        dispatcher = self.token_dispatcher
        ranks = exact.row_engine_dp_ranks(selected.shape[0])
        with torch.no_grad():
            combined = ep_combine(output, dispatcher.routing_map, selected, ranks, exact.expert_parallel_size)
        combined = combined.view(dispatcher.hidden_shape)
        return kernel_value(combined, lambda: dispatcher.combine_postprocess(output))


class ExactTransformerLayer(TransformerLayer):
    """A Grug decoder layer whose forward hands its input-norm statistic and residual sum to the next consumer."""

    def forward(
        self, hidden_states: torch.Tensor, statistic: torch.Tensor, rotary_pos_emb, exact: ExactForward, keep: bool
    ) -> tuple[torch.Tensor, torch.Tensor, ResidualSum]:
        """The layer output, the next input norm's statistic of its unrounded sum, and the sum's parts; with ``keep``,
        a first forward keeps its attention output in ``exact`` for the recompute."""
        kept = exact.attention.pop(self.layer_number, None)
        normalized = self.input_layernorm(hidden_states, statistic)
        hidden_states, attention = self.self_attention(normalized, hidden_states, rotary_pos_emb, kept)
        if keep and kept is None:
            exact.attention[self.layer_number] = attention
        routed, shared = self.mlp(self.pre_mlp_layernorm(hidden_states), exact, recompute=kept is not None)
        output, square_sum = regions.residual_square_sum(hidden_states, routed, shared)
        output = kernel_value(output, lambda: hidden_states + (routed + shared))
        return output, square_sum, ResidualSum(hidden_states.detach(), routed.detach(), shared.detach())
