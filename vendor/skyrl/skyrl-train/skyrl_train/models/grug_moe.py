# SPDX-FileCopyrightText: 2026 NovaSkyAI
# SPDX-License-Identifier: Apache-2.0

"""PyTorch Grug MoE policy-training implementation.

The eager attention path remains the correctness reference. The supported
FlashAttention path preserves the same checkpoint and Snowball semantics
without materializing dense sequence-by-sequence scores or masks.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, Protocol

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, initialization as init
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from skyrl_train.models.ep_gradient import ExpertGradientAveraging
from skyrl_train.models.grug_query_bias import (
    GrugQueryBiasLayerObservation,
    GrugQueryBiasObservation,
)
from skyrl_train.models.layers.moe_routing import TokenReorderer, grouped_expert_contributions
from skyrl_train.models.router_instrumentation import NativeRouterObserverEmitter, emit_router_forward
from skyrl_train.utils.flash_attention import (
    FLASH_ATTN_IMPORT_ERROR,
    flash_attn_func,
    flash_attn_varlen_func,
    flash_index_first_axis,
    flash_pad_input,
    flash_unpad_input,
)


GRUG_MOE_MODEL_TYPE = "grug_moe"
GRUG_EP_COMM_BACKEND = "torch"
GRUG_ROUTER_BIAS_SUFFIX = ".mlp.router.bias"
GRUG_MOE_ARCHITECTURE = "GrugMoeForCausalLM"
GRUG_MOE_ATTENTION_MODE = "production"
GRUG_STACKED_EXPERT_SCHEMA_VERSION = 1
GRUG_SPLIT_EXPERT_SCHEMA_VERSION = 2
GRUG_DEFAULT_GLOBAL_EVERY = 4
GRUG_EAGER_ATTENTION_BACKEND = "eager"
GRUG_FLASH_ATTENTION_BACKEND = "flash_attention_2"
GRUG_SUPPORTED_ATTENTION_BACKENDS = frozenset({GRUG_EAGER_ATTENTION_BACKEND, GRUG_FLASH_ATTENTION_BACKEND})
GRUG_GATED_NORM_RANK = 128
GRUG_ROUTING_RENORM_SUM = 2.5
# Per-head output gate: sigmoid scaled so an untrained gate passes attention through unchanged.
GRUG_ATTN_GATE_SCALE = 2.0
GRUG_QK_RMS_NORM_EPS = 1e-6
GRUG_XSA_EPS = 1e-6
GRUG_ROUTER_RENORM_EPS = 1e-9
GRUG_SUPPORTED_TRAINING_STRATEGIES = frozenset({"megatron"})


def is_grug_router_bias(model_type: str | None, name: str) -> bool:
    """Return whether a state entry is Grug's persistent FP32 router bias."""

    return model_type == GRUG_MOE_MODEL_TYPE and name.endswith(GRUG_ROUTER_BIAS_SUFFIX)


def validate_grug_training_strategy(model_type: str | None, training_strategy: str | None) -> None:
    """Reject Grug before an unsupported trainer backend allocates the model."""

    if model_type == GRUG_MOE_MODEL_TYPE and training_strategy not in GRUG_SUPPORTED_TRAINING_STRATEGIES:
        raise ValueError(
            f"Grug policy training requires trainer.strategy in {sorted(GRUG_SUPPORTED_TRAINING_STRATEGIES)}"
        )


def grug_long_layer_flags(num_layers: int, global_every: int = GRUG_DEFAULT_GLOBAL_EVERY) -> tuple[bool, ...]:
    """Return which decoder layers use full causal attention without RoPE.

    Every ``global_every``-th layer and the final layer are "long" layers; the
    remaining layers use sliding-window attention with half-RoPE.
    """

    return tuple((idx + 1) % global_every == 0 or idx == num_layers - 1 for idx in range(num_layers))


def _validate_flash_attention_mask(attention_mask: torch.Tensor) -> None:
    valid = attention_mask.to(torch.bool)
    torch._assert_async(
        valid.any(dim=-1).all(),
        "Grug FlashAttention requires at least one valid token in each attention-mask row",
    )
    span_starts = valid[:, 0].to(torch.int64) + ((~valid[:, :-1]) & valid[:, 1:]).sum(dim=-1)
    torch._assert_async(
        (span_starts == 1).all(),
        "Grug FlashAttention requires valid tokens to form one contiguous span in each attention-mask row",
    )


def validate_grug_expert_parallel_options(
    model_type: str | None,
    *,
    expert_model_parallel_size: int,
    use_grouped_mm: bool,
    ep_comm_backend: str,
) -> None:
    """Reject unsupported execution choices when native Grug EP is enabled."""

    if model_type != GRUG_MOE_MODEL_TYPE or expert_model_parallel_size <= 1:
        return
    validate_grug_expert_parallel_runtime(
        use_grouped_mm=use_grouped_mm,
        ep_comm_backend=ep_comm_backend,
    )


def validate_grug_expert_parallel_runtime(*, use_grouped_mm: bool, ep_comm_backend: str) -> None:
    """Validate the execution choices required by native Grug expert sharding."""

    if not use_grouped_mm:
        raise ValueError("Grug expert_model_parallel_size>1 requires use_grouped_mm=true")
    if ep_comm_backend != GRUG_EP_COMM_BACKEND:
        raise ValueError(f"Grug expert parallelism supports only ep_comm_backend={GRUG_EP_COMM_BACKEND}")


def jax_top_k(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return JAX-compatible top-k results, including its lower-index tie rule."""

    sorted_values, sorted_indices = torch.sort(values, dim=-1, descending=True, stable=True)
    return sorted_values[..., :k], sorted_indices[..., :k]


def _resolve_aliases(*values: int | None, default: int, label: str) -> int:
    specified = {int(value) for value in values if value is not None}
    if len(specified) > 1:
        raise ValueError(f"conflicting {label} aliases: {sorted(specified)}")
    return specified.pop() if specified else default


class GrugMoeConfig(PretrainedConfig):
    model_type = GRUG_MOE_MODEL_TYPE

    def __init__(
        self,
        *,
        vocab_size: int = 128256,
        hidden_size: int | None = None,
        hidden_dim: int | None = None,
        intermediate_size: int | None = None,
        intermediate_dim: int | None = None,
        moe_intermediate_size: int | None = None,
        shared_expert_intermediate_size: int | None = None,
        shared_expert_intermediate_dim: int | None = None,
        num_local_experts: int | None = None,
        num_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        num_experts_per_token: int | None = None,
        num_hidden_layers: int | None = None,
        num_layers: int | None = None,
        num_attention_heads: int | None = None,
        num_heads: int | None = None,
        num_key_value_heads: int | None = None,
        num_kv_heads: int | None = None,
        head_dim: int | None = 128,
        max_position_embeddings: int | None = None,
        max_seq_len: int | None = None,
        sliding_window: int = 2048,
        rms_norm_eps: float | None = None,
        layer_norm_eps: float | None = None,
        initializer_range: float | None = None,
        initializer_std: float | None = None,
        qk_mult: float = 1.5703274004183786,
        qk_mult_long_scale: float = 1.0,
        rope_theta: float = 10000.0,
        disable_pko: bool = True,
        disable_long_rope: bool = True,
        router_z_loss_coef: float = 0.0,
        grugmoe_attention_mode: str = GRUG_MOE_ATTENTION_MODE,
        grugmoe_artifact_schema_version: int = GRUG_STACKED_EXPERT_SCHEMA_VERSION,
        latent_dim: int | None = None,
        num_shared_experts: int = 1,
        local_kv_heads: int | None = None,
        global_kv_heads: int | None = None,
        global_every: int = GRUG_DEFAULT_GLOBAL_EVERY,
        rope_fused: bool = False,
        sconv: bool = False,
        sconv_kernel: int = 4,
        sconv_sites: tuple[str, ...] = ("k", "attn", "mlp"),
        use_cache: bool = False,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        hidden_size = _resolve_aliases(hidden_size, hidden_dim, default=2560, label="hidden size")
        intermediate_size = _resolve_aliases(
            intermediate_size,
            intermediate_dim,
            moe_intermediate_size,
            default=1280,
            label="intermediate size",
        )
        shared_expert_intermediate_size = _resolve_aliases(
            shared_expert_intermediate_size,
            shared_expert_intermediate_dim,
            default=2560,
            label="shared expert intermediate size",
        )
        num_local_experts = _resolve_aliases(num_local_experts, num_experts, default=256, label="expert count")
        num_experts_per_tok = _resolve_aliases(
            num_experts_per_tok,
            num_experts_per_token,
            default=4,
            label="experts per token",
        )
        num_hidden_layers = _resolve_aliases(
            num_hidden_layers,
            num_layers,
            default=26,
            label="layer count",
        )
        num_attention_heads = _resolve_aliases(
            num_attention_heads,
            num_heads,
            default=20,
            label="attention head count",
        )
        num_key_value_heads = _resolve_aliases(
            num_key_value_heads,
            num_kv_heads,
            default=5,
            label="KV head count",
        )
        max_position_embeddings = _resolve_aliases(
            max_position_embeddings,
            max_seq_len,
            default=65536,
            label="maximum sequence length",
        )
        rms_norm_eps = float(
            rms_norm_eps if rms_norm_eps is not None else layer_norm_eps if layer_norm_eps is not None else 1e-5
        )
        initializer_range = float(
            initializer_range
            if initializer_range is not None
            else initializer_std
            if initializer_std is not None
            else 0.0098821
        )

        if num_attention_heads % num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if qk_mult <= 0 or qk_mult_long_scale <= 0:
            raise ValueError("qk_mult and qk_mult_long_scale must be positive")
        if rope_theta <= 0 or rms_norm_eps <= 0 or initializer_range <= 0:
            raise ValueError("rope_theta, rms_norm_eps, and initializer_range must be positive")
        if num_experts_per_tok >= num_local_experts:
            raise ValueError("Grug top-(K+1) routing requires num_experts_per_tok < num_local_experts")
        if head_dim is None:
            if hidden_size % num_attention_heads:
                raise ValueError("hidden_size must be divisible by num_attention_heads when head_dim is omitted")
            head_dim = hidden_size // num_attention_heads
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by four for half-RoPE")
        if shared_expert_intermediate_size <= 0:
            raise ValueError("Grug requires an always-on shared expert")
        if sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if not disable_pko:
            raise ValueError("Grug training supports only disable_pko=true")
        if not disable_long_rope:
            raise ValueError("Grug training supports only disable_long_rope=true")
        if grugmoe_attention_mode != GRUG_MOE_ATTENTION_MODE:
            raise ValueError(f"unsupported Grug attention mode {grugmoe_attention_mode!r}")
        if int(grugmoe_artifact_schema_version) not in (
            GRUG_STACKED_EXPERT_SCHEMA_VERSION,
            GRUG_SPLIT_EXPERT_SCHEMA_VERSION,
        ):
            raise ValueError(
                f"unsupported Grug artifact schema {grugmoe_artifact_schema_version}; "
                "expected 1 (stacked experts) or 2 (split experts)"
            )
        if latent_dim is not None and not 0 < latent_dim <= hidden_size:
            raise ValueError("latent_dim must be in (0, hidden_size]")
        if num_shared_experts < 1 or global_every < 1 or sconv_kernel < 1:
            raise ValueError("shared expert count, global period, and convolution width must be positive")
        if (local_kv_heads is None) != (global_kv_heads is None):
            raise ValueError("local_kv_heads and global_kv_heads must be specified together")
        if local_kv_heads is not None:
            if any(n <= 0 or num_attention_heads % n for n in (local_kv_heads, global_kv_heads)):
                raise ValueError("local and global KV head counts must divide the query head count")
            if max(local_kv_heads, global_kv_heads) != num_key_value_heads:
                raise ValueError("num_key_value_heads must equal the stored maximum local/global KV heads")
        if set(sconv_sites) - {"k", "attn", "mlp"}:
            raise ValueError("ShortConv sites must be k, attn, or mlp")
        if use_cache:
            raise ValueError("Grug training does not support KV cache")
        if tie_word_embeddings:
            raise ValueError("Grug checkpoints use untied embeddings")
        if router_z_loss_coef != 0.0:
            raise ValueError("Grug RL requires router_z_loss_coef=0")

        architectures = kwargs.pop("architectures", None)
        if architectures not in (None, [GRUG_MOE_ARCHITECTURE]):
            raise ValueError(f"unsupported Grug architectures value {architectures!r}")
        super().__init__(
            use_cache=False,
            tie_word_embeddings=False,
            architectures=[GRUG_MOE_ARCHITECTURE],
            **kwargs,
        )
        self.vocab_size = int(vocab_size)
        self.hidden_size = self.hidden_dim = hidden_size
        self.intermediate_size = self.intermediate_dim = self.moe_intermediate_size = intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.shared_expert_intermediate_dim = shared_expert_intermediate_size
        self.num_local_experts = self.num_experts = num_local_experts
        self.num_experts_per_tok = self.num_experts_per_token = num_experts_per_tok
        self.num_hidden_layers = self.num_layers = num_hidden_layers
        self.num_attention_heads = self.num_heads = num_attention_heads
        self.num_key_value_heads = self.num_kv_heads = num_key_value_heads
        self.head_dim = int(head_dim)
        self.max_position_embeddings = self.max_seq_len = max_position_embeddings
        self.sliding_window = int(sliding_window)
        self.rms_norm_eps = self.layer_norm_eps = rms_norm_eps
        self.initializer_range = self.initializer_std = initializer_range
        self.qk_mult = float(qk_mult)
        self.qk_mult_long_scale = float(qk_mult_long_scale)
        self.rope_theta = float(rope_theta)
        self.disable_pko = True
        self.disable_long_rope = True
        self.router_z_loss_coef = float(router_z_loss_coef)
        self.grugmoe_attention_mode = grugmoe_attention_mode
        self.grugmoe_artifact_schema_version = int(grugmoe_artifact_schema_version)
        self.latent_dim = latent_dim
        self.num_shared_experts = num_shared_experts
        self.local_kv_heads = local_kv_heads
        self.global_kv_heads = global_kv_heads
        self.global_every = global_every
        self.rope_fused = rope_fused
        self.sconv = sconv
        self.sconv_kernel = sconv_kernel
        self.sconv_sites = tuple(sconv_sites)

    @property
    def uses_hero_architecture(self) -> bool:
        return (
            self.latent_dim is not None
            or self.sconv
            or self.local_kv_heads is not None
            or self.num_shared_experts != 1
            or self.rope_fused
            or self.global_every != GRUG_DEFAULT_GLOBAL_EVERY
            or self.grugmoe_artifact_schema_version == GRUG_SPLIT_EXPERT_SCHEMA_VERSION
        )


class GrugMoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        fp32 = hidden_states.float()
        variance = fp32.square().mean(dim=-1, keepdim=True)
        normalized = fp32 * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


def grug_rms_norm_no_weight(hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    fp32 = hidden_states.float()
    return (fp32 * torch.rsqrt(fp32.square().mean(dim=-1, keepdim=True) + GRUG_QK_RMS_NORM_EPS)).to(input_dtype)


class GrugMoeGatedNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, GRUG_GATED_NORM_RANK, bias=False)
        self.up_proj = nn.Linear(GRUG_GATED_NORM_RANK, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.up_proj(F.silu(self.down_proj(hidden_states))))
        return hidden_states * gate.to(hidden_states.dtype)


class GrugMoeDenseMLP(nn.Module):
    def __init__(self, config: GrugMoeConfig, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class _GrugStackedLinear(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape))


class _GrugZeroInitLinear(nn.Linear):
    pass


class GrugMoeExperts(nn.Module, ExpertGradientAveraging):
    def __init__(self, config: GrugMoeConfig) -> None:
        super().__init__()
        num_experts = config.num_local_experts
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.num_experts = num_experts
        self.gate_proj = _GrugStackedLinear((num_experts, intermediate_size, hidden_size))
        self.up_proj = _GrugStackedLinear((num_experts, intermediate_size, hidden_size))
        self.down_proj = _GrugStackedLinear((num_experts, hidden_size, intermediate_size))
        self.use_grouped_mm = False
        self.ep_size = 1
        self._grouped_mm_logged = False

    def validate_expert_parallel_runtime(self, ep_comm_backend: str) -> None:
        validate_grug_expert_parallel_runtime(
            use_grouped_mm=self.use_grouped_mm,
            ep_comm_backend=ep_comm_backend,
        )

    def forward_eager(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        combine_weights: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden_states, dtype=torch.float32)
        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_input = hidden_states.index_select(0, token_idx)
            gate = F.linear(expert_input, self.gate_proj.weight[expert_idx])
            up = F.linear(expert_input, self.up_proj.weight[expert_idx])
            expert_output = F.linear(F.silu(gate) * up, self.down_proj.weight[expert_idx])
            weight = combine_weights[token_idx, slot_idx].unsqueeze(-1).to(expert_output.dtype)
            output.index_add_(0, token_idx, (expert_output * weight).float())
        return output.to(hidden_states.dtype)

    def forward(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Run routed rows grouped by expert and return them in the same order."""

        if not self.use_grouped_mm:
            raise RuntimeError("Grug grouped expert forward called without use_grouped_mm=true")
        if not self._grouped_mm_logged:
            logger.info(
                "[Grug-MoE-PATH] native grouped_mm "
                f"(gate={type(self.gate_proj.weight).__name__}, routed_rows={routed_input.shape[0]})"
            )
            self._grouped_mm_logged = True

        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
        gate = torch._grouped_mm(
            routed_input.bfloat16(), self.gate_proj.weight.bfloat16().transpose(-2, -1), offs=offsets
        )
        up = torch._grouped_mm(routed_input.bfloat16(), self.up_proj.weight.bfloat16().transpose(-2, -1), offs=offsets)
        hidden = F.silu(gate) * up
        return torch._grouped_mm(hidden, self.down_proj.weight.bfloat16().transpose(-2, -1), offs=offsets).to(
            routed_input.dtype
        )


class GrugMoeRouterOutput(NamedTuple):
    router_logits: torch.Tensor
    selected_experts: torch.Tensor
    combine_weights: torch.Tensor


class GrugMoeRouter(nn.Module, NativeRouterObserverEmitter):
    def __init__(self, config: GrugMoeConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.num_local_experts, config.hidden_size))
        self.register_buffer("bias", torch.zeros(config.num_local_experts, dtype=torch.float32), persistent=True)
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self._capture_q: int | None = None
        self._capture_active = False
        self._capture_mask: torch.Tensor | None = None
        self._observation: GrugQueryBiasLayerObservation | None = None

    def begin_query_bias_capture(self, candidate_count: int, token_mask: torch.Tensor) -> None:
        if candidate_count < 1:
            raise ValueError(f"candidate_count must be positive, got {candidate_count}")
        self._capture_q = candidate_count
        self._capture_active = True
        self._capture_mask = token_mask.reshape(-1).to(dtype=torch.bool)
        self._observation = None

    def begin_loss_free_bias_capture(self, token_mask: torch.Tensor) -> None:
        self._capture_q = None
        self._capture_active = True
        self._capture_mask = token_mask.reshape(-1).to(dtype=torch.bool)
        self._observation = None

    def take_query_bias_observation(self) -> GrugQueryBiasLayerObservation:
        observation = self._observation
        self._capture_q = None
        self._capture_active = False
        self._capture_mask = None
        self._observation = None
        if observation is None:
            raise RuntimeError("query-bias observation was not produced by the loss forward")
        return observation

    def forward(self, hidden_states: torch.Tensor) -> GrugMoeRouterOutput:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            router_logits = F.linear(hidden_states.float(), self.weight.float())
            bias = self.bias.detach().to(device=router_logits.device, dtype=torch.float32)
            biased_logits = router_logits + bias
            topk_logits, topk_indices = jax_top_k(biased_logits, self.top_k + 1)
            alpha = topk_logits[:, -1:]
            selected_experts = topk_indices[:, :-1]
            selected_logits = torch.gather(router_logits, dim=-1, index=selected_experts)
            combine_weights = torch.sigmoid(selected_logits)
            combine_weights = combine_weights * (
                GRUG_ROUTING_RENORM_SUM / (combine_weights.sum(dim=-1, keepdim=True) + GRUG_ROUTER_RENORM_EPS)
            )

            emit_router_forward(
                router=self,
                router_inputs=hidden_states,
                selection_logits=biased_logits,
                natural_selected_experts=selected_experts,
                selected_experts=selected_experts,
                combine_weights=combine_weights,
            )

            if self._capture_active:
                mask = self._capture_mask
                if mask is None or mask.numel() != router_logits.shape[0]:
                    raise RuntimeError("query-bias token mask does not match the router token dimension")
                with torch.no_grad():
                    valid_selected_experts = selected_experts[mask].detach()
                    if valid_selected_experts.shape[0] == 0:
                        raise RuntimeError("query-bias capture requires at least one valid token")
                    if self._capture_q:
                        valid_scores = (router_logits.detach() - alpha.detach())[mask]
                        keep = min(self._capture_q, valid_scores.shape[0])
                        candidates = torch.topk(valid_scores.transpose(0, 1), k=keep, dim=-1, sorted=True).values
                    else:
                        candidates = router_logits.new_empty((self.num_experts, 0))
                    self._observation = GrugQueryBiasLayerObservation(
                        candidates=candidates,
                        selected_experts=valid_selected_experts,
                        combine_weights=combine_weights[mask].detach(),
                    )

        return GrugMoeRouterOutput(router_logits, selected_experts, combine_weights)


class _GrugExpertExecution(Protocol):
    """Run sparse experts through either eager or grouped execution."""

    def run(
        self,
        experts: GrugMoeExperts,
        reorderer: TokenReorderer,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        combine_weights: torch.Tensor,
    ) -> torch.Tensor: ...


class _GrugEagerExpertExecution:
    def run(
        self,
        experts: GrugMoeExperts,
        reorderer: TokenReorderer,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        combine_weights: torch.Tensor,
    ) -> torch.Tensor:
        del reorderer
        return experts.forward_eager(hidden_states, selected_experts, combine_weights.to(hidden_states.dtype))


class _GrugGroupedExpertExecution:
    def run(
        self,
        experts: GrugMoeExperts,
        reorderer: TokenReorderer,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        combine_weights: torch.Tensor,
    ) -> torch.Tensor:
        routed_indices, routed_output = grouped_expert_contributions(
            experts,
            hidden_states,
            combine_weights.to(hidden_states.dtype),
            selected_experts,
            reorderer,
        )
        return (
            torch.zeros_like(hidden_states, dtype=torch.float32)
            .scatter_add(
                dim=0,
                index=routed_indices,
                src=routed_output.float(),
            )
            .to(hidden_states.dtype)
        )


_GRUG_EAGER_EXPERT_EXECUTION = _GrugEagerExpertExecution()
_GRUG_GROUPED_EXPERT_EXECUTION = _GrugGroupedExpertExecution()


class GrugMoeSparseMoeBlock(nn.Module):
    def __init__(self, config: GrugMoeConfig) -> None:
        super().__init__()
        self.router = GrugMoeRouter(config)
        self.experts = GrugMoeExperts(config)
        self.reorderer = TokenReorderer(config.num_local_experts, config.num_experts_per_tok)
        self._expert_execution: _GrugExpertExecution = _GRUG_EAGER_EXPERT_EXECUTION

    def enable_grouped_mm(self) -> None:
        self.experts.use_grouped_mm = True
        self._expert_execution = _GRUG_GROUPED_EXPERT_EXECUTION

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        _, selected_experts, combine_weights = self.router(flat)
        output = self._expert_execution.run(
            self.experts,
            self.reorderer,
            flat,
            selected_experts,
            combine_weights,
        )
        return output.reshape(shape)


def enable_grug_grouped_mm(model: nn.Module) -> int:
    """Enable native grouped execution and return the number of changed blocks."""

    blocks = [module for module in model.modules() if isinstance(module, GrugMoeSparseMoeBlock)]
    for block in blocks:
        block.enable_grouped_mm()
    return len(blocks)


def _apply_half_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = query.shape[-1] // 2
    rope_half = rotary_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(rope_half, device=query.device, dtype=torch.float32) / rope_half))
    angles = position_ids.float().unsqueeze(-1) * inv_freq
    cos = angles.cos().unsqueeze(2)
    sin = angles.sin().unsqueeze(2)

    def rotate(first_half: torch.Tensor) -> torch.Tensor:
        dtype = first_half.dtype
        first, second = first_half.float().chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(dtype)

    return (
        torch.cat((rotate(query[..., :rotary_dim]), query[..., rotary_dim:]), dim=-1),
        torch.cat((rotate(key[..., :rotary_dim]), key[..., rotary_dim:]), dim=-1),
    )


class GrugMoeAttention(nn.Module):
    def __init__(self, config: GrugMoeConfig) -> None:
        super().__init__()
        self.config = config
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.attn_gate = _GrugZeroInitLinear(config.hidden_size, config.num_attention_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        *,
        is_long: bool,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.config.num_attention_heads, self.config.head_dim)
        k = self.k_proj(hidden_states).view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim)
        v = self.v_proj(hidden_states).view(batch, seq_len, self.config.num_key_value_heads, self.config.head_dim)
        q = grug_rms_norm_no_weight(q)
        k = grug_rms_norm_no_weight(k)
        if not is_long:
            q, k = _apply_half_rope(q, k, position_ids, self.config.rope_theta)
        scale = self.config.qk_mult * (self.config.qk_mult_long_scale if is_long else 1.0)
        q = q * scale

        attn_implementation = self.config._attn_implementation
        if attn_implementation == GRUG_EAGER_ATTENTION_BACKEND:
            attn_output, v_for_xsa = self._eager_attention(q, k, v, attention_mask, is_long=is_long)
        elif attn_implementation == GRUG_FLASH_ATTENTION_BACKEND:
            attn_output, v_for_xsa = self._flash_attention(q, k, v, attention_mask, is_long=is_long)
        else:
            raise ValueError(f"unsupported Grug attention backend {attn_implementation!r}")

        dot = (attn_output * v_for_xsa).sum(dim=-1, keepdim=True)
        v_norm_sq = v_for_xsa.square().sum(dim=-1, keepdim=True)
        attn_output = attn_output - (dot / (v_norm_sq + GRUG_XSA_EPS)) * v_for_xsa
        gate = GRUG_ATTN_GATE_SCALE * torch.sigmoid(self.attn_gate(hidden_states)).unsqueeze(-1)
        attn_output = attn_output * gate.to(attn_output.dtype)
        return self.o_proj(attn_output.reshape(batch, seq_len, -1))

    def _repeat_kv_heads(self, states: torch.Tensor) -> torch.Tensor:
        repeats = self.config.num_attention_heads // self.config.num_key_value_heads
        return states if repeats == 1 else states.repeat_interleave(repeats, dim=2)

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        is_long: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the attention output and GQA-expanded value used by XSA."""

        key = self._repeat_kv_heads(key)
        value = self._repeat_kv_heads(value)

        seq_len = query.shape[1]
        scores = torch.einsum(
            "bqhd,bkhd->bhqk",
            query.float() / math.sqrt(self.config.head_dim),
            key.float(),
        )
        query_pos = torch.arange(seq_len, device=query.device).view(seq_len, 1)
        key_pos = torch.arange(seq_len, device=query.device).view(1, seq_len)
        allowed = key_pos <= query_pos
        if not is_long:
            allowed = allowed & (key_pos >= query_pos - (self.config.sliding_window - 1))
        allowed = allowed.view(1, 1, seq_len, seq_len)
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].to(torch.bool)
        scores = torch.where(allowed, scores, torch.tensor(-1e9, dtype=scores.dtype, device=scores.device))
        weights = torch.softmax(scores, dim=-1).to(value.dtype)
        return torch.einsum("bhqk,bkhd->bqhd", weights, value), value

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        is_long: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fused output and GQA-expanded value used by XSA."""

        if FLASH_ATTN_IMPORT_ERROR is not None:
            raise ImportError(
                "Grug FlashAttention was requested, but the flash-attn CUDA extension could not be imported"
            ) from FLASH_ATTN_IMPORT_ERROR
        value_for_xsa = self._repeat_kv_heads(value)
        window_size = (-1, -1) if is_long else (self.config.sliding_window - 1, 0)
        softmax_scale = 1.0 / math.sqrt(self.config.head_dim)
        if attention_mask is None:
            return (
                flash_attn_func(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                ),
                value_for_xsa,
            )

        batch, seq_len = attention_mask.shape
        valid = attention_mask.to(torch.bool)
        unpadded_query, indices, cu_seqlens, max_seqlen, *_ = flash_unpad_input(query, valid)
        unpadded_key = flash_index_first_axis(key.flatten(0, 1), indices)
        unpadded_value = flash_index_first_axis(value.flatten(0, 1), indices)
        unpadded_output = flash_attn_varlen_func(
            unpadded_query,
            unpadded_key,
            unpadded_value,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
        )
        return flash_pad_input(unpadded_output, indices, batch, seq_len), value_for_xsa


class GrugMoeDecoderLayer(nn.Module):
    def __init__(self, config: GrugMoeConfig) -> None:
        super().__init__()
        self.self_attn = GrugMoeAttention(config)
        self.input_layernorm = GrugMoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_gated_norm = GrugMoeGatedNorm(config.hidden_size)
        self.post_attention_layernorm = GrugMoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp_gated_norm = GrugMoeGatedNorm(config.hidden_size)
        self.mlp = GrugMoeSparseMoeBlock(config)
        self.shared_expert = GrugMoeDenseMLP(config, config.shared_expert_intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
        is_long: bool,
    ) -> torch.Tensor:
        attn_input = self.attn_gated_norm(self.input_layernorm(hidden_states))
        hidden_states = hidden_states + self.self_attn(attn_input, attention_mask, position_ids, is_long=is_long)
        mlp_input = self.mlp_gated_norm(self.post_attention_layernorm(hidden_states))
        return hidden_states + self.mlp(mlp_input) + self.shared_expert(mlp_input)


class GrugMoePreTrainedModel(PreTrainedModel):
    config_class = GrugMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GrugMoeDecoderLayer"]
    _supports_sdpa = False
    _supports_flash_attn = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, _GrugZeroInitLinear):
            init.zeros_(module.weight)
        elif isinstance(module, (nn.Linear, nn.Embedding, _GrugStackedLinear)):
            init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, GrugMoeRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.bias)
        elif isinstance(module, GrugMoeRMSNorm):
            init.ones_(module.weight)


class GrugMoeModel(GrugMoePreTrainedModel):
    def __init__(self, config: GrugMoeConfig) -> None:
        super().__init__(config)
        if config.uses_hero_architecture:
            raise ValueError("Hero architecture requires the Megatron backend; the eager model supports Snowball only")
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = GrugMoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_gated_norm = GrugMoeGatedNorm(config.hidden_size)
        self.layers = nn.ModuleList([GrugMoeDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = GrugMoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.final_gated_norm = GrugMoeGatedNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def begin_query_bias_capture(self, candidate_count: int, token_mask: torch.Tensor) -> None:
        for layer in self.layers:
            layer.mlp.router.begin_query_bias_capture(candidate_count, token_mask)

    def begin_loss_free_bias_capture(self, token_mask: torch.Tensor) -> None:
        for layer in self.layers:
            layer.mlp.router.begin_loss_free_bias_capture(token_mask)

    def take_query_bias_observation(self, *, candidate_count: int) -> GrugQueryBiasObservation:
        return GrugQueryBiasObservation(
            layers=tuple(layer.mlp.router.take_query_bias_observation() for layer in self.layers),
            candidate_count=candidate_count,
        )

    def take_loss_free_bias_observation(self) -> GrugQueryBiasObservation:
        return GrugQueryBiasObservation(
            layers=tuple(layer.mlp.router.take_query_bias_observation() for layer in self.layers),
            candidate_count=0,
        )

    def get_query_bias(self) -> torch.Tensor:
        """Return the stacked FP32 router query-bias buffers."""

        return torch.stack([layer.mlp.router.bias for layer in self.layers])

    @torch.no_grad()
    def set_query_bias(self, bias: torch.Tensor) -> None:
        expected = (len(self.layers), self.config.num_local_experts)
        if tuple(bias.shape) != expected:
            raise ValueError(f"query bias must have shape {expected}, got {tuple(bias.shape)}")
        for layer_idx, layer in enumerate(self.layers):
            layer.mlp.router.bias.copy_(bias[layer_idx].to(layer.mlp.router.bias.device, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        **_: Any,
    ) -> BaseModelOutputWithPast | tuple[torch.Tensor, ...]:
        if use_cache:
            raise ValueError("Grug training does not support KV cache")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must be rank two, got shape={tuple(attention_mask.shape)}")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch, -1)
            else:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
        if attention_mask is not None and self.config._attn_implementation == GRUG_FLASH_ATTENTION_BACKEND:
            _validate_flash_attention_mask(attention_mask)
        if tuple(position_ids.shape) != (batch, seq_len):
            raise ValueError(f"position_ids must have shape {(batch, seq_len)}, got {tuple(position_ids.shape)}")

        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        return_dict = self.config.return_dict if return_dict is None else return_dict
        hidden_states = self.embed_gated_norm(self.embed_norm(inputs_embeds))
        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        long_layer_flags = grug_long_layer_flags(len(self.layers))
        for layer_idx, layer in enumerate(self.layers):
            is_long = long_layer_flags[layer_idx]
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__, hidden_states, attention_mask, position_ids, is_long
                )
            else:
                hidden_states = layer(hidden_states, attention_mask, position_ids, is_long)
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

        hidden_states = self.final_gated_norm(self.norm(hidden_states))
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(value for value in (hidden_states, all_hidden_states) if value is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


class GrugMoeForCausalLM(GrugMoePreTrainedModel):
    _tied_weights_keys: list[str] = []

    def __init__(self, config: GrugMoeConfig) -> None:
        super().__init__(config)
        self.model = GrugMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def begin_query_bias_capture(self, candidate_count: int, token_mask: torch.Tensor) -> None:
        self.model.begin_query_bias_capture(candidate_count, token_mask)

    def begin_loss_free_bias_capture(self, token_mask: torch.Tensor) -> None:
        self.model.begin_loss_free_bias_capture(token_mask)

    def take_query_bias_observation(self, *, candidate_count: int) -> GrugQueryBiasObservation:
        return self.model.take_query_bias_observation(candidate_count=candidate_count)

    def take_loss_free_bias_observation(self) -> GrugQueryBiasObservation:
        return self.model.take_loss_free_bias_observation()

    def get_query_bias(self) -> torch.Tensor:
        return self.model.get_query_bias()

    def set_query_bias(self, bias: torch.Tensor) -> None:
        self.model.set_query_bias(bias)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_for_logits = hidden_states[:, -logits_to_keep:, :]
        elif isinstance(logits_to_keep, torch.Tensor):
            hidden_for_logits = hidden_states[:, logits_to_keep, :]
        else:
            hidden_for_logits = hidden_states
        logits = self.lm_head(hidden_for_logits)
        loss = None
        if labels is not None:
            if not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError("labels require logits_to_keep=0")
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            values = (logits, outputs.hidden_states)
            return ((loss,) + values) if loss is not None else values
        return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
