# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness-first GPU and TPU implementation of Marin GrugMoE."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.grugmoe import (
    grug_moe_layer_types,
    grug_moe_rope_theta,
)
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    SupportsEagle,
    SupportsEagle3,
    SupportsPP,
)

logger = init_logger(__name__)

# The trained GrugMoE checkpoint uses a fixed rank-128 gated-norm bottleneck.
_GATED_NORM_RANK = 128
_ROUTER_COMBINE_WEIGHT_SUM = 2.5
_ROUTER_COMBINE_WEIGHT_EPS = 1e-9
_UNQUANTIZED_METHOD = "unquantized"
_HERO_ARTIFACT_SCHEMA_VERSION = 2
_SUPPORTED_SCONV_SITES = frozenset({"k", "attn", "mlp"})


def _config_attr(config: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x_float = x.float()
    variance = torch.mean(torch.square(x_float), dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps)).to(dtype)


def _align_kv_heads(value: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    num_kv_heads = value.shape[1]
    if num_q_heads == num_kv_heads:
        return value
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    repeat = num_q_heads // num_kv_heads
    value = value.unsqueeze(2).expand(
        value.shape[0], num_kv_heads, repeat, value.shape[2]
    )
    return value.reshape(value.shape[0], num_q_heads, value.shape[-1]).contiguous()


def _format_int_ranges(values: list[int]) -> str:
    if not values:
        return "[]"
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(str(start) if start == prev else f"{start}..{prev}")
        start = prev = value
    ranges.append(str(start) if start == prev else f"{start}..{prev}")
    return "[" + ", ".join(ranges) + "]"


def get_grug_moe_runtime_info(
    _vllm_config: VllmConfig,
    model: nn.Module,
) -> dict[str, Any]:
    """Return worker-local GrugMoE runtime state for logs and diagnostics."""
    grug_model = getattr(model, "model", model)
    layers = getattr(grug_model, "layers", ())
    first_layer = next(
        (layer for layer in layers if isinstance(layer, GrugMoeDecoderLayer)),
        None,
    )
    if first_layer is None:
        raise RuntimeError("GrugMoE runtime info requires a local decoder layer")
    moe_runner = first_layer.mlp.experts
    moe_config = moe_runner.moe_config
    moe_parallel_config = moe_config.moe_parallel_config
    expert_map_manager = moe_runner.expert_map_manager
    local_expert_ids = [
        int(expert_id) for expert_id in expert_map_manager.get_local_expert_ids()
    ]
    attention_backend = first_layer.self_attn.attn.attn_backend.get_name()
    pp_group = get_pp_group()
    return {
        "tp_size": int(moe_parallel_config.tp_size),
        "tp_rank": int(moe_parallel_config.tp_rank),
        "pp_size": int(pp_group.world_size),
        "pp_rank": int(pp_group.rank_in_group),
        "start_layer": int(grug_model.start_layer),
        "end_layer": int(grug_model.end_layer),
        "dp_size": int(moe_parallel_config.dp_size),
        "dp_rank": int(moe_parallel_config.dp_rank),
        "ep_size": int(moe_parallel_config.ep_size),
        "ep_rank": int(moe_parallel_config.ep_rank),
        "use_ep": bool(moe_parallel_config.use_ep),
        "num_experts": int(moe_config.num_experts),
        "num_logical_experts": int(moe_config.num_logical_experts),
        "num_local_experts": int(moe_config.num_local_experts),
        "top_k": int(moe_config.experts_per_token),
        "local_expert_ids": local_expert_ids,
        "local_expert_ownership": _format_int_ranges(local_expert_ids),
        "expert_placement_strategy": str(moe_runner.expert_placement_strategy),
        "all2all_backend": str(moe_parallel_config.all2all_backend),
        "attention_backend": str(attention_backend),
    }


def _log_grug_moe_runtime_info(
    vllm_config: VllmConfig,
    model: nn.Module,
) -> None:
    info = get_grug_moe_runtime_info(vllm_config, model)
    logger.info(
        "GrugMoE effective config: TP=%d/%d PP=%d/%d layers=[%d,%d) "
        "DP=%d/%d EP=%d/%d "
        "use_ep=%s experts=%d local=%d top_k=%d ownership=%s "
        "placement=%s all2all=%s attention=%s",
        info["tp_rank"],
        info["tp_size"],
        info["pp_rank"],
        info["pp_size"],
        info["start_layer"],
        info["end_layer"],
        info["dp_rank"],
        info["dp_size"],
        info["ep_rank"],
        info["ep_size"],
        info["use_ep"],
        info["num_experts"],
        info["num_local_experts"],
        info["top_k"],
        info["local_expert_ownership"],
        info["expert_placement_strategy"],
        info["all2all_backend"],
        info["attention_backend"],
    )


@dataclass(frozen=True)
class GrugMoeRuntimeConfig:
    vocab_size: int
    artifact_schema_version: int = 1
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    shared_expert_intermediate_dim: int = 5632
    num_shared_experts: int = 1
    num_experts: int = 8
    num_experts_per_token: int = 2
    latent_dim: int | None = None
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    local_kv_heads: int | None = None
    global_kv_heads: int | None = None
    head_dim: int | None = None
    max_seq_len: int = 4096
    sliding_window: int = 4096
    global_every: int = 4
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.0
    qk_mult_long_scale: float = 1.0
    disable_pko: bool = True
    disable_long_rope: bool = True
    rope_fused: bool = False
    sconv: bool = False
    sconv_kernel: int = 4
    sconv_sites: tuple[str, ...] = ("k", "attn", "mlp")
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0

    @classmethod
    def from_hf_config(cls, config: Any) -> "GrugMoeRuntimeConfig":
        max_seq_len = int(
            _config_attr(config, ("max_seq_len", "max_position_embeddings"), 4096)
        )
        num_heads = int(_config_attr(config, ("num_heads", "num_attention_heads"), 16))
        num_layers = int(_config_attr(config, ("num_layers", "num_hidden_layers"), 24))
        latent_dim = _config_attr(config, ("latent_dim",))
        local_kv_heads = _config_attr(config, ("local_kv_heads",))
        global_kv_heads = _config_attr(config, ("global_kv_heads",))
        return cls(
            vocab_size=int(_config_attr(config, ("vocab_size",))),
            artifact_schema_version=int(
                _config_attr(config, ("grugmoe_artifact_schema_version",), 1)
            ),
            hidden_dim=int(_config_attr(config, ("hidden_dim", "hidden_size"), 2048)),
            intermediate_dim=int(
                _config_attr(
                    config,
                    ("intermediate_dim", "moe_intermediate_size", "intermediate_size"),
                    5632,
                )
            ),
            shared_expert_intermediate_dim=int(
                _config_attr(
                    config,
                    (
                        "shared_expert_intermediate_dim",
                        "shared_expert_intermediate_size",
                    ),
                    5632,
                )
            ),
            num_shared_experts=int(
                _config_attr(config, ("num_shared_experts",), 1)
            ),
            num_experts=int(
                _config_attr(config, ("num_experts", "num_local_experts"), 8)
            ),
            num_experts_per_token=int(
                _config_attr(
                    config, ("num_experts_per_token", "num_experts_per_tok"), 2
                )
            ),
            latent_dim=None if latent_dim is None else int(latent_dim),
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=int(
                _config_attr(config, ("num_kv_heads", "num_key_value_heads"), num_heads)
            ),
            local_kv_heads=(
                None if local_kv_heads is None else int(local_kv_heads)
            ),
            global_kv_heads=(
                None if global_kv_heads is None else int(global_kv_heads)
            ),
            head_dim=_config_attr(config, ("head_dim", "attention_head_dim")),
            max_seq_len=max_seq_len,
            sliding_window=int(_config_attr(config, ("sliding_window",), max_seq_len)),
            global_every=int(_config_attr(config, ("global_every",), 4)),
            layer_norm_eps=float(
                _config_attr(config, ("layer_norm_eps", "rms_norm_eps"), 1e-5)
            ),
            initializer_std=float(
                _config_attr(config, ("initializer_std", "initializer_range"), 0.02)
            ),
            qk_mult=float(_config_attr(config, ("qk_mult",), 1.0)),
            qk_mult_long_scale=float(
                _config_attr(config, ("qk_mult_long_scale",), 1.0)
            ),
            disable_pko=bool(_config_attr(config, ("disable_pko",), True)),
            disable_long_rope=bool(_config_attr(config, ("disable_long_rope",), True)),
            rope_fused=bool(_config_attr(config, ("rope_fused",), False)),
            sconv=bool(_config_attr(config, ("sconv",), False)),
            sconv_kernel=int(_config_attr(config, ("sconv_kernel",), 4)),
            sconv_sites=tuple(
                _config_attr(config, ("sconv_sites",), ("k", "attn", "mlp"))
            ),
            tie_word_embeddings=bool(
                _config_attr(config, ("tie_word_embeddings",), False)
            ),
            rope_theta=grug_moe_rope_theta(config),
        ).validate()

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return int(self.head_dim)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by "
                f"num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads

    @property
    def attention_layer_types(self) -> tuple[str, ...]:
        return tuple(grug_moe_layer_types(self.num_layers, self.global_every))

    @property
    def is_hero(self) -> bool:
        return self.artifact_schema_version == _HERO_ARTIFACT_SCHEMA_VERSION

    @property
    def expert_dim(self) -> int:
        return self.hidden_dim if self.latent_dim is None else self.latent_dim

    def logical_kv_heads(self, *, is_global: bool) -> int:
        if self.local_kv_heads is None or self.global_kv_heads is None:
            return self.num_kv_heads
        return self.global_kv_heads if is_global else self.local_kv_heads

    def validate(self) -> "GrugMoeRuntimeConfig":
        if self.artifact_schema_version not in (
            1,
            _HERO_ARTIFACT_SCHEMA_VERSION,
        ):
            raise ValueError(
                "unsupported grugmoe_artifact_schema_version="
                f"{self.artifact_schema_version}"
            )
        if self.artifact_schema_version == 1:
            hero_fields = []
            if self.num_shared_experts != 1:
                hero_fields.append("num_shared_experts")
            if self.latent_dim is not None:
                hero_fields.append("latent_dim")
            if self.local_kv_heads is not None or self.global_kv_heads is not None:
                hero_fields.extend(("local_kv_heads", "global_kv_heads"))
            if self.global_every != 4:
                hero_fields.append("global_every")
            if self.rope_fused:
                hero_fields.append("rope_fused")
            if self.sconv:
                hero_fields.append("sconv")
            if self.sconv_kernel != 4:
                hero_fields.append("sconv_kernel")
            if self.sconv_sites != ("k", "attn", "mlp"):
                hero_fields.append("sconv_sites")
            if hero_fields:
                raise ValueError(
                    ", ".join(hero_fields)
                    + " require grugmoe_artifact_schema_version=2"
                )
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.intermediate_dim <= 0:
            raise ValueError("intermediate_dim must be positive")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")
        if self.num_shared_experts <= 0:
            raise ValueError("num_shared_experts must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token >= self.num_experts:
            raise ValueError(
                "num_experts_per_token must be < num_experts for QB routing"
            )
        if self.latent_dim is not None and not 0 < self.latent_dim <= self.hidden_dim:
            raise ValueError(
                f"latent_dim must be in (0, hidden_dim={self.hidden_dim}]"
            )
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (self.local_kv_heads is None) != (self.global_kv_heads is None):
            raise ValueError("local_kv_heads and global_kv_heads must be set together")
        if self.local_kv_heads is not None and self.global_kv_heads is not None:
            if self.local_kv_heads <= 0 or self.global_kv_heads <= 0:
                raise ValueError("local_kv_heads and global_kv_heads must be positive")
            if (
                self.num_heads % self.local_kv_heads != 0
                or self.num_heads % self.global_kv_heads != 0
            ):
                raise ValueError(
                    "num_heads must be divisible by both local and global "
                    "KV-head counts"
                )
            if self.num_kv_heads != max(
                self.local_kv_heads, self.global_kv_heads
            ):
                raise ValueError(
                    "num_kv_heads must equal the stored maximum of local/global "
                    "KV heads"
                )
        if self.inferred_head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.inferred_head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.sliding_window <= 1:
            raise ValueError("sliding_window must be greater than 1")
        if self.global_every <= 0:
            raise ValueError("global_every must be positive")
        if self.sconv and not 2 <= self.sconv_kernel <= 5:
            raise ValueError("sconv_kernel must be between 2 and 5")
        unknown_sconv_sites = set(self.sconv_sites) - _SUPPORTED_SCONV_SITES
        if unknown_sconv_sites:
            raise ValueError(
                "unsupported sconv_sites: "
                + ", ".join(sorted(unknown_sconv_sites))
            )
        _ = self.attention_layer_types
        return self


class GrugMoeGatedNorm(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        params_dtype: torch.dtype,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.down_proj = ReplicatedLinear(
            hidden_dim,
            _GATED_NORM_RANK,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.down_proj",
        )
        self.up_proj = ReplicatedLinear(
            _GATED_NORM_RANK,
            hidden_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.up_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        gate_hidden = _apply_grug_linear(self.down_proj, x)
        gate_hidden = F.silu(gate_hidden)
        gate = _apply_grug_linear(self.up_proj, gate_hidden)
        return x * torch.sigmoid(gate).to(dtype)


class GrugMoeDenseMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        params_dtype: torch.dtype,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = ReplicatedLinear(
            hidden_dim,
            intermediate_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ReplicatedLinear(
            hidden_dim,
            intermediate_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = ReplicatedLinear(
            intermediate_dim,
            hidden_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out


class GrugMoeRouter(BaseRouter):
    """Grug QB router: biased top-(K+1), drop threshold, normalized sigmoid."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        bias: torch.Tensor,
    ) -> None:
        super().__init__(top_k=top_k, global_num_experts=global_num_experts)
        self.bias = bias

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Custom

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states, input_ids
        _, selected = torch.topk(
            router_logits.float() + self.bias.float(),
            k=self.top_k + 1,
            dim=-1,
        )
        selected = selected[:, : self.top_k]
        topk_weights = torch.sigmoid(
            torch.gather(router_logits.float(), dim=-1, index=selected)
        )
        denom = topk_weights.sum(dim=-1, keepdim=True) + _ROUTER_COMBINE_WEIGHT_EPS
        topk_weights = topk_weights * (_ROUTER_COMBINE_WEIGHT_SUM / denom)
        topk_ids = selected.to(torch.int32 if indices_type is None else indices_type)
        return topk_weights, topk_ids


def _apply_grug_linear(layer: ReplicatedLinear, x: torch.Tensor) -> torch.Tensor:
    """Apply a Grug projection without adding its separately handled bias."""
    output, _ = layer(x)
    return output


class GrugMoeMLP(nn.Module):
    """QB-routed MoE with normalized sigmoid combine weights."""

    def __init__(
        self,
        cfg: GrugMoeRuntimeConfig,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        params_dtype = params_dtype or torch.get_default_dtype()
        self.router = ReplicatedLinear(
            cfg.hidden_dim,
            cfg.num_experts,
            bias=True,
            params_dtype=torch.float32,
            quant_config=quant_config,
            disable_tp=True,
            skip_bias_add=True,
            prefix=f"{prefix}.router",
        )
        moe_router = GrugMoeRouter(
            top_k=cfg.num_experts_per_token,
            global_num_experts=cfg.num_experts,
            bias=self.router.bias,
        )
        self.latent_down_proj = (
            ReplicatedLinear(
                cfg.hidden_dim,
                cfg.expert_dim,
                bias=False,
                params_dtype=params_dtype,
                quant_config=quant_config,
                disable_tp=True,
                prefix=f"{prefix}.latent_down_proj",
            )
            if cfg.latent_dim is not None
            else None
        )
        self.latent_norm = (
            RMSNorm(cfg.expert_dim, eps=cfg.layer_norm_eps)
            if cfg.latent_dim is not None
            else None
        )
        self.latent_up_proj = (
            ReplicatedLinear(
                cfg.expert_dim,
                cfg.hidden_dim,
                bias=False,
                params_dtype=params_dtype,
                quant_config=quant_config,
                disable_tp=True,
                prefix=f"{prefix}.latent_up_proj",
            )
            if cfg.latent_dim is not None
            else None
        )
        self.experts = FusedMoEFactory(
            num_experts=cfg.num_experts,
            top_k=cfg.num_experts_per_token,
            hidden_size=cfg.expert_dim,
            intermediate_size=cfg.intermediate_dim,
            params_dtype=params_dtype,
            renormalize=False,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            router=moe_router,
            router_logits_dtype=torch.float32,
        )

    def _routed_input(self, x_flat: torch.Tensor) -> torch.Tensor:
        if self.latent_down_proj is None or self.latent_norm is None:
            return x_flat
        return self.latent_norm(_apply_grug_linear(self.latent_down_proj, x_flat))

    def _expand_routed_output(self, routed_output: torch.Tensor) -> torch.Tensor:
        if self.latent_up_proj is None:
            return routed_output
        return _apply_grug_linear(self.latent_up_proj, routed_output)

    def _forward_torch_reference(
        self,
        x_flat: torch.Tensor,
        routed_input: torch.Tensor,
    ) -> torch.Tensor:
        router_logits = _apply_grug_linear(self.router, x_flat.float())
        combine_weights, selected = self.experts.router.select_experts(
            hidden_states=x_flat,
            router_logits=router_logits,
        )
        selected = selected.long()
        routed_experts = self.experts.routed_experts
        w13_weight = routed_experts.w13_weight
        gate_proj_weight = w13_weight[:, : self.cfg.intermediate_dim, :]
        up_proj_weight = w13_weight[:, self.cfg.intermediate_dim :, :]
        down_proj_weight = routed_experts.w2_weight

        gate_weight = gate_proj_weight[selected]
        up_weight = up_proj_weight[selected]
        down_weight = down_proj_weight[selected]
        gate = torch.einsum(
            "td,tkid->tki",
            routed_input,
            gate_weight.to(routed_input.dtype),
        )
        up = torch.einsum(
            "td,tkid->tki",
            routed_input,
            up_weight.to(routed_input.dtype),
        )
        expert_out = torch.einsum(
            "tki,tkdi->tkd",
            F.silu(gate) * up,
            down_weight.to(routed_input.dtype),
        )
        routed_output = torch.sum(
            expert_out * combine_weights.unsqueeze(-1), dim=1
        )
        return self._expand_routed_output(routed_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        hidden_dim = orig_shape[-1]
        x_flat = x.reshape(-1, hidden_dim)
        # TPU functional_call replaces registered parameters with sharded
        # tensors. Keep the non-Module custom router bound to that current
        # bias rather than the CPU Parameter captured during construction.
        self.experts.router.bias = self.router.bias
        routed_input = self._routed_input(x_flat)
        if not x_flat.is_cuda and not current_platform.is_tpu():
            out = self._forward_torch_reference(x_flat, routed_input)
            return out.to(x.dtype).reshape(orig_shape)
        router_logits = _apply_grug_linear(self.router, x_flat.float())
        out = self.experts(
            hidden_states=routed_input,
            router_logits=router_logits,
        )
        out = self._expand_routed_output(out)
        return out.to(x.dtype).reshape(orig_shape)


class GrugMoeShortConv(MambaBase, nn.Module):
    """Stateful depthwise causal convolution for exported GrugMoE weights."""

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        params_dtype: torch.dtype,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.params_dtype = params_dtype
        self.cache_config = cache_config
        self.prefix = prefix
        # Marin exports [lag, channel], with lag zero being the current token.
        self.weight = nn.Parameter(
            torch.empty(kernel_size, dim, dtype=params_dtype),
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]),)

    def _conv_weight(self) -> torch.Tensor:
        # causal_conv1d follows torch.conv1d's oldest-to-current convention.
        return self.weight.transpose(0, 1).flip(-1).contiguous()

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            # Profiling only needs the right shape. The lag-zero path is also
            # the exact result for the zero history used by identity init.
            output.copy_(hidden_states * self.weight[0])
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, ShortConvAttentionMetadata)
        conv_state = (
            self.kv_cache[0]
            if is_conv_state_dim_first()
            else self.kv_cache[0].transpose(-1, -2)
        )
        conv_weight = self._conv_weight()

        num_decodes = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_actual_tokens = num_decodes + num_prefill_tokens
        decode, prefill = torch.split(
            hidden_states[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        pieces: list[torch.Tensor] = []

        if attn_metadata.num_prefills > 0:
            assert attn_metadata.state_indices_tensor_p is not None
            prefill_output = causal_conv1d_fn(
                prefill.transpose(0, 1),
                conv_weight,
                None,
                conv_state,
                attn_metadata.query_start_loc_p,
                cache_indices=attn_metadata.state_indices_tensor_p,
                has_initial_state=attn_metadata.has_initial_states_p,
                activation=None,
                metadata=(
                    attn_metadata if attn_metadata.nums_dict is not None else None
                ),
            ).transpose(0, 1)[:num_prefill_tokens]
            pieces.append(prefill_output)

        if num_decodes > 0:
            assert attn_metadata.state_indices_tensor_d is not None
            decode_output = torch.empty_like(decode)
            decode_output = causal_conv1d_update(
                decode.contiguous(),
                conv_state,
                conv_weight,
                None,
                activation=None,
                conv_state_indices=attn_metadata.state_indices_tensor_d,
                out=decode_output,
            )
            pieces.insert(0, decode_output)

        output[:num_actual_tokens] = torch.vstack(pieces)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(hidden_states)
        torch.ops.vllm.grug_moe_short_conv(hidden_states, output, self.prefix)
        return output

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.params_dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, int]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.dim,
            conv_kernel=self.kernel_size,
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV


def _grug_moe_short_conv(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(hidden_states, output)


def _grug_moe_short_conv_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="grug_moe_short_conv",
    op_func=_grug_moe_short_conv,
    mutates_args=["output"],
    fake_impl=_grug_moe_short_conv_fake,
)


class GrugMoeAttention(nn.Module):
    def __init__(
        self,
        cfg: GrugMoeRuntimeConfig,
        cache_config: CacheConfig | None,
        params_dtype: torch.dtype,
        sliding_window: int | None,
        use_rope: bool,
        qk_mult_scale: float,
        is_global: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.inferred_head_dim
        self.q_size = cfg.num_heads * self.head_dim
        self.kv_size = cfg.num_kv_heads * self.head_dim
        self.use_rope = use_rope
        self.logical_kv_heads = cfg.logical_kv_heads(is_global=is_global)
        self.qk_mult_scale = qk_mult_scale

        self.q_proj = ReplicatedLinear(
            cfg.hidden_dim,
            self.q_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ReplicatedLinear(
            cfg.hidden_dim,
            self.kv_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ReplicatedLinear(
            cfg.hidden_dim,
            self.kv_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = ReplicatedLinear(
            self.q_size,
            cfg.hidden_dim,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.o_proj",
        )
        self.attn_gate = ReplicatedLinear(
            cfg.hidden_dim,
            cfg.num_heads,
            bias=False,
            params_dtype=params_dtype,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.attn_gate",
        )
        self.sconv_k = (
            GrugMoeShortConv(
                self.kv_size,
                cfg.sconv_kernel,
                params_dtype,
                cache_config,
                prefix=f"{prefix}.sconv_k",
            )
            if cfg.is_hero and cfg.sconv and "k" in cfg.sconv_sites
            else None
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=cfg.max_seq_len,
            rope_parameters={
                "rope_theta": cfg.rope_theta,
                "partial_rotary_factor": 0.5,
            },
            is_neox_style=not cfg.rope_fused,
        )
        self.attn = Attention(
            cfg.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=cfg.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def apply_xsa(
        self,
        hidden_states: torch.Tensor,
        attn_output: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        value_heads = value_states.view(
            num_tokens, self.cfg.num_kv_heads, self.head_dim
        )
        aligned_value = _align_kv_heads(value_heads, self.cfg.num_heads)
        attn_heads = attn_output.view(num_tokens, self.cfg.num_heads, self.head_dim)

        dot = torch.sum(attn_heads * aligned_value, dim=-1, keepdim=True)
        value_norm_sq = torch.sum(aligned_value * aligned_value, dim=-1, keepdim=True)
        attn_heads = attn_heads - (dot / (value_norm_sq + 1e-6)) * aligned_value
        gate = _apply_grug_linear(self.attn_gate, hidden_states)
        gate = 2 * torch.sigmoid(gate)
        attn_heads = gate[..., None].to(attn_heads.dtype) * attn_heads
        return attn_heads.reshape(num_tokens, self.q_size)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)
        if self.sconv_k is not None:
            k = self.sconv_k(k)

        num_tokens = hidden_states.shape[0]
        q = _rms_norm(q.view(num_tokens, self.cfg.num_heads, self.head_dim))
        k = k.view(num_tokens, self.cfg.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.cfg.num_kv_heads, self.head_dim)
        if self.logical_kv_heads != self.cfg.num_kv_heads:
            k = _align_kv_heads(k[:, : self.logical_kv_heads], self.cfg.num_kv_heads)
            v = _align_kv_heads(v[:, : self.logical_kv_heads], self.cfg.num_kv_heads)
        k = _rms_norm(k)
        q = q.reshape(num_tokens, self.q_size)
        k = k.reshape(num_tokens, self.kv_size)
        v = v.reshape(num_tokens, self.kv_size)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        q = q * self.cfg.qk_mult * self.qk_mult_scale

        attn_output = self.attn(q, k, v)
        attn_output = self.apply_xsa(hidden_states, attn_output, v)
        output, _ = self.o_proj(attn_output)
        return output


class GrugMoeDecoderLayer(nn.Module):
    def __init__(
        self,
        cfg: GrugMoeRuntimeConfig,
        cache_config: CacheConfig | None,
        params_dtype: torch.dtype,
        layer_index: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        is_long = cfg.attention_layer_types[layer_index] == "full_attention"
        sliding_window = None if is_long else cfg.sliding_window
        self.input_layernorm = RMSNorm(cfg.hidden_dim, eps=cfg.layer_norm_eps)
        self.attn_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim,
            params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.attn_gated_norm",
        )
        self.self_attn = GrugMoeAttention(
            cfg,
            cache_config,
            params_dtype,
            sliding_window,
            use_rope=not (is_long and cfg.disable_long_rope),
            qk_mult_scale=cfg.qk_mult_long_scale if is_long else 1.0,
            is_global=is_long,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(
            cfg.hidden_dim,
            eps=cfg.layer_norm_eps,
        )
        self.mlp_gated_norm = GrugMoeGatedNorm(
            cfg.hidden_dim,
            params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp_gated_norm",
        )
        self.mlp = GrugMoeMLP(
            cfg,
            params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.shared_expert = (
            GrugMoeDenseMLP(
                cfg.hidden_dim,
                cfg.shared_expert_intermediate_dim,
                params_dtype,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_expert",
            )
            if not cfg.is_hero and cfg.shared_expert_intermediate_dim > 0
            else None
        )
        self.shared_experts = nn.ModuleList(
            [
                GrugMoeDenseMLP(
                    cfg.hidden_dim,
                    cfg.shared_expert_intermediate_dim,
                    params_dtype,
                    quant_config=quant_config,
                    prefix=f"{prefix}.shared_experts.{shared_index}",
                )
                for shared_index in range(cfg.num_shared_experts)
            ]
            if cfg.is_hero and cfg.shared_expert_intermediate_dim > 0
            else []
        )
        self.sconv_attn = (
            GrugMoeShortConv(
                cfg.hidden_dim,
                cfg.sconv_kernel,
                params_dtype,
                cache_config,
                prefix=f"{prefix}.sconv_attn",
            )
            if cfg.is_hero and cfg.sconv and "attn" in cfg.sconv_sites
            else None
        )
        self.sconv_mlp = (
            GrugMoeShortConv(
                cfg.hidden_dim,
                cfg.sconv_kernel,
                params_dtype,
                cache_config,
                prefix=f"{prefix}.sconv_mlp",
            )
            if cfg.is_hero and cfg.sconv and "mlp" in cfg.sconv_sites
            else None
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        attn_in = self.attn_gated_norm(self.input_layernorm(hidden_states))
        attn_out = self.self_attn(positions, attn_in)
        if self.sconv_attn is not None:
            attn_out = self.sconv_attn(attn_out)
        hidden_states = hidden_states + attn_out

        mlp_in = self.mlp_gated_norm(self.post_attention_layernorm(hidden_states))
        mlp_out = self.mlp(mlp_in)
        if self.shared_expert is not None:
            mlp_out = mlp_out + self.shared_expert(mlp_in)
        for shared_expert in self.shared_experts:
            mlp_out = mlp_out + shared_expert(mlp_in)
        if self.sconv_mlp is not None:
            mlp_out = self.sconv_mlp(mlp_out)
        return hidden_states + mlp_out


@support_torch_compile
class GrugMoeModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = vllm_config.model_config.hf_config
        self.config = GrugMoeRuntimeConfig.from_hf_config(hf_config)
        if not self.config.disable_pko:
            raise NotImplementedError(
                "GrugMoE does not support Partial Key Offset; export a checkpoint "
                "with disable_pko=true"
            )
        self.params_dtype = vllm_config.model_config.dtype
        self.quant_config = vllm_config.quant_config

        if get_pp_group().is_first_rank or (
            self.config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_dim,
                params_dtype=self.params_dtype,
                quant_config=self.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        if get_pp_group().is_first_rank:
            self.embed_norm = RMSNorm(
                self.config.hidden_dim,
                eps=self.config.layer_norm_eps,
            )
            self.embed_gated_norm = GrugMoeGatedNorm(
                self.config.hidden_dim,
                self.params_dtype,
                quant_config=self.quant_config,
                prefix=f"{prefix}.embed_gated_norm",
            )
        else:
            self.embed_norm = PPMissingLayer()
            self.embed_gated_norm = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_layers,
            lambda prefix: GrugMoeDecoderLayer(
                self.config,
                vllm_config.cache_config,
                self.params_dtype,
                layer_index=extract_layer_index(prefix),
                quant_config=self.quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(
                self.config.hidden_dim,
                eps=self.config.layer_norm_eps,
            )
            self.final_gated_norm = GrugMoeGatedNorm(
                self.config.hidden_dim,
                self.params_dtype,
                quant_config=self.quant_config,
                prefix=f"{prefix}.final_gated_norm",
            )
        else:
            self.norm = PPMissingLayer()
            self.final_gated_norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_dim
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError(
                        "input_ids must be provided when inputs_embeds is None"
                    )
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = self.embed_gated_norm(self.embed_norm(hidden_states))
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, None
        )
        for layer_index, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states = layer(positions, hidden_states)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_index + 1, hidden_states, None
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states = self.final_gated_norm(self.norm(hidden_states))
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


def _raise_for_unsupported_modes(vllm_config: VllmConfig) -> None:
    parallel_config = vllm_config.parallel_config
    is_tpu = current_platform.is_tpu()
    if vllm_config.lora_config is not None:
        raise NotImplementedError("GrugMoE does not support LoRA")
    quant_config = vllm_config.quant_config
    if quant_config is not None and (
        not is_tpu or quant_config.get_name() != _UNQUANTIZED_METHOD
    ):
        raise NotImplementedError("GrugMoE does not support quantization")
    if not is_tpu and parallel_config.tensor_parallel_size != 1:
        raise NotImplementedError(
            "GrugMoE expert-parallel GPU serving requires tensor_parallel_size=1"
        )
    if (
        not is_tpu
        and parallel_config.tensor_parallel_size
        != get_tensor_model_parallel_world_size()
    ):
        raise RuntimeError(
            "GrugMoE tensor_parallel_size does not match the initialized "
            "tensor-parallel world size: "
            f"{parallel_config.tensor_parallel_size} != "
            f"{get_tensor_model_parallel_world_size()}"
        )


_EXPERT_WEIGHT_MAPPING: tuple[tuple[str, str, str], ...] = (
    (
        "experts.gate_proj.weight",
        "experts.routed_experts.w13_weight",
        "w1",
    ),
    (
        "experts.down_proj.weight",
        "experts.routed_experts.w2_weight",
        "w2",
    ),
    (
        "experts.up_proj.weight",
        "experts.routed_experts.w13_weight",
        "w3",
    ),
)


def _try_load_grug_expert_weight(
    name: str,
    loaded_weight: torch.Tensor,
    params_dict: dict[str, nn.Parameter],
) -> str | None:
    for weight_name, param_name, shard_id in _EXPERT_WEIGHT_MAPPING:
        if weight_name not in name:
            continue
        mapped_name = name.replace(weight_name, param_name)
        param = params_dict.get(mapped_name)
        if param is None:
            raise ValueError(
                "Mapped Grug expert weight "
                f"{name!r} to missing parameter {mapped_name!r}"
            )
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is None:
            raise ValueError(
                f"Grug expert parameter {mapped_name!r} has no FusedMoE weight_loader"
            )
        if loaded_weight.dim() == 3:
            loaded_experts = loaded_weight.unbind(dim=0)
        else:
            loaded_experts = (loaded_weight,)
        for expert_id, loaded_expert in enumerate(loaded_experts):
            weight_loader(
                param,
                loaded_expert,
                mapped_name,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            )
        return mapped_name
    return None


class GrugMoeForCausalLM(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
):
    fall_back_to_pt_during_load = False

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int]]:
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = vllm_config.model_config.hf_config
        cfg = GrugMoeRuntimeConfig.from_hf_config(hf_config)
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=cfg.hidden_dim,
            conv_kernel=cfg.sconv_kernel,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.short_conv_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        _raise_for_unsupported_modes(vllm_config)
        hf_config = getattr(vllm_config.model_config, "hf_text_config", None)
        if hf_config is None:
            hf_config = vllm_config.model_config.hf_config
        self.config = GrugMoeRuntimeConfig.from_hf_config(hf_config)
        self.tie_word_embeddings = self.config.tie_word_embeddings
        self.quant_config = vllm_config.quant_config
        self.model = GrugMoeModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        _log_grug_moe_runtime_info(vllm_config, self)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_dim,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if self.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        unused_substrings = (
            "rotary_pos_emb.inv_freq",
            "rotary_emb.inv_freq",
            "rotary_emb.cos_cached",
            "rotary_emb.sin_cached",
        )

        for name, loaded_weight in weights:
            if self.tie_word_embeddings and name.startswith("lm_head."):
                continue
            if any(substring in name for substring in unused_substrings):
                continue
            if is_pp_missing_parameter(name, self):
                continue

            if (
                mapped_name := _try_load_grug_expert_weight(
                    name, loaded_weight, params_dict
                )
            ) is not None:
                loaded_params.add(mapped_name)
                continue

            param = params_dict.get(name)
            if param is None:
                raise ValueError(
                    f"Unexpected GrugMoE weight {name!r}; parameter was not found"
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


__all__ = [
    "GrugMoeAttention",
    "GrugMoeDecoderLayer",
    "GrugMoeDenseMLP",
    "GrugMoeForCausalLM",
    "GrugMoeGatedNorm",
    "GrugMoeMLP",
    "GrugMoeModel",
    "GrugMoeRuntimeConfig",
    "GrugMoeRouter",
    "get_grug_moe_runtime_info",
]
