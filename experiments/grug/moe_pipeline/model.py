# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model.

Architecture: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights.
No load-balancing loss; router z-loss only. All layers are MoE (no dense layers).
"""

import dataclasses
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax import Axis
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
)
from levanter.grug.grug_moe import (
    MOE_REMAT_SAVE_NAMES,
    MoeActivation,
    MoeImplementation,
    resolve_moe_implementation,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head
from levanter.tracker.histogram import Histogram, SummaryStats
from levanter.utils.activation import ActivationFunctionEnum
from transformers import PretrainedConfig as HfConfig

from experiments.grug.moe.model import _BATCH_AXES as BATCH_AXES
from experiments.grug.moe.model import GRUG_MOE_ARCHITECTURE as GRUG_MOE_ARCHITECTURE
from experiments.grug.moe.model import GRUG_MOE_ARTIFACT_SCHEMA_VERSION as GRUG_MOE_ARTIFACT_SCHEMA_VERSION
from experiments.grug.moe.model import (
    GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY as GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY,
)
from experiments.grug.moe.model import GRUG_MOE_MODEL_TYPE as GRUG_MOE_MODEL_TYPE
from experiments.grug.moe.model import CausalSelfAttention as CausalSelfAttention
from experiments.grug.moe.model import DenseMLP as DenseMLP
from experiments.grug.moe.model import GatedNorm as GatedNorm
from experiments.grug.moe.model import GrugMoeHfConfig as GrugMoeHfConfig
from experiments.grug.moe.model import MoEMLP as MoEMLP
from experiments.grug.moe.model import RMSNorm as RMSNorm

RematMode = Literal["recompute_all", "save_moe"]


def _batch_spec() -> P:
    return P(BATCH_AXES)


def _layer_attention_masks(mask: AttentionMask, *, sliding_window: int) -> tuple[AttentionMask, AttentionMask]:
    return mask.with_sliding_window(sliding_window // 2), mask.with_sliding_window(sliding_window)


def _hf_config_attr(config: HfConfig, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    return default


@dataclass(frozen=True)
class GrugModelConfig:
    """Architecture and execution settings for the Grug MoE transformer."""

    vocab_size: int
    hidden_dim: int = 512
    intermediate_dim: int = 256
    shared_expert_intermediate_dim: int = 512
    num_experts: int = 256
    num_experts_per_token: int = 4
    num_layers: int = 6
    num_heads: int = 4
    num_kv_heads: int = 1
    head_dim: int | None = None
    max_seq_len: int = 8192
    sliding_window: int = 2048
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.3
    router_z_loss_coef: float = 0.0
    disable_pko: bool = True
    """When True (default), the every-4th + last 'long' layers skip Partial
    Key Offset (no shift of the second half of K, no doc-start zeroing). Short
    layers never had PKO. Set to False to re-enable PKO on long layers."""
    disable_long_rope: bool = True
    """When True (default), the every-4th + last 'long' layers skip rotary
    embedding entirely (Q and K go into attention un-rotated). Short layers
    still apply half-RoPE. Set to False to keep RoPE on long layers."""
    attention_implementation: GrugAttentionImplementation | None = None
    moe_implementation: MoeImplementation | None = None
    capacity_factor: float = 1.0
    remat_mode: RematMode = "recompute_all"
    """Per-block gradient checkpointing. "recompute_all" reruns the whole block in
    backward (lowest memory); "save_moe" keeps the tagged MoE dispatch tensors so
    backward skips re-running expert dispatch and its EP collectives."""
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        resolve_moe_implementation(self.moe_implementation)

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def model_type(self) -> type["Transformer"]:
        return Transformer

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> "Transformer":
        cfg = self if Vocab.size == self.vocab_size else dataclasses.replace(self, vocab_size=Vocab.size)
        return Transformer.init(cfg, key=key)

    def hf_checkpoint_converter(
        self,
        ref_checkpoint: str | None = None,
    ) -> HFCheckpointConverter["GrugModelConfig"]:  # type: ignore[type-var]
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=ref_checkpoint,
            HfConfigClass=GrugMoeHfConfig,
            tokenizer=ref_checkpoint,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "GrugModelConfig":
        rope = RotaryConfig(theta=float(_hf_config_attr(hf_config, ("rope_theta",), 10000.0)))
        return cls(
            vocab_size=int(_hf_config_attr(hf_config, ("vocab_size",))),
            hidden_dim=int(_hf_config_attr(hf_config, ("hidden_dim", "hidden_size"), 2048)),
            intermediate_dim=int(
                _hf_config_attr(hf_config, ("intermediate_dim", "moe_intermediate_size", "intermediate_size"), 5632)
            ),
            shared_expert_intermediate_dim=int(
                _hf_config_attr(
                    hf_config,
                    ("shared_expert_intermediate_dim", "shared_expert_intermediate_size"),
                    5632,
                )
            ),
            num_experts=int(_hf_config_attr(hf_config, ("num_experts", "num_local_experts"), 8)),
            num_experts_per_token=int(_hf_config_attr(hf_config, ("num_experts_per_token", "num_experts_per_tok"), 2)),
            num_layers=int(_hf_config_attr(hf_config, ("num_layers", "num_hidden_layers"), 24)),
            num_heads=int(_hf_config_attr(hf_config, ("num_heads", "num_attention_heads"), 16)),
            num_kv_heads=int(_hf_config_attr(hf_config, ("num_kv_heads", "num_key_value_heads"), 16)),
            head_dim=_hf_config_attr(hf_config, ("head_dim", "attention_head_dim")),
            max_seq_len=int(_hf_config_attr(hf_config, ("max_seq_len", "max_position_embeddings"), 4096)),
            sliding_window=int(_hf_config_attr(hf_config, ("sliding_window",), 4096)),
            layer_norm_eps=float(_hf_config_attr(hf_config, ("layer_norm_eps", "rms_norm_eps"), 1e-5)),
            initializer_std=float(_hf_config_attr(hf_config, ("initializer_std", "initializer_range"), 0.02)),
            qk_mult=float(_hf_config_attr(hf_config, ("qk_mult",), 1.0)),
            rope=rope,
        )

    def to_hf_config(self, vocab_size: int, config_overrides: dict[str, Any] | None = None) -> GrugMoeHfConfig:
        # One name per field: core fields take the universal transformers spelling, MoE fields the
        # most common public spelling, and grug-specific extras keep their bare names. from_hf_config
        # stays tolerant of the older spellings so existing artifacts keep loading.
        config = {
            "architectures": [GRUG_MOE_ARCHITECTURE],
            "vocab_size": vocab_size,
            # core — universal transformers names
            "hidden_size": self.hidden_dim,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.inferred_head_dim,
            "max_position_embeddings": self.max_seq_len,
            "sliding_window": self.sliding_window,
            "rms_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_std,
            "rope_theta": self.rope.theta,
            "tie_word_embeddings": False,
            # MoE — most common public spelling per field
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_token,
            "moe_intermediate_size": self.intermediate_dim,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_dim,
            # grug-specific (no public equivalent)
            "qk_mult": self.qk_mult,
            "grugmoe_attention_mode": "production",
            GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY: GRUG_MOE_ARTIFACT_SCHEMA_VERSION,
        }
        if config_overrides is not None:
            config.update(config_overrides)
        return GrugMoeHfConfig(**config)


def _summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | SummaryStats]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    load_balancing_loss = router_metrics["load_balancing_loss_per_layer"]
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    capacity_overflow = router_metrics["capacity_overflow_per_layer"]
    num_layers = int(routing_entropy.shape[0])

    # Per-layer total assignments = sum of routing_counts over experts (= tokens * k).
    assignments_per_layer = jnp.sum(routing_counts.astype(jnp.float32), axis=-1)
    capacity_overflow_rate = capacity_overflow.astype(jnp.float32) / jnp.maximum(assignments_per_layer, 1.0)

    out: dict[str, jax.Array | SummaryStats] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        "train/router/load_balancing_loss": jnp.mean(load_balancing_loss),
        "train/router/router_z_loss": jnp.mean(router_z_loss),
        "train/router/routing_counts_per_layer": routing_counts,
        "train/router/capacity_overflow_rate_mean": jnp.mean(capacity_overflow_rate),
        "qb_beta_per_layer": router_metrics.get("qb_beta_per_layer"),
    }
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/load_balancing_loss"] = load_balancing_loss[i]
        out[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
        out[f"train/router/layer_{i}/routing_hist"] = _histogram_from_expert_counts(routing_counts[i])
        out[f"train/router/layer_{i}/capacity_overflow_rate"] = capacity_overflow_rate[i]
    return out


def _histogram_from_expert_counts(expert_counts: jax.Array) -> SummaryStats:
    counts = jnp.asarray(expert_counts, dtype=jnp.float32)
    num_experts = counts.shape[0]
    expert_ids = jnp.arange(num_experts, dtype=jnp.float32)
    num = jnp.sum(counts)
    sum_values = jnp.sum(counts * expert_ids)
    sum_squares = jnp.sum(counts * expert_ids * expert_ids)
    nonzero = counts > 0
    min_value = jnp.where(nonzero, expert_ids, jnp.inf).min()
    max_value = jnp.where(nonzero, expert_ids, -jnp.inf).max()
    min_value = jnp.where(num > 0, min_value, 0.0)
    max_value = jnp.where(num > 0, max_value, 0.0)
    bucket_limits = jnp.arange(num_experts + 1, dtype=jnp.float32)
    histogram = Histogram(bucket_limits=bucket_limits, bucket_counts=counts)
    return SummaryStats.from_reduced_values(
        min=min_value,
        max=max_value,
        num=num,
        nonzero_count=jnp.sum(nonzero),
        sum=sum_values,
        sum_squares=sum_squares,
        histogram=histogram,
    )


class LayerAttentionMode(StrEnum):
    SHORT = "short"
    LONG = "long"


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: MoEMLP
    shared: DenseMLP | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        shared = None
        if cfg.shared_expert_intermediate_dim > 0:
            shared = DenseMLP.init(
                cfg.hidden_dim, cfg.shared_expert_intermediate_dim, cfg.initializer_std, key=shared_key
            )
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=MoEMLP.init(cfg, key=mlp_key),
            shared=shared,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        attention_mode: LayerAttentionMode,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        is_long = attention_mode == LayerAttentionMode.LONG
        with jax.named_scope("attention"):
            attn_in = self.attn_gated_norm(self.rms_attn(x))
            x = x + self.attn(
                attn_in,
                mask,
                use_pko=is_long and not self.attn.cfg.disable_pko,
                disable_rope=is_long and self.attn.cfg.disable_long_rope,
            )
        with jax.named_scope("routed_moe"):
            mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
            mlp_out, router_stats = self.mlp(mlp_in)
        if self.shared is not None:
            with jax.named_scope("shared_expert"):
                mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(
        cfg_or_vocab: GrugModelConfig | Axis,
        config: GrugModelConfig | None = None,
        *,
        key: PRNGKeyArray,
    ) -> "Transformer":
        if isinstance(cfg_or_vocab, Axis):
            if config is None:
                raise ValueError("config must be provided when initializing with a Vocab axis")
            cfg = (
                config
                if cfg_or_vocab.size == config.vocab_size
                else dataclasses.replace(config, vocab_size=cfg_or_vocab.size)
            )
        else:
            if config is not None:
                raise ValueError("config must not be provided when initializing directly from GrugModelConfig")
            cfg = cfg_or_vocab

        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        blocks = tuple(Block.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        return Transformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            blocks=blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            config=cfg,
        )

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        batch_spec = _batch_spec()
        cfg = self.config
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        # Short layers: sliding window. Long layers (every 4th + last): full causal.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        num_blocks = len(self.blocks)
        moe_router_stats: list[dict[str, jax.Array]] = []
        for i, block in enumerate(self.blocks):
            is_last = i == num_blocks - 1
            is_long = i % 4 == 3 or is_last
            layer_mask = long_mask if is_long else short_mask
            hidden, router_stats = eqx.filter_checkpoint(block, policy=remat_policy)(
                hidden,
                layer_mask,
                LayerAttentionMode.LONG if is_long else LayerAttentionMode.SHORT,
            )
            moe_router_stats.append(router_stats)

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
            "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
            "capacity_overflow_per_layer": jnp.stack([s["capacity_overflow"] for s in moe_router_stats], axis=0),
        }
        hidden = self.final_gated_norm(self.final_norm(hidden))
        return hidden, router_metrics

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=batch_spec)

    def to_state_dict(self, prefix: str | None = None) -> dict[str, jax.Array]:
        return grugmoe_inference_state_dict(self, prefix=prefix)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        return_router_metrics: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        hidden, router_metrics = self(token_ids, mask=mask)
        labels = jnp.pad(token_ids[:, 1:], ((0, 0), (0, 1))).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
        # No load-balancing loss; router z-loss only.
        num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
        router_z_loss = jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
        aux_loss = self.config.router_z_loss_coef * router_z_loss
        loss = cross_entropy_loss + aux_loss if reduction != "none" else cross_entropy_loss
        if return_router_metrics:
            summarized_metrics = _summarize_router_metrics(router_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            summarized_metrics["train/router/aux_loss_weighted"] = aux_loss
            return loss, summarized_metrics
        return loss


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def debug_mesh_and_token_pspec(num_devices: int) -> tuple[jax.sharding.AbstractMesh, P]:
    """Return a small abstract mesh and token sharding for lowering contract tests."""
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    expert = 2 if num_devices % 2 == 0 else 1
    data = max(1, num_devices // expert)
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, data, expert, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(BATCH_AXES, None)


def _with_state_dict_prefix(prefix: str | None, name: str) -> str:
    return name if prefix is None else f"{prefix}.{name}"


def _linear_inference_tensor(value: jax.Array) -> jax.Array:
    return jnp.swapaxes(value, -1, -2)


def grugmoe_inference_state_dict(model: Transformer, prefix: str | None = None) -> dict[str, jax.Array]:
    tensors: dict[str, jax.Array] = {
        "model.embed_tokens.weight": model.token_embed,
        "model.embed_norm.weight": model.embed_norm.weight,
        "model.embed_gated_norm.down_proj.weight": _linear_inference_tensor(model.embed_gated_norm.w_down),
        "model.embed_gated_norm.up_proj.weight": _linear_inference_tensor(model.embed_gated_norm.w_up),
        "model.norm.weight": model.final_norm.weight,
        "model.final_gated_norm.down_proj.weight": _linear_inference_tensor(model.final_gated_norm.w_down),
        "model.final_gated_norm.up_proj.weight": _linear_inference_tensor(model.final_gated_norm.w_up),
        "lm_head.weight": _linear_inference_tensor(model.output_proj),
    }

    for layer_index, block in enumerate(model.blocks):
        layer_prefix = f"model.layers.{layer_index}"
        tensors.update(
            {
                f"{layer_prefix}.input_layernorm.weight": block.rms_attn.weight,
                f"{layer_prefix}.attn_gated_norm.down_proj.weight": _linear_inference_tensor(
                    block.attn_gated_norm.w_down
                ),
                f"{layer_prefix}.attn_gated_norm.up_proj.weight": _linear_inference_tensor(block.attn_gated_norm.w_up),
                f"{layer_prefix}.self_attn.q_proj.weight": _linear_inference_tensor(block.attn.w_q),
                f"{layer_prefix}.self_attn.k_proj.weight": _linear_inference_tensor(block.attn.w_k),
                f"{layer_prefix}.self_attn.v_proj.weight": _linear_inference_tensor(block.attn.w_v),
                f"{layer_prefix}.self_attn.o_proj.weight": _linear_inference_tensor(block.attn.w_o),
                f"{layer_prefix}.self_attn.attn_gate.weight": _linear_inference_tensor(block.attn.attn_gate),
                f"{layer_prefix}.post_attention_layernorm.weight": block.rms_mlp.weight,
                f"{layer_prefix}.mlp_gated_norm.down_proj.weight": _linear_inference_tensor(block.mlp_gated_norm.w_down),
                f"{layer_prefix}.mlp_gated_norm.up_proj.weight": _linear_inference_tensor(block.mlp_gated_norm.w_up),
                f"{layer_prefix}.mlp.router.weight": _linear_inference_tensor(block.mlp.router),
                f"{layer_prefix}.mlp.router.bias": block.mlp.router_bias,
                f"{layer_prefix}.mlp.experts.gate_proj.weight": _linear_inference_tensor(block.mlp.expert_mlp.w_gate),
                f"{layer_prefix}.mlp.experts.up_proj.weight": _linear_inference_tensor(block.mlp.expert_mlp.w_up),
                f"{layer_prefix}.mlp.experts.down_proj.weight": _linear_inference_tensor(block.mlp.expert_mlp.w_down),
            }
        )
        if block.shared is not None:
            tensors.update(
                {
                    f"{layer_prefix}.shared_expert.gate_proj.weight": _linear_inference_tensor(block.shared.w_gate),
                    f"{layer_prefix}.shared_expert.up_proj.weight": _linear_inference_tensor(block.shared.w_up),
                    f"{layer_prefix}.shared_expert.down_proj.weight": _linear_inference_tensor(block.shared.w_down),
                }
            )

    return {_with_state_dict_prefix(prefix, name): value for name, value in tensors.items()}


__all__ = [
    "GRUG_MOE_ARCHITECTURE",
    "GRUG_MOE_ARTIFACT_SCHEMA_VERSION",
    "GRUG_MOE_ARTIFACT_SCHEMA_VERSION_KEY",
    "GRUG_MOE_MODEL_TYPE",
    "Block",
    "CausalSelfAttention",
    "DenseMLP",
    "GatedNorm",
    "GrugModelConfig",
    "GrugMoeHfConfig",
    "MoEMLP",
    "MoeActivation",
    "RMSNorm",
    "Transformer",
    "debug_mesh_and_token_pspec",
    "grugmoe_inference_state_dict",
]
