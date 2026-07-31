# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy Grug-MoE model layout for evaling pre-PR #5718 checkpoints.

The live MoE model stores expert weights under ``mlp.expert_mlp``. Older
checkpoints store the same tensors directly under ``mlp.w_gate_up`` and
``mlp.w_down``. This module preserves that old parameter tree for checkpoint
loading while reusing the current attention/norm/loss code.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import moe_mlp
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head
from levanter.tracker.histogram import Histogram
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.model import (
    CausalSelfAttention,
    DenseMLP,
    GatedNorm,
    GrugModelConfig,
    RMSNorm,
    _batch_spec,
    _init_weight,
    _mesh_axis_size,
    _routing_stats,
    _summarize_router_metrics,
)

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map


class LegacyMoEMLP(eqx.Module):
    """QB-routed MoE using the legacy flat expert parameter names."""

    router: jax.Array
    router_bias: jax.Array
    w_gate_up: jax.Array
    w_down: jax.Array
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> LegacyMoEMLP:
        k_router, k_gate, k_up, k_down = random.split(key, 4)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e, i = cfg.hidden_dim, cfg.num_experts, cfg.intermediate_dim
        w_gate = _init_weight(k_gate, (e, d, i), cfg.initializer_std)
        w_up = _init_weight(k_up, (e, d, i), cfg.initializer_std)
        w_gate_up = jnp.concatenate([w_gate, w_up], axis=-1)

        return LegacyMoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((e,)),
            w_gate_up=reshard(w_gate_up, P("expert", "data", "model")),
            w_down=reshard(_init_weight(k_down, (e, i, d), cfg.initializer_std), P("expert", "model", "data")),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        topk_logits, selected_experts = jax.lax.top_k(biased_logits, self.cfg.num_experts_per_token + 1)
        qb_alpha = topk_logits[:, -1:]
        selected_experts = selected_experts[:, :-1]
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights = jax.nn.sigmoid(unbiased_topk).astype(x.dtype)
        router_stats = _routing_stats(
            selected_experts,
            router_probs,
            router_logits,
            num_experts=self.cfg.num_experts,
            num_experts_per_token=self.cfg.num_experts_per_token,
        )

        s_minus_alpha = router_logits - qb_alpha
        mesh = get_abstract_mesh()
        batch_axes = ("data", "expert")
        num_devices = 1
        for axis_name in batch_axes:
            if axis_name in mesh.shape:
                num_devices *= mesh.shape[axis_name]
        local_tokens = s_minus_alpha.shape[0] // num_devices
        qb_count = max(1, local_tokens * self.cfg.num_experts_per_token // self.cfg.num_experts)

        def _local_qb_beta(s_ma):
            topk_vals, _ = jax.lax.top_k(s_ma.T, qb_count)
            beta = topk_vals[:, -1]
            return jax.lax.pmean(beta, axis_name=batch_axes)

        router_stats["qb_beta"] = shard_map(
            _local_qb_beta,
            mesh=mesh,
            in_specs=(P(batch_axes, None),),
            out_specs=P(),
        )(s_minus_alpha)

        routed_flat = moe_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            self.w_gate_up,
            self.w_down,
            activation=ActivationFunctionEnum.silu,
            implementation=self.cfg.moe_implementation,
            mesh=get_abstract_mesh(),
            # Old training used a hardcoded capacity factor; eval intentionally honors the override.
            capacity_factor=self.cfg.capacity_factor,
        )

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        return reshard(routed, _batch_spec()), router_stats


class LegacyBlock(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: LegacyMoEMLP
    shared: DenseMLP | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> LegacyBlock:
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        shared = None
        if cfg.shared_expert_intermediate_dim > 0:
            shared = DenseMLP.init(
                cfg.hidden_dim, cfg.shared_expert_intermediate_dim, cfg.initializer_std, key=shared_key
            )
        return LegacyBlock(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=LegacyMoEMLP.init(cfg, key=mlp_key),
            shared=shared,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        x = x + self.attn(attn_in, mask)
        mlp_in = self.mlp_gated_norm(self.rms_mlp(x))
        mlp_out, router_stats = self.mlp(mlp_in)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        return x + mlp_out, router_stats


class LegacyTransformer(eqx.Module):
    """Transformer with the legacy flat MoE expert parameter tree."""

    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    blocks: tuple[LegacyBlock, ...]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> LegacyTransformer:
        embed_key, out_key, embed_gn_key, final_gn_key, *block_keys = random.split(key, cfg.num_layers + 4)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        blocks = tuple(LegacyBlock.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        return LegacyTransformer(
            token_embed=token_embed,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=output_proj,
            blocks=blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            config=cfg,
        )

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

        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window // 2, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)

        moe_router_stats: list[dict[str, jax.Array]] = []
        for i, block in enumerate(self.blocks):
            layer_mask = long_mask if i % 4 == 3 else short_mask
            hidden, router_stats = eqx.filter_checkpoint(block)(hidden, layer_mask)
            moe_router_stats.append(router_stats)

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
            "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
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
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | Histogram]]:
        hidden, router_metrics = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
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
        num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
        rzl = jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
        aux_loss = self.config.router_z_loss_coef * rzl
        loss = cross_entropy_loss + aux_loss if reduction != "none" else cross_entropy_loss
        if return_router_metrics:
            summarized_metrics = _summarize_router_metrics(router_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            summarized_metrics["train/router/aux_loss_weighted"] = aux_loss
            return loss, summarized_metrics
        return loss


__all__ = [
    "LegacyBlock",
    "LegacyMoEMLP",
    "LegacyTransformer",
]
