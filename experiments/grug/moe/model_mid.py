# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimalist shared-expert + routed-MoE transformer for throughput experiments.

A mid-point between ``model_basic`` (dense / a few MoE layers) and the full
``model.py`` (RMSNorm + GatedNorm + PKO/XSA/half-RoPE). *Every* layer here is::

    x = x + attn(norm_attn(x))
    x = x + routed_moe(norm_mlp(x)) + shared_expert(norm_mlp(x))

with learnable-gain :class:`RMSNorm` pre-attention and pre-MLP, plus an input norm
(after the embedding) and an output norm (before the LM head). The shared expert is a SwiGLU FFN at width
``hidden_dim`` (grug's :class:`~experiments.grug.moe.model.DenseMLP`); the routed
component is grug's QB-routed :class:`~experiments.grug.moe.model.MoEMLP` picking
``num_experts_per_token`` of ``num_experts`` (e.g. 4 of 64). Shared and routed
both read the post-attention hidden and their sum feeds a single residual add,
matching ``model.py``'s block minus the norms.

Attention is full multi-head (``num_kv_heads == num_heads``) and reuses
``model_basic``'s no-norm :class:`BasicAttention`. The stack is homogeneous, so
the blocks are an unrolled tuple or, under ``SCALE_SCAN_LAYERS``, a single block
whose leaves carry a leading layer axis for ``jax.lax.scan``.
"""

import dataclasses
import os

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax import Axis
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.model import (
    DenseMLP,
    GrugModelConfig,
    MoEMLP,
    RMSNorm,
    _batch_spec,
    _init_weight,
)
from experiments.grug.moe.model_basic import (
    _FSDP_AXES,
    BasicAttention,
    embed_router_std,
    moe_expert_intermediate,
    moe_remat_policy,
    moe_scan_save,
    remat_every_k,
    remat_policy,
    scan_chunk_size,
    scan_layers_enabled,
)


def shared_expert_intermediate(cfg: GrugModelConfig) -> int:
    """Width of the per-layer shared SwiGLU expert (SCALE_SHARED_EXPERT_INTERMEDIATE).

    Defaults to ``hidden_dim`` (a full-width dense SwiGLU alongside the routed experts).
    """
    return int(os.environ.get("SCALE_SHARED_EXPERT_INTERMEDIATE", str(cfg.hidden_dim)))


class MoeMidBlock(eqx.Module):
    """Residual (pre-norm attention) + (pre-norm routed MoE + shared SwiGLU expert).

    Reuses ``model_basic``'s :class:`BasicAttention`, grug's :class:`MoEMLP` (routed,
    top-(K+1) QB gating, ragged-dot dispatch), grug's :class:`DenseMLP` (SwiGLU shared
    expert), and grug's learnable-gain :class:`RMSNorm` applied pre-attention and
    pre-MLP. Forward-only QB: the router bias stays zero and the returned router stats
    are discarded.
    """

    attn: BasicAttention
    norm_attn: RMSNorm
    norm_mlp: RMSNorm
    shared: DenseMLP
    moe: MoEMLP

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoeMidBlock":
        attn_key, shared_key, moe_key, router_key = random.split(key, 4)
        # MoEMLP reads cfg.intermediate_dim as the *expert* width; swap in the (narrower)
        # routed-expert width so the config's own intermediate_dim is untouched.
        moe_cfg = dataclasses.replace(cfg, intermediate_dim=moe_expert_intermediate(cfg))
        moe = MoEMLP.init(moe_cfg, key=moe_key)
        # Nonzero router init even under zero-init so routing spreads across experts
        # (experts stay zero for stability, but the dispatch load is realistic).
        router = reshard(_init_weight(router_key, moe.router.shape, embed_router_std(cfg)), P(None, None))
        moe = eqx.tree_at(lambda m: m.router, moe, router)
        shared = DenseMLP.init(cfg.hidden_dim, shared_expert_intermediate(cfg), cfg.initializer_std, key=shared_key)
        return MoeMidBlock(
            attn=BasicAttention.init(cfg, key=attn_key),
            norm_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            norm_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            shared=shared,
            moe=moe,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        x = x + self.attn(self.norm_attn(x), mask)
        mlp_in = self.norm_mlp(x)
        moe_out, _router_stats = self.moe(mlp_in)  # forward-only QB routing; stats unused
        shared_out = self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        return x + moe_out + shared_out


class Transformer(eqx.Module):
    """Minimalist all-MoE transformer: embed -> shared+routed MoE blocks -> lm head.

    The stack is homogeneous (every layer is a :class:`MoeMidBlock`), so ``blocks`` is
    an unrolled tuple in the default path, or under ``SCALE_SCAN_LAYERS`` a single block
    whose leaves carry a leading layer axis.
    """

    token_embed: Float[Array, "V D"]
    output_proj: Float[Array, "D V"]
    norm_input: RMSNorm
    norm_output: RMSNorm
    blocks: tuple[MoeMidBlock, ...] | MoeMidBlock
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, *block_keys = random.split(key, cfg.num_layers + 2)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), embed_router_std(cfg)), P("model", _FSDP_AXES)
        )
        output_proj = reshard(
            _init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), P(_FSDP_AXES, "model")
        )
        block_list = [MoeMidBlock.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers)]
        blocks: tuple[MoeMidBlock, ...] | MoeMidBlock
        if scan_layers_enabled():
            # Stack homogeneous per-layer weights along a new leading layer axis so the
            # forward can scan over them. Static (config) fields are shared, not stacked.
            blocks = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *block_list)
        else:
            blocks = tuple(block_list)
        return Transformer(
            token_embed=token_embed,
            output_proj=output_proj,
            norm_input=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            norm_output=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            blocks=blocks,
            config=cfg,
        )

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.config.vocab_size)

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> "Transformer":
        cfg = (
            self.config
            if Vocab.size == self.config.vocab_size
            else dataclasses.replace(self.config, vocab_size=Vocab.size)
        )
        return Transformer.init(cfg, key=key)

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()
        hidden = self.norm_input(self.token_embed.at[token_ids].get(out_sharding=_batch_spec()))
        base_remat = self.config.remat_mode != "none"
        policy = remat_policy()
        # Every block is MoE: save the tagged dispatch tensors (save_moe) only when asked,
        # else full remat. Under scan save_moe stacks the saved tensors across all layers
        # (memory ~ batch), so recompute is the safe default.
        block_policy = moe_remat_policy() if moe_scan_save() else policy

        if scan_layers_enabled():
            block_params, block_static = eqx.partition(self.blocks, eqx.is_array)

            def run_layer(carry, layer_params):
                block = eqx.combine(layer_params, block_static)
                return block(carry, mask), None

            group = scan_chunk_size()
            if group <= 1:
                # Checkpoint every layer (full remat) — the safe minimum-memory path.
                step = eqx.filter_checkpoint(run_layer, policy=block_policy) if base_remat else run_layer
                hidden, _ = jax.lax.scan(step, hidden, block_params)
            else:
                # Nested scan: checkpoint every `group` layers. The inner scan runs the group
                # (activations kept within it); the outer body is checkpointed so only the
                # group input is stacked -> fewer recomputes than per-layer full remat.
                if self.config.num_layers % group != 0:
                    raise ValueError(f"SCALE_SCAN_CHUNK={group} must divide num_layers={self.config.num_layers}")
                grouped = jax.tree_util.tree_map(
                    lambda x: x.reshape(self.config.num_layers // group, group, *x.shape[1:]), block_params
                )

                def run_group(carry, group_params):
                    out, _ = jax.lax.scan(run_layer, carry, group_params)
                    return out, None

                step = eqx.filter_checkpoint(run_group, policy=block_policy) if base_remat else run_group
                hidden, _ = jax.lax.scan(step, hidden, grouped)
        else:
            every_k = remat_every_k()
            for i, block in enumerate(self.blocks):
                do_remat = (i % every_k == 0) if every_k > 0 else base_remat
                hidden = (eqx.filter_checkpoint(block, policy=block_policy) if do_remat else block)(hidden, mask)

        return self.norm_output(hidden)

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=_batch_spec())

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array:
        hidden = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight.astype(loss_dtype),
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )


def build(cfg: GrugModelConfig, Vocab: Axis, *, key: PRNGKeyArray) -> Transformer:
    """Build a minimalist shared-expert + routed-MoE transformer, matching Vocab size."""
    resolved = cfg if Vocab.size == cfg.vocab_size else dataclasses.replace(cfg, vocab_size=Vocab.size)
    return Transformer.init(resolved, key=key)


__all__ = [
    "MoeMidBlock",
    "Transformer",
    "build",
    "shared_expert_intermediate",
]
