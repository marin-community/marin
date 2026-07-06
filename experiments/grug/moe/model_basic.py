# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimalist dense transformer for smoke/throughput experiments.

Deliberately stripped to the bare skeleton: token embedding, a stack of
``(basic multi-head attention + dense MLP)`` residual blocks with *no*
normalization layers, and an untied LM head. None of the grug MoE machinery
(routing, QB betas, GatedNorm, XSA, PKO, half-RoPE, sliding windows) is present.

It reuses :class:`~experiments.grug.moe.model.GrugModelConfig` for shape knobs so
the FLOP/launch plumbing is shared; the MoE-only config fields (``num_experts`` &c)
are ignored here. Attention is full multi-head (``num_kv_heads == num_heads``).
"""

import dataclasses
import os

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax import Axis
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import AttentionMask, apply_rotary_embedding, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head

from experiments.grug.moe.model import (
    _BATCH_AXES,
    GrugModelConfig,
    _batch_spec,
    _init_weight,
)


class BasicAttention(eqx.Module):
    """Full multi-head causal self-attention with RoPE. No GQA, gating, or XSA."""

    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D NH"]
    w_v: Float[Array, "D NH"]
    w_o: Float[Array, "NH D"]
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "BasicAttention":
        if cfg.num_heads != cfg.num_kv_heads:
            raise ValueError("model_basic uses full multi-head attention; set num_kv_heads == num_heads")
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, h = cfg.hidden_dim, cfg.num_heads, cfg.inferred_head_dim
        return BasicAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), P("model", "data")),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (n d) -> ... n d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (n d) -> ... n d", d=head_dim)

        if not no_rope_enabled():
            q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        # Tag the flash-attention output so a remat policy can save it (expensive to
        # recompute — a real kernel — but cheap to store: [B,S,n,h]).
        attn_out = jax.ad_checkpoint.checkpoint_name(attn_out, "attn_out")
        attn_out = reshard(attn_out, P(_BATCH_AXES, None, "model", None))
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=_batch_spec())


class BasicMLP(eqx.Module):
    """Plain two-matrix MLP with GELU. No gated/GLU path."""

    w_up: Float[Array, "D I"]
    w_down: Float[Array, "I D"]

    @staticmethod
    def init(hidden_dim: int, intermediate_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "BasicMLP":
        k_up, k_down = random.split(key)
        return BasicMLP(
            w_up=reshard(_init_weight(k_up, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_down=reshard(_init_weight(k_down, (intermediate_dim, hidden_dim), initializer_std), P("model", "data")),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        hidden = jax.nn.gelu(jnp.einsum("td,di->ti", x_flat, self.w_up))
        out_flat = jnp.einsum("ti,id->td", hidden, self.w_down, out_sharding=_batch_spec())
        # out_flat already carries _batch_spec(); the rearrange is a free bitcast, so no
        # trailing reshard is needed (dropping the redundant round-trip).
        return rearrange(out_flat, "(b s) d -> b s d", b=b, s=s)


class BasicBlock(eqx.Module):
    """Residual attention + MLP. No pre/post normalization."""

    attn: BasicAttention
    mlp: BasicMLP

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "BasicBlock":
        attn_key, mlp_key = random.split(key)
        return BasicBlock(
            attn=BasicAttention.init(cfg, key=attn_key),
            mlp=BasicMLP.init(cfg.hidden_dim, cfg.intermediate_dim, cfg.initializer_std, key=mlp_key),
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        x = x + self.attn(x, mask)
        x = x + self.mlp(x)
        return x


class BasicMLPBlock(eqx.Module):
    """Residual MLP only, no attention. For isolating MLP + CE throughput."""

    mlp: BasicMLP

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "BasicMLPBlock":
        return BasicMLPBlock(mlp=BasicMLP.init(cfg.hidden_dim, cfg.intermediate_dim, cfg.initializer_std, key=key))

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        return x + self.mlp(x)


def mlp_only_enabled() -> bool:
    """Env toggle (SCALE_MLP_ONLY=1) selecting attention-free MLP blocks."""
    return os.environ.get("SCALE_MLP_ONLY") == "1"


def no_rope_enabled() -> bool:
    """Env toggle (SCALE_NO_ROPE=1): skip rotary position embedding (NoPE attention)."""
    return os.environ.get("SCALE_NO_ROPE") == "1"


def remat_every_k() -> int:
    """Selective remat stride (SCALE_REMAT_EVERY_K): checkpoint block ``i`` iff ``i % K == 0``.

    K=1 checkpoints every block (equivalent to full remat); larger K checkpoints
    fewer blocks -> more activation memory but less recompute. 0 (default) disables
    this and falls back to the binary ``remat_mode``.
    """
    return int(os.environ.get("SCALE_REMAT_EVERY_K", "0"))


def scan_layers_enabled() -> bool:
    """Env toggle (SCALE_SCAN_LAYERS=1): run the (homogeneous) blocks under a single
    ``jax.lax.scan`` instead of a Python for-loop. One compiled layer body + a clean
    sequential memory schedule, vs. an unrolled 26-layer graph that XLA over-allocates.
    """
    return os.environ.get("SCALE_SCAN_LAYERS") == "1"


def scan_chunk_size() -> int:
    """SCALE_SCAN_CHUNK=G: under scan, checkpoint every G layers (nested scan).

    G=1 (default) checkpoints every layer = full remat (min memory, max recompute);
    larger G checkpoints fewer boundaries -> more activation memory, less recompute
    (recovers MFU). Must divide num_layers.
    """
    return int(os.environ.get("SCALE_SCAN_CHUNK", "1"))


def remat_policy():
    """SCALE_REMAT_POLICY selects the checkpoint policy for the rematerialized body.

    '' / 'none' -> default (nothing saved: recompute everything, incl. matmuls).
    'dots' -> save all matmul outputs, recompute only cheap elementwise (GELU/residual/RoPE):
              eliminates the GEMM recompute at the cost of storing matmul outputs.
    'dots_nobatch' -> save only batch-dim-free dots (cheaper memory, less recompute saved).
    'everything' -> save all intermediates (equivalent to no remat).
    """
    name = os.environ.get("SCALE_REMAT_POLICY", "").lower()
    if name in ("", "none"):
        return None
    policies = {
        "dots": jax.checkpoint_policies.dots_saveable,
        "dots_nobatch": jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
        "everything": jax.checkpoint_policies.everything_saveable,
        # Save only the flash-attention output: recover its recompute (a real kernel)
        # for [T,d] storage, while still recomputing the cheap-to-redo matmuls.
        "attn": jax.checkpoint_policies.save_only_these_names("attn_out"),
    }
    if name not in policies:
        raise ValueError(f"SCALE_REMAT_POLICY={name!r} must be one of {sorted(policies)} or none")
    return policies[name]


class Transformer(eqx.Module):
    """Minimalist dense transformer: embed -> blocks -> lm head, no norms.

    ``blocks`` is a tuple of per-layer modules in the default (unrolled) path; under
    ``SCALE_SCAN_LAYERS`` it is a single block whose leaves carry a leading layer axis.
    """

    token_embed: Float[Array, "V D"]
    output_proj: Float[Array, "D V"]
    blocks: tuple[BasicBlock | BasicMLPBlock, ...] | BasicBlock | BasicMLPBlock
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, *block_keys = random.split(key, cfg.num_layers + 2)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        block_cls: type[BasicBlock] | type[BasicMLPBlock] = BasicMLPBlock if mlp_only_enabled() else BasicBlock
        block_list = [block_cls.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers)]
        blocks: tuple[BasicBlock | BasicMLPBlock, ...] | BasicBlock | BasicMLPBlock
        if scan_layers_enabled() and block_list:
            # Stack homogeneous per-layer weights along a new leading layer axis so the
            # forward can scan over them. Static (config) fields are shared, not stacked.
            blocks = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *block_list)
        else:
            blocks = tuple(block_list)
        return Transformer(token_embed=token_embed, output_proj=output_proj, blocks=blocks, config=cfg)

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
        hidden = self.token_embed.at[token_ids].get(out_sharding=_batch_spec())
        base_remat = self.config.remat_mode != "none"
        policy = remat_policy()

        if scan_layers_enabled() and self.config.num_layers > 0:
            # Scan the stacked blocks: one compiled body, sequential schedule.
            block_params, block_static = eqx.partition(self.blocks, eqx.is_array)

            def run_layer(carry, layer_params):
                block = eqx.combine(layer_params, block_static)
                return block(carry, mask), None

            group = scan_chunk_size()
            if group <= 1:
                # Checkpoint every layer (full remat) — the safe minimum-memory path.
                step = eqx.filter_checkpoint(run_layer, policy=policy) if base_remat else run_layer
                hidden, _ = jax.lax.scan(step, hidden, block_params)
                return hidden

            # Nested scan: checkpoint every `group` layers. Inner scan runs the group
            # (activations kept within it); the outer body is checkpointed so only the
            # group input is stacked -> fewer recomputes than per-layer full remat.
            num_layers = self.config.num_layers
            if num_layers % group != 0:
                raise ValueError(f"SCALE_SCAN_CHUNK={group} must divide num_layers={num_layers}")
            grouped = jax.tree_util.tree_map(lambda x: x.reshape(num_layers // group, group, *x.shape[1:]), block_params)

            def run_group(carry, group_params):
                out, _ = jax.lax.scan(run_layer, carry, group_params)
                return out, None

            step = eqx.filter_checkpoint(run_group, policy=policy) if base_remat else run_group
            hidden, _ = jax.lax.scan(step, hidden, grouped)
            return hidden

        every_k = remat_every_k()
        for i, block in enumerate(self.blocks):
            do_remat = (i % every_k == 0) if every_k > 0 else base_remat
            hidden = (eqx.filter_checkpoint(block, policy=policy) if do_remat else block)(hidden, mask)
        return hidden

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
    """Build a minimalist transformer, matching Vocab size to the tokenizer."""
    resolved = cfg if Vocab.size == cfg.vocab_size else dataclasses.replace(cfg, vocab_size=Vocab.size)
    return Transformer.init(resolved, key=key)


__all__ = [
    "BasicAttention",
    "BasicBlock",
    "BasicMLP",
    "BasicMLPBlock",
    "Transformer",
    "build",
    "mlp_only_enabled",
]
