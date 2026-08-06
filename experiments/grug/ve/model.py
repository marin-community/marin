# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.analysis.backward_flow import (
    is_backward_flow_active,
    log_backward_activation,
    trace_backward_activation,
    trace_grads,
)
from levanter.grug.attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pbatch, Pembed_vocab, Plm_head, Plogits, unshard


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the Grug Llama-style transformer with value embeddings.

    ``value_emb_lambda_init`` is the one knob this variant adds. When it is None the model
    is the base grug transformer exactly; when it is a float, a second token-indexed
    embedding table is blended into every layer's attention value path, gated by a learnable
    per-layer scalar initialized to this value. The A/B arm of the ablation is therefore the
    same code path with the knob set to None -- not a different file.
    """

    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    value_emb_lambda_init: float | None = None  # None disables value embeddings.

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads

    @property
    def value_emb_dim(self) -> int:
        """Width of a value-embedding row: it is blended into ``v``, so it matches the KV width.

        Under grouped-query attention ``v`` carries ``num_kv_heads`` heads, not ``num_heads``,
        so sizing the table to the KV width (rather than ``hidden_dim``) both makes the blend
        shape-correct and keeps the table as small as GQA already makes the value path.
        """
        return self.num_kv_heads * self.inferred_head_dim


class CausalSelfAttention(eqx.Module):
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    # Blend coefficient between the context-computed value and the per-token stored value
    # embedding. One learnable scalar per layer: each depth finds its own mix, and a layer
    # that wants no stored memory can drive it to zero. None when value embeddings are off.
    lambda_ve: jax.Array | None
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d_model, n_heads, n_kv_heads, head_dim = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        lambda_init = cfg.value_emb_lambda_init
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d_model, n_heads * head_dim), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d_model, n_kv_heads * head_dim), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d_model, n_kv_heads * head_dim), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n_heads * head_dim, d_model), cfg.initializer_std), P("model", "data")),
            lambda_ve=None if lambda_init is None else jnp.array(lambda_init, dtype=jnp.float32),
            cfg=cfg,
        )

    @trace_grads
    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        ve: Float[Array, "B S Dkv"] | None = None,
    ) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)

        # Blend the stored per-token memory into the value path only. q and k are untouched,
        # so the attention pattern -- who attends to whom -- is exactly the base model's; only
        # the payload a token delivers when attended to carries the token-identity side channel.
        if ve is not None:
            if self.lambda_ve is None:
                raise ValueError("value embeddings were passed to a layer with no lambda_ve gate")
            lam = unshard(self.lambda_ve).astype(v.dtype)
            ve_heads = rearrange(ve, "... (m d) -> ... m d", d=head_dim)
            v = (1.0 - lam) * v + lam * ve_heads.astype(v.dtype)

        attn_out = attention(q, k, v, mask)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=Pbatch)


class MLP(eqx.Module):
    mlp_up: jax.Array
    mlp_down: jax.Array

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MLP":
        k_up, k_down = random.split(key, 2)
        d_model, d_ff = cfg.hidden_dim, cfg.intermediate_dim
        return MLP(
            mlp_up=reshard(_init_weight(k_up, (d_model, d_ff), cfg.initializer_std), P("data", "model")),
            mlp_down=reshard(_init_weight(k_down, (d_ff, d_model), cfg.initializer_std), P("model", "data")),
        )

    @trace_grads
    @named_call
    def __call__(self, x: Float[Array, "B S D"]) -> Float[Array, "B S D"]:
        up = jnp.einsum("bsh,hm->bsm", x, self.mlp_up)
        activated = jax.nn.relu(up)
        return jnp.einsum("bsm,mh->bsh", activated, self.mlp_down, out_sharding=Pbatch)


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp: MLP

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key = random.split(key, 2)
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp=MLP.init(cfg, key=mlp_key),
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        ve: Float[Array, "B S Dkv"] | None = None,
    ) -> Float[Array, "B S D"]:
        x = trace_backward_activation(x, "resid_in")
        x = x + self.attn(self.rms_attn(x), mask, ve)
        x = trace_backward_activation(x, "resid_post_attn")
        x = x + self.mlp(self.rms_mlp(x))
        return trace_backward_activation(x, "resid_out")


class Transformer(eqx.Module):
    token_embed: jax.Array
    # The value-embedding table: a second vector per vocabulary entry, read at every layer.
    # One table serves all depths, so its parameter cost is paid once, and it is indexed by
    # the raw token ids rather than the hidden state -- the signal never travels the residual
    # stream and so cannot be degraded by the layers in between. A lookup, not a matmul: it
    # buys parameters at zero added FLOPs. None when value embeddings are off.
    value_embed: jax.Array | None
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, ve_key, out_key, *block_keys = random.split(key, cfg.num_layers + 3)
        token_embed = reshard(
            _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
        )
        value_embed = None
        if cfg.value_emb_lambda_init is not None:
            value_embed = reshard(
                _init_weight(ve_key, (cfg.vocab_size, cfg.value_emb_dim), cfg.initializer_std), Pembed_vocab
            )
        output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head)
        blocks = tuple(Block.init(cfg, key=layer_key) for layer_key in block_keys)
        final_norm = RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps)
        return Transformer(
            token_embed=token_embed,
            value_embed=value_embed,
            output_proj=output_proj,
            blocks=blocks,
            final_norm=final_norm,
            config=cfg,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S D"]:
        if mask is None:
            mask = AttentionMask.causal()

        with jax.named_scope("token_embed"):
            hidden = self.token_embed.at[token_ids].get(out_sharding=Pbatch)
            hidden = log_backward_activation(hidden)

        # One lookup for the whole network: every block is handed the same `ve` tensor.
        ve = None
        if self.value_embed is not None:
            with jax.named_scope("value_embed"):
                ve = self.value_embed.at[token_ids].get(out_sharding=Pbatch)

        for i, block in enumerate(self.blocks):
            with jax.named_scope(f"block_{i}"):
                block_fn = block if is_backward_flow_active() else eqx.filter_checkpoint(block)
                hidden = block_fn(hidden, mask, ve)
        with jax.named_scope("final_norm"):
            hidden = self.final_norm(hidden)
            return log_backward_activation(hidden)

    def value_emb_lambdas(self) -> tuple[jax.Array, ...]:
        """The learned per-layer blend coefficients, shallow to deep.

        The headline diagnostic: plotted against depth, this is the model's own answer to
        where a token-identity side channel is worth having. Empty when the model has none.
        """
        return tuple(block.attn.lambda_ve for block in self.blocks if block.attn.lambda_ve is not None)

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        hidden = self(token_ids, mask=mask)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=Plogits)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        loss_implementation: str | tuple[str, ...] | None = None,
    ) -> jax.Array:
        """Compute next-token cross-entropy loss for a batch."""
        hidden = self(token_ids, mask=mask)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        return fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
            implementation=loss_implementation,
        )


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def debug_mesh_and_token_pspec(num_devices: int, model_axis_size: int = 1) -> tuple[jax.sharding.AbstractMesh, P]:
    """Return a small abstract mesh and token sharding for lowering contract tests."""
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    if model_axis_size <= 0:
        raise ValueError(f"model_axis_size must be positive, got {model_axis_size}")
    if num_devices % model_axis_size != 0:
        raise ValueError(f"num_devices ({num_devices}) must be divisible by model_axis_size ({model_axis_size})")
    data_axis_size = num_devices // model_axis_size
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, data_axis_size, model_axis_size),
        axis_names=("replica_dcn", "data", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(("replica_dcn", "data"), None)


__all__ = [
    "MLP",
    "Block",
    "CausalSelfAttention",
    "GrugModelConfig",
    "RMSNorm",
    "Transformer",
    "debug_mesh_and_token_pspec",
]
