# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from dataclasses import dataclass
from functools import partial
from typing import Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from jax import random
from jax.sharding import PartitionSpec as P, reshard
from jax.tree_util import register_dataclass
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .attention import AttentionMask, RotaryConfig, apply_rotary_embedding, attention
from .loss import linear_softmax_cross_entropy_loss_and_logz
from .sharding import Pbatch, Pvocab, unshard


#### Conventions

# Mesh meanings:
# - "data": data parallel sharding axis. We also shard parameters over this axis.
# - "model": model parallel sharding axis. TP

# Dim names:
# - B = batch
# - D = embedding / hidden dim
# - S = sequence length
# - N = num heads
# - M = num kv heads
# - H = head dim
# - I = intermediate dim


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the Grug Llama-style transformer."""

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

    # Controls how we compute logsumexp over the vocab in `levanter.grug.loss_fn`.
    #
    # - `None` means "single full-vocab block" (often faster for small-ish models/vocabs).
    # - Smaller values reduce peak memory, but can be significantly slower in practice.
    #
    # TODO(grug): Replace with a faster large-vocab CE kernel so we don't have to pick between
    # speed and memory.
    cross_entropy_block_size: int | None = 32768

    # Clipped gated activation (GPT-OSS style)
    use_clipped_gated_activation: bool = False
    clipped_gated_alpha: float = 1.702

    # Attention sinks
    use_attention_sinks: bool = False

    # Mixture of Experts
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_load_balancing_coef: float | None = 0.01
    moe_router_z_loss_coef: float | None = 0.001

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


@register_dataclass
@dataclass(frozen=True)
class GrugAttentionParams:
    w_q: jax.Array
    w_k: jax.Array
    w_v: jax.Array
    w_o: jax.Array
    sinks: jax.Array | None = None  # shape (num_heads,)


@register_dataclass
@dataclass(frozen=True)
class GrugMoEParams:
    router: jax.Array  # (D, num_experts)
    # Expert weights stacked on the leading experts axis.
    gate: jax.Array  # (E, D, I)
    up: jax.Array  # (E, D, I)
    down: jax.Array  # (E, I, D)
    router_bias: jax.Array | None = None  # (num_experts,)
    gate_bias: jax.Array | None = None  # (E, I)
    up_bias: jax.Array | None = None  # (E, I)
    down_bias: jax.Array | None = None  # (E, D)


@register_dataclass
@dataclass(frozen=True)
class GrugBlockParams:
    attn: GrugAttentionParams
    rms_attn: jax.Array
    rms_mlp: jax.Array
    mlp_gate: jax.Array | None = None
    mlp_up: jax.Array | None = None
    mlp_down: jax.Array | None = None
    moe: GrugMoEParams | None = None


@register_dataclass
@dataclass(frozen=True)
class GrugModelParameters:
    token_embed: jax.Array
    output_proj: jax.Array
    blocks: tuple[GrugBlockParams, ...]
    final_norm: jax.Array


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def _experts_axis_name() -> str | None:
    mesh = jax.sharding.get_abstract_mesh()
    if mesh is None or getattr(mesh, "empty", False):
        return None
    return "experts" if "experts" in mesh.axis_names else None


def _maybe_expert_spec(*rest: str | None) -> P:
    experts_axis = _experts_axis_name()
    return P(experts_axis, *rest)


def _ragged_moe_linear(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    lhs_contract_dim: int,
    rhs_contract_dim: int,
) -> jax.Array:
    """Apply a ragged per-expert matmul using JAX ragged_dot_general.

    lhs: (tokens, in_dim) grouped by expert
    rhs: (experts, in_dim, out_dim) or (experts, out_dim, in_dim)
    group_sizes: (experts,) sizes for each expert in lhs
    """
    dim_numbers = jax.lax.RaggedDotDimensionNumbers(
        dot_dimension_numbers=(
            (  # contracting dims
                (lhs_contract_dim,),
                (rhs_contract_dim,),
            ),
            ((), ()),  # batch dims
        ),
        lhs_ragged_dimensions=(0,),
        rhs_group_dimensions=(0,),
    )
    def _ragged(lhs_, rhs_, group_sizes_):
        return jax.lax.ragged_dot_general(lhs_, rhs_, group_sizes_, dim_numbers)

    # ragged_dot_general has no sharding rule; run under auto axes with replicated output.
    ragged_fn = jax.sharding.auto_axes(_ragged, out_sharding=P(None, None))
    return ragged_fn(lhs, rhs, group_sizes)


@partial(jax.jit, static_argnames=("cfg",))
def init_parameters(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> GrugModelParameters:
    head_dim = cfg.inferred_head_dim
    key, embed_key, out_key = random.split(key, 3)
    layer_keys = random.split(key, cfg.num_layers)

    token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pvocab)
    output_proj = reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Pvocab)
    final_norm = reshard(jnp.ones((cfg.hidden_dim,), dtype=jnp.float32), P(None))

    blocks: list[GrugBlockParams] = []
    # extract shape sizes for brevity and consistency
    D, N, M, H, I = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.intermediate_dim
    E = cfg.num_experts  # number of experts for MoE
    for i in range(cfg.num_layers):
        k_q, k_k, k_v, k_o, k_mlp = random.split(layer_keys[i], 5)

        sinks = reshard(jnp.zeros((N,), dtype=jnp.float32), P(None)) if cfg.use_attention_sinks else None
        attn = GrugAttentionParams(
            w_q=reshard(_init_weight(k_q, (D, N * H), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (D, M * H), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (D, M * H), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (N * H, D), cfg.initializer_std), P("model", "data")),
            sinks=sinks,
        )
        # keep rms replicated
        rms_attn = jnp.ones((D,), dtype=jnp.float32)
        rms_mlp = jnp.ones((D,), dtype=jnp.float32)

        if cfg.use_moe:
            # MoE: router + multiple experts
            k_router, k_experts = random.split(k_mlp, 2)
            router = reshard(_init_weight(k_router, (D, E), cfg.initializer_std), P("data", _experts_axis_name()))
            router_bias = reshard(jnp.zeros((E,), dtype=jnp.float32), P(_experts_axis_name()))
            k_gate, k_up, k_down = random.split(k_experts, 3)
            gate = reshard(_init_weight(k_gate, (E, D, I), cfg.initializer_std), _maybe_expert_spec("data", "model"))
            up = reshard(_init_weight(k_up, (E, D, I), cfg.initializer_std), _maybe_expert_spec("data", "model"))
            down = reshard(_init_weight(k_down, (E, I, D), cfg.initializer_std), _maybe_expert_spec("model", "data"))
            gate_bias = reshard(jnp.zeros((E, I), dtype=jnp.float32), _maybe_expert_spec(None))
            up_bias = reshard(jnp.zeros((E, I), dtype=jnp.float32), _maybe_expert_spec(None))
            down_bias = reshard(jnp.zeros((E, D), dtype=jnp.float32), _maybe_expert_spec(None))
            moe = GrugMoEParams(
                router=router,
                router_bias=router_bias,
                gate=gate,
                up=up,
                down=down,
                gate_bias=gate_bias,
                up_bias=up_bias,
                down_bias=down_bias,
            )
            blocks.append(
                GrugBlockParams(
                    attn=attn,
                    rms_attn=rms_attn,
                    rms_mlp=rms_mlp,
                    moe=moe,
                )
            )
        else:
            # Dense MLP
            k_gate, k_up, k_down = random.split(k_mlp, 3)
            mlp_gate = reshard(_init_weight(k_gate, (D, I), cfg.initializer_std), P("data", "model"))
            mlp_up = reshard(_init_weight(k_up, (D, I), cfg.initializer_std), P("data", "model"))
            mlp_down = reshard(_init_weight(k_down, (I, D), cfg.initializer_std), P("model", "data"))
            blocks.append(
                GrugBlockParams(
                    attn=attn,
                    rms_attn=rms_attn,
                    rms_mlp=rms_mlp,
                    mlp_gate=mlp_gate,
                    mlp_up=mlp_up,
                    mlp_down=mlp_down,
                )
            )

    return GrugModelParameters(
        token_embed=token_embed,
        output_proj=output_proj,
        blocks=tuple(blocks),
        final_norm=final_norm,
    )


def rms_norm(x: Float[Array, "... D"], weight: Float[Array, "D"], eps: float) -> Float[Array, "... D"]:
    weight = unshard(weight)
    variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(variance + eps)
    return normed * weight


def clipped_gated_activation(gate: jax.Array, up: jax.Array, alpha: float = 1.702) -> jax.Array:
    """GPT-OSS style clipped gated activation."""
    gate = jnp.clip(gate, a_max=7.0)
    up = jnp.clip(up, a_min=-7.0, a_max=7.0)
    glu = gate * jax.nn.sigmoid(gate * alpha)
    return (up + 1) * glu


def mlp(block: GrugBlockParams, x: Float[Array, "B S D"], cfg: GrugModelConfig | None = None) -> Float[Array, "B S D"]:
    gate = jnp.einsum("bsh,hm->bsm", x, block.mlp_gate)
    up = jnp.einsum("bsh,hm->bsm", x, block.mlp_up)
    if cfg is not None and cfg.use_clipped_gated_activation:
        activated = clipped_gated_activation(gate, up, cfg.clipped_gated_alpha)
    else:
        activated = jax.nn.silu(gate) * up
    return jnp.einsum("bsm,mh->bsh", activated, block.mlp_down, out_sharding=Pbatch)


def moe_forward(
    moe: GrugMoEParams,
    x: Float[Array, "B S D"],
    cfg: GrugModelConfig,
) -> tuple[Float[Array, "B S D"], dict]:
    """Mixture of Experts forward pass with auxiliary losses."""
    B, S, D = x.shape
    num_experts = cfg.num_experts
    k = cfg.num_experts_per_tok

    # Flatten batch and sequence for routing
    x_flat = x.reshape(-1, D)  # (B*S, D)

    # Router logits and probabilities
    # Router is (D, E) with sharding P("data", None), x_flat is (B*S, D)
    # Result should be (B*S, E) - replicated on E dimension
    router_logits = jnp.einsum("td,de->te", x_flat, moe.router)
    if moe.router_bias is not None:
        router_logits = router_logits + moe.router_bias
    router_probs = jax.nn.softmax(router_logits, axis=-1)
    # top_k does not support sharded expert dimension; replicate before routing.
    router_probs = reshard(router_probs, P(None, None))

    # Top-k selection
    topk_weights, topk_indices = jax.lax.top_k(router_probs, k)  # (B*S, k)
    # Normalize weights to sum to 1
    topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)

    # Sparse MoE: route tokens to experts and compute only assigned expert MLPs.
    if moe.gate is None or moe.up is None or moe.down is None:
        raise ValueError("MoE parameters must include stacked expert weights (gate/up/down).")

    tokens = B * S
    topk_indices_flat = topk_indices.reshape(-1)
    topk_weights_flat = topk_weights.reshape(-1)
    token_indices = jnp.arange(tokens * k) // k

    # Sort by expert id to group tokens per expert.
    sort_idx = jnp.argsort(topk_indices_flat)
    inv_sort = jnp.argsort(sort_idx)
    x_repeat = x_flat.at[token_indices].get(out_sharding=P(None, None))
    x_repeat_sort = x_repeat[sort_idx]
    topk_indices_sorted = topk_indices_flat[sort_idx]
    group_sizes = jnp.bincount(topk_indices_sorted, length=num_experts).astype(jnp.int32)
    group_sizes = reshard(group_sizes, P(_experts_axis_name()))

    # Expert-linear projections using ragged matmuls.
    gate = _ragged_moe_linear(x_repeat_sort, moe.gate, group_sizes, lhs_contract_dim=1, rhs_contract_dim=1)
    up = _ragged_moe_linear(x_repeat_sort, moe.up, group_sizes, lhs_contract_dim=1, rhs_contract_dim=1)

    if moe.gate_bias is not None:
        gate_bias_flat = moe.gate_bias.at[topk_indices_flat].get(out_sharding=P(None, None))
        gate = gate + gate_bias_flat[sort_idx]
    if moe.up_bias is not None:
        up_bias_flat = moe.up_bias.at[topk_indices_flat].get(out_sharding=P(None, None))
        up = up + up_bias_flat[sort_idx]

    if cfg.use_clipped_gated_activation:
        activated = clipped_gated_activation(gate, up, cfg.clipped_gated_alpha)
    else:
        activated = jax.nn.silu(gate) * up

    output_sorted = _ragged_moe_linear(activated, moe.down, group_sizes, lhs_contract_dim=1, rhs_contract_dim=1)
    if moe.down_bias is not None:
        down_bias_flat = moe.down_bias.at[topk_indices_flat].get(out_sharding=P(None, None))
        output_sorted = output_sorted + down_bias_flat[sort_idx]

    # Unpermute back to original token order.
    output_repeat = output_sorted.at[inv_sort].get(out_sharding=P(None, None))
    output_repeat = output_repeat.reshape(tokens, k, D)

    # Weighted sum
    output_flat = jnp.einsum("bkd,bk->bd", output_repeat, topk_weights)
    output = output_flat.reshape(B, S, D)
    output = reshard(output, Pbatch)

    # Compute auxiliary losses
    extras = {}

    # Load balancing loss: encourages even distribution across experts
    if cfg.moe_load_balancing_coef is not None and cfg.moe_load_balancing_coef > 0:
        # f_i = fraction of tokens routed to expert i (using top-1)
        # p_i = average router probability for expert i
        top1_indices = topk_indices[:, 0]  # (B*S,)
        f = jnp.zeros(num_experts)
        for e in range(num_experts):
            f = f.at[e].set(jnp.mean(top1_indices == e))
        p = jnp.mean(router_probs, axis=0)  # (E,)
        load_balance_loss = cfg.moe_load_balancing_coef * num_experts * jnp.sum(f * p)
        extras["load_balancing_loss"] = load_balance_loss

    # Router z-loss: penalizes large router logits
    if cfg.moe_router_z_loss_coef is not None and cfg.moe_router_z_loss_coef > 0:
        router_z = jax.scipy.special.logsumexp(router_logits, axis=-1)  # (B*S,)
        router_z_loss = cfg.moe_router_z_loss_coef * jnp.mean(router_z**2)
        extras["router_z_loss"] = router_z_loss

    return output, extras


def _transformer_hidden(
    params: GrugModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugModelConfig,
    *,
    mask: AttentionMask | jax.Array | None,
) -> tuple[Float[Array, "B S D"], dict]:
    head_dim = cfg.inferred_head_dim
    seq_len = token_ids.shape[1]

    if mask is None:
        mask = AttentionMask.causal()

    hidden = params.token_embed.at[token_ids].get(out_sharding=Pbatch)
    extras: dict = {}

    for block in params.blocks:
        attn_in = rms_norm(hidden, block.rms_attn, cfg.layer_norm_eps)
        q = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", attn_in, block.attn.w_v), "... (m d) -> ... m d", d=head_dim)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=cfg.rope)
        attn_out = attention(q, k, v, mask, sinks=block.attn.sinks)
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        attn_out = jnp.einsum("bsh,hd->bsd", attn_out, block.attn.w_o, out_sharding=Pbatch)

        hidden = hidden + attn_out
        mlp_in = rms_norm(hidden, block.rms_mlp, cfg.layer_norm_eps)

        if block.moe is not None:
            mlp_out, moe_extras = moe_forward(block.moe, mlp_in, cfg)
            # Accumulate aux losses
            for key, val in moe_extras.items():
                extras[key] = extras.get(key, 0.0) + val
        else:
            mlp_out = mlp(block, mlp_in, cfg)

        hidden = hidden + mlp_out

    hidden = rms_norm(hidden, params.final_norm, cfg.layer_norm_eps)
    return hidden, extras


def forward(
    params: GrugModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
) -> Float[Array, "B S V"]:
    hidden, _ = _transformer_hidden(params, token_ids, cfg, mask=mask)
    logits = jnp.einsum("bsh,hd->bsd", hidden, params.output_proj, out_sharding=Pbatch)
    return logits


def activations(
    params: GrugModelParameters,
    token_ids: Int[Array, "B S"],
    cfg: GrugModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
) -> Float[Array, "B S D"]:
    """Return final hidden states with shape (batch, seq, hidden_dim)."""
    hidden, _ = _transformer_hidden(params, token_ids, cfg, mask=mask)
    return hidden


def loss_fn(
    params: GrugModelParameters,
    token_ids: Int[Array, "B S"],
    loss_weight: Float[Array, "B S"],
    cfg: GrugModelConfig,
    *,
    mask: AttentionMask | jax.Array | None = None,
    reduction: str = "mean",
    logsumexp_weight: float | None = None,
    loss_dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, dict]:
    """Compute next-token cross-entropy loss for a batch.

    This is the "activations vs lm_head" friendly path: it avoids materializing full logits.

    Args:
        params: Model parameters.
        token_ids: Integer array with shape (batch, seq).
        loss_weight: Float array with shape (batch, seq), typically 1 except last position (0).
        cfg: Model config (uses `cfg.cross_entropy_block_size`).
        mask: Optional attention mask spec.
        reduction: One of {"mean", "sum", "none"}.
        logsumexp_weight: Optional z-loss weight (logsumexp^2 term).
        loss_dtype: Accumulator dtype for logsumexp / loss.

    Returns:
        Tuple of (loss, extras):
        - loss: If reduction=="none": array with shape (batch, seq). Else: scalar array.
        - extras: Dict with auxiliary losses (e.g., load_balancing_loss, router_z_loss for MoE).
    """
    hidden, extras = _transformer_hidden(params, token_ids, cfg, mask=mask)
    labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
    loss_weight = loss_weight.astype(loss_dtype)
    # Ensure loss weights match the batch sharding used by per-position losses.
    loss_weight = reshard(loss_weight, P(("data",), None))

    # NOTE: `block_size=None` corresponds to a single full-vocab block. On the 125M speedrun,
    # disabling blockwise chunking doubled observed MFU (~20 -> ~40). We'll likely need a better
    # large-vocab loss kernel eventually (esp. for sharded vocab / padding weights), but this is
    # good enough for now.
    block_size = cfg.cross_entropy_block_size

    per_pos_loss, logz = linear_softmax_cross_entropy_loss_and_logz(
        hidden,
        params.output_proj,
        labels,
        block_size=block_size,
        dtype=loss_dtype,
    )
    per_pos_loss = per_pos_loss.astype(loss_dtype) * loss_weight
    if logsumexp_weight is not None and logsumexp_weight != 0.0:
        per_pos_loss = per_pos_loss + logsumexp_weight * (logz.astype(loss_dtype) ** 2) * loss_weight

    if reduction == "none":
        loss = per_pos_loss
    elif reduction == "sum":
        loss = jnp.sum(per_pos_loss)
    elif reduction == "mean":
        denom = jnp.sum(loss_weight)
        loss = jnp.sum(per_pos_loss) / jnp.maximum(denom, jnp.array(1.0, dtype=loss_dtype))
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Add auxiliary losses from MoE
    aux_loss = jnp.array(0.0, dtype=loss_dtype)
    for key in ["load_balancing_loss", "router_z_loss"]:
        if key in extras:
            aux_loss = aux_loss + extras[key]
    if reduction != "none":
        loss = loss + aux_loss

    return loss, extras


__all__ = [
    "GrugAttentionParams",
    "GrugBlockParams",
    "GrugMoEParams",
    "GrugModelConfig",
    "GrugModelParameters",
    "clipped_gated_activation",
    "init_parameters",
    "activations",
    "forward",
    "loss_fn",
    "mlp",
    "moe_forward",
    "rms_norm",
]
