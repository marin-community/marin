# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The June 67B-A2B model as selected for AMD MI355X, written as straight code.

Same architecture, parameters and checkpoint layout as ``experiments/june_tpu_67b_a2b/moe/model.py``, whose
``GrugModelConfig``, norm and MLP parameter layouts this module imports. What differs is how it computes:

- attention normalizes, half-RoPE-rotates and scales q and k in one Pallas-Triton kernel per direction
  (``kernels/qk_norm_rope.py``);
- every RMSNorm + GatedNorm pair (two per block, the embedding norm and the final norm) is one custom VJP over the
  residual add, the norm and the gate, with Pallas-Triton kernels (``kernels/add_rms_norm.py``,
  ``kernels/gated_norm.py``);
- the MoE always uses Levanter's ``ring_filtered``, which picks its own kernels for the device;
- the routing statistics, QB thresholds and capacity drops stay per batch shard in each layer and are reduced once
  after the layer scan; of June's router diagnostics only the router z-loss is computed (and always logged);
- one remat policy saves the MoE dispatch and routing tensors, the attention residuals, GatedNorm's down projection
  and the head-gate logits;
- the blocks always run in one array-stacked ``lax.scan`` whose checkpoint has ``prevent_cse=False``;
- the small replicated block weights' gradients are reduced once after the layer scan, and the scan routes the
  trainer's microbatch reductions and ZeRO-1 gathers (``parallel.LayerRouting``).

``Transformer.init`` raises for config values this copy does not implement.
"""

import functools
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random, shard_map
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from levanter.grug._moe.common import CapacityDrops, padding_skipped_assignments
from levanter.grug._moe.ep_ring_filtered import _moe_mlp_ep_ring_filtered_local
from levanter.grug._moe.routing_top_k import routing_top_k
from levanter.grug.attention import AttentionMask, align_kv_heads, attention, token_validity_from_attention_mask
from levanter.grug.grug_moe import (
    MOE_DROPPED_ASSIGNMENTS_METRIC,
    MOE_REMAT_SAVE_NAMES,
    MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC,
    MoEExpertMlp,
    qb_beta_topk_shard,
    qb_topk_physical_count,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import (
    Pembed_vocab,
    Plm_head,
    _axis_names,
    _mesh_axis_size,
    _reshard_for_shard_map,
    _token_spec_from_x,
    keep_reduced,
    replicate_keeping_reduced,
    unshard,
)
from levanter.layers.stacked import ArrayStacked
from levanter.utils.activation import ActivationFunctionEnum

from experiments.amd.june_67b_a2b.kernels.add_rms_norm import (
    add_rms_norm_bwd,
    add_rms_norm_fwd,
    pallas_add_rms_norm_supported,
)
from experiments.amd.june_67b_a2b.kernels.gated_norm import (
    gated_norm_gate_bwd,
    gated_norm_gate_fwd,
    gated_norm_silu,
    pallas_gated_norm_supported,
)
from experiments.amd.june_67b_a2b.kernels.qk_norm_rope import pallas_qk_norm_rope_supported, qk_norm_rope, rope_tables
from experiments.amd.june_67b_a2b.parallel import (
    LayerRouting,
    apply_gathers,
    call_routed,
    local_to_unreduced,
    mark_replicated_reduced,
)
from experiments.june_tpu_67b_a2b.moe.model import (
    _BATCH_AXES,
    _GATED_NORM_RANK,
    DenseMLP,
    GatedNorm,
    GrugModelConfig,
    RMSNorm,
    _batch_spec,
    _init_weight,
    _long_layer_schedule,
)

# The only MoE backend this copy runs; June's None (the backend default) means it too.
MOE_IMPLEMENTATION = "ring_filtered"
# jax-aiter names its attention residuals (output, LSE, q/k/v) "context", as Transformer Engine does; saving the name
# keeps the backward from re-running the attention forward.
ATTENTION_REMAT_SAVE_NAME = "context"
# GatedNorm's rank-128 down projection ``n w_down``: saving it skips the down GEMM when the backward reruns the block.
GATED_NORM_REMAT_SAVE_NAME = "gated_norm_hidden"
# The attention's head-gate logits ``x attn_gate`` ([B, S, num_heads]).
HEAD_GATE_REMAT_SAVE_NAME = "attn_head_gate"
REMAT_SAVE_NAMES = (
    *MOE_REMAT_SAVE_NAMES,
    ATTENTION_REMAT_SAVE_NAME,
    GATED_NORM_REMAT_SAVE_NAME,
    HEAD_GATE_REMAT_SAVE_NAME,
)
# The q/k head norm's epsilon (June's ``rms_norm`` default).
_QK_NORM_EPS = 1e-6
# QBRoutedMoE's normalization: the selected sigmoid combine weights of a token sum to this.
_ROUTING_RENORM_SUM = 2.5
# float32 holds every integer up to 2**24 exactly.
_FLOAT32_EXACT_INTEGERS = 2**24


def check_supported(cfg: GrugModelConfig) -> None:
    """Raise ``ValueError`` if ``cfg`` asks for something this copy does not implement."""
    if cfg.moe_implementation not in (None, MOE_IMPLEMENTATION):
        raise ValueError(f"the AMD June model runs only {MOE_IMPLEMENTATION!r}, got {cfg.moe_implementation!r}")
    if cfg.pooled_transport_capacity_factor is None:
        raise ValueError(f"{MOE_IMPLEMENTATION} requires pooled_transport_capacity_factor")
    if not cfg.use_array_stacked_blocks:
        raise ValueError("the AMD June model runs only array-stacked blocks (use_array_stacked_blocks=True)")
    if not cfg.disable_pko:
        raise ValueError("the AMD June model's stacked blocks need disable_pko=True, as June's do")
    if cfg.ce_implementation is not None:
        raise ValueError(f"the AMD June model uses the backend's cross-entropy, got {cfg.ce_implementation!r}")
    if cfg.remat_mode != "recompute_all":
        raise ValueError(f"the AMD June model has a fixed remat policy; leave remat_mode default, got {cfg.remat_mode}")
    if cfg.scan_unroll != 1:
        raise ValueError(f"the AMD June model scans one layer per iteration, got scan_unroll={cfg.scan_unroll}")
    if not pallas_qk_norm_rope_supported(cfg.inferred_head_dim):
        raise ValueError(f"the q/k norm kernels cannot tile head_dim={cfg.inferred_head_dim}")
    hidden = cfg.hidden_dim
    if not (pallas_add_rms_norm_supported(hidden) and pallas_gated_norm_supported(hidden, _GATED_NORM_RANK)):
        raise ValueError(f"the norm kernels cannot tile hidden_dim={hidden} (needs a multiple of 128)")


def _qk_head_norm(
    x: Float[Array, "B S H D"], positions: jax.Array, *, cfg: GrugModelConfig, rope: bool, scales: tuple[float, ...]
) -> Float[Array, "B S H D"]:
    """``rope(rms_norm(x)) * scales`` per head, in the Pallas kernels, per device under ``shard_map``."""

    def local(x, positions):
        cos, sin = rope_tables(positions, x.shape[-1] // 4, cfg.rope.theta)
        return qk_norm_rope(x, cos, sin, eps=_QK_NORM_EPS, rope=rope, scales=scales)

    mesh = get_abstract_mesh()
    if mesh.empty:
        return local(x, positions)
    x_spec, positions_spec = (P(*(*jax.typeof(a).sharding.spec, *(None,) * a.ndim)[: a.ndim]) for a in (x, positions))
    return shard_map(local, mesh=mesh, in_specs=(x_spec, positions_spec), out_specs=x_spec, check_vma=False)(
        x, positions
    )


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"]
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        in_spec = P(None, None) if cfg.replicate_attn_weights else P("data", "model")
        out_spec = P(None, None) if cfg.replicate_attn_weights else P("model", "data")
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), in_spec),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), in_spec),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), in_spec),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), out_spec),
            attn_gate=reshard(jnp.zeros((d, n)), P(None, None)),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
        disable_rope: bool,
        qk_mult_scale: float,
        position_ids: jax.Array | None,
    ) -> Float[Array, "B S D"]:
        cfg = self.cfg
        head_dim = cfg.inferred_head_dim
        batch, seq_len = x.shape[:2]
        head_spec = P(_BATCH_AXES, None, "model", None)
        q, k, v = (
            jnp.einsum("bsh,hd->bsd", x, w).reshape((batch, seq_len, -1, head_dim), out_sharding=head_spec)
            for w in (self.w_q, self.w_k, self.w_v)
        )
        if position_ids is None:
            positions = jnp.arange(seq_len, dtype=jnp.float32)[None, :]
        else:
            positions = position_ids.astype(jnp.float32)
        rope = not disable_rope
        q = _qk_head_norm(q, positions, cfg=cfg, rope=rope, scales=(cfg.qk_mult, qk_mult_scale))
        k = _qk_head_norm(k, positions, cfg=cfg, rope=rope, scales=())
        attn_out = attention(q, k, v, mask, implementation=cfg.attention_implementation)
        attn_out = reshard(attn_out, head_spec)
        aligned_v = reshard(align_kv_heads(v, num_q_heads=attn_out.shape[2]), head_spec)
        # Exclusive Self Attention: subtract the component of y_i parallel to v_i, per head.
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Headwise gating: 2 * sigmoid(x @ attn_gate), one scalar per head.
        gate_logits = checkpoint_name(jnp.einsum("bsd,dn->bsn", x, self.attn_gate), HEAD_GATE_REMAT_SAVE_NAME)
        attn_out = 2 * jax.nn.sigmoid(gate_logits)[..., None] * attn_out
        attn_out = attn_out.reshape((batch, seq_len, -1), out_sharding=P(_BATCH_AXES, None, "model"))
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=_batch_spec())


def _gated_norm_hidden(x: jax.Array, w_down: jax.Array) -> jax.Array:
    """GatedNorm's down projection, named for the remat policy."""
    return checkpoint_name(jnp.einsum("...d,dr->...r", x, w_down), GATED_NORM_REMAT_SAVE_NAME)


def _per_device(fn, *args, out_specs):
    """``fn`` on each device's rows: ``args`` are activations [B, S, D] (batch-sharded; None passes through) or
    replicated vectors and matrices; ``fn`` sees activations as rows [b * s, D] and returns rows (or [n, D] partial
    sums, ``out_specs`` "partials", one per device in the result's leading dimension)."""
    mesh = get_abstract_mesh()
    first = next(a for a in args if a is not None and a.ndim == 3)
    act_spec = jax.typeof(first).sharding.spec
    s = first.shape[1]
    if len(act_spec) > 2 and act_spec[2] is not None:
        raise ValueError(f"the fused norm needs an unsharded hidden dimension, got {act_spec}")

    def local(*xs):
        rows = [x.reshape(-1, x.shape[-1]) if x is not None and x.ndim == 3 else x for x in xs]
        outs = fn(*rows)
        return tuple(
            o.reshape(-1, s, o.shape[-1]) if spec == "rows" else o[None] for o, spec in zip(outs, out_specs, strict=True)
        )

    if mesh.empty:
        outs = local(*args)
        return tuple(o if spec == "rows" else o[0] for o, spec in zip(outs, out_specs, strict=True))
    batch = act_spec[0] if len(act_spec) else None
    in_specs = tuple(
        None if a is None else act_spec if a.ndim == 3 else keep_reduced(P(*(None,) * a.ndim), a) for a in args
    )
    specs = tuple(act_spec if spec == "rows" else P(batch, None) for spec in out_specs)
    return jax.shard_map(local, in_specs=in_specs, out_specs=specs, check_vma=False)(*args)


def _weight_gradient(partials: jax.Array, weight: jax.Array) -> jax.Array:
    """The weight gradient from per-device partial sums [n, D] (or the sum itself without a mesh): ``unreduced`` over
    the batch axes for a weight marked ``reduced`` (its reduction waits for the caller), otherwise summed here."""
    partials = partials.astype(jnp.float32)
    if partials.ndim == 1:
        return partials.astype(weight.dtype)
    if jax.typeof(weight).sharding.spec.reduced:
        return local_to_unreduced(partials.astype(weight.dtype))
    return reshard(jnp.sum(partials, axis=0).astype(weight.dtype), jax.typeof(weight).sharding.spec)


def _token_row_sums(ids: jax.Array, rows: jax.Array, first_id: jax.Array, num_ids: int) -> jax.Array:
    """``out[v - first_id]`` = the float32 sum of ``rows[i]`` over ``ids[i] == v`` for the ``num_ids`` ids from
    ``first_id``, [num_ids, D], in a fixed order: a stable sort of the ids, a segmented scan over the sorted rows, and a
    gather of each id's last running sum."""
    order = jnp.argsort(ids, stable=True)
    sorted_ids = ids[order]
    sorted_rows = rows[order].astype(jnp.float32)
    starts = jnp.concatenate([jnp.ones((1,), bool), sorted_ids[1:] != sorted_ids[:-1]])

    def combine(a, b):
        a_start, a_sum = a
        b_start, b_sum = b
        return a_start | b_start, jnp.where(b_start[..., None], b_sum, a_sum + b_sum)

    _, running = jax.lax.associative_scan(combine, (starts, sorted_rows))
    vocab = first_id.astype(sorted_ids.dtype) + jnp.arange(num_ids, dtype=sorted_ids.dtype)
    end = jnp.maximum(jnp.searchsorted(sorted_ids, vocab, side="right") - 1, 0)
    return jnp.where((sorted_ids[end] == vocab)[:, None], running[end], 0.0)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def embedding_lookup(embed: jax.Array, token_ids: jax.Array, out_sharding: P) -> jax.Array:
    """``embed.at[token_ids].get(out_sharding=...)`` for token ids [B, S], with a bitwise-deterministic backward.

    The gather's own backward is a scatter-add over repeated ids, whose bits vary from run to run on MI355X even with
    ``--xla_gpu_deterministic_ops=true``. Here each device sums its own rows per id for its shard of the vocabulary
    (``_token_row_sums``), and the per-device sums are reduced like any replicated weight's (``_weight_gradient``)."""
    return embed.at[token_ids].get(out_sharding=out_sharding)


def _embedding_lookup_fwd(embed, token_ids, out_sharding):
    return embedding_lookup(embed, token_ids, out_sharding), (embed, token_ids)


def _embedding_lookup_bwd(out_sharding, residuals, cotangent):
    embed, token_ids = residuals
    vocab_size, hidden = embed.shape
    mesh = get_abstract_mesh()
    if mesh.empty:
        grad = _token_row_sums(token_ids.reshape(-1), cotangent.reshape(-1, hidden), jnp.int32(0), vocab_size)
        return grad.astype(embed.dtype), None
    batch = jax.typeof(cotangent).sharding.spec.partitions[0]
    vocab_entry = jax.typeof(embed).sharding.spec.partitions[0]
    vocab_axes = _axis_names(vocab_entry)
    shard_rows = vocab_size // math.prod(mesh.shape[a] for a in vocab_axes)

    def local(ids, rows):
        first_id = jax.lax.axis_index(vocab_axes) * shard_rows if vocab_axes else jnp.int32(0)
        return _token_row_sums(ids.reshape(-1), rows.reshape(-1, hidden), first_id, shard_rows)[None]

    partials = shard_map(
        local,
        in_specs=(P(batch, None), P(batch, None, None)),
        out_specs=P(batch, vocab_entry, None),
        check_vma=False,
    )(reshard(token_ids, P(batch, None)), cotangent)
    return _weight_gradient(partials, embed), None


embedding_lookup.defvjp(_embedding_lookup_fwd, _embedding_lookup_bwd)


def _norm_and_gate_fwd(x, residual, weight, w_down, w_up, eps, return_h):
    def norm(x, residual, weight):
        h, n, inv_rms = add_rms_norm_fwd(x, residual, weight, eps=eps)
        return (n, inv_rms[:, None]) if residual is None else (h, n, inv_rms[:, None])

    if residual is None:
        h = x
        n, inv_rms = _per_device(norm, x, None, weight, out_specs=("rows", "rows"))
    else:
        h, n, inv_rms = _per_device(norm, x, residual, weight, out_specs=("rows", "rows", "rows"))
    a = _gated_norm_hidden(n, w_down).astype(n.dtype)
    out, gate = _per_device(gated_norm_gate_fwd, n, a, w_up, out_specs=("rows", "rows"))
    return (h, out) if return_h else out, (h, inv_rms, weight, n, a, gate, w_down, w_up)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _norm_and_gate(x, residual, weight, w_down, w_up, eps: float, return_h: bool):
    return _norm_and_gate_fwd(x, residual, weight, w_down, w_up, eps, return_h)[0]


def _norm_and_gate_bwd(eps, return_h, residuals, cotangents):
    del eps
    h, inv_rms, weight, n, a, gate, w_down, w_up = residuals
    d_h, d_out = cotangents if return_h else (None, cotangents)
    # out = n * gate: the gate's cotangent goes back through GatedNorm's matmuls (its d_n term), n's own term is
    # d_out * gate; the RMS-norm backward adds both, and the residual's cotangent, in one pass. The gate kernel forms
    # the up projection's cotangent and its rank-128 GEMM; the other GatedNorm matmuls stay in XLA.
    dz, da = _per_device(
        lambda n, a, gate, w_up, d_out: gated_norm_gate_bwd(n, a, gate, w_up, d_out),
        n,
        a,
        gate,
        w_up,
        d_out,
        out_specs=("rows", "rows"),
    )
    _, down_vjp = jax.vjp(lambda n, w: jnp.einsum("...d,dr->...r", n, w).astype(n.dtype), n, w_down)
    d_n_from_gate, d_w_down = down_vjp(da)
    _, up_vjp = jax.vjp(lambda w: jnp.einsum("...r,rd->...d", gated_norm_silu(a), w).astype(n.dtype), w_up)
    (d_w_up,) = up_vjp(dz)

    def norm_bwd(h, inv_rms, weight, d_out, gate, d_n_from_gate, d_h):
        return add_rms_norm_bwd(h, inv_rms[:, 0], weight, d_out, gate=gate, d_n_extra=d_n_from_gate, d_h=d_h)

    dx, dw = _per_device(norm_bwd, h, inv_rms, weight, d_out, gate, d_n_from_gate, d_h, out_specs=("rows", "partials"))
    # h = x + residual: both get h's cotangent (``None`` stands for the absent residual's).
    return dx, dx, _weight_gradient(dw, weight), d_w_down, d_w_up


_norm_and_gate.defvjp(_norm_and_gate_fwd, _norm_and_gate_bwd)


def fused_norm_and_gate(
    x: Float[Array, "B S D"],
    residual: Float[Array, "B S D"] | None,
    norm: RMSNorm,
    gated: GatedNorm,
    *,
    return_h: bool,
):
    """``out = gated(norm(h))`` with ``h = x + residual`` (``h = x`` without a residual); ``(h, out)`` if
    ``return_h``, else ``out``.

    One ``jax.custom_vjp`` over the add, the RMS norm and the GatedNorm. Its backward adds ``h``'s own cotangent (from
    the caller's later uses of ``h``) inside the norm's backward kernel, so callers that use ``x`` again must use the
    returned ``h`` instead. The kernels run per device under a ``shard_map`` over the activations' batch sharding; the
    GatedNorm's other matmuls stay global. The norm is computed in float32 from the unrounded residual sum and rounds
    its outputs once, so it differs from June's separate modules at bfloat16 rounding level. A norm weight marked
    ``reduced`` gets an ``unreduced`` gradient, as without the fusion."""
    weight = norm.weight if jax.typeof(norm.weight).sharding.spec.reduced else unshard(norm.weight)
    return _norm_and_gate(x, residual, weight, gated.w_down, gated.w_up, norm.eps, return_h)


def _router_z_loss_local(
    router_logits: Float[Array, "T E"], token_valid: Bool[Array, " T"], mesh: jax.sharding.AbstractMesh
) -> dict[str, jax.Array]:
    """This batch shard's router z-loss partials, one row per shard: the valid tokens' sum of squared log-partition
    functions and their count."""

    def local(logits: jax.Array, valid: jax.Array) -> dict[str, jax.Array]:
        valid_f = valid.astype(jnp.float32)
        log_partition = jsp.special.logsumexp(logits.astype(jnp.float32), axis=-1)
        return {
            "router_z_sq_sum_local": jnp.sum(log_partition**2 * valid_f)[None],
            "valid_tokens_local": jnp.sum(valid_f)[None],
        }

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(P(_BATCH_AXES, None), P(_BATCH_AXES)),
        out_specs={"router_z_sq_sum_local": P(_BATCH_AXES), "valid_tokens_local": P(_BATCH_AXES)},
    )(router_logits, token_valid)


def _qb_beta_topk_local(
    s_minus_alpha: Float[Array, "T E"],
    token_valid: Bool[Array, " T"],
    mesh: jax.sharding.AbstractMesh,
    *,
    num_experts_per_token: int,
    num_experts: int,
) -> tuple[Float[Array, "S E"], Int[Array, " S"]]:
    """This batch shard's QB thresholds and its valid-token count, which weights them, one row per shard.

    June's ``estimate_qb_beta_topk`` without its cross-shard weighted mean, which ``_reduce_routing_stats`` takes once
    after the layer scan."""

    def local(s_local: jax.Array, valid_local: jax.Array) -> tuple[jax.Array, jax.Array]:
        physical_count = qb_topk_physical_count(
            s_local.shape[0], num_experts_per_token=num_experts_per_token, num_experts=num_experts
        )
        beta, valid_count = qb_beta_topk_shard(
            s_local,
            valid_local,
            physical_count=physical_count,
            num_experts_per_token=num_experts_per_token,
            num_experts=num_experts,
        )
        return beta[None, :], valid_count[None]

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(P(_BATCH_AXES, None), P(_BATCH_AXES)),
        out_specs=(P(_BATCH_AXES, None), P(_BATCH_AXES)),
    )(s_minus_alpha, token_valid)


def _ring_filtered_experts(
    expert_mlp: MoEExpertMlp,
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    token_valid: Bool[Array, " T"],
    mesh: jax.sharding.AbstractMesh,
) -> tuple[Float[Array, "T D"], CapacityDrops]:
    """Levanter's ``ring_filtered`` expert-parallel MoE, as ``moe_mlp`` calls it, except that the sender and receiver
    drop counts stay per token shard ([S] arrays) for ``_reduce_routing_stats`` to sum after the layer scan."""
    token_spec = _token_spec_from_x(x, mesh)
    drops_spec = P(_axis_names(token_spec[0]))
    w_gate_up = jnp.concatenate([expert_mlp.w_gate, expert_mlp.w_up], axis=-1)
    w_gate_up_spec = keep_reduced(P("expert", None, None), w_gate_up)
    w_down_spec = keep_reduced(P("expert", None, None), expert_mlp.w_down)
    num_experts = w_gate_up.shape[0]

    def local(x, selected_experts, combine_weights, token_valid, w_gate_up, w_down):
        out, drops = _moe_mlp_ep_ring_filtered_local(
            x,
            selected_experts,
            combine_weights,
            token_valid,
            w_gate_up,
            w_down,
            activation_fn=ActivationFunctionEnum.silu.to_jax_fn(),
            num_experts=num_experts,
            capacity_factor=expert_mlp.capacity_factor,
            transport_capacity_factor=expert_mlp.pooled_transport_capacity_factor,
            # No token axes: the drop counts come back unsummed.
            token_sharding_axes=(),
        )
        return out, CapacityDrops(drops.sender_dropped[None], drops.receiver_dropped[None])

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(token_spec, token_spec, token_spec, token_spec, w_gate_up_spec, w_down_spec),
        out_specs=(token_spec, CapacityDrops(sender_dropped=drops_spec, receiver_dropped=drops_spec)),
        check_vma=False,
    )(
        *(_reshard_for_shard_map(a, mesh, token_spec) for a in (x, selected_experts, combine_weights, token_valid)),
        _reshard_for_shard_map(w_gate_up, mesh, w_gate_up_spec),
        _reshard_for_shard_map(expert_mlp.w_down, mesh, w_down_spec),
    )


def _reduce_routing_stats(stacked: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """The per-layer routing statistics from the layer scan's stacked per-shard partials ([L, S, ...]): QB thresholds
    (the valid-token-weighted mean of the shards'), capacity drops and the router z-loss. Every sum over the shards
    here is one cross-device reduction for the whole layer stack."""
    sender = jnp.sum(stacked["routing_sender_drops_local"], axis=1)
    receiver = jnp.sum(stacked["routing_receiver_drops_local"], axis=1)
    qb_weights = stacked["qb_beta_weight_local"].astype(jnp.float32)
    valid_tokens = jnp.maximum(jnp.sum(stacked["valid_tokens_local"], axis=1), 1.0)
    return {
        "capacity_overflow": (sender + receiver).astype(jnp.float32),
        "qb_beta": (
            jnp.sum(stacked["qb_beta_local"] * qb_weights[:, :, None], axis=1)
            / jnp.maximum(jnp.sum(qb_weights, axis=1)[:, None], 1)
        ),
        "router_z_loss": jnp.sum(stacked["router_z_sq_sum_local"], axis=1) / valid_tokens,
    }


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights, through ``ring_filtered``, keeping its routing statistics per batch
    shard for one reduction after the layer scan."""

    router: jax.Array
    router_bias: jax.Array
    expert_mlp: MoEExpertMlp
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_expert = random.split(key, 2)
        expert_axis_size = _mesh_axis_size(get_abstract_mesh(), "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")
        if not cfg.split_w_gate_up:
            raise ValueError("the current Levanter MoE API only supports split w_gate and w_up weights")
        return MoEMLP(
            router=reshard(
                _init_weight(k_router, (cfg.hidden_dim, cfg.num_experts), cfg.initializer_std), P(None, None)
            ),
            router_bias=jnp.zeros((cfg.num_experts,)),
            expert_mlp=MoEExpertMlp.init(
                num_experts=cfg.num_experts,
                hidden_dim=cfg.hidden_dim,
                intermediate_dim=cfg.intermediate_dim,
                initializer_std=cfg.initializer_std,
                key=k_expert,
                implementation=MOE_IMPLEMENTATION,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=cfg.capacity_factor,
                pooled_transport_capacity_factor=cfg.pooled_transport_capacity_factor,
                num_expert_waves=cfg.num_expert_waves,
            ),
            cfg=cfg,
        )

    @named_call
    def __call__(
        self, x: Float[Array, "B S D"], token_valid: Bool[Array, "B S"]
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """The routed output and this layer's per-shard routing statistics (see ``_reduce_routing_stats``)."""
        cfg = self.cfg
        k = cfg.num_experts_per_token
        mesh = get_abstract_mesh()
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        valid = rearrange(token_valid, "b s -> (b s)")
        token_spec = P(_BATCH_AXES, None)
        valid_sharded = reshard(valid, P(_BATCH_AXES))
        with jax.named_scope("moe_route"):
            router_logits = jnp.einsum("td,de->te", x_flat, replicate_keeping_reduced(self.router)).astype(jnp.float32)
            biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
            # One expert past the top k: its logit is the QB threshold's offset.
            topk_logits, selected_experts = routing_top_k(biased_logits, k + 1, mesh=mesh, batch_axes=_BATCH_AXES)
            qb_alpha = topk_logits[:, -1:]
            selected_experts = selected_experts[:, :-1]
            combine_weights = jax.nn.sigmoid(jnp.take_along_axis(router_logits, selected_experts, axis=-1))
            # Keep sigmoid rounding consistent when AD fuses expert-weight renormalization.
            combine_weights = jax.lax.optimization_barrier(combine_weights)
            combine_weights *= _ROUTING_RENORM_SUM / (jnp.sum(combine_weights, axis=-1, keepdims=True) + 1e-9)
            combine_weights = combine_weights.astype(x.dtype)

        with jax.named_scope("moe_router_stats"):
            stats = _router_z_loss_local(reshard(router_logits, token_spec), valid_sharded, mesh)
        with jax.named_scope("moe_qb_beta"):
            stats["qb_beta_local"], stats["qb_beta_weight_local"] = _qb_beta_topk_local(
                reshard(router_logits - qb_alpha, token_spec),
                valid_sharded,
                mesh,
                num_experts_per_token=k,
                num_experts=cfg.num_experts,
            )
        with jax.named_scope("moe_experts"):
            selected_experts = selected_experts.astype(jnp.int32)
            if _mesh_axis_size(mesh, "expert") > 1:
                routed_flat, drops = _ring_filtered_experts(
                    self.expert_mlp, x_flat, selected_experts, combine_weights, valid, mesh
                )
            else:
                # Without expert parallelism Levanter's local path runs, and its drop counts come back summed.
                routed_flat, counts = self.expert_mlp(
                    x_flat,
                    selected_experts,
                    combine_weights,
                    token_valid=valid,
                    mesh=mesh,
                    report_capacity_overflow=True,
                )
                drops = CapacityDrops(counts.sender_dropped[None], counts.receiver_dropped[None])
        # With expert parallelism, in float32 the drop counts join the other statistics' float32 all-reduce after
        # the layer scan instead of taking an int32 one of their own (the local path above has already summed them).
        # A layer drops at most its b * s * k assignments, so they stay exact.
        drops_dtype = jnp.float32 if b * s * k <= _FLOAT32_EXACT_INTEGERS else jnp.int32
        stats["routing_sender_drops_local"] = drops.sender_dropped.astype(drops_dtype)
        stats["routing_receiver_drops_local"] = drops.receiver_dropped.astype(drops_dtype)
        stats["skipped_assignments"] = padding_skipped_assignments(valid, topk=k).astype(jnp.float32)
        routed = reshard(rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s), _batch_spec())
        return routed, stats


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
        short_mask: AttentionMask,
        long_mask: AttentionMask,
        use_long_mask: Bool[Array, ""],
        position_ids: jax.Array | None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        cfg = self.attn.cfg
        # x comes back as h, so its cotangent from the residual path joins the norm's backward kernel.
        x, attn_in = fused_norm_and_gate(x, None, self.rms_attn, self.attn_gated_norm, return_h=True)
        # lax.cond keeps the scan body uniform: long layers use the full causal mask (and may drop RoPE and scale q
        # further), short layers the sliding window.
        attn_out = jax.lax.cond(
            use_long_mask,
            lambda: self.attn(attn_in, long_mask, cfg.disable_long_rope, cfg.qk_mult_long_scale, position_ids),
            lambda: self.attn(attn_in, short_mask, False, 1.0, position_ids),
        )
        x, mlp_in = fused_norm_and_gate(x, attn_out, self.rms_mlp, self.mlp_gated_norm, return_h=True)
        token_valid = token_validity_from_attention_mask(short_mask, batch_size=x.shape[0], sequence_length=x.shape[1])
        mlp_out, router_stats = self.mlp(mlp_in, token_valid)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        return x + mlp_out, router_stats


class Transformer(eqx.Module):
    """June's ``Transformer`` with its blocks always stacked; same parameter tree and checkpoint paths."""

    token_embed: jax.Array
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    output_proj: jax.Array
    stacked_blocks: ArrayStacked[Block]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        check_supported(cfg)
        # 4 module-level keys + per-layer keys, as June splits them.
        keys = random.split(key, cfg.num_layers + 4)
        embed_key, out_key, embed_gn_key, final_gn_key = keys[:4]
        return Transformer(
            token_embed=reshard(
                _init_weight(embed_key, (cfg.vocab_size, cfg.hidden_dim), cfg.initializer_std), Pembed_vocab
            ),
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            output_proj=reshard(_init_weight(out_key, (cfg.hidden_dim, cfg.vocab_size), cfg.initializer_std), Plm_head),
            stacked_blocks=ArrayStacked.init(cfg.num_layers, Block)(cfg=cfg, key=keys[4:]),
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            config=cfg,
        )

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
        *,
        position_ids: jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        """The final hidden states and the per-layer router metrics (``<name>_per_layer``)."""
        hidden, router_metrics, _ = self._forward(token_ids, mask, position_ids=position_ids)
        return hidden, router_metrics

    def _forward(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None,
        *,
        position_ids: jax.Array | None = None,
        routing: LayerRouting = LayerRouting(),
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array], LayerRouting]:
        """The final hidden states, the per-layer router metrics, and ``routing`` with the gather buffers and gradient
        sums as the layer scan returns them (see ``LayerRouting``)."""
        if (routing.gather_sinks is None) != (routing.gather_buffers is None):
            raise ValueError("gather_sinks and gather_buffers go together")
        if mask is None:
            mask = AttentionMask.causal()
        cfg = self.config
        hidden = embedding_lookup(self.token_embed, token_ids, _batch_spec())
        # Keep gather fusion from changing normalization rounding between scoring and AD.
        hidden = jax.lax.optimization_barrier(hidden)
        hidden = fused_norm_and_gate(hidden, None, self.embed_norm, self.embed_gated_norm, return_h=False)

        # Short layers: sliding window. Long layers (every 4th + last): full causal.
        segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
        short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
        long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)
        # Inside the scan the backward cannot reuse the forward's values, so the checkpoint needs no CSE barriers;
        # without them XLA fuses the backward's slices of the saved residuals (about 100 ms per MI355X step).
        checkpoint = functools.partial(
            eqx.filter_checkpoint,
            policy=jax.checkpoint_policies.save_only_these_names(*REMAT_SAVE_NAMES),
            prevent_cse=False,
        )

        def scan_layer(carry, layer_inputs):
            hidden, buffers, sums = carry
            layer, use_long_mask, offsets, sinks, gather_sinks, index = layer_inputs
            if gather_sinks is not None:
                layer, buffers = apply_gathers(layer, gather_sinks, buffers, index)
            args = (hidden, short_mask, long_mask, use_long_mask, position_ids)
            if sums is None and offsets is None:
                hidden, stats = checkpoint(layer)(*args)
            else:
                hidden, stats, *rest = checkpoint(call_routed)(layer, sums, offsets, sinks, index, *args)
                sums = rest[0] if rest else None
            return (hidden, buffers, sums), stats

        # The layer index locates each layer's rows in the gathers and sums; without them the scan does not carry it.
        routed = any(x is not None for x in (routing.gather_sinks, routing.grad_sums, routing.grad_offsets))
        layer_indices = jnp.arange(cfg.num_layers, dtype=jnp.int32) if routed else None
        # The small replicated weights' gradients wait for one reduction after the scan.
        stacked = mark_replicated_reduced(self.stacked_blocks.stacked)
        (hidden, buffers, sums), stacked_stats = jax.lax.scan(
            scan_layer,
            (hidden, routing.gather_buffers, routing.grad_sums),
            xs=(
                stacked,
                _long_layer_schedule(cfg.num_layers),
                routing.grad_offsets,
                routing.grad_sinks,
                routing.gather_sinks,
                layer_indices,
            ),
        )
        # One cross-device reduction of the routing statistics for the whole layer stack.
        reduced = _reduce_routing_stats(stacked_stats)
        router_metrics = {
            "qb_beta_per_layer": reduced["qb_beta"],
            "capacity_overflow_per_layer": reduced["capacity_overflow"],
            "skipped_assignments_per_layer": stacked_stats["skipped_assignments"],
            "router_z_loss_per_layer": reduced["router_z_loss"],
        }
        hidden = fused_norm_and_gate(hidden, None, self.final_norm, self.final_gated_norm, return_h=False)
        return hidden, router_metrics, routing._replace(gather_buffers=buffers, grad_sums=sums)

    @named_call
    def logits(self, token_ids: Int[Array, "B S"], mask: AttentionMask | jax.Array | None = None) -> jax.Array:
        hidden, _ = self(token_ids, mask=mask)
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
        return_router_metrics: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array]]:
        terms = self.next_token_loss_terms(
            token_ids,
            loss_weight,
            mask=mask,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            loss_dtype=loss_dtype,
        )
        loss = terms.cross_entropy_loss + terms.aux_loss if reduction != "none" else terms.cross_entropy_loss
        if not return_router_metrics:
            return loss
        return loss, summarize_loss_metrics(terms.router_metrics, terms.cross_entropy_loss, terms.aux_loss)

    def next_token_loss_terms(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        routing: LayerRouting = LayerRouting(),
    ) -> "LossTerms":
        """The loss terms, the raw per-layer router metrics and ``routing`` as the layer scan returns it.

        Callers that combine several batches, such as the trainer's microbatches, combine the raw metrics and
        summarize them once (``summarize_loss_metrics``)."""
        hidden, router_metrics, routing = self._forward(token_ids, mask, routing=routing)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight.astype(loss_dtype),
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
        # No load-balancing loss. The router z-loss is always logged; at weight 0 it stays out of the loss, so the
        # backward does not differentiate it. A mean over the valid tokens.
        router_z_loss = router_metrics["router_z_loss_per_layer"]
        if self.config.router_z_loss_coef == 0:
            aux_loss = jnp.zeros((), jnp.float32)
        else:
            aux_loss = self.config.router_z_loss_coef * jnp.mean(router_z_loss)
        return LossTerms(cross_entropy_loss, aux_loss, router_metrics, routing)


class LossTerms(NamedTuple):
    """``Transformer.next_token_loss_terms``."""

    cross_entropy_loss: jax.Array
    aux_loss: jax.Array
    router_metrics: dict[str, jax.Array]
    routing: LayerRouting


def summarize_loss_metrics(
    router_metrics: dict[str, jax.Array], cross_entropy_loss: jax.Array, aux_loss: jax.Array
) -> dict[str, jax.Array]:
    """The training metrics ``next_token_loss`` returns from the raw per-layer router metrics and the loss terms."""
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    metrics: dict[str, jax.Array] = {
        "qb_beta_per_layer": router_metrics["qb_beta_per_layer"],
        "train/cross_entropy_loss": cross_entropy_loss,
        "train/router/aux_loss_weighted": aux_loss,
        MOE_DROPPED_ASSIGNMENTS_METRIC: router_metrics["capacity_overflow_per_layer"],
        MOE_SKIPPED_PADDING_ASSIGNMENTS_METRIC: router_metrics["skipped_assignments_per_layer"],
        "train/router/router_z_loss": jnp.mean(router_z_loss),
    }
    for i in range(router_z_loss.shape[0]):
        metrics[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
    return metrics
