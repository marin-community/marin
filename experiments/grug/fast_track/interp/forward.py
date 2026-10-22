# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Instrumented forward of the Ckpt 1 d512 model (KDA + MLA/Inkling layers, one Block AttnRes block per layer, latent
ReLU^2 MoE with one shared expert), exposing every intermediate the study reads and taking traced ablation inputs.

It re-implements ``Transformer.__call__`` for exactly this config (``check_config`` refuses any other) from the model's
own parameters and kernels: the AttnRes gates over ``[embed, L0, ..., L(i-1)]`` (attention) and ``[..., attn_i]``
(MLP), KDA through the fused kernel, MLA through the FA4 kernel, and the MoE as a dropless sorted (``ragged_dot``)
dispatch whose hidden units (the "neurons") are materialized. ``verify.py`` checks it against
``Transformer.next_token_loss``.

Ablations are traced inputs (one compile serves all of them): every head, sublayer and neuron has a keep multiplier
``m`` and its output becomes ``m * out + (1 - m) * mean``.
"""

import functools
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from haliax.nn.ragged_dot import ragged_dot
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from levanter.grug.attention import AttentionMask, attention, fa4_cute_segment_bounds

from experiments.grug.fast_track.key_offset import partial_key_offset
from experiments.grug.fast_track.model import (
    KDA_MIN_LOG_DECAY,
    Block,
    CausalSelfAttention,
    GrugModelConfig,
    KdaKernelOptions,
    KimiDeltaAttention,
    RouterCombine,
    Transformer,
    ValueEmbeds,
    _embedding_gather,
    _gated_rms_norm,
    _is_long_layer,
    _kda_kernel,
    _small_top_k,
    rms_norm,
)
from experiments.grug.moe.kda_prep_pallas import GateFn, GateKind, gate_log_decay

NUM_LAYERS = 6
NUM_HEADS = 4
HEAD_DIM = 128
NUM_EXPERTS = 384
TOP_K = 10
EXPERT_WIDTH = 384
SHARED_WIDTH = 384
LATENT = 256
HIDDEN = 512
VOCAB = 16384
KDA_LAYERS = (0, 1, 2, 4)
MLA_LAYERS = (3, 5)
# Sublayer ablation slots: attention (whole branch), shared expert, routed experts, MLA value embeddings.
SUB_ATTN, SUB_SHARED, SUB_ROUTED, SUB_VE = range(4)
NUM_SUB = 4
_BATCH = ("replica_dcn", "data", "expert")


def _bspec(x: jax.Array) -> jax.Array:
    """``x`` sharded on its leading (batch) axis only."""
    return reshard(x, P(_BATCH, *([None] * (x.ndim - 1))))


def check_config(cfg: GrugModelConfig) -> None:
    """Refuse any config this forward does not reproduce exactly."""
    expected = dict(
        num_layers=NUM_LAYERS,
        hidden_dim=HIDDEN,
        num_heads=NUM_HEADS,
        mla_num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOP_K,
        num_shared_experts=1,
        intermediate_dim=EXPERT_WIDTH,
        shared_expert_intermediate_dim=SHARED_WIDTH,
        latent_dim=LATENT,
        vocab_size=VOCAB,
        attn_res=True,
        attn_res_num_blocks=8,
        attn_res_plain_sum=False,
        moe_ungated_relu2=True,
        shared_ungated_relu2=True,
        router_combine=RouterCombine.SQRT_SOFTPLUS_RENORM,
        value_embeds=ValueEmbeds.GATED,
        mla=True,
        inkling_relpos=True,
        mla_key_offset=True,
        mla_k_norm=True,
        mla_q_norm=True,
        qk_norm=False,
        learnable_qk_mult=True,
        attn_output_gate=True,
        attn_xsa=True,
        sconv=True,
        sconv_sites=("attn", "mlp"),
        sublayer_gated_norm=True,
        embed_gated_norm=False,
        final_gated_norm=False,
        kda_onchip_decay_gate=True,
        kda_write_gate=False,
        kda_erase_gate=False,
        num_null_experts=0,
        second_embed=False,
        logit_soft_cap=10,
    )
    wrong = {k: (getattr(cfg, k), v) for k, v in expected.items() if getattr(cfg, k) != v}
    if wrong:
        raise ValueError(f"interp forward does not reproduce this config: {wrong}")
    layers = tuple(i for i in range(NUM_LAYERS) if not _is_long_layer(i, NUM_LAYERS, cfg.global_every))
    if layers != KDA_LAYERS:
        raise ValueError(f"expected KDA layers {KDA_LAYERS}, got {layers}")


class Ablation(NamedTuple):
    """Keep multipliers (1 = untouched) and the replacement means (``m * out + (1 - m) * mean``)."""

    head_mult: jax.Array  # [L, H]
    head_mean: jax.Array  # [L, H, D] mean per-head contribution (after w_o, before the branch SConv)
    sub_mult: jax.Array  # [L, NUM_SUB]
    sub_mean: jax.Array  # [L, NUM_SUB, D]
    neuron_mult: jax.Array  # [L, E, I] routed hidden units
    neuron_mean: jax.Array  # [L, E, I] their replacement activation (0: zero ablation)
    shared_mult: jax.Array  # [L, I] shared hidden units
    shared_mean: jax.Array  # [L, I] their replacement activation (0: zero ablation)
    slot_mult: jax.Array  # [L, K] per top-k rank (rank 0 = largest combine weight)
    slot_renorm: jax.Array  # [] 1: renormalize the kept combine weights to the original sum
    ve_head_mult: jax.Array  # [L, H] MLA value-embedding term per head (times ``sub_mult[:, SUB_VE]``)
    ve_mean: jax.Array  # [L, H, head_dim] its replacement: the mean gated value embedding
    kda_early_tokens: jax.Array  # [] KDA writes of each document's first this-many tokens are ablated
    kda_early_mode: jax.Array  # [] 0: untouched, 1: zero (beta = 0), 2: mean (their values replaced by the mean value)


def no_ablation() -> Ablation:
    return Ablation(
        head_mult=jnp.ones((NUM_LAYERS, NUM_HEADS), jnp.float32),
        head_mean=jnp.zeros((NUM_LAYERS, NUM_HEADS, HIDDEN), jnp.float32),
        sub_mult=jnp.ones((NUM_LAYERS, NUM_SUB), jnp.float32),
        sub_mean=jnp.zeros((NUM_LAYERS, NUM_SUB, HIDDEN), jnp.float32),
        neuron_mult=jnp.ones((NUM_LAYERS, NUM_EXPERTS, EXPERT_WIDTH), jnp.float32),
        neuron_mean=jnp.zeros((NUM_LAYERS, NUM_EXPERTS, EXPERT_WIDTH), jnp.float32),
        shared_mult=jnp.ones((NUM_LAYERS, SHARED_WIDTH), jnp.float32),
        shared_mean=jnp.zeros((NUM_LAYERS, SHARED_WIDTH), jnp.float32),
        slot_mult=jnp.ones((NUM_LAYERS, TOP_K), jnp.float32),
        slot_renorm=jnp.zeros((), jnp.float32),
        ve_head_mult=jnp.ones((NUM_LAYERS, NUM_HEADS), jnp.float32),
        ve_mean=jnp.zeros((NUM_LAYERS, NUM_HEADS, HEAD_DIM), jnp.float32),
        kda_early_tokens=jnp.zeros((), jnp.int32),
        kda_early_mode=jnp.zeros((), jnp.int32),
    )


class RoutingOverride(NamedTuple):
    """Per-layer routing edits over the ``T = B * S`` flattened tokens (traced, so one compile serves all of them).

    ``bias_scale`` multiplies the QB bias in the top-k selection (0: unbiased selection); where ``sel_mask``, the
    token's selection is ``sel_override`` (distinct experts; combine weights follow from their unbiased logits as
    usual); ``combine_mult`` scales each slot's final combine weight (a capacity drop is 0); ``latent_delta`` is added
    to the routed latent output (before ``w_latent_up``), the point the routing studies differentiate at.
    """

    bias_scale: jax.Array  # [L]
    sel_mask: jax.Array  # [L, T] bool
    sel_override: jax.Array  # [L, T, K] int32
    combine_mult: jax.Array  # [L, T, K]
    latent_delta: jax.Array  # [L, T, R]


def no_override(tokens: int) -> RoutingOverride:
    return RoutingOverride(
        bias_scale=jnp.ones((NUM_LAYERS,), jnp.float32),
        sel_mask=jnp.zeros((NUM_LAYERS, tokens), bool),
        sel_override=jnp.zeros((NUM_LAYERS, tokens, TOP_K), jnp.int32),
        combine_mult=jnp.ones((NUM_LAYERS, tokens, TOP_K), jnp.float32),
        latent_delta=jnp.zeros((NUM_LAYERS, tokens, LATENT), jnp.float32),
    )


def position_in_document(seg: jax.Array) -> jax.Array:
    """``[B, S]`` each token's offset from its document's first token."""
    idx = jnp.broadcast_to(jnp.arange(seg.shape[1]), seg.shape)
    start = jnp.concatenate([jnp.ones_like(seg[:, :1], bool), seg[:, 1:] != seg[:, :-1]], axis=1)
    return idx - jax.lax.cummax(jnp.where(start, idx, 0), axis=1)


def _mix(m: jax.Array, out: jax.Array, mean: jax.Array) -> jax.Array:
    return (m * out.astype(jnp.float32) + (1.0 - m) * mean).astype(out.dtype)


def _gate(sources: list[jax.Array], query: jax.Array, eps: float) -> tuple[jax.Array, jax.Array]:
    """One AttnRes gate: softmax over ``q . rms_norm(src)`` (float32) and the mix, as ``_attn_res_mix``."""
    logits = []
    for src in sources:
        inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(src.astype(jnp.float32)), axis=-1) + eps)
        dots = jnp.einsum("bsd,d->bs", src, query.astype(src.dtype), preferred_element_type=jnp.float32)
        logits.append(dots * inv_rms)
    weights = jax.nn.softmax(jnp.stack(logits), axis=0)
    mixed = weights[0][..., None] * sources[0].astype(jnp.float32)
    for w, src in zip(weights[1:], sources[1:], strict=True):
        mixed = mixed + w[..., None] * src.astype(jnp.float32)
    return weights, mixed.astype(sources[0].dtype)


def _sconv(conv, x: jax.Array, seg: jax.Array) -> jax.Array:
    return conv(x, seg)


def _heads_through(conv, contrib: jax.Array, seg: jax.Array) -> jax.Array:
    """``[B, S, H, D]`` per-head contributions through a channelwise (linear) SConv, heads folded into the batch."""
    h = contrib.shape[2]
    folded = rearrange(reshard(contrib, P(None, None, None, None)), "b s h d -> (h b) s d")
    out = conv(folded, jnp.tile(reshard(seg, P(None, None)), (h, 1)))
    return _bspec(rearrange(out, "(h b) s d -> b s h d", h=h))


def _kda_branch(attn: KimiDeltaAttention, x, seg, cfg: GrugModelConfig, head_mult, head_mean, abl: Ablation, out: dict):
    """KDA (as ``KimiDeltaAttention.__call__``) with per-head outputs; records q, k, v, log-decay g and beta."""

    def project(w, conv):
        y = jax.nn.silu(conv(jnp.einsum("bsd,de->bse", x, w), seg))
        return rearrange(y, "... (n d) -> ... n d", d=HEAD_DIM)

    q = project(attn.w_q, attn.sconv_q)
    k = project(attn.w_k, attn.sconv_k)
    v = project(attn.w_v, attn.sconv_v)
    a = jnp.einsum("bsr,re->bse", jnp.einsum("bsd,dr->bsr", x, attn.w_a_down), attn.w_a_up)
    a = rearrange(a, "... (n d) -> ... n d", d=attn.dt_bias.shape[-1])
    scale = jnp.exp(attn.a_log.astype(jnp.float32))[:, None]
    gate_p = jnp.broadcast_to(scale, attn.dt_bias.shape)
    gate_bias = attn.dt_bias.astype(jnp.float32)
    beta = jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, attn.w_beta).astype(jnp.float32))
    out["v_clean"] = v
    early = (position_in_document(seg) < abl.kda_early_tokens)[..., None]  # [B, S, 1]
    beta = jnp.where(early & (abl.kda_early_mode == 1), 0.0, beta)
    mean_v = jnp.mean(v.astype(jnp.float32), axis=(0, 1), keepdims=True).astype(v.dtype)
    v = jnp.where(early[..., None] & (abl.kda_early_mode == 2), mean_v, v)
    options = KdaKernelOptions(
        cfg.kda_chunk_size,
        False,
        cfg.kda_state_grads,
        cfg.kda_state_block_v,
        cfg.kda_state_primal_block_v,
        cfg.kda_state_num_stages,
        cfg.kda_prep_chunks,
        cfg.kda_state_saved_bf16,
        cfg.kda_state_grads_chunks,
    )
    spec4, spec3, head_spec = P(_BATCH, None, "model", None), P(_BATCH, None, "model"), P("model", None)
    extras = {"gate_p": reshard(gate_p, head_spec), "gate_bias": reshard(gate_bias, head_spec), "segment_ids": seg}
    extra_specs = {"gate_p": head_spec, "gate_bias": head_spec, "segment_ids": P(_BATCH, None)}
    args = (*(reshard(t, spec4) for t in (q, k, v, a)), reshard(beta, spec3), extras)
    # As the model: the Pallas custom VJPs are not vma-annotated.
    kernel = jax.shard_map(
        functools.partial(_kda_kernel, options=options),
        mesh=get_abstract_mesh(),
        in_specs=(spec4,) * 4 + (spec3, extra_specs),
        out_specs=spec4,
        check_vma=False,
    )
    o = _bspec(kernel(*args))
    out["kda_o"] = o
    o = attn.o_norm(o.astype(x.dtype))
    gate = jax.nn.sigmoid(jnp.einsum("bsd,de->bse", x, attn.w_g))
    o = o * rearrange(gate, "... (n d) -> ... n d", d=HEAD_DIM)
    w_o = rearrange(attn.w_o, "(n d) e -> n d e", d=HEAD_DIM)
    contrib = jnp.einsum("bsnd,nde->bsne", o, w_o, preferred_element_type=jnp.float32)
    contrib = _mix(head_mult[None, None, :, None], contrib, head_mean[None, None]).astype(x.dtype)
    out.update(
        q=q,
        k=k,
        v=v,
        log_decay=gate_log_decay(GateFn(GateKind.SIGMOID, KDA_MIN_LOG_DECAY), a.astype(jnp.float32), gate_p, gate_bias),
        beta=beta,
        head_contrib=contrib,
        head_gate=jnp.mean(rearrange(gate, "... (n d) -> ... n d", d=HEAD_DIM).astype(jnp.float32), axis=-1),
    )
    return jnp.sum(contrib.astype(jnp.float32), axis=2).astype(x.dtype), contrib


def mla_qkv(attn: CausalSelfAttention, x, seg, token_ids, ve_mult, ve_mean):
    """MLA q, k (offset, normed, qk_mult-scaled as the kernel sees them), the projected value and the gated value
    embedding (``v = v_proj + ve``), and the Inkling bias band."""
    latent = attn.kv_latent_norm(jnp.einsum("bsd,dl->bsl", x, attn.w_dkv))
    q = rearrange(jnp.einsum("bsd,de->bse", x, attn.w_q), "... (n d) -> ... n d", d=HEAD_DIM)
    k = rearrange(jnp.einsum("bsl,le->bse", latent, attn.w_uk), "... (n d) -> ... n d", d=HEAD_DIM)
    v_proj = rearrange(jnp.einsum("bsl,le->bse", latent, attn.w_uv), "... (n d) -> ... n d", d=HEAD_DIM)
    table = _embedding_gather(attn.value_embed.astype(x.dtype), token_ids)
    ve = rearrange(table, "... (n d) -> ... n d", d=HEAD_DIM)
    ve_w = jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, attn.ve_gate.astype(x.dtype)))[..., None]
    lam = attn.ve_lambda.astype(x.dtype)
    v_proj = lam[0] * v_proj
    m = ve_mult.astype(jnp.float32)[None, None, :, None]
    ve = (m * (ve_w * ve).astype(jnp.float32) + (1.0 - m) * ve_mean[None, None]).astype(x.dtype)
    k = partial_key_offset(k, seg, P(_BATCH, None, None, None), ())
    k = rms_norm(k)
    q = rms_norm(q)
    q = q * attn.qk_mult.astype(q.dtype)
    rel_bias = attn.rel_pos(x)
    return _bspec(q), _bspec(k), _bspec(v_proj), _bspec(ve), rel_bias, ve_w[..., 0]


def _xsa(y: jax.Array, v: jax.Array) -> jax.Array:
    """The component of ``y`` along ``v`` (per head), removed by Exclusive Self Attention."""
    dot = jnp.sum(y * v, axis=-1, keepdims=True)
    return (dot / (jnp.sum(v * v, axis=-1, keepdims=True) + 1e-6)) * v


def _mla_branch(attn: CausalSelfAttention, x, seg, token_ids, mask, head_mult, head_mean, ve_mult, ve_mean, out: dict):
    """MLA (as ``CausalSelfAttention.__call__``) with per-head outputs split into the projected-value and the
    value-embedding parts (XSA, the output gate and ``w_o`` are linear in the attended value)."""
    q, k, v_proj, ve, rel_bias, ve_w = mla_qkv(attn, x, seg, token_ids, ve_mult, ve_mean)
    v = v_proj + ve
    impl = attn.cfg.gpu_attention if jax.default_backend() == "gpu" else None
    y_proj = _bspec(attention(q, k, v_proj, mask, implementation=impl, rel_bias=rel_bias)[..., :HEAD_DIM])
    y_ve = _bspec(attention(q, k, ve, mask, implementation=impl, rel_bias=rel_bias)[..., :HEAD_DIM])
    v = _bspec(v)
    y_proj = y_proj - _xsa(y_proj, v)
    y_ve = y_ve - _xsa(y_ve, v)
    gate_logit = jnp.einsum("bsd,dn->bsn", x, attn.attn_gate).astype(jnp.float32)
    gate = (2 * jax.nn.sigmoid(gate_logit)).astype(y_proj.dtype)[..., None]
    w_o = rearrange(attn.w_o, "(n d) e -> n d e", d=HEAD_DIM)
    c_proj = jnp.einsum("bsnd,nde->bsne", gate * y_proj, w_o, preferred_element_type=jnp.float32)
    c_ve = jnp.einsum("bsnd,nde->bsne", gate * y_ve, w_o, preferred_element_type=jnp.float32)
    hm = head_mult[None, None, :, None]
    contrib = _mix(hm, c_proj + c_ve, head_mean[None, None]).astype(x.dtype)
    # The value-embedding share of each (possibly ablated) head: the mean replacement is all projection.
    ve_contrib = (hm * c_ve).astype(x.dtype)
    out.update(
        q=q, k=k, v_proj=v_proj, ve=ve, ve_weight=ve_w, rel_bias=rel_bias, head_gate=gate[..., 0], head_contrib=contrib
    )
    return contrib, ve_contrib


def _dispatch(selected: jax.Array):
    """Dropless sorted dispatch of ``[T, K]`` assignments: each assignment's expert, token and slot in expert order,
    and the per-expert counts and segment starts."""
    k = selected.shape[1]
    flat = selected.reshape(-1)
    order = jnp.argsort(flat, stable=True)
    counts = jnp.zeros((NUM_EXPERTS,), jnp.int32).at[flat].add(1)
    starts = jnp.cumsum(counts) - counts
    return flat[order], order // k, order % k, counts, starts


def _local_ragged_dot(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array) -> jax.Array:
    """``ragged_dot`` on replicated operands inside a ``shard_map`` (no explicit-sharding rules needed)."""
    return jax.shard_map(
        ragged_dot,
        mesh=get_abstract_mesh(),
        in_specs=(P(None, None), P(None, None, None), P(None)),
        out_specs=P(None, None),
        check_vma=False,
    )(lhs, rhs, group_sizes)


def _moe_branch(block: Block, x, seg, layer: int, abl: Ablation, routing: RoutingOverride, out: dict):
    """The routed MoE plus the shared expert, as ``moe_and_shared_fused`` (dropless)."""
    mlp = block.mlp
    shared = block.shared[0]
    b, s, d = x.shape
    t = b * s
    # One device: the dispatch gathers and scatters run on replicated token arrays.
    x_flat = reshard(x.reshape(t, d), P(None, None))
    router_logits = jnp.einsum("td,de->te", x_flat, mlp.router).astype(jnp.float32)
    biased = router_logits + routing.bias_scale[layer] * mlp.router_bias.astype(jnp.float32)
    _, selected = _small_top_k(biased, TOP_K)
    selected = jnp.where(routing.sel_mask[layer][:, None], routing.sel_override[layer], selected)
    unbiased_topk = jnp.take_along_axis(router_logits, selected, axis=-1)
    raw = jnp.sqrt(jnp.maximum(jax.nn.softplus(unbiased_topk), 1e-30))
    combine = raw * (mlp.cfg.routing_renorm_sum / (jnp.sum(raw, axis=-1, keepdims=True) + 1e-9))
    # Rank of each slot by combine weight (0 = largest) for the slot ablation.
    rank = jnp.argsort(jnp.argsort(-combine, axis=-1), axis=-1)
    slot_keep = abl.slot_mult[layer][rank]
    kept = combine * slot_keep
    renormed = kept * (jnp.sum(combine, axis=-1, keepdims=True) / (jnp.sum(kept, axis=-1, keepdims=True) + 1e-9))
    used = (abl.slot_renorm * renormed + (1.0 - abl.slot_renorm) * kept) * routing.combine_mult[layer]
    routed_input = mlp.latent_norm(jnp.einsum("td,dl->tl", x_flat, mlp.w_latent_down))

    expert_sorted, token_sorted, slot_sorted, counts, starts = _dispatch(selected)
    em = mlp.expert_mlp
    # haliax's ragged_dot: the Triton grouped GEMM on GPU (XLA's ragged_dot_general lowers to a dense masked product).
    pre = _local_ragged_dot(routed_input[token_sorted], em.w_up, counts).astype(jnp.float32)
    keep = abl.neuron_mult[layer][expert_sorted]
    act = keep * jnp.square(jax.nn.relu(pre)) + (1.0 - keep) * abl.neuron_mean[layer][expert_sorted]
    y = _local_ragged_dot(act.astype(x.dtype), em.w_down, counts).astype(jnp.float32)  # [TK, R]
    per_slot = jnp.zeros((t, TOP_K, LATENT), jnp.float32).at[token_sorted, slot_sorted].set(y)
    routed_latent = jnp.einsum("tk,tko->to", used, per_slot) + routing.latent_delta[layer]
    routed = jnp.einsum("to,od->td", routed_latent.astype(x.dtype), mlp.w_latent_up, preferred_element_type=jnp.float32)

    shared_pre = jnp.einsum("td,di->ti", x_flat, shared.w_up, preferred_element_type=jnp.float32)
    s_keep = abl.shared_mult[layer][None, :]
    shared_act = s_keep * jnp.square(jax.nn.relu(shared_pre)) + (1.0 - s_keep) * abl.shared_mean[layer][None, :]
    shared_out = jnp.einsum("ti,id->td", shared_act.astype(x.dtype), shared.w_down, preferred_element_type=jnp.float32)

    routed = _mix(abl.sub_mult[layer, SUB_ROUTED], routed, abl.sub_mean[layer, SUB_ROUTED]).reshape(b, s, d)
    shared_out = _mix(abl.sub_mult[layer, SUB_SHARED], shared_out, abl.sub_mean[layer, SUB_SHARED]).reshape(b, s, d)
    routed = _bspec(routed.astype(x.dtype))
    shared_out = _bspec(shared_out.astype(x.dtype))
    out.update(
        router_logits=router_logits,
        router_bias=mlp.router_bias.astype(jnp.float32),
        selected=selected,
        combine=combine,
        used_combine=used,
        weight_rank=rank,
        routed_input=routed_input,
        expert_sorted=expert_sorted,
        token_sorted=token_sorted,
        slot_sorted=slot_sorted,
        expert_starts=starts,
        expert_pre=pre,
        expert_act=act,
        expert_counts=counts,
        slot_latent_out=per_slot,
        routed_latent=routed_latent,
        shared_pre=shared_pre,
        shared_act=shared_act,
        routed_pre_conv=routed,
        shared_pre_conv=shared_out,
    )
    return routed, shared_out


def replicate(tree):
    """One device: every weight replicated, so the per-head and per-expert einsums need no sharding rules."""
    return jax.tree.map(lambda x: reshard(x, P(*([None] * x.ndim))) if eqx.is_array(x) else x, tree)


def _final_logits(model: Transformer, hidden: jax.Array) -> jax.Array:
    return jnp.einsum("bsd,dv->bsv", hidden, model.output_proj, preferred_element_type=jnp.float32)


def soft_cap(z: jax.Array, cap: float = 10.0) -> jax.Array:
    return cap * jnp.tanh(z / cap)


def trace_forward(
    model: Transformer,
    tokens: jax.Array,
    segment_ids: jax.Array,
    abl: Ablation,
    routing: RoutingOverride | None = None,
) -> dict:
    """The forward of ``model`` (compute dtype) on ``[B, S]`` tokens, returning every intermediate: ``layers[i]`` holds
    layer ``i``'s gate weights, attention internals, MoE internals and branch outputs, and the top level the final
    gate, the normed hidden state, the capped logits' loss, top-1 and the label. ``routing`` edits the MoE routing
    (``no_override`` when None)."""
    cfg = model.config
    eps = cfg.layer_norm_eps
    b, s = tokens.shape
    if routing is None:
        routing = no_override(b * s)
    model = replicate(model)
    seg = reshard(segment_ids, P(_BATCH, None))
    long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=(seg, seg))
    lower, valid = fa4_cute_segment_bounds(long_mask, batch_size=b, seq_len=s, sliding_window=None)
    mla_mask = long_mask.with_fa4_bounds(reshard(lower, P(_BATCH, None)), reshard(valid, P(_BATCH, None)))

    embed = _gated_rms_norm(
        model.embed_norm, None, _embedding_gather(model.token_embed, tokens), cfg.gated_norm_implementation
    )
    blocks = [embed]
    layers_out = []
    for i, layer in enumerate(model.layers()):
        out: dict = {}
        w_attn, h = _gate(blocks, layer.attn_res_query_attn, eps)
        attn_in = layer._attn_input(h)
        if i in KDA_LAYERS:
            _, contrib = _kda_branch(layer.attn, attn_in, seg, cfg, abl.head_mult[i], abl.head_mean[i], abl, out)
            ve_contrib = jnp.zeros_like(contrib)
        else:
            contrib, ve_contrib = _mla_branch(
                layer.attn,
                attn_in,
                seg,
                tokens,
                mla_mask,
                abl.head_mult[i],
                abl.head_mean[i],
                abl.sub_mult[i, SUB_VE] * abl.ve_head_mult[i],
                abl.ve_mean[i],
                out,
            )
            # The branch-output SConv (MLA only) is linear and channelwise: per-head parts go through it alone.
            contrib = _heads_through(layer.sconv_attn, contrib, seg)
            ve_contrib = _heads_through(layer.sconv_attn, ve_contrib, seg)
        attn_out = jnp.sum(contrib.astype(jnp.float32), axis=2)
        attn_out = _mix(abl.sub_mult[i, SUB_ATTN], attn_out, abl.sub_mean[i, SUB_ATTN]).astype(embed.dtype)
        w_mlp, h2 = _gate([*blocks, attn_out], layer.attn_res_query_mlp, eps)
        mlp_in = layer._mlp_input(h2)
        routed, shared_out = _moe_branch(layer, mlp_in, seg, i, abl, routing, out)
        routed_sc = layer.sconv_mlp(routed, seg)
        shared_sc = layer.sconv_mlp(shared_out, seg)
        mlp_out = layer.sconv_mlp(routed + shared_out, seg)
        block_out = attn_out + mlp_out
        out.update(
            w_attn=w_attn,
            w_mlp=w_mlp,
            attn_gate_in=h,
            attn_in=attn_in,
            head_out=contrib,
            ve_out=jnp.sum(ve_contrib.astype(jnp.float32), axis=2),
            ve_head_out=ve_contrib,
            attn_out=attn_out,
            mlp_gate_in=h2,
            mlp_in=mlp_in,
            routed_out=routed_sc,
            shared_out=shared_sc,
            mlp_out=mlp_out,
            block=block_out,
        )
        layers_out.append(out)
        blocks.append(block_out)
    w_final, mixed = _gate(blocks, model.attn_res_query_final, eps)
    mixed_f = mixed.astype(jnp.float32)
    inv_rms = jax.lax.rsqrt(jnp.mean(jnp.square(mixed_f), axis=-1, keepdims=True) + eps)
    hidden = model.final_norm(mixed)
    logits = soft_cap(_final_logits(model, hidden))
    labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    logp = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.take_along_axis(logp, labels[..., None], axis=-1)[..., 0]
    top = jax.lax.top_k(logp, 5)
    return dict(
        layers=layers_out,
        embed=embed,
        blocks=blocks,
        w_final=w_final,
        mixed=mixed,
        inv_rms=inv_rms[..., 0],
        hidden=hidden,
        loss=loss,
        top_logp=top[0],
        top_ids=top[1],
        labels=labels,
        entropy=-jnp.sum(jnp.exp(logp) * logp, axis=-1),
    )


def final_gain(model: Transformer) -> jax.Array:
    return model.final_norm.gain().astype(jnp.float32)


def unembed_direction(model: Transformer) -> jax.Array:
    """``gain * W_out`` ``[D, V]`` in float32: a term ``t`` adds ``inv_rms * t @ unembed_direction`` to the logits."""
    return final_gain(model)[:, None] * model.output_proj.astype(jnp.float32)
