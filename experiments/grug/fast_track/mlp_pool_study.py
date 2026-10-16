# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analysis study: can adjacent tokens share one MLP evaluation via mean pooling?

Loads a trained Belay Ckpt 1 fast_track checkpoint, runs a Paloma sample, and for one MoE layer records the MLP input
``x_t`` exactly as the MLP sees it (AttnRes mix -> rms_mlp -> GatedNorm), the routed/shared hidden activations, the
logit-lens of every routed/shared neuron's output direction, pairwise pooling statistics, and the loss when that
layer's MLP output (before the branch-output SConv) is replaced by pooled variants.

Run as a one-GPU Iris job in the checkpoint's region; ``--smoke`` runs a random-init model on random tokens on CPU.
"""

import concurrent.futures
import dataclasses
import functools
import io
import json
import logging
import math
import time

import click
import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from einops import rearrange
from haliax import Axis
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.checkpoint import load_checkpoint
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer

from experiments.grug.fast_track import belay_ckpt1, launch
from experiments.grug.fast_track import model as model_mod
from experiments.grug.fast_track.model import Block, GrugModelConfig, Transformer
from experiments.grug.fast_track.train import _apply_qb_betas, _cast_to_compute, _to_dropless_local
from experiments.grug.moe.launch_datakit_moe_mix import _val_component

logger = logging.getLogger(__name__)

CHECKPOINT = "s3://marin-us-east-02a/marin/grug/lc1-l171-d512/2026.09.25/checkpoints/step-8821"
OUTPUT_DIR = "s3://marin-us-east-02a/tmp/ttl=7d/larry/mlppool"
PALOMA_PREFIX = "s3://marin-us-east-02a/marin/paloma"
PALOMA_TAG = "hero-bpe-v16384"
PALOMA_SETS = (
    "4chan",
    "c4_100_domains",
    "c4_en",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
)
MP_POLICY = "params=float32,compute=bfloat16,output=bfloat16"
DROPLESS_IMPLEMENTATION = "scatter"
TOP_LENS_TOKENS = 5
TOP_CONTRIBUTORS = 8
MASS_FRACTION = 0.9
REF_CHUNK = 64
LENS_CHUNK = 4096


# ---------------------------------------------------------------------------------------------------------------------
# Model config and checkpoint.


def ckpt1_model_config(size: str) -> GrugModelConfig:
    """The Belay Ckpt 1 model config at ``size``, built the way ``launch.build_h100_ladder_run`` builds it."""
    args = belay_ckpt1.recipe_args(size)
    settings = tuple(args[i + 1] for i, a in enumerate(args) if a == "--model-set")
    rung = launch._h100_ladder_rung(size)
    cfg = dataclasses.replace(launch._h100_ladder_model(rung), vocab_size=launch.V16384_VOCAB)
    cfg = launch._kma_model(cfg)
    return launch._apply_settings(cfg, launch._parse_settings(settings))


# ---------------------------------------------------------------------------------------------------------------------
# Forward hooks. The d512 forward runs the layers in an unrolled Python loop (no scan, no custom_vjp in the forward
# with attn_res_layer_backward=save), so a hook can read the traced layer index and hand back traced captures.


@dataclasses.dataclass
class PoolPlan:
    """Replace a layer's MLP (routed + shared, before the branch-output SConv) by the MLP of a pooled input.

    Position ``t`` reads ``z_t = (v[i1[t]] + v[i2[t]]) / 2``; ``v`` is the MLP input ``x`` (``pool_resid`` 0) or the
    pre-norm AttnRes mix ``h`` re-normed through rms_mlp/GatedNorm (``pool_resid`` 1). ``routed_on``/``shared_on``
    pick which part reads ``z`` (the other reads ``x``). All four are traced so one compile serves every variant."""

    layer: int
    i1: jax.Array
    i2: jax.Array
    routed_on: jax.Array
    shared_on: jax.Array
    pool_resid: jax.Array


@dataclasses.dataclass
class Hook:
    capture_layers: tuple[int, ...] = ()
    plan: PoolPlan | None = None
    captures: dict = dataclasses.field(default_factory=dict)


_HOOK: list[Hook | None] = [None]
_CURRENT_LAYER: list[int | None] = [None]
_ORIG_SUBLAYERS = model_mod._attn_res_sublayers
_ORIG_MLP_BRANCH = Block.mlp_branch


def _hooked_sublayers(*args, **kwargs):
    _CURRENT_LAYER[0] = args[6]
    return _ORIG_SUBLAYERS(*args, **kwargs)


def _gather_rows(v: jax.Array, idx: jax.Array) -> jax.Array:
    return jax.vmap(lambda row, i: row[i])(v, idx)


def _pooled_mlp_branch(self: Block, h: jax.Array, mask, plan: PoolPlan) -> tuple[jax.Array, dict]:
    """``Block.mlp_branch`` for this recipe (no sum streams, no input SConvs, one ungated shared expert), with the
    routed and shared experts reading ``PoolPlan``'s pooled input."""
    assert self.sconv_mlp_in is None and self.sconv_shared_in is None and self.sconv_shared_out is None
    assert self.shared is not None and len(self.shared) == 1 and self.shared[0].w_gate is None
    segment_ids = model_mod._sconv_segment_ids(mask)
    x = self._mlp_input(h)
    spec = model_mod._batch_spec()
    i1, i2 = reshard(plan.i1, spec), reshard(plan.i2, spec)
    zx = (0.5 * (_gather_rows(x, i1).astype(jnp.float32) + _gather_rows(x, i2).astype(jnp.float32))).astype(x.dtype)
    hbar = 0.5 * (_gather_rows(h, i1).astype(jnp.float32) + _gather_rows(h, i2).astype(jnp.float32))
    zh = self._mlp_input(hbar.astype(h.dtype))
    z = jnp.where(plan.pool_resid > 0, zh, zx)
    routed_in = jnp.where(plan.routed_on > 0, z, x)
    shared_in = jnp.where(plan.shared_on > 0, z, x)
    routed, stats = self.mlp(routed_in, segment_ids=segment_ids)
    shared = self.shared[0]
    pre = jnp.einsum("bsd,di->bsi", shared_in, shared.w_up.astype(x.dtype), out_sharding=spec)
    shared_out = jnp.einsum(
        "bsi,id->bsd", jnp.square(jax.nn.relu(pre)), shared.w_down.astype(x.dtype), out_sharding=spec
    )
    stats = {**stats, model_mod._SHARED_PRE: rearrange(pre, "b s i -> (b s) i")}
    out = routed + shared_out
    if self.sconv_mlp is not None:
        out = self.sconv_mlp(out, segment_ids)
    return out, stats


def _hooked_mlp_branch(self: Block, h: jax.Array, mask, **kwargs):
    hook, layer = _HOOK[0], _CURRENT_LAYER[0]
    if hook is None:
        return _ORIG_MLP_BRANCH(self, h, mask, **kwargs)
    if hook.plan is not None and hook.plan.layer == layer:
        out, stats = _pooled_mlp_branch(self, h, mask, hook.plan)
    else:
        out, stats = _ORIG_MLP_BRANCH(self, h, mask, **kwargs)
    if layer in hook.capture_layers:
        hook.captures[layer] = {"h": h, "x": self._mlp_input(h), "out": out}
    return out, stats


model_mod._attn_res_sublayers = _hooked_sublayers
Block.mlp_branch = _hooked_mlp_branch


def token_losses(model: Transformer, tokens: jax.Array, labels: jax.Array, segment_ids: jax.Array) -> jax.Array:
    """Per-position next-token loss ``[B, S]`` against ``labels`` (the eval head: final norm, lm_head, soft cap)."""
    mask = AttentionMask.causal().with_segment_ids(segment_ids)
    hidden, _ = model(tokens, mask=mask)
    head_in, lm_head = model._lm_head_operands(hidden, tokens)
    logits = jnp.einsum("bsd,dv->bsv", head_in, lm_head, preferred_element_type=jnp.float32)
    logits = model_mod._soft_capped(logits, model_mod._logit_cap(model.config))
    logits = reshard(logits, P(None, None, None))
    labels = reshard(labels, P(None, None))
    lse = jax.nn.logsumexp(logits, axis=-1)
    picked = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    return lse - picked


@eqx.filter_jit
def baseline_forward(model, tokens, labels, segment_ids, capture_layers: tuple[int, ...]):
    hook = Hook(capture_layers=capture_layers)
    _HOOK[0] = hook
    try:
        loss = token_losses(model, tokens, labels, segment_ids)
        mask = AttentionMask.causal().with_segment_ids(segment_ids)
        weight = jnp.ones(tokens.shape, jnp.float32)
        hook_off = Hook()
        _HOOK[0] = hook_off
        model_loss = model.next_token_loss(tokens, weight, mask=mask, reduction="none")
    finally:
        _HOOK[0] = None
    return loss, model_loss, hook.captures


@eqx.filter_jit
def pooled_forward(model, tokens, labels, segment_ids, layer: int, i1, i2, routed_on, shared_on, pool_resid):
    _HOOK[0] = Hook(plan=PoolPlan(layer, i1, i2, routed_on, shared_on, pool_resid))
    try:
        return token_losses(model, tokens, labels, segment_ids)
    finally:
        _HOOK[0] = None


# ---------------------------------------------------------------------------------------------------------------------
# Pooling index plans (per row, document-relative; never pool across a document boundary).


def doc_starts_of(segment_ids: np.ndarray) -> np.ndarray:
    starts = np.zeros(segment_ids.shape, bool)
    starts[:, 0] = True
    starts[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    return starts


def pool_indices(doc_start: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """``(i1, i2)`` ``[B, S]`` for ``variant``:

    - ``identity``: ``(t, t)``.
    - ``causal_pool`` (a): doc-relative pairs ``(2k-1, 2k)`` for k >= 1 serve positions ``2k`` and ``2k+1``; group 0
      is the document's first token alone, serving doc positions 0 and 1.
    - ``lag1`` (b): ``(t-1, t-1)``; a document's first token uses itself.
    - ``pool_self`` (c): ``(t-1, t)``; a document's first token uses itself.
    - ``oracle`` (d, non-causal): doc-relative pairs ``(2k, 2k+1)`` share their mean; an odd document's last token
      is alone.
    """
    b, s = doc_start.shape
    i1 = np.zeros((b, s), np.int32)
    i2 = np.zeros((b, s), np.int32)
    t = np.arange(s)
    for row in range(b):
        start = np.maximum.accumulate(np.where(doc_start[row], t, 0))
        ends = np.empty(s, np.int64)
        nxt = s
        for pos in range(s - 1, -1, -1):
            ends[pos] = nxt - 1
            if doc_start[row, pos]:
                nxt = pos
        r = t - start
        if variant == "identity":
            a, c = t, t
        elif variant == "causal_pool":
            g = r // 2
            a = np.where(g == 0, start, start + 2 * g - 1)
            c = np.where(g == 0, start, start + 2 * g)
        elif variant == "lag1":
            a = np.where(r == 0, t, t - 1)
            c = a
        elif variant == "pool_self":
            a, c = np.where(r == 0, t, t - 1), t
        elif variant == "oracle":
            q = r // 2
            a = start + 2 * q
            c = np.minimum(a + 1, ends)
        else:
            raise ValueError(variant)
        assert np.all(a >= start) and np.all(c >= start) and np.all(a <= ends) and np.all(c <= ends)
        i1[row], i2[row] = a, c
    return i1, i2


# (name, layer-independent index variant, routed_on, shared_on, pool_resid)
LAYER3_VARIANTS = (
    ("ref_identity", "identity", 1, 1, 0),
    ("a_causal_pool", "causal_pool", 1, 1, 0),
    ("a_causal_pool_shared_only", "causal_pool", 0, 1, 0),
    ("a_causal_pool_routed_only", "causal_pool", 1, 0, 0),
    ("a_causal_pool_resid", "causal_pool", 1, 1, 1),
    ("b_lag1", "lag1", 1, 1, 0),
    ("c_pool_self", "pool_self", 1, 1, 0),
    ("c_pool_self_resid", "pool_self", 1, 1, 1),
    ("d_oracle_noncausal", "oracle", 1, 1, 0),
)
CONTRAST_VARIANTS = (
    ("ref_identity", "identity", 1, 1, 0),
    ("a_causal_pool", "causal_pool", 1, 1, 0),
    ("b_lag1", "lag1", 1, 1, 0),
    ("c_pool_self", "pool_self", 1, 1, 0),
    ("d_oracle_noncausal", "oracle", 1, 1, 0),
)
LEAK_VARIANTS = ("ref_identity", "a_causal_pool", "b_lag1", "c_pool_self", "d_oracle_noncausal")


# ---------------------------------------------------------------------------------------------------------------------
# Reference MLP (float32, plain jnp, no mesh) for the per-token statistics.


@dataclasses.dataclass(frozen=True)
class RefMlp:
    router: np.ndarray  # [D, E]
    router_bias: np.ndarray  # [E]
    w_latent_down: np.ndarray  # [D, R]
    latent_gain: np.ndarray  # [R]
    latent_eps: float
    w_up: np.ndarray  # [E, R, I]
    w_down: np.ndarray  # [E, I, R]
    w_latent_up: np.ndarray  # [R, D]
    shared_up: np.ndarray  # [D, J]
    shared_down: np.ndarray  # [J, D]
    sconv_w0: np.ndarray  # [D], the branch-output SConv's lag-0 tap
    sconv_weight: np.ndarray  # [W, D]
    k: int
    renorm_sum: float


def ref_mlp_of(block: Block, cfg: GrugModelConfig) -> RefMlp:
    mlp = block.mlp
    assert mlp.latent_out_norm is None and mlp.latent_mix_out_gate is None and mlp.sconv_latent_out is None
    assert mlp.latent_mix_in_gate is None and mlp.sconv_latent_in is None and mlp.expert_mlp.w_gate is None
    assert cfg.router_combine == model_mod.RouterCombine.SQRT_SOFTPLUS_RENORM and not cfg.num_null_experts
    assert block.sconv_mlp is not None and block.shared is not None and len(block.shared) == 1

    def host(x):
        return np.asarray(jax.device_get(x), np.float32)

    sconv = host(block.sconv_mlp.weight)
    return RefMlp(
        router=host(mlp.router),
        router_bias=host(mlp.router_bias),
        w_latent_down=host(mlp.w_latent_down),
        latent_gain=host(mlp.latent_norm.gain()),
        latent_eps=float(mlp.latent_norm.eps),
        w_up=host(mlp.expert_mlp.w_up),
        w_down=host(mlp.expert_mlp.w_down),
        w_latent_up=host(mlp.w_latent_up),
        shared_up=host(block.shared[0].w_up),
        shared_down=host(block.shared[0].w_down),
        sconv_w0=sconv[0],
        sconv_weight=sconv,
        k=cfg.num_experts_per_token,
        renorm_sum=float(cfg.routing_renorm_sum),
    )


def ref_forward(p: dict, x: jax.Array, k: int, renorm_sum: float, eps: float) -> dict:
    """``x [N, D]`` -> routing, hidden activations and outputs of the routed and shared experts (float32)."""
    logits = x @ p["router"]
    _, sel = jax.lax.top_k(logits + p["router_bias"], k)
    gate = jnp.sqrt(jnp.maximum(jax.nn.softplus(jnp.take_along_axis(logits, sel, axis=-1)), 1e-30))
    w = gate * (renorm_sum / (jnp.sum(gate, axis=-1, keepdims=True) + 1e-9))
    lat = x @ p["w_latent_down"]
    lat = lat * jax.lax.rsqrt(jnp.mean(jnp.square(lat), axis=-1, keepdims=True) + eps) * p["latent_gain"]
    n = x.shape[0]
    chunks = n // REF_CHUNK

    def chunk(args):
        lat_c, sel_c, w_c = args
        pre = jnp.einsum("cr,ckri->cki", lat_c, p["w_up"][sel_c])
        hr = jnp.square(jax.nn.relu(pre))
        out = jnp.einsum("cki,ckio,ck->co", hr, p["w_down"][sel_c], w_c)
        return hr, out

    hr, out_lat = jax.lax.map(
        chunk,
        (
            lat.reshape(chunks, REF_CHUNK, -1),
            sel.reshape(chunks, REF_CHUNK, k),
            w.reshape(chunks, REF_CHUNK, k),
        ),
    )
    hr = hr.reshape(n, k, -1)
    yr = out_lat.reshape(n, -1) @ p["w_latent_up"]
    hs = jnp.square(jax.nn.relu(x @ p["shared_up"]))
    ys = hs @ p["shared_down"]
    return {"sel": sel, "w": w, "hr": hr, "hs": hs, "yr": yr, "ys": ys}


def _mass_set(a: jax.Array, frac: float) -> jax.Array:
    """Units (last axis) that carry ``frac`` of the activation mass: the largest ones until the cumulative sum
    reaches ``frac`` of the total."""
    order = jnp.argsort(-a, axis=-1)
    sorted_a = jnp.take_along_axis(a, order, axis=-1)
    before = jnp.cumsum(sorted_a, axis=-1) - sorted_a
    keep_sorted = (before < frac * jnp.sum(a, axis=-1, keepdims=True)) & (sorted_a > 0)
    inv = jnp.argsort(order, axis=-1)
    return jnp.take_along_axis(keep_sorted, inv, axis=-1)


def _jaccard(a: jax.Array, b: jax.Array) -> jax.Array:
    inter = jnp.sum(a & b, axis=-1).astype(jnp.float32)
    union = jnp.sum(a | b, axis=-1).astype(jnp.float32)
    return jnp.where(union > 0, inter / jnp.maximum(union, 1.0), jnp.nan)


def _routed_jaccard(sel_p, set_p, sel_q, set_q) -> jax.Array:
    """Mean over P's experts that Q also selected of the Jaccard of their unit sets ``[N, K, I]``; NaN if none."""
    match = sel_p[:, :, None] == sel_q[:, None, :]  # [N, K, K]
    has = jnp.any(match, axis=-1)
    idx = jnp.argmax(match, axis=-1)
    set_q_m = jnp.take_along_axis(set_q, idx[..., None], axis=1)
    jac = _jaccard(set_p, set_q_m)  # [N, K]
    valid = has & ~jnp.isnan(jac)
    total = jnp.sum(jnp.where(valid, jac, 0.0), axis=-1)
    count = jnp.sum(valid, axis=-1)
    return jnp.where(count > 0, total / jnp.maximum(count, 1), jnp.nan)


def _overlap(sel_a, sel_b) -> jax.Array:
    return jnp.sum(jnp.any(sel_a[:, :, None] == sel_b[:, None, :], axis=-1), axis=-1)


def _cos(a, b):
    return jnp.sum(a * b, axis=-1) / (jnp.linalg.norm(a, axis=-1) * jnp.linalg.norm(b, axis=-1) + 1e-30)


def _rel(a, ref):
    return jnp.linalg.norm(a - ref, axis=-1) / (jnp.linalg.norm(ref, axis=-1) + 1e-30)


def _sets(r: dict) -> dict:
    return {
        "fire_r": r["hr"] > 0,
        "fire_s": r["hs"] > 0,
        "mass_r": _mass_set(r["hr"], MASS_FRACTION),
        "mass_s": _mass_set(r["hs"], MASS_FRACTION),
    }


def _take(r: dict, idx: jax.Array) -> dict:
    return {k: v[idx] for k, v in r.items()}


def pair_metrics(pooled: dict, q1: dict, q2: dict) -> dict:
    """Pooled result ``P`` against the two inputs' own results ``Q1`` (token t) and ``Q2`` (its partner)."""
    sp, s1, s2 = _sets(pooled), _sets(q1), _sets(q2)
    yp, y1, y2 = pooled["yr"] + pooled["ys"], q1["yr"] + q1["ys"], q2["yr"] + q2["ys"]
    ymean = 0.5 * (y1 + y2)
    out = {
        "ov_p_t": _overlap(pooled["sel"], q1["sel"]),
        "ov_p_prev": _overlap(pooled["sel"], q2["sel"]),
        "ov_p_union": jnp.sum(
            jnp.any(pooled["sel"][:, :, None] == q1["sel"][:, None, :], axis=-1)
            | jnp.any(pooled["sel"][:, :, None] == q2["sel"][:, None, :], axis=-1),
            axis=-1,
        ),
        "ov_t_prev": _overlap(q1["sel"], q2["sel"]),
        "cos_p_t": _cos(yp, y1),
        "cos_p_prev": _cos(yp, y2),
        "cos_p_mean": _cos(yp, ymean),
        "cos_t_prev": _cos(y1, y2),
        "rel_p_t": _rel(yp, y1),
        "rel_p_prev": _rel(yp, y2),
        "rel_p_mean": _rel(yp, ymean),
        "rel_prev_t": _rel(y2, y1),
        "cos_shared_p_t": _cos(pooled["ys"], q1["ys"]),
        "cos_routed_p_t": _cos(pooled["yr"], q1["yr"]),
    }
    for kind in ("fire", "mass"):
        out[f"jac_shared_{kind}_p_t"] = _jaccard(sp[f"{kind}_s"], s1[f"{kind}_s"])
        out[f"jac_shared_{kind}_p_prev"] = _jaccard(sp[f"{kind}_s"], s2[f"{kind}_s"])
        out[f"jac_shared_{kind}_t_prev"] = _jaccard(s1[f"{kind}_s"], s2[f"{kind}_s"])
        out[f"jac_routed_{kind}_p_t"] = _routed_jaccard(pooled["sel"], sp[f"{kind}_r"], q1["sel"], s1[f"{kind}_r"])
        out[f"jac_routed_{kind}_p_prev"] = _routed_jaccard(pooled["sel"], sp[f"{kind}_r"], q2["sel"], s2[f"{kind}_r"])
        out[f"jac_routed_{kind}_t_prev"] = _routed_jaccard(q1["sel"], s1[f"{kind}_r"], q2["sel"], s2[f"{kind}_r"])
    return out


def _norm_input(block_norms: dict, h: jax.Array, eps_attn: float) -> jax.Array:
    """rms_mlp then GatedNorm (``Block._mlp_input``) in float32."""
    x = h * jax.lax.rsqrt(jnp.mean(jnp.square(h), axis=-1, keepdims=True) + eps_attn) * block_norms["rms_gain"]
    if "gn_down" in block_norms:
        gate = jax.nn.sigmoid(jax.nn.silu(x @ block_norms["gn_down"]) @ block_norms["gn_up"])
        x = x * gate
    return x


def sequence_analysis(p, norms, unit_norms, x, h, prev, rand, k, renorm_sum, eps, norm_eps):
    """Per-token statistics of one sequence ``x, h [S, D]``; ``prev``/``rand`` are each token's partner indices."""
    run = functools.partial(ref_forward, k=k, renorm_sum=renorm_sum, eps=eps)
    base = run(p, x)
    xbar = 0.5 * (x + x[prev])
    hbar_normed = _norm_input(norms, 0.5 * (h + h[prev]), norm_eps)
    xrand = 0.5 * (x + x[rand])
    pooled = run(p, xbar)
    resid = run(p, hbar_normed)
    rpool = run(p, xrand)
    q_prev, q_rand = _take(base, prev), _take(base, rand)
    out = {}
    for name, res, partner in (("adj", pooled, q_prev), ("resid", resid, q_prev), ("rand", rpool, q_rand)):
        for key, value in pair_metrics(res, base, partner).items():
            out[f"{name}/{key}"] = value
    # The re-normed residual mean against the plain input mean.
    out["resid/cos_x_vs_xbar_in"] = _cos(hbar_normed, xbar)
    out["adj/cos_in_t_prev"] = _cos(x, x[prev])
    out["adj/norm_ratio_xbar_x"] = jnp.linalg.norm(xbar, axis=-1) / (jnp.linalg.norm(x, axis=-1) + 1e-30)
    sets = _sets(base)
    out["tok/fire_frac_shared"] = jnp.mean(sets["fire_s"], axis=-1)
    out["tok/fire_frac_routed"] = jnp.mean(sets["fire_r"], axis=(-1, -2))
    out["tok/mass_size_shared"] = jnp.sum(sets["mass_s"], axis=-1)
    out["tok/mass_size_routed"] = jnp.mean(jnp.sum(sets["mass_r"], axis=-1), axis=-1)
    out["tok/sel"] = base["sel"]
    out["tok/w"] = base["w"]
    # Top contributors by (combine weight x) activation x output-direction norm; routed unit id e * I + j, shared
    # unit id E * I + j.
    num_e, num_i = p["w_up"].shape[0], p["w_up"].shape[2]
    routed_norms = unit_norms[: num_e * num_i].reshape(num_e, num_i)[base["sel"]]  # [S, K, I]
    contrib_r = base["w"][..., None] * base["hr"] * routed_norms
    contrib_s = base["hs"] * unit_norms[num_e * num_i :]
    ids_r = base["sel"][..., None] * num_i + jnp.arange(num_i)
    contrib = jnp.concatenate([contrib_r.reshape(x.shape[0], -1), contrib_s], axis=-1)
    ids = jnp.concatenate(
        [ids_r.reshape(x.shape[0], -1), jnp.broadcast_to(num_e * num_i + jnp.arange(num_i), contrib_s.shape)], axis=-1
    )
    top_v, top_pos = jax.lax.top_k(contrib, TOP_CONTRIBUTORS)
    out["tok/top_ids"] = jnp.take_along_axis(ids, top_pos, axis=-1)
    out["tok/top_vals"] = top_v
    out["tok/contrib_total"] = jnp.sum(contrib, axis=-1)
    out["tok/yr_norm"] = jnp.linalg.norm(base["yr"], axis=-1)
    out["tok/ys_norm"] = jnp.linalg.norm(base["ys"], axis=-1)
    out["tok/y_full"] = base["yr"] + base["ys"]
    return out


_sequence_analysis_jit = jax.jit(sequence_analysis, static_argnames=("k", "renorm_sum", "eps", "norm_eps"))


def unit_directions(ref: RefMlp) -> np.ndarray:
    """Every routed unit's (``w_down[e, j] @ w_latent_up``) and shared unit's (``shared_down[j]``) model-space output
    direction ``[E * I + J, D]``. The routed combine is linear here (no latent_out_norm), so these are exact."""
    routed = np.einsum("eir,rd->eid", ref.w_down, ref.w_latent_up).reshape(-1, ref.w_latent_up.shape[1])
    return np.concatenate([routed, ref.shared_down], axis=0)


@jax.jit
def _lens_chunk(dirs, scale, lm_head):
    logits = (dirs * scale) @ lm_head
    _, top = jax.lax.top_k(logits, TOP_LENS_TOKENS)
    return top


def logit_lens(dirs: np.ndarray, scale: np.ndarray, lm_head: np.ndarray) -> np.ndarray:
    lm = jnp.asarray(lm_head)
    sc = jnp.asarray(scale)
    out = []
    for start in range(0, dirs.shape[0], LENS_CHUNK):
        chunk = dirs[start : start + LENS_CHUNK]
        pad = LENS_CHUNK - chunk.shape[0]
        chunk = np.pad(chunk, ((0, pad), (0, 0)))
        out.append(np.asarray(_lens_chunk(jnp.asarray(chunk), sc, lm))[: LENS_CHUNK - pad])
    return np.concatenate(out)


# ---------------------------------------------------------------------------------------------------------------------
# Data.


def paloma_sample(num_sequences: int, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """``num_sequences`` held-out sequences round-robin over the Paloma sets in name order (index 0 of every set,
    then index 1, ...), with segment ids and the eval loss weights."""
    components = {
        "train": DatasetComponent(
            source=None,
            cache_dir=launch.V16384_CACHE_DIR,
            format=TextLmDatasetFormat(),
            tags=["train"],
            flat_cache=True,
        )
    }
    for name in PALOMA_SETS:
        components[f"paloma/{name}"] = _val_component(
            f"{PALOMA_PREFIX}/{name}-{PALOMA_TAG}/{launch.PALOMA_DETOK_VERSION}"
        )
    config = LmDataConfig(
        tokenizer=launch.V16384_TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights={"train": 1.0, **{n: 0.0 for n in components if n != "train"}},
        auto_build_caches=False,
    )
    sets = config.validation_sets(Axis("position", seq_len))
    names = sorted(sets)
    datasets = {name: sets[name].as_sync_dataset() for name in names}
    lengths = {name: len(datasets[name]) for name in names}
    picks = [(n, i) for i in range(max(lengths.values())) for n in names if i < lengths[n]][:num_sequences]
    fetched = {}
    for name in names:
        indices = [i for n, i in picks if n == name]
        if indices:
            examples = [grug_lm_example_from_named(ex) for ex in datasets[name].get_batch(indices)]
            fetched.update({(name, i): ex for i, ex in zip(indices, examples, strict=True)})
    tokens = np.stack([np.asarray(fetched[p].tokens, np.int32) for p in picks])
    segs = np.stack(
        [
            np.zeros(seq_len, np.int32) if (s := fetched[p].attn_mask.segment_ids) is None else np.asarray(s[0])
            for p in picks
        ]
    ).astype(np.int32)
    weights = np.stack([np.asarray(fetched[p].loss_weight, np.float32) for p in picks])
    return tokens, segs, weights, picks


# ---------------------------------------------------------------------------------------------------------------------
# Main.


def _ci95(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, float("nan")
    # Student t 0.975 quantiles for small n; normal beyond.
    t_table = {8: 2.365, 16: 2.131, 32: 2.040}
    t = t_table.get(n, 1.96 if n > 32 else 2.2)
    return mean, float(t * np.std(values, ddof=1) / math.sqrt(n))


def _write(path: str, data: bytes) -> None:
    with fsspec.open(path, "wb") as f:
        f.write(data)


@click.command()
@click.option("--smoke", is_flag=True, help="Random-init model on random tokens (local CPU check).")
@click.option("--num-sequences", default=32, show_default=True)
@click.option("--seq-len", default=4096, show_default=True)
@click.option("--batch", default=4, show_default=True)
@click.option("--layers", default="3,1", show_default=True, help="Layers studied; the first gets every variant.")
@click.option("--output-dir", default=OUTPUT_DIR, show_default=True)
@click.option("--tag", default="d512_step8821", show_default=True)
@click.option("--skip-interventions", is_flag=True, help="Only the baseline forward and the per-token statistics.")
def main(
    smoke: bool,
    num_sequences: int,
    seq_len: int,
    batch: int,
    layers: str,
    output_dir: str,
    tag: str,
    skip_interventions: bool,
) -> None:
    logging.basicConfig(level=logging.INFO)
    started = time.time()
    study_layers = tuple(int(v) for v in layers.split(","))
    cfg = ckpt1_model_config("d512")
    cfg = dataclasses.replace(cfg, max_seq_len=max(seq_len, 128) if smoke else cfg.max_seq_len)
    logger.info("devices: %s", jax.devices())
    mp = jmp.get_policy(MP_POLICY)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):
        params = jax.jit(lambda key: mp.cast_to_param(Transformer.init(cfg, key=key)))(jax.random.PRNGKey(0))
        qb = jnp.zeros((cfg.num_layers, cfg.qb_num_experts))
        step = -1
        if not smoke:
            loaded = load_checkpoint(
                {"step": jnp.zeros((), jnp.int32), "params": params, "pending_qb_betas": qb},
                CHECKPOINT,
                axis_mapping=None,
                mesh=mesh,
                allow_partial=False,
            )
            params, qb, step = loaded["params"], loaded["pending_qb_betas"], int(loaded["step"])
            logger.info("loaded %s at step %d", CHECKPOINT, step)
        params = _apply_qb_betas(params, qb)
        model = _to_dropless_local(_cast_to_compute(mp, params), implementation=DROPLESS_IMPLEMENTATION)

    if smoke:
        rng = np.random.default_rng(0)
        tokens = rng.integers(1, cfg.vocab_size, (num_sequences, seq_len)).astype(np.int32)
        segs = np.zeros((num_sequences, seq_len), np.int32)
        segs[:, seq_len // 3 :] = 1
        segs[:, seq_len // 3 + 17 :] = 2
        weights = np.ones((num_sequences, seq_len), np.float32)
        weights[:, -1] = 0
        picks = [("random", i) for i in range(num_sequences)]
        vocab_tokens = [str(i) for i in range(cfg.vocab_size)]
        vocab_text = vocab_tokens
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            tokens, segs, weights, picks = pool.submit(paloma_sample, num_sequences, seq_len).result()
        tok = load_tokenizer(launch.V16384_TOKENIZER)
        vocab_tokens = [str(t) for t in tok.convert_ids_to_tokens(list(range(cfg.vocab_size)))]
        vocab_text = [tok.decode([i]) for i in range(cfg.vocab_size)]
    logger.info("sample: %s tokens %s", picks, tokens.shape)
    labels = np.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    doc_start = doc_starts_of(segs)

    def batches():
        for s in range(0, num_sequences, batch):
            yield s, slice(s, s + batch)

    # Baseline forward with captures.
    base_loss = np.zeros(tokens.shape, np.float32)
    model_loss = np.zeros(tokens.shape, np.float32)
    caps = {
        layer: {k: np.zeros((*tokens.shape, cfg.hidden_dim), np.float32) for k in ("h", "x", "out")}
        for layer in study_layers
    }
    with set_mesh(mesh):
        for _, sl in batches():
            loss, mloss, captured = baseline_forward(
                model, jnp.asarray(tokens[sl]), jnp.asarray(labels[sl]), jnp.asarray(segs[sl]), study_layers
            )
            base_loss[sl] = np.asarray(loss)
            model_loss[sl] = np.asarray(mloss)
            for layer in study_layers:
                for k in ("h", "x", "out"):
                    caps[layer][k][sl] = np.asarray(captured[layer][k].astype(jnp.float32))
    logger.info("baseline done (%.0fs)", time.time() - started)

    w = weights
    results: dict = {
        "checkpoint": None if smoke else CHECKPOINT,
        "step": step,
        "picks": [list(p) for p in picks],
        "num_tokens_scored": float(w.sum()),
        "baseline_mean_loss": float((base_loss * w).sum() / w.sum()),
        "baseline_vs_model_next_token_loss_max_abs": float(np.max(np.abs((base_loss - model_loss) * w))),
        "num_doc_starts": int(doc_start.sum()),
        "variants": {},
        "leakage": {},
        "config": {
            "hidden_dim": cfg.hidden_dim,
            "latent_dim": cfg.latent_dim,
            "intermediate_dim": cfg.intermediate_dim,
            "shared_expert_intermediate_dim": cfg.shared_expert_intermediate_dim,
            "num_experts": cfg.num_experts,
            "k": cfg.num_experts_per_token,
            "layer_kinds": list(model.layer_kinds()),
            "routing_renorm_sum": cfg.routing_renorm_sum,
            "logit_soft_cap": cfg.logit_soft_cap,
            "sublayer_gated_norm": cfg.sublayer_gated_norm,
        },
    }
    per_seq_base = (base_loss * w).sum(1) / w.sum(1)
    arrays: dict[str, np.ndarray] = {
        "tokens": tokens,
        "segment_ids": segs,
        "loss_weight": w.astype(np.float16),
        "base_loss": base_loss.astype(np.float16),
    }

    # Loss interventions.
    for layer in () if skip_interventions else study_layers:
        variants = LAYER3_VARIANTS if layer == study_layers[0] else CONTRAST_VARIANTS
        for name, index_variant, routed_on, shared_on, pool_resid in variants:
            i1, i2 = pool_indices(doc_start, index_variant)
            losses = np.zeros(tokens.shape, np.float32)
            with set_mesh(mesh):
                for _, sl in batches():
                    losses[sl] = np.asarray(
                        pooled_forward(
                            model,
                            jnp.asarray(tokens[sl]),
                            jnp.asarray(labels[sl]),
                            jnp.asarray(segs[sl]),
                            layer,
                            jnp.asarray(i1[sl]),
                            jnp.asarray(i2[sl]),
                            jnp.asarray(routed_on, jnp.int32),
                            jnp.asarray(shared_on, jnp.int32),
                            jnp.asarray(pool_resid, jnp.int32),
                        )
                    )
            per_seq = (losses * w).sum(1) / w.sum(1)
            delta = per_seq - per_seq_base
            mean, half = _ci95(delta)
            results["variants"][f"L{layer}/{name}"] = {
                "mean_loss": float((losses * w).sum() / w.sum()),
                "delta_token_weighted": float(((losses - base_loss) * w).sum() / w.sum()),
                "delta_seq_mean": mean,
                "delta_seq_ci95": half,
                "delta_per_seq": delta.tolist(),
            }
            arrays[f"loss/L{layer}/{name}"] = (losses - base_loss).astype(np.float16)
            logger.info("L%d %s: delta %.5f +- %.5f (%.0fs)", layer, name, mean, half, time.time() - started)

    # Leakage: perturb tokens from position p on (keeping labels and segments), compare earlier positions' losses.
    rng = np.random.default_rng(1)
    leak_rows = slice(0, batch)
    for layer in () if skip_interventions else study_layers[:1]:
        for cut in (seq_len // 2, seq_len // 2 + 1):
            pert = tokens[leak_rows].copy()
            pert[:, cut:] = rng.integers(1, cfg.vocab_size, pert[:, cut:].shape)
            for name in LEAK_VARIANTS:
                spec = next(v for v in LAYER3_VARIANTS if v[0] == name)
                i1, i2 = pool_indices(doc_start[leak_rows], spec[1])
                outs = []
                with set_mesh(mesh):
                    for toks in (tokens[leak_rows], pert):
                        outs.append(
                            np.asarray(
                                pooled_forward(
                                    model,
                                    jnp.asarray(toks),
                                    jnp.asarray(labels[leak_rows]),
                                    jnp.asarray(segs[leak_rows]),
                                    layer,
                                    jnp.asarray(i1),
                                    jnp.asarray(i2),
                                    jnp.asarray(spec[2], jnp.int32),
                                    jnp.asarray(spec[3], jnp.int32),
                                    jnp.asarray(spec[4], jnp.int32),
                                )
                            )
                        )
                diff = np.abs(outs[1] - outs[0])
                results["leakage"][f"L{layer}/{name}/cut{cut}"] = {
                    "max_abs_before_cut_minus_1": float(diff[:, : cut - 1].max()),
                    "max_abs_at_cut_minus_1": float(diff[:, cut - 1].max()),
                    "mean_abs_after_cut": float(diff[:, cut:].mean()),
                }
    logger.info("leakage done (%.0fs)", time.time() - started)

    # Per-token statistics, lm_head alignment.
    with set_mesh(mesh):
        lm_head = np.asarray(jax.device_get(params.output_proj), np.float32)
        final_gain = np.asarray(jax.device_get(params.final_norm.gain()), np.float32)
        host_layers = params.layers()
        refs = {}
        for layer in study_layers:
            block = host_layers[layer]
            norms = {"rms_gain": np.asarray(jax.device_get(block.rms_mlp.gain()), np.float32)}
            if block.mlp_gated_norm is not None:
                norms["gn_down"] = np.asarray(jax.device_get(block.mlp_gated_norm.w_down), np.float32)
                norms["gn_up"] = np.asarray(jax.device_get(block.mlp_gated_norm.w_up), np.float32)
            refs[layer] = (ref_mlp_of(block, cfg), norms, float(block.rms_mlp.eps))
    rng = np.random.default_rng(2)
    for layer in study_layers:
        ref, norms, norm_eps = refs[layer]
        dirs = unit_directions(ref)
        unit_norms = np.linalg.norm(dirs * ref.sconv_w0, axis=-1)
        lens = logit_lens(dirs, ref.sconv_w0 * final_gain, lm_head)
        arrays[f"L{layer}/lens_top"] = lens.astype(np.int16)
        arrays[f"L{layer}/unit_norm"] = unit_norms.astype(np.float16)
        p = {
            k: jnp.asarray(getattr(ref, k))
            for k in (
                "router",
                "router_bias",
                "w_latent_down",
                "latent_gain",
                "w_up",
                "w_down",
                "w_latent_up",
                "shared_up",
                "shared_down",
            )
        }
        per_tok: dict[str, list[np.ndarray]] = {}
        out_cos = []
        for row in range(num_sequences):
            t = np.arange(seq_len)
            prev = np.where(doc_start[row], t, t - 1).astype(np.int32)
            rand = rng.integers(0, seq_len - 2, seq_len)
            rand = np.where(rand >= t - 1, rand + 3, rand)  # skip t-1, t, t+1
            rand = np.clip(rand, 0, seq_len - 1).astype(np.int32)
            stats = _sequence_analysis_jit(
                p,
                {k: jnp.asarray(v) for k, v in norms.items()},
                jnp.asarray(unit_norms),
                jnp.asarray(caps[layer]["x"][row]),
                jnp.asarray(caps[layer]["h"][row]),
                jnp.asarray(prev),
                jnp.asarray(rand),
                k=ref.k,
                renorm_sum=ref.renorm_sum,
                eps=ref.latent_eps,
                norm_eps=norm_eps,
            )
            stats = {k: np.asarray(v) for k, v in stats.items()}
            # The reference MLP through the branch-output SConv against the forward's actual MLP branch output.
            y = stats.pop("tok/y_full")
            conv = np.zeros_like(y)
            for lag in range(ref.sconv_weight.shape[0]):
                shifted = np.zeros_like(y)
                shifted[lag:] = y[: seq_len - lag]
                same_doc = np.zeros(seq_len, bool)
                same_doc[lag:] = segs[row, lag:] == segs[row, : seq_len - lag]
                conv += ref.sconv_weight[lag] * shifted * same_doc[:, None]
            actual = caps[layer]["out"][row]
            out_cos.append(
                np.sum(conv * actual, -1) / (np.linalg.norm(conv, axis=-1) * np.linalg.norm(actual, axis=-1) + 1e-30)
            )
            stats["tok/rand_partner"] = rand
            for k, v in stats.items():
                per_tok.setdefault(k, []).append(v)
        for k, v in per_tok.items():
            arr = np.stack(v)
            if k in ("tok/sel", "tok/top_ids", "tok/rand_partner"):
                arrays[f"L{layer}/{k}"] = arr.astype(np.int32)
            elif k.startswith(("adj/ov", "resid/ov", "rand/ov")):
                arrays[f"L{layer}/{k}"] = arr.astype(np.int8)
            else:
                arrays[f"L{layer}/{k}"] = arr.astype(np.float16)
        out_cos = np.stack(out_cos)
        results[f"L{layer}/ref_vs_forward_mlp_out_cos"] = {
            "median": float(np.median(out_cos)),
            "p01": float(np.quantile(out_cos, 0.01)),
            "min": float(out_cos.min()),
        }
        results[f"L{layer}/expert_width"] = int(ref.w_up.shape[2])
        results[f"L{layer}/latent_out_norm"] = "none (routed combine and w_latent_up are linear)"
        logger.info("L%d analysis done (%.0fs)", layer, time.time() - started)

    results["vocab_tokens"] = vocab_tokens
    results["vocab_text"] = vocab_text
    results["elapsed"] = time.time() - started
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write(f"{output_dir}/{tag}/arrays.npz", buffer.getvalue())
    _write(f"{output_dir}/{tag}/results.json", json.dumps(results).encode())
    summary = {k: v for k, v in results.items() if k not in ("vocab_tokens", "vocab_text")}
    logger.info("RESULTS %s", json.dumps({k: v for k, v in summary.items() if k != "picks"})[:20000])
    logger.info("wrote %s/%s", output_dir, tag)


if __name__ == "__main__":
    main()
