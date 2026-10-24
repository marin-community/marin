# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron exclusion study, magnitude-weighted pass: with ReLU^2 a barely positive unit contributes almost nothing,
so this pass weighs each unit by its contribution ``c_i = act_i * ||W_down[i]||`` (the norm of its output in the
expert's output space) instead of treating it as on/off.

Passes (routing frozen to the unpatched selections wherever units are patched):

1. Calibration (the stats draw): per unit, a log-spaced histogram of its activation over its expert's routed tokens
   (the shared expert's over all tokens) -> each unit's 90th percentile.
2. Concentration (held-out draw): per routed token, how many units carry 50/80/90/95/99% of ``sum c``; the
   participation ratio ``(sum c)^2 / sum c^2``; the top-1 and top-4 shares; and the relative error of the expert
   output rebuilt from the top-k units by ``c``. The same for the shared expert. Strong-firing co-occurrence for two
   definitions -- "q90": a unit above its own 90th percentile; "c5": ``c_i >= 5%`` of the token's ``sum c`` -- plus
   ``P(j active | i strong)`` for q90.
3. Host: pair census of the strong definitions against the shuffle null (hypergeometric, pairs whose expected
   co-occurrence is at least ``MIN_EXPECTED``), and a recursive spectral bisection per expert on each definition's
   co-occurrence correlation.
4. Group pass: per token, the share of ``sum c`` in the heavier half / quarter of those orders and of a random order.
5. Sweep: keep the top-k units by ``c`` (oracle; needs the full up-projection), the top-k by a low-rank sketch of
   the up-projection (``exclusion_skip.lowrank``; exact activations for the kept units) and 2:4 by ``c`` in the
   original and the greedy WTA order, in all layers or one.

Writes ``excl_mass.npz`` (histograms, census, example matrices) and ``excl_mass.json`` (sweep).
"""

import argparse
import dataclasses
import logging
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import common, forward
from experiments.grug.fast_track.interp.exclusion_skip import (
    CALIB_SEED,
    EVAL_SEED,
    LAYER_FIELDS,
    base_selection,
    bisection_order,
    lowrank,
)
from experiments.grug.fast_track.interp.exclusion_stats import CAPACITY, padded_rows, scatter_padded
from experiments.grug.fast_track.interp.ngram import load_npz

logger = logging.getLogger(__name__)

BATCH = 4
L, E, I, R, D = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.LATENT, forward.HIDDEN
MAX_RANK = 64
QUANTILES = (0.5, 0.8, 0.9, 0.95, 0.99)
RECON_K = (1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256)
ACT_BINS = np.concatenate([[0.0], np.logspace(-6, 3, 91)])  # bin 0: exactly zero; then log-spaced
STRONG_SHARE = 0.05
STRONG_QUANTILE = 0.9
MIN_EXPECTED = 20.0
LIFT_THRESHOLDS = (0.5, 0.2, 0.1, 0.05)
LOG_LIFT_BINS = np.linspace(-3.0, 1.5, 91)
SHARE_BINS = np.linspace(0.0, 1.0, 41)
ERR_BINS = np.linspace(0.0, 1.2, 61)
TOPK = (8, 16, 32, 64, 96, 128, 192)
PRED = tuple((r, k) for r in (16, 32, 64) for k in (32, 64, 96, 128, 192))
METHODS = ("none", "topk", "pred", "wta24")
ORDERS = ("identity", "wta4")
NUM_EXAMPLES = 3  # example matrices per layer and definition


def unit_norms(model) -> tuple[jax.Array, jax.Array]:
    """``[L, E, I]`` routed ``||W_down[i]||`` and ``[L, I]`` shared."""
    m = forward.replicate(model)
    routed = jnp.stack([jnp.linalg.norm(lay.mlp.expert_mlp.w_down.astype(jnp.float32), axis=-1) for lay in m.layers()])
    shared = jnp.stack([jnp.linalg.norm(lay.shared[0].w_down.astype(jnp.float32), axis=-1) for lay in m.layers()])
    return routed, shared


def _bin(act: jax.Array) -> jax.Array:
    return jnp.where(act > 0, jnp.clip(jnp.searchsorted(jnp.asarray(ACT_BINS[1:]), act, side="right"), 1, 90), 0)


def make_hist_step():
    @jax.jit
    def step(state, model, tokens, seg, weight):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        valid_tok = reshard(weight.reshape(-1), P(None)) > 0
        new = dict(state)
        units = jnp.arange(I)[None, :]
        for li, lay in enumerate(tr["layers"]):
            experts, toks, act = lay["expert_sorted"], lay["token_sorted"], lay["expert_act"]
            v = valid_tok[toks].astype(jnp.float32)[:, None] * jnp.ones_like(act)
            new["hist"] = new["hist"].at[li, experts[:, None], units, _bin(act)].add(v)
            s_act = lay["shared_act"]
            sv = valid_tok.astype(jnp.float32)[:, None] * jnp.ones_like(s_act)
            new["s_hist"] = new["s_hist"].at[li, units, _bin(s_act)].add(sv)
        return new

    return step


def quantile_from_hist(hist: np.ndarray, q: float) -> np.ndarray:
    """``[..., B]`` histograms over ``ACT_BINS`` -> the value with fraction ``1 - q`` of the mass at or above it
    (log-interpolated inside its bin)."""
    total = hist.sum(-1, keepdims=True)
    above = np.cumsum(hist[..., ::-1], axis=-1)[..., ::-1]  # mass at or above each bin
    target = (1 - q) * total
    idx = np.clip(np.sum(above >= target, axis=-1) - 1, 1, hist.shape[-1] - 1)
    lo = np.log10(ACT_BINS[idx])
    hi = np.log10(ACT_BINS[np.minimum(idx + 1, len(ACT_BINS) - 1)])
    a_hi = np.take_along_axis(above, np.minimum(idx + 1, hist.shape[-1] - 1)[..., None], -1)[..., 0]
    h = np.take_along_axis(hist, idx[..., None], -1)[..., 0]
    frac = np.clip((target[..., 0] - a_hi) / np.maximum(h, 1e-9), 0.0, 1.0)
    return (10 ** (hi - frac * (hi - lo))).astype(np.float32)


def _concentration(c: jax.Array, v: jax.Array):
    """Per-row ``c [N, I]`` (valid weights ``v [N]``) -> histograms: units for each quantile of ``sum c``
    ``[Q, I + 1]``, participation ratio ``[I + 1]``, top-1 and top-4 share ``[2, S]``; and the descending order."""
    order = jnp.argsort(-c, axis=-1)
    srt = jnp.take_along_axis(c, order, axis=-1)
    total = jnp.sum(srt, axis=-1)
    ok = (total > 0).astype(jnp.float32) * v
    cum = jnp.cumsum(srt, axis=-1) / jnp.maximum(total, 1e-30)[:, None]
    kq = jnp.stack([jnp.minimum(jnp.sum(cum < q, axis=-1) + 1, I) for q in QUANTILES])
    kq_hist = jnp.stack([jnp.zeros(I + 1).at[k].add(ok) for k in kq])
    pr = jnp.square(total) / jnp.maximum(jnp.sum(jnp.square(srt), axis=-1), 1e-30)
    pr_hist = jnp.zeros(I + 1).at[jnp.clip(pr.astype(jnp.int32), 0, I)].add(ok)
    top1 = srt[:, 0] / jnp.maximum(total, 1e-30)
    top4 = jnp.sum(srt[:, :4], axis=-1) / jnp.maximum(total, 1e-30)
    shares = jnp.stack([jnp.histogram(top1, SHARE_BINS, weights=ok)[0], jnp.histogram(top4, SHARE_BINS, weights=ok)[0]])
    rank = jnp.argsort(order, axis=-1)
    return kq_hist, pr_hist, shares, rank, ok


def _recon(act, rank, ok, y_full, down):
    """Relative error of the output rebuilt from the top-k units for every ``RECON_K``: histograms ``[K, B]``, sums
    ``[K]`` and the per-quantile smallest grid k reaching the captured fraction ``[Q, K + 1]``."""
    norm = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(y_full), axis=-1), 1e-30))
    errs = []
    for k in RECON_K:
        y = down((act * (rank < k)).astype(jnp.bfloat16))
        errs.append(jnp.sqrt(jnp.sum(jnp.square(y - y_full), axis=-1)) / norm)
    errs = jnp.stack(errs)  # [K, N]
    hists = jnp.stack([jnp.histogram(jnp.clip(e, 0, 1.199), ERR_BINS, weights=ok)[0] for e in errs])
    sums = jnp.sum(errs * ok[None], axis=-1)
    reach = []
    for q in QUANTILES:
        hit = (1.0 - errs) >= q  # [K, N]
        first = jnp.where(jnp.any(hit, axis=0), jnp.argmax(hit, axis=0), len(RECON_K))
        reach.append(jnp.zeros(len(RECON_K) + 1).at[first].add(ok))
    return hists, sums, jnp.stack(reach)


def make_conc_step():
    @jax.jit
    def step(state, model, tokens, seg, weight, norms, s_norms, q90, s_q90):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        valid_tok = reshard(weight.reshape(-1), P(None)) > 0
        new = dict(state)
        for li, (lay, blk) in enumerate(zip(tr["layers"], model.layers(), strict=True)):
            experts, toks, act, pre = lay["expert_sorted"], lay["token_sorted"], lay["expert_act"], lay["expert_pre"]
            valid = valid_tok[toks]
            v = valid.astype(jnp.float32)
            c = act * norms[li][experts]
            kq, pr, shares, rank, ok = _concentration(c, v)
            counts = lay["expert_counts"]
            w_down = blk.mlp.expert_mlp.w_down

            def down(a, w_down=w_down, counts=counts):
                return forward._local_ragged_dot(a, w_down, counts).astype(jnp.float32)

            y_full = down(act.astype(jnp.bfloat16))
            rh, rs, reach = _recon(act, rank, ok, y_full, down)
            for name, val in (("kq", kq), ("pr", pr), ("shares", shares), ("recon_hist", rh), ("recon_sum", rs)):
                new[name] = new[name].at[li].add(val)
            new["reach"] = new["reach"].at[li].add(reach)
            new["n"] = new["n"].at[li].add(jnp.sum(ok))
            # Strong-firing co-occurrence (padded per expert).
            rows, _ = padded_rows(experts, lay["expert_starts"], valid, CAPACITY)
            total = jnp.sum(c, axis=-1, keepdims=True)
            strong_q = (act >= q90[li][experts]) & (pre > 0)
            strong_c = (c >= STRONG_SHARE * total) & (total > 0)
            active = pre > 0
            pq = scatter_padded(strong_q.astype(jnp.bfloat16), rows, CAPACITY)
            pc = scatter_padded(strong_c.astype(jnp.bfloat16), rows, CAPACITY)
            pa = scatter_padded(active.astype(jnp.bfloat16), rows, CAPACITY)
            ein = lambda a, b: jnp.einsum("eci,ecj->eij", a, b, preferred_element_type=jnp.float32)  # noqa: E731
            new["co_q"] = new["co_q"].at[li].add(ein(pq, pq))
            new["co_c"] = new["co_c"].at[li].add(ein(pc, pc))
            new["co_qa"] = new["co_qa"].at[li].add(ein(pq, pa))
            new["rows"] = (
                new["rows"]
                .at[li]
                .add(jax.ops.segment_sum((rows < E * CAPACITY).astype(jnp.float32), experts, num_segments=E))
            )
            new["act_count"] = (
                new["act_count"]
                .at[li]
                .add(jax.ops.segment_sum(active.astype(jnp.float32) * v[:, None], experts, num_segments=E))
            )
            # Shared expert.
            s_act = lay["shared_act"]
            sv = valid_tok.astype(jnp.float32)
            sc = s_act * s_norms[li][None, :]
            skq, spr, sshares, srank, sok = _concentration(sc, sv)
            s_down_w = blk.shared[0].w_down

            def s_down(a, s_down_w=s_down_w):
                return jnp.einsum("ti,id->td", a, s_down_w, preferred_element_type=jnp.float32)

            sy = s_down(s_act.astype(jnp.bfloat16))
            srh, srs, sreach = _recon(s_act, srank, sok, sy, s_down)
            for name, val in (
                ("s_kq", skq),
                ("s_pr", spr),
                ("s_shares", sshares),
                ("s_recon_hist", srh),
                ("s_recon_sum", srs),
                ("s_reach", sreach),
            ):
                new[name] = new[name].at[li].add(val)
            new["s_n"] = new["s_n"].at[li].add(jnp.sum(sok))
            s_strong_q = ((s_act >= s_q90[li][None, :]) & (lay["shared_pre"] > 0)).astype(jnp.bfloat16) * sv[
                :, None
            ].astype(jnp.bfloat16)
            new["s_co_q"] = (
                new["s_co_q"]
                .at[li]
                .add(jnp.einsum("ti,tj->ij", s_strong_q, s_strong_q, preferred_element_type=jnp.float32))
            )
        return new

    return step


def conc_state() -> dict:
    nk, nq = len(RECON_K), len(QUANTILES)
    z = jnp.zeros
    return dict(
        kq=z((L, nq, I + 1)),
        pr=z((L, I + 1)),
        shares=z((L, 2, len(SHARE_BINS) - 1)),
        recon_hist=z((L, nk, len(ERR_BINS) - 1)),
        recon_sum=z((L, nk)),
        reach=z((L, nq, nk + 1)),
        n=z((L,)),
        co_q=z((L, E, I, I)),
        co_c=z((L, E, I, I)),
        co_qa=z((L, E, I, I)),
        rows=z((L, E)),
        act_count=z((L, E, I)),
        s_kq=z((L, nq, I + 1)),
        s_pr=z((L, I + 1)),
        s_shares=z((L, 2, len(SHARE_BINS) - 1)),
        s_recon_hist=z((L, nk, len(ERR_BINS) - 1)),
        s_recon_sum=z((L, nk)),
        s_reach=z((L, nq, nk + 1)),
        s_n=z((L,)),
        s_co_q=z((L, I, I)),
    )


@jax.jit
def strong_census(co: jax.Array, n: jax.Array) -> dict:
    """Census of one layer's strong co-occurrence ``[E, I, I]`` over ``n [E]`` rows: pairs (i < j) whose expected
    co-occurrence under the shuffle null is at least ``MIN_EXPECTED``."""
    nn = jnp.maximum(n, 1.0)[:, None, None]
    diag = jnp.diagonal(co, axis1=1, axis2=2)
    ni, nj = diag[:, :, None], diag[:, None, :]
    mu = ni * nj / nn
    var = ni * nj * (nn - ni) * (nn - nj) / (nn * nn * jnp.maximum(nn - 1, 1.0))
    z = (co - mu) / jnp.sqrt(jnp.maximum(var, 1e-6))
    lift = co / jnp.maximum(mu, 1e-9)
    upper = jnp.triu(jnp.ones((I, I), bool), k=1)[None]
    elig = (upper & (mu >= MIN_EXPECTED)).astype(jnp.float32)
    log_lift = jnp.clip(jnp.log10(jnp.maximum(lift, 1e-3)), LOG_LIFT_BINS[0], LOG_LIFT_BINS[-1] - 1e-6)
    return dict(
        eligible=jnp.sum(elig),
        exclusive=jnp.stack([jnp.sum(elig * (lift < t)) for t in LIFT_THRESHOLDS]),
        never=jnp.sum(elig * (co == 0)),
        sig_neg=jnp.sum(elig * (z < -5.0)),
        sig_pos=jnp.sum(elig * (z > 5.0)),
        hist=jnp.histogram(log_lift, LOG_LIFT_BINS, weights=elig)[0],
        rate=diag / nn[:, :, 0],
        per_expert_excl=jnp.sum(elig * (lift < 0.2), axis=(1, 2)),
    )


@jax.jit
def mixed_lift(co_qa: jax.Array, n: jax.Array, act_count: jax.Array, co_q: jax.Array) -> dict:
    """``P(j active | i strong) / P(j active)`` histogram (i != j), and its minimum per expert."""
    nn = jnp.maximum(n, 1.0)[:, None]
    p_act = act_count / nn  # [E, I]
    n_strong = jnp.diagonal(co_q, axis1=1, axis2=2)  # [E, I]
    cond = co_qa / jnp.maximum(n_strong, 1.0)[:, :, None]
    lift = cond / jnp.maximum(p_act[:, None, :], 1e-9)
    ok = ((n_strong[:, :, None] * p_act[:, None, :]) >= MIN_EXPECTED) & ~jnp.eye(I, dtype=bool)[None]
    okf = ok.astype(jnp.float32)
    log_lift = jnp.clip(jnp.log10(jnp.maximum(lift, 1e-3)), LOG_LIFT_BINS[0], LOG_LIFT_BINS[-1] - 1e-6)
    return dict(
        hist=jnp.histogram(log_lift, LOG_LIFT_BINS, weights=okf)[0],
        eligible=jnp.sum(okf),
        below=jnp.stack([jnp.sum(okf * (lift < t)) for t in LIFT_THRESHOLDS]),
    )


def phi_from(co: np.ndarray, n: np.ndarray) -> np.ndarray:
    nn = np.maximum(n, 1.0)[:, None, None]
    p = np.diagonal(co, axis1=1, axis2=2) / nn[:, :, 0]
    sd = np.sqrt(np.maximum(p * (1 - p), 1e-9))
    return np.clip((co / nn - p[:, :, None] * p[:, None, :]) / (sd[:, :, None] * sd[:, None, :]), -1, 1)


def make_share_step():
    """Per layer and order: histogram of the share of ``sum c`` in the heaviest half and quarter, and the per-expert
    mean heavier-half share."""

    @jax.jit
    def step(state, model, tokens, seg, weight, norms, perms):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        valid_tok = reshard(weight.reshape(-1), P(None)) > 0
        new = dict(state)
        for li, lay in enumerate(tr["layers"]):
            experts, toks, act = lay["expert_sorted"], lay["token_sorted"], lay["expert_act"]
            v = valid_tok[toks].astype(jnp.float32)
            c = act * norms[li][experts]
            n = c.shape[0]
            total = jnp.maximum(jnp.sum(c, axis=-1), 1e-30)
            hs, ex = [], []
            for o in range(perms.shape[0]):
                ordered = jnp.take_along_axis(c, perms[o, li][experts], axis=1)
                for size in (192, 96):
                    share = jnp.max(jnp.sum(ordered.reshape(n, I // size, size), axis=-1), axis=-1) / total
                    hs.append(jnp.histogram(share, SHARE_BINS, weights=v)[0])
                    if size == 192:
                        ex.append(jax.ops.segment_sum(share * v, experts, num_segments=E))
            new["hist"] = new["hist"].at[li].add(jnp.stack(hs))
            new["expert"] = new["expert"].at[li].add(jnp.stack(ex))
        return new

    return step


# ---------------------------------------------------------------------------------------------------------- sweep


class MassPlan(NamedTuple):
    norms: jax.Array  # [L, E, I]
    lr_a: jax.Array  # [L, E, R, MAX_RANK]
    lr_b: jax.Array  # [L, E, MAX_RANK, I]
    perms: jax.Array  # [O, L, E, I]
    inv: jax.Array


class MassParams(NamedTuple):
    layer_on: jax.Array
    method: jax.Array
    k: jax.Array
    rank_mask: jax.Array
    order: jax.Array


@dataclasses.dataclass(frozen=True)
class MassConfig:
    name: str
    method: str
    layers: tuple[int, ...]
    k: int = 384
    rank: int = MAX_RANK
    order: str = "identity"

    def params(self) -> MassParams:
        on = np.zeros(L, np.float32)
        on[list(self.layers)] = 1.0
        return MassParams(
            layer_on=jnp.asarray(on),
            method=jnp.asarray(METHODS.index(self.method), jnp.int32),
            k=jnp.asarray(self.k, jnp.int32),
            rank_mask=jnp.asarray(np.arange(MAX_RANK) < self.rank, jnp.float32),
            order=jnp.asarray(ORDERS.index(self.order), jnp.int32),
        )


def make_mass_gate(gp: MassParams, plan: MassPlan) -> forward.RoutedGate:
    def gate(layer: int, expert: jax.Array, x: jax.Array, pre: jax.Array) -> jax.Array:
        norms = plan.norms[layer][expert]
        c = jnp.square(jax.nn.relu(pre)) * norms

        def predicted():
            counts = jnp.zeros((E,), jnp.int32).at[expert].add(1)
            h = forward._local_ragged_dot(x.astype(jnp.bfloat16), plan.lr_a[layer], counts).astype(jnp.float32)
            h = (h * gp.rank_mask[None, :]).astype(jnp.bfloat16)
            pred = forward._local_ragged_dot(h, plan.lr_b[layer], counts).astype(jnp.float32)
            return jnp.square(jax.nn.relu(pred)) * norms

        score = jax.lax.cond(gp.method == METHODS.index("pred"), predicted, lambda: c)
        rank = jnp.argsort(jnp.argsort(-score, axis=-1), axis=-1)
        topk = rank < gp.k
        perm = plan.perms[gp.order, layer][expert]
        inv = plan.inv[gp.order, layer][expert]
        n = c.shape[0]
        blocks = jnp.take_along_axis(c, perm, axis=1).reshape(n, I // 4, 4)
        wta = (jnp.argsort(jnp.argsort(-blocks, axis=-1), axis=-1) < 2).reshape(n, I)
        wta = jnp.take_along_axis(wta, inv, axis=1)
        keep = jnp.where(gp.method == METHODS.index("wta24"), wta, topk)
        keep = jnp.where(gp.method == METHODS.index("none"), True, keep).astype(jnp.float32)
        return jnp.where(gp.layer_on[layer] > 0, keep, 1.0)

    return gate


def make_mass_eval_step():
    @jax.jit
    def step(model, tokens, seg, weight, selected, gp: MassParams, plan: MassPlan):
        model = forward.replicate(model)
        frozen = forward.no_override(selected.shape[1])._replace(
            sel_mask=jnp.ones(selected.shape[:2], bool), sel_override=selected
        )
        tr = forward.trace_forward(
            model, tokens, seg, forward.no_ablation(), routing=frozen, routed_gate=make_mass_gate(gp, plan)
        )
        w = reshard(weight.reshape(-1), P(None))
        valid_tok = w > 0
        loss = reshard(tr["loss"].reshape(-1), P(None))
        per_layer = []
        for lay, blk in zip(tr["layers"], model.layers(), strict=True):
            toks = lay["token_sorted"]
            v = valid_tok[toks].astype(jnp.float32)
            pre, act, keep = lay["expert_pre"], lay["expert_act"], lay["expert_gate"]
            full = jnp.square(jax.nn.relu(pre))
            on = (pre > 0).astype(jnp.float32)
            counts = lay["expert_counts"]
            w_down = blk.mlp.expert_mlp.w_down
            y_full = forward._local_ragged_dot(full.astype(jnp.bfloat16), w_down, counts).astype(jnp.float32)
            y = forward._local_ragged_dot(act.astype(jnp.bfloat16), w_down, counts).astype(jnp.float32)
            err = jnp.sum(jnp.square(y - y_full), axis=-1)
            norm = jnp.sum(jnp.square(y_full), axis=-1)
            per_layer.append(
                jnp.stack(
                    [
                        jnp.sum(v),
                        jnp.sum(v[:, None] * (1.0 - keep)),
                        jnp.sum(v[:, None] * on),
                        jnp.sum(v[:, None] * on * (1.0 - keep)),
                        jnp.sum(v[:, None] * full),
                        jnp.sum(v[:, None] * act),
                        jnp.sum(v * err),
                        jnp.sum(v * norm),
                        jnp.sum(v * jnp.sqrt(err / jnp.maximum(norm, 1e-12))),
                        jnp.zeros(()),
                    ]
                )
            )
        return jnp.stack([jnp.sum(loss * w), jnp.sum(w)]), jnp.stack(per_layer)

    return step


def sweep_configs() -> list[MassConfig]:
    all_l = tuple(range(L))
    cfgs = [MassConfig("base", "none", ())]
    per_layer = []
    for k in TOPK:
        cfgs.append(MassConfig(f"topk_{k}", "topk", all_l, k=k))
        per_layer.append(cfgs[-1])
    for r, k in PRED:
        cfgs.append(MassConfig(f"pred_r{r}_k{k}", "pred", all_l, k=k, rank=r))
    per_layer += [MassConfig("pred_r32_k64", "pred", (), k=64, rank=32), MassConfig("pred_r64_k128", "pred", (), k=128)]
    for o in ORDERS:
        cfgs.append(MassConfig(f"wta24_{o}", "wta24", all_l, order=o))
        per_layer.append(cfgs[-1])
    for c in per_layer:
        for li in range(L):
            cfgs.append(dataclasses.replace(c, name=f"{c.name}@L{li}", layers=(li,)))
    return cfgs


def run(step, state, model, batches, mesh, *extra):
    for batch in batches:
        tokens, seg, w = common.device_batch(batch, mesh)
        state = step(state, model, tokens, seg, w, *extra)
    return state


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--calib-sequences", type=int, default=64)
    parser.add_argument("--eval-sequences", type=int, default=128)
    args = parser.parse_args()
    t0 = time.time()
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    skip_plan = load_npz(args.tag, "excl_plan")
    skip_orders = list(skip_plan["orders"])
    lr_a, lr_b = [], []
    for li in range(L):
        layer = load_npz(args.tag, f"excl_cofire_L{li}")
        a, b, _ = lowrank(layer["w_eff"], layer["lat_m2"] / np.maximum(layer["cnt"], 1.0)[:, None, None])
        lr_a.append(a)
        lr_b.append(b)
    logger.info("low-rank factors (%.0fs)", time.time() - t0)
    calib = common.train_batches(BATCH, args.calib_sequences, seed=CALIB_SEED)
    evals = common.train_batches(BATCH, args.eval_sequences, seed=EVAL_SEED)
    out: dict = {}
    with set_mesh(mesh):
        model = common.compute_model(params)
        norms, s_norms = unit_norms(model)
        hstate = dict(hist=jnp.zeros((L, E, I, len(ACT_BINS) - 1)), s_hist=jnp.zeros((L, I, len(ACT_BINS) - 1)))
        hstate = run(make_hist_step(), hstate, model, calib, mesh)
        q90 = quantile_from_hist(np.asarray(hstate["hist"]), STRONG_QUANTILE)
        s_q90 = quantile_from_hist(np.asarray(hstate["s_hist"]), STRONG_QUANTILE)
        del hstate
        logger.info("quantiles (%.0fs)", time.time() - t0)
        cstate = run(
            make_conc_step(), conc_state(), model, evals, mesh, norms, s_norms, jnp.asarray(q90), jnp.asarray(s_q90)
        )
        logger.info("concentration pass (%.0fs)", time.time() - t0)
        perms_strong = {"q90": [], "c5": []}
        examples = {}
        for li in range(L):
            n = cstate["rows"][li]
            for name, key in (("q90", "co_q"), ("c5", "co_c")):
                cen = {k: np.asarray(v) for k, v in strong_census(cstate[key][li], n).items()}
                for k, v in cen.items():
                    out.setdefault(f"{name}_{k}", []).append(v)
                co = np.asarray(cstate[key][li])
                phi = phi_from(co, np.asarray(n))
                perms_strong[name].append(np.stack([bisection_order(phi[e]) for e in range(E)]))
                # examples: experts with the most strongly exclusive pairs, plus a median one
                score = cen["per_expert_excl"]
                pick = [*np.argsort(-score)[: NUM_EXAMPLES - 1], int(np.argsort(score)[E // 2])]
                nn = np.maximum(np.asarray(n), 1.0)
                for e in pick:
                    p = np.diagonal(co[e]) / nn[e]
                    lift = (co[e] / nn[e]) / np.maximum(np.outer(p, p), 1e-12)
                    order = perms_strong[name][-1][e]
                    examples.setdefault(name, []).append(
                        dict(layer=li, expert=int(e), lift=lift[np.ix_(order, order)].astype(np.float16), rate=p[order])
                    )
            mix = {
                k: np.asarray(v)
                for k, v in mixed_lift(
                    cstate["co_qa"][li], cstate["rows"][li], cstate["act_count"][li], cstate["co_q"][li]
                ).items()
            }
            for k, v in mix.items():
                out.setdefault(f"mixed_{k}", []).append(v)
            s_cen = {
                k: np.asarray(v) for k, v in strong_census(cstate["s_co_q"][li][None], cstate["s_n"][li][None]).items()
            }
            for k, v in s_cen.items():
                out.setdefault(f"s_q90_{k}", []).append(v)
            logger.info("strong census layer %d (%.0fs)", li, time.time() - t0)
        host = {k: np.asarray(v) for k, v in cstate.items() if k not in ("co_q", "co_c", "co_qa", "s_co_q")}
        del cstate
        rng = np.random.default_rng(7)
        rand = np.stack([np.stack([rng.permutation(I) for _ in range(E)]) for _ in range(L)])
        share_orders = ("q90", "c5", "binary", "random")
        sperms = np.stack(
            [
                np.stack(perms_strong["q90"]),
                np.stack(perms_strong["c5"]),
                skip_plan["perms"][skip_orders.index("bisection")].astype(np.int64),
                rand,
            ]
        ).astype(np.int32)
        sstate = dict(
            hist=jnp.zeros((L, 2 * len(share_orders), len(SHARE_BINS) - 1)), expert=jnp.zeros((L, len(share_orders), E))
        )
        sstate = {
            k: np.asarray(v)
            for k, v in run(make_share_step(), sstate, model, evals, mesh, norms, jnp.asarray(sperms)).items()
        }
        logger.info("share pass (%.0fs)", time.time() - t0)
        wta_perm = skip_plan["perms"][skip_orders.index("wta4")].astype(np.int32)
        perms = np.stack([np.tile(np.arange(I, dtype=np.int32), (L, E, 1)), wta_perm])
        plan = MassPlan(
            norms=norms,
            lr_a=jnp.asarray(np.stack(lr_a), jnp.bfloat16),
            lr_b=jnp.asarray(np.stack(lr_b), jnp.bfloat16),
            perms=jnp.asarray(perms),
            inv=jnp.asarray(np.argsort(perms, axis=-1).astype(np.int32)),
        )
        device_batches = [common.device_batch(b, mesh) for b in evals]
        selections = [base_selection(model, tokens, seg) for tokens, seg, _ in device_batches]
        step = make_mass_eval_step()
        results = []
        for ci, c in enumerate(sweep_configs()):
            gp = c.params()
            tot = jnp.zeros((2,))
            lay = jnp.zeros((L, len(LAYER_FIELDS)))
            for (tokens, seg, w), sel in zip(device_batches, selections, strict=True):
                a, b = step(model, tokens, seg, w, sel, gp, plan)
                tot, lay = tot + a, lay + b
            tot, lay = np.asarray(tot), np.asarray(lay)
            row = dataclasses.asdict(c)
            row.update(loss=float(tot[0] / tot[1]), tokens=float(tot[1]))
            row["layer"] = {f: lay[:, j].tolist() for j, f in enumerate(LAYER_FIELDS)}
            results.append(row)
            if ci % 10 == 0:
                logger.info("config %d %s loss %.5f (%.0fs)", ci, c.name, row["loss"], time.time() - t0)
    arrays = {k: np.stack(v) for k, v in out.items()}
    for name, rows in examples.items():
        arrays[f"ex_{name}_lift"] = np.stack([r["lift"] for r in rows])
        arrays[f"ex_{name}_rate"] = np.stack([r["rate"] for r in rows])
        arrays[f"ex_{name}_id"] = np.asarray([[r["layer"], r["expert"]] for r in rows])
    common.save_npz(
        args.tag,
        "excl_mass",
        **arrays,
        **{f"conc_{k}": v for k, v in host.items()},
        **{f"share_{k}": v for k, v in sstate.items()},
        share_orders=np.asarray(share_orders),
        q90=q90,
        s_q90=s_q90,
        quantiles=np.asarray(QUANTILES),
        recon_k=np.asarray(RECON_K),
        share_bins=SHARE_BINS,
        err_bins=ERR_BINS,
        log_lift_bins=LOG_LIFT_BINS,
        lift_thresholds=np.asarray(LIFT_THRESHOLDS),
    )
    common.save_json(args.tag, "excl_mass", dict(configs=results, eval_sequences=args.eval_sequences))
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
