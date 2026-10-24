# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Neuron exclusion study, skip pass: can the co-firing structure inside a routed expert predict which of its ReLU^2
units are zero, and what does skipping them cost?

Plan (host, from ``exclusion_stats``' full co-firing matrices, firing moments and input moments), per expert:

- unit orders: identity; a seriation (average-linkage clustering on ``1 - phi``, ``phi`` the units' binary
  correlation) so co-firing units are contiguous; a recursive spectral bisection (balanced halves, then quarters, on
  the positive part of ``phi``; seriated within each quarter); greedy block-WTA orders for block sizes 4, 8 and 16
  (units assigned to blocks so the expected mass of non-top units, ``P(i, j) min(m_i, m_j)`` summed over block
  pairs, is small); a random order.
- a trigger set: ``NUM_TRIGGERS`` units picked greedily by deflated ``phi^2`` coverage; their joint on/off pattern
  indexes a table of every unit's firing rate given the pattern, filled on calibration tokens.
- a low-rank predictor: ``W V_r V_r^T`` with ``V_r`` the top right singular vectors of ``M^(1/2) W`` (``M`` the input
  second moment), the data-optimal rank-``r`` approximation of the up-projection; scores are normalized by each
  unit's RMS pre-activation.

Group pass (held-out tokens, unpatched): per token, the share of an expert's activation mass in its heaviest half
and quarter of the bisection order, against the original contiguous halves and a random order.

Skip sweep (the same held-out tokens, routing frozen to the unpatched selections): every config patches the routed
units' keep (``forward.RoutedGate``) in one or all layers and records the loss and, per layer, the skipped unit
fraction, active units skipped, activation mass kept, the expert-output relative error and block-WTA violations.
Methods: oracle (skip a block when every unit in it is zero), random (skip blocks at a rate), low-rank and trigger
(skip a block when its maximum score is under a threshold), block WTA (keep the top-k units by activation in every
block of B in a given order) and top-group (keep only the heaviest half / quarter of the bisection order, or of the
original order: oracle sub-expert dispatch). Block granularities 1, 32, 64 and 128 in the seriation order.

Writes ``excl_plan.npz`` (orders, triggers, labels) and ``excl_skip.json``.
"""

import argparse
import dataclasses
import json
import logging
import time
from typing import NamedTuple

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from experiments.grug.fast_track.interp import common, forward
from experiments.grug.fast_track.interp.exclusion_stats import CAPACITY, padded_rows, scatter_padded
from experiments.grug.fast_track.interp.ngram import load_npz

logger = logging.getLogger(__name__)

BATCH = 4
L, E, I, R, K = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.LATENT, forward.TOP_K
GRANULARITIES = (1, 32, 64, 128)
WTA_BLOCKS = (4, 8, 16)
WTA_CONFIGS = ((2, 4), (1, 4), (1, 8), (1, 16), (2, 16), (4, 16))
GROUP_SIZES = (192, 96)  # halves and quarters
MAX_RANK = 64
RANKS = (16, 32, 64)
NUM_TRIGGERS = 6
TABLE_PRIOR = 4.0  # pseudo-count pulling a pattern's firing rate to the unit's overall rate
CALIB_SEED = 21  # the stats pass's draw (in-distribution, disjoint from the eval draw)
EVAL_SEED = 22
MIN_TOKENS = 2000  # experts routed fewer stats tokens than this are excluded from the plan's structure summaries
MASS_BINS = np.linspace(0.0, 1.0, 41)


ORDERS = ("identity", "seriation", "bisection", "wta4", "wta8", "wta16", "random")
METHODS = ("none", "oracle", "random", "lowrank", "trigger", "wta", "topgroup")


class Plan(NamedTuple):
    """Device arrays of the per-expert plan, all layers stacked."""

    perms: jax.Array  # [O, L, E, I] unit at each ordered position
    inv: jax.Array  # [O, L, E, I] ordered position of each unit
    lr_a: jax.Array  # [L, E, R, MAX_RANK]
    lr_b: jax.Array  # [L, E, MAX_RANK, I]
    lr_rms: jax.Array  # [L, E, I]
    trig: jax.Array  # [L, E, NUM_TRIGGERS]
    trig_mask: jax.Array  # [L, E, I]
    table: jax.Array  # [L, E, 2^NUM_TRIGGERS, I] firing rate given the trigger pattern


class GateParams(NamedTuple):
    """One sweep config (all traced)."""

    layer_on: jax.Array  # [L]
    method: jax.Array  # [] index into METHODS
    order: jax.Array  # [] index into ORDERS
    gran: jax.Array  # [] index into GRANULARITIES
    thr: jax.Array  # [] score threshold (random: skip rate)
    rank_mask: jax.Array  # [MAX_RANK]
    wta_block: jax.Array  # [] index into WTA_BLOCKS
    wta_k: jax.Array  # []
    group: jax.Array  # [] index into GROUP_SIZES


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    method: str
    layers: tuple[int, ...]
    order: str = "seriation"
    gran: int = 1
    thr: float = 0.0
    rank: int = MAX_RANK
    wta: tuple[int, int] = (1, 4)  # (k, B)
    group: int = 192

    def params(self) -> GateParams:
        on = np.zeros(L, np.float32)
        on[list(self.layers)] = 1.0
        return GateParams(
            layer_on=jnp.asarray(on),
            method=jnp.asarray(METHODS.index(self.method), jnp.int32),
            order=jnp.asarray(ORDERS.index(self.order), jnp.int32),
            gran=jnp.asarray(GRANULARITIES.index(self.gran), jnp.int32),
            thr=jnp.asarray(self.thr, jnp.float32),
            rank_mask=jnp.asarray(np.arange(MAX_RANK) < self.rank, jnp.float32),
            wta_block=jnp.asarray(WTA_BLOCKS.index(self.wta[1]), jnp.int32),
            wta_k=jnp.asarray(self.wta[0], jnp.int32),
            group=jnp.asarray(GROUP_SIZES.index(self.group), jnp.int32),
        )


# ---------------------------------------------------------------------------------------------------------- plan


def phi_matrix(cofire: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """``[E, I, I]`` binary (phi) correlation of the units' firing."""
    n = np.maximum(cnt, 1.0)[:, None, None]
    p = np.diagonal(cofire, axis1=1, axis2=2) / n[:, :, 0]
    pij = cofire / n
    sd = np.sqrt(np.maximum(p * (1 - p), 1e-9))
    phi = (pij - p[:, :, None] * p[:, None, :]) / (sd[:, :, None] * sd[:, None, :])
    return np.clip(phi, -1.0, 1.0)


def seriation(phi: np.ndarray) -> np.ndarray:
    """Units ordered by the leaves of an average-linkage tree on ``1 - phi``."""
    dist = np.clip(1.0 - phi, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    return leaves_list(linkage(squareform(dist, checks=False), method="average"))


def fiedler_split(phi: np.ndarray, units: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Balanced bisection of ``units`` at the median of the second eigenvector of the normalized positive-``phi``
    affinity."""
    w = np.maximum(phi[np.ix_(units, units)], 0.0)
    np.fill_diagonal(w, 0.0)
    d = np.maximum(w.sum(1), 1e-9)
    norm = w / np.sqrt(d[:, None] * d[None, :])
    _, vecs = np.linalg.eigh(norm)
    f = vecs[:, -2] / np.sqrt(d)
    order = np.argsort(f, kind="stable")
    half = len(units) // 2
    return units[order[:half]], units[order[half:]]


def bisection_order(phi: np.ndarray, depth: int = 2) -> np.ndarray:
    parts = [np.arange(phi.shape[0])]
    for _ in range(depth):
        parts = [half for part in parts for half in fiedler_split(phi, part)]
    return np.concatenate([part[seriation(phi[np.ix_(part, part)])] for part in parts])


def wta_orders(p_joint: np.ndarray, mean_active: np.ndarray, block: int) -> np.ndarray:
    """``[E, I]`` greedy block assignment for top-k-of-``block`` WTA: units (heaviest expected mass first) go to the
    open block whose members they co-fire with least, weighted by the smaller conditional mean activation."""
    e, n = mean_active.shape
    cost = p_joint * np.minimum(mean_active[:, :, None], mean_active[:, None, :])
    nb = n // block
    member = np.zeros((e, n, nb), np.float32)
    fill = np.zeros((e, nb), np.int64)
    weight = np.diagonal(p_joint, axis1=1, axis2=2) * mean_active
    seq = np.argsort(-weight, axis=1)
    rows = np.arange(e)
    for step in range(n):
        u = seq[:, step]
        added = np.einsum("ej,ejb->eb", cost[rows, u], member)
        added = np.where(fill >= block, np.inf, added)
        b = np.argmin(added + 1e-9 * np.arange(nb)[None, :], axis=1)
        member[rows, u, b] = 1.0
        fill[rows, b] += 1
    block_of = np.argmax(member, axis=2)
    return np.argsort(block_of, axis=1, kind="stable")


def pick_triggers(phi: np.ndarray, count: int) -> np.ndarray:
    """``[E, count]`` units chosen greedily by the summed squared correlation with the rest, deflating the
    correlation by each pick."""
    c = phi.copy()
    e = c.shape[0]
    rows = np.arange(e)
    picks = []
    for _ in range(count):
        score = np.sum(np.square(c), axis=2)
        for prev in picks:
            score[rows, prev] = -1.0
        t = np.argmax(score, axis=1)
        picks.append(t)
        ct = c[rows, :, t]  # [E, I]
        c = c - ct[:, :, None] * ct[:, None, :]
    return np.stack(picks, axis=1)


def lowrank(w_eff: np.ndarray, m2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Data-optimal rank-``MAX_RANK`` factors ``A = W V_r``, ``B = V_r^T`` (``V_r`` from the SVD of ``M^(1/2) W``) and
    each unit's RMS pre-activation."""
    vals, vecs = np.linalg.eigh(m2)
    root = np.einsum("erk,ek,esk->ers", vecs, np.sqrt(np.maximum(vals, 0.0)), vecs)
    _, _, vt = np.linalg.svd(np.einsum("ers,esi->eri", root, w_eff), full_matrices=False)
    v = np.swapaxes(vt[:, :MAX_RANK], 1, 2)  # [E, I, r]
    a = np.einsum("eri,eik->erk", w_eff, v)
    rms = np.sqrt(np.maximum(np.einsum("eri,ers,esi->ei", w_eff, m2, w_eff), 1e-12))
    return a.astype(np.float32), np.swapaxes(v, 1, 2).astype(np.float32), rms.astype(np.float32)


def layer_plan(layer: dict, asum: np.ndarray, seed: int) -> dict:
    t0 = time.time()
    cofire, cnt = layer["cofire"], layer["cnt"]
    n = np.maximum(cnt, 1.0)
    phi = phi_matrix(cofire, cnt)
    diag = np.diagonal(cofire, axis1=1, axis2=2)
    mean_active = asum / np.maximum(diag, 1.0)
    p_joint = cofire / n[:, None, None]
    rng = np.random.default_rng(seed)
    orders = {
        "identity": np.tile(np.arange(I), (E, 1)),
        "seriation": np.stack([seriation(phi[e]) for e in range(E)]),
        "bisection": np.stack([bisection_order(phi[e]) for e in range(E)]),
        **{f"wta{b}": wta_orders(p_joint, mean_active, b) for b in WTA_BLOCKS},
        "random": np.stack([rng.permutation(I) for _ in range(E)]),
    }
    logger.info("orders (%.0fs)", time.time() - t0)
    m2 = layer["lat_m2"] / n[:, None, None]
    a, b, rms = lowrank(layer["w_eff"], m2)
    trig = pick_triggers(phi, NUM_TRIGGERS)
    return dict(perms=np.stack([orders[o] for o in ORDERS]), lr_a=a, lr_b=b, lr_rms=rms, trig=trig)


def build_plan(tag: str) -> dict:
    stats = load_npz(tag, "excl_stats")
    parts = []
    for li in range(L):
        layer = load_npz(tag, f"excl_cofire_L{li}")
        parts.append(layer_plan(layer, stats["asum"][li], seed=li))
        logger.info("plan layer %d", li)
    plan = {k: np.stack([p[k] for p in parts]) for k in parts[0]}
    plan["cnt"] = stats["cnt"]
    return plan


def device_plan(plan: dict, table: np.ndarray | None) -> Plan:
    perms = plan["perms"].astype(np.int32)
    inv = np.argsort(perms, axis=-1).astype(np.int32)
    trig_mask = np.zeros((L, E, I), bool)
    np.put_along_axis(trig_mask, plan["trig"], True, axis=-1)
    if table is None:
        table = np.zeros((L, E, 2**NUM_TRIGGERS, I), np.float32)
    return Plan(
        perms=jnp.asarray(perms),
        inv=jnp.asarray(inv),
        lr_a=jnp.asarray(plan["lr_a"], jnp.bfloat16),
        lr_b=jnp.asarray(plan["lr_b"], jnp.bfloat16),
        lr_rms=jnp.asarray(plan["lr_rms"]),
        trig=jnp.asarray(plan["trig"].astype(np.int32)),
        trig_mask=jnp.asarray(trig_mask),
        table=jnp.asarray(table, jnp.bfloat16),
    )


# ---------------------------------------------------------------------------------------------------------- gate


def _counts(expert: jax.Array) -> jax.Array:
    return jnp.zeros((E,), jnp.int32).at[expert].add(1)


def trigger_pattern(pre: jax.Array, trig: jax.Array) -> jax.Array:
    """``[N]`` the trigger units' on/off pattern as an integer (``trig`` the rows' ``[N, NUM_TRIGGERS]`` units)."""
    bits = (jnp.take_along_axis(pre, trig, axis=1) > 0).astype(jnp.int32)
    return jnp.sum(bits << jnp.arange(NUM_TRIGGERS)[None, :], axis=1)


def _ordered(x: jax.Array, perm: jax.Array) -> jax.Array:
    return jnp.take_along_axis(x, perm, axis=1)


def _block_max(x: jax.Array, size: int) -> jax.Array:
    n = x.shape[0]
    return jnp.repeat(jnp.max(x.reshape(n, I // size, size), axis=-1), size, axis=1)


def _wta_keep(act: jax.Array, block: int, k: jax.Array) -> jax.Array:
    n = act.shape[0]
    blocks = act.reshape(n, I // block, block)
    rank = jnp.argsort(jnp.argsort(-blocks, axis=-1), axis=-1)
    return (rank < k).reshape(n, I)


def _wta_violation(pre: jax.Array, block: int, k: jax.Array) -> jax.Array:
    n = pre.shape[0]
    active = jnp.sum((pre > 0).reshape(n, I // block, block), axis=-1)
    return jnp.mean((active > k).astype(jnp.float32), axis=-1)


def _top_group_keep(act: jax.Array, size: int) -> jax.Array:
    n = act.shape[0]
    mass = jnp.sum(act.reshape(n, I // size, size), axis=-1)
    top = jnp.argmax(mass, axis=-1)
    return jnp.repeat(jax.nn.one_hot(top, I // size, dtype=jnp.bool_), size, axis=1)


def make_gate(gp: GateParams, plan: Plan, key: jax.Array, stash: dict) -> forward.RoutedGate:
    """The keep multipliers of one config; ``stash[layer]`` receives the rows' block-WTA violation rate."""

    def gate(layer: int, expert: jax.Array, x: jax.Array, pre: jax.Array) -> jax.Array:
        n = pre.shape[0]
        perm = plan.perms[gp.order, layer][expert]
        inv = plan.inv[gp.order, layer][expert]
        act = jnp.square(jax.nn.relu(pre))
        method = gp.method

        def lowrank_score():
            counts = _counts(expert)
            h = forward._local_ragged_dot(x.astype(jnp.bfloat16), plan.lr_a[layer], counts).astype(jnp.float32)
            h = (h * gp.rank_mask[None, :]).astype(jnp.bfloat16)
            pred = forward._local_ragged_dot(h, plan.lr_b[layer], counts).astype(jnp.float32)
            return pred / plan.lr_rms[layer][expert]

        def trigger_score():
            pattern = trigger_pattern(pre, plan.trig[layer][expert])
            prob = plan.table[layer][expert, pattern].astype(jnp.float32)
            return jnp.where(plan.trig_mask[layer][expert], 2.0, prob)

        branch = jnp.where(
            method == METHODS.index("lowrank"), 2, jnp.where(method == METHODS.index("trigger"), 3, 0)
        ) + (method == METHODS.index("random"))
        score = jax.lax.switch(branch, [lambda: pre, lambda: jnp.zeros_like(pre), lowrank_score, trigger_score])
        ordered = _ordered(score, perm)
        thr = jnp.where(method == METHODS.index("oracle"), 1e-30, gp.thr)
        lkey = jax.random.fold_in(key, layer)
        skips = []
        for size in GRANULARITIES:
            by_score = _block_max(ordered, size) < thr
            u = jax.random.uniform(jax.random.fold_in(lkey, size), (n, I // size))
            by_rand = jnp.repeat(u < gp.thr, size, axis=1)
            skips.append(jnp.where(method == METHODS.index("random"), by_rand, by_score))
        skip = jnp.stack(skips)[gp.gran]
        ordered_act = _ordered(act, perm)
        ordered_pre = _ordered(pre, perm)
        wta = jnp.stack([_wta_keep(ordered_act, b, gp.wta_k) for b in WTA_BLOCKS])[gp.wta_block]
        stash[layer] = jnp.stack([_wta_violation(ordered_pre, b, gp.wta_k) for b in WTA_BLOCKS])[gp.wta_block]
        top = jnp.stack([_top_group_keep(ordered_act, s) for s in GROUP_SIZES])[gp.group]
        keep = jnp.where(
            method == METHODS.index("wta"),
            wta,
            jnp.where(method == METHODS.index("topgroup"), top, ~skip),
        )
        keep = jnp.where(method == METHODS.index("none"), True, keep)
        keep = _ordered(keep, inv).astype(jnp.float32)
        return jnp.where(gp.layer_on[layer] > 0, keep, 1.0)

    return gate


# ---------------------------------------------------------------------------------------------------------- passes


@jax.jit
def base_selection(model, tokens, seg) -> jax.Array:
    """``[L, T, K]`` the unpatched forward's expert selections (the sweep freezes routing to them)."""
    tr = forward.trace_forward(forward.replicate(model), tokens, seg, forward.no_ablation())
    return jnp.stack([lay["selected"] for lay in tr["layers"]])


def make_eval_step():
    @jax.jit
    def step(model, tokens, seg, weight, selected, gp: GateParams, plan: Plan, key):
        model = forward.replicate(model)
        stash: dict = {}
        frozen = forward.no_override(selected.shape[1])._replace(
            sel_mask=jnp.ones(selected.shape[:2], bool), sel_override=selected
        )
        tr = forward.trace_forward(
            model, tokens, seg, forward.no_ablation(), routing=frozen, routed_gate=make_gate(gp, plan, key, stash)
        )
        w = reshard(weight.reshape(-1), P(None))
        valid_tok = w > 0
        loss = reshard(tr["loss"].reshape(-1), P(None))
        per_layer = []
        for li, (lay, blk) in enumerate(zip(tr["layers"], model.layers(), strict=True)):
            toks = lay["token_sorted"]
            v = valid_tok[toks].astype(jnp.float32)
            pre, act, keep = lay["expert_pre"], lay["expert_act"], lay["expert_gate"]
            full = jnp.square(jax.nn.relu(pre))
            on = (pre > 0).astype(jnp.float32)
            w_down = blk.mlp.expert_mlp.w_down
            counts = lay["expert_counts"]
            y_full = forward._local_ragged_dot(full.astype(jnp.bfloat16), w_down, counts).astype(jnp.float32)
            y = forward._local_ragged_dot(act.astype(jnp.bfloat16), w_down, counts).astype(jnp.float32)
            err = jnp.sum(jnp.square(y - y_full), axis=-1)
            norm = jnp.sum(jnp.square(y_full), axis=-1)
            rel = jnp.sqrt(err / jnp.maximum(norm, 1e-12))
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
                        jnp.sum(v * rel),
                        jnp.sum(v * stash[li]),
                    ]
                )
            )
        return jnp.stack([jnp.sum(loss * w), jnp.sum(w)]), jnp.stack(per_layer)

    return step


LAYER_FIELDS = (
    "assignments",
    "units_skipped",
    "active",
    "active_skipped",
    "mass",
    "mass_kept",
    "out_err2",
    "out_norm2",
    "rel_err_sum",
    "wta_violation_sum",
)


def make_table_step():
    """Calibration: per layer and expert, the count of each trigger pattern and of each unit firing under it."""

    @jax.jit
    def step(state, model, tokens, seg, weight, plan: Plan):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        valid_tok = reshard(weight.reshape(-1), P(None)) > 0
        new = dict(state)
        for li, lay in enumerate(tr["layers"]):
            experts, toks, pre = lay["expert_sorted"], lay["token_sorted"], lay["expert_pre"]
            rows, _ = padded_rows(experts, lay["expert_starts"], valid_tok[toks], CAPACITY)
            pattern = trigger_pattern(pre, plan.trig[li][experts])
            onehot = scatter_padded(jax.nn.one_hot(pattern, 2**NUM_TRIGGERS, dtype=jnp.bfloat16), rows, CAPACITY)
            bits = scatter_padded((pre > 0).astype(jnp.bfloat16), rows, CAPACITY)
            new["fire"] = (
                new["fire"].at[li].add(jnp.einsum("ecp,eci->epi", onehot, bits, preferred_element_type=jnp.float32))
            )
            new["seen"] = new["seen"].at[li].add(jnp.sum(onehot.astype(jnp.float32), axis=1))
        return new

    return step


def make_group_step():
    """Per layer: histograms over (valid) assignments of the activation-mass share of the heaviest group, for the
    halves and quarters of the bisection order, the original order's contiguous halves and quarters and a random
    order's; and per expert the mean heaviest-half share of the bisection and random orders."""
    cases = (("bisection", 192), ("bisection", 96), ("identity", 192), ("identity", 96), ("random", 192), ("random", 96))

    @jax.jit
    def step(state, model, tokens, seg, weight, plan: Plan):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        valid_tok = reshard(weight.reshape(-1), P(None)) > 0
        new = dict(state)
        for li, lay in enumerate(tr["layers"]):
            experts, toks, act = lay["expert_sorted"], lay["token_sorted"], lay["expert_act"]
            v = valid_tok[toks].astype(jnp.float32)
            total = jnp.maximum(jnp.sum(act, axis=-1), 1e-12)
            hists, units_hists, per_expert = [], [], []
            for name, size in cases:
                ordered = _ordered(act, plan.perms[ORDERS.index(name), li][experts])
                n = act.shape[0]
                mass = jnp.sum(ordered.reshape(n, I // size, size), axis=-1)
                share = jnp.max(mass, axis=-1) / total
                hists.append(jnp.histogram(share, MASS_BINS, weights=v)[0])
                on = jnp.sum((ordered > 0).reshape(n, I // size, size), axis=-1)
                ushare = jnp.max(on, axis=-1) / jnp.maximum(jnp.sum(on, axis=-1), 1)
                units_hists.append(jnp.histogram(ushare, MASS_BINS, weights=v)[0])
                per_expert.append(jax.ops.segment_sum(share * v, experts, num_segments=E, indices_are_sorted=True))
            new["share_hist"] = new["share_hist"].at[li].add(jnp.stack(hists))
            new["unit_share_hist"] = new["unit_share_hist"].at[li].add(jnp.stack(units_hists))
            new["expert_share"] = new["expert_share"].at[li].add(jnp.stack(per_expert))
            new["expert_n"] = (
                new["expert_n"].at[li].add(jax.ops.segment_sum(v, experts, num_segments=E, indices_are_sorted=True))
            )
        return new

    return step, cases


def table_from_counts(fire: np.ndarray, seen: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (fire + TABLE_PRIOR * p[:, :, None, :]) / (seen[..., None] + TABLE_PRIOR)


def sweep_configs() -> list[Config]:
    all_l = tuple(range(L))
    cfgs = [Config("base", "none", ())]
    for g in GRANULARITIES:
        for order in ("seriation", "identity"):
            cfgs.append(Config(f"oracle_{order}_g{g}", "oracle", all_l, order=order, gran=g))
        for rate in (0.05, 0.1, 0.2, 0.3, 0.45, 0.6):
            cfgs.append(Config(f"random_g{g}_{rate}", "random", all_l, gran=g, thr=rate))
        for rank in RANKS:
            for thr in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25):
                cfgs.append(Config(f"lowrank_r{rank}_g{g}_{thr}", "lowrank", all_l, gran=g, thr=thr, rank=rank))
        for thr in (0.01, 0.03, 0.1, 0.2, 0.3, 0.4):
            cfgs.append(Config(f"trigger_g{g}_{thr}", "trigger", all_l, gran=g, thr=thr))
    for k, b in WTA_CONFIGS:
        for order in ("identity", f"wta{b}"):
            cfgs.append(Config(f"wta_{k}of{b}_{order}", "wta", all_l, order=order, wta=(k, b)))
    for size in GROUP_SIZES:
        for order in ("bisection", "identity", "random"):
            cfgs.append(Config(f"topgroup_{size}_{order}", "topgroup", all_l, order=order, group=size))
    # Per-layer breakdowns (one layer patched at a time).
    per_layer = [
        Config("random_g1_0.2", "random", (), thr=0.2),
        Config("lowrank_r32_g1_-0.5", "lowrank", (), thr=-0.5, rank=32),
        Config("lowrank_r32_g32_-0.5", "lowrank", (), gran=32, thr=-0.5, rank=32),
        Config("trigger_g1_0.1", "trigger", (), thr=0.1),
        *(
            Config(f"wta_{k}of{b}_{o}", "wta", (), order=o, wta=(k, b))
            for k, b in WTA_CONFIGS
            for o in ("identity", f"wta{b}")
        ),
        *(Config(f"topgroup_{s}_bisection", "topgroup", (), order="bisection", group=s) for s in GROUP_SIZES),
        Config("topgroup_192_identity", "topgroup", (), order="identity", group=192),
    ]
    for c in per_layer:
        for li in range(L):
            cfgs.append(dataclasses.replace(c, name=f"{c.name}@L{li}", layers=(li,)))
    return cfgs


def run_pass(step, state, model, batches, plan, mesh):
    for batch in batches:
        tokens, seg, w = common.device_batch(batch, mesh)
        state = step(state, model, tokens, seg, w, plan)
    return state


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--calib-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=128)
    parser.add_argument("--only", default="", help="comma-separated config name prefixes (default: all)")
    args = parser.parse_args()
    t0 = time.time()
    plan_host = build_plan(args.tag)
    logger.info("plan built (%.0fs)", time.time() - t0)
    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    stats = load_npz(args.tag, "excl_stats")
    p_fire = stats["c_p"]  # [L, E, I]
    eval_batches = common.train_batches(BATCH, args.eval_sequences, seed=EVAL_SEED)
    calib_batches = common.train_batches(BATCH, args.calib_sequences, seed=CALIB_SEED)
    with set_mesh(mesh):
        model = common.compute_model(params)
        plan = device_plan(plan_host, None)
        tstate = dict(fire=jnp.zeros((L, E, 2**NUM_TRIGGERS, I)), seen=jnp.zeros((L, E, 2**NUM_TRIGGERS)))
        tstate = run_pass(make_table_step(), tstate, model, calib_batches, plan, mesh)
        fire, seen = np.asarray(tstate["fire"]), np.asarray(tstate["seen"])
        table = table_from_counts(fire, seen, p_fire)
        plan = device_plan(plan_host, table)
        logger.info("trigger tables (%.0fs)", time.time() - t0)
        gstep, cases = make_group_step()
        nc = len(cases)
        gstate = dict(
            share_hist=jnp.zeros((L, nc, len(MASS_BINS) - 1)),
            unit_share_hist=jnp.zeros((L, nc, len(MASS_BINS) - 1)),
            expert_share=jnp.zeros((L, nc, E)),
            expert_n=jnp.zeros((L, E)),
        )
        gstate = {k: np.asarray(v) for k, v in run_pass(gstep, gstate, model, eval_batches, plan, mesh).items()}
        logger.info("group pass (%.0fs)", time.time() - t0)
        step = make_eval_step()
        key = jax.random.PRNGKey(EVAL_SEED)
        device_batches = [common.device_batch(b, mesh) for b in eval_batches]
        selections = [base_selection(model, tokens, seg) for tokens, seg, _ in device_batches]
        prefixes = tuple(p for p in args.only.split(",") if p)
        results = []
        for ci, c in enumerate(sweep_configs()):
            if prefixes and c.name != "base" and not c.name.startswith(prefixes):
                continue
            gp = c.params()
            tot = jnp.zeros((2,))
            lay = jnp.zeros((L, len(LAYER_FIELDS)))
            for bi, ((tokens, seg, w), sel) in enumerate(zip(device_batches, selections, strict=True)):
                a, b = step(model, tokens, seg, w, sel, gp, plan, jax.random.fold_in(key, bi))
                tot, lay = tot + a, lay + b
            tot, lay = np.asarray(tot), np.asarray(lay)
            row = dataclasses.asdict(c)
            row["loss"] = float(tot[0] / tot[1])
            row["tokens"] = float(tot[1])
            row["layer"] = {f: lay[:, j].tolist() for j, f in enumerate(LAYER_FIELDS)}
            results.append(row)
            if ci % 10 == 0:
                logger.info("config %d %s loss %.5f (%.0fs)", ci, c.name, row["loss"], time.time() - t0)
    out = dict(
        configs=results,
        group_cases=[list(c) for c in cases],
        granularities=list(GRANULARITIES),
        ranks=list(RANKS),
        num_triggers=NUM_TRIGGERS,
        eval_sequences=args.eval_sequences,
        calib_sequences=args.calib_sequences,
    )
    common.save_json(args.tag, "excl_skip", out)
    common.save_npz(
        args.tag,
        "excl_plan",
        perms=plan_host["perms"].astype(np.int16),
        trig=plan_host["trig"].astype(np.int16),
        lr_rms=plan_host["lr_rms"],
        trig_seen=seen.astype(np.float32),
        orders=np.asarray(ORDERS),
        **{f"group_{k}": v for k, v in gstate.items()},
        mass_bins=MASS_BINS,
    )
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/excl_skip_done.txt", "w") as f:
        f.write(json.dumps({"seconds": time.time() - t0, "configs": len(results)}))
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
