# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Routing problems of the d512 model: missed experts (routing regret), load-balancing distortion, router-expert
mismatch, topic units skipped on in-topic tokens, and n-gram units whose expert is not selected.

Sample phase, per ``BATCH``-sequence batch of a domain-tagged sample (unseen training sequences plus every Paloma
subset): the gradient ``g`` of the batch's summed next-token loss with respect to each layer's routed latent output
(``RoutingOverride.latent_delta``), then every one of the 384 experts evaluated densely on each token's routed input.
With ``r_e = sqrt(softplus(logit_e))``, the selected set ``S`` and ``s = 2.5 / sum_S r``, the linear loss effects
are (nats, positive = lower loss):

- an expert's gain ``-s r_e g . y_e``: for a selected expert what removing it would cost, for an unselected one
  what adding it at the weight it would get would save (``regret`` = best unselected minus worst selected gain;
  ``regret_fix`` uses a fixed weight, the median smallest combine weight, for the candidate);
- the renormalized swap of the worst selected expert ``k`` for an unselected ``e`` (``swap_gain``);
- restoring the unbiased top-10 ``U`` (``bias_lin`` = loss(U) - loss(S)).

"Strong" units are activations above ``atlas.HIGH_FRACTION`` of the unit's mean top-16 routed activation; an expert
"responds" to a token when at least one of its units is strong. Topic units (input file, from the atlas
categories) are strong-and-unrouted on in-topic tokens (their domain group, or their token set) when skipped.

Swap phase: end-to-end checks of the regret swaps, bias restores, topic swaps and no-op controls, each a copy of
its sequence with one (layer, position) routing override, scored against an unedited copy in the same batch.

N-gram phase: at every (a, b) occurrence of an eligible n-gram unit in fresh training sequences, whether the unit's
expert is selected, its rank, the unit's dense activation, and end-to-end swaps that add the expert where it was
skipped (replacing the smallest-weight slot) or remove it where it was selected (replaced by the 11th choice).

Bias phase: the sample's mean loss with the QB bias removed from selection on every layer and on each layer alone.
"""

import argparse
import dataclasses
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import atlas, common, forward
from experiments.grug.fast_track.model import _small_top_k

logger = logging.getLogger(__name__)

L, E, I, K, V, R = (
    forward.NUM_LAYERS,
    forward.NUM_EXPERTS,
    forward.EXPERT_WIDTH,
    forward.TOP_K,
    forward.VOCAB,
    forward.LATENT,
)
D = len(atlas.SOURCES)
G = len(atlas.GROUPS)
NT = atlas.TOP_TOKENS + 1
BATCH = 4
SWAP_ROWS = 16  # one unedited copy plus 15 edited ones per swap batch
CHUNK = 1024
NGRAM_BATCH = 8
NGRAM_MAX_MATCHES = 4096
LARGE_REGRET = 0.1
NEG = -1e9
STRONG_BINS = (1, 2, 4, 8, 16)  # n_strong buckets [0], [1], [2, 4), [4, 8), [8, 16), [16, inf)


class Kind:
    REGRET = 0
    BIAS = 1
    TOPIC = 2
    NOOP = 3
    NGRAM_ADD = 4
    NGRAM_REMOVE = 5


RECORD_FIELDS = (
    "regret",
    "regret_fix",
    "best_e",
    "best_gain",
    "worst_e",
    "worst_gain",
    "swap_e",
    "swap_gain",
    "bias_lin",
    "best_in_u",
    "swap_in_u",
    "ns_best",
    "ns_swap",
    "ns_worst",
    "topic_gain",
    "topic_e",
    "n_topic",
    "n_bias_only",
    "moe_gain",
)


def _layer_weights(model, layer: int):
    em = model.layers()[layer].mlp.expert_mlp
    return em.w_up, em.w_down  # [E, R, I], [E, I, R]


def init_acc() -> dict:
    z = jnp.zeros
    return dict(
        routed=z((L, E)),
        routed_resp=z((L, E)),
        unrouted_resp=z((L, E)),
        sel_negative=z((L, E)),
        sel_gain=z((L, E)),
        best_missed=z((L, E)),
        best_missed_regret=z((L, E)),
        worst_selected=z((L, E)),
        swap_useful=z((L, E)),
        latent_useful=z((L, E)),
        inserted=z((L, E)),
        displaced=z((L, E)),
        strong_hist=z((L, 2, len(STRONG_BINS) + 1)),
        resp_unrouted_tok=jnp.zeros((L, E, V), jnp.int32),
        wasted_tok=jnp.zeros((L, E, V), jnp.int32),
        missed_tok=jnp.zeros((L, E, V), jnp.int32),
        unit_strong_routed=z((L, E, I)),
        unit_strong_unrouted=z((L, E, I, D)),
        unit_topic_routed=z((L, E, I)),
        unit_topic_skipped=z((L, E, I)),
        unit_topic_gain=z((L, E, I)),
    )


def dense_chunk(c: dict, w_up, w_down, thr, topic_dom_g, topic_slot_t, slot_table, w_fix):
    """Per-token records and accumulator increments for one chunk of one layer (see the module docstring)."""
    x, g, logits, sel, unb, tok, valid = c["x"], c["g"], c["logits"], c["sel"], c["unb"], c["tok"], c["valid"]
    n = x.shape[0]
    rows = jnp.arange(n)[:, None]
    act = jnp.square(jax.nn.relu(jnp.einsum("tr,eri->tei", x, w_up, preferred_element_type=jnp.float32)))
    gw = jnp.einsum("eir,tr->tei", w_down, g.astype(w_down.dtype), preferred_element_type=jnp.float32)
    gy = jnp.sum(act.astype(w_down.dtype).astype(jnp.float32) * gw, axis=-1)  # [T, E]
    strong = act > thr[None]
    n_strong = jnp.sum(strong, axis=-1).astype(jnp.int32)  # [T, E]
    raw = jnp.sqrt(jnp.maximum(jax.nn.softplus(logits), 1e-30))
    rmask = jnp.zeros((n, E), bool).at[rows, sel].set(True)
    umask = jnp.zeros((n, E), bool).at[rows, unb].set(True)
    total = jnp.sum(raw * rmask, axis=-1)
    s = 2.5 / total
    gain = -(s[:, None] * raw) * gy
    best_e = jnp.argmax(jnp.where(rmask, NEG, gain), axis=-1)
    best = jnp.max(jnp.where(rmask, NEG, gain), axis=-1)
    worst_e = jnp.argmin(jnp.where(rmask, -NEG, gain), axis=-1)
    worst = jnp.min(jnp.where(rmask, -NEG, gain), axis=-1)
    gain_fix = -w_fix * gy
    regret_fix = jnp.max(jnp.where(rmask, NEG, gain_fix), axis=-1) - worst
    g_a = jnp.sum(raw * gy * rmask, axis=-1)
    take = lambda a, idx: jnp.take_along_axis(a, idx[:, None], axis=-1)[:, 0]  # noqa: E731
    raw_k, gy_k = take(raw, worst_e), take(gy, worst_e)
    new_total = total[:, None] - raw_k[:, None] + raw
    swap_delta = 2.5 / new_total * (g_a[:, None] - (raw_k * gy_k)[:, None] + raw * gy) - (s * g_a)[:, None]
    swap_gain = jnp.where(rmask, NEG, -swap_delta)
    swap_e = jnp.argmax(swap_gain, axis=-1)
    swap_best = jnp.max(swap_gain, axis=-1)
    total_u = jnp.sum(raw * umask, axis=-1)
    bias_lin = 2.5 / total_u * jnp.sum(raw * gy * umask, axis=-1) - s * g_a

    slot = slot_table[jnp.arange(E)[None, :], tok[:, None]].astype(jnp.int32)  # [T, E]
    in_topic = topic_dom_g[None] | topic_slot_t[jnp.arange(E)[None, :], slot]  # [T, E, I]
    vb = valid > 0
    topical = strong & in_topic & vb[:, None, None]
    flagged = ~rmask & jnp.any(topical, axis=-1)
    topic_gain = jnp.where(flagged, swap_gain, NEG)
    topic_e = jnp.argmax(topic_gain, axis=-1)
    topic_best = jnp.maximum(jnp.max(topic_gain, axis=-1), 0.0)

    rec = dict(
        regret=best - worst,
        regret_fix=regret_fix,
        best_e=best_e,
        best_gain=best,
        worst_e=worst_e,
        worst_gain=worst,
        swap_e=swap_e,
        swap_gain=swap_best,
        bias_lin=bias_lin,
        best_in_u=take(umask, best_e),
        swap_in_u=take(umask, swap_e),
        ns_best=take(n_strong, best_e),
        ns_swap=take(n_strong, swap_e),
        ns_worst=take(n_strong, worst_e),
        topic_gain=jnp.where(jnp.any(flagged, axis=-1), topic_best, 0.0),
        topic_e=jnp.where(jnp.any(flagged, axis=-1), topic_e, -1),
        n_topic=jnp.sum(flagged, axis=-1),
        n_bias_only=jnp.sum(rmask & ~umask, axis=-1),
        moe_gain=-s * g_a,
    )
    vf = vb.astype(jnp.float32)
    vr = rmask * vf[:, None]
    vu = (~rmask) * vf[:, None]
    resp = n_strong >= 1
    large = (rec["regret"] > LARGE_REGRET) & vb
    eidx = jnp.broadcast_to(jnp.arange(E)[None, :], (n, E))
    tidx = jnp.broadcast_to(tok[:, None], (n, E))
    bucket = jnp.searchsorted(jnp.asarray(STRONG_BINS), n_strong, side="right")
    strong_f = strong.astype(jnp.float32)
    topical_f = topical.astype(jnp.float32)
    inc = dict(
        routed=jnp.sum(vr, 0),
        routed_resp=jnp.sum(vr * resp, 0),
        unrouted_resp=jnp.sum(vu * resp, 0),
        sel_negative=jnp.sum(vr * (gain < 0), 0),
        sel_gain=jnp.sum(vr * gain, 0),
        best_missed=jnp.zeros((E,)).at[best_e].add(large),
        best_missed_regret=jnp.zeros((E,)).at[best_e].add(large * rec["regret"]),
        worst_selected=jnp.zeros((E,)).at[worst_e].add(large),
        swap_useful=jnp.sum(vu * (swap_gain > LARGE_REGRET), 0),
        latent_useful=jnp.sum(vu * resp * (swap_gain > LARGE_REGRET), 0),
        inserted=jnp.sum((rmask & ~umask) * vf[:, None], 0),
        displaced=jnp.sum((umask & ~rmask) * vf[:, None], 0),
        strong_hist=jnp.stack(
            [
                jnp.zeros((len(STRONG_BINS) + 1,)).at[bucket.reshape(-1)].add(vr.reshape(-1)),
                jnp.zeros((len(STRONG_BINS) + 1,)).at[bucket.reshape(-1)].add(vu.reshape(-1)),
            ]
        ),
        resp_unrouted_tok=jnp.zeros((E, V), jnp.int32).at[eidx, tidx].add((vu * resp).astype(jnp.int32)),
        wasted_tok=jnp.zeros((E, V), jnp.int32).at[eidx, tidx].add((vr * ~resp).astype(jnp.int32)),
        missed_tok=jnp.zeros((E, V), jnp.int32).at[best_e, tok].add(large.astype(jnp.int32)),
        unit_strong_routed=jnp.einsum("tei,te->ei", strong_f, vr),
        unit_strong_unrouted=jnp.einsum("tei,te->ei", strong_f, vu),
        unit_topic_routed=jnp.einsum("tei,te->ei", topical_f, rmask.astype(jnp.float32)),
        unit_topic_skipped=jnp.einsum("tei,te->ei", topical_f, (~rmask).astype(jnp.float32)),
        unit_topic_gain=jnp.einsum("tei,te->ei", topical_f, jnp.where(rmask, 0.0, jnp.maximum(swap_gain, 0.0))),
    )
    return rec, inc


def make_sample_step(tab: dict, w_fix: jax.Array):
    @jax.jit
    def step(acc, model, tokens, seg, weight, source, group):
        model = forward.replicate(model)
        t = tokens.shape[0] * tokens.shape[1]
        base = forward.no_override(t)

        def loss_fn(delta):
            tr = forward.trace_forward(model, tokens, seg, forward.no_ablation(), base._replace(latent_delta=delta))
            aux = dict(
                loss=tr["loss"],
                x=[lay["routed_input"] for lay in tr["layers"]],
                logits=[lay["router_logits"] for lay in tr["layers"]],
                sel=[lay["selected"] for lay in tr["layers"]],
            )
            return jnp.sum(tr["loss"] * weight), aux

        (_, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(base.latent_delta)
        tok = tokens.reshape(-1)
        valid = weight.reshape(-1)
        records = []
        new = dict(acc)
        for i in range(L):
            w_up, w_down = _layer_weights(model, i)
            _, unb = _small_top_k(aux["logits"][i], K)
            fields = dict(
                x=aux["x"][i], g=grad[i], logits=aux["logits"][i], sel=aux["sel"][i], unb=unb, tok=tok, valid=valid
            )
            chunks = jax.tree.map(lambda a: a.reshape(t // CHUNK, CHUNK, *a.shape[1:]), fields)
            topic_dom_g = tab["topic_dom"][i][:, :, group]  # [E, I]

            def body(carry, c, i=i, w_up=w_up, w_down=w_down, topic_dom_g=topic_dom_g):
                rec, inc = dense_chunk(
                    c, w_up, w_down, tab["thr"][i], topic_dom_g, tab["topic_slot_t"][i], tab["slot"][i], w_fix[i]
                )
                carry = jax.tree.map(jnp.add, carry, inc)
                return carry, rec

            first = jax.tree.map(lambda a: a[0], chunks)
            shapes = jax.eval_shape(
                lambda c, i=i, w_up=w_up, w_down=w_down, topic_dom_g=topic_dom_g: dense_chunk(
                    c, w_up, w_down, tab["thr"][i], topic_dom_g, tab["topic_slot_t"][i], tab["slot"][i], w_fix[i]
                )[1],
                first,
            )
            zero = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), shapes)
            inc, rec = jax.lax.scan(body, zero, chunks)
            records.append(jax.tree.map(lambda a: a.reshape(t, *a.shape[2:]), rec))
            for key in acc:
                if key == "unit_strong_unrouted":
                    new[key] = new[key].at[i, :, :, source].add(inc[key])
                else:
                    new[key] = new[key].at[i].add(inc[key])
            records[-1]["sel"] = aux["sel"][i]
            records[-1]["unb"] = unb
        records = {k: jnp.stack([r[k] for r in records]) for k in records[0]}
        return new, records, aux["loss"]

    return step


def smallest_weights(model, batch: common.TokenBatch, mesh) -> np.ndarray:
    """Per layer, the median over tokens of the smallest selected combine weight (the fixed candidate weight)."""

    @jax.jit
    def run(model, tokens, seg):
        tr = forward.trace_forward(forward.replicate(model), tokens, seg, forward.no_ablation())
        return jnp.stack([jnp.min(lay["combine"], axis=-1) for lay in tr["layers"]])

    tokens, seg, w = common.device_batch(batch, mesh)
    mins = np.asarray(run(model, tokens, seg))
    ok = np.asarray(w).reshape(-1) > 0
    return np.median(mins[:, ok], axis=1).astype(np.float32)


def make_swap_forward():
    """Losses ``[ROWS, S]`` of ``ROWS`` sequence copies, row ``j`` with its routing at (``layer[j]``, ``pos[j]``)
    replaced by ``new_sel[j]`` (``layer < 0``: unedited)."""

    @jax.jit
    def run(model, tokens, seg, layer, pos, new_sel):
        rows, s = tokens.shape
        base = forward.no_override(rows * s)
        flat = jnp.arange(rows) * s + pos  # [ROWS]
        hit = (jnp.arange(L)[:, None] == layer[None, :]) & (layer[None, :] >= 0)  # [L, ROWS]
        mask = jnp.zeros((L, rows * s), bool).at[:, flat].set(hit)
        override = jnp.zeros((L, rows * s, K), jnp.int32).at[:, flat].set(jnp.broadcast_to(new_sel, (L, rows, K)))
        routing = base._replace(sel_mask=mask, sel_override=override)
        tr = forward.trace_forward(forward.replicate(model), tokens, seg, forward.no_ablation(), routing)
        return tr["loss"]

    return run


@dataclasses.dataclass(frozen=True)
class Swap:
    seq: int  # index into the swap phase's sequence store
    layer: int
    pos: int
    new_sel: np.ndarray  # [K]
    kind: int
    ref: int  # the record (or n-gram match) index it belongs to


def _swap_batches(swaps: list[Swap]) -> list[list[tuple[int, int | None]]]:
    """Pack the swaps into ``SWAP_ROWS``-row batches of (sequence, swap index or None for the unedited copy): every
    sequence's swaps travel with an unedited copy of it in the same batch."""
    by_seq: dict[int, list[int]] = {}
    for j, sw in enumerate(swaps):
        by_seq.setdefault(sw.seq, []).append(j)
    batches, current = [], []
    for seq, idx in by_seq.items():
        for start in range(0, len(idx), SWAP_ROWS - 1):
            piece = [(seq, None), *((seq, j) for j in idx[start : start + SWAP_ROWS - 1])]
            if len(current) + len(piece) > SWAP_ROWS:
                batches.append(current)
                current = []
            current += piece
    if current:
        batches.append(current)
    return batches


def run_swaps(model, mesh, swaps: list[Swap], seqs: dict[int, common.TokenBatch]) -> np.ndarray:
    """``[len(swaps), 4]``: per swap, the loss change at the edited position, over it and the next 15, over the rest
    of the sequence from it on, and before it (should be 0), against an unedited copy in the same batch."""
    run = make_swap_forward()
    out = np.zeros((len(swaps), 4), np.float32)
    t0 = time.time()
    packed = _swap_batches(swaps)
    for n, rows in enumerate(packed):
        rows = rows + [rows[0]] * (SWAP_ROWS - len(rows))
        tokens = np.concatenate([seqs[q].tokens for q, _ in rows])
        seg = np.concatenate([seqs[q].segment_ids for q, _ in rows])
        layer = np.asarray([-1 if j is None else swaps[j].layer for _, j in rows], np.int32)
        pos = np.asarray([0 if j is None else swaps[j].pos for _, j in rows], np.int32)
        new_sel = np.stack([np.zeros(K, np.int32) if j is None else swaps[j].new_sel for _, j in rows]).astype(np.int32)
        with set_mesh(mesh):
            loss = np.asarray(
                run(
                    model,
                    jnp.asarray(tokens),
                    jnp.asarray(seg),
                    jnp.asarray(layer),
                    jnp.asarray(pos),
                    jnp.asarray(new_sel),
                )
            )
        base_row = {}
        for r, (q, j) in enumerate(rows):
            if j is None:
                base_row[q] = r
            else:
                weight = seqs[q].loss_weight[0] > 0
                diff = (loss[r] - loss[base_row[q]]) * weight
                p = swaps[j].pos
                out[j] = (diff[p], diff[p : p + 16].sum(), diff[p:].sum(), diff[:p].sum())
        if n % 100 == 0:
            logger.info("swap batch %d/%d %.0fs", n, len(packed), time.time() - t0)
    return out


def sample_batches(train_sequences: int, paloma_sequences: int, seed: int) -> list[common.TokenBatch]:
    return [
        *common.paloma_batches(BATCH, max_sequences=paloma_sequences),
        *common.train_batches(BATCH, train_sequences, seed=seed),
    ]


def as_rows(batches: list[common.TokenBatch]) -> list[common.TokenBatch]:
    """Each sequence of ``batches`` as its own one-row batch."""
    return [
        common.TokenBatch(b.source, b.tokens[r : r + 1], b.segment_ids[r : r + 1], b.loss_weight[r : r + 1])
        for b in batches
        for r in range(b.tokens.shape[0])
    ]


def choose_swaps(
    rec: dict, valid: np.ndarray, rng: np.random.Generator, n_top: int, n_random: int, seq_len: int
) -> list[Swap]:
    """Regret swaps (top ``swap_gain`` plus random), bias restores (top ``-bias_lin`` plus random among tokens whose
    selection the bias changed), topic swaps (top ``topic_gain``) and no-op controls, per layer."""
    s = seq_len
    swaps = []
    ok = np.flatnonzero(valid)
    for layer in range(L):

        def pick(score, eligible):
            cand = ok[eligible[ok]]
            if len(cand) == 0:
                return np.zeros(0, int)
            top = cand[np.argsort(-score[cand])[:n_top]]
            rest = np.setdiff1d(cand, top)
            rand = rng.choice(rest, size=min(n_random, len(rest)), replace=False) if len(rest) else np.zeros(0, int)
            return np.concatenate([top, rand])

        sel, unb = rec["sel"][layer], rec["unb"][layer]
        for t in pick(rec["swap_gain"][layer], np.ones(len(valid), bool)):
            new = sel[t].copy()
            new[new == rec["worst_e"][layer][t]] = rec["swap_e"][layer][t]
            swaps.append(Swap(t // s, layer, t % s, new, Kind.REGRET, t))
        for t in pick(-rec["bias_lin"][layer], rec["n_bias_only"][layer] > 0):
            swaps.append(Swap(t // s, layer, t % s, unb[t].copy(), Kind.BIAS, t))
        topical = np.flatnonzero((rec["topic_e"][layer] >= 0) & valid)
        for t in topical[np.argsort(-rec["topic_gain"][layer][topical])[:n_top]]:
            new = sel[t].copy()
            new[new == rec["worst_e"][layer][t]] = rec["topic_e"][layer][t]
            swaps.append(Swap(t // s, layer, t % s, new, Kind.TOPIC, t))
        for t in rng.choice(ok, size=min(n_top // 4, len(ok)), replace=False):
            swaps.append(Swap(t // s, layer, t % s, sel[t].copy(), Kind.NOOP, t))
    return swaps


def load_ngram_units(tag: str) -> dict[str, np.ndarray]:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/ngram_units.npz", "rb") as f:
        z = np.load(f)
        return {k: z[k] for k in z.files}


def load_topic_units(tag: str) -> dict[str, np.ndarray]:
    with fsspec.open(f"{common.OUTPUT_ROOT}/{tag}/topic_units.npz", "rb") as f:
        z = np.load(f)
        return {k: z[k] for k in z.files}


def ngram_matches(batch: common.TokenBatch, units: dict[str, np.ndarray], rng) -> np.ndarray:
    """``[M, 3]`` (row, position, unit) for every position whose (previous, current) token pair in the same document
    is an eligible unit's (a, b), capped at ``NGRAM_MAX_MATCHES``."""
    tok = batch.tokens.astype(np.int64)
    prev = np.pad(tok[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    same = np.pad(batch.segment_ids[:, 1:] == batch.segment_ids[:, :-1], ((0, 0), (1, 0)))
    key = np.where(same & (prev >= 0) & (batch.loss_weight > 0), prev * V + tok, -1)
    ukey = units["a"].astype(np.int64) * V + units["b"]
    order = np.argsort(ukey, kind="stable")
    sk = ukey[order]
    lo = np.searchsorted(sk, key.reshape(-1), side="left")
    hi = np.searchsorted(sk, key.reshape(-1), side="right")
    hits = np.flatnonzero(hi > lo)
    out = [(h // tok.shape[1], h % tok.shape[1], int(u)) for h in hits for u in order[lo[h] : hi[h]]]
    out = np.asarray(out, np.int64).reshape(-1, 3)
    if len(out) > NGRAM_MAX_MATCHES:
        out = out[np.sort(rng.choice(len(out), NGRAM_MAX_MATCHES, replace=False))]
    return out


def make_ngram_step():
    @jax.jit
    def run(model, tokens, seg, rows, pos, layer, expert, unit):
        model = forward.replicate(model)
        tr = forward.trace_forward(model, tokens, seg, forward.no_ablation())
        s = tokens.shape[1]
        flat = rows * s + pos
        out = {k: [] for k in ("selected", "rank", "weight", "act", "biased_rank", "unbiased_rank", "eleventh", "low")}
        for i, lay in enumerate(tr["layers"]):
            w_up, _ = _layer_weights(model, i)
            sel = lay["selected"][flat]  # [M, K]
            hit = sel == expert[:, None]
            comb = lay["combine"][flat]
            x = lay["routed_input"][flat]  # [M, R]
            col = w_up[expert, :, unit]  # [M, R]
            act = jnp.square(jax.nn.relu(jnp.sum(x.astype(jnp.float32) * col.astype(jnp.float32), axis=-1)))
            logits = lay["router_logits"][flat]
            biased = logits + lay["router_bias"][None]
            mine = jnp.take_along_axis(logits, expert[:, None], axis=-1)
            mine_b = jnp.take_along_axis(biased, expert[:, None], axis=-1)
            _, top11 = _small_top_k(biased, K + 1)
            out["selected"].append(jnp.any(hit, axis=-1))
            out["rank"].append(jnp.sum(jnp.where(hit, lay["weight_rank"][flat], 0), axis=-1))
            out["weight"].append(jnp.sum(jnp.where(hit, comb, 0.0), axis=-1))
            out["act"].append(act)
            out["biased_rank"].append(jnp.sum(biased > mine_b, axis=-1))
            out["unbiased_rank"].append(jnp.sum(logits > mine, axis=-1))
            out["eleventh"].append(top11[:, K])
            out["low"].append(jnp.take_along_axis(sel, jnp.argmin(comb, axis=-1)[:, None], axis=-1)[:, 0])
        picked = {k: jnp.take_along_axis(jnp.stack(v), layer[None, :], axis=0)[0] for k, v in out.items()}
        picked["sel"] = jnp.stack([lay["selected"][flat] for lay in tr["layers"]])  # [L, M, K]
        picked["loss"] = reshard(tr["loss"].reshape(-1), P(None))[flat]
        return picked

    return run


def make_bias_forward():
    @jax.jit
    def run(model, tokens, seg, weight, bias_scale):
        base = forward.no_override(tokens.shape[0] * tokens.shape[1])
        tr = forward.trace_forward(
            forward.replicate(model), tokens, seg, forward.no_ablation(), base._replace(bias_scale=bias_scale)
        )
        return jnp.sum(tr["loss"] * weight), jnp.sum(weight)

    return run


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--train-sequences", type=int, default=64)
    parser.add_argument("--paloma-sequences", type=int, default=4, help="per Paloma set")
    parser.add_argument("--ngram-sequences", type=int, default=1024)
    parser.add_argument("--swap-top", type=int, default=256)
    parser.add_argument("--swap-random", type=int, default=512)
    parser.add_argument("--ngram-swaps", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    tab_host = atlas.tables(atlas.load_stats(args.tag))
    topic = load_topic_units(args.tag)
    units = load_ngram_units(args.tag)
    t0 = time.time()
    out: dict[str, np.ndarray] = {}
    with set_mesh(mesh):
        model = common.compute_model(params)
        batches = sample_batches(args.train_sequences, args.paloma_sequences, args.seed)
        w_fix = smallest_weights(model, batches[-1], mesh)
        logger.info("fixed candidate weights %s", w_fix)
        tab = dict(
            thr=jnp.asarray(tab_host["thr"]),
            slot=jnp.asarray(tab_host["slot"]),
            topic_dom=jnp.asarray(topic["topic_dom"]),  # [L, E, I, G]
            topic_slot_t=jnp.asarray(np.swapaxes(topic["topic_slot"], 2, 3)),  # [L, E, NT, I]
        )
        step = make_sample_step(tab, jnp.asarray(w_fix))
        acc = init_acc()
        recs, losses = [], []
        for n, batch in enumerate(batches):
            tokens, seg, w = common.device_batch(batch, mesh)
            source = jnp.asarray(atlas.SOURCES.index(batch.source), jnp.int32)
            group = jnp.asarray(atlas.GROUPS.index(atlas.SOURCE_GROUP[batch.source]), jnp.int32)
            acc, rec, loss = step(acc, model, tokens, seg, w, source, group)
            recs.append(jax.tree.map(np.asarray, rec))
            losses.append(np.asarray(loss).reshape(-1))
            if n % 8 == 0:
                logger.info("sample batch %d/%d %.0fs", n, len(batches), time.time() - t0)
        rec = {k: np.concatenate([r[k] for r in recs], axis=1) for k in recs[0]}
        rows = as_rows(batches)
        valid = np.concatenate([b.loss_weight.reshape(-1) > 0 for b in rows])
        out.update({f"acc/{k}": np.asarray(v) for k, v in acc.items()})
        out.update({f"rec/{k}": v for k, v in rec.items()})
        out["sample/loss"] = np.concatenate(losses).astype(np.float32)
        out["sample/valid"] = valid
        out["sample/tokens"] = np.concatenate([b.tokens for b in rows]).astype(np.uint16)
        out["sample/segment_ids"] = np.concatenate([b.segment_ids for b in rows]).astype(np.int32)
        out["sample/sources"] = np.asarray([atlas.SOURCES.index(b.source) for b in rows], np.int8)
        out["w_fix"] = w_fix
        logger.info("sample phase done %.0fs", time.time() - t0)

        bias_run = make_bias_forward()
        configs = [np.ones(L, np.float32), np.zeros(L, np.float32)]
        for layer in range(L):
            c = np.ones(L, np.float32)
            c[layer] = 0.0
            configs.append(c)
        bias_loss = np.zeros((len(configs), 2))
        for batch in batches:
            tokens, seg, w = common.device_batch(batch, mesh)
            for j, c in enumerate(configs):
                total, count = bias_run(model, tokens, seg, w, jnp.asarray(c))
                bias_loss[j] += (float(total), float(count))
        out["bias/configs"] = np.stack(configs)
        out["bias/loss"] = bias_loss
        logger.info("bias phase done %.0fs: %s", time.time() - t0, bias_loss[:, 0] / bias_loss[:, 1])

        swaps = choose_swaps(rec, valid, rng, args.swap_top, args.swap_random, common.SEQ_LEN)
        seqs = dict(enumerate(rows))
        deltas = run_swaps(model, mesh, swaps, seqs)
        out["swap/kind"] = np.asarray([s.kind for s in swaps], np.int8)
        out["swap/layer"] = np.asarray([s.layer for s in swaps], np.int8)
        out["swap/ref"] = np.asarray([s.ref for s in swaps], np.int64)
        out["swap/new_sel"] = np.stack([s.new_sel for s in swaps]).astype(np.int16)
        out["swap/delta"] = deltas
        logger.info("swap phase done %.0fs (%d swaps)", time.time() - t0, len(swaps))
        common.save_npz(args.tag, "regret", **out)

        ngram_run = make_ngram_step()
        gram_batches = common.train_batches(NGRAM_BATCH, args.ngram_sequences, seed=args.seed + 1)
        found = {k: [] for k in ("seq", "pos", "unit", "selected", "rank", "weight", "act", "biased_rank")}
        found.update({k: [] for k in ("unbiased_rank", "eleventh", "low", "loss", "next")})
        gram_rows, gram_sel = [], []
        for n, batch in enumerate(gram_batches):
            m = ngram_matches(batch, units, rng)
            if len(m) == 0:
                continue
            pad = NGRAM_MAX_MATCHES - len(m)
            mm = np.concatenate([m, np.repeat(m[:1], pad, axis=0)])
            u = mm[:, 2]
            tokens, seg, _ = common.device_batch(batch, mesh)
            res = ngram_run(
                model,
                tokens,
                seg,
                *(
                    jnp.asarray(a, jnp.int32)
                    for a in (mm[:, 0], mm[:, 1], units["layer"][u], units["expert"][u], units["unit"][u])
                ),
            )
            res = {k: np.asarray(v)[..., : len(m)] if k != "sel" else np.asarray(v)[:, : len(m)] for k, v in res.items()}
            for k in ("selected", "rank", "weight", "act", "biased_rank", "unbiased_rank", "eleventh", "low", "loss"):
                found[k].append(res[k])
            sel_l = res["sel"][units["layer"][m[:, 2]], np.arange(len(m))]  # [M, K]
            gram_sel.append(sel_l)
            base = len(gram_rows)
            gram_rows += [
                common.TokenBatch(
                    batch.source, batch.tokens[r : r + 1], batch.segment_ids[r : r + 1], batch.loss_weight[r : r + 1]
                )
                for r in range(batch.tokens.shape[0])
            ]
            found["seq"].append(base + m[:, 0])
            found["pos"].append(m[:, 1])
            found["unit"].append(m[:, 2])
            nxt = np.pad(batch.tokens[:, 1:], ((0, 0), (0, 1)), constant_values=-1)
            found["next"].append(nxt[m[:, 0], m[:, 1]])
            if n % 20 == 0:
                logger.info("ngram batch %d %.0fs", n, time.time() - t0)
        found = {k: np.concatenate(v) for k, v in found.items()}
        gram_sel = np.concatenate(gram_sel)
        out.update({f"ngram/{k}": v for k, v in found.items()})
        hit_c = found["next"] == units["c"][found["unit"]]
        miss = np.flatnonzero(~found["selected"] & hit_c)
        kept = np.flatnonzero(found["selected"] & hit_c)
        miss = rng.choice(miss, size=min(args.ngram_swaps, len(miss)), replace=False) if len(miss) else miss
        kept = rng.choice(kept, size=min(args.ngram_swaps // 3, len(kept)), replace=False) if len(kept) else kept
        gswaps = []
        for j in miss:
            new = gram_sel[j].copy()
            new[new == found["low"][j]] = units["expert"][found["unit"][j]]
            gswaps.append(
                Swap(
                    int(found["seq"][j]),
                    int(units["layer"][found["unit"][j]]),
                    int(found["pos"][j]),
                    new,
                    Kind.NGRAM_ADD,
                    int(j),
                )
            )
        for j in kept:
            new = gram_sel[j].copy()
            new[new == units["expert"][found["unit"][j]]] = found["eleventh"][j]
            gswaps.append(
                Swap(
                    int(found["seq"][j]),
                    int(units["layer"][found["unit"][j]]),
                    int(found["pos"][j]),
                    new,
                    Kind.NGRAM_REMOVE,
                    int(j),
                )
            )
        gdeltas = run_swaps(model, mesh, gswaps, dict(enumerate(gram_rows)))
        out["ngram_swap/kind"] = np.asarray([s.kind for s in gswaps], np.int8)
        out["ngram_swap/ref"] = np.asarray([s.ref for s in gswaps], np.int64)
        out["ngram_swap/delta"] = gdeltas
        logger.info("ngram phase done %.0fs (%d matches, %d swaps)", time.time() - t0, len(found["unit"]), len(gswaps))
    common.save_npz(args.tag, "regret", **out)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
