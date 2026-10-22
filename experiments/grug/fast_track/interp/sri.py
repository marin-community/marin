# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Case study: how the d512 model predicts " L" (Lanka) after " S" "ri", with " Hong" -> " Kong" and " Los" ->
" Angeles" as comparison bigrams.

Positions ("queries") are the trigger token ("ri", " Hong", " Los") in unseen training-cache sequences and every
Paloma set, in categories (``CATEGORIES``): the target follows and the bigram is the document's first (visible) one or
a repeat, "S ri" followed by something else, and "ri" after any other token. At each query:

- direct logit attribution to the case target (" L" for every "ri" category): the final AttnRes mix with the final
  RMSNorm scale held fixed splits the centred pre-cap logit exactly into terms (embedding; per layer each head without
  its value embedding, the value embeddings, the shared expert, the routed experts); the routed and shared outputs are
  further split into every hidden unit at each of the MLP SConv's lags (activation x combine weight x the unit's
  direction through ``w_down``, ``w_latent_up``, the SConv tap and the final gain), summed per category on the device;
- gate and tuned lenses (``attrib.npz``) of the target's log-prob and rank at every sublayer boundary;
- for up to ``F`` focus neurons per case (``L5.E304.n0`` and each case's top-DLA neurons): the router logit, biased and
  unbiased rank, selection, combine weight and its rank, whether the QB bias alone selected it, the margin to the
  top-k boundary, the pre-activation and activation, the neuron's DLA;
- for the first two focus neurons, the router logit and the pre-activation decomposed (with the MLP input's RMSNorm
  and GatedNorm scales and the latent RMSNorm scale held fixed) into the sources of the layer's MLP AttnRes mix:
  embedding, every earlier layer's heads / value embeddings / shared expert / routed slots (per lag and expert), and
  the layer's own attention heads.

Interventions: zero / mean ablation of the focus neurons, their experts, the top-DLA neurons, sublayers and heads
(change in the target log-prob per query); replacing the " S" before "ri" with other tokens and putting " S" before
other "ri"; the trigger with only its last ``k`` tokens of context (packed as separate documents after a BOS).

Writes ``sri.npz``.
"""

import argparse
import dataclasses
import io
import json
import logging
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from haliax.partitioning import set_mesh

from experiments.grug.fast_track.interp import common, forward

logger = logging.getLogger(__name__)

BATCH = 8
Q = 64  # query slots per batch
F = 4  # focus neurons per query
NUM_DECOMP = 2  # focus neurons whose router logit and pre-activation are decomposed
TOP_NEURONS = 16
L, H, E, I, K = forward.NUM_LAYERS, forward.NUM_HEADS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.TOP_K
S = common.SEQ_LEN
D = forward.HIDDEN
R = forward.LATENT
MAX_SOURCES = L + 1
NUM_STAGES = 2 * L + 1
CONTEXT_LEFT = 24
TRUNC_LENGTHS = (2, 3, 4, 8, 16, 64, 256, 1024)
SWAP_PREFIXES = (" M", " T", " A", " P", " s", "F")
N0 = (5, 304, 0)
SRI_ROWS, HONG_ROWS, LOS_ROWS, RANDOM_ROWS = 1200, 120, 120, 64
RANDOM_EVERY = 400
ABLATE_TOP = 10


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    prefix: str | None  # the token before the trigger, if the case requires one
    trigger: str
    target: str


CASES = (Case("sri", " S", "ri", " L"), Case("hong", None, " Hong", " Kong"), Case("los", None, " Los", " Angeles"))
CATEGORIES = (
    "sri_first",
    "sri_repeat",
    "sri_other_next",
    "ri_other",
    "hong_first",
    "hong_repeat",
    "los_first",
    "los_repeat",
)
CAT_CASE = (0, 0, 0, 0, 1, 1, 2, 2)
POSITIVE_CATS = (0, 1, 4, 5, 6, 7)
C = len(CATEGORIES)


def term_names() -> list[str]:
    names = ["embed"]
    for i in range(L):
        names += [f"L{i}.h{h}" for h in range(H)] + [f"L{i}.ve", f"L{i}.shared", f"L{i}.routed"]
    return names


TERMS = term_names()
NT = len(TERMS)


@dataclasses.dataclass
class Row:
    source: str
    group: str
    tokens: np.ndarray
    segment_ids: np.ndarray
    loss_weight: np.ndarray
    queries: list[dict]


@dataclasses.dataclass(frozen=True)
class Ids:
    case_prefix: tuple[int, ...]  # -1: none
    case_trigger: tuple[int, ...]
    case_target: tuple[int, ...]
    bos: int


def resolve_ids(vocab: list[str]) -> Ids:
    index = {}
    for i, s in enumerate(vocab):
        index.setdefault(s, i)

    def tid(s: str) -> int:
        if s not in index:
            raise ValueError(f"{s!r} is not a single token")
        return index[s]

    return Ids(
        case_prefix=tuple(-1 if c.prefix is None else tid(c.prefix) for c in CASES),
        case_trigger=tuple(tid(c.trigger) for c in CASES),
        case_target=tuple(tid(c.target) for c in CASES),
        bos=tid("<|begin_of_text|>"),
    )


def row_queries(tokens: np.ndarray, seg: np.ndarray, weight: np.ndarray, ids: Ids) -> list[dict]:
    """Every query in one row: the trigger positions, their category, document offset and history."""
    out = []
    n = len(tokens)
    start = np.concatenate([[True], seg[1:] != seg[:-1]])
    doc_start = np.maximum.accumulate(np.where(start, np.arange(n), 0))
    for ci, case in enumerate(CASES):
        trig = np.flatnonzero(tokens[:-1] == ids.case_trigger[ci])
        seen: dict[int, int] = {}
        for t in trig:
            if weight[t] <= 0:
                continue
            ds = doc_start[t]
            prev = tokens[t - 1] if t > ds else -1
            nxt = tokens[t + 1] if seg[t + 1] == seg[t] else -1
            positive = nxt == ids.case_target[ci]
            if case.prefix is not None and prev != ids.case_prefix[ci]:
                if case.name != "sri":
                    continue
                cat = 3
            elif case.name == "sri":
                cat = (1 if seen.get(ds, 0) else 0) if positive else 2
            elif not positive:
                continue
            else:
                cat = 2 * ci + 2 + (1 if seen.get(ds, 0) else 0)
            if positive and cat != 3:
                seen[ds] = seen.get(ds, 0) + 1
            out.append(
                dict(
                    pos=int(t),
                    cat=cat,
                    doc_offset=int(t - ds),
                    doc_start_visible=bool(ds > 0 or tokens[ds] == ids.bos),
                    prior_mentions=int(seen.get(ds, 0) - (1 if positive and cat != 3 else 0)),
                )
            )
    return out


def row_group(queries: list[dict]) -> str | None:
    cats = {q["cat"] for q in queries}
    if cats & {0, 1, 2}:
        return "sri"
    if cats & {4, 5}:
        return "hong"
    if cats & {6, 7}:
        return "los"
    return None


def _keep_rows(batch: common.TokenBatch, ids: Ids, counts: dict, rand_ok: bool, offset: int) -> list[Row]:
    caps = {"sri": SRI_ROWS, "hong": HONG_ROWS, "los": LOS_ROWS, "rand": RANDOM_ROWS}
    kept = []
    for r in range(batch.tokens.shape[0]):
        if batch.loss_weight[r].sum() == 0:
            continue
        qs = row_queries(batch.tokens[r], batch.segment_ids[r], batch.loss_weight[r], ids)
        group = row_group(qs)
        if group is None and rand_ok and (offset + r) % RANDOM_EVERY == 0:
            group = "rand"
        if group is None or counts[group] >= caps[group]:
            continue
        counts[group] += 1
        kept.append(Row(batch.source, group, batch.tokens[r], batch.segment_ids[r], batch.loss_weight[r], qs))
    return kept


@common.off_mesh
def scan_train(start: int, num: int, chunk: int, ids: Ids, counts: dict):
    """Unseen contiguous training-cache sequences ``[start, start + num)`` (the cache is document-shuffled), keeping
    the rows with queries; stops once every group is full."""
    ds = common.data_config().train_sets(Axis("position", S), key=jax.random.PRNGKey(0))["train"].as_sync_dataset()
    t0 = time.time()
    end = min(start + num, len(ds))
    for s in range(start, end, chunk):
        idx = list(range(s, min(end, s + chunk)))
        batch = common.examples_to_batch("train", ds.get_batch(idx))
        yield from _keep_rows(batch, ids, counts, True, s)
        logger.info("scanned train %d/%d (%.0fs) kept %s", s + len(idx) - start, end - start, time.time() - t0, counts)
        if counts["sri"] >= SRI_ROWS and counts["hong"] >= HONG_ROWS and counts["los"] >= LOS_ROWS:
            return


def make_batches(rows: list[Row], rng: np.random.Generator) -> list[tuple[common.TokenBatch, np.ndarray, list[Row]]]:
    """Rows grouped by case in batches of ``BATCH`` with their query table ``[Q, 6]``: row, position, category,
    document offset, doc-start-visible, prior mentions (unused slots: row -1). Excess "ri_other" queries are dropped."""
    out = []
    for group in ("sri", "hong", "los", "rand"):
        g = [r for r in rows if r.group == group]
        for s in range(0, len(g), BATCH):
            chunk = g[s : s + BATCH]
            qs = [(bi, q) for bi, r in enumerate(chunk) for q in r.queries]
            main = [x for x in qs if x[1]["cat"] != 3]
            other = [x for x in qs if x[1]["cat"] == 3]
            room = Q - len(main)
            if room < 0:
                main = main[:Q]
                room = 0
            if len(other) > room:
                other = [other[j] for j in sorted(rng.choice(len(other), room, replace=False))]
            table = np.full((Q, 6), -1, np.int32)
            for j, (bi, q) in enumerate(main + other):
                table[j] = (bi, q["pos"], q["cat"], q["doc_offset"], int(q["doc_start_visible"]), q["prior_mentions"])
            pad = BATCH - len(chunk)
            tokens = np.stack([r.tokens for r in chunk] + [np.zeros(S, np.int32)] * pad)
            seg = np.stack([r.segment_ids for r in chunk] + [np.zeros(S, np.int32)] * pad)
            weight = np.stack([r.loss_weight for r in chunk] + [np.zeros(S, np.float32)] * pad)
            out.append((common.TokenBatch(group, tokens, seg, weight), table, chunk))
    return out


def _pq(x: jax.Array, qflat: jax.Array) -> jax.Array:
    """``[B, S, ...]`` at the flat query positions."""
    return x.reshape(-1, *x.shape[2:])[qflat]


def _stage_sources(tr) -> list[list]:
    blocks = tr["blocks"]
    stages = [[blocks[0]]]
    for i, lay in enumerate(tr["layers"]):
        stages.append([*blocks[: i + 1], lay["attn_out"]])
        stages.append(blocks[: i + 2])
    return stages


def _logp_rank(logits: jax.Array, y: jax.Array) -> tuple[jax.Array, jax.Array]:
    logp = jax.nn.log_softmax(forward.soft_cap(logits), axis=-1)
    ly = jnp.take_along_axis(logp, y[:, None], axis=-1)[:, 0]
    return ly, jnp.sum(logp > ly[:, None], axis=-1)


@jax.jit
def neuron_directions(model, targets: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``R[y, l, lag, e, i]``: routed unit ``(l, e, i)``'s centred logit for target ``y`` per unit activation and
    combine weight, through the MLP SConv tap ``lag`` and the final gain; ``Rs[y, l, lag, i]`` for the shared units."""
    m = forward.replicate(model)
    unembed = forward.unembed_direction(m)
    u = unembed[:, targets].T - jnp.mean(unembed, axis=1)[None]  # [Y, D]
    rs, rss = [], []
    for lay in m.layers():
        tap = lay.sconv_mlp.weight.astype(jnp.float32)  # [J, D]
        v = jnp.einsum("rd,jd,yd->yjr", lay.mlp.w_latent_up.astype(jnp.float32), tap, u)
        rs.append(jnp.einsum("eir,yjr->yjei", lay.mlp.expert_mlp.w_down.astype(jnp.float32), v))
        rss.append(jnp.einsum("id,jd,yd->yji", lay.shared[0].w_down.astype(jnp.float32), tap, u))
    return jnp.stack(rs, axis=1), jnp.stack(rss, axis=1)


def _stacked(model) -> dict:
    layers = model.layers()
    f32 = lambda x: x.astype(jnp.float32)  # noqa: E731
    return dict(
        router=jnp.stack([f32(lay.mlp.router) for lay in layers]),  # [L, D, E]
        router_bias=jnp.stack([f32(lay.mlp.router_bias) for lay in layers]),
        w_ld=jnp.stack([f32(lay.mlp.w_latent_down) for lay in layers]),  # [L, D, R]
        w_lu=jnp.stack([f32(lay.mlp.w_latent_up) for lay in layers]),  # [L, R, D]
        lat_gain=jnp.stack([f32(lay.mlp.latent_norm.gain()) for lay in layers]),
        w_up=jnp.stack([lay.mlp.expert_mlp.w_up for lay in layers]),  # [L, E, R, I]
        w_down=jnp.stack([lay.mlp.expert_mlp.w_down for lay in layers]),  # [L, E, I, R]
        mlp_gain=jnp.stack([f32(lay.rms_mlp.gain()) for lay in layers]),
        gn_down=jnp.stack([f32(lay.mlp_gated_norm.w_down) for lay in layers]),
        gn_up=jnp.stack([f32(lay.mlp_gated_norm.w_up) for lay in layers]),
        tap=jnp.stack([f32(lay.sconv_mlp.weight) for lay in layers]),  # [L, J, D]
    )


def analyse(model, tokens, seg, qflat, qmask, qy, qyi, qcat, focus, tuned, rdir, rsdir):
    """Everything measured at the queries of one batch (see the module docstring). ``focus`` is ``[Q, F, 3]``
    ``(layer, expert, unit)``."""
    eps = model.config.layer_norm_eps
    m = forward.replicate(model)
    tr = forward.replicate(forward.trace_forward(m, tokens, seg, forward.no_ablation()))
    st = _stacked(m)
    nlag = st["tap"].shape[1]
    seg_flat = seg.reshape(-1)
    pos = qflat % S
    lags = jnp.arange(nlag)
    lag_flat = jnp.maximum(qflat[:, None] - lags[None], 0)  # [Q, J]
    lag_ok = ((pos[:, None] - lags[None]) >= 0) & (seg_flat[lag_flat] == seg_flat[qflat][:, None])
    lag_okf = lag_ok.astype(jnp.float32)
    qm = qmask.astype(jnp.float32)
    unembed = forward.unembed_direction(m)
    u_q = unembed.T[qy] - jnp.mean(unembed, axis=1)[None]  # [Q, D]
    inv_q = _pq(tr["inv_rms"], qflat)
    wf_q = jnp.stack([_pq(w, qflat) for w in tr["w_final"]])  # [7, Q]
    scale = wf_q * inv_q[None]  # [7, Q]
    out: dict = {}

    # Per-layer components at the queries.
    comps = []
    neuron_acc = {k: [] for k in ("dla", "act", "fire")}
    shared_acc = []
    top_vals, top_meta = [], []
    expert_dla = []
    recon = []
    qidx = jnp.arange(Q)
    for li, lay in enumerate(tr["layers"]):
        heads_full = _pq(lay["head_out"], qflat).astype(jnp.float32)
        ve_heads = _pq(lay["ve_head_out"], qflat).astype(jnp.float32)
        lat = lay["slot_latent_out"][lag_flat]  # [Q, J, K, R]
        used = lay["used_combine"][lag_flat]  # [Q, J, K]
        slot_vec = jnp.einsum("qjkr,qjk,rd->qjkd", lat, used, st["w_lu"][li]) * (st["tap"][li][None, :, None, :])
        slot_vec = slot_vec * lag_okf[..., None, None]
        slot_expert = lay["selected"][lag_flat]  # [Q, J, K]
        routed_q = _pq(lay["routed_out"], qflat).astype(jnp.float32)
        recon.append(jnp.max(jnp.abs(jnp.sum(slot_vec, axis=(1, 2)) - routed_q) * qm[:, None]))
        comps.append(
            dict(
                heads=heads_full - ve_heads,
                ve=jnp.sum(ve_heads, axis=1),
                shared=_pq(lay["shared_out"], qflat).astype(jnp.float32),
                slots=slot_vec,
                slot_expert=slot_expert,
            )
        )
        # Routed units: each (lag, slot) assignment's activations.
        index = lay["token_sorted"] * K + lay["slot_sorted"]
        inv = jnp.zeros((index.shape[0],), jnp.int32).at[index].set(jnp.arange(index.shape[0], dtype=jnp.int32))
        rows = inv[lag_flat[..., None] * K + jnp.arange(K)]  # [Q, J, K]
        act = lay["expert_act"][rows].astype(jnp.float32) * lag_okf[..., None, None]  # [Q, J, K, I]
        dir_q = rdir[qyi[:, None, None], li, lags[None, :, None], slot_expert]  # [Q, J, K, I]
        ndla = act * used[..., None] * dir_q * scale[li + 1][:, None, None, None]
        w_c = jax.nn.one_hot(qcat, C) * qm[:, None]  # [Q, C]
        ce = jnp.broadcast_to(qcat[:, None, None], slot_expert.shape)
        wq = jnp.broadcast_to(qm[:, None, None], slot_expert.shape)[..., None]
        neuron_acc["dla"].append(jnp.zeros((C, E, I)).at[ce, slot_expert].add(ndla * wq))
        a0 = act[:, 0]
        neuron_acc["act"].append(jnp.zeros((C, E, I)).at[ce[:, 0], slot_expert[:, 0]].add(a0 * wq[:, 0]))
        neuron_acc["fire"].append(
            jnp.zeros((C, E, I)).at[ce[:, 0], slot_expert[:, 0]].add((a0 > 0).astype(jnp.float32) * wq[:, 0])
        )
        sact = lay["shared_act"].reshape(-1, I)[lag_flat].astype(jnp.float32) * lag_okf[..., None]  # [Q, J, I]
        sdla = jnp.sum(sact * rsdir[qyi, li] * scale[li + 1][:, None, None], axis=1)  # [Q, I]
        shared_acc.append(jnp.einsum("qc,qi->ci", w_c, sdla))
        expert_dla.append(jnp.zeros((Q, E)).at[qidx[:, None, None], slot_expert].add(jnp.sum(ndla, axis=-1)))
        flat = ndla.reshape(Q, -1)
        tv, ti = jax.lax.top_k(flat, TOP_NEURONS)
        j_, k_, i_ = jnp.unravel_index(ti, (nlag, K, I))
        top_vals.append(tv)
        top_meta.append(jnp.stack([jnp.full_like(ti, li), j_, slot_expert[qidx[:, None], j_, k_], i_], axis=-1))
    tv = jnp.concatenate(top_vals, axis=1)
    tm = jnp.concatenate(top_meta, axis=1)
    tv, pick = jax.lax.top_k(tv, TOP_NEURONS)
    out["top_val"] = tv
    out["top_meta"] = jnp.take_along_axis(tm, pick[..., None], axis=1)  # (layer, lag, expert, unit)
    out["routed_recon_err"] = jnp.max(jnp.stack(recon))
    out["expert_dla"] = jnp.stack(expert_dla, axis=1)  # [Q, L, E]
    acc = {f"neuron_{k}": jnp.stack(v, axis=1) for k, v in neuron_acc.items()}  # [C, L, E, I]
    acc["shared_dla"] = jnp.stack(shared_acc, axis=1)  # [C, L, I]

    # Final-logit DLA terms.
    def dla(vec, src):
        return jnp.sum(vec * u_q, axis=-1) * scale[src]

    terms = [dla(_pq(tr["embed"], qflat).astype(jnp.float32), 0)]
    for li, cp in enumerate(comps):
        terms += [dla(cp["heads"][:, h], li + 1) for h in range(H)]
        terms += [dla(cp["ve"], li + 1), dla(cp["shared"], li + 1), dla(jnp.sum(cp["slots"], axis=(1, 2)), li + 1)]
    out["dla_terms"] = jnp.stack(terms, axis=1)  # [Q, NT]
    hidden_q = _pq(tr["hidden"], qflat).astype(jnp.float32)
    raw = jnp.sum(hidden_q * u_q / forward.final_gain(m), axis=-1)
    out["dla_raw"] = raw
    out["dla_recon_err"] = jnp.max(jnp.abs(jnp.sum(out["dla_terms"], axis=1) - raw) * qm)

    # Output and lenses.
    logits = jnp.einsum("qd,dv->qv", hidden_q, m.output_proj.astype(jnp.float32))
    out["logp_y"], out["rank_y"] = _logp_rank(logits, qy)
    labels = _pq(tr["labels"], qflat)
    out["logp_label"], _ = _logp_rank(logits, labels)
    out["top1"] = jnp.argmax(logits, axis=-1)
    out["entropy"] = _pq(tr["entropy"], qflat)
    final_q = m.attn_res_query_final
    lens = {k: [] for k in ("gate_logp", "gate_rank", "tuned_logp", "tuned_rank")}
    for k, srcs in enumerate(_stage_sources(tr)):
        _, mixed = forward._gate(srcs, final_q, eps)
        mq = _pq(mixed, qflat)
        h_gate = m.final_norm(mq)
        tuned_in = jnp.concatenate([forward.rms_norm(mq.astype(jnp.float32), eps), jnp.ones((Q, 1))], axis=-1)
        h_tuned = (tuned_in @ tuned[k]).astype(mq.dtype)
        for kind, h in (("gate", h_gate), ("tuned", h_tuned)):
            lg = jnp.einsum("qd,dv->qv", h.astype(jnp.float32), m.output_proj.astype(jnp.float32))
            lp, rk = _logp_rank(lg, qy)
            lens[f"{kind}_logp"].append(lp)
            lens[f"{kind}_rank"].append(rk)
    for k, v in lens.items():
        out[f"lens_{k}"] = jnp.stack(v, axis=1)

    # Focus neurons: routing, activation, DLA.
    fl, fe, fi = focus[..., 0], focus[..., 1], focus[..., 2]  # [Q, F]
    logits_all = jnp.stack([lay["router_logits"] for lay in tr["layers"]])  # [L, T, E]
    sel_all = jnp.stack([lay["selected"] for lay in tr["layers"]])  # [L, T, K]
    comb_all = jnp.stack([lay["combine"] for lay in tr["layers"]])
    wrank_all = jnp.stack([lay["weight_rank"] for lay in tr["layers"]])
    rin_all = jnp.stack([lay["routed_input"] for lay in tr["layers"]]).astype(jnp.float32)  # [L, T, R]
    qf = jnp.broadcast_to(qflat[:, None], fl.shape)
    lg = logits_all[fl, qf]  # [Q, F, E]
    biased = lg + st["router_bias"][fl]
    lg_e = jnp.take_along_axis(lg, fe[..., None], axis=-1)[..., 0]
    b_e = jnp.take_along_axis(biased, fe[..., None], axis=-1)[..., 0]
    out["f_logit"] = lg_e
    out["f_rank"] = jnp.sum(lg > lg_e[..., None], axis=-1)
    out["f_biased_rank"] = jnp.sum(biased > b_e[..., None], axis=-1)
    others = jnp.where(jnp.arange(E) == fe[..., None], -jnp.inf, biased)
    out["f_margin"] = b_e - jax.lax.top_k(others, K)[0][..., K - 1]
    sel = sel_all[fl, qf]  # [Q, F, K]
    hit = sel == fe[..., None]
    out["f_selected"] = jnp.any(hit, axis=-1)
    out["f_weight"] = jnp.sum(jnp.where(hit, comb_all[fl, qf], 0.0), axis=-1)
    out["f_weight_rank"] = jnp.where(out["f_selected"], jnp.sum(jnp.where(hit, wrank_all[fl, qf], 0), axis=-1), -1)
    out["f_bias_driven"] = out["f_selected"] & (out["f_rank"] >= K)
    w_up_col = jnp.take_along_axis(st["w_up"][fl, fe].astype(jnp.float32), fi[..., None, None], axis=-1)[
        ..., 0
    ]  # [Q, F, R]
    pre = jnp.einsum("qfr,qfr->qf", rin_all[fl, qf], w_up_col)
    out["f_pre"] = pre
    act_f = jnp.square(jax.nn.relu(pre))
    out["f_act"] = act_f
    w_down_row = jnp.take_along_axis(st["w_down"][fl, fe].astype(jnp.float32), fi[..., None, None], axis=-2)[
        ..., 0, :
    ]  # [Q, F, R]
    lu = jnp.einsum("qfr,qfrd->qfd", w_down_row, st["w_lu"][fl]) * st["tap"][fl, 0]
    out["f_dla"] = (
        out["f_weight"] * act_f * jnp.einsum("qfd,qd->qf", lu, u_q) * jnp.take_along_axis(scale.T, fl + 1, axis=1)
    )

    # Decomposition of the first focus neurons' router logit and pre-activation over the MLP mix's sources.
    wm = jnp.stack(
        [jnp.pad(lay["w_mlp"], ((0, MAX_SOURCES - lay["w_mlp"].shape[0]), (0, 0), (0, 0))) for lay in tr["layers"]]
    )  # [L, 7, B, S]
    h2_all = jnp.stack([lay["mlp_gate_in"] for lay in tr["layers"]])  # [L, B, S, D]
    mi_all = jnp.stack([lay["mlp_in"] for lay in tr["layers"]])
    embed_q = _pq(tr["embed"], qflat).astype(jnp.float32)
    lat_eps = model.layers()[0].mlp.latent_norm.eps
    mlp_eps = model.layers()[0].rms_mlp.eps
    dec = {k: [] for k in ("router_terms", "router_slots", "pre_terms", "pre_slots", "router_actual", "pre_actual")}
    m_err = []
    for d in range(NUM_DECOMP):
        dl, de, di = fl[:, d], fe[:, d], fi[:, d]
        w_s = wm[dl, :, qflat // S, pos]  # [Q, 7]
        h2 = h2_all[dl, qflat // S, pos].astype(jnp.float32)  # [Q, D]
        inv = jax.lax.rsqrt(jnp.mean(jnp.square(h2), axis=-1, keepdims=True) + mlp_eps)
        normed = h2 * inv * st["mlp_gain"][dl]
        gate = jax.nn.sigmoid(
            jnp.einsum("qr,qrd->qd", jax.nn.silu(jnp.einsum("qd,qdr->qr", normed, st["gn_down"][dl])), st["gn_up"][dl])
        )
        mult = inv * st["mlp_gain"][dl] * gate  # [Q, D]
        mi = mi_all[dl, qflat // S, pos].astype(jnp.float32)
        m_err.append(jnp.max(jnp.abs(mult * h2 - mi) * qm[:, None]))
        dir_router = st["router"][dl, :, de]  # [Q, D]
        lat_raw = jnp.einsum("qd,qdr->qr", mi, st["w_ld"][dl])
        inv_lat = jax.lax.rsqrt(jnp.mean(jnp.square(lat_raw), axis=-1, keepdims=True) + lat_eps)
        col = jnp.take_along_axis(st["w_up"][dl, de].astype(jnp.float32), di[:, None, None], axis=-1)[..., 0]
        dir_pre = jnp.einsum("qdr,qr->qd", st["w_ld"][dl], st["lat_gain"][dl] * col) * inv_lat
        dec["router_actual"].append(logits_all[dl, qflat, de])
        dec["pre_actual"].append(jnp.einsum("qr,qr->q", rin_all[dl, qflat], col))

        def project(vec, weight, dvec, mult=mult):
            return jnp.sum(vec * mult * dvec, axis=-1) * weight

        for name, dvec in (("router", dir_router), ("pre", dir_pre)):
            t_list = [project(embed_q, w_s[:, 0], dvec)]
            slot_list = []
            for li, cp in enumerate(comps):
                below = (li < dl).astype(jnp.float32)
                own = (li == dl).astype(jnp.float32)
                w_l = w_s[:, jnp.minimum(li + 1, MAX_SOURCES - 1)] * (below + own)
                t_list += [project(cp["heads"][:, h], w_l, dvec) for h in range(H)]
                t_list.append(project(cp["ve"], w_l, dvec))
                w_b = w_s[:, jnp.minimum(li + 1, MAX_SOURCES - 1)] * below
                t_list.append(project(cp["shared"], w_b, dvec))
                sl = jnp.einsum("qjkd,qd->qjk", cp["slots"], mult * dvec) * w_b[:, None, None]
                slot_list.append(sl)
                t_list.append(jnp.sum(sl, axis=(1, 2)))
            dec[f"{name}_terms"].append(jnp.stack(t_list, axis=1))  # [Q, NT]
            dec[f"{name}_slots"].append(jnp.stack(slot_list, axis=1))  # [Q, L, J, K]
    for k, v in dec.items():
        out[f"dec_{k}"] = jnp.stack(v, axis=1)
    out["dec_mult_err"] = jnp.max(jnp.stack(m_err))
    out["slot_expert"] = jnp.stack([cp["slot_expert"] for cp in comps], axis=1)  # [Q, L, J, K]

    # L5 E304 traffic and n0 by current token (all valid tokens of the batch).
    lay5 = tr["layers"][N0[0]]
    tok_flat = tokens.reshape(-1)
    rows304 = lay5["expert_sorted"] == N0[1]
    t304 = tok_flat[lay5["token_sorted"]]
    a304 = lay5["expert_act"][:, N0[2]].astype(jnp.float32)
    r304 = rows304.astype(jnp.float32)
    acc["e304_count"] = jnp.zeros((forward.VOCAB,)).at[t304].add(r304)
    acc["n0_fire"] = jnp.zeros((forward.VOCAB,)).at[t304].add(r304 * (a304 > 0))
    acc["n0_act"] = jnp.zeros((forward.VOCAB,)).at[t304].add(r304 * a304)
    acc["token_count"] = jnp.zeros((forward.VOCAB,)).at[tok_flat].add(1.0)
    return out, acc


analyse_jit = jax.jit(analyse)


@jax.jit
def ablated_logp(model, tokens, seg, qflat, qy, abl):
    m = forward.replicate(model)
    tr = forward.replicate(forward.trace_forward(m, tokens, seg, abl))
    hidden_q = _pq(tr["hidden"], qflat)
    logits = jnp.einsum("qd,dv->qv", hidden_q.astype(jnp.float32), m.output_proj.astype(jnp.float32))
    return _logp_rank(logits, qy)[0]


@jax.jit
def ve_mean_of(model, tokens, seg, weight):
    tr = forward.trace_forward(forward.replicate(model), tokens, seg, forward.no_ablation())
    return jnp.stack(
        [jnp.einsum("bshd,bs->hd", tr["layers"][li]["ve"].astype(jnp.float32), weight) for li in forward.MLA_LAYERS]
    ), jnp.sum(weight)


def query_arrays(table: np.ndarray, ids: Ids) -> dict[str, np.ndarray]:
    valid = table[:, 0] >= 0
    cat = np.where(valid, table[:, 2], 0)
    yi = np.asarray(CAT_CASE)[cat]
    y = np.asarray(ids.case_target)[yi]
    qflat = np.where(valid, table[:, 0] * S + table[:, 1], 0)
    return dict(qflat=qflat, qmask=valid, qy=y, qyi=yi, qcat=cat)


def focus_table(cat: np.ndarray, focus_by_case: np.ndarray) -> np.ndarray:
    return focus_by_case[np.asarray(CAT_CASE)[cat]].astype(np.int32)  # [Q, F, 3]


def truncated_batches(
    rows_q: list[tuple[np.ndarray, int, int, int]], bos: int
) -> list[tuple[common.TokenBatch, np.ndarray]]:
    """Windows (each the last ``k`` tokens of a document ending at a query, after a BOS unless the window reaches the
    document start), packed as separate documents. ``rows_q``: (row tokens, row segment start, position, query id).
    Returns batches and their query tables ``[Q, 3]``: flat position, query id, window length index."""
    windows = []
    for tokens, doc_start, t, qid in rows_q:
        for ki, k in enumerate(TRUNC_LENGTHS):
            s0 = max(doc_start, t - k + 1)
            w = tokens[s0 : t + 1]
            if s0 > doc_start or tokens[doc_start] != bos:
                w = np.concatenate([[bos], w])
            windows.append((w.astype(np.int32), qid, ki))
    windows.sort(key=lambda x: -len(x[0]))
    seqs: list[list] = []
    for w in windows:
        for sq in seqs:
            if sq[0] + len(w[0]) <= S and len(sq[1]) < Q // BATCH:
                sq[0] += len(w[0])
                sq[1].append(w)
                break
        else:
            seqs.append([len(w[0]), [w]])
    out = []
    for s in range(0, len(seqs), BATCH):
        chunk = seqs[s : s + BATCH]
        tokens = np.zeros((BATCH, S), np.int32)
        seg = np.zeros((BATCH, S), np.int32)
        table = np.full((Q, 3), -1, np.int32)
        j = 0
        for bi, (_, ws) in enumerate(chunk):
            off = 0
            for wi, (w, qid, ki) in enumerate(ws):
                tokens[bi, off : off + len(w)] = w
                seg[bi, off : off + len(w)] = wi + 1
                if j < Q:
                    table[j] = (bi * S + off + len(w) - 1, qid, ki)
                    j += 1
                off += len(w)
            seg[bi, off:] = len(ws) + 1
        out.append((common.TokenBatch("trunc", tokens, seg, np.ones((BATCH, S), np.float32)), table))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scan-train", type=int, default=200000, help="unseen train sequences to scan at most")
    parser.add_argument("--scan-chunk", type=int, default=512)
    parser.add_argument("--swap-batches", type=int, default=24)
    args = parser.parse_args()
    t0 = time.time()
    cfg = common.model_config()
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/attrib.npz", "rb") as f:
        attrib = np.load(io.BytesIO(f.read()))
        tuned = jnp.asarray(attrib["tuned"], jnp.float32)
        head_mean, sub_mean = jnp.asarray(attrib["head_mean"]), jnp.asarray(attrib["sub_mean"])
    with fsspec.open(f"{common.OUTPUT_ROOT}/{args.tag}/stats.npz", "rb") as f:
        stats = np.load(io.BytesIO(f.read()))
        unit_mean = stats["act_sum"] / np.maximum(stats["expert_tokens"][..., None], 1.0)  # mean act when routed
    vocab = common.vocab_strings()
    ids = resolve_ids(vocab)
    logger.info("ids %s (%.0fs)", ids, time.time() - t0)

    # Scan: every Paloma set, then unseen training sequences.
    counts = {"sri": 0, "hong": 0, "los": 0, "rand": 0}
    rows: list[Row] = []
    for b in common.paloma_batches(64):
        rows += _keep_rows(b, ids, counts, False, 0)
    logger.info("paloma kept %s (%.0fs)", counts, time.time() - t0)
    rows += scan_train(common.TRAINED_SEQUENCES, args.scan_train, args.scan_chunk, ids, counts)
    logger.info("scan done: %s rows (%.0fs)", counts, time.time() - t0)
    rng = np.random.default_rng(0)
    batches = make_batches(rows, rng)
    logger.info("%d batches", len(batches))

    targets = jnp.asarray(ids.case_target)
    with set_mesh(mesh):
        model = common.compute_model(params)
        rdir, rsdir = neuron_directions(model, targets)

        def run_pass(focus_by_case: np.ndarray, which=None) -> tuple[dict, dict]:
            per_q: dict[str, list] = {}
            acc: dict = {}
            for bi, (b, table, _) in enumerate(batches):
                if which is not None and bi not in which:
                    continue
                qa = query_arrays(table, ids)
                tokens, seg, _ = common.device_batch(b, mesh)
                out, a = analyse_jit(
                    model,
                    tokens,
                    seg,
                    jnp.asarray(qa["qflat"]),
                    jnp.asarray(qa["qmask"]),
                    jnp.asarray(qa["qy"]),
                    jnp.asarray(qa["qyi"]),
                    jnp.asarray(qa["qcat"]),
                    jnp.asarray(focus_table(qa["qcat"], focus_by_case)),
                    tuned,
                    rdir,
                    rsdir,
                )
                keep = qa["qmask"]
                for k, v in out.items():
                    v = np.asarray(v)
                    per_q.setdefault(k, []).append(v[keep] if v.ndim and v.shape[0] == Q else v[None])
                for k, v in a.items():
                    acc[k] = v if k not in acc else acc[k] + v
                if bi % 10 == 0:
                    logger.info("pass batch %d/%d (%.0fs)", bi, len(batches), time.time() - t0)
            return {k: np.concatenate(v) for k, v in per_q.items()}, {k: np.asarray(v) for k, v in acc.items()}

        # Pass 1: neuron DLA sums with L5.E304.n0 as the focus everywhere.
        default_focus = np.tile(np.asarray(N0, np.int32)[None, None], (len(CASES), F, 1))
        _, acc1 = run_pass(default_focus)
        logger.info("pass 1 done (%.0fs)", time.time() - t0)
        n_cat = np.bincount(np.concatenate([t[:, 2][t[:, 0] >= 0] for _, t, _ in batches]), minlength=C).astype(
            np.float64
        )
        mean_dla = acc1["neuron_dla"] / np.maximum(n_cat, 1)[:, None, None, None]  # [C, L, E, I]
        focus_by_case = np.zeros((len(CASES), F, 3), np.int32)
        top_by_case = {}
        for ci in range(len(CASES)):
            cats = [c for c in POSITIVE_CATS if CAT_CASE[c] == ci]
            score = np.einsum("c,clei->lei", n_cat[cats], mean_dla[cats]) / n_cat[cats].sum()
            order = np.argsort(-score.reshape(-1))[:50]
            top = np.stack(np.unravel_index(order, score.shape), axis=1)
            top_by_case[ci] = top
            chosen = [N0] if ci == 0 else []
            for t in top:
                if len(chosen) >= F:
                    break
                if tuple(int(x) for x in t) not in chosen:
                    chosen.append(tuple(int(x) for x in t))
            focus_by_case[ci] = np.asarray(chosen, np.int32)
            logger.info("case %s focus %s", CASES[ci].name, chosen)

        # Pass 2: per-query everything with the case focus neurons.
        per_q, acc2 = run_pass(focus_by_case)
        logger.info("pass 2 done (%.0fs)", time.time() - t0)
        meta = np.concatenate([t[t[:, 0] >= 0] for _, t, _ in batches])
        src, ctx, nxt = [], [], []
        for _, t, chunk in batches:
            for r in t[t[:, 0] >= 0]:
                row = chunk[r[0]]
                src.append(row.source)
                ctx.append(
                    np.pad(row.tokens[max(0, r[1] - CONTEXT_LEFT + 1) : r[1] + 1], (CONTEXT_LEFT, 0))[-CONTEXT_LEFT:]
                )
                nxt.append(np.pad(row.tokens[r[1] + 1 : r[1] + 5], (0, 4))[:4])

        # Ablations.
        ve_sum = np.zeros((len(forward.MLA_LAYERS), H, forward.HEAD_DIM))
        n_ve = 0.0
        for b, _, _ in batches:
            if b.source != "rand":
                continue
            tokens, seg, w = common.device_batch(b, mesh)
            vs, n = ve_mean_of(model, tokens, seg, w)
            ve_sum += np.asarray(vs)
            n_ve += float(n)
        ve_mean = np.zeros((L, H, forward.HEAD_DIM), np.float32)
        for j, li in enumerate(forward.MLA_LAYERS):
            ve_mean[li] = ve_sum[j] / max(n_ve, 1.0)
        base = forward.no_ablation()
        nmean = jnp.asarray(unit_mean, jnp.float32)

        def units(spec, mode):
            mult = base.neuron_mult
            for li, e, i in spec:
                mult = mult.at[li, e, i].set(0.0)
            return base._replace(neuron_mult=mult, neuron_mean=nmean if mode == "mean" else base.neuron_mean)

        def expert(li, e, mode):
            return units([(li, e, i) for i in range(I)], mode)

        def sub(li, slot, mode):
            return base._replace(
                sub_mult=base.sub_mult.at[li, slot].set(0.0), sub_mean=sub_mean if mode == "mean" else base.sub_mean
            )

        def head(li, h):
            return base._replace(head_mult=base.head_mult.at[li, h].set(0.0), head_mean=head_mean)

        configs = [("none", "all", base)]
        for mode in ("zero", "mean"):
            configs.append((f"n0:{mode}", "sri+rand", units([N0], mode)))
            configs.append((f"E304:{mode}", "sri+rand", expert(5, 304, mode)))
        names_case = {0: "sri", 1: "hong", 2: "los"}
        for ci, top in top_by_case.items():
            grp = names_case[ci] + ("+rand" if ci == 0 else "")
            top_t = [tuple(int(x) for x in t) for t in top]
            for t in top_t[:ABLATE_TOP]:
                configs.append((f"{names_case[ci]}:L{t[0]}.E{t[1]}.n{t[2]}:zero", grp, units([t], "zero")))
            for n in (5, 10, 25, 50):
                configs.append((f"{names_case[ci]}:top{n}:zero", grp, units(top_t[:n], "zero")))
                configs.append((f"{names_case[ci]}:top{n}:mean", grp, units(top_t[:n], "mean")))
            no_n0 = [t for t in top_t if t != N0]
            configs.append((f"{names_case[ci]}:top10-no-n0:zero", grp, units(no_n0[:10], "zero")))
            top_e = top_t[0][:2]
            configs.append((f"{names_case[ci]}:E{top_e[1]}@L{top_e[0]}:zero", grp, expert(top_e[0], top_e[1], "zero")))
        for li in range(L):
            configs.append((f"L{li}.routed:mean", "all", sub(li, forward.SUB_ROUTED, "mean")))
            configs.append((f"L{li}.attn:mean", "sri+rand", sub(li, forward.SUB_ATTN, "mean")))
        configs.append(("L5.shared:mean", "sri+rand", sub(5, forward.SUB_SHARED, "mean")))
        configs.append(("L5.ve:zero", "all", base._replace(sub_mult=base.sub_mult.at[5, forward.SUB_VE].set(0.0))))
        # The two routes together: the n0 / E304 unit path and the L5 copy path (VE, whole attention, L3 induction).
        no_ve = base.sub_mult.at[5, forward.SUB_VE].set(0.0)
        configs.append(("n0+L5.ve:zero", "sri+rand", units([N0], "zero")._replace(sub_mult=no_ve)))
        configs.append(("E304+L5.ve:zero", "sri+rand", expert(5, 304, "zero")._replace(sub_mult=no_ve)))
        no_attn = base.sub_mult.at[5, forward.SUB_ATTN].set(0.0)
        configs.append(
            ("n0:zero+L5.attn:mean", "sri+rand", units([N0], "zero")._replace(sub_mult=no_attn, sub_mean=sub_mean))
        )
        configs.append(
            (
                "n0:zero+L3.h0:mean",
                "sri+rand",
                units([N0], "zero")._replace(head_mult=base.head_mult.at[3, 0].set(0.0), head_mean=head_mean),
            )
        )
        configs.append(
            (
                "L5.ve:mean",
                "all",
                base._replace(sub_mult=base.sub_mult.at[5, forward.SUB_VE].set(0.0), ve_mean=jnp.asarray(ve_mean)),
            )
        )
        configs.append(
            (
                "L3.ve:mean",
                "all",
                base._replace(sub_mult=base.sub_mult.at[3, forward.SUB_VE].set(0.0), ve_mean=jnp.asarray(ve_mean)),
            )
        )
        for li in (0, 1, 2, 3, 4, 5):
            for h in range(H):
                configs.append((f"L{li}.h{h}:mean", "sri+rand", head(li, h)))
        abl_names, abl_logp = [], []
        for name, groups, abl in configs:
            allowed = None if groups == "all" else set(groups.split("+"))
            vals = []
            for b, table, _ in batches:
                qa = query_arrays(table, ids)
                if allowed is not None and b.source not in allowed:
                    vals.append(np.full(int(qa["qmask"].sum()), np.nan, np.float32))
                    continue
                tokens, seg, _ = common.device_batch(b, mesh)
                lp = ablated_logp(model, tokens, seg, jnp.asarray(qa["qflat"]), jnp.asarray(qa["qy"]), abl)
                vals.append(np.asarray(lp)[qa["qmask"]])
            abl_names.append(name)
            abl_logp.append(np.concatenate(vals))
            logger.info("ablation %s (%.0fs)", name, time.time() - t0)

        # Prefix swaps: other tokens before "S ri" queries' "ri"; " S" before other "ri".
        swap_ids = [vocab.index(s) for s in SWAP_PREFIXES]
        s_id = ids.case_prefix[0]
        swap_out: dict[str, list] = {}
        sri_batches = [bi for bi, (b, _, _) in enumerate(batches) if b.source == "sri"][: args.swap_batches]
        all_q_index = np.cumsum([0] + [int((t[:, 0] >= 0).sum()) for _, t, _ in batches])
        for label, new_id, sel_cats, which in [
            *[(f"prefix={SWAP_PREFIXES[j]}", sid, (0, 1, 2), sri_batches) for j, sid in enumerate(swap_ids)],
            ("prefix= S", s_id, (3,), list(range(len(batches)))),
        ]:
            vals: dict[str, list] = {}
            for bi in which:
                b, table, _ = batches[bi]
                qa = query_arrays(table, ids)
                pick = qa["qmask"] & np.isin(qa["qcat"], sel_cats) & (table[:, 3] > 0)
                if not pick.any():
                    continue
                tokens = b.tokens.copy()
                for r in table[pick]:
                    tokens[r[0], r[1] - 1] = new_id
                tb = dataclasses.replace(b, tokens=tokens)
                tok_d, seg_d, _ = common.device_batch(tb, mesh)
                out, _ = analyse_jit(
                    model,
                    tok_d,
                    seg_d,
                    jnp.asarray(qa["qflat"]),
                    jnp.asarray(qa["qmask"]),
                    jnp.asarray(qa["qy"]),
                    jnp.asarray(qa["qyi"]),
                    jnp.asarray(qa["qcat"]),
                    jnp.asarray(focus_table(qa["qcat"], focus_by_case)),
                    tuned,
                    rdir,
                    rsdir,
                )
                kept = pick[qa["qmask"]]
                vals.setdefault("query", []).append(all_q_index[bi] + np.flatnonzero(kept))
                for k in (
                    "logp_y",
                    "rank_y",
                    "f_selected",
                    "f_weight",
                    "f_rank",
                    "f_act",
                    "f_logit",
                    "f_dla",
                    "dec_router_terms",
                    "dec_pre_terms",
                ):
                    vals.setdefault(k, []).append(np.asarray(out[k])[pick])
            for k, v in vals.items():
                swap_out[f"swap/{label}/{k}"] = np.concatenate(v)
            logger.info("swap %s (%.0fs)", label, time.time() - t0)

        # Context truncation of every positive query.
        rows_q = []
        q_global = 0
        for _, table, chunk in batches:
            for r in table[table[:, 0] >= 0]:
                if r[2] in POSITIVE_CATS:
                    row = chunk[r[0]]
                    rows_q.append((row.tokens, int(r[1] - r[3]), int(r[1]), q_global))
                q_global += 1
        trunc_out: dict[str, list] = {}
        cat_all = meta[:, 2]
        for tb, ttable in truncated_batches(rows_q, ids.bos):
            valid = ttable[:, 0] >= 0
            qid = np.where(valid, ttable[:, 1], 0)
            cat = cat_all[qid]
            yi = np.asarray(CAT_CASE)[cat]
            tok_d, seg_d, _ = common.device_batch(tb, mesh)
            out, _ = analyse_jit(
                model,
                tok_d,
                seg_d,
                jnp.asarray(np.where(valid, ttable[:, 0], 0)),
                jnp.asarray(valid),
                jnp.asarray(np.asarray(ids.case_target)[yi]),
                jnp.asarray(yi),
                jnp.asarray(cat),
                jnp.asarray(focus_table(cat, focus_by_case)),
                tuned,
                rdir,
                rsdir,
            )
            trunc_out.setdefault("trunc/query", []).append(qid[valid])
            trunc_out.setdefault("trunc/length_index", []).append(ttable[valid, 2])
            for k in ("logp_y", "rank_y", "f_selected", "f_weight", "f_rank", "f_act", "f_logit", "f_dla", "dla_terms"):
                trunc_out.setdefault(f"trunc/{k}", []).append(np.asarray(out[k])[valid])
        logger.info("truncation done (%.0fs)", time.time() - t0)

    results: dict[str, np.ndarray] = {f"q/{k}": v for k, v in per_q.items()}
    results.update({f"acc/{k}": v for k, v in acc2.items()})
    results["q/meta"] = meta  # row, pos, cat, doc_offset, doc_start_visible, prior_mentions
    results["q/source"] = np.asarray(src)
    results["q/context"] = np.stack(ctx)
    results["q/next"] = np.stack(nxt)
    results["n_cat"] = n_cat
    results["focus_by_case"] = focus_by_case
    results["top_by_case"] = np.stack([top_by_case[c] for c in range(len(CASES))])
    results["abl/names"] = np.asarray(abl_names)
    results["abl/logp"] = np.stack(abl_logp)
    results.update(swap_out)
    results.update({k: np.concatenate(v) for k, v in trunc_out.items()})
    results["trunc_lengths"] = np.asarray(TRUNC_LENGTHS)
    results["categories"] = np.asarray(CATEGORIES)
    results["terms"] = np.asarray(TERMS)
    results["unit_mean"] = unit_mean[N0[0], N0[1]]
    results["ids_json"] = np.asarray(json.dumps(dataclasses.asdict(ids)))
    common.save_npz(args.tag, "sri", **results)
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
