# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""N-gram neuron search, validation pass: interventions on the top candidate units from ``ngram_score.npz``.

Each candidate is a unit (routed ``(layer, expert, unit)`` or shared ``(layer, -1, unit)``) with its context
``(a, b)``, next token ``c`` and modal second-previous token ``z``. Occurrences of ``(a, b)`` come from a scan of unseen
training sequences at the end of the cache (disjoint from the stats and scoring samples); each becomes a window of the
document's last ``WINDOW`` tokens ending at ``b`` (after a BOS unless it reaches the document start), packed as
separate documents. Per occurrence:

- the unit's activation and (routed) whether its expert is selected, log p(c) and log p(actual next token);
- the same with the unit mean-ablated (routed: its mean activation when routed; shared: its mean activation);
- with ``a`` replaced by each of ``NUM_SWAPS`` other frequent predecessors of ``b`` (prefix swaps), and with ``z``
  replaced by another frequent predecessor of ``(a, b)``: the unit's activation, selection and log p(c).

Then population ablations: every candidate whose single-unit necessity (mean drop of log p(c) at first mentions)
passes each of ``POPULATION_THRESHOLDS``, mean-ablated together, on random unseen sequences (loss per token) and on
the candidates' own occurrences.

Writes ``ngram_validate.npz``.
"""

import argparse
import dataclasses
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from haliax.partitioning import set_mesh
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard

from experiments.grug.fast_track.interp import common, forward, ngram

logger = logging.getLogger(__name__)

BATCH = 8
S = common.SEQ_LEN
WINDOW = 512
PER_ROW = S // WINDOW
Q = BATCH * PER_ROW  # windows per batch
L, E, I, V = forward.NUM_LAYERS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH, forward.VOCAB
NUM_SWAPS = 3
OCC_WITH_C = 48  # occurrences per bigram followed by c
OCC_OTHER = 16  # and followed by something else
SCAN_CHUNK = 512
POPULATION_THRESHOLDS = (0.1, 0.25, 0.5, 1.0)  # nats of first-mention necessity
RANDOM_SEQUENCES = 256
RANDOM_SEED = 2
# Candidate selection from the scoring pass (see ``select_candidates``).
NUM_TOP = 250
NUM_RANDOM = 350
SELECT_SEED = 0
SRI_UNIT = (5, 304, 0)
MIN_OCCURRENCES = 8
MIN_CONCENTRATION = 0.3
MIN_SELECTIVITY = 3.0
MAX_C_RANK = 50


@dataclasses.dataclass(frozen=True)
class Candidate:
    layer: int
    expert: int  # -1: shared expert
    unit: int
    a: int
    b: int
    c: int
    z: int


def score_metrics(score: dict) -> dict[str, np.ndarray]:
    """Per-unit metrics from the scoring pass, flattened over routed units then shared units: occurrence count and
    mean activation of each group, the variance explained by the (a, b) / b-after-other / rest partition beyond the
    b / not-b partition, the high-activation concentration on (a, b), selectivity (mean act on (a, b) over mean act on
    b after another token), the mean (z, a, b) vs (a, b)-without-z activations, and the output test."""

    def cat_units(r, s):
        rr = np.moveaxis(score[r], 0, -3).reshape(*score[r].shape[1:-2], -1)  # [..., L*E*I] layer-major
        ss = np.moveaxis(score[s], 0, -2).reshape(*score[s].shape[1:-1], -1)
        return np.concatenate([rr, ss], axis=-1)

    g = cat_units("groups", "s_groups").astype(np.float64)  # [4, 3, N]
    high = cat_units("high", "s_high").astype(np.float64)  # [2, N]
    n_ab, n_bna, n_zab, n_all = g[:, 0]
    s_ab, s_bna, s_zab, s_all = g[:, 1]
    total_ss = g[3, 2] - s_all**2 / np.maximum(n_all, 1)
    n_rest = n_all - n_ab - n_bna
    s_rest = s_all - s_ab - s_bna
    n_b, s_b = n_ab + n_bna, s_ab + s_bna

    def between(ns, ss):
        mean = s_all / np.maximum(n_all, 1)
        return sum(n * (s / np.maximum(n, 1) - mean) ** 2 for n, s in zip(ns, ss, strict=True))

    r2_uni = between((n_b, n_rest), (s_b, s_rest)) / np.maximum(total_ss, 1e-12)
    r2_bi = between((n_ab, n_bna, n_rest), (s_ab, s_bna, s_rest)) / np.maximum(total_ss, 1e-12)
    mean_ab = s_ab / np.maximum(n_ab, 1)
    mean_bna = s_bna / np.maximum(n_bna, 1)
    mean_zab = s_zab / np.maximum(n_zab, 1)
    mean_ab_noz = (s_ab - s_zab) / np.maximum(n_ab - n_zab, 1)
    mean_rest = s_rest / np.maximum(n_rest, 1)
    keys = ("cand_a", "cand_b", "cand_c", "cand_z", "top_share_ab", "top_share_b", "top_share_zab")
    out = {k: np.concatenate([score[k].reshape(-1), score["s_" + k].reshape(-1)]) for k in keys}
    for k in ("c_z", "c_rank", "top_promoted"):
        out["ot_" + k] = np.concatenate([score["ot_" + k].reshape(-1), score["ot_s_" + k].reshape(-1)])
    dla = np.concatenate([score["dla_ab"].reshape(-1), score["s_dla_ab"].reshape(-1)])
    hit = np.concatenate([score["ab_next_c"].reshape(-1), np.full(L * I, np.nan)])
    routed_idx = np.stack(np.unravel_index(np.arange(L * E * I), (L, E, I)), axis=1)
    s_idx = np.stack(np.unravel_index(np.arange(L * I), (L, I)), axis=1)
    unit_index = np.concatenate([routed_idx, np.stack([s_idx[:, 0], np.full(L * I, -1), s_idx[:, 1]], axis=1)])
    out.update(
        unit_index=unit_index,
        n_ab=n_ab,
        n_bna=n_bna,
        n_zab=n_zab,
        n_all=n_all,
        mean_ab=mean_ab,
        mean_bna=mean_bna,
        mean_rest=mean_rest,
        mean_zab=mean_zab,
        mean_ab_noz=mean_ab_noz,
        r2_uni=r2_uni,
        r2_bi=r2_bi,
        r2_gain=r2_bi - r2_uni,
        concentration=high[0] / np.maximum(high[1], 1e-12),
        selectivity=mean_ab / np.maximum(mean_bna, 1e-3 * np.maximum(mean_ab, 1e-12)),
        dla_ab=dla / np.maximum(n_ab, 1),
        p_c_given_ab=hit / np.maximum(n_ab, 1),
    )
    return out


def eligible(m: dict[str, np.ndarray]) -> np.ndarray:
    """Units whose (a, b) context is a real bigram with enough occurrences, holding a large share of the unit's high
    activations, selective over b in other contexts, and whose output direction promotes c."""
    return (
        (m["cand_a"] >= 0)
        & (m["cand_c"] >= 0)
        & (m["n_ab"] >= MIN_OCCURRENCES)
        & (m["concentration"] >= MIN_CONCENTRATION)
        & (m["selectivity"] >= MIN_SELECTIVITY)
        & (m["ot_c_rank"] < MAX_C_RANK)
    )


def select_candidates(m: dict[str, np.ndarray]) -> tuple[list[Candidate], np.ndarray]:
    """The eligible units' top ``NUM_TOP`` by concentration times variance gain, a uniform random ``NUM_RANDOM`` of
    the other eligible units (for population estimates), and the Sri unit. Returns the candidates and each one's tier
    (0: top, 1: random, 2: Sri)."""
    ok = eligible(m)
    score = np.where(ok, m["concentration"] * np.maximum(m["r2_gain"], 0.0), -1.0)
    order = np.argsort(-score)
    top = order[:NUM_TOP]
    rest = np.setdiff1d(np.flatnonzero(ok), top)
    rng = np.random.default_rng(SELECT_SEED)
    rand = np.sort(rng.choice(rest, size=min(NUM_RANDOM, len(rest)), replace=False))
    sri = np.flatnonzero(np.all(m["unit_index"] == np.asarray(SRI_UNIT), axis=1))
    picks = [*top, *rand, *np.setdiff1d(sri, np.concatenate([top, rand]))]
    tiers = np.asarray([0] * len(top) + [1] * len(rand) + [2] * (len(picks) - len(top) - len(rand)), np.int32)
    cands = [
        Candidate(
            *(int(x) for x in m["unit_index"][j]),
            int(m["cand_a"][j]),
            int(m["cand_b"][j]),
            int(m["cand_c"][j]),
            int(m["cand_z"][j]),
        )
        for j in picks
    ]
    return cands, tiers


def alternatives(tokens: np.ndarray, seg: np.ndarray, cands: list[Candidate]) -> tuple[np.ndarray, np.ndarray]:
    """Per candidate: the ``NUM_SWAPS`` most frequent predecessors of ``b`` other than ``a`` (``-1``: none), and the
    most frequent predecessor of ``(a, b)`` other than ``z``."""
    prev, prev2, _ = ngram.context_arrays(tokens, seg)
    tok = tokens.astype(np.int64).reshape(-1)
    prev, prev2 = prev.reshape(-1), prev2.reshape(-1)
    swaps = np.full((len(cands), NUM_SWAPS), -1, np.int64)
    zswap = np.full(len(cands), -1, np.int64)
    wanted_b = np.unique([c.b for c in cands])
    on_b = np.isin(tok, wanted_b) & (prev >= 0)
    pb, bb, p2 = prev[on_b], tok[on_b], prev2[on_b]
    for j, c in enumerate(cands):
        here = bb == c.b
        vals, counts = np.unique(pb[here & (pb != c.a)], return_counts=True)
        top = vals[np.argsort(-counts)][:NUM_SWAPS]
        swaps[j, : len(top)] = top
        zs = p2[here & (pb == c.a) & (p2 >= 0) & (p2 != c.z)]
        if len(zs):
            zv, zc = np.unique(zs, return_counts=True)
            zswap[j] = zv[np.argmax(zc)]
    return swaps, zswap


@dataclasses.dataclass
class Occurrence:
    window: np.ndarray  # tokens ending at b
    next_token: int
    first: bool  # no earlier (a, b) in the window
    prev2: int  # the token before a in the document (-1: none)


def windows_for(tokens: np.ndarray, seg: np.ndarray, t: int, bos: int) -> np.ndarray:
    start = t
    while start > 0 and seg[start - 1] == seg[t] and t - start + 1 < WINDOW - 1:
        start -= 1
    w = tokens[start : t + 1].astype(np.int32)
    doc_start = start == 0 or seg[start - 1] != seg[t]
    if not doc_start or tokens[start] != bos:
        w = np.concatenate([[bos], w]).astype(np.int32)
    return w


@common.off_mesh
def collect(cands: list[Candidate], num_sequences: int, bos: int):
    """Scan the end of the training cache for every candidate (a, b, c); yields one dict of occurrence lists keyed
    by triple."""
    wanted = {(c.a, c.b, c.c) for c in cands}
    by_ab: dict[int, list[int]] = {}
    for a, b, c in wanted:
        by_ab.setdefault(a * V + b, []).append(c)
    keys = np.asarray(sorted(by_ab), np.int64)
    occ: dict[tuple[int, int, int], list[Occurrence]] = {k: [] for k in wanted}
    other: dict[tuple[int, int, int], int] = {k: 0 for k in wanted}

    def full(k):
        return len(occ[k]) - other[k] >= OCC_WITH_C and other[k] >= OCC_OTHER

    t0 = time.time()
    for batch in _cache_tail(num_sequences):
        prev, prev2, nxt = ngram.context_arrays(batch.tokens, batch.segment_ids)
        key = prev * V + batch.tokens.astype(np.int64)
        hits = np.isin(key, keys) & (prev >= 0) & (nxt >= 0) & (batch.loss_weight > 0)
        for r, t in zip(*np.nonzero(hits), strict=True):
            ab = int(key[r, t])
            a, b = divmod(ab, V)
            for c in by_ab[ab]:
                k = (a, b, c)
                is_c = int(nxt[r, t]) == c
                if full(k) or (is_c and len(occ[k]) - other[k] >= OCC_WITH_C) or (not is_c and other[k] >= OCC_OTHER):
                    continue
                w = windows_for(batch.tokens[r], batch.segment_ids[r], int(t), bos)
                wk = np.asarray(w, np.int64)
                first = not np.any((wk[:-2] == a) & (wk[1:-1] == b))
                occ[k].append(Occurrence(w, int(nxt[r, t]), first, int(prev2[r, t])))
                other[k] += 0 if is_c else 1
        done = sum(full(k) for k in wanted)
        logger.info("scan chunk: %d/%d triples full (%.0fs)", done, len(wanted), time.time() - t0)
        if done == len(wanted):
            break
    yield occ


def _cache_tail(num_sequences: int):
    ds = common.data_config().train_sets(Axis("position", S), key=jax.random.PRNGKey(0))["train"].as_sync_dataset()
    total = len(ds)
    start = max(common.TRAINED_SEQUENCES, total - num_sequences)
    for s in range(start, total, SCAN_CHUNK):
        yield common.examples_to_batch("train", ds.get_batch(list(range(s, min(total, s + SCAN_CHUNK)))))


def pack(windows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Up to ``Q`` windows packed ``PER_ROW`` per row as separate documents -> tokens, segment ids ``[BATCH, S]`` and
    the flat query position of each window's last token ``[Q]`` (unused: 0)."""
    tokens = np.zeros((BATCH, S), np.int32)
    seg = np.zeros((BATCH, S), np.int32)
    qflat = np.zeros(Q, np.int32)
    for j, w in enumerate(windows):
        r, slot = divmod(j, PER_ROW)
        off = slot * WINDOW
        tokens[r, off : off + len(w)] = w
        seg[r, off : off + WINDOW] = slot + 1
        qflat[j] = r * S + off + len(w) - 1
    return tokens, seg, qflat


@jax.jit
def measure(model, tokens, seg, qflat, qc, qnext, unit, abl):
    """At the query positions: log p(c), log p(next), the unit's activation and its expert's selection. ``unit`` is
    ``(layer, expert, index)`` (expert -1: shared)."""
    m = forward.replicate(model)
    tr = forward.trace_forward(m, tokens, seg, abl)
    q = reshard(qflat, P(None))
    hidden = reshard(tr["hidden"].reshape(-1, tr["hidden"].shape[-1]), P(None, None))[q]
    logits = forward.soft_cap(jnp.einsum("qd,dv->qv", hidden.astype(jnp.float32), m.output_proj.astype(jnp.float32)))
    logp = jax.nn.log_softmax(logits, axis=-1)
    lc = jnp.take_along_axis(logp, qc[:, None], axis=-1)[:, 0]
    ln = jnp.take_along_axis(logp, qnext[:, None], axis=-1)[:, 0]
    li, e, i = unit[0], unit[1], unit[2]
    routed_in = jnp.stack([lay["routed_input"] for lay in tr["layers"]])[li][q]  # [Q, R]
    w_up = jnp.stack([lay.mlp.expert_mlp.w_up for lay in m.layers()])[li, jnp.maximum(e, 0), :, i]  # [R]
    pre = jnp.einsum("qr,r->q", routed_in, w_up, preferred_element_type=jnp.float32)
    selected = jnp.any(jnp.stack([lay["selected"] for lay in tr["layers"]])[li][q] == e, axis=-1)
    routed_act = jnp.where(selected, jnp.square(jax.nn.relu(pre)), 0.0)
    shared_act = jnp.stack([lay["shared_act"] for lay in tr["layers"]])[li][q][:, i]
    act = jnp.where(e < 0, shared_act, routed_act)
    return dict(logp_c=lc, logp_next=ln, act=act, selected=jnp.where(e < 0, True, selected))


@jax.jit
def sequence_loss(model, tokens, seg, weight, abl):
    tr = forward.trace_forward(forward.replicate(model), tokens, seg, abl)
    return jnp.sum(tr["loss"].astype(jnp.float32) * weight), jnp.sum(weight)


def unit_ablation(base: forward.Ablation, units: list[Candidate], routed_mean, shared_mean) -> forward.Ablation:
    nm, sm = base.neuron_mult, base.shared_mult
    for u in units:
        if u.expert < 0:
            sm = sm.at[u.layer, u.unit].set(0.0)
        else:
            nm = nm.at[u.layer, u.expert, u.unit].set(0.0)
    return base._replace(neuron_mult=nm, neuron_mean=routed_mean, shared_mult=sm, shared_mean=shared_mean)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--scan-sequences", type=int, default=100000)
    args = parser.parse_args()
    t0 = time.time()
    score = ngram.load_npz(args.tag, "ngram_score")
    stats = ngram.load_npz(args.tag, "stats")
    metrics = score_metrics(score)
    cands, tiers = select_candidates(metrics)
    logger.info("%d candidates (%.0fs)", len(cands), time.time() - t0)
    sample = ngram.load_npz(args.tag, "ngram_tokens")
    swaps, zswap = alternatives(sample["tokens"], sample["segment_ids"], cands)
    vocab = common.vocab_strings()
    bos = vocab.index("<|begin_of_text|>")
    (occ,) = collect(cands, args.scan_sequences, bos)
    logger.info("occurrences collected (%.0fs)", time.time() - t0)
    routed_mean = jnp.asarray(stats["act_sum"] / np.maximum(stats["expert_tokens"][..., None], 1.0), jnp.float32)
    shared_mean = jnp.asarray(stats["s_act_sum"] / np.maximum(stats["tokens"], 1.0), jnp.float32)

    cfg = common.model_config(args.tag)
    forward.check_config(cfg)
    mesh = common.one_gpu_mesh()
    params = common.load_params(cfg, mesh, common.CHECKPOINTS[args.tag])
    fields = ("logp_c", "logp_next", "act", "selected")
    configs = ["base", "ablate", *[f"swap{k}" for k in range(NUM_SWAPS)], "zswap"]
    res = {f"{cfg_}/{f}": np.full((len(cands), Q), np.nan, np.float32) for cfg_ in configs for f in fields}
    is_c = np.zeros((len(cands), Q), bool)
    first = np.zeros((len(cands), Q), bool)
    has_z = np.zeros((len(cands), Q), bool)
    valid = np.zeros((len(cands), Q), bool)
    with set_mesh(mesh):
        model = common.compute_model(params)
        base = forward.no_ablation()

        def run(tokens, seg, qflat, qc, qnext, unit, abl):
            out = measure(
                model,
                *common.device_batch(common.TokenBatch("x", tokens, seg, np.ones_like(seg, np.float32)), mesh)[:2],
                jnp.asarray(qflat),
                jnp.asarray(qc),
                jnp.asarray(qnext),
                jnp.asarray(unit, jnp.int32),
                abl,
            )
            return {k: np.asarray(v) for k, v in out.items()}

        for j, cand in enumerate(cands):
            os_ = occ[(cand.a, cand.b, cand.c)][:Q]
            if not os_:
                continue
            n = len(os_)
            valid[j, :n] = True
            is_c[j, :n] = [o.next_token == cand.c for o in os_]
            first[j, :n] = [o.first for o in os_]
            has_z[j, :n] = [cand.z >= 0 and o.prev2 == cand.z for o in os_]
            qc = np.full(Q, cand.c, np.int32)
            qnext = np.zeros(Q, np.int32)
            qnext[:n] = [o.next_token for o in os_]
            unit = (cand.layer, cand.expert, cand.unit)
            windows = [o.window for o in os_]
            tokens, seg, qflat = pack(windows)
            outs = {
                "base": run(tokens, seg, qflat, qc, qnext, unit, base),
                "ablate": run(
                    tokens, seg, qflat, qc, qnext, unit, unit_ablation(base, [cand], routed_mean, shared_mean)
                ),
            }
            for k in range(NUM_SWAPS):
                if swaps[j, k] < 0:
                    continue
                sw = [np.concatenate([w[:-2], [swaps[j, k], w[-1]]]).astype(np.int32) for w in windows]
                outs[f"swap{k}"] = run(*pack(sw), qc, qnext, unit, base)
            if cand.z >= 0 and zswap[j] >= 0:
                zw = [
                    np.concatenate([w[:-3], [zswap[j]], w[-2:]]).astype(np.int32) if z else w
                    for w, z in zip(windows, has_z[j, :n], strict=True)
                ]
                outs["zswap"] = run(*pack(zw), qc, qnext, unit, base)
            for cfg_, out in outs.items():
                for f in fields:
                    res[f"{cfg_}/{f}"][j, :n] = out[f][:n]
            if j % 20 == 0:
                logger.info("candidate %d/%d (%.0fs)", j, len(cands), time.time() - t0)

        # Single-unit necessity at first mentions followed by c.
        drop = res["base/logp_c"] - res["ablate/logp_c"]
        sel = valid & is_c & first
        necessity = np.where(sel.any(1), np.nansum(np.where(sel, drop, 0), 1) / np.maximum(sel.sum(1), 1), np.nan)

        # Population ablations.
        random = list(common.train_batches(BATCH, RANDOM_SEQUENCES, seed=RANDOM_SEED))
        pop_loss = []
        pop_own = []
        for thr in (None, *POPULATION_THRESHOLDS):
            members = [] if thr is None else [c for c, nec in zip(cands, necessity, strict=True) if nec >= thr]
            abl = unit_ablation(base, members, routed_mean, shared_mean)
            tot, cnt = 0.0, 0.0
            for b in random:
                tokens, seg, w = common.device_batch(b, mesh)
                s_, n_ = sequence_loss(model, tokens, seg, w, abl)
                tot += float(s_)
                cnt += float(n_)
            pop_loss.append(tot / cnt)
            own = np.full((len(cands), Q), np.nan, np.float32)
            if thr is not None:
                for j, cand in enumerate(cands):
                    if not valid[j].any():
                        continue
                    os_ = occ[(cand.a, cand.b, cand.c)][:Q]
                    qnext = np.zeros(Q, np.int32)
                    qnext[: len(os_)] = [o.next_token for o in os_]
                    out = run(
                        *pack([o.window for o in os_]),
                        np.full(Q, cand.c, np.int32),
                        qnext,
                        (cand.layer, cand.expert, cand.unit),
                        abl,
                    )
                    own[j, : len(os_)] = out["logp_c"][: len(os_)]
            pop_own.append(own)
            logger.info(
                "population thr %s: %d units, loss %.5f (%.0fs)", thr, len(members), pop_loss[-1], time.time() - t0
            )

    table = np.asarray([dataclasses.astuple(c) for c in cands], np.int32).reshape(-1, 7)
    common.save_npz(
        args.tag,
        "ngram_validate",
        candidates=table,
        tiers=tiers,
        swaps=swaps,
        zswap=zswap,
        valid=valid,
        is_c=is_c,
        first=first,
        has_z=has_z,
        necessity=necessity,
        pop_thresholds=np.asarray((np.nan, *POPULATION_THRESHOLDS)),
        pop_loss=np.asarray(pop_loss),
        pop_own_logp_c=np.stack(pop_own),
        **res,
    )
    logger.info("done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
