# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side analysis of ``skip_tokens``: which tokens need the second half, by category and mechanism, and how well
first-half signals predict it (probes, AUROC, the skip-rate vs loss-cost frontier). Runs on CPU from local copies of
``skip_tokens.npz`` / ``skip_probe.npz``; writes one JSON for the report page.

usage: python -m experiments.grug.fast_track.interp.skip_report --tokens skip_tokens.npz --probe skip_probe.npz
       --vocab vocab.json --ngram-units ngram_units.npz --out skip_report.json
"""

import argparse
import itertools
import json
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.grug.fast_track.interp import categories

logger = logging.getLogger(__name__)

NEED = 0.1  # "needs the second half": benefit above this many nats
FREE = (0.05, 0.1)
SKIP_RATES = (0.3, 0.5, 0.7)
CURVE_POINTS = 101
POS_BINS = (0, 16, 128, 1024, 4096)
FREQ_BINS = (0, 100, 1000, 4000, 16384)
PRIMARY = ("block2", "full")


def load(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC of ``score`` for boolean ``label`` (ties averaged)."""
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(len(s), np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks over ties.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks[order])
    ranks[order] = (sums / counts)[inv]
    pos = label.sum()
    neg = len(label) - pos
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / max(pos * neg, 1))


def roc_curve(score: np.ndarray, label: np.ndarray, points: int = CURVE_POINTS) -> dict[str, list[float]]:
    order = np.argsort(-score, kind="stable")
    lab = label[order]
    tp = np.cumsum(lab) / max(lab.sum(), 1)
    fp = np.cumsum(~lab) / max((~lab).sum(), 1)
    idx = np.unique(np.linspace(0, len(lab) - 1, points).round().astype(int))
    return dict(fpr=[0.0, *fp[idx].tolist()], tpr=[0.0, *tp[idx].tolist()])


def skip_cost(score: np.ndarray, benefit: np.ndarray, rates) -> list[float]:
    """Mean added loss per token (over all tokens) when the ``rate`` fraction with the lowest ``score`` skip."""
    order = np.argsort(score, kind="stable")
    csum = np.concatenate([[0.0], np.cumsum(benefit[order])])
    n = len(benefit)
    return [float(csum[int(r * n + 0.5)] / n) for r in rates]


def concentration(benefit: np.ndarray, points: int = CURVE_POINTS) -> dict[str, list[float]]:
    """Share of the total benefit from the top x fraction of tokens (sorted by benefit, descending)."""
    b = np.sort(benefit)[::-1]
    csum = np.concatenate([[0.0], np.cumsum(b)])
    x = np.linspace(0, 1, points)
    return dict(x=x.tolist(), share=(csum[(x * len(b)).round().astype(int)] / csum[-1]).tolist())


def doc_context(tokens: np.ndarray, seg: np.ndarray) -> dict[str, np.ndarray]:
    """Per position: offset in document; whether the current token, the target, and the bigram occurred earlier in
    the document; and the induction guess (the token that followed the current token's previous occurrence) agreeing
    with the target."""
    b, s = tokens.shape
    pos = np.zeros((b, s), np.int32)
    cur_seen = np.zeros((b, s), bool)
    tgt_seen = np.zeros((b, s), bool)
    ind_hit = np.zeros((b, s), bool)
    ind_avail = np.zeros((b, s), bool)
    for i in range(b):
        row, sg = tokens[i], seg[i]
        last_next: dict[int, int] = {}
        seen: set[int] = set()
        start = 0
        for t in range(s):
            if t > 0 and sg[t] != sg[t - 1]:
                start = t
                last_next = {}
                seen = set()
            pos[i, t] = t - start
            tok = int(row[t])
            cur_seen[i, t] = tok in seen
            seen.add(tok)
            if t + 1 < s and sg[t + 1] == sg[t]:
                nxt = int(row[t + 1])
                tgt_seen[i, t] = nxt in seen
                if tok in last_next:
                    ind_avail[i, t] = True
                    ind_hit[i, t] = last_next[tok] == nxt
                last_next[tok] = nxt
    return dict(pos=pos, cur_seen=cur_seen, tgt_seen=tgt_seen, ind_avail=ind_avail, ind_hit=ind_hit)


def ngram_marks(tokens: np.ndarray, seg: np.ndarray, units: dict) -> tuple[np.ndarray, np.ndarray]:
    """``[B, S]``: the (prev, current) pair is a census n-gram unit's (a, b); and also its target is the unit's c."""
    ab = set(zip(units["a"].tolist(), units["b"].tolist(), strict=True))
    abc = set(zip(units["a"].tolist(), units["b"].tolist(), units["c"].tolist(), strict=True))
    prev = np.pad(tokens[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    prev = np.where(np.pad(seg[:, 1:] == seg[:, :-1], ((0, 0), (1, 0))), prev, -1)
    nxt = np.pad(tokens[:, 1:], ((0, 0), (0, 1)), constant_values=-1)
    in_ab = np.zeros(tokens.shape, bool)
    in_abc = np.zeros(tokens.shape, bool)
    for idx in zip(*np.nonzero(prev >= 0), strict=True):
        key = (int(prev[idx]), int(tokens[idx]))
        if key in ab:
            in_ab[idx] = True
            in_abc[idx] = (*key, int(nxt[idx])) in abc
    return in_ab, in_abc


def summary(benefit: np.ndarray) -> dict:
    q = np.quantile(benefit, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    out = dict(
        mean=float(benefit.mean()),
        quantiles=q.tolist(),
        frac_negative=float((benefit < 0).mean()),
        **{f"free_{f}": float((benefit < f).mean()) for f in FREE},
        need_frac=float((benefit > NEED).mean()),
    )
    pos = np.clip(benefit, 0, None)
    srt = np.sort(benefit)[::-1]
    for top in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        out[f"top{int(top * 100)}_share"] = float(srt[: int(top * len(srt))].sum() / benefit.sum())
    out["positive_total_over_net"] = float(pos.sum() / benefit.sum())
    return out


def histogram(benefit: np.ndarray) -> dict:
    counts, _ = np.histogram(np.clip(benefit, -2.05, 8.05), bins=np.arange(-2.1, 8.11, 0.1))
    return dict(lo=-2.1, width=0.1, counts=counts.tolist())


def logistic_probe(x_tr, y_tr, x_te, steps: int = 400, lr: float = 0.05, l2: float = 1e-4) -> np.ndarray:
    """Logistic regression (standardized features, full-batch Adam); returns test logits."""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xt, xe = jnp.asarray((x_tr - mu) / sd), jnp.asarray((x_te - mu) / sd)
    y = jnp.asarray(y_tr, jnp.float32)
    params = (jnp.zeros(x_tr.shape[1]), jnp.zeros(()))

    def loss(p):
        z = xt @ p[0] + p[1]
        return jnp.mean(optax.sigmoid_binary_cross_entropy(z, y)) + l2 * jnp.sum(p[0] ** 2)

    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        g = jax.grad(loss)(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s

    for _ in range(steps):
        params, state = step(params, state)
    return np.asarray(xe @ params[0] + params[1])


def ridge_probe(x_tr, y_tr, x_te, l2: float = 1e-2) -> np.ndarray:
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    xt = np.concatenate([(x_tr - mu) / sd, np.ones((len(x_tr), 1))], 1)
    xe = np.concatenate([(x_te - mu) / sd, np.ones((len(x_te), 1))], 1)
    w = np.linalg.solve(xt.T @ xt + l2 * len(xt) * np.eye(xt.shape[1]), xt.T @ y_tr)
    return xe @ w


def token_table(ids_tr, y_tr, ids_te, prior_weight: float = 20.0) -> np.ndarray:
    """Smoothed per-token-id mean of ``y`` (fit on train, applied to test)."""
    n = np.bincount(ids_tr, minlength=16384)
    s = np.bincount(ids_tr, weights=y_tr, minlength=16384)
    mean = y_tr.mean()
    table = (s + prior_weight * mean) / (n + prior_weight)
    return table[ids_te]


def depth_report(z: dict[str, np.ndarray]) -> dict:
    """Per exit of a ``skip_depth`` file: benefit summary, concentration, and the scalar-signal skip frontier."""
    w = z["weight"].astype(bool)
    tokens, seg = z["tokens"], z["segment_ids"]
    loss_full = z["loss_full"].astype(np.float32)[w]
    ctx = doc_context(tokens, seg)
    seq_id = np.broadcast_to(np.arange(tokens.shape[0])[:, None], tokens.shape)[w]
    test = seq_id % 2 == 1
    cur = tokens[w]
    rates = np.linspace(0, 1, 51)
    out = dict(
        num_layers=int(z["num_layers"]), exits=z["exits"].tolist(), loss_full=float(loss_full.mean()), per_exit=[]
    )
    for k in range(len(z["exits"])):
        benefit = z["exit_loss"][k].astype(np.float32)[w] - loss_full
        need = benefit > NEED
        ent = z["exit_entropy"][k].astype(np.float32)[w]
        maxp = z["exit_maxp"][k].astype(np.float32)[w]
        tt = np.zeros(len(cur), np.float32)
        for half in (0, 1):
            tr_m = (seq_id % 2) != half
            tt[~tr_m] = token_table(cur[tr_m], benefit[tr_m], cur[~tr_m])
        x = np.stack([ent, -maxp, tt, ctx["cur_seen"][w], np.log1p(ctx["pos"][w])], 1).astype(np.float32)
        reg = ridge_probe(x[~test], benefit[~test], x[test])
        logit = logistic_probe(x[~test], need[~test], x[test])
        b_te = benefit[test]
        out["per_exit"].append(
            dict(
                exit=int(z["exits"][k]),
                lens_loss=float(z["exit_lens_loss"][k].astype(np.float32)[w].mean()),
                head_loss=float(z["exit_loss"][k].astype(np.float32)[w].mean()),
                summary=summary(benefit),
                kl_mean=float(z["exit_kl"][k].astype(np.float32)[w].mean()),
                agree=float(z["exit_agree"][k][w].mean()),
                concentration=concentration(benefit),
                auroc=dict(
                    exit_entropy=auroc(ent[test], need[test]),
                    token_id_table=auroc(tt[test], need[test]),
                    scalars_logistic=auroc(logit, need[test]),
                    scalars_ridge=auroc(reg, need[test]),
                ),
                frontier=dict(
                    rates=rates.tolist(),
                    oracle=skip_cost(b_te, b_te, rates),
                    random=(rates * b_te.mean()).tolist(),
                    scalars_ridge=skip_cost(reg, b_te, rates),
                    exit_entropy=skip_cost(ent[test], b_te, rates),
                ),
            )
        )
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens")
    parser.add_argument("--probe")
    parser.add_argument("--vocab")
    parser.add_argument("--ngram-units")
    parser.add_argument("--out", required=True)
    parser.add_argument("--depth", nargs="*", default=[], help="skip_depth files: only their depth reports")
    args = parser.parse_args()
    if args.depth:
        with open(args.out, "w") as f:
            json.dump({path: depth_report(load(path)) for path in args.depth}, f, indent=1)
        return
    z = load(args.tokens)
    pz = load(args.probe)
    with open(args.vocab) as f:
        vocab = json.load(f)
    units = load(args.ngram_units)
    stages, heads = list(z["stages"]), list(z["heads"])
    tokens, seg, sources = z["tokens"], z["segment_ids"], z["sources"]
    n_seq, s = tokens.shape
    w = z["weight"].astype(bool)
    loss_full = z["loss_full"].astype(np.float32)
    exit_loss = z["exit_loss"].astype(np.float32)
    ctx = doc_context(tokens, seg)
    in_ab, in_abc = ngram_marks(tokens, seg, units)
    feats = categories.token_features(vocab)
    train_rows = sources == "train"
    counts = np.bincount(tokens[train_rows].reshape(-1), minlength=len(vocab))
    freq_rank = np.empty(len(vocab), np.int64)
    freq_rank[np.argsort(-counts, kind="stable")] = np.arange(len(vocab))
    cat = {}
    for src in np.unique(sources):
        rows = sources == src
        m = categories.category_masks(tokens[rows], seg[rows], np.ones(tokens[rows].shape), src, feats, freq_rank)
        for c, mask in zip(categories.CATEGORIES, m, strict=True):
            cat.setdefault(c, np.zeros(tokens.shape, bool))[rows] = mask > 0
    target = np.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    cat["target_seen"] = ctx["tgt_seen"]
    cat["induction_hit"] = ctx["ind_hit"]
    cat["ngram_prefix"] = in_ab
    cat["ngram_completion"] = in_abc
    cat["frequent"] = freq_rank[target] < 1000
    for lo, hi in itertools.pairwise(POS_BINS):
        cat[f"pos_{lo}_{hi}"] = (ctx["pos"] >= lo) & (ctx["pos"] < hi)
    for lo, hi in itertools.pairwise(FREQ_BINS):
        cat[f"freq_{lo}_{hi}"] = (freq_rank[target] >= lo) & (freq_rank[target] < hi)
    for src in np.unique(sources):
        cat[f"src_{src}"] = np.broadcast_to((sources == src)[:, None], tokens.shape)

    report: dict = dict(n_tokens=int(w.sum()), stages=stages, heads=heads, loss_full=float(loss_full[w].mean()))
    report["exit_loss"] = {
        f"{st}/{h}": float(exit_loss[i, j][w].mean()) for i, st in enumerate(stages) for j, h in enumerate(heads)
    }
    kl, agree = z["exit_kl"].astype(np.float32), z["exit_agree"].astype(bool)
    report["exits"] = {}
    for i, st in enumerate(stages):
        for j, h in enumerate(heads):
            ben = exit_loss[i, j][w] - loss_full[w]
            report["exits"][f"{st}/{h}"] = dict(
                summary=summary(ben),
                kl_mean=float(kl[i, j][w].mean()),
                kl_quantiles=np.quantile(kl[i, j][w], [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).tolist(),
                agree=float(agree[i, j][w].mean()),
                correct=float(z["exit_correct"][i, j][w].mean()),
            )
    report["correct_full"] = float(z["correct_full"][w].mean())
    pi, ph = stages.index(PRIMARY[0]), heads.index(PRIMARY[1])
    benefit_all = exit_loss[pi, ph] - loss_full  # [N, S]
    benefit = benefit_all[w]
    report["primary"] = dict(stage=PRIMARY[0], head=PRIMARY[1])
    report["histogram"] = histogram(benefit)
    report["concentration"] = {
        f"{st}/full": concentration(exit_loss[i, ph][w] - loss_full[w]) for i, st in enumerate(stages)
    }
    report["concentration"]["block2/tuned"] = concentration(exit_loss[pi, heads.index("tuned")][w] - loss_full[w])
    # How the benefit splits over L3 attention, L3 MoE, L4 and L5 (full heads): per stage increments.
    seq = [exit_loss[i, ph] for i in range(len(stages))] + [loss_full]
    names = ["L3.attn", "L3.moe", "L4", "L5"]
    inc = np.stack([(seq[k] - seq[k + 1])[w] for k in range(4)])
    report["stage_split"] = dict(
        names=names,
        mean=inc.mean(1).tolist(),
        # Among tokens that need the second half: the share of their benefit per increment and the dominant one.
        need_mean=inc[:, benefit > NEED].mean(1).tolist(),
        need_dominant=np.bincount(np.argmax(inc[:, benefit > NEED], 0), minlength=4).tolist(),
        free_mean=inc[:, benefit < FREE[1]].mean(1).tolist(),
    )
    # Categories.
    total = benefit.sum()
    rows = []
    for c, mask in cat.items():
        m = mask[w]
        if m.sum() < 100:
            continue
        b = benefit[m]
        rows.append(
            dict(
                name=c,
                n=int(m.sum()),
                token_share=float(m.mean()),
                benefit_share=float(b.sum() / total),
                mean_benefit=float(b.mean()),
                free_0_1=float((b < FREE[1]).mean()),
                need=float((b > NEED).mean()),
                loss_full=float(loss_full[w][m].mean()),
                loss_exit=float(exit_loss[pi, ph][w][m].mean()),
                stage_increments=inc[:, m].mean(1).tolist(),
            )
        )
    report["categories"] = rows
    # Mechanisms: ablation deltas and DLA by benefit bucket.
    abl = z["abl_loss"].astype(np.float32) - loss_full[None]
    hard_thr = np.quantile(benefit, 0.9)
    buckets = dict(
        free=benefit < FREE[1],
        mid=(benefit >= FREE[1]) & (benefit < hard_thr),
        hard=benefit >= hard_thr,
        all=np.ones_like(benefit, bool),
    )
    report["hard_threshold"] = float(hard_thr)
    report["ablations"] = dict(
        names=list(z["ablations"]),
        by_bucket={k: abl[:, w][:, m].mean(1).tolist() for k, m in buckets.items()},
        share_of_benefit={
            k: (abl[:, w][:, m].sum(1) / max(benefit[m].sum(), 1e-9)).tolist() for k, m in buckets.items()
        },
    )
    dla = z["dla"].astype(np.float32)
    report["dla"] = dict(
        terms=list(z["dla_terms"]),
        by_bucket={k: dla[:, w][:, m].mean(1).tolist() for k, m in buckets.items()},
    )
    # Router proxy.
    report["router3_overlap"] = dict(
        mean=float(z["router3_overlap"][w].mean()),
        hist=np.bincount(z["router3_overlap"][w].astype(int), minlength=11).tolist(),
    )

    # Probes. Labels and scalar signals over all tokens; sequences split in halves (even / odd) for train / test.
    need = benefit > NEED
    seq_id = np.broadcast_to(np.arange(n_seq)[:, None], tokens.shape)[w]
    test = seq_id % 2 == 1
    ent = z["exit_entropy"].astype(np.float32)[pi, ph][w]
    maxp = z["exit_maxp"].astype(np.float32)[pi, ph][w]
    cur = tokens[w]
    scalars = dict(
        exit_entropy=ent,
        neg_exit_maxp=-maxp,
        neg_tuned_maxp=-z["exit_maxp"].astype(np.float32)[pi, heads.index("tuned")][w],
        router3_proxy_entropy=z["router3_proxy_entropy"].astype(np.float32)[w],
        neg_router3_proxy_max=-z["router3_proxy_max"].astype(np.float32)[w],
        cur_seen=ctx["cur_seen"][w].astype(np.float32),
        induction_available=ctx["ind_avail"][w].astype(np.float32),
        log_pos=np.log1p(ctx["pos"][w]).astype(np.float32),
        cur_freq_rank=np.log1p(freq_rank[cur]).astype(np.float32),
    )
    tt = np.zeros(len(cur), np.float32)
    for half in (0, 1):
        tr_m = (seq_id % 2) != half
        tt[~tr_m] = token_table(cur[tr_m], benefit[tr_m], cur[~tr_m])
    scalars["token_id_table"] = tt
    probes = {}
    for name, sc in scalars.items():
        probes[name] = dict(auroc=auroc(sc[test], need[test]))
    x_sc = np.stack(list(scalars.values()), 1)
    combo_logit = logistic_probe(x_sc[~test], need[~test], x_sc[test])
    combo_reg = ridge_probe(x_sc[~test], benefit[~test], x_sc[test])
    probes["scalars_logistic"] = dict(auroc=auroc(combo_logit, need[test]))
    probes["scalars_ridge_benefit"] = dict(auroc=auroc(combo_reg, need[test]))
    report["need_rate"] = float(need.mean())
    rates = np.linspace(0, 1, 51)
    frontier = dict(
        rates=rates.tolist(),
        oracle=skip_cost(benefit[test], benefit[test], rates),
        random=(rates * benefit[test].mean()).tolist(),
        scalars_ridge=skip_cost(combo_reg, benefit[test], rates),
        scalars_logistic=skip_cost(combo_logit, benefit[test], rates),
        exit_entropy=skip_cost(ent[test], benefit[test], rates),
        neg_exit_maxp=skip_cost(-maxp[test], benefit[test], rates),
        token_id_table=skip_cost(tt[test], benefit[test], rates),
    )
    # Vector probes on the probe subset (the first sequences of the evaluation, all from the training cache).
    n_probe = pz["x"].shape[0]
    wp = w[:n_probe]
    sub_seq = np.broadcast_to(np.arange(n_probe)[:, None], (n_probe, s))[wp]
    sub_test = sub_seq % 2 == 1
    sub_ben = benefit_all[:n_probe][wp]
    sub_need = sub_ben > NEED
    flat_index = np.flatnonzero(w.reshape(-1))
    sub_rows = flat_index < n_probe * s
    sc_sub = x_sc[sub_rows]
    xv = pz["x"].astype(np.float32)[wp]
    rv = pz["router3"].astype(np.float32)[wp]
    sets = dict(
        stream=xv,
        router3=rv,
        stream_router3_scalars=np.concatenate([xv, rv, sc_sub], 1),
        scalars=sc_sub,
    )
    sub_frontier = dict(
        rates=rates.tolist(),
        oracle=skip_cost(sub_ben[sub_test], sub_ben[sub_test], rates),
        random=(rates * sub_ben[sub_test].mean()).tolist(),
    )
    roc = {}
    for name, x in sets.items():
        logit = logistic_probe(x[~sub_test], sub_need[~sub_test], x[sub_test])
        reg = ridge_probe(x[~sub_test], sub_ben[~sub_test], x[sub_test])
        probes[f"vec_{name}_logistic"] = dict(auroc=auroc(logit, sub_need[sub_test]))
        probes[f"vec_{name}_ridge_benefit"] = dict(auroc=auroc(reg, sub_need[sub_test]))
        sub_frontier[f"{name}_ridge"] = skip_cost(reg, sub_ben[sub_test], rates)
        sub_frontier[f"{name}_logistic"] = skip_cost(logit, sub_ben[sub_test], rates)
        roc[name] = roc_curve(logit, sub_need[sub_test])
        logger.info("probe %s: auroc %.3f", name, probes[f"vec_{name}_logistic"]["auroc"])
    roc["exit_entropy"] = roc_curve(ent[test], need[test])
    roc["token_id_table"] = roc_curve(tt[test], need[test])
    # In-job probes (trained on 49M separate tokens, so every evaluated token is a test token).
    if "probe_pred" in z:
        pp = z["probe_pred"].astype(np.float32)
        inj = {}
        for e, ex_name in enumerate(z["probe_exits"]):
            b_e = exit_loss[stages.index(str(ex_name)), ph][w] - loss_full[w]
            entry = dict(
                oracle=skip_cost(b_e, b_e, rates),
                random=(rates * b_e.mean()).tolist(),
                need_rate=float((b_e > NEED).mean()),
            )
            for kk, kind in enumerate(z["probe_kinds"]):
                score = pp[kk, e][w]
                entry[str(kind)] = skip_cost(score, b_e, rates)
                entry[f"{kind}_auroc"] = auroc(score, b_e > NEED)
                entry[f"{kind}_corr"] = float(np.corrcoef(score, b_e)[0, 1])
                if e == 0:
                    roc[f"injob_{kind}"] = roc_curve(score, b_e > NEED)
            entry["exit_entropy"] = skip_cost(ent, b_e, rates)
            inj[str(ex_name)] = entry
        report["injob_probes"] = inj
        frontier["injob_mlp"] = inj["block2"]["mlp"]
        frontier["injob_linear"] = inj["block2"]["linear"]
    report["probes"] = probes
    report["frontier"] = frontier
    report["probe_subset_frontier"] = sub_frontier
    report["roc"] = roc
    report["skip_rate_table"] = {
        k: {f"{r:.1f}": v[int(r * 50 + 0.5)] for r in SKIP_RATES}
        for k, v in {**frontier, **sub_frontier}.items()
        if k != "rates"
    }
    # Worked examples: a few windows from the training evaluation with per-token benefit.
    rng = np.random.default_rng(0)
    examples = []
    for i in rng.choice(np.flatnonzero(sources == "train"), 6, replace=False):
        t0 = int(rng.integers(64, s - 96))
        sl = slice(t0, t0 + 64)
        examples.append(
            dict(
                seq=int(i),
                start=t0,
                tokens=[vocab[t] for t in tokens[i, sl]],
                target=[vocab[t] for t in target[i, sl]],
                benefit=np.round(benefit_all[i, sl], 3).tolist(),
                loss_full=np.round(loss_full[i, sl], 3).tolist(),
                weight=w[i, sl].tolist(),
            )
        )
    report["examples"] = examples
    # The hardest tokens (largest benefit) with a short context.
    flat = np.where(w, benefit_all, -np.inf).reshape(-1)
    top = np.argsort(-flat)[:40]
    report["hardest"] = [
        dict(
            context="".join(vocab[t] for t in tokens[i // s, max(0, i % s - 12) : i % s + 1]),
            target=vocab[target[i // s, i % s]],
            benefit=float(flat[i]),
            loss_full=float(loss_full.reshape(-1)[i]),
            source=str(sources[i // s]),
        )
        for i in top
    ]
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
