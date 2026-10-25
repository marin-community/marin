# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side analysis of the ``early`` study (``early_*.npz``) into the page's ``early.json``.

- token categories of every scored position (by its target ``y = tokens[t + 1]`` and context) and the mean loss
  change of each ablation / path patch per category;
- probe accuracy by site with count-based baselines (the best predictor from the current token alone);
- the KDA head taxonomy;
- routing entropy and mutual information with the current and the previous token (``early_probe`` routes, and the
  atlas expert-token counts);
- an expert census of L0-L2 from the atlas.
"""

import argparse
import json
import logging
import os

import numpy as np

from experiments.grug.fast_track.interp import atlas as atlas_pass
from experiments.grug.fast_track.interp import categories, early, forward

logger = logging.getLogger(__name__)

DETERMINED = 0.8  # P(y | context) for an n-gram-determined target
MIN_CONTEXT = 50  # occurrences of a bigram context (20 for a trigram one) to call it determined
EXAMPLE_WINDOW = 40  # tokens of each example attention pattern
DOC_START = 8  # "first tokens of a document": document positions below this
CATS = (
    "all",
    "word_start",
    "continuation",
    "bigram_cont",
    "copied",
    "novel",
    "format",
    "number",
    "code",
    "doc_start",
    "bigram_det",
    "trigram_only",
    "function_word",
)
CAT_LABELS = {
    "all": "all tokens",
    "word_start": "word start",
    "continuation": "word-piece continuation",
    "bigram_cont": "continuation fixed by prev token (produ→cer)",
    "copied": "copied bigram (seen earlier in doc)",
    "novel": "novel token (not yet in doc)",
    "format": "punctuation / whitespace / newline",
    "number": "number",
    "code": "code (programming-language docs)",
    "doc_start": f"first {DOC_START} tokens of a doc",
    "bigram_det": "bigram-determined (P≥0.8)",
    "trigram_only": "(a,b)→c determined, b alone not",
    "function_word": "function word",
}


def ngram_tables(tokens: np.ndarray, seg: np.ndarray, valid: np.ndarray):
    """Bigram ``P(y | x)`` and trigram ``P(c | a, b)`` lookups (dicts keyed by packed ids) from training rows."""
    v = np.int64(16384)
    x, y = tokens[:, :-1].astype(np.int64), tokens[:, 1:].astype(np.int64)
    ok = (seg[:, 1:] == seg[:, :-1]) & valid[:, 1:].astype(bool)
    pair = (x * v + y)[ok]
    keys, counts = np.unique(pair, return_counts=True)
    ctx_tot = np.bincount(keys // v, weights=counts, minlength=int(v))
    a, b, c = tokens[:, :-2].astype(np.int64), tokens[:, 1:-1].astype(np.int64), tokens[:, 2:].astype(np.int64)
    ok3 = ok[:, 1:] & ok[:, :-1]
    tri = ((a * v + b) * v + c)[ok3]
    tkeys, tcounts = np.unique(tri, return_counts=True)
    ctx = tkeys // v
    ukeys, inv = np.unique(ctx, return_inverse=True)
    tctx_tot = np.bincount(inv, weights=tcounts)
    return dict(keys=keys, counts=counts, ctx_tot=ctx_tot, tkeys=tkeys, tcounts=tcounts, tctx=ukeys, tctx_tot=tctx_tot)


def _lookup(keys, vals, query, default=0.0):
    i = np.searchsorted(keys, query)
    i = np.minimum(i, len(keys) - 1)
    return np.where(keys[i] == query, vals[i], default)


def bigram_p(tab, x, y):
    v = np.int64(16384)
    cnt = _lookup(tab["keys"], tab["counts"], x.astype(np.int64) * v + y)
    tot = tab["ctx_tot"][x]
    return np.where(tot >= MIN_CONTEXT, cnt / np.maximum(tot, 1), np.nan), tot


def trigram_p(tab, a, b, c):
    v = np.int64(16384)
    ctx = a.astype(np.int64) * v + b
    cnt = _lookup(tab["tkeys"], tab["tcounts"], ctx * v + c)
    tot = _lookup(tab["tctx"], tab["tctx_tot"], ctx)
    return np.where(tot >= 20, cnt / np.maximum(tot, 1), np.nan)


def token_categories(tokens, seg, weight, sources, vocab, tab) -> dict[str, np.ndarray]:
    """``{category: [N, S] bool}`` over scored positions (the target in the same document)."""
    pf = early.piece_features(vocab)
    feats = categories.token_features(vocab)
    widx = early.word_index(tokens, seg, pf)
    pos = early.position_in_doc(seg)
    target = np.pad(tokens[:, 1:], ((0, 0), (0, 1)))
    same = np.pad(seg[:, 1:] == seg[:, :-1], ((0, 0), (0, 1)))
    scored = (weight > 0) & same
    cont = np.pad(widx[:, 1:] >= 1, ((0, 0), (0, 1)))
    p_bi, _ = bigram_p(tab, tokens, target)
    prev = np.pad(tokens[:, :-1], ((0, 0), (1, 0)))
    p_tri = trigram_p(tab, prev, tokens, target)
    p_tri = np.where(pos >= 1, p_tri, np.nan)
    copied = categories.copy_targets(tokens, seg)
    seen_next = np.pad(early.seen_before(tokens, seg)[:, 1:], ((0, 0), (0, 1)))
    bi_det = np.nan_to_num(p_bi) >= DETERMINED
    out = {
        "all": np.ones_like(scored),
        "word_start": feats["word_start"][target] | feats["function_word"][target],
        "continuation": cont,
        "bigram_cont": cont & bi_det,
        "copied": copied,
        "novel": ~seen_next,
        "format": pf["delim"][target],
        "number": feats["digit"][target],
        "code": np.broadcast_to((sources == early.CODE)[:, None], tokens.shape),
        "doc_start": pos < DOC_START,
        "bigram_det": bi_det,
        "trigram_only": (np.nan_to_num(p_tri) >= DETERMINED) & (np.nan_to_num(p_bi, nan=1.0) < 0.5),
        "function_word": feats["function_word"][target],
    }
    return {k: v & scored for k, v in out.items()}


def category_effects(loss: np.ndarray, names, cats: dict[str, np.ndarray], base: int = 0) -> dict:
    """Mean loss change (config minus ``base``) per category: ``{"configs", "cats", "delta" [C, K], "n" [K]}``."""
    l32 = loss.astype(np.float32)
    delta = np.zeros((len(names), len(CATS)))
    n = np.zeros(len(CATS))
    for k, c in enumerate(CATS):
        m = cats[c][: loss.shape[1]]
        n[k] = m.sum()
        if n[k] == 0:
            continue
        b = l32[base][m].mean()
        delta[:, k] = [l32[ci][m].mean() - b for ci in range(len(names))]
    return dict(
        configs=[str(x) for x in names],
        cats=list(CATS),
        delta=delta.round(4).tolist(),
        n=n.astype(int).tolist(),
        base_loss=[float(l32[base][cats[c][: loss.shape[1]]].mean()) if n[k] else 0.0 for k, c in enumerate(CATS)],
    )


def count_baseline(cond: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, num_cond: int) -> float:
    """Test accuracy of ``argmax_y count(cond, y)`` fit on ``train`` (falling back to the global mode)."""
    ok_tr = train & (y >= 0)
    ok_te = test & (y >= 0)
    if ok_te.sum() == 0:
        return float("nan")
    key = cond[ok_tr].astype(np.int64) * (int(y.max()) + 1) + y[ok_tr]
    keys, counts = np.unique(key, return_counts=True)
    c_of, y_of = keys // (int(y.max()) + 1), keys % (int(y.max()) + 1)
    order = np.lexsort((-counts, c_of))
    first = np.ones(len(order), bool)
    first[1:] = c_of[order][1:] != c_of[order][:-1]
    best = np.full(num_cond, np.bincount(y[ok_tr]).argmax())
    best[c_of[order][first]] = y_of[order][first]
    return float(np.mean(best[cond[ok_te]] == y[ok_te]))


def probe_section(probe: dict, toks: dict, vocab) -> dict:
    tokens, seg, weight, sources = toks["tokens"].astype(np.int32), toks["segment_ids"], toks["weight"], toks["sources"]
    n, s = tokens.shape
    test_rows = np.arange(n) % 3 == 0
    pf = early.piece_features(vocab)
    labels = early.probe_labels(tokens, seg, weight, sources, pf, ~test_rows)
    test = np.repeat(test_rows, s)
    train = ~test
    cur = tokens.reshape(-1)
    prev = np.pad(tokens[:, :-1], ((0, 0), (1, 0))).reshape(-1)
    base = {}
    for name in early.NUM_CLASSES:
        y = labels[name].reshape(-1)
        base[name] = {
            "majority": count_baseline(np.zeros_like(cur), y, train, test, 1),
            "from_cur": count_baseline(cur, y, train, test, 16384),
        }
        if name in ("word", "inword", "word_first"):
            pair = cur.astype(np.int64) * 16384 + prev
            uniq, inv = np.unique(pair, return_inverse=True)
            base[name]["from_cur_prev"] = count_baseline(inv, y, train, test, len(uniq))
    sites = list(early.SITE_NAMES)
    targets = [*early.NUM_CLASSES, "logpos_doc", "logpos_abs", "entropy"]
    clean = {t: [float(probe[f"clean/{site}/{t}"][0]) for site in sites] for t in targets}
    clean_xent = {t: [float(probe[f"clean/{site}/{t}"][1]) for site in sites] for t in early.NUM_CLASSES}
    ablated = {}
    for key in probe:
        parts = key.split("/")
        if len(parts) == 3 and parts[0] != "clean":
            ablated.setdefault(parts[0], {}).setdefault(parts[1], {})[parts[2]] = float(probe[key][0])
    counts = {name: int((labels[name].reshape(-1)[test] >= 0).sum()) for name in early.NUM_CLASSES}
    return dict(sites=sites, clean=clean, clean_xent=clean_xent, baseline=base, ablated=ablated, test_counts=counts)


def entropy_bits(p: np.ndarray, axis=-1) -> np.ndarray:
    p = p / np.maximum(p.sum(axis=axis, keepdims=True), 1e-30)
    return -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-30)), 0.0), axis=axis)


def mutual_info(counts: np.ndarray) -> float:
    """``I(E; X)`` in bits from a ``[X, E]`` count table."""
    pxy = counts / counts.sum()
    px = pxy.sum(1, keepdims=True)
    py = pxy.sum(0, keepdims=True)
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / (px @ py)[nz])))


def route_table(x: np.ndarray, e: np.ndarray) -> np.ndarray:
    """``[distinct x, E]`` slot counts of the ``[T, K]`` routes ``e`` by the token feature ``x [T]``."""
    uniq, inv = np.unique(x, return_inverse=True)
    t = np.zeros((len(uniq), 384))
    np.add.at(t, (np.repeat(inv, e.shape[1]), e.reshape(-1)), 1.0)
    return t


def routing_section(probe: dict, toks: dict, atlas: dict, freq_min: int = 200) -> dict:
    """Routing entropy per layer and ``I(expert; token)``, ``I(expert; previous token)`` and their
    shuffled-token bias controls on the probe set (frequent tokens only); the atlas' 24.5M-token ``I(E; token)``."""
    tokens = toks["tokens"].astype(np.int64)
    valid = (toks["weight"] > 0).reshape(-1)
    seg = toks["segment_ids"]
    cur = tokens.reshape(-1)
    prev = np.pad(tokens[:, :-1], ((0, 0), (1, 0))).reshape(-1)
    same = np.pad(seg[:, 1:] == seg[:, :-1], ((0, 0), (1, 0))).reshape(-1)
    sel = probe["selected"].astype(np.int64)  # [L, T, K]
    cnt_c = np.bincount(cur[valid], minlength=16384)
    cnt_p = np.bincount(prev[valid & same], minlength=16384)
    ok = valid & same & (cnt_c[cur] >= freq_min) & (cnt_p[prev] >= freq_min)
    rng = np.random.default_rng(0)
    out = {
        "H_route": [],
        "I_cur": [],
        "I_prev": [],
        "I_cur_shuffled": [],
        "I_prev_shuffled": [],
        "atlas_I_cur": [],
        "atlas_H": [],
        "top_share": [],
        "n": int(ok.sum()),
    }
    idx = np.flatnonzero(ok)
    shuffled = rng.permutation(idx)
    for i in range(early.L):
        e = sel[i][idx]  # [T, K]
        flat_e = e.reshape(-1)

        out["H_route"].append(float(entropy_bits(np.bincount(flat_e, minlength=384).astype(float))))
        out["I_cur"].append(mutual_info(route_table(cur[idx], e)))
        out["I_prev"].append(mutual_info(route_table(prev[idx], e)))
        out["I_cur_shuffled"].append(mutual_info(route_table(cur[shuffled], e)))
        out["I_prev_shuffled"].append(mutual_info(route_table(prev[shuffled], e)))
        et = atlas["expert_token_ids"][i].astype(np.float64).T  # [V, E]
        keep = et.sum(1) >= freq_min * 10
        out["atlas_I_cur"].append(mutual_info(et[keep]))
        out["atlas_H"].append(float(entropy_bits(et[keep].sum(0))))
        # Per token: the share of its slots that go to its 10 most-used experts.
        tk = et[keep]
        top10 = -np.sort(-tk, axis=1)[:, :10].sum(1)
        out["top_share"].append(float(top10.sum() / tk.sum()))
    return out


def expert_census(atlas: dict, vocab: list[str]) -> dict:
    """Each L0-L2 expert's traffic profile and a category (first match): ``token`` (its 20 most-routed token ids
    carry at least half its traffic), ``format`` (at least 60% punctuation / whitespace / newline tokens),
    ``number`` (at least 50% digit tokens), ``continuation`` (at least 70% in-word pieces without a leading space),
    ``function`` (at least 60% function words), ``domain`` (at least 40% of its Paloma traffic from one domain group at
    2.5x that group's base share), else ``broad``."""
    pf = early.piece_features(vocab)
    feats = categories.token_features(vocab)
    groups = ("web", "code", "social", "science", "wiki", "news")
    src_group = np.asarray(
        [groups.index(g) if (g := atlas_pass.SOURCE_GROUP[s]) in groups else -1 for s in early.SOURCES]
    )
    out = {"layers": []}
    cats = ("token", "format", "number", "continuation", "function", "domain", "broad")
    for i in early.EARLY:
        et = atlas["expert_token_ids"][i].astype(np.float64)  # [E, V]
        tot = et.sum(1)
        top20 = -np.sort(-et, axis=1)[:, :20].sum(1) / tot
        fmt = et[:, pf["delim"]].sum(1) / tot
        num = et[:, feats["digit"]].sum(1) / tot
        cont = et[:, pf["cont"]].sum(1) / tot
        func = et[:, feats["function_word"]].sum(1) / tot
        dom = atlas["expert_dom"][i].astype(np.float64)  # [E, 17]
        pal = np.stack([dom[:, src_group == g].sum(1) for g in range(len(groups))], 1)
        pal_share = pal / np.maximum(pal.sum(1, keepdims=True), 1)
        base = pal.sum(0) / pal.sum()
        lift = pal_share / base
        dom_ok = (pal_share >= 0.4) & (lift >= 2.5)
        label = np.full(384, len(cats) - 1)
        for k, mask in reversed(
            list(enumerate([top20 >= 0.5, fmt >= 0.6, num >= 0.5, cont >= 0.7, func >= 0.6, dom_ok.any(1)]))
        ):
            label[mask] = k
        counts = {c: int((label == k).sum()) for k, c in enumerate(cats)}
        examples = {}
        for k, c in enumerate(cats):
            ids = np.flatnonzero(label == k)
            ids = ids[np.argsort(-tot[ids])][:6]
            examples[c] = [
                dict(
                    e=int(e),
                    share=float(tot[e] / tot.sum()),
                    top=[vocab[j] for j in np.argsort(-et[e])[:6]],
                    top20=float(top20[e]),
                    dom=groups[int(np.argmax(lift[e]))] if dom_ok[e].any() else "",
                )
                for e in ids
            ]
        out["layers"].append(
            dict(
                layer=i,
                counts=counts,
                traffic={c: float(tot[label == k].sum() / tot.sum()) for k, c in enumerate(cats)},
                median_top20=float(np.median(top20)),
                mean_format=float(np.mean(fmt)),
                examples=examples,
            )
        )
    out["categories"] = list(cats)
    return out


def heads_section(hz: dict) -> dict:
    """Per head of every layer with head statistics (``early.HEAD_LAYERS``; L0-L2 in the d512 data): distance
    profile, self / previous / doc-start / same-token / delimiter / in-word masses (both weightings), a label, and
    the induction table."""
    out = {"heads": [], "edges": [0, 1, 2, 5, 17, 65, 257, 1025, 4097]}
    for i in [li for li in range(early.L) if f"stats/L{li}/n" in hz]:
        p = f"stats/L{i}/"
        n = hz[p + "n"]
        for h in range(early.H):
            d = {}
            for pre in ("", "v_"):
                q = p + pre
                d[pre + "by_dist"] = (hz[q + "by_dist"][h] / n).round(4).tolist()
                for k in ("self_mass", "prev_mass", "doc_start", "entropy", "mean_dist", "delim_mass"):
                    d[pre + k] = float(hz[q + k][h] / n)
                d[pre + "same_token"] = float(hz[q + "same_token"][h] / max(hz[q + "n_same_token"], 1))
                d[pre + "delim_base"] = float(hz[q + "delim_base"] / n)
                d[pre + "inword_mass"] = float(hz[q + "inword_mass"][h] / max(hz[q + "n_cont"], 1))
                d[pre + "inword_prev"] = float(hz[q + "inword_prev"][h] / max(hz[q + "n_cont"], 1))
                d[pre + "wstart_mass"] = float(hz[q + "wstart_mass"][h] / max(hz[q + "n_deep"], 1))
                d[pre + "wstart_prev"] = float(hz[q + "wstart_prev"][h] / max(hz[q + "n_deep"], 1))
            d["neg_frac"] = float(hz[p + "neg_frac"][h] / n) if p + "neg_frac" in hz else None
            d["label"] = head_label(d)
            d["name"] = f"L{i}.h{h}"
            out["heads"].append(d)
    out["induction"] = dict(
        names=[str(x) for x in hz["ind_names"]],
        first=hz["ind_first"].round(4).tolist(),
        second=hz["ind_second"].round(4).tolist(),
        l3_mass=hz["ind_l3_mass"].round(4).tolist(),
    )
    ex = {}
    for k in hz:
        if k.startswith("example/"):
            ex[k[len("example/") :]] = hz[k]
    out["example_source"] = [early.SOURCES[int(s)] for s in ex["source"]]
    return out, ex


def head_label(d: dict) -> str:
    """The head's class from its value-weighted profile: ``previous-token`` (at least 40% at distance 1),
    ``self`` (at least 40% at distance 0), ``local`` (at least 70% within 4 tokens), ``delimiter`` (delimiter mass at
    least twice the base rate and 25%), ``sink`` (at least 20% on the document start), ``long-range`` (at least 30%
    beyond 64 tokens), else ``mid-range``."""
    bd = d["v_by_dist"]
    if bd[1] >= 0.4:
        return "previous-token"
    if bd[0] >= 0.4:
        return "self"
    if d["v_doc_start"] >= 0.2:
        return "sink"
    if d["v_delim_mass"] >= max(2 * d["v_delim_base"], 0.25):
        return "delimiter"
    if sum(bd[:3]) >= 0.7:
        return "local"
    if sum(bd[5:]) >= 0.3:
        return "long-range"
    return "mid-range"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        required=True,
        help="directory with the early_*.npz and atlas.npz + atlas_tokens.npz (else stats.npz + tokens.npz)",
    )
    parser.add_argument("--vocab", required=True, help="vocab.json (list of token strings)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.vocab) as f:
        vocab = json.load(f)
    d = args.data
    toks = dict(np.load(f"{d}/early_tokens.npz"))
    has_atlas = os.path.exists(f"{d}/atlas.npz")
    at = np.load(f"{d}/atlas_tokens.npz" if has_atlas else f"{d}/tokens.npz")
    train = at["sources"] == (0 if has_atlas else "train")
    tab = ngram_tables(at["tokens"][train].astype(np.int64), at["segment_ids"][train], at["valid"][train])
    cats = token_categories(
        toks["tokens"].astype(np.int64), toks["segment_ids"], toks["weight"], toks["sources"], vocab, tab
    )
    page: dict = {"cat_labels": CAT_LABELS}
    ab = np.load(f"{d}/early_ablate.npz")
    page["ablate"] = category_effects(ab["loss"], ab["names"], cats)
    co = np.load(f"{d}/early_consume.npz")
    page["consume"] = category_effects(co["loss"], co["names"], cats)
    page["consume"]["overlap"] = co["overlap"].round(4).tolist()
    page["consume"]["gate_w"] = co["gate_w"].round(5).tolist()
    page["consume"]["gate_names"] = [str(x) for x in co["gate_names"]]
    page["consume"]["units"] = unit_dependence(co, vocab)
    pr = np.load(f"{d}/early_probe.npz")
    page["probe"] = probe_section(dict(pr), toks, vocab)
    # Without an atlas the stats pass's expert-token counts stand in (no per-domain counts: no "domain" experts).
    atlas = np.load(f"{d}/atlas.npz" if has_atlas else f"{d}/stats.npz", mmap_mode="r")
    page["routing"] = routing_section(dict(pr), toks, {"expert_token_ids": atlas["expert_token_ids"]})
    dom = (
        atlas["expert_dom"]
        if has_atlas
        else np.zeros((*atlas["expert_token_ids"].shape[:2], len(early.SOURCES)), np.float32)
    )
    page["census"] = expert_census({"expert_token_ids": atlas["expert_token_ids"], "expert_dom": dom}, vocab)
    hz = np.load(f"{d}/early_heads.npz")
    page["heads"], examples = heads_section(dict(hz))
    rows = [r for r, src in enumerate(page["heads"]["example_source"]) if src in ("c4_en", "train")][:2]
    page["head_examples"] = {
        "rows": rows,
        "source": [page["heads"]["example_source"][r] for r in rows],
        "tokens": [[vocab[int(t)] for t in examples["tokens"][r][:EXAMPLE_WINDOW]] for r in rows],
        **{
            k: np.round(v[rows][:, :, :EXAMPLE_WINDOW, :EXAMPLE_WINDOW], 3).tolist()
            for k, v in examples.items()
            if k.startswith("L") and k.endswith("_v")
        },
    }
    with open(args.out, "w") as f:
        json.dump(page, f, default=float)
    logger.info("wrote %s", args.out)


def unit_dependence(co, vocab) -> dict:
    """Per L3-L5 layer and frozen-routing patch: the share of the routed units' clean activation mass lost, the
    experts with the largest lost share (among the 100 with the most mass), and named units of interest."""
    clean = co["unit_clean"]  # [3, E, I]
    signed = co["unit_signed"]  # [F, 3, E, I]
    names = [str(x) for x in co["frozen_names"]]
    out = {"names": names, "layers": [3, 4, 5], "lost_share": [], "shared_lost_share": [], "top_experts": [], "e375": {}}
    for fi, name in enumerate(names):
        out["lost_share"].append([float(-signed[fi, j].sum() / clean[j].sum()) for j in range(3)])
        out["shared_lost_share"].append(
            [float(-co["shared_signed"][fi, j].sum() / co["shared_clean"][j].sum()) for j in range(3)]
        )
        tops = []
        for j in range(3):
            mass = clean[j].sum(1)
            cand = np.argsort(-mass)[:100]
            lost = -signed[fi, j].sum(1)[cand] / mass[cand]
            order = np.argsort(-lost)[:8]
            tops.append(
                [dict(e=int(cand[o]), lost=float(lost[o]), mass=float(mass[cand[o]] / mass.sum())) for o in order]
            )
        out["top_experts"].append(tops)
        e = 375  # the d512 L4 ics expert (its named units are d512's)
        if forward.NUM_LAYERS != 6:
            continue
        out["e375"][name] = dict(
            expert=float(-signed[fi, 1, e].sum() / max(clean[1, e].sum(), 1e-9)),
            units={str(u): float(-signed[fi, 1, e, u] / max(clean[1, e, u], 1e-9)) for u in (79, 218, 263, 306, 372)},
        )
    return out


if __name__ == "__main__":
    main()
