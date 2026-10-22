# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Condense the study's npz outputs (``verify``, ``stats``, ``attrib``, ``heads``, downloaded to a local directory)
into one JSON document for the report page.

    python -m experiments.grug.fast_track.interp.report_data <data dir> <out.json> [--compare <other data dir>]
"""

import argparse
import json
import os

import numpy as np

from experiments.grug.fast_track.interp import forward

L, H, E, I = forward.NUM_LAYERS, forward.NUM_HEADS, forward.NUM_EXPERTS, forward.EXPERT_WIDTH
DEAD_RATE = 1e-4  # fires on fewer than 1 in 10k of its expert's tokens
DENSE_RATE = 0.5
NEURON_CARDS_PER_LAYER = 8
CONTEXT_WINDOW = 12  # tokens of left context shown per top activation
CARD_CONTEXTS = 6


def _r(x, n=4):
    return np.round(np.asarray(x, np.float64), n).tolist()


def heads_section(h) -> dict:
    out = {"layers": [], "dist_edges": h["dist_edges"].tolist()}
    for i in range(L):
        kind = "MLA" if i in forward.MLA_LAYERS else "KDA"
        per = {}
        for src in ("natural", "induction"):
            p = f"{src}/L{i}/"
            n = float(h[p + "n"])
            d = dict(
                entropy=_r(h[p + "entropy"] / n),
                mean_dist=_r(h[p + "mean_dist"] / n, 1),
                by_dist=_r(h[p + "by_dist"] / n),
                prev=_r(h[p + "prev_mass"] / n),
                self=_r(h[p + "self_mass"] / n),
                doc_start=_r(h[p + "doc_start"] / n),
                induction=_r(h[p + "induction"] / max(float(h[p + "n_induction"]), 1)),
                same_token=_r(h[p + "same_token"] / max(float(h[p + "n_same_token"]), 1)),
            )
            if kind == "MLA":
                d["e_prev"] = _r(h[p + "e_prev"] / n, 3)
                d["e_cur"] = _r(h[p + "e_cur"] / n, 3)
                d["e_bias"] = _r(h[p + "e_bias"] / n, 3)
                bias = h[p + "bias_by_dist"] / n
                keep = np.unique(np.round(np.geomspace(1, bias.shape[1], 60)).astype(int) - 1)
                d["bias_dist"] = keep.tolist()
                d["bias_by_dist"] = _r(bias[:, keep], 3)
            else:
                g = h[p + "log_decay_sum"] / n  # [H, D] mean log decay per channel
                half = np.log(0.5) / np.minimum(g, -1e-9)
                d["half_life"] = _r(np.sort(half, axis=1), 2)
                d["beta"] = _r(h[p + "beta_sum"] / n, 3)
                d["beta_hist"] = _r(h[p + "beta_hist"] / n, 4)
                d["neg_frac"] = _r(h[p + "neg_frac"] / n, 3)
            per[src] = d
        out["layers"].append(dict(layer=i, kind=kind, **per))
    names = [str(x) for x in h["induction_names"]]
    first, second = h["induction_first"], h["induction_second"]
    out["induction"] = dict(
        names=names,
        first=_r(first, 4),
        second=_r(second, 4),
        pos_loss_base=_r(h["induction_pos_loss"][0], 3),
    )
    out["examples"] = {
        k.removeprefix("example/"): _r(h[k][0], 3) for k in h.files if k.startswith("example/") and "tokens" not in k
    }
    out["example_tokens"] = {
        k.removeprefix("example/").removesuffix("/tokens"): h[k][0].tolist() for k in h.files if k.endswith("/tokens")
    }
    out["kda_recon_err_max"] = float(h["kda_recon_err"].max())
    return out


def _context(tokens, flat_pos, vocab, seq_len) -> dict:
    seq, pos = divmod(int(flat_pos), seq_len)
    lo = max(0, pos - CONTEXT_WINDOW)
    row = tokens[seq]
    return dict(
        left="".join(vocab[t] for t in row[lo:pos]),
        tok=vocab[row[pos]],
        next=vocab[row[pos + 1]] if pos + 1 < seq_len else "",
        tok_id=int(row[pos]),
    )


def _concentration(ids: np.ndarray) -> float:
    """Share of the most common current token among a neuron's top activations."""
    if len(ids) == 0:
        return 0.0
    _, counts = np.unique(ids, return_counts=True)
    return float(counts.max() / len(ids))


def neurons_section(s, tok, vocab, seq_len: int, rng: np.random.Generator) -> dict:
    tokens = tok["tokens"].astype(np.int64)
    expert_tokens = s["expert_tokens"]  # [L, E]
    rate = s["fire"] / np.maximum(expert_tokens[..., None], 1)  # [L, E, I]
    mean_act = s["act_sum"] / np.maximum(expert_tokens[..., None], 1)
    total = float(s["tokens"])
    s_rate = s["s_fire"] / total
    out: dict = {"tokens": total, "mean_loss": float(s["loss_sum"] / total)}
    bins = np.linspace(-6, 0, 61)
    out["rate_bins"] = _r(bins, 2)
    layers = []
    for i in range(L):
        alive_experts = expert_tokens[i] > 0
        rr = rate[i][alive_experts].reshape(-1)
        hist_r = np.histogram(np.log10(np.maximum(rr, 1e-7)), bins=bins)[0]
        hist_s = np.histogram(np.log10(np.maximum(s_rate[i], 1e-7)), bins=bins)[0]
        load = expert_tokens[i] / expert_tokens[i].sum()
        ent = -np.sum(load * np.log(np.maximum(load, 1e-30)))
        # Token specialization: per expert, entropy of its current-token distribution vs the corpus's.
        ids = s["expert_token_ids"][i].astype(np.float64)  # [E, V]
        p_tok = ids / np.maximum(ids.sum(1, keepdims=True), 1)
        h_tok = -np.sum(p_tok * np.log(np.maximum(p_tok, 1e-30)), axis=1)
        corpus = s["token_counts"] / s["token_counts"].sum()
        h_corpus = float(-np.sum(corpus * np.log(np.maximum(corpus, 1e-30))))
        # Mutual information between expert choice and current token, per assignment.
        joint = ids / ids.sum()
        pe = joint.sum(1, keepdims=True)
        pv = joint.sum(0, keepdims=True)
        mi = float(np.sum(np.where(joint > 0, joint * np.log(np.maximum(joint, 1e-30) / np.maximum(pe * pv, 1e-30)), 0)))
        top_tok_share = ids.max(1) / np.maximum(ids.sum(1), 1)
        top_tok = ids.argmax(1)
        layers.append(
            dict(
                layer=i,
                routed_rate_hist=hist_r.tolist(),
                shared_rate_hist=hist_s.tolist(),
                routed_dead=int(np.sum(rr < DEAD_RATE)),
                routed_dense=int(np.sum(rr > DENSE_RATE)),
                routed_mean_rate=float(np.mean(rr)),
                routed_median_rate=float(np.median(rr)),
                shared_dead=int(np.sum(s_rate[i] < DEAD_RATE)),
                shared_dense=int(np.sum(s_rate[i] > DENSE_RATE)),
                shared_mean_rate=float(np.mean(s_rate[i])),
                shared_median_rate=float(np.median(s_rate[i])),
                act_hist=_r(s["act_hist"][i] / max(s["act_hist"][i].sum(), 1), 5),
                s_act_hist=_r(s["s_act_hist"][i] / max(s["s_act_hist"][i].sum(), 1), 5),
                load_sorted=_r(np.sort(load)[::-1] * E, 3),
                load_entropy_ratio=float(ent / np.log(E)),
                unused_experts=int(np.sum(~alive_experts)),
                expert_token_entropy=_r(np.sort(h_tok), 3),
                corpus_token_entropy=h_corpus,
                mi_expert_token=mi,
                top_token_share_sorted=_r(np.sort(top_tok_share)[::-1], 3),
                most_specialized=[
                    dict(
                        expert=int(e),
                        share=float(top_tok_share[e]),
                        token=vocab[int(top_tok[e])],
                        tokens=int(ids[e].sum()),
                    )
                    for e in np.argsort(-top_tok_share)[:12]
                ],
            )
        )
    out["layers"] = layers
    out["rate_by_layer"] = dict(
        routed_mean=[float(np.mean(rate[i])) for i in range(L)],
        shared_mean=[float(np.mean(s_rate[i])) for i in range(L)],
    )
    # Neuron cards: a random sample per layer (routed and shared), with top contexts and logit-lens tokens.
    cards = []
    conc_all = {"routed": [], "shared": []}
    for i in range(L):
        alive = np.argwhere((rate[i] > 1e-3) & (expert_tokens[i][:, None] > 50))
        picks = alive[rng.choice(len(alive), size=NEURON_CARDS_PER_LAYER, replace=False)]
        # Monosemanticity proxy over every neuron: top-context token concentration.
        tp = s["top_pos"][i]  # [E, I, k]
        tv = s["top_val"][i]
        cur = tokens.reshape(-1)[np.clip(tp, 0, tokens.size - 1)]
        conc = np.zeros((E, I))
        for e in range(E):
            for n in range(I):
                ok = tv[e, n] > 0
                conc[e, n] = _concentration(cur[e, n][ok])
        conc_all["routed"].append(np.histogram(conc[rate[i] > 1e-3], bins=np.linspace(0, 1, 21))[0].tolist())
        for e, n in picks:
            cards.append(_card("routed", i, int(e), int(n), s, tokens, vocab, seq_len, rate, mean_act, conc[e, n]))
        stp, stv = s["s_top_pos"][i], s["s_top_val"][i]
        s_conc = np.array([_concentration(tokens.reshape(-1)[stp[n][stv[n] > 0]]) for n in range(I)])
        conc_all["shared"].append(np.histogram(s_conc, bins=np.linspace(0, 1, 21))[0].tolist())
        for n in rng.choice(I, size=3, replace=False):
            cards.append(_card("shared", i, None, int(n), s, tokens, vocab, seq_len, rate, mean_act, s_conc[n]))
    out["cards"] = cards
    out["concentration_hist"] = conc_all
    return out


def _card(kind, layer, expert, neuron, s, tokens, vocab, seq_len, rate, mean_act, conc) -> dict:
    if kind == "routed":
        tv, tp = s["top_val"][layer, expert, neuron], s["top_pos"][layer, expert, neuron]
        lens = [s[f"lens_{k}"][layer, expert, neuron] for k in ("top_id", "top_val", "bot_id", "bot_val")]
        r, m = float(rate[layer, expert, neuron]), float(mean_act[layer, expert, neuron])
        mx = float(s["act_max"][layer, expert, neuron])
        std = float(s["lens_logit_std"][layer, expert, neuron])
    else:
        tv, tp = s["s_top_val"][layer, neuron], s["s_top_pos"][layer, neuron]
        lens = [s[f"s_lens_{k}"][layer, neuron] for k in ("top_id", "top_val", "bot_id", "bot_val")]
        r = float(s["s_fire"][layer, neuron] / s["tokens"])
        m = float(s["s_act_sum"][layer, neuron] / s["tokens"])
        mx = float(s["s_act_max"][layer, neuron])
        std = float(s["s_lens_logit_std"][layer, neuron])
    contexts = [
        dict(act=float(v), **_context(tokens, p, vocab, seq_len))
        for v, p in zip(tv[:CARD_CONTEXTS], tp, strict=False)
        if v > 0
    ]
    return dict(
        kind=kind,
        layer=layer,
        expert=expert,
        neuron=neuron,
        rate=r,
        mean_act=m,
        max_act=mx,
        concentration=float(conc),
        contexts=contexts,
        promotes=[[vocab[int(t)], float(v)] for t, v in zip(lens[0][:8], lens[1][:8], strict=True)],
        suppresses=[[vocab[int(t)], float(v)] for t, v in zip(lens[2][:8], lens[3][:8], strict=True)],
        logit_std=std,
    )


def _quantiles(centers: np.ndarray, cdf: np.ndarray, p: float) -> list[float]:
    return [float(centers[np.searchsorted(c, p)]) for c in cdf]


def routing_section(s) -> dict:
    total = float(s["tokens"])
    out = {"layers": []}
    wbins = np.linspace(0, 2.5, s["weight_hist"].shape[-1] + 1)
    out["weight_bins"] = _r(wbins, 3)
    out["eff_bins"] = _r(np.linspace(1, 11, s["eff_hist"].shape[-1] + 1), 2)
    out["norm_bins"] = _r(np.linspace(-4, 4, s["norm_hist"].shape[-1] + 1), 2)
    for i in range(L):
        wh = s["weight_hist"][i]  # [K, bins]
        centers = (wbins[1:] + wbins[:-1]) / 2
        mean_by_rank = (wh * centers).sum(1) / np.maximum(wh.sum(1), 1)
        cdf = np.cumsum(wh, axis=1) / np.maximum(wh.sum(1, keepdims=True), 1)
        eh = s["eff_hist"][i]
        ec = np.linspace(1, 11, len(eh) + 1)
        ec = (ec[1:] + ec[:-1]) / 2
        selections = s["expert_tokens"][i]
        bias = s["router_bias"][i]
        mean_logit = s["logit_sum"][i] / total
        sel_logit = s["selected_logit_sum"][i] / np.maximum(selections, 1)
        bias_frac = s["bias_driven"][i] / np.maximum(selections, 1)
        pas = s["passengers"][i] / np.maximum(selections[None], 1)  # [fractions, E]
        mean_w = s["slot_weight_sum"][i] / np.maximum(selections, 1)
        mean_norm = s["slot_norm_sum"][i] / np.maximum(selections, 1)
        bdh = s["bias_driven_hist"][i]  # [2, bins]
        out["layers"].append(
            dict(
                layer=i,
                weight_hist_by_rank=_r(wh / np.maximum(wh.sum(1, keepdims=True), 1), 5),
                weight_mean_by_rank=_r(mean_by_rank, 4),
                weight_p10_by_rank=_r(_quantiles(centers, cdf, 0.1), 3),
                weight_p90_by_rank=_r(_quantiles(centers, cdf, 0.9), 3),
                eff_hist=_r(eh / eh.sum(), 5),
                eff_mean=float((eh * ec).sum() / eh.sum()),
                norm_hist_by_rank=_r(s["norm_hist"][i] / np.maximum(s["norm_hist"][i].sum(1, keepdims=True), 1), 5),
                frac_norm_by_rank=_r(s["frac_norm_sum"][i] / total, 4),
                passenger_rate=_r(s["passengers"][i].sum(1) / (total * forward.TOP_K), 5),
                passenger_norm_share=_r(
                    s["passenger_norm"][i].sum(1) / max(float(s["slot_norm_sum"][i].sum()), 1e-9), 5
                ),
                bias_driven_rate=float(s["bias_driven"][i].sum() / (total * forward.TOP_K)),
                bias_driven_rank=_r(s["bias_driven_rank"][i] / max(float(s["bias_driven"][i].sum()), 1), 4),
                bias_driven_weight_hist=_r(bdh / np.maximum(bdh.sum(1, keepdims=True), 1), 5),
                bias_driven_mean_weight=_r((bdh * centers).sum(1) / np.maximum(bdh.sum(1), 1), 4),
                experts=dict(
                    bias=_r(bias, 4),
                    mean_logit=_r(mean_logit, 4),
                    selected_logit=_r(sel_logit, 4),
                    freq=_r(selections / total, 5),
                    bias_frac=_r(bias_frac, 4),
                    passenger_frac=_r(pas[1], 4),
                    mean_weight=_r(mean_w, 4),
                    mean_norm=_r(mean_norm, 4),
                ),
                corr=dict(
                    bias_vs_mean_logit=float(np.corrcoef(bias, mean_logit)[0, 1]),
                    bias_vs_freq=float(np.corrcoef(bias, selections)[0, 1]),
                    bias_vs_bias_frac=float(np.corrcoef(bias, bias_frac)[0, 1]),
                    bias_frac_vs_passenger=float(np.corrcoef(bias_frac, pas[1])[0, 1]),
                ),
                sublayer_norm=dict(
                    attn=float(s["attn_norm_sum"][i] / total),
                    shared=float(s["shared_norm_sum"][i] / total),
                    routed=float(s["routed_norm_sum"][i] / total),
                ),
            )
        )
    return out


def attrib_section(a) -> dict:
    cats = [str(c) for c in a["categories"]]
    n = a["n"]
    out = dict(categories=cats, terms=[str(t) for t in a["terms"]], stages=[str(s) for s in a["stages"]])
    out["n"] = n.tolist()
    out["loss"] = _r(a["loss"] / n)
    out["top1_acc"] = _r(a["top1_correct"] / n)
    out["entropy"] = _r(a["entropy"] / n)
    out["dla"] = _r(a["dla_y"] / n[:, None], 4)
    out["dla_std"] = _r(np.sqrt(np.maximum(a["dla_y_sq"] / n[:, None] - (a["dla_y"] / n[:, None]) ** 2, 0)), 4)
    out["dla_top"] = _r(a["dla_top"] / n[:, None], 4)
    out["dla_pos_frac"] = _r(a["dla_y_pos"] / n[:, None], 4)
    for k in ("lens_gate_loss", "lens_gate_agree", "lens_gate_kl", "lens_tuned_loss", "lens_tuned_agree"):
        out[k] = _r(a[k] / n[:, None], 4)
    out["gate_w"] = _r(a["gate_w"] / n[:, None, None], 4)
    out["gate_entropy"] = _r(a["gate_entropy"] / n[:, None], 4)
    out["eff_experts"] = _r(a["eff_experts"] / n[:, None], 4)
    out["top_weight"] = _r(a["top_weight"] / n[:, None], 4)
    out["term_norm"] = _r(a["term_norm"] / n[:, None], 4)
    an = a["ablation_n"]
    base = a["ablation_base_loss"] / an
    out["ablation_names"] = [str(x) for x in a["ablation_names"]]
    out["ablation_base"] = _r(base, 4)
    out["ablation_delta"] = _r(a["ablation_loss"] / an[None] - base[None], 4)
    out["recon_err"] = float(a["recon_err"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--compare", default=None, help="another tag's data dir (heads/attrib/routing summaries)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    with open(os.path.join(args.data, "vocab.json")) as f:
        vocab = json.load(f)
    doc: dict = {}
    with open(os.path.join(args.data, "verify.json")) as f:
        verify = json.load(f)
    doc["verify"] = dict(
        checkpoint=verify["checkpoint"],
        paloma_macro=verify["paloma_macro"],
        logged=verify["logged_paloma_macro"],
        short=verify["chunks"]["short"],
        arrays=verify["chunks"]["arrays"],
        paloma=verify["paloma_loss"],
    )
    for name, fn in (("heads", heads_section), ("attrib", attrib_section)):
        path = os.path.join(args.data, f"{name}.npz")
        if os.path.exists(path):
            doc[name] = fn(np.load(path))
    stats_path = os.path.join(args.data, "stats.npz")
    if os.path.exists(stats_path):
        s = np.load(stats_path)
        tok = np.load(os.path.join(args.data, "tokens.npz"))
        doc["neurons"] = neurons_section(s, tok, vocab, tok["tokens"].shape[1], rng)
        doc["routing"] = routing_section(s)
    if args.compare:
        other: dict = {}
        for name, fn in (("heads", heads_section), ("attrib", attrib_section)):
            path = os.path.join(args.compare, f"{name}.npz")
            if os.path.exists(path):
                other[name] = fn(np.load(path))
        stats_path = os.path.join(args.compare, "stats.npz")
        if os.path.exists(stats_path):
            other["routing"] = routing_section(np.load(stats_path))
        doc["compare"] = other
    with open(args.out, "w") as f:
        json.dump(doc, f)


if __name__ == "__main__":
    main()
