# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report data for the routing-problems section of the expert atlas, from local copies of ``regret.npz``,
``capacity.npz``, ``ngram_units.npz``, ``vocab.json`` and the atlas outputs (``unit_metrics.npz``,
``expert_labels.json``).

Signs: a swap's ``delta`` is loss(edited) - loss(unedited), so its end-to-end gain is ``-delta``; the linear
``swap_gain`` / ``regret`` are already gains. ``bias_lin`` is loss(unbiased top-10) - loss(actual).

Usage: ``python -m experiments.grug.fast_track.interp.regret_report <data dir> <atlas dir> <out.json>``.
"""

import json
import sys

import numpy as np

from experiments.grug.fast_track.interp import atlas, atlas_report, regret
from experiments.grug.fast_track.interp.atlas_report import read_json, write_json

L, E = regret.L, regret.E
LARGE = regret.LARGE_REGRET
QS = (50, 90, 99, 99.9)
NUM_EXAMPLES = 24
TOP_TOKENS = 8
CONTEXT = 10


def _q(x: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in np.percentile(x, QS)] if len(x) else []


def _fit(lin: np.ndarray, e2e: np.ndarray) -> dict:
    if len(lin) < 3:
        return dict(n=len(lin))
    slope = float(np.sum(lin * e2e) / max(np.sum(lin * lin), 1e-12))
    return dict(
        n=len(lin),
        corr=round(float(np.corrcoef(lin, e2e)[0, 1]), 3),
        slope=round(slope, 3),
        lin_mean=round(float(lin.mean()), 4),
        e2e_mean=round(float(e2e.mean()), 4),
        e2e_large=round(float(np.mean(e2e > LARGE)), 3),
        e2e_positive=round(float(np.mean(e2e > 0)), 3),
    )


def context_text(tokens: np.ndarray, segs: np.ndarray, t: int, vocab: list[str]) -> dict:
    """The ``CONTEXT`` tokens up to and including flat position ``t`` (same document) and the next one."""
    s = tokens.shape[1]
    row, pos = divmod(int(t), s)
    lo = pos
    while lo > 0 and pos - lo < CONTEXT - 1 and segs[row, lo - 1] == segs[row, pos]:
        lo -= 1
    nxt = vocab[int(tokens[row, pos + 1])] if pos + 1 < s and segs[row, pos + 1] == segs[row, pos] else ""
    return dict(pre="".join(vocab[int(x)] for x in tokens[row, lo:pos]), cur=vocab[int(tokens[row, pos])], next=nxt)


def regret_section(z: dict, labels: list, vocab: list[str]) -> dict:
    valid = z["sample/valid"]
    sources = np.repeat(z["sample/sources"], z["sample/tokens"].shape[1])
    groups = atlas_report.GROUP_OF_SOURCE[sources]
    tokens, segs = z["sample/tokens"], z["sample/segment_ids"]
    layers = []
    for layer in range(L):
        reg = z["rec/regret"][layer][valid]
        swap = z["rec/swap_gain"][layer][valid]
        fix = z["rec/regret_fix"][layer][valid]
        in_u = z["rec/swap_in_u"][layer][valid].astype(bool)
        large = swap > LARGE
        bias = z["rec/bias_lin"][layer][valid]
        disp = z["rec/n_bias_only"][layer][valid] > 0
        by_group = []
        for g in range(len(atlas.GROUPS)):
            m = groups[valid] == g
            if m.sum() >= 1000:
                by_group.append([atlas.GROUPS[g], int(m.sum()), round(float(np.mean(large[m])), 4)])
        layers.append(
            dict(
                regret_q=_q(reg),
                regret_large=round(float(np.mean(reg > LARGE)), 4),
                regret_fix_q=_q(fix),
                regret_fix_large=round(float(np.mean(fix > LARGE)), 4),
                swap_q=_q(swap),
                swap_large=round(float(np.mean(large)), 4),
                swap_mean_pos=round(float(np.mean(np.maximum(swap, 0))), 4),
                large_bias_share=round(float(np.mean(in_u[large])), 3) if large.any() else None,
                large_best_strong=(
                    round(float(np.mean(z["rec/ns_swap"][layer][valid][large] >= 1)), 3) if large.any() else None
                ),
                displaced_tokens=round(float(np.mean(disp)), 4),
                bias_lin_mean=round(float(bias.mean()), 5),
                bias_lin_q=_q(-bias[disp]),
                moe_gain_mean=round(float(z["rec/moe_gain"][layer][valid].mean()), 4),
                by_group=by_group,
            )
        )
    kinds = z["swap/kind"]
    lay = z["swap/layer"]
    ref = z["swap/ref"]
    gain = -z["swap/delta"][:, 2]
    checks = {}
    for kind, name, field in (
        (regret.Kind.REGRET, "regret", "rec/swap_gain"),
        (regret.Kind.BIAS, "bias", "rec/bias_lin"),
        (regret.Kind.TOPIC, "topic", "rec/topic_gain"),
    ):
        rows = []
        for layer in range(L):
            m = (kinds == kind) & (lay == layer)
            lin = z[field][layer][ref[m]]
            if kind == regret.Kind.BIAS:
                lin = -lin  # gain of restoring the unbiased selection
            rows.append(_fit(lin.astype(np.float64), gain[m].astype(np.float64)))
        checks[name] = rows
    noop = kinds == regret.Kind.NOOP
    checks["noise"] = dict(
        noop_abs_mean=round(float(np.abs(z["swap/delta"][noop, 2]).mean()), 5) if noop.any() else None,
        pre_abs_max=round(float(np.abs(z["swap/delta"][:, 3]).max()), 5),
    )
    # The bias-restore study: per layer, tokens whose selection the bias changed, mean e2e gain of restoring.
    examples = []
    m = kinds == regret.Kind.REGRET
    order = np.flatnonzero(m)[np.argsort(-gain[m])][:NUM_EXAMPLES]
    for j in order:
        layer, t = int(lay[j]), int(ref[j])
        best, worst = int(z["rec/swap_e"][layer][t]), int(z["rec/worst_e"][layer][t])
        examples.append(
            dict(
                layer=layer,
                ctx=context_text(tokens, segs, t, vocab),
                source=atlas.SOURCES[int(z["sample/sources"][t // tokens.shape[1]])],
                missed=best,
                missed_label=labels[layer][best],
                dropped=worst,
                dropped_label=labels[layer][worst],
                lin=round(float(z["rec/swap_gain"][layer][t]), 3),
                e2e=round(float(gain[j]), 3),
                e2e_here=round(float(-z["swap/delta"][j, 0]), 3),
                in_unbiased=bool(z["rec/swap_in_u"][layer][t]),
                strong_units=int(z["rec/ns_swap"][layer][t]),
                loss=round(float(z["sample/loss"][t]), 3),
            )
        )
    bias_loss = z["bias/loss"]
    mean = bias_loss[:, 0] / bias_loss[:, 1]
    return dict(
        tokens=int(valid.sum()),
        w_fix=[round(float(x), 4) for x in z["w_fix"]],
        layers=layers,
        checks=checks,
        examples=examples,
        bias_global=dict(
            base=round(float(mean[0]), 5),
            none=round(float(mean[1]), 5),
            per_layer=[round(float(x), 5) for x in mean[2:]],
        ),
    )


def expert_section(z: dict, vocab: list[str]) -> list:
    out = []
    for layer in range(L):
        a = {k: z[f"acc/{k}"][layer] for k in ("routed", "routed_resp", "unrouted_resp", "sel_negative")}
        a.update({k: z[f"acc/{k}"][layer] for k in ("best_missed", "inserted", "displaced", "latent_useful")})
        resp_tok = z["acc/resp_unrouted_tok"][layer]
        waste_tok = z["acc/wasted_tok"][layer]
        rows = []
        for e in range(E):
            routed = max(float(a["routed"][e]), 1.0)
            resp = float(a["routed_resp"][e] + a["unrouted_resp"][e])
            lt = np.argsort(-resp_tok[e])[:TOP_TOKENS]
            wt = np.argsort(-waste_tok[e])[:TOP_TOKENS]
            rows.append(
                dict(
                    bm=int(a["best_missed"][e]),
                    lu=float(a["latent_useful"][e]) / max(z["sample/valid"].sum(), 1),
                    lc=float(a["unrouted_resp"][e]) / max(resp, 1.0),
                    ws=1.0 - float(a["routed_resp"][e]) / routed,
                    ng=float(a["sel_negative"][e]) / routed,
                    ins=float(a["inserted"][e]) / routed,
                    dis=int(a["displaced"][e]),
                    lt=[[int(t), int(resp_tok[e, t])] for t in lt if resp_tok[e, t] > 0],
                    wt=[[int(t), int(waste_tok[e, t])] for t in wt if waste_tok[e, t] > 0],
                )
            )
        out.append(rows)
    return out


def mismatch_section(experts: list, labels: list) -> dict:
    """Per layer: the experts with the most unused latent capacity and the most wasted slots."""
    rows = []
    for layer in range(L):
        ex = experts[layer]
        lc = np.array([r["lc"] for r in ex])
        ws = np.array([r["ws"] for r in ex])
        lu = np.array([r["lu"] for r in ex])
        top_lu = np.argsort(-lu)[:6]
        top_ws = np.argsort(-ws)[:6]
        rows.append(
            dict(
                lc_median=round(float(np.median(lc)), 3),
                ws_median=round(float(np.median(ws)), 3),
                ws_q90=round(float(np.percentile(ws, 90)), 3),
                lu_total=round(float(lu.sum()), 4),
                latent=[[int(e), labels[layer][e], round(float(lu[e]), 5), round(float(lc[e]), 3)] for e in top_lu],
                wasted=[[int(e), labels[layer][e], round(float(ws[e]), 3)] for e in top_ws],
            )
        )
    return dict(layers=rows)


def topic_section(z: dict, um: dict) -> dict:
    """Topic units: how often they are strong on in-topic tokens their expert was not routed, and the linear and
    end-to-end cost of those skips."""
    valid = z["sample/valid"]
    topic = um["category"] == atlas_report.CATEGORIES.index("topic")
    rows = []
    for layer in range(L):
        routed = z["acc/unit_topic_routed"][layer][topic[layer]]
        skipped = z["acc/unit_topic_skipped"][layer][topic[layer]]
        tot = routed + skipped
        ok = tot >= 5
        flagged = z["rec/topic_e"][layer][valid] >= 0
        tg = z["rec/topic_gain"][layer][valid]
        rows.append(
            dict(
                units=int(topic[layer].sum()),
                skip_rate_median=round(float(np.median(skipped[ok] / tot[ok])), 3) if ok.any() else None,
                skip_share=round(float(skipped.sum() / max(tot.sum(), 1)), 3),
                flagged_tokens=round(float(flagged.mean()), 4),
                linear_cost=round(float(tg.mean()), 5),
                large=round(float(np.mean(tg > LARGE)), 4),
            )
        )
    return dict(layers=rows)


def ngram_section(z: dict, units: dict, vocab: list[str], labels: list) -> dict:
    u = z["ngram/unit"]
    hit_c = z["ngram/next"] == units["c"][u]
    sel = z["ngram/selected"].astype(bool)
    layer = units["layer"][u]
    rows = []
    for lyr in range(L):
        m = hit_c & (layer == lyr)
        rows.append(dict(n=int(m.sum()), selected=round(float(sel[m].mean()), 4) if m.any() else None))
    kinds = z["ngram_swap/kind"]
    ref = z["ngram_swap/ref"]
    gain = -z["ngram_swap/delta"][:, 2]
    here = -z["ngram_swap/delta"][:, 0]
    add = kinds == regret.Kind.NGRAM_ADD
    rem = kinds == regret.Kind.NGRAM_REMOVE
    clear = units["clear"][u]
    skipped_any = hit_c & ~sel
    per_unit_rate = np.bincount(u[skipped_any], minlength=len(units["layer"])) / np.maximum(
        np.bincount(u[hit_c], minlength=len(units["layer"])), 1
    )
    examples = []
    for j in np.flatnonzero(add)[np.argsort(-gain[add])][:12]:
        k = int(ref[j])
        unit = int(u[k])
        examples.append(
            dict(
                unit=f"L{units['layer'][unit]} E{units['expert'][unit]} n{units['unit'][unit]}",
                label=labels[int(units["layer"][unit])][int(units["expert"][unit])],
                a=vocab[int(units["a"][unit])],
                b=vocab[int(units["b"][unit])],
                c=vocab[int(units["c"][unit])],
                act=round(float(z["ngram/act"][k]), 2),
                rank=int(z["ngram/biased_rank"][k]),
                urank=int(z["ngram/unbiased_rank"][k]),
                e2e=round(float(gain[j]), 3),
                here=round(float(here[j]), 3),
            )
        )
    return dict(
        occurrences=int(hit_c.sum()),
        selected=round(float(sel[hit_c].mean()), 4),
        selected_clear=round(float(sel[hit_c & (clear == 1)].mean()), 4) if (hit_c & (clear == 1)).any() else None,
        units_never_skipped=round(
            float(np.mean(per_unit_rate[np.bincount(u[hit_c], minlength=len(per_unit_rate)) > 0] == 0)), 3
        ),
        skipped_rank_q=_q(z["ngram/biased_rank"][skipped_any].astype(np.float64)),
        skipped_act_ratio=(
            round(float(np.median(z["ngram/act"][skipped_any]) / max(np.median(z["ngram/act"][hit_c & sel]), 1e-9)), 3)
            if skipped_any.any()
            else None
        ),
        add=(
            _fit(np.zeros(add.sum()), gain[add]) | dict(here_mean=round(float(here[add].mean()), 4)) if add.any() else {}
        ),
        remove=(
            _fit(np.zeros(rem.sum()), gain[rem]) | dict(here_mean=round(float(here[rem].mean()), 4)) if rem.any() else {}
        ),
        layers=rows,
        examples=examples,
    )


def capacity_section(c: dict, vocab: list[str], labels: list) -> dict:
    assign = c["assignments"]  # (batch, layer, total, dropped, tokens with a drop)
    rows = []
    for layer in range(L):
        m = assign[:, 1] == layer
        tot, drop, tok = assign[m, 2].sum(), assign[m, 3].sum(), assign[m, 4].sum()
        rows.append(
            dict(
                drop_rate=float(drop / tot),
                token_rate=float(tok / (tot / regret.K)),
                batches_with_drops=round(float(np.mean(assign[m, 3] > 0)), 3),
                max_load=round(float(c["receiver_load"][m].max()), 4),
            )
        )
    d = {k[5:]: v for k, v in c.items() if k.startswith("drop/")}
    top_tok = []
    if len(d["token"]):
        ids, n = np.unique(d["token"], return_counts=True)
        for t in ids[np.argsort(-n)][:12]:
            top_tok.append([vocab[int(t)], int(n[ids == t][0])])
        ek = d["layer"].astype(np.int64) * E + d["expert"]
        eids, en = np.unique(ek, return_counts=True)
        top_exp = [
            [labels[int(k // E)][int(k % E)], int(v)]
            for k, v in zip(eids[np.argsort(-en)][:8], np.sort(en)[::-1][:8], strict=True)
        ]
        rank_hist = np.bincount(d["rank"], minlength=regret.K).tolist()
    else:
        top_exp, rank_hist = [], []
    loss = c["loss"]
    return dict(
        batches=int(loss.shape[0]),
        layers=rows,
        loss_dropless=round(float(loss[:, 0].mean()), 5),
        loss_drops=round(float(loss[:, 1].mean()), 5),
        loss_delta_q=_q(loss[:, 1] - loss[:, 0]),
        top_tokens=top_tok,
        top_experts=top_exp,
        rank_hist=rank_hist,
        weight_mean=round(float(d["weight"].mean()), 4) if len(d["weight"]) else None,
    )


def main() -> None:
    data, atlas_dir, out_path = sys.argv[1:4]
    z = dict(np.load(f"{data}/regret.npz"))
    vocab = read_json(f"{data}/vocab.json")
    labels = read_json(f"{atlas_dir}/expert_labels.json")
    um = dict(np.load(f"{atlas_dir}/unit_metrics.npz"))
    units = dict(np.load(f"{data}/ngram_units.npz"))
    experts = expert_section(z, vocab)
    report = dict(
        regret=regret_section(z, labels, vocab),
        experts=experts,
        mismatch=mismatch_section(experts, labels),
        topic=topic_section(z, um),
        ngram=ngram_section(z, units, vocab, labels),
    )
    try:
        report["capacity"] = capacity_section(dict(np.load(f"{data}/capacity.npz")), vocab, labels)
    except FileNotFoundError:
        report["capacity"] = None
    write_json(report, out_path, separators=(",", ":"), default=float)
    print(json.dumps({k: v for k, v in report.items() if k != "experts"}, default=float)[:6000])


if __name__ == "__main__":
    main()
