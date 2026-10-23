# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Router vs hardness from the eval-only token dumps (frac-0 jobs: every token keeps; skipped_loss = every token
skips L3-L5).

With strict first-half KV, a token's skip changes only its own prediction, so per-token losses give EXACT skip
frontiers for any selection rule (oracle, router, random); for plain-skip50 the frontier is approximate (skipped
tokens' K/V would change too).

usage: python -m experiments.grug.fast_track.interp.skip_router_report <ev-name>...
(writes skip_router_report.json under ``common.OUTPUT_ROOT``/skipkv). CPU only, runs next to the data.
"""

import sys

import fsspec
import numpy as np

from experiments.grug.fast_track.interp import common

DUMP = "s3://marin-us-east-02a/marin/grug/lc1-skipkv-ev-{name}/2026.09.25/token_dump.npz"
RATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def load(name: str) -> dict:
    with fsspec.open(DUMP.format(name=name), "rb") as f:
        z = np.load(f)
        return {k: z[k] for k in z.files}


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.sum()
    neg = len(label) - pos
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def frontier(benefit: np.ndarray, rank_score: np.ndarray) -> list[float]:
    """Added loss per token (over all tokens) when the lowest-``rank_score`` fraction skips."""
    order = np.argsort(rank_score, kind="stable")
    csum = np.cumsum(benefit[order])
    n = len(benefit)
    return [float(csum[int(r * n) - 1] / n) for r in RATES]


def categories(tokens: np.ndarray, seg: np.ndarray) -> dict[str, np.ndarray]:
    """Per scored position (predicting token s+1 at s): copyable (bigram (x_s, x_s+1) seen earlier in the document),
    induction-right (the successor of the previous occurrence of x_s is x_s+1), document position, rare target."""
    b, s = tokens.shape
    nxt = np.pad(tokens[:, 1:], ((0, 0), (0, 1)), constant_values=-1)
    copy = np.zeros((b, s), bool)
    induct = np.zeros((b, s), bool)
    pos = np.zeros((b, s), np.int32)
    for i in range(b):
        seen_bigram, last_succ = set(), {}
        start = 0
        for t in range(s):
            if t > 0 and seg[i, t] != seg[i, t - 1]:
                seen_bigram, last_succ, start = set(), {}, t
            pos[i, t] = t - start
            cur, n = int(tokens[i, t]), int(nxt[i, t])
            copy[i, t] = (cur, n) in seen_bigram
            induct[i, t] = last_succ.get(cur, -2) == n
            if t + 1 < s and seg[i, t + 1] == seg[i, t]:
                seen_bigram.add((cur, n))
                last_succ[cur] = n
    counts = np.bincount(tokens.reshape(-1), minlength=int(tokens.max()) + 2)
    rare = counts[np.maximum(nxt, 0)] < np.quantile(counts[counts > 0], 0.5)
    return {
        "copyable": copy,
        "induction_right": induct,
        "pos_lt16": pos < 16,
        "pos_ge1024": pos >= 1024,
        "rare_target": rare,
    }


def analyse(name: str) -> dict:
    z = load(name)
    tokens, seg = z["tokens"], z["segment_ids"]
    valid = np.zeros(tokens.shape, bool)
    valid[:, :-1] = seg[:, 1:] == seg[:, :-1]
    full = z["next_token_loss"].astype(np.float32)[valid]
    skipped = z["skipped_loss"].astype(np.float32)[valid]
    benefit = skipped - full
    rng = np.random.default_rng(0)
    out = {
        "tokens": int(valid.sum()),
        "loss_full": float(full.mean()),
        "loss_all_skip": float(skipped.mean()),
        "mean_benefit": float(benefit.mean()),
        "need_frac": float((benefit > 0.1).mean()),
        "benefit_quantiles": [float(q) for q in np.quantile(benefit, [0.1, 0.25, 0.5, 0.75, 0.9])],
        "rates": list(RATES),
        "frontier_oracle": frontier(benefit, benefit),
        "frontier_random": frontier(benefit, rng.random(len(benefit))),
    }
    cats = {k: v[valid] for k, v in categories(tokens, seg).items()}
    out["category_benefit"] = {k: [float(m.mean()), float(benefit[m].mean())] for k, m in cats.items()}
    if "skip_router_score" in z:
        score = z["skip_router_score"].astype(np.float32)[valid]
        out["frontier_router"] = frontier(benefit, score)
        out["router_auroc_need"] = auroc(score, benefit > 0.1)
        out["router_spearman_benefit"] = spearman(score, benefit)
        keep50 = score > np.quantile(score, 0.5)
        out["router_keep50_by_category"] = {k: float(keep50[m].mean()) for k, m in cats.items()}
        out["router_keep50_need_precision"] = float((benefit[keep50] > 0.1).mean())
        out["oracle_keep50_overlap"] = float(np.mean(keep50 == (benefit > np.quantile(benefit, 0.5))))
    if "entropy_pred" in z:
        ent = z["entropy_pred"].astype(np.float32)[valid]
        out["frontier_entropy_pred"] = frontier(benefit, ent)
        out["entropy_pred_auroc_need"] = auroc(ent, benefit > 0.1)
    return out


def main() -> None:
    res = {}
    for name in sys.argv[1:]:
        res[name] = analyse(name)
        r = res[name]
        print(
            name, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items() if not isinstance(v, list | dict)}
        )
        for k in [k for k in r if k.startswith("frontier")]:
            print("  ", k, [round(v, 3) for v in r[k]])
    common.save_json("skipkv", "skip_router_report", res)


if __name__ == "__main__":
    main()
