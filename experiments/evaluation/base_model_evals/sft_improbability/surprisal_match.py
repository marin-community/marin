# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Surprisal-matched decomposition of the post-trained-Qwen gap.

For each Qwen token, gap = post NLL - Qwen3.5-base NLL. The reference curve is the mean gap
of reference-author tokens in bins of Qwen3.5-base NLL. An author's predicted gap applies
that curve to its own tokens: what its gap would be if each token were penalized like a
reference token of equal base surprisal. Everything is per character covered by tokens.

  author gap - ref gap = (pred - ref)   composition: the author has more surprising tokens
                       + (obs - pred)   residual: its tokens are penalized beyond their surprisal

Prompts are resampled (4,000 bootstrap replicates) jointly for curve and author.
"""

import numpy as np
from fulldata import QS, load_full

REF = ["gemma4", "glm47flash", "kimilinear", "kolibri", "linglite", "nemotron3nano", "trinitymini"]
EDGES = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 13, 1e9])
NB = len(EDGES) - 1
POS = [0, 20, 100, 300, 1000, 2000, 4001]
SFT = "grugsft471"


def tokens_in(f, cut: int):
    keep = f.offsets[:, 1] <= cut
    return f.offsets[keep], {s: v[keep] for s, v in f.tok.items()}


def per_prompt_tables(F, scorer: str, cut_attr: str, base: str = QS[0], cond=None):
    """Per author: arrays over prompts of bin counts (chars-weighted not needed: token counts),
    obs gap sums, chars. cond(f, offs, tok) optionally returns the conditioning NLL per token."""
    prompts = sorted({f.prompt_key for f in F})
    pidx = {p: i for i, p in enumerate(prompts)}
    authors = sorted({f.author for f in F})
    T = {
        a: dict(
            cnt=np.zeros((len(prompts), NB)),
            gsum=np.zeros((len(prompts), NB)),
            obs=np.zeros(len(prompts)),
            chars=np.zeros(len(prompts)),
        )
        for a in authors
    }
    for f in F:
        cut = getattr(f, cut_attr) if cut_attr else len(f.text)
        offs, tok = tokens_in(f, cut)
        g = tok[scorer] - tok[base]
        c = tok[base] if cond is None else cond(f, offs, tok)
        b = np.clip(np.digitize(c, EDGES) - 1, 0, NB - 1)
        d = T[f.author]
        i = pidx[f.prompt_key]
        d["cnt"][i] += np.bincount(b, minlength=NB)
        d["gsum"][i] += np.bincount(b, weights=g, minlength=NB)
        d["obs"][i] += g.sum()
        d["chars"][i] += (offs[:, 1] - offs[:, 0]).sum()
    return T


def curve_from(T, idx):
    cnt = sum(T[a]["cnt"][idx].sum(0) for a in REF)
    gs = sum(T[a]["gsum"][idx].sum(0) for a in REF)
    return np.where(cnt > 0, gs / np.maximum(cnt, 1), 0.0)


def ref_gap(T, idx):
    return sum(T[a]["obs"][idx].sum() for a in REF) / sum(T[a]["chars"][idx].sum() for a in REF)


def decompose(T, author: str, n: int = 4000, seed: int = 0):
    """Returns dict of point estimate and 95% CI for obs, pred, ref, composition, residual."""
    k = len(T[author]["obs"])
    rng = np.random.default_rng(seed)
    reps = [np.arange(k)] + [rng.integers(0, k, k) for _ in range(n)]
    rows = []
    for idx in reps:
        c = curve_from(T, idx)
        d = T[author]
        ch = d["chars"][idx].sum()
        obs = d["obs"][idx].sum() / ch
        pred = (d["cnt"][idx].sum(0) * c).sum() / ch
        ref = ref_gap(T, idx)
        rows.append((obs, pred, ref, pred - ref, obs - pred, obs - ref))
    rows = np.array(rows)
    names = ["obs", "pred", "ref", "composition", "residual", "excess"]
    return {nm: (rows[0, j], *np.percentile(rows[1:, j], [2.5, 97.5])) for j, nm in enumerate(names)}


def fmt(x):
    return f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"


if __name__ == "__main__":
    F = load_full()
    for cut in ["cut_line", "cut_fused", None]:
        for s in QS[1:]:
            T = per_prompt_tables(F, s, cut)
            print(f"\n== cut={cut} scorer={s}")
            for a in sorted(T):
                r = decompose(T, a, n=1000 if a != SFT else 4000)
                base_tok = None
                print(
                    f"{a:14s} obs {fmt(r['obs'])} pred {fmt(r['pred'])} resid {fmt(r['residual'])}"
                    f"  excess {fmt(r['excess'])} comp {fmt(r['composition'])}"
                )
