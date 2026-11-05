# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Markdown tables for REPORT.md: SFT gap contributions by segment x position x condensed class."""

import numpy as np
from cells import boot_contrib, cell_sums, pos_bin
from fulldata import QS, load_full
from surprisal_match import POS, SFT

GROUP = {
    "word interior": "words",
    "word start": "words",
    "space": "spaces",
    "punctuation": "punctuation",
    "digit": "digits",
    "newline (blank-line run)": "newlines",
    "newline (single)": "newlines",
    "list marker": "markdown",
    "header #": "markdown",
    "bold/italic *": "markdown",
    "other markdown": "markdown",
    "dash (em/en)": "dashes/quotes",
    "curly quote": "dashes/quotes",
    "CJK": "non-ASCII",
    "non-ASCII letter": "non-ASCII",
}
F = load_full()
for scorer in QS[1:]:
    sft = cell_sums(F, scorer, {SFT}, "cut_line")
    tot = sum(v[:, 0] for v in sft.values())
    T = tot.sum()
    gap = sum(v[:, 1].sum() for v in sft.values()) / T
    print(f"\n### {scorer}: SFT line-cut gap {gap:+.4f}; cells are contribution x1e-4 (obs / residual)\n")
    poss = [pos_bin(p) for p in POS[:-1]]
    groups = ["words", "spaces", "punctuation", "newlines", "markdown", "dashes/quotes", "non-ASCII", "digits"]
    print("| class | " + " | ".join(poss) + " | total [95% CI] | share of gap |")
    print("|---|" + "---:|" * (len(poss) + 2))

    def agg(sel):
        out = np.zeros((100, 4))
        for k, v in sft.items():
            if sel(k):
                out += v
        return out

    for g in groups:
        cells = []
        for p in poss:
            v = agg(lambda k: k[0] == "answer" and k[1] == p and GROUP[k[2]] == g)
            cells.append(f"{1e4 * v[:, 1].sum() / T:+.0f} / {1e4 * (v[:, 1].sum() - v[:, 2].sum()) / T:+.0f}")
        v = agg(lambda k: k[0] == "answer" and GROUP[k[2]] == g)
        lo, hi = boot_contrib(v[:, 1], tot)
        print(
            f"| {g} | "
            + " | ".join(cells)
            + f" | {v[:, 1].sum() / T:+.4f} [{lo:+.4f}, {hi:+.4f}] | {v[:, 1].sum() / T / gap:.0%} |"
        )
    for name, sel in [("answer, all", lambda k: k[0] == "answer"), ("fused-leak tail", lambda k: k[0] != "answer")]:
        v = agg(sel)
        lo, hi = boot_contrib(v[:, 1], tot)
        print(
            f"| **{name}** | "
            + " | ".join(f"{1e4 * agg(lambda k, p=p: sel(k) and k[1] == p)[:, 1].sum() / T:+.0f}" for p in poss)
            + f" | {v[:, 1].sum() / T:+.4f} [{lo:+.4f}, {hi:+.4f}] | {v[:, 1].sum() / T / gap:.0%} |"
        )
    print(
        "| char share | " + " | ".join(f"{agg(lambda k, p=p: k[1] == p)[:, 0].sum() / T:.3f}" for p in poss) + " | | |"
    )
