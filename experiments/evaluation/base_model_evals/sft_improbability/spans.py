# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Highest-gap spans in SFT responses (line-start cut), per scorer, with base NLL."""

import sys

import numpy as np
from fulldata import QS, load_full
from surprisal_match import SFT, tokens_in

F = load_full()
scorer = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # tokens per window
spans = []
for f in F:
    if f.author != SFT:
        continue
    offs, tok = tokens_in(f, f.cut_line)
    g = tok[scorer] - tok[QS[0]]
    if len(g) < W:
        continue
    cs = np.convolve(g, np.ones(W), "valid")
    used = np.zeros(len(cs), bool)
    for i in np.argsort(-cs)[:3]:
        if used[max(0, i - W) : i + W].any():
            continue
        used[i] = True
        s, e = offs[i][0], offs[i + W - 1][1]
        spans.append(
            (
                cs[i],
                tok[QS[0]][i : i + W].sum(),
                f.trace_id,
                s,
                f.text[max(0, s - 40) : s].replace("\n", "⏎"),
                f.text[s:e].replace("\n", "⏎"),
                "LEAKTAIL" if s >= f.cut_fused else "",
            )
        )
spans.sort(key=lambda x: -x[0])
for sp in spans[:40]:
    print(f"{sp[0]:+6.2f} base {sp[1]:5.1f} {sp[2]:>14s} @{sp[3]:<5d}{sp[6]:9s}| …{sp[4]}【{sp[5]}】")
