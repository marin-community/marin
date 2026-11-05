# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Word-salad stretches: windows where the generator's own mean NLL over 20 of its tokens exceeds
4 nats (p ~ 0.02 per token on average), which greedy decoding cannot produce. Share of chars,
responses affected, and share of each Qwen post gap carried there."""

import gzip
import json

import numpy as np
from common import TL
from fulldata import QS, load_full
from surprisal_match import SFT, tokens_in

F = load_full()
SELF = {SFT: "grug-67b-a2b-datakit-sft", "snowballbase": "snowball-67b-a2b-base-skew8"}
W, THR = 20, 4.0
for a, s in SELF.items():
    raw = {}
    for line in gzip.open(TL / "data" / f"{s}.jsonl.gz", "rt"):
        r = json.loads(line)
        if r["author"] == a:
            raw[r["trace_id"]] = (np.array(r["offsets"]).reshape(-1, 2), np.array(r["nll"]))
    for cut in ["cut_fused", "cut_line"]:
        P = []
        first = []
        for f in F:
            if f.author != a:
                continue
            c = getattr(f, cut)
            so, sn = raw[f.trace_id]
            keep = so[:, 1] <= c
            so, sn = so[keep], sn[keep]
            mark = np.zeros(len(f.text), bool)
            if len(sn) >= W:
                m = np.convolve(sn, np.ones(W) / W, "valid")
                for i in np.flatnonzero(m > THR):
                    mark[so[i, 0] : so[i + W - 1, 1]] = True
            first.append(np.flatnonzero(mark)[0] if mark.any() else -1)
            offs, tok = tokens_in(f, c)
            inside = np.array([mark[x:y].any() for x, y in offs]) if len(offs) else np.zeros(0, bool)
            L = offs[:, 1] - offs[:, 0]
            g5 = tok[QS[1]] - tok[QS[0]]
            g6 = tok[QS[2]] - tok[QS[0]]
            P.append([L.sum(), g5.sum(), g6.sum(), L[inside].sum(), g5[inside].sum(), g6[inside].sum()])
        P = np.array(P)
        rng = np.random.default_rng(0)

        def ci(fn):
            v = [fn(P[rng.integers(0, len(P), len(P))]) for _ in range(2000)]
            return f"{fn(P):+.4f} [{np.percentile(v, 2.5):+.4f},{np.percentile(v, 97.5):+.4f}]"

        aff = np.array(first) >= 0
        print(
            f"{a:12s} {cut:9s} salad: {aff.sum()} responses, {P[:, 3].sum() / P[:, 0].sum():.3f} of chars (median onset char {np.median(np.array(first)[aff]):.0f}); "
            f"gap share q35 {P[:, 4].sum() / P[:, 1].sum():.2f} q36 {P[:, 5].sum() / P[:, 2].sum():.2f}; gap/char inside q35 {P[:, 4].sum() / P[:, 3].sum():+.3f} q36 {P[:, 5].sum() / P[:, 3].sum():+.3f}; "
            f"gap/char with salad removed q35 {ci(lambda Q: (Q[:, 1].sum() - Q[:, 4].sum()) / (Q[:, 0].sum() - Q[:, 3].sum()))} q36 {ci(lambda Q: (Q[:, 2].sum() - Q[:, 5].sum()) / (Q[:, 0].sum() - Q[:, 3].sum()))}"
        )
