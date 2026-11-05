# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokens the generator itself found improbable. The SFT and Snowball base scored their own
IFBench outputs; under greedy decoding in the same context, an emitted token is the generator's
argmax and can essentially never have p < 0.01. This script measures how often that happens and
how much of the post-trained-Qwen gap sits on Qwen tokens overlapping such generator-tail tokens."""

import gzip
import json

import numpy as np
from common import TL
from fulldata import QS, load_full
from surprisal_match import SFT, tokens_in

F = load_full()
SELF = {SFT: "grug-67b-a2b-datakit-sft", "snowballbase": "snowball-67b-a2b-base-skew8"}
raw = {}
for a, s in SELF.items():
    for line in gzip.open(TL / "data" / f"{s}.jsonl.gz", "rt"):
        r = json.loads(line)
        if r["author"] == a:
            raw[r["trace_id"]] = (np.array(r["offsets"]).reshape(-1, 2), np.array(r["nll"]))

rng = np.random.default_rng(0)
for a in SELF:
    for cut in ["cut_fused", "cut_line"]:
        for thr_name, thr in [("p<0.01", np.log(100)), ("p<0.001", np.log(1000))]:
            per = []  # per prompt: chars, g35, g36, tail chars, tail g35, tail g36, after g35, after g36
            for f in F:
                if f.author != a:
                    continue
                c = getattr(f, cut)
                so, sn = raw[f.trace_id]
                selfmax = np.zeros(len(f.text))
                for (s, e), x in zip(so, sn):
                    if e > s:
                        selfmax[s:e] = np.maximum(selfmax[s:e], x)
                offs, tok = tokens_in(f, c)
                tail = np.array([selfmax[s:e].max() > thr for s, e in offs]) if len(offs) else np.zeros(0, bool)
                after = np.zeros_like(tail)
                for k in np.flatnonzero(tail):
                    after[k + 1 : k + 4] = True
                after &= ~tail
                L = offs[:, 1] - offs[:, 0]
                g5 = tok[QS[1]] - tok[QS[0]]
                g6 = tok[QS[2]] - tok[QS[0]]
                per.append(
                    [
                        L.sum(),
                        g5.sum(),
                        g6.sum(),
                        L[tail].sum(),
                        g5[tail].sum(),
                        g6[tail].sum(),
                        g5[after].sum(),
                        g6[after].sum(),
                        tail.sum(),
                        len(tail),
                    ]
                )
            P = np.array(per)

            def ci(fn):
                v = [fn(P[rng.integers(0, len(P), len(P))]) for _ in range(2000)]
                return fn(P), *np.percentile(v, [2.5, 97.5])

            s35 = ci(lambda Q: Q[:, 4].sum() / Q[:, 1].sum())
            s36 = ci(lambda Q: Q[:, 5].sum() / Q[:, 2].sum())
            a35 = ci(lambda Q: Q[:, 6].sum() / Q[:, 1].sum())
            a36 = ci(lambda Q: Q[:, 7].sum() / Q[:, 2].sum())
            rest35 = (P[:, 1].sum() - P[:, 4].sum()) / (P[:, 0].sum() - P[:, 3].sum())
            rest36 = (P[:, 2].sum() - P[:, 5].sum()) / (P[:, 0].sum() - P[:, 3].sum())
            print(
                f"{a:12s} {cut:9s} generator {thr_name:8s}: {P[:, 8].sum() / P[:, 9].sum():.3f} of Qwen tokens, {P[:, 3].sum() / P[:, 0].sum():.3f} of chars | "
                f"share of gap on them q35 {s35[0]:.2f} [{s35[1]:.2f},{s35[2]:.2f}] q36 {s36[0]:.2f} [{s36[1]:.2f},{s36[2]:.2f}] | "
                f"next-3-token share q35 {a35[0]:.2f} q36 {a36[0]:.2f} | gap/char on non-tail chars q35 {rest35:+.4f} q36 {rest36:+.4f}"
            )
