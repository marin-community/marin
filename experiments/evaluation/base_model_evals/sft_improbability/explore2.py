# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import re

import numpy as np
from common import *

raw = {t["trace_id"]: t for t in map(json.loads, TRACES.read_text().splitlines())}
per = {}
for s in SCORERS:
    for line in gzip.open(TL / "data" / f"{s}.jsonl.gz", "rt"):
        r = json.loads(line)
        per[(s, r["trace_id"])] = char_nll(len(raw[r["trace_id"]]["response"]), r["offsets"], r["nll"])
P1 = re.compile(r"(^|\n)\s*(user|assistant|User|Assistant)\s*(\n|:)")


def cutA(x):
    return len(x)


def cutB(x):
    m = P1.search(x)
    return m.start() if m else len(x)


def cutC(x):
    return leak_cut(x)


for a in ["grugsft471", "snowballbase"]:
    for name, f in [("none", cutA), ("line-start role", cutB), ("fused user", cutC)]:
        res = {}
        for s in SCORERS[1:]:
            num = den = 0
            for tid, t in raw.items():
                if t["author"] != a:
                    continue
                x = t["response"]
                c = f(x)
                arr = np.array([per[(sc, tid)][:c] for sc in SCORERS])
                m = ~np.isnan(arr).any(0)
                num += (arr[SCORERS.index(s)][m] - arr[0][m]).sum()
                den += m.sum()
            res[SHORT[s]] = num / den
        print(a, name, {k: round(v, 3) for k, v in res.items()}, den)
# segment between fused cut and line-start cut, and the post-leak remainder
for a in ["grugsft471"]:
    for s in SCORERS[1:]:
        num = den = 0
        num2 = den2 = 0
        for tid, t in raw.items():
            if t["author"] != a:
                continue
            x = t["response"]
            c = cutC(x)
            b = cutB(x)
            if c == len(x):
                continue
            arr = np.array([per[(sc, tid)] for sc in SCORERS])
            m = ~np.isnan(arr).any(0)
            seg = np.zeros(len(x), bool)
            seg[c:b] = True
            seg &= m
            num += (arr[SCORERS.index(s)][seg] - arr[0][seg]).sum()
            den += seg.sum()
            rest = np.zeros(len(x), bool)
            rest[c:] = True
            rest &= m
            num2 += (arr[SCORERS.index(s)][rest] - arr[0][rest]).sum()
            den2 += rest.sum()
        print(
            "fake-user segment", SHORT[s], round(num / max(den, 1), 3), den, "all post-cut", round(num2 / den2, 3), den2
        )
