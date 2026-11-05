# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import unicodedata

import numpy as np
from common import *

T = load()
REF = ["gemma4", "glm47flash", "kimilinear", "kolibri", "linglite", "nemotron3nano", "trinitymini"]


def tot(ts, s):
    return sum(np.nansum(gap_arrays(t, s)) for t in ts) / sum(t.mask.sum() for t in ts)


for a in ["grugsft471", "snowballbase"] + REF:
    ts = [t for t in T if t.author == a]
    nonascii = sum(sum(ord(c) > 127 for c in t.text) for t in ts) / sum(len(t.text) for t in ts)
    cjk = sum(
        sum(
            (
                "CJK" in unicodedata.name(c, "")
                or "HIRAGANA" in unicodedata.name(c, "")
                or "KATAKANA" in unicodedata.name(c, "")
                or "HANGUL" in unicodedata.name(c, "")
            )
            for c in t.text
        )
        for t in ts
    ) / sum(len(t.text) for t in ts)
    L = np.array([len(t.text) for t in ts])
    for s in ["qwen3.5-35b-a3b", "qwen3.6-35b-a3b"]:
        per = np.array([np.nanmean(gap_arrays(t, s)) if t.mask.sum() else np.nan for t in ts])
        nums = np.array([np.nansum(gap_arrays(t, s)) for t in ts])
        dens = np.array([t.mask.sum() for t in ts])
        order = np.argsort(-nums)  # largest total-nats contributors
        drop = [(k, nums[order[k:]].sum() / dens[order[k:]].sum()) for k in (0, 1, 3, 5, 10)]
        print(
            f"{a:13s} {SHORT[s]} nonascii={nonascii:.3f} cjk={cjk:.4f} medlen={np.median(L):.0f} median_resp_gap={np.nanmedian(per):+.3f} frac>0={np.nanmean(per>0):.2f} drop-topk:",
            " ".join(f"{k}:{v:+.3f}" for k, v in drop),
        )
