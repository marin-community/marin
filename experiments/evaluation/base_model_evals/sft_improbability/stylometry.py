# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H6: Burrows' delta over the 150 most frequent words, plus punctuation/typography profile distance."""

import re
from collections import Counter

import numpy as np
from fulldata import load_full

F = load_full()
AUTHORS = sorted({f.author for f in F})
W = re.compile(r"[a-z']+")
cnt = {a: Counter() for a in AUTHORS}
for f in F:
    cnt[f.author].update(W.findall(f.text[: f.cut_fused].lower()))
total = Counter()
for a in AUTHORS:
    total.update(cnt[a])
top = [w for w, _ in total.most_common(150)]
M = np.array([[cnt[a][w] / sum(cnt[a].values()) for w in top] for a in AUTHORS])
Z = (M - M.mean(0)) / M.std(0)
D = np.abs(Z[:, None, :] - Z[None, :, :]).mean(-1)
for tgt in ["grugsft471", "snowballbase", "nemotron3nano", "qwen36"]:
    i = AUTHORS.index(tgt)
    order = np.argsort(D[i])
    print(f"Burrows delta from {tgt:13s}:", ", ".join(f"{AUTHORS[j]} {D[i, j]:.2f}" for j in order if j != i))
