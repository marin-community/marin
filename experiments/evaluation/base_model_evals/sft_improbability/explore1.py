# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from common import *

T = load()
authors = sorted({t.author for t in T})
for a in authors:
    ts = [t for t in T if t.author == a]
    row = []
    for s in SCORERS[1:]:
        num = np.array([np.nansum(gap_arrays(t, s)) for t in ts])
        den = np.array([t.mask.sum() for t in ts])
        g, lo, hi = bootstrap(num, den)
        row.append(f"{SHORT[s]} {g:+.3f}[{lo:+.3f},{hi:+.3f}]")
    base = sum(np.nansum(t.nll[SCORERS[0]]) for t in ts) / sum(t.mask.sum() for t in ts)
    print(f"{a:14s} base={base:.3f} chars={sum(t.mask.sum() for t in ts):6d} leaked={sum(t.leaked for t in ts)} ", *row)
