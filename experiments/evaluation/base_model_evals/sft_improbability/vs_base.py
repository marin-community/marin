# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT vs Snowball base: same surprisal profile, different penalty. Paired-by-prompt bootstrap of
the gap difference and its composition/residual parts, and where the residual difference sits."""

import numpy as np
from cells import ext_class, list_mask
from fulldata import QS, load_full
from surprisal_match import EDGES, NB, POS, REF, SFT, curve_from, per_prompt_tables, tokens_in

F = load_full()
BASE = "snowballbase"
rng = np.random.default_rng(1)
for s in QS[1:]:
    for cut in ["cut_fused", "cut_line"]:
        T = per_prompt_tables(F, s, cut)
        reps = [np.arange(100)] + [rng.integers(0, 100, 100) for _ in range(4000)]
        out = []
        for idx in reps:
            c = curve_from(T, idx)
            v = []
            for a in (SFT, BASE):
                d = T[a]
                ch = d["chars"][idx].sum()
                obs = d["obs"][idx].sum() / ch
                pred = (d["cnt"][idx].sum(0) * c).sum() / ch
                v += [obs, pred, obs - pred]
            out.append([v[0] - v[3], v[1] - v[4], v[2] - v[5]])
        out = np.array(out)
        print(
            f"{s:18s} {cut:9s} SFT-base: gap {out[0,0]:+.4f} [{np.percentile(out[1:,0],2.5):+.4f},{np.percentile(out[1:,0],97.5):+.4f}]"
            f"  composition {out[0,1]:+.4f} [{np.percentile(out[1:,1],2.5):+.4f},{np.percentile(out[1:,1],97.5):+.4f}]"
            f"  residual {out[0,2]:+.4f} [{np.percentile(out[1:,2],2.5):+.4f},{np.percentile(out[1:,2],97.5):+.4f}]"
        )

# residual by position and class, fused cut, for SFT, BASE, REF
for s in QS[1:]:
    T = per_prompt_tables(F, s, "cut_fused")
    curve = curve_from(T, np.arange(100))
    print(f"\n{s}: residual (obs - pred) per char of the cell, and cell share; fused cut")
    res = {}
    for grp, name in (([SFT], "SFT"), ([BASE], "base"), (REF, "REF")):
        acc = {}
        tot = 0
        for f in F:
            if f.author not in grp:
                continue
            offs, tok = tokens_in(f, f.cut_fused)
            b = np.clip(np.digitize(tok[QS[0]], EDGES) - 1, 0, NB - 1)
            lm = list_mask(f.text)
            for (st, e), g, p in zip(offs, tok[s] - tok[QS[0]], curve[b]):
                pb = next(f"{lo}-{hi}" for lo, hi in zip(POS, POS[1:]) if lo <= st < hi)
                cl = ext_class(f.text, st if not f.text[st].isspace() or e - st == 1 else st + 1, lm)
                for key in (("pos", pb), ("cls", cl)):
                    a = acc.setdefault(key, np.zeros(3))
                    a += (e - st, g, p)
                tot += e - st
        res[name] = (acc, tot)
    keys = sorted(res["SFT"][0], key=lambda k: (k[0], -res["SFT"][0][k][0]))
    print(f"{'cell':32s} {'SFT share/resid/contrib':>26s} {'base share/resid':>18s} {'REF share/resid':>18s}")
    for k in keys:
        row = []
        for name in ("SFT", "base", "REF"):
            acc, tot = res[name]
            a = acc.get(k, np.zeros(3))
            row.append((a[0] / tot, (a[1] - a[2]) / max(a[0], 1), (a[1] - a[2]) / tot))
        if row[0][0] < 0.003:
            continue
        print(
            f"{k[0]+':'+k[1]:32s} {row[0][0]:.3f} {row[0][1]:+.3f} {row[0][2]:+.4f}   {row[1][0]:.3f} {row[1][1]:+.3f}   {row[2][0]:.3f} {row[2][1]:+.3f}"
        )
