# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Robustness of the surprisal match: (a) position-specific curves, (b) conditioning on
Snowball-base NLL of the token's characters instead of Qwen3.5-base NLL, (c) residual by bin."""

import numpy as np
from fulldata import QS, load_full
from surprisal_match import EDGES, NB, POS, REF, SFT, decompose, fmt, per_prompt_tables, tokens_in

F = load_full()
NP = len(POS) - 1


def pos_cond(f, offs, tok):
    # composite bin id = pos_bin * NB + nll_bin, encoded so digitize on EDGES still works: return
    # value used only via custom path below
    raise NotImplementedError


def tables_2d(scorer, cut_attr):
    prompts = sorted({f.prompt_key for f in F})
    pidx = {p: i for i, p in enumerate(prompts)}
    T = {}
    for f in F:
        d = T.setdefault(
            f.author,
            dict(cnt=np.zeros((100, NP * NB)), gsum=np.zeros((100, NP * NB)), obs=np.zeros(100), chars=np.zeros(100)),
        )
        cut = getattr(f, cut_attr)
        offs, tok = tokens_in(f, cut)
        g = tok[scorer] - tok[QS[0]]
        b = np.clip(np.digitize(tok[QS[0]], EDGES) - 1, 0, NB - 1)
        pb = np.clip(np.digitize(offs[:, 0], POS) - 1, 0, NP - 1)
        k = pb * NB + b
        i = pidx[f.prompt_key]
        d["cnt"][i] += np.bincount(k, minlength=NP * NB)
        d["gsum"][i] += np.bincount(k, weights=g, minlength=NP * NB)
        d["obs"][i] += g.sum()
        d["chars"][i] += (offs[:, 1] - offs[:, 0]).sum()
    return T


def sb_cond(f, offs, tok):
    c = f.char["snowball-67b-a2b-base-skew8"]
    out = np.array([np.nansum(c[s:e]) for s, e in offs])
    return out


for s in QS[1:]:
    for cut in ["cut_line", "cut_fused"]:
        print(f"\n== {s} {cut}")
        r = decompose(per_prompt_tables(F, s, cut), SFT)
        print("qwen-base bins       ", "obs", fmt(r["obs"]), "pred", fmt(r["pred"]), "resid", fmt(r["residual"]))
        import surprisal_match as sm

        sm.NB_SAVE = sm.NB
        T2 = tables_2d(s, cut)
        r = decompose(T2, SFT)
        print("pos x qwen-base bins ", "obs", fmt(r["obs"]), "pred", fmt(r["pred"]), "resid", fmt(r["residual"]))
        r = decompose(per_prompt_tables(F, s, cut, cond=sb_cond), SFT)
        print("snowball-base bins   ", "obs", fmt(r["obs"]), "pred", fmt(r["pred"]), "resid", fmt(r["residual"]))
        if cut == "cut_fused":
            for a in ["snowballbase", "kimilinear", "nemotron3nano", "trinitymini"]:
                r = decompose(T2, a, n=500)
                print(
                    f"   {a:14s} pos x qwen-base  obs",
                    fmt(r["obs"]),
                    "pred",
                    fmt(r["pred"]),
                    "resid",
                    fmt(r["residual"]),
                )
    # residual by base-nll bin, fused cut
    T = per_prompt_tables(F, s, "cut_fused")
    cnt_r = sum(T[a]["cnt"].sum(0) for a in REF)
    g_r = sum(T[a]["gsum"].sum(0) for a in REF)
    cnt_s = T[SFT]["cnt"].sum(0)
    g_s = T[SFT]["gsum"].sum(0)
    ch = T[SFT]["chars"].sum()
    print("bin lo  | SFT tok share  ref tok share | gap/token SFT ref | SFT contrib to gap/char")
    for j in range(NB):
        print(
            f"{EDGES[j]:6.3f} | {cnt_s[j]/cnt_s.sum():.3f} {cnt_r[j]/cnt_r.sum():.3f} | {g_s[j]/max(cnt_s[j],1):+.3f} {g_r[j]/max(cnt_r[j],1):+.3f} | {g_s[j]/ch:+.4f}"
        )
