# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report data for the Sri -> Lanka section: summarizes ``sri.npz`` (with ``stats.npz``, ``tokens.npz`` and
``vocab.json`` for unit dedication) into one JSON file.

Usage: ``python -m experiments.grug.fast_track.interp.sri_report <local tag dir> <out.json>``.
"""

import json
import sys

import numpy as np

D = sys.argv[1].rstrip("/") + "/"
OUT = sys.argv[2]
z = np.load(D + "sri.npz")
with open(D + "vocab.json") as f:
    v = json.load(f)
st = np.load(D + "stats.npz")
tk = np.load(D + "tokens.npz")["tokens"].reshape(-1)
cat = z["q/meta"][:, 2]
C = [str(c) for c in z["categories"]]
T = [str(t) for t in z["terms"]]
n_cat = z["n_cat"]
r = lambda x, d=3: round(float(x), d)  # noqa: E731
out = {"categories": C, "n": [int((cat == i).sum()) for i in range(len(C))]}
src = z["q/source"]
out["sources"] = {
    C[i]: dict(zip(*[x.tolist() for x in np.unique(src[cat == i], return_counts=True)], strict=True)) for i in (0, 1)
}
out["sources"] = {k: {str(a): int(b) for a, b in d.items()} for k, d in out["sources"].items()}
lp, rk = z["q/logp_y"], z["q/rank_y"]
out["pred"] = [
    dict(
        cat=C[i],
        p_mean=r(np.exp(lp[m]).mean()),
        p_median=r(np.median(np.exp(lp[m]))),
        top1=r((rk[m] == 0).mean()),
        nll=r(-lp[m].mean()),
    )
    for i in range(len(C))
    for m in [cat == i]
]
# n0 routing and activation
f = {
    k: z[f"q/f_{k}"]
    for k in (
        "selected",
        "weight",
        "rank",
        "weight_rank",
        "bias_driven",
        "act",
        "pre",
        "dla",
        "margin",
        "logit",
        "biased_rank",
    )
}
rows = []
for i in range(4):
    m = cat == i
    s = f["selected"][m, 0]
    rows.append(
        dict(
            cat=C[i],
            n=int(m.sum()),
            selected=r(s.mean()),
            weight=r(f["weight"][m, 0][s].mean()),
            weight_rank0=r((f["weight_rank"][m, 0][s] == 0).mean()),
            logit_rank0=r((f["rank"][m, 0] == 0).mean()),
            bias_driven=r(f["bias_driven"][m, 0].mean()),
            margin=r(np.median(f["margin"][m, 0]), 2),
            act=r(f["act"][m, 0].mean(), 2),
            fire=r((f["pre"][m, 0] > 0).mean()),
            dla=r(f["dla"][m, 0].mean(), 2),
            act_hist=np.histogram(np.clip(f["act"][m, 0], 0, 24), bins=24, range=(0, 24))[0].tolist(),
        )
    )
out["n0"] = rows
a = z["acc/e304_count"]
out["traffic"] = dict(
    sample_ri_share=r(a[390] / a.sum(), 4),
    sample_ri_to_e304=r(a[390] / z["acc/token_count"][390]),
    sample_n0_fire_ri=r(z["acc/n0_fire"][390] / a[390]),
    sample_n0_fire_all=r(z["acc/n0_fire"].sum() / a.sum()),
    stats_ri_share=r(st["expert_token_ids"][5, 304, 390] / st["expert_tokens"][5, 304], 4),
    stats_ri_to_e304=r(st["expert_token_ids"][5, 304, 390] / st["token_counts"][390]),
    stats_e304_tokens=int(st["expert_tokens"][5, 304]),
    stats_tokens=int(st["tokens"]),
    stats_n0_fire=r(st["fire"][5, 304, 0] / st["expert_tokens"][5, 304]),
    stats_n0_mean=r(st["act_sum"][5, 304, 0] / st["expert_tokens"][5, 304]),
    top_tokens=[
        [v[k], int(st["expert_token_ids"][5, 304, k])] for k in np.argsort(-st["expert_token_ids"][5, 304])[:12]
    ],
    n0_act_by_token=[[v[k], r(z["acc/n0_act"][k] / z["acc/n0_act"].sum())] for k in np.argsort(-z["acc/n0_act"])[:8]],
)
# ablations
names = [str(x) for x in z["abl/names"]]
L = z["abl/logp"]
abl = []
for j, n in enumerate(names):
    d = L[j] - L[0]
    abl.append(
        [n] + [None if not ((cat == i) & ~np.isnan(d)).any() else r(np.nanmean(d[cat == i])) for i in range(len(C))]
    )
out["abl"] = abl
# DLA terms
dt = z["q/dla_terms"]
out["terms"] = T
out["dla"] = {C[i]: [r(x, 3) for x in dt[cat == i].mean(0)] for i in range(len(C))}
ed = z["q/expert_dla"]
out["expert_dla"] = {
    C[i]: [
        [l, e, r(ed[cat == i, l, e].mean())]
        for l, e in [divmod(int(k), 384) for k in np.argsort(-ed[cat == i].mean(0).ravel())[:8]]
    ]
    for i in (0, 1, 4, 5, 6, 7)
}
# lens
out["lens"] = {
    kind: {
        C[i]: dict(
            p=[r(x) for x in np.exp(z[f"q/lens_{kind}_logp"][cat == i]).mean(0)],
            top1=[r(x) for x in (z[f"q/lens_{kind}_rank"][cat == i] == 0).mean(0)],
        )
        for i in range(len(C))
    }
    for kind in ("gate", "tuned")
}
# neurons ranked by DLA, with dedication from the stats sample's top contexts
nd = z["acc/neuron_dla"]
na, nf = z["acc/neuron_act"], z["acc/neuron_fire"]


def dedication(l, e, i):
    pos = st["top_pos"][l, e, i]
    val = st["top_val"][l, e, i]
    ok = val > 0
    cur = tk[pos[ok]]
    prev = tk[np.maximum(pos[ok] - 1, 0)]
    nxt = tk[np.minimum(pos[ok] + 1, len(tk) - 1)]
    return dict(
        top_ri=r((cur == 390).mean() if ok.any() else 0),
        top_sri=r(((cur == 390) & (prev == 337)).mean() if ok.any() else 0),
        top_tokens=[v[t] for t in cur[:6]],
        top_next=[v[t] for t in nxt[:6]],
        top_prev=[v[t] for t in prev[:6]],
        fire_rate=r(st["fire"][l, e, i] / max(st["expert_tokens"][l, e], 1)),
        mean_act=r(st["act_sum"][l, e, i] / max(st["expert_tokens"][l, e], 1)),
        max_act=r(st["act_max"][l, e, i], 1),
        lens=[v[t] for t in st["lens_top_id"][l, e, i][:4]],
    )


units = {}
for name, cats in (("sri_first", [0]), ("sri_repeat", [1]), ("hong_first", [4]), ("los_first", [6])):
    s = nd[cats].sum(0) / n_cat[cats].sum()
    flat = s.ravel()
    pos = np.sort(flat[flat > 0])[::-1]
    cs = np.cumsum(pos) / pos.sum()
    top = []
    for k in np.argsort(-flat)[:15]:
        l, e, i = (int(x) for x in np.unravel_index(k, s.shape))
        top.append(
            dict(
                l=l,
                e=e,
                i=i,
                dla=r(flat[k]),
                act=r(na[cats, l, e, i].sum() / n_cat[cats].sum(), 2),
                fire=r(nf[cats, l, e, i].sum() / n_cat[cats].sum()),
                **dedication(l, e, i),
            )
        )
    units[name] = dict(
        total=r(flat.sum(), 2),
        positive=r(pos.sum(), 2),
        top1_share=r(pos[0] / pos.sum()),
        top10_share=r(cs[9]),
        n50=int(np.searchsorted(cs, 0.5) + 1),
        n80=int(np.searchsorted(cs, 0.8) + 1),
        cum=[r(cs[k - 1]) for k in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)],
        top=top,
        by_layer=[r(s[l].sum(), 2) for l in range(6)],
    )
out["units"] = units
sd = z["acc/shared_dla"]
out["shared"] = {
    C[i]: dict(
        by_layer=[r(x, 2) for x in (sd[i] / n_cat[i]).sum(1)],
        top=[
            [int(l), int(u), r((sd[i] / n_cat[i])[l, u])]
            for l, u in [divmod(int(k), 384) for k in np.argsort(-(sd[i] / n_cat[i]).ravel())[:5]]
        ],
    )
    for i in (0, 1)
}
# router / pre decomposition for n0 (focus 0)
se = z["q/slot_expert"]
dec = {}
for key in ("router", "pre"):
    t = z[f"q/dec_{key}_terms"][:, 0]
    act = z[f"q/dec_{key}_actual"][:, 0]
    sl = z[f"q/dec_{key}_slots"][:, 0]
    dd = {}
    for i in range(4):
        m = cat == i
        acc = {}
        for q in np.flatnonzero(m):
            for l in range(6):
                for j in range(4):
                    for k in range(10):
                        key2 = (l, j, int(se[q, l, j, k]))
                        acc[key2] = acc.get(key2, 0.0) + float(sl[q, l, j, k])
        top = sorted(acc.items(), key=lambda x: -abs(x[1]))[:8]
        dd[C[i]] = dict(
            actual=r(act[m].mean(), 2),
            terms=[r(x, 3) for x in t[m].mean(0)],
            slots=[[k[0], k[1], k[2], r(x / m.sum())] for k, x in top],
        )
    dec[key] = dd
out["dec"] = dec
# swaps
base = z["q/logp_y"]
fa, fs = f["act"][:, 0], f["selected"][:, 0]
sw = []
for k in sorted(z.files):
    if not (k.startswith("swap/") and k.endswith("/query")):
        continue
    lab = k.split("/")[1]
    q = z[k]
    c = cat[q]
    for i in sorted(set(c.tolist())):
        m = c == i
        sw.append(
            dict(
                prefix=lab.split("=")[1],
                cat=C[i],
                n=int(m.sum()),
                p_before=r(np.exp(base[q][m]).mean()),
                p_after=r(np.exp(z[f"swap/{lab}/logp_y"][m]).mean()),
                act_before=r(fa[q][m].mean(), 2),
                act_after=r(z[f"swap/{lab}/f_act"][m, 0].mean(), 2),
                sel_after=r(z[f"swap/{lab}/f_selected"][m, 0].mean()),
                w_after=r(z[f"swap/{lab}/f_weight"][m, 0].mean(), 2),
                router_embed_after=r(z[f"swap/{lab}/dec_router_terms"][m, 0, 0].mean(), 2),
                router_after=r(z[f"swap/{lab}/f_logit"][m, 0].mean(), 2),
                router_before=r(f["logit"][q][m, 0].mean(), 2),
            )
        )
out["swaps"] = sw
# truncation
tq, tl = z["trunc/query"], z["trunc/length_index"]
Ls = z["trunc_lengths"].tolist()
tr = {}
for i in (0, 1, 4, 5, 6, 7):
    rows = []
    for li in range(len(Ls)):
        m = (cat[tq] == i) & (tl == li)
        rows.append(
            dict(
                k=Ls[li],
                p=r(np.exp(z["trunc/logp_y"][m]).mean()),
                act=r(z["trunc/f_act"][m, 0].mean(), 2),
                sel=r(z["trunc/f_selected"][m, 0].mean()),
                dla_routed5=r(z["trunc/dla_terms"][m, T.index("L5.routed")].mean(), 2),
                dla_ve5=r(z["trunc/dla_terms"][m, T.index("L5.ve")].mean(), 2),
                dla_embed=r(z["trunc/dla_terms"][m, 0].mean(), 2),
            )
        )
    rows.append(
        dict(
            k="full",
            p=r(np.exp(base[cat == i]).mean()),
            act=r(fa[cat == i].mean(), 2),
            sel=r(fs[cat == i].mean()),
            dla_routed5=r(dt[cat == i, T.index("L5.routed")].mean(), 2),
            dla_ve5=r(dt[cat == i, T.index("L5.ve")].mean(), 2),
            dla_embed=r(dt[cat == i, 0].mean(), 2),
        )
    )
    tr[C[i]] = rows
out["trunc"] = tr
out["focus"] = z["focus_by_case"].tolist()
# worked examples
meta, ctx, nxt = z["q/meta"], z["q/context"], z["q/next"]
tm, tv = z["q/top_meta"], z["q/top_val"]
rng = np.random.default_rng(3)
ex = []
for i, k in ((0, 4), (1, 3), (2, 3), (3, 2), (4, 1), (6, 1)):
    pool = np.flatnonzero((cat == i) & (meta[:, 4] == 1))
    for q in rng.choice(pool, size=min(k, len(pool)), replace=False):
        off = int(meta[q, 3])
        c = ctx[q][-min(len(ctx[q]), off + 1) :]
        terms = sorted(range(len(T)), key=lambda t: -abs(dt[q, t]))[:5]
        ex.append(
            dict(
                cat=C[i],
                source=str(src[q]),
                left="".join(v[t] for t in c),
                next="".join(v[t] for t in nxt[q][:3]),
                p=r(np.exp(lp[q])),
                prior=int(meta[q, 5]),
                doc_offset=off,
                n0=dict(
                    sel=bool(f["selected"][q, 0]),
                    w=r(f["weight"][q, 0], 2),
                    act=r(f["act"][q, 0], 2),
                    dla=r(f["dla"][q, 0], 2),
                ),
                abl_n0=r(np.exp(L[names.index("n0:zero")][q])) if not np.isnan(L[names.index("n0:zero")][q]) else None,
                abl_ve=r(np.exp(L[names.index("L5.ve:zero")][q])),
                terms=[[T[t], r(dt[q, t], 2)] for t in terms],
                units=[
                    [int(a), int(b), int(e), int(u), r(x, 2)]
                    for (a, b, e, u), x in zip(tm[q][:5], tv[q][:5], strict=True)
                ],
            )
        )
out["examples"] = ex
with open(OUT, "w") as f:
    json.dump(out, f)
print("wrote", OUT, len(json.dumps(out)) / 1e3, "kB")
print(json.dumps({k: out[k] for k in ("traffic", "n0")}, indent=0)[:3000])
for nm, u in units.items():
    print(nm, {k: u[k] for k in ("top1_share", "top10_share", "n50", "n80", "by_layer")})
    for t in u["top"][:6]:
        print("   ", t)
