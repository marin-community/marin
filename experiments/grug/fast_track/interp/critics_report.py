# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report data for the critics case studies: ``critics.npz`` (``critics.py``) and ``halt_pairs.npz`` (the halting
branch's ``halt_pairs.py``) plus the tag's ``stats.npz`` and vocabulary -> one JSON (and ``window.DATA`` JS) for the
case-study page, with a printed summary.

usage: critics_report.py <data dir> <interp data dir of the tag> <out dir> [suffix]
"""

import itertools
import json
import os
import sys

import numpy as np

D, TAG_DIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
suf = sys.argv[4] if len(sys.argv) > 4 else ""
z = np.load(f"{D}/critics{suf}.npz")
h = np.load(f"{D}/halt_pairs{suf}.npz")
st = np.load(f"{TAG_DIR}/stats.npz")
with open(f"{TAG_DIR}/vocab.json") as fh:
    V = json.load(fh)
L, E, I, SW = 6, 384, 384, 384
NR = L * E * I
CATS = [str(c) for c in z["categories"]]
meta = z["q/meta"]
cat = meta[:, 2]
pick = z["w/pick"]
cw = meta[pick, 2]
TERMS = [str(t) for t in z["terms"]]
STAGES = ["embed"] + [f"L{i} {p}" for i in range(6) for p in ("attn", "MLP")]
out = {}


def tokstr(i):
    return V[int(i)]


def uid_name(u):
    u = int(u)
    if u >= NR:
        l, i = divmod(u - NR, SW)
        return dict(layer=l, expert=-1, unit=i, name=f"L{l}.shared.n{i}")
    l, r = divmod(u, E * I)
    e, i = divmod(r, I)
    return dict(layer=l, expert=e, unit=i, name=f"L{l}.E{e}.n{i}")


def unit_promotes(info, k=6):
    l, e, i = info["layer"], info["expert"], info["unit"]
    if e < 0:
        ids = st["s_lens_top_id"][l, i][:k]
    else:
        ids = st["lens_top_id"][l, e, i][:k]
    return [tokstr(t) for t in ids]


# ---- counts
nh = z["count/next_hist"]  # [src, trig, V]
trig = [str(t) for t in z["count/count_triggers"]]
counts = {"tokens": z["count/tokens_scanned"].tolist(), "triggers": {}}
for j, t in enumerate(trig):
    tot_src = nh[:, j].sum(1)
    hsum = nh[:, j].sum(0)
    order = np.argsort(-hsum)[:9]
    rows = [dict(tok=tokstr(i), train=int(nh[0, j, i]), paloma=int(nh[1, j, i])) for i in order]
    rest = [int(tot_src[0] - sum(r["train"] for r in rows)), int(tot_src[1] - sum(r["paloma"] for r in rows))]
    counts["triggers"][t] = dict(total=[int(x) for x in tot_src], rows=rows, other=rest)
counts["cat_counts"] = {CATS[c]: z["count/cat_counts"][:, c].astype(int).tolist() for c in range(len(CATS))}
out["counts"] = counts

# ---- per category model A basics
qcat = {}
for c, name in enumerate(CATS):
    m = cat == c
    if not m.any():
        continue
    p = np.exp(z["q/logp_y"][m])
    qcat[name] = dict(
        n=int(m.sum()),
        p_mean=float(p.mean()),
        p_median=float(np.median(p)),
        loss=float((-z["q/logp_label"][m]).mean()),
        top1=float((z["q/rank_y"][m] == 0).mean()),
        tuned_p=np.exp(z["q/lens_tuned_logp"][m]).mean(0).tolist(),
        tuned_p_med=np.median(np.exp(z["q/lens_tuned_logp"][m]), 0).tolist(),
        tuned_top1=(z["q/lens_tuned_rank"][m] == 0).mean(0).tolist(),
        gate_p=np.exp(z["q/lens_gate_logp"][m]).mean(0).tolist(),
        dla=z["q/dla_terms"][m].mean(0).tolist(),
        dla_total=float(z["q/dla_raw"][m].mean()),
    )
out["A"] = dict(cats=qcat, stages=STAGES, terms=TERMS)

# ---- interventions
sk = z["suff_k"].tolist()
nk = z["nec_k"].tolist()
gk = z["global_k"].tolist()


def k_at(v, full, base, ks, fracs=(0.5, 0.8, 0.9)):
    """Per window, the smallest k recovering frac of the log-prob gap (ablated -> clean); median over windows."""
    rec = (v - base[:, None]) / np.maximum(full - base, 1e-6)[:, None]
    res = []
    for f in fracs:
        hit = rec >= f
        k = np.where(hit.any(1), np.asarray(ks)[np.argmax(hit, 1)], np.inf)
        res.append(float(np.median(k)))
    return res


interv = {}
for c, name in enumerate(CATS):
    m = cw == c
    if not m.any():
        continue
    full = z["w/logp"][m]
    d = dict(n=int(m.sum()), p_full=float(np.median(np.exp(full))))
    for mode in ("zero", "mean"):
        v = z[f"w/suff_{mode}"][m]
        d[f"suff_{mode}_p"] = np.median(np.exp(v), 0).tolist()
        d[f"suff_{mode}_k"] = k_at(v[:, :-1], full, v[:, 0], sk[:-1])
        # the smallest k whose p reaches frac x the clean p, median over windows
        hitp = [np.exp(v[:, :-1]) >= f * np.exp(full)[:, None] for f in (0.5, 0.8, 0.9)]
        d[f"suff_{mode}_kp"] = [
            float(np.median(np.where(hh.any(1), np.asarray(sk[:-1])[np.argmax(hh, 1)], np.inf))) for hh in hitp
        ]
        n = z[f"w/nec_{mode}"][m]
        d[f"nec_{mode}_p"] = np.median(np.exp(n), 0).tolist()
        d[f"nec_{mode}_dlogp"] = (n - full[:, None]).mean(0).tolist()
    gs = z["w/gsuff_zero"][m]
    gn = z["w/gnec_zero"][m]
    d["gsuff_p"] = np.median(np.exp(gs), 0).tolist()
    d["gnec_p"] = np.median(np.exp(gn), 0).tolist()
    d["gnec_dlogp"] = (gn - full[:, None]).mean(0).tolist()
    d["attr_k0"] = np.median(z["w/k0"][m], 0).tolist()
    d["attr_km"] = np.median(z["w/km"][m], 0).tolist()
    d["single_dlogp"] = (z["w/single_zero"][m] - full[:, None]).mean(0).tolist()
    d["single_mean_dlogp"] = (z["w/single_mean"][m] - full[:, None]).mean(0).tolist()
    d["single_present"] = z["w/single_present"][m].mean(0).tolist()
    d["attr_layer"] = (z["w/attr_routed_l"][m] + z["w/attr_shared_l"][m]).mean(0).tolist()
    d["attr_head_layer"] = z["w/attr_head"][m].sum(-1).mean(0).tolist()
    d["attr_embed"] = float(z["w/attr_embed"][m].mean())
    interv[name] = d
out["interv"] = dict(cats=interv, suff_k=sk, nec_k=nk, global_k=gk)

# ---- top units per case (global ranking) with roles
units = {}
case_cats = {"crit": (0, 1), "wasn": (5,), "wasn_curly": (6,)}
for case, cs in case_cats.items():
    top = z[f"global/{case}/top"][:24]
    gsum = z[f"global/{case}/sum"][:24]
    m = np.isin(cw, cs)
    npos = m.sum()
    single = (z["w/single_zero"][m] - z["w/logp"][m][:, None]).mean(0)
    single_m = (z["w/single_mean"][m] - z["w/logp"][m][:, None]).mean(0)
    rows = []
    for r, u in enumerate(top):
        info = uid_name(u)
        rows.append(
            dict(
                **info,
                attr=float(gsum[r] / max(npos, 1)),
                abl_zero=float(single[r]),
                abl_mean=float(single_m[r]),
                present=float(z["w/single_present"][m][:, r].mean()),
                promotes=unit_promotes(info),
            )
        )
    # the share of the case's attribution held by each layer
    units[case] = rows
out["units"] = units

# per-case top DLA units (focus) and their routing info
focus = {}
for p, (case, cs) in enumerate([("crit", (0, 1)), (None, None), (None, None), ("wasn", (5,)), ("wasn_curly", (6,))]):
    if case is None:
        continue
    m = np.isin(cat, cs)
    rows = []
    for f in range(4):
        l, e, i = (int(x) for x in z["focus"][p, f])
        rows.append(
            dict(
                name=f"L{l}.E{e}.n{i}",
                selected=float(z["q/f_selected"][m, f].mean()),
                weight=float(z["q/f_weight"][m, f].mean()),
                act=float(np.median(z["q/f_act"][m, f])),
                dla=float(z["q/f_dla"][m, f].mean()),
                bias_driven=float(z["q/f_bias_driven"][m, f].mean()),
                rank=float(np.median(z["q/f_rank"][m, f])),
                promotes=unit_promotes(dict(layer=l, expert=e, unit=i)),
            )
        )
    pre = z["q/dec_pre_terms"][m, 0].mean(0)
    order = np.argsort(-np.abs(pre))[:6]
    focus[case] = dict(
        units=rows,
        pre_terms=[(TERMS[j], float(pre[j])) for j in order],
        pre_actual=float(z["q/dec_pre_actual"][m, 0].mean()),
    )
out["focus"] = focus

# ---- routing at the trigger
se = z["q/slot_expert"][:, :, 0, :]
routing = {}
for case, cs, tok in (("crit", (0, 1, 2), 1927), ("wasn", (5, 6, 7), 7691)):
    m = np.isin(cat, cs)
    layers = []
    for l in range(6):
        c = np.bincount(se[m, l].reshape(-1), minlength=E) / m.sum()
        o = np.argsort(-c)[:10]
        et = st["expert_token_ids"][l]
        layers.append([dict(expert=int(e), freq=float(c[e]), share=float(et[e, tok] / max(et[e].sum(), 1))) for e in o])
    routing[case] = dict(
        layers=layers,
        always=[int((np.bincount(se[m, l].reshape(-1), minlength=E) / m.sum() >= 0.95).sum()) for l in range(6)],
    )
out["routing"] = routing

# ---- ablations everywhere
names = [str(n) for n in z["abl/names"]]
al = z["abl/logp"]
abl = {}
for case, cs in case_cats.items():
    m = np.isin(cat, cs)
    base = al[0, m]
    abl[case] = {
        n: float((al[k, m] - base).mean()) for k, n in enumerate(names) if k > 0 and not np.isnan(al[k, m]).any()
    }
out["abl"] = abl

# truncation
tq = z["trunc/query"]
tl = z["trunc/length_index"]
tlp = z["trunc/logp_y"]
trunc = {}
for case, cs in case_cats.items():
    m = np.isin(cat[tq], cs)
    trunc[case] = [float(np.median(np.exp(tlp[m & (tl == k)]))) for k in range(len(z["trunc_lengths"]))]
out["trunc"] = dict(lengths=z["trunc_lengths"].tolist(), p=trunc)

# ---- model B
q = h["queries"]
hcat = q[:, 2]
SK = ["0-1-2-3-4-5", "1-2-3-4-5", "2-3-4-5", "3-4-5", "4-5", "5"]
B = {}
for name in CATS:
    c = CATS.index(name)
    m = hcat == c
    if not m.any():
        continue
    d = {"n": int(m.sum())}
    for r in ("ctl", "dampen", "probe"):
        rr = dict(
            loss=float((-h[f"{r}/final/logp_label"][m]).mean()),
            p=float(np.median(np.exp(h[f"{r}/final/logp_y"][m]))),
            skip45=float((-h[f"{r}/skip/4-5/logp_label"][m]).mean()),
            depth_p=[float(np.median(np.exp(h[f"{r}/skip/{s}/logp_y"][m]))) for s in SK]
            + [float(np.median(np.exp(h[f"{r}/final/logp_y"][m])))],
            depth_top1=[
                float((h[f"{r}/skip/{s}/top1"][m] == h["targets"][[0, 0, 0, 1, 2, 3, 4, 3][c]]).mean()) for s in SK
            ],
            loo=[float((h[f"{r}/skip/{j}/logp_label"][m] - h[f"{r}/final/logp_label"][m]).mean()) for j in range(5)],
        )
        if f"{r}/halt/g" in h.files:
            g = h[f"{r}/halt/g"][0][m]
            lg = h[f"{r}/halt/logit"][0][m]
            y = h[f"{r}/exit/y"][m]
            mp = h[f"{r}/exit/maxp"][m]
            full = -h[f"{r}/final/logp_label"][m]
            skp = -h[f"{r}/skip/4-5/logp_label"][m]
            votes = h[f"{r}/votes/routed"][:, m]
            bias = float(h[f"{r}/halt_bias"][3])
            rr.update(
                G_med=float(np.median(g)),
                G_mean=float(g.mean()),
                G_q=np.quantile(g, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
                G_lt05=float((g < 0.05).mean()),
                G_lt2=float((g < 0.2).mean()),
                G_lt5=float((g < 0.5).mean()),
                logit_med=float(np.median(lg)),
                z_med=float(np.median(lg - bias)),
                bias=bias,
                votes_med=np.median(votes, 1).tolist(),
                votes_mean=votes.mean(1).tolist(),
                other_votes_med=float(np.median(lg - bias - votes.sum(0))),
                pexit_med=float(np.median(np.exp(h[f"{r}/exit/logp_y"][m]))),
                pexit_mean=float(np.exp(h[f"{r}/exit/logp_y"][m]).mean()),
                exit_top1=float((h[f"{r}/exit/top_ids"][m, 0] == h[f"{r}/final/top_ids"][m, 0]).mean()),
                exit_correct=float(
                    (h[f"{r}/exit/top_ids"][m, 0] == q[m, 0] * 0 + h["targets"][[0, 0, 0, 1, 2, 3, 4, 3][c]]).mean()
                ),
                y=float(y.mean()),
                maxp_med=float(np.median(mp)),
                skip_cost=float((skp - full).mean()),
                hard05=float(np.where(g < 0.05, skp, full).mean() - full.mean()),
                hist_G=np.histogram(g, bins=np.linspace(0, 1, 21))[0].tolist(),
            )
        d[r] = rr
    B[name] = d
# binned by exit max-prob for crit_ics, wasn_t and all crit
bins = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.01]
binned = {}
for case, cs in (("crit_ics", (0, 1)), ("wasn_apos_t", (5,)), ("wasn_curly_t", (6,)), ("crit_other", (2,))):
    m = np.isin(hcat, cs)
    rows = []
    mp = h["dampen/exit/maxp"]
    g = h["dampen/halt/g"][0]
    lg = h["dampen/halt/logit"][0]
    y = h["dampen/exit/y"]
    full = -h["dampen/final/logp_label"]
    skp = -h["dampen/skip/4-5/logp_label"]
    for lo, hi in itertools.pairwise(bins):
        mm = m & (mp >= lo) & (mp < hi)
        if mm.any():
            rows.append(
                dict(
                    lo=lo,
                    hi=min(hi, 1.0),
                    n=int(mm.sum()),
                    y=float(y[mm].mean()),
                    G=float(np.median(g[mm])),
                    logit=float(np.median(lg[mm])),
                    skip_cost=float((skp[mm] - full[mm]).mean()),
                )
            )
    binned[case] = rows
out["B"] = dict(
    cats=B,
    binned=binned,
    dump_err_g=float(h["dampen/dump_err_g"].mean()),
    dump_err_loss=float(h["dampen/dump_err_loss"].mean()),
)


# AUC of the dampening logit for y over all trigger positions
def auc(s, y):
    y = y.astype(bool)
    r = np.argsort(np.argsort(s))
    n1 = y.sum()
    n0 = (~y).sum()
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


out["B"]["auc"] = {
    case: auc(h["dampen/halt/logit"][0][np.isin(hcat, cs)], h["dampen/exit/y"][np.isin(hcat, cs)])
    for case, cs in (("crit", (0, 1, 2)), ("wasn", (5, 6, 7)), ("produ", (3,)), ("ex", (4,)))
}

# top voting units at the crit / wasn / produ positions (summed over positions)
voters = {}
for case, cs in (("crit_ics", (0, 1)), ("wasn_apos_t", (5,)), ("produ_cer", (3,))):
    m = np.isin(hcat, cs)
    acc = {}
    for l in range(4):
        ids = h["dampen/votes/top_ids"][l, m]
        vals = h["dampen/votes/top_vals"][l, m].astype(np.float64)
        for a, b in zip(ids.reshape(-1), vals.reshape(-1), strict=True):
            acc[(l, int(a))] = acc.get((l, int(a)), 0.0) + b
    top = sorted(acc.items(), key=lambda kv: -kv[1])[:10]
    voters[case] = [dict(name=f"L{l}.E{u // 384}.n{u % 384}", vote=v / m.sum()) for (l, u), v in top]
out["B"]["voters"] = voters

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/data.js", "w") as f:
    f.write("window.DATA = " + json.dumps(out, default=float) + ";\n")
with open(f"{OUT}/report{suf}.json", "w") as f:
    json.dump(out, f, default=float, indent=0)
print("wrote", len(json.dumps(out, default=float)) / 1e3, "KB")
for k in ("crit_ics_first", "crit_ics_repeat", "wasn_apos_t", "wasn_curly_t"):
    iv = interv[k]
    print(
        k,
        "n",
        iv["n"],
        "suff_mean k50/80/90",
        iv["suff_mean_k"],
        "p-scale",
        iv["suff_mean_kp"],
        iv["suff_zero_kp"],
        "suff_zero",
        iv["suff_zero_k"],
        "attr k0",
        iv["attr_k0"],
    )
for case in units:
    print(case, [(u["name"], round(u["attr"], 3), round(u["abl_zero"], 3), u["promotes"][:3]) for u in units[case][:8]])
print("abl crit", sorted(abl["crit"].items(), key=lambda kv: kv[1])[:12])
print("abl wasn", sorted(abl["wasn"].items(), key=lambda kv: kv[1])[:12])
print("trunc", out["trunc"])
print("focus", json.dumps(focus)[:1500])
print("auc", out["B"]["auc"], "voters", voters)
