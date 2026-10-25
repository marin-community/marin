# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decider-neuron analysis on the host: unit rankings from ``decider_discover.npz``, the decider registry the
``replicate`` pass reads, and (after it) the page data and the halting hand-off list.

Usage::

    python -m experiments.grug.fast_track.interp.decider_report registry --data DIR
    python -m experiments.grug.fast_track.interp.decider_report page --data DIR --out DIR

A unit's strong fires are its main level (top 1% of its nonzero activations). Precision is P(confident & right | strong
fire) with a Wilson 95% lower bound; deciders need ``MIN_SUPPORT`` strong fires and are ranked by the lower bound.
"""

import argparse
import json
import os

import numpy as np

from experiments.grug.fast_track.interp import decider, forward

L, E, I, SI = decider.L, decider.E, decider.I, decider.SI
NR = L * E * I
MIN_SUPPORT = 100
Z = 1.96
EARLY_LAYERS = 4  # L0-L3
ATLAS_CATEGORIES = ("ngram", "structural", "token", "output", "topic", "dense", "polysemantic")
DIRECT_SHARE = 0.05
SEED = 0
# Registry decider sets (bit j): name -> (outcome stat, min Wilson LB, min lift over the unit's own expert's routed
# rate). "random" is a random control matched per layer to cr90_lb80.
SET_RULES = (
    ("cr90_lb95", "cr90", 0.95, 0.0),
    ("cr90_lb90", "cr90", 0.90, 0.0),
    ("cr90_lb80", "cr90", 0.80, 0.0),
    ("cr90_lb70", "cr90", 0.70, 0.0),
    ("cr90_lb80_xlift1.5", "cr90", 0.80, 1.5),
    ("cw50_lb20", "cw50", 0.20, 0.0),
    ("cw50_lb15", "cw50", 0.15, 0.0),
    ("random", None, None, None),
)
REGISTRY_TOP = 2400
REGISTRY_EARLY = 1000
REGISTRY_CW = 600
ABLATE_TOP, ABLATE_EARLY, ABLATE_CW = 30, 20, 10


def wilson_lb(k: np.ndarray, n: np.ndarray, z: float = Z) -> np.ndarray:
    n = np.maximum(n, 1e-9)
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - rad) / den


def wilson_ub(k: np.ndarray, n: np.ndarray, z: float = Z) -> np.ndarray:
    n = np.maximum(n, 1e-9)
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    rad = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + rad) / den


def load(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def stat(levels: np.ndarray, name: str) -> np.ndarray:
    return levels[..., decider.STATS.index(name)]


def unit_levels(z: dict) -> np.ndarray:
    """``[NR + L * SI, levels, stats]``: routed units then shared units."""
    return np.concatenate(
        [
            z["levels"].reshape(NR, len(decider.LEVEL_NAMES), -1),
            z["s_levels"].reshape(L * SI, len(decider.LEVEL_NAMES), -1),
        ]
    )


def unit_keys() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Layer, expert (-1: shared) and unit of every row of ``unit_levels``."""
    li, ei, ui = np.meshgrid(np.arange(L), np.arange(E), np.arange(I), indexing="ij")
    sl, su = np.meshgrid(np.arange(L), np.arange(SI), indexing="ij")
    return (
        np.concatenate([li.ravel(), sl.ravel()]),
        np.concatenate([ei.ravel(), np.full(sl.size, -1)]),
        np.concatenate([ui.ravel(), su.ravel()]),
    )


def base_rates(z: dict) -> dict[str, float]:
    src = z["source"].sum(0)
    n = src[0]
    return {k: float(src[j] / n) for j, k in enumerate(decider.TOKEN_STATS) if k != "n"} | {"n": float(n)}


def unit_table(z: dict, level: int = decider.MAIN_LEVEL) -> dict[str, np.ndarray]:
    """Per unit (rows of ``unit_levels``) stats at ``level``: support, precisions with Wilson bounds, lifts,
    coverage, direct-logit share and top-promoted hit rate, the expert-conditional base rate."""
    lv = unit_levels(z)
    base = base_rates(z)
    cur, routed = lv[:, level], lv[:, 0]
    n = stat(cur, "n")
    out = {"n": n, "fires": stat(lv[:, 1], "n"), "routed": stat(routed, "n")}
    totals = z["source"].sum(0)
    for k in ("cr70", "cr90", "cr99", "cw50", "cw90", "top1"):
        kk = stat(cur, k)
        out[f"{k}_k"] = kk
        out[f"{k}_p"] = kk / np.maximum(n, 1)
        out[f"{k}_lb"] = np.where(n > 0, wilson_lb(kk, n), 0.0)
        out[f"{k}_ub"] = np.where(n > 0, wilson_ub(kk, n), 0.0)
        out[f"{k}_lift"] = out[f"{k}_p"] / base[k]
        out[f"{k}_cov"] = kk / totals[decider.TOKEN_STATS.index(k)]
        rk, rn = stat(routed, k), stat(routed, "n")
        out[f"{k}_routed_p"] = rk / np.maximum(rn, 1)
    out["logp"] = stat(cur, "logp") / np.maximum(n, 1)
    out["dla"] = stat(cur, "dla") / np.maximum(n, 1)
    out["tlogit"] = stat(cur, "tlogit") / np.maximum(n, 1)
    out["dla_share"] = stat(cur, "dla") / np.maximum(stat(cur, "tlogit"), 1e-6)
    out["is_top"] = stat(cur, "is_top") / np.maximum(n, 1)
    out["mean_act"] = stat(routed, "act") / np.maximum(stat(routed, "n"), 1)
    out["strong_act"] = stat(cur, "act") / np.maximum(n, 1)
    thr = np.concatenate([z["thr"].reshape(NR, -1), z["s_thr"].reshape(L * SI, -1)])
    out["thr"] = thr[:, level]
    out["layer"], out["expert"], out["unit"] = unit_keys()
    return out


def atlas_category(atlas_metrics: str) -> np.ndarray:
    """Atlas category index per row of ``unit_levels`` (-1 for shared units)."""
    cat = np.load(atlas_metrics)["category"].reshape(NR).astype(np.int16)
    return np.concatenate([cat, np.full(L * SI, -1, np.int16)])


def build_registry(z: dict) -> dict:
    t = unit_table(z)
    ok = t["n"] >= MIN_SUPPORT
    rng = np.random.default_rng(SEED)
    bits = np.zeros(len(t["n"]), np.int32)
    for j, (_name, key, lb, xlift) in enumerate(SET_RULES):
        if key is not None:
            xl = t[f"{key}_p"] / np.maximum(t[f"{key}_routed_p"], 1e-9)
            bits |= (ok & (t[f"{key}_lb"] >= lb) & (xl >= xlift)).astype(np.int32) << j
    # Random control: as many supported units per layer as the cr90_lb80 set has.
    ref_set = (bits >> 2) & 1
    for layer in range(L):
        want = int(ref_set[t["layer"] == layer].sum())
        pool = np.flatnonzero(ok & (t["layer"] == layer))
        pick = rng.choice(pool, size=min(want, len(pool)), replace=False)
        bits[pick] |= 1 << (len(SET_RULES) - 1)
    lb = np.where(ok, t["cr90_lb"], -1)
    cw = np.where(ok, t["cw50_lb"], -1)
    order = np.lexsort((-t["n"], -lb))
    early = order[t["layer"][order] < EARLY_LAYERS]
    cw_order = np.argsort(-cw)
    reg_rows = list(dict.fromkeys([*order[:REGISTRY_TOP], *early[:REGISTRY_EARLY], *cw_order[:REGISTRY_CW]]))
    reg_rows = reg_rows[: decider.NUM_REGISTRY]
    ablate = set(order[:ABLATE_TOP].tolist()) | set(early[:ABLATE_EARLY].tolist()) | set(cw_order[:ABLATE_CW].tolist())
    reg_index = {int(r): j for j, r in enumerate(reg_rows)}
    units = []
    for r in np.flatnonzero((bits > 0) | np.isin(np.arange(len(bits)), reg_rows)):
        layer, ex, un = int(t["layer"][r]), int(t["expert"][r]), int(t["unit"][r])
        u = dict(
            layer=layer, expert=ex, unit=un, thr=float(t["thr"][r]), bits=int(bits[r]), reg=reg_index.get(int(r), -1)
        )
        if int(r) in ablate:
            width = I if ex >= 0 else SI
            ctrl = int(rng.integers(width - 1))
            ctrl += ctrl >= un
            ctrl_row = r - un + ctrl
            u.update(ablate=True, mean=float(t["mean_act"][r]), ctrl=ctrl, ctrl_mean=float(t["mean_act"][ctrl_row]))
        units.append(u)
    return {"sets": [s[0] for s in SET_RULES], "units": units, "reg_rows": [int(r) for r in reg_rows]}


BANDS = (
    ("zero", 0, 1),
    ("<50%", 1, 2),
    ("50-20%", 2, 3),
    ("20-10%", 3, 4),
    ("10-3%", 4, 5),
    ("3-1%", 5, 6),
    ("1-0.3%", 6, 7),
    ("0.3-0.1%", 7, 8),
    ("top 0.1%", 8, None),
)
TABLE_TOP = 300
TABLE_EARLY = 150
TABLE_CW = 60
EARLY_JSON = 200


def band_precision(lv: np.ndarray, key: str) -> tuple[np.ndarray, np.ndarray]:
    """``[N, levels, stats]`` cumulative stats -> per-band ``key`` rate and count ``[N, len(BANDS)]``."""
    n, k = stat(lv, "n"), stat(lv, key)
    rates, counts = [], []
    for _, lo, hi in BANDS:
        bn = n[:, lo] - (n[:, hi] if hi is not None else 0)
        bk = k[:, lo] - (k[:, hi] if hi is not None else 0)
        rates.append(bk / np.maximum(bn, 1))
        counts.append(bn)
    return np.stack(rates, 1), np.stack(counts, 1)


def decode(vocab: list[str], ids) -> list[str]:
    return [vocab[i] if i >= 0 else "" for i in ids]


def unit_name(layer: int, expert: int, unit: int) -> str:
    return f"L{layer}.shared.{unit}" if expert < 0 else f"L{layer}.E{expert}.{unit}"


def round_list(x, digits: int = 4) -> list:
    return [round(float(v), digits) for v in x]


def page_data(z: dict, rep: dict, cov: dict, abl: list[dict], reg: dict, atlas_metrics: str, vocab: list[str]) -> dict:
    t = unit_table(z)
    h = unit_table(rep)
    ok = t["n"] >= MIN_SUPPORT
    cat = atlas_category(atlas_metrics)
    lb = np.where(ok, t["cr90_lb"], -1)
    order = np.lexsort((-t["n"], -lb))
    base = base_rates(z)
    hbase = base_rates(rep)
    direct = t["dla_share"] >= DIRECT_SHARE
    layer = t["layer"]
    shared = t["expert"] < 0
    res: dict = {"base": base, "held_out_base": hbase, "min_support": MIN_SUPPORT, "direct_share": DIRECT_SHARE}
    # Counts by layer and threshold; direct vs indirect; atlas categories.
    counts = {}
    for th in (0.95, 0.9, 0.8, 0.7):
        s = ok & (lb >= th)
        counts[f"{th:g}"] = {
            "routed": [int((s & ~shared & (layer == li)).sum()) for li in range(L)],
            "shared": [int((s & shared & (layer == li)).sum()) for li in range(L)],
            "direct": [int((s & direct & (layer == li)).sum()) for li in range(L)],
            "held_out_lb_ge_0.8": float((h["cr90_lb"][s] >= 0.8).mean()) if s.any() else 0.0,
            "held_out_p_median": float(np.median(h["cr90_p"][s])) if s.any() else 0.0,
        }
    res["counts"] = counts
    s90 = ok & (lb >= 0.9) & ~shared
    cats = {}
    for ci, cname in enumerate(ATLAS_CATEGORIES):
        all_c = (cat == ci) & ~shared
        cats[cname] = {
            "deciders": [int((s90 & (cat == ci) & (layer == li)).sum()) for li in range(L)],
            "all": [int((all_c & (layer == li)).sum()) for li in range(L)],
        }
    res["atlas"] = cats
    # Expert-conditional lift distribution per layer (deciders at LB >= 0.9).
    xl = t["cr90_p"] / np.maximum(t["cr90_routed_p"], 1e-9)
    res["expert_lift"] = {
        "median": [float(np.median(xl[s90 & (layer == li)])) if (s90 & (layer == li)).any() else 0 for li in range(L)],
        "frac_ge_1.5": [
            float((xl[s90 & (layer == li)] >= 1.5).mean()) if (s90 & (layer == li)).any() else 0 for li in range(L)
        ],
        "expert_rate_median": [
            float(np.median(t["cr90_routed_p"][s90 & (layer == li)])) if (s90 & (layer == li)).any() else 0
            for li in range(L)
        ],
    }
    # Dose response, held out (replicate levels at the discover cuts).
    hl = unit_levels(rep)
    rates, nb = band_precision(hl, "cr90")
    dose = {"bands": [b[0] for b in BANDS], "layers": {}, "cw": {}}
    for li in range(L):
        sel = s90 & (layer == li)
        rnd = ok & ~shared & (layer == li)
        dose["layers"][li] = {
            "deciders": round_list(np.sum(rates[sel] * nb[sel], 0) / np.maximum(nb[sel].sum(0), 1)),
            "all_units": round_list(np.sum(rates[rnd] * nb[rnd], 0) / np.maximum(nb[rnd].sum(0), 1)),
            "n_units": int(sel.sum()),
        }
    cw_rates, cw_nb = band_precision(hl, "cw50")
    cw_sel = ok & (t["cw50_lb"] >= 0.15)
    dose["cw"] = {
        "cw_units_cw50": round_list(np.sum(cw_rates[cw_sel] * cw_nb[cw_sel], 0) / np.maximum(cw_nb[cw_sel].sum(0), 1)),
        "all_units_cw50": round_list(np.sum(cw_rates[ok] * cw_nb[ok], 0) / np.maximum(cw_nb[ok].sum(0), 1)),
    }
    res["dose"] = dose
    # Registry joins: examples, target types, ablations.
    reg_rows = np.asarray(reg["reg_rows"])
    reg_of_row = {int(r): j for j, r in enumerate(reg_rows)}
    abl_by_reg = {a["reg"]: a for a in abl}

    def ablation(r: int) -> dict | None:
        a = abl_by_reg.get(reg_of_row.get(r, -1))
        if a is None or a["n"] == 0:
            return None
        n = a["n"]
        return {
            "n": int(n),
            "seqs": a["sequences"],
            "dlogp": (a["abl_logp"] - a["base_logp"]) / n,
            "dlogp_ctl": (a["ctl_logp"] - a["base_logp"]) / n,
            "base_cr90": a["base_cr90"] / n,
            "abl_cr90": a["abl_cr90"] / n,
            "ctl_cr90": a["ctl_cr90"] / n,
            "base_top1": a["base_top1"] / n,
            "abl_top1": a["abl_top1"] / n,
            "other_dlogp": (a["abl_other_logp"] - a["base_other_logp"]) / max(a["n_other"], 1),
        }

    def row(r: int) -> dict:
        li, ex, un = int(t["layer"][r]), int(t["expert"][r]), int(t["unit"][r])
        promoted = int(z["s_promoted"][li, un] if ex < 0 else z["promoted"][li, ex, un])
        out = {
            "id": unit_name(li, ex, un),
            "layer": li,
            "expert": ex,
            "unit": un,
            "n": int(t["n"][r]),
            "p": round(float(t["cr90_p"][r]), 4),
            "lb": round(float(t["cr90_lb"][r]), 4),
            "ub": round(float(t["cr90_ub"][r]), 4),
            "p70": round(float(t["cr70_p"][r]), 4),
            "p99": round(float(t["cr99_p"][r]), 4),
            "cw50": round(float(t["cw50_p"][r]), 4),
            "cw50_lb": round(float(t["cw50_lb"][r]), 4),
            "lift": round(float(t["cr90_lift"][r]), 3),
            "xlift": round(float(xl[r]), 3),
            "expert_rate": round(float(t["cr90_routed_p"][r]), 4),
            "cov": float(t["cr90_cov"][r]),
            "held_n": int(h["n"][r]),
            "held_p": round(float(h["cr90_p"][r]), 4),
            "held_lb": round(float(h["cr90_lb"][r]), 4),
            "held_cw50": round(float(h["cw50_p"][r]), 4),
            "share": round(float(t["dla_share"][r]), 4),
            "dla": round(float(t["dla"][r]), 4),
            "is_top": round(float(t["is_top"][r]), 3),
            "promoted": vocab[promoted],
            "cat": ATLAS_CATEGORIES[cat[r]] if cat[r] >= 0 else "shared",
            "thr": round(float(t["thr"][r]), 4),
            "dose": round_list(rates[r], 3),
            "abl": ablation(r),
        }
        j = reg_of_row.get(r)
        if j is not None:
            n_pos = int(rep["ex_n_positions"][j])
            out["n_ex"] = n_pos
            out["types"] = {
                name: round(float(rep["ex_type_counts"][j, k] / max(n_pos, 1)), 3)
                for k, name in enumerate(decider.TYPE_NAMES)
            }
            exs = []
            for w in range(rep["ex_windows"].shape[1]):
                ids = rep["ex_windows"][j, w]
                if ids[-2] < 0:
                    continue
                code = int(rep["ex_window_code"][j, w])
                exs.append(
                    {
                        "ctx": decode(vocab, [int(i) for i in ids[:-1] if i >= 0]),
                        "next": vocab[int(ids[-1])] if ids[-1] >= 0 else "",
                        "act": round(float(rep["ex_window_act"][j, w] / max(t["thr"][r], 1e-9)), 2),
                        "cr90": bool((code >> decider.OUTCOME_BITS.index("cr90")) & 1),
                        "top1": bool(code & 1),
                    }
                )
            out["examples"] = exs[:4]
        return out

    res["top"] = [row(int(r)) for r in order[:TABLE_TOP]]
    early_order = order[layer[order] < EARLY_LAYERS]
    early_order = early_order[lb[early_order] >= 0]
    res["early"] = [row(int(r)) for r in early_order[:TABLE_EARLY]]
    cw_order = np.lexsort((-t["n"], -np.where(ok, t["cw50_lb"], -1)))
    res["cw"] = [row(int(r)) for r in cw_order[:TABLE_CW]]
    res["ablations"] = [
        dict(
            id=unit_name(int(t["layer"][r]), int(t["expert"][r]), int(t["unit"][r])),
            layer=int(t["layer"][r]),
            p=float(t["cr90_p"][r]),
            share=float(t["dla_share"][r]),
            cw=bool(t["cw50_lb"][r] >= 0.15),
            **ablation(int(r)),
        )
        for r in reg_rows
        if ablation(int(r)) is not None
    ]
    res["coverage"] = cov
    res["early_json"] = [row(int(r)) for r in early_order[:EARLY_JSON]]
    return res


def early_deciders(res: dict) -> list[dict]:
    keys = (
        "id",
        "layer",
        "expert",
        "unit",
        "thr",
        "n",
        "p",
        "lb",
        "held_p",
        "held_lb",
        "lift",
        "xlift",
        "cov",
        "share",
        "cat",
        "promoted",
    )
    return [{k: r[k] for k in keys} | {"abl": r["abl"]} for r in res["early_json"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("registry", "page"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--atlas-metrics", help="page: atlas unit_metrics.npz")
    parser.add_argument("--vocab", help="page: vocab.json")
    parser.add_argument("--out", help="page: output directory")
    args = parser.parse_args()
    z = load(os.path.join(args.data, "decider_discover.npz"))
    if args.cmd == "registry":
        reg = build_registry(z)
        with open(os.path.join(args.data, "decider_registry.json"), "w") as f:
            json.dump(reg, f)
        counts = {s: sum((u["bits"] >> j) & 1 for u in reg["units"]) for j, s in enumerate(reg["sets"])}
        print("units", len(reg["units"]), "registry", len(reg["reg_rows"]), counts)
        print("ablate", sum(bool(u.get("ablate")) for u in reg["units"]), "layers", forward.NUM_LAYERS)

    if args.cmd == "page":
        rep = load(os.path.join(args.data, "decider_replicate.npz"))
        with open(os.path.join(args.data, "decider_replicate_coverage.json")) as f:
            cov = json.load(f)
        with open(os.path.join(args.data, "decider_replicate_ablation.json")) as f:
            abl = json.load(f)
        with open(os.path.join(args.data, "decider_registry.json")) as f:
            reg = json.load(f)
        with open(args.vocab) as f:
            vocab = json.load(f)
        res = page_data(z, rep, cov, abl, reg, args.atlas_metrics, vocab)
        early = early_deciders(res)
        del res["early_json"]
        with open(os.path.join(args.out, "page_data.json"), "w") as f:
            json.dump(res, f, default=float)
        with open(os.path.join(args.out, "early_deciders.json"), "w") as f:
            json.dump(early, f, indent=1, default=float)
        print("wrote page_data.json, early_deciders.json")


if __name__ == "__main__":
    main()
