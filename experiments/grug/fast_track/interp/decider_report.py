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
# Registry decider sets (bit j): name -> (outcome stat, min Wilson LB). "random" is a matched random control.
SET_RULES = (
    ("cr90_lb95", "cr90", 0.95),
    ("cr90_lb90", "cr90", 0.90),
    ("cr90_lb80", "cr90", 0.80),
    ("cr90_lb70", "cr90", 0.70),
    ("cr99_lb80", "cr99", 0.80),
    ("cw50_lb50", "cw50", 0.50),
    ("cw50_lb30", "cw50", 0.30),
    ("random", None, None),
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
    for j, (_name, key, lb) in enumerate(SET_RULES):
        if key is not None:
            bits |= (ok & (t[f"{key}_lb"] >= lb)).astype(np.int32) << j
    # Random control: as many supported units per layer as the cr90_lb80 set has.
    ref_set = (bits >> 2) & 1
    for layer in range(L):
        want = int(ref_set[t["layer"] == layer].sum())
        pool = np.flatnonzero(ok & (t["layer"] == layer))
        pick = rng.choice(pool, size=min(want, len(pool)), replace=False)
        bits[pick] |= 1 << (len(SET_RULES) - 1)
    lb = np.where(ok, t["cr90_lb"], -1)
    cw = np.where(ok, t["cw50_lb"], -1)
    order = np.argsort(-lb)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=("registry",))
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    z = load(os.path.join(args.data, "decider_discover.npz"))
    if args.cmd == "registry":
        reg = build_registry(z)
        with open(os.path.join(args.data, "decider_registry.json"), "w") as f:
            json.dump(reg, f)
        counts = {s: sum((u["bits"] >> j) & 1 for u in reg["units"]) for j, s in enumerate(reg["sets"])}
        print("units", len(reg["units"]), "registry", len(reg["reg_rows"]), counts)
        print("ablate", sum(bool(u.get("ablate")) for u in reg["units"]), "layers", forward.NUM_LAYERS)


if __name__ == "__main__":
    main()
