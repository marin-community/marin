# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Exact minimizer of the tuned RegMix surrogate over the simplex, as a mixed-integer program.

RegMix proposes the mean of the best 128 of a million uniform-simplex samples, which need not be where its own
gradient-boosted trees predict the lowest loss (at 3e18 the trees rated MARINER's mixture below their own
proposal). This module refits the tuned trees of ``materialize_delphi_comparator_proposals_20260909`` (registry
entry ``lightgbm_regmix``, every component of an objective on all 280 swarm runs, the heldout-stage inner folds)
and minimizes the objective's aggregate prediction exactly with the split-variable formulation of Mišić (2020):
one binary per feature threshold, one leaf variable per tree leaf, and the mixture as integer counts on the 1/2048
runtime grid (the cells of the threshold grid can be thinner than a grid step, so a continuous optimum rounded to
the grid can leave its cell). The optimum is a cell of the threshold grid; two grid points of it are written, the
one farthest from its faces (``center``) and the one nearest the proportional mixture (``nearest_proportional``).
Predictions of the trees and of MARINER's reference fit at the argmin, at RegMix's own top-k proposal and at
MARINER's policy are recorded for the decision whether to train it.

Progress (fits done, elapsed, projected total, Pacific ETA; then the solver's log) goes to ``progress.log`` in the
output directory.

usage: OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 uv run --offline --no-sync --with lightgbm==4.7.0 python \\
           optimize_regmix_tuned_argmin_20260930.py [--targets uncheatable table9] [--time-limit 1800] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import zoneinfo
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_comparator_proposals_20260909 as proposals,
)

ms = proposals.ms
OUTPUT = SCRIPT_DIR / "reference_outputs" / "regmix_tuned_argmin_3e18_20260930"
MODEL_ID = "lightgbm_regmix"
TOPK_CANDIDATES = {"uncheatable": "cmp_u_lgbm_cap08", "table9": "cmp_t9_lgbm_cap12"}
CANDIDATE_KEYS = {"uncheatable": "rgx_u_argmin", "table9": "rgx_t9_argmin"}
THRESHOLD_DECIMALS = 12
STRICT_MARGIN = 1e-9
PACIFIC = zoneinfo.ZoneInfo("America/Los_Angeles")


def log(path: Path, message: str) -> None:
    stamp = datetime.datetime.now(PACIFIC).strftime("%H:%M:%S")
    with path.open("a") as handle:
        handle.write(f"{stamp} {message}\n")
    print(message, flush=True)


def leaves_of(node: dict, path: list[tuple[int, float, bool]], out: list[dict]) -> None:
    """Flatten a LightGBM tree: every leaf with its value and the (feature, threshold, went_left) path."""
    if "leaf_value" in node:
        out.append({"value": float(node["leaf_value"]), "path": list(path)})
        return
    if node.get("decision_type") != "<=":
        raise ValueError(f"unsupported decision type {node.get('decision_type')}")
    feature, threshold = int(node["split_feature"]), float(node["threshold"])
    leaves_of(node["left_child"], [*path, (feature, threshold, True)], out)
    leaves_of(node["right_child"], [*path, (feature, threshold, False)], out)


def ensemble_terms(fits: list, weights: np.ndarray) -> tuple[list[dict], float]:
    """Every leaf of every tree of every component, with its objective coefficient, thresholds in weight space."""
    terms: list[dict] = []
    constant = 0.0
    for aggregation, fitted in zip(weights, fits, strict=True):
        head = fitted.head
        scale = float(aggregation * head.y_scale)
        constant += float(aggregation * head.y_mean)
        dump = head.estimator.booster_.dump_model()
        for tree in dump["tree_info"]:
            leaves: list[dict] = []
            leaves_of(tree["tree_structure"], [], leaves)
            for leaf in leaves:
                leaf["coefficient"] = scale * leaf["value"]
                # Splits are on standardized weights; move them to the weight simplex.
                leaf["path"] = [
                    (
                        feature,
                        round(float(head.x_mean[feature] + head.x_scale[feature] * threshold), THRESHOLD_DECIMALS),
                        left,
                    )
                    for feature, threshold, left in leaf["path"]
                ]
            terms.append({"leaves": leaves})
    return terms, constant


def solve_argmin(terms: list[dict], buckets: int, time_limit: float, log_path: Path) -> dict:
    """Mišić's split-variable program: binaries x[j,k] = [w_j <= v_jk], leaf variables per tree, simplex on w."""
    thresholds: dict[int, list[float]] = {}
    for tree in terms:
        for leaf in tree["leaves"]:
            for feature, threshold, _ in leaf["path"]:
                thresholds.setdefault(feature, []).append(threshold)
    grid = {feature: sorted(set(values)) for feature, values in thresholds.items()}
    x_index: dict[tuple[int, float], int] = {}
    for feature, values in sorted(grid.items()):
        for value in values:
            x_index[(feature, value)] = buckets + len(x_index)
    n_x = len(x_index)
    leaf_offset = buckets + n_x
    n_leaves = sum(len(tree["leaves"]) for tree in terms)
    n_var = leaf_offset + n_leaves
    cost = np.zeros(n_var)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    def add(entries: list[tuple[int, float]], lo: float, hi: float) -> None:
        nonlocal row
        for col, val in entries:
            rows.append(row)
            cols.append(col)
            vals.append(val)
        lower.append(lo)
        upper.append(hi)
        row += 1

    block = ms.MIXTURE_BLOCK_SIZE
    add([(j, 1.0) for j in range(buckets)], block, block)  # counts sum to the block size
    for feature, values in grid.items():
        for k, value in enumerate(values):
            xi = x_index[(feature, value)]
            if k + 1 < len(values):  # w_j <= v_k implies w_j <= v_{k+1}
                add([(xi, 1.0), (x_index[(feature, values[k + 1])], -1.0)], -np.inf, 0.0)
            below = int(np.floor(value * block + STRICT_MARGIN))  # largest count with count/block <= v
            add([(feature, 1.0), (xi, block - below)], -np.inf, block)  # x=1: count <= below
            add([(feature, 1.0), (xi, below + 1)], below + 1, np.inf)  # x=0: count >= below + 1
    leaf = leaf_offset
    for tree in terms:
        first = leaf
        node_sides: dict[tuple[int, float, bool], list[int]] = {}
        for item in tree["leaves"]:
            cost[leaf] = item["coefficient"]
            for feature, threshold, left in item["path"]:
                node_sides.setdefault((feature, threshold, left), []).append(leaf)
            leaf += 1
        add([(col, 1.0) for col in range(first, leaf)], 1.0, 1.0)
        for (feature, threshold, left), members in node_sides.items():
            xi = x_index[(feature, threshold)]
            if left:  # leaves left of the split need x = 1
                add([*((col, 1.0) for col in members), (xi, -1.0)], -np.inf, 0.0)
            else:  # leaves right of the split need x = 0
                add([*((col, 1.0) for col in members), (xi, 1.0)], -np.inf, 1.0)
    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(row, n_var))
    integrality = np.zeros(n_var)
    integrality[: buckets + n_x] = 1
    upper_bounds = np.ones(n_var)
    upper_bounds[:buckets] = block
    bounds = Bounds(np.zeros(n_var), upper_bounds)
    log(log_path, f"MILP: {buckets} weights, {n_x} threshold binaries, {n_leaves} leaves, {row} constraints")
    started = time.monotonic()
    result = milp(
        cost,
        constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
        integrality=integrality,
        bounds=bounds,
        options={"disp": True, "time_limit": time_limit, "mip_rel_gap": 1e-6},
    )
    seconds = time.monotonic() - started
    if result.x is None:
        raise RuntimeError(f"MILP returned no solution: {result.message}")
    # The optimal cell as count bounds: count_low < count <= count_high on the grid.
    count_low = np.full(buckets, -1)
    count_high = np.full(buckets, block)
    for (feature, value), xi in x_index.items():
        below = int(np.floor(value * block + STRICT_MARGIN))
        if result.x[xi] > 0.5:
            count_high[feature] = min(count_high[feature], below)
        else:
            count_low[feature] = max(count_low[feature], below)
    return {
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "bound": float(getattr(result, "mip_dual_bound", np.nan)),
        "gap": float(getattr(result, "mip_gap", np.nan)),
        "seconds": seconds,
        "thresholds": n_x,
        "leaves": n_leaves,
        "constraints": row,
        "count_low": count_low,
        "count_high": count_high,
        "counts_milp": np.rint(result.x[:buckets]).astype(int),
    }


def cell_points(count_low: np.ndarray, count_high: np.ndarray, natural: np.ndarray) -> dict[str, np.ndarray]:
    """Two grid points of the optimal cell: the most interior (max-min slack) and the nearest to proportional."""
    n = len(count_low)
    block = ms.MIXTURE_BLOCK_SIZE
    lo = count_low + 1
    hi = count_high
    # Center: maximize t with lo + t <= c <= hi - t, sum c = block, c integer.
    cost = np.zeros(n + 1)
    cost[-1] = -1.0
    rows = []
    for j in range(n):
        left = np.zeros(n + 1)
        left[j], left[-1] = -1.0, 1.0
        rows.append((left, -np.inf, -lo[j]))
        right = np.zeros(n + 1)
        right[j], right[-1] = 1.0, 1.0
        rows.append((right, -np.inf, hi[j]))
    total = np.zeros(n + 1)
    total[:n] = 1.0
    rows.append((total, block, block))
    matrix = np.vstack([r[0] for r in rows])
    center = milp(
        cost,
        constraints=LinearConstraint(matrix, np.array([r[1] for r in rows]), np.array([r[2] for r in rows])),
        integrality=np.concatenate([np.ones(n), [0]]),
        bounds=Bounds(np.zeros(n + 1), np.concatenate([np.full(n, block), [block]])),
    )
    if center.x is None:
        raise RuntimeError(f"cell center program failed: {center.message}")
    # Nearest to proportional in L1: c = p + u - v with u, v >= 0, minimize sum(u + v), c integer in the cell.
    target = natural * block
    cost2 = np.concatenate([np.zeros(n), np.ones(2 * n)])
    a_eq = np.zeros((1 + n, 3 * n))
    a_eq[0, :n] = 1.0
    for j in range(n):
        a_eq[1 + j, j], a_eq[1 + j, n + j], a_eq[1 + j, 2 * n + j] = 1.0, -1.0, 1.0
    b_eq = np.concatenate([[block], target])
    nearest = milp(
        cost2,
        constraints=LinearConstraint(a_eq, b_eq, b_eq),
        integrality=np.concatenate([np.ones(n), np.zeros(2 * n)]),
        bounds=Bounds(np.concatenate([lo, np.zeros(2 * n)]), np.concatenate([hi, np.full(2 * n, np.inf)])),
    )
    if nearest.x is None:
        raise RuntimeError(f"nearest-proportional program failed: {nearest.message}")
    return {"center": np.rint(center.x[:n]).astype(int), "nearest_proportional": np.rint(nearest.x[:n]).astype(int)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", nargs="+", default=["uncheatable", "table9"])
    parser.add_argument("--time-limit", type=float, default=1800.0, help="solver seconds per target")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.output_dir / "progress.log"
    panel = bench.load_panel(proposals.PANEL)
    buckets = tuple(str(bucket) for bucket in panel.buckets)
    inventory = panel.features.inventory
    natural = 1.0 / inventory
    natural = natural / natural.sum()
    inner = bench.heldout_inner_folds(panel)
    swarm, to_panel = proposals.reference_swarm_in_panel_order(panel)
    from_panel = np.argsort(to_panel)
    launcher_order = [buckets.index(domain) for domain in proposals.base.DOMAIN_NAMES]
    earlier = pd.read_csv(proposals.OUTPUT / "solutions.csv")
    earlier_fits = json.loads((proposals.OUTPUT / "fits.json").read_text())
    total_components = sum(len(panel.group(t).components) for t in args.targets)
    done = 0
    fit_started = time.monotonic()
    summary: dict[str, dict] = {}
    solutions = []
    rows: list[dict] = []
    fit_records: dict[str, list[dict]] = {}
    for target in args.targets:
        group = panel.group(target)
        fits = []
        for index in range(len(group.components)):
            fits.append(proposals.fit_component(panel, MODEL_ID, target, index, inner))
            done += 1
            elapsed = time.monotonic() - fit_started
            projected = elapsed / done * total_components
            eta = datetime.datetime.now(PACIFIC) + datetime.timedelta(seconds=projected - elapsed)
            log(
                progress,
                f"fits {done}/{total_components} elapsed {elapsed:.0f}s projected {projected:.0f}s ETA {eta:%H:%M} PDT",
            )
        key = f"cmp_{proposals.TARGET_KEYS[target]}_lgbm"
        shapes = [(int(f.shape["n_estimators"]), int(f.shape["num_leaves"])) for f in fits]
        earlier_shapes = [(int(r["shape"]["n_estimators"]), int(r["shape"]["num_leaves"])) for r in earlier_fits[key]]
        reproduced = shapes == earlier_shapes
        log(progress, f"{target}: inner-CV shapes {'reproduce' if reproduced else 'DIFFER FROM'} the 2026-09-09 fits")
        fit_records[target] = [
            {
                "component": str(c),
                "n_estimators": s[0],
                "num_leaves": s[1],
                "inner_cv_rmse": float(f.diagnostics["inner_cv_rmse"]),
            }
            for c, s, f in zip(group.components, shapes, fits, strict=True)
        ]
        surrogate = proposals.ObservatorySurrogate(panel, MODEL_ID, target, fits)
        terms, constant = ensemble_terms(fits, np.asarray(group.aggregation_weights, dtype=float))
        solved = solve_argmin(terms, len(buckets), args.time_limit, progress)
        log(
            progress,
            f"{target}: MILP {solved['message']} objective {solved['objective'] + constant:.6f} "
            f"in {solved['seconds']:.0f}s",
        )
        points = cell_points(solved["count_low"], solved["count_high"], natural)
        mariner = proposals.reference_fit(swarm, target, None)
        mariner_predict = lambda w, fit=mariner: fit.predict(np.atleast_2d(w)[:, from_panel])  # noqa: E731
        mariner_policy = proposals.reference_policy(proposals.MARINER_POLICY[target], buckets)
        topk_id = TOPK_CANDIDATES[target]
        topk = (
            earlier[earlier.candidate_id.eq(topk_id)].set_index("bucket").loc[list(buckets), "runtime"].to_numpy(float)
        )
        record: dict = {
            "target": target,
            "model_id": MODEL_ID,
            "fits_reproduce_20260909": reproduced,
            "milp": {k: v for k, v in solved.items() if k not in ("count_low", "count_high", "counts_milp")},
            "milp_objective_with_constant": float(solved["objective"] + constant),
            "cell_count_low_exclusive": dict(zip(buckets, map(int, solved["count_low"]), strict=True)),
            "cell_count_high": dict(zip(buckets, map(int, solved["count_high"]), strict=True)),
            "predictions": {},
        }
        grid_points = {**points, "milp_vertex": solved["counts_milp"]}
        candidates = {
            **{name: counts / ms.MIXTURE_BLOCK_SIZE for name, counts in grid_points.items()},
            "topk_proposal": topk,
            "mariner_policy": mariner_policy,
        }
        for name, w in candidates.items():
            counts = ms.constrained_counts(np.asarray(w, dtype=float), np.full(len(buckets), ms.MIXTURE_BLOCK_SIZE))
            if name in grid_points and not np.array_equal(counts, grid_points[name]):
                raise ValueError(f"{name}: grid point changed under constrained_counts")
            runtime = counts / ms.MIXTURE_BLOCK_SIZE
            record["predictions"][name] = {
                "trees_continuous": float(surrogate.predict(np.asarray(w, dtype=float)[None])[0]),
                "trees_runtime": float(surrogate.predict(runtime[None])[0]),
                "mariner_runtime": float(mariner_predict(runtime[None])[0]),
                **proposals.summarize(runtime, inventory, natural),
                "tv_to_mariner": float(np.abs(runtime - mariner_policy).sum() / 2),
                "tv_to_topk": float(np.abs(runtime - topk).sum() / 2),
            }
            if name in points:
                cap = proposals.nominal_cap(float((runtime * inventory).max()))
                candidate_id = f"{CANDIDATE_KEYS[target]}_{name}_cap{cap:02d}"
                record["predictions"][name]["candidate_id"] = candidate_id
                solutions.append(
                    pd.DataFrame({"candidate_id": candidate_id, "bucket": buckets, "continuous": w, "runtime": runtime})
                )
                for index in launcher_order:
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "target": target,
                            "target_label": proposals.TARGET_LABELS[target],
                            "epoch_cap": cap,
                            "surrogate": MODEL_ID,
                            "domain": buckets[index],
                            "runtime_count": int(counts[index]),
                            "weight": float(runtime[index]),
                            "materialized_epochs": float(runtime[index] * inventory[index]),
                        }
                    )
        summary[target] = record
        log(
            progress,
            f"{target}: predictions "
            + json.dumps({k: round(v["trees_runtime"], 5) for k, v in record["predictions"].items()}),
        )
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.concat(solutions).to_csv(args.output_dir / "solutions.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "candidate_weights.csv", index=False)
    (args.output_dir / "fits.json").write_text(json.dumps(fit_records, indent=1))
    log(progress, "DONE")


if __name__ == "__main__":
    main()
