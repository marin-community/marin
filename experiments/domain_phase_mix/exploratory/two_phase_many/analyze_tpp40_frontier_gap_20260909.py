# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0", "matplotlib==3.10.8",
# ]
# ///
"""Audit a partial TPP40 swarm's HPR frontier and conditional phase advantage."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import audit_two_phase_link_softmax_solver_20260907 as softmax_solver
import benchmark_two_phase_link_controls_20260907 as controls
import numpy as np
import optimize_two_phase_hpr_transfer_20260907 as hpr_solver
import optimize_two_phase_link_transfer_20260907 as raw_solver
import pandas as pd
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent / "reference_outputs"
OUTPUT = BASE / "tpp40_frontier_gap_20260909"
OLD = BASE / "two_phase_link_transfer_20260907"
HPR = controls.baseline.hierarchical_grp
SEED = 20260909
ALPHA = 21856 / 27336
CONTEXTS = ("full", "outer0", "outer1", "outer2", "east5", "europe")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity() -> dict[str, str]:
    paths = [
        Path(__file__),
        OUTPUT / "snapshot.json",
        OUTPUT / "PROTOCOL.md",
        OUTPUT / "COVERAGE_ADDENDUM.md",
        Path(HPR.__file__),
        Path(HPR.family_grp.__file__),
        Path(controls.baseline.observatory.__file__),
        Path(hpr_solver.__file__),
        Path(raw_solver.__file__),
        Path(softmax_solver.__file__),
    ]
    return {str(path): sha(path) for path in paths}


def data() -> tuple[Any, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(OUTPUT / "inputs/outcomes.csv")
    specs = json.loads((OUTPUT / "inputs/run_specs.json").read_text())
    buckets = pd.read_csv(OUTPUT / "inputs/buckets.csv")
    domains = tuple(buckets.bucket)
    weights = np.asarray(
        [
            [[specs[order]["phase_weights"][phase][name] for name in domains] for phase in ("phase_0", "phase_1")]
            for order in frame.order
        ]
    )
    weights /= weights.sum(axis=2, keepdims=True)
    inventory = buckets.epochs_per_unit_weight.to_numpy(float)
    objectives = pd.read_csv(OUTPUT / "inputs/objectives.csv")
    objective = objectives[objectives.objective.eq("uncheatable")]
    outcome = frame[objective.component].to_numpy() @ objective.weight.to_numpy()
    error = np.max(np.abs(outcome - frame["eval/uncheatable_eval/bpb"].to_numpy()))
    assert error < 3e-6, ("component aggregation mismatch", error)
    frame["measured"] = outcome
    with (OLD / "controls/hierarchical_phase_replay/uncheatable/full/model.pkl").open("rb") as handle:
        template = pickle.load(handle).dataset
    assert domains == template.domains, "bucket order changed"
    assert np.max(np.abs(inventory - template.c0 - template.c1)) < 1e-7
    dataset = replace(
        template, frame=frame, target=outcome, weights=weights, c0=ALPHA * inventory, c1=(1 - ALPHA) * inventory
    )
    aggregate = ALPHA * weights[:, 0] + (1 - ALPHA) * weights[:, 1]
    natural = 1 / inventory
    natural /= natural.sum()
    calibration = np.max(np.abs(aggregate - natural), axis=1) < 1e-10
    assert calibration.sum() == 1 and frame.loc[calibration, "order"].item() == 0
    return dataset, frame, aggregate, calibration, inventory


def labels_for(aggregate: np.ndarray, calibration: np.ndarray, seed: int) -> np.ndarray:
    coordinates, inverse = np.unique(np.round(aggregate, 10), axis=0, return_inverse=True)
    pinned = np.unique(inverse[calibration])
    use = ~np.isin(np.arange(len(coordinates)), pinned)
    labels = np.full(len(coordinates), -1, dtype=int)
    labels[use] = KMeans(n_clusters=3, random_state=seed, n_init=20).fit_predict(np.sqrt(coordinates[use]))
    result = labels[inverse]
    assert np.array_equal(result == -1, calibration), "unhandled calibration aliases"
    assert set(result) == {-1, 0, 1, 2}
    return result


def training_rows(context: str, frame: pd.DataFrame, outer: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    if context == "full":
        return np.arange(len(frame))
    if context.startswith("outer"):
        return np.flatnonzero(outer != int(context[-1]))
    return np.flatnonzero(frame.region.eq(context).to_numpy() | calibration)


def fitted_hpr(dataset: Any, labels: np.ndarray) -> tuple[Any, dict]:
    folds = tuple((np.flatnonzero(labels != fold), np.flatnonzero(labels == fold)) for fold in range(3))
    covered = labels >= 0
    assert covered.sum() == dataset.n - 1
    shapes = controls.baseline.observatory.hierarchical_phase_replay_shape_candidates(
        controls.baseline.observatory.TWO_PHASE
    )

    def select(configs: list) -> tuple[Any, list]:
        scores = []
        for config in configs:
            prediction = np.full(dataset.n, np.nan)
            for train, test in folds:
                fit = HPR.fit_model(dataset, config, train)
                prediction[test] = fit.predict(dataset.weights[test])
            assert np.isfinite(prediction[covered]).all()
            metrics = HPR.metric_summary(dataset.target[covered], prediction[covered])
            scores.append({"config": asdict(config), **metrics})
        best = min(range(len(scores)), key=lambda k: (scores[k]["rmse"], -scores[k]["spearman"], k))
        return configs[best], scores

    _, screen = select(HPR.baseline_configs(shapes))
    by_shape = {}
    for score in screen:
        k = score["config"]["shape_index"]
        by_shape[k] = min(by_shape.get(k, float("inf")), score["rmse"])
    indices = [k for k, _ in sorted(by_shape.items(), key=lambda row: row[1])[:3]]
    config, sweep = select(HPR.structural_configs(HPR.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY, shapes, indices))
    fit = HPR.fit_model(dataset, config, np.arange(dataset.n))
    return fit, {"screen": screen, "sweep": sweep, "selected": asdict(config), "inner_labels": labels.tolist()}


def fit_context(context: str) -> None:
    with threadpool_limits(limits=1):
        destination = OUTPUT / "fits" / context
        protocol = identity()
        if (destination / "complete.json").exists():
            record = json.loads((destination / "complete.json").read_text())
            assert record["identity"] == protocol, "fit source changed; choose a new output root"
            for name, digest in record["sha256"].items():
                assert sha(destination / name) == digest
            print(f"reuse fit {context}", flush=True)
            return
        dataset, frame, aggregate, calibration, _ = data()
        outer = labels_for(aggregate, calibration, SEED)
        train = training_rows(context, frame, outer, calibration)
        local = replace(
            dataset,
            frame=frame.iloc[train].reset_index(drop=True),
            target=dataset.target[train],
            weights=dataset.weights[train],
        )
        labels = labels_for(aggregate[train], calibration[train], SEED + 1)
        started = time.monotonic()
        print(f"fit {context}: {len(train)} rows; inner sizes {np.bincount(labels[labels >= 0]).tolist()}", flush=True)
        fitted, selection = fitted_hpr(local, labels)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.pkl").write_bytes(pickle.dumps(fitted))
        write_json_atomic(destination / "selection.json", selection)
        prediction = fitted.predict(dataset.weights)
        tied = fitted.predict(np.repeat(aggregate[:, None, :], 2, axis=1))
        table = frame[["order", "run_name", "region", "tied", "measured"]].copy()
        table["prediction"], table["tied_prediction"] = prediction, tied
        table["outer_fold"], table["in_train"] = outer, np.isin(np.arange(len(frame)), train)
        table.to_csv(destination / "predictions.csv", index=False)
        write_json_atomic(
            destination / "complete.json",
            {
                "identity": protocol,
                "train_rows": train.tolist(),
                "outer_labels": outer.tolist(),
                "seconds": time.monotonic() - started,
                "sha256": {name: sha(destination / name) for name in ("model.pkl", "selection.json", "predictions.csv")},
            },
        )
        print(f"finished {context}: {time.monotonic() - started:.1f} seconds", flush=True)


@dataclass(frozen=True)
class HprSurface:
    response: hpr_solver.HprResponse

    @property
    def buckets(self) -> int:
        return len(self.response.c0)

    @property
    def phase_fraction(self) -> np.ndarray:
        return self.response.c0 / (self.response.c0 + self.response.c1)

    def value_gradient(self, weights: np.ndarray) -> tuple[float, np.ndarray, dict]:
        return self.response.value_gradient(weights)


def optimize_context(context: str) -> None:
    with threadpool_limits(limits=1):
        path = OUTPUT / "optima" / f"{context}.json"
        source = OUTPUT / "fits" / context / "model.pkl"
        protocol = {**identity(), str(source): sha(source)}
        if path.exists():
            assert json.loads(path.read_text())["identity"] == protocol
            print(f"reuse optima {context}", flush=True)
            return
        dataset, frame, aggregate, calibration, inventory = data()
        outer = labels_for(aggregate, calibration, SEED)
        train = training_rows(context, frame, outer, calibration)
        with source.open("rb") as handle:
            fitted = pickle.load(handle)
        surface = HprSurface(hpr_solver.hpr_response(fitted))
        parity = max(
            abs(surface.value_gradient(w)[0] - y)
            for w, y in zip(dataset.weights, fitted.predict(dataset.weights), strict=True)
        )
        assert parity < 1e-10, ("analytic HPR parity failed", parity)
        # Check directional derivatives in the interior, including the tied restriction.
        rng = np.random.default_rng(SEED)
        derivative_errors = []
        for tied in (False, True):
            w = rng.dirichlet(np.ones(dataset.m), size=2)
            direction = rng.normal(size=w.shape)
            direction -= direction.mean(axis=1, keepdims=True)
            if tied:
                w[1], direction[1] = w[0], direction[0]
            epsilon = 1e-8
            numeric = (
                surface.value_gradient(w + epsilon * direction)[0] - surface.value_gradient(w - epsilon * direction)[0]
            ) / (2 * epsilon)
            analytic = np.sum(surface.value_gradient(w)[1] * direction)
            derivative_errors.append(abs(float(numeric - analytic)))
        assert max(derivative_errors) < 1e-5
        results = {}
        for mode in ("tied", "two_phase"):
            tied = mode == "tied"
            cloud = np.repeat(aggregate[train, None, :], 2, axis=1) if tied else dataset.weights[train]
            good = np.argsort(fitted.predict(cloud))[:3]
            random = rng.choice(len(cloud), 4, replace=False)
            natural = np.repeat(dataset.weights[calibration][0, 0][None], 2, axis=0)
            starts = [natural, *cloud[good], *cloud[random]]
            if not tied:
                starts.append(np.asarray(results["tied"]["weights"]))
            records = []
            for index, start in enumerate(starts):
                _, record = raw_solver.solve_start(surface, start, tied, f"start{index}")
                records.append(record)
            best = min(records, key=lambda row: row["retained_bpb"])
            soft = softmax_solver.optimize_start(surface, np.asarray(best["weights"]), tied, "best_slsqp")
            soft_better = soft["predicted_bpb"] < best["retained_bpb"]
            weights = np.asarray(soft["weights"] if soft_better else best["weights"])
            prediction = surface.value_gradient(weights)[0]
            a = ALPHA * weights[0] + (1 - ALPHA) * weights[1]
            support = raw_solver.support_audit(weights, dataset.weights[train], ALPHA)
            tied_support = raw_solver.support_audit(weights, dataset.weights[train][frame.tied.to_numpy()[train]], ALPHA)
            results[mode] = {
                "prediction": prediction,
                "weights": weights.tolist(),
                "phase_tv": float(np.abs(weights[0] - weights[1]).sum() / 2),
                "max_epochs": float(np.max(a * inventory)),
                "tied_at_selected_aggregate": surface.value_gradient(np.stack([a, a]))[0],
                "support": support,
                "tied_training_support": tied_support,
                "selected_success": soft["success"] if soft_better else best["success"],
                "successful_slsqp_starts": sum(row["success"] for row in records),
                "slsqp_starts": records,
                "softmax_check": soft,
            }
        gap = results["tied"]["prediction"] - results["two_phase"]["prediction"]
        assert gap >= -1e-8, "two-phase solve is worse despite a tied feasible start"
        write_json_atomic(
            path,
            {
                "context": context,
                "identity": protocol,
                "results": results,
                "gap": gap,
                "value_parity_error": parity,
                "directional_derivative_errors": derivative_errors,
            },
        )
        print(
            f"optima {context}: tied {results['tied']['prediction']:.6f}, "
            f"2p {results['two_phase']['prediction']:.6f}, gap {gap:.6f}",
            flush=True,
        )


def metrics(frame: pd.DataFrame) -> dict:
    residual = frame.measured.to_numpy() - frame.prediction.to_numpy()
    best = int(np.argmin(frame.prediction.to_numpy()))
    measured = frame.measured.to_numpy()
    return {
        "n": len(frame),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "optimism": float(np.mean(residual)),
        "median_optimism": float(np.median(residual)),
        "spearman": float(spearmanr(measured, frame.prediction).statistic) if len(frame) > 2 else None,
        "selected_optimism": float(residual[best]),
        "selected_regret": float(measured[best] - measured.min()),
        "selected_run": frame.iloc[best].run_name,
        "predicted_min": float(frame.prediction.min()),
        "measured_min": float(frame.measured.min()),
    }


def frontier_tables(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        head = dict(zip(keys, values, strict=True))
        for population, chosen in (
            ("all", group),
            ("predicted_best20pct", group.nsmallest(max(1, int(np.ceil(len(group) * 0.2))), "prediction")),
            ("measured_best20pct_descriptive", group.nsmallest(max(1, int(np.ceil(len(group) * 0.2))), "measured")),
        ):
            rows.append({**head, "population": population, **metrics(chosen)})
    return pd.DataFrame(rows)


def old_frontier() -> None:
    old = pd.read_csv(BASE / "two_phase_hpr_transfer_20260907/aggregate_replacement/predictions.csv")
    old = old[old.context.ne("final") & old.scored & old.objective.eq("uncheatable")].copy()
    old = old.rename(columns={"predicted": "prediction", "run": "run_name"})
    old["policy_class"] = np.where(old.tied, "tied", "asymmetric")
    table = frontier_tables(old, ["model", "policy_class", "context"])
    table.to_csv(OUTPUT / "previous_520_frontier_metrics.csv", index=False)
    primary = old[(old.model.eq("aggregate") & old.tied) | (old.model.eq("hpr_aggregate_replacement") & ~old.tied)]
    chosen = (
        primary.groupby(["model", "policy_class", "context"], group_keys=False)
        .apply(lambda group: group.nsmallest(max(1, int(np.ceil(len(group) * 0.2))), "prediction"), include_groups=False)
        .reset_index(drop=True)
    )
    chosen.to_csv(OUTPUT / "previous_520_frontier_rows.csv", index=False)
    source = OUTPUT / "inputs/mariner.py"
    previous = OLD / "inputs/single_phase.py"
    trees = [ast.parse(path.read_text()) for path in (source, previous)]
    before, after = [
        {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        for tree in trees
    ]
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    assert set(changed) <= {"continuous_optimum", "runtime_policy", "bucket_upper_bounds", "self_test", "main"}
    # Reconstruct all saved seven-task fits with the latest prediction code.
    current = load_module(source, "tpp40_mariner_snapshot")
    with np.load(OLD / "inputs/panel.npz", allow_pickle=False) as panel:
        component_names = panel["uncheatable_components"].tolist()
        saved = pd.read_csv(BASE / "two_phase_hpr_transfer_20260907/aggregate_replacement/predictions.csv")
        replay_errors = {}
        for context in ("final", "outer0", "outer1", "outer2"):
            tasks = tuple(
                current.TaskFit.from_json(
                    json.loads((OLD / "spines" / context / f"uncheatable_c{index}.json").read_text())
                )
                for index in range(7)
            )
            model = current.ObjectiveFit(
                "uncheatable",
                tuple(panel["buckets"]),
                panel["inventory"],
                panel["uncheatable_aggregation_weights"],
                tasks,
            )
            reference = saved[
                saved.context.eq(context) & saved.model.eq("aggregate") & saved.objective.eq("uncheatable")
            ].sort_values("row")
            error = float(np.max(np.abs(model.predict(panel["aggregate"]) - reference.predicted.to_numpy())))
            assert error < 1e-12
            replay_errors[context] = error
    write_json_atomic(
        OUTPUT / "mariner_source_parity.json",
        {
            "current_sha256": sha(source),
            "previous_sha256": sha(previous),
            "changed_functions": changed,
            "fit_and_prediction_definitions_identical": True,
            "source_changes": (
                "The simplex optimizer now defaults to no epoch cap; fitted law and selection are unchanged."
            ),
            "component_names": component_names,
            "module_loaded": current.__name__,
            "prediction_replay_max_errors": replay_errors,
        },
    )


def summarize() -> None:
    old_frontier()
    parts = []
    optima = []
    for context in CONTEXTS:
        p = OUTPUT / "fits" / context / "predictions.csv"
        if p.exists():
            frame = pd.read_csv(p)
            if context.startswith("outer"):
                frame = frame[frame.outer_fold.eq(int(context[-1]))].copy()
                frame["context"] = context
                parts.append(frame)
        p = OUTPUT / "optima" / f"{context}.json"
        if p.exists():
            result = json.loads(p.read_text())
            for mode, record in result["results"].items():
                optima.append(
                    {
                        "context": context,
                        "mode": mode,
                        "predicted_bpb": record["prediction"],
                        "gap": result["gap"],
                        "max_epochs": record["max_epochs"],
                        "phase_tv": record["phase_tv"],
                        "selected_success": record["selected_success"],
                        **record["support"],
                    }
                )
    if not parts:
        return
    oof = pd.concat(parts, ignore_index=True)
    assert oof.order.nunique() == len(oof) == 157
    oof["policy_class"] = np.where(oof.tied, "tied", "asymmetric")
    oof.to_csv(OUTPUT / "tpp40_oof_predictions.csv", index=False)
    frontier = frontier_tables(oof, ["policy_class", "context"])
    frontier.to_csv(OUTPUT / "tpp40_frontier_metrics.csv", index=False)
    opt = pd.DataFrame(optima)
    opt.to_csv(OUTPUT / "optimum_summary.csv", index=False)
    print(frontier[frontier.population.eq("predicted_best20pct")].to_string(index=False))
    print(opt.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "optimize", "summarize"))
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=list(CONTEXTS))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "summarize":
        summarize()
        return
    function = fit_context if args.stage == "fit" else optimize_context
    if args.workers == 1:
        for context in args.contexts:
            function(context)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(function, args.contexts))


if __name__ == "__main__":
    main()
