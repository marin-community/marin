# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Research-only HPR contrast estimator with exact tied-policy restriction.

The earlier decoupled-budget audit already used HPR feature differences, but
fitted an unconstrained head with a free intercept. This estimator keeps the
source HPR nonnegative cone and hierarchical ridge, omits the intercept, and
never centers the differenced design. It is an estimator repair, not a new
temporal architecture. Its CLI runs design and synthetic checks only; real-data
use requires a separately registered protocol and a caller supplying its folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hpr,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_two_phase_link_spines_20260907 import (  # noqa: E402
    write_json_atomic,
)

DEFAULT_ROOT = SCRIPT_DIR / "reference_outputs" / "two_phase_link_transfer_20260907"
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "two_phase_hpr_transfer_20260907" / "contrast_synthetic"
NULL_COLUMN_NORM = 1e-14


def candidate_configs(shape_indices: tuple[int, ...] | None = None) -> tuple[hpr.Config, ...]:
    """Return the unchanged source HPR grid, with deterministic conservative ties."""
    shapes = observatory.hierarchical_phase_replay_shape_candidates(observatory.TWO_PHASE)
    indices = list(range(len(shapes))) if shape_indices is None else list(shape_indices)
    configs = hpr.structural_configs(hpr.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY, shapes, indices)
    return tuple(sorted(configs, key=lambda config: (-config.l2, -config.residual_shrink, config.shape_index)))


@dataclass(frozen=True)
class ContrastHead:
    coefficients: np.ndarray
    feature_names: tuple[str, ...]
    active_columns: np.ndarray
    ridge_multipliers: np.ndarray
    l2: float
    objective: float
    kkt_violation: float

    def predict(self, design: hpr.Design) -> np.ndarray:
        assert design.names == self.feature_names, "contrast feature order changed"
        return design.values @ self.coefficients


def tied_weights(dataset: hpr.family_grp.Dataset, weights: np.ndarray) -> np.ndarray:
    """Return each policy's physical exposure-matched tied counterpart."""
    weights = np.asarray(weights, dtype=float)
    assert weights.ndim == 3 and weights.shape[1:] == (2, dataset.m)
    assert np.isfinite(weights).all() and np.min(weights) >= -1e-12
    assert np.max(np.abs(weights.sum(axis=2) - 1.0)) < 1e-10
    rates = dataset.c0 + dataset.c1
    assert np.isfinite(rates).all() and np.min(rates) > 0.0
    aggregate = (dataset.c0 * weights[:, 0] + dataset.c1 * weights[:, 1]) / rates
    assert np.max(np.abs(aggregate.sum(axis=1) - 1.0)) < 1e-10, "physical counterpart leaves the simplex"
    result = np.stack([aggregate, aggregate], axis=1)
    physically_tied = np.all(weights[:, 0] == weights[:, 1], axis=1)
    result[physically_tied] = weights[physically_tied]
    return result


def contrast_design(dataset: hpr.family_grp.Dataset, config: hpr.Config, weights: np.ndarray) -> hpr.Design:
    """Difference the unchanged source features before fitting nonnegative coefficients."""
    candidate = replace(dataset, weights=np.asarray(weights, dtype=float), target=np.zeros(len(weights)))
    tied = replace(candidate, weights=tied_weights(dataset, weights))
    ordinary = hpr.build_design(candidate, config)
    reference = hpr.build_design(tied, config)
    assert ordinary.names == reference.names
    assert np.array_equal(ordinary.ridge_multipliers, reference.ridge_multipliers)
    difference = ordinary.values - reference.values
    assert np.isfinite(difference).all()
    return hpr.Design(difference, ordinary.names, ordinary.ridge_multipliers)


def fit_contrast_head(design: hpr.Design, deltas: np.ndarray, l2: float) -> ContrastHead:
    """Fit signed observed differences in the original HPR nonnegative cone.

    Signed features allow signed predictions without relaxing coefficient signs.
    The ridge uses raw source-feature units. Zero training columns receive zero
    coefficients. No projection or mean subtraction alters the feasible cone.
    """
    values = np.asarray(design.values, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    assert values.ndim == 2 and deltas.shape == (len(values),) and len(values) > 0
    assert np.isfinite(values).all() and np.isfinite(deltas).all()
    assert len(design.names) == values.shape[1] == len(design.ridge_multipliers)
    assert l2 >= 0.0 and np.isfinite(l2)
    active = np.linalg.norm(values, axis=0) > NULL_COLUMN_NORM
    coefficients = np.zeros(values.shape[1])
    if active.any():
        matrix = values[:, active]
        response = deltas
        if l2 > 0.0:
            matrix = np.vstack([matrix, np.diag(np.sqrt(l2 * design.ridge_multipliers[active]))])
            response = np.concatenate([deltas, np.zeros(int(active.sum()))])
        coefficients[active], _ = nnls(matrix, response, maxiter=40 * matrix.shape[1])
    residual = values @ coefficients - deltas
    ridge = l2 * design.ridge_multipliers
    gradient = values.T @ residual + ridge * coefficients
    positive = coefficients > 1e-12
    stationarity = float(np.max(np.abs(gradient[positive]))) if positive.any() else 0.0
    dual_feasibility = float(np.max(np.maximum(-gradient, 0.0)))
    return ContrastHead(
        coefficients,
        design.names,
        active,
        design.ridge_multipliers,
        l2,
        float(residual @ residual + np.sum(ridge * coefficients**2)),
        max(stationarity, dual_feasibility),
    )


def select_contrast_config(
    dataset: hpr.family_grp.Dataset,
    weights: np.ndarray,
    deltas: np.ndarray,
    groups: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    configs: tuple[hpr.Config, ...],
) -> tuple[hpr.Config | None, ContrastHead, list[dict[str, Any]]]:
    """Select on held-pair SSE within caller-supplied aggregate-grouped folds.

    Inputs contain one asymmetric policy and one observed difference per pair.
    Calibration-only groups must be removed by the caller before constructing
    these arrays. Every pair must appear in exactly one test fold; folds with no
    pairs are allowed and skipped. The exact zero head wins score ties, followed
    by candidate order. A selected configuration of None means the zero head.
    """
    assert len(weights) == len(deltas) == len(groups)
    coverage = np.zeros(len(weights), dtype=int)
    for train, test in folds:
        assert len(train) and np.intersect1d(train, test).size == 0
        assert np.intersect1d(groups[train], groups[test]).size == 0, "aggregate group crosses split"
        assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(weights)))
        coverage[test] += 1
    assert np.all(coverage == 1), "each pair must be scored once"
    assert configs, "empty HPR contrast grid"
    assert all(config.variant is hpr.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY for config in configs)
    null_score = float(np.mean(deltas**2))
    best = (null_score, -1)
    sweep = [{"config": None, "paired_mse": null_score, "n_pairs": len(deltas)}]
    cached: dict[tuple[hpr.Variant, hpr.family_grp.Shape, float, float], hpr.Design] = {}
    for index, config in enumerate(configs):
        key = (config.variant, config.shape, config.undercoverage_fraction, config.coverage_gate_ratio)
        if key not in cached:
            cached[key] = contrast_design(dataset, config, weights)
        base = cached[key]
        ridge = np.array(
            [config.residual_shrink if name.startswith("bucket_excess_signal:") else 1.0 for name in base.names]
        )
        design = replace(base, ridge_multipliers=ridge)
        squared_error = 0.0
        count = 0
        for train, test in folds:
            if not len(test):
                continue
            head = fit_contrast_head(replace(design, values=design.values[train]), deltas[train], config.l2)
            residual = head.predict(replace(design, values=design.values[test])) - deltas[test]
            squared_error += float(residual @ residual)
            count += len(test)
        score = (squared_error / count, index)
        sweep.append({"config": asdict(config), "paired_mse": score[0], "n_pairs": count})
        if score < best:
            best = score
    if best[1] == -1:
        design = contrast_design(dataset, configs[0], weights)
        zero = np.zeros(len(design.names))
        head = ContrastHead(
            zero,
            design.names,
            np.zeros(len(zero), dtype=bool),
            design.ridge_multipliers,
            float("inf"),
            float(deltas @ deltas),
            0.0,
        )
        return None, head, sweep
    selected = configs[best[1]]
    design = contrast_design(dataset, selected, weights)
    return selected, fit_contrast_head(design, deltas, selected.l2), sweep


def audit(root: Path) -> dict[str, Any]:
    """Check source parity, the tied null, and constrained synthetic recovery."""
    records = []
    random = np.random.default_rng(20260907)
    for objective in ("uncheatable", "table9"):
        path = root / "controls" / "hierarchical_phase_replay" / objective / "full" / "model.pkl"
        with path.open("rb") as handle:
            source = pickle.load(handle)
        weights = source.dataset.weights
        design = contrast_design(source.dataset, source.config, weights)
        tied = tied_weights(source.dataset, weights)
        null_design = contrast_design(source.dataset, source.config, tied)
        source_difference = source.predict(weights) - source.predict(tied)
        parity = float(np.max(np.abs(design.values @ source.coefficients - source_difference)))
        assert parity < 1e-10
        assert np.max(np.abs(null_design.values)) == 0.0
        use = np.flatnonzero(np.any(weights[:, 0] != weights[:, 1], axis=1))
        synthetic_coefficients = random.uniform(0.0, 0.002, design.values.shape[1])
        synthetic = design.values[use] @ synthetic_coefficients
        train_design = replace(design, values=design.values[use])
        recovered = fit_contrast_head(train_design, synthetic, 0.0)
        recovery_error = float(np.max(np.abs(recovered.predict(train_design) - synthetic)))
        assert recovery_error < 1e-8
        penalized = fit_contrast_head(train_design, synthetic, 0.1)
        assert np.min(penalized.coefficients) >= 0.0
        assert penalized.kkt_violation < 1e-7
        assert np.max(np.abs(penalized.predict(null_design))) == 0.0
        nonsingletons = sum(len(members) > 1 for members in source.dataset.family_members)
        expected_columns = source.dataset.m + 2 * nonsingletons + 2 * len(source.dataset.family_names) + 1
        assert expected_columns == design.values.shape[1]
        records.append(
            {
                "objective": objective,
                "source_model": str(path),
                "n_buckets": source.dataset.m,
                "n_families": len(source.dataset.family_names),
                "n_nonsingleton_families": nonsingletons,
                "n_raw_coefficients": expected_columns,
                "source_prediction_difference_max_error": parity,
                "tied_prediction_max_abs": 0.0,
                "synthetic_in_span_prediction_max_error": recovery_error,
                "synthetic_penalized_kkt_violation": penalized.kkt_violation,
                "training_null_columns": [
                    name for name, active in zip(design.names, recovered.active_columns, strict=True) if not active
                ],
                "real_outcome_fits": 0,
            }
        )
    return {"checks": records, "real_outcome_fits": 0}


def read_policy_inputs(root: Path) -> tuple[hpr.family_grp.Dataset, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read the prepared policy geometry and source family schema without outcomes."""
    path = root / "controls" / "hierarchical_phase_replay" / "uncheatable" / "full" / "model.pkl"
    with path.open("rb") as handle:
        source = pickle.load(handle)
    dataset = replace(source.dataset, target=np.zeros(source.dataset.n))
    with np.load(root / "inputs/panel.npz", allow_pickle=False) as archive:
        keys = ("weights", "runs", "groups", "pair_asymmetric_rows", "pair_tied_rows", "anchor_components")
        panel = {name: archive[name] for name in keys}
    with np.load(root / "inputs/splits.npz", allow_pickle=False) as archive:
        splits = {name: archive[name] for name in archive.files}
    assert np.array_equal(dataset.weights, panel["weights"])
    return dataset, panel, splits


def synthetic_shapes(root: Path) -> tuple[int, ...]:
    """Choose three shapes by source index and outcome-free full-rank geometry."""
    dataset, panel, _ = read_policy_inputs(root)
    selected = []
    seen = set()
    for config in sorted(candidate_configs(), key=lambda config: config.shape_index):
        if config.shape_index in seen:
            continue
        seen.add(config.shape_index)
        values = contrast_design(dataset, config, panel["weights"][panel["pair_asymmetric_rows"]]).values
        norms = np.linalg.norm(values, axis=0)
        scaled = values[:, norms > NULL_COLUMN_NORM] / norms[norms > NULL_COLUMN_NORM]
        if np.linalg.matrix_rank(scaled) == 49:
            selected.append(config.shape_index)
        if len(selected) == 3:
            break
    assert len(selected) == 3
    return tuple(selected)


def synthetic_cell(argument: tuple[str, str, dict[str, Any], str, int, int, int]) -> dict[str, Any]:
    """Fit one synthetic draw using nested source folds and measured local noise."""
    root_string, output_string, identity, objective, shape_index, draw, fold = argument
    root, output = Path(root_string), Path(output_string)
    scenario = "null" if shape_index < 0 else f"shape{shape_index}"
    path = output / "cells" / f"{objective}_{scenario}_draw{draw}_outer{fold}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached["identity"] == identity, "synthetic cache protocol changed"
        return cached
    dataset, panel, splits = read_policy_inputs(root)
    asymmetric, tied = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
    pair_weights = panel["weights"][asymmetric]
    train_mask = np.isin(asymmetric, splits[f"outer{fold}_train"])
    assert np.array_equal(train_mask, np.isin(tied, splits[f"outer{fold}_train"]))
    test_mask = np.isin(asymmetric, splits[f"outer{fold}_test"])
    train, test = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    assert np.array_equal(np.sort(np.concatenate([train, test])), np.arange(len(asymmetric)))
    inner = []
    for index in range(3):
        inner_train = np.flatnonzero(np.isin(asymmetric[train], splits[f"outer{fold}_inner{index}_train"]))
        inner_test = np.flatnonzero(np.isin(asymmetric[train], splits[f"outer{fold}_inner{index}_test"]))
        inner.append((inner_train, inner_test))
    with np.load(root / "noise/noise.npz", allow_pickle=False) as noise:
        assert np.array_equal(noise["components"], panel["anchor_components"])
        macro_noise = noise["centered_outcomes"] @ noise["objective_weights"].T
    random = np.random.default_rng(20260907100 + draw)
    sampled = random.integers(len(macro_noise), size=len(panel["runs"]))
    objective_index = ("uncheatable", "table9").index(objective)
    noise_delta = macro_noise[sampled[asymmetric], objective_index] - macro_noise[sampled[tied], objective_index]
    true_shape = identity["truth_shape_indices"][0] if shape_index < 0 else shape_index
    known_configs = candidate_configs((true_shape,))
    truth_design = contrast_design(dataset, known_configs[0], pair_weights)
    rms = np.sqrt(np.mean(truth_design.values**2, axis=0))
    direction = np.divide(1.0, rms, out=np.zeros_like(rms), where=rms > NULL_COLUMN_NORM)
    raw_truth = truth_design.values @ direction
    desired = (0.0013, 0.0039)[objective_index]
    truth = np.zeros_like(raw_truth) if shape_index < 0 else raw_truth * desired / np.sqrt(np.mean(raw_truth**2))
    observed = truth + noise_delta
    records = []
    for mode, configs in (("known_shape", known_configs), ("selected_shape", candidate_configs())):
        selected, head, sweep = select_contrast_config(
            dataset, pair_weights[train], observed[train], panel["groups"][asymmetric[train]], tuple(inner), configs
        )
        design = contrast_design(dataset, configs[0] if selected is None else selected, pair_weights[test])
        prediction = head.predict(design)
        assert np.isfinite(prediction).all()
        record = {
            "objective": objective,
            "scenario": scenario,
            "true_shape_index": shape_index,
            "draw": draw,
            "fold": fold,
            "mode": mode,
            "n_train_pairs": len(train),
            "n_test_pairs": len(test),
            "selected_config": None if selected is None else asdict(selected),
            "selected_inner_mse": min(float(row["paired_mse"]) for row in sweep),
            "truth_rms_prescribed": 0.0 if shape_index < 0 else desired,
            "pair_indices": test.tolist(),
            "truth": truth[test].tolist(),
            "prediction": prediction.tolist(),
            "coefficients": head.coefficients.tolist(),
            "kkt_violation": head.kkt_violation,
            "real_outcome_fits": 0,
        }
        records.append(record)
    result = {"identity": identity, "records": records, "noise_sample_indices": sampled.tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, result)
    return result


def run_synthetic(root: Path, output: Path, workers: int) -> None:
    """Run the fixed signal/null screen and save pooled held-pair diagnostics."""
    shapes = synthetic_shapes(root)
    paths = [
        Path(__file__),
        Path(hpr.__file__),
        Path(hpr.family_grp.__file__),
        Path(observatory.__file__),
        root / "inputs/panel.npz",
        root / "inputs/splits.npz",
        root / "noise/noise.npz",
    ]
    identity = {
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "truth_shape_indices": list(shapes),
        "truth_coefficients": "positive inverse column RMS; one common multiplier fixes macro pair RMS",
        "noise": "whole centered 11-repeat vectors, independent draw per physical run, shared across objectives",
        "truth_rms": {"uncheatable": 0.0013, "table9": 0.0039},
        "draws": 4,
        "seed": 20260907100,
        "grid": [asdict(config) for config in candidate_configs()],
        "exact_zero_head_included": True,
        "real_outcome_fits": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        assert json.loads((output / "protocol.json").read_text()) == identity, "synthetic protocol changed"
    write_json_atomic(output / "protocol.json", identity)
    arguments = [
        (str(root), str(output), identity, objective, shape, draw, fold)
        for objective in ("uncheatable", "table9")
        for shape in (-1, *shapes)
        for draw in range(4)
        for fold in (0, 2)
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(synthetic_cell, argument) for argument in arguments]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            for record in result["records"]:
                for position, pair in enumerate(record["pair_indices"]):
                    rows.append(
                        {
                            name: record[name]
                            for name in ("objective", "scenario", "draw", "fold", "mode", "truth_rms_prescribed")
                        }
                        | {
                            "pair_index": pair,
                            "truth": record["truth"][position],
                            "prediction": record["prediction"][position],
                            "zero_head": record["selected_config"] is None,
                        }
                    )
            print(f"synthetic {done}/{len(futures)}", flush=True)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "pair_predictions.csv", index=False)
    summaries = []
    for key, group in predictions.groupby(["objective", "scenario", "mode"], sort=True):
        error = group["prediction"].to_numpy() - group["truth"].to_numpy()
        truth = group["truth"].to_numpy()
        prediction = group["prediction"].to_numpy()
        signal = float(np.sqrt(np.mean(truth**2)))
        rmse = float(np.sqrt(np.mean(error**2)))
        summaries.append(
            {
                "objective": key[0],
                "scenario": key[1],
                "mode": key[2],
                "n_predictions": len(group),
                "truth_rms": signal,
                "rmse": rmse,
                "rmse_over_signal": None if signal == 0.0 else rmse / signal,
                "prediction_rms": float(np.sqrt(np.mean(prediction**2))),
                "positive_gain_rms": float(np.sqrt(np.mean(np.maximum(-prediction, 0.0) ** 2))),
                "max_false_gain": float(max(0.0, np.max(truth - prediction))),
                "prediction_max_abs": float(np.max(np.abs(prediction))),
                "zero_head_fraction": float(group["zero_head"].mean()),
            }
        )
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    write_json_atomic(output / "summary.json", {"identity": identity, "summary": summaries, "real_outcome_fits": 0})
    print(json.dumps(summaries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.synthetic:
        run_synthetic(args.root, args.output, args.workers)
        return
    result = audit(args.root)
    payload = json.dumps(result, indent=2) + "\n"
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
