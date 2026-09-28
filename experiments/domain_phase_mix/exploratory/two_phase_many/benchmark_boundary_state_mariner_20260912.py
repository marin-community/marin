# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "threadpoolctl==3.6.0", "tabulate==0.9.0",
# ]
# ///
"""Compare policy, predicted boundary, and feedback features on local paired gains."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REFERENCES = HERE / "reference_outputs"
DEFAULT_OUTPUT = REFERENCES / "two_phase_mariner_transfer_20260912/boundary"
ARCHIVE = Path("/Users/calvinxu/Projects/Work/Marin/worktree-archives/20260905-dirty-worktrees")
BOUNDARY_SOURCE = (
    ARCHIVE
    / "marin-delphi-phase0-bitwise/files/experiments/domain_phase_mix/exploratory/two_phase_many"
    / "reference_outputs/delphi_3e18_phase0_prefix_replay_20260820/materialized_boundary_metrics"
    / "prefix_boundary_fit_matrix.csv"
)
MARINER_ROOT = Path("/Users/calvinxu/Projects/Work/Marin/mixture-selection")
CANONICAL = REFERENCES / "two_phase_surrogate_collaborator_packet_20260721/data/canonical"
SOURCE_PATHS = {
    "boundary.csv": BOUNDARY_SOURCE,
    "endpoint.csv": CANONICAL / "delphi_3e18_two_phase_fit.csv",
    "tied.csv": CANONICAL / "delphi_3e18_one_phase_fit.csv",
    "components.csv": REFERENCES / "delphi_3e18_observed_components_20260724/observed_component_panel.csv",
    "mixture_selection.py": MARINER_ROOT / "mixture_selection.py",
    "anchors.csv": MARINER_ROOT / "data/anchors.csv",
    "objectives.csv": MARINER_ROOT / "data/objectives.csv",
}
EXPECTED_HASHES = {
    "boundary.csv": "43032db14ae6ab0ee83eaeee24cf8ecab58740db0eae3d938534ff9088b1a646",
    "endpoint.csv": "7936ec3d59582bbfe8c7c10c19b8520829b5441ab8695bd50d3a4fad1321eadd",
    "mixture_selection.py": "61b7c017accb920d6874db848bc65fc0fdb729301a00a2fcf69a8595f1474396",
}
ALPHA = 2400 / 3007
FOLD_SEED = 20260912
OUTER_FOLDS = 5
INNER_FOLDS = 3
POLICY_COLUMNS = 196
BOUNDARY_RIDGE = 1.0
GAIN_RIDGES = (0.01, 0.1, 1.0, 10.0, float("inf"))
MODEL_NAMES = {
    "MTP-101": "mariner_aggregate",
    "MTP-102": "policy_gain",
    "MTP-103": "predicted_boundary_gain",
    "MTP-104": "measured_boundary_gain_feedback",
}
BOOTSTRAP_DRAWS = 3000
LOGGER = logging.getLogger("boundary_state")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@cache
def response_source(output: str) -> ModuleType:
    path = Path(output) / "inputs/mixture_selection.py"
    spec = importlib.util.spec_from_file_location("boundary_mariner_source", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclasses.dataclass(frozen=True)
class Panel:
    rows: pd.DataFrame
    buckets: tuple[str, ...]
    components: tuple[str, ...]
    aggregation: np.ndarray
    w0: np.ndarray
    w1: np.ndarray
    aggregate: np.ndarray
    inventory: np.ndarray
    boundary: np.ndarray
    endpoint: np.ndarray
    tied: np.ndarray
    policy: np.ndarray
    boundary_policy: np.ndarray
    distance: np.ndarray

    @property
    def gain(self) -> np.ndarray:
        return self.endpoint - self.tied


def unique_join(frame: pd.DataFrame, key: str, identities: list[str]) -> pd.DataFrame:
    if frame[key].duplicated().any():
        raise ValueError(f"duplicate {key} identities")
    indexed = frame.set_index(key, drop=False)
    if set(identities) != set(indexed.index.astype(str)):
        raise ValueError(f"source identity coverage differs for {key}")
    return indexed.loc[identities].reset_index(drop=True)


def load_panel(output: Path) -> Panel:
    inputs = output / "inputs"
    module = response_source(str(output))
    endpoint = pd.read_csv(inputs / "endpoint.csv")
    identities = endpoint.row_id.astype(str).tolist()
    boundary = unique_join(pd.read_csv(inputs / "boundary.csv"), "source_run_name", identities)
    tied = unique_join(pd.read_csv(inputs / "tied.csv"), "group_id", endpoint.group_id.astype(str).tolist())
    component_source = pd.read_csv(inputs / "components.csv")
    endpoint_components = unique_join(
        component_source[component_source.panel.eq("two_phase_fit")], "row_name", identities
    )
    tied_components = unique_join(
        component_source[component_source.panel.eq("one_phase_fit")], "row_name", tied.row_id.astype(str).tolist()
    )
    objective = module.read_objectives(inputs / "objectives.csv")["uncheatable"]
    components = objective.components
    buckets = tuple(
        str(column).removeprefix("phase_0_weight::") for column in endpoint if str(column).startswith("phase_0_weight::")
    )
    assert len(identities) == 280 and len(buckets) == 39 and len(components) == 7
    w0 = endpoint[[f"phase_0_weight::{bucket}" for bucket in buckets]].to_numpy(float)
    w1 = endpoint[[f"phase_1_weight::{bucket}" for bucket in buckets]].to_numpy(float)
    replay_weights = boundary[[f"phase_0_weight::{bucket}" for bucket in buckets]].to_numpy(float)
    np.testing.assert_allclose(w0, replay_weights, atol=1e-12, rtol=0)
    for phase, weights in ((0, w0), (1, w1)):
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            weights,
            endpoint_components[[f"phase_{phase}_{bucket}" for bucket in buckets]].to_numpy(float),
            atol=1e-12,
            rtol=0,
        )
    aggregate = ALPHA * w0 + (1 - ALPHA) * w1
    for phase in (0, 1):
        np.testing.assert_allclose(
            aggregate, tied[[f"phase_{phase}_weight::{bucket}" for bucket in buckets]], atol=1e-12, rtol=0
        )
    exposure0 = boundary[[f"phase_0_materialized_epochs::{bucket}" for bucket in buckets]].to_numpy(float)
    rates = []
    for index in range(len(buckets)):
        mask = w0[:, index] > 1e-12
        values = exposure0[mask, index] / w0[mask, index]
        rate = float(np.median(values))
        np.testing.assert_allclose(values, rate, atol=1e-8, rtol=1e-8)
        rates.append(rate)
    inventory = np.asarray(rates) / ALPHA
    exposure1 = (1 - ALPHA) * inventory * w1
    aggregate_exposure = inventory * aggregate
    component_tokens = tuple(component.split("/")[-2] for component in components)
    z = boundary[[f"{token}_bpb" for token in component_tokens]].to_numpy(float)
    y = endpoint_components[list(components)].to_numpy(float)
    t = tied_components[list(components)].to_numpy(float)
    for frame, matrix in ((boundary, z), (endpoint, y), (tied, t)):
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(matrix @ objective.weights, frame.uncheatable_bpb, atol=3e-7, rtol=0)
    distance = 0.5 * np.abs(w1 - w0).sum(axis=1)
    distance[distance < 1e-12] = 0
    np.testing.assert_allclose(y[distance == 0], t[distance == 0], atol=1e-12, rtol=0)
    b0, b1 = ALPHA * aggregate_exposure, (1 - ALPHA) * aggregate_exposure
    policy = np.column_stack(
        [
            distance,
            w1 - w0,
            module.benefit(exposure0, 0.25, 1.0) - module.benefit(b0, 0.25, 1.0),
            module.benefit(exposure1, 0.25, 1.0) - module.benefit(b1, 0.25, 1.0),
            module.harm(exposure0, 3.0) - module.harm(b0, 3.0),
            module.harm(exposure1, 3.0) - module.harm(b1, 3.0),
        ]
    )
    policy[distance == 0] = 0
    assert policy.shape == (280, POLICY_COLUMNS)
    boundary_policy = np.column_stack([np.sqrt(w0), module.benefit(exposure0, 0.25, 1.0), module.harm(exposure0, 3.0)])
    rows = pd.DataFrame(
        {
            "row": np.arange(len(endpoint)),
            "row_id": endpoint.row_id,
            "group_id": endpoint.group_id,
            "cohort": boundary.panel_source,
            "source_experiment": boundary.source_experiment,
            "boundary_run": boundary.run_name,
            "boundary_checkpoint": boundary.checkpoint_path,
            "boundary_data_seed": boundary.data_seed,
            "boundary_trainer_seed": boundary.trainer_seed,
            "endpoint_run": endpoint_components.training_run_id,
            "tied_run": tied_components.training_run_id,
            "physical_tied": distance == 0,
            "calibration": endpoint.row_id.eq("baseline_proportional"),
        }
    )
    assert int(rows.calibration.sum()) == 1 and len(rows.group_id.unique()) == 280
    aliases: dict[str, str] = {}
    for row in rows.itertuples():
        for value in (row.endpoint_run, row.tied_run, row.boundary_run):
            run, group = str(value), str(row.group_id)
            if run in aliases and aliases[run] != group:
                raise ValueError(f"training-run alias crosses groups: {run}")
            aliases[run] = group
    return Panel(
        rows,
        buckets,
        components,
        objective.weights,
        w0,
        w1,
        aggregate,
        inventory,
        z,
        y,
        t,
        policy,
        boundary_policy,
        distance,
    )


def blocked_folds(panel: Panel, rows: np.ndarray, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    scored = rows[~panel.rows.calibration.to_numpy()[rows]]
    calibration = rows[panel.rows.calibration.to_numpy()[rows]]
    labels = KMeans(n_clusters=count, n_init=50, random_state=seed).fit_predict(np.sqrt(panel.w0[scored]))
    assert len(np.unique(labels)) == count
    return [(np.sort(np.r_[scored[labels != fold], calibration]), scored[labels == fold]) for fold in range(count)]


def make_splits(panel: Panel) -> dict[str, np.ndarray]:
    splits = {}
    outer = blocked_folds(panel, np.arange(len(panel.rows)), OUTER_FOLDS, FOLD_SEED)
    for fold, (train, test) in enumerate(outer):
        name = f"outer{fold}"
        splits[f"{name}_train"], splits[f"{name}_test"] = train, test
        for index, (inner_train, inner_test) in enumerate(
            blocked_folds(panel, train, INNER_FOLDS, FOLD_SEED + 100 + fold)
        ):
            inner = f"{name}_inner{index}"
            splits[f"{inner}_train"], splits[f"{inner}_test"] = inner_train, inner_test
            nested = blocked_folds(panel, inner_train, INNER_FOLDS, FOLD_SEED + 1000 + fold * 10 + index)
            for nested_index, (nested_train, nested_test) in enumerate(nested):
                key = f"{inner}_boundary{nested_index}"
                splits[f"{key}_train"], splits[f"{key}_test"] = nested_train, nested_test
    return splits


def get_splits(output: Path) -> dict[str, np.ndarray]:
    with np.load(output / "splits.npz", allow_pickle=False) as values:
        return {key: values[key].copy() for key in values.files}


def run_identity(output: Path) -> dict[str, str]:
    paths = [Path(__file__), output / "PROTOCOL.md", output / "splits.npz", *sorted((output / "inputs").glob("*"))]
    return {str(path): file_hash(path) for path in paths if path.is_file()}


def prepare(output: Path) -> None:
    if not (output / "PROTOCOL.md").exists():
        raise ValueError("write and register PROTOCOL.md before preparing fits")
    (output / "inputs").mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, source in SOURCE_PATHS.items():
        digest = file_hash(source)
        if name in EXPECTED_HASHES and digest != EXPECTED_HASHES[name]:
            raise ValueError(f"changed registered source: {source}")
        destination = output / "inputs" / name
        if destination.exists() and file_hash(destination) != digest:
            raise ValueError(f"changed input snapshot: {destination}")
        if not destination.exists():
            shutil.copyfile(source, destination)
        sources[name] = {"source": str(source), "sha256": digest}
    panel = load_panel(output)
    if not (output / "splits.npz").exists():
        np.savez_compressed(output / "splits.npz", **make_splits(panel))
    splits = get_splits(output)
    outer_labels = np.full(len(panel.rows), -1, dtype=int)
    for fold in range(OUTER_FOLDS):
        train, test = splits[f"outer{fold}_train"], splits[f"outer{fold}_test"]
        assert not np.intersect1d(train, test).size
        outer_labels[test] = fold
    assert np.array_equal(outer_labels < 0, panel.rows.calibration.to_numpy())
    panel.rows.assign(outer_fold=outer_labels).to_csv(output / "joined_rows.csv", index=False)
    write_json(output / "split_membership.json", splits)
    write_json(
        output / "input_audit.json",
        {
            "sources": sources,
            "rows": len(panel.rows),
            "asymmetric_rows": int(np.sum(panel.distance > 0)),
            "tied_rows": int(np.sum(panel.distance == 0)),
            "scored_rows": int(np.sum(outer_labels >= 0)),
            "fold_sizes": np.bincount(outer_labels[outer_labels >= 0]),
            "components": panel.components,
            "aggregation_weights": panel.aggregation,
            "buckets": panel.buckets,
            "inventory": panel.inventory,
            "phase0_fraction": ALPHA,
            "crossed_boundary_status": "No provenance-preserving local nine-prefix boundary-component join found.",
            "policy_columns": panel.policy.shape[1],
            "boundary_predictor_columns": panel.boundary_policy.shape[1],
            "alias_groups_disjoint": True,
        },
    )
    LOGGER.info("prepared %s: 279 scored paired groups", output)


@dataclasses.dataclass(frozen=True)
class RidgeFit:
    center: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    ridge: float
    effective_dof: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + ((x - self.center) / self.scale) @ self.coefficients


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float, *, centered: bool) -> RidgeFit:
    center = x.mean(axis=0) if centered else np.zeros(x.shape[1])
    matrix = x - center
    scale = np.sqrt(np.mean(matrix**2, axis=0))
    scale[scale < 1e-12] = 1
    matrix = matrix / scale
    intercept = y.mean(axis=0) if centered else np.zeros(y.shape[1])
    if math.isinf(penalty):
        return RidgeFit(center, scale, intercept, np.zeros((x.shape[1], y.shape[1])), penalty, 0)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    regularization = len(x) * penalty
    coefficients = right.T @ ((singular / (singular**2 + regularization))[:, None] * (left.T @ (y - intercept)))
    dof = float(np.sum(singular**2 / (singular**2 + regularization))) + int(centered)
    return RidgeFit(center, scale, intercept, coefficients, penalty, dof)


def ridge_record(fitted: RidgeFit) -> dict[str, Any]:
    record = dataclasses.asdict(fitted)
    record["ridge"] = "infinity" if math.isinf(fitted.ridge) else fitted.ridge
    return record


def gain_design(panel: Panel, rows: np.ndarray, state: np.ndarray | None, state_mean: np.ndarray) -> np.ndarray:
    if state is None:
        return panel.policy[rows]
    return np.column_stack([panel.policy[rows], panel.distance[rows, None] * (state - state_mean)])


def spectrum(matrix: np.ndarray, reference_top: float | None = None) -> dict[str, Any]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular) or singular[0] == 0:
        return {"rank": 0, "stable_rank": 0.0, "entropy_rank": 0.0, "singular_values": singular}
    cutoff = (singular[0] if reference_top is None else reference_top) * 1e-8
    rank = int(np.sum(singular > cutoff))
    mass = singular**2 / np.sum(singular**2)
    nonzero = mass > 0
    return {
        "rank": rank,
        "absolute_rank_cutoff": float(cutoff),
        "stable_rank": float(np.sum(singular**2) / singular[0] ** 2) if rank else 0.0,
        "entropy_rank": float(np.exp(-np.sum(mass[nonzero] * np.log(mass[nonzero])))) if rank else 0.0,
        "singular_values": singular,
    }


def preflight(output: Path) -> None:
    panel, splits = load_panel(output), get_splits(output)
    rank_rows, synthetic_rows = [], []
    bucket = panel.buckets.index("dolma3_stack_edu")
    synthetic = 0.02 * (panel.w1[:, bucket] - panel.w0[:, bucket])[:, None] * np.arange(1, 8)[None, :] / 7
    for fold in range(OUTER_FOLDS):
        train, test = splits[f"outer{fold}_train"], splits[f"outer{fold}_test"]
        x = panel.policy[train]
        scale = np.sqrt(np.mean(x**2, axis=0))
        scale[scale < 1e-12] = 1
        x = x / scale
        state = panel.distance[train, None] * (panel.boundary[train] - panel.boundary[train].mean(axis=0))
        state_scale = np.sqrt(np.mean(state**2, axis=0))
        state_scale[state_scale < 1e-12] = 1
        state = state / state_scale
        residual = state - x @ np.linalg.lstsq(x, state, rcond=1e-8)[0]
        state_top = float(np.linalg.svd(state, compute_uv=False)[0])
        for kind, matrix in (("policy", x), ("state", state), ("state_after_policy", residual)):
            rank_rows.append(
                {
                    "fold": fold,
                    "design": kind,
                    "rows": len(train),
                    "columns": matrix.shape[1],
                    **spectrum(matrix, state_top if kind == "state_after_policy" else None),
                }
            )
        coefficients = np.linalg.lstsq(x, synthetic[train], rcond=1e-10)[0]
        ridge = fit_ridge(panel.policy[train], synthetic[train], GAIN_RIDGES[0], centered=False)
        for method in ("least_squares_span", "ridge_0.01"):
            for population, selected in (("train", train), ("held_prefix", test)):
                predicted = (
                    panel.policy[selected] / scale @ coefficients
                    if method == "least_squares_span"
                    else ridge.predict(panel.policy[selected])
                )
                synthetic_rows.append(
                    {
                        "fold": fold,
                        "method": method,
                        "population": population,
                        "rmse": float(np.sqrt(np.mean((predicted - synthetic[selected]) ** 2))),
                        "max_error": float(np.max(np.abs(predicted - synthetic[selected]))),
                        "outcome_sd": float(np.std(synthetic[selected])),
                    }
                )
    write_json(output / "rank_audit.json", rank_rows)
    pd.DataFrame(synthetic_rows).to_csv(output / "synthetic_recovery.csv", index=False)
    train_recovery = [
        row["max_error"]
        for row in synthetic_rows
        if row["method"] == "least_squares_span" and row["population"] == "train"
    ]
    assert max(train_recovery) < 1e-8, "synthetic policy signal is not recoverable on its generating design"
    write_json(output / "preflight.json", {"identity": run_identity(output), "synthetic_train_span_passed": True})
    LOGGER.info("rank and synthetic preflight complete")


def boundary_crossfit(
    panel: Panel, splits: dict[str, np.ndarray], context: str, prefix: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train, test = splits[f"{context}_train"], splits[f"{context}_test"]
    full = fit_ridge(panel.boundary_policy[train], panel.boundary[train], BOUNDARY_RIDGE, centered=True)
    train_predictions = full.predict(panel.boundary_policy[train])
    positions = {int(row): index for index, row in enumerate(train)}
    records = {"full": ridge_record(full)}
    covered = []
    for index in range(INNER_FOLDS):
        name = f"{context}_{prefix}{index}"
        inner_train, inner_test = splits[f"{name}_train"], splits[f"{name}_test"]
        assert not np.intersect1d(inner_train, inner_test).size
        assert np.isin(np.r_[inner_train, inner_test], train).all()
        fitted = fit_ridge(
            panel.boundary_policy[inner_train], panel.boundary[inner_train], BOUNDARY_RIDGE, centered=True
        )
        train_predictions[[positions[int(row)] for row in inner_test]] = fitted.predict(
            panel.boundary_policy[inner_test]
        )
        covered.extend(inner_test.tolist())
        records[f"crossfit{index}"] = ridge_record(fitted)
    assert set(covered) == set(train[~panel.rows.calibration.to_numpy()[train]])
    return train_predictions, full.predict(panel.boundary_policy[test]), records


def fit_gain_fold(output: Path, fold: int) -> None:
    destination = output / "gain_fits" / f"outer{fold}.json"
    identity = run_identity(output)
    if destination.exists():
        if json.loads(destination.read_text())["identity"] != identity:
            raise ValueError(f"stale gain fit: {destination}")
        LOGGER.info("reuse gain fold %d", fold)
        return
    panel, splits = load_panel(output), get_splits(output)
    context = f"outer{fold}"
    train, test = splits[f"{context}_train"], splits[f"{context}_test"]
    state_train, state_test, boundary_records = boundary_crossfit(panel, splits, context, "inner")
    inner_cache = []
    for index in range(INNER_FOLDS):
        name = f"{context}_inner{index}"
        inner_train, inner_test = splits[f"{name}_train"], splits[f"{name}_test"]
        predicted_train, predicted_test, fitted = boundary_crossfit(panel, splits, name, "boundary")
        boundary_records[f"inner{index}"] = fitted
        inner_cache.append((inner_train, inner_test, predicted_train, predicted_test))
    records = {}
    predictions = {"MTP-101": np.zeros((len(test), len(panel.components)))}
    for model in ("MTP-102", "MTP-103", "MTP-104"):
        scores = []
        for penalty in GAIN_RIDGES:
            squared_error, count = 0.0, 0
            for inner_train, inner_test, predicted_train, predicted_test in inner_cache:
                z_train = None
                z_test = None
                if model == "MTP-103":
                    z_train, z_test = predicted_train, predicted_test
                if model == "MTP-104":
                    z_train, z_test = panel.boundary[inner_train], panel.boundary[inner_test]
                mean = np.zeros(7) if z_train is None else z_train.mean(axis=0)
                x_train = gain_design(panel, inner_train, z_train, mean)
                x_test = gain_design(panel, inner_test, z_test, mean)
                fitted = fit_ridge(x_train, panel.gain[inner_train], penalty, centered=False)
                error = (fitted.predict(x_test) - panel.gain[inner_test]) @ panel.aggregation
                squared_error += float(error @ error)
                count += len(error)
            scores.append(squared_error / count)
        best = min(range(len(scores)), key=lambda index: (scores[index], -index))
        z_train = None
        z_test = None
        if model == "MTP-103":
            z_train, z_test = state_train, state_test
        if model == "MTP-104":
            z_train, z_test = panel.boundary[train], panel.boundary[test]
        mean = np.zeros(7) if z_train is None else z_train.mean(axis=0)
        fitted = fit_ridge(
            gain_design(panel, train, z_train, mean), panel.gain[train], GAIN_RIDGES[best], centered=False
        )
        predictions[model] = fitted.predict(gain_design(panel, test, z_test, mean))
        np.testing.assert_allclose(predictions[model][panel.distance[test] == 0], 0, atol=1e-12, rtol=0)
        records[model] = {
            "fit": ridge_record(fitted),
            "state_mean": mean,
            "inner_mse": scores,
            "selected_ridge_index": best,
        }
    write_json(
        destination,
        {
            "identity": identity,
            "fold": fold,
            "train": train,
            "test": test,
            "models": records,
            "boundary_models": boundary_records,
            "gain_predictions": predictions,
            "boundary_predictions": state_test,
        },
    )
    LOGGER.info("gain fold %d complete", fold)


def fit_aggregate_task(argument: tuple[str, int, int]) -> dict[str, Any]:
    output_string, fold, component_index = argument
    output = Path(output_string)
    destination = output / "aggregate_fits" / f"outer{fold}_component{component_index}.json"
    identity = run_identity(output)
    if destination.exists():
        if json.loads(destination.read_text())["identity"] != identity:
            raise ValueError(f"stale aggregate fit: {destination}")
        return {"fold": fold, "component": component_index, "cached": True}
    with threadpool_limits(limits=1):
        panel, splits = load_panel(output), get_splits(output)
        module = response_source(output_string)
        component = panel.components[component_index]
        train, test = splits[f"outer{fold}_train"], splits[f"outer{fold}_test"]
        positions = {int(row): index for index, row in enumerate(train)}
        inner = tuple(
            (
                np.asarray([positions[int(row)] for row in splits[f"outer{fold}_inner{index}_train"]]),
                np.asarray([positions[int(row)] for row in splits[f"outer{fold}_inner{index}_test"]]),
            )
            for index in range(INNER_FOLDS)
        )
        swarm = module.Swarm(
            tuple(panel.rows.row_id.iloc[train]),
            panel.buckets,
            panel.aggregate[train],
            panel.inventory,
            pd.DataFrame(panel.tied[train], columns=panel.components),
            panel.rows.calibration.to_numpy()[train],
        )
        anchor = module.read_anchors(output / "inputs/anchors.csv", "uncheatable")[component]
        started = time.monotonic()
        fitted = module.fit_task(swarm, component, anchor, inner)
        predicted = fitted.head.predict(module.design_matrix(panel.aggregate[test] * panel.inventory, fitted.shape))
        elapsed = time.monotonic() - started
        assert np.isfinite(predicted).all()
        write_json(
            destination,
            {
                "identity": identity,
                "fold": fold,
                "component_index": component_index,
                "component": component,
                "train": train,
                "test": test,
                "fit": fitted.to_json(),
                "anchor": dataclasses.asdict(anchor),
                "prediction": predicted,
                "elapsed_seconds": elapsed,
            },
        )
    return {"fold": fold, "component": component_index, "elapsed_seconds": round(elapsed, 2)}


def metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - observed
    result = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "spearman": (
            float(stats.spearmanr(observed, prediction).statistic)
            if np.std(prediction) > 1e-14 and np.std(observed) > 1e-14
            else float("nan")
        ),
    }
    if np.std(prediction) > 1e-14:
        slope, intercept = np.polyfit(prediction, observed, 1)
        result.update(calibration_slope=float(slope), calibration_intercept=float(intercept))
    else:
        result.update(calibration_slope=float("nan"), calibration_intercept=float("nan"))
    return result


def selection_metrics(
    observed: np.ndarray, prediction: np.ndarray, gain: np.ndarray, predicted_gain: np.ndarray
) -> dict[str, float]:
    selected = int(np.argmin(prediction))
    chosen_delta = np.where(predicted_gain < 0, gain, 0)
    return {
        "endpoint_regret1": float(observed[selected] - observed.min()),
        "endpoint_regret5": float(np.min(observed[np.argsort(prediction, kind="stable")[:5]]) - observed.min()),
        "endpoint_selection_optimism": float(observed[selected] - prediction[selected]),
        "matched_decision_regret": float(np.mean(chosen_delta - np.minimum(gain, 0))),
        "gain_sign_accuracy": float(np.mean((predicted_gain < 0) == (gain < 0))),
        "asymmetric_choice_fraction": float(np.mean(predicted_gain < 0)),
    }


def summarize(output: Path) -> None:
    panel, splits = load_panel(output), get_splits(output)
    identity = run_identity(output)
    rows, boundary_rows, fit_rows = [], [], []
    for fold in range(OUTER_FOLDS):
        gain_record = json.loads((output / "gain_fits" / f"outer{fold}.json").read_text())
        assert gain_record["identity"] == identity
        test = splits[f"outer{fold}_test"]
        aggregate = np.empty((len(test), 7))
        for index in range(7):
            record = json.loads((output / "aggregate_fits" / f"outer{fold}_component{index}.json").read_text())
            assert record["identity"] == identity and record["test"] == test.tolist()
            aggregate[:, index] = record["prediction"]
            fit_rows.append(
                {
                    "fold": fold,
                    "component": panel.components[index],
                    "elapsed_seconds": record["elapsed_seconds"],
                    **{key: record["fit"][key] for key in ("ridge", "kappa", "flat_profile", "inner_cv_rmse")},
                }
            )
        for model in MODEL_NAMES:
            predicted_gain = np.asarray(gain_record["gain_predictions"][model], float)
            prediction = aggregate + predicted_gain
            for position, row in enumerate(test):
                metadata = panel.rows.iloc[row]
                for target, weights in [
                    ("uncheatable", panel.aggregation),
                    *[(name, np.eye(7)[index]) for index, name in enumerate(panel.components)],
                ]:
                    rows.append(
                        {
                            "model": model,
                            "model_name": MODEL_NAMES[model],
                            "fold": fold,
                            "row": int(row),
                            "row_id": str(metadata.row_id),
                            "group_id": str(metadata.group_id),
                            "target": target,
                            "physical_tied": bool(metadata.physical_tied),
                            "cohort": str(metadata.cohort),
                            "observed": float(panel.endpoint[row] @ weights),
                            "prediction": float(prediction[position] @ weights),
                            "observed_tied": float(panel.tied[row] @ weights),
                            "predicted_tied": float(aggregate[position] @ weights),
                            "observed_gain": float(panel.gain[row] @ weights),
                            "predicted_gain": float(predicted_gain[position] @ weights),
                        }
                    )
        for position, row in enumerate(test):
            for index, component in enumerate(panel.components):
                boundary_rows.append(
                    {
                        "fold": fold,
                        "row": int(row),
                        "row_id": panel.rows.row_id.iloc[row],
                        "target": component,
                        "observed": panel.boundary[row, index],
                        "prediction": gain_record["boundary_predictions"][position][index],
                    }
                )
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(output / "boundary_predictions.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(output / "aggregate_fit_summary.csv", index=False)
    metric_rows = []
    for (model, target), frame in predictions.groupby(["model", "target"], sort=False):
        for fold in ["oof", *range(OUTER_FOLDS)]:
            selected = frame if fold == "oof" else frame[frame.fold.eq(fold)]
            for population in ("all", "asymmetric", "qsplit_signal", "domain_deletion", "frontier15"):
                mask = np.ones(len(selected), dtype=bool)
                if population == "asymmetric":
                    mask = ~selected.physical_tied.to_numpy()
                elif population in ("qsplit_signal", "domain_deletion"):
                    mask = selected.cohort.eq(population).to_numpy()
                elif population == "frontier15":
                    asymmetric = selected[~selected.physical_tied]
                    best = asymmetric.nsmallest(max(1, math.ceil(0.15 * len(asymmetric))), "observed").index
                    mask = selected.index.isin(best)
                subset = selected[mask]
                if not len(subset):
                    continue
                for kind in ("endpoint", "gain", "tied"):
                    observed_column = "observed" if kind == "endpoint" else f"observed_{kind}"
                    predicted_column = "prediction" if kind == "endpoint" else f"predicted_{kind}"
                    metric_rows.append(
                        {
                            "model": model,
                            "target": target,
                            "fold": fold,
                            "population": population,
                            "kind": kind,
                            "rows": len(subset),
                            **metrics(subset[observed_column].to_numpy(), subset[predicted_column].to_numpy()),
                        }
                    )
                if target == "uncheatable":
                    record = selection_metrics(
                        subset.observed.to_numpy(),
                        subset.prediction.to_numpy(),
                        subset.observed_gain.to_numpy(),
                        subset.predicted_gain.to_numpy(),
                    )
                    metric_rows.append(
                        {
                            "model": model,
                            "target": target,
                            "fold": fold,
                            "population": population,
                            "kind": "selection",
                            "rows": len(subset),
                            **record,
                        }
                    )
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(output / "metrics.csv", index=False)
    bootstrap = paired_bootstrap(predictions)
    bootstrap.to_csv(output / "paired_bootstrap.csv", index=False)
    write_report(output, metric_frame, bootstrap)
    artifacts = [path for path in output.iterdir() if path.is_file() and path.name != "complete.json"]
    write_json(
        output / "complete.json",
        {
            "identity": identity,
            "output_sha256": {path.name: file_hash(path) for path in artifacts},
            "fit_shards": 35,
            "gain_shards": 5,
        },
    )


def paired_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(FOLD_SEED + 10000)
    result = []
    canonical = predictions[predictions.target.eq("uncheatable")]
    for population in ("all", "asymmetric"):
        frames = {
            model: (
                canonical[
                    canonical.model.eq(model) & (True if population == "all" else ~canonical.physical_tied)
                ].sort_values("row")
            )
            for model in MODEL_NAMES
        }
        count = len(frames["MTP-101"])
        samples = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
        for left, right in (
            ("MTP-102", "MTP-101"),
            ("MTP-103", "MTP-102"),
            ("MTP-104", "MTP-102"),
            ("MTP-104", "MTP-103"),
        ):
            first, second = frames[left], frames[right]
            assert np.array_equal(first.row, second.row)
            gain = first.observed_gain.to_numpy()
            for name in ("gain_rmse", "endpoint_rmse", "matched_decision_regret"):
                if name == "matched_decision_regret":
                    first_errors = np.where(first.predicted_gain.to_numpy() < 0, gain, 0) - np.minimum(gain, 0)
                    second_errors = np.where(second.predicted_gain.to_numpy() < 0, gain, 0) - np.minimum(gain, 0)
                    difference = first_errors.mean() - second_errors.mean()
                    draws = (first_errors - second_errors)[samples].mean(axis=1)
                else:
                    actual = first.observed.to_numpy() if name == "endpoint_rmse" else gain
                    key = "prediction" if name == "endpoint_rmse" else "predicted_gain"
                    first_errors = (first[key].to_numpy() - actual) ** 2
                    second_errors = (second[key].to_numpy() - actual) ** 2
                    difference = np.sqrt(first_errors.mean()) - np.sqrt(second_errors.mean())
                    draws = np.sqrt(first_errors[samples].mean(axis=1)) - np.sqrt(second_errors[samples].mean(axis=1))
                low, high = np.quantile(draws, [0.025, 0.975])
                result.append(
                    {
                        "left": left,
                        "right": right,
                        "population": population,
                        "metric": name,
                        "difference": difference,
                        "ci_low": low,
                        "ci_high": high,
                        "groups": count,
                        "draws": BOOTSTRAP_DRAWS,
                    }
                )
    return pd.DataFrame(result)


def write_report(output: Path, metrics_frame: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    summary_rows = []
    for model in MODEL_NAMES:
        selected = metrics_frame[
            metrics_frame.model.eq(model)
            & metrics_frame.target.eq("uncheatable")
            & metrics_frame.fold.eq("oof")
            & metrics_frame.population.eq("asymmetric")
        ]
        endpoint = selected[selected.kind.eq("endpoint")].iloc[0]
        gain = selected[selected.kind.eq("gain")].iloc[0]
        decision = selected[selected.kind.eq("selection")].iloc[0]
        summary_rows.append(
            {
                "candidate": model,
                "endpoint RMSE": endpoint.rmse,
                "gain RMSE": gain.rmse,
                "matched regret": decision.matched_decision_regret,
                "gain slope": gain.calibration_slope,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)
    measured = summary[summary.candidate.eq("MTP-104")].iloc[0]
    policy = summary[summary.candidate.eq("MTP-102")].iloc[0]
    predicted = summary[summary.candidate.eq("MTP-103")].iloc[0]
    helps_policy = bool(
        measured["gain RMSE"] < policy["gain RMSE"] and measured["matched regret"] < policy["matched regret"]
    )
    helps_predicted = bool(
        measured["gain RMSE"] < predicted["gain RMSE"] and measured["matched regret"] < predicted["matched regret"]
    )
    write_json(
        output / "gates.json",
        {
            "measured_beats_policy_on_gain_and_matched_regret": helps_policy,
            "measured_beats_predicted_on_gain_and_matched_regret": helps_predicted,
            "temporal_model_promotion": False,
            "scope": "High-dimensional predictive diagnostic on archived one-continuation-per-prefix data.",
        },
    )
    conclusion = (
        "Measured boundary state passed" if helps_policy and helps_predicted else "Measured boundary state did not pass"
    )
    contrasts = bootstrap[
        bootstrap.population.eq("asymmetric")
        & bootstrap.left.eq("MTP-104")
        & bootstrap.metric.isin(["gain_rmse", "matched_decision_regret"])
    ]
    joined = pd.read_csv(output / "joined_rows.csv")
    folds = (
        joined[joined.outer_fold >= 0].groupby("outer_fold").agg(rows=("row", "count"), tied=("physical_tied", "sum"))
    )
    folds["asymmetric"] = folds.rows - folds.tied
    ranks = pd.DataFrame(json.loads((output / "rank_audit.json").read_text()))
    rank_table = ranks.pivot(index="fold", columns="design", values="rank")
    synthetic = pd.read_csv(output / "synthetic_recovery.csv")
    synthetic_train_max = synthetic[
        (synthetic.method == "least_squares_span") & (synthetic.population == "train")
    ].max_error.max()
    synthetic_test_max = synthetic[
        (synthetic.method == "least_squares_span") & (synthetic.population == "held_prefix")
    ].rmse.max()
    text = f"""# Boundary state diagnostic

{conclusion} the prespecified paired-gain prediction and selection gate. The comparison covers 238 asymmetric
policies across five held-prefix mixture blocks. Forty-one additional tied policies are scored in the full
tables; the proportional calibration policy is always training data. These are archived outcomes, one
continuation per unique prefix, and a single trainer seed. The result concerns prediction from this gated
seven-loss readout.

{summary.to_markdown(index=False, floatfmt='.6f')}

The held-out fold sizes are below. Fold 1 contains only tied policies, including every domain-deletion row;
the phase-gain comparison therefore has nonzero actions in four folds. The asymmetric rows all come from the
qsplit signal cohort. This design does not establish phase-gain transfer between data-generation cohorts.

{folds.to_markdown()}

BPB means bits per byte; lower is better. Gain is the asymmetric endpoint minus its paired constant-mixture
endpoint, so a negative value is an improvement. Matched regret is the average loss from choosing the worse of
those two measured alternatives. MTP-101 fits the full MARINER aggregate procedure to each fold's paired tied
training rows. MTP-102 adds a policy/exposure ridge gain prediction. MTP-103 also uses cross-fitted predicted
boundary losses. MTP-104 instead receives measured boundary losses and is a feedback procedure.

The seven components are fitted separately by MARINER and jointly by the multi-output ridge heads; Uncheatable
is reconstructed with fixed canonical byte weights. Every MARINER shape, ridge, floor multiplier, and final
coefficient is fitted inside the outer training data. Boundary predictions used for gain-head training are
themselves cross-fitted; gain-head inner validation rebuilds the boundary prediction procedure inside that
inner training set.

The following differences are measured-state minus comparator. Negative values favor measured state. The 95%
intervals resample matched source groups conditional on this fixed set of fitted folds. They omit refitting
and independent trainer-seed uncertainty.

{contrasts[['left', 'right', 'metric', 'difference', 'ci_low', 'ci_high']].to_markdown(index=False, floatfmt='.6f')}

Training-design numerical ranks are below. The projected state residual uses a cutoff of 1e-8 times the largest
unprojected state singular value, so numerical roundoff is counted as zero. In three folds, the policy columns
already span every nonzero training action. The state comparison there can change ridge generalization even
though it adds no training-set span.

{rank_table.to_markdown()}

The synthetic policy signal's largest training recovery error was {synthetic_train_max:.3g} BPB. Its largest
held-prefix RMSE was {synthetic_test_max:.3g} BPB under least-squares recovery. Full per-fold results and the
minimum-ridge shrinkage comparison are in `synthetic_recovery.csv`; this check establishes only the tested span.

The 196-column policy design is high-dimensional relative to 279 scored source groups. All gain features
vanish on tied policies, including the seven added state columns. The residual head remains a predictive
diagnostic even if it wins. Failure cannot exclude richer state/action interactions. Prefix-blocked validation
does not establish source-cohort-disjoint transfer, held-action transfer, or ranking among common actions at
an unseen checkpoint. The locally found crossed-nine-state endpoint file lacks a verified boundary-component
join and was excluded.

[PROTOCOL.md](PROTOCOL.md) records the frozen model equations, prior route, and selection rule.
[input_audit.json](input_audit.json) records exact source paths, hashes, joins, and component weights.
[rank_audit.json](rank_audit.json) and [synthetic_recovery.csv](synthetic_recovery.csv) record the prefit span
checks. [predictions.csv](predictions.csv) contains every scored component and aggregate prediction.
[metrics.csv](metrics.csv) contains per-fold and pooled prediction, calibration, and selection metrics,
including the retrospective best-15% frontier stratum and the two source cohorts.
[paired_bootstrap.csv](paired_bootstrap.csv) retains all paired contrasts. Component fits and gain/auxiliary
fits are saved in `aggregate_fits/` and `gain_fits/`.

Reproduce from the Marin root with `uv run --offline
experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_boundary_state_mariner_20260912.py run
--workers 2`. The script checks immutable inputs and reuses completed fit shards only when their identities
agree. All operations are local; no remote data, training, or evaluation jobs are requested.
"""
    (output / "REPORT.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "preflight", "fit", "summarize", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output = args.output.resolve()
    with threadpool_limits(limits=1):
        if args.stage in ("prepare", "run"):
            prepare(output)
        if args.stage in ("preflight", "run"):
            preflight(output)
        if args.stage in ("fit", "run"):
            audit = json.loads((output / "preflight.json").read_text())
            if audit["identity"] != run_identity(output):
                raise ValueError("preflight identity changed; rerun before fitting")
            for fold in range(OUTER_FOLDS):
                fit_gain_fold(output, fold)
            jobs = [(str(output), fold, component) for fold in range(OUTER_FOLDS) for component in range(7)]
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(fit_aggregate_task, job) for job in jobs]
                for future in as_completed(futures):
                    LOGGER.info("MARINER %s", future.result())
        if args.stage in ("summarize", "run"):
            summarize(output)


if __name__ == "__main__":
    main()
