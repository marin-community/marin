# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Nested endpoint diagnostics for observed and policy-predicted boundary state.

These direct-BPB ridge regressions test whether low-dimensional boundary state
helps endpoint prediction. They are not MARINER replacements or promotable
surrogates. They share a policy design and add only state-by-phase-contrast
interactions. The response is final loss, never final-minus-boundary loss.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_boundary_state_mariner_20260912 import RidgeFit, fit_ridge, write_json
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from tpp40_temporal_data_20261014 import INNER_FOLDS, OUTER_FOLDS, SPLIT_SEED, TemporalData, load_complete_tpp40

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "reference_outputs/tpp40_temporal_round_20261014/boundary"
RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0)
RANKS = (1, 2)
PREFIX_RIDGE = 1.0
MODELS = ("policy", "feedback", "open_loop")


@dataclasses.dataclass(frozen=True)
class Policy:
    """Pre-training policy inputs for the diagnostic models."""

    w0: np.ndarray
    w1: np.ndarray
    e0: np.ndarray
    e1: np.ndarray
    alpha: float

    def subset(self, rows: np.ndarray) -> Policy:
        return Policy(self.w0[rows], self.w1[rows], self.e0[rows], self.e1[rows], self.alpha)


@dataclasses.dataclass(frozen=True)
class StateProjection:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def predict(self, boundary: np.ndarray) -> np.ndarray:
        return ((boundary - self.mean) / self.scale) @ self.components.T


@dataclasses.dataclass(frozen=True)
class DiagnosticFit:
    """A direct endpoint diagnostic with per-task inner-selected ridge and rank."""

    mode: str
    selected: tuple[tuple[int, float], ...]
    fits: dict[tuple[int, float], RidgeFit]
    projection: StateProjection | None
    prefix_fit: RidgeFit | None

    def predict(self, policy: Policy, boundary: np.ndarray | None = None) -> np.ndarray:
        state = None
        if self.mode == "feedback":
            if boundary is None or self.projection is None:
                raise ValueError("Feedback predictions require measured boundary state")
            state = self.projection.predict(boundary)
        elif self.mode == "open_loop":
            if boundary is not None or self.prefix_fit is None or self.projection is None:
                raise ValueError("Open-loop predictions use policy inputs and no measured test state")
            state = self.projection.predict(self.prefix_fit.predict(prefix_design(policy)))
        elif boundary is not None:
            raise ValueError("Policy-only predictions do not consume boundary measurements")
        predictions = {}
        for key, fitted in self.fits.items():
            rank, _ = key
            predictions[key] = fitted.predict(endpoint_design(policy, state, rank))
        return np.column_stack([predictions[key][:, task] for task, key in enumerate(self.selected)])


def policy_inputs(data: TemporalData) -> Policy:
    panel = data.panel
    return Policy(panel.phase0, panel.phase1, panel.phase0_epochs, panel.phase1_epochs, panel.alpha)


def fit_record(fitted: DiagnosticFit) -> dict:
    """Serialize coefficients and transforms needed to replay held-out predictions."""
    return {
        "mode": fitted.mode,
        "selected_per_task": fitted.selected,
        "projection": None if fitted.projection is None else dataclasses.asdict(fitted.projection),
        "prefix_fit": None if fitted.prefix_fit is None else dataclasses.asdict(fitted.prefix_fit),
        "endpoint_fits": [
            {"rank": rank, "ridge": ridge, "fit": dataclasses.asdict(value)}
            for (rank, ridge), value in fitted.fits.items()
        ],
    }


def prefix_design(policy: Policy) -> np.ndarray:
    """Encode only the realized first-phase allocation and exposure."""
    return np.column_stack([np.sqrt(policy.w0), np.log1p(policy.e0)])


def endpoint_design(policy: Policy, state: np.ndarray | None, rank: int) -> np.ndarray:
    """Append low-rank state interactions that vanish on exactly tied policies."""
    aggregate = policy.alpha * policy.w0 + (1 - policy.alpha) * policy.w1
    contrast = policy.w1 - policy.w0
    base = np.column_stack(
        [np.sqrt(aggregate), np.log1p(policy.e0 + policy.e1), contrast, 0.5 * np.abs(contrast).sum(axis=1)]
    )
    if rank == 0:
        return base
    if state is None or state.shape[1] < rank:
        raise ValueError("State interaction rank exceeds the supplied state dimension")
    interaction = (contrast[:, :, None] * state[:, None, :rank]).reshape(len(contrast), -1)
    return np.column_stack([base, interaction])


def project_state(boundary: np.ndarray) -> StateProjection:
    """Fit standardized boundary-state PCA using training observations only."""
    mean = boundary.mean(axis=0)
    scale = boundary.std(axis=0)
    scale[scale < 1e-12] = 1
    _, singular, components = np.linalg.svd((boundary - mean) / scale, full_matrices=False)
    variance = singular**2
    ratio = variance[: max(RANKS)] / variance.sum()
    return StateProjection(mean, scale, components[: max(RANKS)], ratio)


def prefix_folds(data: TemporalData, rows: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Block boundary cross-fitting by prefix neighborhoods within its training set."""
    groups = data.rows.group_id.to_numpy(str)
    unique = np.unique(groups[rows])
    points = np.asarray([data.panel.phase0[rows[groups[rows] == group]].mean(axis=0) for group in unique])
    labels = KMeans(n_clusters=INNER_FOLDS, n_init=20, random_state=seed, algorithm="lloyd").fit_predict(np.sqrt(points))
    by_group = dict(zip(unique, labels, strict=True))
    row_labels = np.asarray([by_group[group] for group in groups[rows]])
    if len(np.unique(row_labels)) != INNER_FOLDS:
        raise ValueError("Degenerate prefix neighborhoods")
    return [(rows[row_labels != index], rows[row_labels == index]) for index in range(INNER_FOLDS)]


def predicted_state(
    policy: Policy,
    boundary: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, RidgeFit]:
    """Predict every training state out of fold, and test state from training only."""
    features = prefix_design(policy)
    full = fit_ridge(features[train], boundary[train], PREFIX_RIDGE, centered=True)
    train_prediction = np.full((len(train), boundary.shape[1]), np.nan)
    positions = {int(row): index for index, row in enumerate(train)}
    covered = []
    for inner_train, inner_test in folds:
        if np.intersect1d(inner_train, inner_test).size or not np.isin(np.r_[inner_train, inner_test], train).all():
            raise ValueError("Boundary cross-fit rows cross the training boundary")
        fitted = fit_ridge(features[inner_train], boundary[inner_train], PREFIX_RIDGE, centered=True)
        train_prediction[[positions[int(row)] for row in inner_test]] = fitted.predict(features[inner_test])
        covered.extend(inner_test.tolist())
    if sorted(covered) != sorted(train.tolist()) or not np.isfinite(train_prediction).all():
        raise ValueError("Boundary cross-fitting must score every training row exactly once")
    return train_prediction, full.predict(features[test]), full


def fit_context(
    data: TemporalData,
    train: np.ndarray,
    test: np.ndarray,
    inner_folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[dict[str, DiagnosticFit], dict]:
    """Fit matched diagnostics with fully nested prefix predictions and transforms."""
    if data.panel.tied[train].any():
        raise ValueError("Tied outcome rows are evaluation-only")
    policy = policy_inputs(data)
    y, boundary = data.final_targets, data.boundary_atoms
    if np.intersect1d(train, test).size:
        raise ValueError("Endpoint training and scoring rows overlap")
    if sorted(np.concatenate([right for _, right in inner_folds]).tolist()) != sorted(train.tolist()):
        raise ValueError("Endpoint inner folds do not cover outer training rows exactly once")
    candidates = {"policy": [(0, ridge) for ridge in RIDGES]}
    candidates.update({mode: [(rank, ridge) for rank in RANKS for ridge in RIDGES] for mode in MODELS[1:]})
    errors = {mode: np.zeros((len(keys), y.shape[1])) for mode, keys in candidates.items()}
    inner_records = []
    for index, (inner_train, inner_test) in enumerate(inner_folds):
        if np.intersect1d(inner_train, inner_test).size or not np.isin(np.r_[inner_train, inner_test], train).all():
            raise ValueError("Endpoint inner fold crosses outer training boundary")
        projection = project_state(boundary[inner_train])
        nested = prefix_folds(data, inner_train, seed + 100 + index)
        predicted_train, predicted_test, _ = predicted_state(policy, boundary, inner_train, inner_test, nested)
        states = {
            "policy": (None, None),
            "feedback": (projection.predict(boundary[inner_train]), projection.predict(boundary[inner_test])),
            "open_loop": (projection.predict(predicted_train), projection.predict(predicted_test)),
        }
        for mode, keys in candidates.items():
            state_train, state_test = states[mode]
            for candidate, (rank, ridge) in enumerate(keys):
                fitted = fit_ridge(
                    endpoint_design(policy.subset(inner_train), state_train, rank), y[inner_train], ridge, centered=True
                )
                prediction = fitted.predict(endpoint_design(policy.subset(inner_test), state_test, rank))
                errors[mode][candidate] += np.sum((prediction - y[inner_test]) ** 2, axis=0)
        inner_records.append(
            {
                "train": inner_train,
                "test": inner_test,
                "boundary_crossfits": [{"train": left, "test": right} for left, right in nested],
                "state_explained_variance_ratio": projection.explained_variance_ratio,
            }
        )
    projection = project_state(boundary[train])
    outer_prefix_folds = prefix_folds(data, train, seed)
    predicted_train, predicted_test, prefix_fit = predicted_state(policy, boundary, train, test, outer_prefix_folds)
    states = {
        "policy": None,
        "feedback": projection.predict(boundary[train]),
        "open_loop": projection.predict(predicted_train),
    }
    results, selections = {}, {}
    for mode, keys in candidates.items():
        selected = tuple(
            keys[min(range(len(keys)), key=lambda index: (errors[mode][index, task], -keys[index][1], keys[index][0]))]
            for task in range(y.shape[1])
        )
        fits = {
            (rank, ridge): fit_ridge(
                endpoint_design(policy.subset(train), states[mode], rank), y[train], ridge, centered=True
            )
            for rank, ridge in sorted(set(selected))
        }
        results[mode] = DiagnosticFit(
            mode, selected, fits, projection if mode != "policy" else None, prefix_fit if mode == "open_loop" else None
        )
        selections[mode] = {
            "candidates": keys,
            "inner_mse": errors[mode] / len(train),
            "selected_per_task": selected,
            "fit_effective_dof_per_task": [fits[key].effective_dof for key in selected],
        }
    records = {
        "train": train,
        "test": test,
        "inner_folds": inner_records,
        "outer_boundary_crossfits": [{"train": left, "test": right} for left, right in outer_prefix_folds],
        "state_explained_variance_ratio": projection.explained_variance_ratio,
        "boundary_prediction": predicted_test,
        "boundary_observed": boundary[test],
        "boundary_train_crossfit_prediction": predicted_train,
        "selections": selections,
    }
    return results, records


def run(output: Path, protocol: Path) -> None:
    """Run local nested diagnostics against a pre-existing protocol artifact."""
    if not protocol.is_file():
        raise ValueError("Register the offline-round protocol before fitting")
    data = load_complete_tpp40()
    identity = {
        "protocol": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ridge_implementation": (
            hashlib.sha256((BASE / "benchmark_boundary_state_mariner_20260912.py").read_bytes()).hexdigest()
        ),
        "data_adapter": hashlib.sha256((BASE / "tpp40_temporal_data_20261014.py").read_bytes()).hexdigest(),
        "input_hashes": data.panel.input_hashes,
        "folds": hashlib.sha256(data.folds.to_csv(index=False).encode()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    predictions = []
    with threadpool_limits(limits=1):
        for outer in range(OUTER_FOLDS):
            destination = output / f"outer{outer}.json"
            if destination.exists():
                record = json.loads(destination.read_text())
                if record["identity"] != identity:
                    raise ValueError(f"Stale boundary diagnostic cache: {destination}")
                predictions.extend(record["predictions"])
                continue
            rows = data.folds.loc[data.folds.outer_fold.eq(outer)]
            training = rows.loc[rows.role.eq("train")]
            train = training.row_index.to_numpy(int)
            test_rows = rows.loc[rows.role.ne("train")]
            test = test_rows.row_index.to_numpy(int)
            inner = [
                (
                    training.loc[training.inner_fold.ne(index), "row_index"].to_numpy(int),
                    training.loc[training.inner_fold.eq(index), "row_index"].to_numpy(int),
                )
                for index in range(INNER_FOLDS)
            ]
            fitted, diagnostics = fit_context(data, train, test, inner, SPLIT_SEED + 10000 + 1000 * outer)
            fold_predictions = []
            for mode, model in fitted.items():
                prediction = model.predict(
                    policy_inputs(data).subset(test), data.boundary_atoms[test] if mode == "feedback" else None
                )
                for index, row in enumerate(test):
                    for task, name in enumerate(data.task_names):
                        fold_predictions.append(
                            {
                                "outer_fold": outer,
                                "row_index": int(row),
                                "row_id": str(data.rows.row_id.iloc[row]),
                                "role": str(test_rows.role.iloc[index]),
                                "task": name,
                                "model": mode,
                                "observed": float(data.final_targets[row, task]),
                                "prediction": float(prediction[index, task]),
                            }
                        )
            write_json(
                destination,
                {
                    "identity": identity,
                    "models": {mode: fit_record(model) for mode, model in fitted.items()},
                    "diagnostics": diagnostics,
                    "predictions": fold_predictions,
                },
            )
            predictions.extend(fold_predictions)
    frame = pd.DataFrame(predictions)
    if (
        len(frame) != len(data.rows) * len(data.task_names) * len(MODELS)
        or frame.duplicated(["row_id", "task", "model"]).any()
    ):
        raise ValueError("Incomplete or duplicated boundary diagnostic predictions")
    frame.to_csv(output / "predictions.csv", index=False)
    write_json(
        output / "manifest.json",
        {
            "identity": identity,
            "models": MODELS,
            "ridges": RIDGES,
            "ranks": RANKS,
            "prefix_ridge": PREFIX_RIDGE,
            "interpretation": "Matched direct-BPB diagnostics only; no MARINER nesting and no surrogate promotion.",
            "rows": len(frame),
            "outcome": "final atom BPB, never final-minus-boundary",
            "state_interaction": "standardized boundary PCA times signed phase-weight contrast; no state main effect",
            "outer_split": "aggregate neighborhoods; exact aggregate/prefix aliases blocked",
            "prefix_crossfit_split": "prefix neighborhoods within the current training partition",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.protocol)


if __name__ == "__main__":
    main()
