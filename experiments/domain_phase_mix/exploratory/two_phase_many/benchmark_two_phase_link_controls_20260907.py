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
"""Refit established two-phase controls on the frozen link-transfer folds.

The source models and candidate grids are unchanged. Their historical selectors
required every row to receive OOF predictions, so the scoring adapter below
instead scores exactly the noncalibration rows covered by the supplied folds.
Fits are resumable by target, model, and outer fold; every prediction uses the
prepared canonical component aggregate rather than a historical prediction.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)

ROOT = SCRIPT_DIR / "reference_outputs" / "two_phase_link_transfer_20260907"
DEFAULT_INPUTS = ROOT / "inputs"
DEFAULT_OUTPUT = ROOT / "controls"
MODEL_IDS = ("hierarchical_phase_replay", "separate_heads", "effective_exposure_dsp")
TARGETS = ("uncheatable", "table9")
FOLD_IDS = (0, 1, 2, -1)
PROTOCOL_VERSION = "two-phase-link-controls-v1"


def calibrated_selection(
    observed: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    candidates: list[Any],
    fit: Callable[[Any, np.ndarray], Any],
    predict: Callable[[Any, np.ndarray], np.ndarray],
    metadata: Callable[[Any], dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]]]:
    """Use source candidate ordering and RMSE/Spearman selection on scored rows only."""
    covered = np.concatenate([test for _, test in folds])
    assert len(np.unique(covered)) == len(covered), "inner test rows overlap"
    assert len(covered) == len(observed) - 2, "exactly two calibration rows must remain unscored"
    best: tuple[float, float, int] | None = None
    records = []
    for index, candidate in enumerate(candidates):
        prediction = np.full(len(observed), np.nan)
        for train, test in folds:
            model = fit(candidate, train)
            prediction[test] = predict(model, test)
        assert np.isfinite(prediction[covered]).all(), "incomplete scored prediction"
        metrics = baseline.hierarchical_grp.metric_summary(observed[covered], prediction[covered])
        records.append({**metadata(candidate), **metrics})
        score = (float(metrics["rmse"]), -float(metrics["spearman"]), index)
        if best is None or score < best:
            best = score
    assert best is not None, "no control candidates"
    return candidates[best[2]], records


def fit_control(
    model_id: str,
    dataset: baseline.pooled.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    target: str,
    family_index: np.ndarray,
) -> baseline.FitResult:
    """Fit source controls with calibration-excluding hyperparameter selection."""
    all_rows = np.arange(dataset.n)
    if model_id == "effective_exposure_dsp":
        return baseline.fit_model(model_id, dataset, folds, target, family_index, rpl_workers=1)
    if model_id == "separate_heads":
        selected, sweep = calibrated_selection(
            dataset.y,
            folds,
            list(baseline.observatory.SEPARATE_L2_GRID),
            lambda l2, rows: baseline.observatory.separate_fit(dataset, rows, l2, baseline.observatory.TWO_PHASE),
            lambda model, rows: baseline.predict_model(model_id, model, dataset, dataset.weights[rows]),
            lambda l2: {"l2": l2},
        )
        model = baseline.observatory.separate_fit(dataset, all_rows, selected, baseline.observatory.TWO_PHASE)
        selection = {"l2": selected, "candidate_sweep": sweep}
    elif model_id == "hierarchical_phase_replay":
        structured = baseline.observatory.family_dataset(dataset)
        shapes = baseline.observatory.hierarchical_phase_replay_shape_candidates(baseline.observatory.TWO_PHASE)
        source = baseline.hierarchical_grp

        def select(
            configs: list[baseline.hierarchical_grp.Config],
        ) -> tuple[baseline.hierarchical_grp.Config, list[dict[str, Any]]]:
            return calibrated_selection(
                structured.target,
                folds,
                configs,
                lambda config, rows: source.fit_model(structured, config, rows),
                lambda fitted, rows: fitted.predict(structured.weights[rows]),
                lambda config: source.config_record(config, {}),
            )

        _, screen = select(source.baseline_configs(shapes))
        best_by_shape: dict[int, float] = {}
        for row in screen:
            index = int(row["shape_index"])
            best_by_shape[index] = min(best_by_shape.get(index, float("inf")), float(row["rmse"]))
        shape_indices = [
            index
            for index, _ in sorted(best_by_shape.items(), key=lambda item: item[1])[
                : baseline.observatory.HIERARCHICAL_PHASE_REPLAY_TOP_SHAPES
            ]
        ]
        selected, sweep = select(
            source.structural_configs(source.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY, shapes, shape_indices)
        )
        model = source.fit_model(structured, selected, all_rows)
        selection = {
            "selected_config": asdict(selected),
            "top_shape_indices": shape_indices,
            "baseline_shape_screen": screen,
            "candidate_sweep": sweep,
        }
    else:
        raise ValueError(f"unknown model: {model_id}")
    return baseline.FitResult(model, selection, baseline.parameter_diagnostics(model_id, model))


def protocol(inputs: Path) -> dict[str, Any]:
    sources = {Path(__file__), *(Path(path) for path in baseline.SOURCE_FILES)}
    # Include transitive local source imports used by the original model builders.
    for module in sys.modules.values():
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(REPO_ROOT):
                sources.add(path)
    paths = {str(path.relative_to(REPO_ROOT)): baseline.file_hash(path) for path in sorted(sources)}
    return {
        "version": PROTOCOL_VERSION,
        "source_sha256": paths,
        "input_sha256": {
            name: baseline.file_hash(inputs / name) for name in ("panel.npz", "splits.npz", "manifest.json")
        },
        "scoring": "All noncalibration inner test rows, weighted equally; RMSE, then Spearman, then source grid order.",
        "head_targets": "Fixed canonical component aggregates from the prepared520 bundle.",
        "final_fit": "All520 rows; final inner folds retain the two calibration rows in every training split.",
        "thread_limit": 1,
    }


def global_to_local_folds(
    splits: dict[str, np.ndarray], train: np.ndarray, prefix: str, calibration: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Map frozen global row indices to a local training dataset without changing membership."""
    reverse = np.full(len(calibration), -1, dtype=int)
    reverse[train] = np.arange(len(train))
    result = []
    for index in range(3):
        inner_train = reverse[splits[f"{prefix}{index}_train"]]
        inner_test = reverse[splits[f"{prefix}{index}_test"]]
        assert (inner_train >= 0).all() and (inner_test >= 0).all(), "inner split escapes outer training"
        assert not calibration[train[inner_test]].any()
        assert calibration[train[inner_train]].sum() == 2
        result.append((inner_train, inner_test))
    return tuple(result)


def run_cell(inputs: Path, output: Path, model_id: str, target: str, fold: int, identity: dict[str, Any]) -> None:
    """Fit one independently resumable cell and save predictions plus its model."""
    destination = output / model_id / target / ("full" if fold == -1 else f"outer{fold}")
    if destination.exists():
        complete = json.loads((destination / "complete.json").read_text())
        if complete["protocol"] != identity:
            raise ValueError(f"changed protocol for {destination}; choose a new output directory")
        for name, digest in complete["sha256"].items():
            assert baseline.file_hash(destination / name) == digest, f"changed control artifact: {name}"
        print(f"reuse {model_id}/{target}/{fold}", flush=True)
        return
    with np.load(inputs / "panel.npz", allow_pickle=False) as archive:
        panel = {name: archive[name] for name in archive.files}
    with np.load(inputs / "splits.npz", allow_pickle=False) as archive:
        splits = {name: archive[name] for name in archive.files}
    expanded = baseline.expanded.load_300m(target)
    assert np.array_equal(expanded.frame["run_name"].astype(str).to_numpy(), panel["runs"])
    assert np.max(np.abs(expanded.weights - panel["weights"])) < 1e-12
    response = panel[f"{target}_aggregate"]
    assert np.max(np.abs(expanded.y - response)) <= 3e-6
    expanded = replace(expanded, y=response)
    dataset = baseline.as_pooled(expanded)
    train = np.arange(len(response)) if fold == -1 else splits[f"outer{fold}_train"]
    test = np.flatnonzero(~panel["calibration_mask"]) if fold == -1 else splits[f"outer{fold}_test"]
    local = baseline.subset_dataset(dataset, train, f"control_{fold}")
    prefix = "final_inner" if fold == -1 else f"outer{fold}_inner"
    inner = global_to_local_folds(splits, train, prefix, panel["calibration_mask"])
    print(f"fit {model_id}/{target}/{fold}: {len(train)} train, {len(test)} test", flush=True)
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = fit_control(model_id, local, inner, target, expanded.family_index)
        prediction = baseline.predict_model(model_id, fitted.model, local, panel["weights"])
    elapsed = time.monotonic() - started
    assert prediction.shape == response.shape and np.isfinite(prediction).all()
    prediction_table = pd.DataFrame(
        {
            "model": model_id,
            "target": target,
            "outer_fold": fold,
            "row_index": np.arange(len(response)),
            "run": panel["runs"],
            "group": panel["groups"],
            "observed": response,
            "prediction": prediction,
            "training_row": np.isin(np.arange(len(response)), train),
            "scored_oof": np.isin(np.arange(len(response)), test) & (fold >= 0),
            "calibration_only": panel["calibration_mask"],
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".control-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "cell"
        stage.mkdir()
        np.savez_compressed(stage / "prediction.npz", prediction=prediction, observed=response, train=train, test=test)
        prediction_table.to_csv(stage / "predictions.csv", index=False)
        with (stage / "model.pkl").open("wb") as handle:
            pickle.dump(fitted.model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        baseline.write_json(
            stage / "fit.json",
            {
                "model": model_id,
                "target": target,
                "fold": fold,
                "elapsed_seconds": elapsed,
                "selection": fitted.selection,
                "parameter_diagnostics": fitted.parameter_diagnostics,
                "inner_global_indices": [{"train": train[a], "test": train[b]} for a, b in inner],
                "prediction_metrics": baseline.scalar_metrics(response[test], prediction[test]),
                "metrics_kind": "in-sample descriptive" if fold == -1 else "out-of-fold",
            },
        )
        baseline.write_json(
            stage / "complete.json",
            {"protocol": identity, "sha256": {path.name: baseline.file_hash(path) for path in sorted(stage.iterdir())}},
        )
        os.replace(stage, destination)
    print(f"completed {model_id}/{target}/{fold} in {elapsed:.2f} seconds", flush=True)


def collect(output: Path) -> None:
    frames = []
    for model_id in MODEL_IDS:
        for target in TARGETS:
            for fold in FOLD_IDS:
                path = output / model_id / target / ("full" if fold == -1 else f"outer{fold}") / "predictions.csv"
                if path.exists():
                    frames.append(pd.read_csv(path))
    if frames:
        predictions = pd.concat(frames, ignore_index=True)
        predictions.to_csv(output / "predictions.csv", index=False)
        predictions.loc[predictions["scored_oof"]].to_csv(output / "oof_predictions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default=",".join(MODEL_IDS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--folds", default=",".join(str(value) for value in FOLD_IDS))
    args = parser.parse_args()
    models = tuple(args.models.split(","))
    targets = tuple(args.targets.split(","))
    folds = tuple(int(value) for value in args.folds.split(","))
    if not set(models) <= set(MODEL_IDS) or not set(targets) <= set(TARGETS) or not set(folds) <= set(FOLD_IDS):
        raise ValueError("unknown model, target, or fold")
    identity = protocol(args.inputs)
    for model_id in models:
        for target in targets:
            for fold in folds:
                run_cell(args.inputs, args.output_dir, model_id, target, fold, identity)
                collect(args.output_dir)


if __name__ == "__main__":
    main()
