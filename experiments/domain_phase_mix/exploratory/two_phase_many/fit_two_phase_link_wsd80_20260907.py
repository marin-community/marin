# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""WSD80 secondary screen of a tied-trained log-link temporal departure.

Preparation snapshots the surface, repeated runs, provenance, and fiber folds.
Fitting uses only training tied rows for the standalone aggregate procedure and
unpaired BPB residuals for the temporal head. Nominal 80/20 fibers are grouped
throughout, but the response reads physical 3040/3814 exposures. Thus nominal
fiber differences are not exact physical aggregate-matched contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    two_phase_link_residual_20260907 as temporal,
)

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/two_phase_link_transfer_20260907/wsd80"
STANDALONE = REPO_ROOT.parent / "mixture-selection/mixture_selection.py"
METRIC_CSV = wsd80.SURFACE_DIR / "wsd80_all_bpb_metrics.csv"
TARGETS = {
    "programming_languages": "eval/paloma/dolma_100_programing_languages-llama3/bpb",
    "c4": "eval/paloma/c4_en-llama3/bpb",
    "refinedweb": "eval/paloma/falcon-refinedweb-llama3/bpb",
}
OUTER_FOLDS = 5
INNER_FOLDS = 3
FOLD_SEED = 20260907
ARMS = ("spine-only", "damage-only", "benefit+damage")
COORDINATE_DECIMALS = 10
PROPORTIONAL_TOLERANCE = 1e-8  # The inherited cosine epoch multiplier has six decimal digits.


@dataclass(frozen=True)
class Snapshot:
    surface: pd.DataFrame
    repeats: pd.DataFrame
    weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_standalone(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("wsd80_frozen_mixture_selection", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import standalone method: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def phase_weights(frame: pd.DataFrame) -> np.ndarray:
    shares = frame[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(float)
    return np.stack([1.0 - shares, shares], axis=2)


def grouped_labels(aggregate: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Cluster distinct nominal aggregates so repeated fibers have equal weight."""
    unique, inverse = np.unique(np.round(aggregate, COORDINATE_DECIMALS), return_inverse=True)
    features = np.sqrt(np.column_stack([1.0 - unique, unique]))
    labels = KMeans(n_clusters=count, n_init=50, random_state=seed).fit_predict(features)
    if len(np.unique(labels)) != count:
        raise ValueError("A requested aggregate fold is empty")
    return labels[inverse]


def prepare(output: Path, standalone: Path) -> Snapshot:
    output.mkdir(parents=True, exist_ok=True)
    panel = wsd80.load_surface()
    metrics = pd.read_csv(METRIC_CSV)
    columns = ["wandb_run_id", *TARGETS.values()]
    surface = panel.frame.merge(metrics[columns], on="wandb_run_id", how="left", validate="one_to_one")
    repeats = wsd80.load_fiber_replicates().merge(metrics[columns], on="wandb_run_id", how="left", validate="one_to_one")
    if surface[list(TARGETS.values())].isna().any().any() or repeats[list(TARGETS.values())].isna().any().any():
        raise ValueError("Missing canonical target outcomes; do not discard incomplete rows silently")
    weights = phase_weights(surface)
    surface["row"] = np.arange(len(surface))
    surface["tied"] = np.max(np.abs(weights[:, 0] - weights[:, 1]), axis=1) < 1e-12
    surface["nominal_aggregate"] = np.round(panel.aggregate[:, 1], COORDINATE_DECIMALS)
    physical = panel.epochs / (panel.c0 + panel.c1)
    surface["physical_aggregate"] = physical[:, 1]
    proportional = 1.0 / (panel.c0 + panel.c1)
    proportional /= proportional.sum()
    surface["calibration"] = surface.tied & np.isclose(
        surface.phase_0_starcoder, proportional[1], atol=PROPORTIONAL_TOLERANCE, rtol=0
    )
    if surface.calibration.sum() != 1:
        raise ValueError("Expected exactly one observed proportional tied calibration coordinate")
    calibration_group = surface.loc[surface.calibration, "nominal_aggregate"].item()
    noncalibration = ~surface.nominal_aggregate.eq(calibration_group)
    surface["outer_fold"] = -1
    surface.loc[noncalibration, "outer_fold"] = grouped_labels(
        surface.loc[noncalibration, "nominal_aggregate"].to_numpy(), OUTER_FOLDS, FOLD_SEED
    )
    tied_nominal = surface.loc[surface.tied, "nominal_aggregate"].to_numpy()
    nominal_pairs = np.isclose(
        surface.nominal_aggregate.to_numpy()[:, None], tied_nominal[None, :], atol=1e-10, rtol=0
    ).any(axis=1)
    physical_pairs = np.isclose(physical[:, 1, None], physical[surface.tied, 1][None, :], atol=1e-10, rtol=0).any(axis=1)
    surface["nominal_tied_counterpart"] = np.asarray(nominal_pairs, dtype=bool)
    surface["physical_tied_counterpart"] = physical_pairs
    surface.to_csv(output / "surface_snapshot.csv", index=False)
    repeats.to_csv(output / "repeat_snapshot.csv", index=False)
    weights.reshape(len(surface), -1).tofile(output / "weights.float64")
    source_paths = [wsd80.SURFACE_CSV, wsd80.FIBER_CSV, METRIC_CSV, standalone, Path(wsd80.__file__)]
    manifest = {
        "schema_version": 1,
        "surface_rows": len(surface),
        "tied_rows": int(surface.tied.sum()),
        "asymmetric_rows": int((~surface.tied).sum()),
        "nominal_aggregate_groups": int(surface.nominal_aggregate.nunique()),
        "exact_physical_pairs": int((physical_pairs & ~surface.tied.to_numpy()).sum()),
        "nominal_pairs": int((nominal_pairs & ~surface.tied.to_numpy()).sum()),
        "repeat_rows": len(repeats),
        "repeat_seeds": {str(k): int(v) for k, v in repeats.data_seed.value_counts().items()},
        "c0": panel.c0.tolist(),
        "c1": panel.c1.tolist(),
        "proportional": proportional.tolist(),
        "calibration_row": int(surface.loc[surface.calibration, "row"].item()),
        "calibration_share_inventory_rounding_difference": float(
            surface.loc[surface.calibration, "phase_0_starcoder"].item() - proportional[1]
        ),
        "anchor_contract": "one observed proportional tied outcome; repeat_sd=0 unavailable",
        "phase_fit_contract": "unpaired BPB residual y_asymmetric - frozen A(physical aggregate)",
        "grouping": "five outer / three inner blocks of unique noncalibration nominal aggregate; calibration pinned",
        "targets": TARGETS,
        "sources": {str(path): sha256(path) for path in source_paths},
        "limitations": [
            "No exact physical aggregate-matched asymmetric/tied pairs exist.",
            "Nominal fiber pairs differ slightly in realized aggregate exposure.",
            "Surface outcomes are development-used reference-seed outcomes.",
            "Repeat coordinates overlap development coordinates; fresh seeds diagnose fixed-policy calibration only.",
            "Grid selection regret is descriptive and not fresh validation of a continuous optimum.",
        ],
    }
    write_json(output / "data_manifest.json", manifest)
    return Snapshot(surface, repeats, weights, panel.c0, panel.c1)


def inner_labels(frame: pd.DataFrame, seed: int) -> np.ndarray:
    calibration_groups = frame.loc[frame.calibration, "nominal_aggregate"]
    noncalibration = ~frame.nominal_aggregate.isin(calibration_groups).to_numpy()
    labels = np.full(len(frame), -1, dtype=int)
    labels[noncalibration] = grouped_labels(frame.nominal_aggregate.to_numpy()[noncalibration], INNER_FOLDS, seed)
    return labels


def fit_spine(module: ModuleType, snapshot: Snapshot, rows: np.ndarray, task: str, seed: int):
    frame = snapshot.surface.iloc[rows]
    tied = frame[frame.tied]
    if tied.calibration.sum() != 1:
        raise ValueError("Every aggregate fit must retain the proportional calibration row")
    labels = inner_labels(tied, seed)
    if any(np.sum(labels == fold) == 0 for fold in range(INNER_FOLDS)):
        raise ValueError("A tied-only inner validation fold is empty")
    swarm = module.Swarm(
        tuple(tied.wandb_run_id),
        tuple(wsd80.DOMAIN_NAMES),
        snapshot.weights[tied.row.to_numpy(), 0],
        snapshot.c0 + snapshot.c1,
        tied[[task]].reset_index(drop=True),
        tied.calibration.to_numpy(),
    )
    anchor = module.Anchor(float(tied.loc[tied.calibration, task].item()), 0.0)
    folds = module.folds_from_labels(np.arange(len(tied)), labels, INNER_FOLDS)
    return module.fit_task(swarm, task, anchor, folds)


def basis_and_spine(module: ModuleType, fit, weights: np.ndarray, c0: np.ndarray, c1: np.ndarray):
    basis = temporal.temporal_basis(
        weights[:, 0],
        weights[:, 1],
        c0,
        c1,
        fit.head.coefficients[:2],
        fit.head.coefficients[2:],
        partial(module.benefit, rate=fit.shape["rate"], power=fit.shape["power"]),
        partial(module.harm, threshold=fit.shape["threshold"]),
    )
    matrix = module.design_matrix(basis.total_exposure, fit.shape)
    prediction = fit.head.predict(matrix)
    linear = fit.head.intercept + matrix @ fit.head.coefficients
    return basis, prediction, linear


def phase_training_rows(snapshot: Snapshot, rows: np.ndarray) -> np.ndarray:
    return rows[~snapshot.surface.iloc[rows].tied.to_numpy()]


def fit_phase(module: ModuleType, snapshot: Snapshot, spine, rows: np.ndarray, task: str, penalty: float, arm: str):
    selected = phase_training_rows(snapshot, rows)
    basis, prediction, _ = basis_and_spine(module, spine, snapshot.weights[selected], snapshot.c0, snapshot.c1)
    response = snapshot.surface.iloc[selected][task].to_numpy(float) - prediction
    fitted = temporal.fit_bpb_contrasts(prediction - spine.head.floor, basis.columns, response, penalty, arm)
    if not fitted.success:
        raise ValueError(f"Temporal optimizer did not converge for {task}, {arm}, penalty={penalty}")
    return fitted


def score(response: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    error = prediction - response
    correlation = float(spearmanr(response, prediction).statistic) if len(response) > 2 else float("nan")
    return {
        "rows": len(response),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "spearman": correlation if math.isfinite(correlation) else None,
    }


def select_penalties(module: ModuleType, snapshot: Snapshot, rows: np.ndarray, task: str, seed: int):
    frame = snapshot.surface.iloc[rows]
    labels = inner_labels(frame, seed)
    error = np.zeros((2, len(temporal.PENALTY_GRID)))
    count = np.zeros_like(error)
    details = []
    for fold in range(INNER_FOLDS):
        train = rows[labels != fold]
        valid = phase_training_rows(snapshot, rows[labels == fold])
        if not len(valid):
            raise ValueError("A phase inner validation fold is empty")
        spine = fit_spine(module, snapshot, train, task, seed + 100 + fold)
        basis, base, linear = basis_and_spine(module, spine, snapshot.weights[valid], snapshot.c0, snapshot.c1)
        observed = snapshot.surface.iloc[valid][task].to_numpy(float)
        fits = []
        for arm_index, arm in enumerate(ARMS[1:]):
            for penalty_index, penalty in enumerate(temporal.PENALTY_GRID):
                fitted = fit_phase(module, snapshot, spine, train, task, penalty, arm)
                prediction = base + temporal.predict_bpb_delta(base - spine.head.floor, basis.columns, fitted.theta)
                error[arm_index, penalty_index] += float(np.sum((prediction - observed) ** 2))
                count[arm_index, penalty_index] += len(valid)
                fits.append({**fitted.to_json(), "heldout": score(observed, prediction)})
        details.append(
            {
                "inner_fold": fold,
                "train_rows": train.tolist(),
                "validation_rows": valid.tolist(),
                "spine": spine.to_json(),
                "spine_clip_rows": int(np.sum(np.abs(linear) >= module.LOG_CLIP)),
                "phase_fits": fits,
            }
        )
    rmse = np.sqrt(error / count)
    # Equal scores favor the simpler, more strongly shrunk procedure, including the exact null.
    selected = {
        arm: temporal.PENALTY_GRID[min(range(len(temporal.PENALTY_GRID)), key=lambda k: (rmse[i, k], -k))]
        for i, arm in enumerate(ARMS[1:])
    }
    return selected, {"rmse": rmse.tolist(), "folds": details}


def predict_arm(module: ModuleType, spine, phase_fit, weights: np.ndarray, snapshot: Snapshot):
    basis, base, linear = basis_and_spine(module, spine, weights, snapshot.c0, snapshot.c1)
    theta = np.zeros(2) if phase_fit is None else phase_fit.theta
    correction = basis.columns @ theta
    prediction = base + temporal.predict_bpb_delta(base - spine.head.floor, basis.columns, theta)
    clips = {
        "base_clip_rows": int(np.sum(np.abs(linear) >= module.LOG_CLIP)),
        "correction_clip_rows": int(np.sum(np.abs(correction) >= temporal.LOG_CLIP)),
        "combined_clip_rows": int(np.sum(np.abs(linear + correction) >= module.LOG_CLIP)),
        "max_abs_combined_linear": float(np.max(np.abs(linear + correction))),
    }
    return prediction, clips


def grid_audit(module: ModuleType, snapshot: Snapshot, spine, fits: dict, task: str, size: int):
    axis = np.linspace(0.0, 1.0, size)
    early, late = np.meshgrid(axis, axis, indexing="ij")
    dense_frame = pd.DataFrame({"phase_0_starcoder": early.ravel(), "phase_1_starcoder": late.ravel()})
    dense_weights = phase_weights(dense_frame)
    records = []
    observed = snapshot.surface[task].to_numpy(float)
    tied = snapshot.surface.tied.to_numpy()
    for arm in ARMS:
        prediction, clips = predict_arm(module, spine, fits.get(arm), snapshot.weights, snapshot)
        selected = int(np.argmin(prediction))
        selected_tied = int(np.flatnonzero(tied)[np.argmin(prediction[tied])])
        dense, dense_clips = predict_arm(module, spine, fits.get(arm), dense_weights, snapshot)
        index = int(np.argmin(dense))
        dense_tied = dense.reshape(size, size).diagonal()
        distance = np.max(np.abs(snapshot.weights - dense_weights[index]), axis=(1, 2))
        record = {
            "arm": arm,
            "selected_observed_row": selected,
            "selected_observed_tied_row": selected_tied,
            "selected_observed_prediction": float(prediction[selected]),
            "selected_observed_bpb": float(observed[selected]),
            "selected_observed_regret": float(observed[selected] - observed.min()),
            "selected_observed_gain_over_selected_tied": float(observed[selected_tied] - observed[selected]),
            "raw_grid_phase0": float(early.ravel()[index]),
            "raw_grid_phase1": float(late.ravel()[index]),
            "raw_grid_prediction": float(dense[index]),
            "raw_grid_predicted_gain": float(dense_tied.min() - dense[index]),
            "raw_grid_boundary": bool(early.ravel()[index] in (0.0, 1.0) or late.ravel()[index] in (0.0, 1.0)),
            "raw_grid_nearest_observed_linf": float(distance.min()),
            "surface_clips": clips,
            "dense_clips": dense_clips,
        }
        records.append(record)
        dense_frame[arm] = dense
    return records, dense_frame


def evaluate_nominal_pairs(frame: pd.DataFrame, prediction: np.ndarray, task: str) -> list[dict]:
    """Descriptive contrasts only: nominal fibers do not exactly match physical exposure."""
    rows = []
    for aggregate, block in frame.assign(prediction=prediction).groupby("nominal_aggregate"):
        tied = block[block.tied]
        if len(tied) != 1:
            continue
        reference = tied.iloc[0]
        for row in block[~block.tied].itertuples():
            row_values = block.loc[row.Index]
            rows.append(
                {
                    "nominal_aggregate": aggregate,
                    "row": int(row_values["row"]),
                    "tied_row": int(reference["row"]),
                    "physical_aggregate_difference": float(
                        row_values["physical_aggregate"] - reference["physical_aggregate"]
                    ),
                    "observed_delta": float(row_values[task] - reference[task]),
                    "predicted_delta": float(row_values["prediction"] - reference["prediction"]),
                }
            )
    return rows


def run_target(module: ModuleType, snapshot: Snapshot, key: str, output: Path, grid_size: int) -> None:
    task = TARGETS[key]
    target_dir = output / key
    target_dir.mkdir(exist_ok=True)
    predictions = []
    summaries = []
    for fold in [-1, *range(OUTER_FOLDS)]:
        result_path = target_dir / f"fold_{fold}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            predictions.extend(result["predictions"])
            summaries.extend(result["scores"])
            continue
        rows = snapshot.surface.row.to_numpy()
        if fold >= 0:
            train = rows[snapshot.surface.outer_fold.to_numpy() != fold]
            validation = rows[snapshot.surface.outer_fold.to_numpy() == fold]
        else:
            train, validation = rows, rows
        seed = FOLD_SEED + 1000 * (fold + 1)
        penalties, inner = select_penalties(module, snapshot, train, task, seed)
        spine = fit_spine(module, snapshot, train, task, seed + 500)
        phase_fits = {arm: fit_phase(module, snapshot, spine, train, task, penalties[arm], arm) for arm in ARMS[1:]}
        prediction_rows, scores, pair_rows = [], [], []
        clip_diagnostics = {}
        for arm in ARMS:
            prediction, clips = predict_arm(module, spine, phase_fits.get(arm), snapshot.weights[validation], snapshot)
            clip_diagnostics[arm] = clips
            observed = snapshot.surface.iloc[validation][task].to_numpy(float)
            for row, value in zip(validation, prediction, strict=True):
                prediction_rows.append({"fold": fold, "row": int(row), "arm": arm, "prediction": float(value)})
            for stratum, mask in (
                ("all", np.ones(len(validation), dtype=bool)),
                ("tied", snapshot.surface.iloc[validation].tied.to_numpy()),
                ("asymmetric", ~snapshot.surface.iloc[validation].tied.to_numpy()),
            ):
                if mask.any():
                    scores.append(
                        {"fold": fold, "arm": arm, "stratum": stratum, **score(observed[mask], prediction[mask])}
                    )
            pairs = evaluate_nominal_pairs(snapshot.surface.iloc[validation], prediction, task)
            pair_rows.extend({"fold": fold, "arm": arm, **row} for row in pairs)
            if pairs:
                scores.append(
                    {
                        "fold": fold,
                        "arm": arm,
                        "stratum": "nominal_pair_delta_approximate_exposure",
                        **score(
                            np.asarray([r["observed_delta"] for r in pairs]),
                            np.asarray([r["predicted_delta"] for r in pairs]),
                        ),
                    }
                )
        result = {
            "task": task,
            "fold": fold,
            "train_rows": train.tolist(),
            "validation_rows": validation.tolist(),
            "spine": spine.to_json(),
            "selected_phase_fits": {arm: fit.to_json() for arm, fit in phase_fits.items()},
            "inner_selection": inner,
            "predictions": prediction_rows,
            "scores": scores,
            "clip_diagnostics": clip_diagnostics,
            "nominal_pair_diagnostics": pair_rows,
        }
        if fold == -1:
            result["raw_grid_audit"], grid = grid_audit(module, snapshot, spine, phase_fits, task, grid_size)
            grid.to_csv(target_dir / "raw_grid_predictions.csv", index=False)
            repeated_weights = phase_weights(snapshot.repeats)
            repeat_predictions = snapshot.repeats.copy()
            for arm in ARMS:
                repeat_predictions[arm], _ = predict_arm(module, spine, phase_fits.get(arm), repeated_weights, snapshot)
            repeat_predictions.to_csv(target_dir / "repeat_predictions.csv", index=False)
        write_json(result_path, result)
        predictions.extend(prediction_rows)
        summaries.extend(scores)
        print(f"{key}: completed fold {fold}", flush=True)
    pd.DataFrame(predictions).to_csv(target_dir / "predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(target_dir / "scores.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--standalone", type=Path, default=STANDALONE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    parser.add_argument("--grid-size", type=int, default=201)
    args = parser.parse_args()
    snapshot = prepare(args.output_dir, args.standalone)
    print(json.dumps({"rows": len(snapshot.surface), "tied": int(snapshot.surface.tied.sum())}), flush=True)
    if args.prepare_only:
        return
    protocol = args.output_dir.parent / "PROTOCOL.md"
    if not protocol.exists():
        raise ValueError("The parent study protocol must be registered before fitting")
    fingerprint = {
        "data_manifest": sha256(args.output_dir / "data_manifest.json"),
        "script": sha256(Path(__file__)),
        "temporal_module": sha256(Path(temporal.__file__)),
        "protocol": sha256(protocol),
        "grid_size": args.grid_size,
    }
    fingerprint_path = args.output_dir / "fit_fingerprint.json"
    if fingerprint_path.exists() and json.loads(fingerprint_path.read_text()) != fingerprint:
        raise ValueError("Inputs or implementation changed; use a new output directory for the changed study")
    write_json(fingerprint_path, fingerprint)
    module = load_standalone(args.standalone)
    for key in args.targets:
        run_target(module, snapshot, key, args.output_dir, args.grid_size)


if __name__ == "__main__":
    main()
