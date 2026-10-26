# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Audit actual-design identifiability and synthetic recovery without observed labels.

The four fixed worlds are diagnostics, not promotion gates. Recovery uses the
native candidate estimator, including its floor and hyperparameter selection,
on the frozen first StarCoder outer partition. TPP40 receives rank checks only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_surrogate_20260921 as surrogate
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from two_phase_benchmark_data_20260921 import BenchPanel

SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 3.0}
FORGETTING = 4.0
CONSOLIDATION = 1.0
NOISE_SEED = 20260921
MODELS = ("NG2-001", "NG2-002", "NG2-003")
SCRIPT_DIR = Path(__file__).resolve().parent
PREPARED = SCRIPT_DIR / "reference_outputs/two_phase_benchmark_20260921/v3/prepared"


@dataclasses.dataclass(frozen=True)
class World:
    name: str
    floor: float
    intercept: float
    noise_sd: float
    phase: str


WORLDS = (
    World("strong", 0.65, -0.35, 0.0, "protected"),
    World("null_close_floor", 0.9, -1.8, 0.002, "aggregate"),
    World("null_far_floor", 0.1, 0.1, 0.01, "aggregate"),
    World("noisy_protected", 0.65, -0.35, 0.002, "protected"),
)


def geometry_panel(prepared: Path, name: str) -> BenchPanel:
    """Read frozen geometry and folds without opening the real outcomes file."""
    folder = prepared / name
    meta = json.loads((folder / "metadata.json").read_text())
    rows = pd.read_csv(folder / "rows.csv", keep_default_na=False, float_precision="round_trip")
    rows["tied"] = rows.tied.astype(str).str.lower().eq("true")
    with np.load(folder / "policies.npz") as arrays:
        metadata = dict(meta["metadata"])
        metadata["realized_phase0_epochs"] = arrays["phase0_epochs"]
        metadata["realized_phase1_epochs"] = arrays["phase1_epochs"]
        return BenchPanel(
            name=name,
            buckets=tuple(meta["buckets"]),
            phase0=arrays["phase0"],
            phase1=arrays["phase1"],
            alpha=float(meta["alpha"]),
            c0=arrays["c0"],
            c1=arrays["c1"],
            targets={},
            rows=rows,
            input_hashes=meta["source_hashes"],
            metadata=metadata,
        )


def tied_queries(panel: BenchPanel) -> BenchPanel:
    """Construct physically tied counterfactuals with newly computed doses."""
    aggregate = panel.aggregate
    if not np.allclose(panel.phase0_epochs, panel.phase0 * panel.c0, atol=1e-12, rtol=1e-12):
        raise ValueError("Synthetic tied queries require exact constant-exposure geometry")
    if not np.allclose(panel.phase1_epochs, panel.phase1 * panel.c1, atol=1e-12, rtol=1e-12):
        raise ValueError("Synthetic tied queries require exact constant-exposure geometry")
    metadata = {
        **panel.metadata,
        "realized_phase0_epochs": aggregate * panel.c0,
        "realized_phase1_epochs": aggregate * panel.c1,
    }
    rows = panel.rows.copy()
    rows["tied"] = True
    return dataclasses.replace(panel, phase0=aggregate, phase1=aggregate, rows=rows, metadata=metadata)


def generator_coefficients(panel: BenchPanel) -> np.ndarray:
    """Return prespecified, target-independent nonnegative response coefficients."""
    if len(panel.buckets) == 2:
        return np.asarray([0.8, 0.45, 0.03, 0.06])
    values = np.asarray([int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) / (2**32 - 1) for name in panel.buckets])
    scale = 2.0 / len(panel.buckets)
    return scale * np.concatenate([0.25 + 0.5 * values, 0.02 + 0.06 * values])


def world_mean(panel: BenchPanel, world: World) -> np.ndarray:
    """Evaluate a fixed synthetic law; no outcomes or fitted quantities are used."""
    queries = tied_queries(panel) if world.phase == "aggregate" else panel
    indices = np.arange(len(panel.rows))
    matrix = surrogate.design(queries, indices, SHAPE, "NG2-002", FORGETTING, CONSOLIDATION)
    return world.floor + np.exp(world.intercept + matrix @ generator_coefficients(panel))


def spectrum(matrix: np.ndarray) -> dict[str, Any]:
    """Describe a column-scaled matrix while retaining its absolute column norms."""
    norms = np.linalg.norm(matrix, axis=0)
    active = norms > np.finfo(float).eps
    scaled = matrix[:, active] / norms[active]
    singular = np.linalg.svd(scaled, compute_uv=False)
    if not len(singular):
        return {"rows": len(matrix), "columns": matrix.shape[1], "column_norms": norms.tolist(), "rank": 0}
    tolerance = np.finfo(float).eps * max(scaled.shape) * singular[0]
    rank = int(np.sum(singular > tolerance))
    return {
        "rows": len(matrix),
        "columns": matrix.shape[1],
        "active_columns": int(active.sum()),
        "column_norms": norms.tolist(),
        "singular_values": singular.tolist(),
        "rank": rank,
        "rank_at_relative_1e6": int(np.sum(singular > singular[0] * 1e-6)),
        "relative_smallest_singular": float(singular[-1] / singular[0]),
    }


def residualize(matrix: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """Project out a nuisance span using a rank-revealing SVD."""
    norms = np.linalg.norm(nuisance, axis=0)
    active = norms > np.finfo(float).eps
    scaled = nuisance[:, active] / norms[active]
    u, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    keep = singular > (singular[0] * max(scaled.shape) * np.finfo(float).eps)
    basis = u[:, keep]
    projected = matrix - basis @ (basis.T @ matrix)
    tolerance = 10 * max(nuisance.shape) * np.finfo(float).eps * np.linalg.norm(matrix, axis=0)
    projected[:, np.linalg.norm(projected, axis=0) <= tolerance] = 0.0
    return projected


def design_rank(panel: BenchPanel, indices: np.ndarray, model: str) -> dict[str, Any]:
    """Audit feature and kinetic directions before seeing response labels."""
    consolidation = 0.0 if model == "NG2-001" else CONSOLIDATION
    aggregate = surrogate.design(tied_queries(panel), indices, SHAPE, model, FORGETTING, consolidation)
    scalar = surrogate.design(panel, indices, SHAPE, "NG2-001", FORGETTING, 0.0)
    protected = surrogate.design(panel, indices, SHAPE, model, FORGETTING, consolidation)
    nuisance = np.column_stack([np.ones(len(indices)), aggregate, scalar - aggregate])
    difference = protected - scalar
    step = 1e-4
    beta = generator_coefficients(panel)
    derivatives = []
    for position in range(2):
        if model == "NG2-001" and position == 1:
            derivatives.append(np.zeros(len(indices)))
            continue
        upper, lower = [FORGETTING, consolidation], [FORGETTING, consolidation]
        upper[position] += step
        lower[position] -= step
        high = surrogate.design(panel, indices, SHAPE, model, *upper) @ beta
        low = surrogate.design(panel, indices, SHAPE, model, *lower) @ beta
        derivatives.append((high - low) / (2 * step))
    kinetics = np.column_stack(derivatives)
    projected = residualize(kinetics, nuisance)
    raw_norm = np.linalg.norm(kinetics, axis=0)
    projected_norm = np.linalg.norm(projected, axis=0)
    deficit = np.exp(-0.35 + protected @ beta)
    joint_jacobian = np.column_stack(
        [np.ones(len(indices)), deficit, protected * deficit[:, None], kinetics * deficit[:, None]]
    )
    return {
        "model": model,
        "nuisance": spectrum(nuisance),
        "protected_design": spectrum(protected),
        "protected_minus_scalar": spectrum(difference),
        "projected_protected_minus_scalar": spectrum(residualize(difference, nuisance)),
        "kinetics": spectrum(kinetics),
        "projected_kinetics": spectrum(projected),
        "kinetic_retained_norm_fraction": (
            np.divide(projected_norm, raw_norm, out=np.zeros_like(raw_norm), where=raw_norm > 0).tolist()
        ),
        "joint_response_jacobian_including_floor": spectrum(joint_jacobian),
        "kinetic_parameter_order": ["forgetting", "consolidation"],
        "derivative_step": step,
        "interpretation": "Local design diagnostic; not coefficient recovery or global identifiability",
    }


def frozen_partition(prepared: Path, panel: BenchPanel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = pd.read_csv(prepared / panel.name / "folds.csv")
    cell = folds[(folds["repeat"] == 0) & (folds.outer_fold == 0)]
    indexed = cell.set_index("row_id").reindex(panel.rows.row_id)
    train = np.flatnonzero(indexed.role.eq("train"))
    test = np.flatnonzero(indexed.role.isin(["test", "tied_test"]))
    inner = indexed.iloc[train].inner_fold.to_numpy(int)
    if panel.tied[train].any() or not len(train) or not len(test):
        raise ValueError("Invalid asymmetric-only synthetic fitting partition")
    return train, test, inner


def identity(prepared: Path, names: list[str]) -> dict[str, Any]:
    inputs = [prepared / "manifest.json"]
    for name in names:
        inputs.extend(
            prepared / name / filename for filename in ("rows.csv", "policies.npz", "metadata.json", "folds.csv")
        )
    manifest = json.loads(inputs[0].read_text())
    expected = {
        str(path.relative_to(prepared)): manifest["outputs"][str(path.relative_to(prepared))] for path in inputs[1:]
    }
    benchmark.verify_hashes(expected, prepared)
    return {
        "inputs": {str(path.resolve()): benchmark.sha256(path) for path in inputs},
        "sources": {**surrogate.source_hashes(), str(Path(__file__).resolve()): benchmark.sha256(Path(__file__))},
        "shape": SHAPE,
        "forgetting": FORGETTING,
        "consolidation": CONSOLIDATION,
        "noise_seed": NOISE_SEED,
        "worlds": [dataclasses.asdict(world) for world in WORLDS],
        "partition": {"repeat": 0, "outer_fold": 0},
        "environment": benchmark.environment_identity(),
    }


def check_completed(folder: Path, config: dict[str, Any]) -> bool:
    manifest = folder / "complete.json"
    if not manifest.exists():
        return False
    saved = json.loads(manifest.read_text())
    if saved["config"] != config:
        raise ValueError(f"Completed audit changed: {folder}; use a new output directory")
    benchmark.verify_hashes(saved["outputs"], folder)
    return True


def rank_audit(prepared: Path, output: Path, models: list[str]) -> None:
    names = ["starcoder_wsd80_dense", "tpp40"]
    config = {**identity(prepared, names), "mode": "rank", "models": models}
    if check_completed(output, config):
        return
    records = []
    for name in names:
        panel = geometry_panel(prepared, name)
        train, _, inner = frozen_partition(prepared, panel)
        partitions = [("outer_train", train)] + [
            (f"inner_train_{fold}", train[inner != fold]) for fold in sorted(np.unique(inner))
        ]
        for label, indices in partitions:
            for model in models:
                records.append({"panel": name, "partition": label, **design_rank(panel, indices, model)})
    benchmark.write_json(output / "rank.json", {"config": config, "records": records})
    benchmark.write_json(
        output / "complete.json", {"config": config, "outputs": {"rank.json": benchmark.sha256(output / "rank.json")}}
    )


def recovery_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Score endpoint, counterfactual gain, and within-menu policy selection."""
    held = frame[frame.role != "train"]
    asymmetric = held[~held.tied]
    truth_gain = asymmetric.tied_truth - asymmetric.truth
    predicted_gain = asymmetric.tied_prediction - asymmetric.prediction
    observed_best = float(held.truth.min())
    chosen = held.sort_values(["prediction", "row_id"], kind="stable").iloc[0]
    best_truth_tied = float(held.tied_truth.min())
    tied_chosen = held.sort_values(["tied_prediction", "row_id"], kind="stable").iloc[0]
    choose_tied = bool(tied_chosen.tied_prediction <= chosen.prediction)
    selected_truth = float(tied_chosen.tied_truth if choose_tied else chosen.truth)
    selected_prediction = float(tied_chosen.tied_prediction if choose_tied else chosen.prediction)
    return {
        "held_rows": len(held),
        "held_asymmetric_rows": len(asymmetric),
        "held_truth_range": float(held.truth.max() - held.truth.min()),
        "held_truth_sd": float(held.truth.std(ddof=0)),
        "endpoint_rmse": float(np.sqrt(np.mean((held.prediction - held.truth) ** 2))),
        "synthetic_tied_rmse": float(np.sqrt(np.mean((held.tied_prediction - held.tied_truth) ** 2))),
        "gain_rmse": float(np.sqrt(np.mean((predicted_gain - truth_gain) ** 2))),
        "gain_bias": float(np.mean(predicted_gain - truth_gain)),
        "true_max_matched_gain": float(truth_gain.max()),
        "true_matched_gain_rms": float(np.sqrt(np.mean(truth_gain**2))),
        "predicted_max_matched_gain": float(predicted_gain.max()),
        "held_policy_regret": float(chosen.truth - observed_best),
        "selected_optimism": float(chosen.truth - chosen.prediction),
        "joint_with_synthetic_tied_regret": selected_truth - min(observed_best, best_truth_tied),
        "joint_with_synthetic_tied_optimism": selected_truth - selected_prediction,
        "joint_selected_tied": choose_tied,
        "interpretation": (
            "Fixed finite-menu synthetic diagnostic, not continuous optimization or statistical certification"
        ),
    }


def recover(prepared: Path, output: Path, models: list[str], world_names: list[str]) -> None:
    name = "starcoder_wsd80_dense"
    panel = geometry_panel(prepared, name)
    tied = tied_queries(panel)
    train, test, inner = frozen_partition(prepared, panel)
    base_config = identity(prepared, [name])
    all_indices = np.arange(len(panel.rows))
    for world in WORLDS:
        if world.name not in world_names:
            continue
        truth = world_mean(panel, world)
        tied_truth = world_mean(tied, world)
        noise = np.random.default_rng(NOISE_SEED).normal(0, world.noise_sd, len(truth))
        response = truth + noise
        for model in models:
            folder = output / world.name / model
            config = {**base_config, "mode": "recover", "world": dataclasses.asdict(world), "model": model}
            if check_completed(folder, config):
                continue
            benchmark.write_json(folder / "requested_config.json", config)
            started = time.monotonic()
            with threadpool_limits(limits=1):
                fitted = surrogate.fit_model(panel, response, train, inner, model)
            frame = panel.rows[["row_id", "policy_id", "group_id", "tied"]].copy()
            frame["role"] = "excluded"
            frame.loc[train, "role"] = "train"
            frame.loc[test, "role"] = "test"
            frame["truth"] = truth
            frame["response"] = response
            frame["prediction"] = fitted.predict(panel, all_indices)
            frame["tied_truth"] = tied_truth
            frame["tied_prediction"] = fitted.predict(tied, all_indices)
            metrics = recovery_metrics(frame[frame.role != "excluded"])
            benchmark.write_json(folder / "fit.json", fitted.to_json())
            benchmark.write_json(
                folder / "world.json",
                {
                    **config,
                    "coefficients": generator_coefficients(panel).tolist(),
                    "train_ids": panel.rows.iloc[train].row_id.tolist(),
                    "inner_labels": inner.tolist(),
                    "noiseless_outer_floor_gamma": float(
                        (np.median(truth[train]) - world.floor) / (np.median(truth[train]) - truth[train].min())
                    ),
                },
            )
            frame.to_csv(folder / "predictions.csv", index=False)
            benchmark.write_json(folder / "metrics.json", {**metrics, "elapsed_seconds": time.monotonic() - started})
            paths = ("requested_config.json", "fit.json", "world.json", "predictions.csv", "metrics.json")
            benchmark.write_json(
                folder / "complete.json",
                {"config": config, "outputs": {file: benchmark.sha256(folder / file) for file in paths}},
            )
            print(json.dumps({"world": world.name, "model": model, **metrics}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("rank", "recover"))
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=["NG2-001", "NG2-002"])
    parser.add_argument(
        "--worlds", nargs="+", choices=[world.name for world in WORLDS], default=[world.name for world in WORLDS]
    )
    args = parser.parse_args()
    if args.mode == "rank":
        rank_audit(args.prepared, args.output, args.models)
    else:
        recover(args.prepared, args.output, args.models, args.worlds)


if __name__ == "__main__":
    main()
