# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Attribute exact-pair phase predictions to frozen HPR and RPL response blocks.

This diagnostic does not select nonlinear configurations or fit a new model.
It reconstructs the common-fold expanded 300M baseline, verifies its persisted
OOF predictions, and then removes physically named response blocks with and
without refitting the original linear head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hpr,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_repaired_rpl_300m_20260731 as repaired_rpl_benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_estimator_repair_20260731 as repaired_rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "hpr_rpl_phase_block_attribution_20260731"
BASELINE_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731"
REPAIRED_RPL_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_20260731"
PREREGISTRATION = DEFAULT_OUTPUT_DIR / "preregistration.md"
TARGETS = ("uncheatable", "table9")
MODELS = ("hierarchical_phase_replay", "retained_power_law_repaired")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 731_510
PREDICTION_TOLERANCE = 1e-9
ACTIVE_TOLERANCE = 1e-10
PROTOCOL_VERSION = "hpr-rpl-phase-block-attribution-v1"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ReconstructedModel:
    full_prediction: np.ndarray
    intercept_prediction: np.ndarray
    block_prediction: dict[str, np.ndarray]
    refit_omission_prediction: dict[str, np.ndarray]
    coefficient_rows: list[dict[str, Any]]
    reconstruction_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_payload(bootstrap_samples: int) -> dict[str, Any]:
    sources = [
        Path(__file__),
        PREREGISTRATION,
        Path(baseline.__file__),
        Path(hpr.__file__),
        Path(repaired_rpl_benchmark.__file__),
        Path(rpl.__file__),
        Path(repaired_rpl.__file__),
    ]
    for target in TARGETS:
        for model in MODELS:
            cell = cell_dir(target, model)
            sources.extend(
                [
                    cell / "fold_selections.json",
                    cell / "predictions.csv",
                    cell / "complete.json",
                ]
            )
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen attribution inputs: {missing}")
    payload = {
        "version": PROTOCOL_VERSION,
        "parent_protocol": "e30c84f654eb55e9d428eb9ee1afeac69a111d629abe45de6f96eb81db026185",
        "targets": TARGETS,
        "models": MODELS,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "prediction_tolerance": PREDICTION_TOLERANCE,
        "source_hashes": {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in sources},
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def completed(output_dir: Path, protocol_hash: str) -> bool:
    path = output_dir / "complete.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    required = (
        "block_metrics.csv",
        "pair_block_contributions.csv",
        "fold_block_metrics.csv",
        "report.md",
        "phase_block_ablation.html",
    )
    return payload.get("protocol_hash") == protocol_hash and all((output_dir / name).exists() for name in required)


def hpr_config(record: dict[str, Any]) -> hpr.Config:
    raw = record["selection"]["selected_config"]
    return hpr.Config(
        variant=hpr.Variant(raw["variant"]),
        shape_index=int(raw["shape_index"]),
        shape=hpr.family_grp.Shape(**raw["shape"]),
        l2=float(raw["l2"]),
        residual_shrink=float(raw["residual_shrink"]),
        undercoverage_fraction=float(raw["undercoverage_fraction"]),
        coverage_gate_ratio=float(raw["coverage_gate_ratio"]),
    )


def rpl_shape(record: dict[str, Any]) -> tuple[rpl.Shape, float]:
    return rpl.Shape(**record["shape"]), float(record["ridge"])


def cell_dir(target: str, model: str) -> Path:
    if model == "hierarchical_phase_replay":
        return BASELINE_DIR / "cells" / target / model
    if model == "retained_power_law_repaired":
        return REPAIRED_RPL_DIR / "cells" / target / model
    raise ValueError(f"unsupported attribution model {model}")


def hpr_blocks(names: tuple[str, ...]) -> dict[str, np.ndarray]:
    prefixes = {
        "retained_bucket_benefit": (
            "singleton_signal:",
            "pooled_base_signal:",
            "bucket_excess_signal:",
        ),
        "retained_family_benefit": ("family_coverage_signal:",),
        "family_overexposure": ("family_overexposure:",),
        "family_member_replay": ("family_member_replay:",),
        "global_phase_tv": ("phase_shift_tv",),
    }
    blocks = {
        block: np.asarray([any(name.startswith(prefix) for prefix in accepted) for name in names], dtype=bool)
        for block, accepted in prefixes.items()
    }
    coverage = np.sum(np.stack(list(blocks.values()), axis=0), axis=0)
    if not np.all(coverage == 1):
        unknown = [name for name, count in zip(names, coverage, strict=True) if count != 1]
        raise ValueError(f"HPR block partition is not exhaustive and disjoint: {unknown}")
    return blocks


def rpl_blocks(
    geometry: rpl.Geometry,
    shape: rpl.Shape,
    layout: repaired_rpl.FeatureLayout,
) -> dict[str, np.ndarray]:
    families = len(np.unique(geometry.families))
    hierarchical = families + len(geometry.excess_domains)
    widths = [
        ("retained_benefit", hierarchical),
        ("aggregate_damage", hierarchical),
        ("concentration", 1),
    ]
    if shape.ordering_channel:
        widths.extend(
            [
                ("ordering_benefit", families),
                ("ordering_damage", families),
                ("asymmetry", 1),
            ]
        )
    total = sum(width for _name, width in widths)
    if total != layout.total_count:
        raise ValueError(f"repaired RPL block widths {total} do not match design {layout.total_count}")
    blocks: dict[str, np.ndarray] = {}
    start = 0
    for name, width in widths:
        mask = np.zeros(total, dtype=bool)
        mask[start : start + width] = True
        blocks[name] = mask
        start += width
    return blocks


def fit_hpr_head(
    design: np.ndarray,
    target: np.ndarray,
    l2: float,
    ridge_multipliers: np.ndarray,
) -> tuple[float, np.ndarray]:
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(target.mean())
    centered_design = design - design_mean
    centered_target = target - target_mean
    if l2 > 0.0:
        ridge = np.sqrt(l2 * ridge_multipliers)
        centered_design = np.vstack([centered_design, np.diag(ridge)])
        centered_target = np.concatenate([centered_target, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_design, centered_target, maxiter=40 * design.shape[1])
    intercept = target_mean - float((design_mean @ coefficients).item())
    return intercept, coefficients


def fold_map(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    mapped = {int(record["outer_fold"]): record for record in records}
    if sorted(mapped) != list(range(baseline.OUTER_SPLITS)):
        raise ValueError(f"unexpected frozen outer folds: {sorted(mapped)}")
    return mapped


def test_hpr_dataset(
    train_dataset: hpr.family_grp.Dataset,
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> hpr.family_grp.Dataset:
    return replace(
        train_dataset,
        frame=frame.reset_index(drop=True),
        target=np.zeros(len(weights), dtype=float),
        weights=np.asarray(weights, dtype=float),
    )


def reconstruct_hpr(
    dataset: baseline.expanded.Dataset,
    pooled_dataset: baseline.pooled.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    records: list[dict[str, Any]],
    persisted: np.ndarray,
) -> ReconstructedModel:
    full = np.full(dataset.n, np.nan, dtype=float)
    intercept_prediction = np.full(dataset.n, np.nan, dtype=float)
    block_prediction: dict[str, np.ndarray] = {}
    refit_omission: dict[str, np.ndarray] = {}
    coefficient_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    by_fold = fold_map(records)

    for fold_id, (train, test) in enumerate(folds):
        local = baseline.subset_dataset(pooled_dataset, train, f"sur051_hpr_outer{fold_id}")
        structured = baseline.observatory.family_dataset(local)
        config = hpr_config(by_fold[fold_id])
        train_design = hpr.build_design(structured, config)
        test_structured = test_hpr_dataset(structured, dataset.frame.iloc[test], dataset.weights[test])
        test_design = hpr.build_design(test_structured, config)
        blocks = hpr_blocks(train_design.names)
        intercept, coefficients = fit_hpr_head(
            train_design.values,
            structured.target,
            config.l2,
            train_design.ridge_multipliers,
        )
        original = hpr.fit_model(structured, config, np.arange(structured.n))
        if not np.allclose(coefficients, original.coefficients, atol=1e-11, rtol=1e-11):
            raise AssertionError("generic HPR head does not reproduce source coefficients")
        if not math.isclose(intercept, original.intercept, abs_tol=1e-11, rel_tol=1e-11):
            raise AssertionError("generic HPR head does not reproduce source intercept")

        fold_prediction = intercept + test_design.values @ coefficients
        full[test] = fold_prediction
        intercept_prediction[test] = intercept
        error = float(np.max(np.abs(fold_prediction - persisted[test])))
        reconstruction_rows.append(
            {
                "model": "hierarchical_phase_replay",
                "outer_fold": fold_id,
                "max_abs_prediction_error": error,
            }
        )
        if error > PREDICTION_TOLERANCE:
            raise AssertionError(f"HPR fold {fold_id} reconstruction error {error:.3e}")

        for block, mask in blocks.items():
            block_prediction.setdefault(block, np.full(dataset.n, np.nan, dtype=float))[test] = (
                test_design.values[:, mask] @ coefficients[mask]
            )
            keep = ~mask
            omitted_intercept, omitted_coefficients = fit_hpr_head(
                train_design.values[:, keep],
                structured.target,
                config.l2,
                train_design.ridge_multipliers[keep],
            )
            refit_omission.setdefault(block, np.full(dataset.n, np.nan, dtype=float))[test] = (
                omitted_intercept + test_design.values[:, keep] @ omitted_coefficients
            )
            coefficient_rows.append(
                {
                    "model": "hierarchical_phase_replay",
                    "outer_fold": fold_id,
                    "block": block,
                    "column_count": int(mask.sum()),
                    "active_coefficient_count": int(np.sum(np.abs(coefficients[mask]) > ACTIVE_TOLERANCE)),
                    "coefficient_l2_norm": float(np.linalg.norm(coefficients[mask])),
                }
            )

        block_sum = sum(block_prediction[block][test] for block in blocks)
        if not np.allclose(block_sum, fold_prediction - intercept, atol=1e-11, rtol=1e-11):
            raise AssertionError(f"HPR fold {fold_id} row-level block partition failed")

    return ReconstructedModel(
        full,
        intercept_prediction,
        block_prediction,
        refit_omission,
        coefficient_rows,
        reconstruction_rows,
    )


def reconstruct_rpl(
    dataset: baseline.expanded.Dataset,
    pooled_dataset: baseline.pooled.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    records: list[dict[str, Any]],
    persisted: np.ndarray,
) -> ReconstructedModel:
    full = np.full(dataset.n, np.nan, dtype=float)
    intercept_prediction = np.full(dataset.n, np.nan, dtype=float)
    block_prediction: dict[str, np.ndarray] = {}
    refit_omission: dict[str, np.ndarray] = {}
    coefficient_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    by_fold = fold_map(records)

    for fold_id, (train, test) in enumerate(folds):
        local = baseline.subset_dataset(pooled_dataset, train, f"sur051_rpl_outer{fold_id}")
        geometry = baseline.retained_geometry(local, dataset.family_index)
        shape, ridge = rpl_shape(by_fold[fold_id])
        train_design, layout = repaired_rpl.design_matrix(local.weights, geometry, shape)
        test_design, test_layout = repaired_rpl.design_matrix(dataset.weights[test], geometry, shape)
        if test_layout != layout:
            raise AssertionError("repaired RPL train and test layouts disagree")
        multipliers = repaired_rpl.penalty_multipliers(geometry, layout)
        blocks = rpl_blocks(geometry, shape, layout)
        if train_design.shape[1] != len(multipliers):
            raise AssertionError("RPL design and penalty lengths disagree")
        if train_design.shape[1] != len(next(iter(blocks.values()))):
            raise AssertionError("RPL block partition and design lengths disagree")
        coverage = np.sum(np.stack(list(blocks.values()), axis=0), axis=0)
        if not np.all(coverage == 1):
            raise AssertionError("RPL block partition is not exhaustive and disjoint")

        intercept, aggregate_coefficients, phase_coefficients = repaired_rpl.solve_head(
            train_design,
            local.y,
            ridge,
            multipliers,
            layout,
        )
        coefficients = np.concatenate([aggregate_coefficients, phase_coefficients])
        fold_prediction = intercept + test_design @ coefficients
        full[test] = fold_prediction
        intercept_prediction[test] = intercept
        error = float(np.max(np.abs(fold_prediction - persisted[test])))
        reconstruction_rows.append(
            {
                "model": "retained_power_law_repaired",
                "outer_fold": fold_id,
                "max_abs_prediction_error": error,
            }
        )
        if error > PREDICTION_TOLERANCE:
            raise AssertionError(f"RPL fold {fold_id} reconstruction error {error:.3e}")

        for block, mask in blocks.items():
            block_prediction.setdefault(block, np.full(dataset.n, np.nan, dtype=float))[test] = (
                test_design[:, mask] @ coefficients[mask]
            )
            keep = ~mask
            omitted_layout = repaired_rpl.FeatureLayout(
                aggregate_count=int(np.sum(keep[: layout.aggregate_count])),
                phase_count=int(np.sum(keep[layout.aggregate_count :])),
            )
            omitted_intercept, omitted_aggregate, omitted_phase = repaired_rpl.solve_head(
                train_design[:, keep],
                local.y,
                ridge,
                multipliers[keep],
                omitted_layout,
            )
            omitted_coefficients = np.concatenate([omitted_aggregate, omitted_phase])
            refit_omission.setdefault(block, np.full(dataset.n, np.nan, dtype=float))[test] = (
                omitted_intercept + test_design[:, keep] @ omitted_coefficients
            )
            coefficient_rows.append(
                {
                    "model": "retained_power_law_repaired",
                    "outer_fold": fold_id,
                    "block": block,
                    "column_count": int(mask.sum()),
                    "active_coefficient_count": int(np.sum(np.abs(coefficients[mask]) > ACTIVE_TOLERANCE)),
                    "coefficient_l2_norm": float(np.linalg.norm(coefficients[mask])),
                }
            )

        block_sum = sum(block_prediction[block][test] for block in blocks)
        if not np.allclose(block_sum, fold_prediction - intercept, atol=1e-11, rtol=1e-11):
            mismatch = float(np.max(np.abs(block_sum - (fold_prediction - intercept))))
            raise AssertionError(f"RPL fold {fold_id} row-level block partition failed by {mismatch:.3e}")

    return ReconstructedModel(
        full,
        intercept_prediction,
        block_prediction,
        refit_omission,
        coefficient_rows,
        reconstruction_rows,
    )


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def covariance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean((left - left.mean()) * (right - right.mean())))


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-15 else float("nan")


def paired_bootstrap(
    observed: np.ndarray,
    full: np.ndarray,
    omitted: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(observed)
    changes = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = rng.integers(0, n, size=n)
        changes[index] = rmse(observed[draw], omitted[draw]) - rmse(observed[draw], full[draw])
    low, median, high = np.quantile(changes, [0.025, 0.5, 0.975])
    return {
        "bootstrap_change_low": float(low),
        "bootstrap_change_median": float(median),
        "bootstrap_change_high": float(high),
        "bootstrap_probability_positive": float(np.mean(changes > 0.0)),
    }


def analyze_reconstruction(
    target: str,
    model: str,
    dataset: baseline.expanded.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    reconstructed: ReconstructedModel,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tied_rows, asymmetric_rows, keys = baseline.pair_indices(dataset)
    pair_fold = np.full(len(keys), -1, dtype=int)
    row_fold = np.full(dataset.n, -1, dtype=int)
    for fold_id, (_train, test) in enumerate(folds):
        row_fold[test] = fold_id
    if not np.all(row_fold[tied_rows] == row_fold[asymmetric_rows]):
        raise AssertionError("an exact aggregate-matched pair crosses outer folds")
    intercept_delta = reconstructed.intercept_prediction[asymmetric_rows] - reconstructed.intercept_prediction[tied_rows]
    if not np.allclose(intercept_delta, 0.0, atol=1e-14, rtol=0.0):
        mismatch = float(np.max(np.abs(intercept_delta)))
        raise AssertionError(f"paired rows have different fitted intercepts by up to {mismatch:.3e}")
    pair_fold[:] = row_fold[tied_rows]

    observed_delta = dataset.y[asymmetric_rows] - dataset.y[tied_rows]
    full_delta = reconstructed.full_prediction[asymmetric_rows] - reconstructed.full_prediction[tied_rows]
    block_deltas = {
        block: prediction[asymmetric_rows] - prediction[tied_rows]
        for block, prediction in reconstructed.block_prediction.items()
    }
    summed = np.sum(np.stack(list(block_deltas.values()), axis=0), axis=0)
    if not np.allclose(summed, full_delta, atol=1e-10, rtol=1e-10):
        mismatch = float(np.max(np.abs(summed - full_delta)))
        component_max = max(float(np.max(np.abs(values))) for values in block_deltas.values())
        raise AssertionError(
            f"{target}/{model} block deltas do not sum to full prediction; max mismatch {mismatch:.3e}, "
            f"max block contribution {component_max:.3e}"
        )

    pair_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    asymmetric_mask = ~baseline.expanded.replay_control.tied_rows(dataset.weights)
    full_pair_rmse = rmse(observed_delta, full_delta)
    full_all_rmse = rmse(dataset.y, reconstructed.full_prediction)
    full_asymmetric_rmse = rmse(
        dataset.y[asymmetric_mask],
        reconstructed.full_prediction[asymmetric_mask],
    )
    total_covariance = covariance(full_delta, observed_delta)
    total_variance = float(np.var(full_delta))

    coefficient_frame = pd.DataFrame(reconstructed.coefficient_rows)
    for block_index, block in enumerate(sorted(block_deltas)):
        contribution = block_deltas[block]
        frozen_omitted_delta = full_delta - contribution
        refit_prediction = reconstructed.refit_omission_prediction[block]
        refit_omitted_delta = refit_prediction[asymmetric_rows] - refit_prediction[tied_rows]
        frozen_change = rmse(observed_delta, frozen_omitted_delta) - full_pair_rmse
        refit_change = rmse(observed_delta, refit_omitted_delta) - full_pair_rmse
        block_coefficients = coefficient_frame.loc[coefficient_frame["block"].eq(block)]
        cov = covariance(contribution, observed_delta)
        bootstrap = paired_bootstrap(
            observed_delta,
            full_delta,
            refit_omitted_delta,
            bootstrap_samples,
            BOOTSTRAP_SEED + TARGETS.index(target) * 100 + MODELS.index(model) * 20 + block_index,
        )
        metric_rows.append(
            {
                "target": target,
                "model": model,
                "block": block,
                "n_pairs": len(keys),
                "full_pair_rmse": full_pair_rmse,
                "contribution_rms": float(np.sqrt(np.mean(contribution**2))),
                "contribution_mean": float(np.mean(contribution)),
                "contribution_spearman": baseline.safe_spearman(observed_delta, contribution),
                "contribution_covariance": cov,
                "covariance_fraction": safe_ratio(cov, total_covariance),
                "variance_fraction": safe_ratio(float(np.var(contribution)), total_variance),
                "frozen_omission_pair_rmse": rmse(observed_delta, frozen_omitted_delta),
                "frozen_omission_pair_rmse_change": frozen_change,
                "refit_omission_pair_rmse": rmse(observed_delta, refit_omitted_delta),
                "refit_omission_pair_rmse_change": refit_change,
                "full_all_rmse": full_all_rmse,
                "refit_omission_all_rmse": rmse(dataset.y, refit_prediction),
                "refit_omission_all_rmse_change": rmse(dataset.y, refit_prediction) - full_all_rmse,
                "full_asymmetric_rmse": full_asymmetric_rmse,
                "refit_omission_asymmetric_rmse": rmse(
                    dataset.y[asymmetric_mask],
                    refit_prediction[asymmetric_mask],
                ),
                "refit_omission_asymmetric_rmse_change": (
                    rmse(
                        dataset.y[asymmetric_mask],
                        refit_prediction[asymmetric_mask],
                    )
                    - full_asymmetric_rmse
                ),
                "mean_active_coefficient_count": float(block_coefficients["active_coefficient_count"].mean()),
                "mean_coefficient_l2_norm": float(block_coefficients["coefficient_l2_norm"].mean()),
                **bootstrap,
            }
        )
        for pair_index, key in enumerate(keys):
            pair_rows.append(
                {
                    "target": target,
                    "model": model,
                    "block": block,
                    "phase_correspondence_key": key,
                    "outer_fold": int(pair_fold[pair_index]),
                    "observed_delta": observed_delta[pair_index],
                    "full_predicted_delta": full_delta[pair_index],
                    "block_contribution": contribution[pair_index],
                    "frozen_omission_delta": frozen_omitted_delta[pair_index],
                    "refit_omission_delta": refit_omitted_delta[pair_index],
                }
            )
        for fold_id in range(len(folds)):
            local = pair_fold == fold_id
            fold_covariance = covariance(contribution[local], observed_delta[local])
            fold_rows.append(
                {
                    "target": target,
                    "model": model,
                    "block": block,
                    "outer_fold": fold_id,
                    "n_pairs": int(local.sum()),
                    "contribution_mean": float(np.mean(contribution[local])),
                    "contribution_covariance": fold_covariance,
                    "contribution_covariance_sign": int(np.sign(fold_covariance)),
                    "full_pair_rmse": rmse(observed_delta[local], full_delta[local]),
                    "refit_omission_pair_rmse": rmse(observed_delta[local], refit_omitted_delta[local]),
                    "refit_omission_pair_rmse_change": (
                        rmse(
                            observed_delta[local],
                            refit_omitted_delta[local],
                        )
                        - rmse(observed_delta[local], full_delta[local])
                    ),
                }
            )
    return metric_rows, pair_rows, fold_rows


def numeric_gate(metrics: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, block), group in metrics.groupby(["model", "block"], sort=True):
        by_target = {str(row.target): row for row in group.itertuples(index=False)}
        if set(by_target) != set(TARGETS):
            continue
        target_details: dict[str, Any] = {}
        positive_both = True
        fold_gate = True
        bootstrap_gate = False
        for target in TARGETS:
            row = by_target[target]
            local_folds = folds.loc[folds["model"].eq(model) & folds["block"].eq(block) & folds["target"].eq(target)]
            positive_folds = int(np.sum(local_folds["refit_omission_pair_rmse_change"] > 0.0))
            positive_both &= float(row.refit_omission_pair_rmse_change) > 0.0
            target_details[target] = {
                "refit_change": float(row.refit_omission_pair_rmse_change),
                "positive_folds": positive_folds,
                "bootstrap_low": float(row.bootstrap_change_low),
                "bootstrap_high": float(row.bootstrap_change_high),
            }
            bootstrap_gate |= float(row.bootstrap_change_low) > 0.0
        positive_fold_counts = [target_details[target]["positive_folds"] for target in TARGETS]
        fold_gate = max(positive_fold_counts) == baseline.OUTER_SPLITS and min(positive_fold_counts) >= 2
        non_tv = block != "global_phase_tv"
        rows.append(
            {
                "model": model,
                "block": block,
                "positive_on_both_targets": positive_both,
                "fold_stability_gate": fold_gate,
                "bootstrap_gate": bootstrap_gate,
                "non_tv_block": non_tv,
                "numeric_gate": positive_both and fold_gate and bootstrap_gate and non_tv,
                "target_details": json.dumps(target_details, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def render(metrics: pd.DataFrame, output_dir: Path) -> None:
    rows = len(TARGETS)
    cols = len(MODELS)
    titles = [f"{target} · {model}" for target in TARGETS for model in MODELS]
    figure = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.12)
    for target_index, target in enumerate(TARGETS, start=1):
        for model_index, model in enumerate(MODELS, start=1):
            local = metrics.loc[metrics["target"].eq(target) & metrics["model"].eq(model)].sort_values(
                "refit_omission_pair_rmse_change"
            )
            colors = [
                "#1a9850" if value > 0 else "#d73027"
                for value in local["refit_omission_pair_rmse_change"].to_numpy(float)
            ]
            figure.add_trace(
                go.Bar(
                    x=local["refit_omission_pair_rmse_change"],
                    y=local["block"],
                    orientation="h",
                    marker_color=colors,
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": local["bootstrap_change_high"] - local["refit_omission_pair_rmse_change"],
                        "arrayminus": local["refit_omission_pair_rmse_change"] - local["bootstrap_change_low"],
                    },
                    customdata=np.column_stack(
                        [
                            local["contribution_rms"],
                            local["contribution_spearman"],
                            local["covariance_fraction"],
                            local["mean_active_coefficient_count"],
                        ]
                    ),
                    hovertemplate=(
                        "%{y}<br>refit omission ΔRMSE=%{x:.6f}"
                        "<br>contribution RMS=%{customdata[0]:.6f}"
                        "<br>contribution Spearman=%{customdata[1]:.3f}"
                        "<br>covariance fraction=%{customdata[2]:.3f}"
                        "<br>mean active coefficients=%{customdata[3]:.1f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=target_index,
                col=model_index,
            )
            figure.add_vline(x=0.0, line_width=1, line_color="#17324d", row=target_index, col=model_index)
    figure.update_layout(
        title="Exact-pair phase attribution: refit-omission change in RMSE",
        template="plotly_white",
        height=950,
        width=1500,
        margin={"l": 190, "r": 50, "t": 100, "b": 70},
        font={"family": "Avenir Next, sans-serif", "color": "#17324d"},
    )
    figure.update_xaxes(title_text="Omitted-block RMSE - full-model RMSE (BPB)")
    figure.write_html(
        output_dir / "phase_block_ablation.html",
        include_plotlyjs=True,
        full_html=True,
        config=PLOT_CONFIG,
    )


def write_report(
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    reconstruction: pd.DataFrame,
    gates: pd.DataFrame,
    protocol: dict[str, Any],
    output_dir: Path,
) -> None:
    lines = [
        "# HPR/RPL Phase-Block Attribution",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Parent baseline: `{protocol['parent_protocol']}`",
        f"- Paired bootstrap samples: {protocol['bootstrap_samples']:,}",
        "- Positive omission change means the full model needed that block.",
        "- Variance and covariance fractions are descriptive and need not sum to one because blocks are correlated.",
        "",
        "## Reconstruction",
        "",
        "| Target | Model | Maximum absolute OOF mismatch |",
        "|:--|:--|--:|",
    ]
    reconstruction_summary = reconstruction.groupby(["target", "model"], as_index=False)[
        "max_abs_prediction_error"
    ].max()
    for row in reconstruction_summary.itertuples(index=False):
        lines.append(f"| {row.target} | {row.model} | {row.max_abs_prediction_error:.3e} |")

    lines.extend(
        [
            "",
            "## Refit-Omission Results",
            "",
            "| Target | Model | Block | Pair ΔRMSE | 95% interval | Positive folds | Covariance fraction |",
            "|:--|:--|:--|--:|:--|--:|--:|",
        ]
    )
    ordered_metrics = metrics.sort_values(
        ["target", "model", "refit_omission_pair_rmse_change"],
        ascending=[True, True, False],
    )
    for row in ordered_metrics.itertuples(index=False):
        local = folds.loc[folds["target"].eq(row.target) & folds["model"].eq(row.model) & folds["block"].eq(row.block)]
        positive_folds = int(np.sum(local["refit_omission_pair_rmse_change"] > 0.0))
        lines.append(
            f"| {row.target} | {row.model} | {row.block} | "
            f"{row.refit_omission_pair_rmse_change:+.6f} | "
            f"[{row.bootstrap_change_low:+.6f}, {row.bootstrap_change_high:+.6f}] | "
            f"{positive_folds}/{len(local)} | {row.covariance_fraction:+.3f} |"
        )

    lines.extend(["", "## Frozen Numeric Gate", ""])
    passing = gates.loc[gates["numeric_gate"]]
    if passing.empty:
        lines.append(
            "No block passes the preregistered numeric gate. This is a negative attribution result unless "
            "manual review finds a protocol error."
        )
    else:
        lines.append("The following blocks pass the numeric portion of the preregistered gate:")
        lines.append("")
        for row in passing.itertuples(index=False):
            lines.append(f"- `{row.model}/{row.block}`")
        lines.extend(
            [
                "",
                "Passing the numeric gate does not promote a model. The block still needs a common physical "
                "interpretation on both targets and a preregistered WSD80 analogue.",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This diagnostic identifies predictive necessity conditional on the frozen nonlinear configurations. "
            "It does not establish that a block is causal, that HPR's transition law is correct, or that combining "
            "HPR and RPL will improve policy selection. A new mechanism requires a separate preregistration.",
            "",
            "Artifacts:",
            "",
            "- `block_metrics.csv`",
            "- `fold_block_metrics.csv`",
            "- `pair_block_contributions.csv`",
            "- `coefficient_diagnostics.csv`",
            "- `reconstruction_checks.csv`",
            "- `numeric_gate.csv`",
            "- `phase_block_ablation.html`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    protocol = protocol_payload(args.bootstrap_samples)
    write_json(args.output_dir / "protocol.json", protocol)
    if not args.force and completed(args.output_dir, str(protocol["protocol_hash"])):
        print(f"skip complete attribution {protocol['protocol_hash']}", flush=True)
        return

    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        print(f"load expanded 300M {target}", flush=True)
        dataset = baseline.expanded.load_300m(target)
        folds = baseline.correspondence_folds(dataset.frame, baseline.OUTER_SEED, baseline.OUTER_SPLITS)
        pooled_dataset = baseline.as_pooled(dataset)
        for model in MODELS:
            print(f"reconstruct {target}/{model}", flush=True)
            cell = cell_dir(target, model)
            records = json.loads((cell / "fold_selections.json").read_text())
            persisted_frame = pd.read_csv(cell / "predictions.csv").sort_values("row_index")
            persisted = persisted_frame["predicted"].to_numpy(float)
            persisted_folds = persisted_frame["outer_fold"].to_numpy(int)
            expected_folds = np.full(dataset.n, -1, dtype=int)
            for fold_id, (_train, test) in enumerate(folds):
                expected_folds[test] = fold_id
            if not np.array_equal(persisted_folds, expected_folds):
                raise AssertionError(f"{target}/{model} persisted fold assignment changed")

            if model == "hierarchical_phase_replay":
                result = reconstruct_hpr(dataset, pooled_dataset, folds, records, persisted)
            else:
                result = reconstruct_rpl(dataset, pooled_dataset, folds, records, persisted)
            if not np.isfinite(result.full_prediction).all():
                raise AssertionError(f"{target}/{model} has incomplete reconstructed predictions")
            for prediction in [*result.block_prediction.values(), *result.refit_omission_prediction.values()]:
                if not np.isfinite(prediction).all():
                    raise AssertionError(f"{target}/{model} has incomplete block predictions")

            local_metrics, local_pairs, local_folds = analyze_reconstruction(
                target,
                model,
                dataset,
                folds,
                result,
                args.bootstrap_samples,
            )
            metric_rows.extend(local_metrics)
            pair_rows.extend(local_pairs)
            fold_rows.extend(local_folds)
            coefficient_rows.extend({"target": target, **row} for row in result.coefficient_rows)
            reconstruction_rows.extend({"target": target, **row} for row in result.reconstruction_rows)

    metrics = pd.DataFrame(metric_rows)
    pairs = pd.DataFrame(pair_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    gates = numeric_gate(metrics, fold_metrics)

    metrics.to_csv(args.output_dir / "block_metrics.csv", index=False)
    pairs.to_csv(args.output_dir / "pair_block_contributions.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "fold_block_metrics.csv", index=False)
    coefficients.to_csv(args.output_dir / "coefficient_diagnostics.csv", index=False)
    reconstruction.to_csv(args.output_dir / "reconstruction_checks.csv", index=False)
    gates.to_csv(args.output_dir / "numeric_gate.csv", index=False)
    render(metrics, args.output_dir)
    write_report(metrics, fold_metrics, reconstruction, gates, protocol, args.output_dir)
    write_json(
        args.output_dir / "complete.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "rows": {
                "block_metrics": len(metrics),
                "pair_block_contributions": len(pairs),
                "fold_block_metrics": len(fold_metrics),
            },
            "numeric_gate_passes": gates.loc[gates["numeric_gate"], ["model", "block"]].to_dict("records"),
        },
    )
    print(f"completed attribution {protocol['protocol_hash']}", flush=True)


if __name__ == "__main__":
    main()
