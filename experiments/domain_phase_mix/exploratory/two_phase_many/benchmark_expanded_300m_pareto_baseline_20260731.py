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
"""Freeze the expanded 300M surrogate Pareto baseline on correspondence-safe folds.

The discovery panel contains 520 observations but only 280 independently
sampled aggregate coordinates. Aggregate-matched tied and asymmetric rows must
therefore remain together in both outer evaluation folds and nested
hyperparameter-selection folds. This harness calls the existing source model
implementations while replacing their heterogeneous fold protocols with one
frozen correspondence-grouped protocol.

Each target/model cell is independently resumable. A cell is reused only when
its completion record matches the hash of this script, its model sources, the
input data, and the frozen protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.optimize import lsq_linear, minimize
from scipy.special import softmax
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as expanded,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    hierarchical_band_model_20260726 as hierarchical_band,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as retained,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731"
TARGETS = ("uncheatable", "table9")
MODEL_IDS = (
    "canonical_dsp",
    "effective_exposure_dsp",
    "effective_exposure_geometry",
    "separate_heads",
    "original_grp",
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_replay",
    "hpr_band",
    "retained_power_law",
)
DSP_MODEL_IDS = {
    "canonical_dsp": "canonical",
    "effective_exposure_dsp": "effective_exposure",
    "effective_exposure_geometry": "effective_exposure_geometry",
}
REFERENCE_ONLY_MODELS = {"hpr_band"}
OUTER_SEED = 7310
INNER_SEED_BASE = 731_000
FULL_FIT_SEED = 731_999
OUTER_SPLITS = 3
INNER_SPLITS = 3
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
OPTIMIZER_STARTS = 16
OPTIMIZER_MAX_ITERATIONS = 750
OPTIMIZER_LOGIT_BOUND = 12.0
ACTIVE_COEFFICIENT_TOLERANCE = 1e-10
PROTOCOL_VERSION = "expanded-300m-pareto-v1-correspondence-nested"

SOURCE_FILES = (
    Path(__file__),
    Path(expanded.__file__),
    Path(observatory.__file__),
    Path(hierarchical_grp.__file__),
    Path(hierarchical_band.__file__),
    Path(retained.__file__),
    Path(observatory.coverage_dsp.__file__),
    Path(observatory.separate_heads.__file__),
    Path(observatory.compact_retained.__file__),
    Path(observatory.family_grp.__file__),
    Path(observatory.legacy_exporter.__file__),
    Path(observatory.grp_calibration.__file__),
    Path(pooled.__file__),
    expanded.PACKET,
    expanded.ONE_PHASE_SOURCE,
)


class Predictable(Protocol):
    def predict(self, weights: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class FitResult:
    model: Any
    selection: dict[str, Any]
    parameter_diagnostics: dict[str, int | float | str]


def parse_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--models", default=",".join(MODEL_IDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-splits", type=int, default=OUTER_SPLITS)
    parser.add_argument("--inner-splits", type=int, default=INNER_SPLITS)
    parser.add_argument("--rpl-workers", type=int, default=1)
    parser.add_argument("--optimizer-starts", type=int, default=OPTIMIZER_STARTS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-optimum", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
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


def protocol_payload(outer_splits: int, inner_splits: int, optimizer_starts: int) -> dict[str, Any]:
    missing = [str(path) for path in SOURCE_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing protocol sources: {missing}")
    payload = {
        "version": PROTOCOL_VERSION,
        "targets": TARGETS,
        "models": MODEL_IDS,
        "reference_only_models": sorted(REFERENCE_ONLY_MODELS),
        "outer_seed": OUTER_SEED,
        "inner_seed_base": INNER_SEED_BASE,
        "full_fit_seed": FULL_FIT_SEED,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "optimizer_starts": optimizer_starts,
        "lower_tail_fraction": LOWER_TAIL_FRACTION,
        "lower_tail_min_count": LOWER_TAIL_MIN_COUNT,
        "optimizer_max_iterations": OPTIMIZER_MAX_ITERATIONS,
        "optimizer_logit_bound": OPTIMIZER_LOGIT_BOUND,
        "source_hashes": {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in SOURCE_FILES},
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def as_pooled(dataset: expanded.Dataset) -> pooled.Dataset:
    return pooled.Dataset(
        name=dataset.name,
        frame=dataset.frame.copy(),
        y=np.asarray(dataset.y, dtype=float),
        weights=np.asarray(dataset.weights, dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        domain_names=list(dataset.domain_names),
    )


def subset_dataset(dataset: pooled.Dataset, indices: np.ndarray, suffix: str) -> pooled.Dataset:
    return pooled.Dataset(
        name=f"{dataset.name}_{suffix}",
        frame=dataset.frame.iloc[indices].reset_index(drop=True),
        y=np.asarray(dataset.y[indices], dtype=float),
        weights=np.asarray(dataset.weights[indices], dtype=float),
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        domain_names=list(dataset.domain_names),
    )


def correspondence_folds(
    frame: pd.DataFrame,
    seed: int,
    n_splits: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    groups = frame["phase_correspondence_key"].astype(str)
    resolved_splits = min(n_splits, groups.nunique())
    if resolved_splits < 2:
        raise ValueError("at least two correspondence groups are required")
    folds = expanded.grouped_folds(frame, seed, resolved_splits)
    assert_fold_integrity(frame, folds)
    return folds


def assert_fold_integrity(
    frame: pd.DataFrame,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> None:
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()
    seen = np.zeros(len(frame), dtype=int)
    group_to_fold: dict[str, int] = {}
    for fold_id, (train, test) in enumerate(folds):
        if np.intersect1d(train, test).size:
            raise ValueError(f"fold {fold_id} has train/test row overlap")
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        overlap = train_groups & test_groups
        if overlap:
            raise ValueError(f"fold {fold_id} splits correspondence groups: {sorted(overlap)[:5]}")
        seen[test] += 1
        for group in test_groups:
            previous = group_to_fold.setdefault(group, fold_id)
            if previous != fold_id:
                raise ValueError(f"correspondence group {group} appears in multiple test folds")
    if not np.all(seen == 1):
        raise ValueError("outer test folds must cover every row exactly once")


def prepare_target(
    output_dir: Path,
    target: str,
    outer_splits: int,
) -> tuple[expanded.Dataset, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    dataset = expanded.load_300m(target)
    folds = correspondence_folds(dataset.frame, OUTER_SEED, outer_splits)
    assignment = np.full(dataset.n, -1, dtype=int)
    for fold_id, (_train, test) in enumerate(folds):
        assignment[test] = fold_id
    if np.any(assignment < 0):
        raise ValueError(f"incomplete fold assignment for {target}")

    tied = expanded.replay_control.tied_rows(dataset.weights)
    beta0 = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    aggregate = beta0 * dataset.weights[:, 0, :] + (1.0 - beta0) * dataset.weights[:, 1, :]
    phase_tv = 0.5 * np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]).sum(axis=1)
    manifest = dataset.frame.copy()
    manifest.insert(0, "row_index", np.arange(dataset.n))
    manifest["outer_fold"] = assignment
    manifest["physical_tied"] = tied
    manifest["phase_tv"] = phase_tv
    manifest["aggregate_hhi"] = np.sum(aggregate**2, axis=1)
    manifest["target_value"] = dataset.y
    manifest.to_csv(output_dir / f"rows_{target}.csv", index=False)

    fold_rows = []
    for fold_id, (train, test) in enumerate(folds):
        fold_rows.append(
            {
                "target": target,
                "outer_fold": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_groups": dataset.frame.iloc[train]["phase_correspondence_key"].nunique(),
                "test_groups": dataset.frame.iloc[test]["phase_correspondence_key"].nunique(),
            }
        )
    pd.DataFrame(fold_rows).to_csv(output_dir / f"folds_{target}.csv", index=False)
    return dataset, folds


def acceptance_gate() -> dict[str, Any]:
    return {
        "frozen_before_candidate_evaluation": True,
        "primary_panel": "expanded high-TPP 300M / 6B-token, 39 buckets",
        "primary_targets": list(TARGETS),
        "candidate_requirements": {
            "phase_sensitive_diagnostic": "improve beyond paired-bootstrap uncertainty on at least one target",
            "other_target": "preserve",
            "maximum_core_oof_rmse_ratio": 1.05,
            "wsd80_programming_languages_observed_gain_bpb": 0.009594,
            "wsd80_observed_optimum": {"phase_0": 0.10, "phase_1": 0.50},
            "wsd80_negative_controls": "phase gain must shrink toward zero on broad-text targets",
            "raw_optimum": "plausible and stable before deployment regularization",
        },
        "insufficient_evidence": [
            "pooled RMSE alone",
            "Spearman alone",
            "predicted optimum value alone",
            "HPR-band improvement as a headline model",
        ],
    }


def prepare_protocol(
    output_dir: Path,
    outer_splits: int,
    inner_splits: int,
    optimizer_starts: int,
) -> tuple[dict[str, Any], dict[str, expanded.Dataset], dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(outer_splits, inner_splits, optimizer_starts)
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "acceptance_gate.json", acceptance_gate())
    datasets: dict[str, expanded.Dataset] = {}
    folds: dict[str, tuple[tuple[np.ndarray, np.ndarray], ...]] = {}
    for target in TARGETS:
        datasets[target], folds[target] = prepare_target(output_dir, target, outer_splits)
    return protocol, datasets, folds


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.std(observed) <= 0.0 or np.std(predicted) <= 0.0:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def calibration_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    centered = predicted - np.mean(predicted)
    denominator = float(centered @ centered)
    if denominator <= 1e-18:
        return float("nan")
    return float(centered @ (observed - np.mean(observed)) / denominator)


def scalar_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - observed
    optimism = observed - predicted
    tail_count = min(len(observed), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))))
    tail = np.argsort(predicted)[:tail_count]
    tail_error = error[tail]
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "spearman": safe_spearman(observed, predicted),
        "calibration_slope": calibration_slope(observed, predicted),
        "low_tail_rmse": float(np.sqrt(np.mean(tail_error**2))),
        "lower_tail_optimism": float(np.mean(np.maximum(-tail_error, 0.0))),
        "optimism_over_0p05": int(np.sum(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
    }


def prefixed_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    values = scalar_metrics(observed[mask], predicted[mask])
    return {f"{prefix}_{key}": value for key, value in values.items()}


def fold_regret(
    observed: np.ndarray,
    predicted: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    eligible_mask: np.ndarray,
    k: int,
) -> float:
    regrets = []
    for _train, test in folds:
        eligible = test[eligible_mask[test]]
        if not len(eligible):
            continue
        selected = eligible[np.argsort(predicted[eligible])[: min(k, len(eligible))]]
        regrets.append(float(np.min(observed[selected]) - np.min(observed[eligible])))
    return float(np.mean(regrets)) if regrets else float("nan")


def metric_summary(
    dataset: expanded.Dataset,
    predicted: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> dict[str, float | int]:
    tied = expanded.replay_control.tied_rows(dataset.weights)
    asymmetric = ~tied
    metrics = {
        **prefixed_metrics(dataset.y, predicted, np.ones(dataset.n, dtype=bool), "all"),
        **prefixed_metrics(dataset.y, predicted, tied, "tied"),
        **prefixed_metrics(dataset.y, predicted, asymmetric, "asymmetric"),
    }
    for k in (1, 3, 5):
        metrics[f"all_regret_at_{k}"] = fold_regret(
            dataset.y,
            predicted,
            folds,
            np.ones(dataset.n, dtype=bool),
            k,
        )
        metrics[f"asymmetric_regret_at_{k}"] = fold_regret(dataset.y, predicted, folds, asymmetric, k)
    return metrics


def pair_indices(dataset: expanded.Dataset) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = dataset.frame.reset_index()
    indexed = frame.set_index(["phase_correspondence_key", "policy_family"])["index"]
    keys = sorted(
        set(frame.loc[frame["policy_family"].eq("single_phase"), "phase_correspondence_key"].astype(str))
        & set(frame.loc[frame["policy_family"].eq("two_phase"), "phase_correspondence_key"].astype(str))
    )
    tied = np.asarray([indexed.loc[(key, "single_phase")] for key in keys], dtype=int)
    asymmetric = np.asarray([indexed.loc[(key, "two_phase")] for key in keys], dtype=int)
    genuinely_asymmetric = ~expanded.replay_control.tied_rows(dataset.weights[asymmetric])
    return (
        tied[genuinely_asymmetric],
        asymmetric[genuinely_asymmetric],
        [key for key, keep in zip(keys, genuinely_asymmetric, strict=True) if keep],
    )


def pair_summary(
    dataset: expanded.Dataset,
    predicted: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    tied, asymmetric, keys = pair_indices(dataset)
    observed_delta = dataset.y[asymmetric] - dataset.y[tied]
    predicted_delta = predicted[asymmetric] - predicted[tied]
    summary = {
        "n_pairs": len(keys),
        "delta_rmse": float(np.sqrt(np.mean((predicted_delta - observed_delta) ** 2))),
        "delta_spearman": safe_spearman(observed_delta, predicted_delta),
        "delta_bias": float(np.mean(predicted_delta - observed_delta)),
        "sign_accuracy": float(np.mean(np.sign(predicted_delta) == np.sign(observed_delta))),
    }
    rows = pd.DataFrame(
        {
            "phase_correspondence_key": keys,
            "tied_row": tied,
            "asymmetric_row": asymmetric,
            "tied_observed": dataset.y[tied],
            "asymmetric_observed": dataset.y[asymmetric],
            "observed_delta": observed_delta,
            "tied_predicted": predicted[tied],
            "asymmetric_predicted": predicted[asymmetric],
            "predicted_delta": predicted_delta,
        }
    )
    return summary, rows


def candidate_score(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    return float(np.sqrt(np.mean((predicted - observed) ** 2))), -safe_spearman(observed, predicted)


def select_candidates(
    dataset: pooled.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    candidates: list[tuple[Any, dict[str, Any]]],
    fit: Callable[[pooled.Dataset, np.ndarray, Any], Any],
    predict: Callable[[Any, pooled.Dataset, np.ndarray], np.ndarray],
) -> tuple[Any, list[dict[str, Any]]]:
    rows = []
    best: tuple[float, float, int, Any] | None = None
    for candidate_index, (candidate, metadata) in enumerate(candidates):
        oof = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds:
            fitted = fit(dataset, train, candidate)
            oof[test] = predict(fitted, dataset, dataset.weights[test])
        if not np.isfinite(oof).all():
            raise RuntimeError(f"candidate {metadata} produced incomplete predictions")
        rmse, negative_spearman = candidate_score(dataset.y, oof)
        row = {
            "candidate_index": candidate_index,
            **metadata,
            "oof_rmse": rmse,
            "oof_spearman": -negative_spearman,
        }
        rows.append(row)
        scored = (rmse, negative_spearman, candidate_index, candidate)
        if best is None or scored[:3] < best[:3]:
            best = scored
    if best is None:
        raise RuntimeError("no hyperparameter candidates were scored")
    return best[3], rows


def hpr_selection(
    dataset: pooled.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> tuple[hierarchical_grp.Config, list[hierarchical_grp.Config], dict[str, Any]]:
    structured = observatory.family_dataset(dataset)
    shapes = observatory.hierarchical_phase_replay_shape_candidates(observatory.TWO_PHASE)
    baseline_configs = hierarchical_grp.baseline_configs(shapes)
    _baseline, _prediction, baseline_rows = hierarchical_grp.score_configs(
        structured,
        baseline_configs,
        list(folds),
    )
    best_by_shape: dict[int, float] = {}
    for row in baseline_rows:
        shape_index = int(row["shape_index"])
        best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), float(row["rmse"]))
    shape_indices = [
        shape_index
        for shape_index, _score in sorted(best_by_shape.items(), key=lambda item: item[1])[
            : observatory.HIERARCHICAL_PHASE_REPLAY_TOP_SHAPES
        ]
    ]
    structural = hierarchical_grp.structural_configs(
        hierarchical_grp.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
        shapes,
        shape_indices,
    )
    selected, _prediction, candidate_rows = hierarchical_grp.score_configs(
        structured,
        structural,
        list(folds),
    )
    detail = {
        "top_shape_indices": shape_indices,
        "baseline_shape_screen": baseline_rows,
        "candidate_sweep": candidate_rows,
    }
    return selected, structural, detail


def retained_geometry(dataset: pooled.Dataset, family_index: np.ndarray) -> retained.Geometry:
    phase_0_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    return retained.Geometry(
        c0=np.asarray(dataset.c0, dtype=float),
        c1=np.asarray(dataset.c1, dtype=float),
        phase_0_fraction=phase_0_fraction,
        family_index=np.asarray(family_index, dtype=int),
    )


def numeric_scalar_count(value: Any) -> int:
    if isinstance(value, (bool, str, Enum, Path)) or value is None:
        return 0
    if isinstance(value, (int, float, np.number)):
        return 1
    if isinstance(value, np.ndarray):
        return int(value.size)
    if is_dataclass(value):
        return sum(numeric_scalar_count(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, dict):
        return sum(numeric_scalar_count(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(numeric_scalar_count(item) for item in value)
    return 0


def active_count(values: np.ndarray) -> int:
    return int(np.sum(np.abs(np.asarray(values, dtype=float)) > ACTIVE_COEFFICIENT_TOLERANCE))


def parameter_diagnostics(model_id: str, model: Any) -> dict[str, int | float | str]:
    nonlinear = 0
    linear_nominal = 0
    linear_active = 0
    if model_id in DSP_MODEL_IDS:
        nonlinear = numeric_scalar_count(model.base.params)
        coefficients = np.concatenate(
            [
                np.asarray(model.base.benefit_coef),
                np.asarray(model.base.penalty_coef),
                np.asarray(model.coverage_coef),
            ]
        )
        linear_nominal = len(coefficients)
        linear_active = active_count(coefficients)
    elif model_id == "separate_heads":
        nonlinear = int(model.mu0.size + model.mu1.size)
        linear_nominal = int(model.coefficients.size)
        linear_active = active_count(model.coefficients)
    elif model_id == "original_grp":
        nonlinear = numeric_scalar_count(model.params)
        coefficients = np.asarray(model.coef_, dtype=float)
        linear_nominal = len(coefficients)
        linear_active = active_count(coefficients)
    elif model_id == "compact_retained_state":
        nonlinear = numeric_scalar_count(model.shape)
        coefficients = np.concatenate([np.asarray(model.signal_coef), np.asarray(model.replay_coef)])
        linear_nominal = len(coefficients)
        linear_active = active_count(coefficients)
    elif model_id == "bucket_family_grp":
        nonlinear = numeric_scalar_count(model.shape)
        coefficients = np.asarray(model.head.coefficients)
        linear_nominal = len(coefficients)
        linear_active = active_count(coefficients)
    elif model_id == "hierarchical_phase_replay":
        nonlinear = numeric_scalar_count(model.config.shape)
        coefficients = np.asarray(model.coefficients)
        linear_nominal = len(coefficients)
        linear_active = active_count(coefficients)
    elif model_id == "hpr_band":
        member_diagnostics = [parameter_diagnostics("hierarchical_phase_replay", fitted) for fitted in model.fitted]
        nonlinear = sum(int(row["nonlinear_parameter_count"]) for row in member_diagnostics)
        linear_nominal = sum(int(row["linear_parameter_count"]) for row in member_diagnostics)
        linear_active = sum(int(row["active_linear_parameter_count"]) for row in member_diagnostics)
        nonlinear += max(len(model.members) - 1, 0)
    elif model_id == "retained_power_law":
        nonlinear = numeric_scalar_count(model.shape)
        coefficients = np.asarray(model.coefficients)
        linear_nominal = len(coefficients)
        linear_active = active_count(coefficients)
    else:
        raise ValueError(f"unsupported model diagnostics for {model_id}")
    nominal = 1 + nonlinear + linear_nominal
    active_proxy = 1 + nonlinear + linear_active
    return {
        "nominal_parameter_count": nominal,
        "nonlinear_parameter_count": nonlinear,
        "linear_parameter_count": linear_nominal,
        "active_linear_parameter_count": linear_active,
        "effective_df_active_set_proxy": active_proxy,
        "effective_df_note": "intercept + selected nonlinear scalars + active constrained linear coefficients",
    }


def fit_model(
    model_id: str,
    dataset: pooled.Dataset,
    inner_folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    target: str,
    family_index: np.ndarray,
    rpl_workers: int,
) -> FitResult:
    all_indices = np.arange(dataset.n)
    if model_id in DSP_MODEL_IDS:
        model = observatory.dsp_fit(
            dataset,
            all_indices,
            DSP_MODEL_IDS[model_id],
            observatory.TWO_PHASE,
        )
        selection = {
            "model_source": "benchmark_nested_coverage_dsp.fit_model",
            "variant": DSP_MODEL_IDS[model_id],
            "selection": "nonlinear profile fit on the outer training rows; no discrete external grid",
        }
    elif model_id == "separate_heads":
        candidates = [(l2, {"l2": float(l2)}) for l2 in observatory.SEPARATE_L2_GRID]
        selected, sweep = select_candidates(
            dataset,
            inner_folds,
            candidates,
            lambda local, indices, l2: observatory.separate_fit(
                local,
                indices,
                float(l2),
                observatory.TWO_PHASE,
            ),
            lambda fitted, local, weights: observatory.separate_predict(
                fitted,
                local,
                weights,
                observatory.TWO_PHASE,
            ),
        )
        model = observatory.separate_fit(dataset, all_indices, float(selected), observatory.TWO_PHASE)
        selection = {"l2": float(selected), "candidate_sweep": sweep}
    elif model_id == "original_grp":
        candidates = [(l2, {"l2": float(l2)}) for l2 in observatory.legacy_exporter.GRP_L2_GRID]
        selected, sweep = select_candidates(
            dataset,
            inner_folds,
            candidates,
            lambda local, indices, l2: observatory.grp_300m_fit(
                local,
                indices,
                float(l2),
                observatory.TWO_PHASE,
            ),
            lambda fitted, _local, weights: fitted.predict(weights),
        )
        model = observatory.grp_300m_fit(dataset, all_indices, float(selected), observatory.TWO_PHASE)
        selection = {"l2": float(selected), "candidate_sweep": sweep}
    elif model_id == "compact_retained_state":
        candidates = [(l2, {"l2": float(l2)}) for l2 in observatory.COMPACT_L2_GRID]
        selected, sweep = select_candidates(
            dataset,
            inner_folds,
            candidates,
            lambda local, indices, l2: observatory.compact_fit(
                local,
                indices,
                float(l2),
                observatory.TWO_PHASE,
            ),
            lambda fitted, _local, weights: fitted.predict(weights),
        )
        model = observatory.compact_fit(dataset, all_indices, float(selected), observatory.TWO_PHASE)
        selection = {
            "l2": float(selected),
            "candidate_sweep": sweep,
            "selected_shape": asdict(model.shape),
        }
    elif model_id == "bucket_family_grp":
        candidates = [
            (
                (shape, l2),
                {
                    "shape_index": shape_index,
                    "shape": asdict(shape),
                    "l2": float(l2),
                },
            )
            for shape_index, shape in enumerate(observatory.bucket_shape_candidates(observatory.TWO_PHASE))
            for l2 in observatory.BUCKET_FAMILY_L2_GRID
        ]
        selected, sweep = select_candidates(
            dataset,
            inner_folds,
            candidates,
            lambda local, indices, candidate: observatory.bucket_fit(
                local,
                indices,
                candidate[0],
                float(candidate[1]),
            ),
            lambda fitted, _local, weights: fitted.predict(weights),
        )
        shape, l2 = selected
        model = observatory.bucket_fit(dataset, all_indices, shape, float(l2))
        selection = {"shape": asdict(shape), "l2": float(l2), "candidate_sweep": sweep}
    elif model_id in {"hierarchical_phase_replay", "hpr_band"}:
        selected, candidates, detail = hpr_selection(dataset, inner_folds)
        structured = observatory.family_dataset(dataset)
        if model_id == "hierarchical_phase_replay":
            model = hierarchical_grp.fit_model(structured, selected, all_indices)
            selection = {"selected_config": asdict(selected), **detail}
        else:
            model, band_detail = hierarchical_band.build_band(
                structured,
                candidates,
                list(inner_folds),
                target,
                all_indices,
            )
            selection = {
                "reference_only": True,
                "single_config_argmin": asdict(selected),
                "band": band_detail,
                **detail,
            }
    elif model_id == "retained_power_law":
        model = retained.fit(
            dataset.weights,
            dataset.y,
            retained_geometry(dataset, family_index),
            inner_folds,
            workers=rpl_workers,
        )
        selection = {
            "shape": asdict(model.shape),
            "ridge": model.ridge,
            "candidate_count": len(retained.shape_grid()) * len(retained.RIDGE_GRID),
        }
    else:
        raise ValueError(f"unsupported model {model_id}")
    return FitResult(
        model=model,
        selection=selection,
        parameter_diagnostics=parameter_diagnostics(model_id, model),
    )


def predict_model(
    model_id: str,
    model: Any,
    dataset: pooled.Dataset,
    weights: np.ndarray,
) -> np.ndarray:
    if model_id in DSP_MODEL_IDS:
        return observatory.dsp_predict(model, dataset, weights)
    if model_id == "separate_heads":
        return observatory.separate_predict(model, dataset, weights, observatory.TWO_PHASE)
    return np.asarray(model.predict(weights), dtype=float)


def support_distances(
    candidate: np.ndarray,
    observed: np.ndarray,
    phase_0_fraction: float,
) -> dict[str, float]:
    candidate_aggregate = phase_0_fraction * candidate[0] + (1.0 - phase_0_fraction) * candidate[1]
    observed_aggregate = phase_0_fraction * observed[:, 0, :] + (1.0 - phase_0_fraction) * observed[:, 1, :]
    policy_tv = 0.5 * (
        phase_0_fraction * np.abs(observed[:, 0, :] - candidate[0]).sum(axis=1)
        + (1.0 - phase_0_fraction) * np.abs(observed[:, 1, :] - candidate[1]).sum(axis=1)
    )
    aggregate_tv = 0.5 * np.abs(observed_aggregate - candidate_aggregate).sum(axis=1)

    observed_flat = observed.reshape(len(observed), -1)
    candidate_flat = candidate.ravel()
    penalty = 100.0
    augmented = np.vstack([observed_flat.T, penalty * np.ones((1, len(observed)))])
    response = np.concatenate([candidate_flat, [penalty]])
    solved = lsq_linear(augmented, response, bounds=(0.0, np.inf), tol=1e-10, max_iter=5_000)
    coefficients = np.maximum(solved.x, 0.0)
    coefficients /= max(coefficients.sum(), 1e-12)
    projection = coefficients @ observed_flat
    return {
        "nearest_policy_tv": float(np.min(policy_tv)),
        "nearest_aggregate_tv": float(np.min(aggregate_tv)),
        "convex_support_l2_approx": float(np.linalg.norm(projection - candidate_flat)),
    }


def raw_optimum(
    model_id: str,
    fit: FitResult,
    dataset: pooled.Dataset,
    starts: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    basis = helmert(dataset.m, full=False).T

    def to_coordinates(weights: np.ndarray) -> np.ndarray:
        return (np.log(np.maximum(weights, 1e-12)) @ basis).ravel()

    def to_weights(coordinates: np.ndarray) -> np.ndarray:
        logits = np.asarray(coordinates, dtype=float).reshape(2, dataset.m - 1) @ basis.T
        return softmax(logits, axis=1)

    def objective(coordinates: np.ndarray) -> float:
        weights = to_weights(coordinates)
        return float(predict_model(model_id, fit.model, dataset, weights[None, :, :])[0])

    phase_0_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    natural = observatory.natural_weights(dataset, phase_0_fraction)
    seed_weights = [np.stack([natural, natural])]
    observed_order = np.argsort(dataset.y)
    seed_weights.extend(dataset.weights[index] for index in observed_order[: min(7, len(observed_order))])
    rng = np.random.default_rng(seed)
    while len(seed_weights) < starts:
        concentration = float(rng.choice([0.3, 1.0, 3.0]))
        seed_weights.append(np.stack([rng.dirichlet(np.full(dataset.m, concentration)) for _ in range(2)]))

    candidates = []
    bounds = [(-OPTIMIZER_LOGIT_BOUND, OPTIMIZER_LOGIT_BOUND)] * (2 * (dataset.m - 1))
    for start_id, weights in enumerate(seed_weights[:starts]):
        start_prediction = float(predict_model(model_id, fit.model, dataset, weights[None, :, :])[0])
        candidates.append(
            {
                "start_id": start_id,
                "prediction": start_prediction,
                "weights": weights,
                "success": False,
                "message": "unoptimized_start",
                "iterations": 0,
            }
        )
        result = minimize(
            objective,
            to_coordinates(weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": OPTIMIZER_MAX_ITERATIONS,
                "maxfun": 250_000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 60,
            },
        )
        if np.isfinite(result.fun) and np.isfinite(result.x).all():
            candidates.append(
                {
                    "start_id": start_id,
                    "prediction": float(result.fun),
                    "weights": to_weights(np.asarray(result.x, dtype=float)),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "iterations": int(result.nit),
                }
            )
    finite = [row for row in candidates if np.isfinite(row["prediction"])]
    if not finite:
        raise RuntimeError(f"raw optimization found no finite endpoint for {model_id}/{dataset.name}")
    best = min(finite, key=lambda row: row["prediction"])
    weights = np.asarray(best["weights"], dtype=float)
    aggregate = phase_0_fraction * weights[0] + (1.0 - phase_0_fraction) * weights[1]
    epochs = weights[0] * dataset.c0 + weights[1] * dataset.c1
    tied_weights = np.stack([aggregate, aggregate])
    tied_prediction = float(predict_model(model_id, fit.model, dataset, tied_weights[None, :, :])[0])
    diagnostics = {
        "predicted_bpb": float(best["prediction"]),
        "matched_aggregate_tied_prediction": tied_prediction,
        "predicted_phase_gain_on_fiber": tied_prediction - float(best["prediction"]),
        "max_bucket_weight": float(np.max(weights)),
        "max_materialized_epochs": float(np.max(epochs)),
        "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "aggregate_hhi": float(np.sum(aggregate**2)),
        "successful_optimizer_endpoints": int(sum(bool(row["success"]) for row in finite)),
        "finite_optimizer_endpoints": len(finite),
        "best_start_id": int(best["start_id"]),
        "best_optimizer_success": bool(best["success"]),
        "best_optimizer_message": str(best["message"]),
        **support_distances(weights, dataset.weights, phase_0_fraction),
    }
    policy = pd.DataFrame(
        {
            "domain": dataset.domain_names,
            "phase_0_weight": weights[0],
            "phase_1_weight": weights[1],
            "aggregate_weight": aggregate,
            "materialized_epochs": epochs,
        }
    )
    return diagnostics, policy


def test_support_columns(
    dataset: expanded.Dataset,
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    beta0 = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    policy_distance = np.empty(len(test), dtype=float)
    aggregate_distance = np.empty(len(test), dtype=float)
    train_aggregate = beta0 * dataset.weights[train, 0, :] + (1.0 - beta0) * dataset.weights[train, 1, :]
    for output_index, row in enumerate(test):
        policy_distance[output_index] = np.min(
            0.5
            * (
                beta0 * np.abs(dataset.weights[train, 0, :] - dataset.weights[row, 0, :]).sum(axis=1)
                + (1.0 - beta0) * np.abs(dataset.weights[train, 1, :] - dataset.weights[row, 1, :]).sum(axis=1)
            )
        )
        aggregate = beta0 * dataset.weights[row, 0, :] + (1.0 - beta0) * dataset.weights[row, 1, :]
        aggregate_distance[output_index] = np.min(0.5 * np.abs(train_aggregate - aggregate).sum(axis=1))
    return policy_distance, aggregate_distance


def cell_dir(output_dir: Path, target: str, model_id: str) -> Path:
    return output_dir / "cells" / target / model_id


def cell_complete(path: Path, protocol_hash: str, *, require_optimum: bool = False) -> bool:
    marker = path / "complete.json"
    required = (
        path / "predictions.csv",
        path / "metrics.json",
        path / "pair_metrics.json",
        path / "pair_predictions.csv",
        path / "fold_selections.json",
        path / "parameter_diagnostics.csv",
    )
    if not marker.exists() or any(not item.exists() for item in required):
        return False
    payload = json.loads(marker.read_text())
    if payload.get("protocol_hash") != protocol_hash:
        return False
    return not require_optimum or bool(payload.get("has_raw_optimum"))


def run_cell(
    output_dir: Path,
    protocol: dict[str, Any],
    target: str,
    model_id: str,
    dataset: expanded.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    inner_splits: int,
    rpl_workers: int,
    optimizer_starts: int,
    skip_optimum: bool,
    force: bool,
) -> None:
    path = cell_dir(output_dir, target, model_id)
    if not force and cell_complete(
        path,
        str(protocol["protocol_hash"]),
        require_optimum=not skip_optimum,
    ):
        print(f"skip complete cell {target}/{model_id}", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    pooled_dataset = as_pooled(dataset)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    outer_fold = np.full(dataset.n, -1, dtype=int)
    nearest_policy_tv = np.full(dataset.n, np.nan, dtype=float)
    nearest_aggregate_tv = np.full(dataset.n, np.nan, dtype=float)
    selections = []
    parameter_rows = []

    for fold_id, (train, test) in enumerate(folds):
        print(
            f"{target}/{model_id}: outer fold {fold_id + 1}/{len(folds)} " f"({len(train)} train, {len(test)} test)",
            flush=True,
        )
        local = subset_dataset(pooled_dataset, train, f"outer{fold_id}_train")
        inner = correspondence_folds(
            local.frame,
            INNER_SEED_BASE + fold_id,
            inner_splits,
        )
        fit = fit_model(
            model_id,
            local,
            inner,
            target,
            dataset.family_index,
            rpl_workers,
        )
        prediction[test] = predict_model(model_id, fit.model, local, dataset.weights[test])
        outer_fold[test] = fold_id
        nearest_policy_tv[test], nearest_aggregate_tv[test] = test_support_columns(dataset, train, test)
        selections.append(
            {
                "outer_fold": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_correspondence_groups": local.frame["phase_correspondence_key"].nunique(),
                "inner_fold_test_groups": [
                    int(local.frame.iloc[inner_test]["phase_correspondence_key"].nunique())
                    for _inner_train, inner_test in inner
                ],
                "selection": fit.selection,
            }
        )
        parameter_rows.append({"outer_fold": fold_id, **fit.parameter_diagnostics})

    if not np.isfinite(prediction).all() or np.any(outer_fold < 0):
        raise RuntimeError(f"incomplete OOF predictions for {target}/{model_id}")
    tied = expanded.replay_control.tied_rows(dataset.weights)
    beta0 = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    phase_tv = 0.5 * np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]).sum(axis=1)
    aggregate = beta0 * dataset.weights[:, 0, :] + (1.0 - beta0) * dataset.weights[:, 1, :]
    predictions = pd.DataFrame(
        {
            "row_index": np.arange(dataset.n),
            "run_name": dataset.frame["run_name"].astype(str),
            "phase_correspondence_key": dataset.frame["phase_correspondence_key"].astype(str),
            "policy_family": dataset.frame["policy_family"].astype(str),
            "physical_tied": tied,
            "outer_fold": outer_fold,
            "observed": dataset.y,
            "predicted": prediction,
            "residual": prediction - dataset.y,
            "optimism": dataset.y - prediction,
            "phase_tv": phase_tv,
            "aggregate_hhi": np.sum(aggregate**2, axis=1),
            "nearest_train_policy_tv": nearest_policy_tv,
            "nearest_train_aggregate_tv": nearest_aggregate_tv,
        }
    )
    predictions.to_csv(path / "predictions.csv", index=False)
    metrics = {
        "target": target,
        "model": model_id,
        "reference_only": model_id in REFERENCE_ONLY_MODELS,
        **metric_summary(dataset, prediction, folds),
    }
    pair_metrics, pair_predictions = pair_summary(dataset, prediction)
    write_json(path / "metrics.json", metrics)
    write_json(path / "pair_metrics.json", pair_metrics)
    pair_predictions.to_csv(path / "pair_predictions.csv", index=False)
    write_json(path / "fold_selections.json", selections)
    pd.DataFrame(parameter_rows).to_csv(path / "parameter_diagnostics.csv", index=False)

    print(f"{target}/{model_id}: fitting full 520-row model", flush=True)
    full_inner = correspondence_folds(pooled_dataset.frame, FULL_FIT_SEED, inner_splits)
    full_fit = fit_model(
        model_id,
        pooled_dataset,
        full_inner,
        target,
        dataset.family_index,
        rpl_workers,
    )
    write_json(
        path / "full_fit.json",
        {
            "selection": full_fit.selection,
            "parameter_diagnostics": full_fit.parameter_diagnostics,
        },
    )
    if not skip_optimum:
        print(f"{target}/{model_id}: raw optimum audit with {optimizer_starts} starts", flush=True)
        optimum, policy = raw_optimum(
            model_id,
            full_fit,
            pooled_dataset,
            optimizer_starts,
            seed=FULL_FIT_SEED,
        )
        write_json(path / "raw_optimum.json", optimum)
        policy.to_csv(path / "raw_optimum_policy.csv", index=False)

    write_json(
        path / "complete.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "target": target,
            "model": model_id,
            "has_raw_optimum": (path / "raw_optimum.json").exists(),
        },
    )
    print(f"completed cell {target}/{model_id}", flush=True)


def collect_results(output_dir: Path, protocol: dict[str, Any]) -> None:
    metric_rows = []
    pair_rows = []
    parameter_rows = []
    optimum_rows = []
    complete_cells = []
    for target in TARGETS:
        for model_id in MODEL_IDS:
            path = cell_dir(output_dir, target, model_id)
            if not cell_complete(path, str(protocol["protocol_hash"])):
                continue
            complete_cells.append(f"{target}/{model_id}")
            metrics = json.loads((path / "metrics.json").read_text())
            pair = json.loads((path / "pair_metrics.json").read_text())
            metric_rows.append(metrics)
            pair_rows.append({"target": target, "model": model_id, **pair})
            diagnostics = pd.read_csv(path / "parameter_diagnostics.csv")
            parameter_rows.append(
                {
                    "target": target,
                    "model": model_id,
                    **{
                        f"mean_{column}": float(diagnostics[column].mean())
                        for column in (
                            "nominal_parameter_count",
                            "nonlinear_parameter_count",
                            "linear_parameter_count",
                            "active_linear_parameter_count",
                            "effective_df_active_set_proxy",
                        )
                    },
                }
            )
            optimum_path = path / "raw_optimum.json"
            if optimum_path.exists():
                optimum_rows.append(
                    {
                        "target": target,
                        "model": model_id,
                        **json.loads(optimum_path.read_text()),
                    }
                )
    metrics_frame = pd.DataFrame(metric_rows)
    pairs_frame = pd.DataFrame(pair_rows)
    parameters_frame = pd.DataFrame(parameter_rows)
    optima_frame = pd.DataFrame(optimum_rows)
    metrics_frame.to_csv(output_dir / "baseline_metrics.csv", index=False)
    pairs_frame.to_csv(output_dir / "baseline_pair_metrics.csv", index=False)
    parameters_frame.to_csv(output_dir / "baseline_parameter_counts.csv", index=False)
    optima_frame.to_csv(output_dir / "baseline_raw_optima.csv", index=False)
    write_json(
        output_dir / "baseline_status.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "complete_cells": complete_cells,
            "expected_cells": [f"{target}/{model}" for target in TARGETS for model in MODEL_IDS],
            "frozen_complete": len(complete_cells) == len(TARGETS) * len(MODEL_IDS),
        },
    )

    report = [
        "# Expanded 300M Pareto Baseline",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Complete cells: {len(complete_cells)}/{len(TARGETS) * len(MODEL_IDS)}",
        "- Rows: 520 observations, 280 correspondence groups, 238 exact asymmetric contrasts.",
        "- Outer and nested inner folds are grouped by `phase_correspondence_key`.",
        "- HPR-band is a reference-only non-mechanistic ensemble.",
        "- `effective_df_active_set_proxy` is not a formal generalized degree of freedom.",
        "",
    ]
    if not metrics_frame.empty:
        columns = [
            "target",
            "model",
            "all_rmse",
            "asymmetric_rmse",
            "asymmetric_regret_at_1",
            "all_low_tail_rmse",
            "all_calibration_slope",
        ]
        report.extend(
            [
                "## OOF Metrics",
                "",
                metrics_frame[columns].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    if not pairs_frame.empty:
        report.extend(
            [
                "## Exact Aggregate-Matched Contrasts",
                "",
                pairs_frame[
                    ["target", "model", "delta_rmse", "delta_spearman", "delta_bias", "sign_accuracy"]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    if not optima_frame.empty:
        report.extend(
            [
                "## Raw Optima",
                "",
                optima_frame[
                    [
                        "target",
                        "model",
                        "predicted_bpb",
                        "predicted_phase_gain_on_fiber",
                        "phase_tv",
                        "max_bucket_weight",
                        "nearest_policy_tv",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(report))


def main() -> None:
    args = parse_args()
    targets = parse_csv(args.targets)
    models = parse_csv(args.models)
    unknown_targets = sorted(set(targets) - set(TARGETS))
    unknown_models = sorted(set(models) - set(MODEL_IDS))
    if unknown_targets:
        raise ValueError(f"unknown targets: {unknown_targets}")
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    if args.outer_splits < 2 or args.inner_splits < 2:
        raise ValueError("outer and inner split counts must be at least two")
    if args.rpl_workers < 1:
        raise ValueError("rpl workers must be positive")
    if args.optimizer_starts < 8:
        raise ValueError("optimizer starts must be at least eight")

    protocol, datasets, folds = prepare_protocol(
        args.output_dir,
        args.outer_splits,
        args.inner_splits,
        args.optimizer_starts,
    )
    if args.prepare_only:
        collect_results(args.output_dir, protocol)
        print(
            f"prepared protocol {protocol['protocol_hash']} in {args.output_dir}",
            flush=True,
        )
        return

    for target in targets:
        for model_id in models:
            run_cell(
                args.output_dir,
                protocol,
                target,
                model_id,
                datasets[target],
                folds[target],
                args.inner_splits,
                args.rpl_workers,
                args.optimizer_starts,
                args.skip_optimum,
                args.force,
            )
            if not args.no_collect:
                collect_results(args.output_dir, protocol)
    if not args.no_collect:
        collect_results(args.output_dir, protocol)


if __name__ == "__main__":
    main()
