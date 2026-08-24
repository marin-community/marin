# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Benchmark compact surrogates on the exact Delphi phase-0 prefix panel.

The deployment target is phase-boundary Uncheatable BPB under a hard ten-epoch
cap. Model selection nevertheless uses every row: the cap is an optimisation
constraint, not a reason to discard observations of repetition damage. GitHub
C++ is a frozen diagnostic component of Uncheatable, not an independent target.

Every reported prediction is from an outer mixture-blocked fold. Shape and
shrinkage choices are selected again inside each outer training fold. This
keeps the prefix shortlist independent of its own reported validation scores.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import benchmark_dsp_single_phase_ladder_20260824 as dsp  # noqa: E402
import benchmark_single_phase_surrogates_20260824 as single_phase  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize, nnls  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

PANEL_PATH = (
    SCRIPT_DIR
    / "reference_outputs"
    / "delphi_3e18_phase0_prefix_replay_20260820"
    / "materialized_boundary_metrics"
    / "prefix_boundary_fit_matrix.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase0_prefix_surrogates_20260824"
TARGETS = ("uncheatable_bpb", "github_cpp_bpb")
PRIMARY_TARGET = "uncheatable_bpb"
CAP_COLUMN = "phase_0_epoch_cap_10_admissible"
CAP_EPOCHS = 10.0
OUTER_FOLDS = 5
INNER_FOLDS = 3
PARTITION_SEEDS = (0, 1, 2)
SHRINKAGE_GRID = (0.0, 1.0, 10.0)
LINEAR_RIDGE = 1e-6
LOG_RATE_BOUND = (float(np.log(1e-4)), float(np.log(2.0)))
THRESHOLD_BOUND = (-2.0, 8.0)
LOG_POWER_OFFSET_BOUND = (float(np.log(0.05)), float(np.log(20.0)))
LOG_POWER_EXPONENT_BOUND = (float(np.log(0.1)), float(np.log(5.0)))


@dataclasses.dataclass(frozen=True)
class Variant:
    name: str
    benefit: str
    damage: str
    note: str


VARIANTS = (
    Variant("shared_shape", "exponential", "canonical", "shared-shape canonical DSP"),
    Variant("bounded_shape", "exponential", "bounded", "shared exponential benefit plus bounded damage"),
    Variant("benefit_only", "exponential", "none", "shared exponential benefit without damage"),
    Variant("power_shape", "power", "canonical", "shared offset-power benefit plus canonical damage"),
    Variant("power_benefit_only", "power", "none", "shared offset-power benefit without damage"),
)


@dataclasses.dataclass(frozen=True)
class Fit:
    variant: Variant
    shape: np.ndarray
    shrinkage: float
    intercept: float
    coefficients: np.ndarray


def load_panel(path: Path = PANEL_PATH) -> tuple[pd.DataFrame, tuple[str, ...], np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    weight_columns = tuple(column for column in frame if column.startswith("phase_0_weight::"))
    epoch_columns = tuple(column for column in frame if column.startswith("phase_0_materialized_epochs::"))
    buckets = tuple(column.removeprefix("phase_0_weight::") for column in weight_columns)
    if buckets != tuple(column.removeprefix("phase_0_materialized_epochs::") for column in epoch_columns):
        raise ValueError("Prefix weight and epoch columns disagree")
    weights = frame.loc[:, weight_columns].to_numpy(dtype=float)
    exposure = frame.loc[:, epoch_columns].to_numpy(dtype=float)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Prefix weights do not sum to one")
    if len(frame) != 280 or not np.isfinite(exposure).all():
        raise ValueError(f"Unexpected prefix panel shape or missing exposure: {frame.shape}")
    if frame.run_order.duplicated().any() or len(np.unique(weights, axis=0)) != len(frame):
        raise ValueError("Prefix rows or mixtures are duplicated")
    computed_admissible = frame.max_phase_0_materialized_epoch.to_numpy(dtype=float) <= CAP_EPOCHS + 1e-12
    if not np.array_equal(frame[CAP_COLUMN].to_numpy(dtype=bool), computed_admissible):
        raise ValueError("Stored phase-0 cap labels disagree with materialized epochs")
    for target in TARGETS:
        if not np.isfinite(frame[target]).all():
            raise ValueError(f"Missing target values in {target}")
    return frame, buckets, weights, exposure


def quality_pairs(buckets: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    groups: dict[str, list[int]] = {}
    for position, bucket in enumerate(buckets):
        groups.setdefault(single_phase.domain_of(bucket), []).append(position)
    return tuple((members[0], members[1]) for members in groups.values() if len(members) == 2)


def mixture_blocks(weights: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = KMeans(n_clusters=folds, random_state=20260824 + seed, n_init=50).fit_predict(np.sqrt(weights))
    rows = np.arange(len(weights))
    result = []
    for label in range(folds):
        test = rows[labels == label]
        train = rows[labels != label]
        if len(test) == 0 or len(train) == 0:
            raise ValueError("Empty mixture block")
        result.append((train, test))
    return result


def benefit(exposure: np.ndarray, shape: np.ndarray, kind: str) -> np.ndarray:
    if kind == "exponential":
        return 1.0 - np.exp(-np.exp(shape[0]) * exposure)
    if kind == "power":
        offset = np.exp(shape[0])
        exponent = np.exp(shape[1])
        return 1.0 - (1.0 + exposure / offset) ** (-exponent)
    raise ValueError(f"Unknown benefit kind: {kind}")


def design(exposure: np.ndarray, variant: Variant, shape: np.ndarray) -> np.ndarray:
    columns = [-benefit(exposure, shape, variant.benefit)]
    if variant.damage == "canonical":
        columns.append(dsp.canonical_penalty(exposure, np.full(exposure.shape[1], shape[-1])))
    elif variant.damage == "bounded":
        columns.append(dsp.bounded_penalty(exposure, np.full(exposure.shape[1], shape[-1])))
    elif variant.damage != "none":
        raise ValueError(f"Unknown damage kind: {variant.damage}")
    return np.hstack(columns)


def bounds(variant: Variant) -> list[tuple[float, float]]:
    if variant.benefit == "exponential":
        box = [LOG_RATE_BOUND]
    else:
        box = [LOG_POWER_OFFSET_BOUND, LOG_POWER_EXPONENT_BOUND]
    if variant.damage == "canonical":
        box.append(THRESHOLD_BOUND)
    elif variant.damage == "bounded":
        box.append(dsp.LOG_EXPONENT_BOUND)
    return box


def solve_head(
    rows: np.ndarray,
    response: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    shrinkage: float,
    bucket_count: int,
) -> tuple[float, np.ndarray]:
    centre = rows.mean(axis=0, keepdims=True)
    response_mean = float(response.mean())
    fit_rows = rows - centre
    fit_target = response - response_mean
    width = rows.shape[1]
    fit_rows = np.vstack([fit_rows, np.sqrt(LINEAR_RIDGE) * np.eye(width)])
    fit_target = np.concatenate([fit_target, np.zeros(width)])
    if shrinkage > 0:
        blocks = width // bucket_count
        tie_rows = []
        for block in range(blocks):
            offset = block * bucket_count
            for first, second in pairs:
                row = np.zeros(width)
                row[offset + first] = np.sqrt(shrinkage)
                row[offset + second] = -np.sqrt(shrinkage)
                tie_rows.append(row)
        fit_rows = np.vstack([fit_rows, np.asarray(tie_rows)])
        fit_target = np.concatenate([fit_target, np.zeros(len(tie_rows))])
    coefficients, _ = nnls(fit_rows, fit_target, maxiter=300 * width)
    intercept = response_mean - float((centre @ coefficients).item())
    return intercept, coefficients


def fit_shape(
    exposure: np.ndarray,
    response: np.ndarray,
    variant: Variant,
    folds: list[tuple[np.ndarray, np.ndarray]],
    pairs: tuple[tuple[int, int], ...],
    shrinkage: float,
    seed: int,
) -> Fit:
    box = bounds(variant)

    def objective(shape: np.ndarray) -> float:
        total = 0.0
        for train, test in folds:
            train_design = design(exposure[train], variant, shape)
            intercept, coefficients = solve_head(train_design, response[train], pairs, shrinkage, exposure.shape[1])
            residual = intercept + design(exposure[test], variant, shape) @ coefficients - response[test]
            total += float(residual @ residual)
        return total

    lows = np.array([low for low, _ in box])
    highs = np.array([high for _, high in box])
    generator = np.random.default_rng(20260824 + seed)
    starts = [0.5 * (lows + highs), generator.uniform(lows, highs), generator.uniform(lows, highs)]
    best_shape = starts[0]
    best_value = float("inf")
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=box, options={"maxiter": 100})
        if float(result.fun) < best_value:
            best_shape = np.asarray(result.x, dtype=float)
            best_value = float(result.fun)
    full_design = design(exposure, variant, best_shape)
    intercept, coefficients = solve_head(full_design, response, pairs, shrinkage, exposure.shape[1])
    return Fit(variant, best_shape, shrinkage, intercept, coefficients)


def predict(fit: Fit, exposure: np.ndarray) -> np.ndarray:
    return fit.intercept + design(exposure, fit.variant, fit.shape) @ fit.coefficients


def inner_select(
    exposure: np.ndarray,
    weights: np.ndarray,
    response: np.ndarray,
    variant: Variant,
    pairs: tuple[tuple[int, int], ...],
    seed: int,
) -> tuple[Fit, float]:
    folds = mixture_blocks(weights, INNER_FOLDS, seed)
    candidates = []
    for shrinkage in SHRINKAGE_GRID:
        fitted = fit_shape(exposure, response, variant, folds, pairs, shrinkage, seed)
        residuals = []
        for train, test in folds:
            nested = fit_shape(
                exposure[train],
                response[train],
                variant,
                mixture_blocks(weights[train], INNER_FOLDS, seed + 17),
                pairs,
                shrinkage,
                seed + 31,
            )
            residuals.extend((predict(nested, exposure[test]) - response[test]).tolist())
        candidates.append((float(np.mean(np.square(residuals))), fitted, shrinkage))
    _, fitted, shrinkage = min(candidates, key=lambda item: item[0])
    return fitted, shrinkage


def scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((observed - observed.mean()) ** 2)),
        "spearman": float(stats.spearmanr(observed, predicted).statistic),
        "pearson": float(stats.pearsonr(observed, predicted).statistic),
    }


def outer_benchmark(
    frame: pd.DataFrame,
    buckets: tuple[str, ...],
    weights: np.ndarray,
    exposure: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = quality_pairs(buckets)
    predictions = []
    metrics = []
    for target in TARGETS:
        response = frame[target].to_numpy(dtype=float)
        for variant in VARIANTS:
            for partition_seed in PARTITION_SEEDS:
                fold_predictions = np.full(len(frame), np.nan)
                selected_shrinkage = []
                fold_cap_regrets = []
                for fold, (train, test) in enumerate(mixture_blocks(weights, OUTER_FOLDS, partition_seed)):
                    fitted, shrinkage = inner_select(
                        exposure[train],
                        weights[train],
                        response[train],
                        variant,
                        pairs,
                        seed=1000 * partition_seed + fold,
                    )
                    fold_predictions[test] = predict(fitted, exposure[test])
                    selected_shrinkage.append(shrinkage)
                    fold_admissible = test[frame.iloc[test][CAP_COLUMN].to_numpy(dtype=bool)]
                    if len(fold_admissible):
                        chosen = fold_admissible[np.argmin(fold_predictions[fold_admissible])]
                        fold_cap_regrets.append(float(response[chosen] - response[fold_admissible].min()))
                    for row, value in zip(test, fold_predictions[test], strict=True):
                        predictions.append(
                            {
                                "target": target,
                                "variant": variant.name,
                                "partition_seed": partition_seed,
                                "outer_fold": fold,
                                "run_order": int(frame.iloc[row]["run_order"]),
                                "observed": response[row],
                                "predicted": value,
                                "cap_admissible": bool(frame.iloc[row][CAP_COLUMN]),
                            }
                        )
                if not np.isfinite(fold_predictions).all():
                    raise ValueError("Outer predictions are incomplete")
                admissible = frame[CAP_COLUMN].to_numpy(dtype=bool)
                chosen = np.flatnonzero(admissible)[np.argmin(fold_predictions[admissible])]
                best = float(response[admissible].min())
                cap_scores = {
                    f"cap_{name}": value
                    for name, value in scores(response[admissible], fold_predictions[admissible]).items()
                }
                metrics.append(
                    {
                        "target": target,
                        "variant": variant.name,
                        "partition_seed": partition_seed,
                        **scores(response, fold_predictions),
                        **cap_scores,
                        "pooled_cap_regret_at_1": float(response[chosen] - best),
                        "mean_fold_cap_regret_at_1": float(np.mean(fold_cap_regrets)),
                        "worst_fold_cap_regret_at_1": float(np.max(fold_cap_regrets)),
                        "pooled_cap_selected_run_order": int(frame.iloc[chosen]["run_order"]),
                        "shrinkage_mode": float(stats.mode(selected_shrinkage, keepdims=False).mode),
                    }
                )
    return pd.DataFrame(metrics), pd.DataFrame(predictions)


def full_fits(
    frame: pd.DataFrame,
    buckets: tuple[str, ...],
    weights: np.ndarray,
    exposure: np.ndarray,
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Fit]]:
    pairs = quality_pairs(buckets)
    primary = (
        metrics[metrics.target.eq(PRIMARY_TARGET)]
        .groupby("variant")
        .agg(
            rmse=("rmse", "mean"),
            cap_rmse=("cap_rmse", "mean"),
            cap_spearman=("cap_spearman", "mean"),
            mean_fold_cap_regret=("mean_fold_cap_regret_at_1", "mean"),
        )
    )
    primary = primary.sort_values(
        ["cap_spearman", "cap_rmse", "mean_fold_cap_regret", "rmse"],
        ascending=[False, True, True, True],
    )
    fits = {}
    rows = []
    for variant_name in primary.index:
        variant = next(item for item in VARIANTS if item.name == variant_name)
        shrinkage = float(
            stats.mode(
                metrics.loc[metrics.target.eq(PRIMARY_TARGET) & metrics.variant.eq(variant_name), "shrinkage_mode"],
                keepdims=False,
            ).mode
        )
        response = frame[PRIMARY_TARGET].to_numpy(dtype=float)
        fitted = fit_shape(
            exposure,
            response,
            variant,
            mixture_blocks(weights, OUTER_FOLDS, 97),
            pairs,
            shrinkage,
            97,
        )
        fits[variant_name] = fitted
        predicted = predict(fitted, exposure)
        admissible = frame[CAP_COLUMN].to_numpy(dtype=bool)
        selected = np.flatnonzero(admissible)[np.argmin(predicted[admissible])]
        rows.append(
            {
                "variant": variant_name,
                "selected_run_order": int(frame.iloc[selected]["run_order"]),
                "selected_observed_uncheatable": float(frame.iloc[selected][PRIMARY_TARGET]),
                "selected_predicted_uncheatable": float(predicted[selected]),
                "selected_github_cpp": float(frame.iloc[selected]["github_cpp_bpb"]),
                "selected_max_phase0_epoch": float(frame.iloc[selected]["max_phase_0_materialized_epoch"]),
                "shrinkage": shrinkage,
                "shape": json.dumps(fitted.shape.tolist()),
            }
        )
    return pd.DataFrame(rows), fits


def write_report(output_dir: Path, metrics: pd.DataFrame, selections: pd.DataFrame, panel_hash: str) -> None:
    summary = metrics.groupby(["target", "variant"]).agg(
        rmse=("rmse", "mean"),
        spearman=("spearman", "mean"),
        cap_rmse=("cap_rmse", "mean"),
        cap_spearman=("cap_spearman", "mean"),
        mean_fold_cap_regret=("mean_fold_cap_regret_at_1", "mean"),
        worst_fold_cap_regret=("worst_fold_cap_regret_at_1", "max"),
        pooled_cap_regret=("pooled_cap_regret_at_1", "mean"),
    )
    report = f"""# Delphi phase-0 prefix surrogate audit

Primary target: exact-boundary Uncheatable BPB. GitHub C++ is a diagnostic component of that macro, not an
independent guardrail. The ten-epoch
cap is applied only to policy selection; all 280 rows inform each fit. Every metric below is outer
mixture-blocked OOF with all shape and quality-shrinkage choices repeated inside the outer fold.

The `cap_*` columns score only the 44 deployment-admissible rows. Fold-cap regret selects and scores within
each held-out fold, avoiding comparisons between predictions with independently fitted fold intercepts.

Panel SHA-256: `{panel_hash}`

## OOF summary

```
{summary.to_string(float_format=lambda value: f"{value:.6f}")}
```

## Full-panel admissible selections

```
{selections.to_string(index=False)}
```

No model is promoted from fit alone. Continuous constrained candidates and historical one-phase transfer
must agree before training; boundary validation then decides whether a prefix remains in the branch panel.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, buckets, weights, exposure = load_panel()
    metrics, predictions = outer_benchmark(frame, buckets, weights, exposure)
    selections, fits = full_fits(frame, buckets, weights, exposure, metrics)
    panel_hash = hashlib.sha256(PANEL_PATH.read_bytes()).hexdigest()
    metrics.to_csv(args.output_dir / "outer_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "outer_predictions.csv", index=False)
    selections.to_csv(args.output_dir / "admissible_observed_selections.csv", index=False)
    model_payload = {
        name: {
            "variant": dataclasses.asdict(fit.variant),
            "shape": fit.shape.tolist(),
            "shrinkage": fit.shrinkage,
            "intercept": fit.intercept,
            "coefficients": fit.coefficients.tolist(),
        }
        for name, fit in fits.items()
    }
    (args.output_dir / "full_models.json").write_text(json.dumps(model_payload, indent=2, sort_keys=True))
    write_report(args.output_dir, metrics, selections, panel_hash)
    print(metrics.groupby(["target", "variant"]).mean(numeric_only=True).to_string())
    print("\n", selections.to_string(index=False))


if __name__ == "__main__":
    main()
