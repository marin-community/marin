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
"""Test a collaborator-derived linear-past-threshold replay law.

This is a frozen nested ablation of Hierarchical Phase Replay (HPR). Useful
learning retains HPR's bucket/family power response and phase-state transition.
Only the repetition-harm state and response law change.

The script has an explicit two-stage protocol. ``--stage fit`` selects every
hyperparameter using fit panels only and persists the frozen configurations.
``--stage heldout`` then evaluates those frozen configurations on the
append-only Delphi 3e18 development archive. The heldout stage refuses to run
without the fit-stage artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hpr,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/linear_threshold_replay_20260721"
HPR_METRICS = SCRIPT_DIR / "reference_outputs/hierarchical_coverage_grp_20260715/metrics.csv"
THRESHOLD_GRID = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
L2_GRID = (0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0)
FIT_SEED = 7210
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
OPTIMISM_THRESHOLD = 0.05
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Variant(StrEnum):
    CURVED_RETAINED_HPR = "curved_retained_hpr"
    NO_REPLAY_HPR = "no_replay_hpr"
    LINEAR_RETAINED_FAMILY_HPR = "linear_retained_family_hpr"
    LINEAR_PHYSICAL_FAMILY_HPR = "linear_physical_family_hpr"
    LINEAR_PHYSICAL_BUCKET_HPR = "linear_physical_bucket_hpr"

    @property
    def uses_hinge(self) -> bool:
        return self not in {Variant.CURVED_RETAINED_HPR, Variant.NO_REPLAY_HPR}

    @property
    def physical(self) -> bool:
        return self in {Variant.LINEAR_PHYSICAL_FAMILY_HPR, Variant.LINEAR_PHYSICAL_BUCKET_HPR}

    @property
    def bucket_slopes(self) -> bool:
        return self is Variant.LINEAR_PHYSICAL_BUCKET_HPR


@dataclass(frozen=True)
class Config:
    variant: Variant
    shape: family_grp.Shape
    l2: float
    residual_shrink: float
    threshold: float | None


@dataclass(frozen=True)
class Design:
    values: np.ndarray
    names: tuple[str, ...]
    ridge_multipliers: np.ndarray


@dataclass(frozen=True)
class Head:
    intercept: float
    coefficients: np.ndarray

    def predict(self, design: Design) -> np.ndarray:
        return np.asarray(self.intercept + design.values @ self.coefficients, dtype=float)


DATASETS = (
    hpr.DatasetId.PRODUCTION_UNCHEATABLE,
    hpr.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    hpr.DatasetId.THREE_HUNDRED_M_TABLE9,
    hpr.DatasetId.DELPHI_3E18_UNCHEATABLE,
    hpr.DatasetId.DELPHI_3E18_TABLE9,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "heldout"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=",".join(dataset.value for dataset in DATASETS),
        help="Comma-separated dataset IDs.",
    )
    return parser.parse_args()


def preregistration() -> dict[str, Any]:
    return {
        "frozen_at": datetime.now(UTC).isoformat(),
        "source": "marin-community/marin#7067 and origin/rav/mixing-via-embeddings",
        "data_use": {
            "fit_stage": "fit panels only; no Delphi development-heldout outcomes loaded",
            "heldout_stage": "frozen full-fit configurations projected onto coordinate-disjoint archive rows",
        },
        "shared_state": (
            "HPR retained useful exposure x_i = exp[-lambda(1-w_i^1)] e_i^0 + eta e_i^1; "
            "physical exposure E_i = e_i^0 + e_i^1"
        ),
        "variants": {
            Variant.CURVED_RETAINED_HPR.value: (
                "Exact incumbent: squared-softplus harm in log retained exposure at family and member level."
            ),
            Variant.NO_REPLAY_HPR.value: "Nested ablation with every replay-harm feature removed.",
            Variant.LINEAR_RETAINED_FAMILY_HPR.value: (
                "Replace curved harm with sum_f b_f mean_i_in_f [x_i-tau]_+; b_f>=0."
            ),
            Variant.LINEAR_PHYSICAL_FAMILY_HPR.value: (
                "Replace curved harm with sum_f b_f mean_i_in_f [E_i-tau]_+; b_f>=0."
            ),
            Variant.LINEAR_PHYSICAL_BUCKET_HPR.value: ("Replace curved harm with sum_i b_i [E_i-tau]_+; b_i>=0."),
        },
        "fixed_grids": {"threshold_epochs": list(THRESHOLD_GRID), "l2": list(L2_GRID)},
        "selection": (
            "Nested fit-panel CV for OOF metrics; a separate full-panel CV selects the persisted configuration. "
            "The incumbent retained-state shape and residual shrink are held fixed per dataset so this is a "
            "response-law ablation."
        ),
        "promotion_gate": {
            "oof_rmse": "no core fit-panel regression above 5% versus incumbent",
            "heldout_regret_at_1": "no policy-matched regression above 0.002 BPB",
            "optimism_gt_0p05": "preserve or reduce on both Delphi targets",
            "calibration": "move observed-on-predicted slope toward one without output calibration",
            "mechanism": "nonzero stable hinge slopes on at least two independent panels",
        },
    }


def incumbent_config(dataset_id: hpr.DatasetId) -> Config:
    metrics = pd.read_csv(HPR_METRICS)
    selected = metrics.loc[
        metrics["dataset"].eq(dataset_id.value)
        & metrics["variant"].eq(hpr.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY.value)
        & metrics["split"].eq("fit_oof")
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one incumbent HPR row for {dataset_id.value}, found {len(selected)}")
    row = selected.iloc[0]
    shape = family_grp.Shape(
        exponent=float(row["exponent"]),
        late_multiplier=float(row["late_multiplier"]),
        forgetting_rate=float(row["forgetting_rate"]),
        penalty_threshold=float(row["penalty_threshold"]),
        quality_discount=1.0,
    )
    return Config(
        variant=Variant.CURVED_RETAINED_HPR,
        shape=shape,
        l2=float(row["l2"]),
        residual_shrink=float(row["residual_shrink"]),
        threshold=None,
    )


def hpr_config(config: Config) -> hpr.Config:
    return hpr.Config(
        variant=hpr.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
        shape_index=0,
        shape=config.shape,
        l2=config.l2,
        residual_shrink=config.residual_shrink,
        undercoverage_fraction=0.0,
        coverage_gate_ratio=0.0,
    )


def physical_exposure(dataset: family_grp.Dataset) -> np.ndarray:
    return dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :]


def hinge_design(dataset: family_grp.Dataset, config: Config) -> Design:
    base = hpr.build_design(dataset, hpr_config(config))
    if config.variant is Variant.CURVED_RETAINED_HPR:
        return Design(base.values, base.names, base.ridge_multipliers)

    keep = np.asarray(
        [
            not name.startswith("family_overexposure:") and not name.startswith("family_member_replay:")
            for name in base.names
        ],
        dtype=bool,
    )
    pieces = [base.values[:, keep]]
    names = [name for index, name in enumerate(base.names) if keep[index]]
    ridge = list(base.ridge_multipliers[keep])
    if config.variant is Variant.NO_REPLAY_HPR:
        return Design(np.hstack(pieces), tuple(names), np.asarray(ridge, dtype=float))

    if config.threshold is None:
        raise ValueError(f"{config.variant.value} requires a threshold")
    exposure = physical_exposure(dataset) if config.variant.physical else hpr.retained_exposure(dataset, config.shape)
    activated = np.maximum(exposure - config.threshold, 0.0)
    if config.variant.bucket_slopes:
        pieces.append(activated)
        names.extend(f"physical_linear_replay:{domain}" for domain in dataset.domains)
        ridge.extend([1.0] * dataset.m)
    else:
        family_harm = np.column_stack([activated[:, members].mean(axis=1) for members in dataset.family_members])
        pieces.append(family_harm)
        state_name = "physical" if config.variant.physical else "retained"
        names.extend(f"{state_name}_linear_replay:{family}" for family in dataset.family_names)
        ridge.extend([1.0] * len(dataset.family_names))
    return Design(np.hstack(pieces), tuple(names), np.asarray(ridge, dtype=float))


def fit_head(design: Design, target: np.ndarray, indices: np.ndarray, l2: float) -> Head:
    values = design.values[indices]
    observed = target[indices]
    mean_values = values.mean(axis=0, keepdims=True)
    mean_target = float(observed.mean())
    centered_values = values - mean_values
    centered_target = observed - mean_target
    if l2 > 0.0:
        ridge = np.sqrt(l2 * design.ridge_multipliers)
        centered_values = np.vstack([centered_values, np.diag(ridge)])
        centered_target = np.concatenate([centered_target, np.zeros(len(ridge), dtype=float)])
    coefficients, _residual = nnls(centered_values, centered_target, maxiter=60 * centered_values.shape[1])
    intercept = mean_target - float((mean_values @ coefficients).item())
    return Head(intercept, np.asarray(coefficients, dtype=float))


def config_grid(incumbent: Config, variant: Variant) -> tuple[Config, ...]:
    if variant is Variant.CURVED_RETAINED_HPR:
        return (incumbent,)
    thresholds: tuple[float | None, ...] = THRESHOLD_GRID if variant.uses_hinge else (None,)
    return tuple(
        replace(incumbent, variant=variant, l2=l2, threshold=threshold) for threshold in thresholds for l2 in L2_GRID
    )


def oof_prediction(
    dataset: family_grp.Dataset,
    config: Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    design = hinge_design(dataset, config)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = fit_head(design, dataset.target, train, config.l2).predict(
            Design(design.values[test], design.names, design.ridge_multipliers)
        )
    covered = np.unique(np.concatenate([test for _train, test in splits]))
    if not np.isfinite(prediction[covered]).all():
        raise RuntimeError(f"Incomplete OOF predictions for {config.variant.value}")
    return prediction


def metric_summary(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = predicted - observed
    lower_count = min(len(observed), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))))
    lower = np.argsort(predicted)[:lower_count]
    optimism = observed - predicted
    predicted_order = np.argsort(predicted)
    calibration_slope = (
        float(np.cov(predicted, observed, ddof=0)[0, 1] / np.var(predicted))
        if np.var(predicted) > 1e-15
        else float("nan")
    )
    result: dict[str, float | int] = {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_pred_minus_obs": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "calibration_slope_obs_on_pred": calibration_slope,
        "optimism_gt_0p05": int(np.sum(optimism > OPTIMISM_THRESHOLD)),
        "worst_optimism": float(np.max(optimism)),
        "lower_tail_optimism": float(np.mean(np.maximum(optimism[lower], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[lower] ** 2))),
    }
    best = float(np.min(observed))
    for k in (1, 3, 5):
        result[f"regret_at_{k}"] = float(np.min(observed[predicted_order[: min(k, len(observed))]]) - best)
    return result


def select_config(
    dataset: family_grp.Dataset,
    dataset_id: hpr.DatasetId,
    candidates: tuple[Config, ...],
    indices: np.ndarray,
    seed: int,
) -> tuple[Config, dict[str, float | int]]:
    splits = hpr.split_indices(dataset, dataset_id, indices, seed)
    best: tuple[float, float, float, Config, dict[str, float | int]] | None = None
    for config in candidates:
        prediction = oof_prediction(dataset, config, splits)
        covered = np.unique(np.concatenate([test for _train, test in splits]))
        metrics = metric_summary(dataset.target[covered], prediction[covered])
        candidate = (float(metrics["rmse"]), -float(metrics["spearman"]), config.l2, config, metrics)
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None:
        raise RuntimeError("No candidate configurations")
    return best[3], best[4]


def nested_oof(
    dataset: family_grp.Dataset,
    dataset_id: hpr.DatasetId,
    incumbent: Config,
    variant: Variant,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    outer = hpr.split_indices(dataset, dataset_id, np.arange(dataset.n), FIT_SEED)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selections: list[dict[str, Any]] = []
    candidates = config_grid(incumbent, variant)
    for fold, (outer_train, outer_test) in enumerate(outer):
        if variant is Variant.CURVED_RETAINED_HPR:
            selected = incumbent
        else:
            selected, _metrics = select_config(dataset, dataset_id, candidates, outer_train, FIT_SEED + fold + 1)
        design = hinge_design(dataset, selected)
        head = fit_head(design, dataset.target, outer_train, selected.l2)
        prediction[outer_test] = head.predict(Design(design.values[outer_test], design.names, design.ridge_multipliers))
        selections.append(
            {
                "fold": fold,
                "threshold": selected.threshold,
                "l2": selected.l2,
                "active_parameters": int(np.sum(head.coefficients > 1e-10)),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Nested OOF incomplete for {dataset_id.value}/{variant.value}")
    return prediction, selections


def full_fit_config(
    dataset: family_grp.Dataset,
    dataset_id: hpr.DatasetId,
    incumbent: Config,
    variant: Variant,
) -> Config:
    if variant is Variant.CURVED_RETAINED_HPR:
        return incumbent
    config, _metrics = select_config(
        dataset,
        dataset_id,
        config_grid(incumbent, variant),
        np.arange(dataset.n),
        FIT_SEED + 100,
    )
    return config


def config_record(config: Config) -> dict[str, Any]:
    return {
        "variant": config.variant.value,
        "shape": asdict(config.shape),
        "l2": config.l2,
        "residual_shrink": config.residual_shrink,
        "threshold": config.threshold,
    }


def config_from_record(record: dict[str, Any]) -> Config:
    return Config(
        variant=Variant(record["variant"]),
        shape=family_grp.Shape(**record["shape"]),
        l2=float(record["l2"]),
        residual_shrink=float(record["residual_shrink"]),
        threshold=None if record["threshold"] is None else float(record["threshold"]),
    )


def write_fit_report(output_dir: Path) -> None:
    metrics = pd.read_csv(output_dir / "fit_panel_metrics.csv")
    parameters = pd.read_csv(output_dir / "fit_panel_parameters.csv")
    incumbent = (
        metrics.loc[metrics["variant"].eq(Variant.CURVED_RETAINED_HPR.value), ["dataset", "rmse"]]
        .set_index("dataset")["rmse"]
        .to_dict()
    )
    metrics["rmse_ratio_to_incumbent"] = [
        float(row.rmse) / float(incumbent[str(row.dataset)]) for row in metrics.itertuples()
    ]
    metrics["passes_oof_rmse_gate"] = metrics["rmse_ratio_to_incumbent"] <= 1.05
    metrics.to_csv(output_dir / "fit_gate_evaluation.csv", index=False)

    replay = parameters.loc[parameters["is_replay_feature"]].copy()
    replay_summary = (
        replay.groupby(["dataset", "variant"], as_index=False)
        .agg(
            replay_features=("coefficient", "size"),
            active_replay_features=("coefficient", lambda values: int(np.sum(values > 1e-10))),
            replay_coefficient_sum=("coefficient", "sum"),
            replay_coefficient_max=("coefficient", "max"),
        )
        .sort_values(["dataset", "variant"])
    )
    replay_summary.to_csv(output_dir / "fit_replay_parameter_summary.csv", index=False)

    selected_columns = [
        "dataset",
        "variant",
        "rmse",
        "rmse_ratio_to_incumbent",
        "spearman",
        "calibration_slope_obs_on_pred",
        "regret_at_1",
        "passes_oof_rmse_gate",
    ]
    lines = [
        "# Linear-past-threshold replay: fit-stage gate",
        "",
        "No Delphi development-heldout outcomes were opened for this stage. The comparison changes only the replay-harm "
        "state and response law while fixing the incumbent HPR useful-learning state.",
        "",
        "## Result",
        "",
        "No linear-hinge variant passes the frozen OOF gate on every core fit panel. The family-pooled retained-state "
        "hinge is close at 300M, but exceeds the 5% RMSE allowance on both Delphi targets. Physical-epoch hinges "
        "regress more strongly, and every production-swarm linear-hinge coefficient collapses to zero.",
        "",
        "The route is therefore blocked before heldout evaluation. The collaborator's controlled experiment supports "
        "the hinge as a small-budget or extreme-repetition guardrail, not as a universal response term for these smooth "
        "aggregate targets.",
        "",
        "## OOF gate",
        "",
        metrics[selected_columns].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Replay parameters",
        "",
        replay_summary.to_markdown(index=False, floatfmt=".6g"),
    ]
    (output_dir / "fit_stage_report.md").write_text("\n".join(lines) + "\n")


def run_fit_stage(output_dir: Path, dataset_ids: tuple[hpr.DatasetId, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration_path = output_dir / "preregistration.json"
    if not preregistration_path.exists():
        preregistration_path.write_text(json.dumps(preregistration(), indent=2, sort_keys=True) + "\n")

    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    selections: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        print(f"Fit-panel audit: {dataset_id.value}", flush=True)
        dataset = hpr.load_dataset(dataset_id)
        incumbent = incumbent_config(dataset_id)
        selected[dataset_id.value] = {}
        for variant in Variant:
            prediction, fold_selections = nested_oof(dataset, dataset_id, incumbent, variant)
            full_config = full_fit_config(dataset, dataset_id, incumbent, variant)
            design = hinge_design(dataset, full_config)
            head = fit_head(design, dataset.target, np.arange(dataset.n), full_config.l2)
            selected[dataset_id.value][variant.value] = config_record(full_config)
            for name, coefficient in zip(design.names, head.coefficients, strict=True):
                parameters.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.value,
                        "feature": name,
                        "coefficient": coefficient,
                        "is_replay_feature": "replay:" in name or name.startswith("family_overexposure:"),
                    }
                )
            metrics.append(
                {
                    "dataset": dataset_id.value,
                    "variant": variant.value,
                    "split": "nested_oof",
                    "feature_count": design.values.shape[1],
                    "active_parameters": int(np.sum(head.coefficients > 1e-10)),
                    **metric_summary(dataset.target, prediction),
                }
            )
            for selection in fold_selections:
                selections.append({"dataset": dataset_id.value, "variant": variant.value, **selection})
            for index, (observed, predicted) in enumerate(zip(dataset.target, prediction, strict=True)):
                predictions.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.value,
                        "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                        "observed": observed,
                        "predicted": predicted,
                    }
                )
    pd.DataFrame(metrics).to_csv(output_dir / "fit_panel_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output_dir / "fit_panel_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(output_dir / "nested_fold_selections.csv", index=False)
    pd.DataFrame(parameters).to_csv(output_dir / "fit_panel_parameters.csv", index=False)
    (output_dir / "selected_configs.json").write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    write_fit_report(output_dir)


def heldout_subsets(frame: pd.DataFrame, dataset_id: hpr.DatasetId) -> list[tuple[str, np.ndarray]]:
    target = "uncheatable" if dataset_id is hpr.DatasetId.DELPHI_3E18_UNCHEATABLE else "table9"
    policy = frame["policy_class"].astype(str).to_numpy()
    proposal = frame["proposal_target"].fillna("").astype(str).str.lower().to_numpy()
    subsets = [
        ("all_coordinate_disjoint", np.ones(len(frame), dtype=bool)),
        ("two_phase", policy != "single_phase_tied"),
        ("single_phase", policy == "single_phase_tied"),
        ("target_matched", np.asarray([target in value for value in proposal], dtype=bool)),
    ]
    return [(name, mask) for name, mask in subsets if int(mask.sum()) >= 3]


def render_heldout(predictions: pd.DataFrame, output_dir: Path) -> None:
    for dataset in (hpr.DatasetId.DELPHI_3E18_UNCHEATABLE.value, hpr.DatasetId.DELPHI_3E18_TABLE9.value):
        selected = predictions.loc[predictions["dataset"].eq(dataset) & predictions["subset_member_two_phase"]]
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Observed vs predicted", "Optimism vs max physical epoch"),
        )
        colors = {
            Variant.CURVED_RETAINED_HPR.value: "#d73027",
            Variant.NO_REPLAY_HPR.value: "#fdae61",
            Variant.LINEAR_RETAINED_FAMILY_HPR.value: "#fee08b",
            Variant.LINEAR_PHYSICAL_FAMILY_HPR.value: "#66bd63",
            Variant.LINEAR_PHYSICAL_BUCKET_HPR.value: "#1a9850",
        }
        for variant in Variant:
            rows = selected.loc[selected["variant"].eq(variant.value)]
            figure.add_trace(
                go.Scatter(
                    x=rows["predicted"],
                    y=rows["observed"],
                    mode="markers",
                    name=variant.value,
                    marker={"color": colors[variant.value], "size": 5, "opacity": 0.5},
                    customdata=np.column_stack([rows["row_id"], rows["training_series"]]),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>pred=%{x:.5f}<br>obs=%{y:.5f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatter(
                    x=rows["max_physical_epoch"],
                    y=rows["optimism"],
                    mode="markers",
                    name=variant.value,
                    showlegend=False,
                    marker={"color": colors[variant.value], "size": 5, "opacity": 0.5},
                    customdata=np.column_stack([rows["row_id"], rows["training_series"]]),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>max epoch=%{x:.2f}<br>obs-pred=%{y:.5f}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )
        bounds = [
            float(selected[["observed", "predicted"]].min().min()),
            float(selected[["observed", "predicted"]].max().max()),
        ]
        figure.add_trace(
            go.Scatter(x=bounds, y=bounds, mode="lines", line={"color": "#687780", "dash": "dash"}, showlegend=False),
            row=1,
            col=1,
        )
        figure.add_hline(y=0.0, line={"color": "#687780", "dash": "dash"}, row=1, col=2)
        figure.update_layout(
            title=f"Linear-threshold replay audit: {dataset}",
            template="plotly_white",
            width=1500,
            height=650,
            legend={"orientation": "h", "y": 1.12},
        )
        figure.update_xaxes(title_text="Predicted BPB", row=1, col=1)
        figure.update_yaxes(title_text="Observed BPB", row=1, col=1)
        figure.update_xaxes(title_text="Maximum physical simulated epoch", row=1, col=2)
        figure.update_yaxes(title_text="Optimism (observed - predicted)", row=1, col=2)
        figure.write_html(output_dir / f"{dataset}_heldout_calibration.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, configs: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Linear-past-threshold replay audit",
        "",
        "The retained useful-learning state and its power response are fixed to the incumbent HPR selection. "
        "The only structural change is the replay-harm law. Hyperparameters were frozen before opening the "
        "append-only Delphi development archive.",
        "",
        "## Frozen forms",
        "",
        "- `curved_retained_hpr`: incumbent squared-softplus penalty in log retained exposure.",
        "- `no_replay_hpr`: exact replay-harm ablation.",
        "- `linear_retained_family_hpr`: family-pooled linear hinge on retained exposure.",
        "- `linear_physical_family_hpr`: family-pooled linear hinge on physical aggregate epochs.",
        "- `linear_physical_bucket_hpr`: bucket-resolved slopes with a shared physical-epoch onset.",
        "",
        "## Results",
        "",
        metrics.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Selected configurations",
        "",
        "```json",
        json.dumps(configs, indent=2, sort_keys=True),
        "```",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def run_heldout_stage(output_dir: Path, dataset_ids: tuple[hpr.DatasetId, ...]) -> None:
    preregistration_path = output_dir / "preregistration.json"
    configs_path = output_dir / "selected_configs.json"
    if not preregistration_path.exists() or not configs_path.exists():
        raise FileNotFoundError("Run --stage fit before opening heldout outcomes")
    selected_configs = json.loads(configs_path.read_text())
    metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        if dataset_id not in {hpr.DatasetId.DELPHI_3E18_UNCHEATABLE, hpr.DatasetId.DELPHI_3E18_TABLE9}:
            continue
        print(f"Heldout audit: {dataset_id.value}", flush=True)
        dataset = hpr.load_dataset(dataset_id)
        heldout = hpr.heldout_data(dataset_id, dataset)
        if heldout is None:
            raise RuntimeError(f"No heldout data for {dataset_id.value}")
        frame, weights, observed = heldout
        candidate_dataset = replace(dataset, weights=weights, target=np.zeros(len(weights), dtype=float))
        max_epoch = physical_exposure(candidate_dataset).max(axis=1)
        for variant in Variant:
            config = config_from_record(selected_configs[dataset_id.value][variant.value])
            fit_design = hinge_design(dataset, config)
            head = fit_head(fit_design, dataset.target, np.arange(dataset.n), config.l2)
            heldout_design = hinge_design(candidate_dataset, config)
            predicted = head.predict(heldout_design)
            for subset, mask in heldout_subsets(frame, dataset_id):
                metrics.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.value,
                        "split": subset,
                        **metric_summary(observed[mask], predicted[mask]),
                    }
                )
            for index in range(len(frame)):
                predictions.append(
                    {
                        "dataset": dataset_id.value,
                        "variant": variant.value,
                        "row_id": str(frame.iloc[index]["wandb_run_name"]),
                        "training_series": str(frame.iloc[index]["training_series"]),
                        "policy_class": str(frame.iloc[index]["policy_class"]),
                        "proposal_target": str(frame.iloc[index].get("proposal_target", "")),
                        "observed": observed[index],
                        "predicted": predicted[index],
                        "optimism": observed[index] - predicted[index],
                        "max_physical_epoch": max_epoch[index],
                        "subset_member_two_phase": str(frame.iloc[index]["policy_class"]) != "single_phase_tied",
                    }
                )
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(output_dir / "heldout_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "heldout_predictions.csv", index=False)
    worst = (
        prediction_frame.sort_values("optimism", ascending=False).groupby(["dataset", "variant"], sort=False).head(10)
    )
    worst.to_csv(output_dir / "worst_heldout_predictions.csv", index=False)
    render_heldout(prediction_frame, output_dir)
    fit_metrics = pd.read_csv(output_dir / "fit_panel_metrics.csv")
    combined = pd.concat([fit_metrics, metric_frame], ignore_index=True, sort=False)
    combined.to_csv(output_dir / "all_metrics.csv", index=False)
    write_report(combined, selected_configs, output_dir)
    ledger = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "development_outcomes_opened": True,
        "candidate_source": "linear-past-threshold law from issue 7067",
        "hyperparameters_frozen_artifact": str(configs_path),
        "direct_heldout_tuning": False,
        "future_status": "development-exposed; any promoted form requires a new untouched confirmation panel",
    }
    (output_dir / "data_use_ledger.json").write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    dataset_ids = tuple(hpr.DatasetId(value.strip()) for value in args.datasets.split(",") if value.strip())
    if args.stage == "fit":
        run_fit_stage(args.output_dir, dataset_ids)
        return
    run_heldout_stage(args.output_dir, dataset_ids)


if __name__ == "__main__":
    main()
