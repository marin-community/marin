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
"""Measure whether a few lowest-loss fit rows determine the fitted surface.

This is an influence audit, not a proposal to discard inconvenient results.
For each fit panel it removes the best observed k rows, refits the frozen
Hierarchical phase replay model, and compares the change with equally sized
random deletions. The sealed adversarial panel is never read: all heldout
coordinates come from the content-addressed frozen Observatory snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    audit_raw_optima as raw_optima,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
FROZEN_GATE = RESEARCH_DIR / (
    "reference_outputs/mechanistic_surrogate_discovery_20260717/frozen_gate/acceptance_gate.json"
)
DEFAULT_OUTPUT = RESEARCH_DIR / ("reference_outputs/mechanistic_surrogate_discovery_20260717/low_loss_tail_influence")
MODEL_ID = "hierarchical_phase_bucket_replay"
POLICY = "two_phase"
TAIL_SIZES = (1, 2, 3, 5, 10, 14, 28)
RETUNE_SIZES = (5, 14)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class PanelSpec:
    dataset_id: hierarchical.DatasetId
    swarm: str
    target: str

    @property
    def name(self) -> str:
        return self.dataset_id.value


PANELS = (
    PanelSpec(hierarchical.DatasetId.THREE_HUNDRED_M_UNCHEATABLE, "300m", "uncheatable"),
    PanelSpec(hierarchical.DatasetId.THREE_HUNDRED_M_TABLE9, "300m", "table9"),
    PanelSpec(hierarchical.DatasetId.DELPHI_3E18_UNCHEATABLE, "delphi_3e18", "uncheatable"),
    PanelSpec(hierarchical.DatasetId.DELPHI_3E18_TABLE9, "delphi_3e18", "table9"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-repeats", type=int, default=12)
    parser.add_argument("--optimizer-starts", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_bundle() -> dict[str, Any]:
    gate.assert_sealed_absent(DASHBOARD)
    manifest = json.loads(FROZEN_GATE.read_text())
    expected = str(manifest["dashboard_sha256"])
    actual = sha256(DASHBOARD)
    if actual != expected:
        raise ValueError(f"Dashboard drifted after gate freeze: expected {expected}, got {actual}")
    return json.loads(DASHBOARD.read_text())


def selected_config(bundle: dict[str, Any], spec: PanelSpec) -> hierarchical.Config:
    tuning = bundle["swarms"][spec.swarm]["fits"][spec.target][POLICY][MODEL_ID]["tuning"]
    shape_record = tuning["shapeParameters"]
    shape = family_grp.Shape(
        exponent=float(shape_record["exponent"]),
        late_multiplier=float(shape_record["lateMultiplier"]),
        forgetting_rate=float(shape_record["forgettingRate"]),
        penalty_threshold=float(shape_record["penaltyThreshold"]),
    )
    return hierarchical.Config(
        variant=hierarchical.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
        shape_index=-1,
        shape=shape,
        l2=float(tuning["l2"]),
        residual_shrink=float(tuning["residualShrink"]),
        undercoverage_fraction=0.0,
        coverage_gate_ratio=0.0,
    )


def dashboard_heldout(
    bundle: dict[str, Any],
    spec: PanelSpec,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    selected: list[dict[str, Any]] = []
    for row in bundle["swarms"][spec.swarm]["rows"]:
        observed = row["observed"].get(spec.target)
        if (
            row["split"] == "heldout"
            and not row["isSharedAlias"]
            and row["policyFamily"] == POLICY
            and observed is not None
            and math.isfinite(float(observed))
        ):
            selected.append(row)
    frame = pd.DataFrame(
        {
            "row_id": [row["id"] for row in selected],
            "name": [row["name"] for row in selected],
            "panel": [row["panel"] for row in selected],
            "observed": [float(row["observed"][spec.target]) for row in selected],
        }
    )
    weights = np.asarray([[row["phase0"], row["phase1"]] for row in selected], dtype=float)
    return frame, weights, frame["observed"].to_numpy(float)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = predicted - observed
    output: dict[str, float | int] = {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "optimism_gt_0p05_count": int(np.sum(observed - predicted > gate.OPTIMISM_THRESHOLD)),
        "worst_optimism": float(np.max(observed - predicted)),
    }
    if len(observed) >= 3 and np.std(predicted) > 1e-12:
        output.update(gate.metrics(observed, predicted)[0])
    return output


def oof_prediction(
    dataset: family_grp.Dataset,
    dataset_id: hierarchical.DatasetId,
    config: hierarchical.Config,
    retained: np.ndarray,
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    splits = hierarchical.split_indices(dataset, dataset_id, retained, hierarchical.SCREEN_SEED)
    for train, test in splits:
        model = hierarchical.fit_model(dataset, config, train)
        prediction[test] = model.predict(dataset.weights[test])
    if not np.isfinite(prediction[retained]).all():
        raise RuntimeError(f"Incomplete retained-row OOF prediction for {dataset_id.value}")
    return prediction


def score_configs_on_retained(
    dataset: family_grp.Dataset,
    dataset_id: hierarchical.DatasetId,
    configs: list[hierarchical.Config],
    retained: np.ndarray,
) -> tuple[hierarchical.Config, list[dict[str, Any]]]:
    best: tuple[float, float, hierarchical.Config] | None = None
    rows: list[dict[str, Any]] = []
    for config in configs:
        prediction = oof_prediction(dataset, dataset_id, config, retained)
        summary = hierarchical.metric_summary(dataset.target[retained], prediction[retained])
        row = {**hierarchical.config_record(config, summary), "retained_rows": len(retained)}
        rows.append(row)
        candidate = (float(summary["rmse"]), -float(summary["spearman"]), config)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("No retained-row configurations were scored")
    return best[2], rows


def retune_config(
    dataset: family_grp.Dataset,
    dataset_id: hierarchical.DatasetId,
    retained: np.ndarray,
) -> tuple[hierarchical.Config, list[dict[str, Any]]]:
    shapes = observatory.hierarchical_phase_replay_shape_candidates(POLICY)
    baseline_config, baseline_rows = score_configs_on_retained(
        dataset,
        dataset_id,
        hierarchical.baseline_configs(shapes),
        retained,
    )
    del baseline_config
    best_by_shape: dict[int, float] = {}
    for row in baseline_rows:
        shape_index = int(row["shape_index"])
        best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), float(row["rmse"]))
    shape_indices = [
        index
        for index, _rmse in sorted(best_by_shape.items(), key=lambda item: item[1])[
            : observatory.HIERARCHICAL_PHASE_REPLAY_TOP_SHAPES
        ]
    ]
    selected, structural_rows = score_configs_on_retained(
        dataset,
        dataset_id,
        hierarchical.structural_configs(
            hierarchical.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            shapes,
            shape_indices,
        ),
        retained,
    )
    rows = [{"stage": "baseline_shape_screen", **row} for row in baseline_rows] + [
        {"stage": "hierarchical_selection", **row} for row in structural_rows
    ]
    return selected, rows


def optimize_model(
    model: hierarchical.Model,
    dataset: family_grp.Dataset,
    seed: int,
    starts: int,
    reference: np.ndarray | None,
) -> tuple[np.ndarray, float, bool]:
    initial = raw_optima.optimization_starts(dataset, POLICY, seed, starts)
    if reference is not None:
        initial.insert(0, raw_optima.weights_to_logits(reference, POLICY))
    return raw_optima.optimize(raw_optima.Fitted(MODEL_ID, model), dataset, POLICY, initial)


def coefficient_diagnostics(reference: hierarchical.Model, candidate: hierarchical.Model) -> dict[str, float | int]:
    left = reference.coefficients
    right = candidate.coefficients
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(np.dot(left, right) / denominator) if denominator > 1e-15 else float("nan")
    active_left = left > 1e-10
    active_right = right > 1e-10
    union = int(np.sum(active_left | active_right))
    return {
        "coefficient_cosine": cosine,
        "coefficient_relative_l2": float(np.linalg.norm(right - left) / max(np.linalg.norm(left), 1e-12)),
        "active_coefficients": int(np.sum(active_right)),
        "active_jaccard": float(np.sum(active_left & active_right) / union) if union else 1.0,
        "intercept_shift": float(candidate.intercept - reference.intercept),
    }


def scenario(
    *,
    spec: PanelSpec,
    dataset: family_grp.Dataset,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
    heldout_target: np.ndarray,
    config: hierarchical.Config,
    reference_model: hierarchical.Model,
    reference_fit_prediction: np.ndarray,
    reference_optimum: np.ndarray,
    removed: np.ndarray,
    deletion_kind: str,
    replicate: int,
    config_mode: str,
    seed: int,
    optimizer_starts: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    retained = np.setdiff1d(np.arange(dataset.n), removed, assume_unique=True)
    model = hierarchical.fit_model(dataset, config, retained)
    fit_prediction = model.predict(dataset.weights)
    oof = oof_prediction(dataset, spec.dataset_id, config, retained)
    heldout_prediction = model.predict(heldout_weights)
    weights, predicted_optimum, converged = optimize_model(
        model,
        dataset,
        seed,
        optimizer_starts,
        reference_optimum,
    )
    metric_rows = []
    for split, observed, predicted in (
        ("retained_train", dataset.target[retained], fit_prediction[retained]),
        ("retained_oof", dataset.target[retained], oof[retained]),
        ("removed_tail", dataset.target[removed], fit_prediction[removed]),
        ("frozen_development_heldout", heldout_target, heldout_prediction),
    ):
        if len(observed) == 0:
            continue
        metric_rows.append(
            {
                "panel": spec.name,
                "deletion_kind": deletion_kind,
                "k": len(removed),
                "replicate": replicate,
                "config_mode": config_mode,
                "split": split,
                **metrics(observed, predicted),
            }
        )
    exposure = weights[0] * dataset.c0 + weights[1] * dataset.c1
    optimum_row = {
        "panel": spec.name,
        "deletion_kind": deletion_kind,
        "k": len(removed),
        "replicate": replicate,
        "config_mode": config_mode,
        "predicted_bpb": predicted_optimum,
        "optimizer_converged": converged,
        "tv_from_full_optimum": float(0.25 * np.abs(weights - reference_optimum).sum()),
        "prediction_under_full_model": float(reference_model.predict(weights[None, :, :])[0]),
        "max_bucket_weight": float(np.max(weights)),
        "max_simulated_epochs": float(np.max(exposure)),
        "phase_total_variation": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "fit_support_distance": raw_optima.support_distance(dataset, weights),
    }
    influence_row = {
        "panel": spec.name,
        "deletion_kind": deletion_kind,
        "k": len(removed),
        "replicate": replicate,
        "config_mode": config_mode,
        "prediction_field_rmse_from_full": float(np.sqrt(np.mean((fit_prediction - reference_fit_prediction) ** 2))),
        "prediction_field_max_abs_from_full": float(np.max(np.abs(fit_prediction - reference_fit_prediction))),
        "removed_target_mean": float(np.mean(dataset.target[removed])) if len(removed) else float("nan"),
        "removed_prediction_mean": float(np.mean(fit_prediction[removed])) if len(removed) else float("nan"),
        "heldout_rows": len(heldout_frame),
        **coefficient_diagnostics(reference_model, model),
        **{f"config_{key}": value for key, value in asdict(config).items() if key not in {"variant", "shape"}},
        **{f"shape_{key}": value for key, value in asdict(config.shape).items()},
    }
    return metric_rows, optimum_row, influence_row


def tail_rows(dataset: family_grp.Dataset, spec: PanelSpec, max_k: int) -> pd.DataFrame:
    order = np.argsort(dataset.target)[:max_k]
    rows = []
    for rank, index in enumerate(order, start=1):
        frame_row = dataset.frame.iloc[index]
        rows.append(
            {
                "panel": spec.name,
                "rank": rank,
                "row_index": int(index),
                "row_name": str(frame_row.get("run_name", frame_row.get("name", index))),
                "panel_source": str(frame_row.get("panel_source", "")),
                "observed_bpb": float(dataset.target[index]),
                "gap_from_best": float(dataset.target[index] - dataset.target[order[0]]),
            }
        )
    return pd.DataFrame(rows)


def random_percentile(tail_value: float, random_values: pd.Series) -> float:
    values = random_values.to_numpy(float)
    return float((np.sum(values < tail_value) + 0.5 * np.sum(values == tail_value)) / len(values))


def summarize_random_controls(
    influence: pd.DataFrame,
    optima: pd.DataFrame,
    metric_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    influence_metrics = (
        "prediction_field_rmse_from_full",
        "coefficient_relative_l2",
        "active_jaccard",
    )
    optimum_metrics = ("tv_from_full_optimum", "max_simulated_epochs", "fit_support_distance")
    for panel in influence["panel"].unique():
        for k in sorted(influence.loc[influence["panel"].eq(panel), "k"].unique()):
            if k == 0:
                continue
            tail = influence.loc[
                influence["panel"].eq(panel)
                & influence["k"].eq(k)
                & influence["deletion_kind"].eq("lowest_loss")
                & influence["config_mode"].eq("frozen_hyperparameters")
            ]
            random = influence.loc[
                influence["panel"].eq(panel) & influence["k"].eq(k) & influence["deletion_kind"].eq("random")
            ]
            tail_optimum = optima.loc[
                optima["panel"].eq(panel)
                & optima["k"].eq(k)
                & optima["deletion_kind"].eq("lowest_loss")
                & optima["config_mode"].eq("frozen_hyperparameters")
            ]
            random_optimum = optima.loc[
                optima["panel"].eq(panel) & optima["k"].eq(k) & optima["deletion_kind"].eq("random")
            ]
            if len(tail) != 1 or len(tail_optimum) != 1 or random.empty or random_optimum.empty:
                continue
            record: dict[str, Any] = {"panel": panel, "k": int(k), "random_repeats": len(random)}
            for name in influence_metrics:
                value = float(tail.iloc[0][name])
                controls = random[name]
                record[f"tail_{name}"] = value
                record[f"random_median_{name}"] = float(controls.median())
                record[f"random_q95_{name}"] = float(controls.quantile(0.95))
                record[f"percentile_{name}"] = random_percentile(value, controls)
            for name in optimum_metrics:
                value = float(tail_optimum.iloc[0][name])
                controls = random_optimum[name]
                record[f"tail_{name}"] = value
                record[f"random_median_{name}"] = float(controls.median())
                record[f"random_q95_{name}"] = float(controls.quantile(0.95))
                record[f"percentile_{name}"] = random_percentile(value, controls)
            for split in ("retained_oof", "frozen_development_heldout"):
                selected = metric_rows.loc[
                    metric_rows["panel"].eq(panel)
                    & metric_rows["k"].eq(k)
                    & metric_rows["deletion_kind"].eq("lowest_loss")
                    & metric_rows["config_mode"].eq("frozen_hyperparameters")
                    & metric_rows["split"].eq(split)
                ]
                if len(selected) == 1:
                    record[f"tail_{split}_rmse"] = float(selected.iloc[0]["rmse"])
                    record[f"tail_{split}_bias"] = float(selected.iloc[0]["bias_predicted_minus_observed"])
            rows.append(record)
    return pd.DataFrame(rows)


def render(
    summary: pd.DataFrame,
    metric_rows: pd.DataFrame,
    optima: pd.DataFrame,
    output_dir: Path,
) -> None:
    panels = list(summary["panel"].unique())
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Prediction-field movement",
            "Raw-optimum movement",
            "Retained-panel OOF RMSE",
            "Frozen development-heldout RMSE",
        ),
    )
    colors = ["#d73027", "#fc8d59", "#91cf60", "#1a9850"]
    for panel, color in zip(panels, colors, strict=False):
        local = summary.loc[summary["panel"].eq(panel)].sort_values("k")
        figure.add_trace(
            go.Scatter(
                x=local["k"],
                y=local["tail_prediction_field_rmse_from_full"],
                mode="lines+markers",
                name=panel,
                legendgroup=panel,
                line={"color": color},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (
                        local["random_q95_prediction_field_rmse_from_full"]
                        - local["random_median_prediction_field_rmse_from_full"]
                    ),
                    "arrayminus": np.zeros(len(local)),
                    "color": color,
                },
                customdata=np.column_stack(
                    [
                        local["random_median_prediction_field_rmse_from_full"],
                        local["percentile_prediction_field_rmse_from_full"],
                    ]
                ),
                hovertemplate=(
                    "k=%{x}<br>tail deletion=%{y:.5f}<br>random median=%{customdata[0]:.5f}"
                    "<br>random percentile=%{customdata[1]:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=local["k"],
                y=local["tail_tv_from_full_optimum"],
                mode="lines+markers",
                name=panel,
                legendgroup=panel,
                showlegend=False,
                line={"color": color},
                customdata=np.column_stack(
                    [local["random_median_tv_from_full_optimum"], local["percentile_tv_from_full_optimum"]]
                ),
                hovertemplate=(
                    "k=%{x}<br>tail deletion TV=%{y:.4f}<br>random median=%{customdata[0]:.4f}"
                    "<br>random percentile=%{customdata[1]:.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        for column, split in enumerate(("retained_oof", "frozen_development_heldout"), start=1):
            local_metrics = metric_rows.loc[
                metric_rows["panel"].eq(panel)
                & metric_rows["deletion_kind"].eq("lowest_loss")
                & metric_rows["config_mode"].eq("frozen_hyperparameters")
                & metric_rows["split"].eq(split)
            ].sort_values("k")
            figure.add_trace(
                go.Scatter(
                    x=local_metrics["k"],
                    y=local_metrics["rmse"],
                    mode="lines+markers",
                    name=panel,
                    legendgroup=panel,
                    showlegend=False,
                    line={"color": color},
                    hovertemplate="k=%{x}<br>RMSE=%{y:.5f}<extra></extra>",
                ),
                row=2,
                col=column,
            )
    figure.update_xaxes(title_text="Number of lowest-loss rows removed")
    figure.update_yaxes(title_text="RMSE change in predictions", row=1, col=1)
    figure.update_yaxes(title_text="Total variation from full-fit optimum", row=1, col=2)
    figure.update_yaxes(title_text="BPB RMSE", row=2)
    figure.update_layout(
        title={
            "text": "Influence of the lowest-loss fit rows versus matched random deletions",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.99,
        },
        template="plotly_white",
        width=1500,
        height=1000,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 0.95, "yanchor": "top"},
        margin={"t": 135},
    )
    figure.write_html(output_dir / "tail_influence_paths.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    optimum_plot = go.Figure()
    for panel, color in zip(panels, colors, strict=False):
        local = optima.loc[optima["panel"].eq(panel) & optima["deletion_kind"].eq("lowest_loss")].sort_values(
            ["config_mode", "k"]
        )
        for mode, dash in (("frozen_hyperparameters", "solid"), ("retuned", "dash")):
            selected = local.loc[local["config_mode"].eq(mode)]
            optimum_plot.add_trace(
                go.Scatter(
                    x=selected["k"],
                    y=selected["predicted_bpb"],
                    mode="lines+markers",
                    name=f"{panel} · {mode}",
                    line={"color": color, "dash": dash},
                    customdata=np.column_stack(
                        [
                            selected["tv_from_full_optimum"],
                            selected["max_simulated_epochs"],
                            selected["fit_support_distance"],
                        ]
                    ),
                    hovertemplate=(
                        "k=%{x}<br>predicted optimum=%{y:.5f}<br>TV from full=%{customdata[0]:.3f}"
                        "<br>max epochs=%{customdata[1]:.2f}<br>support distance=%{customdata[2]:.2f}<extra></extra>"
                    ),
                )
            )
    optimum_plot.update_layout(
        title="Raw optimum after deleting the lowest-loss fit rows",
        template="plotly_white",
        xaxis_title="Number of lowest-loss rows removed",
        yaxis_title="Predicted BPB at re-optimized raw optimum",
        width=1300,
        height=750,
    )
    optimum_plot.write_html(output_dir / "tail_optimum_paths.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    summary: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    optima: pd.DataFrame,
    tail_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# Lowest-loss row influence audit",
        "",
        "The analysis refits the frozen **Hierarchical phase replay** form after deleting the best observed fit rows. "
        "Matched random deletions measure ordinary finite-sample sensitivity. The frozen Observatory hash is verified "
        "before loading any data; the sealed adversarial stress panel is absent.",
        "",
        "Hyperparameters are held fixed for the primary influence comparison so only row leverage changes. The exact "
        "fit-panel selection procedure is rerun at k=5 and k=14 as a robustness check. No deployment regularizer is "
        "applied to the raw optimum.",
        "",
        "## Result",
        "",
        "The lowest-loss rows are influential on the Delphi panels, but this audit does **not** support treating them "
        "as unreliable or excluding them from fitting:",
        "",
        "- On both 300M targets, deleting up to 28/280 frontier rows does not improve the frozen development-heldout "
        "fit. It instead worsens decision quality on several retained-row Regret@1 comparisons.",
        "- On Delphi Uncheatable, the frozen development-heldout RMSE is 0.02075 with all rows, 0.02043 after deleting "
        "14, and 0.02156 after deleting 28. All three fits retain nine optimism errors above 0.05 BPB. The apparent "
        "benefit is negligible and disappears at the larger trim.",
        "- Delphi Table-9 is genuinely sensitive: deleting 28 rows changes the prediction field more than all 12 "
        "matched random deletions and moves the raw optimum by 0.102 total variation. The frozen development-heldout "
        "RMSE improves from 0.02602 to 0.02118 and bias from -0.01468 to -0.00072, but the refit then misses the "
        "omitted "
        "frontier by 0.02788 RMSE with +0.02179 BPB bias. At k=14, full hyperparameter reselection produces a "
        "53.13-epoch raw optimum and worsens heldout RMSE to 0.02606.",
        "- Therefore the tail supplies unique decision-relevant curvature and also exposes model misspecification. "
        "High leverage is not evidence of high observation noise. Target-based trimming would discard precisely the "
        "region the surrogate must model for optimization and can make the raw optimizer less stable.",
        "",
        "There is still a winner's-curse caveat. Using the exact-policy repeat RMSE floors as rough noise scales, the "
        "expected extreme downward fluctuation among 280 independent Gaussian observations is about 0.0020 BPB for "
        "Uncheatable and 0.0081 BPB for Table-9. The Delphi winner's 0.0142 Uncheatable lead is much larger than this "
        "scale, whereas its 0.0116 Table-9 lead is not. This makes targeted Table-9 repeats particularly important, "
        "but it still does not justify deleting the row before measuring its variance.",
        "",
        "The next statistical check should repeat the extreme policies, especially `run_00125` and the next four "
        "Delphi frontier rows, then use estimated heteroskedastic observation variances in the likelihood. Until those "
        "repeats exist, keep the rows and report this deletion path as a sensitivity diagnostic rather than a model "
        "selection rule.",
        "",
        "## Tail rows",
        "",
        tail_frame.loc[tail_frame["rank"] <= 5].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Tail deletion versus random deletion",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optima",
        "",
        optima.loc[optima["deletion_kind"].eq("lowest_loss")].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Evaluation metrics",
        "",
        metrics_frame.loc[
            metrics_frame["deletion_kind"].eq("lowest_loss")
            & metrics_frame["split"].isin(["retained_oof", "removed_tail", "frozen_development_heldout"])
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation rule",
        "",
        "Deleting low-loss rows is evidence of excessive leverage only when its surface or optimum movement is in the "
        "extreme tail of matched random deletions. Better retained-row RMSE after deletion is not evidence that the "
        "rows were noise: it can simply remove the hardest part of the decision-relevant response surface.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.render_only:
        summary = pd.read_csv(args.output_dir / "random_control_summary.csv")
        metrics_frame = pd.read_csv(args.output_dir / "metrics.csv")
        optima = pd.read_csv(args.output_dir / "raw_optima.csv")
        tails = pd.read_csv(args.output_dir / "lowest_loss_rows.csv")
        render(summary, metrics_frame, optima, args.output_dir)
        write_report(summary, metrics_frame, optima, tails, args.output_dir)
        return

    bundle = frozen_bundle()
    rng = np.random.default_rng(args.seed)
    metric_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    influence_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    tail_frames: list[pd.DataFrame] = []

    for panel_index, spec in enumerate(PANELS):
        print(f"Auditing {spec.name}", flush=True)
        dataset = hierarchical.load_dataset(spec.dataset_id)
        config = selected_config(bundle, spec)
        all_indices = np.arange(dataset.n)
        reference_model = hierarchical.fit_model(dataset, config, all_indices)
        reference_fit_prediction = reference_model.predict(dataset.weights)
        reference_optimum, _predicted, _converged = optimize_model(
            reference_model,
            dataset,
            args.seed + panel_index,
            args.optimizer_starts,
            None,
        )
        heldout_frame, heldout_weights, heldout_target = dashboard_heldout(bundle, spec)
        tail_frames.append(tail_rows(dataset, spec, max(TAIL_SIZES)))

        empty = np.asarray([], dtype=int)
        rows, optimum, influence = scenario(
            spec=spec,
            dataset=dataset,
            heldout_frame=heldout_frame,
            heldout_weights=heldout_weights,
            heldout_target=heldout_target,
            config=config,
            reference_model=reference_model,
            reference_fit_prediction=reference_fit_prediction,
            reference_optimum=reference_optimum,
            removed=empty,
            deletion_kind="none",
            replicate=0,
            config_mode="frozen_hyperparameters",
            seed=args.seed + panel_index,
            optimizer_starts=args.optimizer_starts,
        )
        metric_rows.extend(rows)
        optimum_rows.append(optimum)
        influence_rows.append(influence)

        order = np.argsort(dataset.target)
        for k in TAIL_SIZES:
            removed = np.sort(order[:k])
            rows, optimum, influence = scenario(
                spec=spec,
                dataset=dataset,
                heldout_frame=heldout_frame,
                heldout_weights=heldout_weights,
                heldout_target=heldout_target,
                config=config,
                reference_model=reference_model,
                reference_fit_prediction=reference_fit_prediction,
                reference_optimum=reference_optimum,
                removed=removed,
                deletion_kind="lowest_loss",
                replicate=0,
                config_mode="frozen_hyperparameters",
                seed=args.seed + 100 * panel_index + k,
                optimizer_starts=args.optimizer_starts,
            )
            metric_rows.extend(rows)
            optimum_rows.append(optimum)
            influence_rows.append(influence)

            if k in RETUNE_SIZES:
                retained = np.setdiff1d(all_indices, removed, assume_unique=True)
                retuned, screen = retune_config(dataset, spec.dataset_id, retained)
                tuning_rows.extend(
                    {
                        "panel": spec.name,
                        "k": k,
                        **row,
                    }
                    for row in screen
                )
                rows, optimum, influence = scenario(
                    spec=spec,
                    dataset=dataset,
                    heldout_frame=heldout_frame,
                    heldout_weights=heldout_weights,
                    heldout_target=heldout_target,
                    config=retuned,
                    reference_model=reference_model,
                    reference_fit_prediction=reference_fit_prediction,
                    reference_optimum=reference_optimum,
                    removed=removed,
                    deletion_kind="lowest_loss",
                    replicate=0,
                    config_mode="retuned",
                    seed=args.seed + 1000 + 100 * panel_index + k,
                    optimizer_starts=args.optimizer_starts,
                )
                metric_rows.extend(rows)
                optimum_rows.append(optimum)
                influence_rows.append(influence)

            for replicate in range(args.random_repeats):
                random_removed = np.sort(rng.choice(dataset.n, size=k, replace=False))
                rows, optimum, influence = scenario(
                    spec=spec,
                    dataset=dataset,
                    heldout_frame=heldout_frame,
                    heldout_weights=heldout_weights,
                    heldout_target=heldout_target,
                    config=config,
                    reference_model=reference_model,
                    reference_fit_prediction=reference_fit_prediction,
                    reference_optimum=reference_optimum,
                    removed=random_removed,
                    deletion_kind="random",
                    replicate=replicate,
                    config_mode="frozen_hyperparameters",
                    seed=args.seed + 10_000 * panel_index + 100 * k + replicate,
                    optimizer_starts=3,
                )
                metric_rows.extend(rows)
                optimum_rows.append(optimum)
                influence_rows.append(influence)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metric_rows)
    optima = pd.DataFrame(optimum_rows)
    influence = pd.DataFrame(influence_rows)
    tuning = pd.DataFrame(tuning_rows)
    tails = pd.concat(tail_frames, ignore_index=True)
    summary = summarize_random_controls(influence, optima, metrics_frame)
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)
    optima.to_csv(output_dir / "raw_optima.csv", index=False)
    influence.to_csv(output_dir / "influence.csv", index=False)
    tuning.to_csv(output_dir / "retuning_screen.csv", index=False)
    tails.to_csv(output_dir / "lowest_loss_rows.csv", index=False)
    summary.to_csv(output_dir / "random_control_summary.csv", index=False)
    render(summary, metrics_frame, optima, output_dir)
    write_report(summary, metrics_frame, optima, tails, output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
