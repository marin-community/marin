# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Screen a source-identified phase-response direction across model scales."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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
    analyze_olmo_base_easy_per_component_dsp_decision_300m as component_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    transferred_phase_response_models as transferred,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round5_transferred_phase_response"
)
SCREEN_SEED = 20260719
N_SPLITS = 5
SHAPE_COUNT = 12
TOP_SHAPES = 3
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def source_configs() -> list[transferred.SourceConfig]:
    return [transferred.SourceConfig(tau, l2) for tau in (0.03, 0.1, 0.3, 1.0) for l2 in (0.01, 0.1, 1.0, 10.0)]


def target_configs() -> list[transferred.TargetConfig]:
    return [transferred.TargetConfig(l2, include_contrast_cost=True) for l2 in (0.0, 0.01, 0.1, 1.0, 10.0)]


def source_oof(panel: paired.PairedPanel, config: transferred.SourceConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    coefficients: list[np.ndarray] = []
    design, _names = transferred.source_design(panel, panel.weights, config)
    for train, test in paired_screen.folds(panel):
        model = transferred.fit_source_direction(panel, train, config)
        paired_test = test[panel.paired_mask[test]]
        prediction[paired_test] = model.source_head.predict(design[paired_test])
        coefficients.append(model.family_direction)
    return prediction, coefficients


def select_source(panel: paired.PairedPanel) -> tuple[transferred.SourceConfig, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    delta = panel.two_phase_target - panel.one_phase_target
    paired_mask = panel.paired_mask
    for config in source_configs():
        prediction, coefficients = source_oof(panel, config)
        coefficient_matrix = np.stack(coefficients)
        signs = np.sign(coefficient_matrix)
        sign_agreement = np.mean(np.abs(np.mean(signs, axis=0)))
        rows.append(
            {
                "source_panel": panel.name,
                "config": config.key,
                "config_json": json.dumps(asdict(config), sort_keys=True),
                "mean_family_sign_agreement": float(sign_agreement),
                **paired_screen.scalar_metrics(delta[paired_mask], prediction[paired_mask]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    return transferred.SourceConfig(**json.loads(table.iloc[0]["config_json"])), table


def target_delta_oof(
    panel: paired.PairedPanel,
    source: transferred.SourceDirection,
    config: transferred.TargetConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows = []
    for fold, (train, test) in enumerate(paired_screen.folds(panel)):
        model = transferred.fit_target_phase(panel, train, source, config)
        paired_test = test[panel.paired_mask[test]]
        prediction[paired_test] = model.predict_delta(panel, panel.weights[paired_test])
        rows.append(
            {
                "fold": fold,
                "amplitude": model.amplitude,
                "contrast_coefficient": model.contrast_coefficient,
            }
        )
    return prediction, pd.DataFrame(rows)


def select_target(
    panel: paired.PairedPanel,
    source: transferred.SourceDirection,
) -> tuple[transferred.TargetConfig, pd.DataFrame]:
    rows = []
    delta = panel.two_phase_target - panel.one_phase_target
    paired_mask = panel.paired_mask
    for config in target_configs():
        prediction, coefficients = target_delta_oof(panel, source, config)
        rows.append(
            {
                "target_panel": panel.name,
                "config": config.key,
                "config_json": json.dumps(asdict(config), sort_keys=True),
                "amplitude_mean": float(coefficients["amplitude"].mean()),
                "amplitude_std": float(coefficients["amplitude"].std(ddof=1)),
                "contrast_mean": float(coefficients["contrast_coefficient"].mean()),
                "contrast_zero_folds": int(np.sum(coefficients["contrast_coefficient"] < 1e-10)),
                **paired_screen.scalar_metrics(delta[paired_mask], prediction[paired_mask]),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    return transferred.TargetConfig(**json.loads(table.iloc[0]["config_json"])), table


def tied_family_dataset(panel: paired.PairedPanel, subset: np.ndarray | None = None) -> family_grp.Dataset:
    indices = np.arange(panel.n) if subset is None else np.asarray(subset, dtype=int)
    aggregate = panel.aggregate_weights[indices]
    weights = np.stack([aggregate, aggregate], axis=1)
    target = np.nan_to_num(panel.one_phase_target[indices], nan=0.0)
    return family_grp.Dataset(
        frame=panel.frame.iloc[indices].reset_index(drop=True),
        target=target,
        weights=weights,
        c0=np.asarray(panel.c0, dtype=float),
        c1=np.asarray(panel.c1, dtype=float),
        domains=panel.domain_names,
        family_names=panel.family_names,
        family_members=panel.family_members,
        quality=np.full(panel.m, -1, dtype=int),
    )


def tied_shapes() -> tuple[family_grp.Shape, ...]:
    candidates = family_grp.shape_candidates(family_grp.Variant.BUCKET_RESOLVED, SHAPE_COUNT)
    tied = [replace(shape, late_multiplier=1.0, forgetting_rate=0.0) for shape in candidates]
    return tuple(dict.fromkeys(tied))


def select_tied_spine(panel: paired.PairedPanel) -> tuple[hierarchical.Config, pd.DataFrame]:
    paired_indices = np.flatnonzero(panel.paired_mask)
    dataset = tied_family_dataset(panel, paired_indices)
    splits = component_dsp.panel_stratified_folds(dataset.frame, n_splits=N_SPLITS, seed=SCREEN_SEED)
    shapes = tied_shapes()
    _baseline, _prediction, baseline_rows = hierarchical.score_configs(
        dataset,
        hierarchical.baseline_configs(shapes),
        splits,
    )
    best_by_shape: dict[int, float] = {}
    for row in baseline_rows:
        shape_index = int(row["shape_index"])
        best_by_shape[shape_index] = min(best_by_shape.get(shape_index, float("inf")), float(row["rmse"]))
    shape_indices = [index for index, _score in sorted(best_by_shape.items(), key=lambda item: item[1])[:TOP_SHAPES]]
    config, _prediction, structural_rows = hierarchical.score_configs(
        dataset,
        hierarchical.structural_configs(
            hierarchical.Variant.HIERARCHICAL_BUCKET_REPLAY,
            shapes,
            shape_indices,
        ),
        splits,
    )
    rows = pd.DataFrame(
        [{"panel": panel.name, "stage": "baseline_shape", **row} for row in baseline_rows]
        + [{"panel": panel.name, "stage": "hierarchical_spine", **row} for row in structural_rows]
    )
    return config, rows


def combined_oof(
    panel: paired.PairedPanel,
    spine_config: hierarchical.Config,
    source: transferred.SourceDirection,
    target_config: transferred.TargetConfig,
) -> pd.DataFrame:
    dataset = tied_family_dataset(panel)
    aggregate_prediction = np.full(panel.n, np.nan, dtype=float)
    phase_prediction = np.full(panel.n, np.nan, dtype=float)
    combined_prediction = np.full(panel.n, np.nan, dtype=float)
    fold_index = np.full(panel.n, -1, dtype=int)
    coefficient_rows: list[dict[str, float | int]] = []
    for fold, (train, test) in enumerate(paired_screen.folds(panel)):
        paired_train = train[panel.paired_mask[train]]
        spine = hierarchical.fit_model(dataset, spine_config, paired_train)
        aggregate_prediction[test] = spine.predict(dataset.weights[test])
        phase = transferred.fit_target_phase(panel, train, source, target_config)
        phase_prediction[test] = phase.predict_delta(panel, panel.weights[test])
        combined_prediction[test] = aggregate_prediction[test] + phase_prediction[test]
        fold_index[test] = fold
        coefficient_rows.append(
            {
                "fold": fold,
                "amplitude": phase.amplitude,
                "contrast_coefficient": phase.contrast_coefficient,
            }
        )
    if not np.isfinite(combined_prediction).all():
        raise RuntimeError(f"Incomplete combined OOF prediction for {panel.name}")
    result = panel.frame.copy()
    result["panel"] = panel.name
    result["fold"] = fold_index
    result["paired"] = panel.paired_mask
    result["observed_one_phase"] = panel.one_phase_target
    result["observed_two_phase"] = panel.two_phase_target
    result["observed_delta"] = panel.two_phase_target - panel.one_phase_target
    result["predicted_one_phase"] = aggregate_prediction
    result["predicted_delta"] = phase_prediction
    result["predicted_two_phase"] = combined_prediction
    result.attrs["coefficient_rows"] = coefficient_rows
    return result


def metrics_for_prediction(frame: pd.DataFrame) -> list[dict[str, Any]]:
    paired_mask = frame["paired"].to_numpy(bool)
    rows = []
    for evaluation, observed, predicted, mask in (
        (
            "one_phase_spine_oof",
            frame["observed_one_phase"].to_numpy(float),
            frame["predicted_one_phase"].to_numpy(float),
            paired_mask,
        ),
        (
            "phase_delta_oof",
            frame["observed_delta"].to_numpy(float),
            frame["predicted_delta"].to_numpy(float),
            paired_mask,
        ),
        (
            "two_phase_combined_oof",
            frame["observed_two_phase"].to_numpy(float),
            frame["predicted_two_phase"].to_numpy(float),
            np.ones(len(frame), dtype=bool),
        ),
        (
            "two_phase_zero_correction_oof",
            frame["observed_two_phase"].to_numpy(float),
            frame["predicted_one_phase"].to_numpy(float),
            np.ones(len(frame), dtype=bool),
        ),
    ):
        rows.append(
            {
                "panel": frame["panel"].iloc[0],
                "evaluation": evaluation,
                **paired_screen.scalar_metrics(observed[mask], predicted[mask]),
            }
        )
    return rows


def plot_predictions(frame: pd.DataFrame, output_path: Path) -> None:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Two-phase OOF", "Matched phase effect OOF"))
    plots = (
        (frame["observed_two_phase"], frame["predicted_two_phase"], np.ones(len(frame), dtype=bool)),
        (frame["observed_delta"], frame["predicted_delta"], frame["paired"].to_numpy(bool)),
    )
    for column, (observed, predicted, mask) in enumerate(plots, start=1):
        x = np.asarray(observed)[mask]
        y = np.asarray(predicted)[mask]
        minimum = float(min(x.min(), y.min()))
        maximum = float(max(x.max(), y.max()))
        fig.add_trace(
            go.Scatter(
                x=[minimum, maximum],
                y=[minimum, maximum],
                mode="lines",
                line={"dash": "dash", "color": "#87939b"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={"size": 7, "color": "#2f855a", "opacity": 0.7},
                name="OOF prediction",
                showlegend=column == 1,
            ),
            row=1,
            col=column,
        )
        fig.update_xaxes(title_text="Observed BPB or delta", row=1, col=column)
        fig.update_yaxes(title_text="Predicted BPB or delta", row=1, col=column)
    fig.update_layout(template="plotly_white", width=1350, height=570, title=frame["panel"].iloc[0])
    fig.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def evaluate_direction(source_scale: str, target_scale: str, target: str, output_dir: Path) -> dict[str, Any]:
    source_panel = paired_screen.load_panel(source_scale, target)
    target_panel = paired_screen.load_panel(target_scale, target)
    source_config, source_grid = select_source(source_panel)
    source = transferred.fit_source_direction(source_panel, np.arange(source_panel.n), source_config)
    target_config, target_grid = select_target(target_panel, source)
    spine_config, spine_grid = select_tied_spine(target_panel)
    prediction = combined_oof(target_panel, spine_config, source, target_config)

    direction = f"{source_scale}_to_{target_scale}_{target}"
    source_grid.to_csv(output_dir / f"{direction}_source_grid.csv", index=False)
    target_grid.to_csv(output_dir / f"{direction}_target_grid.csv", index=False)
    spine_grid.to_csv(output_dir / f"{direction}_spine_grid.csv", index=False)
    prediction.to_csv(output_dir / f"{direction}_oof_predictions.csv", index=False)
    plot_predictions(prediction, output_dir / f"{direction}_oof_predictions.html")
    coefficient_rows = pd.DataFrame(prediction.attrs["coefficient_rows"])
    coefficient_rows.to_csv(output_dir / f"{direction}_target_coefficients.csv", index=False)
    return {
        "direction": direction,
        "source_config": source_config.key,
        "target_config": target_config.key,
        "source_family_direction": json.dumps(
            dict(zip(source_panel.family_names, source.family_direction.tolist(), strict=True)), sort_keys=True
        ),
        "source_contrast_coefficient": source.source_contrast_coefficient,
        "spine_config": json.dumps(
            {
                "variant": spine_config.variant.value,
                "shape_index": spine_config.shape_index,
                "shape": asdict(spine_config.shape),
                "l2": spine_config.l2,
                "residual_shrink": spine_config.residual_shrink,
            },
            sort_keys=True,
        ),
        "metrics": metrics_for_prediction(prediction),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    metrics = []
    for target in ("uncheatable", "table9"):
        for source_scale, target_scale in (("300m", "delphi_3e18"), ("delphi_3e18", "300m")):
            result = evaluate_direction(source_scale, target_scale, target, args.output_dir)
            metrics.extend({"direction": result["direction"], **row} for row in result.pop("metrics"))
            summaries.append(result)
    summary_frame = pd.DataFrame(summaries)
    metric_frame = pd.DataFrame(metrics)
    summary_frame.to_csv(args.output_dir / "selected_configs_and_directions.csv", index=False)
    metric_frame.to_csv(args.output_dir / "oof_metrics.csv", index=False)
    report = [
        "# Round-five transferred phase-response screen",
        "",
        "The source scale learns a unit-norm three-family signed recency direction. The target scale fits only one signed amplitude and one nonnegative finite-contrast coefficient. The one-phase spine is independently fit with the tied restriction of hierarchical bucket-replay GRP.",
        "",
        "## OOF metrics",
        "",
        metric_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Selected directions",
        "",
        summary_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "No historical or adversarial heldout outcome was read in this screen.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()
