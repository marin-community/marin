# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

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
"""Select nonlinear transition laws jointly across fit panels.

The nonlinear state law is shared across targets and swarms. Linear response
amplitudes and ridge strength remain target specific. Selection minimizes the
worst ratio of panel OOF RMSE to that family's panel-optimal OOF RMSE; mean
log-ratio breaks ties. Delphi heldouts are opened only after selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.mechanistic_models import (  # noqa: E402
    ModelConfig,
    build_design,
    candidate_configs,
    fit_nonnegative_ridge,
    round19_posterior_precision_candidate_configs,
    round20_capacity_gated_candidate_configs,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.screen_portfolio import (  # noqa: E402
    DASHBOARD,
    load_panel,
)

RESEARCH_DIR = Path(__file__).resolve().parent.parent
ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
OUTPUT = ROOT / "shared_transition_audit"
PANEL_IDS = (
    "300m_uncheatable",
    "300m_table9",
    "delphi_3e18_uncheatable",
    "delphi_3e18_table9",
    "production_uncheatable",
    "starcoder_cosine_starcoder_bpb",
    "starcoder_wsd80_starcoder_bpb",
)
FAMILY_SOURCES = {
    "retained_state_ode": ROOT / "initial_screen",
    "posterior_precision_debt": ROOT / "round19_posterior_precision",
    "capacity_gated_precision": ROOT / "round20_capacity_gated",
}
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}}


def config_map() -> dict[str, ModelConfig]:
    configs = (
        *candidate_configs(),
        *round19_posterior_precision_candidate_configs(),
        *round20_capacity_gated_candidate_configs(),
    )
    return {config.key: config for config in configs}


def best_per_panel_rows(family: str, source: Path) -> pd.DataFrame:
    frames = []
    for panel_id in PANEL_IDS:
        path = source / panel_id / "hyperparameter_screen.csv"
        gate.assert_sealed_absent(path)
        frame = pd.read_csv(path)
        frames.append(frame.loc[frame["family"].eq(family)].copy())
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["panel", "config", "rmse", "l2"]).groupby(["panel", "config"], as_index=False).first()


def select_shared(family: str, source: Path) -> tuple[pd.Series, pd.DataFrame]:
    rows = best_per_panel_rows(family, source)
    panel_optimal = rows.groupby("panel")["rmse"].min().rename("panel_optimal_rmse")
    rows = rows.join(panel_optimal, on="panel")
    rows["relative_rmse"] = rows["rmse"] / rows["panel_optimal_rmse"]
    summary = (
        rows.groupby("config", as_index=False)
        .agg(
            worst_relative_rmse=("relative_rmse", "max"),
            mean_log_relative_rmse=("relative_rmse", lambda values: float(np.mean(np.log(values)))),
            mean_relative_rmse=("relative_rmse", "mean"),
        )
        .sort_values(["worst_relative_rmse", "mean_log_relative_rmse", "config"])
    )
    selected = summary.iloc[0]
    selected_rows = rows.loc[rows["config"].eq(selected["config"])].copy()
    return selected, selected_rows


def heldout_evaluation(
    bundle: dict[str, Any],
    target: str,
    config: ModelConfig,
    l2: float,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    panel_id = f"delphi_3e18_{target}"
    panel, _dataset = load_panel(bundle, panel_id)
    fit_design = build_design(panel, panel.weights, config)
    model = fit_nonnegative_ridge(
        fit_design,
        panel.observed,
        np.arange(panel.n),
        config,
        l2,
    )
    rows = bundle["swarms"]["delphi_3e18"]["rows"]
    all_weights = np.asarray([[row["phase0"], row["phase1"]] for row in rows], dtype=float)
    all_design = build_design(panel, all_weights, config)
    prediction = model.predict_design(all_design)
    records = []
    for row, predicted in zip(rows, prediction, strict=True):
        observed = row["observed"].get(target)
        if row["split"] != "heldout" or row["isSharedAlias"] or row["policyFamily"] != "two_phase" or observed is None:
            continue
        records.append(
            {
                "target": target,
                "row_id": row["name"],
                "panel": row["panel"],
                "observed": float(observed),
                "predicted": float(predicted),
                "optimism": float(observed) - float(predicted),
            }
        )
    predictions = pd.DataFrame(records)
    metrics, bins = gate.metrics(predictions["observed"].to_numpy(), predictions["predicted"].to_numpy())
    return metrics, predictions, bins


def main() -> None:
    gate.assert_sealed_absent(DASHBOARD)
    bundle = json.loads(DASHBOARD.read_text())
    configs = config_map()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    selection_records = []
    panel_records = []
    metric_records = []
    prediction_frames = []
    bin_frames = []
    for family, source in FAMILY_SOURCES.items():
        selected, selected_rows = select_shared(family, source)
        config_key = str(selected["config"])
        config = configs[config_key]
        selection_records.append({"family": family, **selected.to_dict()})
        panel_records.extend(selected_rows.to_dict("records"))
        for target in ("uncheatable", "table9"):
            panel_id = f"delphi_3e18_{target}"
            l2 = float(selected_rows.loc[selected_rows["panel"].eq(panel_id), "l2"].iloc[0])
            summary, predictions, bins = heldout_evaluation(bundle, target, config, l2)
            metric_records.append(
                {
                    "family": family,
                    "config": config_key,
                    "target": target,
                    "split": "heldout_policy_matched",
                    "l2": l2,
                    **summary,
                }
            )
            predictions.insert(0, "family", family)
            predictions.insert(1, "config", config_key)
            prediction_frames.append(predictions)
            bin_frame = pd.DataFrame(bins)
            bin_frame.insert(0, "family", family)
            bin_frame.insert(1, "target", target)
            bin_frames.append(bin_frame)

    selection_frame = pd.DataFrame(selection_records)
    panel_frame = pd.DataFrame(panel_records)
    metric_frame = pd.DataFrame(metric_records)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    bin_frame = pd.concat(bin_frames, ignore_index=True)
    selection_frame.to_csv(OUTPUT / "shared_selection.csv", index=False)
    panel_frame.to_csv(OUTPUT / "selected_panel_metrics.csv", index=False)
    metric_frame.to_csv(OUTPUT / "heldout_metrics.csv", index=False)
    prediction_frame.to_csv(OUTPUT / "heldout_predictions.csv", index=False)
    bin_frame.to_csv(OUTPUT / "heldout_calibration_bins.csv", index=False)

    heatmap = panel_frame.pivot(index="family", columns="panel", values="relative_rmse")
    figure = px.imshow(
        heatmap,
        color_continuous_scale="RdYlGn_r",
        color_continuous_midpoint=1.05,
        text_auto=".3f",
        aspect="auto",
        labels={"color": "OOF RMSE / panel-optimal"},
        title="Cost of sharing one transition law across targets and swarms",
    )
    figure.update_layout(template="plotly_white", width=1500, height=520)
    figure.write_html(OUTPUT / "shared_transition_fit_cost.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = [
        "# Shared transition-law audit",
        "",
        "Selection was frozen as minimax panel-relative grouped-OOF RMSE; mean log-ratio breaks ties. "
        "Linear amplitudes and ridge remain panel specific. Delphi heldouts were evaluated only after selection.",
        "",
        "## Selected laws",
        "",
        selection_frame.to_markdown(index=False),
        "",
        "## Frozen Delphi heldouts",
        "",
        metric_frame.to_markdown(index=False),
        "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
