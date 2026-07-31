# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit deployment-region identification among OOF-equivalent surrogates.

This is a diagnostic, not an ensemble or a post-hoc calibration method. The
candidate set is frozen to pre-search models whose grouped OOF RMSE is within
the acceptance gate's five-percent tolerance of the best model for a target.
The coordinate-disjoint Delphi heldouts are scored only after this set is
selected.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    freeze_baseline_gate as gate,
)

DEFAULT_DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_GATE = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/frozen_gate"
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/identification_audit"
TARGETS = ("uncheatable", "table9")
MODEL_COLORS = ("#2166ac", "#67a9cf", "#fdae61", "#b2182b", "#4d9221", "#762a83")
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def assert_sealed_absent(path: Path) -> None:
    gate.assert_sealed_absent(path)


def dashboard_predictions(bundle: dict[str, Any], target: str, model: str) -> pd.DataFrame:
    swarm = bundle["swarms"]["delphi_3e18"]
    rows = swarm["rows"]
    values = swarm["predictions"][target]["two_phase"][model]["prediction"]
    records = []
    for row, prediction in zip(rows, values, strict=True):
        if row["split"] == "heldout" and row["isSharedAlias"]:
            continue
        if row["split"] == "heldout" and row["policyFamily"] != "two_phase":
            continue
        records.append(
            {
                "target": target,
                "model": model,
                "row_id": row["name"],
                "split": "fit_oof" if row["split"] == "fit" else "heldout_policy_matched",
                "observed": row["observed"][target],
                "predicted": prediction,
                "panel": row["panel"],
                "method": row["method"],
                "max_epoch": max(row["totalEpochs"]),
                "max_weight": max(max(row["phase0"]), max(row["phase1"])),
                "phase_tv": 0.5 * sum(abs(a - b) for a, b in zip(row["phase0"], row["phase1"], strict=True)),
                "aggregate_kl": row["diagnostics"].get("aggregateKlToProportional"),
            }
        )
    return pd.DataFrame(records)


def external_predictions(target: str, model: str) -> pd.DataFrame:
    specifications = {
        "inverse_deficit_log_link": (
            "reference_outputs/model_improvement_round2_conditioned_replay_link_20260716/predictions.csv",
            "inverse_power_deficit_conditioned_replay",
            "log_reducible_bpb",
        ),
        "early_family_asymmetric": (
            "reference_outputs/deficit_output_link_asymmetric_20260716/predictions.csv",
            "inverse_power_deficit_early_family_asymmetric_surplus",
            "identity_raw_bpb" if target == "uncheatable" else "log_reducible_bpb",
        ),
    }
    relative, variant, link = specifications[model]
    path = RESEARCH_DIR / relative
    assert_sealed_absent(path)
    frame = pd.read_csv(path)
    dataset = f"delphi_3e18_{target}"
    selected = frame.loc[
        frame["dataset"].eq(dataset) & frame["deficit_variant"].eq(variant) & frame["link"].eq(link)
    ].copy()
    selected["target"] = target
    selected["model"] = model
    selected["split"] = selected["split"].replace({"heldout": "heldout_policy_matched"})
    return selected[["target", "model", "row_id", "split", "observed", "predicted"]]


def equivalent_models(metrics: pd.DataFrame, target: str, tolerance: float) -> list[str]:
    selected = metrics.loc[
        metrics["swarm"].eq("delphi_3e18")
        & metrics["target"].eq(target)
        & metrics["policy"].eq("two_phase")
        & metrics["split"].eq("fit_oof")
    ].copy()
    best = float(selected["rmse"].min())
    return selected.loc[selected["rmse"] <= best * (1.0 + tolerance), "model"].tolist()


def append_row_metadata(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    rows = {row["name"]: row for row in bundle["swarms"]["delphi_3e18"]["rows"] if not row["isSharedAlias"]}
    output = frame.copy()
    for column, getter in (
        ("panel", lambda row: row["panel"]),
        ("method", lambda row: row["method"]),
        ("max_epoch", lambda row: max(row["totalEpochs"])),
        ("max_weight", lambda row: max(max(row["phase0"]), max(row["phase1"]))),
        ("phase_tv", lambda row: 0.5 * sum(abs(a - b) for a, b in zip(row["phase0"], row["phase1"], strict=True))),
        ("aggregate_kl", lambda row: row["diagnostics"].get("aggregateKlToProportional")),
        ("policy_family", lambda row: row["policyFamily"]),
        ("is_shared_alias", lambda row: row["isSharedAlias"]),
    ):
        output[column] = output["row_id"].map(lambda value, getter=getter: getter(rows[str(value)]))
    return output


def prediction_matrix(frame: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.Series]:
    selected = frame.loc[frame["split"].eq(split)]
    matrix = selected.pivot(index="row_id", columns="model", values="predicted").sort_index()
    observed = selected.groupby("row_id", sort=True)["observed"].first().reindex(matrix.index)
    if matrix.isna().any().any():
        raise ValueError(f"Incomplete prediction matrix for {split}")
    return matrix, observed


def pairwise_rows(target: str, split: str, matrix: pd.DataFrame) -> list[dict[str, Any]]:
    output = []
    for left, right in combinations(matrix.columns, 2):
        difference = matrix[left] - matrix[right]
        output.append(
            {
                "target": target,
                "split": split,
                "left_model": left,
                "right_model": right,
                "rms_disagreement": float(np.sqrt(np.mean(difference**2))),
                "max_abs_disagreement": float(np.max(np.abs(difference))),
                "spearman": float(spearmanr(matrix[left], matrix[right]).statistic),
            }
        )
    return output


def selection_rows(target: str, matrix: pd.DataFrame, observed: pd.Series) -> list[dict[str, Any]]:
    best_observed = float(observed.min())
    output = []
    for model in matrix.columns:
        selected_id = str(matrix[model].idxmin())
        output.append(
            {
                "target": target,
                "model": model,
                "selected_row": selected_id,
                "selected_predicted": float(matrix.loc[selected_id, model]),
                "selected_observed": float(observed.loc[selected_id]),
                "regret_at_1": float(observed.loc[selected_id] - best_observed),
                "selected_optimism": float(observed.loc[selected_id] - matrix.loc[selected_id, model]),
            }
        )
    return output


def envelope_rows(target: str, matrix: pd.DataFrame, observed: pd.Series, metadata: pd.DataFrame) -> pd.DataFrame:
    indexed = metadata.drop_duplicates("row_id").set_index("row_id")
    output = pd.DataFrame(
        {
            "target": target,
            "row_id": matrix.index,
            "observed": observed,
            "prediction_min": matrix.min(axis=1),
            "prediction_max": matrix.max(axis=1),
            "prediction_range": matrix.max(axis=1) - matrix.min(axis=1),
            "prediction_std": matrix.std(axis=1, ddof=0),
            "mean_prediction": matrix.mean(axis=1),
        }
    ).reset_index(drop=True)
    for column in ("panel", "method", "max_epoch", "max_weight", "phase_tv", "aggregate_kl"):
        output[column] = output["row_id"].map(indexed[column])
    return output.sort_values("prediction_range", ascending=False)


def plot_calibration(target: str, frame: pd.DataFrame, output: Path) -> None:
    models = list(dict.fromkeys(frame["model"]))
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Fit-panel grouped OOF", "Frozen policy-matched heldouts"))
    for model_index, model in enumerate(models):
        color = MODEL_COLORS[model_index % len(MODEL_COLORS)]
        for column_index, split in enumerate(("fit_oof", "heldout_policy_matched"), start=1):
            selected = frame.loc[frame["model"].eq(model) & frame["split"].eq(split)]
            figure.add_trace(
                go.Scatter(
                    x=selected["predicted"],
                    y=selected["observed"],
                    mode="markers",
                    name=model,
                    legendgroup=model,
                    showlegend=column_index == 1,
                    marker={"color": color, "size": 6, "opacity": 0.7},
                    customdata=np.column_stack([selected["row_id"], selected["panel"]]),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>pred=%{x:.5f}<br>obs=%{y:.5f}<extra>%{fullData.name}</extra>",
                ),
                row=1,
                col=column_index,
            )

    values = np.concatenate([frame["observed"].to_numpy(), frame["predicted"].to_numpy()])
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    for column_index in (1, 2):
        figure.add_trace(
            go.Scatter(
                x=[low, high], y=[low, high], mode="lines", line={"color": "#666", "dash": "dash"}, showlegend=False
            ),
            row=1,
            col=column_index,
        )
    figure.update_xaxes(title_text="Predicted BPB")
    figure.update_yaxes(title_text="Observed BPB")
    figure.update_layout(
        title=f"OOF-equivalent prediction laws: {target}",
        template="plotly_white",
        width=1500,
        height=670,
        legend={"orientation": "h", "y": -0.16},
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".5f")


def main() -> None:
    args = parse_args()
    assert_sealed_absent(args.dashboard)
    bundle = json.loads(args.dashboard.read_text())
    baseline_metrics = pd.read_csv(args.gate_dir / "baseline_metrics.csv")
    gate_record = json.loads((args.gate_dir / "acceptance_gate.json").read_text())
    tolerance = float(gate_record["acceptance_gate"]["core_oof_rmse_relative_tolerance"])

    all_predictions = []
    pairwise = []
    selections = []
    envelopes = []
    selected_models: dict[str, list[str]] = {}
    for target in TARGETS:
        models = equivalent_models(baseline_metrics, target, tolerance)
        selected_models[target] = models
        target_frames = []
        for model in models:
            if model in ("inverse_deficit_log_link", "early_family_asymmetric"):
                target_frames.append(external_predictions(target, model))
            else:
                target_frames.append(dashboard_predictions(bundle, target, model))
        frame = append_row_metadata(pd.concat(target_frames, ignore_index=True), bundle)
        frame = frame.loc[
            frame["split"].eq("fit_oof") | (frame["policy_family"].eq("two_phase") & ~frame["is_shared_alias"])
        ].copy()
        all_predictions.append(frame)
        for split in ("fit_oof", "heldout_policy_matched"):
            matrix, observed = prediction_matrix(frame, split)
            pairwise.extend(pairwise_rows(target, split, matrix))
            if split == "heldout_policy_matched":
                selections.extend(selection_rows(target, matrix, observed))
                envelopes.append(envelope_rows(target, matrix, observed, frame.loc[frame["split"].eq(split)]))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plot_calibration(target, frame, args.output_dir / f"{target}_oof_equivalent_calibration.html")

    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    pairwise_frame = pd.DataFrame(pairwise)
    selection_frame = pd.DataFrame(selections)
    envelope_frame = pd.concat(envelopes, ignore_index=True)
    predictions_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    pairwise_frame.to_csv(args.output_dir / "pairwise_disagreement.csv", index=False)
    selection_frame.to_csv(args.output_dir / "heldout_selections.csv", index=False)
    envelope_frame.to_csv(args.output_dir / "heldout_prediction_envelopes.csv", index=False)

    summary_rows = []
    for target, local in envelope_frame.groupby("target", sort=True):
        summary_rows.append(
            {
                "target": target,
                "models": len(selected_models[target]),
                "heldouts": len(local),
                "median_prediction_range": local["prediction_range"].median(),
                "p90_prediction_range": local["prediction_range"].quantile(0.9),
                "max_prediction_range": local["prediction_range"].max(),
                "rows_range_gt_0p02": int((local["prediction_range"] > 0.02).sum()),
                "rows_range_gt_0p05": int((local["prediction_range"] > 0.05).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    report = [
        "# OOF-equivalent surrogate identification audit",
        "",
        "This artifact is diagnostic only. It does not average models or calibrate their outputs. Model membership was frozen by the pre-search 5% grouped-OOF RMSE tolerance before heldout scoring.",
        "",
        "## Equivalent sets",
        "",
        *[f"- **{target}:** {', '.join(models)}" for target, models in selected_models.items()],
        "",
        "## Heldout prediction envelope",
        "",
        markdown_table(summary, list(summary.columns)),
        "",
        "## Heldout selections",
        "",
        markdown_table(selection_frame, list(selection_frame.columns)),
        "",
        "## Largest row-level disagreements",
        "",
        markdown_table(
            envelope_frame.groupby("target", group_keys=False).head(10),
            [
                "target",
                "row_id",
                "panel",
                "observed",
                "prediction_min",
                "prediction_max",
                "prediction_range",
                "max_epoch",
                "phase_tv",
            ],
        ),
        "",
        "## Interpretation",
        "",
        "If OOF-equivalent models have prediction ranges comparable to or larger than the candidate improvements under discussion, ordinary grouped CV does not identify the deployment surface. This is evidence against selecting an extrapolation law from the current fit panel; it is not evidence for model averaging.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"selected_models": selected_models, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
