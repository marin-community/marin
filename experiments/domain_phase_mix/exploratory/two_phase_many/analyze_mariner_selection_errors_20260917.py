# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///

"""Decompose measured-bank selection disagreements into weighted components."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reference_outputs/mariner_local_validation_20260917"


def leave_one_component_out(atoms: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    results = []
    for (panel_name, target), rows in metrics.groupby(["panel", "target"]):
        panel = bench.load_panel(panel_name)
        group = panel.group(target)
        observed = (
            atoms[atoms.panel.eq(panel_name) & atoms.target.eq(target)]
            .pivot(index="coordinate_id", columns="component", values="bpb_mean")
            .reindex(columns=group.components)
            .dropna()
        )
        matrix = observed.to_numpy()
        predictions = {}
        for model in rows.model:
            parent = "banks/convex" if model.startswith("cv17_") else "banks"
            columns = []
            for i in range(len(group.components)):
                path = ROOT / parent / "fits" / model / panel_name / target / f"component_{i:03d}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    columns.append(
                        pd.Series(saved["prediction"], index=saved["coordinate_id"]).loc[observed.index].to_numpy()
                    )
            predictions[model] = np.asarray(columns).T
        for i, component in enumerate(group.components):
            weights = group.aggregation_weights.copy()
            weights[i] = 0.0
            weights /= weights.sum()
            y = matrix @ weights
            reference = int(np.argmin(predictions["hs17_mariner"] @ weights))
            for model, prediction in predictions.items():
                selected = int(np.argmin(prediction @ weights))
                results.append(
                    {
                        "panel": panel_name,
                        "target": target,
                        "omitted": component,
                        "model": model,
                        "regret": float(y[selected] - y.min()),
                        "regret_difference_vs_mariner": float(y[selected] - y[reference]),
                    }
                )
    frame = pd.DataFrame(results)
    frame.to_csv(output / "leave_one_component_out.csv", index=False)
    summary = (
        frame.assign(win=frame.regret_difference_vs_mariner < -1e-8, loss=frame.regret_difference_vs_mariner > 1e-8)
        .groupby(["panel", "target", "model"])
        .agg(
            variants=("omitted", "size"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
            min_regret_difference=("regret_difference_vs_mariner", "min"),
            max_regret_difference=("regret_difference_vs_mariner", "max"),
        )
    )
    summary.to_csv(output / "leave_one_component_out_summary.csv")


def main() -> None:
    metrics = pd.concat([pd.read_csv(ROOT / folder / "bank_metrics.csv") for folder in ("banks", "banks/convex")])
    metrics = metrics[metrics.labels.eq("current_weights_complete") & metrics.stratum.eq("all")]
    atoms = pd.read_csv(
        HERE / "reference_outputs/single_phase_heldout_benchmark_expanded_20260913/heldout_coordinate_components.csv"
    )
    records = []
    pairs = []
    for (panel_name, target), rows in metrics.groupby(["panel", "target"]):
        panel = bench.load_panel(panel_name)
        group = panel.group(target)
        observed = atoms[atoms.panel.eq(panel_name) & atoms.target.eq(target)].pivot(
            index="coordinate_id", columns="component", values="bpb_mean"
        )
        baseline = rows[rows.model.eq("hs17_mariner")].iloc[0]
        for _, challenger in rows[~rows.model.eq("hs17_mariner")].iterrows():
            if challenger.selected_id == baseline.selected_id:
                continue
            ids = [baseline.selected_id, challenger.selected_id]
            values = observed.loc[ids, list(group.components)].to_numpy()
            measured_delta = (values[1] - values[0]) * group.aggregation_weights
            np.testing.assert_allclose(
                measured_delta.sum(), challenger.selected_measured - baseline.selected_measured, atol=2e-12
            )
            for scoring in ("hs17_mariner", challenger.model):
                parent = "banks/convex" if scoring.startswith("cv17_") else "banks"
                predicted = []
                for i in range(len(group.components)):
                    path = ROOT / parent / "fits" / scoring / panel_name / target / f"component_{i:03d}.npz"
                    with np.load(path, allow_pickle=False) as saved:
                        lookup = pd.Series(saved["prediction"], index=saved["coordinate_id"])
                        predicted.append(lookup.loc[ids].to_numpy())
                prediction = np.asarray(predicted)
                predicted_delta = (prediction[:, 1] - prediction[:, 0]) * group.aggregation_weights
                pairs.append(
                    {
                        "panel": panel_name,
                        "target": target,
                        "challenger": challenger.model,
                        "scoring_model": scoring,
                        "baseline_coordinate": ids[0],
                        "challenger_coordinate": ids[1],
                        "measured_loss_difference": measured_delta.sum(),
                        "predicted_loss_difference": predicted_delta.sum(),
                    }
                )
                for i, component in enumerate(group.components):
                    records.append(
                        {
                            "panel": panel_name,
                            "target": target,
                            "challenger": challenger.model,
                            "scoring_model": scoring,
                            "component": component,
                            "weight": group.aggregation_weights[i],
                            "measured_weighted_difference": measured_delta[i],
                            "predicted_weighted_difference": predicted_delta[i],
                            "weighted_prediction_error": predicted_delta[i] - measured_delta[i],
                        }
                    )
    output = ROOT / "selection_error_components"
    output.mkdir(exist_ok=True)
    pd.DataFrame(pairs).to_csv(output / "pairs.csv", index=False)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "components.csv", index=False)
    own = frame[frame.challenger.eq(frame.scoring_model)]
    tops = (
        own.assign(abs_error=own.weighted_prediction_error.abs())
        .sort_values("abs_error", ascending=False)
        .groupby(["panel", "target", "challenger"])
        .head(8)
    )
    tops.to_csv(output / "largest_errors.csv", index=False)
    leave_one_component_out(atoms, metrics, output)
    print(pd.DataFrame(pairs).drop(columns=["baseline_coordinate", "challenger_coordinate"]).to_string(index=False))
    print(tops[tops.panel.eq("delphi_3e18_39bucket") & tops.target.eq("table9")].to_string(index=False))


if __name__ == "__main__":
    main()
