# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "fsspec==2026.1.0", "threadpoolctl==3.6.0"]
# ///
"""Replay frontier gain arithmetic and diagnose opposite-direction errors."""

from __future__ import annotations

import json
from pathlib import Path

import evaluate_anchored_frontier_transfer_20260917 as evaluation
import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from threadpoolctl import threadpool_limits

ROOT = evaluation.ROOT
OUTPUT = evaluation.OUTPUT


def main() -> None:
    inputs = [
        Path(__file__),
        Path(evaluation.__file__),
        OUTPUT / "manifest.json",
        OUTPUT / "predictions.csv",
        OUTPUT / "matched_pairs.csv",
        OUTPUT / "gain_metrics.csv",
        OUTPUT / "mirror_predictions.csv",
        OUTPUT / "cell_selections.csv",
        evaluation.CALIBRATION / "inputs/coordinates.npz",
        evaluation.CALIBRATION / "inputs/observations.csv",
        evaluation.SOURCE,
        ROOT / "frozen_baselines/manifest.json",
        *[ROOT / "crossed/fits" / state / "MTP-002.json" for state in evaluation.crossed.STATES],
    ]
    identity = {str(path): evaluation.sha(path) for path in inputs}
    path = OUTPUT / "independent_audit.json"
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached["inputs"] == identity
        assert all(evaluation.sha(OUTPUT / name) == digest for name, digest in cached["outputs"].items())
        print("Verified independent frontier audit.")
        return
    predictions = pd.read_csv(OUTPUT / "predictions.csv")
    pairs = pd.read_csv(OUTPUT / "matched_pairs.csv")
    reported = pd.read_csv(OUTPUT / "gain_metrics.csv")
    mirror = pd.read_csv(OUTPUT / "mirror_predictions.csv")
    observations = pd.read_csv(evaluation.CALIBRATION / "inputs/observations.csv")
    archive = observations[observations.cohort.eq("archive")].set_index("archive_id")
    assert predictions.run_id.equals(observations.run_id)
    errors = []
    for _, metric in reported.iterrows():
        group = pairs[pairs.cell.eq(metric.cell) & pairs.model.eq(metric.model)]
        if metric.source != "all":
            group = group[group.source.eq(metric.source)]
        measured = (
            archive.loc[group.control_id, "uncheatable"].to_numpy() - archive.loc[group.row_id, "uncheatable"].to_numpy()
        )
        assert np.max(np.abs(measured - group.observed_gain)) < 1e-12
        predicted = group.predicted_gain.to_numpy()
        chosen_gain = np.where(predicted > 1e-12, measured, 0)
        regret = np.maximum(measured, 0) - chosen_gain
        rebuilt = {
            "gain_rmse": np.linalg.norm(predicted - measured) / np.sqrt(len(measured)),
            "matched_regret": regret.mean(),
            "gain_optimism": predicted.mean() - measured.mean(),
        }
        errors.extend(abs(float(metric[key]) - value) for key, value in rebuilt.items())
    source = load_module(evaluation.SOURCE, "independent_frontier_primitives")
    anchor = source.ObjectiveFit.from_json(
        json.loads((evaluation.CALIBRATION / "fits/mariner_uncheatable.json").read_text())
    )
    scale = json.loads((ROOT / "frozen_baselines/manifest.json").read_text())["coordinate_sensitivity"][
        "mariner_inventory_scale"
    ]
    with np.load(evaluation.CALIBRATION / "inputs/coordinates.npz") as stored:
        weights, aggregate = stored["weights"].copy(), stored["aggregate"].copy()
    inventory = anchor.inventory * scale
    total = aggregate * inventory
    prefix = evaluation.ALPHA * weights[:, 0] * inventory
    tied_prefix = evaluation.ALPHA * total
    prediction_error = []
    with threadpool_limits(limits=1):
        for index, state in enumerate(evaluation.crossed.STATES):
            fit = json.loads((ROOT / "crossed/fits" / state / "MTP-002.json").read_text())["fit"]
            shape = fit["shape"]
            benefit_amplitude, harm_amplitude = np.split(np.asarray(fit["coefficients"]), 2)
            latent = []
            for start in (prefix, tied_prefix):
                benefit = source.benefit(total, shape["rate"], shape["power"]) - source.benefit(
                    start, shape["rate"], shape["power"]
                )
                harm = source.harm(total, shape["threshold"]) - source.harm(start, shape["threshold"])
                value = fit["intercept"] - benefit @ benefit_amplitude + harm @ harm_amplitude
                assert np.max(np.abs(value)) < source.LOG_CLIP
                latent.append(value)
            # The floor cancels exactly. expm1 independently avoids subtracting two close levels.
            gain = -np.exp(latent[1]) * np.expm1(latent[0] - latent[1])
            prediction_error.extend(np.abs(gain - predictions[f"gain::F{index}"]).tolist())
    assert max(errors) < 1e-12 and max(prediction_error) < 1e-12

    lookup = pairs.set_index(["model", "row_id"])
    decomposition = []
    for row in mirror.itertuples():
        left = lookup.loc[(str(row.model), str(row.left))]
        right = lookup.loc[(str(row.model), str(row.right))]
        assert left.control_id == right.control_id
        decomposition.append(
            {
                "cell": row.cell,
                "model": row.model,
                "left": row.left,
                "right": row.right,
                "control_id": left.control_id,
                "observed_odd": (left.observed_gain - right.observed_gain) / 2,
                "predicted_odd": (left.predicted_gain - right.predicted_gain) / 2,
                "observed_even": (left.observed_gain + right.observed_gain) / 2,
                "predicted_even": (left.predicted_gain + right.predicted_gain) / 2,
            }
        )
    parts = pd.DataFrame(decomposition)
    assert len(parts) == 192 * 8
    summaries = []
    for (cell, model), group in parts.groupby(["cell", "model"]):
        for component in ("odd", "even"):
            measured, predicted = group[f"observed_{component}"], group[f"predicted_{component}"]
            summaries.append(
                {
                    "cell": cell,
                    "model": model,
                    "component": component,
                    "pairs": len(group),
                    "observed_mean": measured.mean(),
                    "predicted_mean": predicted.mean(),
                    "observed_rms": np.sqrt(np.mean(measured**2)),
                    "predicted_rms": np.sqrt(np.mean(predicted**2)),
                    "rmse": np.sqrt(np.mean((predicted - measured) ** 2)),
                }
            )
    selected = pd.read_csv(OUTPUT / "cell_selections.csv")
    details = []
    mixtures = []
    for _, row in selected[selected.menu.eq("seed_matched") & selected.model.str.startswith("F")].iterrows():
        candidate = predictions[predictions.policy_id.eq(row.selected_policy)]
        fit_index = int(row.model[1:])
        assert candidate.primary.all()
        details.append(
            {
                "cell": row.cell,
                "model": row.model,
                "policy_id": row.selected_policy,
                "predicted_gain": row.predicted_gain_vs_tied,
                "measured_same_seed_gain": row.selected_same_seed_gain,
                "prefix_tv": candidate[f"prefix_tv::{fit_index}"].iloc[0],
                "action_tv": candidate[f"action_tv::{fit_index}"].iloc[0],
                "policy_tv": candidate[f"policy_tv::{fit_index}"].iloc[0],
                "tied_query_tv": candidate[f"tied_query_tv::{fit_index}"].iloc[0],
                "source": candidate.source.iloc[0],
                "candidate": candidate.candidate.iloc[0],
            }
        )
        if row.model != "F0":
            continue
        position = candidate.index[0]
        for index, bucket in enumerate(anchor.buckets):
            mixtures.append(
                {
                    "cell": row.cell,
                    "policy_id": row.selected_policy,
                    "bucket": bucket,
                    "phase0": weights[position, 0, index],
                    "phase1": weights[position, 1, index],
                    "aggregate": aggregate[position, index],
                }
            )
    files = {
        "odd_even_predictions.csv": parts,
        "odd_even_metrics.csv": pd.DataFrame(summaries),
        "selected_policy_details.csv": pd.DataFrame(details),
        "selected_policy_weights.csv": pd.DataFrame(mixtures),
    }
    for name, table in files.items():
        table.to_csv(OUTPUT / name, index=False)
    result = {
        "inputs": identity,
        "outputs": {name: evaluation.sha(OUTPUT / name) for name in files},
        "recomputed_metric_rows": len(reported),
        "metric_max_error": max(errors),
        "independent_prediction_max_error": max(prediction_error),
        "same_control_mirror_pairs": len(parts) // 8,
        "diagnostic_status": "Odd/even decomposition after failed primary test; no coefficients fitted or adjusted.",
    }
    write_json_atomic(path, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("inputs", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
