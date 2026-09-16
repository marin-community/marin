# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Common-row selection diagnostics for the offline two-phase transfer round."""

from __future__ import annotations

import argparse
from pathlib import Path

import fit_two_phase_link_transfer_20260907 as experiment
import numpy as np
import pandas as pd


def geometry(output: Path) -> None:
    module, panel, _ = experiment.inputs(str(output))
    rows = panel["pair_asymmetric_rows"]
    contrast = panel["phase1"] - panel["phase0"]
    records = []
    for objective in experiment.OBJECTIVES:
        for index in range(len(panel[f"{objective}_components"])):
            spine = experiment.load_spine(output, "final", objective, index)
            _, q, columns = experiment.basis_and_prediction(module, panel, spine)
            shape = spine.shape
            exposure = panel["epochs"]
            x = np.maximum(shape["rate"] * exposure, 1e-15)
            benefit_derivative = (
                shape["rate"] * shape["power"] * x ** (shape["power"] - 1) * np.exp(-(x ** shape["power"]))
            )
            offset = np.log1p(exposure) - shape["threshold"]
            harm_derivative = 2 * np.logaddexp(0, offset) / (1 + np.exp(-offset)) / (1 + exposure)
            alpha, beta = np.split(spine.head.coefficients, 2)
            directional_gradient = np.sum(
                (-alpha * benefit_derivative + beta * harm_derivative) * panel["inventory"] * contrast, axis=1
            )
            fisher_energy = np.sum(contrast**2 / np.maximum(panel["aggregate"], 1e-12), axis=1)
            reference = q[rows, None] * np.column_stack([directional_gradient[rows], fisher_energy[rows]])
            design = q[rows, None] * columns[rows]
            norms = np.linalg.norm(design, axis=0)
            scaled = np.divide(design, norms, out=np.zeros_like(design), where=norms > 1e-12)
            left, singular, _ = np.linalg.svd(scaled, full_matrices=False)
            rank = int(np.sum(singular > 1e-8))
            leverage = np.sum(left[:, :rank] ** 2, axis=1)
            projection = reference @ np.linalg.lstsq(reference, design, rcond=None)[0]
            residual_ratio = np.divide(
                np.linalg.norm(design - projection, axis=0), norms, out=np.zeros(2), where=norms > 1e-12
            )
            records.append(
                {
                    "objective": objective,
                    "component": index,
                    "rank": rank,
                    "condition": float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None,
                    "max_leverage": float(np.max(leverage)),
                    "benefit_residual_fraction": residual_ratio[0],
                    "harm_residual_fraction": residual_ratio[1],
                    "reference_scope": (
                        "net aggregate directional gradient plus local Fisher energy; "
                        "not a full reimplementation of SUR-013"
                    ),
                }
            )
    pd.DataFrame(records).to_csv(output / "geometry_span_audit.csv", index=False)


def summarize(output: Path) -> None:
    _, panel, _ = experiment.inputs(str(output))
    controls = pd.read_csv(output / "controls/oof_predictions.csv")
    all_metrics, all_pairs, bootstraps = [], [], []
    rng = np.random.default_rng(20260907)
    for objective in experiment.OBJECTIVES:
        candidate = pd.read_csv(output / f"predictions_{objective}.csv")
        assert len(candidate) == len(panel["runs"]) and candidate.row.nunique() == len(candidate)
        candidate = candidate.set_index("row").loc[np.arange(len(panel["runs"]))].reset_index()
        assert np.array_equal(candidate.run.to_numpy(str), panel["runs"])
        assert np.array_equal(candidate.group.to_numpy(str), panel["groups"])
        assert np.array_equal(candidate.fold.to_numpy(int), panel["outer_fold"])
        truth = panel[f"{objective}_outcomes"] @ panel[f"{objective}_aggregation_weights"]
        assert np.allclose(candidate.measured.to_numpy(float), truth, rtol=0, atol=1e-12)
        predictions = {arm: candidate[arm].to_numpy() for arm in experiment.ARMS}
        for model, group in controls.loc[(controls.target == objective) & controls.scored_oof].groupby("model"):
            assert len(group) == 518 and group.row_index.nunique() == 518
            indices = group.row_index.to_numpy(int)
            assert np.array_equal(group.run.to_numpy(str), panel["runs"][indices])
            assert np.array_equal(group.group.to_numpy(str), panel["groups"][indices])
            assert np.array_equal(group.outer_fold.to_numpy(int), panel["outer_fold"][indices])
            assert np.allclose(group.observed.to_numpy(float), truth[indices], rtol=0, atol=3e-6)
            prediction = np.full(len(truth), np.nan)
            prediction[group.row_index.to_numpy(int)] = group.prediction.to_numpy(float)
            predictions[model] = prediction
        scoreable = ~panel["calibration_mask"]
        a, t = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
        deltas = {}
        observed_delta = truth[a] - truth[t]
        for model, prediction in predictions.items():
            assert np.isfinite(prediction[scoreable]).all()
            for fold in (-1, 0, 1, 2):
                selected = scoreable if fold == -1 else panel["outer_fold"] == fold
                for name, mask in [
                    ("all", selected),
                    ("tied", selected & panel["physical_tied"]),
                    ("asymmetric", selected & ~panel["physical_tied"]),
                ]:
                    if np.any(mask):
                        all_metrics.append(
                            {
                                "objective": objective,
                                "model": model,
                                "fold": fold,
                                "population": name,
                                **experiment.metrics(truth[mask], prediction[mask], panel["physical_tied"][mask]),
                            }
                        )
            delta = prediction[a] - prediction[t]
            delta[np.abs(delta) < 1e-12] = 0
            deltas[model] = delta
            for i in range(len(a)):
                regret = max(observed_delta[i], 0) if delta[i] < 0 else max(-observed_delta[i], 0)
                all_pairs.append(
                    {
                        "objective": objective,
                        "model": model,
                        "group": panel["groups"][a[i]],
                        "fold": panel["outer_fold"][a[i]],
                        "measured_delta": observed_delta[i],
                        "predicted_delta": delta[i],
                        "decision_regret": regret,
                        "squared_error": (delta[i] - observed_delta[i]) ** 2,
                    }
                )
        samples = rng.integers(0, len(a), size=(3000, len(a)))
        for model in experiment.ARMS[1:]:
            for reference in ("aggregate", "hierarchical_phase_replay", "separate_heads", "effective_exposure_dsp"):
                if reference not in deltas:
                    continue
                candidate_delta, reference_delta = deltas[model], deltas[reference]
                se_candidate, se_reference = (candidate_delta - observed_delta) ** 2, (
                    reference_delta - observed_delta
                ) ** 2
                regret_candidate = np.where(
                    candidate_delta < 0, np.maximum(observed_delta, 0), np.maximum(-observed_delta, 0)
                )
                regret_reference = np.where(
                    reference_delta < 0, np.maximum(observed_delta, 0), np.maximum(-observed_delta, 0)
                )
                for metric, observed, bootstrap in [
                    (
                        "delta_rmse",
                        np.sqrt(se_candidate.mean()) - np.sqrt(se_reference.mean()),
                        np.sqrt(se_candidate[samples].mean(axis=1)) - np.sqrt(se_reference[samples].mean(axis=1)),
                    ),
                    (
                        "binary_decision_regret",
                        np.mean(regret_candidate - regret_reference),
                        (regret_candidate - regret_reference)[samples].mean(axis=1),
                    ),
                ]:
                    low, high = np.quantile(bootstrap, [0.025, 0.975])
                    bootstraps.append(
                        {
                            "objective": objective,
                            "candidate": model,
                            "reference": reference,
                            "metric": metric,
                            "difference": observed,
                            "ci_low": low,
                            "ci_high": high,
                            "groups": len(a),
                            "draws": len(samples),
                            "scope": (
                                "paired correspondence-group resampling of fixed OOF outputs; "
                                "not training randomness or source-disjoint confirmation"
                            ),
                        }
                    )
    pd.DataFrame(all_metrics).to_csv(output / "matched_metrics.csv", index=False)
    paired = pd.DataFrame(all_pairs)
    paired.to_csv(output / "matched_pairs.csv", index=False)
    pd.DataFrame(bootstraps).to_csv(output / "paired_bootstrap.csv", index=False)
    pair_summary = []
    for (objective, model), group in paired.groupby(["objective", "model"]):
        pair_summary.append(
            {
                "objective": objective,
                "model": model,
                "pairs": len(group),
                "delta_rmse": np.sqrt(group.squared_error.mean()),
                "decision_regret": group.decision_regret.mean(),
                "delta_bias": (group.predicted_delta - group.measured_delta).mean(),
                "delta_spearman": experiment.safe_rank(
                    group.measured_delta.to_numpy(), group.predicted_delta.to_numpy()
                ),
            }
        )
    summary = pd.DataFrame(pair_summary)
    summary.to_csv(output / "matched_pair_metrics.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("geometry", "summarize"))
    parser.add_argument("--output", type=Path, default=experiment.DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.phase == "geometry":
        geometry(args.output)
    else:
        summarize(args.output)


if __name__ == "__main__":
    main()
