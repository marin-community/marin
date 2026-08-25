# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Audit paired-channel RPL at ordinary-RPL-selected nonlinear configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_aggregate_conditioned_replay_control_20260730 as benchmark  # noqa: E402
import paired_channel_retained_power_law_20260730 as paired  # noqa: E402
import retained_power_law_model_20260728 as rpl  # noqa: E402
import starcoder_wsd80_panel_20260728 as wsd80  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "paired_channel_rpl_estimation_20260730"
BASELINE_DIR = SCRIPT_DIR / "reference_outputs" / "rpl_repaired_baseline_screen_20260730"
TARGETS = ("uncheatable", "table9")
OUTER_SEED = 0
OUTER_SPLITS = 3
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260730
PRIMARY_TARGET = "uncheatable"


def shape_from_row(row: pd.Series) -> rpl.Shape:
    return rpl.Shape(
        benefit_exponent=float(row["benefit_exponent"]),
        benefit_offset=float(row["benefit_offset"]),
        damage_exponent=float(row["damage_exponent"]),
        damage_threshold=0.0,
        retention=float(row["retention"]),
        late_multiplier=float(row["late_multiplier"]),
        ordering_channel=bool(row["ordering_channel"]),
    )


def pair_indices(frame: pd.DataFrame, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tied = paired.tied_rows(weights)
    keys = frame["phase_correspondence_key"].astype(str).to_numpy()
    tied_by_key: dict[str, list[int]] = {}
    for index in np.flatnonzero(tied):
        tied_by_key.setdefault(keys[index], []).append(int(index))
    asymmetric = np.flatnonzero(~tied)
    anchors = []
    for index in asymmetric:
        matches = tied_by_key.get(keys[index], [])
        if len(matches) != 1:
            raise ValueError(f"row {index} has {len(matches)} tied counterparts")
        anchors.append(matches[0])
    return np.asarray(anchors, dtype=int), asymmetric


def bootstrap_pair_improvement(
    ordinary_error: np.ndarray,
    paired_error: np.ndarray,
) -> dict[str, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = len(ordinary_error)
    indices = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    ordinary_rmse = np.sqrt(np.mean(ordinary_error[indices] ** 2, axis=1))
    paired_rmse = np.sqrt(np.mean(paired_error[indices] ** 2, axis=1))
    improvement = (ordinary_rmse - paired_rmse) / ordinary_rmse
    return {
        "relative_improvement": float(
            (np.sqrt(np.mean(ordinary_error**2)) - np.sqrt(np.mean(paired_error**2)))
            / np.sqrt(np.mean(ordinary_error**2))
        ),
        "bootstrap_lower": float(np.quantile(improvement, 0.025)),
        "bootstrap_median": float(np.median(improvement)),
        "bootstrap_upper": float(np.quantile(improvement, 0.975)),
        "probability_positive": float(np.mean(improvement > 0.0)),
    }


def wsd_exact_pair_count() -> int:
    panel = wsd80.load_surface()
    tied = paired.tied_rows(panel.weights)
    aggregate = (
        wsd80.REALIZED_PHASE_0_FRACTION * panel.weights[:, 0, 1]
        + (1.0 - wsd80.REALIZED_PHASE_0_FRACTION) * panel.weights[:, 1, 1]
    )
    tied_aggregate = aggregate[tied]
    return int(sum(np.any(np.isclose(tied_aggregate, value, rtol=0.0, atol=1e-12)) for value in aggregate[~tied]))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    pair_rows = []
    bootstrap_rows = []
    fold_rows = []
    prediction_frames = []

    for target in TARGETS:
        dataset = benchmark.load_300m(target)
        geometry = benchmark.geometry_300m(dataset)
        folds = benchmark.grouped_folds(dataset.frame, OUTER_SEED, OUTER_SPLITS)
        parameters = pd.read_csv(BASELINE_DIR / f"diagnostic_300m_{target}" / "parameters_300m.csv").sort_values("fold")
        if (
            parameters["fold"].tolist() != list(range(OUTER_SPLITS))
            or parameters["seed"].tolist() != [OUTER_SEED] * OUTER_SPLITS
            or parameters["dataset"].tolist() != [f"300m_{target}"] * OUTER_SPLITS
        ):
            raise ValueError(f"missing ordinary nested-RPL selections for {target}")

        predictions = {
            "ordinary_rpl_pinned": np.full(dataset.n, np.nan),
            "paired_channel_rpl_pinned": np.full(dataset.n, np.nan),
        }
        for fold, (train, test) in enumerate(folds):
            row = parameters.iloc[fold]
            shape = shape_from_row(row)
            ridge = float(row["ridge"])
            train_design = rpl.design_matrix(dataset.weights[train], geometry, shape)
            test_design = rpl.design_matrix(dataset.weights[test], geometry, shape)
            multipliers = rpl.penalty_multipliers(geometry, shape)

            intercept, coefficients = rpl.solve_head(
                train_design,
                dataset.y[train],
                ridge,
                multipliers,
            )
            predictions["ordinary_rpl_pinned"][test] = intercept + test_design @ coefficients
            ordinary_active = int(np.count_nonzero(coefficients > 1e-10))

            paired_fit = paired.fit_fixed_shape(
                dataset.weights[train],
                dataset.y[train],
                dataset.frame.iloc[train]["phase_correspondence_key"].astype(str).to_numpy(),
                geometry,
                shape,
                ridge,
            )
            predictions["paired_channel_rpl_pinned"][test] = paired_fit.predict(dataset.weights[test])
            fold_rows.append(
                {
                    "target": target,
                    "fold": fold,
                    "benefit_exponent": shape.benefit_exponent,
                    "benefit_offset": shape.benefit_offset,
                    "damage_exponent": shape.damage_exponent,
                    "retention": shape.retention,
                    "late_multiplier": shape.late_multiplier,
                    "ordering_channel": shape.ordering_channel,
                    "ridge": ridge,
                    "ordinary_active_coefficients": ordinary_active,
                    "paired_active_coefficients": paired_fit.active_coefficients,
                    "aggregate_equations": paired_fit.aggregate_count,
                    "phase_equations": paired_fit.phase_count,
                }
            )

        anchors, asymmetric = pair_indices(dataset.frame, dataset.weights)
        observed_delta = dataset.y[asymmetric] - dataset.y[anchors]
        pair_errors = {}
        for model, predicted in predictions.items():
            if not np.all(np.isfinite(predicted)):
                raise ValueError(f"{target} {model} has missing OOF predictions")
            metric_rows.append(
                benchmark.metric_row(
                    dataset.name,
                    model,
                    OUTER_SEED,
                    dataset.y,
                    predicted,
                    dataset.weights,
                    folds,
                )
            )
            pair_rows.append(
                benchmark.paired_metric_row(
                    dataset.name,
                    model,
                    OUTER_SEED,
                    dataset.y,
                    predicted,
                    dataset.frame,
                    dataset.weights,
                )
            )
            predicted_delta = predicted[asymmetric] - predicted[anchors]
            pair_errors[model] = predicted_delta - observed_delta
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "target": target,
                        "model": model,
                        "row": np.arange(dataset.n),
                        "run_name": dataset.frame["run_name"].astype(str),
                        "phase_correspondence_key": dataset.frame["phase_correspondence_key"].astype(str),
                        "physically_tied": paired.tied_rows(dataset.weights),
                        "observed": dataset.y,
                        "predicted": predicted,
                    }
                )
            )

        bootstrap_rows.append(
            {
                "target": target,
                **bootstrap_pair_improvement(
                    pair_errors["ordinary_rpl_pinned"],
                    pair_errors["paired_channel_rpl_pinned"],
                ),
            }
        )

        ordinary_residual = predictions["ordinary_rpl_pinned"] - dataset.y
        anchor_residual = ordinary_residual[anchors]
        phase_innovation = ordinary_residual[asymmetric] - anchor_residual
        tied_count = int(paired.tied_rows(dataset.weights).sum())
        bootstrap_rows[-1].update(
            {
                "empirical_group_to_innovation_variance_ratio": float(
                    np.var(anchor_residual, ddof=1) / np.var(phase_innovation, ddof=1)
                ),
                "equal_channel_implied_ratio": float(tied_count / len(asymmetric)),
                "empirical_pair_residual_correlation": float(
                    np.corrcoef(anchor_residual, ordinary_residual[asymmetric])[0, 1]
                ),
                "equal_channel_implied_correlation": float(np.sqrt(tied_count / dataset.n)),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    pairs = pd.DataFrame(pair_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    folds = pd.DataFrame(fold_rows)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    pairs.to_csv(OUTPUT_DIR / "paired_metrics.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "paired_bootstrap.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "fold_parameters.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    summary = {}
    for target in TARGETS:
        ordinary = metrics[(metrics["dataset"] == f"300m_{target}") & (metrics["model"] == "ordinary_rpl_pinned")].iloc[
            0
        ]
        candidate = metrics[
            (metrics["dataset"] == f"300m_{target}") & (metrics["model"] == "paired_channel_rpl_pinned")
        ].iloc[0]
        ordinary_pair = pairs[(pairs["dataset"] == f"300m_{target}") & (pairs["model"] == "ordinary_rpl_pinned")].iloc[0]
        candidate_pair = pairs[
            (pairs["dataset"] == f"300m_{target}") & (pairs["model"] == "paired_channel_rpl_pinned")
        ].iloc[0]
        uncertainty = bootstrap[bootstrap["target"] == target].iloc[0]
        summary[target] = {
            "ordinary_rmse": float(ordinary["rmse"]),
            "candidate_rmse": float(candidate["rmse"]),
            "ordinary_tied_rmse": float(ordinary["one_phase_rmse"]),
            "candidate_tied_rmse": float(candidate["one_phase_rmse"]),
            "ordinary_pair_rmse": float(ordinary_pair["delta_rmse"]),
            "candidate_pair_rmse": float(candidate_pair["delta_rmse"]),
            "pair_relative_improvement": float(uncertainty["relative_improvement"]),
            "pair_bootstrap_lower": float(uncertainty["bootstrap_lower"]),
            "ordinary_regret": float(ordinary["fold_regret_at_1"]),
            "candidate_regret": float(candidate["fold_regret_at_1"]),
        }

    primary = summary[PRIMARY_TARGET]
    table9 = summary["table9"]
    passed = (
        primary["pair_relative_improvement"] >= 0.02
        and primary["pair_bootstrap_lower"] > 0.0
        and table9["candidate_pair_rmse"] <= table9["ordinary_pair_rmse"]
        and all(
            values["candidate_rmse"] <= 1.05 * values["ordinary_rmse"]
            and values["candidate_tied_rmse"] <= 1.05 * values["ordinary_tied_rmse"]
            and values["candidate_regret"] - values["ordinary_regret"] <= 0.002
            for values in summary.values()
        )
    )
    result = {
        "primary_target": PRIMARY_TARGET,
        "passed_frozen_stage1_gate": passed,
        "wsd80_exact_aggregate_pairs": wsd_exact_pair_count(),
        "targets": summary,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# Paired-channel retained-power-law audit",
        "",
        "This frozen Stage-1 test changes only the linear-head estimation loss. Nonlinear shape and ridge",
        "are pinned to the ordinary nested-RPL selections in every outer fold, preventing the candidate",
        "from winning through selection-mediated shape changes.",
        "",
        f"**Stage-1 gate: {'PASS' if passed else 'FAIL'}.**",
        "",
        "| Target | Ordinary RMSE | Paired RMSE | Ordinary delta RMSE | Paired delta RMSE "
        "| Delta change | Bootstrap 95% interval | Regret change |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        values = summary[target]
        interval = bootstrap[bootstrap["target"] == target].iloc[0]
        lines.append(
            f"| {target} | {values['ordinary_rmse']:.6f} | {values['candidate_rmse']:.6f} | "
            f"{values['ordinary_pair_rmse']:.6f} | {values['candidate_pair_rmse']:.6f} | "
            f"{100 * values['pair_relative_improvement']:+.2f}% | "
            f"[{100 * interval['bootstrap_lower']:+.2f}%, {100 * interval['bootstrap_upper']:+.2f}%] | "
            f"{values['candidate_regret'] - values['ordinary_regret']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"WSD80 has **{result['wsd80_exact_aggregate_pairs']}** asymmetric rows with an exactly observed",
            "tied aggregate counterpart under the realized 80/20 fractions. It therefore cannot evaluate",
            "this estimator; unchanged WSD80 behavior is an identity consequence of leaving the model form",
            "untouched, not supporting evidence for the paired loss.",
            "",
            "Independent review confirmed rejection. The pair bootstrap conditions on only three shared",
            "outer fits and is optimistic; its interval therefore cannot be used to rescue the failed gate.",
            "Changing only the two channel variances cannot match the observed full pair covariance, and a",
            "full-covariance successor is underpowered on 238 pairs. No successor weighting is promoted.",
            "",
            "The moment covariance diagnostics are descriptive only and did not select the equal-channel",
            "weight. No channel-weight sweep was performed.",
        ]
    )
    (OUTPUT_DIR / "stage1_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
