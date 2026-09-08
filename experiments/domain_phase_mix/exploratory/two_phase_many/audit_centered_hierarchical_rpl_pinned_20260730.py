# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Separate centered-prior effects from shape-selection effects on 300M pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmark_aggregate_conditioned_replay_control_20260730 as benchmark  # noqa: E402
import benchmark_centered_hierarchical_rpl_20260730 as centered_benchmark  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    centered_hierarchical_rpl_20260730 as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "centered_hierarchical_rpl_physical_20260730"
CANDIDATE_PARAMETERS = {
    target: DEFAULT_OUTPUT_DIR / f"parameters_300m_{target}.csv" for target in ("uncheatable", "table9")
}
RPL_PARAMETERS = centered_benchmark.RPL_PARAMETERS
CONFIGURATION_SOURCES = ("rpl", "centered")
HEADS = ("rpl", "centered")


def shape_from_row(row: dict[str, object]) -> rpl.Shape:
    return rpl.Shape(
        benefit_exponent=float(row["benefit_exponent"]),
        benefit_offset=float(row["benefit_offset"]),
        damage_exponent=float(row["damage_exponent"]),
        damage_threshold=0.0,
        retention=float(row["retention"]),
        late_multiplier=float(row["late_multiplier"]),
        ordering_channel=bool(int(float(row["ordering_channel"]))),
    )


def parameter_rows(path: Path) -> dict[int, dict[str, object]]:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["seed"].eq(0)]
    rows = {int(row["fold"]): row for row in frame.to_dict("records")}
    if set(rows) != {0, 1, 2}:
        raise ValueError(f"expected folds 0, 1, and 2 in {path}")
    return rows


def fit_and_predict(
    head: str,
    shape: rpl.Shape,
    ridge: float,
    train_weights: np.ndarray,
    train_target: np.ndarray,
    test_weights: np.ndarray,
    geometry: rpl.Geometry,
) -> np.ndarray:
    if head == "rpl":
        train_design = rpl.design_matrix(train_weights, geometry, shape)
        test_design = rpl.design_matrix(test_weights, geometry, shape)
        intercept, coefficients = rpl.solve_head(
            train_design,
            train_target,
            ridge,
            rpl.penalty_multipliers(geometry, shape),
        )
        return intercept + test_design @ coefficients
    if head == "centered":
        train_design = centered.design_matrix(train_weights, geometry, shape)
        test_design = centered.design_matrix(test_weights, geometry, shape)
        intercept, coefficients = centered.solve_head(
            train_design,
            train_target,
            ridge,
            centered.penalty_operator(geometry, shape),
            geometry,
        )
        return intercept + test_design @ coefficients
    raise ValueError(f"unknown head: {head}")


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    one: np.ndarray,
    two: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float | int]:
    selected = np.zeros(len(observed), dtype=bool)
    selected[rows] = True
    tied = replay_control.tied_rows(weights)
    asymmetric = ~tied
    residual = predicted - observed
    in_pairs = selected[two]
    if not np.array_equal(in_pairs, selected[one]):
        raise ValueError("pair members must remain in the same outer fold")
    pair_one = one[in_pairs]
    pair_two = two[in_pairs]
    observed_delta = observed[pair_two] - observed[pair_one]
    predicted_delta = predicted[pair_two] - predicted[pair_one]
    pair_error = predicted_delta - observed_delta
    pair_spearman = float(spearmanr(observed_delta, predicted_delta).statistic)
    return {
        "n_rows": int(selected.sum()),
        "n_tied": int((selected & tied).sum()),
        "n_asymmetric": int((selected & asymmetric).sum()),
        "n_pairs": len(pair_one),
        "rmse": rmse(residual[selected]),
        "tied_rmse": rmse(residual[selected & tied]),
        "asymmetric_rmse": rmse(residual[selected & asymmetric]),
        "pair_rmse": rmse(pair_error),
        "pair_bias": float(np.mean(pair_error)),
        "pair_spearman": pair_spearman,
        "pair_sign_accuracy": float(np.mean(np.sign(predicted_delta) == np.sign(observed_delta))),
    }


def evaluate_target(
    target: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    dataset = benchmark.load_300m(target)
    geometry = benchmark.geometry_300m(dataset)
    folds = benchmark.grouped_folds(dataset.frame, seed=0, n_splits=3)
    configurations = {
        "rpl": parameter_rows(RPL_PARAMETERS[target]),
        "centered": parameter_rows(CANDIDATE_PARAMETERS[target]),
    }
    one, two = centered_benchmark.exact_pair_indices(dataset.frame, dataset.weights)
    predictions = {
        (configuration, head): np.full(dataset.n, np.nan) for configuration in CONFIGURATION_SOURCES for head in HEADS
    }
    records: list[dict[str, object]] = []

    for fold_id, (train, test) in enumerate(folds):
        for configuration_source, rows_by_fold in configurations.items():
            configuration = rows_by_fold[fold_id]
            shape = shape_from_row(configuration)
            ridge = float(configuration["ridge"])
            for head in HEADS:
                predicted = fit_and_predict(
                    head,
                    shape,
                    ridge,
                    dataset.weights[train],
                    dataset.y[train],
                    dataset.weights[test],
                    geometry,
                )
                predictions[(configuration_source, head)][test] = predicted
                records.append(
                    {
                        "target": target,
                        "scope": "fold",
                        "fold": fold_id,
                        "configuration_source": configuration_source,
                        "head": head,
                        "benefit_exponent": shape.benefit_exponent,
                        "benefit_offset": shape.benefit_offset,
                        "damage_exponent": shape.damage_exponent,
                        "retention": shape.retention,
                        "late_multiplier": shape.late_multiplier,
                        "ordering_channel": int(shape.ordering_channel),
                        "ridge": ridge,
                        **metrics(
                            dataset.y,
                            predictions[(configuration_source, head)],
                            dataset.weights,
                            one,
                            two,
                            test,
                        ),
                    }
                )

    for (configuration_source, head), predicted in predictions.items():
        if not np.all(np.isfinite(predicted)):
            raise ValueError(f"{target}/{configuration_source}/{head} did not predict every row")
        records.append(
            {
                "target": target,
                "scope": "pooled",
                "fold": -1,
                "configuration_source": configuration_source,
                "head": head,
                "benefit_exponent": np.nan,
                "benefit_offset": np.nan,
                "damage_exponent": np.nan,
                "retention": np.nan,
                "late_multiplier": np.nan,
                "ordering_channel": -1,
                "ridge": np.nan,
                **metrics(
                    dataset.y,
                    predicted,
                    dataset.weights,
                    one,
                    two,
                    np.arange(dataset.n),
                ),
            }
        )

    prediction_frame = pd.DataFrame(
        {
            "target": target,
            "row": np.arange(dataset.n),
            "run_name": dataset.frame["run_name"].astype(str),
            "policy_family": dataset.frame["policy_family"].astype(str),
            "phase_correspondence_key": dataset.frame["phase_correspondence_key"].astype(str),
            "observed": dataset.y,
            **{
                f"predicted_{configuration}_{head}": predicted
                for (configuration, head), predicted in predictions.items()
            },
        }
    )
    return records, prediction_frame


def decision(metrics_frame: pd.DataFrame) -> dict[str, object]:
    pooled = metrics_frame.loc[metrics_frame["scope"].eq("pooled")].set_index(["target", "configuration_source", "head"])
    folded = metrics_frame.loc[metrics_frame["scope"].eq("fold")]
    target_results: dict[str, object] = {}
    qualifying_targets = []
    all_pair_guardrails = True
    all_overall_guardrails = True
    for target in ("uncheatable", "table9"):
        configuration_results = {}
        target_qualifies = True
        for configuration in CONFIGURATION_SOURCES:
            rpl_row = pooled.loc[(target, configuration, "rpl")]
            centered_row = pooled.loc[(target, configuration, "centered")]
            pair_ratio = float(centered_row["pair_rmse"] / rpl_row["pair_rmse"])
            overall_ratio = float(centered_row["rmse"] / rpl_row["rmse"])
            fold_rows = folded.loc[folded["target"].eq(target) & folded["configuration_source"].eq(configuration)].pivot(
                index="fold", columns="head", values="pair_rmse"
            )
            improved_folds = int((fold_rows["centered"] < fold_rows["rpl"]).sum())
            configuration_results[configuration] = {
                "rpl_pair_rmse": float(rpl_row["pair_rmse"]),
                "centered_pair_rmse": float(centered_row["pair_rmse"]),
                "pair_ratio": pair_ratio,
                "rpl_rmse": float(rpl_row["rmse"]),
                "centered_rmse": float(centered_row["rmse"]),
                "overall_ratio": overall_ratio,
                "improved_folds": improved_folds,
            }
            target_qualifies &= pair_ratio <= 0.98 and improved_folds >= 2
            all_pair_guardrails &= pair_ratio <= 1.02
            all_overall_guardrails &= overall_ratio <= 1.02
        if target_qualifies:
            qualifying_targets.append(target)
        target_results[target] = configuration_results

    other_target_guardrails = False
    for qualifying_target in qualifying_targets:
        other_targets = {"uncheatable", "table9"} - {qualifying_target}
        other_target_guardrails |= all(
            all(
                float(target_results[other_target][configuration]["pair_ratio"]) <= 1.02  # type: ignore[index]
                and float(target_results[other_target][configuration]["overall_ratio"]) <= 1.02  # type: ignore[index]
                for configuration in CONFIGURATION_SOURCES
            )
            for other_target in other_targets
        )

    passed = bool(qualifying_targets and other_target_guardrails and all_pair_guardrails and all_overall_guardrails)
    return {
        "targets": target_results,
        "qualifying_targets": qualifying_targets,
        "all_pair_guardrails": all_pair_guardrails,
        "all_overall_guardrails": all_overall_guardrails,
        "other_target_guardrails": other_target_guardrails,
        "passed": passed,
        "next_action": (
            "run additional nested-CV seeds"
            if passed
            else "reject the centered-prior route; do not run additional seeds"
        ),
    }


def write_report(
    output_dir: Path,
    metrics_frame: pd.DataFrame,
    verdict: dict[str, object],
) -> None:
    pooled = metrics_frame.loc[metrics_frame["scope"].eq("pooled")]
    rows = []
    for target in ("uncheatable", "table9"):
        for configuration in CONFIGURATION_SOURCES:
            selected = pooled.loc[
                pooled["target"].eq(target) & pooled["configuration_source"].eq(configuration)
            ].set_index("head")
            rpl_row = selected.loc["rpl"]
            centered_row = selected.loc["centered"]
            rows.append(
                "| "
                + " | ".join(
                    [
                        target,
                        configuration,
                        f'{rpl_row["pair_rmse"]:.6f}',
                        f'{centered_row["pair_rmse"]:.6f}',
                        f'{centered_row["pair_rmse"] / rpl_row["pair_rmse"] - 1:+.2%}',
                        f'{rpl_row["rmse"]:.6f}',
                        f'{centered_row["rmse"]:.6f}',
                        f'{centered_row["rmse"] / rpl_row["rmse"] - 1:+.2%}',
                    ]
                )
                + " |"
            )

    report = """# Shape-pinned centered-hierarchy falsification

This 2 x 2 comparison uses only the high-TPP 300M paired panel. It crosses both
linear heads with both configurations selected in the prior nested screen, so
the head contrast at a fixed configuration isolates the centered prior.

| Target | Configuration | RPL pair RMSE | Centered pair RMSE | Pair change | RPL RMSE | Centered RMSE | Overall change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    report += "\n".join(rows)
    report += f"""

## Decision

**{"CONTINUE TO ADDITIONAL SEEDS" if verdict["passed"] else "REJECT CENTERED PRIOR"}**.

Next action: {verdict["next_action"]}.

The outcome-free WSD80 invariant remains exact, but it cannot identify this
prior because WSD80 has singleton families. Delphi 3e18 is deliberately not a
selection panel for this test.
"""
    (output_dir / "pinned_2x2_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    predictions = []
    for target in ("uncheatable", "table9"):
        target_records, target_predictions = evaluate_target(target)
        records.extend(target_records)
        predictions.append(target_predictions)
    metrics_frame = pd.DataFrame(records)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    verdict = decision(metrics_frame)

    metrics_frame.to_csv(args.output_dir / "pinned_2x2_metrics.csv", index=False)
    prediction_frame.to_csv(args.output_dir / "pinned_2x2_predictions.csv", index=False)
    (args.output_dir / "pinned_2x2_summary.json").write_text(json.dumps(verdict, indent=2) + "\n")
    write_report(args.output_dir, metrics_frame, verdict)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
