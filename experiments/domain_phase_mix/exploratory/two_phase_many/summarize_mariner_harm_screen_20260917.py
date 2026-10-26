# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///

"""Summarize independently audited full-objective results for the five finalists.

Run after the independent audit completes, with the repository's pinned Python
environment. Reads audit CSVs only and writes a paired table plus a Markdown
readout; does not fit models or inspect an external proposal bank.
"""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "reference_outputs/mariner_harm_screen_20260917/fixed_basis"
CONTROL = "hs17_mariner"
FINALISTS = (CONTROL, "hs17_bounded_exp_c1", "hs17_bounded_rat_c1", "hs17_coupled_c4", "hs17_log1")
LABELS = {
    CONTROL: "MARINER",
    "hs17_bounded_exp_c1": "Bounded exponential, C=1",
    "hs17_bounded_rat_c1": "Bounded rational, C=1",
    "hs17_coupled_c4": "Coupled inflection, c=4",
    "hs17_log1": "Logarithmic harm",
}
GROUP_KEYS = ["panel", "target", "repeat"]
METRICS = ("spearman", "regret_at_1", "selection_optimism", "rmse")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def paired_objectives(pooled: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Compare objective-level menus on identical scored rows and outer folds."""
    require(set(pooled["model"]) == set(FINALISTS), "Expected exactly the five registered finalists")
    require(not pooled.duplicated(["model", *GROUP_KEYS]).any(), "Duplicate pooled objective identities")
    require(not folds.duplicated(["model", *GROUP_KEYS, "fold"]).any(), "Duplicate fold objective identities")
    baseline = pooled[pooled["model"].eq(CONTROL)].set_index(GROUP_KEYS).sort_index()
    baseline_folds = folds[folds["model"].eq(CONTROL)].set_index([*GROUP_KEYS, "fold"]).sort_index()
    rows = []
    for model_id in FINALISTS:
        current = pooled[pooled["model"].eq(model_id)].set_index(GROUP_KEYS).sort_index()
        current_folds = folds[folds["model"].eq(model_id)].set_index([*GROUP_KEYS, "fold"]).sort_index()
        require(current.index.equals(baseline.index), f"Pooled objective identities differ: {model_id}")
        require(current_folds.index.equals(baseline_folds.index), f"Fold objective identities differ: {model_id}")
        require(np.array_equal(current["n"], baseline["n"]), f"Pooled menu sizes differ: {model_id}")
        require(np.array_equal(current_folds["n"], baseline_folds["n"]), f"Fold menu sizes differ: {model_id}")
        for index, row in current.iterrows():
            assert isinstance(index, tuple)
            identity = (str(index[0]), str(index[1]), int(index[2]))
            control = baseline.xs(identity)
            fold_rows = current_folds.xs(identity, level=GROUP_KEYS)
            fold_controls = baseline_folds.xs(identity, level=GROUP_KEYS)
            require(len(fold_rows) == 5, f"Expected five outer folds: {identity}")
            fold_regret_delta = fold_rows["regret_at_1"] - fold_controls["regret_at_1"]
            record: dict[str, str | int | float] = {
                "model": model_id,
                "label": LABELS[model_id],
                **dict(zip(GROUP_KEYS, identity, strict=True)),
            }
            record["n"] = int(row["n"])
            for metric in METRICS:
                record[metric] = float(row[metric])
                record[f"delta_{metric}"] = float(row[metric] - control[metric])
                record[f"mean_fold_{metric}"] = float(fold_rows[metric].mean())
                record[f"delta_mean_fold_{metric}"] = float(fold_rows[metric].mean() - fold_controls[metric].mean())
            record.update(
                {
                    "fold_regret_wins": int((fold_regret_delta < -1e-8).sum()),
                    "fold_regret_losses": int((fold_regret_delta > 1e-8).sum()),
                    "fold_regret_ties": int((fold_regret_delta.abs() <= 1e-8).sum()),
                    "max_fold_regret_regression": float(fold_regret_delta.max()),
                    "largest_fold_regret_improvement": float(fold_regret_delta.min()),
                }
            )
            rows.append(record)
    return pd.DataFrame(rows)


def derived_thresholds(fits: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Describe the coupled form's selected onset range without changing any fit."""
    selected = fits[fits["model"].eq("hs17_coupled_c4")].copy()
    require(len(selected) > 0 and selected["status"].eq("ok").all(), "Missing successful coupled fits")
    shapes = [json.loads(value) for value in selected["shape_json"]]
    selected["rate"] = [shape["rate"] for shape in shapes]
    selected["power"] = [shape["power"] for shape in shapes]
    values = np.asarray(
        [math.log1p(4 * math.log(2) ** (1 / shape["power"]) / shape["rate"]) - math.log(math.e - 1) for shape in shapes]
    )
    selected["derived_tau"] = values
    selected["outside_canonical_tau_range"] = (values < 1) | (values > 6)
    summary = {
        "fits": len(selected),
        "minimum": float(values.min()),
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "maximum": float(values.max()),
        "below_one": int((values < 1).sum()),
        "above_six": int((values > 6).sum()),
        "outside_fraction": float(selected["outside_canonical_tau_range"].mean()),
        "canonical_range": [1, 6],
        "formula": "log1p(4*log(2)**(1/power)/rate)-log(e-1)",
    }
    columns = ["model", *GROUP_KEYS, "component", "fold", "rate", "power", "derived_tau", "outside_canonical_tau_range"]
    return selected[columns], summary


def readout_lines(paired: pd.DataFrame, checks: dict, thresholds: dict) -> list[str]:
    lines = [
        "# Full-objective MARINER harm screen",
        "",
        f"The independent audit verified {checks['completed_fits']:,} component fits and "
        f"{checks['aggregate_prediction_rows']:,} objective predictions. Each finalist uses the same "
        "five outer folds on all three panels, with all seven Uncheatable and all 51 Table9 components.",
        "",
        "These are archived out-of-fold predictions. Pooled-menu regret selects the lowest predicted loss "
        "across all scored rows, whose predictions come from different outer-fold fits. Mean fold regret "
        "first selects within each fold and then averages its five regrets. Both compare against the "
        "best observed loss in the corresponding menu. No external proposal bank or newly optimized "
        "training mixture is evaluated here.",
        "",
        "Positive selection optimism means the chosen row's loss was underpredicted: observed minus "
        "predicted loss. Lower regret and higher Spearman are better; optimism near zero indicates "
        "calibration of the selected loss. Deltas compare "
        "each form with MARINER on the same objective. BPB differences should be compared within an "
        "objective because component scales and aggregation weights differ.",
    ]
    for identity, group in paired.groupby(GROUP_KEYS, sort=False):
        panel, target, repeat = identity
        lines.extend(
            [
                "",
                f"## {panel} / {target} (outer repeat {repeat})",
                "",
                "| Form | Spearman | Delta | Pooled regret | Delta | Mean fold regret | Delta | "
                "Fold regret W/L/T | Selection optimism |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in group.iterrows():
            lines.append(
                f"| {row['label']} | {row['spearman']:.4f} | {row['delta_spearman']:+.4f} | "
                f"{row['regret_at_1']:.5f} | {row['delta_regret_at_1']:+.5f} | "
                f"{row['mean_fold_regret_at_1']:.5f} | {row['delta_mean_fold_regret_at_1']:+.5f} | "
                f"{row['fold_regret_wins']}/{row['fold_regret_losses']}/{row['fold_regret_ties']} | "
                f"{row['selection_optimism']:+.5f} |"
            )
    lines.extend(
        [
            "",
            "The five forms were advanced using a six-component anchor screen on the same archived "
            "panels. This broader objective evaluation remains part of model development; it is not "
            "an independent prospective confirmation. One fixed outer partition supports paired "
            "diagnostics but does not establish variation across new partitions or training seeds.",
            "",
            f"The coupled c=4 form selects derived tau values from {thresholds['minimum']:.3f} to "
            f"{thresholds['maximum']:.3f}, with median {thresholds['median']:.3f}. "
            f"{thresholds['below_one'] + thresholds['above_six']}/{thresholds['fits']} "
            f"({100 * thresholds['outside_fraction']:.1f}%) lie outside MARINER's canonical tau range [1, 6]. "
            "Tau is derived without clipping, so this candidate changes the available onset range as "
            "well as tying the benefit and harm clocks. Its results do not isolate a causal effect of "
            "clock tying. Per-fit values are in `coupled_c4_derived_thresholds.csv`.",
            "",
            "Exact values, RMSE, fold optimism, and the largest fold regret regressions are in "
            "`full_objective_paired.csv`. Source identities are in `full_objective_readout_manifest.json`; "
            "the preceding verification is in `audit.json`.",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    directory = args.output / "independent_audit_certify"
    audit_path = directory / "audit.json"
    checks = json.loads(audit_path.read_text())
    require(checks["complete_exact_coverage"], "Full-objective audit has incomplete fits")
    require(set(checks["models"]) == set(FINALISTS), "Audit must cover all five registered finalists")
    require(checks["aggregate_prediction_rows"] > 0, "Audit contains no full-objective predictions")
    require(
        all(
            checks["report_comparisons"].get(name)
            for name in ("aggregate_predictions", "aggregate_metrics", "aggregate_fold_metrics")
        ),
        "Full-objective report has not been verified against independent reconstruction",
    )
    inputs = (
        audit_path,
        directory / "aggregate_metrics.csv",
        directory / "aggregate_fold_metrics.csv",
        args.output / "report_certify" / "fits.csv",
    )
    pooled, folds, fits = (pd.read_csv(path) for path in inputs[1:])
    paired = paired_objectives(pooled, folds)
    thresholds, threshold_summary = derived_thresholds(fits)
    paired.to_csv(directory / "full_objective_paired.csv", index=False)
    thresholds.to_csv(directory / "coupled_c4_derived_thresholds.csv", index=False)
    (directory / "coupled_c4_threshold_summary.json").write_text(json.dumps(threshold_summary, indent=2) + "\n")
    (directory / "full_objective_readout.md").write_text(
        "\n".join(readout_lines(paired, checks, threshold_summary)) + "\n"
    )
    manifest = {
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "sources": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), *inputs)
        },
        "comparison": "paired archived OOF full objectives; no external bank or training runs",
        "models": FINALISTS,
    }
    (directory / "full_objective_readout_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"report": str(directory / "full_objective_readout.md"), "objective_rows": len(paired)}))


if __name__ == "__main__":
    main()
