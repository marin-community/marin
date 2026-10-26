# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "tabulate"]
# ///

"""Read-only reconstruction and decision-oriented summary of MARINER refits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    audit_mariner_harm_screen_20260917 as audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_harm_forms_20260917 as screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_refit_robustness_20260917 as validation,
)


def audit_artifacts(output: Path) -> dict:
    manifest = validation.prepare(output)
    validation.register_models()
    tasks = validation.planned_tasks("all")
    replayed = set()
    records = []
    max_error = 0.0
    split_records = []
    for panel_name in bench.THIRTY_NINE_BUCKET_PANELS:
        panel = bench.load_panel(panel_name)
        calibration = set(bench.calibration_rows(panel))
        for split in bench.panel_splits(panel, validation.REPEATS):
            train, test = set(split.train), set(split.test)
            if train & test or not calibration <= train or calibration & test:
                raise ValueError(f"Outer split/calibration leakage: {panel_name}/{split.repeat}/{split.fold}")
            for inner_train, inner_test in split.inner:
                a, b = set(inner_train), set(inner_test)
                if a & b or not a | b <= train or not calibration <= a or calibration & b:
                    raise ValueError(f"Inner split/calibration leakage: {panel_name}/{split.repeat}/{split.fold}")
            split_records.append(
                {
                    "panel": panel_name,
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "split_fingerprint": bench.split_fingerprint(panel_name, split.repeat, split.fold),
                }
            )
    for task in tasks:
        path = bench.shard_path(output, task)
        saved = bench.load_shard(path)
        if saved is None or str(saved["status"]) != "ok":
            raise ValueError(f"Missing successful fit: {task}")
        panel = bench.load_panel(task.panel)
        split = next(
            s for s in bench.panel_splits(panel, task.repeat + 1) if (s.repeat, s.fold) == (task.repeat, task.fold)
        )
        head_path = validation.task_head_path(output, task)
        head_payload = bench.load_shard(head_path)
        if head_payload is None or str(head_payload["prediction_shard_sha256"]) != screen.digest(path):
            raise ValueError(f"Final head provenance differs: {head_path}")
        fitted = validation.load_fitted(head_path)
        features, model = audit.component_model(task, panel)
        prediction = model.predict(fitted, features, split.test)
        error = float(np.max(np.abs(prediction - saved["prediction"])))
        max_error = max(error, max_error)
        np.testing.assert_allclose(prediction, saved["prediction"], rtol=1e-9, atol=1e-10)
        np.testing.assert_array_equal(
            saved["observed"], panel.group(task.target).outcomes[split.test, task.component_index]
        )
        audit.check_selected_grid(task, panel, saved)
        replay_key = task.model_id, task.panel, task.target, task.repeat
        if replay_key not in replayed:
            records.append(audit.replay_selected(task, panel, split, saved))
            replayed.add(replay_key)
    result = {
        "fit_records": len(tasks),
        "verified_final_head_records": len(tasks),
        "max_saved_head_prediction_error": max_error,
        "sampled_full_inner_cv_replays": len(records),
        "protocol_sha256": screen.digest(output / "protocol.json"),
        "audit_source_sha256": screen.digest(Path(__file__)),
        "split_hash": manifest["split_hash"],
        "outer_and_inner_split_membership": split_records,
        "replays": records,
    }
    (output / "independent_reconstruction.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def regularization_summary(output: Path, components: pd.DataFrame) -> pd.DataFrame:
    anchors = bench._anchor_components()
    rows = components[
        components["repeat"].eq(0) & components.apply(lambda row: row.component in anchors[row.target], axis=1)
    ].copy()
    labels = rows.model.str.split("__", n=1, expand=True)
    rows["family"] = labels[0]
    rows["setting"] = labels[1].fillna("canonical")
    keys = ["setting", "panel", "target", "component", "repeat"]
    baseline = rows[rows.family.eq("hs17_mariner")].set_index(keys)
    paired = rows.join(baseline[["spearman", "rmse"]].add_prefix("matched_mariner_"), on=keys)
    paired["delta_spearman"] = paired.spearman - paired.matched_mariner_spearman
    paired["rmse_ratio"] = paired.rmse / paired.matched_mariner_rmse
    paired.to_csv(output / "regularization_component_contrasts.csv", index=False)
    grouped = paired.groupby(["family", "setting", "panel", "target"])[["delta_spearman", "rmse_ratio"]].mean()
    grouped.to_csv(output / "regularization_panel_objective_contrasts.csv")
    summary = grouped.groupby(["family", "setting"]).mean()
    summary.to_csv(output / "regularization_summary.csv")
    return summary


def component_removal(output: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    main = predictions[predictions.model.isin(validation.MODEL_IDS)]
    records = []
    keys = ["panel", "target", "repeat"]
    for (panel_name, target, repeat), group in main.groupby(keys):
        panel = bench.load_panel(panel_name)
        objective = panel.group(target)
        observed = group[group.model.eq("hs17_mariner")].pivot(index="row_index", columns="component", values="observed")
        observed = observed.loc[:, list(objective.components)]
        weights = objective.aggregation_weights
        values = {
            model: (
                rows.pivot(index="row_index", columns="component", values="prediction")
                .loc[observed.index, list(objective.components)]
                .to_numpy()
            )
            for model, rows in group.groupby("model")
        }
        y = objective.aggregate[observed.index.to_numpy(int)]
        aggregates = {model: value @ weights for model, value in values.items()}
        for model in validation.MODEL_IDS[1:]:
            intact = float(
                stats.spearmanr(y, aggregates[model]).statistic
                - stats.spearmanr(y, aggregates["hs17_mariner"]).statistic
            )
            for index, component in enumerate(objective.components):
                w = weights[index]
                retained_y = (y - w * observed.to_numpy()[:, index]) / (1.0 - w)
                candidate = (aggregates[model] - w * values[model][:, index]) / (1.0 - w)
                baseline = (aggregates["hs17_mariner"] - w * values["hs17_mariner"][:, index]) / (1.0 - w)
                records.append(
                    {
                        "panel": panel_name,
                        "target": target,
                        "repeat": repeat,
                        "model": model,
                        "removed_component": component,
                        "component_weight": w,
                        "intact_delta_spearman": intact,
                        "remaining_delta_spearman": float(
                            stats.spearmanr(retained_y, candidate).statistic
                            - stats.spearmanr(retained_y, baseline).statistic
                        ),
                    }
                )
    result = pd.DataFrame(records)
    result.to_csv(output / "leave_one_component_out.csv", index=False)
    return result


def fold_selection_comparisons(output: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    keys = ["model", "panel", "target", "repeat", "fold"]
    for identity, group in predictions.groupby(keys):
        selected = group.loc[group.prediction.idxmin()]
        records.append(
            dict(
                zip(keys, identity, strict=True),
                selected_row=int(selected.row_index),
                selected_observed=float(selected.observed),
                regret=float(selected.observed - group.observed.min()),
            )
        )
    rows = pd.DataFrame(records)
    baseline = rows[rows.model.eq("hs17_mariner")].set_index(keys[1:])
    paired = rows.join(baseline[["regret", "selected_row"]].add_prefix("baseline_"), on=keys[1:])
    paired["delta_regret"] = paired.regret - paired.baseline_regret
    paired["better"] = paired.delta_regret < -1e-12
    paired["worse"] = paired.delta_regret > 1e-12
    paired["tied"] = paired.delta_regret.abs() <= 1e-12
    paired.to_csv(output / "fold_selection_comparisons.csv", index=False)
    counts = paired.groupby("model")[["better", "worse", "tied"]].sum()
    counts.to_csv(output / "fold_selection_counts.csv")
    return counts


def summarize(output: Path) -> None:
    destination = output / "report_all"
    if not destination.is_dir():
        raise FileNotFoundError(destination)
    checks = audit_artifacts(output)
    components = pd.read_csv(destination / "component_metrics.csv")
    sensitivity = regularization_summary(output, components)
    predictions = pd.read_csv(destination / "component_predictions.csv")
    removals = component_removal(output, predictions)
    paired = pd.read_csv(destination / "paired_partition_metrics.csv")
    selection_counts = fold_selection_comparisons(output, pd.read_csv(destination / "aggregate_predictions.csv"))
    statistics = paired.groupby(["model", "panel", "target"])[
        ["spearman", "mean_fold_regret", "delta_spearman", "delta_mean_fold_regret"]
    ].agg(["mean", "min", "max"])
    statistics.to_csv(output / "final_partition_summary.csv")
    display = pd.DataFrame(index=statistics.index)
    display["Mean Spearman"] = statistics[("spearman", "mean")].round(4)
    display["Delta Spearman range"] = [
        f"{low:+.4f} to {high:+.4f}"
        for low, high in zip(statistics[("delta_spearman", "min")], statistics[("delta_spearman", "max")], strict=True)
    ]
    display["Mean fold regret BPB"] = statistics[("mean_fold_regret", "mean")].round(6)
    display["Delta regret range BPB"] = [
        f"{low:+.6f} to {high:+.6f}"
        for low, high in zip(
            statistics[("delta_mean_fold_regret", "min")],
            statistics[("delta_mean_fold_regret", "max")],
            strict=True,
        )
    ]
    component_summary = removals.groupby(["model", "panel", "target", "repeat"]).agg(
        intact_delta_spearman=("intact_delta_spearman", "first"),
        smallest_removal_delta=("remaining_delta_spearman", "min"),
        largest_removal_delta=("remaining_delta_spearman", "max"),
    )
    component_summary.to_csv(output / "component_removal_summary.csv")
    text = [
        "# Local refit robustness",
        "",
        "The forms and all fitting sensitivities were frozen before the new partition scores were observed. "
        "All panels are archived development data already used to shortlist the two challengers.",
        "",
        "## Three paired grouped partitions",
        "",
        "The same canonical calibration rows remain in training. Other mixtures form geometric blocks. "
        "Each outer model chooses shape/ridge/floor using only that outer training set and its inner folds. "
        "These are three overlapping partitions, not three independent datasets; "
        "ranges are descriptive split sensitivity.",
        "",
        display.to_markdown(),
        "",
        "Across the 90 objective/fold menus (six objectives times three partitions times five folds), "
        "the following counts compare measured selected loss against MARINER. These menus overlap across "
        "partitions and are not independent trials.",
        "",
        selection_counts.to_markdown(),
        "",
        "## Predeclared regularization sensitivity",
        "",
        "Six anchors per panel, repeat zero only. RMS column scaling and an expanded ridge grid "
        "were tested separately against canonical unscaled fitting. Each challenger is compared to MARINER "
        "under the same setting; objective groups and panels receive equal weight in this summary. "
        "Anchor scores do not establish full-objective selection quality.",
        "",
        sensitivity.round(6).to_markdown(),
        "",
        "## Component dependence",
        "",
        "leave_one_component_out.csv removes each component from both measured and predicted aggregates "
        "without refitting. This diagnoses concentration of ranking gains; it does not select a new objective.",
        "",
        "## Reconstruction",
        "",
        f"All {checks['fit_records']} selected grids and portable heads passed reconstruction. "
        f"{checks['sampled_full_inner_cv_replays']} sampled selected-shape/ridge/floor inner-CV replays passed. "
        "Benchmark aggregation was independently rebuilt from explicit component weights. Stored aggregate "
        "observations are used consistently; tiny archival rounding differences from sums of components "
        "are recorded separately in report_all/verification.json.",
    ]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n")
    print(
        json.dumps(
            {
                "report": str(output / "RESULTS.md"),
                "fits": checks["fit_records"],
                "replays": checks["sampled_full_inner_cv_replays"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=validation.DEFAULT_OUTPUT)
    args = parser.parse_args()
    summarize(args.output)


if __name__ == "__main__":
    main()
