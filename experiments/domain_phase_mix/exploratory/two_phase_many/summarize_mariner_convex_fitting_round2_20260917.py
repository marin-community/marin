# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "tabulate"]
# ///

"""Independent saved-grid/NNLS audit and matched fitting-procedure readout."""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_convex_fitting_round2_20260917 as screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_refit_robustness_20260917 as portable,
)


def model_label(model_id: str) -> str:
    family, *procedure = model_id.split("__")
    family = family.replace("cf2_", "").replace("cv17_", "")
    label = {"mariner": "MARINER", "huber_d8": "Convex Huber (width 8)", "smooth_d1": "Convex smooth hinge (width 1)"}[
        family
    ]
    setting = {"bounded_rms": "bounded RMS", "selection": "selection score", "joint": "joint reselection"}
    return label + (": " + setting[procedure[0]] if procedure else ": canonical fitting")


def audit(output: Path) -> dict:
    screen.prepare(output)
    screen.register_candidates()
    tasks = {task for phase in ("anchors", "full") for task in screen.planned_tasks(phase, output)}
    records = []
    for task in sorted(
        tasks, key=lambda item: (item.model_id, item.panel, item.target, item.component_index, item.fold)
    ):
        saved = bench.load_shard(bench.shard_path(output, task))
        head_saved = bench.load_shard(portable.task_head_path(output, task))
        if saved is None or head_saved is None or str(saved["status"]) != "ok":
            raise ValueError(f"Incomplete fit: {task}")
        if str(head_saved["prediction_shard_sha256"]) != screen.previous.previous.digest(bench.shard_path(output, task)):
            raise ValueError(f"Final head provenance mismatch: {task}")
        panel = bench.load_panel(task.panel)
        features = dataclasses.replace(panel.features, component=task.component)
        model = registry.ENTRY_BY_ID[task.model_id].build(features)
        fitted = portable.load_fitted(portable.task_head_path(output, task))
        grid = model.base
        shape_index, ridge_index = np.unravel_index(np.argmin(saved["cv_table"]), saved["cv_table"].shape)
        if grid.candidate_shapes(features)[shape_index] != fitted.shape or grid.ridge_grid[ridge_index] != fitted.ridge:
            raise ValueError(f"Selected grid differs from recorded argmin: {task}")
        selection_key = (
            "selection_inner_cv_objective"
            if "selection_inner_cv_objective" in fitted.diagnostics
            else "selection_inner_cv_rmse"
        )
        np.testing.assert_allclose(
            saved["cv_table"][shape_index, ridge_index], fitted.diagnostics[selection_key], atol=1e-12
        )
        rows = np.asarray(head_saved["train"], dtype=int)
        test = np.asarray(head_saved["test"], dtype=int)
        np.testing.assert_allclose(model.predict(fitted, features, test), saved["prediction"], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            model.predict(fitted, features, rows), saved["train_prediction"], rtol=1e-9, atol=1e-10
        )
        design = grid.design(features, fitted.shape).values[rows]
        response = panel.group(task.target).outcomes[rows, task.component_index]
        target = np.log(np.maximum(response - fitted.head.floor, 1e-9))
        residual = design @ fitted.head.coefficients + fitted.head.intercept - target
        gradient = (
            design.T @ residual + fitted.ridge * head_saved["training_ridge_multipliers"] * fitted.head.coefficients
        )
        active = fitted.head.coefficients > 1e-8
        violation = max(
            abs(float(np.sum(residual))),
            float(np.max(np.abs(gradient[active]), initial=0.0)),
            float(np.max(np.maximum(-gradient[~active], 0.0), initial=0.0)),
        )
        scale = max(1.0, float(np.linalg.norm(design, ord=2) * np.linalg.norm(target)))
        if violation > 1e-7 * scale or np.any(fitted.head.coefficients < -1e-12):
            raise ValueError(f"NNLS stationarity/complementarity failure {violation / scale}: {task}")
        records.append(dataclasses.asdict(task) | {"relative_kkt_violation": violation / scale})
    frame = pd.DataFrame(records)
    frame.to_csv(output / "independent_head_checks.csv", index=False)
    result = {
        "unique_fit_records": len(tasks),
        "selected_grids_verified": len(tasks),
        "final_heads_verified": len(tasks),
        "max_relative_nnls_kkt_violation": float(frame.relative_kkt_violation.max()),
        "source_inputs_protocol_checked": True,
        "analysis_source_sha256": screen.previous.previous.digest(Path(__file__)),
    }
    (output / "independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def summarize(output: Path) -> None:
    checks = audit(output)
    anchor = pd.read_csv(output / "anchor_balanced_summary.csv")
    advancement = json.loads((output / "advancement.json").read_text())
    destination = output / "report_full"
    pooled = pd.read_csv(destination / "aggregate_metrics.csv")
    folds = pd.read_csv(destination / "aggregate_fold_metrics.csv")
    keys = ["model", "panel", "target"]
    summary = pooled.set_index(keys)[["spearman", "rmse"]].join(
        folds.groupby(keys).regret_at_1.mean().rename("mean_fold_regret")
    )
    rows = []
    for model_id in advancement["selected_convex"]:
        procedure = model_id.split("__")[1]
        matched = f"cf2_mariner__{procedure}"
        comparison = summary.loc[model_id].join(summary.loc[matched].add_prefix("mariner_"))
        comparison["delta_spearman"] = comparison.spearman - comparison.mariner_spearman
        comparison["delta_mean_fold_regret"] = comparison.mean_fold_regret - comparison.mariner_mean_fold_regret
        comparison["model"] = model_id
        rows.append(comparison.reset_index())
    paired = pd.concat(rows, ignore_index=True)
    paired.to_csv(output / "full_matched_metrics.csv", index=False)
    summary.to_csv(output / "full_model_metrics.csv")
    head_audit = pd.read_csv(output / "report_anchors/head_audit.csv")
    bounds = head_audit.groupby("model_id")[["fraction_rms_below_floor", "fraction_rms_above_ceiling"]].mean()
    bounds.to_csv(output / "anchor_penalty_bound_summary.csv")
    lead = []
    for model_id, comparisons in paired.groupby("model"):
        difference = comparisons.delta_mean_fold_regret
        wins, losses = int((difference < -1e-12).sum()), int((difference > 1e-12).sum())
        lead.append(
            f"{model_label(model_id)} lowers mean held-out-menu regret on {wins} of six objectives, "
            f"raises it on {losses}, and ties on {6 - wins - losses}, compared with MARINER under the same fitting rule."
        )
    display_anchor = anchor.assign(model=anchor.model.map(model_label))
    display_paired = paired.assign(model=paired.model.map(model_label))
    text = [
        "# Convex fitting procedure screen",
        "",
        " ".join(lead),
        "",
        "MARINER uses squared log-epoch harm. The two convex alternatives retain their fixed smooth-hinge "
        "and Huber-hinge harm functions in raw epochs; this screen changes how their coefficients, shape, "
        "ridge, and floor are chosen.",
        "",
        "Three fitting changes were frozen before anchor scoring, each applied to the same two convex equations "
        "and MARINER. Source, inputs, and grouped folds were fixed. The anchor screen selected at most two "
        "convex variants by mean component decision regret and then correlation. The full objective comparisons "
        "use matching MARINER fitting controls.",
        "",
        "All observations are archived development data. Anchors and full-objective scores share data; these "
        "are exploratory checks, not untouched confirmation. Component-level inner regret need not optimize "
        "the weighted aggregate objective's decision. No measured bank outcomes entered fitting or advancement.",
        "",
        "## Anchor screen",
        "",
        "The bounded RMS rule multiplies each coefficient penalty by its training column's mean square, "
        "clipped to [.25, 4]. The selection score adds half the measured selected-point regret to pooled "
        "BPB RMSE in every inner shape/ridge/floor search. Joint reselection makes one additional shape/ridge "
        "search at the fitted floor. All choices use training folds only. Each panel contributes three "
        "components from each objective to the anchor screen; six panel/objective groups receive equal weight.",
        "",
        display_anchor.round(6).to_markdown(index=False),
        "",
        "## Advanced variants: matched full objectives",
        "",
        display_paired.round(6).to_markdown(index=False),
        "",
        "Regret is the average of five held-out outer-fold menus. The geometric folds are not independent "
        "experiments; no iid confidence interval is claimed. Rank improvement is reported separately from selection.",
        "",
        "## Bounded penalty behavior",
        "",
        bounds.round(4).to_markdown(),
        "",
        "These are the fractions of training column RMS-squared values beyond .25 and 4 before clipping. "
        "A lower-bound-dominated setting mostly lowers canonical ridge penalties by fourfold; this does not "
        "demonstrate that feature-wise normalization is the cause of any change.",
        "",
        "## Verification",
        "",
        json.dumps(checks, indent=2),
        "",
        "Each successful record additionally has an independent returned-inner-score replay and saved numeric head. "
        "Explicit component weights independently reconstruct the full objective predictions. Failures are retained "
        "in report_anchors/failures.csv and report_full/failures.csv.",
    ]
    (output / "RESULTS.md").write_text("\n".join(text) + "\n")
    print(json.dumps({"report": str(output / "RESULTS.md"), **checks}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=screen.DEFAULT_OUTPUT)
    args = parser.parse_args()
    summarize(args.output)


if __name__ == "__main__":
    main()
