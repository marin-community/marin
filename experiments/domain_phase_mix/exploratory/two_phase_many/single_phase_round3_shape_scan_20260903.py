# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Bank selection value of every fixed (shape, ridge, link) of the successor, amplitudes fitted on the panel.

For each target and component the successor's design is built at every grid shape, the nonnegative head is fitted
on the canonical 280-run panel at every ridge and link, and the Delphi bank is predicted. Shapes are shared across
components in this scan (one row per shape, ridge, link); the row labelled ``inner_cv`` reproduces the frozen model
(per-component shapes chosen by inner CV). Selecting a row by its bank score is development evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)

PANEL = "delphi_3e18_39bucket"
LINKS = (models.LinkKind.IDENTITY, models.LinkKind.LOG_DEFICIT_BOUNDED)


def scan_component(model_id: str, target: str, component_index: int, bank_features) -> dict[str, np.ndarray]:
    panel = harness.load_panel(PANEL)
    entry = registry.ENTRY_BY_ID[model_id]
    group = panel.group(target)
    component = group.components[component_index]
    features = dataclasses.replace(registry.apply_transform(panel.features, entry), component=str(component))
    query = dataclasses.replace(registry.apply_transform(bank_features, entry), component=str(component))
    model = entry.build(features)
    response = group.outcomes[:, component_index]
    rows = np.arange(panel.rows)
    shapes = model.candidate_shapes(features)
    grid = np.full((len(shapes), len(model.ridge_grid), len(LINKS), query.rows), np.nan)
    cv_table = np.full((len(shapes), len(model.ridge_grid), len(LINKS)), np.inf)
    inner = harness.heldout_inner_folds(panel)
    for shape_index, shape in enumerate(shapes):
        design = model.design(features, shape)
        bank_design = model.design(query, shape)
        for link_index, link in enumerate(LINKS):
            spec = model.head_for(shape, link)
            for ridge_index, ridge in enumerate(model.ridge_grid):
                head = models.fit_head(design, response, ridge, spec)
                grid[shape_index, ridge_index, link_index] = models.predict_head(head, bank_design.values, spec)
                cv_table[shape_index, ridge_index, link_index] = models._cv_rmse(design, response, ridge, spec, inner)
    fitted = model.fit(
        features,
        response,
        rows,
        harness.heldout_inner_folds(panel),
        harness._seed(harness.FitTask(model_id, PANEL, target, component_index, component, 0, 0)),
    )
    cv_prediction = np.asarray(model.predict(fitted, query, np.arange(query.rows)), dtype=float)
    return {
        "grid": grid,
        "cv_table": cv_table,
        "inner_cv": cv_prediction,
        "cv_shape": fitted.shape,
        "cv_ridge": fitted.ridge,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-dir", type=Path, required=True, help="heldout registry directory (use the corrected view)"
    )
    parser.add_argument("--output-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR / "heldout_round3_corrected")
    parser.add_argument("--model", default="weibull_softplus_unscaled")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    panel = harness.load_panel(PANEL)
    entry = registry.ENTRY_BY_ID[args.model]
    model = entry.build(panel.features)
    shapes = model.candidate_shapes(panel.features)
    rows = []
    for target in ("uncheatable", "table9"):
        bank = selection.load_bank(panel, target)
        _frame, bank_features = harness.heldout_features(panel, target)
        group = panel.group(target)
        with harness.parallel_config(backend="loky", inner_max_num_threads=1):
            parts = Parallel(n_jobs=args.workers)(
                delayed(scan_component)(args.model, target, index, bank_features)
                for index in range(len(group.components))
            )
        weights = group.aggregation_weights
        grid = sum(weights[index] * part["grid"] for index, part in enumerate(parts))
        cv = sum(weights[index] * part["inner_cv"] for index, part in enumerate(parts))
        archive = np.array([selection.DOSE_SOURCE not in source for source in bank.sources])
        strata = (("pooled", np.ones(len(bank.measured), dtype=bool)), ("archive", archive))
        for stratum, mask in strata:
            row = {"target": target, "stratum": stratum, "shape": "inner_cv", "ridge": np.nan, "link": "per_component"}
            row.update(selection.selection_row(bank.measured[mask], cv[mask], bank.tolerance))
            rows.append(row)
            for shape_index, shape in enumerate(shapes):
                for ridge_index, ridge in enumerate(model.ridge_grid):
                    for link_index, link in enumerate(LINKS):
                        guess = grid[shape_index, ridge_index, link_index]
                        if not np.isfinite(guess).all():
                            continue
                        row = {
                            "target": target,
                            "stratum": stratum,
                            "shape": str(shape),
                            "ridge": ridge,
                            "link": str(link),
                        }
                        row.update(selection.selection_row(bank.measured[mask], guess[mask], bank.tolerance))
                        rows.append(row)
        chosen = pd.Series([str(part["cv_shape"]) for part in parts]).value_counts()
        print(f"{target}: inner-CV shapes per component: {chosen.head(5).to_dict()}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_dir / f"shape_scan_{args.model.replace('@', '_')}_{target}.npz",
            grid=grid,
            inner_cv=cv,
            measured=bank.measured,
            coordinate_id=bank.coordinate_id,
            sources=bank.sources,
            distance=bank.distance,
            shapes=np.array([str(shape) for shape in shapes]),
            ridges=np.array(model.ridge_grid),
            links=np.array([str(link) for link in LINKS]),
            cv_shapes=np.array([str(part["cv_shape"]) for part in parts]),
            cv_ridges=np.array([part["cv_ridge"] for part in parts]),
            component_grid=np.stack([part["grid"] for part in parts]).astype(np.float32),
            component_cv=np.stack([part["cv_table"] for part in parts]),
            aggregation_weights=weights,
        )
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / f"shape_scan_{args.model.replace('@', '_')}.csv", index=False)
    pd.set_option("display.width", 250)
    columns = [
        "target",
        "shape",
        "ridge",
        "link",
        "regret_at_1",
        "top5_regret",
        "frontier_predicted_rank",
        "selected_rank",
        "bias",
        "rmse",
        "spearman_best_quartile",
    ]
    for target in ("uncheatable", "table9"):
        subset = table[table["target"].eq(target) & table["stratum"].eq("archive")]
        print(
            f"\n=== {target} / archive stratum: inner-CV row, then the 12 best fixed rows by (regret@1, frontier rank)"
        )
        print(subset[subset["shape"].eq("inner_cv")][columns].round(4).to_string(index=False))
        best = (
            subset[subset["shape"].ne("inner_cv")]
            .sort_values(["regret_at_1", "frontier_predicted_rank", "top5_regret"])
            .head(12)
        )
        print(best[columns].round(4).to_string(index=False))
        share = (subset["regret_at_1"] <= subset[subset["shape"].eq("inner_cv")]["regret_at_1"].iloc[0]).mean()
        print(f"share of fixed rows at or below the inner-CV regret: {share:.3f} of {len(subset) - 1}")


if __name__ == "__main__":
    main()
