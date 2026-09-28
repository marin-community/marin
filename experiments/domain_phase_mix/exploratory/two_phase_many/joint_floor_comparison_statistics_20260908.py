# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Paired evaluation uncertainty and floor diagnostics for the joint-floor ablation.

Resampling conditions on the fitted models and observed evaluation populations.
It does not estimate variability from fitting data or training seeds. Source-block
resampling preserves all coordinates connected through shared source membership;
fresh-run blocks are their six historical launch families. Neither bank is a new
prospective test. Outputs supplement, rather than tune, the prespecified models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

DEFAULT_OUTPUT = Path(__file__).parent / "reference_outputs/joint_floor_comparison_20260908"
VARIANTS = ("frozen", "response_fixed", "response_joint")
PAIRS = (("response_fixed", "frozen"), ("response_joint", "frozen"), ("response_joint", "response_fixed"))
INTERVENTIONS = {"conditional_epoch_dose_response", "archive::delphi_baseline_mixtures_issue6607_20260623"}
SEED = 20260908
DRAWS = 3000
BOUND_TOLERANCE = 1e-5


def source_memberships(sources: pd.Series) -> list[set[str]]:
    return [set(json.loads(x) if str(x).startswith("[") else str(x).split(";")) for x in sources]


def source_blocks(sources: pd.Series) -> np.ndarray:
    memberships = source_memberships(sources)
    adjacency = np.array([[bool(left & right) for right in memberships] for left in memberships])
    return connected_components(adjacency, directed=False)[1]


def paired_bootstrap(
    objective: str,
    population: str,
    table: pd.DataFrame,
    clusters: np.ndarray,
    scheme: str,
    rng: np.random.Generator,
) -> list[dict]:
    """Bootstrap pooled coordinate RMSE, drawing whole clusters with replacement."""
    measured = table.measured.to_numpy(float)
    predicted = table[list(VARIANTS)].to_numpy(float)
    squared = (predicted - measured[:, None]) ** 2
    labels, inverse = np.unique(clusters, return_inverse=True)
    group_sums = np.zeros((len(labels), len(VARIANTS)))
    np.add.at(group_sums, inverse, squared)
    group_sizes = np.bincount(inverse)
    draws = rng.integers(0, len(labels), size=(DRAWS, len(labels)))
    boot_rmse = np.sqrt(group_sums[draws].sum(axis=1) / group_sizes[draws].sum(axis=1)[:, None])
    observed_rmse = np.sqrt(squared.mean(axis=0))
    records = []
    for candidate, reference in PAIRS:
        ci, ri = VARIANTS.index(candidate), VARIANTS.index(reference)
        difference = boot_rmse[:, ci] - boot_rmse[:, ri]
        records.append(
            {
                "objective": objective,
                "population": population,
                "scheme": scheme,
                "candidate": candidate,
                "reference": reference,
                "n": len(table),
                "clusters": len(labels),
                "candidate_rmse": observed_rmse[ci],
                "reference_rmse": observed_rmse[ri],
                "rmse_difference": observed_rmse[ci] - observed_rmse[ri],
                "ci_low": np.quantile(difference, 0.025),
                "ci_high": np.quantile(difference, 0.975),
                "bootstrap_fraction_below_zero": np.mean(difference < 0),
                "draws": DRAWS,
                "uncertainty_scope": "evaluation population conditional on fixed fitted models; not training variance",
            }
        )
    return records


def bootstrap_tables(output: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records = []
    for objective in ("uncheatable", "table9"):
        bank = pd.read_csv(output / f"bank_predictions_{objective}.csv")
        memberships = source_memberships(bank.sources)
        clusters = source_blocks(bank.sources)
        table = bank.rename(columns={"measured_mean_bpb": "measured", **{f"prediction_{v}": v for v in VARIANTS}})
        optima = np.array([not bool(sources & INTERVENTIONS) for sources in memberships])
        for population, mask in (("bank_all", np.ones(len(table), bool)), ("bank_optima", optima)):
            selected = table.loc[mask]
            records.extend(paired_bootstrap(objective, population, selected, clusters[mask], "source_block", rng))
            records.extend(
                paired_bootstrap(objective, population, selected, np.arange(len(selected)), "coordinate", rng)
            )
    fresh = pd.read_csv(output / "fresh_predictions.csv")
    for objective, rows in fresh.groupby("objective", sort=True):
        keys = ["launch", "candidate_id", "target", "measured"]
        table = rows.pivot(index=keys, columns="variant", values="predicted").reset_index()
        for population, selected in (("fresh_all", table), ("fresh_targeted", table[table.target.eq(objective)])):
            records.extend(paired_bootstrap(objective, population, selected, selected.launch.to_numpy(), "launch", rng))
            records.extend(paired_bootstrap(objective, population, selected, np.arange(len(selected)), "run", rng))
    return pd.DataFrame(records)


def floor_tables(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits = pd.read_csv(output / "component_fits.csv")
    anchors = pd.read_csv(output / "inputs/anchors.csv").set_index(["objective", "component"])
    swarm = pd.read_csv(output / "inputs/swarm_outcomes.csv")
    fresh = pd.read_csv(output / "inputs/fresh_components.csv")
    targets = pd.read_csv(output / "inputs/fresh_runs.csv")[["launch", "candidate_id", "target"]]
    fresh = fresh.merge(targets, on=["launch", "candidate_id"], validate="one_to_one")
    records = []
    for fit in fits.itertuples():
        anchor = anchors.loc[(fit.objective, fit.component)]
        y = swarm[fit.component].to_numpy(float)
        proportional = float(anchor.proportional_bpb)
        gap = proportional - y.min()
        if gap <= 0:
            gap = max(float(y.std()), 1e-3 * abs(proportional), 1e-6)
        noise_depth = 3 * float(anchor.repeat_sd)
        below = fresh[fit.component].to_numpy(float) < fit.floor
        targeted = fresh.target.eq(fit.objective).to_numpy()
        records.append(
            {
                "objective": fit.objective,
                "component": fit.component,
                "variant": fit.variant,
                "gamma": fit.gamma,
                "floor": fit.floor,
                "at_lower_bound": abs(fit.gamma - 1) <= BOUND_TOLERANCE,
                "at_upper_bound": abs(fit.gamma - 6) <= BOUND_TOLERANCE,
                "noise_margin_active": noise_depth >= fit.gamma * gap - 1e-12,
                "noise_margin_strictly_active": noise_depth > fit.gamma * gap + 1e-12,
                "noise_margin_gamma_threshold": noise_depth / gap,
                "fresh_floor_violations": int(below.sum()),
                "targeted_floor_violations": int((below & targeted).sum()),
                "fresh_floor_max_excess": max(0.0, fit.floor - float(fresh[fit.component].min())),
                "fresh_floor_violating_component": bool(below.any()),
                "targeted_floor_violating_component": bool((below & targeted).any()),
            }
        )
    table = pd.DataFrame(records)
    summary = (
        table.groupby(["objective", "variant"])
        .agg(
            components=("component", "size"),
            lower_boundary_components=("at_lower_bound", "sum"),
            upper_boundary_components=("at_upper_bound", "sum"),
            noise_margin_active_components=("noise_margin_active", "sum"),
            gamma_locally_unidentified_by_floor_components=("noise_margin_strictly_active", "sum"),
            fresh_floor_violating_components=("fresh_floor_violating_component", "sum"),
            fresh_floor_violating_outcomes=("fresh_floor_violations", "sum"),
            targeted_floor_violating_components=("targeted_floor_violating_component", "sum"),
            targeted_floor_violating_outcomes=("targeted_floor_violations", "sum"),
            worst_floor_excess=("fresh_floor_max_excess", "max"),
        )
        .reset_index()
    )
    return table, summary


def task_group(component: str) -> str:
    name = component.split("/")[2] if "/" in component else component
    if name.startswith("minerva") or name == "basic_skills_arithmetic":
        return "math"
    if name.startswith("mt_mbpp") or name in {"codex_humaneval", "mbpp", "basic_skills_coding"}:
        return "code"
    if name.startswith("mmlu"):
        return "mmlu"
    return "qa_other"


def oof_tables(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    objectives = pd.read_csv(output / "inputs/objectives.csv")
    swarm = pd.read_csv(output / "inputs/swarm_outcomes.csv")
    component_rows, group_rows = [], []
    for objective, spec in objectives.groupby("objective", sort=False):
        components = spec.component.tolist()
        truth = swarm[components].to_numpy(float)
        predictions = {v: np.full_like(truth, np.nan) for v in VARIANTS}
        folds = np.full(len(swarm), -1)
        for index, component in enumerate(components):
            for fold in range(5):
                path = output / "tasks" / objective / f"r0_f{fold}_c{index}.json"
                if not path.exists():
                    raise FileNotFoundError(f"Complete five-fold diagnostics require {path}")
                task = json.loads(path.read_text())
                test = np.array(task["test"], int)
                folds[test] = fold
                assert task["component"] == component
                for variant in VARIANTS:
                    predictions[variant][test, index] = task["variants"][variant]["test_prediction"]
        valid = np.logical_and.reduce([np.isfinite(predictions[v]).all(axis=1) for v in VARIANTS])
        errors = {v: predictions[v][valid] - truth[valid] for v in VARIANTS}
        for index, component in enumerate(components):
            row = {"objective": objective, "component": component, "group": task_group(component), "n": int(valid.sum())}
            for variant in VARIANTS:
                row[f"rmse_{variant}"] = float(np.sqrt(np.mean(errors[variant][:, index] ** 2)))
                row[f"bias_{variant}"] = float(np.mean(errors[variant][:, index]))
            component_rows.append(row)
        weights = spec.weight.to_numpy(float)
        groups = np.array([task_group(c) for c in components])
        for variant in VARIANTS:
            weighted_errors = errors[variant] * weights[None, :]
            total_error = weighted_errors.sum(axis=1)
            for group in np.unique(groups):
                contribution = weighted_errors[:, groups == group].sum(axis=1)
                group_rows.append(
                    {
                        "objective": objective,
                        "variant": variant,
                        "group": group,
                        "components": int((groups == group).sum()),
                        "aggregate_mse": float(np.mean(total_error**2)),
                        "signed_mse_contribution": float(np.mean(contribution * total_error)),
                        "group_weighted_error_rmse": float(np.sqrt(np.mean(contribution**2))),
                        "group_weighted_bias": float(contribution.mean()),
                    }
                )
    return pd.DataFrame(component_rows), pd.DataFrame(group_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-oof", action="store_true", help="Summarize completed all-row fits before outer fits finish."
    )
    args = parser.parse_args()
    bootstrap = bootstrap_tables(args.output)
    floors, summary = floor_tables(args.output)
    bootstrap.to_csv(args.output / "paired_bootstrap_rmse.csv", index=False)
    floors.to_csv(args.output / "floor_diagnostics.csv", index=False)
    summary.to_csv(args.output / "floor_summary.csv", index=False)
    if not args.skip_oof:
        components, groups = oof_tables(args.output)
        components.to_csv(args.output / "oof_component_diagnostics.csv", index=False)
        groups.to_csv(args.output / "oof_group_contributions.csv", index=False)
    note = {
        "seed": SEED,
        "draws": DRAWS,
        "negative_rmse_difference_favors_candidate": True,
        "bootstrap": (
            "Paired percentile intervals. Bank blocks are connected source memberships; "
            "fresh blocks are launch families. Also show individual coordinate/run resampling as sensitivity."
        ),
        "limitations": (
            "Conditional on fitted models and evaluation populations; no model-fitting or training-seed variance. "
            "Retrospective development data, with only six fresh launch clusters. "
            "No multiplicity adjustment and no inferential selection of an improved method."
        ),
        "floor_diagnostics": (
            "A strictly active three-SD margin makes gamma locally unidentified through the floor; "
            "a boundary gamma on that plateau has no distinct predictive meaning."
        ),
        "oof_groups": (
            "Signed MSE contributions E[group error times total error] sum to aggregate MSE. "
            "Negative contributions indicate error cancellation, not negative error variance. "
            "QA/other is the residual task category."
        ),
        "complete_oof_requested": not args.skip_oof,
    }
    (args.output / "statistics_summary.json").write_text(json.dumps(note, indent=2) + "\n")
    print(bootstrap[bootstrap.scheme.isin(["source_block", "launch"])].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
