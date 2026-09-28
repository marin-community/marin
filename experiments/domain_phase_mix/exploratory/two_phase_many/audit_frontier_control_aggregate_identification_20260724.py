# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "joblib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Identify the drop, control, and net effects of frontier-control allocation.

The strict 280-run comparison changes two things at once: it drops eight tied
coordinates and adds eight repeated frontier controls. This diagnostic fits
the same frozen aggregate procedures to:

* all 280 tied rows;
* the exact 272 tied-row subset for each seed;
* that same subset plus the eight frontier controls.

It also separates the original two-phase fit swarm from the append-only
archive. No phase-treatment outcomes or sealed-panel outcomes are accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from joblib import Parallel, delayed

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_aggregate_comparators_20260724 as comparators,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_pooled_acquisition_protocol_20260724 as strict_protocol,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "frontier_control_aggregate_identification_20260724"
DEFAULT_SEEDS = strict_protocol.DEFAULT_SEEDS
MODEL_WORKERS = 8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
COMPACT_OPTIMUM_CLUSTER = "archive::delphi_compact_sub280_optimum_validation_panel_20260721"

FROZEN_POOLED_CONFIG = orthogonal.AggregateConfig(
    name="family_none_huber_frozen",
    include_families=True,
    replay=orthogonal.ReplayKind.NONE,
    loss="huber",
)
FROZEN_POOLED_SHAPE = orthogonal.AggregateShape(rho=0.25, power=1.0)
FROZEN_POOLED_L2 = 0.1


@dataclass(frozen=True)
class DiagnosticTask:
    """One target, subset seed, and aggregate-training design."""

    target: str
    seed: int
    design: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap-draws", type=int, default=strict_protocol.BOOTSTRAP_DRAWS)
    parser.add_argument("--workers", type=int, default=MODEL_WORKERS)
    return parser.parse_args()


def tied_subset_dataset(
    target: str,
    single: pooled.Dataset,
    count: int,
    seed: int,
) -> pooled.Dataset:
    selected = strict_protocol.fixed_budget.tied_training_indices(single, count, seed)
    frame = single.frame.iloc[selected].copy().reset_index(drop=True)
    frame["budget_role"] = "tied_swarm"
    return pooled.Dataset(
        name=f"delphi_3e18_{target}_tied_subset{count}_seed{seed}",
        frame=frame,
        y=np.asarray(single.y[selected], dtype=float),
        weights=np.asarray(single.weights[selected], dtype=float),
        c0=np.asarray(single.c0, dtype=float),
        c1=np.asarray(single.c1, dtype=float),
        domain_names=list(single.domain_names),
    )


def training_dataset(
    target: str,
    single: pooled.Dataset,
    controls: pooled.Dataset,
    design: str,
    seed: int,
) -> pooled.Dataset:
    if design == "all_tied_280":
        frame = single.frame.copy()
        frame["budget_role"] = "tied_swarm"
        return pooled.Dataset(
            name=f"delphi_3e18_{target}_all_tied_280",
            frame=frame,
            y=np.asarray(single.y, dtype=float),
            weights=np.asarray(single.weights, dtype=float),
            c0=np.asarray(single.c0, dtype=float),
            c1=np.asarray(single.c1, dtype=float),
            domain_names=list(single.domain_names),
        )
    if design == "tied_272":
        return tied_subset_dataset(target, single, 272, seed)
    if design == "tied_272_plus_controls":
        return strict_protocol.fixed_budget.aggregate_training_dataset(
            target,
            single,
            controls,
            272,
            seed,
        )
    raise ValueError(f"Unknown diagnostic design {design!r}")


def frozen_pooled_fit(
    dataset: pooled.Dataset,
    fold: np.ndarray,
) -> comparators.AggregateFit:
    all_indices = np.arange(dataset.n)
    families = orthogonal.family_partition(dataset.domain_names)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for fold_index in range(strict_protocol.N_FOLDS):
        test = np.flatnonzero(fold == fold_index)
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        model = orthogonal.fit_aggregate(
            dataset,
            train,
            FROZEN_POOLED_CONFIG,
            FROZEN_POOLED_SHAPE,
            FROZEN_POOLED_L2,
            families,
        )
        prediction[test] = model.predict(dataset.weights[test])
    model = orthogonal.fit_aggregate(
        dataset,
        all_indices,
        FROZEN_POOLED_CONFIG,
        FROZEN_POOLED_SHAPE,
        FROZEN_POOLED_L2,
        families,
    )
    active = int(np.sum(model.bucket_coef > 1e-12) + np.sum(model.family_coef > 1e-12) + 1)
    metrics = orthogonal.regression_metrics(dataset.y, prediction)
    return comparators.AggregateFit(
        model=model,
        oof_prediction=prediction,
        selected_l2=FROZEN_POOLED_L2,
        parameter_count=active,
        selection_rows=(
            {
                "rho": FROZEN_POOLED_SHAPE.rho,
                "power": FROZEN_POOLED_SHAPE.power,
                "l2": FROZEN_POOLED_L2,
                "loss": FROZEN_POOLED_CONFIG.loss,
                **metrics,
                "selected": True,
            },
        ),
    )


def evaluation_scope(frame: pd.DataFrame) -> np.ndarray:
    source = frame["source"].astype(str)
    cluster = frame["cluster"].astype(str)
    scope = np.full(len(frame), "all", dtype=object)
    scope[source.eq("original_two_phase_swarm").to_numpy()] = "original_two_phase_swarm"
    scope[source.eq("append_only_archive").to_numpy()] = "append_only_archive"
    scope[source.eq("append_only_archive").to_numpy() & ~cluster.eq(COMPACT_OPTIMUM_CLUSTER).to_numpy()] = (
        "append_only_without_compact_optimum"
    )
    return scope


def scope_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    source = frame["source"].astype(str)
    cluster = frame["cluster"].astype(str)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "original_two_phase_swarm": source.eq("original_two_phase_swarm").to_numpy(),
        "append_only_archive": source.eq("append_only_archive").to_numpy(),
        "append_only_without_compact_optimum": (
            source.eq("append_only_archive").to_numpy() & ~cluster.eq(COMPACT_OPTIMUM_CLUSTER).to_numpy()
        ),
    }


def run_task(
    task: DiagnosticTask,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    (
        _reference,
        _heldout_frame,
        _heldout_weights,
        single,
        controls,
        evaluation_frame,
        evaluation_weights,
        observed,
        clusters,
    ) = comparators.target_data(task.target)
    training = training_dataset(task.target, single, controls, task.design, task.seed)
    fold = strict_protocol.grouped_stratified_folds(training, task.seed)
    fits = {
        "physical_pooled_acquisition_frozen": frozen_pooled_fit(training, fold),
        "compact_retained_state_tied": comparators.compact_fit(training, fold),
        "canonical_dsp_tied": comparators.canonical_fit(training, fold),
    }
    metric_rows = []
    prediction_frames = []
    for model_name, fit in fits.items():
        prediction = fit.model.predict(evaluation_weights)
        metadata = {
            "target": task.target,
            "seed": task.seed,
            "design": task.design,
            "fit_rows": training.n,
            "aggregate_model": model_name,
            "selected_l2": fit.selected_l2,
            "parameter_count": fit.parameter_count,
            **{
                f"oof_{key}": value
                for key, value in orthogonal.regression_metrics(training.y, fit.oof_prediction).items()
            },
        }
        local = evaluation_frame.copy()
        local["target"] = task.target
        local["cluster"] = clusters
        local["observed"] = observed
        local["aggregate_model"] = model_name
        local["design"] = task.design
        local["seed"] = task.seed
        local["predicted"] = prediction
        local["residual"] = prediction - observed
        local["evaluation_row"] = np.arange(len(local))
        local["scope"] = evaluation_scope(local)
        prediction_frames.append(local)
        for scope, mask in scope_masks(local).items():
            metric_rows.append(
                {
                    **metadata,
                    "scope": scope,
                    **orthogonal.regression_metrics(observed[mask], prediction[mask]),
                }
            )
    return metric_rows, pd.concat(prediction_frames, ignore_index=True)


def prediction_slice(
    predictions: pd.DataFrame,
    target: str,
    model: str,
    design: str,
    seed: int,
    scope: str,
) -> pd.DataFrame:
    selected = predictions[
        predictions["target"].eq(target)
        & predictions["aggregate_model"].eq(model)
        & predictions["design"].eq(design)
        & predictions["seed"].eq(seed)
    ].copy()
    mask = scope_masks(selected)[scope]
    selected = selected.loc[mask].reset_index(drop=True)
    if selected.empty:
        raise KeyError(f"Missing prediction slice for {target}/{model}/{design}/{seed}/{scope}")
    return selected


def bootstrap_contrasts(
    predictions: pd.DataFrame,
    seeds: tuple[int, ...],
    draws: int,
) -> pd.DataFrame:
    records = []
    models = (
        "physical_pooled_acquisition_frozen",
        "compact_retained_state_tied",
        "canonical_dsp_tied",
    )
    scopes = ("all", "append_only_archive", "append_only_without_compact_optimum")
    contrasts = (
        ("drop_8_tied", "tied_272", "all_tied_280"),
        ("add_8_controls", "tied_272_plus_controls", "tied_272"),
        ("net_fixed_budget_swap", "tied_272_plus_controls", "all_tied_280"),
    )
    for target_index, target in enumerate(orthogonal.TARGETS):
        for model_index, model in enumerate(models):
            for seed_index, seed in enumerate(seeds):
                for scope_index, scope in enumerate(scopes):
                    for contrast_index, (name, candidate_design, reference_design) in enumerate(contrasts):
                        candidate = prediction_slice(
                            predictions,
                            target,
                            model,
                            candidate_design,
                            seed,
                            scope,
                        )
                        reference = prediction_slice(
                            predictions,
                            target,
                            model,
                            reference_design,
                            seed,
                            scope,
                        )
                        records.append(
                            {
                                "target": target,
                                "aggregate_model": model,
                                "seed": seed,
                                "scope": scope,
                                "contrast": name,
                                "candidate_design": candidate_design,
                                "reference_design": reference_design,
                                **strict_protocol.cluster_bootstrap(
                                    candidate["observed"].to_numpy(dtype=float),
                                    candidate["predicted"].to_numpy(dtype=float),
                                    reference["predicted"].to_numpy(dtype=float),
                                    candidate["cluster"].to_numpy(dtype=object),
                                    draws,
                                    20260724
                                    + 100_000 * target_index
                                    + 10_000 * model_index
                                    + 1000 * seed_index
                                    + 100 * scope_index
                                    + contrast_index,
                                ),
                            }
                        )
    return pd.DataFrame(records)


def direct_model_contrasts(
    predictions: pd.DataFrame,
    seeds: tuple[int, ...],
    draws: int,
) -> pd.DataFrame:
    records = []
    pairs = (
        (
            "frozen_pooled_vs_tied_crs",
            "physical_pooled_acquisition_frozen",
            "compact_retained_state_tied",
        ),
        (
            "frozen_pooled_vs_tied_canonical_dsp",
            "physical_pooled_acquisition_frozen",
            "canonical_dsp_tied",
        ),
    )
    scopes = ("all", "append_only_archive", "append_only_without_compact_optimum")
    for target_index, target in enumerate(orthogonal.TARGETS):
        for seed_index, seed in enumerate(seeds):
            for scope_index, scope in enumerate(scopes):
                for design_index, design in enumerate(("all_tied_280", "tied_272", "tied_272_plus_controls")):
                    for pair_index, (name, candidate_model, reference_model) in enumerate(pairs):
                        candidate = prediction_slice(
                            predictions,
                            target,
                            candidate_model,
                            design,
                            seed,
                            scope,
                        )
                        reference = prediction_slice(
                            predictions,
                            target,
                            reference_model,
                            design,
                            seed,
                            scope,
                        )
                        records.append(
                            {
                                "target": target,
                                "seed": seed,
                                "scope": scope,
                                "design": design,
                                "contrast": name,
                                "candidate_model": candidate_model,
                                "reference_model": reference_model,
                                **strict_protocol.cluster_bootstrap(
                                    candidate["observed"].to_numpy(dtype=float),
                                    candidate["predicted"].to_numpy(dtype=float),
                                    reference["predicted"].to_numpy(dtype=float),
                                    candidate["cluster"].to_numpy(dtype=object),
                                    draws,
                                    20260724
                                    + 100_000 * target_index
                                    + 10_000 * seed_index
                                    + 1000 * scope_index
                                    + 100 * design_index
                                    + pair_index,
                                ),
                            }
                        )
    return pd.DataFrame(records)


def cluster_attribution(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (target, model, seed), group in predictions.groupby(
        ["target", "aggregate_model", "seed"],
        sort=True,
    ):
        controls = group[group["design"].eq("tied_272_plus_controls")].set_index("evaluation_row")
        tied = group[group["design"].eq("tied_272")].set_index("evaluation_row")
        shared = controls.index.intersection(tied.index)
        if len(shared) != len(controls) or len(shared) != len(tied):
            raise ValueError("Control and tied-only predictions do not share the evaluation rows")
        joined = controls.loc[shared]
        tied_prediction = tied.loc[shared, "predicted"].to_numpy(dtype=float)
        observed = joined["observed"].to_numpy(dtype=float)
        candidate_squared_error = (joined["predicted"].to_numpy(dtype=float) - observed) ** 2
        reference_squared_error = (tied_prediction - observed) ** 2
        local = pd.DataFrame(
            {
                "cluster": joined["cluster"].astype(str).to_numpy(),
                "squared_error_delta": candidate_squared_error - reference_squared_error,
            }
        )
        for cluster, values in local.groupby("cluster", sort=True):
            records.append(
                {
                    "target": target,
                    "aggregate_model": model,
                    "seed": seed,
                    "cluster": cluster,
                    "n": len(values),
                    "mean_squared_error_delta": float(values["squared_error_delta"].mean()),
                    "total_squared_error_delta": float(values["squared_error_delta"].sum()),
                }
            )
    return pd.DataFrame(records)


def plot_control_effects(contrasts: pd.DataFrame, output_path: Path) -> None:
    selected = contrasts[contrasts["scope"].eq("append_only_without_compact_optimum")].copy()
    figure = px.scatter(
        selected,
        x="contrast",
        y="rmse_delta",
        error_y=selected["rmse_delta_ci_high"] - selected["rmse_delta"],
        error_y_minus=selected["rmse_delta"] - selected["rmse_delta_ci_low"],
        color="aggregate_model",
        symbol="seed",
        facet_col="target",
        title="Frontier-control allocation: drop, add, and net RMSE effects",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#334155")
    figure.update_layout(template="plotly_white", width=1500, height=700)
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    direct: pd.DataFrame,
) -> None:
    metric_columns = [
        "target",
        "scope",
        "aggregate_model",
        "design",
        "seed",
        "fit_rows",
        "oof_rmse",
        "rmse",
        "spearman",
        "calibration_slope",
        "regret_at_1",
        "optimism_gt_0p05",
    ]
    lines = [
        "# Frontier-control aggregate identification audit",
        "",
        "## Frozen comparison",
        "",
        (
            r"The pooled form is frozen to \(\rho=0.25\), \(p=1\), Huber loss, and \(\ell_2=0.1\). "
            "This removes the earlier grid-selection advantage. CRS and canonical DSP retain their previously "
            "specified fitting procedures."
        ),
        "",
        "## Aggregate metrics",
        "",
        metrics[metric_columns]
        .sort_values(
            ["target", "scope", "design", "rmse"],
        )
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Drop, control, and net contrasts",
        "",
        contrasts.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Direct aggregate-model contrasts",
        "",
        direct.to_markdown(index=False, floatfmt=".6f"),
        "",
        (
            "The add-control contrast compares the same 272 tied coordinates with and without controls. The "
            "archive-only scopes remove the original two-phase fit swarm; the strictest scope also removes the "
            "compact-optimum validation series."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds:
        raise ValueError("At least one subset seed is required")
    tasks = tuple(
        DiagnosticTask(target, seed, design)
        for target in orthogonal.TARGETS
        for seed in seeds
        for design in ("all_tied_280", "tied_272", "tied_272_plus_controls")
    )
    results = Parallel(n_jobs=args.workers, backend="loky")(delayed(run_task)(task) for task in tasks)
    metrics = pd.DataFrame([row for result in results for row in result[0]])
    predictions = pd.concat([result[1] for result in results], ignore_index=True)
    contrasts = bootstrap_contrasts(predictions, seeds, int(args.bootstrap_draws))
    direct = direct_model_contrasts(predictions, seeds, int(args.bootstrap_draws))
    attribution = cluster_attribution(predictions)

    metrics.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    contrasts.to_csv(args.output_dir / "control_allocation_contrasts.csv", index=False)
    direct.to_csv(args.output_dir / "direct_model_contrasts.csv", index=False)
    attribution.to_csv(args.output_dir / "cluster_attribution.csv", index=False)
    plot_control_effects(contrasts, args.output_dir / "control_allocation_effects.html")
    write_report(args.output_dir, metrics, contrasts, direct)

    script_path = Path(__file__).resolve()
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "designs": {
                    "all_tied_280": {"tied": 280, "controls": 0},
                    "tied_272": {"tied": 272, "controls": 0},
                    "tied_272_plus_controls": {"tied": 272, "controls": 8},
                },
                "frozen_pooled_config": {
                    **asdict(FROZEN_POOLED_CONFIG),
                    **asdict(FROZEN_POOLED_SHAPE),
                    "l2": FROZEN_POOLED_L2,
                },
                "phase_treatment_rows_used": 0,
                "sealed_targeted_pairwise_panel_accessed": False,
                "script_sha256": file_sha256(script_path),
                "git": git_metadata(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
