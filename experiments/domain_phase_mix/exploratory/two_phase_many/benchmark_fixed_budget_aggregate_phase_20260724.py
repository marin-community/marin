# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit aggregate-plus-order identification under exactly 280 checkpoints.

Every allocation contains:

* a target-free stratified subset of the independently trained tied swarm;
* eight tied frontier controls, four at each of two frontier aggregates;
* complete antithetic ``+d/-d`` treatment pairs around those controls.

The tied examples fit the aggregate spine. The treatment-control differences
fit a bounded acquisition/retention phase correction. The evaluation archive
excludes the entire phase-fiber series and every tied coordinate, so no
checkpoint used by either fit appears in the combined heldout metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_tied_backbone_phase_order_20260724 as backbone_benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "fixed_budget_aggregate_phase_20260724"
TOTAL_CHECKPOINT_BUDGET = 280
CONTROL_COUNT = 8
TIED_COUNTS = (240, 200, 160, 120, 80)
DEFAULT_SEEDS = (20260724, 20260725, 20260726)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", choices=("all", *orthogonal.TARGETS), default="all")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    return parser.parse_args()


def fiber_control_dataset(target: str, template: pooled.Dataset) -> pooled.Dataset:
    results = pd.read_csv(orthogonal.FIBER_RESULTS)
    controls = results[results["contrast_family"].eq("center_control")].copy()
    if len(controls) != CONTROL_COUNT:
        raise ValueError(f"Expected {CONTROL_COUNT} phase-fiber controls, found {len(controls)}")
    weights = orthogonal.weights_from_long(
        orthogonal.FIBER_PANEL / "phase_weights.csv",
        controls["candidate_id"].astype(str).tolist(),
        template.domain_names,
    )
    if not np.allclose(weights[:, 0, :], weights[:, 1, :], atol=1e-12):
        raise ValueError("Phase-fiber controls are not tied")
    target_column = orthogonal.TARGET_COLUMNS[target]
    return pooled.Dataset(
        name=f"delphi_3e18_{target}_fiber_controls",
        frame=controls.reset_index(drop=True),
        y=controls[target_column].to_numpy(dtype=float),
        weights=weights,
        c0=template.c0,
        c1=template.c1,
        domain_names=template.domain_names,
    )


def tied_training_indices(single: pooled.Dataset, count: int, seed: int) -> np.ndarray:
    if count < 42:
        raise ValueError("The tied design must retain three baselines and all 39 deletion interventions")
    pinned = np.flatnonzero(
        single.frame["panel_source"].eq("domain_deletion").to_numpy()
        | single.frame["source_run_name"]
        .isin(("baseline_proportional", "baseline_unimax", "baseline_stratified"))
        .to_numpy()
    )
    if len(pinned) != 42:
        raise ValueError(f"Expected 42 pinned tied designs, found {len(pinned)}")
    candidates = np.setdiff1d(np.arange(single.n), pinned, assume_unique=True)
    rng = np.random.default_rng(seed)
    selected = np.concatenate([pinned, rng.permutation(candidates)[: count - len(pinned)]])
    return np.sort(selected)


def aggregate_training_dataset(
    target: str,
    single: pooled.Dataset,
    controls: pooled.Dataset,
    tied_count: int,
    seed: int,
) -> pooled.Dataset:
    selected = tied_training_indices(single, tied_count, seed)
    frame = pd.concat(
        [
            single.frame.iloc[selected].assign(budget_role="tied_swarm"),
            controls.frame.assign(budget_role="frontier_control"),
        ],
        ignore_index=True,
        sort=False,
    )
    weights = np.concatenate([single.weights[selected], controls.weights], axis=0)
    target_values = np.concatenate([single.y[selected], controls.y])
    if len(frame) != tied_count + CONTROL_COUNT:
        raise AssertionError("Aggregate training count is inconsistent")
    return pooled.Dataset(
        name=f"delphi_3e18_{target}_budget_tied{tied_count}_seed{seed}",
        frame=frame,
        y=target_values,
        weights=weights,
        c0=single.c0,
        c1=single.c1,
        domain_names=single.domain_names,
    )


def balanced_fiber_pair_order(rows: orthogonal.PhaseRows, seed: int) -> list[tuple[int, int]]:
    pairs = backbone_benchmark.antithetic_pair_indices(rows, np.arange(len(rows.frame)))
    records: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for plus, minus in zip(pairs.plus, pairs.minus, strict=True):
        row = rows.frame.iloc[plus]
        if row["panel"] != "frontier_fiber":
            continue
        group = (str(row["source_anchor_key"]), int(row["seed_block"]))
        records.setdefault(group, []).append((int(plus), int(minus)))
    if len(records) != CONTROL_COUNT:
        raise ValueError(f"Expected {CONTROL_COUNT} frontier anchor/seed groups, found {len(records)}")
    rng = np.random.default_rng(seed)
    queues = {group: [values[index] for index in rng.permutation(len(values))] for group, values in records.items()}
    order: list[tuple[int, int]] = []
    groups = sorted(queues)
    while any(queues.values()):
        for group in groups:
            if queues[group]:
                order.append(queues[group].pop())
    if len(order) != 96:
        raise ValueError(f"Expected 96 complete frontier pairs, found {len(order)}")
    return order


def phase_training_indices(rows: orthogonal.PhaseRows, treatment_count: int, seed: int) -> np.ndarray:
    if treatment_count % 2:
        raise ValueError("Antithetic treatment count must be even")
    pair_order = balanced_fiber_pair_order(rows, seed)
    selected_pairs = pair_order[: treatment_count // 2]
    return np.asarray([index for pair in selected_pairs for index in pair], dtype=int)


def phase_configs(target: str) -> tuple[orthogonal.PhaseConfig, ...]:
    huber = 0.001 if target == "uncheatable" else 0.002
    return (
        orthogonal.PhaseConfig(
            orthogonal.PhaseKind.NULL,
            orthogonal.PhaseShiftKind.NONE,
            huber,
        ),
        orthogonal.PhaseConfig(
            orthogonal.PhaseKind.NULL,
            orthogonal.PhaseShiftKind.HELLINGER,
            huber,
        ),
        orthogonal.PhaseConfig(
            orthogonal.PhaseKind.GLOBAL_RETENTION,
            orthogonal.PhaseShiftKind.HELLINGER,
            huber,
        ),
        orthogonal.PhaseConfig(
            orthogonal.PhaseKind.TWO_GROUP_RETENTION,
            orthogonal.PhaseShiftKind.HELLINGER,
            huber,
        ),
    )


def aggregate_backbones(
    target: str,
    training: pooled.Dataset,
    families: orthogonal.FamilyPartition,
    include_canonical: bool,
) -> tuple[backbone_benchmark.PhaseBackbone, ...]:
    physical = backbone_benchmark.physical_backbone(target, training, families)
    compact_model = observatory.compact_fit(
        training,
        np.arange(training.n),
        l2=1.0,
        policy_class="single_phase",
    )
    backbones = [
        physical,
        replace(
            physical,
            name="compact_retained_state_tied",
            aggregate_predictor=compact_model,
        ),
    ]
    if include_canonical:
        canonical_model = observatory.dsp_fit(
            training,
            np.arange(training.n),
            model_id="canonical",
            policy_class="single_phase",
        )
        backbones.insert(
            1,
            replace(
                physical,
                name="canonical_dsp_tied",
                aggregate_predictor=backbone_benchmark.DSPAggregatePredictor(
                    canonical_model,
                    training,
                ),
            ),
        )
    return tuple(backbones)


def phase_holdout_records(
    target: str,
    rows: orthogonal.PhaseRows,
    backbone: backbone_benchmark.PhaseBackbone,
    model: backbone_benchmark.FittedPhase,
    tied_count: int,
    treatment_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    prediction = model.predict_delta(rows.weights)
    evaluation = ~rows.frame["panel"].eq("frontier_fiber").to_numpy()
    records = []
    groups: tuple[tuple[str, np.ndarray], ...] = (
        ("all_nonfiber", np.flatnonzero(evaluation)),
        *tuple(
            (str(panel), np.flatnonzero(rows.frame["panel"].eq(panel).to_numpy()))
            for panel in sorted(rows.frame.loc[evaluation, "panel"].unique())
        ),
    )
    for scope, indices in groups:
        records.append(
            {
                "target": target,
                "seed": seed,
                "tied_count": tied_count,
                "control_count": CONTROL_COUNT,
                "treatment_count": treatment_count,
                "total_checkpoints": tied_count + CONTROL_COUNT + treatment_count,
                "aggregate_model": backbone.name,
                "phase_model": model.config.name,
                "scope": scope,
                **orthogonal.regression_metrics(rows.target_delta[indices], prediction[indices]),
            }
        )
    return records


def combined_records(
    target: str,
    reference: pooled.Dataset,
    single: pooled.Dataset,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
    backbone: backbone_benchmark.PhaseBackbone,
    phase_model: backbone_benchmark.FittedPhase,
    tied_count: int,
    control_count: int,
    treatment_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    metrics, _predictions = backbone_benchmark.combined_metrics(
        target,
        reference,
        single,
        heldout_frame,
        heldout_weights,
        backbone,
        phase_model,
    )
    wanted = metrics[
        metrics["model"].isin(
            (
                f"{backbone.name}_aggregate_only",
                f"{backbone.name}_plus_phase",
            )
        )
    ].copy()
    records = []
    for row in wanted.to_dict("records"):
        records.append(
            {
                "target": target,
                "seed": seed,
                "tied_count": tied_count,
                "control_count": control_count,
                "treatment_count": treatment_count,
                "total_checkpoints": tied_count + control_count + treatment_count,
                "aggregate_model": backbone.name,
                "phase_model": phase_model.config.name,
                "prediction_kind": "plus_phase" if str(row["model"]).endswith("_plus_phase") else "aggregate_only",
                **{key: value for key, value in row.items() if key != "model"},
            }
        )
    return records


def run_target(target: str, seeds: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, _single_evaluation_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    controls = fiber_control_dataset(target, single)
    families = orthogonal.family_partition(single.domain_names)
    phase_rows = orthogonal.load_phase_rows(
        target,
        single.domain_names,
        float(np.mean(single.c0 / (single.c0 + single.c1))),
    )
    combined: list[dict[str, Any]] = []
    phase_holdout: list[dict[str, Any]] = []
    null_config = phase_configs(target)[0]
    for backbone in aggregate_backbones(target, single, families, include_canonical=True):
        null_phase = backbone_benchmark.FittedPhase(
            config=null_config,
            params=np.asarray([], dtype=float),
            backbone=backbone,
        )
        combined.extend(
            combined_records(
                target,
                reference,
                single,
                heldout_frame,
                heldout_weights,
                backbone,
                null_phase,
                TOTAL_CHECKPOINT_BUDGET,
                0,
                0,
                -1,
            )
        )
    for seed in seeds:
        for tied_count in TIED_COUNTS:
            treatment_count = TOTAL_CHECKPOINT_BUDGET - CONTROL_COUNT - tied_count
            phase_indices = phase_training_indices(phase_rows, treatment_count, seed)
            training = aggregate_training_dataset(target, single, controls, tied_count, seed)
            if training.n + treatment_count != TOTAL_CHECKPOINT_BUDGET:
                raise AssertionError("Allocation does not sum to the checkpoint budget")
            for backbone in aggregate_backbones(
                target,
                training,
                families,
                include_canonical=False,
            ):
                for config in phase_configs(target):
                    phase_model = backbone_benchmark.fit_phase(
                        phase_rows,
                        phase_indices,
                        backbone,
                        config,
                    )
                    phase_holdout.extend(
                        phase_holdout_records(
                            target,
                            phase_rows,
                            backbone,
                            phase_model,
                            tied_count,
                            treatment_count,
                            seed,
                        )
                    )
                    combined.extend(
                        combined_records(
                            target,
                            reference,
                            single,
                            heldout_frame,
                            heldout_weights,
                            backbone,
                            phase_model,
                            tied_count,
                            CONTROL_COUNT,
                            treatment_count,
                            seed,
                        )
                    )
    return pd.DataFrame(combined), pd.DataFrame(phase_holdout)


def plot_learning_curve(frame: pd.DataFrame, output_path: Path, metric: str, title: str) -> None:
    selected = frame[
        frame["scope"].eq("all")
        & frame["prediction_kind"].eq("plus_phase")
        & frame["phase_model"].str.startswith("global_retention_hellinger")
    ].copy()
    figure = px.line(
        selected,
        x="treatment_count",
        y=metric,
        color="aggregate_model",
        facet_col="target",
        markers=True,
        error_y=None,
        line_group="seed",
        hover_data=["seed", "tied_count", "phase_model"],
        title=title,
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
    )
    figure.update_layout(template="plotly_white", width=1300, height=620)
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(output_dir: Path, combined: pd.DataFrame, phase: pd.DataFrame) -> None:
    combined_all = combined[combined["scope"].eq("all") & combined["prediction_kind"].eq("plus_phase")].copy()
    combined_summary = (
        combined_all.groupby(
            ["target", "tied_count", "treatment_count", "aggregate_model", "phase_model"],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            spearman_mean=("spearman", "mean"),
            regret_at_1_mean=("regret_at_1", "mean"),
            optimism_gt_0p05_mean=("optimism_gt_0p05", "mean"),
        )
        .sort_values(["target", "rmse_mean", "regret_at_1_mean"])
    )
    phase_all = phase[phase["scope"].eq("all_nonfiber")].copy()
    phase_summary = (
        phase_all.groupby(
            ["target", "tied_count", "treatment_count", "aggregate_model", "phase_model"],
            as_index=False,
        )
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            spearman_mean=("spearman", "mean"),
            regret_at_1_mean=("regret_at_1", "mean"),
        )
        .sort_values(["target", "rmse_mean", "regret_at_1_mean"])
    )
    combined_summary.to_csv(output_dir / "combined_summary.csv", index=False)
    phase_summary.to_csv(output_dir / "phase_holdout_summary.csv", index=False)
    lines = [
        "# Fixed-budget aggregate plus phase-order audit",
        "",
        (
            "Every row uses exactly 280 checkpoints: tied swarm rows + eight repeated frontier controls + "
            "complete antithetic phase treatments. Hyperparameter forms were frozen before this sweep; "
            "the three deterministic seeds vary only target-free row selection."
        ),
        "",
    ]
    for target in orthogonal.TARGETS:
        lines.extend(
            [
                f"## {target}",
                "",
                "Best combined heldout configurations:",
                "",
                combined_summary[combined_summary["target"].eq(target)].head(15).to_markdown(index=False),
                "",
                "Best non-fiber phase-transfer configurations:",
                "",
                phase_summary[phase_summary["target"].eq(target)].head(15).to_markdown(index=False),
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    targets = orthogonal.TARGETS if args.target == "all" else (args.target,)
    combined_frames = []
    phase_frames = []
    for target in targets:
        combined, phase = run_target(target, seeds)
        combined.to_csv(args.output_dir / f"{target}_combined_metrics.csv", index=False)
        phase.to_csv(args.output_dir / f"{target}_phase_holdout_metrics.csv", index=False)
        combined_frames.append(combined)
        phase_frames.append(phase)
    all_combined = pd.concat(combined_frames, ignore_index=True)
    all_phase = pd.concat(phase_frames, ignore_index=True)
    write_report(args.output_dir, all_combined, all_phase)
    plot_learning_curve(
        all_combined,
        args.output_dir / "combined_rmse_by_budget.html",
        "rmse",
        "Fixed 280-checkpoint budget: aggregate and phase-order allocation",
    )
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "total_checkpoint_budget": TOTAL_CHECKPOINT_BUDGET,
                "control_count": CONTROL_COUNT,
                "tied_counts": (TOTAL_CHECKPOINT_BUDGET, *TIED_COUNTS),
                "treatment_counts": [
                    0,
                    *[TOTAL_CHECKPOINT_BUDGET - CONTROL_COUNT - count for count in TIED_COUNTS],
                ],
                "seeds": seeds,
                "aggregate_models": [
                    "physical_pooled_acquisition",
                    "compact_retained_state_tied",
                ],
                "all_tied_only_comparator": "canonical_dsp_tied",
                "phase_models": [config.name for config in phase_configs("uncheatable")],
                "evaluation_excludes_phase_fiber": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
