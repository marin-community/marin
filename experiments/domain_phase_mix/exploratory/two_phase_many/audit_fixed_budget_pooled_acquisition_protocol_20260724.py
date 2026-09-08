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
"""Audit pooled acquisition and phase ordering under a strict 280-row budget.

This audit fixes two ambiguities in the first fixed-budget experiment:

* aggregate hyperparameters are selected only from rows charged to each arm;
* explicit controls isolate repeated frontier anchors from phase treatments.

Every phase-treatment subset is balanced across both frontier anchors before
cycling through seed blocks. Coordinate repeats stay in one aggregate CV fold.
The evaluation archive excludes every training coordinate and every phase
population used to estimate the phase correction.
"""

from __future__ import annotations

import argparse
import json
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
    benchmark_fixed_budget_aggregate_phase_20260724 as fixed_budget,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_tied_backbone_phase_order_20260724 as phase_benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "fixed_budget_pooled_acquisition_protocol_20260724"
DEFAULT_SEEDS = (20260724, 20260725, 20260726)
TOTAL_BUDGET = 280
N_FOLDS = 5
BOOTSTRAP_DRAWS = 20_000
AGGREGATE_WORKERS = 8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class BudgetArm:
    """One exact-budget allocation."""

    name: str
    tied_count: int
    control_count: int
    treatment_count: int

    def __post_init__(self) -> None:
        if self.tied_count + self.control_count + self.treatment_count != TOTAL_BUDGET:
            raise ValueError(f"{self.name} does not sum to {TOTAL_BUDGET}")


ARMS = (
    BudgetArm("all_tied", 280, 0, 0),
    BudgetArm("frontier_controls_only", 272, 8, 0),
    BudgetArm("phase_probe_8", 264, 8, 8),
    BudgetArm("phase_probe_32", 240, 8, 32),
    BudgetArm("phase_probe_112", 160, 8, 112),
)


@dataclass(frozen=True)
class AggregateSelection:
    """Nested-CV selection and full-data pooled-acquisition fit."""

    model: orthogonal.AggregateModel
    selected_row: dict[str, Any]
    sweep: pd.DataFrame
    oof_prediction: np.ndarray
    fold: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def aggregate_candidates() -> tuple[tuple[orthogonal.AggregateConfig, orthogonal.AggregateShape, float], ...]:
    """Return the preregistered physical pooled-acquisition grid."""
    return tuple(
        (
            orthogonal.AggregateConfig(
                name=f"family_none_{loss}",
                include_families=True,
                replay=orthogonal.ReplayKind.NONE,
                loss=loss,
            ),
            orthogonal.AggregateShape(rho, power),
            l2,
        )
        for loss in orthogonal.LOSS_KINDS
        for rho in orthogonal.RHO_GRID
        for power in orthogonal.POWER_GRID
        for l2 in orthogonal.L2_GRID
    )


def coordinate_groups(dataset: pooled.Dataset) -> np.ndarray:
    return np.asarray([orthogonal.coordinate_key(weights).hex() for weights in dataset.weights], dtype=object)


def row_strata(dataset: pooled.Dataset) -> np.ndarray:
    role = dataset.frame.get("budget_role")
    panel = dataset.frame.get("panel_source")
    strata = []
    for index in range(dataset.n):
        if role is not None and str(role.iloc[index]) == "frontier_control":
            strata.append("frontier_control")
        elif panel is not None and pd.notna(panel.iloc[index]):
            strata.append(str(panel.iloc[index]))
        else:
            strata.append("tied_swarm")
    return np.asarray(strata, dtype=object)


def grouped_stratified_folds(dataset: pooled.Dataset, seed: int) -> np.ndarray:
    """Assign coordinate groups to folds while balancing source strata."""
    groups = coordinate_groups(dataset)
    strata = row_strata(dataset)
    unique_records = (
        pd.DataFrame({"group": groups, "stratum": strata})
        .groupby("group", sort=True, as_index=False)
        .agg(stratum=("stratum", "first"))
    )
    rng = np.random.default_rng(seed)
    group_fold: dict[str, int] = {}
    for _stratum, indices in unique_records.groupby("stratum", sort=True).indices.items():
        local = np.asarray(indices, dtype=int)
        shuffled = local[rng.permutation(len(local))]
        offset = int(rng.integers(0, N_FOLDS))
        for rank, index in enumerate(shuffled):
            group_fold[str(unique_records.iloc[index]["group"])] = (offset + rank) % N_FOLDS
    fold = np.asarray([group_fold[str(group)] for group in groups], dtype=int)
    if set(fold) != set(range(N_FOLDS)):
        raise ValueError("Grouped aggregate CV produced an empty fold")
    for group in np.unique(groups):
        if len(np.unique(fold[groups == group])) != 1:
            raise AssertionError("Coordinate repeats leaked across aggregate CV folds")
    return fold


def select_aggregate_model(
    dataset: pooled.Dataset,
    families: orthogonal.FamilyPartition,
    seed: int,
) -> AggregateSelection:
    """Select nonlinear shape and ridge only from the charged aggregate rows."""
    fold = grouped_stratified_folds(dataset, seed)
    all_indices = np.arange(dataset.n)

    def evaluate_candidate(
        indexed_candidate: tuple[
            int,
            tuple[orthogonal.AggregateConfig, orthogonal.AggregateShape, float],
        ],
    ) -> tuple[dict[str, Any], np.ndarray]:
        candidate_index, (config, shape, l2) = indexed_candidate
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for fold_index in range(N_FOLDS):
            test = np.flatnonzero(fold == fold_index)
            train = np.setdiff1d(all_indices, test, assume_unique=True)
            fitted = orthogonal.fit_aggregate(dataset, train, config, shape, l2, families)
            prediction[test] = fitted.predict(dataset.weights[test])
        if not np.isfinite(prediction).all():
            raise ValueError(f"Non-finite grouped OOF prediction for candidate {candidate_index}")
        return (
            {
                "candidate_index": candidate_index,
                **asdict(config),
                **asdict(shape),
                "l2": l2,
                "active_parameter_count": 1,
                **orthogonal.regression_metrics(dataset.y, prediction),
            },
            prediction,
        )

    indexed_candidates = tuple(enumerate(aggregate_candidates()))
    evaluated = Parallel(n_jobs=AGGREGATE_WORKERS, backend="loky")(
        delayed(evaluate_candidate)(indexed_candidate) for indexed_candidate in indexed_candidates
    )
    records = [record for record, _prediction in evaluated]
    predictions = [prediction for _record, prediction in evaluated]
    sweep = pd.DataFrame(records)
    selected = sweep.sort_values(
        ["rmse", "regret_at_1", "optimism_gt_0p05", "l2", "rho", "power"],
        ascending=[True, True, True, False, True, True],
        ignore_index=True,
    ).iloc[0]
    config = orthogonal.AggregateConfig(
        name=str(selected["name"]),
        include_families=bool(selected["include_families"]),
        replay=orthogonal.ReplayKind(str(selected["replay"])),
        loss=str(selected["loss"]),
    )
    shape = orthogonal.AggregateShape(float(selected["rho"]), float(selected["power"]))
    model = orthogonal.fit_aggregate(
        dataset,
        np.arange(dataset.n),
        config,
        shape,
        float(selected["l2"]),
        families,
    )
    active = int(np.sum(model.bucket_coef > 1e-12) + np.sum(model.family_coef > 1e-12) + 1)
    sweep.loc[sweep["candidate_index"].eq(int(selected["candidate_index"])), "active_parameter_count"] = active
    return AggregateSelection(
        model=model,
        selected_row={**selected.to_dict(), "active_parameter_count": active},
        sweep=sweep,
        oof_prediction=predictions[int(selected["candidate_index"])],
        fold=fold,
    )


def aggregate_training_dataset(
    target: str,
    single: pooled.Dataset,
    controls: pooled.Dataset,
    arm: BudgetArm,
    seed: int,
) -> pooled.Dataset:
    if arm.control_count == 0:
        if arm.tied_count != single.n:
            raise ValueError("A control-free arm must use the complete tied panel")
        frame = single.frame.copy()
        frame["budget_role"] = "tied_swarm"
        return pooled.Dataset(
            name=f"delphi_3e18_{target}_{arm.name}",
            frame=frame,
            y=np.asarray(single.y, dtype=float),
            weights=np.asarray(single.weights, dtype=float),
            c0=np.asarray(single.c0, dtype=float),
            c1=np.asarray(single.c1, dtype=float),
            domain_names=list(single.domain_names),
        )
    if arm.control_count != fixed_budget.CONTROL_COUNT:
        raise ValueError("Only the complete eight-control block is supported")
    return fixed_budget.aggregate_training_dataset(
        target,
        single,
        controls,
        arm.tied_count,
        seed,
    )


def physical_backbone(
    model: orthogonal.AggregateModel,
    families: orthogonal.FamilyPartition,
) -> phase_benchmark.PhaseBackbone:
    channel_count = len(model.bucket_coef) + len(model.family_coef)
    return phase_benchmark.PhaseBackbone(
        name="physical_pooled_acquisition",
        aggregate_predictor=model,
        phase_fraction=model.phase_fraction,
        c_total=model.c_total,
        families=families,
        rho=np.full(channel_count, model.shape.rho, dtype=float),
        power=np.full(channel_count, model.shape.power, dtype=float),
        value_coef=np.concatenate([model.bucket_coef, model.family_coef]),
        channel_group=np.concatenate(
            [
                families.bucket_group,
                np.arange(len(model.family_coef), dtype=int),
            ]
        ),
        include_family_channels=bool(len(model.family_coef)),
    )


def balanced_phase_pair_order(rows: orthogonal.PhaseRows, seed: int) -> list[tuple[int, int]]:
    """Interleave anchors before seed blocks and directions."""
    pairs = phase_benchmark.antithetic_pair_indices(rows, np.arange(len(rows.frame)))
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for plus, minus in zip(pairs.plus, pairs.minus, strict=True):
        row = rows.frame.iloc[plus]
        if str(row["panel"]) != "frontier_fiber":
            continue
        key = (str(row["source_anchor_key"]), int(row["seed_block"]))
        grouped.setdefault(key, []).append((int(plus), int(minus)))
    anchors = sorted({anchor for anchor, _seed in grouped})
    seed_blocks = sorted({seed_block for _anchor, seed_block in grouped})
    if len(anchors) != 2 or len(seed_blocks) != 4 or len(grouped) != 8:
        raise ValueError("Expected two frontier anchors with four seed blocks each")
    rng = np.random.default_rng(seed)
    queues = {key: [values[index] for index in rng.permutation(len(values))] for key, values in grouped.items()}
    per_anchor_seeds = {
        anchor: [seed_blocks[index] for index in rng.permutation(len(seed_blocks))] for anchor in anchors
    }
    order: list[tuple[int, int]] = []
    for depth in range(max(map(len, queues.values()))):
        for seed_rank in range(len(seed_blocks)):
            for anchor in anchors:
                seed_block = per_anchor_seeds[anchor][seed_rank]
                queue = queues[(anchor, seed_block)]
                if depth < len(queue):
                    order.append(queue[depth])
    if len(order) != 96:
        raise ValueError(f"Expected 96 complete frontier pairs, found {len(order)}")
    return order


def phase_training_indices(rows: orthogonal.PhaseRows, treatment_count: int, seed: int) -> np.ndarray:
    if treatment_count == 0:
        return np.asarray([], dtype=int)
    if treatment_count % 2:
        raise ValueError("Treatment count must contain complete antithetic pairs")
    pairs = balanced_phase_pair_order(rows, seed)[: treatment_count // 2]
    selected = np.asarray([index for pair in pairs for index in pair], dtype=int)
    anchors = rows.frame.iloc[selected]["source_anchor_key"].value_counts()
    if anchors.max() - anchors.min() > 2:
        raise ValueError("Phase treatment allocation is not balanced across anchors")
    return selected


def phase_configs(target: str, treatment_count: int) -> tuple[orthogonal.PhaseConfig, ...]:
    huber = 0.001 if target == "uncheatable" else 0.002
    null = orthogonal.PhaseConfig(orthogonal.PhaseKind.NULL, orthogonal.PhaseShiftKind.NONE, huber)
    if treatment_count == 0:
        return (null,)
    hellinger = orthogonal.PhaseConfig(
        orthogonal.PhaseKind.NULL,
        orthogonal.PhaseShiftKind.HELLINGER,
        huber,
    )
    if treatment_count < 32:
        return null, hellinger
    retention = orthogonal.PhaseConfig(
        orthogonal.PhaseKind.GLOBAL_RETENTION,
        orthogonal.PhaseShiftKind.HELLINGER,
        huber,
    )
    return null, hellinger, retention


def evaluation_clusters(
    frame: pd.DataFrame,
    positions: np.ndarray,
    reference: pooled.Dataset,
    heldout_frame: pd.DataFrame,
) -> np.ndarray:
    clusters = []
    for source, position in zip(frame["source"], positions, strict=True):
        if source == "original_two_phase_swarm":
            row = reference.frame.iloc[int(position)]
            clusters.append(f"reference::{row.get('panel_source', 'fit_swarm')}")
            continue
        row = heldout_frame.iloc[int(position) - reference.n]
        anchor = row.get("anchor_id")
        anchor_suffix = "" if pd.isna(anchor) else f"::{anchor}"
        clusters.append(f"archive::{row['training_series']}{anchor_suffix}")
    return np.asarray(clusters, dtype=object)


def cluster_bootstrap(
    observed: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    clusters: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Paired cluster bootstrap over proposal series and phase anchors."""
    labels = np.unique(clusters)
    candidate_error = (candidate - observed) ** 2
    reference_error = (reference - observed) ** 2
    candidate_sums = np.asarray([candidate_error[clusters == label].sum() for label in labels])
    reference_sums = np.asarray([reference_error[clusters == label].sum() for label in labels])
    counts = np.asarray([np.sum(clusters == label) for label in labels], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(labels), size=(draws, len(labels)))
    sampled_count = counts[sampled].sum(axis=1)
    candidate_rmse = np.sqrt(candidate_sums[sampled].sum(axis=1) / sampled_count)
    reference_rmse = np.sqrt(reference_sums[sampled].sum(axis=1) / sampled_count)
    delta = candidate_rmse - reference_rmse
    point_candidate = float(np.sqrt(np.mean(candidate_error)))
    point_reference = float(np.sqrt(np.mean(reference_error)))
    return {
        "cluster_count": len(labels),
        "candidate_rmse": point_candidate,
        "reference_rmse": point_reference,
        "rmse_delta": point_candidate - point_reference,
        "rmse_delta_ci_low": float(np.quantile(delta, 0.025)),
        "rmse_delta_ci_high": float(np.quantile(delta, 0.975)),
        "probability_candidate_better": float(np.mean(delta < 0.0)),
    }


def model_key(arm: str, seed: int, phase_model: str) -> str:
    return f"{arm}__seed{seed}__{phase_model}"


def run_target(
    target: str,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, _single_evaluation_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    controls = fixed_budget.fiber_control_dataset(target, single)
    families = orthogonal.family_partition(single.domain_names)
    phase_rows = orthogonal.load_phase_rows(
        target,
        single.domain_names,
        float(np.mean(single.c0 / (single.c0 + single.c1))),
    )
    evaluation_frame, evaluation_weights, observed, positions = phase_benchmark.coordinate_disjoint_combined_rows(
        target,
        reference,
        single,
        heldout_frame,
        heldout_weights,
    )
    clusters = evaluation_clusters(evaluation_frame, positions, reference, heldout_frame)
    baseline_path = orthogonal.OBSERVATORY_CACHE / target / "two_phase" / "compact_retained_state.json"
    baseline_prediction = np.asarray(json.loads(baseline_path.read_text())["prediction"], dtype=float)[positions]

    selections: list[dict[str, Any]] = []
    sweeps: list[pd.DataFrame] = []
    prediction_columns: dict[str, np.ndarray] = {"observatory_compact_retained_state": baseline_prediction}
    metadata: dict[str, dict[str, Any]] = {
        "observatory_compact_retained_state": {
            "target": target,
            "arm": "observatory",
            "seed": -1,
            "tied_count": 0,
            "control_count": 0,
            "treatment_count": 0,
            "phase_model": "incumbent",
        }
    }

    for arm in ARMS:
        arm_seeds = (seeds[0],) if arm.name == "all_tied" else seeds
        for seed in arm_seeds:
            training = aggregate_training_dataset(target, single, controls, arm, seed)
            selection = select_aggregate_model(training, families, seed)
            backbone = physical_backbone(selection.model, families)
            selection_record = {
                "target": target,
                "arm": arm.name,
                "seed": seed,
                **asdict(arm),
                "fit_rows": training.n,
                **selection.selected_row,
                "oof_rmse": orthogonal.regression_metrics(training.y, selection.oof_prediction)["rmse"],
                "oof_spearman": orthogonal.regression_metrics(training.y, selection.oof_prediction)["spearman"],
            }
            selections.append(selection_record)
            local_sweep = selection.sweep.copy()
            local_sweep.insert(0, "seed", seed)
            local_sweep.insert(0, "arm", arm.name)
            local_sweep.insert(0, "target", target)
            sweeps.append(local_sweep)

            training_indices = phase_training_indices(phase_rows, arm.treatment_count, seed)
            aggregate_prediction = selection.model.predict(evaluation_weights)
            for config in phase_configs(target, arm.treatment_count):
                phase_model = phase_benchmark.fit_phase(
                    phase_rows,
                    training_indices,
                    backbone,
                    config,
                )
                key = model_key(arm.name, seed, config.name)
                prediction_columns[key] = aggregate_prediction + phase_model.predict_delta(evaluation_weights)
                metadata[key] = {
                    "target": target,
                    "arm": arm.name,
                    "seed": seed,
                    **asdict(arm),
                    "phase_model": config.name,
                    "phase_params": json.dumps(phase_model.params.tolist()),
                }

    predictions = evaluation_frame.copy()
    predictions["target"] = target
    predictions["cluster"] = clusters
    predictions["observed"] = observed
    predictions["position"] = positions
    long_predictions = []
    metric_records = []
    for key, prediction in prediction_columns.items():
        local = predictions.copy()
        local["model_key"] = key
        local["predicted"] = prediction
        local["residual"] = prediction - observed
        for name, value in metadata[key].items():
            local[name] = value
        long_predictions.append(local)
        metric_records.append(
            {
                "model_key": key,
                **metadata[key],
                **orthogonal.regression_metrics(observed, prediction),
            }
        )
    return (
        pd.DataFrame(selections),
        pd.concat(sweeps, ignore_index=True),
        pd.DataFrame(metric_records),
        pd.concat(long_predictions, ignore_index=True),
    )


def prediction_lookup(predictions: pd.DataFrame, target: str, key: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = predictions[predictions["target"].eq(target) & predictions["model_key"].eq(key)].copy()
    if frame.empty:
        raise KeyError(f"Missing prediction key {target}/{key}")
    return frame, frame["predicted"].to_numpy(dtype=float)


def preregistered_contrasts(
    predictions: pd.DataFrame,
    seeds: tuple[int, ...],
    draws: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target_index, target in enumerate(orthogonal.TARGETS):
        all_tied_key = model_key("all_tied", seeds[0], "phase_null")
        for seed_index, seed in enumerate(seeds):
            contrasts = (
                (
                    "frontier_controls_vs_all_tied",
                    model_key("frontier_controls_only", seed, "phase_null"),
                    all_tied_key,
                ),
                (
                    "phase_probe_32_hellinger_vs_crs",
                    model_key(
                        "phase_probe_32",
                        seed,
                        f"phase_null_hellinger_h{0.001 if target == 'uncheatable' else 0.002:g}",
                    ),
                    "observatory_compact_retained_state",
                ),
                (
                    "phase_probe_32_hellinger_increment",
                    model_key(
                        "phase_probe_32",
                        seed,
                        f"phase_null_hellinger_h{0.001 if target == 'uncheatable' else 0.002:g}",
                    ),
                    model_key("phase_probe_32", seed, "phase_null"),
                ),
                (
                    "phase_probe_112_retention_increment",
                    model_key(
                        "phase_probe_112",
                        seed,
                        f"global_retention_hellinger_h{0.001 if target == 'uncheatable' else 0.002:g}",
                    ),
                    model_key(
                        "phase_probe_112",
                        seed,
                        f"phase_null_hellinger_h{0.001 if target == 'uncheatable' else 0.002:g}",
                    ),
                ),
            )
            for contrast_index, (name, candidate_key, reference_key) in enumerate(contrasts):
                frame, candidate = prediction_lookup(predictions, target, candidate_key)
                _reference_frame, reference = prediction_lookup(predictions, target, reference_key)
                records.append(
                    {
                        "target": target,
                        "seed": seed,
                        "contrast": name,
                        "candidate": candidate_key,
                        "reference": reference_key,
                        **cluster_bootstrap(
                            frame["observed"].to_numpy(dtype=float),
                            candidate,
                            reference,
                            frame["cluster"].to_numpy(dtype=object),
                            draws,
                            20260724 + 100 * target_index + 10 * seed_index + contrast_index,
                        ),
                    }
                )
    return pd.DataFrame(records)


def plot_budget_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    selected = metrics[metrics["arm"].ne("observatory")].copy()
    figure = px.scatter(
        selected,
        x="treatment_count",
        y="rmse",
        color="phase_model",
        symbol="arm",
        facet_col="target",
        hover_data=["seed", "spearman", "regret_at_1", "optimism_gt_0p05"],
        title="Strict 280-row budget: pooled acquisition plus phase-order correction",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
    )
    figure.update_layout(template="plotly_white", width=1400, height=650)
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    selections: pd.DataFrame,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    selected_columns = [
        "target",
        "arm",
        "seed",
        "fit_rows",
        "rho",
        "power",
        "l2",
        "loss",
        "active_parameter_count",
        "oof_rmse",
        "oof_spearman",
    ]
    metric_columns = [
        "target",
        "arm",
        "seed",
        "phase_model",
        "rmse",
        "spearman",
        "calibration_slope",
        "regret_at_1",
        "optimism_gt_0p05",
        "worst_optimism",
    ]
    lines = [
        "# Strict-budget pooled acquisition and phase-order audit",
        "",
        "## Frozen aggregate form",
        "",
        r"With aggregate exposure \(q_i=c_i[\alpha_0w_i^{(0)}+\alpha_1w_i^{(1)}]\),",
        "",
        r"\[L_{\mathrm{agg}}=b-\sum_i\beta_i(1-e^{-(\rho q_i)^p})" r"-\sum_f B_f(1-e^{-(\rho Q_f)^p}).\]",
        "",
        (
            "All coefficients are nonnegative. The nonlinear shape, ridge, and Gaussian/Huber loss are selected "
            "by coordinate-grouped CV using only aggregate rows charged to that arm. No replay coefficient is fit."
        ),
        "",
        "## Budget arms",
        "",
        pd.DataFrame([asdict(arm) for arm in ARMS]).to_markdown(index=False),
        "",
        "## Nested aggregate selections",
        "",
        selections[selected_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Combined heldout metrics",
        "",
        metrics[metric_columns]
        .sort_values(["target", "rmse", "regret_at_1"])
        .to_markdown(
            index=False,
            floatfmt=".6f",
        ),
        "",
        "## Preregistered source-and-anchor cluster bootstrap contrasts",
        "",
        contrasts.to_markdown(index=False, floatfmt=".6f"),
        "",
        (
            "The `frontier_controls_vs_all_tied` contrast isolates the effect of spending eight checkpoints on "
            "repeated frontier coordinates. The two phase-increment contrasts compare models with the same "
            "aggregate fit. The CRS contrast tests the combined 240/8/32 procedure against the Observatory incumbent."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds:
        raise ValueError("At least one target-free subset seed is required")

    selection_frames = []
    sweep_frames = []
    metric_frames = []
    prediction_frames = []
    for target in orthogonal.TARGETS:
        selections, sweeps, metrics, predictions = run_target(target, seeds)
        selection_frames.append(selections)
        sweep_frames.append(sweeps)
        metric_frames.append(metrics)
        prediction_frames.append(predictions)
    all_selections = pd.concat(selection_frames, ignore_index=True)
    all_sweeps = pd.concat(sweep_frames, ignore_index=True)
    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    contrasts = preregistered_contrasts(all_predictions, seeds, int(args.bootstrap_draws))

    all_selections.to_csv(args.output_dir / "aggregate_selections.csv", index=False)
    all_sweeps.to_csv(args.output_dir / "aggregate_cv_sweeps.csv", index=False)
    all_metrics.to_csv(args.output_dir / "combined_metrics.csv", index=False)
    all_predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    contrasts.to_csv(args.output_dir / "cluster_bootstrap_contrasts.csv", index=False)
    plot_budget_metrics(all_metrics, args.output_dir / "budget_metrics.html")
    write_report(args.output_dir, all_selections, all_metrics, contrasts)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "total_checkpoint_budget": TOTAL_BUDGET,
                "arms": [asdict(arm) for arm in ARMS],
                "seeds": seeds,
                "aggregate_candidate_count": len(aggregate_candidates()),
                "aggregate_folds": N_FOLDS,
                "coordinate_grouped_cv": True,
                "balanced_phase_anchors": True,
                "bootstrap_draws": int(args.bootstrap_draws),
                "evaluation_excludes_phase_training_series": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
