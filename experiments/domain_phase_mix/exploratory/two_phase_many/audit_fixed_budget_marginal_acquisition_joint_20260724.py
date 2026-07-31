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
"""Audit a marginal-acquisition phase model under an exact 280-row budget.

The aggregate spine is the frozen physical pooled-acquisition model fit only to
phase-tied policies and repeated tied frontier controls. Complete antithetic
phase pairs identify two orthogonal responses:

* an odd family-pooled order potential coupled to aggregate marginal value;
* an even nonnegative family switching cost.

The odd response needs only the two treatments in an antithetic pair. The even
response is fit only on frontier-fiber pairs whose tied center is one of the
eight controls charged to every phase-probe arm. Aggressive-pair controls are
therefore never credited outside the 280-checkpoint accounting.
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
    audit_frontier_control_aggregate_identification_20260724 as aggregate_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_marginal_acquisition_phase_potential_20260724 as phase_potential,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "fixed_budget_marginal_acquisition_joint_20260724"
DEFAULT_SEEDS = (20260724, 20260725, 20260726)
BOOTSTRAP_DRAWS = 20_000
NUMERICAL_FLOOR = 1e-12
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

# Eight treatments contain only two fiber pairs and cannot identify three even
# family coefficients. The 32- and 112-treatment arms are the smallest and
# largest identified allocations in the preregistered strict-budget protocol.
BUDGET_ARMS = tuple(
    arm
    for arm in strict_protocol.ARMS
    if arm.name in {"all_tied", "frontier_controls_only", "phase_probe_32", "phase_probe_112"}
)
ODD_RIDGE = {"uncheatable": 0.0, "table9": 10.0}
EVEN_RIDGE = {"uncheatable": 0.0, "table9": 1.0}
PHASE_MODEL_NAME = "marginal_family_order_plus_family_switching"
ODD_MODEL_NAME = "marginal_family_order"
GLOBAL_PHASE_MODEL_NAME = "marginal_global_order_plus_family_switching"
GLOBAL_ODD_MODEL_NAME = "marginal_global_order"
EVEN_MODEL_NAME = "family_switching"
OBSERVATORY_BASELINES = (
    "compact_retained_state",
    "hierarchical_phase_bucket_replay",
    "separate_heads",
    "effective_exposure",
    "bucket_family_grp",
)


@dataclass(frozen=True)
class FittedPhaseCorrection:
    """Fitted odd order and even switching heads."""

    odd: phase_potential.FitResult
    odd_global: phase_potential.FitResult
    even: phase_potential.FitResult
    selected_pair_indices: np.ndarray
    even_pair_indices: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    return parser.parse_args()


def interleaved_pair_order(
    dataset: phase_potential.PairDataset,
    model: orthogonal.AggregateModel,
    seed: int,
) -> np.ndarray:
    """Interleave phase designs while guaranteeing family coverage.

    Fiber pairs are stratified by the family receiving the most moved mass.
    This gives each family two directions per anchor before another cycle and
    prevents a small random allocation from leaving a family unidentified.
    Aggressive balanced partitions remain a separate stratum.
    """

    rng = np.random.default_rng(seed)
    even = phase_potential.even_design(dataset, model)
    dominant_family = np.argmax(even, axis=1)
    strata = dataset.frame[["panel", "anchor_id"]].copy()
    strata["family_stratum"] = [
        f"family_{family}" if panel == "frontier_phase_fiber" else "balanced_partition"
        for panel, family in zip(strata["panel"], dominant_family, strict=True)
    ]
    queues: dict[tuple[str, str, str], list[int]] = {}
    for key, indices in strata.groupby(
        ["panel", "anchor_id", "family_stratum"],
        sort=True,
    ).indices.items():
        local = dataset.frame.iloc[np.asarray(indices, dtype=int)].copy()
        local["pair_index"] = np.asarray(indices, dtype=int)
        direction_queues: dict[str, list[int]] = {}
        for direction, direction_indices in local.groupby("direction_group", sort=True).indices.items():
            rows = local.iloc[np.asarray(direction_indices, dtype=int)].sort_values("phase_tv")
            values = rows["pair_index"].astype(int).tolist()
            if len(values) > 1:
                offset = int(rng.integers(0, len(values)))
                values = values[offset:] + values[:offset]
            direction_queues[str(direction)] = values
        directions = list(direction_queues)
        directions = [directions[index] for index in rng.permutation(len(directions))]
        queue: list[int] = []
        while any(direction_queues.values()):
            for direction in directions:
                if direction_queues[direction]:
                    queue.append(direction_queues[direction].pop(0))
        queues[(str(key[0]), str(key[1]), str(key[2]))] = queue

    anchor_queues: dict[str, list[int]] = {}
    for anchor in sorted({key[1] for key in queues}):
        group_order = [key for key in sorted(queues) if key[1] == anchor]
        anchor_queue: list[int] = []
        while any(queues[key] for key in group_order):
            for key in group_order:
                if queues[key]:
                    anchor_queue.append(queues[key].pop(0))
        anchor_queues[anchor] = anchor_queue

    ordered: list[int] = []
    anchor_order = sorted(anchor_queues)
    while any(anchor_queues.values()):
        for anchor in anchor_order:
            if anchor_queues[anchor]:
                ordered.append(anchor_queues[anchor].pop(0))
    result = np.asarray(ordered, dtype=int)
    if len(result) != dataset.n or len(np.unique(result)) != dataset.n:
        raise ValueError("Pair ordering is not a permutation")
    return result


def selected_pairs(
    dataset: phase_potential.PairDataset,
    model: orthogonal.AggregateModel,
    treatment_count: int,
    seed: int,
) -> np.ndarray:
    if treatment_count == 0:
        return np.asarray([], dtype=int)
    if treatment_count % 2:
        raise ValueError("Phase treatment count must contain complete pairs")
    selected = interleaved_pair_order(dataset, model, seed)[: treatment_count // 2]
    anchor_counts = dataset.frame.iloc[selected].groupby("anchor_id", sort=True).size()
    if int(anchor_counts.max() - anchor_counts.min()) > 1:
        raise ValueError("Selected phase pairs are not balanced by anchor")
    fiber = selected[dataset.frame.iloc[selected]["panel"].eq("frontier_phase_fiber").to_numpy()]
    even = phase_potential.even_design(dataset, model)
    covered = set(np.argmax(even[fiber], axis=1).tolist())
    if covered != set(range(len(model.families.names))):
        raise ValueError("Selected fiber pairs do not identify every family")
    return selected


def charged_control_even_target(
    dataset: phase_potential.PairDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Center pair averages on the eight controls charged to the budget.

    The fiber panel has four independent tied controls for each anchor. Their
    mean estimates the phase-tied response without leaking the same control
    realization into many direction rows. The resulting even-response noise is
    approximately ``sqrt(3/2)`` times the odd-response noise: two averaged
    treatments plus a four-run control mean.
    """

    fiber = dataset.frame["panel"].eq("frontier_phase_fiber")
    controls = (
        dataset.frame.loc[fiber, ["anchor_id", "seed_block", "center_bpb"]].drop_duplicates().reset_index(drop=True)
    )
    if len(controls) != strict_protocol.fixed_budget.CONTROL_COUNT:
        raise ValueError(f"Expected eight charged controls, found {len(controls)}")
    center_mean = controls.groupby("anchor_id", sort=True)["center_bpb"].mean()
    target = (
        0.5 * (dataset.frame["plus_bpb"] + dataset.frame["minus_bpb"]) - dataset.frame["anchor_id"].map(center_mean)
    ).to_numpy(dtype=float)
    noise = np.asarray(dataset.noise, dtype=float) * np.sqrt(1.5)
    return target, noise


def phase_design(
    weights: np.ndarray,
    model: orthogonal.AggregateModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return family-pooled odd and even phase features."""

    aggregate = orthogonal.aggregate_weights(weights, model.phase_fraction)
    contrast = weights[:, 1, :] - weights[:, 0, :]
    marginal = phase_potential.marginal_bucket_value(model, aggregate)
    alpha0 = model.phase_fraction
    alpha1 = 1.0 - alpha0
    bucket_odd = -alpha0 * alpha1 * marginal * contrast
    moved = alpha0 * alpha1 * np.abs(contrast)
    odd = np.column_stack([bucket_odd[:, members].sum(axis=1) for members in model.families.members])
    even = np.column_stack([moved[:, members].sum(axis=1) ** 2 for members in model.families.members])
    return odd, even


def fit_phase_correction(
    dataset: phase_potential.PairDataset,
    model: orthogonal.AggregateModel,
    treatment_count: int,
    seed: int,
) -> FittedPhaseCorrection | None:
    selected = selected_pairs(dataset, model, treatment_count, seed)
    if not len(selected):
        return None
    candidate = next(item for item in phase_potential.CANDIDATES if item.name == "marginal_family_phase_potential")
    odd_design, odd_family = phase_potential.candidate_design(dataset, model, candidate)
    global_candidate = next(
        item for item in phase_potential.CANDIDATES if item.name == "marginal_global_phase_potential"
    )
    global_design, global_family = phase_potential.candidate_design(
        dataset,
        model,
        global_candidate,
    )
    even_design = phase_potential.even_design(dataset, model)
    even_target, even_noise = charged_control_even_target(dataset)
    even_selected = selected[dataset.frame.iloc[selected]["panel"].eq("frontier_phase_fiber").to_numpy()]
    if len(even_selected) < even_design.shape[1]:
        raise ValueError("The charged fiber controls do not identify the even family head")
    odd = phase_potential.fit_nonnegative_head(
        odd_design[selected],
        dataset.odd[selected],
        dataset.noise[selected],
        odd_family,
        ODD_RIDGE[dataset.target],
        0.0,
    )
    odd_global = phase_potential.fit_nonnegative_head(
        global_design[selected],
        dataset.odd[selected],
        dataset.noise[selected],
        global_family,
        0.0,
        0.0,
    )
    even = phase_potential.fit_nonnegative_head(
        even_design[even_selected],
        even_target[even_selected],
        even_noise[even_selected],
        np.arange(even_design.shape[1], dtype=int),
        EVEN_RIDGE[dataset.target],
        0.0,
    )
    return FittedPhaseCorrection(
        odd=odd,
        odd_global=odd_global,
        even=even,
        selected_pair_indices=selected,
        even_pair_indices=even_selected,
    )


def phase_components(
    weights: np.ndarray,
    model: orthogonal.AggregateModel,
    fitted: FittedPhaseCorrection | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fitted is None:
        zeros = np.zeros(len(weights), dtype=float)
        return zeros, zeros, zeros
    odd, even = phase_design(weights, model)
    odd_prediction = fitted.odd.predict(odd)
    global_odd_prediction = fitted.odd_global.predict(odd.sum(axis=1, keepdims=True))
    even_prediction = fitted.even.predict(even)
    tied = np.all(np.abs(weights[:, 0, :] - weights[:, 1, :]) < 1e-12, axis=1)
    if (
        np.max(np.abs(odd_prediction[tied]), initial=0.0) > 1e-12
        or np.max(np.abs(global_odd_prediction[tied]), initial=0.0) > 1e-12
        or np.max(np.abs(even_prediction[tied]), initial=0.0) > 1e-12
    ):
        raise AssertionError("Phase correction is not zero for tied policies")
    return global_odd_prediction, odd_prediction, even_prediction


def pair_generalization_rows(
    dataset: phase_potential.PairDataset,
    model: orthogonal.AggregateModel,
    fitted: FittedPhaseCorrection | None,
    arm: strict_protocol.BudgetArm,
    seed: int,
) -> list[dict[str, Any]]:
    if fitted is None:
        return []
    candidate = next(item for item in phase_potential.CANDIDATES if item.name == "marginal_family_phase_potential")
    odd_design, _odd_family = phase_potential.candidate_design(dataset, model, candidate)
    even_design = phase_potential.even_design(dataset, model)
    predicted_odd = fitted.odd.predict(odd_design)
    predicted_even = fitted.even.predict(even_design)
    test = np.setdiff1d(
        np.arange(dataset.n),
        fitted.selected_pair_indices,
        assume_unique=True,
    )
    rows: list[dict[str, Any]] = []
    for scope, mask in {
        "all_unselected_pairs": np.ones(len(test), dtype=bool),
        "unselected_fiber": dataset.frame.iloc[test]["panel"].eq("frontier_phase_fiber").to_numpy(),
        "unselected_aggressive": dataset.frame.iloc[test]["panel"].eq("aggressive_balanced_partition").to_numpy(),
    }.items():
        indices = test[mask]
        if not len(indices):
            continue
        observed_treatments = np.concatenate(
            [
                dataset.even[indices] + dataset.odd[indices],
                dataset.even[indices] - dataset.odd[indices],
            ]
        )
        predicted_treatments = np.concatenate(
            [
                predicted_even[indices] + predicted_odd[indices],
                predicted_even[indices] - predicted_odd[indices],
            ]
        )
        rows.extend(
            [
                {
                    "target": dataset.target,
                    "arm": arm.name,
                    "seed": seed,
                    "scope": scope,
                    "response": "odd_order",
                    **phase_potential.metric_row(dataset.odd[indices], predicted_odd[indices]),
                },
                {
                    "target": dataset.target,
                    "arm": arm.name,
                    "seed": seed,
                    "scope": scope,
                    "response": "even_switching",
                    **phase_potential.metric_row(dataset.even[indices], predicted_even[indices]),
                },
                {
                    "target": dataset.target,
                    "arm": arm.name,
                    "seed": seed,
                    "scope": scope,
                    "response": "combined_treatment_delta",
                    **phase_potential.metric_row(observed_treatments, predicted_treatments),
                },
            ]
        )
    return rows


def baseline_predictions(target: str, positions: np.ndarray) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for model_id in OBSERVATORY_BASELINES:
        path = orthogonal.OBSERVATORY_CACHE / target / "two_phase" / f"{model_id}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        predictions[f"observatory_{model_id}"] = np.asarray(
            payload["prediction"],
            dtype=float,
        )[positions]
    return predictions


def metric_rows_for_prediction(
    frame: pd.DataFrame,
    observed: np.ndarray,
    predicted: np.ndarray,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for scope, mask in aggregate_audit.scope_masks(frame).items():
        rows.append(
            {
                **metadata,
                "scope": scope,
                **orthogonal.regression_metrics(observed[mask], predicted[mask]),
            }
        )
    return rows


def run_target(
    target: str,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    ) = comparators.target_data(target)
    positions = evaluation_frame["position"].to_numpy(dtype=int)
    pair_dataset = phase_potential.pair_datasets()[target]
    metrics: list[dict[str, Any]] = []
    pair_metrics: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    base_frame = evaluation_frame.copy()
    base_frame["target"] = target
    base_frame["cluster"] = clusters
    base_frame["observed"] = observed
    base_frame["evaluation_row"] = np.arange(len(base_frame))

    for model_name, prediction in baseline_predictions(target, positions).items():
        metadata = {
            "target": target,
            "arm": "observatory",
            "seed": -1,
            "model": model_name,
            "tied_count": 0,
            "control_count": 0,
            "treatment_count": 0,
        }
        metrics.extend(metric_rows_for_prediction(base_frame, observed, prediction, metadata))
        local = base_frame.copy()
        for name, value in metadata.items():
            local[name] = value
        local["predicted"] = prediction
        local["residual"] = prediction - observed
        prediction_frames.append(local)

    for arm in BUDGET_ARMS:
        arm_seeds = (seeds[0],) if arm.name == "all_tied" else seeds
        for seed in arm_seeds:
            training = strict_protocol.aggregate_training_dataset(
                target,
                single,
                controls,
                arm,
                seed,
            )
            fold = strict_protocol.grouped_stratified_folds(training, seed)
            aggregate_fit = aggregate_audit.frozen_pooled_fit(training, fold)
            model = aggregate_fit.model
            fitted_phase = fit_phase_correction(
                pair_dataset,
                model,
                arm.treatment_count,
                seed,
            )
            aggregate_prediction = model.predict(evaluation_weights)
            global_odd_correction, odd_correction, even_correction = phase_components(
                evaluation_weights,
                model,
                fitted_phase,
            )
            for model_name, prediction in (
                ("physical_pooled_acquisition", aggregate_prediction),
                (
                    f"physical_pooled_acquisition_plus_{ODD_MODEL_NAME}",
                    aggregate_prediction + odd_correction,
                ),
                (
                    f"physical_pooled_acquisition_plus_{GLOBAL_ODD_MODEL_NAME}",
                    aggregate_prediction + global_odd_correction,
                ),
                (
                    f"physical_pooled_acquisition_plus_{EVEN_MODEL_NAME}",
                    aggregate_prediction + even_correction,
                ),
                (
                    f"physical_pooled_acquisition_plus_{GLOBAL_PHASE_MODEL_NAME}",
                    aggregate_prediction + global_odd_correction + even_correction,
                ),
                (
                    f"physical_pooled_acquisition_plus_{PHASE_MODEL_NAME}",
                    aggregate_prediction + odd_correction + even_correction,
                ),
            ):
                metadata = {
                    "target": target,
                    "arm": arm.name,
                    "seed": seed,
                    "model": model_name,
                    **asdict(arm),
                    "aggregate_fit_rows": training.n,
                    "aggregate_oof_rmse": orthogonal.regression_metrics(
                        training.y,
                        aggregate_fit.oof_prediction,
                    )["rmse"],
                }
                metrics.extend(metric_rows_for_prediction(base_frame, observed, prediction, metadata))
                local = base_frame.copy()
                for name, value in metadata.items():
                    local[name] = value
                local["predicted"] = prediction
                local["residual"] = prediction - observed
                prediction_frames.append(local)

            pair_metrics.extend(
                pair_generalization_rows(
                    pair_dataset,
                    model,
                    fitted_phase,
                    arm,
                    seed,
                )
            )
            if fitted_phase is not None:
                for response, labels, coefficients in (
                    ("odd_global_order", ("global",), fitted_phase.odd_global.coefficients),
                    ("odd_order", model.families.names, fitted_phase.odd.coefficients),
                    ("even_switching", model.families.names, fitted_phase.even.coefficients),
                ):
                    for family, coefficient in zip(
                        labels,
                        coefficients,
                        strict=True,
                    ):
                        coefficient_rows.append(
                            {
                                "target": target,
                                "arm": arm.name,
                                "seed": seed,
                                "response": response,
                                "family": family,
                                "coefficient": float(coefficient),
                                "selected_pair_count": len(fitted_phase.selected_pair_indices),
                                "even_pair_count": len(fitted_phase.even_pair_indices),
                            }
                        )

    return (
        pd.DataFrame(metrics),
        pd.DataFrame(pair_metrics),
        pd.DataFrame(coefficient_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def prediction_slice(
    predictions: pd.DataFrame,
    target: str,
    arm: str,
    seed: int,
    model: str,
    scope: str,
) -> pd.DataFrame:
    selected = predictions[
        predictions["target"].eq(target)
        & predictions["arm"].eq(arm)
        & predictions["seed"].eq(seed)
        & predictions["model"].eq(model)
    ].copy()
    if selected.empty:
        raise KeyError(f"Missing prediction slice for {target}/{arm}/{seed}/{model}")
    return selected.loc[aggregate_audit.scope_masks(selected)[scope]].reset_index(drop=True)


def bootstrap_contrasts(
    predictions: pd.DataFrame,
    seeds: tuple[int, ...],
    draws: int,
) -> pd.DataFrame:
    records = []
    scopes = ("all", "append_only_archive", "append_only_without_compact_optimum")
    candidate_names = (
        f"physical_pooled_acquisition_plus_{GLOBAL_ODD_MODEL_NAME}",
        f"physical_pooled_acquisition_plus_{ODD_MODEL_NAME}",
        f"physical_pooled_acquisition_plus_{EVEN_MODEL_NAME}",
        f"physical_pooled_acquisition_plus_{GLOBAL_PHASE_MODEL_NAME}",
        f"physical_pooled_acquisition_plus_{PHASE_MODEL_NAME}",
    )
    for target_index, target in enumerate(orthogonal.TARGETS):
        for arm_index, arm in enumerate(BUDGET_ARMS):
            if arm.treatment_count == 0:
                continue
            for seed_index, seed in enumerate(seeds):
                for scope_index, scope in enumerate(scopes):
                    for candidate_index, candidate_name in enumerate(candidate_names):
                        candidate = prediction_slice(
                            predictions,
                            target,
                            arm.name,
                            seed,
                            candidate_name,
                            scope,
                        )
                        references = (
                            (
                                "aggregate_only",
                                prediction_slice(
                                    predictions,
                                    target,
                                    arm.name,
                                    seed,
                                    "physical_pooled_acquisition",
                                    scope,
                                ),
                            ),
                            (
                                "observatory_compact_retained_state",
                                prediction_slice(
                                    predictions,
                                    target,
                                    "observatory",
                                    -1,
                                    "observatory_compact_retained_state",
                                    scope,
                                ),
                            ),
                        )
                        for reference_index, (reference_name, reference) in enumerate(references):
                            records.append(
                                {
                                    "target": target,
                                    "arm": arm.name,
                                    "seed": seed,
                                    "scope": scope,
                                    "candidate": candidate_name,
                                    "reference": reference_name,
                                    **strict_protocol.cluster_bootstrap(
                                        candidate["observed"].to_numpy(dtype=float),
                                        candidate["predicted"].to_numpy(dtype=float),
                                        reference["predicted"].to_numpy(dtype=float),
                                        candidate["cluster"].to_numpy(dtype=object),
                                        draws,
                                        20260724
                                        + 1_000_000 * target_index
                                        + 100_000 * arm_index
                                        + 10_000 * seed_index
                                        + 1000 * scope_index
                                        + 100 * candidate_index
                                        + reference_index,
                                    ),
                                }
                            )
    return pd.DataFrame(records)


def write_plots(
    output_dir: Path,
    metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
) -> None:
    selected = metrics[
        metrics["arm"].isin([arm.name for arm in BUDGET_ARMS])
        & metrics["scope"].eq("append_only_without_compact_optimum")
    ].copy()
    figure = px.line(
        selected,
        x="treatment_count",
        y="rmse",
        color="model",
        line_dash="seed",
        markers=True,
        facet_col="target",
        hover_data=["arm", "spearman", "calibration_slope", "regret_at_1", "optimism_gt_0p05"],
        title="Exact 280-row budget: aggregate spine and marginal-acquisition phase correction",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
    )
    figure.update_layout(template="plotly_white", width=1450, height=650)
    figure.write_html(
        output_dir / "joint_budget_metrics.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )

    combined = pair_metrics[
        pair_metrics["response"].eq("combined_treatment_delta") & pair_metrics["scope"].eq("all_unselected_pairs")
    ].copy()
    pair_figure = px.line(
        combined,
        x=combined["arm"].map({arm.name: arm.treatment_count for arm in BUDGET_ARMS}),
        y="rmse_ratio",
        color="target",
        line_dash="seed",
        markers=True,
        hover_data=["arm", "spearman", "calibration_slope", "resolved_sign_accuracy"],
        title="Unselected fixed-aggregate pair generalization",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
    )
    pair_figure.add_hline(y=1.0, line_dash="dash", line_color="#334155")
    pair_figure.update_xaxes(title="Phase-treatment rows in the 280-row budget")
    pair_figure.update_layout(template="plotly_white", width=1050, height=650)
    pair_figure.write_html(
        output_dir / "phase_pair_generalization.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    candidate_models = (
        "physical_pooled_acquisition",
        f"physical_pooled_acquisition_plus_{GLOBAL_ODD_MODEL_NAME}",
        f"physical_pooled_acquisition_plus_{ODD_MODEL_NAME}",
        f"physical_pooled_acquisition_plus_{EVEN_MODEL_NAME}",
        f"physical_pooled_acquisition_plus_{GLOBAL_PHASE_MODEL_NAME}",
        f"physical_pooled_acquisition_plus_{PHASE_MODEL_NAME}",
        "observatory_compact_retained_state",
    )
    heldout = metrics[
        metrics["scope"].eq("append_only_without_compact_optimum") & metrics["model"].isin(candidate_models)
    ].copy()
    phase = pair_metrics[
        pair_metrics["scope"].eq("all_unselected_pairs") & pair_metrics["response"].eq("combined_treatment_delta")
    ].copy()
    lines = [
        "# Exact-budget marginal-acquisition joint model",
        "",
        "## Frozen mechanism",
        "",
        (
            r"The phase-invariant aggregate spine is "
            r"\(A(a)=b-\sum_i\beta_iR(c_i a_i)-\sum_fB_fR(E_f(a))\), "
            r"with \(R(x)=1-\exp(-(\rho x)^p)\)."
        ),
        "",
        (
            r"For \(d=w^{(1)}-w^{(0)}\), the phase correction is "
            r"\(\Delta L(a,d)=-\alpha_0\alpha_1\sum_i\gamma_{f(i)}"
            r"m_i(a)d_i+\sum_f\kappa_f[\alpha_0\alpha_1"
            r"\sum_{i\in f}|d_i|]^2\), where "
            r"\(m_i(a)=-\partial A(a)/\partial a_i\)."
        ),
        "",
        (
            "The first term is odd under order reversal and rewards late placement in proportion to aggregate "
            "marginal value. The second is even and prices family-level distribution switching. Both vanish "
            "exactly for phase-tied policies."
        ),
        "",
        "## Heldout metrics",
        "",
        heldout[
            [
                "target",
                "arm",
                "seed",
                "model",
                "treatment_count",
                "rmse",
                "spearman",
                "calibration_slope",
                "regret_at_1",
                "optimism_gt_0p05",
                "worst_optimism",
            ]
        ]
        .sort_values(["target", "rmse"])
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Unselected phase-pair generalization",
        "",
        phase[
            [
                "target",
                "arm",
                "seed",
                "rmse",
                "rmse_ratio",
                "spearman",
                "calibration_slope",
                "resolved_sign_accuracy",
            ]
        ]
        .sort_values(["target", "rmse_ratio"])
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Cluster-bootstrap contrasts",
        "",
        contrasts[
            [
                "target",
                "arm",
                "seed",
                "scope",
                "reference",
                "candidate_rmse",
                "reference_rmse",
                "rmse_delta",
                "rmse_delta_ci_low",
                "rmse_delta_ci_high",
                "probability_candidate_better",
            ]
        ]
        .sort_values(["target", "scope", "rmse_delta"])
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "The sealed targeted pairwise panel was not accessed.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    results = [run_target(target, seeds) for target in orthogonal.TARGETS]
    metrics = pd.concat([result[0] for result in results], ignore_index=True)
    pair_metrics = pd.concat([result[1] for result in results], ignore_index=True)
    coefficients = pd.concat([result[2] for result in results], ignore_index=True)
    predictions = pd.concat([result[3] for result in results], ignore_index=True)
    contrasts = bootstrap_contrasts(predictions, seeds, int(args.bootstrap_draws))
    metrics.to_csv(args.output_dir / "heldout_metrics.csv", index=False)
    pair_metrics.to_csv(args.output_dir / "phase_pair_metrics.csv", index=False)
    coefficients.to_csv(args.output_dir / "phase_coefficients.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    contrasts.to_csv(args.output_dir / "cluster_bootstrap_contrasts.csv", index=False)
    write_plots(args.output_dir, metrics, pair_metrics)
    write_report(args.output_dir, metrics, pair_metrics, contrasts)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "total_checkpoint_budget": strict_protocol.TOTAL_BUDGET,
                "arms": [asdict(arm) for arm in BUDGET_ARMS],
                "seeds": seeds,
                "aggregate": {
                    "form": "physical_pooled_acquisition",
                    "rho": aggregate_audit.FROZEN_POOLED_SHAPE.rho,
                    "power": aggregate_audit.FROZEN_POOLED_SHAPE.power,
                    "l2": aggregate_audit.FROZEN_POOLED_L2,
                    "loss": aggregate_audit.FROZEN_POOLED_CONFIG.loss,
                },
                "phase": {
                    "odd_global": "one shared aggregate-marginal order coefficient",
                    "odd": "family-pooled aggregate-marginal order potential",
                    "even": "nonnegative quadratic family switching cost",
                    "odd_ridge": ODD_RIDGE,
                    "even_ridge": EVEN_RIDGE,
                    "aggressive_controls_used_for_fit": False,
                },
                "evaluation_excludes_phase_training_series": True,
                "sealed_targeted_pairwise_panel_accessed": False,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "git": git_metadata(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
