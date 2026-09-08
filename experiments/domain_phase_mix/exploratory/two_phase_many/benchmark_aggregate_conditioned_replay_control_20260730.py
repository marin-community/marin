# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Benchmark aggregate-conditioned replay control without heldout selection.

The 300M benchmark uses exactly 280 original two-phase rows plus the 240
single-phase qsplit exposure-average collapses. Corresponding tied and
two-phase rows remain in the same fold. The WSD80 benchmark independently
checks shape extrapolation and raw optimum placement on the dense two-bucket
surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import swarm39_harness_20260725 as swarm39  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_replay_control_20260730 as replay_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    staged_retained_phase_control_20260730 as staged_retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import starcoder_wsd80_panel_20260728 as wsd80  # noqa: E402

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "aggregate_conditioned_replay_control_20260730"
PACKET = (
    SCRIPT_DIR
    / "reference_outputs"
    / "two_phase_solver_gap_collaborator_packet_20260701"
    / "data"
    / "all_300m_checkpoint_metrics.csv"
)
ONE_PHASE_SOURCE = (
    SCRIPT_DIR
    / "reference_outputs"
    / "one_phase_swarm_scores_export_300m_20260630"
    / "one_phase_augmented_fit_panel_uncheatable_table9_scores_300m.csv"
)
TARGETS = {
    "uncheatable": "eval_uncheatable_eval_bpb",
    "table9": "table9_macro_bpb",
}
PHASE_CONFIGS = (
    replay_control.PhaseConfig(
        "ordering_only",
        replay_exponent=0.0,
        use_phase_information=False,
        use_replay_jensen=False,
    ),
    replay_control.PhaseConfig("gradient_q0", replay_exponent=0.0),
    replay_control.PhaseConfig("gradient_control_energy", replay_exponent=0.0, use_control_energy=True),
    replay_control.PhaseConfig("replay_q0p5", replay_exponent=0.5),
    replay_control.PhaseConfig("replay_q1", replay_exponent=1.0),
    replay_control.PhaseConfig("replay_q2", replay_exponent=2.0),
    replay_control.PhaseConfig(
        "aggregate_hessian_control",
        replay_exponent=0.0,
        use_phase_information=False,
        use_replay_jensen=False,
        use_aggregate_curvature=True,
    ),
    replay_control.PhaseConfig(
        "late_reactivation_control",
        replay_exponent=0.0,
        use_phase_information=False,
        use_replay_jensen=False,
        use_reactivation_bregman=True,
    ),
)
STAGED_RETAINED_MODEL = "staged_retained_phase"
STAGED_RETAINED_FAMILY_MODEL = "staged_retained_family_phase"
JOINT_RETAINED_FAMILY_MODEL = "joint_retained_family_phase"
JOINT_RETAINED_ORDER_MODEL = "joint_retained_family_order"
JOINT_RETAINED_BALANCED_MODEL = "joint_retained_family_order_balanced"
RETAINED_POWER_LAW_MODEL = "retained_power_law"
ALL_MODELS = (
    "aggregate_only",
    *(config.name for config in PHASE_CONFIGS),
    STAGED_RETAINED_MODEL,
    STAGED_RETAINED_FAMILY_MODEL,
    JOINT_RETAINED_FAMILY_MODEL,
    JOINT_RETAINED_ORDER_MODEL,
    JOINT_RETAINED_BALANCED_MODEL,
    RETAINED_POWER_LAW_MODEL,
)
INNER_SPLITS = 3
OPTIMUM_GRID = 201
FIBER_AGGREGATES = (0.18, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
TRUE_OPTIMUM = (0.100, 0.500)


@dataclass(frozen=True)
class Dataset:
    name: str
    frame: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    domain_names: tuple[str, ...]
    family_index: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)


def parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("expected at least one seed")
    return values


def geometry_300m(dataset: Dataset) -> replay_control.Geometry:
    beta0 = float(np.median(dataset.c0 / (dataset.c0 + dataset.c1)))
    return replay_control.Geometry(
        c0=dataset.c0,
        c1=dataset.c1,
        phase_0_fraction=beta0,
        family_index=dataset.family_index,
    )


def attach_single_phase_weights(frame: pd.DataFrame, domains: tuple[str, ...]) -> pd.DataFrame:
    source = pd.read_csv(ONE_PHASE_SOURCE).set_index("run_name")
    out = frame.copy()
    single = out["policy_family"].eq("single_phase") & out["source_panel"].eq("all")
    for domain in domains:
        values = out.loc[single, "run_name"].map(source[f"weight_{domain}"])
        out.loc[single, f"phase_0_{domain}"] = values.to_numpy()
        out.loc[single, f"phase_1_{domain}"] = values.to_numpy()
    columns = [f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains]
    if out.loc[single, columns].isna().any(axis=None):
        missing = out.loc[single & out[columns].isna().any(axis=1), "run_name"].tolist()
        raise ValueError(f"missing reconstructed single-phase weights: {missing[:5]}")
    return out


def load_300m(target: str) -> Dataset:
    domains, c0, c1, family_index, _family_names = swarm39._exposure("300m_two_phase_fit")
    frame = attach_single_phase_weights(pd.read_csv(PACKET), domains)
    two_phase = frame["split"].eq("train") & frame["packet_panel"].eq("augmented_fit_panel")
    qsplit_single = (
        frame["split"].eq("heldout")
        & frame["packet_panel"].eq("single_phase_augmented_panel")
        & frame["policy_family"].eq("single_phase")
        & frame["source_panel"].eq("all")
    )
    selected = frame.loc[(two_phase | qsplit_single) & frame[TARGETS[target]].notna()].reset_index(drop=True)
    phase0 = selected[[f"phase_0_{domain}" for domain in domains]].to_numpy(dtype=float, copy=True)
    phase1 = selected[[f"phase_1_{domain}" for domain in domains]].to_numpy(dtype=float, copy=True)
    phase0 /= phase0.sum(axis=1, keepdims=True)
    phase1 /= phase1.sum(axis=1, keepdims=True)
    dataset = Dataset(
        name=f"300m_{target}",
        frame=selected,
        y=selected[TARGETS[target]].to_numpy(dtype=float),
        weights=np.stack([phase0, phase1], axis=1),
        c0=c0,
        c1=c1,
        domain_names=domains,
        family_index=family_index,
    )
    counts = dataset.frame["policy_family"].value_counts().to_dict()
    if counts != {"two_phase": 280, "single_phase": 240}:
        raise ValueError(f"unexpected 300M policy counts: {counts}")
    if dataset.frame["phase_correspondence_key"].nunique() != 280:
        raise ValueError("expected 280 correspondence groups")
    physical_tied = replay_control.tied_rows(dataset.weights)
    if (int(physical_tied.sum()), int((~physical_tied).sum())) != (282, 238):
        raise ValueError("expected 282 physically tied and 238 asymmetric policies")
    return dataset


def grouped_folds(frame: pd.DataFrame, seed: int, n_splits: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()
    unique = np.unique(groups)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_groups, test_groups in splitter.split(unique):
        training = set(unique[train_groups])
        testing = set(unique[test_groups])
        train = np.flatnonzero(np.fromiter((group in training for group in groups), dtype=bool))
        test = np.flatnonzero(np.fromiter((group in testing for group in groups), dtype=bool))
        folds.append((train, test))
    return tuple(folds)


def local_folds(
    frame: pd.DataFrame,
    seed: int,
    n_splits: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return grouped_folds(frame.reset_index(drop=True), seed, n_splits)


def paired_tied_target(frame: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    """Observed tied baseline for each exact matched asymmetric row."""
    single = frame["policy_family"].eq("single_phase").to_numpy()
    tied_by_key = {
        str(key): float(value)
        for key, value in zip(
            frame.loc[single, "phase_correspondence_key"],
            target[single],
            strict=True,
        )
    }
    return np.asarray(
        [
            tied_by_key.get(str(key), np.nan) if family == "two_phase" else np.nan
            for key, family in zip(
                frame["phase_correspondence_key"],
                frame["policy_family"],
                strict=True,
            )
        ],
        dtype=float,
    )


def phase_parameter_values(
    fitted: replay_control.Fitted,
) -> dict[str, float]:
    """Name phase amplitudes without assuming every candidate has all costs."""
    coefficients = fitted.phase_coefficients
    cursor = 1
    control_energy = 0.0
    phase_information = 0.0
    replay_jensen = 0.0
    aggregate_curvature = 0.0
    reactivation_bregman = 0.0
    if fitted.phase.use_control_energy:
        control_energy = float(coefficients[cursor])
        cursor += 1
    if fitted.phase.use_phase_information:
        phase_information = float(coefficients[cursor])
        cursor += 1
    if fitted.phase.use_replay_jensen:
        replay_jensen = float(coefficients[cursor])
        cursor += 1
    if fitted.phase.use_aggregate_curvature:
        aggregate_curvature = float(coefficients[cursor])
        cursor += 1
    if fitted.phase.use_reactivation_bregman:
        reactivation_bregman = float(coefficients[cursor])
        cursor += 1
    if cursor != len(coefficients):
        raise ValueError(f"unmapped phase coefficients for {fitted.phase.name}")
    return {
        "phase_control": float(coefficients[0]),
        "control_energy": control_energy,
        "phase_information": phase_information,
        "replay_jensen": replay_jensen,
        "aggregate_curvature": aggregate_curvature,
        "reactivation_bregman": reactivation_bregman,
    }


def fit_models(
    weights: np.ndarray,
    target: np.ndarray,
    frame: pd.DataFrame,
    geometry: replay_control.Geometry,
    seed: int,
    paired_target: np.ndarray | None,
    model_names: tuple[str, ...],
    retained_workers: int,
) -> tuple[
    dict[
        str,
        replay_control.Fitted
        | replay_control.AggregateFitted
        | staged_retained.Fitted
        | staged_retained.JointFitted
        | retained.Fitted,
    ],
    list[dict[str, float | str]],
]:
    folds = local_folds(frame, seed, min(INNER_SPLITS, frame["phase_correspondence_key"].nunique()))
    aggregate = replay_control.fit_aggregate(weights, target, geometry, folds)
    models: dict[
        str,
        replay_control.Fitted
        | replay_control.AggregateFitted
        | staged_retained.Fitted
        | staged_retained.JointFitted
        | retained.Fitted,
    ] = {}
    parameters: list[dict[str, float | str]] = []
    if "aggregate_only" in model_names:
        models["aggregate_only"] = aggregate
        parameters.append(
            {
                "model": "aggregate_only",
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": aggregate.ridge,
                "phase_control": 0.0,
                "control_energy": 0.0,
                "phase_information": 0.0,
                "replay_jensen": 0.0,
                "aggregate_curvature": 0.0,
                "reactivation_bregman": 0.0,
            }
        )
    for config in PHASE_CONFIGS:
        if config.name not in model_names:
            continue
        fitted = replay_control.fit_phase(
            aggregate,
            weights,
            target,
            config,
            paired_tied_target=paired_target,
        )
        models[config.name] = fitted
        parameters.append(
            {
                "model": config.name,
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": aggregate.ridge,
                **phase_parameter_values(fitted),
            }
        )
    if STAGED_RETAINED_MODEL in model_names:
        staged = staged_retained.fit(
            aggregate,
            weights,
            target,
            folds,
            paired_tied_target=paired_target,
        )
        models[STAGED_RETAINED_MODEL] = staged
        parameters.append(
            {
                "model": STAGED_RETAINED_MODEL,
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": aggregate.ridge,
                "retention": staged.shape.retention,
                "late_multiplier": staged.shape.late_multiplier,
                "phase_control": float(staged.phase_coefficients[0]),
                "phase_information": float(staged.phase_coefficients[1]),
                "replay_jensen": 0.0,
                "aggregate_curvature": 0.0,
                "reactivation_bregman": 0.0,
            }
        )
    if JOINT_RETAINED_BALANCED_MODEL in model_names:
        joint_balanced = staged_retained.fit_joint(
            aggregate,
            weights,
            target,
            folds,
            use_ordering=True,
            balance_policy_classes=True,
        )
        models[JOINT_RETAINED_BALANCED_MODEL] = joint_balanced
        parameters.append(
            {
                "model": JOINT_RETAINED_BALANCED_MODEL,
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": joint_balanced.ridge,
                "retention": joint_balanced.shape.retention,
                "late_multiplier": joint_balanced.shape.late_multiplier,
            }
        )
    if JOINT_RETAINED_ORDER_MODEL in model_names:
        joint_order = staged_retained.fit_joint(
            aggregate,
            weights,
            target,
            folds,
            use_ordering=True,
        )
        models[JOINT_RETAINED_ORDER_MODEL] = joint_order
        parameters.append(
            {
                "model": JOINT_RETAINED_ORDER_MODEL,
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": joint_order.ridge,
                "retention": joint_order.shape.retention,
                "late_multiplier": joint_order.shape.late_multiplier,
            }
        )
    if JOINT_RETAINED_FAMILY_MODEL in model_names:
        joint = staged_retained.fit_joint(
            aggregate,
            weights,
            target,
            folds,
        )
        models[JOINT_RETAINED_FAMILY_MODEL] = joint
        parameters.append(
            {
                "model": JOINT_RETAINED_FAMILY_MODEL,
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": joint.ridge,
                "retention": joint.shape.retention,
                "late_multiplier": joint.shape.late_multiplier,
            }
        )
    if STAGED_RETAINED_FAMILY_MODEL in model_names:
        staged_family = staged_retained.fit(
            aggregate,
            weights,
            target,
            folds,
            paired_tied_target=paired_target,
            family_resolved=True,
        )
        models[STAGED_RETAINED_FAMILY_MODEL] = staged_family
        parameters.append(
            {
                "model": STAGED_RETAINED_FAMILY_MODEL,
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": aggregate.ridge,
                "retention": staged_family.shape.retention,
                "late_multiplier": staged_family.shape.late_multiplier,
                "phase_control": float(np.linalg.norm(staged_family.phase_coefficients[:-1])),
                "phase_information": float(staged_family.phase_coefficients[-1]),
                "replay_jensen": 0.0,
                "aggregate_curvature": 0.0,
                "reactivation_bregman": 0.0,
            }
        )
    if RETAINED_POWER_LAW_MODEL in model_names:
        retained_power_law = retained.fit(
            weights,
            target,
            geometry,
            folds,
            workers=retained_workers,
        )
        models[RETAINED_POWER_LAW_MODEL] = retained_power_law
        parameters.append(
            {
                "model": RETAINED_POWER_LAW_MODEL,
                "benefit_exponent": retained_power_law.shape.benefit_exponent,
                "benefit_offset": retained_power_law.shape.benefit_offset,
                "damage_exponent": retained_power_law.shape.damage_exponent,
                "ridge": retained_power_law.ridge,
                "retention": retained_power_law.shape.retention,
                "late_multiplier": retained_power_law.shape.late_multiplier,
                "ordering_channel": float(retained_power_law.shape.ordering_channel),
            }
        )
    return models, parameters


def prediction(
    model: (
        replay_control.Fitted
        | replay_control.AggregateFitted
        | staged_retained.Fitted
        | staged_retained.JointFitted
        | retained.Fitted
    ),
    weights: np.ndarray,
) -> np.ndarray:
    return model.predict(weights)


def metric_row(
    dataset: str,
    model: str,
    seed: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> dict[str, float | int | str]:
    residual = predicted - observed
    one = replay_control.tied_rows(weights)
    two = ~one
    fold_regret = []
    for _train, test in folds:
        eligible = test[two[test]]
        if not len(eligible):
            continue
        selected = eligible[int(np.argmin(predicted[eligible]))]
        fold_regret.append(float(observed[selected] - np.min(observed[eligible])))
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "n_rows": len(observed),
        "n_tied": int(one.sum()),
        "n_asymmetric": int(two.sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "one_phase_rmse": float(np.sqrt(np.mean(residual[one] ** 2))),
        "one_phase_spearman": float(spearmanr(observed[one], predicted[one]).statistic),
        "two_phase_rmse": float(np.sqrt(np.mean(residual[two] ** 2))),
        "two_phase_spearman": float(spearmanr(observed[two], predicted[two]).statistic),
        "fold_regret_at_1": float(np.mean(fold_regret)),
        "mean_residual": float(np.mean(residual)),
    }


def paired_metric_row(
    dataset: str,
    model: str,
    seed: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> dict[str, float | int | str]:
    indexed = frame.reset_index().set_index(["phase_correspondence_key", "policy_family"])["index"]
    keys = sorted(
        set(frame.loc[frame["policy_family"].eq("single_phase"), "phase_correspondence_key"])
        & set(frame.loc[frame["policy_family"].eq("two_phase"), "phase_correspondence_key"])
    )
    one = np.asarray([indexed.loc[(key, "single_phase")] for key in keys], dtype=int)
    two = np.asarray([indexed.loc[(key, "two_phase")] for key in keys], dtype=int)
    asymmetric = ~replay_control.tied_rows(weights[two])
    one = one[asymmetric]
    two = two[asymmetric]
    observed_delta = observed[two] - observed[one]
    predicted_delta = predicted[two] - predicted[one]
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "n_pairs": len(one),
        "delta_rmse": float(np.sqrt(np.mean((predicted_delta - observed_delta) ** 2))),
        "delta_spearman": float(spearmanr(observed_delta, predicted_delta).statistic),
        "delta_bias": float(np.mean(predicted_delta - observed_delta)),
        "sign_accuracy": float(np.mean(np.sign(predicted_delta) == np.sign(observed_delta))),
    }


def benchmark_300m(
    output_dir: Path,
    seeds: tuple[int, ...],
    n_splits: int,
    model_names: tuple[str, ...],
    target_names: tuple[str, ...],
    retained_workers: int,
) -> None:
    metric_rows = []
    pair_rows = []
    parameter_rows = []
    prediction_frames = []
    for target in target_names:
        dataset = load_300m(target)
        geometry = geometry_300m(dataset)
        paired = paired_tied_target(dataset.frame, dataset.y)
        for seed in seeds:
            folds = grouped_folds(dataset.frame, seed, n_splits)
            oof = {name: np.full(dataset.n, np.nan) for name in model_names}
            for fold_id, (train, test) in enumerate(folds):
                print(f"300m_{target}: seed={seed} fold={fold_id + 1}/{len(folds)}", flush=True)
                training = dataset.frame.iloc[train].reset_index(drop=True)
                models, parameters = fit_models(
                    dataset.weights[train],
                    dataset.y[train],
                    training,
                    geometry,
                    seed + 10_000 + fold_id,
                    paired[train],
                    model_names,
                    retained_workers,
                )
                for name, fitted in models.items():
                    oof[name][test] = prediction(fitted, dataset.weights[test])
                for row in parameters:
                    parameter_rows.append({"dataset": f"300m_{target}", "seed": seed, "fold": fold_id, **row})
            for name in model_names:
                if not np.all(np.isfinite(oof[name])):
                    raise ValueError(f"{name} did not predict every 300M row")
                metric_rows.append(
                    metric_row(
                        f"300m_{target}",
                        name,
                        seed,
                        dataset.y,
                        oof[name],
                        dataset.weights,
                        folds,
                    )
                )
                pair_rows.append(
                    paired_metric_row(
                        f"300m_{target}",
                        name,
                        seed,
                        dataset.y,
                        oof[name],
                        dataset.frame,
                        dataset.weights,
                    )
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "dataset": f"300m_{target}",
                            "model": name,
                            "seed": seed,
                            "row": np.arange(dataset.n),
                            "run_name": dataset.frame["run_name"].astype(str),
                            "policy_family": dataset.frame["policy_family"].astype(str),
                            "phase_correspondence_key": dataset.frame["phase_correspondence_key"].astype(str),
                            "observed": dataset.y,
                            "predicted": oof[name],
                        }
                    )
                )
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics_300m.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "paired_metrics_300m.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output_dir / "parameters_300m.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions_300m.csv", index=False)


def wsd_folds(
    weights: np.ndarray,
    indices: np.ndarray,
    n_splits: int,
    seed: int,
    protocol: str,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if protocol == "random":
        return tuple(
            (indices[train], indices[test])
            for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(indices)
        )
    coordinates = np.column_stack([weights[indices, 0, 1], weights[indices, 1, 1]])
    blocks = KMeans(n_clusters=n_splits, n_init=10, random_state=seed).fit_predict(coordinates)
    return tuple(
        (np.setdiff1d(indices, indices[blocks == block]), indices[blocks == block]) for block in np.unique(blocks)
    )


def grid_weights(phase0_share: np.ndarray, phase1_share: np.ndarray) -> np.ndarray:
    phase0 = np.column_stack([1.0 - phase0_share, phase0_share])
    phase1 = np.column_stack([1.0 - phase1_share, phase1_share])
    return np.stack([phase0, phase1], axis=1)


def predicted_optimum(predict, resolution: int, phase_0_fraction: float) -> dict[str, float]:
    axis = np.linspace(0.0, 1.0, resolution)
    phase0, phase1 = np.meshgrid(axis, axis, indexing="ij")
    values = predict(grid_weights(phase0.ravel(), phase1.ravel()))
    best = int(np.argmin(values))
    p0 = float(phase0.ravel()[best])
    p1 = float(phase1.ravel()[best])
    return {
        "phase_0": p0,
        "phase_1": p1,
        "aggregate": phase_0_fraction * p0 + (1.0 - phase_0_fraction) * p1,
        "contrast": p1 - p0,
        "prediction": float(values[best]),
    }


def predicted_phase_gain(
    predict,
    aggregate: float,
    resolution: int,
    phase_0_fraction: float,
) -> dict[str, float]:
    phase_1_fraction = 1.0 - phase_0_fraction
    low = max(-aggregate / phase_0_fraction, (aggregate - 1.0) / phase_1_fraction)
    high = min(aggregate / phase_1_fraction, (1.0 - aggregate) / phase_0_fraction)
    contrast = np.unique(np.append(np.linspace(low, high, resolution), 0.0))
    phase0 = aggregate - phase_1_fraction * contrast
    phase1 = aggregate + phase_0_fraction * contrast
    values = predict(grid_weights(phase0, phase1))
    tied = float(predict(grid_weights(np.asarray([aggregate]), np.asarray([aggregate])))[0])
    best = int(np.argmin(values))
    return {
        "phase_gain": tied - float(values[best]),
        "best_contrast": float(contrast[best]),
    }


def two_phase_advantage(predict, resolution: int) -> dict[str, float]:
    axis = np.linspace(0.0, 1.0, resolution)
    phase0, phase1 = np.meshgrid(axis, axis, indexing="ij")
    tied = predict(grid_weights(axis, axis))
    everywhere = predict(grid_weights(phase0.ravel(), phase1.ravel()))
    return {
        "best_tied_bpb": float(np.min(tied)),
        "best_two_phase_bpb": float(np.min(everywhere)),
        "two_phase_gain": float(np.min(tied) - np.min(everywhere)),
    }


def fit_wsd_models(
    panel: wsd80.Panel,
    indices: np.ndarray,
    seed: int,
    protocol: str,
    model_names: tuple[str, ...],
    retained_workers: int,
) -> tuple[
    dict[
        str,
        replay_control.Fitted
        | replay_control.AggregateFitted
        | staged_retained.Fitted
        | staged_retained.JointFitted
        | retained.Fitted,
    ],
    list[dict[str, float | str]],
]:
    local_weights = panel.weights[indices]
    tied = np.flatnonzero(replay_control.tied_rows(local_weights))
    aggregate_splits = min(INNER_SPLITS, len(tied))
    local = wsd_folds(local_weights, tied, aggregate_splits, seed, protocol)
    phase_local = wsd_folds(local_weights, np.arange(len(local_weights)), INNER_SPLITS, seed, protocol)
    geometry = replay_control.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    aggregate = replay_control.fit_aggregate(
        local_weights,
        panel.y[indices],
        geometry,
        local,
    )
    models: dict[
        str,
        replay_control.Fitted
        | replay_control.AggregateFitted
        | staged_retained.Fitted
        | staged_retained.JointFitted
        | retained.Fitted,
    ] = {}
    parameters = []
    if "aggregate_only" in model_names:
        models["aggregate_only"] = aggregate
        parameters.append(
            {
                "model": "aggregate_only",
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": aggregate.ridge,
                "phase_control": 0.0,
                "phase_information": 0.0,
                "replay_jensen": 0.0,
                "aggregate_curvature": 0.0,
                "reactivation_bregman": 0.0,
            }
        )
    for config in PHASE_CONFIGS:
        if config.name not in model_names:
            continue
        fitted = replay_control.fit_phase(
            aggregate,
            panel.weights[indices],
            panel.y[indices],
            config,
        )
        models[config.name] = fitted
        parameters.append(
            {
                "model": config.name,
                "benefit_exponent": aggregate.shape.benefit_exponent,
                "benefit_offset": aggregate.shape.benefit_offset,
                "damage_exponent": aggregate.shape.damage_exponent,
                "ridge": aggregate.ridge,
                **phase_parameter_values(fitted),
            }
        )
    if STAGED_RETAINED_MODEL in model_names:
        staged = staged_retained.fit(
            aggregate,
            panel.weights[indices],
            panel.y[indices],
            phase_local,
        )
        models[STAGED_RETAINED_MODEL] = staged
        parameters.append(
            {
                "model": STAGED_RETAINED_MODEL,
                "retention": staged.shape.retention,
                "late_multiplier": staged.shape.late_multiplier,
                "phase_control": staged.phase_coefficients[0],
                "phase_information": staged.phase_coefficients[1],
                "replay_jensen": 0.0,
                "aggregate_curvature": 0.0,
                "reactivation_bregman": 0.0,
            }
        )
    if JOINT_RETAINED_FAMILY_MODEL in model_names:
        joint = staged_retained.fit_joint(
            aggregate,
            panel.weights[indices],
            panel.y[indices],
            phase_local,
        )
        models[JOINT_RETAINED_FAMILY_MODEL] = joint
        parameters.append(
            {
                "model": JOINT_RETAINED_FAMILY_MODEL,
                "retention": joint.shape.retention,
                "late_multiplier": joint.shape.late_multiplier,
                "ridge": joint.ridge,
            }
        )
    if JOINT_RETAINED_ORDER_MODEL in model_names:
        joint_order = staged_retained.fit_joint(
            aggregate,
            panel.weights[indices],
            panel.y[indices],
            phase_local,
            use_ordering=True,
        )
        models[JOINT_RETAINED_ORDER_MODEL] = joint_order
        parameters.append(
            {
                "model": JOINT_RETAINED_ORDER_MODEL,
                "retention": joint_order.shape.retention,
                "late_multiplier": joint_order.shape.late_multiplier,
                "ridge": joint_order.ridge,
            }
        )
    if JOINT_RETAINED_BALANCED_MODEL in model_names:
        joint_balanced = staged_retained.fit_joint(
            aggregate,
            panel.weights[indices],
            panel.y[indices],
            phase_local,
            use_ordering=True,
            balance_policy_classes=True,
        )
        models[JOINT_RETAINED_BALANCED_MODEL] = joint_balanced
        parameters.append(
            {
                "model": JOINT_RETAINED_BALANCED_MODEL,
                "retention": joint_balanced.shape.retention,
                "late_multiplier": joint_balanced.shape.late_multiplier,
                "ridge": joint_balanced.ridge,
            }
        )
    if STAGED_RETAINED_FAMILY_MODEL in model_names:
        staged_family = staged_retained.fit(
            aggregate,
            panel.weights[indices],
            panel.y[indices],
            phase_local,
            family_resolved=True,
        )
        models[STAGED_RETAINED_FAMILY_MODEL] = staged_family
        parameters.append(
            {
                "model": STAGED_RETAINED_FAMILY_MODEL,
                "retention": staged_family.shape.retention,
                "late_multiplier": staged_family.shape.late_multiplier,
                "phase_control": float(np.linalg.norm(staged_family.phase_coefficients[:-1])),
                "phase_information": float(staged_family.phase_coefficients[-1]),
                "replay_jensen": 0.0,
                "aggregate_curvature": 0.0,
                "reactivation_bregman": 0.0,
            }
        )
    if RETAINED_POWER_LAW_MODEL in model_names:
        retained_power_law = retained.fit(
            local_weights,
            panel.y[indices],
            geometry,
            phase_local,
            workers=retained_workers,
        )
        models[RETAINED_POWER_LAW_MODEL] = retained_power_law
        parameters.append(
            {
                "model": RETAINED_POWER_LAW_MODEL,
                "benefit_exponent": retained_power_law.shape.benefit_exponent,
                "benefit_offset": retained_power_law.shape.benefit_offset,
                "damage_exponent": retained_power_law.shape.damage_exponent,
                "ridge": retained_power_law.ridge,
                "retention": retained_power_law.shape.retention,
                "late_multiplier": retained_power_law.shape.late_multiplier,
                "ordering_channel": float(retained_power_law.shape.ordering_channel),
            }
        )
    return models, parameters


def benchmark_wsd(
    output_dir: Path,
    seeds: tuple[int, ...],
    n_splits: int,
    model_names: tuple[str, ...],
    retained_workers: int,
) -> None:
    panel = wsd80.load_surface()
    sigma = wsd80.training_seed_sigma(wsd80.load_fiber_replicates())
    metric_rows = []
    diagnostic_rows = []
    parameter_rows = []
    for protocol in ("random", "blocked"):
        for seed in seeds:
            outer = wsd_folds(panel.weights, np.arange(len(panel.y)), n_splits, seed, protocol)
            oof = {name: np.full(len(panel.y), np.nan) for name in model_names}
            for fold_id, (train, test) in enumerate(outer):
                print(f"wsd80_{protocol}: seed={seed} fold={fold_id + 1}/{len(outer)}", flush=True)
                models, parameters = fit_wsd_models(
                    panel,
                    train,
                    seed + 10_000 + fold_id,
                    protocol,
                    model_names,
                    retained_workers,
                )
                for name, fitted in models.items():
                    oof[name][test] = prediction(fitted, panel.weights[test])
                for row in parameters:
                    parameter_rows.append({"protocol": protocol, "seed": seed, "fold": fold_id, **row})
            for name in model_names:
                residual = oof[name] - panel.y
                metric_rows.append(
                    {
                        "protocol": protocol,
                        "model": name,
                        "seed": seed,
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "rmse_sigma": float(np.sqrt(np.mean(residual**2)) / sigma),
                        "median_absolute_sigma": float(np.median(np.abs(residual)) / sigma),
                        "spearman": float(spearmanr(panel.y, oof[name]).statistic),
                    }
                )
        models, parameters = fit_wsd_models(
            panel,
            np.arange(len(panel.y)),
            seed=20_000,
            protocol=protocol,
            model_names=model_names,
            retained_workers=retained_workers,
        )
        for row in parameters:
            parameter_rows.append({"protocol": protocol, "seed": -1, "fold": -1, **row})
        tied = np.isclose(panel.weights[:, 0, 1], panel.weights[:, 1, 1])
        for name, fitted in models.items():
            predict = fitted.predict
            optimum = predicted_optimum(predict, OPTIMUM_GRID, wsd80.REALIZED_PHASE_0_FRACTION)
            advantage = two_phase_advantage(predict, OPTIMUM_GRID)
            row: dict[str, float | str] = {
                "protocol": protocol,
                "model": name,
                **{f"optimum_{key}": value for key, value in optimum.items()},
                **{f"advantage_{key}": value for key, value in advantage.items()},
                "tied_rmse": float(np.sqrt(np.mean((predict(panel.weights[tied]) - panel.y[tied]) ** 2))),
                "optimum_distance": float(
                    np.hypot(
                        optimum["phase_0"] - TRUE_OPTIMUM[0],
                        optimum["phase_1"] - TRUE_OPTIMUM[1],
                    )
                ),
            }
            for aggregate in FIBER_AGGREGATES:
                gain = predicted_phase_gain(
                    predict,
                    aggregate,
                    OPTIMUM_GRID,
                    wsd80.REALIZED_PHASE_0_FRACTION,
                )
                row[f"phase_gain_at_{aggregate:.2f}"] = gain["phase_gain"]
                row[f"best_contrast_at_{aggregate:.2f}"] = gain["best_contrast"]
            diagnostic_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(output_dir / "metrics_wsd80.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(output_dir / "diagnostics_wsd80.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output_dir / "parameters_wsd80.csv", index=False)


def summarize(output_dir: Path) -> None:
    summaries = {}
    for name in ("metrics_300m", "paired_metrics_300m", "metrics_wsd80"):
        path = output_dir / f"{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        keys = [column for column in ("dataset", "protocol", "model") if column in frame]
        numeric = [column for column in frame.select_dtypes(include="number") if column not in {"seed", "fold"}]
        summary = frame.groupby(keys, as_index=False)[numeric].mean()
        summary.to_csv(output_dir / f"{name}_summary.csv", index=False)
        summaries[name] = summary.to_dict(orient="records")
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panels", nargs="+", choices=("300m", "wsd80"), default=("300m", "wsd80"))
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS), default=tuple(TARGETS))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument(
        "--retained-workers",
        type=int,
        default=1,
        help="Process workers for the retained-power-law shape grid; this does not change the estimator.",
    )
    args = parser.parse_args()
    if args.retained_workers < 1:
        raise ValueError("--retained-workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)
    model_names = tuple(args.models)
    if "300m" in args.panels:
        benchmark_300m(
            args.output_dir,
            seeds,
            args.splits,
            model_names,
            tuple(args.targets),
            args.retained_workers,
        )
    if "wsd80" in args.panels:
        benchmark_wsd(args.output_dir, seeds, args.splits, model_names, args.retained_workers)
    summarize(args.output_dir)
    print(f"Wrote aggregate-conditioned replay-control results to {args.output_dir}")


if __name__ == "__main__":
    main()
