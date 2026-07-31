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
"""Compare tied-policy aggregate backbones under one phase-order transition.

This benchmark keeps aggregate quality and phase ordering statistically
orthogonal:

* the aggregate model is fit only to phase-tied policies;
* the phase model is fit only to fixed-aggregate treatment-control deltas;
* combined predictions are evaluated on coordinates used by neither fit.

The phase transition operates only on the bounded acquisition state. Literal
repetition penalties remain functions of aggregate exposure and therefore
cancel under a fixed-aggregate contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "tied_backbone_phase_order_20260724"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
PHASE_TRAINING_SERIES = {
    "delphi_3e18_aggressive_phase_asymmetry_20260722",
    "delphi_3e18_frontier_phase_fiber_20260719",
    "delphi_3e18_frontier_random_phase_population_20260720",
    "delphi_3e18_hybrid_phase_ordering_validation_20260720",
}


class AggregatePredictor(Protocol):
    """Predict BPB from a two-column policy."""

    def predict(self, weights: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class DSPAggregatePredictor:
    """Independently fitted phase-tied canonical DSP."""

    model: Any
    dataset: Any

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return observatory.dsp_predict(self.model, self.dataset, weights)


@dataclass(frozen=True)
class PhaseBackbone:
    """Acquisition channels used by the phase transition."""

    name: str
    aggregate_predictor: AggregatePredictor
    phase_fraction: float
    c_total: np.ndarray
    families: orthogonal.FamilyPartition
    rho: np.ndarray
    power: np.ndarray
    value_coef: np.ndarray
    channel_group: np.ndarray
    include_family_channels: bool

    def channel_exposures(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alpha0 = self.phase_fraction
        alpha1 = 1.0 - alpha0
        q0 = weights[:, 0, :] * (alpha0 * self.c_total)[None, :]
        q1 = weights[:, 1, :] * (alpha1 * self.c_total)[None, :]
        if not self.include_family_channels:
            return q0, q1
        family_q0 = orthogonal.family_epochs(alpha0 * weights[:, 0, :], self.c_total, self.families)
        family_q1 = orthogonal.family_epochs(alpha1 * weights[:, 1, :], self.c_total, self.families)
        return np.hstack([q0, family_q0]), np.hstack([q1, family_q1])


@dataclass(frozen=True)
class FittedPhase:
    """A fitted fixed-aggregate phase-order correction."""

    config: orthogonal.PhaseConfig
    params: np.ndarray
    backbone: PhaseBackbone

    def predict_delta(self, weights: np.ndarray) -> np.ndarray:
        return phase_delta(weights, self.backbone, self.config, self.params)


@dataclass(frozen=True)
class AntitheticPairs:
    """Indices for complete fixed-aggregate +d/-d interventions."""

    plus: np.ndarray
    minus: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", choices=("all", *orthogonal.TARGETS), default="all")
    return parser.parse_args()


def physical_backbone(
    target: str,
    single: Any,
    families: orthogonal.FamilyPartition,
) -> PhaseBackbone:
    sweep = pd.read_csv(
        REFERENCE_OUTPUTS / "orthogonal_aggregate_phase_identification_20260724" / f"{target}_aggregate_cv_sweep.csv"
    )
    model = orthogonal.selected_aggregate_model(single, families, sweep)
    channel_count = single.m + len(model.family_coef)
    return PhaseBackbone(
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


def canonical_backbone(
    single: Any,
    families: orthogonal.FamilyPartition,
) -> PhaseBackbone:
    fitted = observatory.dsp_fit(
        single,
        np.arange(single.n),
        model_id="canonical",
        policy_class="single_phase",
    )
    base = fitted.base
    phase_fraction_by_bucket = single.c0 / (single.c0 + single.c1)
    if np.ptp(phase_fraction_by_bucket) > 1e-12:
        raise ValueError("Canonical tied dataset has bucket-dependent phase fractions")
    return PhaseBackbone(
        name="canonical_dsp_tied",
        aggregate_predictor=DSPAggregatePredictor(fitted, single),
        phase_fraction=float(np.mean(phase_fraction_by_bucket)),
        c_total=single.c0 + single.c1,
        families=families,
        rho=np.asarray(base.params["rho"], dtype=float),
        power=np.ones(single.m, dtype=float),
        value_coef=np.asarray(base.benefit_coef, dtype=float),
        channel_group=families.bucket_group,
        include_family_channels=False,
    )


def compact_aggregate_backbone(
    target: str,
    single: Any,
    phase_template: PhaseBackbone,
) -> PhaseBackbone:
    cache_path = orthogonal.OBSERVATORY_CACHE / target / "single_phase" / "compact_retained_state.json"
    tuning = json.loads(cache_path.read_text())["fitDetail"]["tuning"]
    model = observatory.compact_fit(
        single,
        np.arange(single.n),
        l2=float(tuning["l2"]),
        policy_class="single_phase",
    )
    return replace(
        phase_template,
        name="compact_tied_physical_phase",
        aggregate_predictor=model,
    )


def bounded_phase_state(
    q0: np.ndarray,
    q1: np.ndarray,
    backbone: PhaseBackbone,
    retention: np.ndarray,
) -> np.ndarray:
    h0 = (backbone.rho[None, :] * np.maximum(q0, 0.0)) ** backbone.power[None, :]
    h_total = (backbone.rho[None, :] * np.maximum(q0 + q1, 0.0)) ** backbone.power[None, :]
    late_hazard = np.maximum(h_total - h0, 0.0)
    state0 = -np.expm1(-h0)
    retained = retention[backbone.channel_group][None, :] * state0
    return retained + (1.0 - retained) * (-np.expm1(-late_hazard))


def phase_shift_feature(
    weights: np.ndarray,
    backbone: PhaseBackbone,
    shift: orthogonal.PhaseShiftKind,
) -> np.ndarray:
    if shift is orthogonal.PhaseShiftKind.NONE:
        return np.zeros(len(weights), dtype=float)
    if shift is orthogonal.PhaseShiftKind.HELLINGER:
        return np.sum(
            (np.sqrt(np.maximum(weights[:, 0, :], 0.0)) - np.sqrt(np.maximum(weights[:, 1, :], 0.0))) ** 2,
            axis=1,
        )
    if shift is orthogonal.PhaseShiftKind.EPOCH_CONTRAST:
        aggregate = orthogonal.aggregate_weights(weights, backbone.phase_fraction)
        contrast = backbone.phase_fraction * (weights[:, 0, :] - aggregate)
        shifted_epochs = contrast * backbone.c_total[None, :]
        return np.mean(shifted_epochs**2, axis=1)
    raise ValueError(f"Unknown phase shift {shift}")


def simulated_phase_loss(
    weights: np.ndarray,
    backbone: PhaseBackbone,
    config: orthogonal.PhaseConfig,
    params: np.ndarray,
) -> np.ndarray:
    if config.kind is orthogonal.PhaseKind.FOUNDATION_TRANSFER:
        raise ValueError("Foundation transfer is excluded from the corrected benchmark")
    retention, transfer, shift = orthogonal.decode_phase_params(config, params)
    if transfer != 0.0:
        raise AssertionError("Unexpected transfer coefficient")
    q0, q1 = backbone.channel_exposures(weights)
    state = bounded_phase_state(q0, q1, backbone, retention)
    return -state @ backbone.value_coef + shift * phase_shift_feature(weights, backbone, config.shift)


def phase_delta(
    weights: np.ndarray,
    backbone: PhaseBackbone,
    config: orthogonal.PhaseConfig,
    params: np.ndarray,
) -> np.ndarray:
    if config.kind is orthogonal.PhaseKind.NULL and config.shift is orthogonal.PhaseShiftKind.NONE:
        return np.zeros(len(weights), dtype=float)
    aggregate = orthogonal.aggregate_weights(weights, backbone.phase_fraction)
    tied = np.stack([aggregate, aggregate], axis=1)
    return simulated_phase_loss(weights, backbone, config, params) - simulated_phase_loss(
        tied,
        backbone,
        config,
        params,
    )


def fit_phase(
    rows: orthogonal.PhaseRows,
    indices: np.ndarray,
    backbone: PhaseBackbone,
    config: orthogonal.PhaseConfig,
) -> FittedPhase:
    lower, upper = orthogonal.phase_parameter_bounds(config)
    if len(lower) == 0:
        return FittedPhase(config, np.asarray([], dtype=float), backbone)

    def residual(params: np.ndarray) -> np.ndarray:
        prediction = phase_delta(rows.weights[indices], backbone, config, params)
        return np.sqrt(rows.base_weight[indices]) * (prediction - rows.target_delta[indices])

    best_params = orthogonal.phase_starts(config)[0]
    best_cost = float("inf")
    for start in orthogonal.phase_starts(config):
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            loss="huber",
            f_scale=config.huber_scale,
            x_scale="jac",
            max_nfev=1000,
        )
        if np.isfinite(result.cost) and float(result.cost) < best_cost:
            best_cost = float(result.cost)
            best_params = np.asarray(result.x, dtype=float)
    return FittedPhase(config, best_params, backbone)


def antithetic_pair_indices(
    rows: orthogonal.PhaseRows,
    allowed_indices: np.ndarray,
) -> AntitheticPairs:
    allowed = np.zeros(len(rows.frame), dtype=bool)
    allowed[allowed_indices] = True
    frame = rows.frame.copy()
    frame["row_index"] = np.arange(len(frame))
    frame["phase_tv_key"] = frame["phase_tv"].round(12)
    frame = frame[allowed & frame["sign"].isin(("plus", "minus"))].copy()
    key_columns = ["panel", "anchor_key", "direction_id", "seed_block", "phase_tv_key"]
    plus = []
    minus = []
    for _key, group in frame.groupby(key_columns, sort=True, dropna=False):
        if len(group) != 2 or set(group["sign"]) != {"plus", "minus"}:
            continue
        plus.append(int(group.loc[group["sign"].eq("plus"), "row_index"].iloc[0]))
        minus.append(int(group.loc[group["sign"].eq("minus"), "row_index"].iloc[0]))
    if not plus:
        raise ValueError("No complete antithetic pairs in the requested training split")
    return AntitheticPairs(np.asarray(plus, dtype=int), np.asarray(minus, dtype=int))


def fit_phase_odd_even(
    rows: orthogonal.PhaseRows,
    indices: np.ndarray,
    backbone: PhaseBackbone,
    kind: orthogonal.PhaseKind,
    huber_scale: float,
) -> FittedPhase:
    if kind is orthogonal.PhaseKind.FOUNDATION_TRANSFER:
        raise ValueError("Foundation transfer is excluded")
    pairs = antithetic_pair_indices(rows, indices)
    retention_config = orthogonal.PhaseConfig(kind, orthogonal.PhaseShiftKind.NONE, huber_scale)
    lower, upper = orthogonal.phase_parameter_bounds(retention_config)
    observed_odd = 0.5 * (rows.target_delta[pairs.plus] - rows.target_delta[pairs.minus])
    pair_weight = 0.5 * (rows.base_weight[pairs.plus] + rows.base_weight[pairs.minus])

    if len(lower):

        def odd_residual(params: np.ndarray) -> np.ndarray:
            plus_prediction = phase_delta(rows.weights[pairs.plus], backbone, retention_config, params)
            minus_prediction = phase_delta(rows.weights[pairs.minus], backbone, retention_config, params)
            predicted_odd = 0.5 * (plus_prediction - minus_prediction)
            return np.sqrt(pair_weight) * (predicted_odd - observed_odd)

        result = least_squares(
            odd_residual,
            orthogonal.phase_starts(retention_config)[0],
            bounds=(lower, upper),
            loss="huber",
            f_scale=huber_scale,
            x_scale="jac",
            max_nfev=1000,
        )
        retention_params = np.asarray(result.x, dtype=float)
    else:
        retention_params = np.asarray([], dtype=float)

    plus_retention = phase_delta(rows.weights[pairs.plus], backbone, retention_config, retention_params)
    minus_retention = phase_delta(rows.weights[pairs.minus], backbone, retention_config, retention_params)
    observed_even = 0.5 * (rows.target_delta[pairs.plus] + rows.target_delta[pairs.minus])
    retained_even = 0.5 * (plus_retention + minus_retention)
    plus_shift = phase_shift_feature(rows.weights[pairs.plus], backbone, orthogonal.PhaseShiftKind.HELLINGER)
    minus_shift = phase_shift_feature(rows.weights[pairs.minus], backbone, orthogonal.PhaseShiftKind.HELLINGER)
    even_feature = 0.5 * (plus_shift + minus_shift)

    def even_residual(shift: np.ndarray) -> np.ndarray:
        prediction = retained_even + float(shift[0]) * even_feature
        return np.sqrt(pair_weight) * (prediction - observed_even)

    shift_result = least_squares(
        even_residual,
        np.asarray([0.0]),
        bounds=(np.asarray([0.0]), np.asarray([100.0])),
        loss="huber",
        f_scale=huber_scale,
        x_scale="jac",
        max_nfev=1000,
    )
    config = orthogonal.PhaseConfig(kind, orthogonal.PhaseShiftKind.HELLINGER, huber_scale)
    return FittedPhase(
        config,
        np.concatenate([retention_params, np.asarray(shift_result.x, dtype=float)]),
        backbone,
    )


def cross_validate_odd_even(
    rows: orthogonal.PhaseRows,
    backbone: PhaseBackbone,
    target: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    huber_scales = {
        "uncheatable": (0.001, 0.002, 0.004),
        "table9": (0.002, 0.004, 0.008),
    }[target]
    panels = sorted(rows.frame["panel"].unique())
    anchors = sorted(rows.frame["anchor_key"].unique())
    summaries = []
    predictions = {}
    for kind in (
        orthogonal.PhaseKind.NULL,
        orthogonal.PhaseKind.GLOBAL_RETENTION,
        orthogonal.PhaseKind.TWO_GROUP_RETENTION,
    ):
        for huber_scale in huber_scales:
            name = f"odd_even_{kind.value}_hellinger_h{huber_scale:g}"
            panel_prediction = np.full(len(rows.frame), np.nan, dtype=float)
            panel_params = []
            for panel in panels:
                test = np.flatnonzero(rows.frame["panel"].eq(panel).to_numpy())
                train = np.flatnonzero(~rows.frame["panel"].eq(panel).to_numpy())
                model = fit_phase_odd_even(rows, train, backbone, kind, huber_scale)
                panel_prediction[test] = model.predict_delta(rows.weights[test])
                panel_params.append({"panel": panel, "params": model.params.tolist()})
            anchor_prediction = np.full(len(rows.frame), np.nan, dtype=float)
            anchor_params = []
            for anchor in anchors:
                test = np.flatnonzero(rows.frame["anchor_key"].eq(anchor).to_numpy())
                train = np.flatnonzero(~rows.frame["anchor_key"].eq(anchor).to_numpy())
                model = fit_phase_odd_even(rows, train, backbone, kind, huber_scale)
                anchor_prediction[test] = model.predict_delta(rows.weights[test])
                anchor_params.append({"anchor": anchor, "params": model.params.tolist()})
            panel_metrics = orthogonal.phase_metrics(rows.frame, rows.target_delta, panel_prediction)
            anchor_metrics = orthogonal.phase_metrics(rows.frame, rows.target_delta, anchor_prediction)
            summaries.append(
                {
                    "backbone": backbone.name,
                    "model": name,
                    "kind": kind.value,
                    "huber_scale": huber_scale,
                    "parameter_count": len(model.params),
                    "panel_mean_rmse": orthogonal.phase_mean_panel_rmse(rows, panel_prediction),
                    "panel_fold_params_json": json.dumps(panel_params),
                    "anchor_fold_params_json": json.dumps(anchor_params),
                    **panel_metrics,
                    **{f"lao_{key}": value for key, value in anchor_metrics.items()},
                }
            )
            predictions[name] = panel_prediction
    return (
        pd.DataFrame(summaries).sort_values(
            ["panel_mean_rmse", "lao_regret_at_1", "parameter_count", "model"],
            ignore_index=True,
        ),
        predictions,
    )


def cross_validate_phase(
    rows: orthogonal.PhaseRows,
    backbone: PhaseBackbone,
    target: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    panels = sorted(rows.frame["panel"].unique())
    anchors = sorted(rows.frame["anchor_key"].unique())
    summaries: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for config in orthogonal.phase_configs(target):
        panel_prediction = np.full(len(rows.frame), np.nan, dtype=float)
        panel_params = []
        for panel in panels:
            test = np.flatnonzero(rows.frame["panel"].eq(panel).to_numpy())
            train = np.flatnonzero(~rows.frame["panel"].eq(panel).to_numpy())
            model = fit_phase(rows, train, backbone, config)
            panel_prediction[test] = model.predict_delta(rows.weights[test])
            panel_params.append({"panel": panel, "params": model.params.tolist()})
        anchor_prediction = np.full(len(rows.frame), np.nan, dtype=float)
        anchor_params = []
        for anchor in anchors:
            test = np.flatnonzero(rows.frame["anchor_key"].eq(anchor).to_numpy())
            train = np.flatnonzero(~rows.frame["anchor_key"].eq(anchor).to_numpy())
            model = fit_phase(rows, train, backbone, config)
            anchor_prediction[test] = model.predict_delta(rows.weights[test])
            anchor_params.append({"anchor": anchor, "params": model.params.tolist()})
        panel_metrics = orthogonal.phase_metrics(rows.frame, rows.target_delta, panel_prediction)
        anchor_metrics = orthogonal.phase_metrics(rows.frame, rows.target_delta, anchor_prediction)
        summaries.append(
            {
                "backbone": backbone.name,
                "model": config.name,
                "parameter_count": len(orthogonal.phase_parameter_bounds(config)[0]),
                "huber_scale": config.huber_scale,
                "panel_mean_rmse": orthogonal.phase_mean_panel_rmse(rows, panel_prediction),
                "panel_fold_params_json": json.dumps(panel_params),
                "anchor_fold_params_json": json.dumps(anchor_params),
                **panel_metrics,
                **{f"lao_{key}": value for key, value in anchor_metrics.items()},
            }
        )
        predictions[config.name] = panel_prediction
    summary = pd.DataFrame(summaries).sort_values(
        ["panel_mean_rmse", "lao_regret_at_1", "parameter_count", "model"],
        ignore_index=True,
    )
    return summary, predictions


def phase_control_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    panel_specs = (
        (orthogonal.FIBER_PANEL, ("anchor_id", "seed_block")),
        (orthogonal.AGGRESSIVE_PANEL, ("anchor_id", "seed_block")),
        (orthogonal.RANDOM_PANEL, ("anchor_id", "seed_block")),
    )
    for panel_dir, keys in panel_specs:
        manifest = pd.read_csv(panel_dir / "candidate_manifest.csv")
        controls = manifest[manifest["contrast_family"].eq("center_control")].copy()
        control = controls.set_index(list(keys))["candidate_id"].to_dict()
        for row in manifest[~manifest["contrast_family"].eq("center_control")].itertuples(index=False):
            key = tuple(getattr(row, column) for column in keys)
            mapping[str(row.candidate_id)] = str(control[key])
    hybrid = pd.read_csv(orthogonal.HYBRID_PANEL / "candidate_manifest.csv")
    fixed = hybrid[
        hybrid["candidate_kind"].astype(str).str.startswith("fixed_aggregate_")
        | hybrid["candidate_kind"].eq("tied_separate_heads_anchor")
    ].copy()
    keys = ["target", "aggregate_kl_coefficient"]
    controls = fixed[fixed["candidate_kind"].eq("tied_separate_heads_anchor")].copy()
    control = controls.set_index(keys)["candidate_id"].to_dict()
    for row in fixed[fixed["policy_class"].eq("two_phase")].itertuples(index=False):
        key = (row.target, row.aggregate_kl_coefficient)
        mapping[str(row.candidate_id)] = str(control[key])
    return mapping


def observatory_phase_predictions(
    target: str,
    reference: Any,
    heldout_frame: pd.DataFrame,
    rows: orthogonal.PhaseRows,
) -> dict[str, np.ndarray]:
    candidate_position = {
        str(candidate_id): reference.n + index
        for index, candidate_id in enumerate(heldout_frame["candidate_id"])
        if pd.notna(candidate_id)
    }
    run_base_position = {
        str(run_base): reference.n + index
        for index, run_base in enumerate(heldout_frame["wandb_run_base"])
        if pd.notna(run_base)
    }
    fiber = pd.read_csv(orthogonal.FIBER_RESULTS)
    fiber_run_base = {
        str(row.candidate_id): str(row.training_wandb_name).rsplit("-", 1)[0] for row in fiber.itertuples(index=False)
    }

    def position(candidate_id: str) -> int:
        if candidate_id in candidate_position:
            return candidate_position[candidate_id]
        return run_base_position[fiber_run_base[candidate_id]]

    controls = phase_control_map()
    treatment_positions = np.asarray([position(str(value)) for value in rows.frame["candidate_id"]], dtype=int)
    control_positions = np.asarray(
        [position(controls[str(value)]) for value in rows.frame["candidate_id"]],
        dtype=int,
    )
    predictions: dict[str, np.ndarray] = {}
    for model_id in orthogonal.PARETO_BASELINE_MODELS:
        cache_path = orthogonal.OBSERVATORY_CACHE / target / "two_phase" / f"{model_id}.json"
        if not cache_path.exists():
            continue
        payload = json.loads(cache_path.read_text())
        prediction = np.asarray(payload["prediction"], dtype=float)
        predictions[f"observatory_{model_id}"] = prediction[treatment_positions] - prediction[control_positions]
    return predictions


def coordinate_disjoint_combined_rows(
    target: str,
    reference: Any,
    single: Any,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    training_keys = {orthogonal.coordinate_key(weights) for weights in single.weights}
    reference_mask = np.asarray(
        [orthogonal.coordinate_key(weights) not in training_keys for weights in reference.weights],
        dtype=bool,
    )
    reference_frame = pd.DataFrame(
        {
            "source": "original_two_phase_swarm",
            "candidate_id": reference.frame["run_name"].astype(str),
            "observed": reference.y,
            "position": np.arange(reference.n),
        }
    )
    archive_mask = (
        heldout_frame["fit_panel_overlap"].eq("coordinate_disjoint").to_numpy()
        & heldout_frame["policy_class"].eq("two_phase").to_numpy()
        & np.isfinite(heldout_frame[orthogonal.TARGET_COLUMNS[target]].to_numpy(dtype=float))
        & ~heldout_frame["training_series"].isin(PHASE_TRAINING_SERIES).to_numpy()
        & np.asarray(
            [orthogonal.coordinate_key(weights) not in training_keys for weights in heldout_weights],
            dtype=bool,
        )
    )
    archive_indices = np.flatnonzero(archive_mask)
    archive_frame = pd.DataFrame(
        {
            "source": "append_only_archive",
            "candidate_id": heldout_frame.iloc[archive_indices]["candidate_id"].astype(str).to_numpy(),
            "observed": heldout_frame.iloc[archive_indices][orthogonal.TARGET_COLUMNS[target]].to_numpy(dtype=float),
            "position": reference.n + archive_indices,
        }
    )
    frame = pd.concat(
        [
            reference_frame.loc[reference_mask],
            archive_frame,
        ],
        ignore_index=True,
    )
    weights = np.concatenate([reference.weights[reference_mask], heldout_weights[archive_indices]], axis=0)
    observed = frame["observed"].to_numpy(dtype=float)
    positions = frame["position"].to_numpy(dtype=int)
    return frame, weights, observed, positions


def combined_metrics(
    target: str,
    reference: Any,
    single: Any,
    heldout_frame: pd.DataFrame,
    heldout_weights: np.ndarray,
    backbone: PhaseBackbone,
    phase_model: FittedPhase,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame, weights, observed, positions = coordinate_disjoint_combined_rows(
        target,
        reference,
        single,
        heldout_frame,
        heldout_weights,
    )
    aggregate_prediction = backbone.aggregate_predictor.predict(weights)
    combined_prediction = aggregate_prediction + phase_model.predict_delta(weights)
    records = []
    for name, prediction in (
        (f"{backbone.name}_aggregate_only", aggregate_prediction),
        (f"{backbone.name}_plus_phase", combined_prediction),
    ):
        for source, indices in frame.groupby("source", sort=True).indices.items():
            group = np.asarray(indices, dtype=int)
            records.append(
                {
                    "model": name,
                    "scope": source,
                    **orthogonal.regression_metrics(observed[group], prediction[group]),
                }
            )
        records.append({"model": name, "scope": "all", **orthogonal.regression_metrics(observed, prediction)})
    for model_id in orthogonal.PARETO_BASELINE_MODELS:
        cache_path = orthogonal.OBSERVATORY_CACHE / target / "two_phase" / f"{model_id}.json"
        if not cache_path.exists():
            continue
        payload = json.loads(cache_path.read_text())
        prediction = np.asarray(payload["prediction"], dtype=float)[positions]
        for source, indices in frame.groupby("source", sort=True).indices.items():
            group = np.asarray(indices, dtype=int)
            records.append(
                {
                    "model": f"observatory_{model_id}",
                    "scope": source,
                    **orthogonal.regression_metrics(observed[group], prediction[group]),
                }
            )
        records.append(
            {
                "model": f"observatory_{model_id}",
                "scope": "all",
                **orthogonal.regression_metrics(observed, prediction),
            }
        )
    predictions = frame.copy()
    predictions["aggregate_prediction"] = aggregate_prediction
    predictions["phase_delta"] = phase_model.predict_delta(weights)
    predictions["combined_prediction"] = combined_prediction
    predictions["residual"] = combined_prediction - observed
    return pd.DataFrame(records).drop_duplicates(ignore_index=True), predictions


def plot_combined_calibration(frame: pd.DataFrame, path: Path, title: str) -> None:
    figure = px.scatter(
        frame,
        x="observed",
        y="combined_prediction",
        color="source",
        hover_name="candidate_id",
        hover_data=["phase_delta", "residual"],
        title=title,
        color_discrete_sequence=["#e77836", "#1f736d"],
    )
    lower = min(float(frame["observed"].min()), float(frame["combined_prediction"].min()))
    upper = max(float(frame["observed"].max()), float(frame["combined_prediction"].max()))
    figure.add_shape(type="line", x0=lower, x1=upper, y0=lower, y1=upper, line={"dash": "dash", "color": "#6f7d85"})
    figure.update_layout(template="plotly_white", width=1100, height=760)
    figure.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def run_target(target: str, output_dir: Path) -> None:
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, _single_evaluation_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    families = orthogonal.family_partition(single.domain_names)
    physical = physical_backbone(target, single, families)
    backbones = (
        physical,
        canonical_backbone(single, families),
        compact_aggregate_backbone(target, single, physical),
    )
    rows = orthogonal.load_phase_rows(target, single.domain_names, backbones[0].phase_fraction)
    all_phase_records = []
    all_component_records = []
    all_combined_records = []
    all_combined_sweep_records = []
    all_phase_predictions: dict[str, np.ndarray] = observatory_phase_predictions(
        target,
        reference,
        heldout_frame,
        rows,
    )
    for backbone in backbones:
        phase_summary, predictions = cross_validate_phase(rows, backbone, target)
        phase_summary.to_csv(output_dir / f"{target}_{backbone.name}_phase_cv.csv", index=False)
        odd_even_summary, odd_even_predictions = cross_validate_odd_even(rows, backbone, target)
        odd_even_summary.to_csv(
            output_dir / f"{target}_{backbone.name}_odd_even_phase_cv.csv",
            index=False,
        )
        selected = phase_summary.iloc[0]
        selected_config = next(config for config in orthogonal.phase_configs(target) if config.name == selected["model"])
        full_phase = fit_phase(rows, np.arange(len(rows.frame)), backbone, selected_config)
        keyed_predictions = {f"{backbone.name}:{name}": value for name, value in predictions.items()}
        all_phase_predictions.update(keyed_predictions)
        all_phase_predictions.update({f"{backbone.name}:{name}": value for name, value in odd_even_predictions.items()})
        selected_prediction = predictions[str(selected["model"])]
        all_phase_records.append(
            orthogonal.phase_panel_metrics(
                rows,
                {
                    f"{backbone.name}:{selected['model']}": selected_prediction,
                },
            )
        )
        component_metrics, _pairs = orthogonal.antithetic_component_metrics(
            rows,
            {
                f"{backbone.name}:{selected['model']}": selected_prediction,
            },
        )
        all_component_records.append(component_metrics)
        for config in orthogonal.phase_configs(target):
            candidate_phase = fit_phase(rows, np.arange(len(rows.frame)), backbone, config)
            candidate_metrics, _candidate_predictions = combined_metrics(
                target,
                reference,
                single,
                heldout_frame,
                heldout_weights,
                backbone,
                candidate_phase,
            )
            candidate_metrics = candidate_metrics[candidate_metrics["model"].eq(f"{backbone.name}_plus_phase")].copy()
            candidate_metrics.insert(0, "phase_model", config.name)
            candidate_metrics.insert(0, "backbone", backbone.name)
            all_combined_sweep_records.append(candidate_metrics)
        for row in odd_even_summary.itertuples(index=False):
            candidate_phase = fit_phase_odd_even(
                rows,
                np.arange(len(rows.frame)),
                backbone,
                orthogonal.PhaseKind(row.kind),
                float(row.huber_scale),
            )
            candidate_metrics, _candidate_predictions = combined_metrics(
                target,
                reference,
                single,
                heldout_frame,
                heldout_weights,
                backbone,
                candidate_phase,
            )
            candidate_metrics = candidate_metrics[candidate_metrics["model"].eq(f"{backbone.name}_plus_phase")].copy()
            candidate_metrics.insert(0, "phase_model", str(row.model))
            candidate_metrics.insert(0, "backbone", backbone.name)
            all_combined_sweep_records.append(candidate_metrics)
        metrics, prediction_frame = combined_metrics(
            target,
            reference,
            single,
            heldout_frame,
            heldout_weights,
            backbone,
            full_phase,
        )
        metrics.insert(0, "target", target)
        all_combined_records.append(metrics)
        prediction_frame.to_csv(
            output_dir / f"{target}_{backbone.name}_combined_predictions.csv",
            index=False,
        )
        plot_combined_calibration(
            prediction_frame,
            output_dir / f"{target}_{backbone.name}_combined_calibration.html",
            f"{target}: {backbone.name} aggregate plus fixed-aggregate phase correction",
        )
    phase_baseline_metrics = orthogonal.phase_panel_metrics(rows, all_phase_predictions)
    phase_baseline_metrics.to_csv(output_dir / f"{target}_phase_model_comparison.csv", index=False)
    pd.concat(all_phase_records, ignore_index=True).to_csv(
        output_dir / f"{target}_selected_phase_panel_metrics.csv",
        index=False,
    )
    pd.concat(all_component_records, ignore_index=True).to_csv(
        output_dir / f"{target}_selected_antithetic_metrics.csv",
        index=False,
    )
    pd.concat(all_combined_records, ignore_index=True).drop_duplicates(ignore_index=True).to_csv(
        output_dir / f"{target}_combined_heldout_metrics.csv",
        index=False,
    )
    pd.concat(all_combined_sweep_records, ignore_index=True).to_csv(
        output_dir / f"{target}_combined_phase_config_sweep.csv",
        index=False,
    )


def write_report(output_dir: Path, targets: tuple[str, ...]) -> None:
    lines = [
        "# Tied aggregate backbone plus orthogonal phase order",
        "",
        (
            "The aggregate spine is fit only to the 280 phase-tied policies. The phase correction is fit only to "
            "fixed-aggregate treatment-control deltas. Combined evaluation excludes both training sources."
        ),
        "",
    ]
    for target in targets:
        metrics = pd.read_csv(output_dir / f"{target}_combined_heldout_metrics.csv")
        phase = pd.read_csv(output_dir / f"{target}_phase_model_comparison.csv")
        lines.extend(
            [
                f"## {target}",
                "",
                "### Coordinate-disjoint combined prediction",
                "",
                metrics[metrics["scope"].eq("all")].sort_values("rmse").to_markdown(index=False),
                "",
                "### Fixed-aggregate phase prediction",
                "",
                phase[phase["scope"].eq("all")].sort_values("rmse").to_markdown(index=False),
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = orthogonal.TARGETS if args.target == "all" else (args.target,)
    for target in targets:
        run_target(target, args.output_dir)
    write_report(args.output_dir, targets)


if __name__ == "__main__":
    main()
