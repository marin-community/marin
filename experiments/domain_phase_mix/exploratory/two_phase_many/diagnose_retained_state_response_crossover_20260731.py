# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Cross frozen HPR/RPL retained states and response links on expanded 300M data.

This is diagnostic WSD80-SUR-052. It does not select a new model or tune any
configuration. Only HPR's retained-bucket benefit block changes; every other
HPR column, fold, nonlinear selection, constraint, and ridge stays frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hpr,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_hpr_rpl_phase_blocks_20260731 as attribution,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "retained_state_response_crossover_20260731"
PREREGISTRATION = DEFAULT_OUTPUT_DIR / "preregistration.md"
BASELINE_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731"
REPAIRED_RPL_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_20260731"
TARGETS = ("uncheatable", "table9")
VARIANTS = (
    "H_state__H_link",
    "R_state__H_link",
    "H_state__R_link",
    "R_state__R_link",
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 731_520
PREDICTION_TOLERANCE = 1e-11
REVERSE_TOLERANCE = 0.0001
PARENT_GAPS = {"uncheatable": 0.001134, "table9": 0.002318}
PROTOCOL_VERSION = "retained-state-response-crossover-v1"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FoldFit:
    prediction: dict[str, np.ndarray]
    diagnostics: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hpr_cell(target: str) -> Path:
    return BASELINE_DIR / "cells" / target / "hierarchical_phase_replay"


def rpl_cell(target: str) -> Path:
    return REPAIRED_RPL_DIR / "cells" / target / "retained_power_law_repaired"


def protocol_payload(bootstrap_samples: int) -> dict[str, Any]:
    sources = [
        Path(__file__),
        PREREGISTRATION,
        Path(baseline.__file__),
        Path(hpr.__file__),
        Path(attribution.__file__),
        Path(rpl.__file__),
    ]
    for target in TARGETS:
        sources.extend(
            [
                hpr_cell(target) / "fold_selections.json",
                hpr_cell(target) / "predictions.csv",
                hpr_cell(target) / "complete.json",
                rpl_cell(target) / "fold_selections.json",
                rpl_cell(target) / "complete.json",
            ]
        )
    missing = [str(path) for path in sources if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen crossover inputs: {missing}")
    payload = {
        "version": PROTOCOL_VERSION,
        "parent_protocol": "e30c84f654eb55e9d428eb9ee1afeac69a111d629abe45de6f96eb81db026185",
        "rpl_protocol": "a829181d36a9b3707b307bf802f81966905225304f94e6d6c4dc92ccb5838734",
        "targets": TARGETS,
        "variants": VARIANTS,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "prediction_tolerance": PREDICTION_TOLERANCE,
        "reverse_tolerance": REVERSE_TOLERANCE,
        "parent_pair_rmse_gaps": PARENT_GAPS,
        "source_hashes": {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in sources},
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def completed(output_dir: Path, protocol_hash: str) -> bool:
    marker = output_dir / "complete.json"
    if not marker.exists():
        return False
    payload = json.loads(marker.read_text())
    required = (
        "variant_metrics.csv",
        "factor_effects.csv",
        "fold_metrics.csv",
        "predictions.csv",
        "pair_predictions.csv",
        "diagnostics.csv",
        "decision_gate.csv",
        "report.md",
        "state_response_crossover.html",
    )
    return payload.get("protocol_hash") == protocol_hash and all((output_dir / name).exists() for name in required)


def full_epoch_scale(dataset: hpr.family_grp.Dataset) -> tuple[float, np.ndarray]:
    total = np.maximum(dataset.c0 + dataset.c1, 1e-12)
    beta0 = float(np.median(dataset.c0 / total))
    beta1 = 1.0 - beta0
    scale0 = dataset.c0 / beta0
    scale1 = dataset.c1 / beta1
    mismatch = float(np.max(np.abs(scale0 - scale1)))
    relative = float(np.max(np.abs(scale0 - scale1) / np.maximum(scale0, 1e-12)))
    if mismatch > 1e-8 or relative > 1e-10:
        raise ValueError(f"phase epoch scales disagree: absolute={mismatch:.3e}, relative={relative:.3e}")
    return beta0, 0.5 * (scale0 + scale1)


def retained_loss_block(
    loss_feature: np.ndarray,
    dataset: hpr.family_grp.Dataset,
    config: hpr.Config,
) -> np.ndarray:
    groups = hpr.pooling_groups(dataset, config.variant)
    pieces: list[np.ndarray] = []
    singletons = [members[0] for _name, members in groups if len(members) == 1]
    if singletons:
        pieces.append(loss_feature[:, singletons])
    nonsingletons = [(name, members) for name, members in groups if len(members) > 1]
    for _name, members in nonsingletons:
        pieces.append(loss_feature[:, members].sum(axis=1, keepdims=True))
    if nonsingletons:
        residual_members = np.concatenate([members for _name, members in nonsingletons])
        pieces.append(loss_feature[:, residual_members])
    if not pieces:
        raise ValueError("retained hierarchy produced no columns")
    return np.column_stack(pieces)


def state_share(
    state: str,
    dataset: hpr.family_grp.Dataset,
    hpr_config: hpr.Config,
    rpl_shape: rpl.Shape,
    geometry: rpl.Geometry,
) -> tuple[np.ndarray, np.ndarray]:
    _beta0, epoch_scale = full_epoch_scale(dataset)
    if state == "H":
        exposure = hpr.retained_exposure(dataset, hpr_config.shape)
        return exposure / epoch_scale[None, :], epoch_scale
    if state == "R":
        retained = rpl.retained_share(
            dataset.weights,
            geometry,
            rpl_shape.retention,
            rpl_shape.late_multiplier,
        )
        return retained, epoch_scale
    raise ValueError(f"unknown state {state}")


def crossover_block(
    variant: str,
    dataset: hpr.family_grp.Dataset,
    hpr_config: hpr.Config,
    rpl_shape: rpl.Shape,
    geometry: rpl.Geometry,
) -> np.ndarray:
    state_label, link_label = variant.split("__", maxsplit=1)
    state, link = state_label[0], link_label[0]
    if state == "H" and link == "H":
        exposure = hpr.retained_exposure(dataset, hpr_config.shape)
        loss_feature = -hpr.power_response(exposure, hpr_config.shape.exponent)
        return retained_loss_block(loss_feature, dataset, hpr_config)
    retained, epoch_scale = state_share(state, dataset, hpr_config, rpl_shape, geometry)
    if link == "H":
        loss_feature = -((epoch_scale[None, :] * retained) ** hpr_config.shape.exponent)
    elif link == "R":
        loss_feature = (retained + rpl_shape.benefit_offset) ** (-rpl_shape.benefit_exponent)
    else:
        raise ValueError(f"unknown response link {link}")
    return retained_loss_block(loss_feature, dataset, hpr_config)


def rms_scale(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference_rms = np.sqrt(np.mean(reference**2, axis=0))
    candidate_rms = np.sqrt(np.mean(candidate**2, axis=0))
    if np.any(candidate_rms <= 1e-14):
        bad = np.flatnonzero(candidate_rms <= 1e-14)
        raise ValueError(f"candidate retained columns have negligible scale: {bad.tolist()}")
    return reference_rms / candidate_rms


def condition_number(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(centered, axis=0)
    active = scale > 1e-12
    if not np.any(active):
        return float("inf")
    normalized = centered[:, active] / scale[active]
    singular = np.linalg.svd(normalized, compute_uv=False)
    positive = singular[singular > 1e-12]
    if not len(positive):
        return float("inf")
    return float(positive[0] / positive[-1])


def fit_fold(
    target: str,
    fold_id: int,
    train: np.ndarray,
    test: np.ndarray,
    dataset: baseline.expanded.Dataset,
    pooled_dataset: baseline.pooled.Dataset,
    hpr_record: dict[str, Any],
    rpl_record: dict[str, Any],
) -> FoldFit:
    local = baseline.subset_dataset(pooled_dataset, train, f"sur052_{target}_outer{fold_id}")
    train_dataset = baseline.observatory.family_dataset(local)
    test_dataset = attribution.test_hpr_dataset(
        train_dataset,
        dataset.frame.iloc[test],
        dataset.weights[test],
    )
    hpr_config = attribution.hpr_config(hpr_record)
    rpl_shape, _ridge = attribution.rpl_shape(rpl_record)
    geometry = baseline.retained_geometry(local, dataset.family_index)

    train_design = hpr.build_design(train_dataset, hpr_config)
    test_design = hpr.build_design(test_dataset, hpr_config)
    retained_mask = attribution.hpr_blocks(train_design.names)["retained_bucket_benefit"]
    reference_train = train_design.values[:, retained_mask]
    reference_test = test_design.values[:, retained_mask]

    exact_train = crossover_block("H_state__H_link", train_dataset, hpr_config, rpl_shape, geometry)
    exact_test = crossover_block("H_state__H_link", test_dataset, hpr_config, rpl_shape, geometry)
    if not np.allclose(exact_train, reference_train, atol=PREDICTION_TOLERANCE, rtol=PREDICTION_TOLERANCE):
        mismatch = float(np.max(np.abs(exact_train - reference_train)))
        raise AssertionError(f"{target}/fold{fold_id} HPR train block mismatch {mismatch:.3e}")
    if not np.allclose(exact_test, reference_test, atol=PREDICTION_TOLERANCE, rtol=PREDICTION_TOLERANCE):
        mismatch = float(np.max(np.abs(exact_test - reference_test)))
        raise AssertionError(f"{target}/fold{fold_id} HPR test block mismatch {mismatch:.3e}")

    predictions: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    for variant in VARIANTS:
        candidate_train = crossover_block(variant, train_dataset, hpr_config, rpl_shape, geometry)
        candidate_test = crossover_block(variant, test_dataset, hpr_config, rpl_shape, geometry)
        if candidate_train.shape != reference_train.shape or candidate_test.shape != reference_test.shape:
            raise AssertionError(f"{variant} hierarchy does not align with HPR retained block")
        scale = np.ones(candidate_train.shape[1])
        if variant != "H_state__H_link":
            scale = rms_scale(reference_train, candidate_train)
        candidate_train = candidate_train * scale[None, :]
        candidate_test = candidate_test * scale[None, :]

        values_train = train_design.values.copy()
        values_test = test_design.values.copy()
        values_train[:, retained_mask] = candidate_train
        values_test[:, retained_mask] = candidate_test
        intercept, coefficients = attribution.fit_hpr_head(
            values_train,
            train_dataset.target,
            hpr_config.l2,
            train_design.ridge_multipliers,
        )
        predictions[variant] = intercept + values_test @ coefficients
        diagnostics.append(
            {
                "target": target,
                "outer_fold": fold_id,
                "variant": variant,
                "train_rows": len(train),
                "test_rows": len(test),
                "retained_column_count": int(retained_mask.sum()),
                "active_parameter_count": int(np.sum(np.abs(coefficients) > 1e-10)),
                "condition_number": condition_number(values_train),
                "scale_min": float(np.min(scale)),
                "scale_median": float(np.median(scale)),
                "scale_max": float(np.max(scale)),
                "hpr_exponent": hpr_config.shape.exponent,
                "hpr_forgetting_rate": hpr_config.shape.forgetting_rate,
                "hpr_late_multiplier": hpr_config.shape.late_multiplier,
                "rpl_exponent": rpl_shape.benefit_exponent,
                "rpl_offset": rpl_shape.benefit_offset,
                "rpl_retention": rpl_shape.retention,
                "rpl_late_multiplier": rpl_shape.late_multiplier,
                "l2": hpr_config.l2,
                "residual_shrink": hpr_config.residual_shrink,
            }
        )
    return FoldFit(predictions, diagnostics)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def effect_values(observed: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    errors = {variant: rmse(observed, predictions[variant]) for variant in VARIANTS}
    state_h_link = errors["R_state__H_link"] - errors["H_state__H_link"]
    state_r_link = errors["R_state__R_link"] - errors["H_state__R_link"]
    link_h_state = errors["H_state__R_link"] - errors["H_state__H_link"]
    link_r_state = errors["R_state__R_link"] - errors["R_state__H_link"]
    return {
        "state_under_H_link": state_h_link,
        "state_under_R_link": state_r_link,
        "state_effect": 0.5 * (state_h_link + state_r_link),
        "link_under_H_state": link_h_state,
        "link_under_R_state": link_r_state,
        "link_effect": 0.5 * (link_h_state + link_r_state),
        "interaction": state_r_link - state_h_link,
    }


def bootstrap_effects(
    observed: np.ndarray,
    predictions: dict[str, np.ndarray],
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float, float, float]]:
    rng = np.random.default_rng(seed)
    names = tuple(effect_values(observed, predictions))
    draws = {name: np.empty(samples, dtype=float) for name in names}
    for sample in range(samples):
        indices = rng.integers(0, len(observed), size=len(observed))
        values = effect_values(
            observed[indices],
            {variant: prediction[indices] for variant, prediction in predictions.items()},
        )
        for name in names:
            draws[name][sample] = values[name]
    return {
        name: (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.5)),
            float(np.quantile(values, 0.975)),
            float(np.mean(values > 0.0)),
        )
        for name, values in draws.items()
    }


def analyze_target(
    target: str,
    dataset: baseline.expanded.Dataset,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    predictions: dict[str, np.ndarray],
    bootstrap_samples: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    tied_rows, asymmetric_rows, keys = baseline.pair_indices(dataset)
    observed_delta = dataset.y[asymmetric_rows] - dataset.y[tied_rows]
    pair_prediction = {
        variant: prediction[asymmetric_rows] - prediction[tied_rows] for variant, prediction in predictions.items()
    }
    row_fold = np.full(dataset.n, -1, dtype=int)
    for fold_id, (_train, test) in enumerate(folds):
        row_fold[test] = fold_id
    if not np.all(row_fold[tied_rows] == row_fold[asymmetric_rows]):
        raise AssertionError("exact pair split across folds")

    for variant, prediction in predictions.items():
        metrics = baseline.metric_summary(dataset, prediction, folds)
        pair_metrics, _frame = baseline.pair_summary(dataset, prediction)
        metric_rows.append({"target": target, "variant": variant, **metrics, **pair_metrics})
        for row_index in range(dataset.n):
            prediction_rows.append(
                {
                    "target": target,
                    "variant": variant,
                    "row_index": row_index,
                    "outer_fold": int(row_fold[row_index]),
                    "phase_correspondence_key": dataset.frame.iloc[row_index]["phase_correspondence_key"],
                    "policy_family": dataset.frame.iloc[row_index]["policy_family"],
                    "observed": dataset.y[row_index],
                    "predicted": prediction[row_index],
                    "residual": prediction[row_index] - dataset.y[row_index],
                }
            )
        for pair_index, key in enumerate(keys):
            pair_rows.append(
                {
                    "target": target,
                    "variant": variant,
                    "phase_correspondence_key": key,
                    "outer_fold": int(row_fold[tied_rows[pair_index]]),
                    "observed_delta": observed_delta[pair_index],
                    "predicted_delta": pair_prediction[variant][pair_index],
                    "residual": pair_prediction[variant][pair_index] - observed_delta[pair_index],
                }
            )

    effects = effect_values(observed_delta, pair_prediction)
    bootstraps = bootstrap_effects(
        observed_delta,
        pair_prediction,
        bootstrap_samples,
        BOOTSTRAP_SEED + TARGETS.index(target) * 100,
    )
    effect_rows: list[dict[str, Any]] = []
    for name, value in effects.items():
        low, median, high, probability_positive = bootstraps[name]
        effect_rows.append(
            {
                "target": target,
                "effect": name,
                "value": value,
                "bootstrap_low": low,
                "bootstrap_median": median,
                "bootstrap_high": high,
                "bootstrap_probability_positive": probability_positive,
            }
        )

    pair_fold = row_fold[tied_rows]
    for fold_id in range(len(folds)):
        local = pair_fold == fold_id
        local_effects = effect_values(
            observed_delta[local],
            {variant: prediction[local] for variant, prediction in pair_prediction.items()},
        )
        for name, value in local_effects.items():
            fold_rows.append(
                {
                    "target": target,
                    "outer_fold": fold_id,
                    "effect": name,
                    "value": value,
                    "n_pairs": int(local.sum()),
                }
            )
    return metric_rows, prediction_rows, pair_rows, effect_rows + fold_rows


def decision_gate(effects: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    definitions = {
        "state": ("state_effect", "state_under_H_link", "state_under_R_link"),
        "link": ("link_effect", "link_under_H_state", "link_under_R_state"),
    }
    for factor, (main_name, first_name, second_name) in definitions.items():
        target_details: dict[str, Any] = {}
        positive_targets = True
        no_large_reversal = True
        magnitude_gate = True
        bootstrap_gate = False
        positive_fold_counts: list[int] = []
        for target in TARGETS:
            local = effects.loc[effects["target"].eq(target)].set_index("effect")
            main = float(local.loc[main_name, "value"])
            first = float(local.loc[first_name, "value"])
            second = float(local.loc[second_name, "value"])
            low = float(local.loc[main_name, "bootstrap_low"])
            high = float(local.loc[main_name, "bootstrap_high"])
            target_folds = folds.loc[folds["target"].eq(target) & folds["effect"].eq(main_name)]
            positive_folds = int(np.sum(target_folds["value"] > 0.0))
            positive_fold_counts.append(positive_folds)
            positive_targets &= main > 0.0
            no_large_reversal &= min(first, second) >= -REVERSE_TOLERANCE
            magnitude_gate &= main >= 0.5 * PARENT_GAPS[target]
            bootstrap_gate |= low > 0.0
            target_details[target] = {
                "main_effect": main,
                "first_constituent": first,
                "second_constituent": second,
                "bootstrap_low": low,
                "bootstrap_high": high,
                "positive_folds": positive_folds,
                "half_parent_gap": 0.5 * PARENT_GAPS[target],
            }
        fold_gate = max(positive_fold_counts) == baseline.OUTER_SPLITS and min(positive_fold_counts) >= 2
        passes = positive_targets and no_large_reversal and magnitude_gate and bootstrap_gate and fold_gate
        rows.append(
            {
                "factor": factor,
                "positive_on_both_targets": positive_targets,
                "no_constituent_reversal": no_large_reversal,
                "magnitude_gate": magnitude_gate,
                "bootstrap_gate": bootstrap_gate,
                "fold_stability_gate": fold_gate,
                "numeric_gate": passes,
                "wsd80_analogue": True,
                "eligible_to_motivate_route": passes,
                "target_details": json.dumps(target_details, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def render(metrics: pd.DataFrame, effects: pd.DataFrame, output_dir: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Exact-pair RMSE", "Factorial effects"),
        horizontal_spacing=0.18,
    )
    colors = {
        "H_state__H_link": "#1a9850",
        "R_state__H_link": "#91cf60",
        "H_state__R_link": "#fdae61",
        "R_state__R_link": "#d73027",
    }
    for target in TARGETS:
        local = metrics.loc[metrics["target"].eq(target)].set_index("variant").loc[list(VARIANTS)].reset_index()
        figure.add_trace(
            go.Bar(
                name=target,
                x=[f"{target}<br>{variant}" for variant in local["variant"]],
                y=local["delta_rmse"],
                marker_color=[colors[variant] for variant in local["variant"]],
                customdata=np.column_stack([local["all_rmse"], local["asymmetric_rmse"], local["sign_accuracy"]]),
                hovertemplate=(
                    "%{x}<br>pair RMSE=%{y:.6f}<br>all RMSE=%{customdata[0]:.6f}"
                    "<br>asymmetric RMSE=%{customdata[1]:.6f}<br>sign accuracy=%{customdata[2]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    main_effects = effects.loc[effects["effect"].isin(["state_effect", "link_effect"])]
    for target in TARGETS:
        local = main_effects.loc[main_effects["target"].eq(target)]
        figure.add_trace(
            go.Bar(
                name=target,
                x=local["effect"],
                y=local["value"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": local["bootstrap_high"] - local["value"],
                    "arrayminus": local["value"] - local["bootstrap_low"],
                },
                hovertemplate="%{x}<br>effect=%{y:+.6f}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=2,
        )
    figure.add_hline(y=0.0, line_width=1, line_color="#17324d", row=1, col=2)
    figure.update_yaxes(title_text="BPB RMSE", row=1, col=1)
    figure.update_yaxes(title_text="Positive favors HPR component (BPB)", row=1, col=2)
    figure.update_layout(
        title="Frozen retained-state x response-link crossover",
        template="plotly_white",
        barmode="group",
        height=720,
        width=1500,
        font={"family": "Avenir Next, sans-serif", "color": "#17324d"},
        margin={"l": 90, "r": 50, "t": 100, "b": 150},
    )
    figure.write_html(
        output_dir / "state_response_crossover.html",
        include_plotlyjs=True,
        full_html=True,
        config=PLOT_CONFIG,
    )


def write_report(
    metrics: pd.DataFrame,
    effects: pd.DataFrame,
    folds: pd.DataFrame,
    gate: pd.DataFrame,
    reconstruction: pd.DataFrame,
    protocol: dict[str, Any],
    output_dir: Path,
) -> None:
    lines = [
        "# Retained-State and Response-Link Crossover",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Parent baseline: `{protocol['parent_protocol']}`",
        f"- Bootstrap samples: {protocol['bootstrap_samples']:,}",
        "- Positive factorial effects favor the HPR component.",
        "",
        "## Reconstruction",
        "",
        f"Maximum absolute HPR-baseline OOF mismatch: `{reconstruction['absolute_error'].max():.3e}`.",
        "",
        "## Variant Metrics",
        "",
        "| Target | State x link | All RMSE | Asymmetric RMSE | Pair ΔRMSE | Pair Spearman | Sign accuracy |",
        "|:--|:--|--:|--:|--:|--:|--:|",
    ]
    for row in metrics.sort_values(["target", "delta_rmse"]).itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.variant} | {row.all_rmse:.6f} | {row.asymmetric_rmse:.6f} | "
            f"{row.delta_rmse:.6f} | {row.delta_spearman:.3f} | {row.sign_accuracy:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Factorial Effects",
            "",
            "| Target | Effect | Value | 95% interval | Positive folds |",
            "|:--|:--|--:|:--|--:|",
        ]
    )
    for row in effects.itertuples(index=False):
        local = folds.loc[folds["target"].eq(row.target) & folds["effect"].eq(row.effect)]
        positive = int(np.sum(local["value"] > 0.0))
        lines.append(
            f"| {row.target} | {row.effect} | {row.value:+.6f} | "
            f"[{row.bootstrap_low:+.6f}, {row.bootstrap_high:+.6f}] | {positive}/{len(local)} |"
        )
    lines.extend(["", "## Frozen Decision", ""])
    eligible = gate.loc[gate["eligible_to_motivate_route"]]
    if eligible.empty:
        lines.append(
            "No isolated state or response factor passes the preregistered gate. This is a negative "
            "attribution and does not license a component graft."
        )
    else:
        names = ", ".join(f"`{name}`" for name in eligible["factor"])
        lines.append(
            f"The following factors pass the numerical and WSD80-analogue gates: {names}. "
            "This permits a separately preregistered mechanism proposal; it does not promote one."
        )
    lines.extend(
        [
            "",
            "The interaction is diagnostic only. If it dominates, HPR's state and response cannot be "
            "simplified independently from this evidence.",
            "",
            "Artifacts:",
            "",
            "- `variant_metrics.csv`",
            "- `factor_effects.csv`",
            "- `fold_metrics.csv`",
            "- `predictions.csv`",
            "- `pair_predictions.csv`",
            "- `diagnostics.csv`",
            "- `decision_gate.csv`",
            "- `state_response_crossover.html`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    protocol = protocol_payload(args.bootstrap_samples)
    write_json(args.output_dir / "protocol.json", protocol)
    if not args.force and completed(args.output_dir, str(protocol["protocol_hash"])):
        print(f"skip complete crossover {protocol['protocol_hash']}", flush=True)
        return

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    fold_effect_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        print(f"cross retained state and response for {target}", flush=True)
        dataset = baseline.expanded.load_300m(target)
        folds = baseline.correspondence_folds(dataset.frame, baseline.OUTER_SEED, baseline.OUTER_SPLITS)
        pooled_dataset = baseline.as_pooled(dataset)
        hpr_records = attribution.fold_map(json.loads((hpr_cell(target) / "fold_selections.json").read_text()))
        rpl_records = attribution.fold_map(json.loads((rpl_cell(target) / "fold_selections.json").read_text()))
        persisted_frame = pd.read_csv(hpr_cell(target) / "predictions.csv").sort_values("row_index")
        persisted = persisted_frame["predicted"].to_numpy(float)
        predictions = {variant: np.full(dataset.n, np.nan, dtype=float) for variant in VARIANTS}

        for fold_id, (train, test) in enumerate(folds):
            result = fit_fold(
                target,
                fold_id,
                train,
                test,
                dataset,
                pooled_dataset,
                hpr_records[fold_id],
                rpl_records[fold_id],
            )
            for variant in VARIANTS:
                predictions[variant][test] = result.prediction[variant]
            diagnostic_rows.extend(result.diagnostics)

        for variant, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise AssertionError(f"incomplete {target}/{variant} prediction")
        error = np.abs(predictions["H_state__H_link"] - persisted)
        if float(np.max(error)) > PREDICTION_TOLERANCE:
            raise AssertionError(f"{target} HPR reconstruction error {float(np.max(error)):.3e}")
        reconstruction_rows.extend(
            {
                "target": target,
                "row_index": row_index,
                "absolute_error": value,
            }
            for row_index, value in enumerate(error)
        )

        local_metrics, local_predictions, local_pairs, combined_effects = analyze_target(
            target,
            dataset,
            folds,
            predictions,
            args.bootstrap_samples,
        )
        metric_rows.extend(local_metrics)
        prediction_rows.extend(local_predictions)
        pair_rows.extend(local_pairs)
        for row in combined_effects:
            if "bootstrap_low" in row:
                effect_rows.append(row)
            else:
                fold_effect_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    pairs = pd.DataFrame(pair_rows)
    effects = pd.DataFrame(effect_rows)
    folds_frame = pd.DataFrame(fold_effect_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    gate = decision_gate(effects, folds_frame)

    metrics.to_csv(args.output_dir / "variant_metrics.csv", index=False)
    effects.to_csv(args.output_dir / "factor_effects.csv", index=False)
    folds_frame.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions_frame.to_csv(args.output_dir / "predictions.csv", index=False)
    pairs.to_csv(args.output_dir / "pair_predictions.csv", index=False)
    diagnostics.to_csv(args.output_dir / "diagnostics.csv", index=False)
    reconstruction.to_csv(args.output_dir / "reconstruction_checks.csv", index=False)
    gate.to_csv(args.output_dir / "decision_gate.csv", index=False)
    render(metrics, effects, args.output_dir)
    write_report(metrics, effects, folds_frame, gate, reconstruction, protocol, args.output_dir)
    write_json(
        args.output_dir / "complete.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "status": "complete",
            "eligible_factors": gate.loc[gate["eligible_to_motivate_route"], "factor"].tolist(),
        },
    )
    print(f"complete crossover {protocol['protocol_hash']}", flush=True)


if __name__ == "__main__":
    main()
