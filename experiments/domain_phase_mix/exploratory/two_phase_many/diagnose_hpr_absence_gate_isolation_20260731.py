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
"""Isolate HPR's absence-dependent gate under a common late multiplier."""

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
    diagnose_retained_state_response_crossover_20260731 as crossover,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "hpr_absence_gate_isolation_20260731"
PREREGISTRATION = DEFAULT_OUTPUT_DIR / "preregistration.md"
BASELINE_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731"
REPAIRED_RPL_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_20260731"
TARGETS = ("uncheatable", "table9")
VARIANTS = ("hpr_absence_gate", "contrast_gate_native_rate", "contrast_gate_slope_matched")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 812_503
PREDICTION_TOLERANCE = 1e-11
GATE_THRESHOLDS = {"uncheatable": 0.000567, "table9": 0.001159}
PROTOCOL_VERSION = "hpr-absence-gate-isolation-v1"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FoldFit:
    predictions: dict[str, np.ndarray]
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
        Path(crossover.__file__),
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
        raise FileNotFoundError(f"missing frozen gate-isolation inputs: {missing}")
    payload = {
        "version": PROTOCOL_VERSION,
        "parent_protocol": "e30c84f654eb55e9d428eb9ee1afeac69a111d629abe45de6f96eb81db026185",
        "rpl_protocol": "a829181d36a9b3707b307bf802f81966905225304f94e6d6c4dc92ccb5838734",
        "targets": TARGETS,
        "variants": VARIANTS,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "gate_thresholds": GATE_THRESHOLDS,
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
        "gate_effects.csv",
        "fold_metrics.csv",
        "predictions.csv",
        "pair_predictions.csv",
        "diagnostics.csv",
        "reconstruction_checks.csv",
        "decision_gate.csv",
        "report.md",
        "absence_gate_isolation.html",
    )
    return payload.get("protocol_hash") == protocol_hash and all((output_dir / name).exists() for name in required)


def centered_rms_scale(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    reference_centered = reference - reference.mean(axis=0, keepdims=True)
    candidate_centered = candidate - candidate.mean(axis=0, keepdims=True)
    reference_rms = np.sqrt(np.mean(reference_centered**2, axis=0))
    candidate_rms = np.sqrt(np.mean(candidate_centered**2, axis=0))
    if np.any(candidate_rms <= 1e-14):
        bad = np.flatnonzero(candidate_rms <= 1e-14)
        raise ValueError(f"candidate retained columns have negligible centered scale: {bad.tolist()}")
    return reference_rms / candidate_rms


def contrast_gate_share(
    dataset: hpr.family_grp.Dataset,
    geometry: rpl.Geometry,
    retention: float,
    late_multiplier: float,
) -> np.ndarray:
    phase0, phase1 = dataset.weights[:, 0, :], dataset.weights[:, 1, :]
    survival = np.exp(rpl.GATE_CLIP * np.tanh(retention * (phase1 - phase0) / rpl.GATE_CLIP))
    return survival * geometry.phase_0_fraction * phase0 + late_multiplier * geometry.phase_1_fraction * phase1


def retained_block(
    variant: str,
    dataset: hpr.family_grp.Dataset,
    hpr_config: hpr.Config,
    rpl_shape: rpl.Shape,
    geometry: rpl.Geometry,
) -> np.ndarray:
    if variant == "hpr_absence_gate":
        exposure = hpr.retained_exposure(dataset, hpr_config.shape)
    else:
        beta0, epoch_scale = crossover.full_epoch_scale(dataset)
        if variant == "contrast_gate_native_rate":
            retention = rpl_shape.retention
        elif variant == "contrast_gate_slope_matched":
            retention = beta0 * hpr_config.shape.forgetting_rate
        else:
            raise ValueError(f"unknown gate-isolation variant {variant}")
        share = contrast_gate_share(
            dataset,
            geometry,
            retention,
            hpr_config.shape.late_multiplier,
        )
        exposure = epoch_scale[None, :] * share
    loss_feature = -hpr.power_response(exposure, hpr_config.shape.exponent)
    return crossover.retained_loss_block(loss_feature, dataset, hpr_config)


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
    local = baseline.subset_dataset(pooled_dataset, train, f"sur053_{target}_outer{fold_id}")
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

    exact_train = retained_block(
        "hpr_absence_gate",
        train_dataset,
        hpr_config,
        rpl_shape,
        geometry,
    )
    exact_test = retained_block(
        "hpr_absence_gate",
        test_dataset,
        hpr_config,
        rpl_shape,
        geometry,
    )
    if not np.array_equal(exact_train, reference_train):
        mismatch = float(np.max(np.abs(exact_train - reference_train)))
        raise AssertionError(f"{target}/fold{fold_id} HPR train block mismatch {mismatch:.3e}")
    if not np.array_equal(exact_test, reference_test):
        mismatch = float(np.max(np.abs(exact_test - reference_test)))
        raise AssertionError(f"{target}/fold{fold_id} HPR test block mismatch {mismatch:.3e}")

    predictions: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    for variant in VARIANTS:
        candidate_train = retained_block(variant, train_dataset, hpr_config, rpl_shape, geometry)
        candidate_test = retained_block(variant, test_dataset, hpr_config, rpl_shape, geometry)
        scale = np.ones(candidate_train.shape[1], dtype=float)
        if variant != "hpr_absence_gate":
            scale = centered_rms_scale(reference_train, candidate_train)
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
                "active_parameter_count": int(np.sum(np.abs(coefficients) > 1e-10)),
                "condition_number": crossover.condition_number(values_train),
                "scale_min": float(np.min(scale)),
                "scale_median": float(np.median(scale)),
                "scale_max": float(np.max(scale)),
                "hpr_forgetting_rate": hpr_config.shape.forgetting_rate,
                "hpr_late_multiplier": hpr_config.shape.late_multiplier,
                "native_contrast_rate": rpl_shape.retention,
                "slope_matched_contrast_rate": geometry.phase_0_fraction * hpr_config.shape.forgetting_rate,
                "l2": hpr_config.l2,
                "residual_shrink": hpr_config.residual_shrink,
            }
        )
    return FoldFit(predictions, diagnostics)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def bootstrap_effect(
    observed: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for sample in range(samples):
        indices = rng.integers(0, len(observed), size=len(observed))
        draws[sample] = rmse(observed[indices], candidate_prediction[indices]) - rmse(
            observed[indices],
            baseline_prediction[indices],
        )
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.5)),
        float(np.quantile(draws, 0.975)),
        float(np.mean(draws > 0.0)),
    )


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
    list[dict[str, Any]],
]:
    metrics: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    tied_rows, asymmetric_rows, keys = baseline.pair_indices(dataset)
    observed_delta = dataset.y[asymmetric_rows] - dataset.y[tied_rows]
    pair_predictions = {
        variant: prediction[asymmetric_rows] - prediction[tied_rows] for variant, prediction in predictions.items()
    }
    row_fold = np.full(dataset.n, -1, dtype=int)
    for fold_id, (_train, test) in enumerate(folds):
        row_fold[test] = fold_id
    if not np.all(row_fold[tied_rows] == row_fold[asymmetric_rows]):
        raise AssertionError("exact pair split across folds")

    for variant, prediction in predictions.items():
        summary = baseline.metric_summary(dataset, prediction, folds)
        pair_summary, _frame = baseline.pair_summary(dataset, prediction)
        metrics.append({"target": target, "variant": variant, **summary, **pair_summary})
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
                    "predicted_delta": pair_predictions[variant][pair_index],
                    "residual": pair_predictions[variant][pair_index] - observed_delta[pair_index],
                }
            )

    pair_fold = row_fold[tied_rows]
    baseline_pair = pair_predictions["hpr_absence_gate"]
    for candidate in VARIANTS[1:]:
        effect = rmse(observed_delta, pair_predictions[candidate]) - rmse(
            observed_delta,
            baseline_pair,
        )
        low, median, high, probability_positive = bootstrap_effect(
            observed_delta,
            baseline_pair,
            pair_predictions[candidate],
            bootstrap_samples,
            BOOTSTRAP_SEED + TARGETS.index(target) * 100 + VARIANTS.index(candidate),
        )
        effect_rows.append(
            {
                "target": target,
                "candidate": candidate,
                "effect": effect,
                "bootstrap_low": low,
                "bootstrap_median": median,
                "bootstrap_high": high,
                "bootstrap_probability_positive": probability_positive,
            }
        )
        for fold_id in range(len(folds)):
            local = pair_fold == fold_id
            fold_rows.append(
                {
                    "target": target,
                    "outer_fold": fold_id,
                    "candidate": candidate,
                    "effect": (
                        rmse(observed_delta[local], pair_predictions[candidate][local])
                        - rmse(observed_delta[local], baseline_pair[local])
                    ),
                    "n_pairs": int(local.sum()),
                }
            )
    return metrics, prediction_rows, pair_rows, effect_rows, fold_rows


def decision_gate(effects: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    candidate = "contrast_gate_native_rate"
    target_details: dict[str, Any] = {}
    positive_both = True
    magnitude = True
    bootstrap = False
    positive_folds: list[int] = []
    for target in TARGETS:
        row = effects.loc[effects["target"].eq(target) & effects["candidate"].eq(candidate)].iloc[0]
        local_folds = folds.loc[folds["target"].eq(target) & folds["candidate"].eq(candidate)]
        fold_count = int(np.sum(local_folds["effect"] > 0.0))
        positive_folds.append(fold_count)
        positive_both &= float(row["effect"]) > 0.0
        magnitude &= float(row["effect"]) >= GATE_THRESHOLDS[target]
        bootstrap |= float(row["bootstrap_low"]) > 0.0
        target_details[target] = {
            "effect": float(row["effect"]),
            "bootstrap_low": float(row["bootstrap_low"]),
            "bootstrap_high": float(row["bootstrap_high"]),
            "positive_folds": fold_count,
            "threshold": GATE_THRESHOLDS[target],
        }
    fold_stability = max(positive_folds) == baseline.OUTER_SPLITS and min(positive_folds) >= 2
    passes = positive_both and magnitude and bootstrap and fold_stability
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                "positive_on_both_targets": positive_both,
                "magnitude_gate": magnitude,
                "bootstrap_gate": bootstrap,
                "fold_stability_gate": fold_stability,
                "retains_hpr_absence_gate": passes,
                "licenses_new_model": False,
                "target_details": json.dumps(target_details, sort_keys=True),
            }
        ]
    )


def render(metrics: pd.DataFrame, effects: pd.DataFrame, output_dir: Path) -> None:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Exact-pair RMSE", "Gate replacement cost"),
        horizontal_spacing=0.18,
    )
    colors = {
        "hpr_absence_gate": "#1a9850",
        "contrast_gate_native_rate": "#fdae61",
        "contrast_gate_slope_matched": "#d73027",
    }
    for target in TARGETS:
        local = metrics.loc[metrics["target"].eq(target)].set_index("variant").loc[list(VARIANTS)].reset_index()
        figure.add_trace(
            go.Bar(
                name=target,
                x=[f"{target}<br>{variant}" for variant in local["variant"]],
                y=local["delta_rmse"],
                marker_color=[colors[variant] for variant in local["variant"]],
                showlegend=False,
                hovertemplate="%{x}<br>pair RMSE=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    for target in TARGETS:
        local = effects.loc[effects["target"].eq(target)]
        figure.add_trace(
            go.Bar(
                name=target,
                x=local["candidate"],
                y=local["effect"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": local["bootstrap_high"] - local["effect"],
                    "arrayminus": local["effect"] - local["bootstrap_low"],
                },
                hovertemplate="%{x}<br>effect=%{y:+.6f}<extra>%{fullData.name}</extra>",
            ),
            row=1,
            col=2,
        )
    figure.add_hline(y=0.0, line_width=1, line_color="#17324d", row=1, col=2)
    figure.update_yaxes(title_text="BPB RMSE", row=1, col=1)
    figure.update_yaxes(title_text="Positive favors HPR absence gate (BPB)", row=1, col=2)
    figure.update_layout(
        title="HPR absence-gate isolation under a common late multiplier",
        template="plotly_white",
        barmode="group",
        height=720,
        width=1500,
        font={"family": "Avenir Next, sans-serif", "color": "#17324d"},
        margin={"l": 90, "r": 50, "t": 100, "b": 150},
    )
    figure.write_html(
        output_dir / "absence_gate_isolation.html",
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
        "# HPR Absence-Gate Isolation",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Bootstrap samples: {protocol['bootstrap_samples']:,}",
        f"- Maximum HPR OOF reconstruction error: `{reconstruction['absolute_error'].max():.3e}`.",
        "- Positive gate effects favor HPR's absence-dependent gate.",
        "",
        "## Variant Metrics",
        "",
        "| Target | Gate | All RMSE | Asymmetric RMSE | Pair delta RMSE | Pair Spearman | Sign accuracy |",
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
            "## Gate Effects",
            "",
            "| Target | Replacement | Effect | 95% interval | Positive folds | Threshold |",
            "|:--|:--|--:|:--|--:|--:|",
        ]
    )
    for row in effects.itertuples(index=False):
        local = folds.loc[folds["target"].eq(row.target) & folds["candidate"].eq(row.candidate)]
        positive = int(np.sum(local["effect"] > 0.0))
        threshold = GATE_THRESHOLDS[row.target] if row.candidate == "contrast_gate_native_rate" else float("nan")
        threshold_text = f"{threshold:.6f}" if math.isfinite(threshold) else "interpretive"
        lines.append(
            f"| {row.target} | {row.candidate} | {row.effect:+.6f} | "
            f"[{row.bootstrap_low:+.6f}, {row.bootstrap_high:+.6f}] | "
            f"{positive}/{len(local)} | {threshold_text} |"
        )
    decision = bool(gate.iloc[0]["retains_hpr_absence_gate"])
    lines.extend(["", "## Frozen Decision", ""])
    if decision:
        lines.append(
            "The native-rate common-multiplier arm passes the frozen closure gate. "
            "HPR's existing absence gate remains an empirically necessary baseline component, "
            "but this diagnostic licenses no new model."
        )
    else:
        lines.append(
            "The native-rate common-multiplier arm fails the frozen closure gate. "
            "Retained-state component attribution is closed; this diagnostic licenses no new model."
        )
    lines.extend(
        [
            "",
            "The slope-matched arm is interpretive only and cannot alter this decision.",
            "",
            "Artifacts:",
            "",
            "- `variant_metrics.csv`",
            "- `gate_effects.csv`",
            "- `fold_metrics.csv`",
            "- `predictions.csv`",
            "- `pair_predictions.csv`",
            "- `diagnostics.csv`",
            "- `reconstruction_checks.csv`",
            "- `decision_gate.csv`",
            "- `absence_gate_isolation.html`",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload(args.bootstrap_samples)
    write_json(args.output_dir / "protocol.json", protocol)
    if not args.force and completed(args.output_dir, str(protocol["protocol_hash"])):
        print(f"skip complete gate isolation {protocol['protocol_hash']}", flush=True)
        return

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        print(f"isolate HPR absence gate for {target}", flush=True)
        dataset = baseline.expanded.load_300m(target)
        folds = baseline.correspondence_folds(
            dataset.frame,
            baseline.OUTER_SEED,
            baseline.OUTER_SPLITS,
        )
        pooled_dataset = baseline.as_pooled(dataset)
        hpr_records = attribution.fold_map(json.loads((hpr_cell(target) / "fold_selections.json").read_text()))
        rpl_records = attribution.fold_map(json.loads((rpl_cell(target) / "fold_selections.json").read_text()))
        persisted = (
            pd.read_csv(hpr_cell(target) / "predictions.csv").sort_values("row_index")["predicted"].to_numpy(float)
        )
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
                predictions[variant][test] = result.predictions[variant]
            diagnostic_rows.extend(result.diagnostics)

        for variant, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise AssertionError(f"incomplete {target}/{variant} prediction")
        error = np.abs(predictions["hpr_absence_gate"] - persisted)
        if float(np.max(error)) > PREDICTION_TOLERANCE:
            raise AssertionError(f"{target} HPR reconstruction error {float(np.max(error)):.3e}")
        reconstruction_rows.extend(
            {"target": target, "row_index": index, "absolute_error": value} for index, value in enumerate(error)
        )

        local = analyze_target(
            target,
            dataset,
            folds,
            predictions,
            args.bootstrap_samples,
        )
        local_metrics, local_predictions, local_pairs, local_effects, local_folds = local
        metric_rows.extend(local_metrics)
        prediction_rows.extend(local_predictions)
        pair_rows.extend(local_pairs)
        effect_rows.extend(local_effects)
        fold_rows.extend(local_folds)

    metrics = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    pairs = pd.DataFrame(pair_rows)
    effects = pd.DataFrame(effect_rows)
    folds_frame = pd.DataFrame(fold_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    reconstruction = pd.DataFrame(reconstruction_rows)
    gate = decision_gate(effects, folds_frame)

    metrics.to_csv(args.output_dir / "variant_metrics.csv", index=False)
    effects.to_csv(args.output_dir / "gate_effects.csv", index=False)
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
            "retains_hpr_absence_gate": bool(gate.iloc[0]["retains_hpr_absence_gate"]),
            "licenses_new_model": False,
        },
    )
    print(f"complete gate isolation {protocol['protocol_hash']}", flush=True)


if __name__ == "__main__":
    main()
