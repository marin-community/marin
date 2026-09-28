# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Falsify a bounded Fisher-Rao demand-overlap aggregate law at 300M.

The model treats a tied training allocation ``w`` and a target-demand
distribution ``q`` as points on the probability simplex. Their Bhattacharyya
overlap is a bounded produced-capability state:

    B(w, q) = sum_i sqrt(q_i w_i)
    A(w) = b + M [1 - B(w, q)] + h R(w)

where ``R`` is exact repeated materialized mass. Writing
``a_i = M sqrt(q_i)`` makes policy differences linear in nonnegative
coefficients on ``-sqrt(w_i)``. The fitted coefficients uniquely recover
``M = ||a||_2`` and ``q_i = a_i^2 / ||a||_2^2``.

Only pair-odd effects of the 39 physically tied antithetic proportional log
tilts select the target-specific ridge and demand direction. Pair-even effects
remain unused until the frozen falsification. No tied endpoint or deletion
outcome is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_intervention_identified_component_transfer_20260731 as transfer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_unique_evidence_demand_allocation_20260731 as sur065,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    swarm39_harness_20260725 as swarm39,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "fisher_rao_demand_overlap_20260731"

CANDIDATE_ID = "WSD80-SUR-066"
PROTOCOL_VERSION = "fisher-rao-demand-overlap-parity-v1"
TARGETS = ("uncheatable", "table9")
RIDGE_GRID = (0.01, 0.10, 1.0, 10.0, 100.0)
CV_SPLITS = 5
CV_SEED = 7_316_601
BOOTSTRAP_DRAWS = 4_000
COEFFICIENT_BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 7_316_602
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

GATES = {
    "odd_design_nullity_max": 0,
    "odd_cv_rmse_improvement_min": 0.20,
    "even_rmse_improvement_min": 0.05,
    "even_rmse_improvement_ci_low_min": 0.0,
    "even_spearman_min": 0.25,
    "even_sign_accuracy_min": 0.60,
    "demand_bootstrap_cosine_median_min": 0.80,
}


@dataclass(frozen=True)
class FeatureBundle:
    """Overlap features and exact antithetic contrasts."""

    plus: np.ndarray
    minus: np.ndarray
    anchor: np.ndarray
    odd: np.ndarray
    even: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "evaluate"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def protocol_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": CANDIDATE_ID,
        "version": PROTOCOL_VERSION,
        "purpose": "Parity falsification of a bounded Fisher-Rao demand-overlap aggregate law",
        "mechanism": {
            "overlap": "B(w,q)=sum_i sqrt(q_i w_i)",
            "response": "A(w)=b+M[1-B(w,q)]+h R(w)",
            "replay": "R(w)=sum_i p_i max(E_i(w)-1,0)",
            "linear_head": "a_i=M sqrt(q_i)>=0 on -sqrt(w_i), h>=0 on exact repeated mass",
            "identified_parameters": "M=||a||_2; q_i=a_i^2/||a||_2^2",
        },
        "selection_data": "39 pair-odd effects from central antithetic proportional log tilts",
        "falsification_data": "39 pair-even effects relative to 11 proportional controls",
        "excluded_outcomes": "all tied endpoints and all domain deletions",
        "important_non_evidence": "In-sample odd reconstruction is not evidence; no even outcome selects ridge",
        "selection": {
            "target_specific_ridge_grid": RIDGE_GRID,
            "criterion": "pair-odd cross-validation RMSE",
            "cv_splits": CV_SPLITS,
            "cv_seed": CV_SEED,
        },
        "bootstrap": {
            "paired_metric_draws": BOOTSTRAP_DRAWS,
            "coefficient_draws": COEFFICIENT_BOOTSTRAP_DRAWS,
            "unit": "antithetic direction",
            "seed": BOOTSTRAP_SEED,
        },
        "gates": GATES,
        "decision": (
            "The odd-only design must first have zero coefficient nullity. If it does, both targets must pass "
            "every outcome gate. Passing licenses tied-panel nested CV and raw optimization; failure blocks the "
            "route before endpoint fitting."
        ),
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in (
                Path(__file__),
                Path(sur065.__file__),
                transfer.MANIFEST_PATH,
                transfer.UNCHEATABLE_INTERVENTIONS_PATH,
                transfer.UNCHEATABLE_CONTROLS_PATH,
                transfer.table9.OLMO_FULL_WIDE,
                Path(swarm39.__file__),
            )
        },
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    payload["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def preflight_payload() -> dict[str, Any]:
    manifest, domains, c0, c1, proportional = sur065.geometry()
    plus, minus = sur065.paired_rows(manifest, domains)
    columns = [f"phase_0_{domain}" for domain in domains]
    plus_weights = plus[columns].to_numpy(float)
    minus_weights = minus[columns].to_numpy(float)
    features = feature_bundle(manifest, domains, c0, c1, proportional)
    feature_scale = np.maximum(np.sqrt(np.mean(features.odd**2, axis=0)), 1e-12)
    scaled_odd = features.odd / feature_scale[None, :]
    singular_values = np.linalg.svd(scaled_odd, compute_uv=False)
    numerical_rank = int(np.linalg.matrix_rank(scaled_odd))
    return {
        "directions": len(domains),
        "domains": len(domains),
        "physically_tied": True,
        "plus_weight_sum_error_max": float(np.max(np.abs(plus_weights.sum(axis=1) - 1.0))),
        "minus_weight_sum_error_max": float(np.max(np.abs(minus_weights.sum(axis=1) - 1.0))),
        "feature_columns": features.odd.shape[1],
        "odd_design_rank": numerical_rank,
        "odd_design_nullity": int(features.odd.shape[1] - numerical_rank),
        "odd_design_condition_number_nonzero": float(singular_values[0] / singular_values[numerical_rank - 1]),
        "all_features_finite": bool(np.all(np.isfinite(features.plus)) and np.all(np.isfinite(features.minus))),
        "tied_endpoint_outcomes_used": False,
        "deletion_outcomes_used": False,
    }


def freeze_protocol(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    path = output_dir / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != json_ready(payload):
            raise ValueError(f"Frozen protocol differs from current source: {path}")
    else:
        path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")
    preflight = preflight_payload()
    (output_dir / "preflight.json").write_text(json.dumps(json_ready(preflight), indent=2, sort_keys=True) + "\n")
    design_passed = preflight["odd_design_nullity"] <= GATES["odd_design_nullity_max"]
    design_decision = {
        "candidate_id": CANDIDATE_ID,
        "passed": design_passed,
        "decision": (
            "PASS: outcome evaluation licensed" if design_passed else "BLOCK: odd-only mechanism is underidentified"
        ),
        "measured_nullity": preflight["odd_design_nullity"],
        "maximum_nullity": GATES["odd_design_nullity_max"],
    }
    (output_dir / "design_decision.json").write_text(
        json.dumps(json_ready(design_decision), indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            json_ready({"protocol": payload, "preflight": preflight, "design_decision": design_decision}),
            indent=2,
            sort_keys=True,
        )
    )


def verify_protocol(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Freeze the protocol before evaluation: {path}")
    frozen = json.loads(path.read_text())
    current = json_ready(protocol_payload())
    if frozen != current:
        raise ValueError("Current source or data differs from the frozen protocol")
    decision_path = output_dir / "design_decision.json"
    if not decision_path.exists() or not json.loads(decision_path.read_text())["passed"]:
        raise ValueError("Odd-only mechanism failed its structural identification precondition")
    return frozen


def overlap_features(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    proportional: np.ndarray,
) -> np.ndarray:
    if np.any(weights < -1e-12) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Invalid mixture allocation")
    clipped = np.maximum(weights, 0.0)
    epochs = clipped * (c0 + c1)[None, :]
    repeated_mass = (proportional[None, :] * np.maximum(epochs - 1.0, 0.0)).sum(axis=1)
    return np.column_stack([-np.sqrt(clipped), repeated_mass])


def feature_bundle(
    manifest: pd.DataFrame,
    domains: list[str],
    c0: np.ndarray,
    c1: np.ndarray,
    proportional: np.ndarray,
) -> FeatureBundle:
    plus, minus = sur065.paired_rows(manifest, domains)
    columns = [f"phase_0_{domain}" for domain in domains]
    plus_x = overlap_features(plus[columns].to_numpy(float), c0, c1, proportional)
    minus_x = overlap_features(minus[columns].to_numpy(float), c0, c1, proportional)
    anchor_x = overlap_features(proportional[None, :], c0, c1, proportional)[0]
    return FeatureBundle(
        plus=plus_x,
        minus=minus_x,
        anchor=anchor_x,
        odd=0.5 * (plus_x - minus_x),
        even=0.5 * (plus_x + minus_x) - anchor_x[None, :],
    )


def cv_predictions(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    predictions = np.empty_like(target)
    splitter = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    for train, test in splitter.split(design):
        coefficients = sur065.fit_nonnegative_ridge(design[train], target[train], ridge)
        predictions[test] = design[test] @ coefficients
    return predictions


def select_ridges(
    bundle: FeatureBundle,
    effects: dict[str, sur065.TargetEffects],
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float | str | bool]] = []
    selected: dict[str, float] = {}
    for target in TARGETS:
        observed = effects[target].odd
        null_rmse = float(np.sqrt(np.mean(observed**2)))
        target_rows: list[dict[str, float | str | bool]] = []
        for ridge in RIDGE_GRID:
            predicted = cv_predictions(bundle.odd, observed, ridge)
            rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
            target_rows.append(
                {
                    "target": target,
                    "ridge": ridge,
                    "odd_cv_rmse": rmse,
                    "odd_null_rmse": null_rmse,
                    "normalized_rmse": rmse / null_rmse,
                    "selected": False,
                }
            )
        best = min(target_rows, key=lambda row: (float(row["normalized_rmse"]), float(row["ridge"])))
        best["selected"] = True
        selected[target] = float(best["ridge"])
        rows.extend(target_rows)
    return selected, pd.DataFrame(rows)


def demand_bootstrap(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    full_coefficients: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    amplitudes = full_coefficients[:-1]
    if float(np.linalg.norm(amplitudes)) <= 1e-12:
        return {
            "demand_bootstrap_cosine_median": 0.0,
            "demand_bootstrap_cosine_ci_low": 0.0,
            "demand_bootstrap_cosine_ci_high": 0.0,
        }
    full_demand = amplitudes**2
    full_demand /= max(float(full_demand.sum()), 1e-12)
    cosines = np.empty(COEFFICIENT_BOOTSTRAP_DRAWS)
    count = len(target)
    for draw in range(COEFFICIENT_BOOTSTRAP_DRAWS):
        indices = rng.integers(0, count, count)
        coefficients = sur065.fit_nonnegative_ridge(design[indices], target[indices], ridge)
        demand = coefficients[:-1] ** 2
        demand /= max(float(demand.sum()), 1e-12)
        denominator = max(float(np.linalg.norm(demand) * np.linalg.norm(full_demand)), 1e-12)
        cosines[draw] = float(np.dot(demand, full_demand) / denominator)
    return {
        "demand_bootstrap_cosine_median": float(np.median(cosines)),
        "demand_bootstrap_cosine_ci_low": float(np.quantile(cosines, 0.025)),
        "demand_bootstrap_cosine_ci_high": float(np.quantile(cosines, 0.975)),
    }


def target_decision(row: pd.Series) -> dict[str, Any]:
    checks = {
        "odd_cv_improvement": bool(row["odd_cv_rmse_improvement"] >= GATES["odd_cv_rmse_improvement_min"]),
        "even_improvement": bool(row["even_rmse_improvement"] >= GATES["even_rmse_improvement_min"]),
        "even_improvement_ci": bool(row["even_rmse_improvement_ci_low"] >= GATES["even_rmse_improvement_ci_low_min"]),
        "even_spearman": bool(row["even_spearman"] >= GATES["even_spearman_min"]),
        "even_sign_accuracy": bool(row["even_sign_accuracy"] >= GATES["even_sign_accuracy_min"]),
        "demand_stability": bool(row["demand_bootstrap_cosine_median"] >= GATES["demand_bootstrap_cosine_median_min"]),
    }
    return {"passed": all(checks.values()), "checks": checks}


def write_plot(predictions: pd.DataFrame, output_dir: Path) -> None:
    figure = go.Figure()
    colors = {"uncheatable": "#2166ac", "table9": "#b2182b"}
    for target in TARGETS:
        rows = predictions[predictions["target"].eq(target)]
        figure.add_trace(
            go.Scatter(
                x=rows["observed_even"],
                y=rows["predicted_even"],
                mode="markers+text",
                text=rows["domain"],
                textposition="top center",
                textfont={"size": 8},
                marker={"size": 9, "color": colors[target], "opacity": 0.78},
                name=target,
                hovertemplate=("<b>%{text}</b><br>observed even=%{x:.6f}<br>predicted even=%{y:.6f}<extra></extra>"),
            )
        )
    limits = predictions[["observed_even", "predicted_even"]].to_numpy(float)
    lower = float(np.min(limits))
    upper = float(np.max(limits))
    figure.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            line={"color": "#222", "dash": "dash"},
            name="y=x",
        )
    )
    figure.update_layout(
        title="Fisher-Rao demand overlap: frozen odd-to-even transfer",
        xaxis_title="Observed pair-even BPB effect",
        yaxis_title="Predicted pair-even BPB effect",
        template="plotly_white",
        width=1100,
        height=780,
    )
    figure.write_html(output_dir / "odd_to_even_transfer.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, decision: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Fisher-Rao demand-overlap parity test",
        "",
        f"**Decision: {decision['decision']}**",
        "",
        (
            "Only antithetic pair-odd outcomes selected the ridge and demand direction. "
            "Pair-even effects were held out from selection."
        ),
        "",
        metrics.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]
    if decision["passed"]:
        lines.append(
            "The bounded overlap law predicts independently withheld curvature on both targets. "
            "This licenses, but does not promote, a tied-only nested-CV and raw-optimum audit."
        )
    else:
        lines.append(
            "The bounded Fisher-Rao overlap does not predict independently withheld curvature on both targets. "
            "Do not fit it to tied endpoints or repair it with free linear terms, signed square-root coefficients, "
            "another power, or an output link."
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `protocol.json`: frozen pre-outcome protocol",
            "- `selection.csv`: odd-only ridge selection",
            "- `aggregate_metrics.csv`: frozen target decisions",
            "- `pair_predictions.csv`: all odd/even observations and predictions",
            "- `demand_weights.csv`: fitted target-demand distributions",
            "- `odd_to_even_transfer.html`: independent curvature visualization",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def evaluate(output_dir: Path) -> None:
    protocol = verify_protocol(output_dir)
    manifest, domains, c0, c1, proportional = sur065.geometry()
    effects = sur065.target_effects(manifest, domains)
    bundle = feature_bundle(manifest, domains, c0, c1, proportional)
    ridges, selection = select_ridges(bundle, effects)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    demand_rows: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}
    for target in TARGETS:
        effect = effects[target]
        ridge = ridges[target]
        odd_cv_prediction = cv_predictions(bundle.odd, effect.odd, ridge)
        coefficients = sur065.fit_nonnegative_ridge(bundle.odd, effect.odd, ridge)
        even_prediction = bundle.even @ coefficients
        odd_metrics = sur065.effect_metrics(effect.odd, odd_cv_prediction)
        even_metrics = sur065.effect_metrics(effect.even, even_prediction)
        intervals = sur065.metric_bootstrap(effect.even, even_prediction, rng)
        stability = demand_bootstrap(bundle.odd, effect.odd, ridge, coefficients, rng)
        amplitudes = coefficients[:-1]
        magnitude = float(np.linalg.norm(amplitudes))
        demand = amplitudes**2
        demand /= max(float(demand.sum()), 1e-12)
        metric_row = {
            "target": target,
            "selected_ridge": ridge,
            "odd_cv_rmse": odd_metrics["rmse"],
            "odd_null_rmse": odd_metrics["null_rmse"],
            "odd_cv_rmse_improvement": odd_metrics["rmse_improvement"],
            "odd_cv_spearman": odd_metrics["spearman"],
            "even_rmse": even_metrics["rmse"],
            "even_null_rmse": even_metrics["null_rmse"],
            "even_rmse_improvement": even_metrics["rmse_improvement"],
            "even_rmse_improvement_ci_low": intervals["rmse_improvement_ci"][0],
            "even_rmse_improvement_ci_high": intervals["rmse_improvement_ci"][1],
            "even_spearman": even_metrics["spearman"],
            "even_spearman_ci_low": intervals["spearman_ci"][0],
            "even_spearman_ci_high": intervals["spearman_ci"][1],
            "even_sign_accuracy": even_metrics["sign_accuracy"],
            "even_bias": even_metrics["bias"],
            "even_observed_on_predicted_slope": even_metrics["observed_on_predicted_slope"],
            "overlap_magnitude_bpb": magnitude,
            "replay_coefficient_bpb": float(coefficients[-1]),
            "active_demand_buckets": int(np.sum(amplitudes > 1e-10)),
            "maximum_demand_weight": float(np.max(demand)),
            "demand_entropy": float(stats.entropy(demand + 1e-30)),
            "anchor_bpb": effect.anchor,
            "anchor_sd": effect.anchor_sd,
            **stability,
        }
        metric_rows.append(metric_row)
        decisions[target] = target_decision(pd.Series(metric_row))
        for index, domain in enumerate(domains):
            pair_rows.append(
                {
                    "target": target,
                    "domain": domain,
                    "observed_odd": effect.odd[index],
                    "predicted_odd_oof": odd_cv_prediction[index],
                    "observed_even": effect.even[index],
                    "predicted_even": even_prediction[index],
                    "plus_bpb": effect.plus[index],
                    "minus_bpb": effect.minus[index],
                    "anchor_bpb": effect.anchor,
                }
            )
            demand_rows.append(
                {
                    "target": target,
                    "domain": domain,
                    "amplitude_bpb": amplitudes[index],
                    "demand_weight": demand[index],
                    "proportional_pool_mass": proportional[index],
                }
            )

    metrics = pd.DataFrame(metric_rows)
    passed = all(item["passed"] for item in decisions.values())
    decision = {
        "candidate_id": CANDIDATE_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "passed": passed,
        "targets": decisions,
        "decision": "PASS: license tied-panel evaluation" if passed else "FAIL: production route blocked",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_dir / "selection.csv", index=False)
    metrics.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    predictions = pd.DataFrame(pair_rows)
    predictions.to_csv(output_dir / "pair_predictions.csv", index=False)
    pd.DataFrame(demand_rows).to_csv(output_dir / "demand_weights.csv", index=False)
    (output_dir / "decision.json").write_text(json.dumps(json_ready(decision), indent=2, sort_keys=True) + "\n")
    write_plot(predictions, output_dir)
    write_report(metrics, decision, output_dir)
    print(json.dumps(json_ready(decision), indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        freeze_protocol(args.output_dir)
        return
    evaluate(args.output_dir)


if __name__ == "__main__":
    main()
