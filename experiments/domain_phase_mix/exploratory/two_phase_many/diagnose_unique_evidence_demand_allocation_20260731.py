# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Falsify conserved unique-evidence demand allocation at 300M.

Only the odd effects of the 39 physically tied antithetic proportional log
tilts select the shared evidence pseudocount, target-specific ridge, and demand
amplitudes. The unused even effects test the production law's implied
composition curvature. In-sample odd reconstruction is explicitly non-evidence.

For tied policies, materialized epochs ``E_i`` and proportional pool mass
``p_i`` define unique evidence and its normalized composition:

    m_i = p_i min(E_i, 1)
    U = sum_i m_i
    s_i = (m_i + epsilon p_i) / (U + epsilon)

The aggregate response is

    A = b + M KL(q || s) + h (E_total - U),

represented by nonnegative coefficients ``a_i = M q_i`` on ``-log(s_i)`` and
one nonnegative coefficient on repeated materialized mass. All selection uses
pair-odd outcomes; pair-even outcomes remain a frozen falsification panel.
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
from scipy.optimize import lsq_linear
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_intervention_identified_component_transfer_20260731 as transfer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    swarm39_harness_20260725 as swarm39,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "unique_evidence_demand_allocation_20260731"

CANDIDATE_ID = "WSD80-SUR-065"
PROTOCOL_VERSION = "unique-evidence-demand-allocation-parity-v1"
TARGETS = ("uncheatable", "table9")
TARGET_COLUMNS = {
    "uncheatable": "eval/uncheatable_eval/bpb",
    "table9": "table9_macro_bpb",
}
EPSILON_GRID = (0.01, 0.03, 0.10, 0.30, 1.00)
RIDGE_GRID = (0.01, 0.10, 1.00, 10.0, 100.0)
CV_SPLITS = 5
CV_SEED = 7_316_501
BOOTSTRAP_DRAWS = 4_000
COEFFICIENT_BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEED = 7_316_502
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

GATES = {
    "odd_cv_rmse_improvement_min": 0.20,
    "even_rmse_improvement_min": 0.05,
    "even_rmse_improvement_ci_low_min": 0.0,
    "even_spearman_min": 0.25,
    "even_sign_accuracy_min": 0.60,
    "demand_bootstrap_cosine_median_min": 0.80,
    "shared_epsilon_must_be_interior": True,
}


@dataclass(frozen=True)
class TargetEffects:
    """Observed pair parity effects for one aggregate target."""

    name: str
    odd: np.ndarray
    even: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    anchor: float
    anchor_sd: float


@dataclass(frozen=True)
class FeatureBundle:
    """Production features and exact antithetic contrasts for one epsilon."""

    epsilon: float
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
        "purpose": "Parity falsification of a conserved unique-evidence composition production law",
        "mechanism": {
            "unique_evidence": "m_i = p_i min(E_i, 1)",
            "total_unique_evidence": "U = sum_i m_i",
            "composition": "s_i = (m_i + epsilon p_i) / (U + epsilon)",
            "response": "A = b + M KL(q||s) + h(E_total-U)",
            "linear_head": "a_i=M q_i>=0 on -log(s_i), h>=0 on repeated mass",
        },
        "selection_data": "39 pair-odd effects from central antithetic proportional log tilts",
        "falsification_data": "39 pair-even effects relative to 11 proportional controls",
        "important_non_evidence": "In-sample odd reconstruction is not evidence; no even outcome selects any setting",
        "shared_state_selection": {
            "epsilon_grid": EPSILON_GRID,
            "criterion": "mean target-normalized odd cross-validation RMSE",
            "target_specific_ridge_grid": RIDGE_GRID,
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
            "Both targets must pass every gate. Passing licenses tied-panel nested CV and raw optimization; "
            "failure blocks the route before endpoint fitting."
        ),
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): sha256(path)
            for path in (
                Path(__file__),
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
    print(json.dumps(json_ready({"protocol": payload, "preflight": preflight}), indent=2, sort_keys=True))


def verify_protocol(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"Freeze the protocol before evaluation: {path}")
    frozen = json.loads(path.read_text())
    current = json_ready(protocol_payload())
    if frozen != current:
        raise ValueError("Current source or data differs from the frozen protocol")
    return frozen


def geometry() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
    manifest, manifest_domains = transfer.load_manifest()
    domains, c0, c1, _family_index, _family_names = swarm39._exposure("300m_two_phase_fit")
    if set(manifest_domains) != set(domains):
        raise ValueError("Intervention and 300M exposure domains differ")
    manifest = manifest.copy()
    phase0_columns = [f"phase_0_{domain}" for domain in domains]
    missing = sorted(set(phase0_columns) - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest lacks phase weights: {missing[:3]}")
    proportional = (1.0 / c0) / np.sum(1.0 / c0)
    analytic = transfer.pctrl.load_geometry()
    analytic_p = np.asarray([analytic.p[analytic.domains.index(domain)] for domain in domains])
    if not np.allclose(proportional, analytic_p, atol=2e-7, rtol=0.0):
        raise ValueError("Catalog and intervention proportional policies disagree")
    return manifest, list(domains), c0, c1, proportional


def preflight_payload() -> dict[str, Any]:
    manifest, domains, c0, c1, proportional = geometry()
    tilts = manifest[manifest["intervention_type"].eq("central_log_tilt")]
    pair_counts = tilts.groupby("target_domain")["tilt_sign"].nunique()
    if len(pair_counts) != 39 or not np.all(pair_counts.to_numpy() == 2):
        raise ValueError("Expected 39 complete antithetic directions")
    anchor_epochs = (c0 + c1) * proportional
    return {
        "directions": len(pair_counts),
        "domains": len(domains),
        "physically_tied": True,
        "proportional_weight_sum": float(proportional.sum()),
        "proportional_epoch_min": float(anchor_epochs.min()),
        "proportional_epoch_max": float(anchor_epochs.max()),
        "central_tilt_rows": len(tilts),
        "deletion_outcomes_used": False,
    }


def paired_rows(manifest: pd.DataFrame, domains: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tilts = manifest[manifest["intervention_type"].eq("central_log_tilt")]
    plus = tilts[tilts["tilt_sign"].eq("plus")].set_index("target_domain").loc[domains]
    minus = tilts[tilts["tilt_sign"].eq("minus")].set_index("target_domain").loc[domains]
    return plus, minus


def target_effects(manifest: pd.DataFrame, domains: list[str]) -> dict[str, TargetEffects]:
    bundles = {
        "uncheatable": transfer.load_uncheatable(manifest),
        "table9": transfer.load_table9(manifest),
    }
    out: dict[str, TargetEffects] = {}
    for target, bundle in bundles.items():
        joined, controls = bundle[0], bundle[1]
        plus, minus = paired_rows(joined, domains)
        column = TARGET_COLUMNS[target]
        plus_y = plus[column].to_numpy(dtype=float)
        minus_y = minus[column].to_numpy(dtype=float)
        control_y = controls[column].to_numpy(dtype=float)
        anchor = float(np.mean(control_y))
        out[target] = TargetEffects(
            name=target,
            odd=0.5 * (plus_y - minus_y),
            even=0.5 * (plus_y + minus_y) - anchor,
            plus=plus_y,
            minus=minus_y,
            anchor=anchor,
            anchor_sd=float(np.std(control_y, ddof=1)),
        )
    return out


def production_features(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    proportional: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    epochs = weights * (c0 + c1)[None, :]
    unique_mass = proportional[None, :] * np.minimum(epochs, 1.0)
    total_unique = unique_mass.sum(axis=1)
    composition = (unique_mass + epsilon * proportional[None, :]) / (total_unique[:, None] + epsilon)
    repeated_mass = (proportional[None, :] * np.maximum(epochs - 1.0, 0.0)).sum(axis=1)
    if np.any(composition <= 0.0) or not np.allclose(composition.sum(axis=1), 1.0):
        raise ValueError("Invalid unique-evidence composition")
    return np.column_stack([-np.log(composition), repeated_mass])


def feature_bundle(
    manifest: pd.DataFrame,
    domains: list[str],
    c0: np.ndarray,
    c1: np.ndarray,
    proportional: np.ndarray,
    epsilon: float,
) -> FeatureBundle:
    plus, minus = paired_rows(manifest, domains)
    columns = [f"phase_0_{domain}" for domain in domains]
    plus_x = production_features(plus[columns].to_numpy(float), c0, c1, proportional, epsilon)
    minus_x = production_features(minus[columns].to_numpy(float), c0, c1, proportional, epsilon)
    anchor_x = production_features(proportional[None, :], c0, c1, proportional, epsilon)[0]
    return FeatureBundle(
        epsilon=epsilon,
        plus=plus_x,
        minus=minus_x,
        anchor=anchor_x,
        odd=0.5 * (plus_x - minus_x),
        even=0.5 * (plus_x + minus_x) - anchor_x[None, :],
    )


def fit_nonnegative_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), 1e-10)
    scaled = design / scale[None, :]
    augmented_design = np.vstack([scaled, np.sqrt(ridge) * np.eye(design.shape[1])])
    augmented_target = np.concatenate([target, np.zeros(design.shape[1])])
    result = lsq_linear(augmented_design, augmented_target, bounds=(0.0, np.inf), method="trf", max_iter=2_000)
    if not result.success:
        raise RuntimeError(f"Nonnegative ridge fit failed: {result.message}")
    return result.x / scale


def cv_predictions(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    predictions = np.empty_like(target)
    splitter = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    for train, test in splitter.split(design):
        coefficients = fit_nonnegative_ridge(design[train], target[train], ridge)
        predictions[test] = design[test] @ coefficients
    return predictions


def select_shape(
    features: dict[float, FeatureBundle],
    effects: dict[str, TargetEffects],
) -> tuple[float, dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    for epsilon, bundle in features.items():
        for target in TARGETS:
            observed = effects[target].odd
            null_rmse = float(np.sqrt(np.mean(observed**2)))
            for ridge in RIDGE_GRID:
                predicted = cv_predictions(bundle.odd, observed, ridge)
                rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
                rows.append(
                    {
                        "epsilon": epsilon,
                        "target": target,
                        "ridge": ridge,
                        "odd_cv_rmse": rmse,
                        "odd_null_rmse": null_rmse,
                        "normalized_rmse": rmse / null_rmse,
                    }
                )
    table = pd.DataFrame(rows)
    best_by_target = (
        table.sort_values(["epsilon", "target", "normalized_rmse", "ridge"])
        .groupby(["epsilon", "target"], as_index=False)
        .first()
    )
    shared = best_by_target.groupby("epsilon", as_index=False)["normalized_rmse"].mean()
    selected_epsilon = float(shared.sort_values(["normalized_rmse", "epsilon"]).iloc[0]["epsilon"])
    selected = best_by_target[best_by_target["epsilon"].eq(selected_epsilon)]
    ridges = {row.target: float(row.ridge) for row in selected.itertuples()}
    table["selected"] = table.apply(
        lambda row: bool(row["epsilon"] == selected_epsilon and row["ridge"] == ridges[row["target"]]), axis=1
    )
    return selected_epsilon, ridges, table


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    value = stats.spearmanr(observed, predicted).statistic
    return float(value) if value is not None and np.isfinite(value) else 0.0


def effect_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    null_rmse = float(np.sqrt(np.mean(observed**2)))
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.ptp(predicted) > 1e-12 else 0.0
    return {
        "rmse": rmse,
        "null_rmse": null_rmse,
        "rmse_improvement": 1.0 - rmse / null_rmse,
        "bias": float(np.mean(residual)),
        "spearman": safe_spearman(observed, predicted),
        "sign_accuracy": float(np.mean(np.sign(observed) == np.sign(predicted))),
        "observed_on_predicted_slope": slope,
    }


def metric_bootstrap(observed: np.ndarray, predicted: np.ndarray, rng: np.random.Generator) -> dict[str, list[float]]:
    improvements = np.empty(BOOTSTRAP_DRAWS)
    correlations = np.empty(BOOTSTRAP_DRAWS)
    count = len(observed)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, count, count)
        sample_observed = observed[indices]
        sample_predicted = predicted[indices]
        rmse = np.sqrt(np.mean((sample_predicted - sample_observed) ** 2))
        null_rmse = np.sqrt(np.mean(sample_observed**2))
        improvements[draw] = 1.0 - rmse / max(null_rmse, 1e-12)
        correlations[draw] = safe_spearman(sample_observed, sample_predicted)
    return {
        "rmse_improvement_ci": np.quantile(improvements, [0.025, 0.975]).tolist(),
        "spearman_ci": np.quantile(correlations, [0.025, 0.975]).tolist(),
    }


def demand_bootstrap(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    full_coefficients: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    full_demand = full_coefficients[:-1]
    full_demand = full_demand / max(float(full_demand.sum()), 1e-12)
    cosines = np.empty(COEFFICIENT_BOOTSTRAP_DRAWS)
    count = len(target)
    for draw in range(COEFFICIENT_BOOTSTRAP_DRAWS):
        indices = rng.integers(0, count, count)
        coefficients = fit_nonnegative_ridge(design[indices], target[indices], ridge)
        demand = coefficients[:-1]
        demand = demand / max(float(demand.sum()), 1e-12)
        denominator = max(float(np.linalg.norm(demand) * np.linalg.norm(full_demand)), 1e-12)
        cosines[draw] = float(np.dot(demand, full_demand) / denominator)
    return {
        "demand_bootstrap_cosine_median": float(np.median(cosines)),
        "demand_bootstrap_cosine_ci_low": float(np.quantile(cosines, 0.025)),
        "demand_bootstrap_cosine_ci_high": float(np.quantile(cosines, 0.975)),
    }


def decision_for(target_rows: pd.DataFrame, selected_epsilon: float) -> tuple[bool, dict[str, Any]]:
    target_decisions: dict[str, Any] = {}
    for target in TARGETS:
        row = target_rows[target_rows["target"].eq(target)].iloc[0]
        checks = {
            "odd_cv_improvement": bool(row["odd_cv_rmse_improvement"] >= GATES["odd_cv_rmse_improvement_min"]),
            "even_improvement": bool(row["even_rmse_improvement"] >= GATES["even_rmse_improvement_min"]),
            "even_improvement_ci": bool(
                row["even_rmse_improvement_ci_low"] >= GATES["even_rmse_improvement_ci_low_min"]
            ),
            "even_spearman": bool(row["even_spearman"] >= GATES["even_spearman_min"]),
            "even_sign_accuracy": bool(row["even_sign_accuracy"] >= GATES["even_sign_accuracy_min"]),
            "demand_stability": bool(
                row["demand_bootstrap_cosine_median"] >= GATES["demand_bootstrap_cosine_median_min"]
            ),
        }
        target_decisions[target] = {"passed": all(checks.values()), "checks": checks}
    epsilon_interior = selected_epsilon not in {min(EPSILON_GRID), max(EPSILON_GRID)}
    passed = all(item["passed"] for item in target_decisions.values()) and epsilon_interior
    return passed, {
        "passed": passed,
        "shared_epsilon": selected_epsilon,
        "shared_epsilon_interior": epsilon_interior,
        "targets": target_decisions,
        "decision": "PASS: license tied-panel evaluation" if passed else "FAIL: production route blocked",
    }


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
        go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", line={"color": "#222", "dash": "dash"}, name="y=x")
    )
    figure.update_layout(
        title="Unique-evidence demand allocation: frozen odd-to-even transfer",
        xaxis_title="Observed pair-even BPB effect",
        yaxis_title="Predicted pair-even BPB effect",
        template="plotly_white",
        width=1100,
        height=780,
    )
    figure.write_html(output_dir / "odd_to_even_transfer.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, decision: dict[str, Any], output_dir: Path) -> None:
    lines = [
        "# Unique-evidence demand allocation parity test",
        "",
        f"**Decision: {decision['decision']}**",
        "",
        f"Shared selected evidence pseudocount: `{decision['shared_epsilon']}` "
        f"(interior: `{decision['shared_epsilon_interior']}`).",
        "",
        "Only antithetic pair-odd outcomes selected the evidence state and response head. "
        "Pair-even effects were held out from selection.",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
    ]
    if decision["passed"]:
        lines.extend(
            [
                "The independently implied curvature passes both targets. This licenses, but does not promote, "
                "a tied-only nested-CV and raw-optimum audit of the production law.",
            ]
        )
    else:
        lines.extend(
            [
                "The conserved evidence-composition law does not predict independently withheld curvature on "
                "both targets. Do not fit it to the 282 tied endpoints or repair it by selecting a different "
                "pseudocount, occupancy exponent, replay feature, or output link after this result.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `protocol.json`: frozen pre-outcome protocol",
            "- `selection.csv`: odd-only shape and ridge selection",
            "- `aggregate_metrics.csv`: frozen target decisions",
            "- `pair_predictions.csv`: all odd/even observations and predictions",
            "- `demand_weights.csv`: fitted target-demand distributions",
            "- `odd_to_even_transfer.html`: independent curvature visualization",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def evaluate(output_dir: Path) -> None:
    protocol = verify_protocol(output_dir)
    manifest, domains, c0, c1, proportional = geometry()
    effects = target_effects(manifest, domains)
    features = {epsilon: feature_bundle(manifest, domains, c0, c1, proportional, epsilon) for epsilon in EPSILON_GRID}
    selected_epsilon, ridges, selection = select_shape(features, effects)
    bundle = features[selected_epsilon]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    metric_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    demand_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        effect = effects[target]
        ridge = ridges[target]
        odd_cv_prediction = cv_predictions(bundle.odd, effect.odd, ridge)
        coefficients = fit_nonnegative_ridge(bundle.odd, effect.odd, ridge)
        even_prediction = bundle.even @ coefficients
        odd_metrics = effect_metrics(effect.odd, odd_cv_prediction)
        even_metrics = effect_metrics(effect.even, even_prediction)
        intervals = metric_bootstrap(effect.even, even_prediction, rng)
        stability = demand_bootstrap(bundle.odd, effect.odd, ridge, coefficients, rng)
        demand_amplitudes = coefficients[:-1]
        mass = float(demand_amplitudes.sum())
        demand = demand_amplitudes / max(mass, 1e-12)
        metric_rows.append(
            {
                "target": target,
                "selected_epsilon": selected_epsilon,
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
                "demand_mass_bpb": mass,
                "replay_coefficient_bpb": float(coefficients[-1]),
                "active_demand_buckets": int(np.sum(demand_amplitudes > 1e-10)),
                "maximum_demand_weight": float(np.max(demand)),
                "demand_entropy": float(stats.entropy(demand + 1e-30)),
                "anchor_bpb": effect.anchor,
                "anchor_sd": effect.anchor_sd,
                **stability,
            }
        )
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
                    "amplitude_bpb": demand_amplitudes[index],
                    "demand_weight": demand[index],
                    "proportional_pool_mass": proportional[index],
                }
            )

    metrics = pd.DataFrame(metric_rows)
    passed, decision = decision_for(metrics, selected_epsilon)
    decision["protocol_sha256"] = protocol["protocol_sha256"]
    decision["candidate_id"] = CANDIDATE_ID
    decision["passed"] = passed
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
